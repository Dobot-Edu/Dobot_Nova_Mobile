#!/usr/bin/env python3
"""Nova π0.5 双相机真机推理入口；默认只读，不解析命令行参数。"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import queue
import select
import signal
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import inference as settings


def restart_with_clean_cuda_environment() -> None:
    """在 import torch 前清理 shell 注入的 CUDA 动态库路径。"""

    marker = "_NOVA_PI05_CUDA_ENV_CLEAN"
    inherited = [name for name in ("LD_LIBRARY_PATH", "CUDA_MODULE_LOADING") if os.environ.get(name)]
    if not inherited:
        return
    if os.environ.get(marker) == "1":
        raise RuntimeError(f"无法清除冲突的 CUDA 环境变量：{', '.join(inherited)}")
    environment = os.environ.copy()
    for name in inherited:
        environment.pop(name, None)
    environment[marker] = "1"
    print(f"[环境] 清除 {', '.join(inherited)} 后重新启动", flush=True)
    # 保留包入口和工程路径；直接执行infer.py会丢失Dobot_Inference的父目录。
    project_root = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (project_root, environment.get("PYTHONPATH", "")) if part
    )
    os.execve(
        sys.executable,
        [sys.executable, "-m", "Dobot_Inference.infer", *sys.argv[1:]],
        environment,
    )


# spawn预览进程以__mp_main__导入本文件，不能在子进程中重新启动推理入口。
if __name__ == "__main__":
    restart_with_clean_cuda_environment()

# LeRobot 源码随工程保存；硬件实现统一从 Dobot_Init/Dobot_Collection 导入。
if not settings.LEROBOT_SOURCE.is_dir():
    raise RuntimeError(f"仓库不完整，缺少 LeRobot 源码：{settings.LEROBOT_SOURCE}")
sys.path.insert(0, str(settings.LEROBOT_SOURCE))

# 推理只读取本地检查点和仓库内 tokenizer，禁止运行中访问 Hugging Face。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from Dobot_Inference.chassis import InferenceChassis  # noqa: E402
from Dobot_Inference.control import (  # noqa: E402
    ACTION_NAMES,
    STATE_NAMES,
    ActionLimits,
    build_policy_state,
    next_joint_command,
    sanitize_action,
    tracking_error_rad,
)
from Dobot_Inference.gripper import PolicyGripper  # noqa: E402
from Dobot_Init.chassis_rpc import DEFAULT_PASSWORD, DEFAULT_USERNAME  # noqa: E402
from Dobot_Collection.devices import (  # noqa: E402
    DobotFeedback,
    ModbusHeight,
    TopHttpCamera,
    WristRealSense,
)
from Dobot_Init.robot import NovaRobot, RobotError, controller_has_errors  # noqa: E402


STOP = threading.Event()
# 模型、处理器和训练元数据只与权重相关；将其保留在模块全局变量中，即可在
# 多轮真机任务之间继续占用 GPU，而不用反复读取 9.35 GB safetensors。
MODEL_CACHE: tuple[Any, Any, list[str], Any, Any, Any] | None = None


def request_stop(_signum=None, _frame=None) -> None:
    STOP.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


@dataclass(frozen=True)
class ActionPlan:
    origin_step: int
    submitted_at: float
    model_actions: torch.Tensor
    raw_actions: np.ndarray
    actions: np.ndarray
    inference_s: float
    estimated_delay_steps: int = 0

    def action_at(self, policy_step: int) -> tuple[np.ndarray, int]:
        index = int(policy_step - self.origin_step)
        if not 0 <= index < len(self.actions):
            raise RuntimeError(
                f"动作块已耗尽：step={policy_step}, origin={self.origin_step}, "
                f"horizon={len(self.actions)}"
            )
        return self.actions[index].copy(), index

    def after_delay(self, delay_steps: int, origin_step: int) -> ActionPlan:
        """丢弃推理期间已经过去的动作，并从收到结果的时刻开始执行。"""

        delay = max(0, int(delay_steps))
        if delay >= len(self.actions):
            raise RuntimeError(
                f"RTC 新动作块已被推理延迟耗尽：delay={delay}, horizon={len(self.actions)}"
            )
        return ActionPlan(
            origin_step=int(origin_step),
            submitted_at=self.submitted_at,
            model_actions=self.model_actions[delay:].clone(),
            raw_actions=self.raw_actions[delay:].copy(),
            actions=self.actions[delay:].copy(),
            inference_s=self.inference_s,
            estimated_delay_steps=self.estimated_delay_steps,
        )


@dataclass(frozen=True)
class PlanningFailure:
    error: Exception


class Tee(io.StringIO):
    """既保留模型加载输出用于校验，也把它实时显示在终端。"""

    def __init__(self, target) -> None:
        super().__init__()
        self.target = target

    def write(self, value: str) -> int:
        self.target.write(value)
        self.target.flush()
        return super().write(value)


class AsyncPlanner:
    """所有 policy/CUDA 调用只在一个后台线程中执行。"""

    _STOP = object()

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        limits: ActionLimits,
        *,
        initial_latency_s: float,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.limits = limits
        self._requests: queue.Queue[object] = queue.Queue(maxsize=1)
        self._results: queue.Queue[ActionPlan | PlanningFailure] = queue.Queue(maxsize=1)
        self._pending = False
        self._submitted_at = 0.0
        self._deadline = 0.0
        self._latencies: list[float] = [float(initial_latency_s)]
        self._thread = threading.Thread(target=self._worker, name="pi05-planner", daemon=True)
        self._thread.start()

    @property
    def pending(self) -> bool:
        return self._pending

    def submit(
        self,
        origin_step: int,
        frame: dict[str, Any],
        *,
        prev_chunk_left_over: torch.Tensor | None = None,
        deadline: float | None = None,
    ) -> None:
        if self._pending:
            raise RuntimeError("上一轮 π0.5 推理尚未完成")
        self._submitted_at = time.monotonic()
        self._deadline = self._submitted_at + settings.MAX_INFERENCE_LATENCY_S
        if deadline is not None:
            self._deadline = min(self._deadline, deadline)
        if self._deadline <= self._submitted_at:
            raise RuntimeError("剩余动作时间不足以完成 RTC 重规划")
        estimated_delay_steps = 0
        if prev_chunk_left_over is not None:
            estimated_delay_steps = math.ceil(max(self._latencies) * settings.POLICY_HZ)
            prev_chunk_left_over = normalize_rtc_prefix(
                prev_chunk_left_over,
                settings.RTC_EXECUTION_HORIZON_STEPS,
            )
        self._requests.put_nowait(
            (
                int(origin_step),
                self._submitted_at,
                frame,
                prev_chunk_left_over,
                estimated_delay_steps,
            )
        )
        self._pending = True

    def poll(self) -> ActionPlan | None:
        # 即使结果已经入队，过了执行截止时间也不能接受。
        if self._pending and time.monotonic() >= self._deadline:
            raise RuntimeError("π0.5重规划超过剩余动作窗口允许的截止时间，停止执行")
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        self._pending = False
        if isinstance(result, PlanningFailure):
            raise RuntimeError(f"π0.5 推理失败：{result.error}") from result.error
        return result

    def wait(self) -> ActionPlan:
        while not STOP.is_set():
            result = self.poll()
            if result is not None:
                return result
            STOP.wait(0.01)
        raise RuntimeError("等待动作块时收到停止信号")

    def close(self) -> None:
        try:
            self._requests.put_nowait(self._STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        while True:
            request = self._requests.get()
            if request is self._STOP:
                return
            origin_step, submitted_at, frame, prev_chunk_left_over, estimated_delay_steps = request
            try:
                started = time.monotonic()
                model_actions, raw_actions, actions = predict_action_chunk(
                    self.policy,
                    self.preprocessor,
                    self.postprocessor,
                    frame,
                    self.limits,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay_steps=estimated_delay_steps,
                )
                inference_s = time.monotonic() - started
                self._latencies.append(inference_s)
                del self._latencies[:-settings.RTC_LATENCY_HISTORY]
                result: ActionPlan | PlanningFailure = ActionPlan(
                    origin_step=origin_step,
                    submitted_at=submitted_at,
                    model_actions=model_actions,
                    raw_actions=raw_actions,
                    actions=actions,
                    inference_s=inference_s,
                    estimated_delay_steps=estimated_delay_steps,
                )
            except Exception as error:
                result = PlanningFailure(error)
            self._results.put(result)


class RobotWatchdog:
    """使用独立 Dashboard 连接检查机械臂状态，不阻塞 50 Hz ServoJ。"""

    def __init__(self, robot_config) -> None:
        self.robot_config = robot_config
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, name="nova-watchdog", daemon=True)
        self._thread.start()

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"机械臂状态监控失败：{self._error}") from self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _worker(self) -> None:
        monitor: NovaRobot | None = None
        try:
            monitor = NovaRobot(self.robot_config)
            while not self._stop.is_set():
                mode = monitor.mode()
                errors = monitor.request("GetErrorID()")
                if mode not in (5, 7) or controller_has_errors(errors):
                    raise RuntimeError(f"mode={mode}, errors={errors.strip()}")
                self._stop.wait(1.0)
        except Exception as error:
            self._error = error
        finally:
            if monitor is not None:
                monitor.close()


def validate_thor_runtime() -> None:
    """在 NVIDIA Thor 上检查板型和 nvpmodel；普通电脑只做代码测试时跳过。"""

    compatible_path = Path("/proc/device-tree/compatible")
    if not compatible_path.exists():
        print("[平台] 非 Thor 环境，仅允许静态测试，不能执行真机推理")
        return
    compatible = compatible_path.read_bytes().split(b"\0")
    if b"nvidia,p3834-0000" not in compatible:
        raise RuntimeError("当前 NVIDIA 板型不是已经验收的 p3834-0000 Thor")
    expected_nvpmodel = Path("/etc/nvpmodel/nvpmodel_p3834_0000.conf")
    actual_nvpmodel = Path("/etc/nvpmodel.conf").resolve()
    if actual_nvpmodel != expected_nvpmodel:
        raise RuntimeError(
            f"Thor nvpmodel 配置不匹配：{actual_nvpmodel}，应为 {expected_nvpmodel}"
        )
    service = subprocess.run(
        [
            "systemctl",
            "show",
            "nvpmodel.service",
            "--property=Result",
            "--property=ExecMainStatus",
            "--no-pager",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    if service.returncode or "Result=success" not in service.stdout or "ExecMainStatus=0" not in service.stdout:
        raise RuntimeError(f"nvpmodel.service 状态异常：{service.stdout}{service.stderr}")
    print("[Thor] 板型和 nvpmodel 状态正常")


def force_legacy_cublas_linear(module: torch.nn.Module) -> int:
    """绕过 Thor 上已出现过的 cuBLAS-Lt heuristic 失败。"""

    def forward(layer: torch.nn.Linear, values: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(values, layer.weight.transpose(-1, -2))
        return output if layer.bias is None else output + layer.bias

    count = 0
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            child.forward = types.MethodType(forward, child)
            count += 1
    if count == 0:
        raise RuntimeError("π0.5 中没有找到 Linear 层")
    return count


def load_metadata_and_limits():
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    dataset_root = settings.resolve_dataset_meta_root()
    metadata = LeRobotDatasetMetadata(
        f"local/{dataset_root.name}",
        root=dataset_root,
    )
    expected_shapes = {
        "observation.state": [9],
        "observation.images.top": [480, 640, 3],
        "observation.images.wrist": [480, 640, 3],
        "action": [9],
    }
    actual_shapes = {key: list(metadata.features[key]["shape"]) for key in expected_shapes}
    if actual_shapes != expected_shapes or metadata.fps != int(settings.POLICY_HZ):
        raise RuntimeError(
            f"推理元数据不匹配：shape={actual_shapes}, fps={metadata.fps}"
        )
    if tuple(metadata.features["observation.state"]["names"]) != tuple(STATE_NAMES):
        raise RuntimeError("9维状态名称或顺序与推理代码不一致")
    if tuple(metadata.features["action"]["names"]) != tuple(ACTION_NAMES):
        raise RuntimeError("9 维动作名称或顺序与推理代码不一致")
    action_stats = metadata.stats["action"]
    limits = ActionLimits(
        np.asarray(action_stats["min"], dtype=np.float64),
        np.asarray(action_stats["max"], dtype=np.float64),
    )
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"数据集缺少任务表：{tasks_path}")
    import pandas as pd

    task_table = pd.read_parquet(tasks_path)
    if task_table.index.name == "task" and "task_index" in task_table.columns:
        indexed_tasks = [
            (int(row.task_index), str(task))
            for task, row in task_table.iterrows()
        ]
    elif {"task", "task_index"}.issubset(task_table.columns):
        indexed_tasks = [
            (int(row.task_index), str(row.task))
            for row in task_table.itertuples(index=False)
        ]
    else:
        raise RuntimeError(f"无法识别任务表字段：{list(task_table.columns)}")
    tasks = [task for _, task in sorted(indexed_tasks)]
    if settings.TASK_OVERRIDE is not None:
        tasks = [settings.TASK_OVERRIDE]
    if not tasks:
        raise RuntimeError("数据集任务表为空")
    if not 1 <= settings.INITIAL_TASK_INDEX <= len(tasks):
        raise ValueError(
            f"INITIAL_TASK_INDEX={settings.INITIAL_TASK_INDEX} 超出任务数量 {len(tasks)}"
        )
    print(f"[数据] 元信息：{dataset_root}")
    print(f"[任务] 共 {len(tasks)} 个：")
    for index, task in enumerate(tasks, 1):
        print(f"  {index}. {task}")
    return metadata, limits, tasks


def load_policy(metadata):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    model_file = settings.CHECKPOINT_DIR / "model.safetensors"
    required = (
        model_file,
        settings.CHECKPOINT_DIR / "config.json",
        settings.CHECKPOINT_DIR / "policy_preprocessor.json",
        settings.CHECKPOINT_DIR / "policy_postprocessor.json",
        settings.PALIGEMMA_TOKENIZER_DIR / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("推理文件不完整：\n" + "\n".join(missing))
    # 归一化文件编号由处理器序列决定，从保存的配置读取。
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = json.loads((settings.CHECKPOINT_DIR / filename).read_text(encoding="utf-8"))
        for step in processor["steps"]:
            state_file = step.get("state_file")
            if state_file and not (settings.CHECKPOINT_DIR / state_file).is_file():
                raise FileNotFoundError(f"{filename}引用的处理器状态不存在：{state_file}")
    if model_file.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"model.safetensors 体积异常：{model_file.stat().st_size} bytes")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 没有检测到 CUDA")

    config = PreTrainedConfig.from_pretrained(settings.CHECKPOINT_DIR)
    if config.type != "pi05":
        raise RuntimeError(f"检查点不是 π0.5：{config.type}")
    config.pretrained_path = str(settings.CHECKPOINT_DIR)
    horizon = policy_horizon(config)
    if settings.PLAN_MARGIN_STEPS < 1:
        raise ValueError("余量步数必须为正")
    if not settings.RTC_ENABLED:
        raise ValueError("当前异步动作调度依赖 RTC，RTC_ENABLED 必须为 True")
    if not 1 <= settings.RTC_EXECUTION_HORIZON_STEPS <= horizon:
        raise ValueError("RTC execution_horizon 必须在动作窗口范围内")
    if horizon <= settings.PLAN_MARGIN_STEPS:
        raise ValueError("动作窗口过短，无法容纳时间余量")
    if settings.RTC_LATENCY_HISTORY < 1:
        raise ValueError("RTC 延迟历史长度必须为正")
    config.rtc_config = RTCConfig(
        enabled=settings.RTC_ENABLED,
        execution_horizon=settings.RTC_EXECUTION_HORIZON_STEPS,
        max_guidance_weight=settings.RTC_MAX_GUIDANCE_WEIGHT,
    )
    print(
        f"[调度] checkpoint chunk={config.chunk_size}，执行窗口={horizon}步，"
        f"RTC overlap={settings.RTC_EXECUTION_HORIZON_STEPS}步"
    )
    config.device = "cuda"
    config.gradient_checkpointing = False
    config.compile_model = False

    print(f"[模型] 加载 {settings.CHECKPOINT_DIR}", flush=True)
    captured = Tee(sys.stdout)
    with contextlib.redirect_stdout(captured):
        policy = make_policy(config, ds_meta=metadata)
    # 当前 LeRobot 的 PI05Policy 会吞掉加载异常并返回随机模型，因此必须检查成功标记。
    if "All keys loaded successfully!" not in captured.getvalue():
        raise RuntimeError("π0.5 权重没有严格完整加载，禁止使用随机初始化模型控制真机")

    if settings.FORCE_LEGACY_CUBLAS_LINEAR:
        count = force_legacy_cublas_linear(policy)
        print(f"[CUDA兼容] {count} 个 Linear 已切换为 matmul+bias")
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        str(settings.CHECKPOINT_DIR),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(settings.PALIGEMMA_TOKENIZER_DIR)},
        },
    )
    policy.eval()
    policy.reset()
    return policy, preprocessor, postprocessor


def source_value(source, name: str):
    if source.error is not None:
        raise RuntimeError(f"{name}线程失败：{source.error}") from source.error
    sample = source.latest.get()
    if sample is None:
        raise RuntimeError(f"{name}暂无数据")
    age = time.monotonic() - sample.monotonic
    if age > settings.SOURCE_TIMEOUT_S:
        raise RuntimeError(f"{name}数据过期：{age:.3f}s > {settings.SOURCE_TIMEOUT_S:.3f}s")
    return sample.value


def wait_for_sources(sources: dict[str, Any], chassis: InferenceChassis) -> None:
    deadline = time.monotonic() + settings.STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        failures = [f"{name}: {source.error}" for name, source in sources.items() if source.error]
        if failures:
            raise RuntimeError("；".join(failures))
        try:
            chassis.feedback(maximum_age_s=settings.BASE_FEEDBACK_ABORT_AGE_S)
            chassis_ready = True
        except RuntimeError:
            chassis_ready = False
        if chassis_ready and all(source.latest.get() is not None for source in sources.values()):
            lift_description = "升降台、" if settings.VERIFY_LIFT_HEIGHT else ""
            print(f"[输入] 机械臂、{lift_description}底盘和双相机数据均已就绪")
            return
        time.sleep(0.1)
    raise TimeoutError("等待真机输入超时")


def current_state(sources: dict[str, Any], chassis: InferenceChassis, gripper_position: float):
    feedback = source_value(sources["robot"], "机械臂反馈")
    joints_rad = np.deg2rad(np.asarray(feedback["actual_deg"], dtype=np.float64))
    base = chassis.feedback(maximum_age_s=settings.BASE_FEEDBACK_ABORT_AGE_S)
    state = build_policy_state(
        joints_rad,
        gripper_position,
        base.linear_m_s,
        base.angular_rad_s,
    )
    return state, joints_rad, base


def build_frame(
    metadata,
    sources: dict[str, Any],
    chassis: InferenceChassis,
    gripper_position: float,
    task: str,
):
    from lerobot.policies.utils import build_inference_frame

    state, _, _ = current_state(sources, chassis, gripper_position)
    raw: dict[str, Any] = dict(zip(STATE_NAMES, state, strict=True))
    top = source_value(sources["top"], "top 相机")
    wrist = source_value(sources["wrist"], "wrist 相机")
    raw["top"] = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)
    raw["wrist"] = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
    frame = build_inference_frame(
        observation=raw,
        device=torch.device("cpu"),
        ds_features=metadata.features,
        task=task,
        robot_type=metadata.robot_type,
    )
    return frame, top, wrist


def policy_horizon(config) -> int:
    chunk = int(config.chunk_size)
    horizon = chunk if settings.PLAN_HORIZON_STEPS is None else int(settings.PLAN_HORIZON_STEPS)
    if not 1 <= horizon <= chunk:
        raise ValueError(f"动作窗口{horizon}超出检查点chunk_size={chunk}")
    return horizon


def normalize_rtc_prefix(actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """按 LeRobot RTC 的约定截断或补零旧动作前缀。"""

    if actions.ndim != 2 or actions.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"RTC 前缀应为 (N,{len(ACTION_NAMES)})，实际为 {tuple(actions.shape)}")
    actions = actions.detach().to(device="cpu").clone()
    if len(actions) >= target_steps:
        return actions[:target_steps]
    padded = torch.zeros((target_steps, actions.shape[1]), dtype=actions.dtype)
    padded[: len(actions)] = actions
    return padded


def predict_action_chunk(
    policy,
    preprocessor,
    postprocessor,
    frame,
    limits: ActionLimits,
    *,
    prev_chunk_left_over: torch.Tensor | None = None,
    inference_delay_steps: int = 0,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    policy.reset()
    preprocessed = preprocessor(frame)
    rtc_kwargs: dict[str, Any] = {}
    if prev_chunk_left_over is not None:
        device = next(policy.parameters()).device
        rtc_kwargs = {
            "prev_chunk_left_over": prev_chunk_left_over.to(device=device),
            "inference_delay": max(0, int(inference_delay_steps)),
            "execution_horizon": settings.RTC_EXECUTION_HORIZON_STEPS,
        }
    # RTC 的 denoise_step 会在内部临时启用梯度来计算引导修正；inference_mode
    # 无法被覆盖，因此这里只能使用 no_grad。
    with torch.no_grad():
        model_output = policy.predict_action_chunk(preprocessed, **rtc_kwargs)
        actions = postprocessor(model_output)
    result = actions.detach().float().cpu().numpy()
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[2] != len(ACTION_NAMES):
        raise RuntimeError(f"π0.5 动作块形状应为 (1,N,9)，实际为 {result.shape}")
    horizon = policy_horizon(policy.config)
    if result.shape[1] < horizon:
        raise RuntimeError(
            f"动作块只有 {result.shape[1]} 步，小于 {horizon}"
        )
    model_actions = model_output[0, :horizon].detach().float().cpu().clone()
    raw = result[0, :horizon].astype(np.float64, copy=True)
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("π0.5 原始动作块含非有限数值")
    clean = np.stack(
        [
            sanitize_action(
                action,
                limits,
                base_output_scale=settings.BASE_OUTPUT_SCALE,
                base_linear_limit_m_s=settings.BASE_LINEAR_LIMIT_M_S,
                base_angular_limit_rad_s=settings.BASE_ANGULAR_LIMIT_RAD_S,
            )
            for action in raw
        ]
    )
    return model_actions, raw, clean


def check_lift(sources: dict[str, Any], *, execute: bool) -> None:
    if not settings.VERIFY_LIFT_HEIGHT:
        print(
            "[升降台] 已按现场确认固定为 "
            f"{settings.FIXED_LIFT_HEIGHT_M:.3f}m；跳过 Modbus 高度读取与校验"
        )
        return
    actual_m = float(source_value(sources["lift"], "升降台高度")) / 100.0
    error_m = abs(actual_m - settings.FIXED_LIFT_HEIGHT_M)
    message = (
        f"升降台实际={actual_m:.4f}m，固定目标={settings.FIXED_LIFT_HEIGHT_M:.4f}m，"
        f"误差={error_m * 1000:.1f}mm"
    )
    if error_m > settings.LIFT_HEIGHT_TOLERANCE_M:
        if execute:
            raise RuntimeError(message + "；超出容差，拒绝执行，请先用遥操调整")
        print("[升降台警告] " + message)
    else:
        print("[升降台] " + message + "，检查通过")


def confirm_execution() -> None:
    print(
        "\n即将执行 π0.5 真机动作。请确认急停可用、工作区无人、饮料与场景已复位、"
        "机械臂已退出拖拽/JOG。"
    )
    if input("现场确认后直接按 Enter；输入其他内容取消：").strip():
        raise RuntimeError("用户取消真机执行")


def configure_robot_controller(robot: NovaRobot) -> None:
    for command in (
        f"SpeedFactor({settings.CONTROLLER_SPEED_FACTOR})",
        f"AccJ({settings.CONTROLLER_SPEED_FACTOR})",
    ):
        reply = robot.request(command)
        if not reply.startswith("0,"):
            raise RuntimeError(f"机械臂拒绝 {command}：{reply}")
    print(f"[机械臂] 控制器速度和加速度比例={settings.CONTROLLER_SPEED_FACTOR}%")


def draw_preview(top: np.ndarray, wrist: np.ndarray, step: int, action: np.ndarray) -> None:
    canvas = np.hstack([top, wrist])
    cv2.putText(
        canvas,
        f"step={step} grip={action[6]:.2f} base={action[7]:+.3f}/{action[8]:+.3f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Nova PI05 - top | wrist", canvas)
    cv2.waitKey(1)


def terminal_phase_command() -> str | None:
    """非阻塞读取阶段控制；空行结束阶段，q 结束整个会话。"""

    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    value = sys.stdin.readline().strip().lower()
    return "quit" if value in {"q", "quit", "exit"} else "next"


def choose_task(tasks: list[str], default_index: int) -> tuple[int, str] | None:
    while not STOP.is_set():
        print("\n[阶段选择]")
        for index, task in enumerate(tasks, 1):
            marker = "（下一阶段）" if index - 1 == default_index else ""
            print(f"  {index}. {task}{marker}")
        try:
            if len(tasks) == 1:
                prompt = "移除碰撞/障碍后，直接按 Enter 重新开始；输入 q 退出："
            else:
                prompt = (
                    f"直接按 Enter 选择 {default_index + 1}；"
                    f"输入 1..{len(tasks)} 选择阶段；输入 q 退出："
                )
            answer = input(prompt).strip().lower()
        except EOFError:
            return None
        if answer in {"q", "quit", "exit"}:
            return None
        if not answer:
            return default_index, tasks[default_index]
        if answer.isdigit() and 1 <= int(answer) <= len(tasks):
            selected = int(answer) - 1
            return selected, tasks[selected]
        print("[阶段选择] 输入无效")
    return None


def run_task(
    *,
    task_index: int,
    task: str,
    metadata,
    limits: ActionLimits,
    policy,
    preprocessor,
    postprocessor,
    sources: dict[str, Any],
    chassis: InferenceChassis,
    robot: NovaRobot | None,
    gripper: PolicyGripper | None,
    gripper_position: float,
    initial_latency_s: float,
) -> tuple[bool, float]:
    """执行一个提示词阶段；返回是否退出会话以及当前夹爪位置。"""

    policy.reset()
    planner: AsyncPlanner | None = None
    initial_frame, _, _ = build_frame(
        metadata, sources, chassis, gripper_position, task
    )
    try:
        planner = AsyncPlanner(
            policy,
            preprocessor,
            postprocessor,
            limits,
            initial_latency_s=initial_latency_s,
        )
        planner.submit(0, initial_frame)
        current_plan = planner.wait()
        print(
            f"[规划] 阶段 {task_index + 1} 的 {len(current_plan.actions)} 步动作块就绪，"
            f"耗时={current_plan.inference_s:.3f}s"
        )
        print(f"[阶段 {task_index + 1}] {task}")
        print("[阶段控制] 直接按 Enter 结束当前阶段；输入 q 再按 Enter 结束程序")

        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = settings.LOG_DIR / (
            f"pi05_task{task_index + 1}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        start = time.monotonic()
        deadline = math.inf if settings.RUN_SECONDS is None else start + settings.RUN_SECONDS
        next_servo = start
        last_policy_step = -1
        action = current_plan.actions[0].copy()
        _, initial_joints, _ = current_state(sources, chassis, gripper_position)
        command_rad = initial_joints.copy()
        command_velocity = np.zeros(6, dtype=np.float64)
        last_status = 0.0
        latest_inference_s = current_plan.inference_s
        latest_estimated_delay_steps = 0
        latest_actual_delay_steps = 0
        next_controller_status_check = start
        quit_requested = False

        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            while not STOP.is_set() and time.monotonic() < deadline:
                phase_command = terminal_phase_command()
                if phase_command is not None:
                    quit_requested = phase_command == "quit"
                    break
                tick_started = time.monotonic()
                policy_step = int((tick_started - start) * settings.POLICY_HZ)
                state, actual_rad, base_feedback = current_state(
                    sources,
                    chassis,
                    gripper_position,
                )

                if policy_step != last_policy_step:
                    completed = planner.poll()
                    if completed is not None:
                        request_step = completed.origin_step
                        actual_delay_steps = math.ceil(
                            (tick_started - completed.submitted_at) * settings.POLICY_HZ
                        )
                        current_plan = completed.after_delay(actual_delay_steps, policy_step)
                        if len(current_plan.actions) <= settings.PLAN_MARGIN_STEPS:
                            raise RuntimeError("RTC 新动作块扣除真实推理延迟后没有安全执行余量")
                        latest_inference_s = completed.inference_s
                        latest_estimated_delay_steps = completed.estimated_delay_steps
                        latest_actual_delay_steps = actual_delay_steps
                        print(
                            f"\033[36m[推理时延] request={request_step}, start={policy_step}, "
                            f"compute={completed.inference_s * 1000:.1f}ms, "
                            f"end_to_end={(tick_started - completed.submitted_at) * 1000:.1f}ms, "
                            f"RTC估计/实际延迟={completed.estimated_delay_steps}/{actual_delay_steps}步，"
                            f"剩余={len(current_plan.actions)}步\033[0m",
                            flush=True,
                        )

                    action, plan_index = current_plan.action_at(policy_step)
                    raw_action = current_plan.raw_actions[plan_index].copy()

                    if gripper is not None:
                        gripper.submit(float(action[6]))
                        gripper.check_error()
                        gripper_position = gripper.position
                    else:
                        gripper_position = float(action[6])
                    chassis.set_velocity(float(action[7]), float(action[8]))
                    chassis.check_error()

                    if not planner.pending:
                        frame, _, _ = build_frame(
                            metadata,
                            sources,
                            chassis,
                            gripper_position,
                            task,
                        )
                        deadline_step = (
                            current_plan.origin_step
                            + len(current_plan.actions)
                            - settings.PLAN_MARGIN_STEPS
                        )
                        planner.submit(
                            policy_step,
                            frame,
                            prev_chunk_left_over=current_plan.model_actions[plan_index:],
                            deadline=start + deadline_step / settings.POLICY_HZ,
                        )
                    last_policy_step = policy_step
                else:
                    plan_index = policy_step - current_plan.origin_step

                command_rad, command_velocity = next_joint_command(
                    action[:6],
                    actual_rad,
                    command_rad,
                    command_velocity,
                    servo_hz=settings.SERVO_HZ,
                    max_speed_rad_s=np.deg2rad(settings.MAX_JOINT_SPEED_DEG_S),
                    max_acceleration_rad_s2=np.deg2rad(settings.MAX_JOINT_ACCEL_DEG_S2),
                    max_target_jump_rad=np.deg2rad(settings.MAX_TARGET_JUMP_DEG),
                    enforce_target_jump=settings.ABORT_ON_TARGET_JUMP,
                )
                tracking = tracking_error_rad(actual_rad, command_rad)
                if settings.EXECUTE:
                    if robot is None:
                        raise RuntimeError("EXECUTE=true 但机械臂尚未初始化")
                    if tick_started >= next_controller_status_check:
                        mode = robot.mode()
                        errors = robot.request("GetErrorID()")
                        if mode not in (5, 7) or controller_has_errors(errors):
                            raise RobotError(
                                f"控制柜状态异常：mode={mode}, errors={errors.strip()}"
                            )
                        next_controller_status_check = tick_started + settings.CONTROLLER_STATUS_INTERVAL_S
                    robot.servo_j(command_rad)

                row = {
                    "time_ns": time.time_ns(),
                    "execute": settings.EXECUTE,
                    "task_index": task_index,
                    "task": task,
                    "checkpoint": str(settings.CHECKPOINT_DIR),
                    "policy_step": policy_step,
                    "state": state.tolist(),
                    "raw_action": raw_action.tolist(),
                    "safe_action": action.tolist(),
                    "command_joint_rad": command_rad.tolist(),
                    "tracking_error_deg": float(np.rad2deg(tracking)),
                    "base_feedback_m_s": base_feedback.linear_m_s,
                    "base_feedback_rad_s": base_feedback.angular_rad_s,
                    "plan_origin": current_plan.origin_step,
                    "plan_index": plan_index,
                    "latest_inference_s": latest_inference_s,
                    "rtc_estimated_delay_steps": latest_estimated_delay_steps,
                    "rtc_actual_delay_steps": latest_actual_delay_steps,
                }
                log.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

                if tick_started - last_status >= settings.LOG_EVERY_SECONDS:
                    mode = "EXEC" if settings.EXECUTE else "DRY"
                    print(
                        f"[{mode}] task={task_index + 1} step={policy_step} "
                        f"J实测={np.round(np.rad2deg(actual_rad), 1).tolist()} "
                        f"J模型={np.round(np.rad2deg(action[:6]), 1).tolist()} "
                        f"误差={np.rad2deg(tracking):.2f}° "
                        f"夹爪实测={gripper_position:.3f} 目标={action[6]:.3f} "
                        f"底盘={action[7]:+.3f}m/s,{action[8]:+.3f}rad/s "
                        f"推理={latest_inference_s:.3f}s",
                        flush=True,
                    )
                    last_status = tick_started

                next_servo += 1.0 / settings.SERVO_HZ
                remaining = next_servo - time.monotonic()
                if remaining < -1.0 / settings.SERVO_HZ:
                    next_servo = time.monotonic()
                    remaining = 0.0
                STOP.wait(max(0.0, remaining))

        print(f"[完成] 阶段 {task_index + 1} 推理日志：{log_path}")
        return quit_requested, gripper_position
    finally:
        # 阶段边界只发送零速度并清空规划上下文，控制会话保持可复用。
        try:
            chassis.pause()
        finally:
            if planner is not None:
                planner.close()


def run() -> int:
    global MODEL_CACHE
    validate_thor_runtime()
    if MODEL_CACHE is None:
        metadata, limits, tasks = load_metadata_and_limits()
        policy, preprocessor, postprocessor = load_policy(metadata)
        MODEL_CACHE = (metadata, limits, tasks, policy, preprocessor, postprocessor)
        print("[模型] 已常驻 GPU；切换阶段不会重新加载权重")
    else:
        metadata, limits, tasks, policy, preprocessor, postprocessor = MODEL_CACHE
    task_index = settings.INITIAL_TASK_INDEX - 1
    task = tasks[task_index]
    robot_config = settings.HARDWARE
    sources = {
        "robot": DobotFeedback(settings.ROBOT_IP, settings.ROBOT_FEEDBACK_PORT),
        "top": TopHttpCamera(settings.TOP_CAMERA_URL, settings.SOURCE_TIMEOUT_S),
        "wrist": WristRealSense(
            settings.WRIST_CAMERA_SERIAL,
            settings.CAMERA_WIDTH,
            settings.CAMERA_HEIGHT,
            settings.CAMERA_FPS,
        ),
    }
    if settings.VERIFY_LIFT_HEIGHT:
        sources["lift"] = ModbusHeight(
            settings.LIFT_HOST,
            settings.LIFT_PORT,
            settings.LIFT_UNIT_ID,
            settings.LIFT_HEIGHT_REGISTER,
            settings.LIFT_POLL_HZ,
        )
    chassis = InferenceChassis(
        url=robot_config.chassis.url,
        username=DEFAULT_USERNAME,
        password=DEFAULT_PASSWORD,
        command_hz=settings.POLICY_HZ,
        feedback_hz=settings.BASE_FEEDBACK_HZ,
        command_timeout_s=settings.BASE_COMMAND_TIMEOUT_S,
        linear_limit_m_s=settings.BASE_LINEAR_LIMIT_M_S,
        angular_limit_rad_s=settings.BASE_ANGULAR_LIMIT_RAD_S,
        linear_acceleration_m_s2=settings.BASE_LINEAR_ACCELERATION_M_S2,
        angular_acceleration_rad_s2=settings.BASE_ANGULAR_ACCELERATION_RAD_S2,
        allow_localization_warning=robot_config.chassis.allow_manual_localization_warning,
        rpc_timeout_s=robot_config.chassis.rpc_timeout_s,
        mode_confirm_timeout_s=robot_config.chassis.mode_confirm_timeout_s,
    )
    robot: NovaRobot | None = None
    gripper: PolicyGripper | None = None
    gripper_position = 1.0

    from Dobot_Init.camera_preview import CameraPreview
    preview = CameraPreview(sources, 'Nova 推理')
    for source in sources.values():
        source.start()
    preview.start()
    try:
        chassis.connect_read_only()
        wait_for_sources(sources, chassis)
        check_lift(sources, execute=settings.EXECUTE)

        # 首次推理只用于 CUDA、双相机、tokenizer 和输出形状验收，不下发动作。
        warmup_frame, _, _ = build_frame(metadata, sources, chassis, 1.0, task)
        warmup_started = time.monotonic()
        _, warmup_raw_actions, warmup_actions = predict_action_chunk(
            policy, preprocessor, postprocessor, warmup_frame, limits
        )
        torch.cuda.synchronize()
        warmup_s = time.monotonic() - warmup_started
        print(
            f"[预热] π0.5 双相机推理通过：{warmup_s:.3f}s，"
            f"原始首动作={np.round(warmup_raw_actions[0], 4).tolist()}，"
            f"限幅后={np.round(warmup_actions[0], 4).tolist()}"
        )

        if settings.EXECUTE:
            confirm_execution()
            gripper = PolicyGripper(
                robot_config.gripper,
                execute=True,
            )
            gripper.connect()
            chassis.enable_remote()
            robot = NovaRobot(robot_config.robot)
            robot.prepare_for_control()
            robot.move_to_startup_pose()
            configure_robot_controller(robot)
        else:
            print("[DRY-RUN] 不使能机械臂、夹爪或底盘，只记录预测动作")

        gripper_position = 1.0 if gripper is None else gripper.position
        first_stage = True
        needs_robot_recovery = False
        while not STOP.is_set():
            if not first_stage:
                next_index = task_index if needs_robot_recovery else (task_index + 1) % len(tasks)
                selected = choose_task(tasks, next_index)
                if selected is None:
                    break
                task_index, task = selected

            if settings.EXECUTE and robot is not None and needs_robot_recovery:
                try:
                    print("[机械臂恢复] 正在清除上一阶段报警并重新使能；不会自动运动……")
                    robot.prepare_for_control()
                    configure_robot_controller(robot)
                    print("[机械臂恢复] 完成，可以开始新的推理阶段")
                    needs_robot_recovery = False
                except RobotError as error:
                    print(
                        f"[机械臂恢复失败] {error}\n"
                        "请移除碰撞/障碍并检查机械臂状态，然后重新选择阶段；输入 q 可退出。",
                        file=sys.stderr,
                        flush=True,
                    )
                    first_stage = False
                    continue

            try:
                quit_requested, gripper_position = run_task(
                    task_index=task_index,
                    task=task,
                    metadata=metadata,
                    limits=limits,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    sources=sources,
                    chassis=chassis,
                    robot=robot,
                    gripper=gripper,
                    gripper_position=gripper_position,
                    initial_latency_s=warmup_s,
                )
            except RobotError as error:
                # 控制柜碰撞/报警或 ServoJ 拒绝只结束当前阶段。控制器负责制动；
                # 此处再明确停止运动并保持模型、相机和程序常驻。
                try:
                    if robot is not None:
                        robot.request("Stop()")
                except Exception as stop_error:
                    print(f"[机械臂停止失败] {stop_error}", file=sys.stderr, flush=True)
                print(
                    f"[阶段停止] 机械臂控制器异常：{error}\n"
                    "当前推理阶段已结束，底盘已停车。程序保持运行。",
                    file=sys.stderr,
                    flush=True,
                )
                needs_robot_recovery = True
                quit_requested = False

            first_stage = False
            if quit_requested or STOP.is_set():
                break
        return 0
    finally:
        # 无论模型、CUDA、相机或机械臂哪里异常，底盘都先变成零速度。
        # 某个关闭操作失败也必须继续关闭其他执行器。
        cleanup = [("底盘立即停车", chassis.stop), ("相机预览", preview.close)]
        cleanup.append(("底盘连接", chassis.close))
        if gripper is not None:
            cleanup.append(("夹爪", gripper.close))
        if robot is not None:
            cleanup.append(("机械臂", robot.close))
        cleanup.extend((name, source.close) for name, source in sources.items())
        cleanup.append(("窗口", cv2.destroyAllWindows))
        for name, close in cleanup:
            try:
                close()
            except Exception as error:
                print(f"[关闭失败] {name}: {error}", file=sys.stderr, flush=True)


def main() -> int:
    try:
        return run()
    except Exception as error:
        print(f"[推理停止] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
