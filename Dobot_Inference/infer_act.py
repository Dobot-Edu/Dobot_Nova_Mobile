#!/usr/bin/env python3
"""Nova ACT 双相机试验推理入口；默认 DRY-RUN，与 π0.5 入口独立。"""

from __future__ import annotations

import json
import math
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Dobot_Inference import settings_act as settings


def restart_with_clean_cuda_environment() -> None:
    marker = "_NOVA_ACT_CUDA_ENV_CLEAN"
    inherited = [name for name in ("LD_LIBRARY_PATH", "CUDA_MODULE_LOADING") if os.environ.get(name)]
    if not inherited:
        return
    if os.environ.get(marker) == "1":
        raise RuntimeError(f"无法清除冲突的 CUDA 环境变量：{', '.join(inherited)}")
    environment = os.environ.copy()
    for name in inherited:
        environment.pop(name, None)
    environment[marker] = "1"
    project_root = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (project_root, environment.get("PYTHONPATH", "")) if part
    )
    print(f"[环境] 清除 {', '.join(inherited)} 后重新启动 ACT", flush=True)
    os.execve(sys.executable, [sys.executable, "-m", "Dobot_Inference.infer_act"], environment)


if __name__ == "__main__":
    restart_with_clean_cuda_environment()

if not settings.LEROBOT_SOURCE.is_dir():
    raise RuntimeError(f"缺少 LeRobot 源码：{settings.LEROBOT_SOURCE}")
sys.path.insert(0, str(settings.LEROBOT_SOURCE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from Dobot_Collection.devices import DobotFeedback, TopHttpCamera, WristRealSense  # noqa: E402
from Dobot_Inference.chassis import InferenceChassis  # noqa: E402
from Dobot_Inference.control import (  # noqa: E402
    ACTION_NAMES,
    build_policy_state,
    next_joint_command,
    tracking_error_rad,
)
from Dobot_Inference.gripper import PolicyGripper  # noqa: E402
from Dobot_Init.chassis_rpc import DEFAULT_PASSWORD, DEFAULT_USERNAME  # noqa: E402
from Dobot_Init.robot import NovaRobot, controller_has_errors  # noqa: E402


STOP = threading.Event()


def request_stop(_signum=None, _frame=None) -> None:
    STOP.set()


# multiprocessing 的预览子进程会以 __mp_main__ 导入本模块；只允许真正的
# 推理主进程接管信号，否则 process.terminate() 会被子进程吞掉。
if __name__ == "__main__":
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


@dataclass(frozen=True)
class ActionPlan:
    requested_step: int
    submitted_at: float
    raw_actions: np.ndarray
    actions: np.ndarray
    inference_s: float
    origin_step: int | None = None
    dropped_steps: int = 0

    def aligned(self, policy_step: int, delay_steps: int) -> "ActionPlan":
        delay = max(0, int(delay_steps))
        if delay >= len(self.actions):
            raise RuntimeError(
                f"ACT 动作块被推理延迟耗尽：delay={delay}, chunk={len(self.actions)}"
            )
        return ActionPlan(
            requested_step=self.requested_step,
            submitted_at=self.submitted_at,
            raw_actions=self.raw_actions[delay:].copy(),
            actions=self.actions[delay:].copy(),
            inference_s=self.inference_s,
            origin_step=int(policy_step),
            dropped_steps=delay,
        )

    def at(self, policy_step: int) -> tuple[np.ndarray, np.ndarray, int]:
        if self.origin_step is None:
            raise RuntimeError("ACT 动作块尚未对齐")
        index = int(policy_step - self.origin_step)
        if not 0 <= index < len(self.actions):
            raise RuntimeError(
                f"ACT 动作块耗尽：step={policy_step}, origin={self.origin_step}, remaining={len(self.actions)}"
            )
        return self.raw_actions[index].copy(), self.actions[index].copy(), index


@dataclass(frozen=True)
class PlanningFailure:
    error: Exception


class AsyncACTPlanner:
    _CLOSE = object()

    def __init__(self, policy, preprocessor, postprocessor) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.requests: queue.Queue = queue.Queue(maxsize=1)
        self.results: queue.Queue = queue.Queue(maxsize=1)
        self.pending = False
        self.thread = threading.Thread(target=self._worker, name="act-planner", daemon=True)
        self.thread.start()

    def submit(self, policy_step: int, frame) -> None:
        if self.pending:
            raise RuntimeError("ACT 已有推理请求在途")
        submitted_at = time.monotonic()
        self.requests.put((int(policy_step), submitted_at, frame))
        self.pending = True

    def _worker(self) -> None:
        while True:
            request = self.requests.get()
            if request is self._CLOSE:
                return
            policy_step, submitted_at, frame = request
            try:
                started = time.monotonic()
                self.policy.reset()
                prepared = self.preprocessor(frame)
                with torch.inference_mode():
                    normalized = self.policy.predict_action_chunk(prepared)
                    output = self.postprocessor(normalized)
                torch.cuda.synchronize()
                inference_s = time.monotonic() - started
                raw = output.detach().float().cpu().numpy()
                if raw.shape != (1, self.policy.config.chunk_size, len(ACTION_NAMES)):
                    raise RuntimeError(f"ACT 输出形状异常：{raw.shape}")
                raw = raw[0].astype(np.float64, copy=True)
                clean = np.stack([sanitize_action(action) for action in raw])
                result: ActionPlan | PlanningFailure = ActionPlan(
                    requested_step=policy_step,
                    submitted_at=submitted_at,
                    raw_actions=raw,
                    actions=clean,
                    inference_s=inference_s,
                )
            except Exception as error:
                result = PlanningFailure(error)
            self.results.put(result)

    def poll(self) -> ActionPlan | None:
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.pending = False
        if isinstance(result, PlanningFailure):
            raise RuntimeError(f"ACT 后台推理失败：{result.error}") from result.error
        return result

    def wait(self) -> ActionPlan:
        while not STOP.is_set():
            try:
                result = self.results.get(timeout=0.1)
            except queue.Empty:
                continue
            self.pending = False
            if isinstance(result, PlanningFailure):
                raise RuntimeError(f"ACT 后台推理失败：{result.error}") from result.error
            return result
        raise RuntimeError("等待 ACT 推理时收到停止信号")

    def close(self) -> None:
        if self.thread.is_alive():
            try:
                self.requests.put_nowait(self._CLOSE)
            except queue.Full:
                pass
            self.thread.join(timeout=2.0)


def sanitize_action(action: object) -> np.ndarray:
    result = np.asarray(action, dtype=np.float64)
    if result.shape != (len(ACTION_NAMES),) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"ACT 动作必须是 9 个有限数值：{result!r}")
    result = result.copy()
    result[6] = np.clip(result[6], 0.0, 1.0)
    result[7] = np.clip(
        result[7] * settings.BASE_OUTPUT_SCALE,
        -settings.BASE_LINEAR_LIMIT_M_S,
        settings.BASE_LINEAR_LIMIT_M_S,
    )
    result[8] = np.clip(
        result[8] * settings.BASE_OUTPUT_SCALE,
        -settings.BASE_ANGULAR_LIMIT_RAD_S,
        settings.BASE_ANGULAR_LIMIT_RAD_S,
    )
    return result


def load_policy():
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    checkpoint = settings.CHECKPOINT_DIR
    required = (
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "policy_preprocessor.json",
        checkpoint / "policy_postprocessor.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("ACT checkpoint 不完整：\n" + "\n".join(missing))
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        spec = json.loads((checkpoint / filename).read_text(encoding="utf-8"))
        for step in spec["steps"]:
            state_file = step.get("state_file")
            if state_file and not (checkpoint / state_file).is_file():
                raise FileNotFoundError(f"{filename} 引用文件不存在：{state_file}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 没有检测到 CUDA")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type != "act":
        raise RuntimeError(f"checkpoint 不是 ACT：{config.type}")
    expected_inputs = {
        "observation.state": (9,),
        "observation.images.top": (3, settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH),
        "observation.images.wrist": (3, settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH),
    }
    actual_inputs = {key: tuple(feature.shape) for key, feature in config.input_features.items()}
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"ACT 输入特征不匹配：{actual_inputs}")
    if tuple(config.output_features["action"].shape) != (9,):
        raise RuntimeError(f"ACT 动作维度不是 9：{config.output_features}")
    if not 1 <= settings.REPLAN_AFTER_STEPS < int(config.chunk_size):
        raise ValueError("NOVA_ACT_REPLAN_STEPS 必须小于 chunk_size 且至少为 1")
    print(f"[模型] 加载 ACT：{checkpoint}", flush=True)
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    policy.reset()
    print(
        f"[模型] ACT 加载完成：chunk={config.chunk_size}，"
        f"滚动重规划={settings.REPLAN_AFTER_STEPS}步"
    )
    return policy, preprocessor, postprocessor


def source_value(source, name: str):
    if source.error is not None:
        raise RuntimeError(f"{name}线程失败：{source.error}") from source.error
    sample = source.latest.get()
    if sample is None:
        raise RuntimeError(f"{name}暂无数据")
    age = time.monotonic() - sample.monotonic
    if age > settings.SOURCE_TIMEOUT_S:
        raise RuntimeError(f"{name}数据过期：{age:.3f}s")
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
            print("[输入] 机械臂、底盘和双相机数据均已就绪")
            return
        time.sleep(0.1)
    raise TimeoutError("等待真机输入超时")


def current_state(sources: dict[str, Any], chassis: InferenceChassis, gripper_position: float):
    feedback = source_value(sources["robot"], "机械臂反馈")
    joints_rad = np.deg2rad(np.asarray(feedback["actual_deg"], dtype=np.float64))
    base = chassis.feedback(maximum_age_s=settings.BASE_FEEDBACK_ABORT_AGE_S)
    state = build_policy_state(joints_rad, gripper_position, base.linear_m_s, base.angular_rad_s)
    return state, joints_rad, base


def build_frame(sources: dict[str, Any], chassis: InferenceChassis, gripper_position: float):
    from lerobot.policies.utils import prepare_observation_for_inference

    state, _, _ = current_state(sources, chassis, gripper_position)
    top = source_value(sources["top"], "top 相机")
    wrist = source_value(sources["wrist"], "wrist 相机")
    observation = {
        "observation.state": state,
        "observation.images.top": cv2.cvtColor(top, cv2.COLOR_BGR2RGB),
        "observation.images.wrist": cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB),
    }
    return prepare_observation_for_inference(observation, torch.device("cpu")), top, wrist


def confirm_execution() -> None:
    print("\n即将执行 ACT 真机动作。请确认急停可用、工作区无人、场景已复位。")
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


def run() -> int:
    policy, preprocessor, postprocessor = load_policy()
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
    planner: AsyncACTPlanner | None = None
    gripper_position = 1.0
    from Dobot_Init.camera_preview import CameraPreview

    preview = CameraPreview(sources, "Nova ACT 推理")
    for source in sources.values():
        source.start()
    preview.start()
    try:
        chassis.connect_read_only()
        wait_for_sources(sources, chassis)
        warmup_frame, _, _ = build_frame(sources, chassis, 0.0)
        warmup_started = time.monotonic()
        policy.reset()
        with torch.inference_mode():
            warmup_output = postprocessor(policy.predict_action_chunk(preprocessor(warmup_frame)))
        torch.cuda.synchronize()
        warmup_s = time.monotonic() - warmup_started
        warmup = warmup_output.detach().float().cpu().numpy()
        if warmup.shape != (1, policy.config.chunk_size, len(ACTION_NAMES)) or not np.all(np.isfinite(warmup)):
            raise RuntimeError(f"ACT 预热输出异常：{warmup.shape}")
        print(f"[预热] ACT 双相机推理通过：{warmup_s:.3f}s，首动作={np.round(warmup[0,0],4).tolist()}")

        if settings.EXECUTE:
            confirm_execution()
            gripper = PolicyGripper(robot_config.gripper, execute=True)
            gripper.connect()
            chassis.enable_remote()
            robot = NovaRobot(robot_config.robot)
            robot.prepare_for_control()
            robot.move_to_startup_pose()
            configure_robot_controller(robot)
        else:
            print("[DRY-RUN] 不使能机械臂、夹爪或底盘，只记录 ACT 预测动作")

        initial_frame, _, _ = build_frame(
            sources, chassis, 1.0 if gripper is None else gripper.position
        )
        planner = AsyncACTPlanner(policy, preprocessor, postprocessor)
        planner.submit(0, initial_frame)
        initial = planner.wait()
        if initial.inference_s > settings.MAX_INFERENCE_LATENCY_S:
            raise RuntimeError(
                f"ACT 初始推理超时：{initial.inference_s:.3f}s > {settings.MAX_INFERENCE_LATENCY_S:.3f}s"
            )
        # 控制时钟在初始动作块就绪后才开始，初始块不存在执行延迟。
        delay = 0
        current_plan = initial.aligned(0, delay)
        print(
            f"[规划] 初始 ACT 块耗时={initial.inference_s:.3f}s，"
            f"丢弃={delay}步，剩余={len(current_plan.actions)}步"
        )

        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = settings.LOG_DIR / f"act_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        start = time.monotonic()
        deadline = start + settings.RUN_SECONDS
        next_servo = start
        last_policy_step = -1
        action = current_plan.actions[0].copy()
        raw_action = current_plan.raw_actions[0].copy()
        _, initial_joints, _ = current_state(sources, chassis, gripper_position)
        command_rad = initial_joints.copy()
        command_velocity = np.zeros(6, dtype=np.float64)
        last_status = 0.0
        next_controller_status_check = start
        latest_inference_s = initial.inference_s
        latest_delay_steps = delay

        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            while not STOP.is_set() and time.monotonic() < deadline:
                tick_started = time.monotonic()
                policy_step = int((tick_started - start) * settings.POLICY_HZ)
                state, actual_rad, base_feedback = current_state(sources, chassis, gripper_position)
                if policy_step != last_policy_step:
                    completed = planner.poll()
                    if completed is not None:
                        if completed.inference_s > settings.MAX_INFERENCE_LATENCY_S:
                            raise RuntimeError(
                                f"ACT 推理超时：{completed.inference_s:.3f}s > {settings.MAX_INFERENCE_LATENCY_S:.3f}s"
                            )
                        actual_delay = math.ceil(
                            (tick_started - completed.submitted_at) * settings.POLICY_HZ
                        )
                        current_plan = completed.aligned(policy_step, actual_delay)
                        latest_inference_s = completed.inference_s
                        latest_delay_steps = actual_delay
                        print(
                            f"\033[35m[ACT时延] request={completed.requested_step}, "
                            f"ready={policy_step}, compute={completed.inference_s*1000:.1f}ms, "
                            f"丢弃={actual_delay}步，剩余={len(current_plan.actions)}步\033[0m",
                            flush=True,
                        )
                    raw_action, action, plan_index = current_plan.at(policy_step)
                    if gripper is not None:
                        gripper.submit(float(action[6]))
                        gripper.check_error()
                        gripper_position = gripper.position
                    else:
                        gripper_position = float(action[6])
                    chassis.set_velocity(float(action[7]), float(action[8]))
                    chassis.check_error()
                    if plan_index >= settings.REPLAN_AFTER_STEPS and not planner.pending:
                        frame, _, _ = build_frame(sources, chassis, gripper_position)
                        planner.submit(policy_step, frame)
                    last_policy_step = policy_step
                else:
                    plan_index = policy_step - int(current_plan.origin_step)

                command_rad, command_velocity = next_joint_command(
                    action[:6],
                    actual_rad,
                    command_rad,
                    command_velocity,
                    servo_hz=settings.SERVO_HZ,
                    max_speed_rad_s=np.deg2rad(settings.MAX_JOINT_SPEED_DEG_S),
                    max_acceleration_rad_s2=np.deg2rad(settings.MAX_JOINT_ACCEL_DEG_S2),
                    max_target_jump_rad=np.deg2rad(settings.MAX_TARGET_JUMP_DEG),
                    enforce_target_jump=settings.EXECUTE,
                )
                tracking = tracking_error_rad(actual_rad, command_rad)
                if settings.EXECUTE:
                    if robot is None:
                        raise RuntimeError("ACT EXECUTE=true 但机械臂未初始化")
                    if tick_started >= next_controller_status_check:
                        mode = robot.mode()
                        errors = robot.request("GetErrorID()")
                        if mode not in (5, 7) or controller_has_errors(errors):
                            raise RuntimeError(f"控制柜状态异常：mode={mode}, errors={errors.strip()}")
                        next_controller_status_check = tick_started + settings.CONTROLLER_STATUS_INTERVAL_S
                    robot.servo_j(command_rad)

                log.write(json.dumps({
                    "time_ns": time.time_ns(),
                    "execute": settings.EXECUTE,
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
                    "dropped_delay_steps": latest_delay_steps,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                if tick_started - last_status >= settings.LOG_EVERY_SECONDS:
                    mode = "EXEC" if settings.EXECUTE else "DRY"
                    print(
                        f"[{mode}] step={policy_step} "
                        f"J实测={np.round(np.rad2deg(actual_rad),1).tolist()} "
                        f"J模型={np.round(np.rad2deg(action[:6]),1).tolist()} "
                        f"夹爪={gripper_position:.3f}->{action[6]:.3f} "
                        f"底盘={action[7]:+.3f}m/s,{action[8]:+.3f}rad/s "
                        f"ACT={latest_inference_s:.3f}s",
                        flush=True,
                    )
                    last_status = tick_started
                next_servo += 1.0 / settings.SERVO_HZ
                remaining = next_servo - time.monotonic()
                if remaining < -1.0 / settings.SERVO_HZ:
                    next_servo = time.monotonic()
                    remaining = 0.0
                STOP.wait(max(0.0, remaining))
        print(f"[完成] ACT 推理日志：{log_path}")
        return 0
    finally:
        cleanup = [("底盘立即停车", chassis.stop), ("相机预览", preview.close)]
        if planner is not None:
            cleanup.append(("ACT规划线程", planner.close))
        cleanup.append(("底盘连接", chassis.close))
        if gripper is not None:
            cleanup.append(("夹爪", gripper.close))
        if robot is not None:
            cleanup.append(("机械臂", robot.close))
        cleanup.extend((name, source.close) for name, source in sources.items())
        cleanup.append(("窗口", cv2.destroyAllWindows))
        for name, close in cleanup:
            try:
                print(f"[关闭] {name}……", flush=True)
                close()
                print(f"[关闭] {name}完成", flush=True)
            except Exception as error:
                print(f"[关闭失败] {name}: {error}", file=sys.stderr, flush=True)


def main() -> int:
    try:
        return run()
    except Exception as error:
        print(f"[ACT停止] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
