"""π0.5 推理配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.devices import TOP_CAMERA_URL
from config.paths import PROJECT_ROOT as REPO_DIR

PROJECT_DIR = REPO_DIR / "Dobot_Inference"

# 切换模型通常只需修改此处，或临时设置 NOVA_CHECKPOINT。
DEFAULT_CHECKPOINT = (
    REPO_DIR / "outputs" / "train"
    / "nova_can_20260917_18_pi05_chunk50_030000_inference"
    / "pretrained_model"
)
CHECKPOINT_DIR = Path(os.environ.get("NOVA_CHECKPOINT", DEFAULT_CHECKPOINT)).expanduser().resolve()

# None 时自动检查 CHECKPOINT_DIR、CHECKPOINT_DIR/dataset_meta 和 CHECKPOINT_DIR.parent。
_dataset_override = os.environ.get("NOVA_DATASET_ROOT", "").strip()
DATASET_META_ROOT: Path | None = Path(_dataset_override).expanduser().resolve() if _dataset_override else None
PALIGEMMA_TOKENIZER_DIR = Path(
    os.environ.get("NOVA_TOKENIZER_ROOT", PROJECT_DIR / "assets" / "paligemma_tokenizer")
).expanduser().resolve()
LEROBOT_SOURCE = REPO_DIR / "Dobot_Train" / "lerobot-main" / "src"
LOG_DIR = REPO_DIR / "logs" / "inference"

# 默认采用 meta/tasks.parquet；必要时可在这里或通过 NOVA_TASK 覆盖。
_empty_task = os.environ.get("NOVA_EMPTY_TASK", "0").strip()
if _empty_task not in {"0", "1"}:
    raise ValueError("NOVA_EMPTY_TASK 只能是 0 或 1")
# 诊断开关：NOVA_EMPTY_TASK=1 时向模型传入空字符串，便于仅做 DRY-RUN A/B。
TASK_OVERRIDE = "" if _empty_task == "1" else (os.environ.get("NOVA_TASK", "").strip() or None)
INITIAL_TASK_INDEX = int(os.environ.get("NOVA_TASK_INDEX", "1"))  # 对外使用 1..N

# 默认真机执行。None 表示阶段不限时，在终端按 Enter 结束当前阶段。
EXECUTE = os.environ.get("NOVA_EXECUTE", "1") == "1"
_run_seconds = os.environ.get("NOVA_RUN_SECONDS", "").strip()
RUN_SECONDS: float | None = float(_run_seconds) if _run_seconds else None
if RUN_SECONDS is not None and RUN_SECONDS <= 0:
    raise ValueError("NOVA_RUN_SECONDS 必须为正数")
PREVIEW = True
KEEP_MODEL_LOADED = True
LOG_EVERY_SECONDS = 1.0

POLICY_HZ = 25.0
SERVO_HZ = 50.0
PLAN_HORIZON_STEPS = None
MAX_INFERENCE_LATENCY_S = 2.0
PLAN_MARGIN_STEPS = 1
RTC_ENABLED = True
RTC_EXECUTION_HORIZON_STEPS = 10
RTC_MAX_GUIDANCE_WEIGHT = 10.0
RTC_LATENCY_HISTORY = 20
FORCE_LEGACY_CUBLAS_LINEAR = True

# 机械臂推理执行参数。
MAX_JOINT_SPEED_DEG_S = 10.0
MAX_JOINT_ACCEL_DEG_S2 = 30.0
MAX_TARGET_JUMP_DEG = 90.0  # 仅保留为诊断阈值
ABORT_ON_TARGET_JUMP = False  # 远目标不终止；仍受速度和加速度限制
CONTROLLER_SPEED_FACTOR = 50
CONTROLLER_STATUS_INTERVAL_S = 1.0

# 底盘推理执行参数。
BASE_OUTPUT_SCALE = 1.0
BASE_LINEAR_LIMIT_M_S = 0.10
BASE_ANGULAR_LIMIT_RAD_S = 0.15
BASE_LINEAR_ACCELERATION_M_S2 = 0.10
BASE_ANGULAR_ACCELERATION_RAD_S2 = 0.2
BASE_COMMAND_TIMEOUT_S = 1.0
BASE_FEEDBACK_HZ = 25.0
BASE_FEEDBACK_VALID_AGE_S = 0.30
BASE_FEEDBACK_ABORT_AGE_S = 0.50

# 推理硬件与数据源。
FIXED_LIFT_HEIGHT_M = 0.107
LIFT_HEIGHT_TOLERANCE_M = 0.005
VERIFY_LIFT_HEIGHT = False
ROBOT_IP = "192.168.23.201"
ROBOT_FEEDBACK_PORT = 30004
LIFT_HOST = "192.168.23.254"
LIFT_PORT = 503
LIFT_UNIT_ID = 1
LIFT_HEIGHT_REGISTER = 4615
LIFT_POLL_HZ = 20.0
WRIST_CAMERA_SERIAL = "260522275205"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
SOURCE_TIMEOUT_S = 0.50
STARTUP_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class RobotRuntimeConfig:
    ip: str = ROBOT_IP
    dashboard_port: int = 29999
    rate_hz: float = SERVO_HZ
    speed_factor: int = 70
    max_joint_speed_deg_s: float = 70.0
    max_tcp_speed_mm_s: float = 300.0
    max_tcp_rotation_deg_s: float = 90.0
    auto_clear_error: bool = True
    auto_recover_manual_mode: bool = True
    startup_pose_enabled: bool = True
    startup_joint_deg: np.ndarray | None = None
    startup_speed_factor: int = 40
    startup_tolerance_deg: float = 0.5
    startup_timeout_s: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "startup_joint_deg", np.asarray([90, 0, 90, 0, -90, 0], dtype=float))


@dataclass(frozen=True)
class GripperRuntimeConfig:
    enabled: bool = True
    port: str = "/dev/xtrainer_gripper"
    servo_id: int = 22
    raw_open: int = 2000
    raw_close: int = 1000
    max_close_command: int = 255
    feedback_raw_open: int = 2000
    feedback_raw_close: int = 1000
    speed: int = 30
    target_torque: int = 350
    command_deadband: int = 2
    trigger_press_threshold: float = 0.2
    trigger_release_threshold: float = 0.1
    close_rate_per_s: float = 1.0
    open_rate_per_s: float = 1.0
    stop_on_overload_32: bool = False
    force_feedback_enabled: bool = False
    current_limit_ma: float = 150.0
    current_trip_samples: int = 2
    feedback_interval_s: float = 0.05
    feedback_valid_age_s: float = 0.15
    feedback_abort_age_s: float = 0.5
    hold_torque: int = 250
    mode: str = "toggle"


@dataclass(frozen=True)
class ChassisRuntimeConfig:
    enabled: bool = True
    require_grip: bool = False
    url: str = "http://192.168.23.1"
    linear_speed_mm_s: int = 180
    angular_speed_mdeg_s: int = 15000
    deadzone: float = 0.2
    command_rate_hz: float = 25.0
    feedback_rate_hz: float = 25.0
    mode_confirm_timeout_s: float = 30.0
    rpc_timeout_s: float = 2.0
    ready_wait_s: float = 5.0
    allow_manual_localization_warning: bool = True


@dataclass(frozen=True)
class InferenceHardwareConfig:
    robot: RobotRuntimeConfig
    gripper: GripperRuntimeConfig
    chassis: ChassisRuntimeConfig


HARDWARE = InferenceHardwareConfig(
    robot=RobotRuntimeConfig(),
    gripper=GripperRuntimeConfig(),
    chassis=ChassisRuntimeConfig(),
)


def resolve_dataset_meta_root() -> Path:
    """返回包含 meta/ 的本地数据集根目录。"""
    candidates = (
        (DATASET_META_ROOT,)
        if DATASET_META_ROOT is not None
        else (CHECKPOINT_DIR, CHECKPOINT_DIR / "dataset_meta", CHECKPOINT_DIR.parent)
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError("找不到 checkpoint 对应的数据元信息，已检查：\n" + checked)
