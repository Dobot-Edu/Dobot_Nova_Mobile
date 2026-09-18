"""独立 ACT 试验推理参数；不影响 π0.5 settings.py。"""

from __future__ import annotations

import os
from pathlib import Path

from Dobot_Inference import settings as common


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
CHECKPOINT_DIR = (REPO_DIR / "outputs" / "train" / "pretrained_model").resolve()
LEROBOT_SOURCE = REPO_DIR / "lerobot-main" / "src"
LOG_DIR = REPO_DIR / "logs" / "inference"

# ACT 不使用任务提示词。默认 DRY-RUN，真机必须显式设置 NOVA_ACT_EXECUTE=1。
EXECUTE = os.environ.get("NOVA_ACT_EXECUTE", "0") == "1"
RUN_SECONDS = 10000.0
POLICY_HZ = 20.0
SERVO_HZ = 50.0
REPLAN_AFTER_STEPS = int(os.environ.get("NOVA_ACT_REPLAN_STEPS", "10"))
MAX_INFERENCE_LATENCY_S = float(os.environ.get("NOVA_ACT_MAX_LATENCY_S", "2.0"))
LOG_EVERY_SECONDS = 1.0

# 试运行默认保留当前保守限速，可独立调整，不会改变 π0.5。
MAX_JOINT_SPEED_DEG_S = float(os.environ.get("NOVA_ACT_JOINT_SPEED_DEG_S", "20"))
MAX_JOINT_ACCEL_DEG_S2 = float(os.environ.get("NOVA_ACT_JOINT_ACCEL_DEG_S2", "20"))
MAX_TARGET_JUMP_DEG = float(os.environ.get("NOVA_ACT_MAX_TARGET_JUMP_DEG", "90"))
CONTROLLER_SPEED_FACTOR = int(os.environ.get("NOVA_ACT_CONTROLLER_SPEED_FACTOR", "100"))
CONTROLLER_STATUS_INTERVAL_S = 1.0

BASE_OUTPUT_SCALE = float(os.environ.get("NOVA_ACT_BASE_OUTPUT_SCALE", "1"))
BASE_LINEAR_LIMIT_M_S = float(os.environ.get("NOVA_ACT_BASE_LINEAR_LIMIT_M_S", "0.12"))
BASE_ANGULAR_LIMIT_RAD_S = float(os.environ.get("NOVA_ACT_BASE_ANGULAR_LIMIT_RAD_S", "0.2"))
BASE_LINEAR_ACCELERATION_M_S2 = float(os.environ.get("NOVA_ACT_BASE_LINEAR_ACCELERATION_M_S2", "0.1"))
BASE_ANGULAR_ACCELERATION_RAD_S2 = float(os.environ.get("NOVA_ACT_BASE_ANGULAR_ACCELERATION_RAD_S2", "0.1"))
BASE_COMMAND_TIMEOUT_S = 0.30
BASE_FEEDBACK_HZ = common.BASE_FEEDBACK_HZ
BASE_FEEDBACK_ABORT_AGE_S = common.BASE_FEEDBACK_ABORT_AGE_S

ROBOT_IP = common.ROBOT_IP
ROBOT_FEEDBACK_PORT = common.ROBOT_FEEDBACK_PORT
TOP_CAMERA_URL = common.TOP_CAMERA_URL
WRIST_CAMERA_SERIAL = common.WRIST_CAMERA_SERIAL
CAMERA_WIDTH = common.CAMERA_WIDTH
CAMERA_HEIGHT = common.CAMERA_HEIGHT
CAMERA_FPS = common.CAMERA_FPS
SOURCE_TIMEOUT_S = 0.50
STARTUP_TIMEOUT_S = 20.0
HARDWARE = common.HARDWARE
