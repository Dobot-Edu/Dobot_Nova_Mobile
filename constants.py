"""Nova 项目的唯一配置入口。

普通使用只需要修改本文件。其余目录保存硬件实现、数据流程和训练逻辑，
不再各自维护重复的 IP、相机序列号、数据路径或任务文字。
"""

from pathlib import Path


# ============================== 项目路径 ==============================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATASET_DIR = DATASETS_DIR / "raw"
LEROBOT_DATASET_DIR = DATASETS_DIR / "lerobot" / "nova_pi05_mobile_position_v3"
EXCLUDED_DATASET_DIR = DATASETS_DIR / "excluded"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# 数据流程使用的简短别名。
RAW_ROOT = RAW_DATASET_DIR
LEROBOT_ROOT = LEROBOT_DATASET_DIR


# ============================== 运行模式 ==============================

# full：机械臂、夹爪和底盘启用；训练任务中升降台固定，不允许遥操。
# arm_only：只启用机械臂和夹爪。
# no_chassis：维护模式，机械臂、夹爪、升降台启用，底盘禁用。
TELEOP_PROFILE = "full"

# NVIDIA 上已经配置好的 Python。迁移到其他主机时只需修改这里，
# 也可以在启动前用环境变量 NOVA_PYTHON 覆盖。
PYTHON_BIN = "/home/nvidia/miniforge3/envs/single/bin/python"


# ============================== 任务与采集 ==============================

TASK = 'Pick up the upright drink bottle from the blue basket on the white table. Keep the bottle securely grasped and turn the mobile base 180 degrees in place. Drive forward toward the black shelf, turn to face the shelf, and stop in front of it. Place the bottle upright on the shelf, release it, and retract the robot arm. Finally, turn the mobile base, drive back to the starting position in front of the white table, and stop.'
TARGET_SUCCESSFUL_EPISODES = None  # 统一入口不限制保存条数
DATASET_FPS = 25
MAX_EPISODE_SECONDS = None  # B 控制结束，不限制时长
COUNTDOWN_SECONDS = 0.0  # B 立即开录
JPEG_QUALITY = 92
DATASET_REPO_ID = "local/nova_pi05_mobile_position_v3"
USE_VIDEOS = True
# NVIDIA Thor 优先使用自动检测到的硬件编码器（当前为 h264_nvenc）。
# 训练只依赖可解码的视频帧，不要求必须使用软件AV1；硬件编码可大幅缩短转换时间。
VIDEO_CODEC = "h264"  # CPU编码，避免auto选中NVENC后在子进程中CUDA初始化失败。
# 旧96条示教是在任务文字定稿前采集的。转换时只重标语言指令，不修改图像和动作。
# 关闭后，只要原始 episode 的任务文字与 TASK 不完全一致，转换器就会拒绝处理。
RELABEL_TASK_DURING_CONVERSION = True
# 新版训练固定升降台为0.107m，不进入状态或动作。
FIXED_LIFT_FOR_POLICY = True
FIXED_LIFT_HEIGHT_M = 0.107
STATE_DIM = 9
ACTION_DIM = 9

# 五类数据源的新鲜度限制。底盘状态经独立RPC取得，允许短时网络抖动。
TELEMETRY_TIMEOUT_S = 0.30
ROBOT_FEEDBACK_TIMEOUT_S = 0.20
LIFT_TIMEOUT_S = 0.30
# 相机允许短暂抖动；帧龄超过0.5秒仍停止采集，避免长期使用旧画面。
CAMERA_TIMEOUT_S = 0.50
BASE_EXPECTED_PERIOD_S = 0.04
BASE_VALID_AGE_S = 0.30
BASE_ABORT_AGE_S = 0.50


# ============================== Quest 网络 ==============================

QUEST_LISTEN_HOST = "0.0.0.0"
QUEST_DATA_PORT = 5005
QUEST_DISCOVERY_PORT = 5006
QUEST_ALLOWED_IP = ""
QUEST_ALLOWED_SUBNET = "10.2.150.0/23"
QUEST_PACKET_TIMEOUT_S = 0.25
QUEST_DEADMAN_GRIP_THRESHOLD = 0.65
DATASET_TELEMETRY_HOST = "127.0.0.1"
DATASET_TELEMETRY_PORT = 17890


# ============================== Nova 机械臂 ==============================

ROBOT_IP = "192.168.23.201"
ROBOT_DASHBOARD_PORT = 29999
ROBOT_FEEDBACK_PORT = 30004
ROBOT_CONTROL_HZ = 50.0
ROBOT_SPEED_FACTOR = 70
ROBOT_MAX_JOINT_SPEED_DEG_S = 70.0
ROBOT_MAX_TCP_SPEED_MM_S = 300.0
ROBOT_MAX_TCP_ROTATION_DEG_S = 90.0
ROBOT_AUTO_CLEAR_ERROR = True
# 启动遥操时自动退出拖拽/拖拽录制，并清理残留的 JOG 状态。
# 恢复过程只停止旧控制状态并重新使能，不下发任何位姿或运动目标。
ROBOT_AUTO_RECOVER_MANUAL_MODE = True

# 启动时低速回到训练标准位。
STARTUP_POSE_ENABLED = True
STARTUP_JOINT_DEG = (90, 0, 90, 0, -90, 0)
STARTUP_SPEED_FACTOR = 40
STARTUP_TOLERANCE_DEG = 0.5
STARTUP_TIMEOUT_S = 20.0


# ============================== VR 到机械臂映射 ==============================

# 2026-08-19 实机标定：物理右 -> -X，前 -> -Y，上 -> +Z。
POSITION_AXIS = (
    (-0.99724721, -0.00947366, -0.07354081),
    (0.07161021, 0.13423096, -0.98835926),
    (-0.01923483, 0.99090479, 0.13318304),
)
POSITION_SCALE = 0.9
ORIENTATION_AXIS = ((-1, 0, 0), (0, -1, 0), (0, 0, 1))
MAX_TRANSLATION_MM = 250.0
MAX_ROTATION_DEG = 90.0
MAX_PACKET_TRANSLATION_MM = 50.0
MAX_PACKET_ROTATION_DEG = 25.0


# ============================== 夹爪 ==============================

GRIPPER_PORT = "/dev/xtrainer_gripper"
GRIPPER_SERVO_ID = 22
# 2026-09-11 统一行程：action位置0/1分别对应2000全开、1000全闭。
GRIPPER_RAW_OPEN = 2000
GRIPPER_RAW_CLOSE = 1000
GRIPPER_MAX_CLOSE_COMMAND = 255
# 按同一端点把编码器实际位置归一化为0..1。
GRIPPER_FEEDBACK_RAW_OPEN = 2000
GRIPPER_FEEDBACK_RAW_CLOSE = 1000
GRIPPER_SPEED = 30
GRIPPER_TARGET_TORQUE = 350
GRIPPER_COMMAND_DEADBAND = 2
# hold：按住 Trigger 连续闭合，松开连续张开；toggle：每次点按切换全闭/全开。
GRIPPER_MODE = "hold"
GRIPPER_TRIGGER_PRESS_THRESHOLD = 0.20
GRIPPER_TRIGGER_RELEASE_THRESHOLD = 0.10
# 归一化位置每秒的变化量；1=任务允许范围内全开，0=任务允许范围内全闭。
GRIPPER_CLOSE_RATE_PER_S = 1.0
GRIPPER_OPEN_RATE_PER_S = 1.0
# True：状态32出现时停止继续闭合；False：仅终端提醒并继续下发闭合目标。
# 该开关不能关闭舵机固件自身的堵转/过载保护。
GRIPPER_STOP_ON_OVERLOAD_32 = False
# 关闭额外的软件电流锁停；舵机自身的过流、过温和堵转保护仍然生效。
# 当前夹爪正常开合电流波动较大，软件阈值会造成频繁误停。
GRIPPER_FORCE_FEEDBACK_ENABLED = False
# 仅在重新开启软件限力时使用：相对空载电流基线的增量阈值。
GRIPPER_CURRENT_LIMIT_MA = 150.0
GRIPPER_CURRENT_TRIP_SAMPLES = 2
GRIPPER_FEEDBACK_INTERVAL_S = 0.04  # 25 Hz，与数据采集频率对齐。
GRIPPER_FEEDBACK_VALID_AGE_S = 0.15
GRIPPER_FEEDBACK_ABORT_AGE_S = 0.50
GRIPPER_HOLD_TORQUE = 250


# ============================== 升降台 ==============================

LIFT_HOST = "192.168.23.254"
LIFT_PORT = 503
LIFT_UNIT_ID = 1
LIFT_HEIGHT_REGISTER = 4615
LIFT_HOME_ON_START = False
LIFT_HOME_DI_INDEX = 2
LIFT_HOME_SPEED = 250
LIFT_HOME_TIMEOUT_S = 120.0
LIFT_MAX_HEIGHT_MM = 400.0
LIFT_COMMAND_RATE_HZ = 50.0
LIFT_POLL_HZ = 20.0
LIFT_DRIVE_SPEED = 300
LIFT_RELEASE_COMPENSATION_MM = 10.0
LIFT_SOCKET_TIMEOUT_S = 0.5
LIFT_AUTO_ENABLE_ON_START = True


# ============================== 移动底盘 ==============================

CHASSIS_URL = "http://192.168.23.1"
CHASSIS_REQUIRE_GRIP = False
CHASSIS_LINEAR_SPEED_MM_S = 180
CHASSIS_ANGULAR_SPEED_MDEG_S = 15000
CHASSIS_DEADZONE = 0.20
# 速度指令与真实状态独立配置；当前都按25 Hz更新。前者决定控制平滑度，
# 后者决定数据集中底盘 observation 的时间分辨率。
CHASSIS_COMMAND_RATE_HZ = 25.0
CHASSIS_FEEDBACK_RATE_HZ = 25.0
CHASSIS_MODE_CONFIRM_TIMEOUT_S = 30.0
CHASSIS_RPC_TIMEOUT_S = 2.0
CHASSIS_READY_WAIT_S = 5.0
CHASSIS_ALLOW_LOCALIZATION_WARNING = True


# ============================== 双相机 ==============================

TOP_CAMERA_SERIAL = "254522076021"
WRIST_CAMERA_SERIAL = "260522275205"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_STREAM_FPS = 30  # 两台RealSense均不支持640×480@20；采集按DATASET_FPS取样。
TOP_CAMERA_URL = "http://127.0.0.1:8080/frame.jpg"
TOP_CAMERA_LISTEN_HOST = "0.0.0.0"
TOP_CAMERA_HTTP_PORT = 8080
TOP_CAMERA_PREVIEW = True
# 遥操和推理的双相机窗口；修改后重启程序生效。
CAMERA_PREVIEW_ENABLED = True
CAMERA_PREVIEW_FPS = 20  # 双画面预览目标刷新率；与硬件采流、数据保存频率独立。
TOP_CAMERA_WINDOW_NAME = "Quest top camera"
DUAL_VIEWER_HOST = "127.0.0.1"
DUAL_VIEWER_PORT = 8765


# ============================== π0.5 训练 ==============================

TRAIN_OUTPUT = OUTPUTS_DIR / "pi05_nova_mobile"
POLICY_REPO_ID = "local/pi05_nova_mobile"
PRETRAINED_MODEL = "lerobot/pi05_base"
TRAIN_STEPS = 30000
SAVE_FREQUENCY = 5000
BATCH_SIZE = 16
NUM_WORKERS = 8
POLICY_CHUNK_SIZE = 25
POLICY_ACTION_STEPS = 25
