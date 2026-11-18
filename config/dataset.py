from config.paths import RAW_ROOT, LEROBOT_ROOT

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

DATASET_TELEMETRY_HOST = "127.0.0.1"
DATASET_TELEMETRY_PORT = 17890
