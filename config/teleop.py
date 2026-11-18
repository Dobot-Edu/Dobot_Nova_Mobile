# full：机械臂、夹爪和底盘启用；训练任务中升降台固定，不允许遥操。
# arm_only：只启用机械臂和夹爪。
# test：展示模式，full 全功能启用，并允许长按 A/B 控制升降台上升/下降。
TELEOP_PROFILE = "full"

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
MAX_PACKET_TRANSLATION_MM = 60.0
MAX_PACKET_ROTATION_DEG = 25.0
