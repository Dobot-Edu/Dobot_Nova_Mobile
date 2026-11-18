"""把根目录 ``config/`` 转成强类型硬件配置并执行校验。"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

import numpy as np

from config import dataset, devices, teleop


@dataclass(frozen=True)
class NetworkConfig:
    listen_host: str
    data_port: int
    discovery_port: int
    allowed_quest_ip: str
    allowed_quest_subnet: str
    packet_timeout_s: float
    deadman_grip_threshold: float


@dataclass(frozen=True)
class RobotConfig:
    ip: str
    dashboard_port: int
    rate_hz: float
    speed_factor: int
    max_joint_speed_deg_s: float
    max_tcp_speed_mm_s: float
    max_tcp_rotation_deg_s: float
    auto_clear_error: bool
    auto_recover_manual_mode: bool
    startup_pose_enabled: bool
    startup_joint_deg: np.ndarray
    startup_speed_factor: int
    startup_tolerance_deg: float
    startup_timeout_s: float


@dataclass(frozen=True)
class MappingConfig:
    position_axis: np.ndarray
    position_scale: float
    orientation_axis: np.ndarray
    max_translation_mm: float
    max_rotation_deg: float
    max_packet_translation_mm: float
    max_packet_rotation_deg: float


@dataclass(frozen=True)
class GripperConfig:
    enabled: bool
    port: str
    servo_id: int
    raw_open: int
    raw_close: int
    max_close_command: int
    feedback_raw_open: int
    feedback_raw_close: int
    speed: int
    target_torque: int
    command_deadband: int
    trigger_press_threshold: float
    trigger_release_threshold: float
    close_rate_per_s: float
    open_rate_per_s: float
    stop_on_overload_32: bool
    force_feedback_enabled: bool
    current_limit_ma: float
    current_trip_samples: int
    feedback_interval_s: float
    feedback_valid_age_s: float
    feedback_abort_age_s: float
    hold_torque: int
    mode: str = "hold"


@dataclass(frozen=True)
class LiftConfig:
    enabled: bool
    host: str
    port: int
    unit_id: int
    home_on_start: bool
    home_di_index: int
    home_speed: int
    home_timeout_s: float
    max_height_mm: float
    command_rate_hz: float
    drive_speed: int
    release_compensation_mm: float
    socket_timeout_s: float
    auto_enable_on_start: bool


@dataclass(frozen=True)
class ChassisConfig:
    enabled: bool
    require_grip: bool
    url: str
    linear_speed_mm_s: int
    angular_speed_mdeg_s: int
    deadzone: float
    command_rate_hz: float
    feedback_rate_hz: float
    mode_confirm_timeout_s: float
    rpc_timeout_s: float
    ready_wait_s: float
    allow_manual_localization_warning: bool


@dataclass(frozen=True)
class Config:
    network: NetworkConfig
    robot: RobotConfig
    mapping: MappingConfig
    gripper: GripperConfig
    lift: LiftConfig
    chassis: ChassisConfig


def _vector(values: object, count: int) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.shape != (count,) or not np.all(np.isfinite(value)):
        raise ValueError(f"配置项必须包含 {count} 个有限数值")
    return value


def load_config(profile: str | None = None) -> Config:
    """构造运行配置。

    ``profile`` 只决定哪些执行器启用，所有数值参数均来自根目录
    ``config/``，避免多个 INI 文件长期漂移。
    """
    profile = profile or teleop.TELEOP_PROFILE
    profiles = {
        # 训练/部署任务固定升降台；full 不允许 A/B 误触改变高度。
        "full": (True, not dataset.FIXED_LIFT_FOR_POLICY, True),
        "arm_only": (True, False, False),
        # 临时展示模式：full 的全部功能，加上 A/B 长按升降台控制。
        "test": (True, True, True),
    }
    if profile not in profiles:
        raise ValueError(f"未知遥操模式 {profile!r}，应为 {', '.join(profiles)}")
    gripper_enabled, lift_enabled, chassis_enabled = profiles[profile]
    result = Config(
        network=NetworkConfig(
            listen_host=devices.QUEST_LISTEN_HOST,
            data_port=devices.QUEST_DATA_PORT,
            discovery_port=devices.QUEST_DISCOVERY_PORT,
            allowed_quest_ip=devices.QUEST_ALLOWED_IP,
            allowed_quest_subnet=devices.QUEST_ALLOWED_SUBNET,
            packet_timeout_s=devices.QUEST_PACKET_TIMEOUT_S,
            deadman_grip_threshold=devices.QUEST_DEADMAN_GRIP_THRESHOLD,
        ),
        robot=RobotConfig(
            ip=devices.ROBOT_IP,
            dashboard_port=devices.ROBOT_DASHBOARD_PORT,
            rate_hz=devices.ROBOT_CONTROL_HZ,
            speed_factor=devices.ROBOT_SPEED_FACTOR,
            max_joint_speed_deg_s=devices.ROBOT_MAX_JOINT_SPEED_DEG_S,
            max_tcp_speed_mm_s=devices.ROBOT_MAX_TCP_SPEED_MM_S,
            max_tcp_rotation_deg_s=devices.ROBOT_MAX_TCP_ROTATION_DEG_S,
            auto_clear_error=devices.ROBOT_AUTO_CLEAR_ERROR,
            auto_recover_manual_mode=devices.ROBOT_AUTO_RECOVER_MANUAL_MODE,
            startup_pose_enabled=devices.STARTUP_POSE_ENABLED,
            startup_joint_deg=_vector(devices.STARTUP_JOINT_DEG, 6),
            startup_speed_factor=devices.STARTUP_SPEED_FACTOR,
            startup_tolerance_deg=devices.STARTUP_TOLERANCE_DEG,
            startup_timeout_s=devices.STARTUP_TIMEOUT_S,
        ),
        mapping=MappingConfig(
            position_axis=np.asarray(teleop.POSITION_AXIS, dtype=float),
            position_scale=teleop.POSITION_SCALE,
            orientation_axis=np.asarray(teleop.ORIENTATION_AXIS, dtype=float),
            max_translation_mm=teleop.MAX_TRANSLATION_MM,
            max_rotation_deg=teleop.MAX_ROTATION_DEG,
            max_packet_translation_mm=teleop.MAX_PACKET_TRANSLATION_MM,
            max_packet_rotation_deg=teleop.MAX_PACKET_ROTATION_DEG,
        ),
        gripper=GripperConfig(
            enabled=gripper_enabled,
            port=devices.GRIPPER_PORT,
            servo_id=devices.GRIPPER_SERVO_ID,
            raw_open=devices.GRIPPER_RAW_OPEN,
            raw_close=devices.GRIPPER_RAW_CLOSE,
            max_close_command=devices.GRIPPER_MAX_CLOSE_COMMAND,
            feedback_raw_open=devices.GRIPPER_FEEDBACK_RAW_OPEN,
            feedback_raw_close=devices.GRIPPER_FEEDBACK_RAW_CLOSE,
            speed=devices.GRIPPER_SPEED,
            target_torque=devices.GRIPPER_TARGET_TORQUE,
            command_deadband=devices.GRIPPER_COMMAND_DEADBAND,
            trigger_press_threshold=devices.GRIPPER_TRIGGER_PRESS_THRESHOLD,
            trigger_release_threshold=devices.GRIPPER_TRIGGER_RELEASE_THRESHOLD,
            close_rate_per_s=devices.GRIPPER_CLOSE_RATE_PER_S,
            open_rate_per_s=devices.GRIPPER_OPEN_RATE_PER_S,
            stop_on_overload_32=devices.GRIPPER_STOP_ON_OVERLOAD_32,
            force_feedback_enabled=devices.GRIPPER_FORCE_FEEDBACK_ENABLED,
            current_limit_ma=devices.GRIPPER_CURRENT_LIMIT_MA,
            current_trip_samples=devices.GRIPPER_CURRENT_TRIP_SAMPLES,
            feedback_interval_s=devices.GRIPPER_FEEDBACK_INTERVAL_S,
            feedback_valid_age_s=devices.GRIPPER_FEEDBACK_VALID_AGE_S,
            feedback_abort_age_s=devices.GRIPPER_FEEDBACK_ABORT_AGE_S,
            hold_torque=devices.GRIPPER_HOLD_TORQUE,
            mode=devices.GRIPPER_MODE,
        ),
        lift=LiftConfig(
            enabled=lift_enabled,
            host=devices.LIFT_HOST,
            port=devices.LIFT_PORT,
            unit_id=devices.LIFT_UNIT_ID,
            home_on_start=devices.LIFT_HOME_ON_START,
            home_di_index=devices.LIFT_HOME_DI_INDEX,
            home_speed=devices.LIFT_HOME_SPEED,
            home_timeout_s=devices.LIFT_HOME_TIMEOUT_S,
            max_height_mm=devices.LIFT_MAX_HEIGHT_MM,
            command_rate_hz=devices.LIFT_COMMAND_RATE_HZ,
            drive_speed=devices.LIFT_DRIVE_SPEED,
            release_compensation_mm=devices.LIFT_RELEASE_COMPENSATION_MM,
            socket_timeout_s=devices.LIFT_SOCKET_TIMEOUT_S,
            auto_enable_on_start=devices.LIFT_AUTO_ENABLE_ON_START,
        ),
        chassis=ChassisConfig(
            enabled=chassis_enabled,
            require_grip=devices.CHASSIS_REQUIRE_GRIP,
            url=devices.CHASSIS_URL,
            linear_speed_mm_s=devices.CHASSIS_LINEAR_SPEED_MM_S,
            angular_speed_mdeg_s=devices.CHASSIS_ANGULAR_SPEED_MDEG_S,
            deadzone=devices.CHASSIS_DEADZONE,
            command_rate_hz=devices.CHASSIS_COMMAND_RATE_HZ,
            feedback_rate_hz=devices.CHASSIS_FEEDBACK_RATE_HZ,
            mode_confirm_timeout_s=devices.CHASSIS_MODE_CONFIRM_TIMEOUT_S,
            rpc_timeout_s=devices.CHASSIS_RPC_TIMEOUT_S,
            ready_wait_s=devices.CHASSIS_READY_WAIT_S,
            allow_manual_localization_warning=devices.CHASSIS_ALLOW_LOCALIZATION_WARNING,
        ),
    )
    _validate(result)
    return result


def _validate(config: Config) -> None:
    network, robot, mapping, gripper, lift, chassis = (
        config.network,
        config.robot,
        config.mapping,
        config.gripper,
        config.lift,
        config.chassis,
    )
    if not 1 <= network.data_port <= 65535 or not 1 <= network.discovery_port <= 65535:
        raise ValueError("UDP 端口超出范围")
    if network.packet_timeout_s <= 0 or not 0 <= network.deadman_grip_threshold <= 1:
        raise ValueError("超时或 deadman 阈值无效")
    if network.allowed_quest_ip:
        ipaddress.ip_address(network.allowed_quest_ip)
    if network.allowed_quest_subnet:
        ipaddress.ip_network(network.allowed_quest_subnet, strict=False)
    if (
        robot.rate_hz <= 0
        or robot.max_joint_speed_deg_s <= 0
        or robot.max_tcp_speed_mm_s <= 0
        or robot.max_tcp_rotation_deg_s <= 0
    ):
        raise ValueError("控制频率、关节速度和 TCP 速度必须为正数")
    if not 1 <= robot.speed_factor <= 100:
        raise ValueError("机器人速度比例无效")
    if mapping.position_scale <= 0:
        raise ValueError("位置比例必须为正数")
    for name, matrix in (("position_axis", mapping.position_axis), ("orientation_axis", mapping.orientation_axis)):
        if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
            raise ValueError(f"{name} 必须是正交 3x3 矩阵")
    # 姿态映射必须是正规旋转，不能把反射矩阵当成旋转矩阵使用。
    if not np.isclose(np.linalg.det(mapping.orientation_axis), 1.0, atol=1e-6):
        raise ValueError("orientation_axis 的行列式必须为 +1")
    if gripper.enabled and not gripper.port:
        raise ValueError("gripper.port 不能为空")
    if not 1 <= gripper.servo_id <= 253:
        raise ValueError("gripper.servo_id 必须在 1..253")
    if gripper.raw_open == gripper.raw_close:
        raise ValueError("夹爪开合原始位置不能相同")
    if gripper.feedback_raw_open == gripper.feedback_raw_close:
        raise ValueError("夹爪反馈开闭端点不能相同")
    if not 1 <= gripper.max_close_command <= 255:
        raise ValueError("gripper.max_close_command 必须在 1..255")
    if not 1 <= gripper.speed <= 100:
        raise ValueError("夹爪速度必须在 1..100")
    if not 1 <= gripper.target_torque <= 1000:
        raise ValueError("夹爪目标力矩必须在 1..1000")
    if not 1 <= gripper.command_deadband <= 255:
        raise ValueError("夹爪命令死区必须在 1..255")
    if gripper.mode not in {"hold", "toggle"}:
        raise ValueError("夹爪遥操模式必须为 hold 或 toggle")
    if not (
        0.0 <= gripper.trigger_release_threshold
        < gripper.trigger_press_threshold
        <= 1.0
    ):
        raise ValueError("夹爪切换阈值必须满足 0 <= release < press <= 1")
    if gripper.close_rate_per_s <= 0 or gripper.open_rate_per_s <= 0:
        raise ValueError("夹爪开合位置变化率必须为正数")
    if gripper.current_limit_ma <= 0 or gripper.feedback_interval_s <= 0:
        raise ValueError("夹爪电流阈值和反馈周期必须为正数")
    if not gripper.feedback_interval_s < gripper.feedback_valid_age_s < gripper.feedback_abort_age_s:
        raise ValueError("夹爪反馈周期、有效时限和停止时限必须依次增大")
    if gripper.current_trip_samples < 1:
        raise ValueError("夹爪电流连续触发次数必须大于等于 1")
    if not 1 <= gripper.hold_torque <= gripper.target_torque:
        raise ValueError("夹爪保持力矩必须在 1..target_torque")
    if lift.enabled:
        ipaddress.ip_address(lift.host)
    if not 1 <= lift.port <= 65535 or not 1 <= lift.unit_id <= 247:
        raise ValueError("升降台 Modbus 端口或站号无效")
    if (
        lift.max_height_mm <= 0
        or lift.command_rate_hz <= 0
        or lift.socket_timeout_s <= 0
        or lift.home_timeout_s <= 0
    ):
        raise ValueError("升降台高度、速度、频率和超时必须为正数")
    if not 1 <= lift.drive_speed <= 65535:
        raise ValueError("升降台驱动速度必须在 1..65535")
    if not 0.0 <= lift.release_compensation_mm <= 20.0:
        raise ValueError("升降台松开补偿必须在 0..20 mm")
    if not 1 <= lift.home_speed <= 65535 or not 1 <= lift.home_di_index <= 32:
        raise ValueError("升降台回零速度或 DI 编号无效")
    if not chassis.url.startswith(("http://", "https://")):
        raise ValueError("底盘 RPC 地址必须以 http:// 或 https:// 开头")
    if not 1 <= chassis.linear_speed_mm_s <= 200:
        raise ValueError("底盘线速度必须在 1..200 mm/s")
    if not 1000 <= chassis.angular_speed_mdeg_s <= 30000:
        raise ValueError("底盘角速度必须在 1000..30000（0.001°/s）")
    if not 0 <= chassis.deadzone < 0.9:
        raise ValueError("底盘摇杆死区必须在 [0, 0.9) ")
    if not 5 <= chassis.command_rate_hz <= 30:
        raise ValueError("底盘指令频率必须在 5..30 Hz")
    if not 1 <= chassis.feedback_rate_hz <= chassis.command_rate_hz:
        raise ValueError("底盘反馈频率必须在 1 Hz 到底盘指令频率之间")
    if chassis.mode_confirm_timeout_s <= 0 or chassis.rpc_timeout_s <= 0:
        raise ValueError("底盘模式确认或 RPC 超时必须为正数")
    if not 0 <= chassis.ready_wait_s <= 60:
        raise ValueError("底盘就绪等待时间必须在 0..60 秒")
    if not 1 <= robot.startup_speed_factor <= robot.speed_factor <= 100:
        raise ValueError("启动回位速度必须在 1..robot.speed_factor")
    if robot.startup_tolerance_deg <= 0 or robot.startup_timeout_s <= 0:
        raise ValueError("启动回位到位容差和超时必须为正数")
