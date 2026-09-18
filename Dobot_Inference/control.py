"""π0.5 动作检查、平滑和状态拼接；本文件不依赖真机或 CUDA。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Dobot_Collection.schema import ACTION_NAMES, STATE_NAMES


@dataclass(frozen=True)
class ActionLimits:
    """数据集动作范围；顺序必须与 ``ACTION_NAMES`` 完全一致。"""

    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        if minimum.shape != (len(ACTION_NAMES),) or maximum.shape != (len(ACTION_NAMES),):
            raise ValueError("动作范围必须是两个 9 维数组")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("动作范围含非有限数值")
        if np.any(minimum > maximum):
            raise ValueError("动作最小值不能大于最大值")


def finite_vector(values: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"{name} 必须是 {size} 个有限数值，实际为 {result!r}")
    return result


def build_policy_state(
    joints_rad: object,
    gripper_position: float,
    base_linear_m_s: float,
    base_angular_rad_s: float,
) -> np.ndarray:
    """生成训练时约定的9维状态；通信诊断量不进入模型。"""

    joints = finite_vector(joints_rad, 6, "机械臂关节状态")
    scalars = np.asarray(
        [
            gripper_position,
            base_linear_m_s,
            base_angular_rad_s,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalars)):
        raise RuntimeError("夹爪或底盘状态含非有限数值")
    if not 0.0 <= gripper_position <= 1.0:
        raise RuntimeError(f"夹爪实测位置超出 0..1：{gripper_position}")
    return np.asarray([*joints, *scalars], dtype=np.float32)


def sanitize_action(
    action: object,
    limits: ActionLimits,
    *,
    base_output_scale: float,
    base_linear_limit_m_s: float,
    base_angular_limit_rad_s: float,
) -> np.ndarray:
    """校验模型输出；关节不做角度裁剪，夹爪和底盘仍按各自范围保护。"""

    result = finite_vector(action, len(ACTION_NAMES), "π0.5 动作").copy()
    # Dataset min/max are metadata only, not actuator constraints.
    # Keep gripper physical range and optional configured base limits.
    result[6] = np.clip(result[6], 0.0, 1.0)
    result[7] = np.clip(
        result[7] * base_output_scale,
        -base_linear_limit_m_s,
        base_linear_limit_m_s,
    )
    result[8] = np.clip(
        result[8] * base_output_scale,
        -base_angular_limit_rad_s,
        base_angular_limit_rad_s,
    )
    return result


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def blend_actions(old: object, new: object, weight: float) -> np.ndarray:
    """融合新旧连续位置/速度动作。"""

    old_action = finite_vector(old, len(ACTION_NAMES), "旧动作")
    new_action = finite_vector(new, len(ACTION_NAMES), "新动作")
    alpha = smoothstep(weight)
    return (1.0 - alpha) * old_action + alpha * new_action


def next_joint_command(
    target_rad: object,
    actual_rad: object,
    previous_command_rad: object,
    previous_velocity_rad_s: object,
    *,
    servo_hz: float,
    max_speed_rad_s: float,
    max_acceleration_rad_s2: float,
    max_target_jump_rad: float,
    enforce_target_jump: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """从上一条已发送指令积分，实测角只用于安全检查，避免形成正反馈。"""

    target = finite_vector(target_rad, 6, "模型关节目标")
    actual = finite_vector(actual_rad, 6, "实测关节")
    previous = finite_vector(previous_command_rad, 6, "上一关节指令")
    velocity = finite_vector(previous_velocity_rad_s, 6, "上一指令速度")

    if servo_hz <= 0 or max_speed_rad_s <= 0 or max_acceleration_rad_s2 <= 0:
        raise ValueError("ServoJ 频率、速度和加速度上限必须为正")
    jump = float(np.max(np.abs(target - actual)))
    if enforce_target_jump and jump > max_target_jump_rad:
        raise RuntimeError(
            f"模型关节目标离实测角过远：{np.rad2deg(jump):.2f}° > "
            f"{np.rad2deg(max_target_jump_rad):.2f}°"
        )

    # Explicit unlimited mode avoids inf * zero in the limiter.
    if np.isposinf(max_speed_rad_s) and np.isposinf(max_acceleration_rad_s2):
        return target.copy(), (target - previous) * servo_hz

    dt = 1.0 / servo_hz
    distance = target - previous
    # 根据剩余距离计算可刹停速度，再施加加速度约束，避免越过目标后反复纠偏。
    stopping_speed = np.sqrt(2.0 * max_acceleration_rad_s2 * np.abs(distance))
    desired_velocity = np.sign(distance) * np.minimum(stopping_speed, max_speed_rad_s)
    max_delta_velocity = max_acceleration_rad_s2 * dt
    next_velocity = velocity + np.clip(
        desired_velocity - velocity,
        -max_delta_velocity,
        max_delta_velocity,
    )
    command = previous + next_velocity * dt
    return command, next_velocity


def tracking_error_rad(actual_rad: object, command_rad: object) -> float:
    actual = finite_vector(actual_rad, 6, "实测关节")
    command = finite_vector(command_rad, 6, "关节指令")
    return float(np.max(np.abs(actual - command)))
