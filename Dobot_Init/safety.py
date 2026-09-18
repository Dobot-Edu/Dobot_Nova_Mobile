"""独立于 IK 的笛卡尔与关节指令跳变保护。"""

from __future__ import annotations

import numpy as np

from quest_vr.kinematics import pose6_transform, rotation_matrix, rotation_vector, transform_pose6


class SafetyError(RuntimeError):
    """目标违反关节安全限制。"""


def bounded_cartesian_target(
    requested_pose: np.ndarray,
    current_pose: np.ndarray,
    max_translation_step_mm: float,
    max_rotation_step_deg: float,
) -> np.ndarray:
    """保留六维最终目标，但分成小步交给本地 IK。"""
    if max_translation_step_mm <= 0 or max_rotation_step_deg <= 0:
        raise ValueError("TCP 单步位移和旋转限制必须为正数")
    requested = pose6_transform(np.asarray(requested_pose, dtype=float))
    current = pose6_transform(np.asarray(current_pose, dtype=float))

    translation = requested[:3, 3] - current[:3, 3]
    translation_norm = float(np.linalg.norm(translation))
    if translation_norm > max_translation_step_mm:
        translation *= max_translation_step_mm / translation_norm

    # 在旋转矩阵上限幅，不直接减 RPY，避免 ±180° 包角跳变。
    rotation_step = rotation_vector(requested[:3, :3] @ current[:3, :3].T)
    rotation_norm = float(np.linalg.norm(rotation_step))
    max_rotation_step = float(np.deg2rad(max_rotation_step_deg))
    if rotation_norm > max_rotation_step:
        rotation_step *= max_rotation_step / rotation_norm

    bounded = current.copy()
    bounded[:3, 3] += translation
    bounded[:3, :3] = rotation_matrix(rotation_step) @ current[:3, :3]
    return transform_pose6(bounded)


def bounded_joint_command(
    requested: np.ndarray,
    previous: np.ndarray,
    max_step_rad: float,
) -> np.ndarray:
    """限制单周期关节变化量，但不限制关节的绝对或相对活动范围。"""
    requested = np.asarray(requested, dtype=float)
    previous = np.asarray(previous, dtype=float)
    if any(value.shape != (6,) for value in (requested, previous)):
        raise ValueError("所有关节数组必须包含六个值")
    if max_step_rad <= 0:
        raise ValueError("关节单步限制必须为正数")
    if not all(np.all(np.isfinite(value)) for value in (requested, previous)):
        raise SafetyError("关节目标包含非有限数值")
    return previous + np.clip(requested - previous, -max_step_rad, max_step_rad)
