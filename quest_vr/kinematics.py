"""Forward kinematics and calibrated pose mapping for Cartesian teleoperation."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CartesianSettings:
    calibrated: bool
    leader_zero_joints_deg: tuple[float, ...]
    follower_zero_pose: tuple[float, ...]
    axis_map_rpy_deg: tuple[float, ...]
    translation_scale: float
    max_translation_mm: float
    max_rotation_deg: float
    orientation_axis_signs: tuple[float, ...] = (-1.0, -1.0, 1.0)


def _values(text: str, expected: int) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != expected:
        raise ValueError(f"expected {expected} comma-separated values")
    return values


def load_cartesian_settings(path: str | Path) -> CartesianSettings:
    parser = configparser.ConfigParser()
    if not parser.read(Path(path).expanduser()):
        raise FileNotFoundError(path)
    section = parser["CARTESIAN"]
    settings = CartesianSettings(
        calibrated=section.getboolean("calibrated"),
        leader_zero_joints_deg=_values(section["leader_zero_joints_deg"], 6),
        follower_zero_pose=_values(section["follower_zero_pose"], 6),
        axis_map_rpy_deg=_values(section["axis_map_rpy_deg"], 3),
        translation_scale=section.getfloat("translation_scale"),
        max_translation_mm=section.getfloat("max_translation_mm"),
        max_rotation_deg=section.getfloat("max_rotation_deg"),
        orientation_axis_signs=_values(
            section.get("orientation_axis_signs", "-1, -1, 1"), 3
        ),
    )
    if settings.translation_scale <= 0:
        raise ValueError("translation_scale must be positive")
    if any(abs(value) != 1 for value in settings.orientation_axis_signs):
        raise ValueError("orientation_axis_signs values must be +1 or -1")
    return settings


def dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [[ct, -st * ca, st * sa, a * ct],
         [st, ct * ca, -ct * sa, a * st],
         [0.0, sa, ca, d],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


def nova2_fk(joints_rad: np.ndarray) -> np.ndarray:
    """Nova 2 flange pose; translation is returned in millimetres."""
    q = np.asarray(joints_rad, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("FK requires six finite joint values")
    params = (
        (q[0], 0.2234, 0.0, np.pi / 2),
        (q[1] - np.pi / 2, 0.0, -0.280, 0.0),
        (q[2], 0.0, -0.225, 0.0),
        (q[3] - np.pi / 2, 0.1175, 0.0, np.pi / 2),
        (q[4], 0.120, 0.0, -np.pi / 2),
        (q[5], 0.088, 0.0, 0.0),
    )
    transform = np.eye(4)
    for item in params:
        transform = transform @ dh(*item)
    transform[:3, 3] *= 1000.0
    return transform


def rpy_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]],
        dtype=float,
    )


def matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw])


def pose6_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    transform = np.eye(4)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = rpy_matrix(np.deg2rad(pose[3:]))
    return transform


def transform_pose6(transform: np.ndarray) -> np.ndarray:
    return np.concatenate((transform[:3, 3], np.rad2deg(matrix_rpy(transform[:3, :3]))))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector for a 3x3 rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    angle = np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-10:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Near pi, the skew part is numerically singular.  Recover the axis
        # from the symmetric part and choose signs from the off-diagonals.
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0))
        largest = int(np.argmax(axis))
        for index in range(3):
            if index != largest and axis[largest] > 1e-10:
                axis[index] = (matrix[largest, index] + matrix[index, largest]) / (
                    4.0 * axis[largest]
                )
        axis /= np.linalg.norm(axis)
    else:
        axis = np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        ) / (2.0 * np.sin(angle))
    return axis * angle


def rotation_matrix(vector: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix from an axis-angle vector."""
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle < 1e-10:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def map_orientation_delta(rotation_delta: np.ndarray, signs: tuple[float, ...]) -> np.ndarray:
    """Map calibrated leader-local rotation components into follower-local axes."""
    mapped_vector = np.asarray(signs, dtype=float) * rotation_vector(rotation_delta)
    return rotation_matrix(mapped_vector)


def map_leader_pose(joints_rad: np.ndarray, settings: CartesianSettings) -> np.ndarray:
    """Map leader motion about its calibrated zero into the follower base."""
    leader_zero = nova2_fk(np.deg2rad(settings.leader_zero_joints_deg))
    leader_now = nova2_fk(joints_rad)
    axis = rpy_matrix(np.deg2rad(settings.axis_map_rpy_deg))
    follower_zero = pose6_transform(np.asarray(settings.follower_zero_pose))

    translation_delta = leader_now[:3, 3] - leader_zero[:3, 3]
    rotation_delta = leader_zero[:3, :3].T @ leader_now[:3, :3]
    target = follower_zero.copy()
    target[:3, 3] += settings.translation_scale * axis @ translation_delta
    mapped_rotation = map_orientation_delta(rotation_delta, settings.orientation_axis_signs)
    target[:3, :3] = follower_zero[:3, :3] @ mapped_rotation
    return transform_pose6(target)
