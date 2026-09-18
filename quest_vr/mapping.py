"""把 Quest 右手柄的相对位姿映射为 Nova TCP 相对位姿。"""

from __future__ import annotations

import numpy as np

from Dobot_Init.config import MappingConfig
from quest_vr.kinematics import matrix_rpy, pose6_transform, transform_pose6
from quest_vr.protocol import VRPacket


class MappingError(RuntimeError):
    """VR 位姿不连续或超过配置工作空间。"""


def quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float)
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=float,
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def quaternion_rpy_deg(quaternion: np.ndarray) -> np.ndarray:
    """Convert Quest xyzw quaternion to display-only roll, pitch and yaw degrees."""
    return np.rad2deg(matrix_rpy(quaternion_matrix_xyzw(quaternion)))


def head_yaw_frame(quaternion: np.ndarray) -> np.ndarray:
    """构造头显水平操作者坐标：列向量依次为现实右、上、前。"""
    rotation = quaternion_matrix_xyzw(quaternion)
    forward = rotation[:, 2].copy()
    forward[1] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        raise MappingError("头显朝向接近竖直，无法建立水平操作者坐标")
    forward /= norm
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    return np.column_stack((right, up, forward))


def head_yaw_deg(quaternion: np.ndarray) -> float:
    """返回Unity/OpenXR头显绕竖直轴的水平朝向角，仅用于诊断显示。"""
    forward = head_yaw_frame(quaternion)[:, 2]
    return float(np.rad2deg(np.arctan2(forward[0], forward[2])))


class RelativePoseMapper:
    """Grip 每次由松到按时，重新建立手柄和机器人相对零点。"""

    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self.release()

    def release(self) -> None:
        self._hand_position0: np.ndarray | None = None
        self._hand_rotation0: np.ndarray | None = None
        self._robot_transform0: np.ndarray | None = None
        self._previous_position: np.ndarray | None = None
        self._previous_rotation: np.ndarray | None = None
        self._operator_frame0: np.ndarray | None = None

    def anchor(self, packet: VRPacket, robot_pose: np.ndarray) -> None:
        if not packet.head_tracking_valid or packet.head_quaternion_xyzw is None:
            raise MappingError("Quest APK 未提供有效头显位姿，请安装协议V2版本")
        rotation = quaternion_matrix_xyzw(packet.quaternion_xyzw)
        self._hand_position0 = packet.position_m.copy()
        self._hand_rotation0 = rotation
        self._robot_transform0 = pose6_transform(np.asarray(robot_pose, dtype=float))
        self._previous_position = packet.position_m.copy()
        self._previous_rotation = rotation
        self._operator_frame0 = head_yaw_frame(packet.head_quaternion_xyzw)

    def map(self, packet: VRPacket, robot_pose: np.ndarray) -> np.ndarray:
        # 仅 Grip 重新激活时建立零点；A/B 由上层状态机独占。
        if self._hand_position0 is None:
            self.anchor(packet, robot_pose)
            return np.asarray(robot_pose, dtype=float).copy()

        rotation = quaternion_matrix_xyzw(packet.quaternion_xyzw)
        assert self._previous_position is not None
        assert self._previous_rotation is not None
        step_mm = 1000.0 * float(np.linalg.norm(packet.position_m - self._previous_position))
        step_deg = rotation_angle_deg(self._previous_rotation.T @ rotation)
        if step_mm > self.config.max_packet_translation_mm:
            raise MappingError(f"VR 单帧位移跳变 {step_mm:.1f} mm")
        if step_deg > self.config.max_packet_rotation_deg:
            raise MappingError(f"VR 单帧姿态跳变 {step_deg:.1f}°")
        self._previous_position = packet.position_m.copy()
        self._previous_rotation = rotation

        assert self._hand_rotation0 is not None
        assert self._robot_transform0 is not None
        assert self._operator_frame0 is not None
        # 先将 Quest 世界坐标位移旋回到按下 Grip 时的头显水平坐标。
        # 若追踪世界原点在两次启动之间发生任意偏航旋转，该旋转会在此抵消。
        operator_delta = self._operator_frame0.T @ (
            packet.position_m - self._hand_position0
        )
        translation = (
            1000.0
            * self.config.position_scale
            * (self.config.position_axis @ operator_delta)
        )
        delta_rotation = self._hand_rotation0.T @ rotation
        if np.linalg.norm(translation) > self.config.max_translation_mm:
            raise MappingError("VR 位移超过工作空间限制")
        if rotation_angle_deg(delta_rotation) > self.config.max_rotation_deg:
            raise MappingError("VR 姿态超过工作空间限制")

        target = self._robot_transform0.copy()
        target[:3, 3] += translation
        axis = self.config.orientation_axis
        target[:3, :3] = self._robot_transform0[:3, :3] @ axis @ delta_rotation @ axis.T
        return transform_pose6(target)
