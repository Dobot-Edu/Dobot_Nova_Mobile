import unittest

import numpy as np

from Dobot_Init.config import MappingConfig
from quest_vr.mapping import MappingError, RelativePoseMapper
from quest_vr.protocol import VRPacket


POSITION_AXIS = np.array(
    [
        [-0.99724721, -0.00947366, -0.07354081],
        [0.07161021, 0.13423096, -0.98835926],
        [-0.01923483, 0.99090479, 0.13318304],
    ],
    dtype=float,
)


def packet(
    position=(0, 0, 0),
    quaternion=(0, 0, 0, 1),
    head_quaternion=(0, 0, 0, 1),
    button_a=False,
    button_b=False,
):
    return VRPacket(
        position_m=np.asarray(position, dtype=float),
        quaternion_xyzw=np.asarray(quaternion, dtype=float),
        grip=0.8,
        trigger=0.0,
        button_a=button_a,
        button_b=button_b,
        thumbstick_xy=np.zeros(2),
        tracking_valid=True,
        timestamp_s=0.0,
        head_position_m=np.zeros(3),
        head_quaternion_xyzw=np.asarray(head_quaternion, dtype=float),
        head_tracking_valid=True,
        protocol_version=2,
    )


def axis_angle_quaternion(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = np.deg2rad(angle_deg) / 2.0
    return np.concatenate((axis * np.sin(half), [np.cos(half)]))


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.config = MappingConfig(
            position_axis=POSITION_AXIS,
            position_scale=0.9,
            orientation_axis=np.diag([-1.0, -1.0, 1.0]),
            max_translation_mm=250,
            max_rotation_deg=90,
            max_packet_translation_mm=50,
            max_packet_rotation_deg=25,
        )
        self.mapper = RelativePoseMapper(self.config)
        self.robot_pose = np.array([10, 20, 30, 0, 0, 0], dtype=float)

    def test_first_grip_packet_only_anchors(self):
        np.testing.assert_allclose(self.mapper.map(packet(), self.robot_pose), self.robot_pose)

    def test_openxr_position_axis_mapping(self):
        self.mapper.map(packet(), self.robot_pose)
        target = self.mapper.map(packet((0.01, 0.02, -0.03)), self.robot_pose)
        np.testing.assert_allclose(
            target[:3], [2.83985110, 49.74634919, 44.06723067], atol=1e-8
        )

    def test_measured_physical_translation_directions(self):
        samples = (
            (np.array([1.0, 0.0, 0.0]), np.array([-202.895, -4.559, -0.223])),
            (np.array([0.0, 0.0, 1.0]), np.array([-25.434, -146.606, 23.953])),
            (np.array([0.0, 1.0, 0.0]), np.array([-1.846, 11.607, 108.475])),
        )
        for measured, expected_axis in samples:
            expected_axis = expected_axis / np.linalg.norm(expected_axis)
            mapped = POSITION_AXIS @ measured
            cosine = float(mapped @ expected_axis / np.linalg.norm(mapped))
            self.assertGreater(cosine, 0.99)

    def test_translation_is_invariant_to_quest_world_yaw(self):
        base = RelativePoseMapper(self.config)
        base.map(packet(), self.robot_pose)
        base_target = base.map(packet(position=(0.0, 0.0, 0.01)), self.robot_pose)

        yaw90 = axis_angle_quaternion([0, 1, 0], 90)
        rotated = RelativePoseMapper(self.config)
        rotated.map(
            packet(position=(1.0, 0.0, 2.0), head_quaternion=yaw90),
            self.robot_pose,
        )
        # 同一个现实“向前 10 mm”，在旋转90°的Quest世界坐标中表现为 +X。
        rotated_target = rotated.map(
            packet(position=(1.01, 0.0, 2.0), head_quaternion=yaw90),
            self.robot_pose,
        )
        np.testing.assert_allclose(rotated_target, base_target, atol=1e-8)

    def test_old_apk_without_head_pose_is_rejected(self):
        old = packet()
        old = VRPacket(
            position_m=old.position_m,
            quaternion_xyzw=old.quaternion_xyzw,
            grip=old.grip,
            trigger=old.trigger,
            button_a=old.button_a,
            button_b=old.button_b,
            thumbstick_xy=old.thumbstick_xy,
            tracking_valid=True,
            timestamp_s=0.0,
        )
        with self.assertRaisesRegex(MappingError, "协议V2"):
            self.mapper.map(old, self.robot_pose)

    def test_a_does_not_recenter_mapping(self):
        self.mapper.map(packet(), self.robot_pose)
        target = self.mapper.map(packet((0.01, 0, 0), button_a=True), self.robot_pose)
        self.assertGreater(np.linalg.norm(target[:3] - self.robot_pose[:3]), 1.0)

    def test_a_plus_b_does_not_recenter_arm_mapping(self):
        self.mapper.map(packet(), self.robot_pose)
        target = self.mapper.map(packet((0.01, 0, 0), button_a=True, button_b=True), self.robot_pose)
        self.assertGreater(np.linalg.norm(target[:3] - self.robot_pose[:3]), 1.0)

    def test_rejects_single_frame_jump(self):
        self.mapper.map(packet(), self.robot_pose)
        with self.assertRaisesRegex(MappingError, "跳变"):
            self.mapper.map(packet((0.2, 0, 0)), self.robot_pose)

    def test_openxr_rotation_x_maps_to_nova_rx(self):
        self.mapper.map(packet(), self.robot_pose)
        target = self.mapper.map(
            packet(quaternion=axis_angle_quaternion([1, 0, 0], 10)), self.robot_pose
        )
        np.testing.assert_allclose(target[3:], [-10, 0, 0], atol=1e-8)

    def test_openxr_rotation_y_maps_to_nova_ry(self):
        self.mapper.map(packet(), self.robot_pose)
        target = self.mapper.map(
            packet(quaternion=axis_angle_quaternion([0, 1, 0], 10)), self.robot_pose
        )
        np.testing.assert_allclose(target[3:], [0, -10, 0], atol=1e-8)

    def test_openxr_rotation_z_maps_to_nova_rz(self):
        self.mapper.map(packet(), self.robot_pose)
        target = self.mapper.map(
            packet(quaternion=axis_angle_quaternion([0, 0, 1], 10)), self.robot_pose
        )
        np.testing.assert_allclose(target[3:], [0, 0, 10], atol=1e-8)

    def test_combined_packet_maps_all_six_cartesian_dimensions(self):
        self.mapper.map(packet(), self.robot_pose)
        target = self.mapper.map(
            packet(
                position=(0.01, 0.02, -0.03),
                quaternion=axis_angle_quaternion([1, 0, 0], 10),
            ),
            self.robot_pose,
        )
        np.testing.assert_allclose(
            target,
            [2.83985110, 49.74634919, 44.06723067, -10, 0, 0],
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
