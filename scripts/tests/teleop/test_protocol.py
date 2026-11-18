import json
import unittest

import numpy as np

from quest_vr.protocol import decode_packet


class ProtocolTests(unittest.TestCase):
    def packet(self, **overrides):
        message = {
            "timestamp": 1.0,
            "tracking_valid": True,
            "right_pos": [0.1, 1.2, -0.3],
            "right_rot": [0, 0, 0, 1],
            "protocol_version": 2,
            "head_tracking_valid": True,
            "head_pos": [0.0, 1.6, 0.0],
            "head_rot": [0, 0, 0, 1],
            "right_grip": 0.8,
            "right_trigger": 0.4,
            "button_a": False,
            "button_b": False,
            "right_thumbstick": [0.25, -0.5],
        }
        message.update(overrides)
        return decode_packet(json.dumps(message).encode())

    def test_decodes_quest_right_hand(self):
        packet = self.packet()
        np.testing.assert_allclose(packet.position_m, [0.1, 1.2, -0.3])
        self.assertTrue(packet.deadman(0.65))
        self.assertAlmostEqual(packet.trigger, 0.4)
        np.testing.assert_allclose(packet.thumbstick_xy, [0.25, -0.5])
        np.testing.assert_allclose(packet.head_position_m, [0.0, 1.6, 0.0])
        np.testing.assert_allclose(packet.head_quaternion_xyzw, [0, 0, 0, 1])
        self.assertTrue(packet.head_tracking_valid)
        self.assertEqual(packet.protocol_version, 2)

    def test_normalizes_quaternion(self):
        packet = self.packet(right_rot=[0, 0, 0, 0.8])
        np.testing.assert_allclose(packet.quaternion_xyzw, [0, 0, 0, 1])

    def test_tracking_invalid_disables_deadman(self):
        packet = self.packet(tracking_valid=False, right_grip=1.0)
        self.assertFalse(packet.deadman(0.65))

    def test_rejects_non_finite_position(self):
        with self.assertRaisesRegex(ValueError, "right_pos"):
            self.packet(right_pos=[0, float("nan"), 0])

    def test_old_apk_without_thumbstick_safely_defaults_to_zero(self):
        packet = self.packet()
        message = {
            "timestamp": packet.timestamp_s,
            "tracking_valid": True,
            "right_pos": packet.position_m.tolist(),
            "right_rot": packet.quaternion_xyzw.tolist(),
        }
        decoded = decode_packet(json.dumps(message).encode())
        np.testing.assert_allclose(decoded.thumbstick_xy, [0.0, 0.0])

    def test_v1_packet_without_head_pose_decodes_but_has_no_operator_frame(self):
        packet = self.packet()
        message = {
            "timestamp": packet.timestamp_s,
            "tracking_valid": True,
            "right_pos": packet.position_m.tolist(),
            "right_rot": packet.quaternion_xyzw.tolist(),
        }
        decoded = decode_packet(json.dumps(message).encode())
        self.assertEqual(decoded.protocol_version, 1)
        self.assertFalse(decoded.head_tracking_valid)
        self.assertIsNone(decoded.head_quaternion_xyzw)

    def test_rejects_invalid_thumbstick(self):
        with self.assertRaisesRegex(ValueError, "right_thumbstick"):
            self.packet(right_thumbstick=[2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
