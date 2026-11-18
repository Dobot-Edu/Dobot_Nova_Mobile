from types import SimpleNamespace
import unittest

from Dobot_Init.chassis_rpc import Velocity
from Dobot_Init.chassis import VRChassis


class ChassisDatasetTelemetryTests(unittest.TestCase):
    def test_snapshot_contains_command_feedback_pose_and_timestamp(self):
        chassis = VRChassis(SimpleNamespace(enabled=True), execute=True)
        with chassis._lock:
            chassis._last_sent = Velocity(-50, 10000)
        chassis._store_feedback(
            {
                "Speed": -48,
                "AngularVelocity": 9800,
                "Position": [1234, -5678],
                "Oritation": 90000,
                "Odom": 3210,
            }
        )
        snapshot = chassis.dataset_snapshot()
        self.assertEqual(snapshot["chassis_command_linear_mm_s"], -50)
        self.assertEqual(snapshot["chassis_feedback_speed_mm_s"], -48.0)
        self.assertEqual(snapshot["chassis_feedback_position_mm"], [1234.0, -5678.0])
        self.assertGreater(snapshot["chassis_feedback_monotonic_ns"], 0)
        self.assertTrue(snapshot["chassis_feedback_valid"])

    def test_snapshot_is_invalid_before_first_rpc_feedback(self):
        chassis = VRChassis(SimpleNamespace(enabled=True), execute=True)
        snapshot = chassis.dataset_snapshot()
        self.assertFalse(snapshot["chassis_feedback_valid"])
        self.assertEqual(snapshot["chassis_feedback_monotonic_ns"], 0)


if __name__ == "__main__":
    unittest.main()
