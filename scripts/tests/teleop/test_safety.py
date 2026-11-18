import unittest

import numpy as np

from quest_vr.kinematics import pose6_transform
from quest_vr.mapping import rotation_angle_deg
from Dobot_Init.safety import SafetyError, bounded_cartesian_target, bounded_joint_command


class SafetyTests(unittest.TestCase):
    def test_cartesian_target_limits_translation_and_rotation_step(self):
        current = np.zeros(6)
        requested = np.array([30, 40, 0, 0, 0, 30], dtype=float)
        result = bounded_cartesian_target(requested, current, 5.0, 2.0)
        self.assertAlmostEqual(np.linalg.norm(result[:3]), 5.0, places=8)
        rotation = pose6_transform(result)[:3, :3]
        self.assertAlmostEqual(rotation_angle_deg(rotation), 2.0, places=8)

    def test_cartesian_target_keeps_small_six_dimensional_request(self):
        requested = np.array([1, 2, 3, 1, 2, 3], dtype=float)
        result = bounded_cartesian_target(requested, np.zeros(6), 10.0, 10.0)
        np.testing.assert_allclose(result, requested, atol=1e-8)

    def test_rate_limits_each_joint(self):
        result = bounded_joint_command(np.ones(6), np.zeros(6), 0.1)
        np.testing.assert_allclose(result, np.full(6, 0.1))

    def test_does_not_limit_absolute_or_startup_relative_angle(self):
        result = bounded_joint_command(np.full(6, 10.0), np.zeros(6), 20.0)
        np.testing.assert_allclose(result, np.full(6, 10.0))

    def test_rejects_non_finite_joint_target(self):
        requested = np.zeros(6)
        requested[2] = np.nan
        with self.assertRaisesRegex(SafetyError, "非有限"):
            bounded_joint_command(requested, np.zeros(6), 0.1)


if __name__ == "__main__":
    unittest.main()
