from __future__ import annotations

import unittest

import numpy as np


from Dobot_Inference.control import (
    ACTION_NAMES,
    STATE_NAMES,
    ActionLimits,
    blend_actions,
    build_policy_state,
    next_joint_command,
    sanitize_action,
)


class ControlTest(unittest.TestCase):
    def test_schema_dimensions(self) -> None:
        self.assertEqual(len(STATE_NAMES), 9)
        self.assertEqual(len(ACTION_NAMES), 9)
        self.assertNotIn("lift_height_m", STATE_NAMES)
        self.assertNotIn("lift_target_m", ACTION_NAMES)

    def test_policy_state_uses_si_units(self) -> None:
        state = build_policy_state(np.arange(6) / 10, 0.25, 0.1, -0.2)
        self.assertEqual(state.shape, (9,))
        np.testing.assert_allclose(state[-3:], [0.25, 0.1, -0.2])

    def test_joint_action_is_not_clipped_but_gripper_and_base_are(self) -> None:
        limits = ActionLimits(np.full(9, -1.0), np.full(9, 1.0))
        action = sanitize_action(
            [2, 0, 0, 0, 0, 0, 2, 0.8, -0.8],
            limits,
            base_output_scale=0.25,
            base_linear_limit_m_s=0.1,
            base_angular_limit_rad_s=0.1,
        )
        self.assertEqual(action[0], 2.0)
        self.assertEqual(action[6], 1.0)
        self.assertAlmostEqual(action[7], 0.1)
        self.assertAlmostEqual(action[8], -0.1)

    def test_blend_includes_continuous_gripper_position(self) -> None:
        old = np.zeros(9)
        new = np.ones(9)
        self.assertAlmostEqual(blend_actions(old, new, 0.25)[6], 0.15625)
        self.assertAlmostEqual(blend_actions(old, new, 0.75)[6], 0.84375)

    def test_joint_command_integrates_from_previous_command(self) -> None:
        target = np.ones(6)
        actual = np.full(6, 0.5)
        previous = np.zeros(6)
        command, velocity = next_joint_command(
            target,
            actual,
            previous,
            np.zeros(6),
            servo_hz=50,
            max_speed_rad_s=1.0,
            max_acceleration_rad_s2=2.0,
            max_target_jump_rad=2.0,
            enforce_target_jump=True,
        )
        np.testing.assert_allclose(velocity, np.full(6, 0.04))
        np.testing.assert_allclose(command, np.full(6, 0.0008))


if __name__ == "__main__":
    unittest.main()
