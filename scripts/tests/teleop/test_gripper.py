import unittest

from Dobot_Init.config import GripperConfig
from Dobot_Init.gripper import (
    GripperFeedback,
    GripperStatusError,
    NON_FATAL_STATUS_ERRORS,
    GripperTransportError,
    TriggerGripper,
    describe_servo_status,
    feedback_position,
    position_command,
    raw_position,
    trigger_command,
)


class FeedbackGripper(TriggerGripper):
    def __init__(self, config):
        super().__init__(config, execute=True)
        self._serial = object()
        self.feedback = []
        self.moves = []
        self.overload_recoveries = 0

    def _read_feedback_if_due(self):
        return self.feedback.pop(0) if self.feedback else None

    def _move(self, command, *, torque=None):
        self.moves.append((command, torque))

    def _recover_hardware_overload(self):
        self.overload_recoveries += 1
        self._hardware_overload_latched = False


class TimeoutFeedbackGripper(FeedbackGripper):
    def __init__(self, config):
        super().__init__(config)
        self.feedback_reads = 0

    def _read_feedback_if_due(self):
        self.feedback_reads += 1
        raise GripperTransportError("夹爪串口响应超时")


class Status32CommandGripper(FeedbackGripper):
    def _move(self, command, *, torque=None):
        self.moves.append((command, torque))
        raise GripperStatusError(32)


class GripperTests(unittest.TestCase):
    def setUp(self):
        self.config = GripperConfig(
            enabled=True,
            port="/dev/xtrainer_gripper",
            servo_id=22,
            raw_open=2050,
            raw_close=1000,
            max_close_command=179,
            feedback_raw_open=1990,
            feedback_raw_close=1338,
            speed=15,
            target_torque=80,
            command_deadband=2,
            trigger_press_threshold=0.20,
            trigger_release_threshold=0.10,
            close_rate_per_s=0.60,
            open_rate_per_s=0.80,
            stop_on_overload_32=True,
            force_feedback_enabled=True,
            current_limit_ma=300.0,
            current_trip_samples=2,
            feedback_interval_s=0.05,
            feedback_valid_age_s=0.15,
            feedback_abort_age_s=0.50,
            hold_torque=30,
        )

    def test_trigger_is_continuous_not_binary(self):
        self.assertEqual(trigger_command(0.0), 0)
        self.assertEqual(trigger_command(0.5), 128)
        self.assertEqual(trigger_command(1.0), 255)

    def test_released_is_open_and_pressed_is_closed(self):
        self.assertEqual(raw_position(trigger_command(0.0), self.config), 2050)
        self.assertEqual(raw_position(trigger_command(1.0), self.config), 1000)

    def test_first_valid_packet_only_sets_safe_open_baseline(self):
        gripper = TriggerGripper(self.config, execute=False)
        gripper.update(1.0, input_valid=True)
        self.assertEqual(gripper.last_command, 0)

    def test_hold_closes_gradually_and_release_opens(self):
        gripper = FeedbackGripper(self.config)
        gripper.update(0.0, input_valid=True)
        gripper._last_update_at -= 0.5
        gripper.update(1.0, input_valid=True)
        closed_command = gripper.last_command
        self.assertGreater(closed_command, 0)
        self.assertLess(closed_command, self.config.max_close_command)
        gripper._last_update_at -= 0.5
        gripper.update(0.0, input_valid=True)
        self.assertEqual(gripper.last_command, 0)
        self.assertEqual(gripper.moves[-1], (0, None))

    def test_toggle_press_switches_full_close_and_open_once_per_click(self):
        config = GripperConfig(**{**self.config.__dict__, "mode": "toggle"})
        gripper = FeedbackGripper(config)
        gripper.update(0.0, input_valid=True)
        gripper.update(1.0, input_valid=True)
        self.assertEqual(gripper.last_command, config.max_close_command)
        move_count = len(gripper.moves)
        gripper.update(1.0, input_valid=True)
        self.assertEqual(len(gripper.moves), move_count)
        gripper.update(0.0, input_valid=True)
        gripper.update(1.0, input_valid=True)
        self.assertEqual(gripper.last_command, 0)

    def test_toggle_requires_release_after_invalid_input(self):
        config = GripperConfig(**{**self.config.__dict__, "mode": "toggle"})
        gripper = FeedbackGripper(config)
        gripper.update(0.0, input_valid=True)
        gripper.update(1.0, input_valid=False)
        gripper.update(1.0, input_valid=True)
        self.assertEqual(gripper.last_command, 0)
        gripper.update(0.0, input_valid=True)
        gripper.update(1.0, input_valid=True)
        self.assertEqual(gripper.last_command, config.max_close_command)

    def test_error_12_is_explained_as_temperature_and_current(self):
        self.assertEqual(describe_servo_status(12), "过温保护 + 过流保护")

    def test_invalid_tracking_does_not_change_command(self):
        gripper = TriggerGripper(self.config, execute=False)
        gripper.update(0.25, input_valid=False)
        self.assertIsNone(gripper.last_command)

    def test_position_feedback_is_inverted_to_command(self):
        self.assertEqual(position_command(2050, self.config), 0)
        self.assertEqual(position_command(1000, self.config), 255)
        self.assertAlmostEqual(position_command(1525, self.config), 128, delta=1)

    def test_consecutive_current_samples_stop_closing_at_actual_position(self):
        gripper = FeedbackGripper(self.config)
        gripper.update(0.0, input_valid=True)

        high_current = GripperFeedback(
            position_raw=1500,
            load_raw=0,
            temperature_c=40,
            moving=True,
            current_ma=350.0,
        )
        gripper.feedback.extend([high_current, high_current, high_current])
        gripper.set_target_position(0.0)
        gripper.set_target_position(0.0)
        gripper.set_target_position(0.0)

        hold_command = position_command(1500, self.config)
        self.assertEqual(gripper.last_command, hold_command)
        self.assertEqual(gripper.moves[-1], (hold_command, self.config.hold_torque))
        move_count = len(gripper.moves)
        gripper.update(1.0, input_valid=True)
        self.assertEqual(len(gripper.moves), move_count)

        gripper.set_target_position(1.0)
        self.assertEqual(gripper.moves[-1], (0, None))

    def test_disabled_software_current_limit_never_latches(self):
        config = GripperConfig(**{
            **self.config.__dict__,
            "force_feedback_enabled": False,
        })
        gripper = FeedbackGripper(config)
        gripper.update(0.0, input_valid=True)
        high_current = GripperFeedback(1500, 0, 40, True, 1000.0)
        gripper.feedback.extend([high_current, high_current, high_current])

        gripper.set_target_position(0.0)
        gripper.set_target_position(0.0)
        gripper.set_target_position(0.0)

        self.assertFalse(gripper._force_latched)
        self.assertEqual(gripper.last_command, config.max_close_command)

    def test_pure_hardware_overload_stops_deeper_closing_without_failing(self):
        config = GripperConfig(**{
            **self.config.__dict__,
            "force_feedback_enabled": False,
        })
        gripper = FeedbackGripper(config)
        gripper._last_command = 80
        gripper._closing_active = True
        overload = GripperFeedback(1500, 20, 40, False, 500.0, status_error=32)
        gripper.feedback.append(overload)

        gripper.set_target_position(0.0)

        hold_command = min(position_command(1500, config), config.max_close_command)
        self.assertEqual(gripper.last_command, hold_command)
        self.assertTrue(gripper._hardware_overload_latched)
        self.assertTrue(gripper._force_latched)
        self.assertEqual(gripper.moves, [])

        gripper.set_target_position(0.0)
        self.assertEqual(gripper.moves, [])

    def test_overload_32_switch_off_keeps_closing(self):
        config = GripperConfig(**{
            **self.config.__dict__,
            "stop_on_overload_32": False,
            "force_feedback_enabled": False,
        })
        gripper = FeedbackGripper(config)
        gripper._last_command = 80
        gripper._closing_active = True
        gripper.feedback.append(
            GripperFeedback(1500, 20, 40, False, 500.0, status_error=32)
        )

        gripper.set_target_position(0.0)

        self.assertFalse(gripper._hardware_overload_latched)
        self.assertFalse(gripper._force_latched)
        self.assertEqual(gripper.last_command, config.max_close_command)
        self.assertEqual(gripper.moves[-1], (config.max_close_command, None))

    def test_overload_32_switch_off_does_not_latch_command_error(self):
        config = GripperConfig(**{
            **self.config.__dict__,
            "stop_on_overload_32": False,
            "force_feedback_enabled": False,
        })
        gripper = Status32CommandGripper(config)
        gripper._last_command = 80

        gripper.set_target_position(1.0)
        gripper.set_target_position(0.0)

        self.assertFalse(gripper._hardware_overload_latched)
        self.assertFalse(gripper._force_latched)
        self.assertEqual(len(gripper.moves), 2)

    def test_opening_recovers_pure_hardware_overload(self):
        gripper = FeedbackGripper(self.config)
        gripper._last_command = 120
        gripper._force_latched = True
        gripper._hardware_overload_latched = True

        gripper.set_target_position(1.0)

        self.assertEqual(gripper.overload_recoveries, 1)
        self.assertFalse(gripper._hardware_overload_latched)
        self.assertFalse(gripper._force_latched)
        self.assertEqual(gripper.moves[-1], (0, None))

    def test_position_command_allows_overload_temperature_and_current_status(self):
        gripper = TriggerGripper(self.config, execute=True)
        writes = []
        gripper._write = lambda address, payload, allowed_errors=(0,): writes.append(
            (address, allowed_errors)
        )

        gripper._move(100)

        self.assertEqual(writes, [(41, NON_FATAL_STATUS_ERRORS)])
        combined = GripperStatusError(40)
        self.assertEqual(combined.status, 40)
        self.assertIn("过流保护 + 过载/堵转保护", str(combined))

    def test_encoder_snapshot_is_normalized_to_safe_travel(self):
        gripper = TriggerGripper(self.config, execute=False)
        gripper._store_feedback(
            GripperFeedback(1500, 3, 40, False, 120.0)
        )
        snapshot = gripper.feedback_snapshot()
        expected = feedback_position(1500, self.config)
        self.assertAlmostEqual(snapshot["position"], expected)
        self.assertEqual(snapshot["position_raw"], 1500)

    def test_direct_calibrated_endpoints_cover_exact_zero_to_one(self):
        config = GripperConfig(**{
            **self.config.__dict__,
            "raw_open": 1980,
            "raw_close": 1313,
            "max_close_command": 255,
            "feedback_raw_open": 1980,
            "feedback_raw_close": 1313,
        })
        self.assertEqual(position_command(1980, config), 0)
        self.assertEqual(position_command(1313, config), 255)
        self.assertEqual(raw_position(0, config), 1980)
        self.assertEqual(raw_position(255, config), 1313)
        self.assertEqual(feedback_position(1980, config), 1.0)
        self.assertEqual(feedback_position(1313, config), 0.0)


if __name__ == "__main__":
    unittest.main()
