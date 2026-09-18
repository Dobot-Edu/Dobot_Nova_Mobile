import unittest
from types import SimpleNamespace

import numpy as np

from Dobot_Init.chassis_rpc import RobotRpc, RpcError

from Dobot_Init.chassis import (
    VRChassis,
    chassis_input_enabled,
    connection_safety_problem,
    joystick_velocity,
    manual_safety_problem,
)


class FakeRpc:
    def __init__(self):
        self.calls = []

    def stop(self):
        self.calls.append("stop")

    def logout(self):
        self.calls.append("logout")

    def set_work_mode(self, mode):
        self.calls.append(("set_work_mode", mode))


class RecoveringRpc(FakeRpc):
    def reconnect(self):
        self.calls.append("reconnect")

    def get_state(self):
        self.calls.append("get_state")
        return {
            "State": "Idle",
            "Speed": 0,
            "WorkMode": 3,
            "ExtraStateFirst": 1 << 20,
            "ExtraStateSecond": 0,
        }


class ChassisJoystickTests(unittest.TestCase):
    def command(self, x: float, y: float):
        return joystick_velocity(np.array([x, y]), 0.2, 50, 10000)

    def test_center_and_deadzone_stop(self):
        self.assertEqual(self.command(0.0, 0.0).linear_mm_s, 0)
        self.assertEqual(self.command(0.15, 0.0).angular_mdeg_s, 0)

    def test_forward_and_backward(self):
        self.assertEqual(self.command(0.0, 1.0).linear_mm_s, 50)
        self.assertEqual(self.command(0.0, -1.0).linear_mm_s, -50)

    def test_right_and_left_turn(self):
        self.assertEqual(self.command(1.0, 0.0).angular_mdeg_s, -10000)
        self.assertEqual(self.command(-1.0, 0.0).angular_mdeg_s, 10000)

    def test_deadzone_edge_is_continuous(self):
        inside = self.command(0.2, 0.0)
        just_outside = self.command(0.21, 0.0)
        self.assertEqual(inside.angular_mdeg_s, 0)
        self.assertLess(abs(just_outside.angular_mdeg_s), 200)

    def test_small_cross_axis_noise_does_not_create_unwanted_motion(self):
        right = self.command(1.0, -0.18)
        forward = self.command(-0.14, 1.0)
        self.assertEqual(right.linear_mm_s, 0)
        self.assertEqual(forward.angular_mdeg_s, 0)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            joystick_velocity(np.array([np.nan, 0.0]), 0.2, 50, 10000)

    def test_current_config_does_not_require_grip(self):
        self.assertTrue(
            chassis_input_enabled(packet_valid=True, grip_deadman=False, require_grip=False)
        )
        self.assertFalse(
            chassis_input_enabled(packet_valid=False, grip_deadman=False, require_grip=False)
        )

    def test_connection_allows_only_non_task_bit_before_manual_transition(self):
        state = {
            "State": "Idle",
            "Speed": 0,
            "WorkMode": 4,
            "ExtraStateFirst": 1 << 20,
            "ExtraStateSecond": 0,
        }
        self.assertIsNone(connection_safety_problem(state))
        state["ExtraStateFirst"] |= 1 << 3
        self.assertIn("底盘异常", connection_safety_problem(state))

    def test_manual_direct_velocity_can_tolerate_only_localization_warning(self):
        state = {
            "State": "Idle",
            "Speed": 0,
            "WorkMode": 3,
            "ExtraStateFirst": (1 << 20) | (1 << 31),
            "ExtraStateSecond": 0,
        }
        self.assertIsNone(manual_safety_problem(state, True))
        self.assertIsNotNone(manual_safety_problem(state, False))
        state["ExtraStateFirst"] |= 1 << 5
        self.assertIn("底盘异常", manual_safety_problem(state, True))

    def test_close_stops_but_keeps_manual_mode(self):
        chassis = VRChassis(SimpleNamespace(enabled=True), execute=True)
        rpc = FakeRpc()
        chassis.rpc = rpc
        chassis._remote_active = True
        chassis._mode_requested = True
        chassis.close()
        self.assertEqual(chassis.rpc, None)
        # 退出时只停车和注销，不切回原模式。
        self.assertEqual(rpc.calls, ["stop", "logout"])

    def test_rpc_recovery_stops_reconnects_and_requires_neutral(self):
        config = SimpleNamespace(
            enabled=True,
            ready_wait_s=2.0,
            allow_manual_localization_warning=True,
            deadzone=0.2,
            linear_speed_mm_s=50,
            angular_speed_mdeg_s=10000,
        )
        chassis = VRChassis(config, execute=True)
        rpc = RecoveringRpc()
        chassis.rpc = rpc
        chassis._recover_rpc(RpcError("session invalid"))
        self.assertEqual(rpc.calls[:3], ["reconnect", "stop", "get_state"])

        chassis.update(np.array([0.0, 1.0]), input_valid=True)
        self.assertEqual(chassis.desired, type(chassis.desired)())
        chassis.update(np.array([0.0, 0.0]), input_valid=True)
        chassis.update(np.array([0.0, 1.0]), input_valid=True)
        self.assertEqual(chassis.desired.linear_mm_s, 50)

    def test_session_failure_without_error_field_is_detected(self):
        self.assertTrue(
            RobotRpc._is_invalid_session_response(
                {"id": 123, "session": "expired", "result": False}
            )
        )
        self.assertTrue(
            RobotRpc._is_invalid_session_response(
                {"result": False, "error": {"code": 287637505}}
            )
        )
        self.assertFalse(
            RobotRpc._is_invalid_session_response(
                {"result": False, "error": {"code": 999, "message": "hardware fault"}}
            )
        )

    def test_rpc_call_relogs_and_retries_old_firmware_session_failure(self):
        rpc = RobotRpc("http://example.invalid", "user", "password")
        rpc.session = "old-session"
        responses = iter(
            [
                {"id": 1, "session": "old-session", "result": False},
                {"id": 2, "session": "new-session", "result": True},
            ]
        )
        login_count = []
        rpc._call_unlocked = lambda method, params: next(responses)

        def fake_login():
            login_count.append(1)
            rpc.session = "new-session"

        rpc._login_unlocked = fake_login
        result = rpc.call("Robot.control", {})
        self.assertTrue(result["result"])
        self.assertEqual(len(login_count), 1)

if __name__ == "__main__":
    unittest.main()
