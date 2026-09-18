import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Dobot_Init.config import load_config
from Dobot_Collection.buttons import ButtonRouter
from Dobot_Init.robot import (
    NovaRobot,
    RobotError,
    controller_has_errors,
    parse_values,
    smooth_joint_trajectory,
)


class ConfigAndRobotTests(unittest.TestCase):
    def test_project_config_is_valid(self):
        config = load_config("full")
        self.assertEqual(config.robot.ip, "192.168.23.201")
        self.assertEqual(config.network.data_port, 5005)
        self.assertEqual(config.network.allowed_quest_ip, "")
        self.assertEqual(config.network.allowed_quest_subnet, "10.2.150.0/23")
        self.assertEqual(config.robot.max_tcp_speed_mm_s, 300)
        self.assertEqual(config.robot.max_tcp_rotation_deg_s, 90)
        self.assertTrue(config.robot.auto_clear_error)
        self.assertTrue(config.robot.auto_recover_manual_mode)
        np.testing.assert_allclose(config.robot.startup_joint_deg, [90, 0, 90, 0, -90, 0])
        self.assertEqual(config.robot.startup_speed_factor, 40)
        self.assertTrue(config.gripper.enabled)
        self.assertEqual(config.gripper.port, "/dev/xtrainer_gripper")
        self.assertEqual(config.gripper.servo_id, 22)
        self.assertEqual(config.gripper.raw_open, 2000)
        self.assertEqual(config.gripper.raw_close, 1000)
        self.assertEqual(config.gripper.max_close_command, 255)
        self.assertEqual(config.gripper.feedback_raw_open, 2000)
        self.assertEqual(config.gripper.feedback_raw_close, 1000)
        self.assertEqual(config.gripper.speed, 30)
        self.assertEqual(config.gripper.target_torque, 350)
        self.assertEqual(config.gripper.trigger_press_threshold, 0.20)
        self.assertEqual(config.gripper.trigger_release_threshold, 0.10)
        self.assertEqual(config.gripper.close_rate_per_s, 1.0)
        self.assertEqual(config.gripper.open_rate_per_s, 1.0)
        # 这是供实机对比试验使用的布尔开关，不把测试绑定到True或False。
        self.assertIsInstance(config.gripper.stop_on_overload_32, bool)
        self.assertFalse(config.lift.enabled)
        self.assertEqual(config.lift.host, "192.168.23.254")
        self.assertFalse(config.lift.home_on_start)
        self.assertTrue(config.lift.auto_enable_on_start)
        self.assertEqual(config.lift.home_di_index, 2)
        self.assertEqual(config.lift.max_height_mm, 400)
        self.assertTrue(config.chassis.enabled)
        self.assertFalse(config.chassis.require_grip)
        self.assertEqual(config.chassis.url, "http://192.168.23.1")
        self.assertEqual(config.chassis.linear_speed_mm_s, 180)
        self.assertEqual(config.chassis.angular_speed_mdeg_s, 15000)
        self.assertAlmostEqual(config.chassis.deadzone, 0.2)
        self.assertEqual(config.chassis.command_rate_hz, 25)
        self.assertEqual(config.chassis.feedback_rate_hz, 25)
        self.assertEqual(config.chassis.ready_wait_s, 5)
        self.assertTrue(config.chassis.allow_manual_localization_warning)
        self.assertAlmostEqual(np.linalg.det(config.mapping.position_axis), -1.0)
        self.assertAlmostEqual(np.linalg.det(config.mapping.orientation_axis), 1.0)

    def test_profiles_only_change_enabled_devices(self):
        arm_only = load_config("arm_only")
        no_chassis = load_config("no_chassis")
        self.assertTrue(arm_only.gripper.enabled)
        self.assertFalse(arm_only.lift.enabled)
        self.assertFalse(arm_only.chassis.enabled)
        self.assertTrue(no_chassis.gripper.enabled)
        self.assertTrue(no_chassis.lift.enabled)
        self.assertFalse(no_chassis.chassis.enabled)

    def test_parses_dashboard_reply(self):
        np.testing.assert_allclose(
            parse_values("0,{1,2,3,4,5,6},GetAngle();", 6), [1, 2, 3, 4, 5, 6]
        )

    def test_reads_digital_input(self):
        robot = NovaRobot.__new__(NovaRobot)
        robot.request = lambda command: "0,{1},DI(2);"
        self.assertEqual(robot.digital_input(2), 1)

    def test_rejects_bad_dashboard_reply(self):
        with self.assertRaises(RobotError):
            parse_values("-1,{},GetAngle();", 6)

    def test_get_error_id_parser(self):
        self.assertFalse(controller_has_errors("0,{[]},GetErrorID();"))
        self.assertFalse(controller_has_errors("0,{},GetErrorID();"))
        self.assertTrue(controller_has_errors("0,{[[1001],[2002]]},GetErrorID();"))
        with self.assertRaises(RobotError):
            controller_has_errors("-1,{},GetErrorID();")

    def test_enable_uses_commands_supported_by_nova_firmware(self):
        robot = NovaRobot.__new__(NovaRobot)
        robot.config = SimpleNamespace(speed_factor=10)
        commands = []
        robot.request = lambda command: commands.append(command) or f"0,{{}},{command};"
        robot.mode = lambda: 5

        robot.enable()

        self.assertEqual(commands, ["EnableRobot()", "SpeedFactor(10)", "AccJ(10)"])
        self.assertNotIn("SpeedJ(10)", commands)

    def test_prepare_clears_once_then_enables(self):
        robot = NovaRobot.__new__(NovaRobot)
        robot.config = SimpleNamespace(speed_factor=10, auto_clear_error=True)
        commands = []
        error_queries = 0

        def request(command):
            nonlocal error_queries
            commands.append(command)
            if command == "GetErrorID()":
                error_queries += 1
                return (
                    "0,{[[1001]]},GetErrorID();"
                    if error_queries == 1
                    else "0,{[]},GetErrorID();"
                )
            return f"0,{{}},{command};"

        modes = iter((9, 4, 4, 5, 5, 5, 5))
        robot.request = request
        robot.mode = lambda: next(modes)
        robot.prepare_for_control()

        self.assertEqual(commands.count("ClearError()"), 1)
        self.assertIn("EnableRobot()", commands)
        self.assertIn("SpeedFactor(10)", commands)

    @patch("Dobot_Init.robot.time.sleep", return_value=None)
    def test_prepare_stops_stale_jog_and_reenables(self, _sleep):
        robot = NovaRobot.__new__(NovaRobot)
        robot.config = SimpleNamespace(
            speed_factor=10,
            auto_clear_error=True,
            auto_recover_manual_mode=True,
        )
        commands = []
        robot.request = lambda command: (
            commands.append(command)
            or ("0,{[]},GetErrorID();" if command == "GetErrorID()" else f"0,{{}},{command};")
        )
        # DisableRobot先短暂返回5，再真正变成4；不能提前使能。
        modes = iter((11, 11, 5, 7, 4, 5, 5, 5, 5))
        robot.mode = lambda: next(modes)

        robot.prepare_for_control()

        self.assertIn("ClearError()", commands)
        self.assertIn("StopDrag()", commands)
        self.assertIn("MoveJog()", commands)
        self.assertIn("DisableRobot()", commands)
        self.assertIn("EnableRobot()", commands)
        self.assertLess(commands.index("MoveJog()"), commands.index("DisableRobot()"))

    def test_prepare_stops_drag_and_reenables(self):
        robot = NovaRobot.__new__(NovaRobot)
        robot.config = SimpleNamespace(
            speed_factor=10,
            auto_clear_error=True,
            auto_recover_manual_mode=True,
        )
        commands = []
        robot.request = lambda command: (
            commands.append(command)
            or ("0,{[]},GetErrorID();" if command == "GetErrorID()" else f"0,{{}},{command};")
        )
        modes = iter((6, 6, 4, 5, 5, 5, 5))
        robot.mode = lambda: next(modes)

        robot.prepare_for_control()

        self.assertIn("StopDrag()", commands)
        self.assertNotIn("MoveJog()", commands)
        self.assertIn("DisableRobot()", commands)
        self.assertIn("EnableRobot()", commands)

    @patch("Dobot_Init.robot.time.sleep", return_value=None)
    def test_startup_pose_uses_recordable_servo_trajectory(self, _sleep):
        robot = NovaRobot.__new__(NovaRobot)
        robot.config = SimpleNamespace(
            startup_pose_enabled=True,
            startup_joint_deg=np.array([90, 0, 90, 0, -90, 0], dtype=float),
            startup_speed_factor=10,
            startup_tolerance_deg=0.5,
            startup_timeout_s=2.0,
            speed_factor=30,
            max_joint_speed_deg_s=1000.0,
            rate_hz=10.0,
        )
        # 即使当前位置离标准位很远，也不在应用层拒绝回位；控制柜负责本体行程。
        current = np.deg2rad([40, 30, 40, 30, -30, 30])
        target = np.deg2rad(robot.config.startup_joint_deg)
        joint_reads = iter((current, target, target, target, target))
        robot.joints = lambda: next(joint_reads)
        robot.mode = lambda: 5
        robot.snapshot = lambda: SimpleNamespace(
            joints_rad=target,
            pose=np.zeros(6),
            mode=5,
            errors_reply="0,{[]},GetErrorID();",
        )
        commands = []
        robot.request = lambda command: commands.append(command) or f"0,{{}},{command};"

        progress = []
        snapshot = robot.move_to_startup_pose(progress_callback=progress.append)

        np.testing.assert_allclose(snapshot.joints_rad, target)
        self.assertGreaterEqual(len(progress), 3)
        np.testing.assert_allclose(progress[-1], target)
        self.assertFalse(any(command.startswith("MovJ(") for command in commands))
        self.assertTrue(any(command.startswith("ServoJ(") for command in commands))
        self.assertEqual(commands[:2], ["SpeedFactor(10)", "AccJ(10)"])
        self.assertEqual(commands[-2:], ["SpeedFactor(30)", "AccJ(30)"])

    def test_smooth_joint_trajectory_reaches_target_and_limits_speed(self):
        start = np.zeros(6)
        target = np.deg2rad(np.array([60, -30, 0, 10, 0, 0], dtype=float))
        trajectory = list(
            smooth_joint_trajectory(start, target, rate_hz=50.0, max_speed_deg_s=30.0)
        )
        np.testing.assert_allclose(trajectory[-1], target)
        points = np.vstack([start, *trajectory])
        step_speed = np.max(np.abs(np.diff(np.rad2deg(points), axis=0))) * 50.0
        self.assertLessEqual(step_speed, 30.1)

    def test_activation_requires_a_click_not_held_startup_input(self):
        router = ButtonRouter()
        self.assertFalse(router.update(True, False, valid=True, activated=False, now=0).activate)
        router.update(False, False, valid=True, activated=False, now=1)
        router.update(True, False, valid=True, activated=False, now=2)
        router.update(False, False, valid=True, activated=False, now=2.1)
        self.assertTrue(router.update(False, False, valid=True, activated=False, now=2.5).activate)


if __name__ == "__main__":
    unittest.main()
