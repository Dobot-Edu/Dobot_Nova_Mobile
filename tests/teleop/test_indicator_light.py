import unittest

from Dobot_Collection.session import CollectionSession
from Dobot_Init.indicator_light import (
    IndicatorLight,
    LIGHT_OUTPUTS,
    LightColor,
    collection_state_color,
)
from Dobot_Init.robot import NovaRobot


class FakeRobot:
    def __init__(self):
        self.outputs = []

    def digital_output(self, index, value):
        self.outputs.append((index, value))


class IndicatorLightTests(unittest.TestCase):
    def test_documented_output_mapping(self):
        self.assertEqual(LIGHT_OUTPUTS[LightColor.RED], 6)
        self.assertEqual(LIGHT_OUTPUTS[LightColor.GREEN], 7)
        self.assertEqual(LIGHT_OUTPUTS[LightColor.YELLOW], 8)

    def test_switching_color_turns_all_outputs_off_first(self):
        robot = FakeRobot()
        light = IndicatorLight(robot)
        light.set(LightColor.YELLOW)
        self.assertEqual(robot.outputs, [(6, 0), (7, 0), (8, 0), (8, 1)])

        light.set(LightColor.GREEN)
        self.assertEqual(robot.outputs[-2:], [(8, 0), (7, 1)])

    def test_setting_same_color_does_not_repeat_queue_commands(self):
        robot = FakeRobot()
        light = IndicatorLight(robot)
        light.set(LightColor.YELLOW)
        command_count = len(robot.outputs)
        light.set(LightColor.YELLOW)
        self.assertEqual(len(robot.outputs), command_count)

    def test_collection_states_map_to_physical_colors(self):
        self.assertEqual(collection_state_color("ready"), LightColor.YELLOW)
        self.assertEqual(collection_state_color("saving"), LightColor.YELLOW)
        self.assertEqual(collection_state_color("recording"), LightColor.GREEN)
        self.assertEqual(collection_state_color("error"), LightColor.RED)

    def test_collection_transition_notifies_indicator_callback(self):
        changes = []
        session = CollectionSession(
            state_callback=lambda state, detail: changes.append((state, detail))
        )
        session._transition("recording", "第1条录制中")
        self.assertEqual(session.state, "recording")
        self.assertEqual(changes, [("recording", "第1条录制中")])

    def test_robot_writes_dashboard_do_command(self):
        robot = NovaRobot.__new__(NovaRobot)
        commands = []
        robot.request = lambda command: commands.append(command) or f"0,{{}},{command};"
        robot.digital_output(7, True)
        self.assertEqual(commands, ["DO(7,1)"])

    def test_robot_rejects_invalid_output_value(self):
        robot = NovaRobot.__new__(NovaRobot)
        robot.request = lambda _command: self.fail("无效值不应发给控制柜")
        with self.assertRaises(ValueError):
            robot.digital_output(6, 2)


if __name__ == "__main__":
    unittest.main()
