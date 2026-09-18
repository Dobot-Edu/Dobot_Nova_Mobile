import unittest
from dataclasses import replace
from unittest.mock import patch

from Dobot_Init.config import LiftConfig
from Dobot_Init.lift import (
    LiftColumn,
    LiftDirection,
    button_direction,
    decode_exception_code,
    decode_height_cm,
    encode_height_cm,
)


class FakeClient:
    def __init__(self, height_cm=4.2):
        self.height_cm = height_cm
        self.writes = []

    def read_holding(self, address, count):
        self.assert_request(address, count)
        return list(encode_height_cm(self.height_cm))

    def write_holding(self, address, values):
        self.writes.append((address, values))
        if address == LiftColumn.POSITION_REGISTER:
            self.height_cm = decode_height_cm(*values)

    @staticmethod
    def assert_request(address, count):
        if (address, count) != (LiftColumn.CURRENT_HEIGHT_REGISTER, 2):
            raise AssertionError((address, count))


class LiftTests(unittest.TestCase):
    def setUp(self):
        self.config = LiftConfig(
            enabled=True,
            host="192.168.23.254",
            port=503,
            unit_id=1,
            home_on_start=True,
            home_di_index=2,
            home_speed=250,
            home_timeout_s=120.0,
            max_height_mm=400.0,
            command_rate_hz=10.0,
            drive_speed=300,
            release_compensation_mm=2.0,
            socket_timeout_s=0.5,
            auto_enable_on_start=True,
        )

    def test_button_mapping(self):
        self.assertIs(button_direction(True, True, input_valid=True), LiftDirection.UP)
        self.assertIs(button_direction(False, True, input_valid=True), LiftDirection.DOWN)
        self.assertIs(button_direction(False, False, input_valid=True), LiftDirection.HOLD)
        self.assertIs(button_direction(True, False, input_valid=True), LiftDirection.HOLD)
        self.assertIs(button_direction(True, True, input_valid=False), LiftDirection.HOLD)

    def test_height_codec(self):
        for height in (0.0, 1.0, 5.0, 6.5535, 10.0):
            low, high = encode_height_cm(height)
            self.assertTrue(0 <= low <= 65535)
            self.assertTrue(0 <= high <= 65535)
            self.assertAlmostEqual(decode_height_cm(low, high), height, places=3)

    def test_known_zero_register_sentinels_are_zero_height(self):
        self.assertEqual(decode_height_cm(0, 0), 0.0)
        self.assertEqual(decode_height_cm(1, 0), 0.0)

    def test_gateway_two_byte_busy_exception_is_decoded(self):
        self.assertEqual(decode_exception_code(bytes.fromhex("90 00 06")), 6)
        self.assertEqual(decode_exception_code(bytes.fromhex("90 06")), 6)

    def test_start_without_home_only_reads_current_height_and_does_not_move(self):
        client = FakeClient(14.2)
        config = replace(self.config, home_on_start=False)
        lift = LiftColumn(config, execute=True, client=client)
        lift.connect()
        try:
            self.assertAlmostEqual(lift.height_cm, 14.2, places=3)
            self.assertEqual(client.writes, [])
        finally:
            lift.close()
        self.assertEqual(client.writes, [])

    @patch("Dobot_Init.lift.time.sleep", return_value=None)
    def test_prepare_enables_mode_without_starting_motion(self, _sleep):
        client = FakeClient(14.2)
        config = replace(self.config, home_on_start=False)
        lift = LiftColumn(config, execute=True, client=client)
        lift.connect()
        try:
            lift.prepare_for_control()
            self.assertEqual(
                client.writes,
                [
                    (LiftColumn.ENABLE_REGISTER, [32]),
                    (LiftColumn.MODE_REGISTER, [1]),
                ],
            )
            self.assertTrue(lift._initialized)
        finally:
            lift.close()

    @patch("Dobot_Init.lift.time.sleep", return_value=None)
    def test_up_and_down_use_fixed_endpoints_and_configured_speed(self, _sleep):
        client = FakeClient(4.2)
        lift = LiftColumn(self.config, execute=True, client=client)
        lift._height_cm = 4.2
        lift._step(LiftDirection.UP)
        self.assertAlmostEqual(lift.height_cm, 0.0, places=3)
        self.assertIn((LiftColumn.SPEED_REGISTER, [300]), client.writes)
        lift._step(LiftDirection.DOWN)
        self.assertAlmostEqual(lift.height_cm, 40.0, places=3)

    @patch("Dobot_Init.lift.time.sleep", return_value=None)
    def test_release_compensates_feedback_in_previous_motion_direction(self, _sleep):
        client = FakeClient(12.3)
        lift = LiftColumn(self.config, execute=True, client=client)
        lift._height_cm = 40.0
        lift._applied = LiftDirection.DOWN
        lift._hold_current()
        self.assertIn((LiftColumn.SPEED_REGISTER, [300]), client.writes)
        self.assertIn(
            (LiftColumn.POSITION_REGISTER, list(encode_height_cm(12.5))),
            client.writes,
        )
        self.assertEqual(client.writes[-1], (LiftColumn.START_REGISTER, [16]))
        self.assertAlmostEqual(lift.height_cm, 12.5, places=3)

        client.height_cm = 12.3
        lift.stop_motion(LiftDirection.UP)
        self.assertAlmostEqual(lift.height_cm, 12.1, places=3)

    @patch("Dobot_Init.lift.time.sleep", return_value=None)
    def test_release_compensation_is_clamped_to_travel_limits(self, _sleep):
        client = FakeClient(39.9)
        lift = LiftColumn(self.config, execute=True, client=client)
        lift._height_cm = 40.0
        lift.stop_motion(LiftDirection.DOWN)
        self.assertEqual(lift.height_cm, 40.0)

        client.height_cm = 0.1
        lift.stop_motion(LiftDirection.UP)
        self.assertEqual(lift.height_cm, 0.0)

    @patch("Dobot_Init.lift.time.sleep", return_value=None)
    def test_home_replays_columns_reset_and_waits_for_di2(self, _sleep):
        client = FakeClient(0.0)
        lift = LiftColumn(self.config, execute=True, client=client)
        signals = iter((0, 0, 1))
        requested_di = []

        def read_di(index):
            requested_di.append(index)
            return next(signals)

        lift.lifting_columns("columns_reste", 0.0, 0, home_signal=read_di)
        self.assertEqual(
            client.writes,
            [
                (5888, [0]),
                (0x170D, [250]),
                (0x170A, [18]),
                (5888, [32]),
                (0x1701, [32]),
            ],
        )
        self.assertEqual(requested_di, [2, 2, 2])
        self.assertEqual(lift.height_cm, 0.0)


if __name__ == "__main__":
    unittest.main()
