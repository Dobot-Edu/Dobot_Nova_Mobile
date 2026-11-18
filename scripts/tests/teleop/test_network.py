import json
import socket
import time
import unittest

from quest_vr.network import UDPReceiver


def valid_packet() -> bytes:
    return json.dumps(
        {
            "timestamp": 1.0,
            "tracking_valid": True,
            "right_pos": [0.1, 0.2, 0.3],
            "right_rot": [0.0, 0.0, 0.0, 1.0],
            "right_grip": 0.0,
            "right_trigger": 0.0,
            "button_a": False,
            "button_b": False,
        }
    ).encode("utf-8")


def poll_with_timeout(receiver: UDPReceiver):
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        received = receiver.poll()
        if received is not None:
            return received
        time.sleep(0.005)
    return None


class NetworkTests(unittest.TestCase):
    def test_accepts_and_locks_first_valid_source_in_allowed_subnet(self):
        receiver = UDPReceiver(
            "127.0.0.1", 0, allowed_subnet="127.0.0.0/8"
        )
        first = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        second = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            destination = receiver.socket.getsockname()
            first.sendto(valid_packet(), destination)
            received = poll_with_timeout(receiver)
            self.assertIsNotNone(received)
            self.assertEqual(receiver.locked_source[0], "127.0.0.1")
            self.assertEqual(receiver.locked_source[1], first.getsockname()[1])

            second.sendto(valid_packet(), destination)
            time.sleep(0.01)
            self.assertIsNone(receiver.poll())
            self.assertEqual(receiver.locked_source[0], "127.0.0.1")
            self.assertEqual(receiver.locked_source[1], first.getsockname()[1])
            self.assertGreaterEqual(receiver.invalid_count, 1)
        finally:
            first.close()
            second.close()
            receiver.close()

    def test_rejects_source_outside_allowed_subnet(self):
        receiver = UDPReceiver(
            "127.0.0.1", 0, allowed_subnet="10.2.150.0/23"
        )
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(valid_packet(), receiver.socket.getsockname())
            time.sleep(0.01)
            self.assertIsNone(receiver.poll())
            self.assertIsNone(receiver.locked_source)
            self.assertEqual(receiver.invalid_count, 1)
        finally:
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
