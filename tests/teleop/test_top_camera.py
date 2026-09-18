import unittest

import constants
from Dobot_Init.top_camera import LatestFrame, ViewerStats, load_camera_config


class TopCameraTest(unittest.TestCase):
    def test_latest_frame_replaces_old_data(self):
        frame = LatestFrame()
        self.assertEqual(frame.snapshot()[1], 0)
        self.assertEqual(frame.publish(b"first"), 1)
        self.assertEqual(frame.publish(b"second"), 2)
        data, sequence, captured_at = frame.snapshot()
        self.assertEqual(data, b"second")
        self.assertEqual(sequence, 2)
        self.assertGreater(captured_at, 0)

    def test_load_camera_config(self):
        config = load_camera_config()
        self.assertEqual(config.serial, constants.TOP_CAMERA_SERIAL)
        self.assertEqual(config.http_port, constants.TOP_CAMERA_HTTP_PORT)
        self.assertEqual(config.preview, constants.TOP_CAMERA_PREVIEW)

    def test_viewer_stats_records_client(self):
        stats = ViewerStats()
        self.assertEqual(stats.snapshot(), (0, "", None))
        stats.record("10.2.150.183")
        count, client, age_ms = stats.snapshot()
        self.assertEqual(count, 1)
        self.assertEqual(client, "10.2.150.183")
        self.assertGreaterEqual(age_ms, 0)
