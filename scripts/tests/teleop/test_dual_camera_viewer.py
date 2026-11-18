import unittest
from unittest.mock import patch

from Dobot_Init.dual_camera_viewer import LatestJpeg, load_dual_camera_config


class DualCameraViewerTests(unittest.TestCase):
    def test_latest_jpeg_keeps_newest_frame(self):
        latest = LatestJpeg("test")
        latest.publish(b"first")
        latest.publish(b"second")
        jpeg, sequence, _captured_at, error = latest.snapshot()
        self.assertEqual(jpeg, b"second")
        self.assertEqual(sequence, 2)
        self.assertEqual(error, "")

    def test_config_requires_loopback_viewer(self):
        with patch("constants.DUAL_VIEWER_HOST", "0.0.0.0"):
            with self.assertRaisesRegex(ValueError, "回环地址"):
                load_dual_camera_config()


if __name__ == "__main__":
    unittest.main()
