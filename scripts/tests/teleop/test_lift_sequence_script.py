import unittest
from unittest.mock import patch

from scripts.diagnostics.lift_diagnostic import confirm, wait_until_reached


class FakeLift:
    def __init__(self, heights):
        self.heights = iter(heights)

    def read_height(self):
        return next(self.heights)


class LiftSequenceScriptTests(unittest.TestCase):
    def test_confirmation_requires_exact_lift_word(self):
        with patch("builtins.input", return_value="LIFT"):
            self.assertTrue(confirm())
        with patch("builtins.input", return_value="yes"):
            self.assertFalse(confirm())

    @patch("scripts.diagnostics.lift_diagnostic.time.sleep", return_value=None)
    def test_waits_until_feedback_reaches_tolerance(self, _sleep):
        lift = FakeLift((4.0, 4.7, 4.9))
        actual = wait_until_reached(
            lift,
            5.0,
            tolerance_cm=0.2,
            timeout_s=10.0,
        )
        self.assertEqual(actual, 4.9)


if __name__ == "__main__":
    unittest.main()
