from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.replay_episode import load_episode, resolve_episode


class ReplayEpisodeTest(unittest.TestCase):
    def make_episode(self, root: Path, name: str = "episode_20260911_000000_000000") -> Path:
        path = root / name
        path.mkdir()
        (path / "COMPLETE").write_text("ok\n", encoding="utf-8")
        metadata = {"success": True, "fps": 20, "frame_count": 2, "schema_version": 2}
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        rows = []
        for index in range(2):
            rows.append(
                {
                    "index": index,
                    "timestamp_ns": index + 1,
                    "observation.state": [0.0] * 9,
                    "action": [0.0] * 9,
                    "gripper_diagnostic": {"feedback_valid": index == 0},
                }
            )
        (path / "samples.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_loads_nine_dimensional_actions_and_reports_stale_gripper(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = load_episode(self.make_episode(Path(directory)))
            self.assertEqual(episode.actions.shape, (2, 9))
            self.assertTrue(np.isfinite(episode.actions).all())
            self.assertEqual(episode.invalid_gripper_frames, (1,))

    def test_latest_selects_complete_successful_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_episode(root, "episode_20260911_000000_000000")
            latest = self.make_episode(root, "episode_20260911_000001_000000")
            self.assertEqual(resolve_episode("latest", root), latest.resolve())
            self.assertNotEqual(first, latest)


if __name__ == "__main__":
    unittest.main()
