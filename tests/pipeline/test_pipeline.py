from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import constants as settings
from scripts.convert_to_lerobot_v3 import convert_dataset
from Dobot_Collection.schema import ACTION_NAMES, STATE_NAMES, build_vectors
from lerobot_source import use_local_lerobot

use_local_lerobot()


class PipelineTest(unittest.TestCase):
    def test_legacy_data_is_rejected(self):
        from Dobot_Collection.schema import episode_vectors
        for version, state_size, action_size in [(1, 9, 9), (2, 12, 10), (2, 11, 9)]:
            with self.assertRaises(ValueError):
                episode_vectors(np.zeros(state_size), np.zeros(action_size), version)

    def test_bottom_feedback_alignment_and_units(self):
        telemetry = {
            "joints_command_deg": [0, 10, 20, 30, 40, 50],
            "gripper_position": 0.25,
            "gripper_target_position": 0.30,
            "gripper_position_raw": 1780,
            "gripper_current_ma": 100.0,
            "gripper_load_raw": 2,
            "gripper_temperature_c": 35,
            "gripper_moving": True,
            "gripper_feedback_monotonic_ns": 1_100_000_000,
            "gripper_feedback_valid": True,
            "lift_target_cm": 25.0,
            "chassis_feedback_speed_mm_s": -50,
            "chassis_feedback_angular_mdeg_s": 10000,
            "chassis_command_linear_mm_s": 40,
            "chassis_command_angular_mdeg_s": -5000,
            "chassis_feedback_monotonic_ns": 1_000_000_000,
            "chassis_feedback_valid": True,
            "chassis_feedback_position_mm": [1000, -2000],
            "chassis_feedback_yaw_mdeg": 90000,
            "chassis_feedback_odom_mm": 3000,
        }
        state, action, diagnostic, gripper_diagnostic = build_vectors(
            [0, 10, 20, 30, 40, 50], telemetry, 1_150_000_000,
            base_valid_age_s=0.3, gripper_valid_age_s=0.15,
        )
        self.assertEqual(state.shape, (9,))
        self.assertEqual(action.shape, (9,))
        self.assertNotIn("lift_height_m", STATE_NAMES)
        self.assertNotIn("lift_target_m", ACTION_NAMES)
        self.assertAlmostEqual(float(state[6]), 0.25)
        self.assertAlmostEqual(float(state[7]), -0.05)
        self.assertAlmostEqual(float(state[8]), np.deg2rad(10.0), places=6)
        self.assertAlmostEqual(float(action[6]), 0.30)
        self.assertAlmostEqual(float(action[7]), 0.04)
        self.assertAlmostEqual(diagnostic["yaw_rad"], np.pi / 2)
        self.assertTrue(gripper_diagnostic["feedback_valid"])

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "需要安装 LeRobot")
    def test_mock_episode_converts_to_lerobot_v3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            episode = raw / "episode_mock"
            for camera in ("top", "wrist"):
                (episode / "images" / camera).mkdir(parents=True)
            metadata = {
                "schema_version": 2,
                "fps": settings.DATASET_FPS,
                "task": settings.TASK,
                "success": True,
                "state_names": STATE_NAMES,
                "action_names": ACTION_NAMES,
                "image_keys": ["top", "wrist"],
                "gripper_alignment": {"open_is_one": True},
            }
            (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            with (episode / "samples.jsonl").open("w", encoding="utf-8") as stream:
                for index in range(40):
                    paths = {}
                    for camera in ("top", "wrist"):
                        relative = Path("images") / camera / f"{index:06d}.jpg"
                        cv2.imwrite(str(episode / relative), image + index)
                        paths[camera] = relative.as_posix()
                    state = np.zeros(9, dtype=np.float32)
                    row = {
                        "index": index,
                        "observation.state": state.tolist(),
                        "action": np.zeros(9, dtype=np.float32).tolist(),
                        "images": paths,
                    }
                    stream.write(json.dumps(row) + "\n")
            (episode / "COMPLETE").write_text("ok\n", encoding="utf-8")
            output = root / "dataset"
            episodes, frames = convert_dataset(raw, output, "local/test_mobile", use_videos=True)
            self.assertEqual((episodes, frames), (1, 40))
            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            stats = json.loads((output / "meta" / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], "v3.0")
            self.assertEqual(info["fps"], settings.DATASET_FPS)
            self.assertIn("q01", stats["observation.state"])
            self.assertIn("q99", stats["action"])
            self.assertEqual(len(list(output.rglob("*.mp4"))), 2)
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset("local/test_mobile", root=output, video_backend="pyav")
            self.assertEqual(len(dataset), 40)
            for index in (0, 20, 39):
                row = dataset[index]
                for camera in ("top", "wrist"):
                    image = row[f"observation.images.{camera}"]
                    self.assertEqual(tuple(image.shape), (3, 48, 64))
                    self.assertAlmostEqual(float(image.mean()), index / 255, delta=0.04)


if __name__ == "__main__":
    unittest.main()
