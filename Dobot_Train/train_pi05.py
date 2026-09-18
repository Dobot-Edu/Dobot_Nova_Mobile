#!/usr/bin/env python3
"""在训练服务器上微调 π0.5；所有默认参数集中在 constants.py。"""

from __future__ import annotations

import json
import subprocess
import sys

import constants as settings
from lerobot_source import use_local_lerobot


def preflight() -> None:
    info_path = settings.LEROBOT_ROOT / "meta" / "info.json"
    stats_path = settings.LEROBOT_ROOT / "meta" / "stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        raise RuntimeError(f"找不到完整LeRobot v3数据集：{settings.LEROBOT_ROOT}")
    if settings.TRAIN_OUTPUT.exists():
        raise RuntimeError(f"输出目录已存在，为避免覆盖请修改TRAIN_OUTPUT：{settings.TRAIN_OUTPUT}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise RuntimeError(f"数据集不是LeRobot v3：{info.get('codebase_version')}")
    for key, expected_size in (
        ("observation.state", settings.STATE_DIM),
        ("action", settings.ACTION_DIM),
    ):
        shape = info["features"][key]["shape"]
        if list(shape) != [expected_size]:
            raise RuntimeError(f"{key}形状错误：{shape}")
        if "q01" not in stats[key] or "q99" not in stats[key]:
            raise RuntimeError(f"{key}缺少π0.5量化归一化统计量")


def main() -> int:
    use_local_lerobot()
    preflight()
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={settings.DATASET_REPO_ID}",
        f"--dataset.root={settings.LEROBOT_ROOT}",
        "--dataset.video_backend=pyav",
        "--policy.type=pi05",
        f"--policy.pretrained_path={settings.PRETRAINED_MODEL}",
        f"--policy.repo_id={settings.POLICY_REPO_ID}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false",
        "--policy.train_expert_only=false",
        "--policy.freeze_vision_encoder=false",
        f"--policy.chunk_size={settings.POLICY_CHUNK_SIZE}",
        f"--policy.n_action_steps={settings.POLICY_ACTION_STEPS}",
        "--policy.use_relative_actions=false",
        "--policy.push_to_hub=false",
        f"--batch_size={settings.BATCH_SIZE}",
        f"--steps={settings.TRAIN_STEPS}",
        f"--policy.scheduler_decay_steps={settings.TRAIN_STEPS}",
        "--save_checkpoint=true",
        f"--save_freq={settings.SAVE_FREQUENCY}",
        "--log_freq=20",
        f"--num_workers={settings.NUM_WORKERS}",
        "--wandb.enable=false",
        "--job_name=nova_pi05_mobile",
        f"--output_dir={settings.TRAIN_OUTPUT}",
    ]
    print("开始π0.5训练：")
    print("\n".join(f"  {item}" for item in command))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
