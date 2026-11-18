#!/usr/bin/env python3
"""Compare a PI0.5 action chunk with the recorded future actions of a raw episode."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dobot_Collection.schema import ACTION_NAMES, STATE_NAMES, episode_vectors, read_jsonl
from config import inference as settings

if str(settings.LEROBOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(settings.LEROBOT_SOURCE))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from lerobot.policies.utils import build_inference_frame

from Dobot_Inference.infer import load_metadata_and_limits, load_policy, predict_action_chunk


def resolve_episode(value: str) -> Path:
    supplied = Path(value).expanduser()
    candidates = [supplied, ROOT / value]
    if not supplied.is_absolute():
        candidates.extend((ROOT / "datasets").glob(f"**/{value}"))
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "samples.jsonl").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到包含 samples.jsonl 的 episode：{value}")


def choose_indices(frame_count: int, horizon: int, values: str | None, count: int) -> list[int]:
    last = frame_count - horizon
    if last < 0:
        raise ValueError(f"episode 只有 {frame_count} 帧，少于动作窗口 {horizon} 帧")
    if values:
        indices = sorted(set(int(item.strip()) for item in values.split(",") if item.strip()))
    else:
        indices = np.linspace(0, last, min(count, last + 1), dtype=int).tolist()
    invalid = [index for index in indices if not 0 <= index <= last]
    if invalid:
        raise ValueError(f"采样帧必须在 0..{last}：{invalid}")
    return indices


def build_frame(metadata, episode: Path, row: dict, state: np.ndarray, task: str):
    observation = dict(zip(STATE_NAMES, state, strict=True))
    images = {}
    for camera in ("top", "wrist"):
        path = episode / row["images"][camera]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像：{path}")
        images[camera] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        observation[camera] = images[camera]
    return build_inference_frame(
        observation=observation,
        device=torch.device("cpu"),
        ds_features=metadata.features,
        task=task,
        robot_type=metadata.robot_type,
    ), images


def save_plot(path: Path, index: int, prediction: np.ndarray, target: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(len(target))
    figure, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for feature, axis in enumerate(axes.flat):
        axis.plot(steps, target[:, feature], label="recorded", linewidth=2)
        axis.plot(steps, prediction[:, feature], label="predicted", linewidth=1.5)
        axis.set_title(ACTION_NAMES[feature], fontsize=9)
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    figure.suptitle(f"Raw frame {index}: predicted chunk vs recorded future actions")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="raw episode 目录、相对路径或 episode 名称")
    parser.add_argument("--indices", help="逗号分隔的采样帧，例如 0,100,500")
    parser.add_argument("--samples", type=int, default=6, help="未指定 indices 时均匀采样数量")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="结果目录；默认写入 logs/offline_inference")
    args = parser.parse_args()

    episode = resolve_episode(args.episode)
    metadata_json = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    schema_version = int(metadata_json.get("schema_version", 0))
    rows = list(read_jsonl(episode / "samples.jsonl"))

    print(f"[离线] 加载模型：{settings.CHECKPOINT_DIR}", flush=True)
    metadata, limits, tasks = load_metadata_and_limits()
    task = tasks[settings.INITIAL_TASK_INDEX - 1]
    policy, preprocessor, postprocessor = load_policy(metadata)
    horizon = int(policy.config.chunk_size if settings.PLAN_HORIZON_STEPS is None else settings.PLAN_HORIZON_STEPS)
    indices = choose_indices(len(rows), horizon, args.indices, args.samples)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / "logs" / "offline_inference" / f"{episode.name}_{timestamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []

    for sample_number, index in enumerate(indices, 1):
        state, _ = episode_vectors(rows[index]["observation.state"], rows[index]["action"], schema_version)
        frame, images = build_frame(metadata, episode, rows[index], state, task)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        started = time.monotonic()
        _, prediction, _ = predict_action_chunk(policy, preprocessor, postprocessor, frame, limits)
        inference_s = time.monotonic() - started
        target = np.stack([
            episode_vectors(row["observation.state"], row["action"], schema_version)[1]
            for row in rows[index : index + horizon]
        ])
        error = prediction - target
        mae = np.mean(np.abs(error), axis=0)
        first_error = np.abs(error[0])
        save_plot(output / f"frame_{index:06d}.png", index, prediction, target)
        cv2.imwrite(str(output / f"frame_{index:06d}_top.jpg"), cv2.cvtColor(images["top"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / f"frame_{index:06d}_wrist.jpg"), cv2.cvtColor(images["wrist"], cv2.COLOR_RGB2BGR))
        np.savez_compressed(output / f"frame_{index:06d}.npz", prediction=prediction, target=target, state=state)
        records.append({
            "frame_index": index,
            "time_s": index / float(metadata.fps),
            "inference_s": inference_s,
            "state": state.tolist(),
            "first_prediction": prediction[0].tolist(),
            "first_target": target[0].tolist(),
            "first_absolute_error": first_error.tolist(),
            "chunk_mae": mae.tolist(),
        })
        print(
            f"[{sample_number}/{len(indices)}] frame={index} inference={inference_s:.3f}s "
            f"gripper pred/gt={prediction[0, 6]:.3f}/{target[0, 6]:.3f} "
            f"base pred/gt={prediction[0, 7]:+.3f}/{target[0, 7]:+.3f},"
            f"{prediction[0, 8]:+.3f}/{target[0, 8]:+.3f}",
            flush=True,
        )

    mean_chunk_mae = np.mean([record["chunk_mae"] for record in records], axis=0)
    summary = {
        "episode": str(episode),
        "checkpoint": str(settings.CHECKPOINT_DIR),
        "task_used_for_inference": task,
        "raw_task": metadata_json.get("task"),
        "fps": metadata.fps,
        "horizon": horizon,
        "seed_reset_for_each_sample": args.seed,
        "indices": indices,
        "mean_chunk_mae": dict(zip(ACTION_NAMES, mean_chunk_mae.tolist(), strict=True)),
        "samples": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[结果] 平均动作块 MAE：")
    for name, value in summary["mean_chunk_mae"].items():
        print(f"  {name}: {value:.6f}")
    print(f"[结果] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
