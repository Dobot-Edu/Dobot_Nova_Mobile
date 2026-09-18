#!/usr/bin/env python3
"""在转换前检查所有成功 episode 的结构、同步和底盘反馈适配质量。"""

from __future__ import annotations

import json

import cv2
import numpy as np

import constants as settings
from Dobot_Collection.schema import (
    completed_episodes,
    episode_vectors,
    read_jsonl,
    validate_training_metadata,
)


def validate_episode(path) -> tuple[int, float]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    validate_training_metadata(metadata)
    if metadata.get("fps") != settings.DATASET_FPS or metadata.get("success") is not True:
        raise ValueError(f"{path.name} 的fps或success错误")

    previous_wall_ns = 0
    previous_sample_ns = 0
    sample_intervals_s = []
    previous_base_ns = 0
    base_updates = 0
    ages = []
    previous_gripper_ns = 0
    gripper_updates = 0
    gripper_ages = []
    frames = 0
    for expected_index, row in enumerate(read_jsonl(path / "samples.jsonl")):
        if row.get("index") != expected_index:
            raise ValueError(f"{path.name} 帧序号不连续：{row.get('index')} != {expected_index}")
        state, action = episode_vectors(
            row["observation.state"],
            row["action"],
            int(metadata.get("schema_version", 1)),
        )
        if not 0.0 <= state[6] <= 1.0 or not 0.0 <= action[6] <= 1.0:
            raise ValueError(f"{path.name} 第{expected_index}帧夹爪超出0..1")
        wall_ns = int(row["timestamp_ns"])
        if wall_ns <= previous_wall_ns:
            raise ValueError(f"{path.name} 第{expected_index}帧时间戳未递增")
        previous_wall_ns = wall_ns

        sample_ns = int(row["sample_monotonic_ns"])
        if sample_ns <= previous_sample_ns:
            raise ValueError(f"{path.name} 第{expected_index}帧单调时间戳未递增")
        if previous_sample_ns:
            sample_intervals_s.append((sample_ns - previous_sample_ns) / 1e9)
        previous_sample_ns = sample_ns

        base = row["base_diagnostic"]
        base_ns = int(base["feedback_monotonic_ns"])
        age_s = float(base["feedback_age_s"])
        if base_ns < previous_base_ns or not 0.0 <= age_s <= settings.BASE_ABORT_AGE_S:
            raise ValueError(f"{path.name} 第{expected_index}帧底盘时间或年龄无效")
        if base_ns != previous_base_ns:
            base_updates += 1
        previous_base_ns = base_ns
        ages.append(age_s)

        if int(metadata.get("schema_version", 1)) >= 2:
            gripper = row.get("gripper_diagnostic", {})
            gripper_age_s = float(gripper.get("feedback_age_s", float("inf")))
            if not bool(gripper.get("feedback_valid", False)):
                raise ValueError(f"{path.name} 第{expected_index}帧夹爪编码器反馈无效")
            if not 0.0 <= gripper_age_s <= settings.GRIPPER_FEEDBACK_ABORT_AGE_S:
                raise ValueError(f"{path.name} 第{expected_index}帧夹爪反馈年龄无效")
            gripper_ns = int(gripper["feedback_monotonic_ns"])
            if gripper_ns < previous_gripper_ns:
                raise ValueError(f"{path.name} 第{expected_index}帧夹爪反馈时间倒退")
            if gripper_ns != previous_gripper_ns:
                gripper_updates += 1
            previous_gripper_ns = gripper_ns
            gripper_ages.append(gripper_age_s)

        for camera in ("top", "wrist"):
            image_path = path / row["images"][camera]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            expected_shape = (settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH, 3)
            if image is None or image.shape != expected_shape:
                raise ValueError(f"{path.name} 第{expected_index}帧{camera}图像无效")
        frames += 1

    if frames < settings.DATASET_FPS:
        raise ValueError(f"{path.name} 少于1秒：{frames}帧")
    duration_s = frames / settings.DATASET_FPS
    expected_period_s = 1.0 / settings.DATASET_FPS
    if sample_intervals_s:
        min_interval_s = min(sample_intervals_s)
        max_interval_s = max(sample_intervals_s)
        measured_hz = (frames - 1) / sum(sample_intervals_s)
        if not settings.DATASET_FPS * 0.98 <= measured_hz <= settings.DATASET_FPS * 1.02:
            raise ValueError(
                f"{path.name} 实际采样率{measured_hz:.2f}Hz偏离配置{settings.DATASET_FPS}Hz"
            )
        if min_interval_s < expected_period_s * 0.5 or max_interval_s > expected_period_s * 1.5:
            raise ValueError(
                f"{path.name} 存在补帧或卡顿：帧间隔"
                f"{min_interval_s * 1000:.1f}..{max_interval_s * 1000:.1f}ms"
            )
    update_hz = base_updates / duration_s
    expected_hz = float(metadata.get("base_alignment", {}).get("feedback_hz", 5.0))
    lower_hz = expected_hz * 0.70
    upper_hz = expected_hz * 1.20
    if not lower_hz <= update_hz <= upper_hz:
        raise ValueError(
            f"{path.name} 底盘反馈约{update_hz:.2f} Hz，"
            f"不在配置{expected_hz:g} Hz的容差范围"
        )
    if max(ages) > settings.BASE_VALID_AGE_S + 0.05:
        print(f"[警告] {path.name} 存在valid=0但尚未超时的底盘帧")
    gripper_text = ""
    if int(metadata.get("schema_version", 1)) >= 2:
        expected_gripper_hz = float(metadata["gripper_alignment"]["feedback_hz"])
        gripper_hz = gripper_updates / duration_s
        if not expected_gripper_hz * 0.70 <= gripper_hz <= expected_gripper_hz * 1.20:
            raise ValueError(
                f"{path.name} 夹爪反馈约{gripper_hz:.2f} Hz，"
                f"不在配置{expected_gripper_hz:g} Hz的容差范围"
            )
        gripper_text = f"，夹爪{gripper_hz:.2f}Hz/最大年龄{max(gripper_ages):.3f}s"
    print(
        f"[通过] {path.name}: {frames}帧，{duration_s:.1f}s，"
        f"采样{measured_hz:.2f}Hz/间隔{min_interval_s * 1000:.1f}..{max_interval_s * 1000:.1f}ms，"
        f"底盘{update_hz:.2f}Hz，最大年龄{max(ages):.3f}s{gripper_text}"
    )
    return frames, duration_s


def main() -> int:
    all_completed = completed_episodes(settings.RAW_ROOT, successful_only=False)
    successful = completed_episodes(settings.RAW_ROOT, successful_only=True)
    if not successful:
        raise SystemExit("没有可转换的成功episode")
    total_frames = 0
    for episode in successful:
        frames, _ = validate_episode(episode)
        total_frames += frames
    print(
        f"校验完成：成功 {len(successful)} / 完整 {len(all_completed)} 条，"
        f"总帧数 {total_frames}；失败示教不会进入训练集。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
