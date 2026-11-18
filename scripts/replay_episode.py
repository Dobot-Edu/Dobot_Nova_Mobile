#!/usr/bin/env python3
"""按 episode 原始采样频率在 Nova 真机复现一条示教动作；默认只检查，不连接硬件。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import devices, paths
from Dobot_Collection.schema import ACTION_NAMES, episode_vectors, read_jsonl
from Dobot_Inference.chassis import InferenceChassis
from Dobot_Init.chassis_rpc import DEFAULT_PASSWORD, DEFAULT_USERNAME
from Dobot_Init.config import load_config
from Dobot_Init.gripper import TriggerGripper
from Dobot_Init.robot import NovaRobot, RobotError


START_TOLERANCE_DEG = 3.0
STATUS_CHECK_INTERVAL_S = 1.0


@dataclass(frozen=True)
class Episode:
    path: Path
    fps: float
    actions: np.ndarray
    invalid_gripper_frames: tuple[int, ...]
    gripper_open_is_one: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复现 Nova 原始 episode 的机械臂、夹爪和底盘动作；默认dry-run"
    )
    parser.add_argument(
        "episode",
        nargs="?",
        help="episode目录、目录名或latest；省略时选择最新成功episode",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="允许真机执行；仍需现场输入REPLAY并确认底盘模式",
    )
    return parser.parse_args(argv)


def resolve_episode(value: str | None, raw_root: Path = paths.RAW_ROOT) -> Path:
    if value and value != "latest":
        candidate = Path(value).expanduser()
        if not candidate.is_absolute() and not candidate.is_dir():
            candidate = raw_root / candidate
        return candidate.resolve()
    candidates = sorted(
        path
        for path in raw_root.glob("episode_*")
        if (path / "COMPLETE").is_file()
        and (path / "metadata.json").is_file()
        and json.loads((path / "metadata.json").read_text(encoding="utf-8")).get("success") is True
    )
    if not candidates:
        raise ValueError(f"{raw_root} 中没有成功且完整的episode")
    return candidates[-1].resolve()


def load_episode(path: Path) -> Episode:
    if not path.is_dir() or not (path / "COMPLETE").is_file():
        raise ValueError(f"episode不完整或不存在：{path}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("success") is not True:
        raise ValueError(f"episode未标记成功，拒绝复现：{path.name}")
    fps = float(metadata.get("fps", 0))
    if fps <= 0:
        raise ValueError(f"episode fps无效：{fps}")

    actions: list[np.ndarray] = []
    invalid_gripper: list[int] = []
    previous_timestamp = 0
    for expected_index, row in enumerate(read_jsonl(path / "samples.jsonl")):
        if row.get("index") != expected_index:
            raise ValueError(
                f"第{expected_index}帧序号不连续：实际={row.get('index')}"
            )
        timestamp = int(row.get("timestamp_ns", 0))
        if timestamp <= previous_timestamp:
            raise ValueError(f"第{expected_index}帧时间戳未递增")
        previous_timestamp = timestamp
        _, action = episode_vectors(
            row["observation.state"],
            row["action"],
            int(metadata.get("schema_version", 1)),
        )
        actions.append(action.astype(np.float64))
        if not bool(row.get("gripper_diagnostic", {}).get("feedback_valid", True)):
            invalid_gripper.append(expected_index)

    if not actions:
        raise ValueError("episode没有动作帧")
    result = np.stack(actions)
    if result.shape != (len(actions), len(ACTION_NAMES)) or not np.all(np.isfinite(result)):
        raise ValueError(f"动作数组无效：shape={result.shape}")
    if int(metadata.get("frame_count", len(result))) != len(result):
        raise ValueError(
            f"metadata帧数{metadata.get('frame_count')}与JSONL帧数{len(result)}不一致"
        )
    if np.any((result[:, 6] < 0.0) | (result[:, 6] > 1.0)):
        raise ValueError("夹爪action超出0..1")
    gripper_open_is_one = bool(
        metadata.get("gripper_alignment", {}).get("open_is_one", False)
    )
    return Episode(
        path, fps, result, tuple(invalid_gripper), gripper_open_is_one
    )


def print_summary(episode: Episode) -> None:
    actions = episode.actions
    duration = len(actions) / episode.fps
    joint_step = (
        float(np.max(np.abs(np.diff(np.rad2deg(actions[:, :6]), axis=0))))
        if len(actions) > 1
        else 0.0
    )
    print(f"[轨迹] {episode.path}")
    print(f"[轨迹] {len(actions)}帧，{episode.fps:g}Hz，{duration:.2f}s")
    print(
        f"[轨迹] 最大相邻关节目标变化={joint_step:.2f}°，"
        f"夹爪={actions[:, 6].min():.3f}..{actions[:, 6].max():.3f}"
        f"（{'1=张开' if episode.gripper_open_is_one else '1=闭合'}），"
        f"底盘线速度={actions[:, 7].min():+.3f}..{actions[:, 7].max():+.3f}m/s，"
        f"角速度={actions[:, 8].min():+.3f}..{actions[:, 8].max():+.3f}rad/s"
    )
    if episode.invalid_gripper_frames:
        print(
            "[轨迹提醒] 原始数据存在夹爪反馈短暂失效帧："
            f"{list(episode.invalid_gripper_frames)}；replay使用已保存action，不受影响"
        )


def make_chassis(config) -> InferenceChassis:
    return InferenceChassis(
        url=config.url,
        username=DEFAULT_USERNAME,
        password=DEFAULT_PASSWORD,
        command_hz=devices.CHASSIS_COMMAND_RATE_HZ,
        feedback_hz=devices.CHASSIS_FEEDBACK_RATE_HZ,
        linear_limit_m_s=devices.CHASSIS_LINEAR_SPEED_MM_S / 1000.0,
        angular_limit_rad_s=np.deg2rad(devices.CHASSIS_ANGULAR_SPEED_MDEG_S / 1000.0),
        # replay直接复现数据中的速度目标，不再额外施加推理阶段的加速度滤波。
        linear_acceleration_m_s2=1000.0,
        angular_acceleration_rad_s2=1000.0,
        allow_localization_warning=config.allow_manual_localization_warning,
        rpc_timeout_s=config.rpc_timeout_s,
        mode_confirm_timeout_s=config.mode_confirm_timeout_s,
    )


def confirm_execution(episode: Episode) -> None:
    print(
        "\n即将复现整条真机轨迹，包括机械臂、夹爪和移动底盘。\n"
        "请将场景、底盘朝向和物体恢复到该示教的起始状态；确认工作区无人、"
        "急停可用，且没有其他遥操/推理程序。"
    )
    if input(f"确认复现 {episode.path.name} 后输入 REPLAY：").strip() != "REPLAY":
        raise RuntimeError("用户取消真机复现")


def execute_episode(episode: Episode) -> None:
    config = load_config("full")
    robot: NovaRobot | None = None
    gripper: TriggerGripper | None = None
    chassis = make_chassis(config.chassis)
    try:
        chassis.connect_read_only()
        robot = NovaRobot(config.robot)
        robot.prepare_for_control()
        robot.move_to_startup_pose()
        gripper = TriggerGripper(config.gripper, execute=True)
        gripper.connect()
        chassis.enable_remote()

        actual = robot.joints()
        start_error = float(
            np.max(np.abs(np.rad2deg(episode.actions[0, :6] - actual)))
        )
        if start_error > START_TOLERANCE_DEG:
            raise RuntimeError(
                f"回标准位后仍与轨迹首帧相差{start_error:.2f}° > "
                f"{START_TOLERANCE_DEG:.2f}°，拒绝复现"
            )

        print("[复现] 起始姿态检查通过，1秒后开始；Ctrl+C可停止")
        time.sleep(1.0)
        period = 1.0 / episode.fps
        started = time.monotonic()
        next_status = started
        for index, action in enumerate(episode.actions):
            target = started + index * period
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            robot.servo_j(action[:6])
            gripper_target = float(action[6]) if episode.gripper_open_is_one else 1.0 - float(action[6])
            gripper.set_target_position(gripper_target)
            chassis.set_velocity(float(action[7]), float(action[8]))

            now = time.monotonic()
            if now >= next_status:
                mode = robot.mode()
                if mode not in (5, 7):
                    raise RobotError(f"复现期间RobotMode={mode}")
                gripper.check_error()
                chassis.check_error()
                print(
                    f"[复现] {index + 1}/{len(episode.actions)} "
                    f"t={index / episode.fps:.1f}s grip={action[6]:.3f} "
                    f"base={action[7]:+.3f}/{action[8]:+.3f}"
                )
                next_status = now + STATUS_CHECK_INTERVAL_S
        chassis.stop()
        time.sleep(0.2)
        print("[复现] 整条轨迹执行完成，底盘已停车")
    finally:
        chassis.stop()
        chassis.close()
        if gripper is not None:
            gripper.close()
        if robot is not None:
            try:
                robot.request("Stop()")
            except RobotError:
                pass
            robot.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episode = load_episode(resolve_episode(args.episode))
    print_summary(episode)
    if not args.execute:
        print("[DRY-RUN] 数据检查完成，没有连接或移动任何硬件；加 --execute 才会真机复现")
        return 0
    confirm_execution(episode)
    execute_episode(episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
