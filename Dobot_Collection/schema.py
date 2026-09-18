"""原始 episode 和 LeRobot v3 特征的唯一字段定义。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STATE_NAMES = [
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper_position",
    "base_linear_velocity_m_s",
    "base_angular_velocity_rad_s",
]

ACTION_NAMES = [
    "joint_1_target_rad",
    "joint_2_target_rad",
    "joint_3_target_rad",
    "joint_4_target_rad",
    "joint_5_target_rad",
    "joint_6_target_rad",
    "gripper_target_position",
    "base_linear_velocity_target_m_s",
    "base_angular_velocity_target_rad_s",
]

IMAGE_KEYS = ("top", "wrist")


def validate_training_metadata(metadata: dict) -> None:
    """拒绝把旧夹爪语义或不同字段布局静默混进新训练集。"""
    if int(metadata.get("schema_version", 0)) != 2:
        raise ValueError(f"仅支持原始schema_version=2，实际为{metadata.get('schema_version')}")
    if metadata.get("state_names") != STATE_NAMES:
        raise ValueError("episode 的 state_names 与当前9维定义不一致")
    if metadata.get("action_names") != ACTION_NAMES:
        raise ValueError("episode 的 action_names 与当前9维定义不一致")
    if tuple(metadata.get("image_keys", ())) != IMAGE_KEYS:
        raise ValueError("episode 的 image_keys 必须为 top、wrist")
    alignment = metadata.get("gripper_alignment", {})
    if alignment.get("open_is_one") is not True:
        raise ValueError("拒绝旧夹爪语义；训练数据必须明确声明1=张开、0=闭合")


def finite_vector(values: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是{size}个有限数值，实际为{result!r}")
    return result


def state_vector(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    return finite_vector(result, len(STATE_NAMES), "observation.state")


def action_vector(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    return finite_vector(result, len(ACTION_NAMES), "action")


def episode_vectors(values_state: object, values_action: object, schema_version: int):
    """仅接受schema v2的9维状态和动作；夹爪1开、0闭。"""
    if int(schema_version) != 2:
        raise ValueError(f"仅支持原始schema_version=2，实际为{schema_version}；不再转换旧数据")
    state = state_vector(values_state).copy()
    action = action_vector(values_action).copy()
    if not (0 <= state[6] <= 1 and 0 <= action[6] <= 1):
        raise ValueError("夹爪实测/目标位置必须在0..1")
    return state, action


def completed_episodes(raw_root: Path, *, successful_only: bool = True) -> list[Path]:
    episodes = sorted(path for path in raw_root.glob("episode_*") if (path / "COMPLETE").is_file())
    if not successful_only:
        return episodes
    return [
        path
        for path in episodes
        if json.loads((path / "metadata.json").read_text(encoding="utf-8")).get("success") is True
    ]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} JSON损坏") from error


def build_vectors(
    robot_actual_deg: object,
    telemetry: dict,
    sample_monotonic_ns: int,
    *,
    base_valid_age_s: float,
    gripper_valid_age_s: float,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """把原始设备单位转换为 SI；底盘反馈使用最近一次因果值。"""
    robot_actual_rad = np.deg2rad(finite_vector(robot_actual_deg, 6, "robot_actual_deg"))
    robot_target_rad = np.deg2rad(
        finite_vector(telemetry.get("joints_command_deg"), 6, "joints_command_deg")
    )
    feedback_ns = int(telemetry.get("chassis_feedback_monotonic_ns", 0))
    reported_valid = bool(telemetry.get("chassis_feedback_valid", False))
    if feedback_ns <= 0:
        raise ValueError("尚未收到底盘真实反馈时间戳")
    age_s = max(0.0, (sample_monotonic_ns - feedback_ns) / 1e9)
    valid = reported_valid and age_s <= base_valid_age_s

    gripper = float(telemetry["gripper_position"])
    gripper_target = float(telemetry["gripper_target_position"])
    if not 0.0 <= gripper <= 1.0 or not 0.0 <= gripper_target <= 1.0:
        raise ValueError(f"夹爪实测/目标位置超出0..1：{gripper}/{gripper_target}")
    gripper_feedback_ns = int(telemetry.get("gripper_feedback_monotonic_ns", 0))
    if gripper_feedback_ns <= 0:
        raise ValueError("尚未收到夹爪编码器反馈时间戳")
    gripper_age_s = max(0.0, (sample_monotonic_ns - gripper_feedback_ns) / 1e9)
    gripper_valid = bool(telemetry.get("gripper_feedback_valid", False)) and (
        gripper_age_s <= gripper_valid_age_s
    )
    base_speed_m_s = float(telemetry["chassis_feedback_speed_mm_s"]) / 1000.0
    base_angular_rad_s = np.deg2rad(
        float(telemetry["chassis_feedback_angular_mdeg_s"]) / 1000.0
    )
    command_m_s = float(telemetry["chassis_command_linear_mm_s"]) / 1000.0
    command_rad_s = np.deg2rad(
        float(telemetry["chassis_command_angular_mdeg_s"]) / 1000.0
    )

    state = np.asarray(
        [
            *robot_actual_rad,
            gripper,
            base_speed_m_s,
            base_angular_rad_s,
        ],
        dtype=np.float32,
    )
    action = np.asarray(
        [*robot_target_rad, gripper_target, command_m_s, command_rad_s],
        dtype=np.float32,
    )
    diagnostic = {
        "feedback_monotonic_ns": feedback_ns,
        "feedback_age_s": age_s,
        "feedback_valid": valid,
        "position_m": [
            float(value) / 1000.0
            for value in telemetry.get("chassis_feedback_position_mm", [0.0, 0.0])
        ],
        "yaw_rad": float(
            np.deg2rad(float(telemetry.get("chassis_feedback_yaw_mdeg", 0.0)) / 1000.0)
        ),
        "odom_m": float(telemetry.get("chassis_feedback_odom_mm", 0.0)) / 1000.0,
    }
    gripper_diagnostic = {
        "position_raw": telemetry.get("gripper_position_raw"),
        "position": gripper,
        "target_position": gripper_target,
        "current_ma": telemetry.get("gripper_current_ma"),
        "load_raw": telemetry.get("gripper_load_raw"),
        "temperature_c": telemetry.get("gripper_temperature_c"),
        "moving": bool(telemetry.get("gripper_moving", False)),
        "status_error": telemetry.get("gripper_status_error"),
        "current_baseline_ma": telemetry.get("gripper_current_baseline_ma"),
        "current_threshold_ma": telemetry.get("gripper_current_threshold_ma"),
        "feedback_monotonic_ns": gripper_feedback_ns,
        "feedback_age_s": gripper_age_s,
        "feedback_valid": gripper_valid,
    }
    return state_vector(state), action_vector(action), diagnostic, gripper_diagnostic
