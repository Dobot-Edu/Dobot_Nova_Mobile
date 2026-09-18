"""Quest 右手 JSON 数据包的严格解析。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VRPacket:
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    grip: float
    trigger: float
    button_a: bool
    button_b: bool
    thumbstick_xy: np.ndarray
    tracking_valid: bool
    timestamp_s: float
    head_position_m: np.ndarray | None = None
    head_quaternion_xyzw: np.ndarray | None = None
    head_tracking_valid: bool = False
    protocol_version: int = 1

    def deadman(self, threshold: float) -> bool:
        return self.tracking_valid and self.grip >= threshold


def _vector(message: dict, key: str, count: int) -> np.ndarray:
    value = np.asarray(message[key], dtype=float)
    if value.shape != (count,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} 必须包含 {count} 个有限数值")
    return value


def decode_packet(data: bytes) -> VRPacket:
    if len(data) > 4096:
        raise ValueError("数据包过大")
    try:
        message = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JSON 无效") from error
    if not isinstance(message, dict):
        raise ValueError("JSON 根节点必须是对象")

    position = _vector(message, "right_pos", 3)
    quaternion = _vector(message, "right_rot", 4)
    norm = float(np.linalg.norm(quaternion))
    if not 0.5 <= norm <= 1.5:
        raise ValueError("四元数模长异常")
    grip = float(message.get("right_grip", 0.0))
    trigger = float(message.get("right_trigger", 0.0))
    timestamp = float(message.get("timestamp", 0.0))
    thumbstick = np.asarray(message.get("right_thumbstick", [0.0, 0.0]), dtype=float)
    if not all(np.isfinite(value) for value in (grip, trigger, timestamp)):
        raise ValueError("按键或时间戳不是有限数值")
    if not 0.0 <= grip <= 1.0 or not 0.0 <= trigger <= 1.0:
        raise ValueError("Grip/Trigger 必须在 [0, 1]")
    if thumbstick.shape != (2,) or not np.all(np.isfinite(thumbstick)):
        raise ValueError("right_thumbstick 必须包含 2 个有限数值")
    if np.any(np.abs(thumbstick) > 1.05):
        raise ValueError("right_thumbstick 必须在 [-1, 1]")

    protocol_version = int(message.get("protocol_version", 1))
    if protocol_version not in (1, 2):
        raise ValueError(f"不支持的 Quest 协议版本：{protocol_version}")
    has_head_position = "head_pos" in message
    has_head_rotation = "head_rot" in message
    if has_head_position != has_head_rotation:
        raise ValueError("head_pos 和 head_rot 必须同时出现")
    head_position = None
    head_quaternion = None
    head_tracking_valid = False
    if has_head_position:
        head_position = _vector(message, "head_pos", 3)
        head_quaternion = _vector(message, "head_rot", 4)
        head_norm = float(np.linalg.norm(head_quaternion))
        if not 0.5 <= head_norm <= 1.5:
            raise ValueError("头显四元数模长异常")
        head_quaternion = head_quaternion / head_norm
        head_tracking_valid = bool(message.get("head_tracking_valid", False))

    return VRPacket(
        position_m=position,
        quaternion_xyzw=quaternion / norm,
        grip=grip,
        trigger=trigger,
        button_a=bool(message.get("button_a", False)),
        button_b=bool(message.get("button_b", False)),
        thumbstick_xy=np.clip(thumbstick, -1.0, 1.0),
        tracking_valid=bool(message.get("tracking_valid", False)),
        timestamp_s=timestamp,
        head_position_m=head_position,
        head_quaternion_xyzw=head_quaternion,
        head_tracking_valid=head_tracking_valid,
        protocol_version=protocol_version,
    )
