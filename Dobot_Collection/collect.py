#!/usr/bin/env python3
"""按 DATASET_FPS 采集 Nova 移动操作示教；B键开始/结束当前 episode，正常结束全部保存。"""

from __future__ import annotations

import json
import itertools
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import constants as settings
from Dobot_Init.camera_preview import CameraPreview
from Dobot_Collection.devices import (
    DobotFeedback,
    TelemetryReceiver,
    TopHttpCamera,
    WristRealSense,
)
from Dobot_Collection.schema import (
    ACTION_NAMES,
    IMAGE_KEYS,
    STATE_NAMES,
    build_vectors,
    completed_episodes,
)


def require_fresh(source, name: str, max_age_s: float):
    if source.error is not None:
        raise RuntimeError(f"{name}后台线程失败：{source.error}")
    item = source.latest.get()
    if item is None:
        raise RuntimeError(f"{name}尚无数据")
    age_s = time.monotonic() - item.monotonic
    if age_s > max_age_s:
        raise RuntimeError(f"{name}数据过期：{age_s:.3f}s > {max_age_s:.3f}s")
    return item


class SessionShutdown(Exception):
    pass


def read_command(connection):
    if connection.poll():
        command = connection.recv()
        if command == "shutdown":
            raise SessionShutdown()
        return command
    return None


def make_sources() -> dict:
    return {
        "feedback": DobotFeedback(settings.ROBOT_IP, settings.ROBOT_FEEDBACK_PORT),
        "top": TopHttpCamera(settings.TOP_CAMERA_URL, settings.CAMERA_TIMEOUT_S),
        "wrist": WristRealSense(
            settings.WRIST_CAMERA_SERIAL,
            settings.CAMERA_WIDTH,
            settings.CAMERA_HEIGHT,
            settings.CAMERA_STREAM_FPS,
        ),
        "telemetry": TelemetryReceiver(
            settings.DATASET_TELEMETRY_HOST,
            settings.DATASET_TELEMETRY_PORT,
        ),
    }


def wait_until_ready(sources: dict, timeout_s: float = 20.0) -> None:
    print("等待机械臂、Top、腕部相机、VR及底盘遥测就绪……")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        failures = [f"{name}: {source.error}" for name, source in sources.items() if source.error]
        if failures:
            raise RuntimeError("；".join(failures))
        if all(source.latest.get() is not None for source in sources.values()):
            telemetry = sources["telemetry"].latest.get().value
            if telemetry.get("chassis_feedback_valid"):
                return
        time.sleep(0.05)
    missing = [name for name, source in sources.items() if source.latest.get() is None]
    raise RuntimeError(f"数据源准备超时；缺少={missing}，请确认新版VR遥操正在运行")


def record_episode(sources: dict, connection) -> tuple[Path, int, bool]:
    stem = datetime.now().strftime("episode_%Y%m%d_%H%M%S_%f")
    partial = settings.RAW_ROOT / f".{stem}.partial"
    final = settings.RAW_ROOT / stem
    for camera in IMAGE_KEYS:
        (partial / "images" / camera).mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 2,
        "lerobot_target_version": "v3.0",
        "task": settings.TASK,
        "fps": settings.DATASET_FPS,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "image_keys": list(IMAGE_KEYS),
        "units": "SI: rad, m, m/s, rad/s; gripper position is 1=open, 0=closed",
        "base_alignment": {
            "feedback_hz": settings.CHASSIS_FEEDBACK_RATE_HZ,
            "method": "causal_zero_order_hold",
            "valid_age_s": settings.BASE_VALID_AGE_S,
            "abort_age_s": settings.BASE_ABORT_AGE_S,
        },
        "gripper_alignment": {
            "feedback_hz": 1.0 / settings.GRIPPER_FEEDBACK_INTERVAL_S,
            "observation_source": "servo_encoder",
            "position_semantics": "1=open, 0=closed within configured safe travel",
            "open_is_one": True,
            "valid_age_s": settings.GRIPPER_FEEDBACK_VALID_AGE_S,
            "abort_age_s": settings.GRIPPER_FEEDBACK_ABORT_AGE_S,
        },
        "fixed_lift": {
            "height_m": settings.FIXED_LIFT_HEIGHT_M,
            "included_in_policy": False,
            "reason": "物理高度固定；编码器绝对零点不可靠",
        },
        "success": None,
    }
    metadata_path = partial / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    period_s = 1.0 / settings.DATASET_FPS
    started = time.monotonic()
    frame_count = 0
    end_reason = "right_B_click"
    base_feedback_updates = 0
    previous_base_timestamp = None
    gripper_feedback_updates = 0
    previous_gripper_timestamp = None

    with (partial / "samples.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        connection.send(("recording", "已开录；再次点按 B 结束并保存"))
        for index in itertools.count():
            if read_command(connection) == "stop":
                break
            target = started + index * period_s
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            sample_monotonic_ns = time.monotonic_ns()

            feedback_item = require_fresh(
                sources["feedback"], "机械臂反馈", settings.ROBOT_FEEDBACK_TIMEOUT_S
            )
            top_item = require_fresh(sources["top"], "Top相机", settings.CAMERA_TIMEOUT_S)
            wrist_item = require_fresh(sources["wrist"], "腕部相机", settings.CAMERA_TIMEOUT_S)
            telemetry_item = require_fresh(
                sources["telemetry"], "VR遥测", settings.TELEMETRY_TIMEOUT_S
            )
            telemetry = telemetry_item.value
            state, action, base, gripper_diagnostic = build_vectors(
                feedback_item.value["actual_deg"],
                telemetry,
                sample_monotonic_ns,
                base_valid_age_s=settings.BASE_VALID_AGE_S,
                gripper_valid_age_s=settings.GRIPPER_FEEDBACK_VALID_AGE_S,
            )
            if base["feedback_age_s"] > settings.BASE_ABORT_AGE_S:
                raise RuntimeError(
                    f"底盘反馈已中断{base['feedback_age_s']:.3f}s，超过安全阈值，停止采集"
                )
            if base["feedback_monotonic_ns"] != previous_base_timestamp:
                base_feedback_updates += 1
                previous_base_timestamp = base["feedback_monotonic_ns"]
            if gripper_diagnostic["feedback_age_s"] > settings.GRIPPER_FEEDBACK_ABORT_AGE_S:
                raise RuntimeError(
                    f"夹爪反馈已中断{gripper_diagnostic['feedback_age_s']:.3f}s，"
                    "超过安全阈值，停止采集"
                )
            if gripper_diagnostic["feedback_monotonic_ns"] != previous_gripper_timestamp:
                gripper_feedback_updates += 1
                previous_gripper_timestamp = gripper_diagnostic["feedback_monotonic_ns"]

            image_paths = {}
            for camera, item in (("top", top_item), ("wrist", wrist_item)):
                relative = Path("images") / camera / f"{index:06d}.jpg"
                image = item.value
                if image.shape != (settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH, 3):
                    raise RuntimeError(f"{camera}图像尺寸异常：{image.shape}")
                if not cv2.imwrite(
                    str(partial / relative), image, [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
                ):
                    raise RuntimeError(f"写入{camera}图像失败")
                image_paths[camera] = relative.as_posix()

            row = {
                "index": index,
                "timestamp_ns": time.time_ns(),
                "sample_monotonic_ns": sample_monotonic_ns,
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "images": image_paths,
                "base_diagnostic": base,
                "gripper_diagnostic": gripper_diagnostic,
                "source_age_ms": {
                    "feedback": (time.monotonic() - feedback_item.monotonic) * 1000.0,
                    "top": (time.monotonic() - top_item.monotonic) * 1000.0,
                    "wrist": (time.monotonic() - wrist_item.monotonic) * 1000.0,
                    "telemetry": (time.monotonic() - telemetry_item.monotonic) * 1000.0,
                },
                "arm_active": bool(telemetry.get("arm_active", False)),
                "tracking_valid": bool(telemetry.get("tracking_valid", False)),
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            frame_count = index + 1

            if frame_count % settings.DATASET_FPS == 0:
                elapsed = time.monotonic() - started
                print(
                    f"已录制 {elapsed:.0f}s；"
                    f"底盘/夹爪新反馈 {base_feedback_updates}/{gripper_feedback_updates} 次；"
                    "点按B结束；双击A仅回位"
                )

    recording_duration_s = time.monotonic() - started
    connection.send(("saving", "录制结束，正在保存；此时按 B 不排队开录"))
    success = True
    metadata.update(
        {
            "success": success,
            "acceptance": "all_manual_B_stops_saved_without_success_review",
            "frame_count": frame_count,
            "duration_s": recording_duration_s,
            "end_reason": end_reason,
            "base_feedback_updates": base_feedback_updates,
            "gripper_feedback_updates": gripper_feedback_updates,
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (partial / "COMPLETE").write_text("ok\n", encoding="utf-8")
    partial.rename(final)
    return final, frame_count, success


def check_sources_ready(sources):
    feedback = require_fresh(sources["feedback"], "机械臂", settings.ROBOT_FEEDBACK_TIMEOUT_S)
    require_fresh(sources["top"], "Top", settings.CAMERA_TIMEOUT_S)
    require_fresh(sources["wrist"], "Wrist", settings.CAMERA_TIMEOUT_S)
    telemetry = require_fresh(sources["telemetry"], "遥测", settings.TELEMETRY_TIMEOUT_S).value
    _, _, base, gripper = build_vectors(
        feedback.value["actual_deg"], telemetry, time.monotonic_ns(),
        base_valid_age_s=settings.BASE_VALID_AGE_S,
        gripper_valid_age_s=settings.GRIPPER_FEEDBACK_VALID_AGE_S)
    if base["feedback_age_s"] > settings.BASE_ABORT_AGE_S:
        raise RuntimeError("底盘反馈过期")
    if gripper["feedback_age_s"] > settings.GRIPPER_FEEDBACK_ABORT_AGE_S:
        raise RuntimeError("夹爪反馈过期")


def run_session(connection):
    """由遥操入口创建的独立采集进程；无终端输入，无时长/条数上限。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sources = {}
    preview = None
    try:
        settings.RAW_ROOT.mkdir(parents=True, exist_ok=True)
        sources = make_sources()
        for source in sources.values():
            source.start()
        preview = CameraPreview(sources, 'Nova 遥操 / 数采')
        preview.start()
        wait_until_ready(sources)
        check_sources_ready(sources)
        connection.send(("ready", "相机和反馈就绪；点按 B 立即开录"))
        while True:
            if connection.poll(0.05):
                command = read_command(connection)
                if command == "start":
                    check_sources_ready(sources)
                    path, frames, _ = record_episode(sources, connection)
                    connection.send(("ready", f"已保存 {path.name}，{frames} 帧；点按 B 开始下一条"))
    except (SessionShutdown, EOFError, BrokenPipeError):
        print("[数采退出] 未正常按 B 保存的当前目录保留为 .partial", flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        try:
            connection.send(("error", f"{error}；未完成数据保留 .partial，请重启入口"))
        except (EOFError, OSError):
            pass
    finally:
        if preview is not None:
            preview.close()
        for source in sources.values():
            source.close()
        connection.close()


def main() -> int:
    print("采集已整合到 run_teleop.sh；请启动 full/arm_only，单击 A 初始化，B 开始/结束采集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
