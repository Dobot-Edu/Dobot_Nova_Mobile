#!/usr/bin/env python3
"""只读显示底盘动作和反馈，首次采集前用于确认字段、频率和正负号。"""

from __future__ import annotations

import time

import constants as settings
from Dobot_Collection.devices import TelemetryReceiver


MONITOR_SECONDS = 15.0


def main() -> int:
    receiver = TelemetryReceiver(
        settings.DATASET_TELEMETRY_HOST,
        settings.DATASET_TELEMETRY_PORT,
    )
    receiver.start()
    started = time.monotonic()
    previous_feedback_ns = 0
    first_feedback_ns = 0
    last_feedback_ns = 0
    updates = 0
    try:
        print("监听15秒。请缓慢前进、后退、左转、右转各一次，然后回中停车。")
        while time.monotonic() - started < MONITOR_SECONDS:
            if receiver.error is not None:
                raise RuntimeError(receiver.error)
            item = receiver.latest.get()
            if item is None:
                time.sleep(0.02)
                continue
            data = item.value
            feedback_ns = int(data.get("chassis_feedback_monotonic_ns", 0))
            if feedback_ns and feedback_ns != previous_feedback_ns:
                previous_feedback_ns = feedback_ns
                if first_feedback_ns == 0:
                    first_feedback_ns = feedback_ns
                last_feedback_ns = feedback_ns
                updates += 1
                print(
                    f"cmd=({float(data['chassis_command_linear_mm_s']):7.1f} mm/s, "
                    f"{float(data['chassis_command_angular_mdeg_s']) / 1000:7.2f} deg/s)  "
                    f"actual=({float(data['chassis_feedback_speed_mm_s']):7.1f} mm/s, "
                    f"{float(data['chassis_feedback_angular_mdeg_s']) / 1000:7.2f} deg/s)  "
                    f"position={data.get('chassis_feedback_position_mm')}"
                )
            time.sleep(0.01)
    finally:
        receiver.close()
    # 只统计第一条到最后一条有效反馈之间的间隔，避免遥操尚未启动、等待用户确认
    # 或首次连接耗时被错误计入分母，造成20 Hz被误报为5 Hz。
    if updates < 2 or last_feedback_ns <= first_feedback_ns:
        raise RuntimeError(f"有效底盘反馈不足：仅收到{updates}次更新")
    feedback_elapsed = (last_feedback_ns - first_feedback_ns) / 1e9
    update_hz = (updates - 1) / feedback_elapsed
    expected_hz = settings.CHASSIS_FEEDBACK_RATE_HZ
    lower_hz = expected_hz * 0.70
    upper_hz = expected_hz * 1.20
    if not lower_hz <= update_hz <= upper_hz:
        raise RuntimeError(
            f"底盘反馈更新率异常：{update_hz:.2f} Hz；"
            f"配置目标={expected_hz:.2f} Hz"
        )
    print(f"底盘反馈更新率：{update_hz:.2f} Hz，符合{expected_hz:g} Hz配置范围。")
    print("请人工确认：后退时cmd线速度为负，左右转cmd/actual角速度方向一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
