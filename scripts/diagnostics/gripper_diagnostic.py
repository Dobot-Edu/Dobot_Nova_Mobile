"""只测试夹爪电流限力：闭合、短暂保持，然后自动张开。"""

from __future__ import annotations

import time

from Dobot_Init.config import load_config
from Dobot_Init.gripper import GripperError, TriggerGripper, position_command


def main() -> int:
    raise RuntimeError(
        "旧电流限力验收脚本已停用：不适配当前连续位置控制；"
        "请使用 python -m scripts.gripper_calibrate --status 查看反馈"
    )
    # 保留旧实现供参考；在重新设计验收逻辑前禁止驱动硬件。
    config = load_config("arm_only").gripper
    gripper = TriggerGripper(config, execute=True)
    try:
        gripper.connect()
        # 首帧只建立扳机安全基准，不产生运动。
        gripper.update(0.0, input_valid=True)
        gripper.update(1.0, input_valid=True)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not gripper._force_latched:
            gripper.update(1.0, input_valid=True)
            feedback = gripper._last_feedback
            if feedback is not None:
                print(
                    f"[反馈] 电流={feedback.current_ma:.0f} mA "
                    f"位置={feedback.position_raw} 温度={feedback.temperature_c}°C "
                    f"moving={feedback.moving}"
                )
            time.sleep(config.feedback_interval_s)

        if not gripper._force_latched:
            feedback = gripper._read_feedback()
            hold = min(
                position_command(feedback.position_raw, config),
                config.max_close_command,
            )
            gripper._move(hold, torque=config.hold_torque)
            print(
                f"[结果] 5 秒内未达到 {config.current_limit_ma:.0f} mA；"
                f"已主动停止在 {hold}/255"
            )
        else:
            print(
                f"[结果] 限力成功：停止位置={gripper.last_command}/255，"
                f"触发电流约={gripper.last_current_ma:.0f} mA"
            )

        print("[保持] 观察 2 秒")
        hold_deadline = time.monotonic() + 2.0
        while time.monotonic() < hold_deadline:
            feedback = gripper._read_feedback()
            print(
                f"[保持反馈] 电流={feedback.current_ma:.0f} mA "
                f"温度={feedback.temperature_c}°C"
            )
            time.sleep(0.25)

        print("[释放] 张开夹爪")
        gripper.update(0.0, input_valid=True)
        time.sleep(2.0)
        feedback = gripper._read_feedback()
        print(
            f"[完成] 电流={feedback.current_ma:.0f} mA "
            f"位置={feedback.position_raw} 温度={feedback.temperature_c}°C"
        )
        return 0
    except (GripperError, Exception) as error:
        print(f"[异常] {error}")
        if gripper._serial is not None:
            try:
                # 异常时首先关闭舵机力矩，防止继续夹紧。
                gripper._write(40, bytes([0]), allowed_errors=tuple(range(256)))
                print("[保护] 已关闭夹爪力矩")
            except Exception as stop_error:
                print(f"[保护失败] {stop_error}")
        return 1
    finally:
        gripper.close()


if __name__ == "__main__":
    raise SystemExit(main())
