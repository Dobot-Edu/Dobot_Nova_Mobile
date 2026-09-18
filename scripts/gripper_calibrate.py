#!/usr/bin/env python3
"""交互式标定夹爪安全行程端点，不自动改写工程配置。"""

from __future__ import annotations

import argparse
import time

from Dobot_Init.config import load_config
from Dobot_Init.gripper import GripperError, TriggerGripper


STEP_SMALL = 5
STEP_LARGE = 20
CALIBRATION_TORQUE = 150
SETTLE_S = 0.35


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nova夹爪0/1位置端点标定")
    parser.add_argument("--execute", action="store_true", help="允许夹爪实机小步运动")
    parser.add_argument("--status", action="store_true", help="只读一次位置、电流和温度")
    args = parser.parse_args(argv)
    if not args.execute and not args.status:
        parser.error("标定需添加 --execute；只读检查可添加 --status")
    return args


def wait_feedback(gripper: TriggerGripper) -> dict:
    time.sleep(SETTLE_S)
    gripper.check_error()
    feedback = gripper.feedback_snapshot()
    if not feedback["feedback_valid"] or feedback["position_raw"] is None:
        raise GripperError("未取得有效编码器反馈")
    return feedback


def show(feedback: dict, target_raw: int) -> None:
    print(
        f"目标raw={target_raw}  实测raw={feedback['position_raw']}  "
        f"电流={feedback['current_ma']:.0f}mA  "
        f"温度={feedback['temperature_c']}°C  moving={feedback['moving']}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config("arm_only").gripper
    if args.status:
        gripper = TriggerGripper(config, execute=True)
        try:
            feedback = gripper.connect_read_only()
            print(
                f"夹爪只读状态：raw={feedback.position_raw} "
                f"电流={feedback.current_ma:.0f}mA "
                f"温度={feedback.temperature_c}°C moving={feedback.moving}"
            )
            return 0
        except (GripperError, OSError) as error:
            print(f"[只读失败] {error}")
            return 1
        finally:
            gripper.close()

    answer = input(
        "本测试只移动夹爪。请移除瓶子和障碍物，保持手指远离夹爪，急停可用。\n"
        "确认后输入 CALIBRATE："
    ).strip()
    if answer != "CALIBRATE":
        print("输入不匹配，已取消；未连接和移动夹爪。")
        return 1

    gripper = TriggerGripper(config, execute=True)
    open_endpoint: tuple[int, int] | None = None
    close_endpoint: tuple[int, int] | None = None
    try:
        # 关键：不执行旧的 raw_open，连接后保持当前位置。
        gripper.connect(initialize_open=False)
        feedback = wait_feedback(gripper)
        target_raw = int(feedback["position_raw"])
        show(feedback, target_raw)
        print(
            "命令：o/O=向张开方向+5/+20，c/C=向闭合方向-5/-20；\n"
            "      0=记录当前开端的命令/实测值，1=记录当前闭端的命令/实测值；\n"
            "      p=显示结果，q=退出。电流和温度只显示，不会触发软件停止。"
        )

        while True:
            command = input("标定> ").strip()
            delta = {"o": STEP_SMALL, "O": STEP_LARGE, "c": -STEP_SMALL, "C": -STEP_LARGE}.get(command)
            if delta is not None:
                target_raw = min(max(target_raw + delta, 900), 2150)
                gripper.move_raw_for_calibration(target_raw, torque=CALIBRATION_TORQUE)
                feedback = wait_feedback(gripper)
                show(feedback, target_raw)
                continue
            if command == "0":
                feedback = gripper.feedback_snapshot()
                open_endpoint = (target_raw, int(feedback["position_raw"]))
                print(
                    f"已记录 0=安全全开：命令raw={open_endpoint[0]}，"
                    f"反馈raw={open_endpoint[1]}"
                )
                continue
            if command == "1":
                feedback = gripper.feedback_snapshot()
                close_endpoint = (target_raw, int(feedback["position_raw"]))
                print(
                    f"已记录 1=安全全闭：命令raw={close_endpoint[0]}，"
                    f"反馈raw={close_endpoint[1]}"
                )
                continue
            if command == "p":
                if open_endpoint is None or close_endpoint is None:
                    print(f"尚未完成：open={open_endpoint}, close={close_endpoint}")
                elif (
                    open_endpoint[0] <= close_endpoint[0]
                    or open_endpoint[1] <= close_endpoint[1]
                ):
                    print("端点方向异常：本夹爪应满足 open_raw > close_raw")
                else:
                    print(
                        "建议写入 constants.py：\n"
                        f"GRIPPER_RAW_OPEN = {open_endpoint[0]}\n"
                        f"GRIPPER_RAW_CLOSE = {close_endpoint[0]}\n"
                        "GRIPPER_MAX_CLOSE_COMMAND = 255\n"
                        f"GRIPPER_FEEDBACK_RAW_OPEN = {open_endpoint[1]}\n"
                        f"GRIPPER_FEEDBACK_RAW_CLOSE = {close_endpoint[1]}"
                    )
                continue
            if command == "q":
                return 0
            print("未知命令；请输入 o/O/c/C/0/1/p/q")
    except (GripperError, OSError) as error:
        print(f"[标定停止] {error}")
        return 1
    finally:
        try:
            gripper.disable_torque()
            print("[安全退出] 已关闭夹爪力矩；标定结果未自动写入配置。")
        except (GripperError, OSError):
            pass
        gripper.close()


if __name__ == "__main__":
    raise SystemExit(main())
