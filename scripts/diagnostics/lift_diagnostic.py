#!/usr/bin/env python3
"""不使用 VR，独立验收升降柱：回零、5 cm、10 cm、0 cm。"""

from __future__ import annotations

import argparse
import sys
import time

from Dobot_Init.config import load_config
from Dobot_Init.lift import LiftColumn, LiftError, ModbusTcpClient
from Dobot_Init.robot import NovaRobot, RobotError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="升降柱实机序列测试：回零 -> 5 cm -> 10 cm -> 0 cm"
    )
    parser.add_argument("--execute", action="store_true", help="明确允许升降柱实机运动")
    parser.add_argument("--tolerance-cm", type=float, default=0.2, help="到位误差，默认 0.2 cm")
    parser.add_argument("--timeout-s", type=float, default=90.0, help="每段运动超时，默认 90 秒")
    parser.add_argument("--dwell-s", type=float, default=2.0, help="每个目标到位后的停留时间")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("这是实机运动脚本，必须显式添加 --execute")
    if args.tolerance_cm <= 0 or args.timeout_s <= 0 or args.dwell_s < 0:
        parser.error("误差和超时必须为正数，停留时间不能为负数")
    return args


def confirm() -> bool:
    answer = input(
        "升降柱将回到最高点，然后依次运动到 5、10、0 cm。\n"
        "确认全行程无障碍、人员已远离、急停触手可及后，输入 LIFT 并按 Enter："
    ).strip()
    return answer == "LIFT"


def wait_until_reached(
    lift: LiftColumn,
    target_cm: float,
    *,
    tolerance_cm: float,
    timeout_s: float,
) -> float:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        actual = lift.read_height()
        error = abs(actual - target_cm)
        print(f"  目标={target_cm:6.1f} cm，实际={actual:8.3f} cm，误差={error:6.3f} cm")
        if error <= tolerance_cm:
            return actual
        time.sleep(0.5)
    raise LiftError(f"运动到 {target_cm:.1f} cm 超过 {timeout_s:.1f} 秒")


def hold_at_feedback(lift: LiftColumn) -> None:
    """异常或中断时，尽力读取实际高度并重设为保持目标。"""
    try:
        target = lift.stop_motion()
        print(f"[保持] 已把实际高度 {target:.3f} cm 设为保持目标")
    except (LiftError, OSError) as error:
        print(f"[警告] 无法下发当前位置保持：{error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not confirm():
        print("输入不匹配，已取消；未发送任何运动命令。")
        return 1

    robot: NovaRobot | None = None
    lift: LiftColumn | None = None
    moving = False
    try:
        config = load_config("test")
        robot = NovaRobot(config.robot)
        client = ModbusTcpClient(
            config.lift.host,
            config.lift.port,
            config.lift.unit_id,
            config.lift.socket_timeout_s,
        )
        lift = LiftColumn(config.lift, execute=True, client=client)

        print("\n[1/4] 初始化：回零到物理最高点")
        moving = True
        lift.lifting_columns("columns_reste", 0.0, 0, home_signal=robot.digital_input)
        print("  回零完成，高度变量已置为 0 cm")

        for index, target in enumerate((5.0, 10.0, 0.0), start=2):
            print(f"\n[{index}/4] 运动到 {target:.1f} cm")
            lift.lifting_columns("columns_updomn", target, config.lift.drive_speed)
            actual = wait_until_reached(
                lift,
                target,
                tolerance_cm=args.tolerance_cm,
                timeout_s=args.timeout_s,
            )
            print(f"  到位：{actual:.3f} cm")
            time.sleep(args.dwell_s)

        moving = False
        print("\n升降柱完整序列测试通过：回零 -> 5 cm -> 10 cm -> 0 cm")
        return 0
    except KeyboardInterrupt:
        print("\n收到中断，正在尝试保持当前位置……", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, OSError, LiftError, RobotError) as error:
        print(f"\n测试失败：{error}", file=sys.stderr)
        return 1
    finally:
        if moving and lift is not None:
            hold_at_feedback(lift)
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
