#!/usr/bin/env python3
"""仅将升降台回零到 DI2 物理最高限位，不执行其他测试轨迹。"""

from __future__ import annotations

import argparse
import sys

from Dobot_Init.config import load_config
from Dobot_Init.lift import LiftColumn, LiftDirection, LiftError, ModbusTcpClient
from Dobot_Init.robot import NovaRobot, RobotError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="升降台单次物理最高点回零",
        allow_abbrev=False,
    )
    parser.add_argument("--execute", action="store_true", help="明确允许升降台实机运动")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("这是实机运动脚本，必须显式添加 --execute")
    return args


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    answer = input(
        "升降台将持续物理上升，直到机械臂控制柜 DI2 最高限位触发。\n"
        "确认升降台上方全程无障碍、人员远离且急停触手可及后，输入 HOME："
    ).strip()
    if answer != "HOME":
        print("输入不匹配，已取消；未发送运动命令。")
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
        print("[升降台回零] 正在向物理最高点运动；DI2触发后自动结束……")
        moving = True
        lift.lifting_columns("columns_reste", 0.0, 0, home_signal=robot.digital_input)
        moving = False
        # 回零流程结束后恢复绝对位置模式，并明确保持新零点。
        lift.lifting_columns("columns_updomn", 0.0, config.lift.drive_speed)
        print("[升降台回零] 完成：DI2已触发，新最高点坐标为0 cm。")
        return 0
    except KeyboardInterrupt:
        print("\n收到中断，正在尝试停止并保持当前位置……", file=sys.stderr)
        return 130
    except (ValueError, OSError, LiftError, RobotError) as error:
        print(f"升降台回零失败：{error}", file=sys.stderr)
        if "异常码=6" in str(error):
            print(
                "[安全停止] MAB 升降驱动器持续处于忙状态，回零命令未执行。"
                "请勿连续手工重启；检查驱动器报警和通信占用。",
                file=sys.stderr,
            )
        return 1
    finally:
        if moving and lift is not None:
            try:
                lift.stop_motion(LiftDirection.UP)
            except (LiftError, OSError) as error:
                print(f"[警告] 无法自动保持升降台：{error}", file=sys.stderr)
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
