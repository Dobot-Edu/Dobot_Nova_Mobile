#!/usr/bin/env python3
"""交互测试 Nova 信号灯：1红、2黄、3绿、0熄灭。"""

from __future__ import annotations

import argparse

from Dobot_Init.config import load_config
from Dobot_Init.indicator_light import IndicatorLight, LightColor
from Dobot_Init.robot import NovaRobot, RobotError


KEY_COLORS = {
    "1": (LightColor.RED, "红灯（DO6）"),
    "2": (LightColor.YELLOW, "黄灯（DO8）"),
    "3": (LightColor.GREEN, "绿灯（DO7）"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nova DO6/DO7/DO8 信号灯测试")
    parser.add_argument("--execute", action="store_true", help="允许实际写控制柜数字输出")
    parser.add_argument(
        "--enable-robot",
        action="store_true",
        help="灯不响应时可先使能机械臂；不会发送任何运动指令",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("必须显式添加 --execute 才会写数字输出")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    robot: NovaRobot | None = None
    light: IndicatorLight | None = None
    try:
        robot = NovaRobot(load_config("arm_only").robot)
        mode = robot.mode()
        if args.enable_robot and mode not in (5, 7):
            robot.prepare_for_control()
            mode = robot.mode()
        print(f"已连接控制柜，RobotMode={mode}；不会发送机械臂运动命令。")
        light = IndicatorLight(robot)
        light.off()
        print("输入 1=红色DO6，2=黄色DO8，3=绿色DO7，0=熄灭，q=退出。")
        while True:
            key = input("灯光> ").strip().lower()
            if key == "q":
                break
            if key == "0":
                light.off()
                print("已熄灭 DO6/DO7/DO8")
                continue
            selection = KEY_COLORS.get(key)
            if selection is None:
                print("无效输入，请输入 0、1、2、3 或 q")
                continue
            color, label = selection
            light.set(color)
            print(f"已设置：{label}")
        return 0
    except (OSError, RobotError, ValueError) as error:
        print(f"信号灯测试失败：{error}")
        return 1
    finally:
        if light is not None:
            try:
                light.off()
            except (OSError, RobotError):
                print("警告：退出时无法熄灭 DO6/DO7/DO8")
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
