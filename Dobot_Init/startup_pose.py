"""单独完成 Nova 报警检查、使能和标准位初始化。"""

from __future__ import annotations

import numpy as np

from Dobot_Init.config import load_config
from Dobot_Init.robot import NovaRobot, RobotError


def main() -> int:
    config = load_config("arm_only").robot
    robot: NovaRobot | None = None
    try:
        robot = NovaRobot(config)
        snapshot = robot.snapshot()
        print(f"RobotMode={snapshot.mode}")
        print(f"GetErrorID={snapshot.errors_reply.strip()}")
        print(f"当前关节角={np.round(np.rad2deg(snapshot.joints_rad), 2).tolist()}°")
        print(f"目标关节角={list(config.startup_joint_deg)}°")
        answer = input("确认急停可用且工作区无人后按 Enter；输入其他内容取消：")
        if answer.strip():
            print("已取消，不使能、不运动。")
            return 1
        robot.prepare_for_control()
        result = robot.move_to_startup_pose()
        print(f"初始化完成：{np.round(np.rad2deg(result.joints_rad), 2).tolist()}°")
        return 0
    except (OSError, RobotError, ValueError) as error:
        print(f"初始化失败：{error}")
        return 1
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
