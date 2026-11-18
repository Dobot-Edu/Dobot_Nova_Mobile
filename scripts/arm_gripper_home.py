"""Return the arm to the configured startup pose and open the gripper."""
import argparse

from Dobot_Init.config import load_config
from Dobot_Init.gripper import GripperError, TriggerGripper
from Dobot_Init.robot import NovaRobot, RobotError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    if not parser.parse_args().execute:
        parser.error('实机回位必须明确传入 --execute')
    if input('确认机械臂路径和夹爪周围安全、急停可及后，输入 HOME：').strip() != 'HOME':
        return 1

    config = load_config('arm_only')
    if not config.robot.startup_pose_enabled or not config.gripper.enabled:
        print('机械臂标准位或夹爪未启用，取消回位。')
        return 1
    robot = None
    gripper = TriggerGripper(config.gripper, execute=True)
    try:
        robot = NovaRobot(config.robot)
        robot.prepare_for_control()
        robot.move_to_startup_pose()
        gripper.connect()
        gripper.check_error()
        print('机械臂已到训练标准位，夹爪已执行全开初始化。')
        return 0
    except (RobotError, GripperError, OSError) as error:
        print(f'回位失败：{error}')
        return 1
    finally:
        gripper.close()
        if robot is not None:
            robot.close()


if __name__ == '__main__':
    raise SystemExit(main())
