"""Nova 移动底座机械臂信号灯控制：DO6红、DO7绿、DO8黄。"""

from __future__ import annotations

from enum import Enum

from Dobot_Init.robot import NovaRobot, RobotError


class LightColor(str, Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"


LIGHT_OUTPUTS = {
    LightColor.RED: 6,
    LightColor.GREEN: 7,
    LightColor.YELLOW: 8,
}


def collection_state_color(state: str) -> LightColor:
    """数采状态对应实体灯：录制绿、异常红，其余已激活状态黄。"""
    if state == "recording":
        return LightColor.GREEN
    if state == "error":
        return LightColor.RED
    return LightColor.YELLOW


class IndicatorLight:
    """保证同一时刻最多点亮一种颜色。"""

    def __init__(self, robot: NovaRobot) -> None:
        self.robot = robot
        self._color: LightColor | None = None

    def off(self) -> None:
        if self._color is None:
            for output in LIGHT_OUTPUTS.values():
                self.robot.digital_output(output, 0)
        else:
            self.robot.digital_output(LIGHT_OUTPUTS[self._color], 0)
        self._color = None

    def set(self, color: LightColor | str) -> None:
        selected = LightColor(color)
        if selected is self._color:
            return
        # 第一次先清空三个输出；后续只关闭当前颜色，减少队列 DO 指令。
        self.off()
        try:
            self.robot.digital_output(LIGHT_OUTPUTS[selected], 1)
            self._color = selected
        except (OSError, RobotError):
            # 点亮失败时尽力回到全灭状态，随后保留原异常。
            try:
                self.off()
            except (OSError, RobotError):
                pass
            raise
