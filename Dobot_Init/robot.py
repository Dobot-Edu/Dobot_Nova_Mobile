"""Dobot Dashboard 的最小接口和受控启动恢复。"""

from __future__ import annotations

import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from Dobot_Init.config import RobotConfig


class RobotError(RuntimeError):
    """控制柜连接或指令错误。"""


def parse_values(reply: str, expected: int) -> np.ndarray:
    status = re.match(r"\s*(-?\d+)\s*,", reply or "")
    if status is None:
        raise RobotError(f"控制柜返回格式异常：{reply!r}")
    if int(status.group(1)) != 0:
        raise RobotError(f"控制柜拒绝查询（返回码={status.group(1)}）：{reply!r}；请检查RobotMode和GetErrorID")
    match = re.search(r"\{([^{}]*)\}", reply or "")
    if match is None:
        raise RobotError(f"控制柜返回格式异常：{reply!r}")
    if not match.group(1).strip():
        raise RobotError(f"控制柜未返回数值，不能使用空数据：{reply!r}")
    try:
        values = np.asarray([float(item.strip()) for item in match.group(1).split(",")])
    except ValueError as error:
        raise RobotError(f"控制柜返回非数值：{reply!r}") from error
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise RobotError(f"控制柜返回数量异常：{reply!r}")
    return values


def controller_has_errors(reply: str) -> bool:
    """解析 GetErrorID；空数组表示无报警，出现任意整数表示仍有报警。"""
    match = re.match(r"\s*0,\{(.*)\},GetErrorID\(\);?\s*$", reply or "", re.DOTALL)
    if match is None:
        raise RobotError(f"GetErrorID 返回格式异常：{reply!r}")
    return re.search(r"-?\d+", match.group(1)) is not None


@dataclass(frozen=True)
class RobotSnapshot:
    joints_rad: np.ndarray
    pose: np.ndarray
    mode: int
    errors_reply: str


def smooth_joint_trajectory(
    start_rad: np.ndarray,
    target_rad: np.ndarray,
    *,
    rate_hz: float,
    max_speed_deg_s: float,
):
    """生成三次平滑关节轨迹；每个yield值就是随后实际下发的action。"""
    start = np.asarray(start_rad, dtype=float)
    target = np.asarray(target_rad, dtype=float)
    if start.shape != (6,) or target.shape != (6,):
        raise ValueError("回位轨迹必须包含六个关节")
    if rate_hz <= 0 or max_speed_deg_s <= 0:
        raise ValueError("回位频率和速度必须为正数")
    max_delta_deg = float(np.max(np.abs(np.rad2deg(target - start))))
    # smoothstep最大斜率为1.5，因此据此计算时长，确保不超过指定关节速度。
    duration_s = max(1.0 / rate_hz, 1.5 * max_delta_deg / max_speed_deg_s)
    steps = max(1, int(np.ceil(duration_s * rate_hz)))
    for index in range(1, steps + 1):
        phase = index / steps
        weight = phase * phase * (3.0 - 2.0 * phase)
        yield start + weight * (target - start)


class NovaRobot:
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        try:
            self.socket = socket.create_connection(
                (config.ip, config.dashboard_port), timeout=2.0
            )
        except OSError as error:
            raise RobotError(f"无法连接 Nova {config.ip}:{config.dashboard_port}") from error
        self.socket.settimeout(2.0)

    def request(self, command: str) -> str:
        try:
            self.socket.sendall(command.encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    raise RobotError("控制柜连接断开")
                chunks.append(chunk)
                if b";" in chunk:
                    return b"".join(chunks).decode("utf-8", errors="replace")
        except socket.timeout as error:
            raise RobotError(f"控制柜指令超时：{command}") from error
        except OSError as error:
            raise RobotError(f"控制柜通信失败：{command}") from error

    @staticmethod
    def _require_ok(reply: str, command: str) -> None:
        if not reply.startswith("0,"):
            raise RobotError(f"{command} 被拒绝：{reply}")

    def joints(self) -> np.ndarray:
        return np.deg2rad(parse_values(self.request("GetAngle()"), 6))

    def pose(self) -> np.ndarray:
        return parse_values(self.request("GetPose()"), 6)

    def mode(self) -> int:
        return int(parse_values(self.request("RobotMode()"), 1)[0])

    def digital_input(self, index: int) -> int:
        """读取控制柜 DI，升降柱回零使用 DI2 作为最高点到位信号。"""
        value = int(parse_values(self.request(f"DI({int(index)})"), 1)[0])
        if value not in (0, 1):
            raise RobotError(f"DI({index}) 返回值异常：{value}")
        return value

    def digital_output(self, index: int, value: int | bool) -> None:
        """设置控制柜数字输出；信号灯使用 DO6、DO7、DO8。"""
        output_index = int(index)
        output_value = int(value)
        if output_index <= 0:
            raise ValueError(f"DO 编号必须为正整数：{index}")
        if output_value not in (0, 1):
            raise ValueError(f"DO 输出只能是0或1：{value}")
        command = f"DO({output_index},{output_value})"
        self._require_ok(self.request(command), command)

    def snapshot(self) -> RobotSnapshot:
        # GetErrorID 仅查询，不在这里 ClearError。
        mode = self.mode()
        errors_reply = self.request("GetErrorID()")
        if mode == 9:
            raise RobotError(
                f"控制柜处于报警状态 RobotMode=9，无法读取启动位姿；"
                f"报警详情：{errors_reply.strip()}。请先在控制柜/示教器排查报警，"
                "本次未清错、使能或运动。"
            )
        return RobotSnapshot(
            joints_rad=self.joints(),
            pose=self.pose(),
            mode=mode,
            errors_reply=errors_reply,
        )

    def enable(self) -> None:
        reply = self.request("EnableRobot()")
        self._require_ok(reply, "EnableRobot")
        deadline = time.monotonic() + 3.0
        mode = self.mode()
        while mode not in (5, 7) and time.monotonic() < deadline:
            time.sleep(0.1)
            mode = self.mode()
        if mode not in (5, 7):
            detail = "；请退出拖拽/JOG 并切换到 TCP 远程模式" if mode == 11 else ""
            raise RobotError(f"使能后 RobotMode={mode}，不能发送 ServoJ{detail}")
        self._configure_speed()

    def _configure_speed(self) -> None:
        for command in (
            f"SpeedFactor({self.config.speed_factor})",
            f"AccJ({self.config.speed_factor})",
        ):
            self._require_ok(self.request(command), command)

    def _wait_idle(self) -> None:
        """连续确认空闲，避免异步使能/禁用的瞬态状态被当成完成。"""
        deadline = time.monotonic() + 3.0
        stable = 0
        while True:
            mode = self.mode()
            stable = stable + 1 if mode == 5 else 0
            if stable >= 3:
                return
            if mode not in (4, 5, 7) or time.monotonic() >= deadline:
                raise RobotError(f"等待使能空闲失败：RobotMode={mode}")
            time.sleep(0.1)

    def _recover_manual_mode(self, mode: int) -> int:
        """退出拖拽/拖拽录制或残留JOG；不发送任何运动目标。"""
        mode_names = {6: "拖拽", 8: "拖拽录制", 11: "JOG"}
        if mode not in mode_names:
            return mode
        if not self.config.auto_recover_manual_mode:
            raise RobotError(
                f"机械臂处于{mode_names[mode]}模式（RobotMode={mode}），"
                "且 auto_recover_manual_mode=false"
            )

        print(
            f"[机械臂自检] 检测到{mode_names[mode]}模式（RobotMode={mode}），"
            "正在停止旧控制状态；不会发送运动目标"
        )
        # Nova 固件上残留JOG可能同时保留拖拽状态位；实机验证需要按此顺序
        # 清理状态机，然后禁用一次，使控制权完整释放。
        self._require_ok(self.request("ClearError()"), "ClearError")
        self._require_ok(self.request("StopDrag()"), "StopDrag")
        if mode == 11:
            self._require_ok(self.request("MoveJog()"), "MoveJog停止")
        self._require_ok(self.request("DisableRobot()"), "DisableRobot")

        deadline = time.monotonic() + 3.0
        recovered_mode = self.mode()
        while recovered_mode != 4 and time.monotonic() < deadline:
            time.sleep(0.1)
            recovered_mode = self.mode()
        if recovered_mode != 4:
            raise RobotError(
                f"退出{mode_names[mode]}后未确认禁用完成，RobotMode={recovered_mode}；"
                "请松开末端按键并关闭控制柜界面的JOG/拖拽操作"
            )
        print(
            f"[机械臂自检] 已退出{mode_names[mode]}模式，"
            f"RobotMode={recovered_mode}"
        )
        return recovered_mode

    def move_to_startup_pose(
        self,
        progress_callback: Callable[[np.ndarray], None] | None = None,
    ) -> RobotSnapshot:
        """回标准位；有回调时用可记录的ServoJ轨迹并发布真实下发目标。"""
        if not self.config.startup_pose_enabled:
            print("[机械臂标准位] startup_pose_enabled=false，跳过自动回位")
            snapshot = self.snapshot()
            if progress_callback is not None:
                progress_callback(snapshot.joints_rad.copy())
            return snapshot

        target_deg = np.asarray(self.config.startup_joint_deg, dtype=float)
        current_deg = np.rad2deg(self.joints())
        delta_deg = np.abs(target_deg - current_deg)
        if np.max(delta_deg) <= self.config.startup_tolerance_deg:
            snapshot = self.snapshot()
            if snapshot.mode != 5:
                raise RobotError(
                    f"机械臂虽在标准位，但 RobotMode={snapshot.mode}，尚未空闲"
                )
            print("[机械臂标准位] 当前已在标准位容差内，无需运动")
            if progress_callback is not None:
                progress_callback(snapshot.joints_rad.copy())
            return snapshot

        if progress_callback is not None:
            return self._servo_to_startup_pose(
                np.deg2rad(current_deg),
                np.deg2rad(target_deg),
                progress_callback,
            )

        command = (
            "MovJ(joint={{{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}}},"
            "a={:d},v={:d})"
        ).format(
            *target_deg,
            self.config.startup_speed_factor,
            self.config.startup_speed_factor,
        )
        for speed_command in (
            f"SpeedFactor({self.config.startup_speed_factor})",
            f"AccJ({self.config.startup_speed_factor})",
        ):
            self._require_ok(self.request(speed_command), speed_command)
        print(
            f"[机械臂标准位] 以 {self.config.startup_speed_factor}% 速度回位："
            f"{target_deg.tolist()}°"
        )
        self._require_ok(self.request(command), "MovJ(joint)")

        deadline = time.monotonic() + self.config.startup_timeout_s
        stable_samples = 0
        try:
            while time.monotonic() < deadline:
                actual_deg = np.rad2deg(self.joints())
                if progress_callback is not None:
                    progress_callback(np.deg2rad(actual_deg.copy()))
                error_deg = np.abs(target_deg - actual_deg)
                mode = self.mode()
                if mode not in (5, 7):
                    raise RobotError(f"标准位回位期间 RobotMode={mode}")
                if (
                    np.max(error_deg) <= self.config.startup_tolerance_deg
                    and mode == 5
                ):
                    stable_samples += 1
                    if stable_samples >= 3:
                        snapshot = self.snapshot()
                        if controller_has_errors(snapshot.errors_reply):
                            raise RobotError(
                                "标准位到位后控制柜存在报警："
                                f"{snapshot.errors_reply.strip()}"
                            )
                        print(
                            "[机械臂标准位] 到位："
                            f"{np.round(np.rad2deg(snapshot.joints_rad), 3).tolist()}°"
                        )
                        return snapshot
                else:
                    stable_samples = 0
                time.sleep(0.05)
            raise RobotError(
                f"标准位回位超时 {self.config.startup_timeout_s:.1f}s"
            )
        except Exception:
            try:
                self.request("Stop()")
            except RobotError:
                pass
            raise
        finally:
            self._configure_speed()

    def _servo_to_startup_pose(
        self,
        start_rad: np.ndarray,
        target_rad: np.ndarray,
        progress_callback: Callable[[np.ndarray], None],
    ) -> RobotSnapshot:
        """用50 Hz平滑目标回位，使采集action与实际ServoJ指令完全一致。"""
        target_deg = np.rad2deg(target_rad)
        speed_deg_s = (
            self.config.max_joint_speed_deg_s
            * self.config.startup_speed_factor
            / self.config.speed_factor
        )
        for speed_command in (
            f"SpeedFactor({self.config.startup_speed_factor})",
            f"AccJ({self.config.startup_speed_factor})",
        ):
            self._require_ok(self.request(speed_command), speed_command)
        print(
            f"[机械臂标准位] 以可记录ServoJ轨迹回位，最大关节速度"
            f"{speed_deg_s:.1f}°/s：{target_deg.tolist()}°"
        )

        period = 1.0 / self.config.rate_hz
        next_tick = time.monotonic()
        try:
            for command_rad in smooth_joint_trajectory(
                start_rad,
                target_rad,
                rate_hz=self.config.rate_hz,
                max_speed_deg_s=speed_deg_s,
            ):
                next_tick += period
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                mode = self.mode()
                if mode not in (5, 7):
                    raise RobotError(f"标准位ServoJ回位期间 RobotMode={mode}")
                self.servo_j(command_rad)
                progress_callback(command_rad.copy())

            deadline = time.monotonic() + self.config.startup_timeout_s
            stable_samples = 0
            while time.monotonic() < deadline:
                actual_rad = self.joints()
                # 到位等待期间持续发布最终目标，数采action不回退为实测角。
                progress_callback(target_rad.copy())
                mode = self.mode()
                if mode not in (5, 7):
                    raise RobotError(f"标准位ServoJ到位等待期间 RobotMode={mode}")
                if (
                    np.max(np.abs(np.rad2deg(target_rad - actual_rad)))
                    <= self.config.startup_tolerance_deg
                    and mode == 5
                ):
                    stable_samples += 1
                    if stable_samples >= 3:
                        snapshot = self.snapshot()
                        if controller_has_errors(snapshot.errors_reply):
                            raise RobotError(
                                "标准位到位后控制柜存在报警："
                                f"{snapshot.errors_reply.strip()}"
                            )
                        print(
                            "[机械臂标准位] 到位："
                            f"{np.round(np.rad2deg(snapshot.joints_rad), 3).tolist()}°"
                        )
                        return snapshot
                else:
                    stable_samples = 0
                time.sleep(0.05)
            raise RobotError(
                f"标准位ServoJ回位超时 {self.config.startup_timeout_s:.1f}s"
            )
        except Exception:
            try:
                self.request("Stop()")
            except RobotError:
                pass
            raise
        finally:
            self._configure_speed()

    def prepare_for_control(self) -> None:
        """现场确认后单次清错、复查、使能并设置速度，不发送运动目标。"""
        error_reply = self.request("GetErrorID()")
        mode = self.mode()
        has_errors = controller_has_errors(error_reply)
        if has_errors or mode == 9:
            print(f"[机械臂自检] 检测到报警：RobotMode={mode}，GetErrorID={error_reply.strip()}")
            if not self.config.auto_clear_error:
                raise RobotError("机械臂存在报警，且 auto_clear_error=false")
            self._require_ok(self.request("ClearError()"), "ClearError")
            deadline = time.monotonic() + 3.0
            while True:
                error_reply = self.request("GetErrorID()")
                mode = self.mode()
                if not controller_has_errors(error_reply) and mode != 9:
                    print("[机械臂自检] ClearError 成功，报警已解除")
                    break
                if time.monotonic() >= deadline:
                    raise RobotError(
                        "ClearError 后报警仍存在："
                        f"RobotMode={mode}，GetErrorID={error_reply.strip()}"
                    )
                time.sleep(0.1)

        mode = self._recover_manual_mode(self.mode())
        if mode in (5, 7):
            self._configure_speed()
            self._wait_idle()
            print(f"[机械臂自检] 已处于可运行模式 RobotMode={mode}")
            return
        self.enable()
        self._wait_idle()
        print("[机械臂自检] 已自动使能，RobotMode=5（连续确认空闲）")

    def servo_j(self, joints_rad: np.ndarray) -> None:
        joints = np.asarray(joints_rad, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RobotError("ServoJ 目标必须是六个有限关节角")
        command = "ServoJ({:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f})".format(
            *np.rad2deg(joints)
        )
        self._require_ok(self.request(command), "ServoJ")

    def close(self) -> None:
        self.socket.close()
