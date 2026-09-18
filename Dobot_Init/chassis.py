"""Quest 右手摇杆到华睿底盘的低速、失效安全控制。"""

from __future__ import annotations

import threading
import time

import numpy as np

from Dobot_Init.chassis_rpc import (
    ALLOWED_REMOTE_EXTRA_FIRST,
    DEFAULT_PASSWORD,
    DEFAULT_USERNAME,
    REMOTE_MODE,
    RobotRpc,
    RpcError,
    Velocity,
    state_safety_problem,
)
from Dobot_Init.config import ChassisConfig


class ChassisError(RuntimeError):
    pass


LOCALIZATION_WARNING_FIRST = 1 << 31
EXTRA_STATE_FIRST_NAMES = {
    3: "碰撞",
    4: "脱轨不可恢复",
    5: "急停按下",
    6: "电机未初始化",
    7: "人工干预",
    9: "跌倒",
    10: "遇障",
    11: "机器人本体异常",
    13: "偏航",
    14: "紧急刹车",
    15: "防跌落",
    20: "非任务模式",
    21: "前方障碍",
    22: "后方障碍",
    23: "左方障碍",
    24: "右方障碍",
    30: "报警模式",
    31: "激光导航定位评分异常",
}


def describe_extra_first(value: int) -> str:
    names = [name for bit, name in EXTRA_STATE_FIRST_NAMES.items() if int(value) & (1 << bit)]
    return " + ".join(names) if names else "正常"


def _state_for_manual_safety(state: dict, allow_localization_warning: bool) -> dict:
    masked = dict(state)
    if allow_localization_warning:
        masked["ExtraStateFirst"] = int(state.get("ExtraStateFirst", 0)) & ~LOCALIZATION_WARNING_FIRST
    return masked


def manual_safety_problem(state: dict, allow_localization_warning: bool) -> str | None:
    """直接速度遥控不依赖地图定位；其余硬件和避障状态仍必须正常。"""
    return state_safety_problem(
        _state_for_manual_safety(state, allow_localization_warning),
        remote_mode=True,
    )


def connection_safety_problem(
    state: dict,
    allow_localization_warning: bool = False,
) -> str | None:
    """连接阶段允许非任务位，并可按配置容忍仅影响地图导航的定位告警。"""
    work_mode = int(state.get("WorkMode", -1))
    if work_mode == REMOTE_MODE:
        return manual_safety_problem(state, allow_localization_warning)
    masked = _state_for_manual_safety(state, allow_localization_warning)
    masked["ExtraStateFirst"] = int(state.get("ExtraStateFirst", 0)) & ~ALLOWED_REMOTE_EXTRA_FIRST
    if allow_localization_warning:
        masked["ExtraStateFirst"] &= ~LOCALIZATION_WARNING_FIRST
    return state_safety_problem(masked, remote_mode=False)


def chassis_input_enabled(*, packet_valid: bool, grip_deadman: bool, require_grip: bool) -> bool:
    """只有新鲜且跟踪有效的数据才可控制底盘；Grip 是可配置附加条件。"""
    return packet_valid and (not require_grip or grip_deadman)


def joystick_velocity(
    thumbstick_xy: np.ndarray,
    deadzone: float,
    linear_speed_mm_s: int,
    angular_speed_mdeg_s: int,
) -> Velocity:
    """把 OpenXR 右摇杆映射为底盘速度，每个轴分别滤除机械串扰。"""
    stick = np.asarray(thumbstick_xy, dtype=float)
    if stick.shape != (2,) or not np.all(np.isfinite(stick)):
        raise ValueError("右摇杆必须是两个有限数值")

    stick = np.clip(stick, -1.0, 1.0)

    def filter_axis(value: float) -> float:
        if abs(value) <= deadzone:
            return 0.0
        return float(np.sign(value) * (abs(value) - deadzone) / (1.0 - deadzone))

    filtered = np.asarray([filter_axis(float(value)) for value in stick])
    # OpenXR +Y 是摇杆前推；+X 是右推。底盘正角速度是左转。
    return Velocity(
        int(round(filtered[1] * linear_speed_mm_s)),
        int(round(-filtered[0] * angular_speed_mdeg_s)),
    )


class VRChassis:
    """将主循环中的摇杆目标交给后台 RPC 线程，避免阻塞机械臂 50 Hz 控制。"""

    def __init__(self, config: ChassisConfig, *, execute: bool, read_only: bool = False) -> None:
        self.config = config
        self.read_only = read_only
        self.execute = execute and (config.enabled or read_only)
        self.rpc: RobotRpc | None = None
        self.original_mode = 0
        self._desired = Velocity()
        self._last_sent = Velocity()
        self._feedback_state: dict = {}
        self._feedback_monotonic_ns = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._error: ChassisError | None = None
        self._mode_requested = False
        self._remote_active = False
        self._localization_warning_reported = False
        # RPC 重连后要求摇杆先回中，避免重连成功时突然恢复旧速度。
        self._require_neutral = False

    @property
    def desired(self) -> Velocity:
        with self._lock:
            return self._desired

    def dataset_snapshot(self) -> dict:
        """返回数据采集所需的底盘指令和最近一次真实反馈。"""
        with self._lock:
            state = dict(self._feedback_state)
            command = self._last_sent
            feedback_monotonic_ns = self._feedback_monotonic_ns
        position = state.get("Position", [0, 0])
        if not isinstance(position, list) or len(position) != 2:
            position = [0, 0]
        return {
            "chassis_command_linear_mm_s": int(command.linear_mm_s),
            "chassis_command_angular_mdeg_s": int(command.angular_mdeg_s),
            "chassis_feedback_speed_mm_s": float(state.get("Speed", 0.0)),
            "chassis_feedback_angular_mdeg_s": float(state.get("AngularVelocity", 0.0)),
            "chassis_feedback_position_mm": [float(position[0]), float(position[1])],
            "chassis_feedback_yaw_mdeg": float(state.get("Oritation", 0.0)),
            "chassis_feedback_odom_mm": float(state.get("Odom", 0.0)),
            "chassis_feedback_monotonic_ns": int(feedback_monotonic_ns),
            "chassis_feedback_valid": bool(feedback_monotonic_ns),
        }

    def _store_feedback(self, state: dict) -> None:
        """原样保留RPC状态；时间戳在成功收到完整响应后产生。"""
        with self._lock:
            self._feedback_state = dict(state)
            self._feedback_monotonic_ns = time.monotonic_ns()

    def connect(self) -> None:
        if not self.execute:
            return
        self.rpc = RobotRpc(
            self.config.url,
            DEFAULT_USERNAME,
            DEFAULT_PASSWORD,
            timeout=self.config.rpc_timeout_s,
        )
        try:
            self.rpc.login()
            deadline = time.monotonic() + self.config.ready_wait_s
            while True:
                state = self.rpc.get_state()
                problem = connection_safety_problem(
                    state,
                    self.config.allow_manual_localization_warning,
                )
                idle = str(state.get("State")) == "Idle" and int(state.get("Speed", 0)) == 0
                if problem is None and idle:
                    break
                if time.monotonic() >= deadline:
                    detail = problem or "底盘连接时必须为 Idle 且速度为 0"
                    extra = int(state.get("ExtraStateFirst", 0))
                    raise ChassisError(f"{detail}；状态位含义：{describe_extra_first(extra)}")
                time.sleep(0.2)
            work_mode = int(state.get("WorkMode", -1))
            self.original_mode = work_mode
            self._store_feedback(state)
            self._report_localization_warning(state)
            print(
                f"底盘已连接：State={state.get('State')} "
                f"WorkMode={work_mode} Speed={state.get('Speed')} mm/s"
            )
            if self.read_only:
                self._worker = threading.Thread(target=self._read_only_loop, name="chassis-feedback-only", daemon=True)
                self._worker.start()
                print("[arm_only] 底盘仅读取真实反馈，摇杆禁用")
        except (RpcError, ChassisError) as error:
            # 只读连接阶段失败时只注销，不停车、不改模式。
            if self.rpc is not None:
                self.rpc.logout()
                self.rpc = None
            raise ChassisError(str(error)) from error

    def _read_only_loop(self) -> None:
        """arm_only 仅取真实反馈；退出/错误均不发停车、模式或速度命令。"""
        period = 1.0 / self.config.feedback_rate_hz
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                state = self.rpc.get_state()
                problem = connection_safety_problem(state, self.config.allow_manual_localization_warning)
                if problem:
                    raise ChassisError(problem)
                self._store_feedback(state)
            except (RpcError, ChassisError) as error:
                with self._lock:
                    self._error = ChassisError(str(error))
                return
            self._stop_event.wait(max(0.005, period - (time.monotonic() - started)))

    def enable_remote(self) -> None:
        if not self.execute or self.read_only:
            return
        if self.rpc is None:
            raise ChassisError("底盘尚未连接")
        try:
            state = self.rpc.get_state()
            if int(state.get("WorkMode", -1)) != REMOTE_MODE:
                self._mode_requested = True
                self.rpc.set_work_mode(REMOTE_MODE)
                print("请按底盘实体确认键切换到手动模式……")

            deadline = time.monotonic() + self.config.mode_confirm_timeout_s
            while time.monotonic() < deadline:
                state = self.rpc.get_state()
                if int(state.get("WorkMode", -1)) == REMOTE_MODE:
                    problem = manual_safety_problem(
                        state,
                        self.config.allow_manual_localization_warning,
                    )
                    if problem:
                        raise ChassisError(problem)
                    if str(state.get("State")) != "Idle" or int(state.get("Speed", 0)) != 0:
                        raise ChassisError("底盘进入手动模式时必须为 Idle 且速度为 0")
                    self._remote_active = True
                    self._store_feedback(state)
                    self._report_localization_warning(state)
                    break
                time.sleep(0.2)
            else:
                raise ChassisError("等待底盘实体确认键超时")

            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._control_loop,
                name="vr-chassis-control",
                daemon=True,
            )
            self._worker.start()
            gate = "按住Grip后" if self.config.require_grip else "无需Grip，直接"
            print(f"底盘VR摇杆控制已就绪：{gate}使用右摇杆前后移动、左右转向。")
        except (RpcError, ChassisError) as error:
            try:
                self.rpc.stop()
            except RpcError:
                pass
            raise ChassisError(str(error)) from error

    def update(self, thumbstick_xy: np.ndarray, *, input_valid: bool) -> None:
        input_valid = input_valid and self.config.enabled and not self.read_only
        command = (
            joystick_velocity(
                thumbstick_xy,
                self.config.deadzone,
                self.config.linear_speed_mm_s,
                self.config.angular_speed_mdeg_s,
            )
            if input_valid
            else Velocity()
        )
        with self._lock:
            if self._require_neutral:
                if command == Velocity():
                    self._require_neutral = False
                    print("[底盘重连] 摇杆已回中，恢复控制")
                else:
                    command = Velocity()
            self._desired = command

    def check_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def _control_loop(self) -> None:
        assert self.rpc is not None
        period = 1.0 / self.config.command_rate_hz
        feedback_period = 1.0 / self.config.feedback_rate_hz
        next_state_poll = time.monotonic()
        while not self._stop_event.is_set():
            try:
                started = time.monotonic()
                command = self.desired
                self.rpc.send_velocity(command.linear_mm_s, command.angular_mdeg_s)
                with self._lock:
                    self._last_sent = command
                if started >= next_state_poll:
                    state = self.rpc.get_state()
                    problem = manual_safety_problem(
                        state,
                        self.config.allow_manual_localization_warning,
                    )
                    if problem:
                        raise ChassisError(problem)
                    self._store_feedback(state)
                    self._report_localization_warning(state)
                    next_state_poll = started + feedback_period
                self._stop_event.wait(max(0.005, period - (time.monotonic() - started)))
            except RpcError as error:
                try:
                    self._recover_rpc(error)
                    next_state_poll = time.monotonic()
                except (RpcError, ChassisError) as recovery_error:
                    self._latch_error(recovery_error)
                    return
            except ChassisError as error:
                self._latch_error(error)
                return

    def _recover_rpc(self, original_error: RpcError) -> None:
        """通信/会话瞬断时停车并重连，不终止 VR 遥测主循环。"""
        assert self.rpc is not None
        with self._lock:
            self._desired = Velocity()
            self._require_neutral = True
            self._last_sent = Velocity()
        print(f"[底盘重连] RPC 异常：{original_error}；正在停车并重新登录")

        deadline = time.monotonic() + max(1.0, min(float(self.config.ready_wait_s), 10.0))
        last_error: RpcError = original_error
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            try:
                self.rpc.reconnect()
                self.rpc.stop()
                state = self.rpc.get_state()
                if int(state.get("WorkMode", -1)) != REMOTE_MODE:
                    raise ChassisError(
                        f"底盘重连后不在手动模式：WorkMode={state.get('WorkMode')}"
                    )
                problem = manual_safety_problem(
                    state,
                    self.config.allow_manual_localization_warning,
                )
                if problem:
                    raise ChassisError(problem)
                self._store_feedback(state)
                self._report_localization_warning(state)
                print("[底盘重连] 会话已恢复；请将右摇杆回中后继续控制")
                return
            except ChassisError:
                raise
            except RpcError as error:
                last_error = error
                self._stop_event.wait(0.2)
        raise ChassisError(f"底盘 RPC 重连超时：{last_error}")

    def _latch_error(self, error: RpcError | ChassisError) -> None:
        assert self.rpc is not None
        try:
            self.rpc.stop()
        except RpcError:
            pass
        with self._lock:
            self._desired = Velocity()
            self._error = ChassisError(str(error))

    def _report_localization_warning(self, state: dict) -> None:
        active = bool(int(state.get("ExtraStateFirst", 0)) & LOCALIZATION_WARNING_FIRST)
        if active and not self._localization_warning_reported:
            print(
                "[底盘警告] 激光导航定位评分异常；当前只允许 WorkMode=3 直接速度遥控，"
                "禁止地图导航和任务下发"
            )
            self._localization_warning_reported = True
        elif not active:
            self._localization_warning_reported = False

    def close(self) -> None:
        with self._lock:
            self._desired = Velocity()
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        if self.rpc is None:
            return
        try:
            if self._remote_active or self._mode_requested:
                self.rpc.stop()
        except RpcError as error:
            print(f"[底盘警告] 停车失败：{error}")
        finally:
            self.rpc.logout()
            self.rpc = None
            self._mode_requested = False
            self._remote_active = False
