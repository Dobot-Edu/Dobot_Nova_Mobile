"""通过厂家 HTTP RPC 安全地读取和控制华睿底盘。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


from Dobot_Init.chassis_rpc import REMOTE_MODE, RobotRpc, RpcError
from Dobot_Init.chassis import (
    connection_safety_problem,
    manual_safety_problem,
)


@dataclass(frozen=True)
class ChassisFeedback:
    linear_m_s: float
    angular_rad_s: float
    monotonic_ns: int
    state: dict


class InferenceChassis:
    """只读连接后再按需使能；任一异常都会锁存并尝试发送零速度。"""

    def __init__(
        self,
        *,
        url: str,
        username: str,
        password: str,
        command_hz: float,
        feedback_hz: float,
        linear_limit_m_s: float,
        angular_limit_rad_s: float,
        linear_acceleration_m_s2: float,
        angular_acceleration_rad_s2: float,
        allow_localization_warning: bool,
        rpc_timeout_s: float,
        mode_confirm_timeout_s: float,
        command_timeout_s: float = 0.30,
    ) -> None:
        self.rpc = RobotRpc(url, username, password, timeout=rpc_timeout_s)
        self.command_hz = float(command_hz)
        self.feedback_hz = float(feedback_hz)
        self.linear_limit_m_s = float(linear_limit_m_s)
        self.angular_limit_rad_s = float(angular_limit_rad_s)
        self.linear_acceleration_m_s2 = float(linear_acceleration_m_s2)
        self.angular_acceleration_rad_s2 = float(angular_acceleration_rad_s2)
        self.allow_localization_warning = bool(allow_localization_warning)
        self.mode_confirm_timeout_s = float(mode_confirm_timeout_s)
        if command_timeout_s <= 0:
            raise ValueError("底盘命令超时必须为正")
        self.command_timeout_s = float(command_timeout_s)
        self._command_lock = threading.Lock()
        self._halted = threading.Event()
        self._last_command_at: float | None = None
        self._watchdog: threading.Thread | None = None
        self._lock = threading.Lock()
        self._feedback: ChassisFeedback | None = None
        self._desired = np.zeros(2, dtype=np.float64)
        self._applied = np.zeros(2, dtype=np.float64)
        self._enabled = False
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect_read_only(self) -> None:
        self.rpc.login()
        state = self.rpc.get_state()
        problem = connection_safety_problem(state, self.allow_localization_warning)
        if problem:
            self.rpc.logout()
            raise RuntimeError(f"底盘只读自检失败：{problem}")
        self._store_feedback(state)
        self._thread = threading.Thread(target=self._worker, name="pi05-chassis", daemon=True)
        self._thread.start()
        self._watchdog = threading.Thread(target=self._watch_commands, name="pi05-chassis-watchdog", daemon=True)
        self._watchdog.start()
        print(
            f"[底盘] 已只读连接：State={state.get('State')} "
            f"WorkMode={state.get('WorkMode')} Speed={state.get('Speed')} mm/s"
        )

    def enable_remote(self) -> None:
        """切换厂家手动速度模式；需要现场按底盘实体确认键。"""

        if self._halted.is_set():
            raise RuntimeError("本轮底盘已经停车锁存，需要重新建立控制实例")
        state = self.rpc.get_state()
        if int(state.get("WorkMode", -1)) != REMOTE_MODE:
            self.rpc.set_work_mode(REMOTE_MODE)
            print("[底盘] 请按底盘实体确认键，允许进入手动速度模式……", flush=True)
        deadline = time.monotonic() + self.mode_confirm_timeout_s
        while time.monotonic() < deadline:
            state = self.rpc.get_state()
            if int(state.get("WorkMode", -1)) == REMOTE_MODE:
                problem = manual_safety_problem(state, self.allow_localization_warning)
                if problem:
                    raise RuntimeError(f"底盘不能进入推理控制：{problem}")
                if str(state.get("State")) != "Idle" or abs(float(state.get("Speed", 0))) > 0:
                    raise RuntimeError("底盘进入手动模式时必须为 Idle 且速度为 0")
                with self._lock:
                    self._enabled = True
                    self._desired[:] = 0.0
                    self._applied[:] = 0.0
                self._store_feedback(state)
                print("[底盘] 推理速度控制已启用，初始指令为零")
                return
            time.sleep(0.2)
        raise TimeoutError("等待底盘实体确认键超时")

    def set_velocity(self, linear_m_s: float, angular_rad_s: float) -> None:
        command = np.asarray([linear_m_s, angular_rad_s], dtype=np.float64)
        if not np.all(np.isfinite(command)):
            raise RuntimeError("底盘速度指令含非有限数值")
        command[0] = np.clip(command[0], -self.linear_limit_m_s, self.linear_limit_m_s)
        command[1] = np.clip(command[1], -self.angular_limit_rad_s, self.angular_limit_rad_s)
        with self._lock:
            if self._halted.is_set():
                raise RuntimeError("底盘已经停车锁存，拒绝新的速度指令")
            self._desired = command
            self._last_command_at = time.monotonic()

    def pause(self) -> None:
        """阶段边界立即发送零速度，但保留控制会话供下一阶段继续使用。"""
        with self._command_lock:
            with self._lock:
                if self._halted.is_set():
                    return
                enabled = self._enabled
                self._desired[:] = 0.0
                self._applied[:] = 0.0
                # 暂停期间不启动命令超时；下一条策略命令会重新计时。
                self._last_command_at = None
            if enabled:
                acknowledged = False
                last_error = None
                for _ in range(3):
                    try:
                        self.rpc.send_velocity(0, 0)
                        acknowledged = True
                    except Exception as error:
                        last_error = error
                if not acknowledged:
                    raise RuntimeError(f"底盘阶段停车指令全部发送失败：{last_error}") from last_error

    def stop(self) -> None:
        """锁存停止，串行化正在发送的指令，然后直接发送零速度，不走减速积分。"""
        self._halted.set()
        with self._command_lock:
            with self._lock:
                enabled = self._enabled
                self._desired[:] = 0.0
                self._applied[:] = 0.0
            # 若发送失败，保留enabled，允许close再次尝试停止。
            if enabled:
                acknowledged = False
                last_error = None
                # 底层rpc.stop吞掉RpcError；这里必须明确确认至少一次零速度发送成功。
                for _ in range(3):
                    try:
                        self.rpc.send_velocity(0, 0)
                        acknowledged = True
                    except Exception as error:
                        last_error = error
                if not acknowledged:
                    raise RuntimeError(f"底盘零速度指令全部发送失败：{last_error}") from last_error
            with self._lock:
                self._enabled = False

    def _watch_commands(self) -> None:
        while not self._stop.wait(min(0.05, self.command_timeout_s / 3)):
            with self._lock:
                # 启动回位和初始规划期间尚无策略命令，不启动计时。
                expired = (self._enabled and self._last_command_at is not None
                           and time.monotonic() - self._last_command_at > self.command_timeout_s)
                if expired:
                    self._halted.set()
                    self._error = RuntimeError(f"底盘超过{self.command_timeout_s:.2f}s未收到新策略指令")
            if expired:
                try:
                    self.stop()
                except Exception as error:
                    with self._lock:
                        self._error = RuntimeError(f"底盘命令过期且停车请求失败：{error}")
                return

    def feedback(self, *, maximum_age_s: float) -> ChassisFeedback:
        self.check_error()
        with self._lock:
            value = self._feedback
        if value is None:
            raise RuntimeError("底盘还没有反馈")
        age = (time.monotonic_ns() - value.monotonic_ns) / 1e9
        if age > maximum_age_s:
            raise RuntimeError(f"底盘反馈过期：{age:.3f}s > {maximum_age_s:.3f}s")
        return value

    def check_error(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"底盘控制线程失败：{error}") from error

    def close(self) -> None:
        self._stop.set()
        try:
            self.stop()
        finally:
            for thread in (self._thread, self._watchdog):
                if thread is not None:
                    thread.join(timeout=2.0)
            self.rpc.logout()

    def _store_feedback(self, state: dict) -> None:
        value = ChassisFeedback(
            linear_m_s=float(state.get("Speed", 0.0)) / 1000.0,
            angular_rad_s=float(np.deg2rad(float(state.get("AngularVelocity", 0.0)) / 1000.0)),
            monotonic_ns=time.monotonic_ns(),
            state=dict(state),
        )
        with self._lock:
            self._feedback = value

    @staticmethod
    def _approach(current: float, target: float, maximum_delta: float) -> float:
        return current + float(np.clip(target - current, -maximum_delta, maximum_delta))

    def _worker(self) -> None:
        period = 1.0 / self.command_hz
        feedback_period = 1.0 / self.feedback_hz
        next_feedback = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                with self._lock:
                    enabled = self._enabled
                    desired = self._desired.copy()
                    applied = self._applied.copy()
                if enabled:
                    with self._command_lock:
                        # stop先锁存，再等当前RPC完成；锁存后禁止补发旧非零速度。
                        with self._lock:
                            if self._halted.is_set():
                                break
                            if (self._last_command_at is not None and
                                    time.monotonic() - self._last_command_at > self.command_timeout_s):
                                raise RuntimeError("底盘策略速度指令过期")
                            desired = self._desired.copy()
                            applied = self._applied.copy()
                        applied[0] = self._approach(
                            applied[0], desired[0], self.linear_acceleration_m_s2 * period
                        )
                        applied[1] = self._approach(
                            applied[1], desired[1], self.angular_acceleration_rad_s2 * period
                        )
                        self.rpc.send_velocity(
                            int(round(applied[0] * 1000.0)),
                            int(round(np.rad2deg(applied[1]) * 1000.0)),
                        )
                        with self._lock:
                            self._applied = applied
                if started >= next_feedback:
                    state = self.rpc.get_state()
                    problem = (
                        manual_safety_problem(state, self.allow_localization_warning)
                        if enabled
                        else connection_safety_problem(state, self.allow_localization_warning)
                    )
                    if problem:
                        raise RuntimeError(problem)
                    self._store_feedback(state)
                    next_feedback = started + feedback_period
                self._stop.wait(max(0.002, period - (time.monotonic() - started)))
        except Exception as error:
            try:
                self.stop()
            except Exception:
                pass
            with self._lock:
                self._desired[:] = 0.0
                self._error = error
