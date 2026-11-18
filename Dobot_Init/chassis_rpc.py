#!/usr/bin/env python3
"""使用键盘方向键低速遥控华睿底盘。

接口来自底盘自带“地图实验室”网页实际使用的 RPC：
Robot.setWorkMode + Robot.control/RunCmdVel。

方向键按下运动、松开停车；窗口失焦、通信异常、避障或关闭窗口时均停车。
默认速度很低，程序退出时恢复启动前的工作模式。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_URL = "http://192.168.23.1"
DEFAULT_USERNAME = os.getenv("HUARUI_WEB_USERNAME", "admin")
DEFAULT_PASSWORD = os.getenv("HUARUI_WEB_PASSWORD", "admin123456")
REMOTE_MODE = 3  # 地图实验室打开 AMR 操控窗口时使用的模式
TASK_MODE = 0
ALLOWED_REMOTE_EXTRA_FIRST = 1 << 20  # 非任务模式；遥控模式下属于预期状态


class RpcError(RuntimeError):
    pass


class RobotRpc:
    def __init__(self, base_url: str, username: str, password: str, timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self.session: str | int | None = None
        self._request_id = random.randrange(1, 9_000_000)
        # 登录、重连和普通 RPC 共用一把可重入锁，避免多个线程互相覆盖 session。
        self._lock = threading.RLock()
        self.keepalive_interval = 10.0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _post(self, path: str, message: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(message, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RpcError(f"RPC 通信失败: {exc}") from exc
        if not isinstance(result, dict):
            raise RpcError("RPC 返回值不是 JSON 对象")
        return result

    @staticmethod
    def _md5_upper(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest().upper()

    def login(self) -> None:
        with self._lock:
            self._login_unlocked()

    def _login_unlocked(self) -> None:
        first = self._post(
            "/RPC2_Login",
            {
                "method": "global.login",
                "params": {
                    "userName": self.username,
                    "password": "",
                    "clientType": "Web3.0",
                    "loginType": "Direct",
                    "keepAlive": True,
                },
                "id": self._next_id(),
                "sessionTimeout": 300,
            },
        )
        params = first.get("params", {})
        try:
            realm = str(params["realm"])
            nonce = str(params["random"])
            challenge_session = first["session"]
        except KeyError as exc:
            raise RpcError(f"登录挑战数据不完整: {first}") from exc

        digest1 = self._md5_upper(f"{self.username}:{realm}:{self.password}")
        digest2 = self._md5_upper(f"{self.username}:{nonce}:{digest1}")
        second = self._post(
            "/RPC2_Login",
            {
                "method": "global.login",
                "params": {
                    "userName": self.username,
                    "password": digest2,
                    "clientType": "Web3.0",
                    "loginType": "Direct",
                    "authorityType": "Default",
                    "keepAlive": True,
                },
                "sessionTimeout": 300,
                "id": self._next_id(),
                "session": challenge_session,
            },
        )
        if second.get("result") is not True:
            raise RpcError(f"底盘登录失败: {second.get('error', second)}")
        self.session = second.get("session", challenge_session)
        self.keepalive_interval = float(second.get("params", {}).get("keepAliveInterval", 10))

    def call(self, method: str, params: Any = None) -> dict[str, Any]:
        with self._lock:
            if self.session is None:
                raise RpcError("尚未登录底盘")
            result = self._call_unlocked(method, params)
            # 底盘在人工确认切换工作模式后，可能会立即作废旧会话。
            # 部分固件的 Robot.control 在会话失效时不返回 error，
            # 只返回 id/session/result=false，也必须当作会话失效。
            # 在同一临界区内重登并仅重试一次，不让多线程互相覆盖 session。
            if self._is_invalid_session_response(result):
                self.session = None
                self._login_unlocked()
                result = self._call_unlocked(method, params)
        if result.get("result") is not True:
            raise RpcError(f"{method} 调用失败: {result.get('error', result)}")
        return result

    @staticmethod
    def _is_invalid_session_response(result: dict[str, Any]) -> bool:
        """兼容底盘两种会话失效响应。

        新固件返回 Invalid session 错误码；旧固件只返回
        result=false 和 session，没有 error 字段。
        """
        if result.get("result") is True:
            return False
        error = result.get("error")
        error_text = str(error or result)
        if isinstance(error, dict) and str(error.get("code")) == "287637505":
            return True
        if "Invalid session" in error_text or "invalid session" in error_text.lower():
            return True
        return error is None and "session" in result

    def _call_unlocked(self, method: str, params: Any) -> dict[str, Any]:
        return self._post(
            "/RPC2",
            {
                "method": method,
                "params": params,
                "id": self._next_id(),
                "session": self.session,
            },
        )

    def get_state(self) -> dict[str, Any]:
        result = self.call("Robot.getState", {})
        state = result.get("params", {}).get("State", {})
        if not isinstance(state, dict):
            raise RpcError("Robot.getState 返回数据中没有 State")
        return state

    def keep_alive(self) -> None:
        self.call(
            "global.keepAlive",
            {"timeout": self.keepalive_interval, "active": True},
        )

    def reconnect(self) -> None:
        """会话过期后重新完成挑战登录。"""
        with self._lock:
            self.session = None
            self._login_unlocked()

    def set_work_mode(self, mode: int) -> None:
        self.call("Robot.setWorkMode", {"Mode": int(mode)})

    def send_velocity(self, linear_mm_s: int, angular_mdeg_s: int) -> None:
        self.call(
            "Robot.control",
            {
                "Actions": [
                    {
                        "ID": "keyboard_chassis_control",
                        "Cmd": "RunCmdVel",
                        "Async": 0,
                        "NeedACK": 1,
                        "FailureHandle": 0,
                        "Timeout": 0,
                        "Param0": int(linear_mm_s),
                        "Param1": 0,  # 双轮差速模式
                        "Param2": int(angular_mdeg_s),
                    }
                ]
            },
        )

    def stop(self) -> None:
        # 多发两帧零速度，降低单帧丢失风险。
        for _ in range(3):
            try:
                self.send_velocity(0, 0)
            except RpcError:
                pass
            time.sleep(0.03)

    def logout(self) -> None:
        if self.session is None:
            return
        try:
            self.call("global.logout", {})
        except RpcError:
            pass
        finally:
            self.session = None


@dataclass(frozen=True)
class Velocity:
    linear_mm_s: int = 0
    angular_mdeg_s: int = 0


def velocity_from_keys(keys: set[str], linear_speed: int, angular_speed: int) -> Velocity:
    linear = linear_speed * (int("Up" in keys) - int("Down" in keys))
    angular = angular_speed * (int("Left" in keys) - int("Right" in keys))
    return Velocity(linear, angular)


def state_safety_problem(state: dict[str, Any], remote_mode: bool) -> str | None:
    speed = int(state.get("Speed", 0))
    robot_state = str(state.get("State", "Unknown"))
    work_mode = int(state.get("WorkMode", -1))
    extra_first = int(state.get("ExtraStateFirst", 0))
    extra_second = int(state.get("ExtraStateSecond", 0))
    allowed_first = ALLOWED_REMOTE_EXTRA_FIRST if remote_mode else 0
    unexpected_first = extra_first & ~allowed_first

    if robot_state in {"Fault", "Offline", "Initializing", "InCharging", "InUpgrading"}:
        return f"底盘状态不允许遥控: {robot_state}"
    if remote_mode and work_mode != REMOTE_MODE:
        return f"遥控模式意外变化: WorkMode={work_mode}"
    if unexpected_first or extra_second:
        return f"底盘异常/避障状态: ExtraState=({extra_first:#010x}, {extra_second:#010x})"
    if abs(speed) > 3000:
        return f"状态速度异常: {speed} mm/s"
    return None


class KeyboardController:
    def __init__(
        self,
        rpc: RobotRpc,
        linear_speed: int,
        angular_speed: int,
        send_hz: float,
        original_mode: int,
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import messagebox
        except ImportError as exc:
            raise RuntimeError("缺少 tkinter，无法创建键盘控制窗口") from exc

        self.tk = tk
        self.messagebox = messagebox
        self.rpc = rpc
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.period = 1.0 / send_hz
        self.original_mode = original_mode
        self.keys: set[str] = set()
        self.enabled = False
        self.closing = False
        self.last_ui_heartbeat = time.monotonic()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_sent = Velocity()
        self.mode_transition_lock = threading.Lock()
        self.enable_in_progress = False
        self.enable_request_id = 0

        self.root = tk.Tk()
        self.root.title("华睿底盘键盘遥控（低速）")
        self.root.geometry("620x390")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.bind("<FocusOut>", self.on_focus_out)

        self.status_var = tk.StringVar(value="未启用：底盘保持停止")
        self.command_var = tk.StringVar(value="线速度 0 mm/s　角速度 0.000°/s")
        self.state_var = tk.StringVar(value="正在读取状态……")

        tk.Label(self.root, text="方向键控制底盘", font=("Sans", 20, "bold")).pack(pady=14)
        tk.Label(
            self.root,
            text="↑ 前进　↓ 后退　← 左转　→ 右转\n支持同时按两个方向键；松开立即停车",
            font=("Sans", 14),
        ).pack(pady=8)
        tk.Label(
            self.root,
            text=f"线速度：{linear_speed} mm/s　角速度：{angular_speed / 1000:.1f}°/s",
            font=("Sans", 12),
        ).pack(pady=5)
        tk.Label(self.root, textvariable=self.command_var, fg="#005bbb", font=("Sans", 13)).pack(pady=5)
        tk.Label(self.root, textvariable=self.state_var, font=("Sans", 11)).pack(pady=5)
        tk.Label(self.root, textvariable=self.status_var, fg="#a33", font=("Sans", 12, "bold")).pack(pady=8)

        buttons = tk.Frame(self.root)
        buttons.pack(pady=10)
        self.enable_button = tk.Button(
            buttons, text="启用键盘控制", width=18, height=2, command=self.toggle_enabled
        )
        self.enable_button.pack(side=tk.LEFT, padx=12)
        tk.Button(buttons, text="停车并退出", width=18, height=2, command=self.close).pack(
            side=tk.LEFT, padx=12
        )

        tk.Label(
            self.root,
            text="安全：窗口失焦、Esc、网络异常、避障或程序退出都会停车。\n物理急停优先，程序停车不能替代物理急停。",
            fg="#666",
        ).pack(pady=8)

        self.worker = threading.Thread(target=self.control_loop, name="chassis-control", daemon=True)
        self.worker.start()
        self.root.after(50, self.ui_watchdog)
        self.root.focus_force()

    def post_ui(self, callback: Callable[[], None]) -> None:
        if not self.closing:
            self.root.after(0, callback)

    def desired_velocity(self) -> Velocity:
        with self.lock:
            if not self.enabled or time.monotonic() - self.last_ui_heartbeat > 0.3:
                return Velocity()
            return velocity_from_keys(set(self.keys), self.linear_speed, self.angular_speed)

    def clear_keys(self) -> None:
        with self.lock:
            self.keys.clear()

    def on_key_press(self, event: Any) -> None:
        if event.keysym == "Escape":
            self.disable("已按 Esc，底盘停车")
            return
        if event.keysym not in {"Up", "Down", "Left", "Right"}:
            return
        with self.lock:
            if self.enabled:
                self.keys.add(event.keysym)

    def on_key_release(self, event: Any) -> None:
        if event.keysym in {"Up", "Down", "Left", "Right"}:
            with self.lock:
                self.keys.discard(event.keysym)

    def on_focus_out(self, _event: Any) -> None:
        # Tk 的 FocusOut 也会在焦点从窗口本体移动到内部按钮时触发，
        # 延迟检查当前焦点，只有整个应用真正失焦才停车。
        self.root.after(80, self._disable_if_window_unfocused)

    def _disable_if_window_unfocused(self) -> None:
        if self.closing:
            return
        if self.root.focus_displayof() is None:
            self.disable("窗口失去焦点，底盘停车")

    def ui_watchdog(self) -> None:
        if self.closing:
            return
        with self.lock:
            self.last_ui_heartbeat = time.monotonic()
        self.root.after(50, self.ui_watchdog)

    def toggle_enabled(self) -> None:
        if self.enabled:
            self.disable("已人工停止")
            return
        with self.lock:
            if self.enable_in_progress:
                self.status_var.set("正在等待底盘确认键，无需再次点击")
                return
            self.enable_in_progress = True
            self.enable_request_id += 1
            request_id = self.enable_request_id
            self.keys.clear()
        self.enable_button.configure(state="disabled", text="等待手动模式确认…")
        self.status_var.set("正在请求手动模式……")
        threading.Thread(target=self._request_enable, args=(request_id,), daemon=True).start()

    def _get_state_with_reconnect(self) -> dict[str, Any]:
        try:
            return self.rpc.get_state()
        except RpcError:
            self.rpc.reconnect()
            return self.rpc.get_state()

    def _request_enable(self, request_id: int) -> None:
        if not self.mode_transition_lock.acquire(blocking=False):
            self.post_ui(lambda: self._enable_failed(request_id, "正在完成停车和模式恢复，请稍候再启用"))
            return
        try:
            state = self._get_state_with_reconnect()
            work_mode = int(state.get("WorkMode", -1))
            problem = state_safety_problem(state, remote_mode=(work_mode == REMOTE_MODE))
            if problem:
                raise RpcError(problem)
            if str(state.get("State")) != "Idle" or int(state.get("Speed", 0)) != 0:
                raise RpcError("启用前底盘必须为 Idle 且速度为 0")

            if work_mode != REMOTE_MODE:
                self.rpc.set_work_mode(REMOTE_MODE)
                self.post_ui(
                    lambda: self.status_var.set(
                        "请按底盘物理确认键切换到手动模式；程序会自动继续"
                    )
                )

            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                with self.lock:
                    if request_id != self.enable_request_id or not self.enable_in_progress:
                        return
                state = self._get_state_with_reconnect()
                self.post_ui(lambda s=state: self._display_state(s))
                if int(state.get("WorkMode", -1)) == REMOTE_MODE:
                    problem = state_safety_problem(state, remote_mode=True)
                    if problem:
                        raise RpcError(problem)
                    if str(state.get("State")) != "Idle" or int(state.get("Speed", 0)) != 0:
                        raise RpcError("手动模式就绪时底盘必须为 Idle 且速度为 0")
                    self.post_ui(lambda: self._enable_succeeded(request_id))
                    return
                time.sleep(0.2)
            raise RpcError("等待底盘物理确认键超时（30 秒）")
        except RpcError as exc:
            self.rpc.stop()
            try:
                self.rpc.set_work_mode(self.original_mode)
            except RpcError:
                pass
            self.post_ui(lambda e=str(exc): self._enable_failed(request_id, e))
        finally:
            self.mode_transition_lock.release()

    def _display_state(self, state: dict[str, Any]) -> None:
        self.state_var.set(
            f"状态 {state.get('State')}　速度 {state.get('Speed')} mm/s　"
            f"模式 {state.get('WorkMode')}　异常 {int(state.get('ExtraStateFirst', 0)):#010x}"
        )

    def _enable_succeeded(self, request_id: int) -> None:
        with self.lock:
            if request_id != self.enable_request_id or not self.enable_in_progress:
                return
            self.enable_in_progress = False
            self.enabled = True
            self.keys.clear()
        self.enable_button.configure(state="normal", text="停止键盘控制")
        self.status_var.set("已启用：请保持窗口焦点，松开方向键即停车")
        self.root.focus_force()

    def _enable_failed(self, request_id: int, error: str) -> None:
        with self.lock:
            if request_id != self.enable_request_id:
                return
            self.enable_in_progress = False
            self.enabled = False
            self.keys.clear()
        self.enable_button.configure(state="normal", text="启用键盘控制")
        self.status_var.set(f"无法启用：{error}")
        self.messagebox.showerror("无法启用", error)

    def disable(self, reason: str) -> None:
        with self.lock:
            was_enabled = self.enabled
            was_enabling = self.enable_in_progress
            self.enabled = False
            self.enable_in_progress = False
            self.enable_request_id += 1
            self.keys.clear()
        self.status_var.set(reason)
        self.command_var.set("线速度 0 mm/s　角速度 0.000°/s")
        self.enable_button.configure(state="normal", text="启用键盘控制")
        if was_enabled or was_enabling:
            threading.Thread(target=self._stop_and_restore, daemon=True).start()

    def _stop_and_restore(self) -> None:
        with self.mode_transition_lock:
            self.rpc.stop()
            try:
                self.rpc.set_work_mode(self.original_mode)
            except RpcError as exc:
                self.post_ui(lambda: self.status_var.set(f"已停车；恢复工作模式失败：{exc}"))

    def control_loop(self) -> None:
        next_state_poll = 0.0
        next_keepalive = time.monotonic() + max(2.0, self.rpc.keepalive_interval * 0.5)
        while not self.stop_event.is_set():
            started = time.monotonic()
            command = self.desired_velocity()
            try:
                if started >= next_keepalive:
                    try:
                        self.rpc.keep_alive()
                    except RpcError:
                        # 停用状态下也自动恢复过期会话；不改变工作模式。
                        self.rpc.reconnect()
                        self.rpc.keep_alive()
                    next_keepalive = started + max(2.0, self.rpc.keepalive_interval * 0.5)
                # 运动时持续刷新；停止命令至少在状态变化时发送。
                if command != Velocity() or command != self.last_sent:
                    self.rpc.send_velocity(command.linear_mm_s, command.angular_mdeg_s)
                    self.last_sent = command
                self.post_ui(
                    lambda c=command: self.command_var.set(
                        f"线速度 {c.linear_mm_s} mm/s　角速度 {c.angular_mdeg_s / 1000:.3f}°/s"
                    )
                )

                if self.enabled and started >= next_state_poll:
                    state = self.rpc.get_state()
                    # GUI只用于人工调试，也保持20 Hz状态刷新，避免界面显示明显滞后。
                    next_state_poll = started + 0.05
                    problem = state_safety_problem(state, remote_mode=True)
                    self.post_ui(
                        lambda s=state: self.state_var.set(
                            f"状态 {s.get('State')}　速度 {s.get('Speed')} mm/s　"
                            f"模式 {s.get('WorkMode')}　异常 {int(s.get('ExtraStateFirst', 0)):#010x}"
                        )
                    )
                    if problem:
                        raise RpcError(problem)
            except RpcError as exc:
                self.rpc.stop()
                with self.lock:
                    was_enabled = self.enabled
                    self.enabled = False
                    self.keys.clear()
                self.last_sent = Velocity()
                if was_enabled:
                    self.post_ui(lambda e=str(exc): self._show_fault(e))
                else:
                    self.post_ui(lambda e=str(exc): self.status_var.set(f"底盘会话正在自动恢复：{e}"))

            elapsed = time.monotonic() - started
            self.stop_event.wait(max(0.005, self.period - elapsed))

    def _show_fault(self, error: str) -> None:
        self.enable_button.configure(text="启用键盘控制")
        self.status_var.set(f"安全停车：{error}")
        self.messagebox.showerror("底盘已停车", error)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        with self.lock:
            self.enabled = False
            self.keys.clear()
        self.stop_event.set()
        self.rpc.stop()
        try:
            self.rpc.set_work_mode(self.original_mode)
        except RpcError:
            pass
        self.rpc.logout()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
        if not self.closing:
            self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="华睿底盘方向键低速遥控")
    parser.add_argument("--url", default=DEFAULT_URL, help="底盘网页 RPC 地址")
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--linear-speed", type=int, default=50, help="线速度 mm/s，最大 200")
    parser.add_argument("--angular-speed", type=int, default=10000, help="角速度 0.001°/s，最大 30000")
    parser.add_argument("--send-hz", type=float, default=20.0, help="速度指令刷新频率")
    parser.add_argument("--probe", action="store_true", help="仅登录读取状态，不切模式、不运动")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.linear_speed <= 200:
        raise SystemExit("--linear-speed 必须为 1~200 mm/s")
    if not 1000 <= args.angular_speed <= 30000:
        raise SystemExit("--angular-speed 必须为 1000~30000（即 1~30°/s）")
    if not 5 <= args.send_hz <= 30:
        raise SystemExit("--send-hz 必须为 5~30 Hz")


def main() -> int:
    args = parse_args()
    validate_args(args)
    rpc = RobotRpc(args.url, args.username, args.password)
    try:
        rpc.login()
        state = rpc.get_state()
        print(
            "底盘连接成功："
            f"State={state.get('State')}, WorkMode={state.get('WorkMode')}, "
            f"Speed={state.get('Speed')} mm/s, Position={state.get('Position')}, "
            f"ExtraState=({int(state.get('ExtraStateFirst', 0)):#010x}, "
            f"{int(state.get('ExtraStateSecond', 0)):#010x})"
        )
        if args.probe:
            print("只读检测完成：未切换模式，未发送速度指令。")
            return 0

        problem = state_safety_problem(state, remote_mode=False)
        if problem:
            raise RpcError(problem)
        if str(state.get("State")) != "Idle" or int(state.get("Speed", 0)) != 0:
            raise RpcError("启动键盘程序前底盘必须为 Idle 且速度为 0")
        original_mode = int(state.get("WorkMode", TASK_MODE))
        controller = KeyboardController(
            rpc,
            linear_speed=args.linear_speed,
            angular_speed=args.angular_speed,
            send_hz=args.send_hz,
            original_mode=original_mode,
        )
        # systemctl stop / SIGTERM 也走正常停车、恢复模式和注销流程。
        signal.signal(signal.SIGTERM, lambda *_args: controller.root.after(0, controller.close))
        signal.signal(signal.SIGINT, lambda *_args: controller.root.after(0, controller.close))
        controller.run()
        return 0
    except (RpcError, RuntimeError) as exc:
        print(f"启动失败：{exc}")
        rpc.stop()
        return 1
    except KeyboardInterrupt:
        print("收到 Ctrl+C，底盘停车。")
        rpc.stop()
        return 130
    finally:
        rpc.logout()


if __name__ == "__main__":
    raise SystemExit(main())
