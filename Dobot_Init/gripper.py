"""Nova USB Feetech夹爪的位置控制、编码器反馈和电流限力。"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path

from Dobot_Init.config import GripperConfig


class GripperError(RuntimeError):
    """夹爪串口通信或状态错误。"""


class GripperTransportError(GripperError):
    """可重试的串口超时、帧头或校验错误。"""


class GripperStatusError(GripperError):
    """电机主动上报的过流、过温等保护状态。"""

    def __init__(self, status: int) -> None:
        self.status = int(status)
        super().__init__(
            f"夹爪状态错误：{self.status}（{describe_servo_status(self.status)}）；"
            "请松开扳机并等待舵机降温，勿忽略保护状态"
        )


@dataclass(frozen=True)
class GripperFeedback:
    """舵机只读反馈；电流用于近似判断夹持力。"""

    position_raw: int
    load_raw: int
    temperature_c: int
    moving: bool
    current_ma: float
    status_error: int = 0


SERVO_STATUS_FLAGS = {
    1: "电压异常",
    2: "位置传感器异常",
    4: "过温保护",
    8: "过流保护",
    32: "过载/堵转保护",
}

# 用户要求过温/过流只在终端提醒，不让遥操或标定进程退出。舵机固件
# 自身是否接受运动仍由硬件决定；32继续按夹持接触处理并停止加深闭合。
NON_FATAL_STATUS_ERRORS = (0, 4, 8, 12, 32, 36, 40, 44)


def describe_servo_status(error: int) -> str:
    """把飞特状态位拆成可操作的中文原因；12 即过温(4)+过流(8)。"""
    value = int(error)
    reasons = [name for bit, name in SERVO_STATUS_FLAGS.items() if value & bit]
    known_mask = sum(SERVO_STATUS_FLAGS)
    unknown = value & ~known_mask
    if unknown:
        reasons.append(f"未知状态位 {unknown:#04x}")
    return " + ".join(reasons) if reasons else "正常"


def trigger_command(trigger: float) -> int:
    """把 Quest 的 0..1 模拟扳机转换为 0..255 夹爪命令。"""
    return int(round(min(max(float(trigger), 0.0), 1.0) * 255.0))


def raw_position(command: int, config: GripperConfig) -> int:
    """把 0..255 开合命令换算为当前实物夹爪的原始位置。"""
    value = min(max(int(command), 0), 255)
    return int(
        round(config.raw_open + value / 255.0 * (config.raw_close - config.raw_open))
    )


def position_command(position_raw: int, config: GripperConfig) -> int:
    """把舵机实际位置反算为内部 0..255 闭合命令。"""
    ratio = (float(position_raw) - config.raw_open) / (
        config.raw_close - config.raw_open
    )
    return int(round(min(max(ratio, 0.0), 1.0) * 255.0))


def feedback_position(position_raw: int, config: GripperConfig) -> float:
    """按独立编码器端点得到实测位置：1=张开，0=闭合。"""
    closed_ratio = (float(position_raw) - config.feedback_raw_open) / (
        config.feedback_raw_close - config.feedback_raw_open
    )
    return 1.0 - float(min(max(closed_ratio, 0.0), 1.0))


def feetech_signed(value: int) -> int:
    """解码飞特舵机使用的符号位+数值格式。"""
    value = int(value) & 0xFFFF
    return -(value - 0x8000) if value & 0x8000 else value


class TriggerGripper:
    """按住扳机逐步闭合，松开逐步张开；数据动作始终是目标位置。"""

    def __init__(self, config: GripperConfig, *, execute: bool) -> None:
        self.config = config
        self.execute = bool(execute)
        self._serial = None
        self._last_command: int | None = None
        self._last_status_error = 0
        self._last_feedback_at = 0.0
        self._last_feedback: GripperFeedback | None = None
        self._last_feedback_monotonic_ns = 0
        self._processed_feedback_monotonic_ns = 0
        self._feedback_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._feedback_stop = threading.Event()
        self._feedback_worker: threading.Thread | None = None
        self._feedback_error: GripperError | None = None
        self._over_current_samples = 0
        self._current_baseline_ma = 0.0
        self._force_latched = False
        self._hardware_overload_latched = False
        self._hold_command: int | None = None
        self._closing_active = False
        self._transport_failure_count = 0
        self._last_transport_warning_at = 0.0
        self._last_status_warning = 0
        self._last_status_warning_at = 0.0
        self._trigger_closing: bool | None = None
        self._toggle_armed = False
        self._toggle_closed = False
        self._last_update_at: float | None = None

    @property
    def last_command(self) -> int | None:
        return self._last_command

    @property
    def last_current_ma(self) -> float | None:
        with self._feedback_lock:
            return None if self._last_feedback is None else self._last_feedback.current_ma

    @property
    def target_position(self) -> float:
        """任务允许行程内的目标位置：1=张开，0=闭合。"""
        command = 0 if self._last_command is None else self._last_command
        closed_ratio = min(max(command / self.config.max_close_command, 0.0), 1.0)
        return float(1.0 - closed_ratio)

    def feedback_snapshot(self) -> dict:
        """返回采集所需的实测位置和诊断量，不用目标命令冒充反馈。"""
        with self._feedback_lock:
            feedback = self._last_feedback
            timestamp_ns = self._last_feedback_monotonic_ns
        if feedback is None:
            return {
                "position": self.target_position if not self.execute else 1.0,
                "position_raw": None,
                "current_ma": None,
                "load_raw": None,
                "temperature_c": None,
                "moving": False,
                "status_error": None,
                "current_baseline_ma": self._current_baseline_ma,
                "current_threshold_ma": self._current_baseline_ma + self.config.current_limit_ma,
                "feedback_monotonic_ns": 0,
                "feedback_valid": not self.execute,
            }
        return {
            "position": feedback_position(feedback.position_raw, self.config),
            "position_raw": int(feedback.position_raw),
            "current_ma": float(feedback.current_ma),
            "load_raw": int(feedback.load_raw),
            "temperature_c": int(feedback.temperature_c),
            "moving": bool(feedback.moving),
            "status_error": int(feedback.status_error),
            "current_baseline_ma": float(self._current_baseline_ma),
            "current_threshold_ma": float(
                self._current_baseline_ma + self.config.current_limit_ma
            ),
            "feedback_monotonic_ns": int(timestamp_ns),
            "feedback_valid": bool(timestamp_ns),
        }

    def check_error(self) -> None:
        if self._feedback_error is not None:
            raise self._feedback_error

    def connect(self, *, initialize_open: bool = True) -> None:
        """连接夹爪。

        正常遥操/采集保持默认行为，连接后回到配置的安全全开位。
        只有端点标定工具会传入 ``initialize_open=False``，避免尚未校准的
        旧开端点在连接瞬间继续顶住机械限位。
        """
        if not self.config.enabled:
            return
        if not self.execute:
            self._last_command = 0
            controls = (
                "每次点按Trigger切换全闭/全开"
                if self.config.mode == "toggle"
                else "按住Trigger闭合，松开Trigger张开"
            )
            print(f"夹爪 DRY RUN：初始化全开；{controls}")
            return

        try:
            import serial

            self._serial = serial.Serial(
                self.config.port,
                baudrate=1_000_000,
                timeout=0.10,
                write_timeout=0.10,
            )
            self._tune_latency()
            # 32按接触处理；过温/过流仅提醒，不终止遥操。
            self._transaction(0x01, b"", allowed_errors=NON_FATAL_STATUS_ERRORS)
            if self._last_status_error & 32:
                print("[夹爪自检] 检测到单纯过载/堵转，正在执行一次力矩复位")
            self._write(40, bytes([0]), allowed_errors=NON_FATAL_STATUS_ERRORS)
            time.sleep(0.2)
            self._write(40, bytes([1]), allowed_errors=NON_FATAL_STATUS_ERRORS)
            if initialize_open:
                # 用户确认启动后明确张开，保证位置控制从安全的全开端开始。
                self._move(0)
                self._last_command = 0
                # 等待安全开端落稳，避免把启动运动电流当作空载基线。
                time.sleep(0.4)
            # 首条编码器位置必须真实可读，不能在数据里静默退回目标命令代理。
            feedback = self._read_feedback()
            if not initialize_open:
                self._last_command = min(
                    position_command(feedback.position_raw, self.config),
                    self.config.max_close_command,
                )
            self._current_baseline_ma = feedback.current_ma
            self._store_feedback(feedback)
            self._feedback_stop.clear()
            self._feedback_worker = threading.Thread(
                target=self._feedback_loop,
                name="gripper-feedback",
                daemon=True,
            )
            self._feedback_worker.start()
        except Exception as error:
            self.close()
            if isinstance(error, GripperError):
                raise
            raise GripperError(f"无法连接夹爪 {self.config.port}：{error}") from error
        startup = "初始化全开" if initialize_open else "保持当前位置（标定模式）"
        protection = (
            f"软件限力阈值约={self._current_baseline_ma + self.config.current_limit_ma:.0f}mA"
            if self.config.force_feedback_enabled
            else "软件电流锁停已关闭（舵机硬件保护保留）"
        )
        overload_32 = "状态32锁停=开启" if self.config.stop_on_overload_32 else "状态32锁停=关闭（仅提醒）"
        controls = (
            "每次点按Trigger切换全闭/全开"
            if self.config.mode == "toggle"
            else "按住Trigger闭合，松开Trigger张开"
        )
        print(
            f"夹爪已连接：{self.config.port}，ID={self.config.servo_id}；"
            f"{startup}；{controls}；{1.0 / self.config.feedback_interval_s:.0f}Hz位置反馈已启用；"
            f"{protection}；{overload_32}"
        )

    def connect_read_only(self) -> GripperFeedback:
        """只连接并读取一次状态；不切换力矩，也不发送位置目标。"""
        if not self.config.enabled:
            raise GripperError("当前配置未启用夹爪")
        if not self.execute:
            raise GripperError("只读硬件检查仍需使用实机模式")
        try:
            import serial

            self._serial = serial.Serial(
                self.config.port,
                baudrate=1_000_000,
                timeout=0.10,
                write_timeout=0.10,
            )
            self._tune_latency()
            self._transaction(0x01, b"", allowed_errors=tuple(range(256)))
            feedback = self._read_feedback()
            self._store_feedback(feedback)
            return feedback
        except Exception as error:
            self.close()
            if isinstance(error, GripperError):
                raise
            raise GripperError(f"无法只读连接夹爪 {self.config.port}：{error}") from error

    def move_raw_for_calibration(self, position_raw: int, *, torque: int = 150) -> None:
        """以较低力矩移动到原始编码器目标，仅供交互式端点标定工具使用。"""
        if not self.execute or self._serial is None:
            raise GripperError("夹爪未以实机模式连接")
        raw = int(position_raw)
        if not 900 <= raw <= 2150:
            raise GripperError(f"标定原始目标 {raw} 超出保守范围 [900, 2150]")
        speed = int(round(self.config.speed / 100.0 * 4096.0))
        goal_torque = min(max(int(torque), 1), 1023)
        payload = bytes(
            [
                0,
                raw & 0xFF,
                (raw >> 8) & 0xFF,
                goal_torque & 0xFF,
                (goal_torque >> 8) & 0xFF,
                speed & 0xFF,
                (speed >> 8) & 0xFF,
            ]
        )
        self._write(41, payload, allowed_errors=NON_FATAL_STATUS_ERRORS)

    def disable_torque(self) -> None:
        """关闭夹爪力矩；用于标定异常或退出时卸载。"""
        if self._serial is not None:
            self._write(40, bytes([0]), allowed_errors=tuple(range(256)))

    def update(self, trigger: float, *, input_valid: bool) -> None:
        """按配置执行按住渐变或点按切换；输入失效时保持当前位置。"""
        self.check_error()
        if not self.config.enabled or not input_valid:
            self._last_update_at = time.monotonic()
            if self.config.mode == "toggle":
                self._toggle_armed = False
            return

        now = time.monotonic()
        dt = 0.0 if self._last_update_at is None else min(now - self._last_update_at, 0.1)
        self._last_update_at = now
        trigger = min(max(float(trigger), 0.0), 1.0)
        pressed = trigger >= self.config.trigger_press_threshold
        released = trigger <= self.config.trigger_release_threshold
        if self._trigger_closing is None:
            # 启动时正按住扳机必须先松开，避免连接瞬间意外夹紧。
            self._trigger_closing = False
            if self._last_command is None:
                self._last_command = 0
            self._toggle_closed = self.target_position <= 0.5
            self._toggle_armed = released
            if pressed:
                print(f"[夹爪按键基准] Trigger={trigger:.2f}，请先松开再操作")
            return

        if self.config.mode == "toggle":
            if released:
                self._toggle_armed = True
                return
            if not pressed or not self._toggle_armed:
                return
            self._toggle_armed = False
            self._toggle_closed = not self._toggle_closed
            self._trigger_closing = self._toggle_closed
            target = 0.0 if self._toggle_closed else 1.0
            print(f"[夹爪切换] {'闭合' if self._toggle_closed else '张开'}")
            self.set_target_position(target)
            return

        previous_direction = self._trigger_closing
        if pressed:
            self._trigger_closing = True
        elif released:
            self._trigger_closing = False
        if self._trigger_closing != previous_direction:
            print(f"[夹爪点动] {'按住：闭合' if self._trigger_closing else '松开：张开'}")

        rate = -self.config.close_rate_per_s if self._trigger_closing else self.config.open_rate_per_s
        requested_position = self.target_position + rate * dt
        self.set_target_position(requested_position)

    def set_target_position(self, position: float) -> None:
        """下发连续目标位置（1=张开，0=闭合）；遥操和推理共用。"""
        self.check_error()
        value = min(max(float(position), 0.0), 1.0)
        requested = int(round((1.0 - value) * self.config.max_close_command))
        if self._last_command is None:
            self._last_command = 0

        feedback = self._read_feedback_if_due()
        if feedback is not None:
            self._transport_failure_count = 0
            self._update_force_latch(feedback, requested)

        opening = requested < self._last_command
        if opening:
            if self._hardware_overload_latched:
                self._recover_hardware_overload()
            self._closing_active = False
            self._over_current_samples = 0
            if self._force_latched:
                self._force_latched = False
                self._hold_command = None
                print("[夹爪保持] 已开始张开，解除接触/限力保持")
        elif self._force_latched:
            return

        if abs(requested - self._last_command) < self.config.command_deadband:
            return
        self._closing_active = requested > self._last_command
        if self.execute and not self._move_resilient(requested):
            return
        self._last_command = requested

    def _move_resilient(self, command: int, *, torque: int | None = None) -> bool:
        """偶发串口错误只跳过本帧，不终止遥操和采集。"""
        try:
            self._move(command, torque=torque)
        except GripperStatusError as error:
            if error.status != 32:
                raise
            if self.config.stop_on_overload_32:
                with self._feedback_lock:
                    feedback = self._last_feedback
                self._latch_hardware_overload(feedback)
            else:
                print(
                    "[夹爪硬件状态提醒] 位置指令返回32；"
                    "软件32锁停已关闭，本帧跳过并在下一控制帧继续尝试闭合"
                )
            return False
        except GripperTransportError as error:
            self._report_transport_failure(error)
            return False
        self._transport_failure_count = 0
        return True

    def _latch_hardware_overload(self, feedback: GripperFeedback | None) -> None:
        """把纯32视为已经夹到物体，停止继续加深目标。"""
        if self._hardware_overload_latched:
            return
        if feedback is not None:
            hold_command = min(
                position_command(feedback.position_raw, self.config),
                self.config.max_close_command,
            )
            self._last_command = hold_command
        else:
            hold_command = 0 if self._last_command is None else self._last_command
        self._hold_command = hold_command
        self._force_latched = True
        self._hardware_overload_latched = True
        self._closing_active = False
        print(
            f"[夹爪接触] 舵机上报纯32（夹持/堵转），"
            f"停止继续夹紧并保持在 {hold_command}/255；松开Trigger后自动恢复张开"
        )

    def _recover_hardware_overload(self) -> None:
        """开始张开前复位纯32；其他硬件错误仍由状态检查抛出。"""
        self._write(40, bytes([0]), allowed_errors=NON_FATAL_STATUS_ERRORS)
        time.sleep(0.2)
        self._write(40, bytes([1]), allowed_errors=NON_FATAL_STATUS_ERRORS)
        self._hardware_overload_latched = False
        print("[夹爪接触] 已复位纯32，恢复张开")

    def _report_transport_failure(self, error: GripperTransportError) -> None:
        self._transport_failure_count += 1
        now = time.monotonic()
        if self._transport_failure_count == 1 or now - self._last_transport_warning_at >= 2.0:
            print(
                f"[夹爪通信警告] {error}；已跳过本帧，"
                "VR遥测和数据采集继续"
            )
            self._last_transport_warning_at = now

    def close(self) -> None:
        self._feedback_stop.set()
        if self._feedback_worker is not None:
            self._feedback_worker.join(timeout=1.0)
            self._feedback_worker = None
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def _move(self, command: int, *, torque: int | None = None) -> None:
        raw = raw_position(command, self.config)
        speed = int(round(self.config.speed / 100.0 * 4096.0))
        goal_torque = self.config.target_torque if torque is None else int(torque)
        payload = bytes(
            [
                0,
                raw & 0xFF,
                (raw >> 8) & 0xFF,
                goal_torque & 0xFF,
                (goal_torque >> 8) & 0xFF,
                speed & 0xFF,
                (speed >> 8) & 0xFF,
            ]
        )
        # 32继续作为夹持接触；过温/过流只打印硬件状态，不终止控制进程。
        self._write(41, payload, allowed_errors=NON_FATAL_STATUS_ERRORS)

    def _read_feedback_if_due(self) -> GripperFeedback | None:
        with self._feedback_lock:
            if self._last_feedback_monotonic_ns == self._processed_feedback_monotonic_ns:
                return None
            self._processed_feedback_monotonic_ns = self._last_feedback_monotonic_ns
            return self._last_feedback

    def _store_feedback(self, feedback: GripperFeedback) -> None:
        # 空载基线只在connect()的安全开端落稳后标定，此处禁止持续更新。
        # 否则端点负载会被误学成空载，逐渐抬高限力阈值。
        with self._feedback_lock:
            self._last_feedback = feedback
            self._last_feedback_at = time.monotonic()
            self._last_feedback_monotonic_ns = time.monotonic_ns()

    def _feedback_loop(self) -> None:
        while not self._feedback_stop.is_set():
            started = time.monotonic()
            try:
                self._store_feedback(self._read_feedback())
                self._transport_failure_count = 0
            except GripperTransportError as error:
                self._report_transport_failure(error)
            except GripperError as error:
                self._feedback_error = error
                return
            elapsed = time.monotonic() - started
            self._feedback_stop.wait(max(0.005, self.config.feedback_interval_s - elapsed))

    def _read_feedback(self) -> GripperFeedback:
        # 56..70：位置、速度、负载、电压、温度、状态、移动标志和电流。
        # 32表示夹持接触；过温/过流状态只提醒并继续返回遥测。
        data = self._transaction(
            0x02, bytes([56, 15]), allowed_errors=NON_FATAL_STATUS_ERRORS
        )
        if len(data) != 15:
            raise GripperError(f"夹爪反馈长度异常：{len(data)}")

        def u16(offset: int) -> int:
            return data[offset] | (data[offset + 1] << 8)

        current_raw = feetech_signed(u16(13))
        return GripperFeedback(
            position_raw=feetech_signed(u16(0)),
            load_raw=feetech_signed(u16(4)),
            temperature_c=data[7],
            moving=bool(data[10]),
            # HTS/SMS 配套 SDK 标定为每个原始单位 6.5 mA。
            current_ma=abs(current_raw) * 6.5,
            status_error=self._last_status_error,
        )

    def _update_force_latch(
        self, feedback: GripperFeedback, requested: int
    ) -> None:
        if (
            feedback.status_error & 32
            and self._closing_active
            and self.config.stop_on_overload_32
        ):
            self._latch_hardware_overload(feedback)
            return
        if not self.config.force_feedback_enabled:
            self._over_current_samples = 0
            return
        if self._force_latched:
            return
        # 只有仍在请求闭合时才累计，张开过程中的电流不能触发夹持锁定。
        current_threshold_ma = self._current_baseline_ma + self.config.current_limit_ma
        if self._closing_active and feedback.current_ma >= current_threshold_ma:
            self._over_current_samples += 1
        else:
            self._over_current_samples = 0
        if self._over_current_samples < self.config.current_trip_samples:
            return

        hold_command = min(
            position_command(feedback.position_raw, self.config),
            self.config.max_close_command,
        )
        if not self._move_resilient(hold_command, torque=self.config.hold_torque):
            return
        self._last_command = hold_command
        self._hold_command = hold_command
        self._force_latched = True
        print(
            f"[夹爪限力] 电流连续 {self._over_current_samples} 次达到 "
            f"{feedback.current_ma:.0f} mA（阈值{current_threshold_ma:.0f} mA），"
            f"停止在 {hold_command}/255，"
            f"保持力矩={self.config.hold_torque}"
        )

    def _write(
        self,
        address: int,
        payload: bytes,
        *,
        allowed_errors: tuple[int, ...] = (0,),
    ) -> None:
        self._transaction(
            0x03,
            bytes([address]) + payload,
            allowed_errors=allowed_errors,
        )

    def _transaction(
        self,
        instruction: int,
        parameters: bytes,
        *,
        allowed_errors: tuple[int, ...] = (0,),
    ) -> bytes:
        last_error: GripperTransportError | None = None
        # 一个USB串口不能并发收发；位置反馈线程和运动命令在这里串行化。
        with self._io_lock:
            # USB 串口偶发丢帧时重发幂等指令；三次失败才交给上层隔离。
            for attempt in range(3):
                try:
                    result = self._transaction_once(
                        instruction,
                        parameters,
                        allowed_errors=allowed_errors,
                    )
                    if attempt:
                        print(f"[夹爪通信恢复] 第 {attempt + 1} 次尝试成功")
                    return result
                except GripperTransportError as error:
                    last_error = error
                    if attempt < 2:
                        time.sleep(0.005)
        assert last_error is not None
        raise last_error

    def _transaction_once(
        self,
        instruction: int,
        parameters: bytes,
        *,
        allowed_errors: tuple[int, ...] = (0,),
    ) -> bytes:
        if self._serial is None:
            raise GripperError("夹爪串口尚未连接")
        servo_id = self.config.servo_id
        length = len(parameters) + 2
        body = bytes([servo_id, length, instruction]) + parameters
        packet = b"\xFF\xFF" + body + bytes([(~sum(body)) & 0xFF])
        try:
            self._serial.reset_input_buffer()
            self._serial.write(packet)
            self._serial.flush()
        except (OSError, TimeoutError) as error:
            raise GripperTransportError(f"夹爪串口写入失败：{error}") from error

        header = self._read_exact(4)
        if header[:2] != b"\xFF\xFF" or header[2] != servo_id:
            raise GripperTransportError(f"夹爪返回头异常：{header.hex(' ')}")
        response_length = header[3]
        tail = self._read_exact(response_length)
        response_body = header[2:] + tail[:-1]
        if ((~sum(response_body)) & 0xFF) != tail[-1]:
            raise GripperTransportError("夹爪返回校验失败")
        error = tail[0]
        self._last_status_error = error
        if error not in allowed_errors:
            raise GripperStatusError(error)
        if error:
            now = time.monotonic()
            if error != self._last_status_warning or now - self._last_status_warning_at >= 2.0:
                print(
                    f"[夹爪硬件状态提醒] {error}（{describe_servo_status(error)}）；"
                    "软件继续运行，舵机固件保护仍然有效"
                )
                self._last_status_warning = error
                self._last_status_warning_at = now
        return tail[1:-1]

    def _read_exact(self, size: int) -> bytes:
        assert self._serial is not None
        data = bytearray()
        while len(data) < size:
            try:
                chunk = self._serial.read(size - len(data))
            except (OSError, TimeoutError) as error:
                raise GripperTransportError(f"夹爪串口读取失败：{error}") from error
            if not chunk:
                raise GripperTransportError("夹爪串口响应超时")
            data.extend(chunk)
        return bytes(data)

    def _tune_latency(self) -> None:
        device = Path(self.config.port).resolve().name
        latency = Path("/sys/bus/usb-serial/devices") / device / "latency_timer"
        try:
            latency.write_text("1")
        except (FileNotFoundError, PermissionError, OSError):
            pass
