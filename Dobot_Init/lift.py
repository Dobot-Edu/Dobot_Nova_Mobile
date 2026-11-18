"""Quest A/B 键控制独立升降柱，通信与机械臂控制循环隔离。"""

from __future__ import annotations

import fcntl
import socket
import struct
import threading
import time
from enum import Enum
from typing import Callable

from Dobot_Init.config import LiftConfig


LIFT_MODBUS_LOCK_PATH = "/tmp/nova_lift_modbus.lock"


class LiftError(RuntimeError):
    """升降柱通信、读数或边界错误。"""


class ModbusResponseError(LiftError):
    """从站明确返回的 Modbus 异常响应。"""

    def __init__(
        self,
        *,
        request_function: int,
        address: int | None,
        exception_code: int,
        raw_pdu: bytes,
    ) -> None:
        self.request_function = request_function
        self.address = address
        self.exception_code = exception_code
        self.raw_pdu = raw_pdu
        address_text = f"，寄存器=0x{address:04X}" if address is not None else ""
        super().__init__(
            "升降台 Modbus 异常响应："
            f"请求功能码=0x{request_function:02X}{address_text}，"
            f"异常码={exception_code}，原始PDU={raw_pdu.hex(' ')}"
        )


def decode_exception_code(response: bytes) -> int:
    """兼容标准一字节异常码，以及现场网关返回的两字节异常码。"""
    if len(response) < 2:
        return -1
    # MAB-M006-2-A-0 前的 TCP/RTU 网关实测返回 `90 00 06`；其中
    # 0x0006 才是完整异常码，表示从站忙，而不是异常码0。
    if len(response) >= 3 and response[1] == 0:
        return int.from_bytes(response[1:3], byteorder="big")
    return response[1]


class LiftDirection(Enum):
    HOLD = "保持"
    UP = "上升"
    DOWN = "下降"


def button_direction(button_a: bool, button_b: bool, *, input_valid: bool) -> LiftDirection:
    """A+B 上升，只有 B 下降；松开、A 单独按下或输入失效均保持。"""
    if not input_valid or not button_b:
        return LiftDirection.HOLD
    return LiftDirection.UP if button_a else LiftDirection.DOWN


def encode_height_cm(height_cm: float) -> tuple[int, int]:
    """按原 Dobot Lua 程序编码厘米目标值，返回低字、高字。"""
    if height_cm < 0:
        raise ValueError("升降台高度不能为负数")
    value = int(round(height_cm * 10_000.0))
    if value == 0:
        return 0, 0
    high = int(-value / 65535) + 65535
    low = 65536 - value % 65535
    # 原 Lua 在余数为零时会产生 65536；在线路上的 U16 表示为 0。
    return low & 0xFFFF, high & 0xFFFF


def decode_height_cm(low: int, high: int) -> float:
    """解码厘米高度；归一化设备在机械零点返回的两个已知原始值。"""
    # Treat only observed near-zero positive counts as mechanical zero.
    # Larger positive feedback values still fail the height range check.
    if high == 0 and 0 <= low <= 3:
        return 0.0
    # Lua 计算得到 65536 时写入 U16 会在线路上变成 0，解码时还原该值。
    formula_low = 65536 if low == 0 else low
    return (((high - 65535) * 65535) + (formula_low - 65536)) * -0.0001


class ModbusTcpClient:
    """本项目所需的最小 Modbus TCP 功能：读、写多个保持寄存器。"""

    def __init__(self, host: str, port: int, unit_id: int, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout_s = timeout_s
        self._transaction = 0

    def read_holding(self, address: int, count: int) -> list[int]:
        response = self._exchange(struct.pack(">BHH", 3, address, count))
        if len(response) != 2 + 2 * count or response[1] != 2 * count:
            raise LiftError("升降台读取寄存器返回长度异常")
        return list(struct.unpack(f">{count}H", response[2:]))

    def write_holding(self, address: int, values: list[int]) -> None:
        if not values or any(not 0 <= value <= 65535 for value in values):
            raise LiftError("升降台写入值超出 U16 范围")
        payload = struct.pack(">BHHB", 16, address, len(values), 2 * len(values))
        payload += struct.pack(f">{len(values)}H", *values)
        # 网关在驱动器切换使能/模式时会返回两字节 0x0006（从站忙）。
        # “忙”明确表示本次请求未执行，可以有限等待后安全重发；其他异常不重试。
        for attempt in range(5):
            try:
                response = self._exchange(payload)
                break
            except ModbusResponseError as error:
                if error.exception_code != 6 or attempt == 4:
                    raise
                delay_s = 0.5 * (attempt + 1)
                print(
                    f"[升降台 Modbus] 从站忙，{delay_s:.1f}s 后重试 "
                    f"0x{address:04X}（{attempt + 1}/4）"
                )
                time.sleep(delay_s)
        if response != struct.pack(">BHH", 16, address, len(values)):
            raise LiftError("升降台写入寄存器确认异常")

    def _exchange(self, pdu: bytes) -> bytes:
        self._transaction = self._transaction % 65535 + 1
        request = struct.pack(">HHHB", self._transaction, 0, len(pdu) + 1, self.unit_id) + pdu
        try:
            # 数据采集器会并行读取同一个503端口；跨进程锁避免设备同时处理两个事务。
            with open(LIFT_MODBUS_LOCK_PATH, "a", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                with socket.create_connection(
                    (self.host, self.port), self.timeout_s
                ) as connection:
                    connection.settimeout(self.timeout_s)
                    connection.sendall(request)
                    header = self._receive_exact(connection, 7)
                    transaction, protocol, length, unit = struct.unpack(">HHHB", header)
                    response = self._receive_exact(connection, length - 1)
        except OSError as error:
            raise LiftError(f"升降台 Modbus 通信失败：{error}") from error
        if transaction != self._transaction or protocol != 0 or unit != self.unit_id:
            raise LiftError("升降台 Modbus 响应头不匹配")
        if not response:
            raise LiftError("升降台 Modbus 返回为空")
        if response[0] & 0x80:
            code = decode_exception_code(response)
            address = struct.unpack(">H", pdu[1:3])[0] if len(pdu) >= 3 else None
            raise ModbusResponseError(
                request_function=pdu[0],
                address=address,
                exception_code=code,
                raw_pdu=response,
            )
        if response[0] != pdu[0]:
            raise LiftError("升降台 Modbus 功能码不匹配")
        return response

    @staticmethod
    def _receive_exact(connection: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                raise LiftError("升降台 Modbus 响应提前结束")
            data.extend(chunk)
        return bytes(data)


class LiftColumn:
    """后台控制升降柱方向和速度，避免 Modbus 阻塞机械臂循环。"""

    CURRENT_HEIGHT_REGISTER = 4615
    ENABLE_REGISTER = 5888
    START_REGISTER = 0x1701
    MODE_REGISTER = 0x1800
    POSITION_REGISTER = 0x1801
    SPEED_REGISTER = 0x1803

    def __init__(self, config: LiftConfig, *, execute: bool, client=None) -> None:
        self.config = config
        self.execute = bool(execute)
        self._client = client
        self._height_cm: float | None = None
        self._desired = LiftDirection.HOLD
        self._applied = LiftDirection.HOLD
        self._error: LiftError | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._initialized = False

    @property
    def minimum_cm(self) -> float | None:
        return 0.0 if self._height_cm is not None else None

    @property
    def height_cm(self) -> float | None:
        """回零点为 0；该变量保存最近一次下发的厘米绝对目标。"""
        return self._height_cm

    @property
    def max_height_cm(self) -> float:
        """配置文件用物理毫米表示行程，升降函数接口使用厘米。"""
        return self.config.max_height_mm / 10.0

    def connect(self, home_signal: Callable[[int], int] | None = None) -> None:
        if not self.config.enabled:
            return
        if not self.execute:
            print("升降台 DRY RUN：不会执行回零、连接或写入设备")
            self._height_cm = 0.0
            return
        if self._client is None:
            self._client = ModbusTcpClient(
                self.config.host,
                self.config.port,
                self.config.unit_id,
                self.config.socket_timeout_s,
            )
        if self.config.home_on_start:
            if home_signal is None:
                raise LiftError("启用升降台启动回零时必须提供 DI 到位信号")
            self.lifting_columns("columns_reste", 0.0, 0, home_signal=home_signal)
        else:
            current = self._read_height()
            if current > self.max_height_cm:
                raise LiftError(
                    f"当前高度 {current:.2f} cm 已超过上限 {self.max_height_cm:.2f} cm"
                )
            self._height_cm = current
        self._thread = threading.Thread(target=self._worker, name="lift-column", daemon=True)
        self._thread.start()
        print(
            f"升降台已连接：目标变量 {self._height_cm:.2f} cm，向下限位 "
            f"{self.max_height_cm:.2f} cm（{self.config.max_height_mm:.0f} mm）；"
            "A+B 物理上升，B 物理下降，松开保持"
        )
        if not self.config.home_on_start and self._height_cm <= 0.05:
            print(
                "[升降台提示] 位置寄存器已在0 cm；上升目标同样是0 cm。"
                "如果物理机构尚未到最高限位，说明坐标零点丢失，请先停止遥操并运行 "
                "python -m tests.hardware.lift_home --execute 完成DI2回零。"
            )

    def prepare_for_control(self) -> None:
        """只使能并选择绝对位置模式；不写位置、速度或启动寄存器。"""
        if not self.config.enabled or not self.execute:
            return
        if self._client is None:
            raise LiftError("升降台 Modbus 尚未连接")
        if not self.config.auto_enable_on_start:
            print("[升降台自检] auto_enable_on_start=false，首次按键时再使能")
            return
        before = self._read_height()
        self._client.write_holding(self.ENABLE_REGISTER, [32])
        time.sleep(0.02)
        self._client.write_holding(self.MODE_REGISTER, [1])
        time.sleep(0.05)
        after = self._read_height()
        if abs(after - before) > 0.2:
            raise LiftError(
                f"升降台就绪过程中高度变化 {before:.3f}->{after:.3f} cm，已阻止继续启动"
            )
        self._height_cm = after
        self._initialized = True
        print(f"[升降台自检] 已使能绝对位置模式，高度 {after:.3f} cm，未下发运动")

    def update_buttons(self, button_a: bool, button_b: bool, *, input_valid: bool) -> None:
        if not self.config.enabled:
            return
        desired = button_direction(button_a, button_b, input_valid=input_valid)
        with self._lock:
            changed = desired is not self._desired
            self._desired = desired
        if changed:
            print(f"[升降台] {desired.value}")
            self._wake.set()

    def check_error(self) -> None:
        if self._error is not None:
            raise self._error

    def read_height(self) -> float:
        """读取升降柱实际反馈高度，供独立测试和诊断使用。"""
        return self._read_height()

    def close(self) -> None:
        if self._thread is None:
            return
        with self._lock:
            self._desired = LiftDirection.HOLD
        self._wake.set()
        # 先让后台线程停止改变目标变量，再结束线程。
        deadline = time.monotonic() + max(1.0, 3 * self.config.socket_timeout_s)
        while self._applied is not LiftDirection.HOLD and time.monotonic() < deadline:
            time.sleep(0.01)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=max(1.0, 3 * self.config.socket_timeout_s))
        self._thread = None

    def _worker(self) -> None:
        period = 1.0 / self.config.command_rate_hz
        try:
            while not self._stop.is_set():
                with self._lock:
                    desired = self._desired
                # 只在按键状态变化时写一次指令，避免每周期重复触发绝对定位。
                if desired is not self._applied:
                    if desired is LiftDirection.HOLD:
                        self._hold_current()
                    else:
                        self._step(desired)
                self._wake.wait(period)
                self._wake.clear()
        except LiftError as error:
            self._error = error

    def _step(self, direction: LiftDirection) -> None:
        # 使用绝对位置模式限定运动边界，实际运动快慢由 0x1803 的速度参数控制。
        target = 0.0 if direction is LiftDirection.UP else self.max_height_cm
        self.lifting_columns("columns_updomn", target, self.config.drive_speed)
        self._applied = direction

    def _hold_current(self) -> None:
        # 沿刚才的运动方向补偿反馈滞后，再重设绝对目标，减少反向回弹。
        held_at = self.stop_motion(self._applied)
        if held_at is not None:
            print(f"[升降台] 补偿后保持目标 {held_at:.3f} cm")
        self._applied = LiftDirection.HOLD

    def stop_motion(self, direction: LiftDirection | None = None) -> float | None:
        """按刚才的方向补偿反馈滞后，并把补偿后高度重发为保持目标。"""
        if not self.execute or not self.config.enabled:
            return None
        actual = self._read_height()
        compensation_cm = self.config.release_compensation_mm / 10.0
        if direction is LiftDirection.DOWN:
            compensated = actual + compensation_cm
        elif direction is LiftDirection.UP:
            compensated = actual - compensation_cm
        else:
            compensated = actual
        target = min(self.max_height_cm, max(0.0, compensated))
        self.lifting_columns("columns_updomn", target, self.config.drive_speed)
        return target

    def _read_height(self) -> float:
        if self._client is None:
            raise LiftError("升降台 Modbus 尚未连接")
        low, high = self._client.read_holding(self.CURRENT_HEIGHT_REGISTER, 2)
        height = decode_height_cm(low, high)
        if not 0.0 <= height <= 1000.0:
            raise LiftError(f"升降台返回异常高度：{height:.3f} cm（原始值 {low}, {high}）")
        return height

    def lifting_columns(
        self,
        columns_state: str,
        height_cm: float,
        speed: int,
        *,
        home_signal: Callable[[int], int] | None = None,
    ) -> None:
        """等效实现 lifting_columns(state, hh, speed)，速度写入 0x1803。"""
        if columns_state == "columns_updomn":
            target = float(height_cm)
            if not 0.0 <= target <= self.max_height_cm:
                raise LiftError(
                    f"升降台目标 {target:.3f} cm 超出 0..{self.max_height_cm:.3f} cm"
                )
            if not 1 <= int(speed) <= 65535:
                raise LiftError("升降台运动速度必须在 1..65535")
            self._write_position(target, int(speed))
            self._height_cm = target
            return
        if columns_state != "columns_reste":
            raise LiftError(f"未知升降柱动作：{columns_state}")
        if home_signal is None:
            raise LiftError("升降柱回零缺少 DI 到位信号")

        # columns_reste：回到最高点，并等待控制柜 DI 到位。
        if self._client is None:
            raise LiftError("升降台 Modbus 尚未连接")
        print("[升降台初始化] 正在回零到物理最高点，请勿进入运动范围……")
        self._client.write_holding(self.ENABLE_REGISTER, [0])
        # MAB-M006-2-A-0 切换驱动状态明显慢于普通寄存器写入；20 ms
        # 会使下一条命令收到 0x0006（从站忙）。
        time.sleep(0.5)
        self._client.write_holding(0x170D, [self.config.home_speed])
        time.sleep(0.2)
        self._client.write_holding(0x170A, [18])
        time.sleep(0.2)
        self._client.write_holding(self.ENABLE_REGISTER, [32])
        time.sleep(0.5)
        self._client.write_holding(self.START_REGISTER, [32])

        deadline = time.monotonic() + self.config.home_timeout_s
        while time.monotonic() < deadline:
            if home_signal(self.config.home_di_index) == 1:
                self._initialized = False
                self._height_cm = 0.0
                print("[升降台初始化] DI2 到位，最高点回零完成")
                return
            time.sleep(0.5)
        raise LiftError(
            f"升降台回零超过 {self.config.home_timeout_s:.1f} 秒，"
            f"DI({self.config.home_di_index}) 仍未到位"
        )

    def _write_speed(self, speed: int) -> None:
        if self._client is None:
            raise LiftError("升降台 Modbus 尚未连接")
        if not 0 <= int(speed) <= 65535:
            raise LiftError("升降台速度必须在 0..65535")
        self._client.write_holding(self.SPEED_REGISTER, [int(speed)])

    def _write_position(self, height_cm: float, speed: int) -> None:
        if self._client is None:
            raise LiftError("升降台 Modbus 尚未连接")
        if not self._initialized:
            self._client.write_holding(self.ENABLE_REGISTER, [32])
            time.sleep(0.02)
            self._client.write_holding(self.MODE_REGISTER, [1])
            time.sleep(0.02)
            self._initialized = True
        # 每次开始运动或重设保持位置时，都显式传入本次定位速度。
        self._write_speed(speed)
        low, high = encode_height_cm(height_cm)
        self._client.write_holding(self.POSITION_REGISTER, [low, high])
        time.sleep(0.02)
        self._client.write_holding(self.START_REGISTER, [16])
