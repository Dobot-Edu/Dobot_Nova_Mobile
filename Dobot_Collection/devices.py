"""只读硬件数据源：Dobot 反馈、升降台、顶部 JPEG 和腕部 RealSense。"""

from __future__ import annotations

import fcntl
import socket
import struct
import threading
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np


LIFT_MODBUS_LOCK_PATH = "/tmp/nova_lift_modbus.lock"


@dataclass(frozen=True)
class TimedValue:
    value: object
    monotonic: float


class Latest:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: TimedValue | None = None

    def set(self, value: object) -> None:
        with self._lock:
            self._item = TimedValue(value, time.monotonic())

    def get(self) -> TimedValue | None:
        with self._lock:
            return self._item


class BackgroundSource:
    def __init__(self, name: str) -> None:
        self.name = name
        self.latest = Latest()
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._guarded_run, name=self.name, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _guarded_run(self) -> None:
        try:
            self.run()
        except Exception as error:
            self.error = error

    def run(self) -> None:
        raise NotImplementedError


class DobotFeedback(BackgroundSource):
    """读取 30004 的 1440 字节实时反馈；不会发送任何指令。"""

    PACKET_SIZE = 1440
    Q_TARGET_OFFSET = 192
    Q_ACTUAL_OFFSET = 432
    MAGIC_OFFSET = 48
    MAGIC = 0x123456789ABCDEF

    def __init__(self, host: str, port: int) -> None:
        super().__init__("dobot-feedback")
        self.host, self.port = host, port

    @classmethod
    def decode(cls, packet: bytes) -> dict:
        if len(packet) != cls.PACKET_SIZE:
            raise ValueError(f"Dobot 反馈长度应为 1440，实际 {len(packet)}")
        length = struct.unpack_from("<q", packet, 0)[0]
        magic = struct.unpack_from("<Q", packet, cls.MAGIC_OFFSET)[0]
        if length != cls.PACKET_SIZE or magic != cls.MAGIC:
            raise ValueError(f"Dobot 反馈帧头异常：len={length}, magic={magic:#x}")
        actual = np.frombuffer(packet, dtype="<f8", count=6, offset=cls.Q_ACTUAL_OFFSET).copy()
        target = np.frombuffer(packet, dtype="<f8", count=6, offset=cls.Q_TARGET_OFFSET).copy()
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(target)):
            raise ValueError("Dobot 反馈关节角含非有限值")
        return {"actual_deg": actual, "target_deg": target}

    def run(self) -> None:
        with socket.create_connection((self.host, self.port), timeout=3.0) as connection:
            connection.settimeout(2.0)
            buffer = bytearray()
            while not self._stop.is_set():
                chunk = connection.recv(8192)
                if not chunk:
                    raise ConnectionError("Dobot 30004 连接断开")
                buffer.extend(chunk)
                while len(buffer) >= self.PACKET_SIZE:
                    packet = bytes(buffer[: self.PACKET_SIZE])
                    del buffer[: self.PACKET_SIZE]
                    self.latest.set(self.decode(packet))


class ModbusHeight(BackgroundSource):
    def __init__(
        self,
        host: str,
        port: int,
        unit_id: int,
        register: int,
        poll_hz: float,
        socket_timeout_s: float = 0.25,
        max_consecutive_errors: int = 3,
        retry_delay_s: float = 0.03,
    ) -> None:
        super().__init__("lift-height")
        self.host, self.port, self.unit_id = host, port, unit_id
        self.register, self.poll_hz = register, poll_hz
        self.socket_timeout_s = float(socket_timeout_s)
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.retry_delay_s = float(retry_delay_s)
        if (
            self.poll_hz <= 0
            or self.socket_timeout_s <= 0
            or self.max_consecutive_errors <= 0
            or self.retry_delay_s < 0
        ):
            raise ValueError("升降台采集轮询、超时和重试参数必须为正数")
        self._transaction = 0

    @staticmethod
    def decode_height_cm(low: int, high: int) -> float:
        if (low, high) in {(0, 0), (1, 0)}:
            return 0.0
        formula_low = 65536 if low == 0 else low
        return (((high - 65535) * 65535) + (formula_low - 65536)) * -0.0001

    def _read(self) -> float:
        self._transaction = self._transaction % 65535 + 1
        pdu = struct.pack(">BHH", 3, self.register, 2)
        request = struct.pack(">HHHB", self._transaction, 0, len(pdu) + 1, self.unit_id) + pdu
        # 遥操和数据采集是两个进程，必须避免同时向升降柱发Modbus事务。
        # flock在进程异常退出时由内核自动释放，不会留下永久死锁。
        with open(LIFT_MODBUS_LOCK_PATH, "a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            with socket.create_connection(
                (self.host, self.port), timeout=self.socket_timeout_s
            ) as connection:
                connection.settimeout(self.socket_timeout_s)
                connection.sendall(request)
                response = _recv_exact(connection, 13)
        transaction, protocol, length, unit = struct.unpack(">HHHB", response[:7])
        if (transaction, protocol, length, unit, response[7], response[8]) != (
            self._transaction, 0, 7, self.unit_id, 3, 4
        ):
            raise ValueError("升降台 Modbus 读高返回异常")
        low, high = struct.unpack(">HH", response[9:13])
        height = self.decode_height_cm(low, high)
        if not 0.0 <= height <= 1000.0:
            raise ValueError(f"升降台高度异常 {height:.3f} cm，原始值 {low},{high}")
        return height

    def run(self) -> None:
        period = 1.0 / self.poll_hz
        consecutive_errors = 0
        while not self._stop.wait(0):
            try:
                height = self._read()
            except (OSError, ValueError) as error:
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    raise RuntimeError(
                        f"升降台连续{consecutive_errors}次读取失败，最后错误：{error}"
                    ) from error
                print(
                    f"[升降台采集警告] 读取失败 "
                    f"{consecutive_errors}/{self.max_consecutive_errors}，短暂重试：{error}"
                )
                if self._stop.wait(self.retry_delay_s):
                    return
                continue
            consecutive_errors = 0
            self.latest.set(height)
            self._stop.wait(period)


class TopHttpCamera(BackgroundSource):
    def __init__(self, url: str, timeout_s: float) -> None:
        super().__init__("top-http-camera")
        self.url, self.timeout_s = url, timeout_s

    def run(self) -> None:
        while not self._stop.is_set():
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as response:
                encoded = np.frombuffer(response.read(), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("顶部相机 JPEG 解码失败")
            self.latest.set(image)


class WristRealSense(BackgroundSource):
    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        super().__init__("wrist-realsense")
        self.serial, self.width, self.height, self.fps = serial, width, height, fps

    def run(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        pipeline.start(config)
        try:
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                color = frames.get_color_frame()
                if color:
                    self.latest.set(np.asanyarray(color.get_data()).copy())
        finally:
            pipeline.stop()


class TelemetryReceiver(BackgroundSource):
    def __init__(self, host: str, port: int) -> None:
        super().__init__("vr-dataset-telemetry")
        self.host, self.port = host, port

    def run(self) -> None:
        import json

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.host, self.port))
            sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    payload, source = sock.recvfrom(8192)
                except socket.timeout:
                    continue
                value = json.loads(payload.decode("utf-8"))
                if source[0] not in {"127.0.0.1", "::1"}:
                    continue
                self.latest.set(value)


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("响应提前结束")
        data.extend(chunk)
    return bytes(data)
