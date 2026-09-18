"""UDP 接收和 Quest 主机自动发现。"""

from __future__ import annotations

import json
import ipaddress
import socket
import threading
import time
from dataclasses import dataclass

from quest_vr.protocol import VRPacket, decode_packet


DISCOVERY_REQUEST = b"DOBOT_CRTRAINER_DISCOVER_V1"


@dataclass(frozen=True)
class ReceivedPacket:
    packet: VRPacket
    received_at: float
    source: tuple[str, int]


class UDPReceiver:
    """非阻塞接收；锁定第一个合法发送端，拒绝中途换源。"""

    def __init__(
        self,
        host: str,
        port: int,
        allowed_ip: str = "",
        allowed_subnet: str = "",
    ) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.allowed_ip = allowed_ip.strip()
        self.allowed_network = (
            ipaddress.ip_network(allowed_subnet.strip(), strict=False)
            if allowed_subnet.strip()
            else None
        )
        self.locked_source: tuple[str, int] | None = None
        self.invalid_count = 0

    def poll(self) -> ReceivedPacket | None:
        latest = None
        while True:
            try:
                data, source = self.socket.recvfrom(4097)
            except BlockingIOError:
                break
            try:
                if self.allowed_ip and source[0] != self.allowed_ip:
                    raise ValueError("发送 IP 不是指定 Quest")
                if self.allowed_network is not None and (
                    ipaddress.ip_address(source[0]) not in self.allowed_network
                ):
                    raise ValueError("发送 IP 不在 Quest 允许网段")
                if self.locked_source is not None and source != self.locked_source:
                    raise ValueError("接收器已锁定其他发送端")
                packet = decode_packet(data)
                if self.locked_source is None:
                    self.locked_source = source
                latest = ReceivedPacket(packet, time.monotonic(), source)
            except ValueError:
                self.invalid_count += 1
        return latest

    def close(self) -> None:
        self.socket.close()


class DiscoveryResponder:
    """回应 Quest APK 的 UDP 广播，让 APK 自动找到算力板。"""

    def __init__(self, host: str, discovery_port: int, data_port: int) -> None:
        self.host = host
        self.discovery_port = discovery_port
        self.data_port = data_port
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.discovery_port))
        self._socket.settimeout(0.2)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._socket is not None
        response = json.dumps(
            {"type": "dobot_crtrainer_host", "version": 1, "port": self.data_port}
        ).encode("utf-8")
        while not self._stop.is_set():
            try:
                request, sender = self._socket.recvfrom(512)
                if request == DISCOVERY_REQUEST:
                    self._socket.sendto(response, sender)
            except socket.timeout:
                continue
            except OSError:
                break

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
