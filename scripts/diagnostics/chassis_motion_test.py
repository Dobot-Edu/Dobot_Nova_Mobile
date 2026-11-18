#!/usr/bin/env python3
"""华睿底盘独立通信/短距离移动验证工具。

依据《华睿机器人设备对接协议说明 V1.032》实现：
底盘作为 TCP 客户端主动连接本脚本，随后依次完成注册、登录、心跳、
addTask 和带 Run 动作的 addTaskSlice。

默认仅登录并显示状态，不会让底盘运动。只有同时传入 --execute 并在现场
输入指定确认词，才执行一次低速短距离前进和原路后退测试。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any


DEFAULT_CHASSIS_IP = "192.168.23.1"
DEFAULT_COMPUTE_IP = "192.168.23.204"
DEFAULT_PORT = 3002
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_TEST_DISTANCE_MM = 300
MAX_TEST_SPEED_MM_S = 150


class ProtocolError(RuntimeError):
    """收到不符合协议的数据。"""


def crc16_modbus(data: bytes) -> int:
    """CRC16-MODBUS，初值 0xFFFF，多项式 0xA001。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError("底盘关闭了 TCP 连接")
        data.extend(part)
    return bytes(data)


class JsonFramer:
    """支持协议中的一代帧，并自动识别/支持二代帧。"""

    def __init__(self, sock: socket.socket, generation: str = "auto") -> None:
        self.sock = sock
        self.generation = generation

    def receive(self) -> dict[str, Any]:
        first = recv_exact(self.sock, 1)
        if self.generation == "auto":
            self.generation = "v2" if first == b"\x02" else "v1"
            print(f"[通信] 自动识别为 {self.generation} 帧格式")
        payload = self._receive_v2(first) if self.generation == "v2" else self._receive_v1(first)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"JSON 解码失败: {exc}") from exc
        if not isinstance(value, dict):
            raise ProtocolError("JSON 顶层必须是对象")
        return value

    def send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self.generation == "auto":
            raise ProtocolError("尚未收到注册帧，无法确定发送格式")
        if self.generation == "v1":
            self.sock.sendall(struct.pack(">I", len(payload)) + payload)
            return

        # 二代协议：28 字节小端消息头；消息类型、正文校验及帧尾按大端。
        message_type = b"\x00\x00"
        # 实机抓包确认“消息长度”包含 2 字节消息类型和 JSON 数据。
        body = message_type + payload
        prefix = struct.pack("<BBHIH16s", 0x02, ord("T"), 0, len(body), 0, bytes(16))
        header = prefix + struct.pack("<H", crc16_modbus(prefix))
        body_crc = struct.pack(">H", crc16_modbus(body))
        self.sock.sendall(header + body + body_crc + b"\x03")

    def _receive_v1(self, first: bytes) -> bytes:
        raw_length = first + recv_exact(self.sock, 3)
        length = struct.unpack(">I", raw_length)[0]
        if not 0 < length <= MAX_FRAME_BYTES:
            raise ProtocolError(f"一代帧长度异常: {length}")
        return recv_exact(self.sock, length)

    def _receive_v2(self, first: bytes) -> bytes:
        header = first + recv_exact(self.sock, 27)
        stx, direction, reserved, length, _version, extension, received_crc = struct.unpack(
            "<BBHIH16sH", header
        )
        if stx != 0x02 or direction not in (ord("V"), ord("C")):
            raise ProtocolError("二代帧头或消息方向错误")
        if reserved != 0 or extension != bytes(16):
            raise ProtocolError("二代帧保留字段不为 0")
        if crc16_modbus(header[:26]) != received_crc:
            raise ProtocolError("二代帧消息头 CRC 错误")
        if not 0 < length <= MAX_FRAME_BYTES:
            raise ProtocolError(f"二代帧长度异常: {length}")
        if length < 3:
            raise ProtocolError(f"二代帧消息体长度过小: {length}")
        body = recv_exact(self.sock, length)
        message_type, payload = body[:2], body[2:]
        received_body_crc = struct.unpack(">H", recv_exact(self.sock, 2))[0]
        if crc16_modbus(body) != received_body_crc:
            raise ProtocolError("二代帧消息体 CRC 错误")
        if recv_exact(self.sock, 1) != b"\x03":
            raise ProtocolError("二代帧缺少 ETX 帧尾")
        return payload


@dataclass
class ChassisState:
    x_mm: int
    y_mm: int
    yaw_mdeg: int
    speed_mm_s: int
    state: str
    extra_first: int
    extra_second: int
    work_mode: int
    last_code: str
    task_id: str
    task_state: str

    @classmethod
    def from_report(cls, message: dict[str, Any]) -> "ChassisState":
        body = message.get("Body", {})
        state = body.get("State", {})
        robot_task = body.get("RobotTaskState", {})
        position = state.get("Position", [])
        if not isinstance(position, list) or len(position) != 2:
            raise ProtocolError("DeviceState 中缺少二维 Position")
        return cls(
            x_mm=int(position[0]),
            y_mm=int(position[1]),
            yaw_mdeg=int(state.get("Oritation", 0)),  # 协议原文即拼作 Oritation
            speed_mm_s=int(state.get("Speed", 0)),
            state=str(state.get("State", "Unknown")),
            extra_first=int(state.get("ExtraStateFirst", 0)),
            extra_second=int(state.get("ExtraStateSecond", 0)),
            work_mode=int(state.get("WorkMode", -1)),
            last_code=str(state.get("Last2DCode", "")),
            task_id=str(robot_task.get("TaskId") or state.get("TaskID") or ""),
            task_state=str(robot_task.get("State") or ""),
        )

    def print_summary(self) -> None:
        print(
            "[状态] "
            f"位置=({self.x_mm}, {self.y_mm}) mm, "
            f"朝向={self.yaw_mdeg / 1000:.3f}°, 速度={self.speed_mm_s} mm/s, "
            f"状态={self.state}, 工作模式={self.work_mode}, "
            f"ExtraState=({self.extra_first:#010x}, {self.extra_second:#010x}), "
            f"任务=({self.task_id or '无'}, {self.task_state or '无'})"
        )

    def assert_safe_to_start(self) -> None:
        problems: list[str] = []
        if self.state != "Idle":
            problems.append(f"状态不是 Idle，而是 {self.state}")
        if self.speed_mm_s != 0:
            problems.append(f"当前速度不是 0，而是 {self.speed_mm_s} mm/s")
        if self.work_mode != 0:
            problems.append(f"不是任务模式 WorkMode=0，而是 {self.work_mode}")
        if self.extra_first != 0:
            problems.append(f"ExtraStateFirst 非零: {self.extra_first:#010x}")
        if self.extra_second != 0:
            problems.append(f"ExtraStateSecond 非零: {self.extra_second:#010x}")
        if problems:
            raise RuntimeError("底盘不满足安全启动条件：" + "；".join(problems))


class ChassisServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.connection: socket.socket | None = None
        self.framer: JsonFramer | None = None
        self.device = ""
        self.latest_state: ChassisState | None = None
        self.next_heartbeat = math.inf
        self.active_task_id: str | None = None

    @staticmethod
    def message_id(prefix: str) -> str:
        return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def make_action(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "Action": action,
            "Body": body,
            "Device": self.device,
            "ID": self.message_id(action),
            "Time": int(time.time()),
        }

    def send_action(self, action: str, body: dict[str, Any]) -> str:
        assert self.framer is not None
        message = self.make_action(action, body)
        self.framer.send(message)
        print(f"[发送] {action}, ID={message['ID']}")
        return str(message["ID"])

    def receive_one(self) -> dict[str, Any]:
        assert self.framer is not None
        message = self.framer.receive()
        report = str(message.get("Report", ""))
        if report == "DeviceState":
            self.latest_state = ChassisState.from_report(message)
            if self.args.verbose:
                self.latest_state.print_summary()
        elif self.args.verbose:
            print(f"[接收] {report or '未知消息'}: {message}")
        return message

    def heartbeat_if_due(self) -> None:
        if time.monotonic() < self.next_heartbeat:
            return
        self.send_action("HeartBeating", {})
        self.next_heartbeat = time.monotonic() + 6.0

    def wait_for(self, predicate: Any, timeout: float, description: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        assert self.connection is not None
        self.connection.settimeout(1.0)
        while time.monotonic() < deadline:
            self.heartbeat_if_due()
            try:
                message = self.receive_one()
            except socket.timeout:
                continue
            if predicate(message):
                return message
        raise TimeoutError(f"等待{description}超时（{timeout:.0f} 秒）")

    @staticmethod
    def reply_succeeded(message: dict[str, Any]) -> bool:
        data = message.get("data", {})
        return message.get("code") == 1000 and data.get("result") is True

    def wait_reply(self, report: str, message_id: str, timeout: float = 15.0) -> dict[str, Any]:
        def matches(message: dict[str, Any]) -> bool:
            return message.get("Report") == report and str(message.get("id", "")) == message_id

        reply = self.wait_for(matches, timeout, f"{report} 应答")
        if not self.reply_succeeded(reply):
            raise RuntimeError(
                f"{report} 被底盘拒绝: code={reply.get('code')}, "
                f"desc={reply.get('desc')}, data={reply.get('data')}"
            )
        print(f"[应答] {report} 成功")
        return reply

    def accept_and_login(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.args.listen_host, self.args.listen_port))
            listener.listen(1)
            listener.settimeout(self.args.connect_timeout)
            print(
                f"[监听] {self.args.listen_host}:{self.args.listen_port}，"
                f"等待底盘 {self.args.chassis_ip} 主动连接……"
            )
            connection, address = listener.accept()

        if address[0] != self.args.chassis_ip and not self.args.allow_other_ip:
            connection.close()
            raise RuntimeError(
                f"拒绝非目标设备 {address[0]}，预期 {self.args.chassis_ip}；"
                "如确有需要可使用 --allow-other-ip"
            )
        self.connection = connection
        self.connection.settimeout(10.0)
        self.framer = JsonFramer(connection, self.args.protocol)
        print(f"[连接] 来自 {address[0]}:{address[1]}")

        registration = self.wait_for(
            lambda m: m.get("Report") == "RegDevice", 15.0, "RegDevice 注册包"
        )
        device_info = registration.get("Body", {}).get("Device", {})
        self.device = str(device_info.get("SerialNo") or registration.get("Device") or "")
        if not self.device:
            raise ProtocolError("注册包中没有设备序列号")
        print(f"[注册] 序列号={self.device}, 型号={device_info.get('DeviceType', '未知')}")

        login_id = self.send_action(
            "loginDevice",
            {
                "Username": self.args.username,
                "Password": self.args.password,
                "Ip": self.args.advertise_ip,
                "Port": self.args.listen_port,
            },
        )
        self.wait_reply("loginDevice", login_id)
        if self.framer.generation == "v2":
            self.next_heartbeat = time.monotonic()

    def wait_for_state(self, timeout: float = 15.0) -> ChassisState:
        self.wait_for(lambda m: m.get("Report") == "DeviceState", timeout, "DeviceState 状态")
        assert self.latest_state is not None
        self.latest_state.print_summary()
        return self.latest_state

    def monitor_forever(self) -> None:
        """持续保持登录和心跳，仅监听状态，不发送任务。"""
        print("[保持在线] 正在维持平台连接和心跳；按 Ctrl+C 安全退出。")
        last_summary = 0.0
        assert self.connection is not None
        self.connection.settimeout(1.0)
        while True:
            self.heartbeat_if_due()
            try:
                self.receive_one()
            except socket.timeout:
                pass
            now = time.monotonic()
            if self.latest_state is not None and now - last_summary >= 10.0:
                self.latest_state.print_summary()
                last_summary = now

    @staticmethod
    def predicted_position(state: ChassisState, direction: int, distance_mm: int) -> tuple[int, int]:
        yaw = math.radians(state.yaw_mdeg / 1000.0)
        sign = 1 if direction == 0 else -1
        return (
            round(state.x_mm + sign * distance_mm * math.cos(yaw)),
            round(state.y_mm + sign * distance_mm * math.sin(yaw)),
        )

    def run_motion_task(self, direction: int, distance_mm: int, speed_mm_s: int) -> None:
        state = self.wait_for_state()
        state.assert_safe_to_start()
        end_x, end_y = self.predicted_position(state, direction, distance_mm)
        task_id = self.message_id("chassis_test_task")
        self.active_task_id = task_id

        add_id = self.send_action(
            "addTask",
            {
                "Task": [
                    {
                        "ID": task_id,
                        "Type": 10,
                        "Timeout": 60,
                        "StartPoint": state.last_code,
                        "EndPoint": "",
                        "EndPosition": {
                            "Angle": state.yaw_mdeg,
                            "Position": [end_x, end_y],
                        },
                        "NeedACK": 0,
                        "Source": "Server",
                        "Name": "short_run_validation",
                    }
                ]
            },
        )
        self.wait_reply("addTask", add_id)

        # 按协议流程，addTask 成功后等待任务进入 Running，再下发段任务。
        self.wait_for(
            lambda m: self._task_state_in(m, task_id, {"Running"})
            or (self.latest_state is not None and self.latest_state.state == "InTask"),
            15.0,
            "任务进入 Running",
        )

        slice_id = self.message_id("chassis_test_slice")
        slice_message_id = self.send_action(
            "addTaskSlice",
            {
                "TaskID": task_id,
                "Type": "append",
                "TaskSlice": {
                    "AutoPlanDisable": False,
                    "ID": slice_id,
                    "IsLastSlice": True,
                    "IsConditionalAction": False,
                    "Condition": {},
                    "CurrentPoint": state.last_code,
                    "Actions": [
                        {
                            "Action": [
                                {
                                    "Cmd": "Run",
                                    "ID": self.message_id("run"),
                                    "Async": 0,
                                    "NeedACK": 0,
                                    "FailureHandle": 0,
                                    "Timeout": 30,
                                    "Param0": direction,
                                    "Param1": distance_mm,
                                    "Param2": 0,
                                    "Param3": speed_mm_s,
                                    "Param4": "",
                                    "Param5": "",
                                }
                            ],
                            "Type": "Robot",
                        }
                    ],
                    "WayPoints": [],
                    "ShelfAnglePoints": [],
                },
            },
        )
        reply = self.wait_reply("addTaskSlice", slice_message_id)
        params = reply.get("data", {}).get("params", {})
        if params and params.get("AddTaskSliceResult") is False:
            raise RuntimeError(f"底盘拒绝段任务: {params}")

        def motion_finished(message: dict[str, Any]) -> bool:
            if self.latest_state is not None and (
                self.latest_state.extra_first != 0 or self.latest_state.extra_second != 0
            ):
                raise RuntimeError(
                    "运动中检测到异常状态，立即取消任务："
                    f"ExtraState=({self.latest_state.extra_first:#010x}, "
                    f"{self.latest_state.extra_second:#010x})"
                )
            return self._task_state_in(message, task_id, {"Finished"}) or (
                self.latest_state is not None and self.latest_state.state == "Idle"
            )

        self.wait_for(
            motion_finished,
            self.args.motion_timeout,
            "移动任务完成",
        )
        final_state = self.wait_for_state()
        if final_state.speed_mm_s != 0:
            raise RuntimeError("任务结束后底盘速度仍不为 0")
        self.active_task_id = None
        print(
            f"[完成] {'前进' if direction == 0 else '后退'} {distance_mm} mm，"
            f"指令速度 {speed_mm_s} mm/s"
        )

    @staticmethod
    def _task_state_in(message: dict[str, Any], task_id: str, wanted: set[str]) -> bool:
        if message.get("Report") == "TaskState":
            for state in message.get("Body", {}).get("States", []):
                if str(state.get("TaskID")) != task_id:
                    continue
                if state.get("State") == "Failed":
                    raise RuntimeError(f"任务失败: {state}")
                if state.get("State") in wanted:
                    return True
        if message.get("Report") == "DeviceState":
            task = message.get("Body", {}).get("RobotTaskState", {})
            if str(task.get("TaskId")) == task_id:
                if task.get("State") == "Failed":
                    raise RuntimeError(f"任务失败: {task}")
                if task.get("State") in wanted:
                    return True
        return False

    def emergency_cancel(self) -> None:
        if not self.active_task_id or not self.framer:
            return
        try:
            print("[停车] 正在发送 cancelTask，StopType=1（减速停车）……")
            cancel_id = self.send_action(
                "cancelTask", {"TaskID": [self.active_task_id], "StopType": 1}
            )
            self.wait_reply("cancelTask", cancel_id, timeout=8.0)
            self.active_task_id = None
        except Exception as exc:  # 紧急路径只做最大努力，现场急停优先
            print(f"[警告] 停车指令发送失败: {exc}")

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="华睿底盘通信及低速短距离移动验证（默认只读，不运动）"
    )
    parser.add_argument("--listen-host", default="0.0.0.0", help="本机监听地址")
    parser.add_argument("--listen-port", type=int, default=DEFAULT_PORT, help="本机监听端口")
    parser.add_argument("--advertise-ip", default=DEFAULT_COMPUTE_IP, help="登录报文中的算力板 IP")
    parser.add_argument("--chassis-ip", default=DEFAULT_CHASSIS_IP, help="只允许此底盘 IP 连接")
    parser.add_argument("--username", default=os.getenv("HUARUI_USERNAME", "admin"))
    parser.add_argument(
        "--password", default=os.getenv("HUARUI_PASSWORD", "admin123456"), help="也可用 HUARUI_PASSWORD"
    )
    parser.add_argument("--protocol", choices=("auto", "v1", "v2"), default="auto")
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    parser.add_argument("--motion-timeout", type=float, default=45.0)
    parser.add_argument("--allow-other-ip", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--monitor", action="store_true", help="只读并持续保持平台连接和心跳")
    parser.add_argument("--execute", action="store_true", help="明确启用物理移动测试")
    parser.add_argument(
        "--start-direction",
        choices=("forward", "backward"),
        default="forward",
        help="第一段运动方向，默认 forward",
    )
    parser.add_argument("--cancel-active", action="store_true", help="只取消当前悬空任务，不运动")
    parser.add_argument("--distance-mm", type=int, default=100, help="单程距离，最大 300 mm")
    parser.add_argument("--speed-mm-s", type=int, default=50, help="测试速度，最大 150 mm/s")
    parser.add_argument("--no-return", action="store_true", help="只前进，不自动原路后退")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.listen_port <= 65535:
        raise SystemExit("监听端口必须为 1~65535")
    if not 1 <= args.distance_mm <= MAX_TEST_DISTANCE_MM:
        raise SystemExit(f"测试距离必须为 1~{MAX_TEST_DISTANCE_MM} mm")
    if not 1 <= args.speed_mm_s <= MAX_TEST_SPEED_MM_S:
        raise SystemExit(f"测试速度必须为 1~{MAX_TEST_SPEED_MM_S} mm/s")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    server = ChassisServer(args)
    try:
        server.accept_and_login()
        state = server.wait_for_state()
        if args.cancel_active:
            if not state.task_id:
                print("[恢复] 当前没有活动任务，无需取消。")
                return 0
            print(f"[恢复] 取消当前任务 {state.task_id}")
            server.active_task_id = state.task_id
            server.emergency_cancel()
            server.wait_for(
                lambda _m: server.latest_state is not None
                and server.latest_state.state == "Idle"
                and server.latest_state.speed_mm_s == 0,
                15.0,
                "底盘恢复 Idle",
            )
            server.wait_for_state().print_summary()
            return 0
        if not args.execute:
            print("[只读完成] 注册、登录和状态读取正常；未发送任何移动指令。")
            if args.monitor:
                server.monitor_forever()
            print("如需物理测试，请确认清场、急停可用后增加 --execute。")
            return 0

        state.assert_safe_to_start()
        first_direction = 0 if args.start_direction == "forward" else 1
        first_label = "前进" if first_direction == 0 else "后退"
        return_label = "后退" if first_direction == 0 else "前进"
        print(
            "\n即将进行底盘物理运动：\n"
            f"  1. 低速{first_label} {args.distance_mm} mm（{args.speed_mm_s} mm/s）\n"
            + (f"  2. 低速原路{return_label}相同距离\n" if not args.no_return else "  2. 不自动返回\n")
            + "请清空底盘四周，手握物理急停按钮。"
        )
        if input("确认后输入 MOVE（区分大小写）: ").strip() != "MOVE":
            print("[取消] 未发送移动指令。")
            return 0

        server.run_motion_task(
            direction=first_direction,
            distance_mm=args.distance_mm,
            speed_mm_s=args.speed_mm_s,
        )
        if not args.no_return:
            input(f"确认{return_label}路径仍然安全，按 Enter 原路{return_label}；按 Ctrl+C 停止测试：")
            server.run_motion_task(
                direction=1 - first_direction,
                distance_mm=args.distance_mm,
                speed_mm_s=args.speed_mm_s,
            )
        print("[验证通过] 底盘移动控制链路工作正常。")
        return 0
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C。")
        server.emergency_cancel()
        return 130
    except (ConnectionError, OSError, ProtocolError, RuntimeError, TimeoutError) as exc:
        print(f"[失败] {exc}")
        server.emergency_cancel()
        return 1
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())
