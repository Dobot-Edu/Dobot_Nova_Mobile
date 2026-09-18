"""Quest 右手柄到 Nova 从臂的安全控制主循环。"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import traceback
from Dobot_Collection.buttons import ButtonRouter
from Dobot_Collection.session import CollectionSession

import numpy as np

import constants

from Dobot_Init.chassis import ChassisError, VRChassis, chassis_input_enabled
from Dobot_Init.config import Config, load_config
from Dobot_Init.gripper import GripperError, TriggerGripper
from Dobot_Init.indicator_light import IndicatorLight, LightColor, collection_state_color
from quest_vr.kinematics import transform_pose6
from Dobot_Init.lift import LiftColumn, LiftError
from quest_vr.local_ik import LocalIKSolver
from quest_vr.mapping import (
    MappingError,
    RelativePoseMapper,
    head_yaw_deg,
    head_yaw_frame,
    quaternion_rpy_deg,
)
from quest_vr.network import DiscoveryResponder, ReceivedPacket, UDPReceiver
from Dobot_Init.robot import NovaRobot, RobotError
from Dobot_Init.safety import SafetyError, bounded_cartesian_target, bounded_joint_command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quest 3 右手柄遥操 Nova 从臂")
    parser.add_argument(
        "--profile",
        choices=("full", "arm_only", "no_chassis"),
        default=constants.TELEOP_PROFILE,
        help="执行器组合；默认读取 constants.TELEOP_PROFILE",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--listen-only", action="store_true", help="只收 Quest，不连接机械臂")
    mode.add_argument("--dry-run", action="store_true", help="只读机械臂并计算 IK，不使能、不运动")
    mode.add_argument("--execute", action="store_true", help="允许实机 ServoJ；还必须给 --enable-robot")
    parser.add_argument("--enable-robot", action="store_true", help="与 --execute 同时使用")
    parser.add_argument(
        "--dataset-telemetry-port",
        type=int,
        default=constants.DATASET_TELEMETRY_PORT,
        help="向本机 UDP 端口发布数据集动作遥测；设为0可关闭",
    )
    parser.add_argument(
        "--dataset-double-click-window-s",
        type=float,
        default=0.3,
        help="右手 A 键双击回位的最大间隔，默认 0.3 秒",
    )
    args = parser.parse_args(argv)
    if args.execute != args.enable_robot:
        parser.error("实机控制必须同时提供 --execute 和 --enable-robot")
    if not 0 <= args.dataset_telemetry_port <= 65535:
        parser.error("--dataset-telemetry-port 必须在 0..65535")
    if not 0.2 <= args.dataset_double_click_window_s <= 1.5:
        parser.error("--dataset-double-click-window-s 必须在 0.2..1.5")
    return args


class DatasetTelemetryPublisher:
    """只发布数据遥测；A/B 采集命令经独立 Pipe 发送并确认。"""
    def __init__(self, port: int) -> None:
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if port else None
        if self.socket is not None:
            self.socket.setblocking(False)

    def publish(self, payload: dict) -> None:
        if self.socket is not None:
            try:
                self.socket.sendto(json.dumps(payload, separators=(",", ":")).encode(), ("127.0.0.1", self.port))
            except OSError:
                pass

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()


class VRTransport:
    def __init__(self, config: Config) -> None:
        network = config.network
        self.receiver = UDPReceiver(
            network.listen_host,
            network.data_port,
            network.allowed_quest_ip,
            network.allowed_quest_subnet,
        )
        self.discovery = DiscoveryResponder(
            network.listen_host, network.discovery_port, network.data_port
        )
        self.discovery.start()

    def close(self) -> None:
        self.receiver.close()
        self.discovery.close()


def listen_only(config: Config) -> None:
    transport = VRTransport(config)
    network = config.network
    print(f"只监听 Quest：UDP {network.data_port}，发现端口 {network.discovery_port}")
    print("此模式不连接、不使能机械臂。按 Ctrl+C 退出。")
    last_log = 0.0
    last_packet_at: float | None = None
    hand_position0: np.ndarray | None = None
    operator_frame0: np.ndarray | None = None
    try:
        while True:
            received = transport.receiver.poll()
            now = time.monotonic()
            if received is not None:
                last_packet_at = received.received_at
                if now - last_log >= 0.2:
                    packet = received.packet
                    head_yaw = (
                        f"{head_yaw_deg(packet.head_quaternion_xyzw):.1f}°"
                        if packet.head_tracking_valid
                        and packet.head_quaternion_xyzw is not None
                        else "缺失"
                    )
                    deadman = packet.deadman(network.deadman_grip_threshold)
                    operator_delta = "未按Grip"
                    if (
                        deadman
                        and packet.head_tracking_valid
                        and packet.head_quaternion_xyzw is not None
                    ):
                        if hand_position0 is None or operator_frame0 is None:
                            hand_position0 = packet.position_m.copy()
                            operator_frame0 = head_yaw_frame(packet.head_quaternion_xyzw)
                        delta_mm = 1000.0 * operator_frame0.T @ (
                            packet.position_m - hand_position0
                        )
                        operator_delta = np.round(delta_mm, 1).tolist()
                    else:
                        hand_position0 = None
                        operator_frame0 = None
                    print(
                        f"Quest={received.source[0]} 协议V{packet.protocol_version} "
                        f"XYZ(m)={np.round(packet.position_m, 4).tolist()} "
                        f"RPY(deg)={np.round(quaternion_rpy_deg(packet.quaternion_xyzw), 1).tolist()} "
                        f"HeadYaw={head_yaw} "
                        f"Grip={packet.grip:.2f} Trigger={packet.trigger:.2f} "
                        f"deadman={deadman} "
                        f"操作者Δ[右,上,前](mm)={operator_delta} "
                        f"A={packet.button_a} B={packet.button_b} "
                        f"Stick={np.round(packet.thumbstick_xy, 2).tolist()} "
                        f"tracking={packet.tracking_valid}"
                    )
                    last_log = now
            elif last_packet_at is not None and now - last_packet_at > 1.0:
                print("[HOLD] 超过 1 秒未收到 Quest 合法数据")
                last_packet_at = None
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n监听已停止")
    finally:
        transport.close()


def control(
    config: Config,
    execute: bool,
    dataset_telemetry_port: int = 0,
    dataset_double_click_window_s: float = 0.3,
    profile: str = "full",
) -> None:
    if execute and profile in ("full", "arm_only") and dataset_telemetry_port != constants.DATASET_TELEMETRY_PORT:
        raise ValueError("统一数采入口必须使用 constants.DATASET_TELEMETRY_PORT")
    transport = VRTransport(config)
    dataset_telemetry = DatasetTelemetryPublisher(dataset_telemetry_port)
    buttons = ButtonRouter(profile == "no_chassis", dataset_double_click_window_s)
    activated = False
    require_neutral = True
    robot: NovaRobot | None = None
    indicator: IndicatorLight | None = None
    indicator_enabled = False
    indicator_available = True
    gripper: TriggerGripper | None = None
    lift: LiftColumn | None = None
    chassis: VRChassis | None = None
    failed = True
    arm_fault = None
    maintenance_preview = None
    maintenance_cameras = {}

    def set_indicator(color: LightColor) -> None:
        nonlocal indicator_available
        if indicator is None or not indicator_enabled or not indicator_available:
            return
        try:
            indicator.set(color)
        except (OSError, RobotError) as error:
            indicator_available = False
            print(f"[信号灯警告] {error}；本轮不再控制信号灯")

    def on_collection_state(state: str, _detail: str) -> None:
        set_indicator(LightColor.RED if arm_fault else collection_state_color(state))

    def pause_arm(error):
        nonlocal arm_fault, active, last_command
        arm_fault = str(error)
        active = False
        mapper.release()
        print(f"[机械臂暂停] {error}；底盘遥操继续。排除碰撞后双击A尝试恢复回位。")
        try:
            last_command = robot.joints().copy()
            print(f"[机械臂诊断] {robot.request('GetErrorID()').strip()}")
        except RobotError as diagnostic_error:
            print(f"[机械臂诊断失败] {diagnostic_error}")
        set_indicator(LightColor.RED)

    collection = CollectionSession(state_callback=on_collection_state)
    try:
        print("正在只读连接 Nova；单击 A 前不会清错、使能或运动。")
        robot = NovaRobot(config.robot)
        indicator = IndicatorLight(robot) if execute else None
        snapshot = robot.snapshot()
        print("RobotMode：", snapshot.mode)
        print("GetErrorID：", snapshot.errors_reply.strip())
        print("起始 TCP [mm, deg]：", np.round(snapshot.pose, 3).tolist())
        print("起始关节角 [deg]：", np.round(np.rad2deg(snapshot.joints_rad), 3).tolist())

        solver = LocalIKSolver.from_reference(snapshot.pose, snapshot.joints_rad)
        mapper = RelativePoseMapper(config.mapping)
        period = 1.0 / config.robot.rate_hz
        max_step = np.deg2rad(config.robot.max_joint_speed_deg_s) * period
        max_tcp_step = config.robot.max_tcp_speed_mm_s * period
        max_tcp_rotation_step = config.robot.max_tcp_rotation_deg_s * period
        last_command = snapshot.joints_rad.copy()
        estimated_pose = snapshot.pose.copy()
        gripper = TriggerGripper(config.gripper, execute=execute)
        lift = LiftColumn(config.lift, execute=execute)
        chassis = VRChassis(config.chassis, execute=execute, read_only=profile == "arm_only")
        chassis.connect()

        print("[待机] 确认工作区清空、急停可用后，单击 A 以当前位置开放遥操。")
        print("[按键] 初始化完成后先松开 Grip、让摇杆回中，并松开 A/B。")
        if profile == "no_chassis":
            print("[维护] 初始化后长按 A 上升、长按 B 下降；AB 同按停止，不采集。")
        else:
            print("[按键] B 开始/结束采集；双击 A 回标准位，录制继续。")

        def initialize():
            nonlocal snapshot, solver, last_command, estimated_pose
            nonlocal indicator_enabled
            nonlocal maintenance_preview
            if execute:
                if config.lift.enabled:
                    lift.connect(robot.digital_input)
                    lift.prepare_for_control()
                if config.gripper.enabled:
                    gripper.connect()
                if config.chassis.enabled:
                    chassis.enable_remote()
                robot.prepare_for_control()
                # 单击A只接管当前位置，不自动回标准位；双击A才执行并记录回位轨迹。
                snapshot = robot.snapshot()
                if snapshot.mode != 5:
                    raise RobotError(
                        f"机械臂自检后 RobotMode={snapshot.mode}，尚未空闲，"
                        "拒绝以当前位置开放遥操"
                    )
                # DO() 是队列指令，会短暂令 RobotMode 进入7；先完成上面的
                # Mode=5检查，再开启黄色待机灯。
                indicator_enabled = True
                set_indicator(LightColor.YELLOW)
            else:
                gripper.connect()
                lift.connect()
            solver = LocalIKSolver.from_reference(snapshot.pose, snapshot.joints_rad)
            last_command = snapshot.joints_rad.copy()
            estimated_pose = snapshot.pose.copy()
            mapper.release()
            if execute and profile in ("full", "arm_only"):
                collection.start()
            elif execute and profile == "no_chassis":
                from Dobot_Init.camera_preview import CameraPreview
                from Dobot_Collection.devices import TopHttpCamera, WristRealSense
                maintenance_cameras.update(
                    top=TopHttpCamera(constants.TOP_CAMERA_URL, constants.CAMERA_TIMEOUT_S),
                    wrist=WristRealSense(constants.WRIST_CAMERA_SERIAL, constants.CAMERA_WIDTH,
                                         constants.CAMERA_HEIGHT, constants.CAMERA_STREAM_FPS),
                )
                for source in maintenance_cameras.values():
                    source.start()
                maintenance_preview = CameraPreview(maintenance_cameras, 'Nova 维护遥操')
                maintenance_preview.start()
            print(
                "[遥操已激活] 已以机械臂当前位置为相对零点；"
                "松开 Grip、摇杆回中、松开 A/B 后可操作；双击A才回标准位。"
            )

        latest: ReceivedPacket | None = None
        active = False
        last_log = 0.0

        def publish_dataset(
            joints_command_rad: np.ndarray,
            *,
            arm_active: bool,
            tracking_valid: bool,
            current_packet=None,
        ) -> None:
            chassis_dataset = chassis.dataset_snapshot()
            gripper_dataset = gripper.feedback_snapshot()
            gripper_position = float(gripper_dataset["position"])
            gripper_target_position = float(gripper.target_position)
            dataset_telemetry.publish(
                {
                    "timestamp_ns": time.time_ns(),
                    "joints_command_deg": np.rad2deg(joints_command_rad).tolist(),
                    "gripper_position": gripper_position,
                    "gripper_target_position": gripper_target_position,
                    "gripper_position_raw": gripper_dataset["position_raw"],
                    "gripper_current_ma": gripper_dataset["current_ma"],
                    "gripper_load_raw": gripper_dataset["load_raw"],
                    "gripper_temperature_c": gripper_dataset["temperature_c"],
                    "gripper_moving": gripper_dataset["moving"],
                    "gripper_status_error": gripper_dataset["status_error"],
                    "gripper_current_baseline_ma": gripper_dataset["current_baseline_ma"],
                    "gripper_current_threshold_ma": gripper_dataset["current_threshold_ma"],
                    "gripper_feedback_monotonic_ns": gripper_dataset["feedback_monotonic_ns"],
                    "gripper_feedback_valid": gripper_dataset["feedback_valid"],
                    "lift_target_cm": float(lift.height_cm or 0.0),
                    "arm_active": arm_active,
                    "tracking_valid": tracking_valid,
                    "teleop_activated": activated,
                    "teleop_profile": profile,
                    "arm_fault": arm_fault,
                    **chassis_dataset,
                }
            )

        while True:
            loop_start = time.monotonic()
            received = transport.receiver.poll()
            if received is not None:
                latest = received
            fresh = (
                latest is not None
                and loop_start - latest.received_at <= config.network.packet_timeout_s
            )
            packet = latest.packet if fresh and latest is not None else None
            packet_valid = packet is not None and packet.tracking_valid
            deadman = packet is not None and packet.deadman(config.network.deadman_grip_threshold)
            button_a = packet.button_a if packet is not None else False
            button_b = packet.button_b if packet is not None else False
            collection.poll()
            # 阻塞初始化/回位后先丢掉积压包，并要求新的中立输入。
            if require_neutral and activated:
                neutral = (packet_valid and not deadman and not button_a and not button_b
                           and np.all(np.abs(packet.thumbstick_xy) <= config.chassis.deadzone))
                if neutral:
                    require_neutral = False
                    buttons.reset()
                    print("[就绪] 手柄已回中，遥操已开放")
            events = buttons.update(button_a, button_b,
                                    valid=packet_valid and not (activated and require_neutral),
                                    activated=activated, now=loop_start,
                                    record_allowed=not activated or not execute or collection.state in ("ready", "recording"))
            if events.activate:
                print("[初始化] 单击 A 已确认，开始自检、夹爪张开并接管机械臂当前位置")
                initialize()
                activated = True
                require_neutral = True
                buttons.reset()
                transport.receiver.poll()  # 丢弃初始化期间积压的手柄包
                latest = None
                publish_dataset(last_command, arm_active=False, tracking_valid=False)
                continue
            if events.record:
                if not activated:
                    print("[数采] 请先单击 A 激活遥操")
                elif execute:
                    collection.toggle()
                else:
                    print("[DRY RUN] B 已识别，不创建数据")

            controls_valid = activated and not require_neutral and packet_valid
            reset_requested = events.home and controls_valid
            if activated:
                gripper.update(packet.trigger if packet else 0.0,
                               input_valid=controls_valid and not reset_requested)
                # LiftColumn 仍使用内部 A+B=上升 的低层接口；手柄 AB 不直接透传。
                lift.update_buttons(events.lift_up, events.lift_up or events.lift_down,
                                    input_valid=controls_valid and profile == "no_chassis")
                lift.check_error()
            chassis.update(packet.thumbstick_xy if packet else np.zeros(2),
                           input_valid=controls_valid and config.chassis.enabled
                           and not reset_requested and chassis_input_enabled(
                               packet_valid=packet_valid, grip_deadman=deadman,
                               require_grip=config.chassis.require_grip))
            chassis.check_error()

            if reset_requested:
                lift.update_buttons(False, False, input_valid=False)
                chassis.update(np.zeros(2), input_valid=False)
                mapper.release()
                active = False
                print("[双击A回位] 停止底盘，机械臂低速回标准位；正在录制的数据继续保存")
                if execute:
                    try:
                        if arm_fault:
                            robot.prepare_for_control()
                        snapshot = robot.move_to_startup_pose(
                            progress_callback=lambda joints_rad: publish_dataset(
                                joints_rad, arm_active=True, tracking_valid=packet_valid,
                                current_packet=packet))
                        solver = LocalIKSolver.from_reference(snapshot.pose, snapshot.joints_rad)
                        last_command = snapshot.joints_rad.copy()
                        estimated_pose = snapshot.pose.copy()
                        arm_fault = None
                        indicator_available = True
                        set_indicator(collection_state_color(collection.state))
                    except RobotError as error:
                        pause_arm(error)
                require_neutral = True
                buttons.reset()
                transport.receiver.poll()
                latest = None
                packet = None
                packet_valid = False
                deadman = False
                controls_valid = False
                print("[回位失败] 机械臂保持暂停，手柄回中后底盘可用" if arm_fault else
                      "[回位完成] 松开 Grip、摇杆回中并松开 A/B 后继续；B 可结束录制")

            valid = deadman and controls_valid and arm_fault is None
            if not valid:
                if active:
                    print("[HOLD] Grip 松开、跟踪失效或数据超时")
                mapper.release()
                active = False
            else:
                try:
                    target_pose = mapper.map(packet, estimated_pose)
                    ik_target_pose = bounded_cartesian_target(
                        target_pose,
                        estimated_pose,
                        max_tcp_step,
                        max_tcp_rotation_step,
                    )
                    # 不在应用层限制关节角范围；控制柜负责机械本体的绝对行程保护。
                    # 本循环仍保留下面的单周期关节限速，防止 IK 解或 VR 输入突跳。
                    result = solver.solve(ik_target_pose, last_command)
                    if not result.success:
                        raise SafetyError(
                            f"本地 IK {result.reason}：位置误差 {result.position_error_mm:.2f} mm，"
                            f"姿态误差 {result.orientation_error_deg:.2f}°"
                        )
                    command = bounded_joint_command(
                        result.joints,
                        last_command,
                        max_step,
                    )
                    if execute:
                        robot.servo_j(command)
                    last_command = command
                    estimated_pose = transform_pose6(solver.forward(command))
                    if not active:
                        print("[ACTIVE] 已把当前右手与当前 Nova TCP 设为相对零点")
                    active = True
                    if loop_start - last_log >= 0.5:
                        mode = "EXEC" if execute else "DRY"
                        hand_pose = [
                            *np.round(packet.position_m, 4).tolist(),
                            *np.round(
                                quaternion_rpy_deg(packet.quaternion_xyzw), 1
                            ).tolist(),
                        ]
                        head_yaw = (
                            f"{head_yaw_deg(packet.head_quaternion_xyzw):.1f}°"
                            if packet.head_tracking_valid
                            and packet.head_quaternion_xyzw is not None
                            else "缺失"
                        )
                        print(
                            f"[{mode}] 协议V{packet.protocol_version} HeadYaw={head_yaw} "
                            f"手柄六维[m,deg]={hand_pose} | "
                            f"Nova末端六维[mm,deg]="
                            f"{np.round(estimated_pose, 2).tolist()} | "
                            f"底盘[v(mm/s),w(deg/s)]="
                            f"[{chassis.desired.linear_mm_s}, "
                            f"{chassis.desired.angular_mdeg_s / 1000:.2f}]"
                        )
                        last_log = loop_start
                except MappingError as error:
                    print(f"[HOLD] {error}")
                    mapper.release()
                    active = False
                except RobotError as error:
                    pause_arm(error)
                except SafetyError as error:
                    # 短暂 IK 失败时保留原六维零点，不吞掉手柄已转角度。
                    # 手柄回到可解区域后，同一次 Grip 会继续跟随。
                    print(f"[HOLD] {error}")
                    active = False

            publish_dataset(
                last_command,
                arm_active=active,
                tracking_valid=packet_valid,
                current_packet=packet,
            )

            remaining = period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        failed = False
        print("\nVR 遥操已停止；若机械臂已使能，它会保持最后目标。")
    finally:
        transport.close()
        dataset_telemetry.close()
        # 先停底盘，再关闭其他外设和机械臂连接。
        if chassis is not None:
            chassis.close()
        collection.close()
        if maintenance_preview is not None:
            maintenance_preview.close()
        for source in maintenance_cameras.values():
            source.close()
        if gripper is not None:
            gripper.close()
        if lift is not None:
            lift.close()
        if robot is not None:
            if indicator is not None and indicator_enabled and indicator_available:
                try:
                    indicator.set(LightColor.RED) if failed else indicator.off()
                except (OSError, RobotError) as error:
                    print(f"[信号灯警告] 退出时无法更新信号灯：{error}")
            robot.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.profile)
        if args.listen_only:
            listen_only(config)
        else:
            control(
                config,
                execute=args.execute,
                profile=args.profile,
                dataset_telemetry_port=args.dataset_telemetry_port,
                dataset_double_click_window_s=args.dataset_double_click_window_s,
            )
        return 0
    except (
        ValueError,
        FileNotFoundError,
        OSError,
        RobotError,
        GripperError,
        LiftError,
        ChassisError,
    ) as error:
        # 保留出错位置；入口程序会将 stderr 同步写入轮转日志。
        traceback.print_exc(file=sys.stderr)
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
