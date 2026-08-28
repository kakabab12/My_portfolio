"""Wi-Fi UDP 자이로 장갑을 TurtleBot3 수동 조종 입력에 연결한다.

ESP32는 PDF 원본과 같은 ``ROLL / CTRL_ROLL / PITCH / STATE`` 문자열을 UDP
5005번 포트로 20Hz 전송한다. 수신기는 UDP 5006에 discovery를 보내 ESP32가
현재 TurtleBot 컴퓨터의 IP를 자동으로 찾게 한다.
입력이 ``--stale-sec`` 동안 끊기면 0 Twist만 내보내므로 무선 연결이 끊겨도
TurtleBot3는 안전하게 정지한다.
"""
from __future__ import annotations

import argparse
import socket
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool

from glove_gyro import (
    Orientation,
    average_orientation,
    orientation_to_velocity,
    parse_orientation,
)


class WifiGloveTeleop(Node):
    """UDP 장갑 한 대를 `/cmd_vel_glove`에 연결하는 fail-safe ROS 2 노드."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("wifi_glove_teleop")
        self._stale_sec = args.stale_sec
        self._calibrate_seconds = args.calibrate_seconds
        self._csv_fields = tuple(args.csv_order.split("-"))
        self._mapping = {
            "deadzone_deg": args.deadzone_deg,
            "linear_per_degree": args.linear_per_degree,
            "angular_per_degree": args.angular_per_degree,
            "max_linear": args.max_linear,
            "max_angular": args.max_angular,
            "invert_pitch": args.invert_pitch,
            "invert_roll": args.invert_roll,
        }
        self._publisher = self.create_publisher(Twist, args.output_topic, 10)
        self._mode_publisher = self.create_publisher(Bool, args.mode_topic, 10)
        self._controller_mode_only = args.controller_mode_only
        self._controller_mode_enabled = not self._controller_mode_only
        if self._controller_mode_only:
            self.create_subscription(
                Bool, args.controller_mode_topic, self._on_controller_mode, 10)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((args.bind_host, args.port))
        self._socket.setblocking(False)
        self._discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._discovery_destination = (args.discovery_host, args.discovery_port)
        self._discovery_interval_sec = args.discovery_interval_sec
        self._last_discovery_sec = 0.0
        self._calibration_started_sec: Optional[float] = None
        self._calibration_samples: list[Orientation] = []
        self._neutral: Optional[Orientation] = None
        self._last_orientation: Optional[Orientation] = None
        self._last_data_sec: Optional[float] = None
        self._last_mode: Optional[bool] = None
        self._last_sender: Optional[str] = None
        self._last_parse_warning_sec = 0.0
        self.create_timer(1.0 / args.publish_hz, self._tick)
        self._publish_mode(False)
        self.get_logger().info(
            "Wi-Fi 장갑 대기 — UDP %s:%d, 출력=%s. 연결 뒤 %.1f초 동안 "
            "중립 자세를 유지하세요."
            % (args.bind_host, args.port, args.output_topic, self._calibrate_seconds))

    def _on_controller_mode(self, message: Bool) -> None:
        """전체실행에서는 컨트롤러 모드일 때만 장갑이 제어권을 얻는다."""
        enabled = bool(message.data)
        if enabled == self._controller_mode_enabled:
            return
        self._controller_mode_enabled = enabled
        # 제어권을 끌 때는 새 mux 상태가 전달되기 전에도 즉시 정지를 한 번 보낸다.
        self._publish_stop()
        self._publish_mode(False)
        self.get_logger().info(
            "Wi-Fi 장갑 컨트롤러 모드 -> %s" % ("ON" if enabled else "OFF"))

    def _announce_receiver(self, now: float) -> None:
        if now - self._last_discovery_sec < self._discovery_interval_sec:
            return
        self._last_discovery_sec = now
        try:
            self._discovery_socket.sendto(
                b"IMU_DISCOVER_V1", self._discovery_destination)
        except OSError as error:
            self.get_logger().warning("Wi-Fi 장갑 discovery 전송 실패: %s" % error)

    def _publish_mode(self, enabled: bool, force: bool = False) -> None:
        if force or enabled != self._last_mode:
            message = Bool()
            message.data = enabled
            self._mode_publisher.publish(message)
            self._last_mode = enabled

    def _read_pending_packets(self, now: float) -> None:
        for _ in range(30):
            try:
                raw, sender = self._socket.recvfrom(512)
            except BlockingIOError:
                return
            except OSError as error:
                self.get_logger().error("Wi-Fi 장갑 UDP 수신 오류 — 정지: %s" % error)
                return
            parsed = parse_orientation(
                raw.decode("utf-8", errors="replace"), self._csv_fields)
            if parsed is None:
                if now - self._last_parse_warning_sec >= 2.0:
                    self.get_logger().warning(
                        "Wi-Fi 장갑 데이터를 읽지 못함. JSON pitch/roll 형식이어야 합니다.")
                    self._last_parse_warning_sec = now
                continue
            sender_ip = sender[0]
            if sender_ip != self._last_sender:
                self._last_sender = sender_ip
                self.get_logger().info("Wi-Fi 장갑 수신 시작: %s" % sender_ip)
            self._last_orientation = parsed
            self._last_data_sec = now
            self._update_calibration(parsed, now)

    def _update_calibration(self, orientation: Orientation, now: float) -> None:
        if self._neutral is not None:
            return
        if self._calibration_started_sec is None:
            self._calibration_started_sec = now
            self.get_logger().info("중립 자세 보정 시작 — 장갑을 움직이지 마세요.")
        self._calibration_samples.append(orientation)
        if now - self._calibration_started_sec >= self._calibrate_seconds:
            self._neutral = average_orientation(self._calibration_samples)
            self._calibration_samples = []
            if self._neutral is not None:
                self.get_logger().info(
                    "중립 자세 보정 완료: pitch=%.1f, roll=%.1f. Wi-Fi 장갑 제어 ON."
                    % (self._neutral.pitch, self._neutral.roll))

    def _publish_stop(self) -> None:
        self._publisher.publish(Twist())

    def _tick(self) -> None:
        now = time.monotonic()
        self._announce_receiver(now)
        self._read_pending_packets(now)
        fresh = self._last_data_sec is not None and now - self._last_data_sec <= self._stale_sec
        enabled = (self._neutral is not None and fresh
                   and self._controller_mode_enabled)
        self._publish_mode(enabled)
        if not enabled or self._last_orientation is None or self._neutral is None:
            self._publish_stop()
            return
        if self._last_orientation.enabled is False:
            self._publish_stop()
            return
        velocity = orientation_to_velocity(
            self._last_orientation, self._neutral, **self._mapping)
        message = Twist()
        message.linear.x = velocity.linear_x
        message.angular.z = velocity.angular_z
        self._publisher.publish(message)

    def stop(self) -> None:
        if rclpy.ok():
            self._publish_stop()
            self._publish_mode(False, force=True)
        self._socket.close()
        self._discovery_socket.close()


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("0보다 커야 합니다.")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("0 이상이어야 합니다.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wi-Fi UDP 자이로 장갑으로 TurtleBot3를 조종합니다.")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--discovery-host", default="255.255.255.255")
    parser.add_argument("--discovery-port", type=int, default=5006)
    parser.add_argument("--discovery-interval-sec", type=_positive_float, default=1.0)
    parser.add_argument("--output-topic", default="/cmd_vel_glove")
    parser.add_argument("--mode-topic", default="/glove_mode")
    parser.add_argument(
        "--controller-mode-only", action="store_true",
        help="/joystick_mode가 ON일 때만 장갑 제어권을 활성화합니다.")
    parser.add_argument(
        "--controller-mode-topic", default="/joystick_mode",
        help="--controller-mode-only에서 구독할 컨트롤러 모드 Bool 토픽입니다.")
    parser.add_argument("--publish-hz", type=_positive_float, default=30.0)
    parser.add_argument("--stale-sec", type=_positive_float, default=0.35)
    parser.add_argument("--calibrate-seconds", type=_nonnegative_float, default=2.0)
    parser.add_argument("--csv-order", choices=("pitch-roll", "roll-pitch"), default="pitch-roll")
    parser.add_argument("--deadzone-deg", type=_nonnegative_float, default=10.0)
    parser.add_argument("--linear-per-degree", type=_positive_float, default=0.012)
    parser.add_argument("--angular-per-degree", type=_positive_float, default=0.035)
    parser.add_argument("--max-linear", type=_positive_float, default=0.12)
    parser.add_argument("--max-angular", type=_positive_float, default=0.7)
    parser.add_argument("--invert-pitch", action="store_true")
    parser.add_argument("--invert-roll", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = WifiGloveTeleop(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
