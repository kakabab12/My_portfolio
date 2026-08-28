"""직렬/Bluetooth 자이로 장갑을 TurtleBot3 수동 조종 입력으로 연결한다.

예시(장갑 포트가 /dev/ttyUSB1이고 펌웨어가 115200 baud일 때)::

    /usr/bin/python3 ros2_bridge/glove_gyro_teleop.py \
        --port /dev/ttyUSB1 --baud 115200

장갑은 ``{\"pitch\": 0.0, \"roll\": 0.0}``처럼 한 줄마다 pitch/roll을
전송해야 한다. 시작 후 ``--calibrate-seconds`` 동안 손을 편안한 중립 자세로
유지하면 그 자세를 0도로 보정한다. 입력이 끊기거나 파싱할 수 없으면 이 노드는
계속 0 Twist를 보내므로 로봇은 정지한다.
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    import serial
except ImportError:  # 실행 시 이해하기 쉬운 오류를 내기 위해 지연 처리한다.
    serial = None

from glove_gyro import (
    Orientation,
    average_orientation,
    orientation_to_velocity,
    parse_orientation,
)


class GloveGyroTeleop(Node):
    """한 개의 장갑만 `/cmd_vel_glove`에 연결하는 fail-safe ROS2 노드."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("glove_gyro_teleop")
        self._port = args.port
        self._baud = args.baud
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
        self._serial: Optional[serial.Serial] = None
        self._next_connect_sec = 0.0
        self._calibration_started_sec: Optional[float] = None
        self._calibration_samples: list[Orientation] = []
        self._neutral: Optional[Orientation] = None
        self._last_orientation: Optional[Orientation] = None
        self._last_data_sec: Optional[float] = None
        self._last_mode: Optional[bool] = None
        self._last_parse_warning_sec = 0.0
        self._rx_buffer = bytearray()
        self.create_timer(1.0 / args.publish_hz, self._tick)
        self._publish_mode(False)
        self.get_logger().info(
            "자이로 장갑 대기 — port=%s, baud=%s, 출력=%s. "
            "연결 뒤 %.1f초 동안 중립 자세를 유지하세요."
            % (self._port, self._baud, args.output_topic, self._calibrate_seconds))

    def _publish_mode(self, enabled: bool, force: bool = False) -> None:
        if force or enabled != self._last_mode:
            message = Bool()
            message.data = enabled
            self._mode_publisher.publish(message)
            self._last_mode = enabled

    def _close_serial(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def _connect_if_needed(self, now: float) -> None:
        if self._serial is not None or now < self._next_connect_sec:
            return
        try:
            self._serial = serial.Serial(self._port, self._baud, timeout=0)
            self._serial.reset_input_buffer()
            self._calibration_started_sec = None
            self._calibration_samples = []
            self._neutral = None
            self._rx_buffer.clear()
            self.get_logger().info("장갑 직렬 연결됨: %s (%d baud)" % (self._port, self._baud))
        except Exception as error:
            self._next_connect_sec = now + 2.0
            self.get_logger().warning(
                "장갑 직렬 연결 실패(%s). 2초 후 재시도 — %s" % (self._port, error),
                throttle_duration_sec=2.0)

    def _read_pending_lines(self, now: float) -> None:
        if self._serial is None:
            return
        try:
            # timeout=0에서 readline()은 개행 전의 조각을 반환할 수 있다. 따라서
            # 버퍼에 누적해 완전한 한 줄만 파싱한다.
            waiting = self._serial.in_waiting
            if waiting:
                self._rx_buffer.extend(self._serial.read(min(waiting, 4096)))
            if len(self._rx_buffer) > 8192:
                self._rx_buffer.clear()
                self.get_logger().warning("장갑 수신 버퍼가 너무 길어 비웠습니다.")
                return
            # 1회 타이머에서 너무 많은 누적 데이터를 처리하지 않게 제한한다.
            for _ in range(30):
                try:
                    line_end = self._rx_buffer.index(b"\n")
                except ValueError:
                    break
                raw = bytes(self._rx_buffer[:line_end])
                del self._rx_buffer[:line_end + 1]
                self._on_serial_line(raw, now)
        except Exception as error:
            self.get_logger().warning("장갑 직렬 연결 끊김 — 정지: %s" % error)
            self._close_serial()
            self._next_connect_sec = now + 2.0
            self._neutral = None
            self._calibration_samples = []
            self._calibration_started_sec = None
            self._rx_buffer.clear()

    def _on_serial_line(self, raw: bytes, now: float) -> None:
        parsed = parse_orientation(
            raw.decode("utf-8", errors="replace"), self._csv_fields)
        if parsed is None:
            if now - self._last_parse_warning_sec >= 2.0:
                self.get_logger().warning(
                    "장갑 데이터 형식을 읽지 못함. JSON {\"pitch\":..,\"roll\":..} "
                    "또는 pitch,roll CSV여야 합니다.")
                self._last_parse_warning_sec = now
            return
        self._last_orientation = parsed
        self._last_data_sec = now
        self._update_calibration(parsed, now)

    def _update_calibration(self, orientation: Orientation, now: float) -> None:
        if self._neutral is not None:
            return
        if self._calibration_started_sec is None:
            self._calibration_started_sec = now
            self.get_logger().info("중립 자세 보정 시작 — 손을 움직이지 마세요.")
        self._calibration_samples.append(orientation)
        if now - self._calibration_started_sec >= self._calibrate_seconds:
            self._neutral = average_orientation(self._calibration_samples)
            self._calibration_samples = []
            if self._neutral is not None:
                self.get_logger().info(
                    "중립 자세 보정 완료: pitch=%.1f, roll=%.1f. 장갑 제어 ON."
                    % (self._neutral.pitch, self._neutral.roll))

    def _publish_stop(self) -> None:
        self._publisher.publish(Twist())

    def _tick(self) -> None:
        now = time.monotonic()
        self._connect_if_needed(now)
        self._read_pending_lines(now)

        fresh = self._last_data_sec is not None and now - self._last_data_sec <= self._stale_sec
        enabled = self._neutral is not None and fresh
        self._publish_mode(enabled)
        if not enabled or self._last_orientation is None or self._neutral is None:
            self._publish_stop()
            return
        # 장갑 펌웨어가 deadman/button 값을 함께 보낼 경우 false일 때도 정지한다.
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
        # Ctrl+C 때 ROS의 기본 signal handler가 컨텍스트를 먼저 종료할 수 있다.
        # 그 뒤 publish()하면 RCLError가 나므로, 아직 유효할 때만 정지 상태를
        # 알리고 직렬 포트는 어느 경우에도 닫는다.
        if rclpy.ok():
            self._publish_stop()
            self._publish_mode(False, force=True)
        self._close_serial()


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
    parser = argparse.ArgumentParser(
        description="직렬/Bluetooth 자이로 장갑으로 TurtleBot3를 조종합니다.")
    parser.add_argument("--port", required=True,
                        help="장갑 직렬 포트. OpenCR 포트(/dev/ttyACM0)는 절대 넣지 마세요.")
    parser.add_argument("--baud", type=int, default=115200,
                        help="장갑 펌웨어의 Serial.begin 값 (기본: 115200)")
    parser.add_argument("--output-topic", default="/cmd_vel_glove")
    parser.add_argument("--mode-topic", default="/glove_mode")
    parser.add_argument("--publish-hz", type=_positive_float, default=30.0)
    parser.add_argument("--stale-sec", type=_positive_float, default=0.35,
                        help="이 시간 동안 센서 데이터가 없으면 정지 (기본: 0.35초)")
    parser.add_argument("--calibrate-seconds", type=_nonnegative_float, default=2.0,
                        help="시작 중립 자세 평균 시간 (기본: 2초)")
    parser.add_argument("--csv-order", choices=("pitch-roll", "roll-pitch"),
                        default="pitch-roll", help="JSON이 아닌 CSV 수신 순서")
    parser.add_argument("--deadzone-deg", type=_nonnegative_float, default=10.0)
    parser.add_argument("--linear-per-degree", type=_positive_float, default=0.012)
    parser.add_argument("--angular-per-degree", type=_positive_float, default=0.035)
    parser.add_argument("--max-linear", type=_positive_float, default=0.12,
                        help="최대 전후진 속도 m/s (버거 한계 0.22보다 낮은 안전 기본값)")
    parser.add_argument("--max-angular", type=_positive_float, default=0.7,
                        help="최대 회전 속도 rad/s (버거 한계 2.84보다 낮은 안전 기본값)")
    parser.add_argument("--invert-pitch", action="store_true")
    parser.add_argument("--invert-roll", action="store_true")
    return parser.parse_args()


def main() -> None:
    if serial is None:
        raise SystemExit(
            "pyserial이 없습니다. /usr/bin/python3 -m pip install pyserial 을 먼저 실행하세요.")
    args = parse_args()
    rclpy.init()
    node = GloveGyroTeleop(args)
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
