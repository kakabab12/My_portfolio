#!/usr/bin/env python3
"""Safety 전용 속도 mux.

검증된 기존 CmdVelMux를 상속하고 Safety 우선권과 heartbeat fail-stop만
추가한다. 기존 전체 실행은 이 파일을 사용하지 않는다.
"""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

try:
    from ros2_bridge.cmd_vel_mux import (
        CmdVelMux,
        INPUT_TIMEOUT_SEC,
        select_source,
    )
except ImportError:
    from cmd_vel_mux import CmdVelMux, INPUT_TIMEOUT_SEC, select_source


SAFETY_HEARTBEAT_TIMEOUT_SEC = 0.5
AUTO_SAFETY_PHASES = frozenset(("auto_to_b", "auto_to_a"))


def select_source_safety(
        safety_enabled, safety_heartbeat_fresh,
        safety_active, safety_command_fresh,
        gesture_mode, joystick_mode, glove_mode, nav_fresh, gesture_fresh,
        joystick_fresh, glove_fresh, joystick_active=False):
    """Safety 포함 전체 속도 선택 규칙. None은 반드시 0 Twist를 뜻한다."""
    if safety_enabled:
        if not safety_heartbeat_fresh:
            return None
        if safety_active:
            return "safety" if safety_command_fresh else None
    return select_source(
        gesture_mode, joystick_mode, glove_mode,
        nav_fresh, gesture_fresh, joystick_fresh, glove_fresh,
        joystick_active=joystick_active)


class SafetyCmdVelMux(CmdVelMux):
    """Safety 노드가 살아 있고 비활성일 때만 기존 입력을 통과시킨다."""

    def __init__(self):
        super().__init__()
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._safety_active = True
        self._safety_twist = Twist()
        self._safety_sec = None
        self._heartbeat_sec = None
        self._safety_enabled = False
        self.create_subscription(
            Twist, "/cmd_vel_safety", self._on_safety_command, 10)
        self.create_subscription(
            Bool, "/safety/active", self._on_safety_active, latched_qos)
        self.create_subscription(
            Bool, "/safety/heartbeat", self._on_safety_heartbeat, 10)
        self.create_subscription(
            String, "/safety/mission_phase", self._on_mission_phase,
            latched_qos)
        self.get_logger().info(
            "Safety mux 활성 — A->B/D->A 자율구간에서만 Safety를 적용합니다")

    def _on_safety_command(self, message):
        self._safety_twist = message
        self._safety_sec = time.monotonic()

    def _on_safety_active(self, message):
        enabled = bool(message.data)
        if enabled != self._safety_active:
            self._safety_active = enabled
            # Safety 해제 직후 묵은 Nav2 속도가 한 주기 튀지 않게 0을 먼저 보낸다.
            self._stop_cycles = 1
            self.get_logger().warning(
                "Safety 속도 우선권 -> %s" % ("ON" if enabled else "OFF"))

    def _on_safety_heartbeat(self, message):
        if message.data:
            self._heartbeat_sec = time.monotonic()

    def _on_mission_phase(self, message):
        enabled = message.data.strip() in AUTO_SAFETY_PHASES
        if enabled == self._safety_enabled:
            return
        self._safety_enabled = enabled
        # 모드 경계에서 이전 입력이 한 주기 통과하지 않도록 0을 먼저 보낸다.
        self._stop_cycles = 1
        self.get_logger().info(
            "Safety 적용 구간 -> %s" % ("AUTO" if enabled else "MANUAL"))

    def _tick(self):
        if self._stop_cycles > 0:
            self._stop_cycles -= 1
            self._publisher.publish(Twist())
            return

        now = time.monotonic()
        heartbeat_fresh = (
            self._heartbeat_sec is not None
            and now - self._heartbeat_sec <= SAFETY_HEARTBEAT_TIMEOUT_SEC)
        safety_fresh = (
            self._safety_sec is not None
            and now - self._safety_sec <= INPUT_TIMEOUT_SEC)
        nav_fresh = (
            self._nav_sec is not None
            and now - self._nav_sec <= INPUT_TIMEOUT_SEC)
        gesture_fresh = (
            self._gesture_sec is not None
            and now - self._gesture_sec <= INPUT_TIMEOUT_SEC)
        joystick_fresh = (
            self._joystick_sec is not None
            and now - self._joystick_sec <= INPUT_TIMEOUT_SEC)
        glove_fresh = (
            self._glove_sec is not None
            and now - self._glove_sec <= INPUT_TIMEOUT_SEC)

        source = select_source_safety(
            self._safety_enabled,
            heartbeat_fresh,
            self._safety_active,
            safety_fresh,
            self._gesture_mode,
            self._joystick_mode,
            self._glove_mode,
            nav_fresh,
            gesture_fresh,
            joystick_fresh,
            glove_fresh,
            joystick_active=self._is_active(self._joystick_twist),
        )
        if source == "safety":
            output = self._safety_twist
        elif source == "nav":
            output = self._nav_twist
        elif source == "gesture":
            output = self._gesture_twist
        elif source == "joystick":
            output = self._joystick_twist
        elif source == "glove":
            output = self._glove_twist
        else:
            output = Twist()
        self._publisher.publish(output)

        if self._safety_enabled and not heartbeat_fresh:
            self.get_logger().error(
                "자율구간 Safety heartbeat 없음 — 속도 차단",
                throttle_duration_sec=1.0)
        elif (self._safety_enabled
              and self._safety_active and not safety_fresh):
            self.get_logger().warning(
                "Safety 활성 중 속도 명령 만료 — 정지 유지",
                throttle_duration_sec=1.0)
        elif source is not None:
            self.get_logger().info(
                "속도 전달 %s: linear.x=%.3f angular.z=%.3f" % (
                    source, output.linear.x, output.angular.z),
                throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = SafetyCmdVelMux()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node._publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
