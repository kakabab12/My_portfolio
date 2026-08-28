#!/usr/bin/env python3
"""시연용 Safety mux.

기존 Safety mux를 그대로 상속하되, 미션이 navigating/resumed 상태일 때만
Nav2 속도를 통과시킨다. 미션 오류나 waypoint 전환 중에는 0 속도를 유지한다.
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String

try:
    from ros2_bridge.cmd_vel_mux_safety import SafetyCmdVelMux
except ImportError:
    from cmd_vel_mux_safety import SafetyCmdVelMux


NAV_ALLOWED_STATUSES = frozenset(("navigating", "resumed"))


class DemoSafetyCmdVelMux(SafetyCmdVelMux):
    """Safety 우선권에 미션 상태 기반 Nav2 fail-closed 조건을 추가한다."""

    def __init__(self):
        self._demo_nav_allowed = False
        super().__init__()
        self.create_subscription(
            String, "/safety/nav_status", self._on_demo_nav_status, 10)
        self.get_logger().info(
            "시연용 mux 활성 — 정상 주행 상태에서만 Nav2 속도를 전달합니다")

    def _on_demo_nav_status(self, message):
        allowed = message.data.strip() in NAV_ALLOWED_STATUSES
        if allowed == self._demo_nav_allowed:
            return
        self._demo_nav_allowed = allowed
        self._nav_sec = None
        self._stop_cycles = 1
        self.get_logger().info(
            "시연용 Nav2 속도 허용 -> %s" % ("ON" if allowed else "OFF"))

    def _tick(self):
        if not self._demo_nav_allowed:
            # 부모 mux의 Safety heartbeat/우선권 처리는 유지하면서 Nav2 입력만
            # 만료시켜, 초기화·도착 검증·오류 상태에서 반드시 0이 나가게 한다.
            self._nav_sec = None
        super()._tick()


def main():
    rclpy.init()
    node = DemoSafetyCmdVelMux()
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
