"""Nav2·제스처·조이스틱·자이로 장갑 중 하나만 터틀봇 속도로 전달한다."""
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool

PUBLISH_HZ = 50.0
INPUT_TIMEOUT_SEC = 0.3
INPUT_ACTIVE_EPSILON = 0.001


def select_source(gesture_mode, joystick_mode, glove_mode, nav_fresh,
                  gesture_fresh, joystick_fresh, glove_fresh,
                  joystick_active=False, glove_requires_controller=False):
    """테스트 가능한 순수 선택 규칙. None은 안전 정지를 뜻한다."""
    if joystick_mode:
        # 컨트롤러 모드에서는 사용자가 스틱을 실제로 움직이는 동안만 조이스틱이
        # 우선이다. 스틱이 중립이면 Wi-Fi 장갑을 바로 쓸 수 있어 두 입력 장치를
        # 같은 수동 제어 모드에서 함께 쓸 수 있다.
        if joystick_active:
            return "joystick" if joystick_fresh else None
        if glove_mode:
            return "glove" if glove_fresh else None
        return "joystick" if joystick_fresh else None
    # 전체실행의 장갑은 /joystick_mode에 묶여 있다. OK 사인 1.5초로
    # 컨트롤러 모드를 끈 직후 장갑이 기울어져 있거나 이전 glove_mode=True
    # 메시지가 남아 있어도 AUTO/Nav2 전환을 가로막지 못하게 한다.
    if glove_mode and not glove_requires_controller:
        return "glove" if glove_fresh else None
    if gesture_mode:
        return "gesture" if gesture_fresh else None
    return "nav" if nav_fresh else None


class CmdVelMux(Node):
    def __init__(self):
        super().__init__("gesture_nav_cmd_vel_mux")
        self._publisher = self.create_publisher(Twist, "/cmd_vel_muxed", 10)
        # navigation_with_mux.launch.py에서 Nav2 전체를 Twist 형식으로 고정한다.
        # ROS 2는 동일 토픽에 다른 타입의 구독 둘을 허용하지 않으므로 이 노드는
        # 오직 TurtleBot3와 같은 geometry_msgs/Twist만 구독한다.
        self.create_subscription(Twist, "/cmd_vel", self._on_nav, 10)
        self.create_subscription(Twist, "/cmd_vel_gesture", self._on_gesture, 10)
        self.create_subscription(Twist, "/cmd_vel_joy", self._on_joystick, 10)
        self.create_subscription(Twist, "/cmd_vel_glove", self._on_glove, 10)
        self.create_subscription(Bool, "/gesture_mode", self._on_mode, 10)
        self.create_subscription(Bool, "/joystick_mode", self._on_joystick_mode, 10)
        self.create_subscription(Bool, "/glove_mode", self._on_glove_mode, 10)
        self._gesture_mode = False
        self._joystick_mode = False
        self._glove_mode = False
        # /joystick_mode를 한 번이라도 받으면 전체실행 구성으로 간주한다.
        # 장갑 단독 실행기에는 이 토픽 publisher가 없으므로 기존처럼 장갑만
        # 사용할 수 있다.
        self._controller_mode_seen = False
        self._nav_twist = Twist()
        self._gesture_twist = Twist()
        self._joystick_twist = Twist()
        self._glove_twist = Twist()
        self._nav_sec = None
        self._gesture_sec = None
        self._joystick_sec = None
        self._glove_sec = None
        self._stop_cycles = 1
        self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self.get_logger().info(
            "속도 mux 시작 — AUTO(/cmd_vel), 제스처(/cmd_vel_gesture),"
            " 조이스틱(/cmd_vel_joy), 장갑(/cmd_vel_glove)"
            " -> 터틀봇 전용 /cmd_vel_muxed")

    def _on_nav(self, message):
        self._nav_twist = message
        self._nav_sec = time.monotonic()

    def _on_gesture(self, message):
        self._gesture_twist = message
        self._gesture_sec = time.monotonic()

    def _on_joystick(self, message):
        self._joystick_twist = message
        self._joystick_sec = time.monotonic()

    def _on_glove(self, message):
        self._glove_twist = message
        self._glove_sec = time.monotonic()

    def _on_mode(self, message):
        enabled = bool(message.data)
        if enabled != self._gesture_mode:
            self._gesture_mode = enabled
            self._stop_cycles = 1
            self.get_logger().info(
                "제어권 전환 -> %s" % ("GESTURE" if enabled else "AUTO/Nav2"))

    def _on_joystick_mode(self, message):
        enabled = bool(message.data)
        self._controller_mode_seen = True
        if enabled != self._joystick_mode:
            self._joystick_mode = enabled
            self._stop_cycles = 1
            self.get_logger().info(
                "조이스틱 제어권 -> %s" % ("ON" if enabled else "OFF"))

    def _on_glove_mode(self, message):
        enabled = bool(message.data)
        if enabled != self._glove_mode:
            self._glove_mode = enabled
            self._stop_cycles = 1
            self.get_logger().info(
                "자이로 장갑 제어권 -> %s" % ("ON" if enabled else "OFF"))

    @staticmethod
    def _is_active(command):
        """중립(0 Twist)과 사용자가 조작 중인 수동 입력을 구분한다."""
        return any(abs(value) > INPUT_ACTIVE_EPSILON for value in (
            command.linear.x, command.linear.y, command.linear.z,
            command.angular.x, command.angular.y, command.angular.z,
        ))

    def _tick(self):
        if self._stop_cycles > 0:
            self._stop_cycles -= 1
            self._publisher.publish(Twist())
            return
        now = time.monotonic()
        nav_fresh = self._nav_sec is not None and now - self._nav_sec <= INPUT_TIMEOUT_SEC
        gesture_fresh = (self._gesture_sec is not None
                         and now - self._gesture_sec <= INPUT_TIMEOUT_SEC)
        joystick_fresh = (self._joystick_sec is not None
                          and now - self._joystick_sec <= INPUT_TIMEOUT_SEC)
        glove_fresh = self._glove_sec is not None and now - self._glove_sec <= INPUT_TIMEOUT_SEC
        source = select_source(
            self._gesture_mode, self._joystick_mode, self._glove_mode,
            nav_fresh, gesture_fresh, joystick_fresh, glove_fresh,
            joystick_active=self._is_active(self._joystick_twist),
            glove_requires_controller=self._controller_mode_seen)
        if source == "nav":
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
        if source is not None:
            self.get_logger().info(
                "속도 전달 %s: linear.x=%.3f angular.z=%.3f" % (
                    source, output.linear.x, output.angular.z),
                throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
