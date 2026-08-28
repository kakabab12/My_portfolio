"""ROS2 브리지 — gesture_engine(Flask)의 /cmd를 폴링해 /cmd_vel(Twist)로 발행한다.

젯슨의 ROS2/turtlebot3 환경에서 실행한다 (rclpy는 pip이 아니라 ROS2 설치가
제공하므로 source /opt/ros/<distro>/setup.bash 이후 실행할 것):
    python3 ros2_bridge/cmd_vel_bridge.py --base-url http://127.0.0.1:5000

SLAM/Nav2 없이 제스처 주행만 확인할 때는 다음처럼 직접 /cmd_vel로 보낸다.
이 경우에는 turtlebot3_bringup만 별도로 실행되어 있어야 한다.
    python3 ros2_bridge/cmd_vel_bridge.py --output-topic /cmd_vel \
        --start-gesture-enabled --disable-navigation --disable-joystick

이중 안전정지의 두 번째 층(src/pipeline/gesture_loop.py 모듈독스트링의 첫
번째 층과 짝을 이룬다): HTTP 요청 실패·타임아웃이거나 응답의 age_sec이
--stale-sec을 넘으면 무조건 0 Twist를 발행한다 — gesture_engine 프로세스가
멈추거나 크래시해도, 네트워크가 끊겨도 로봇은 반드시 멈춘다. 정상 응답도
터틀봇3 버거의 실제 물리 한계로 마지막에 한 번 더 클램프한다 — 상대측
(Flask) 설정이 잘못돼 있어도 이 노드가 최종 방어선이 된다.
"""
import argparse
import math
import time

import rclpy
import requests
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool, String

try:
    from nav2_msgs.action import NavigateToPose
except ImportError:  # Nav2를 설치하지 않은 단독 제스처 테스트도 지원한다.
    NavigateToPose = None

POLL_HZ = 30.0
REQUEST_TIMEOUT_SEC = 0.15
COMMAND_STALE_SEC = 0.5
LOG_THROTTLE_SEC = 2.0

# 터틀봇3 버거 공식 물리 한계(configs/config.yaml의 robot 섹션과 같은 값) —
# 제스처 엔진 쪽 설정이 잘못돼도 이 노드가 마지막으로 클램프한다.
BURGER_MAX_LINEAR_MPS = 0.22
BURGER_MAX_ANGULAR_RADPS = 2.84
# 손가락 1개/2개 원샷 이동 거리. /odom 누적 거리로 측정하므로 전·후진 모두 1m다.
STEP_DISTANCE_M = 1.0
STEP_ANGLE_RAD = math.radians(90.0)
STEP_LINEAR_MPS = 0.05
STEP_ANGULAR_RADPS = 0.4
STEP_FIST_CANCEL_HOLD_SEC = 0.25
ODOM_STALE_SEC = 0.5
MODE_OFF_HOLD_SEC = 1.5
GESTURE_INACTIVITY_TIMEOUT_SEC = 15.0
STEP_ZONES = {"up", "down", "left", "right"}
FINGER_COMMANDS = {
    "finger": ("up", STEP_DISTANCE_M, None),
    "two": ("down", STEP_DISTANCE_M, None),
    "three": ("right", None, STEP_ANGLE_RAD),
    "four": ("left", None, STEP_ANGLE_RAD),
    "wave": ("left", None, math.pi),
}


def _clamp(value, limit):
    return max(-limit, min(limit, value))


def _yaw_from_quaternion(orientation):
    siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
    cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_distance(first, second):
    return abs(math.atan2(math.sin(second - first), math.cos(second - first)))


def _gesture_inactivity_expired(last_input_sec, now_sec, timeout_sec):
    """제스처 무입력 제한시간이 실제로 경과했는지 반환한다."""
    return (timeout_sec > 0.0 and last_input_sec is not None
            and now_sec - last_input_sec >= timeout_sec)


class CmdVelBridge(Node):
    def __init__(self, base_url, poll_hz, request_timeout_sec, command_stale_sec,
                 step_distance_m=STEP_DISTANCE_M, step_angle_rad=STEP_ANGLE_RAD,
                 step_linear_mps=STEP_LINEAR_MPS,
                 step_angular_radps=STEP_ANGULAR_RADPS,
                 output_topic="/cmd_vel_gesture", start_gesture_enabled=False,
                 enable_navigation=True, enable_joystick=True,
                 gesture_inactivity_timeout_sec=GESTURE_INACTIVITY_TIMEOUT_SEC):
        super().__init__("gesture_cmd_vel_bridge")
        self._cmd_url = base_url.rstrip("/") + "/cmd"
        self._control_mode_url = base_url.rstrip("/") + "/control_mode"
        self._request_timeout_sec = request_timeout_sec
        self._command_stale_sec = command_stale_sec
        self._output_topic = output_topic
        self._navigation_enabled = enable_navigation
        self._joystick_enabled = enable_joystick
        self._publisher = self.create_publisher(Twist, output_topic, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        if enable_navigation:
            if NavigateToPose is None:
                raise RuntimeError(
                    "nav2_msgs가 없습니다. SLAM/Nav2 없이 실행하려면 --disable-navigation을 사용하세요.")
            self.create_subscription(PoseStamped, "/goal_pose", self._on_goal_pose, 10)
            self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        else:
            self._nav_client = None
        self._mode_publisher = self.create_publisher(Bool, "/gesture_mode", 10)
        self._joystick_mode_publisher = (
            self.create_publisher(Bool, "/joystick_mode", 10)
            if enable_joystick else None)
        # waypoint_handoff_mission.py가 자동 구간에서는 수동 제어를 잠그고,
        # B 도착 후에는 사용자가 짧은 따봉으로 제스처 모드를 켤 수 있게
        # 잠금만 풀 때 사용한다. 이 토픽을 보내는 노드가 없으면 기존
        # 제스처/조이스틱 동작에는 아무 영향이 없다.
        self.create_subscription(
            String, "/mission_control_mode", self._on_mission_control_mode, 10)
        self._gesture_mode_enabled = start_gesture_enabled
        self._joystick_mode_enabled = False
        self._gesture_inactivity_timeout_sec = max(
            0.0, float(gesture_inactivity_timeout_sec))
        self._last_gesture_input_sec = (
            time.monotonic() if start_gesture_enabled else None)
        self._mission_auto_locked = False
        self._mission_hold_locked = False
        self._joystick_toggle_latched = False
        self._joystick_off_hold_start_sec = None
        self._mode_on_hold_start_sec = None
        self._mode_on_off_latched = False
        self._saved_goal_pose = None
        self._nav_goal_handle = None
        self._nav_active = False
        self._session = requests.Session()
        self._last_reported_control_mode = None
        self._step_distance_m = step_distance_m
        self._step_angle_rad = step_angle_rad
        self._step_linear_mps = step_linear_mps
        self._step_angular_radps = step_angular_radps
        self._odom_pose = None
        self._odom_update_sec = None
        self._step_zone = None
        self._step_shape = None
        self._last_command_shape = None
        self._step_start_pose = None
        self._step_target_distance = self._step_distance_m
        self._step_target_angle = self._step_angle_rad
        self._step_last_yaw = None
        self._step_angle_travel = 0.0
        self._step_latched = False
        self._step_exclusive = False
        self._step_fist_start_sec = None
        self.create_timer(1.0 / poll_hz, self._tick)
        self.get_logger().info(
            f"gesture_cmd_vel_bridge 시작 — {self._cmd_url} 을(를) {poll_hz:.0f}Hz로 폴링, "
            f"출력={self._output_topic}, 제스처 시작={'ON' if start_gesture_enabled else 'OFF'}, "
            f"제스처 무입력 전환={self._gesture_inactivity_timeout_sec:.0f}초")

    def _tick(self):
        self._publish_mode()
        if self._joystick_enabled:
            self._publish_joystick_mode()
        self._report_control_mode()
        twist = self._resolve_twist()
        if twist is not None:
            self._publisher.publish(twist)

    def _on_goal_pose(self, message):
        self._saved_goal_pose = message
        self.get_logger().info(
            f"RViz 목표 저장: frame={message.header.frame_id}, "
            f"x={message.pose.position.x:.2f}, y={message.pose.position.y:.2f}")

    def _on_odom(self, message):
        pose = message.pose.pose
        self._odom_pose = (
            float(pose.position.x), float(pose.position.y),
            _yaw_from_quaternion(pose.orientation))
        self._odom_update_sec = time.monotonic()

    def _on_mission_control_mode(self, message):
        """순찰 노드의 명시적 제어권 전환 요청을 적용한다.

        ``auto_lock``은 A->B 같은 자동 구간에서 인식된 손 모양이 우연히
        제어권을 빼앗지 않도록 한다. ``auto``는 자동 잠금은 풀지만
        제스처를 자동으로 켜지 않아, 사용자의 새 따봉 입력을 기다린다.
        ``gesture``가 오면 잠금을 풀고 바로 제스처 모드로 전환한다.
        모든 전환 순간에는 0 Twist를 한 번 내보낸다.
        """
        command = message.data.strip().lower()
        if command == "auto_lock":
            self._mission_auto_locked = True
            self._mission_hold_locked = False
            self._gesture_mode_enabled = False
            self._joystick_mode_enabled = False
        elif command == "auto":
            self._mission_auto_locked = False
            self._mission_hold_locked = False
            self._gesture_mode_enabled = False
            self._joystick_mode_enabled = False
            # B 도착 전부터 따봉을 들고 있었다면 도착 즉시 우연히
            # 제스처 모드가 켜지지 않게 한다. 한 번 손을 풀어야 새
            # 따봉 입력으로 인식된다.
            self._mode_on_off_latched = True
        elif command == "gesture":
            self._mission_auto_locked = False
            self._mission_hold_locked = False
            self._gesture_mode_enabled = True
            self._joystick_mode_enabled = False
        elif command == "joystick":
            if not self._joystick_enabled:
                self.get_logger().warning("순찰 조이스틱 전환 무시 — 조이스틱이 비활성화됨")
                return
            self._mission_auto_locked = False
            self._mission_hold_locked = False
            self._gesture_mode_enabled = False
            self._joystick_mode_enabled = True
        elif command == "hold":
            # 오류 시에는 어떤 최신 제스처 값도 통과시키지 않고, mux에 0 Twist를
            # 계속 공급한다. Nav2 출력도 이 모드에서는 선택되지 않는다.
            self._mission_auto_locked = False
            self._mission_hold_locked = True
            self._gesture_mode_enabled = True
            self._joystick_mode_enabled = False
        else:
            self.get_logger().warning(
                f"알 수 없는 순찰 제어권 명령 무시: {message.data!r}")
            return
        self._last_gesture_input_sec = (
            time.monotonic()
            if command == "gesture" and self._gesture_mode_enabled else None)
        self._cancel_step()
        self._step_exclusive = False
        self._step_latched = False
        self._publish_mode()
        self._publish_joystick_mode()
        self.stop_robot()
        self.get_logger().info(f"순찰 제어권 명령 적용: {command}")

    def _resolve_twist(self):
        """정상·신선한 응답이면 클램프된 속도, 그 외(오류·응답 지연)는 항상 0 Twist."""
        # A->B 자동 구간에서는 손이 카메라에 보여도 Nav2 출력권을 유지한다.
        if self._mission_auto_locked:
            return None
        if self._mission_hold_locked:
            return Twist()
        try:
            response = self._session.get(self._cmd_url, timeout=self._request_timeout_sec)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as error:
            self.get_logger().warning(
                f"gesture_engine 연결 실패 — 정지: {error}",
                throttle_duration_sec=LOG_THROTTLE_SEC)
            self._cancel_step()
            self._switch_to_controller_if_gesture_inactive(time.monotonic())
            return Twist()
        except ValueError as error:
            self.get_logger().warning(
                f"gesture_engine 응답 파싱 실패 — 정지: {error}",
                throttle_duration_sec=LOG_THROTTLE_SEC)
            self._cancel_step()
            self._switch_to_controller_if_gesture_inactive(time.monotonic())
            return Twist()

        age_sec = data.get("age_sec")
        if not isinstance(age_sec, (int, float)) or age_sec > self._command_stale_sec:
            self.get_logger().warning(
                f"gesture_engine 응답 오래됨(age_sec={age_sec}) — 정지",
                throttle_duration_sec=LOG_THROTTLE_SEC)
            self._cancel_step()
            self._switch_to_controller_if_gesture_inactive(time.monotonic())
            return Twist()

        shape = data.get("shape")
        zone = data.get("zone")
        hand_detected = bool(data.get("hand_detected"))
        now_sec = time.monotonic()
        if (self._gesture_mode_enabled and not self._joystick_mode_enabled
                and hand_detected):
            self._last_gesture_input_sec = now_sec
        if self._switch_to_controller_if_gesture_inactive(now_sec):
            return Twist()

        # 제스처 모드 ON 상태에서 따봉을 1.5초 유지하면 제어권을 AUTO/Nav2로
        # 돌려준다. 토글 직후에도 같은 따봉을 계속 들고 있으면 곧바로 다시
        # ON되지 않도록, 손을 풀거나 다른 손모양이 들어와야 재장전한다.
        if not hand_detected or shape != "mode_on":
            self._mode_on_hold_start_sec = None
            self._mode_on_off_latched = False

        if self._joystick_enabled and (not hand_detected or shape != "joystick_toggle"):
            self._joystick_toggle_latched = False
            self._joystick_off_hold_start_sec = None
        if self._joystick_enabled and hand_detected and shape == "joystick_toggle":
            if not self._joystick_mode_enabled and not self._joystick_toggle_latched:
                self._joystick_toggle_latched = True
                self._joystick_mode_enabled = True
                self._cancel_step()
                self._step_exclusive = False
                self._step_latched = False
                self._gesture_mode_enabled = False
                self._last_gesture_input_sec = None
                self._publish_mode()
                self._publish_joystick_mode()
                self.get_logger().info("OK 제스처 — 조이스틱 모드 ON")
                return Twist()
            if self._joystick_mode_enabled:
                # ON에 사용한 같은 OK 자세를 계속 들고 있어도 바로 OFF 타이머를
                # 시작하지 않는다. 한 번 손을 풀고 새 OK 사인을 만들어야 한다.
                if self._joystick_toggle_latched:
                    return None
                now_sec = time.monotonic()
                if self._joystick_off_hold_start_sec is None:
                    self._joystick_off_hold_start_sec = now_sec
                    self.get_logger().info(
                        "컨트롤러 OFF 대기 — OK 사인을 1.5초간 유지하세요")
                if now_sec - self._joystick_off_hold_start_sec >= MODE_OFF_HOLD_SEC:
                    self._joystick_mode_enabled = False
                    self._gesture_mode_enabled = False
                    self._last_gesture_input_sec = None
                    self._joystick_toggle_latched = True
                    self._joystick_off_hold_start_sec = None
                    self._publish_joystick_mode()
                    self._publish_mode()
                    self.get_logger().info(
                        "OK 사인 1.5초 유지 — 컨트롤러·제스처 모드 OFF, "
                        "AUTO/Nav2로 제어권 반환")
                    return Twist()
            return None

        # 컨트롤러 모드 해제는 위의 OK 사인 1.5초 유지로만 처리한다.
        # 엄지 아래와 따봉은 컨트롤러 모드의 제어권을 바꾸지 않는다.
        if self._joystick_enabled and self._joystick_mode_enabled:
            return None

        if self._step_zone is not None:
            return self._advance_step(data)

        if not self._gesture_mode_enabled:
            if (hand_detected and shape == "mode_on"
                    and not self._mode_on_off_latched):
                self._gesture_mode_enabled = True
                self._last_gesture_input_sec = time.monotonic()
                self._publish_mode()
                self.get_logger().info("제스처 모드 ON — SLAM/Nav2 유지, 속도 출력권만 획득")
                return Twist()
            # OFF 동안에는 0 속도조차 계속 발행하지 않아 Nav2 /cmd_vel과 경쟁하지 않는다.
            return None

        if hand_detected and shape == "mode_on":
            now_sec = time.monotonic()
            if self._mode_on_hold_start_sec is None:
                self._mode_on_hold_start_sec = now_sec
                self.get_logger().info("따봉 유지 중 — 1.5초 후 제스처 모드 OFF")
            elif (not self._mode_on_off_latched
                  and now_sec - self._mode_on_hold_start_sec >= MODE_OFF_HOLD_SEC):
                self._cancel_step()
                self._step_exclusive = False
                self._step_latched = False
                self._gesture_mode_enabled = False
                self._last_gesture_input_sec = None
                self._mode_on_off_latched = True
                self._publish_mode()
                self.get_logger().info(
                    "따봉 1.5초 유지 — 제스처 모드 OFF, AUTO/Nav2로 제어권 반환")
                return Twist()
        if self._nav_active:
            if hand_detected and shape == "fist":
                if self._nav_goal_handle is not None:
                    self._nav_goal_handle.cancel_goal_async()
                self._nav_active = False
                self.get_logger().info("Nav2 목표 이동 취소 — 주먹 정지")
                return Twist()
            return None

        if self._step_exclusive:
            # 완료 직후 같은 자세를 계속 들고 있는 동안만 중복 실행을 막는다.
            # 손을 빼지 않아도 다른 명령 자세가 확정되면 즉시 재장전한다.
            if not hand_detected:
                self._step_exclusive = False
                self._step_latched = False
                self._last_command_shape = None
            elif (shape != self._last_command_shape
                  and (shape in FINGER_COMMANDS or shape in ("back", "fist"))):
                self._step_exclusive = False
                self._step_latched = False
            else:
                return Twist()

        if shape == "back" and hand_detected:
            if not self._step_latched:
                self._step_latched = True
                self._start_saved_nav_goal()
            return Twist()

        if shape in FINGER_COMMANDS and hand_detected:
            if self._step_latched:
                return Twist()
            if not self._odom_is_fresh():
                self.get_logger().warning(
                    "손가락 명령을 받았지만 /odom이 없거나 오래됨 — 정지",
                    throttle_duration_sec=LOG_THROTTLE_SEC)
                return Twist()
            command_zone, distance_m, angle_rad = FINGER_COMMANDS[shape]
            self._step_zone = command_zone
            self._step_shape = shape
            self._last_command_shape = shape
            self._step_start_pose = self._odom_pose
            self._step_target_distance = distance_m
            self._step_target_angle = angle_rad
            self._step_last_yaw = self._odom_pose[2]
            self._step_angle_travel = 0.0
            self._step_latched = True
            self._step_exclusive = True
            target = (f"{distance_m:.2f}m" if distance_m is not None
                      else f"{math.degrees(angle_rad):.0f}deg")
            self.get_logger().info(f"손가락 원샷 명령 시작: shape={shape}, target={target}")
            return self._step_twist(command_zone)

        if shape == "finger":
            if not hand_detected or zone not in STEP_ZONES:
                if zone in (None, "center"):
                    self._step_latched = False
                return Twist()
            if self._step_latched:
                return Twist()
            if not self._odom_is_fresh():
                self.get_logger().warning(
                    "단계 이동 요청을 받았지만 /odom이 없거나 오래됨 — 정지",
                    throttle_duration_sec=LOG_THROTTLE_SEC)
                return Twist()
            self._step_zone = zone
            self._step_start_pose = self._odom_pose
            self._step_latched = True
            self._step_exclusive = True
            self.get_logger().info(
                f"한 손가락 단계 이동 시작: zone={zone}, "
                f"distance={self._step_distance_m:.2f}m, "
                f"angle={math.degrees(self._step_angle_rad):.0f}deg")
            return self._step_twist(zone)

        self._cancel_step()
        self._step_latched = False

        twist = Twist()
        twist.linear.x = _clamp(float(data.get("linear_x", 0.0)), BURGER_MAX_LINEAR_MPS)
        twist.angular.z = _clamp(float(data.get("angular_z", 0.0)), BURGER_MAX_ANGULAR_RADPS)
        return twist

    def _switch_to_controller_if_gesture_inactive(self, now_sec):
        """제스처 무입력 15초가 지나면 정지 후 컨트롤러 제어권을 넘긴다.

        원샷 거리·회전 또는 Nav2 동작 중에는 현재 동작을 중간에 끊지 않는다.
        동작이 끝난 다음 폴링 주기에서 제한시간을 다시 판단한다.
        """
        if (not self._joystick_enabled or not self._gesture_mode_enabled
                or self._joystick_mode_enabled or self._mission_auto_locked
                or self._mission_hold_locked or self._step_zone is not None
                or self._nav_active
                or not _gesture_inactivity_expired(
                    self._last_gesture_input_sec, now_sec,
                    self._gesture_inactivity_timeout_sec)):
            return False
        self._cancel_step()
        self._step_exclusive = False
        self._step_latched = False
        self._mode_on_hold_start_sec = None
        self._mode_on_off_latched = False
        self._joystick_toggle_latched = False
        self._joystick_off_hold_start_sec = None
        self._gesture_mode_enabled = False
        self._joystick_mode_enabled = True
        self._last_gesture_input_sec = None
        self._publish_mode()
        self._publish_joystick_mode()
        self.get_logger().info(
            f"제스처 입력 없음 {self._gesture_inactivity_timeout_sec:.0f}초 — "
            "정지 후 컨트롤러 모드로 자동 전환")
        return True

    def _odom_is_fresh(self):
        return (self._odom_pose is not None and self._odom_update_sec is not None
                and time.monotonic() - self._odom_update_sec <= ODOM_STALE_SEC)

    def _advance_step(self, data):
        # 손을 내리면 현재 원샷은 목표까지 계속하지만, 명확한 새 제스처가
        # 들어오면 현재 동작을 버리고 그 순간 위치에서 새 목표를 시작한다.
        if bool(data.get("hand_detected")) and data.get("shape") == "fist":
            now_sec = time.monotonic()
            if self._step_fist_start_sec is None:
                self._step_fist_start_sec = now_sec
            if now_sec - self._step_fist_start_sec >= STEP_FIST_CANCEL_HOLD_SEC:
                self.get_logger().info("한 손가락 단계 이동 취소 — 주먹 비상 정지")
                self._cancel_step()
                return Twist()
        else:
            # 손가락을 거두는 찰나의 1~2프레임 fist 오인을 비상정지로 쓰지 않는다.
            self._step_fist_start_sec = None

        new_shape = data.get("shape") if bool(data.get("hand_detected")) else None
        if new_shape == "back":
            self.get_logger().info("진행 중 상대 이동 중단 — 손등 Nav2 명령으로 전환")
            self._cancel_step()
            self._step_exclusive = False
            self._start_saved_nav_goal()
            return Twist()
        if new_shape in FINGER_COMMANDS and new_shape != self._step_shape:
            command_zone, distance_m, angle_rad = FINGER_COMMANDS[new_shape]
            self._step_zone = command_zone
            self._step_shape = new_shape
            self._last_command_shape = new_shape
            self._step_start_pose = self._odom_pose
            self._step_target_distance = distance_m
            self._step_target_angle = angle_rad
            self._step_last_yaw = self._odom_pose[2]
            self._step_angle_travel = 0.0
            target = (f"{distance_m:.2f}m" if distance_m is not None
                      else f"{math.degrees(angle_rad):.0f}deg")
            self.get_logger().info(
                f"진행 중 명령 전환: shape={new_shape}, target={target}")
            return self._step_twist(command_zone)

        if not self._odom_is_fresh():
            self.get_logger().warning(
                "한 손가락 단계 이동 취소 — /odom이 없거나 오래됨",
                throttle_duration_sec=LOG_THROTTLE_SEC)
            self._cancel_step()
            return Twist()

        start_x, start_y, start_yaw = self._step_start_pose
        x, y, yaw = self._odom_pose
        if self._step_zone in ("up", "down"):
            reached = math.hypot(x - start_x, y - start_y) >= self._step_target_distance
        else:
            if self._step_last_yaw is not None:
                self._step_angle_travel += _angle_distance(self._step_last_yaw, yaw)
            self._step_last_yaw = yaw
            reached = self._step_angle_travel >= self._step_target_angle
        if reached:
            self.get_logger().info(f"한 손가락 단계 이동 완료: zone={self._step_zone}")
            self._cancel_step()
            return Twist()
        return self._step_twist(self._step_zone)

    def _step_twist(self, zone):
        twist = Twist()
        if zone == "up":
            twist.linear.x = self._step_linear_mps
        elif zone == "down":
            twist.linear.x = -self._step_linear_mps
        elif zone == "left":
            twist.angular.z = self._step_angular_radps
        elif zone == "right":
            twist.angular.z = -self._step_angular_radps
        return twist

    def _cancel_step(self):
        self._step_zone = None
        self._step_shape = None
        self._step_start_pose = None
        self._step_fist_start_sec = None
        self._step_last_yaw = None
        self._step_angle_travel = 0.0

    def _start_saved_nav_goal(self):
        if not self._navigation_enabled or self._nav_client is None:
            self.get_logger().warning("손등 명령 무시 — 단독 제스처 테스트에서는 Nav2를 사용하지 않음")
            return
        if self._saved_goal_pose is None:
            self.get_logger().warning("손등 명령 무시 — RViz 2D Goal Pose가 저장되지 않음")
            return
        if not self._nav_client.server_is_ready():
            self.get_logger().warning("손등 명령 무시 — Nav2 navigate_to_pose 서버가 없음")
            return
        goal = NavigateToPose.Goal()
        goal.pose = self._saved_goal_pose
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_nav_goal_response)

    def _on_nav_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warning("Nav2 목표가 거부됨")
            return
        self._nav_goal_handle = handle
        self._nav_active = True
        # 저장 좌표 이동은 Nav2가 담당하므로 제스처 속도 출력권을 반환한다.
        self._gesture_mode_enabled = False
        self._last_gesture_input_sec = None
        self._publish_mode()
        handle.get_result_async().add_done_callback(self._on_nav_result)
        self.get_logger().info("손등 제스처 — 저장된 RViz 목표로 이동 시작")

    def _on_nav_result(self, _future):
        self._nav_active = False
        self._nav_goal_handle = None
        self.get_logger().info("RViz 목표 이동 종료")

    def _publish_mode(self):
        message = Bool()
        message.data = self._gesture_mode_enabled
        self._mode_publisher.publish(message)

    def _publish_joystick_mode(self):
        if self._joystick_mode_publisher is None:
            return
        message = Bool()
        message.data = self._joystick_mode_enabled
        self._joystick_mode_publisher.publish(message)

    def _report_control_mode(self):
        """웹 진단 화면에 mux가 현재 선택할 제어권을 전달한다.

        상태가 바뀔 때만 HTTP 요청을 보내므로 제스처 제어 루프의 30Hz 폴링을
        방해하지 않는다. 서버가 잠시 내려가 있어도 명령 경로와는 독립적으로
        재시도하며, 실패 시에는 안전 정지 규칙에 영향을 주지 않는다.
        """
        mode = ("joystick" if self._joystick_mode_enabled
                else "gesture" if self._gesture_mode_enabled else "auto")
        if mode == self._last_reported_control_mode:
            return
        try:
            response = self._session.post(
                self._control_mode_url, json={"mode": mode},
                timeout=self._request_timeout_sec)
            response.raise_for_status()
        except requests.RequestException as error:
            self.get_logger().warning(
                f"웹 제어 모드 상태 보고 실패: {error}",
                throttle_duration_sec=LOG_THROTTLE_SEC)
            return
        self._last_reported_control_mode = mode

    def stop_robot(self):
        self._publisher.publish(Twist())


def main():
    parser = argparse.ArgumentParser(description="gesture_engine -> /cmd_vel 브리지")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000",
                        help="gesture_engine(Flask) 주소 — 같은 젯슨이면 기본값 그대로")
    parser.add_argument("--poll-hz", type=float, default=POLL_HZ)
    parser.add_argument("--timeout-sec", type=float, default=REQUEST_TIMEOUT_SEC)
    parser.add_argument("--stale-sec", type=float, default=COMMAND_STALE_SEC)
    parser.add_argument(
        "--gesture-inactivity-sec", type=float,
        default=GESTURE_INACTIVITY_TIMEOUT_SEC,
        help="제스처 모드에서 입력이 없을 때 컨트롤러 모드로 바뀌는 시간(초, 0=비활성)")
    parser.add_argument("--step-distance-m", type=float, default=STEP_DISTANCE_M)
    parser.add_argument("--step-angle-deg", type=float, default=math.degrees(STEP_ANGLE_RAD))
    parser.add_argument("--step-linear-mps", type=float, default=STEP_LINEAR_MPS)
    parser.add_argument("--step-angular-radps", type=float, default=STEP_ANGULAR_RADPS)
    parser.add_argument(
        "--output-topic", default="/cmd_vel_gesture",
        help="Twist 출력 토픽 (단독 실주행 테스트는 /cmd_vel, 기본값은 mux 입력)")
    parser.add_argument(
        "--start-gesture-enabled", action="store_true",
        help="시작 직후 제스처 제어권을 켠다 (단독 실주행 테스트용)")
    parser.add_argument(
        "--disable-navigation", action="store_true",
        help="Nav2 목표 수신/실행을 끈다 (SLAM·Nav2 없는 단독 테스트용)")
    parser.add_argument(
        "--disable-joystick", action="store_true",
        help="OK 사인 조이스틱 전환을 끈다 (조이스틱 없는 단독 테스트용)")
    args = parser.parse_args()

    rclpy.init()
    node = CmdVelBridge(
        args.base_url, args.poll_hz, args.timeout_sec, args.stale_sec,
        step_distance_m=args.step_distance_m,
        step_angle_rad=math.radians(args.step_angle_deg),
        step_linear_mps=args.step_linear_mps,
        step_angular_radps=args.step_angular_radps,
        output_topic=args.output_topic,
        start_gesture_enabled=args.start_gesture_enabled,
        enable_navigation=not args.disable_navigation,
        enable_joystick=not args.disable_joystick,
        gesture_inactivity_timeout_sec=args.gesture_inactivity_sec)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop_robot()   # 종료 직전 0 Twist 한 번 더 — 마지막 명령이 남지 않게
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
