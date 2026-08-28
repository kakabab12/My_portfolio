#!/usr/bin/env python3
"""A-B-C-D-A 구간별 제어권 미션.

저장 지도(factory_map_final)에서 다음 순서를 강제한다.

* A -> B: Nav2
* B -> C: 제스처
* C -> D: 조이스틱
* D -> A: Nav2

수동 구간은 /amcl_pose의 map 좌표가 목표점 반경 안에 일정 시간 머물러도
도착 처리한다. 시연 중에는 제어권 전환을 사용자의 명시적 도착 확인으로도
받는다. 즉 C에서 조이스틱 ON, D에서 AUTO/Nav2 전환은 좌표와 관계없이
각각 C/D 도착을 확정한다.
"""
import json
import math
import time
from urllib import error as urlerror
from urllib import request as urlrequest

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


WEB_STATUS_URL = "http://127.0.0.1:5000/mission_status"
# A/B Nav2 구간의 목표 허용오차와 통일한다. AMCL 오차와 손·조이스틱 수동
# 주행 편차를 고려해 C/D 목표점 10cm 이내면 도착으로 처리한다.
ARRIVAL_RADIUS_M = 0.10
ARRIVAL_HOLD_SEC = 1.0
# /navigate_to_pose action이 목록에 나타난 시점은 AMCL lifecycle 활성화보다
# 앞설 수 있다. 그래서 초기 위치 A를 짧게 한 번만 보내면 AMCL 구독 전이라
# 유실되고 A->B가 출발하지 않는다. /amcl_pose가 들어올 때까지 최대 15초간
# 반복 전송한다.
INITIAL_POSE_WAIT_TIMEOUT_SEC = 15.0
INITIAL_POSE_PUBLISH_INTERVAL_SEC = 0.4
STATUS_POST_INTERVAL_SEC = 0.5
NAV_GOAL_ACCEPT_TIMEOUT_SEC = 30.0
NAV_GOAL_RETRY_SEC = 0.5
NAV_GOAL_RESPONSE_TIMEOUT_SEC = 3.0
NAV2_ACTIVE_WAIT_TIMEOUT_SEC = 30.0

# 이 미션은 A에서 출발하는 시연이다. 재실행 전에 로봇을 A로 옮기지 않으면
# /initialpose가 실제 위치를 A로 덮어써 라이다와 정적 지도가 어긋난다. 그 상태로
# Nav2를 시작하면 경로 재계획 실패와 복구 회전이 발생할 수 있으므로, 움직이기 전
# 라이다 끝점이 지도 벽/장애물과 실제로 맞는지 여러 프레임 확인한다.
START_POSE_VALIDATION_TIMEOUT_SEC = 10.0
START_POSE_STATIC_MARGIN_M = 0.12
START_POSE_MIN_MATCH_RATIO = 0.55
START_POSE_MIN_ENDPOINTS = 30
START_POSE_REQUIRED_FRAMES = 3
START_POSE_SCAN_MAX_RANGE_M = 3.0
START_POSE_SCAN_SAMPLE_STEP = 2

# factory_map_final map-frame measurements.
WAYPOINTS = {
    "A": (0.044, -0.115, 0.000),
    "B": (1.429, -0.213, -1.579),
    "C": (1.376, -1.451, -1.559),
    "D": (0.124, -1.482, -3.113),
}


def _yaw_to_quaternion(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _quaternion_to_yaw(quaternion):
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _scan_endpoint_matches_static_map(
        occupancy_grid, point_x, point_y, margin, occupied_threshold=65):
    """Return whether an in-map scan endpoint agrees with static occupancy."""
    info = occupancy_grid.info
    resolution = float(info.resolution)
    width = int(info.width)
    height = int(info.height)
    if resolution <= 0.0 or width <= 0 or height <= 0:
        return False

    origin = info.origin
    origin_yaw = _quaternion_to_yaw(origin.orientation)
    delta_x = float(point_x) - float(origin.position.x)
    delta_y = float(point_y) - float(origin.position.y)
    cosine = math.cos(origin_yaw)
    sine = math.sin(origin_yaw)
    map_x = cosine * delta_x + sine * delta_y
    map_y = -sine * delta_x + cosine * delta_y
    cell_x = math.floor(map_x / resolution)
    cell_y = math.floor(map_y / resolution)
    # 지도 밖 끝점을 일치로 취급하면 잘못된 pose가 높은 점수를 받을 수 있다.
    if cell_x < 0 or cell_x >= width or cell_y < 0 or cell_y >= height:
        return False

    radius_cells = max(0, math.ceil(float(margin) / resolution))
    for y_index in range(
            max(0, cell_y - radius_cells),
            min(height, cell_y + radius_cells + 1)):
        for x_index in range(
                max(0, cell_x - radius_cells),
                min(width, cell_x + radius_cells + 1)):
            value = int(occupancy_grid.data[y_index * width + x_index])
            if value < 0 or value >= occupied_threshold:
                return True
    return False


def scan_map_match_stats(
        scan, occupancy_grid, sensor_pose, static_margin=START_POSE_STATIC_MARGIN_M,
        max_range=START_POSE_SCAN_MAX_RANGE_M,
        sample_step=START_POSE_SCAN_SAMPLE_STEP):
    """Return (matched, usable, ratio) for scan endpoints at a map pose."""
    if scan is None or occupancy_grid is None or sensor_pose is None:
        return 0, 0, 0.0
    step = max(1, int(sample_step))
    sensor_x, sensor_y, sensor_yaw = sensor_pose
    cosine = math.cos(sensor_yaw)
    sine = math.sin(sensor_yaw)
    usable_limit = min(float(max_range), float(scan.range_max))
    matched = 0
    usable = 0
    for index in range(0, len(scan.ranges), step):
        distance = float(scan.ranges[index])
        if (not math.isfinite(distance)
                or distance < float(scan.range_min)
                or distance > usable_limit):
            continue
        angle = float(scan.angle_min) + index * float(scan.angle_increment)
        local_x = distance * math.cos(angle)
        local_y = distance * math.sin(angle)
        point_x = sensor_x + cosine * local_x - sine * local_y
        point_y = sensor_y + sine * local_x + cosine * local_y
        usable += 1
        if _scan_endpoint_matches_static_map(
                occupancy_grid, point_x, point_y, static_margin):
            matched += 1
    ratio = matched / usable if usable else 0.0
    return matched, usable, ratio


def claim_goal_response(mission, future):
    """Reject a late action response after its watchdog already halted.

    Older unit-test stand-ins do not have the watchdog field; those represent
    an ordinary current response and remain accepted.
    """
    if hasattr(mission, "_goal_response_future"):
        if mission._goal_response_future is not future:
            mission.get_logger().warning(
                "시간 초과 뒤 도착한 Nav2 목표 응답을 무시합니다")
            return False
        mission._goal_response_future = None
        mission._goal_response_deadline_sec = None
    return True


class WaypointHandoffMission(Node):
    """Nav2 action과 수동 제어권을 연결하는 단일 미션 상태기계."""

    def __init__(self):
        super().__init__("waypoint_handoff_mission")
        command_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._mode_command_publisher = self.create_publisher(
            String, "/mission_control_mode", command_qos)
        self._initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 20)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid, "/map", self._on_map, map_qos)
        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Bool, "/gesture_mode", self._on_gesture_mode, 20)
        self.create_subscription(Bool, "/joystick_mode", self._on_joystick_mode, 20)
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._nav_state_client = self.create_client(
            GetState, "/bt_navigator/get_state")

        self._phase = "initializing"
        self._label = "Nav2와 AMCL 초기화 중"
        self._target = "B"
        self._checkpoints = {"A": "출발", "B": "이동 예정", "C": "대기", "D": "대기"}
        self._position = None
        self._yaw = None
        self._last_position_sec = None
        self._position_covariance_xy = None
        self._occupancy_grid = None
        self._latest_scan = None
        self._scan_generation = 0
        self._gesture_mode = False
        self._joystick_mode = False
        self._arrival_started_sec = None
        self._active_goal_name = None
        # Action 서버는 Nav2 lifecycle 활성화 전에도 발견될 수 있다. 목표가
        # 거부되면 Nav2가 실제로 활성화될 때까지 이 상태로 재시도한다.
        self._pending_goal_name = None
        self._goal_accept_deadline_sec = None
        self._next_goal_attempt_sec = None
        self._goal_response_future = None
        self._goal_response_deadline_sec = None
        self._last_status_post_sec = 0.0
        self._last_status_signature = None
        self.create_timer(0.1, self._tick)

    def run(self):
        self.get_logger().info(
            "구간별 미션: A->B(Nav2), B->C(제스처), C->D(조이스틱), D->A(Nav2)")
        # 위치 검증이 끝날 때까지 mux를 0 Twist 상태로 잠근다.
        self._publish_mode_command("hold")
        self._post_status(force=True)

        # Nav2는 AMCL이 initial pose를 받아 map->odom TF를 만든 뒤에야
        # navigate_to_pose action을 ACTIVE로 만든다. 따라서 action 서버를
        # 먼저 블로킹 대기하면 initial pose가 한 번도 발행되지 않아 서로
        # 기다리는 상태가 된다. 서버 대기 중에도 A 초기 위치를 반복 발행해
        # Nav2 lifecycle을 완료시킨다.
        if not self._wait_for_nav_server_with_initial_pose(timeout_sec=45.0):
            self._halt("Nav2 navigate_to_pose 서버를 찾지 못했습니다.")
            return

        # AMCL 구독자가 준비되는 시간 동안 A 초기 위치를 반복 발행한다. 이
        # 좌표는 A->B 시작을 위한 초기 위치일 뿐, D->A 복귀 때는 다시 설정하지
        # 않는다. 수동 주행 중 갱신된 AMCL 위치를 그대로 사용해야 하기 때문이다.
        deadline = time.monotonic() + INITIAL_POSE_WAIT_TIMEOUT_SEC
        next_pose_publish_sec = 0.0
        next_lock_publish_sec = 0.0
        while (rclpy.ok() and self._position is None
               and time.monotonic() < deadline):
            now = time.monotonic()
            if now >= next_pose_publish_sec:
                self._publish_initial_pose()
                next_pose_publish_sec = now + INITIAL_POSE_PUBLISH_INTERVAL_SEC
            if now >= next_lock_publish_sec:
                self._publish_mode_command("hold")
                next_lock_publish_sec = now + 0.4
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._position is None:
            self._halt(
                f"AMCL 현재 좌표를 {INITIAL_POSE_WAIT_TIMEOUT_SEC:.0f}초 안에 "
                "받지 못해 A->B를 시작하지 않았습니다.")
            return
        if not self._wait_for_nav2_active():
            self._halt(
                "bt_navigator가 ACTIVE 상태가 되지 않아 A->B를 시작하지 않았습니다.")
            return
        if not self._wait_for_start_pose_validation():
            self._halt(
                "초기 위치와 라이다 지도가 일치하지 않습니다. 로봇을 A 위치와 "
                "A 방향에 정확히 놓은 뒤 전체 실행을 다시 시작하세요.")
            return
        self._start_nav_to_b()
        rclpy.spin(self)

    def _wait_for_nav_server_with_initial_pose(self, timeout_sec):
        """Publish A's initial pose while waiting for Nav2 to become active."""
        deadline = time.monotonic() + float(timeout_sec)
        next_pose_publish_sec = 0.0
        next_lock_publish_sec = 0.0
        self.get_logger().info(
            "Nav2 준비 대기 중 — A 초기 위치를 반복 등록합니다")

        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_pose_publish_sec:
                self._publish_initial_pose()
                next_pose_publish_sec = now + INITIAL_POSE_PUBLISH_INTERVAL_SEC
            if now >= next_lock_publish_sec:
                self._publish_mode_command("hold")
                next_lock_publish_sec = now + 0.4
            if self._nav_client.wait_for_server(timeout_sec=0.2):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _wait_for_start_pose_validation(self):
        """Fail closed unless several stationary scans agree with the map."""
        self._set_phase(
            "validating_start_pose",
            "출발 전 A 위치와 라이다 지도 일치 여부 확인 중",
            "B")
        self._publish_mode_command("hold")
        deadline = time.monotonic() + START_POSE_VALIDATION_TIMEOUT_SEC
        checked_generation = -1
        consecutive_matches = 0
        last_diagnostic_sec = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._scan_generation == checked_generation:
                continue
            checked_generation = self._scan_generation
            pose = None
            if self._position is not None and self._yaw is not None:
                pose = (self._position[0], self._position[1], self._yaw)
            matched, usable, ratio = scan_map_match_stats(
                self._latest_scan, self._occupancy_grid, pose)
            if usable >= START_POSE_MIN_ENDPOINTS and ratio >= START_POSE_MIN_MATCH_RATIO:
                consecutive_matches += 1
                if consecutive_matches >= START_POSE_REQUIRED_FRAMES:
                    self.get_logger().info(
                        "출발 위치 검증 통과 — "
                        f"지도 일치 {matched}/{usable} ({ratio:.0%}), "
                        f"{consecutive_matches}개 연속 scan")
                    return True
            else:
                consecutive_matches = 0

            now = time.monotonic()
            if now - last_diagnostic_sec >= 1.0:
                self.get_logger().warning(
                    "출발 위치 검증 대기 — "
                    f"지도 일치 {matched}/{usable} ({ratio:.0%}), "
                    f"필요 {START_POSE_MIN_MATCH_RATIO:.0%} 이상 / "
                    f"끝점 {START_POSE_MIN_ENDPOINTS}개 이상")
                last_diagnostic_sec = now
        return False

    def _wait_for_nav2_active(self):
        """Wait for lifecycle ACTIVE, not merely action-server discovery."""
        deadline = time.monotonic() + NAV2_ACTIVE_WAIT_TIMEOUT_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            if not self._nav_state_client.wait_for_service(timeout_sec=0.2):
                continue
            future = self._nav_state_client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
            if not future.done():
                continue
            try:
                response = future.result()
            except Exception as exc:
                self.get_logger().warning(
                    f"bt_navigator lifecycle 상태 조회 실패: {exc}",
                    throttle_duration_sec=2.0)
                continue
            if response.current_state.id == State.PRIMARY_STATE_ACTIVE:
                self.get_logger().info(
                    "bt_navigator ACTIVE 확인 — Nav2 목표 전송을 시작합니다")
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _publish_initial_pose(self):
        x, y, yaw = WAYPOINTS["A"]
        message = PoseWithCovarianceStamped()
        message.header.frame_id = "map"
        # AMCL 시작 직후에는 odom TF가 현재 시각보다 약간 늦게 올라온다. 현재
        # 시각을 찍으면 그 TF를 정확히 같은 시각에 찾다가 초기 위치를 거부한다.
        # stamp=0은 AMCL에 "가장 최신 TF"를 쓰도록 요청하는 ROS 표준 방식이다.
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        message.pose.pose.orientation.z, message.pose.pose.orientation.w = _yaw_to_quaternion(yaw)
        # 20cm / 15deg 수준의 초기 불확실성. 0 공분산은 AMCL이 센서를 반영해
        # 보정할 여지를 지나치게 줄이므로 사용하지 않는다.
        message.pose.covariance[0] = 0.04
        message.pose.covariance[7] = 0.04
        message.pose.covariance[35] = math.radians(15.0) ** 2
        self._initial_pose_publisher.publish(message)

    def _on_amcl_pose(self, message):
        pose = message.pose.pose
        self._position = (float(pose.position.x), float(pose.position.y))
        self._yaw = _quaternion_to_yaw(pose.orientation)
        self._last_position_sec = time.monotonic()
        covariance = message.pose.covariance
        covariance_x = float(covariance[0])
        covariance_y = float(covariance[7])
        if math.isfinite(covariance_x) and math.isfinite(covariance_y):
            self._position_covariance_xy = max(
                0.0, covariance_x, covariance_y)
        else:
            self._position_covariance_xy = math.inf

    def _on_map(self, message):
        self._occupancy_grid = message

    def _on_scan(self, message):
        self._latest_scan = message
        self._scan_generation += 1

    def _on_gesture_mode(self, message):
        was_enabled = self._gesture_mode
        self._gesture_mode = bool(message.data)
        if (self._phase == "wait_gesture" and self._gesture_mode
                and not was_enabled):
            self._set_phase(
                "gesture_to_c",
                "제스처 모드 ON — C 지점으로 이동하세요.",
                "C")
        self._maybe_start_return_to_a()

    def _on_joystick_mode(self, message):
        was_enabled = self._joystick_mode
        self._joystick_mode = bool(message.data)
        # C에서 OK 사인을 내거나 제스처 15초 무입력 자동 전환으로 컨트롤러가
        # 켜지면, 사용자가 C 도착을 명시적으로 확인한 것으로 간주한다. 시연
        # 전환은 좌표·AMCL 오차에 영향받지 않도록 반경 검증을 하지 않는다.
        if (self._phase == "gesture_to_c" and self._joystick_mode
                and not was_enabled):
            self.get_logger().info(
                "컨트롤러 전환을 C 도착 확인으로 수신 — 좌표 검증 없이 C 완료 처리")
            self._complete_manual_arrival("C")
        if (self._phase == "wait_controller" and self._joystick_mode
                and not was_enabled):
            self._set_phase(
                "controller_to_d",
                "컨트롤러로 D 지점 이동 중 — D 좌표 도착을 확인합니다.",
                "D")
        self._maybe_start_return_to_a()

    def _tick(self):
        self._retry_pending_nav_goal()
        if self._phase == "gesture_to_c":
            # 제스처 무입력 자동 전환 뒤에는 컨트롤러로 C까지 이어서 이동할 수 있다.
            self._confirm_manual_arrival(
                "C", self._gesture_mode or self._joystick_mode)
        elif self._phase == "controller_to_d":
            self._confirm_manual_arrival("D", self._joystick_mode)
        self._post_status()

    def _confirm_manual_arrival(self, waypoint_name, manual_mode_active):
        if not manual_mode_active or self._position is None:
            self._arrival_started_sec = None
            return
        target_x, target_y, _ = WAYPOINTS[waypoint_name]
        x, y = self._position
        distance = math.hypot(x - target_x, y - target_y)
        if distance > ARRIVAL_RADIUS_M:
            self._arrival_started_sec = None
            return
        now = time.monotonic()
        if self._arrival_started_sec is None:
            self._arrival_started_sec = now
            self.get_logger().info(
                f"{waypoint_name} 도착 반경 진입 ({distance:.2f}m) — "
                f"{ARRIVAL_HOLD_SEC:.0f}초 안정화 확인 중")
            return
        if now - self._arrival_started_sec < ARRIVAL_HOLD_SEC:
            return
        self._arrival_started_sec = None
        self._complete_manual_arrival(waypoint_name)

    def _complete_manual_arrival(self, waypoint_name):
        """수동 구간의 도착 상태를 한 곳에서 확정한다."""
        self._arrival_started_sec = None
        if waypoint_name == "C":
            self._checkpoints["C"] = "도착 완료"
            if self._joystick_mode:
                self._set_phase(
                    "controller_to_d",
                    "C 도착 완료 — 현재 컨트롤러 모드로 D 지점 이동을 계속하세요.",
                    "D")
            else:
                self._set_phase(
                    "wait_controller",
                    "C 도착 완료 — OK 사인으로 컨트롤러 모드로 전환한 뒤 D로 이동하세요.",
                    "D")
        else:
            self._checkpoints["D"] = "도착 완료"
            self._set_phase(
                "wait_auto",
                "D 도착 완료 — OK 사인을 1.5초 유지해 수동 모드를 끄면 Nav2가 A로 복귀합니다.",
                "A")

    def _start_nav_to_b(self):
        self._publish_mode_command("auto_lock")
        self._checkpoints["B"] = "Nav2 이동 중"
        self._set_phase("auto_to_b", "Nav2 자율주행으로 A에서 B로 이동 중", "B")
        self._send_nav_goal("B")

    def _maybe_start_return_to_a(self):
        # D에서 OK 사인을 1.5초 유지하면 bridge가 gesture/joystick 둘 다 False로
        # 발행한다. 이 AUTO/Nav2 전환은 사용자의 명시적 D 도착 확인이므로 좌표
        # 검증 없이 D를 완료한 뒤 A 목표를 전송한다.
        if (self._phase == "controller_to_d" and not self._gesture_mode
                and not self._joystick_mode):
            self.get_logger().info(
                "AUTO/Nav2 전환을 D 도착 확인으로 수신 — 좌표 검증 없이 D 완료 처리")
            self._complete_manual_arrival("D")
        if (self._phase == "wait_auto" and not self._gesture_mode
                and not self._joystick_mode):
            self._checkpoints["A"] = "Nav2 복귀 중"
            self._set_phase("auto_to_a", "Nav2 자율주행으로 D에서 A로 복귀 중", "A")
            self._send_nav_goal("A")

    def _send_nav_goal(self, waypoint_name):
        if self._active_goal_name is not None or self._pending_goal_name is not None:
            active_or_pending = self._active_goal_name or self._pending_goal_name
            self._halt(f"이미 {active_or_pending} Nav2 목표가 활성화되어 있어 새 목표를 보내지 않았습니다.")
            return
        self._pending_goal_name = waypoint_name
        self._goal_accept_deadline_sec = time.monotonic() + NAV_GOAL_ACCEPT_TIMEOUT_SEC
        self._next_goal_attempt_sec = 0.0
        self._attempt_pending_nav_goal()

    def _retry_pending_nav_goal(self):
        """Nav2 lifecycle 활성화 전에 거부된 목표를 제한 시간 동안 재시도한다."""
        waypoint_name = self._pending_goal_name
        if waypoint_name is None:
            return
        now = time.monotonic()
        if self._active_goal_name is not None:
            if (self._goal_response_deadline_sec is not None
                    and now >= self._goal_response_deadline_sec):
                # 응답을 못 받은 목표가 서버에서 수락됐는지는 알 수 없다.
                # 중복 목표를 재전송하지 말고 mux hold로 안전정지한다.
                self._goal_response_future = None
                self._goal_response_deadline_sec = None
                self._active_goal_name = None
                self._clear_pending_goal()
                self._halt(
                    f"Nav2 {waypoint_name} 목표 응답을 "
                    f"{NAV_GOAL_RESPONSE_TIMEOUT_SEC:.0f}초 안에 받지 못했습니다")
            return
        if now >= self._goal_accept_deadline_sec:
            self._clear_pending_goal()
            self._halt(f"Nav2가 {waypoint_name} 목표를 {NAV_GOAL_ACCEPT_TIMEOUT_SEC:.0f}초 안에 수락하지 않았습니다.")
            return
        if now >= self._next_goal_attempt_sec:
            self._attempt_pending_nav_goal()

    def _attempt_pending_nav_goal(self):
        waypoint_name = self._pending_goal_name
        if waypoint_name is None or self._active_goal_name is not None:
            return
        x, y, yaw = WAYPOINTS[waypoint_name]
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z, pose.pose.orientation.w = _yaw_to_quaternion(yaw)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._active_goal_name = waypoint_name
        future = self._nav_client.send_goal_async(goal)
        self._goal_response_future = future
        self._goal_response_deadline_sec = (
            time.monotonic() + NAV_GOAL_RESPONSE_TIMEOUT_SEC)
        future.add_done_callback(
            lambda response, name=waypoint_name: self._on_goal_response(name, response))

    def _clear_pending_goal(self):
        self._pending_goal_name = None
        self._goal_accept_deadline_sec = None
        self._next_goal_attempt_sec = None

    def _on_goal_response(self, waypoint_name, future):
        if not claim_goal_response(self, future):
            return
        try:
            handle = future.result()
        except Exception as exc:  # rclpy action transport failure
            self._active_goal_name = None
            self._next_goal_attempt_sec = time.monotonic() + NAV_GOAL_RETRY_SEC
            self.get_logger().warning(
                f"{waypoint_name} Nav2 목표 전송 실패 — 재시도 대기: {exc}")
            return
        if not handle.accepted:
            self._active_goal_name = None
            self._next_goal_attempt_sec = time.monotonic() + NAV_GOAL_RETRY_SEC
            self.get_logger().info(
                f"Nav2가 {waypoint_name} 목표를 아직 수락하지 않았습니다 — "
                f"{NAV_GOAL_RETRY_SEC:.1f}초 뒤 재시도")
            return
        self._clear_pending_goal()
        self.get_logger().info(f"Nav2 목표 수락: {waypoint_name}")
        handle.get_result_async().add_done_callback(
            lambda result, name=waypoint_name: self._on_nav_result(name, result))

    def _on_nav_result(self, waypoint_name, future):
        self._active_goal_name = None
        try:
            result = future.result()
        except Exception as exc:
            self._halt(f"{waypoint_name} Nav2 결과 수신 실패: {exc}")
            return
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self._halt(f"Nav2가 {waypoint_name} 도착에 실패 또는 취소되었습니다 (status={result.status}).")
            return
        if waypoint_name == "B" and self._phase == "auto_to_b":
            self._checkpoints["B"] = "도착 완료"
            self._set_phase(
                "wait_gesture",
                "B 도착 완료 — 짧은 따봉으로 제스처 모드를 켜세요.",
                "C")
            # A->B 자동 잠금만 풀고 AUTO 상태로 멈춘다. 제스처
            # 모드는 bridge가 사용자의 새 따봉을 인식했을 때만 켜진다.
            self._publish_mode_command("auto")
        elif waypoint_name == "A" and self._phase == "auto_to_a":
            self._checkpoints["A"] = "복귀 완료"
            self._set_phase(
                "complete", "A 복귀 완료 — A-B-C-D-A 미션이 끝났습니다.", "-")
        else:
            self._halt(f"현재 미션 단계와 맞지 않는 {waypoint_name} Nav2 완료 결과입니다.")

    def _publish_mode_command(self, command):
        message = String()
        message.data = command
        self._mode_command_publisher.publish(message)
        self.get_logger().info(f"미션 제어권 요청: {command}")

    def _set_phase(self, phase, label, target):
        self._phase = phase
        self._label = label
        self._target = target
        self._arrival_started_sec = None
        self.get_logger().info(label)
        self._post_status(force=True)

    def _halt(self, reason):
        self._phase = "halted"
        self._label = f"미션 정지: {reason}"
        self._target = "-"
        self._arrival_started_sec = None
        self.get_logger().error(self._label)
        # 오류 상태에서는 mux가 0 Twist만 내보내도록 명시적으로 잠근다.
        self._publish_mode_command("hold")
        self._post_status(force=True)

    def _post_status(self, force=False):
        now = time.monotonic()
        position = (None if self._position is None else {
            "x": round(self._position[0], 4), "y": round(self._position[1], 4)})
        payload = {
            "phase": self._phase,
            "label": self._label,
            "target": self._target,
            "checkpoints": self._checkpoints,
            "position": position,
        }
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if not force and now - self._last_status_post_sec < STATUS_POST_INTERVAL_SEC:
            return
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urlrequest.Request(
            WEB_STATUS_URL, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(request, timeout=0.15):
                pass
        except (OSError, urlerror.URLError):
            self.get_logger().warning(
                "웹 미션 상태 보고 실패", throttle_duration_sec=3.0)
        else:
            self._last_status_post_sec = now
            self._last_status_signature = signature


def main():
    rclpy.init()
    mission = WaypointHandoffMission()
    try:
        mission.run()
    except KeyboardInterrupt:
        mission.get_logger().info("사용자 요청으로 구간별 미션을 종료합니다.")
    finally:
        mission.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
