#!/usr/bin/env python3
"""Safety supervisor for the integrated A-B-C-D-A demonstration.

This node never owns a Nav2 goal and never publishes to Nav2's /cmd_vel.
It detects a temporally new obstacle, asks the waypoint mission to cancel its
goal, drives a straight reverse escape through /cmd_vel_safety, and asks the
mission to resend the interrupted goal.

The 15 cm protective stop is deliberately separate from the surprise-obstacle
recovery. During autonomous segments a close object stops the base, but the
reverse escape is only allowed when map-filtered scan history says that the
object appeared suddenly while the robot was moving forward.
"""

import math
import time
from enum import Enum, auto

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


AUTO_PHASES = frozenset(("auto_to_b", "auto_to_a"))


class SafetyState(Enum):
    MONITORING = auto()
    CANDIDATE = auto()
    PROTECTIVE_STOP = auto()
    WAIT_NAV_CANCEL = auto()
    REVERSE = auto()
    CLEAR_WAIT = auto()
    WAIT_NAV_RESUME = auto()
    COOLDOWN = auto()
    HALT = auto()


def normalize_angle(angle):
    """Return an angle in [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def backward_progress(start_x, start_y, start_yaw, current_x, current_y):
    """Return signed retreat from the obstacle-event pose along its heading."""
    delta_x = float(current_x) - float(start_x)
    delta_y = float(current_y) - float(start_y)
    return -(
        delta_x * math.cos(float(start_yaw))
        + delta_y * math.sin(float(start_yaw))
    )


def yaw_from_quaternion(quaternion):
    """Extract planar yaw from a geometry_msgs quaternion."""
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def points_in_sector(scan, center_angle, half_angle, max_range):
    """Return finite angle/distance points in a LaserScan sector."""
    points = []
    angle = scan.angle_min
    usable_max = min(float(max_range), float(scan.range_max))
    for distance in scan.ranges:
        if (math.isfinite(distance)
                and scan.range_min <= distance <= usable_max
                and abs(normalize_angle(angle - center_angle)) <= half_angle):
            points.append((angle, float(distance)))
        angle += scan.angle_increment
    return points


def points_in_odom(scan_points, odom_x, odom_y, odom_yaw):
    """Project laser points into odom coordinates to compensate base motion."""
    cosine = math.cos(odom_yaw)
    sine = math.sin(odom_yaw)
    projected = []
    for angle, distance in scan_points:
        local_x = distance * math.cos(angle)
        local_y = distance * math.sin(angle)
        projected.append((
            odom_x + cosine * local_x - sine * local_y,
            odom_y + sine * local_x + cosine * local_y,
        ))
    return projected


def count_novel_points(current_points, previous_points, match_distance):
    """Count current odom-frame points not explained by the prior scan."""
    match_distance_sq = float(match_distance) ** 2
    novel = 0
    for current_x, current_y in current_points:
        if not any(
                (current_x - previous_x) ** 2 + (current_y - previous_y) ** 2
                <= match_distance_sq
                for previous_x, previous_y in previous_points):
            novel += 1
    return novel


def count_matching_points(current_points, reference_points, match_distance):
    """Count current map points belonging to a captured obstacle cluster."""
    match_distance_sq = float(match_distance) ** 2
    matching = 0
    for current_x, current_y in current_points:
        if any(
                (current_x - reference_x) ** 2
                + (current_y - reference_y) ** 2
                <= match_distance_sq
                for reference_x, reference_y in reference_points):
            matching += 1
    return matching


def map_point_is_static(
        occupancy_grid, point_x, point_y, margin, occupied_threshold):
    """Return True for mapped walls, unknown cells, and out-of-map points."""
    info = occupancy_grid.info
    resolution = float(info.resolution)
    width = int(info.width)
    height = int(info.height)
    if resolution <= 0.0 or width <= 0 or height <= 0:
        return True

    origin = info.origin
    origin_yaw = yaw_from_quaternion(origin.orientation)
    delta_x = point_x - float(origin.position.x)
    delta_y = point_y - float(origin.position.y)
    cosine = math.cos(origin_yaw)
    sine = math.sin(origin_yaw)
    map_x = cosine * delta_x + sine * delta_y
    map_y = -sine * delta_x + cosine * delta_y
    cell_x = math.floor(map_x / resolution)
    cell_y = math.floor(map_y / resolution)
    if cell_x < 0 or cell_x >= width or cell_y < 0 or cell_y >= height:
        return True

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


def dynamic_scan_points(
        scan_points, sensor_x, sensor_y, sensor_yaw, occupancy_grid,
        static_margin, occupied_threshold):
    """Remove scan endpoints explained by the static occupancy map.

    sensor_x/y/yaw must be the map->scan-frame transform at the exact
    LaserScan timestamp. Using the latest AMCL message here is unsafe while
    the base turns because a mapped wall can be projected into a free cell.
    """
    world_points = points_in_odom(
        scan_points, sensor_x, sensor_y, sensor_yaw)
    return [
        scan_point
        for scan_point, (point_x, point_y) in zip(scan_points, world_points)
        if not map_point_is_static(
            occupancy_grid, point_x, point_y,
            static_margin, occupied_threshold)
    ]


def is_sudden_obstacle_seed(
        have_baseline, previous_front_distance, current_front_distance,
        close_point_count, novel_point_count, forward_speed, angular_speed,
        trigger_distance, minimum_distance_drop, minimum_points,
        minimum_forward_speed, maximum_angular_speed):
    """Pure first-frame classifier used by runtime code and unit tests."""
    return bool(
        have_baseline
        and current_front_distance <= trigger_distance
        and previous_front_distance - current_front_distance
        >= minimum_distance_drop
        and close_point_count >= minimum_points
        and novel_point_count >= minimum_points
        and forward_speed >= minimum_forward_speed
        and abs(angular_speed) <= maximum_angular_speed
    )


def is_new_dynamic_cluster_seed(
        have_baseline, previous_dynamic_point_count,
        current_dynamic_point_count, current_front_distance,
        forward_speed, angular_speed, trigger_distance, minimum_points,
        minimum_forward_speed, maximum_angular_speed):
    """Detect the onset of an unmapped cluster even without a range jump.

    A wide plate can enter several adjacent laser rays at nearly the same
    nearest distance as an already visible side feature.  In that case the
    nearest-range and point-novelty tests can both miss the event.  The static
    occupancy-map filter gives us a safer second signal: a transition from no
    dynamic cluster to a full cluster while Nav2 is moving straight ahead.
    """
    return bool(
        have_baseline
        and previous_dynamic_point_count < minimum_points
        and current_dynamic_point_count >= minimum_points
        and current_front_distance <= trigger_distance
        and forward_speed >= minimum_forward_speed
        and abs(angular_speed) <= maximum_angular_speed
    )


class SafetyMissionManager(Node):
    """Detect, stop, recover, and coordinate Nav2 cancel/resume."""

    def __init__(self):
        super().__init__("turtlebot3_safety_mission_manager")
        self._declare_parameters()
        self._load_parameters()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._cmd_publisher = self.create_publisher(
            Twist, self.cmd_vel_topic, 10)
        self._active_publisher = self.create_publisher(
            Bool, self.active_topic, latched_qos)
        self._heartbeat_publisher = self.create_publisher(
            Bool, self.heartbeat_topic, 10)
        self._request_publisher = self.create_publisher(
            String, self.request_topic, 10)
        self._state_publisher = self.create_publisher(
            String, self.state_topic, latched_qos)

        self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            Twist, self.nav_cmd_vel_topic, self._nav_cmd_callback, 10)
        self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_callback, latched_qos)
        self.create_subscription(
            String, self.mission_phase_topic, self._phase_callback,
            latched_qos)
        self.create_subscription(
            String, self.nav_status_topic, self._nav_status_callback, 10)

        self.state = SafetyState.MONITORING
        self._state_started_sec = time.monotonic()
        self._active = None
        self._last_scan_sec = None
        self._last_odom_sec = None
        self._last_phase_sec = None
        self._ever_received_phase = False
        self._mission_phase = "unknown"
        self._nav_status = "unknown"

        self._odom_x = None
        self._odom_y = None
        self._odom_yaw = None
        self._forward_speed = 0.0
        self._angular_speed = 0.0
        self._last_forward_intent_sec = None
        self._last_forward_intent_speed = 0.0
        self._last_candidate_diagnostic_sec = 0.0
        self._last_tf_diagnostic_sec = 0.0
        self._occupancy_grid = None
        self._previous_front_distance = self.reference_range
        self._previous_dynamic_points_map = []
        self._previous_dynamic_close_point_count = 0
        self._have_scan_baseline = False
        self._candidate_frames = 0
        self._candidate_started_sec = None
        self._protective_clear_frames = 0
        self._rear_blocked = False
        self._front_clear = False
        self._front_clear_started_sec = None

        self._recovery_origin = None
        self._recovery_obstacle_points_map = []

        self.create_timer(0.1, self._publish_heartbeat)
        self.create_timer(0.05, self._control_tick)
        self._publish_active(True)
        self._publish_state()
        self.get_logger().info(
            "Safety supervisor 시작 — 기본 속도 경로와 분리된 "
            "/cmd_vel_safety만 사용합니다.")

    def _declare_parameters(self):
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("nav_cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_safety")
        self.declare_parameter("active_topic", "/safety/active")
        self.declare_parameter("heartbeat_topic", "/safety/heartbeat")
        self.declare_parameter("request_topic", "/safety/request")
        self.declare_parameter("state_topic", "/safety/state")
        self.declare_parameter("mission_phase_topic", "/safety/mission_phase")
        self.declare_parameter("nav_status_topic", "/safety/nav_status")

        self.declare_parameter("front_angle_deg", 15.0)
        self.declare_parameter("protective_angle_deg", 30.0)
        self.declare_parameter("reference_range", 0.90)
        self.declare_parameter("sudden_trigger_distance", 0.35)
        self.declare_parameter("hard_stop_distance", 0.15)
        self.declare_parameter("hard_release_distance", 0.20)
        self.declare_parameter("sudden_distance_drop", 0.10)
        self.declare_parameter("novelty_match_distance", 0.06)
        self.declare_parameter("static_map_margin", 0.15)
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("min_obstacle_points", 3)
        self.declare_parameter("min_surprise_points", 8)
        self.declare_parameter("consecutive_scan_frames", 2)
        self.declare_parameter("candidate_timeout_sec", 0.40)
        self.declare_parameter("minimum_forward_speed", 0.06)
        self.declare_parameter("maximum_angular_speed", 0.30)
        self.declare_parameter("forward_intent_hold_sec", 1.00)
        self.declare_parameter("diagnostic_log_interval_sec", 0.75)
        self.declare_parameter("scan_tf_wait_sec", 0.08)
        self.declare_parameter("protective_clear_frames", 3)
        self.declare_parameter("sensor_timeout_sec", 0.50)
        self.declare_parameter("mission_context_timeout_sec", 1.00)

        self.declare_parameter("cancel_timeout_sec", 3.0)
        self.declare_parameter("escape_speed", 0.10)
        self.declare_parameter("escape_distance", 0.25)
        self.declare_parameter("escape_timeout_sec", 6.0)
        self.declare_parameter("rear_angle_deg", 35.0)
        self.declare_parameter("rear_stop_distance", 0.25)
        self.declare_parameter("resume_angle_deg", 15.0)
        self.declare_parameter("resume_clear_distance", 0.75)
        self.declare_parameter("clear_match_distance", 0.08)
        self.declare_parameter("clear_stable_sec", 1.0)
        self.declare_parameter("clear_wait_timeout_sec", 0.0)
        self.declare_parameter("resume_timeout_sec", 10.0)
        self.declare_parameter("resume_cooldown_sec", 1.0)

    def _load_parameters(self):
        value = lambda name: self.get_parameter(name).value
        self.scan_topic = value("scan_topic")
        self.odom_topic = value("odom_topic")
        self.nav_cmd_vel_topic = value("nav_cmd_vel_topic")
        self.map_topic = value("map_topic")
        self.map_frame = str(value("map_frame"))
        self.cmd_vel_topic = value("cmd_vel_topic")
        self.active_topic = value("active_topic")
        self.heartbeat_topic = value("heartbeat_topic")
        self.request_topic = value("request_topic")
        self.state_topic = value("state_topic")
        self.mission_phase_topic = value("mission_phase_topic")
        self.nav_status_topic = value("nav_status_topic")

        self.front_angle = math.radians(float(value("front_angle_deg")))
        self.protective_angle = math.radians(
            float(value("protective_angle_deg")))
        self.reference_range = float(value("reference_range"))
        self.sudden_trigger_distance = float(value("sudden_trigger_distance"))
        self.hard_stop_distance = float(value("hard_stop_distance"))
        self.hard_release_distance = float(value("hard_release_distance"))
        self.sudden_distance_drop = float(value("sudden_distance_drop"))
        self.novelty_match_distance = float(value("novelty_match_distance"))
        self.static_map_margin = float(value("static_map_margin"))
        self.occupied_threshold = int(value("occupied_threshold"))
        self.min_obstacle_points = int(value("min_obstacle_points"))
        self.min_surprise_points = int(value("min_surprise_points"))
        self.consecutive_scan_frames = int(value("consecutive_scan_frames"))
        self.candidate_timeout_sec = float(value("candidate_timeout_sec"))
        self.minimum_forward_speed = float(value("minimum_forward_speed"))
        self.maximum_angular_speed = float(value("maximum_angular_speed"))
        self.forward_intent_hold_sec = float(value("forward_intent_hold_sec"))
        self.diagnostic_log_interval_sec = float(
            value("diagnostic_log_interval_sec"))
        self.scan_tf_wait_sec = float(value("scan_tf_wait_sec"))
        self.required_protective_clear_frames = int(
            value("protective_clear_frames"))
        self.sensor_timeout_sec = float(value("sensor_timeout_sec"))
        self.mission_context_timeout_sec = float(
            value("mission_context_timeout_sec"))

        self.cancel_timeout_sec = float(value("cancel_timeout_sec"))
        self.escape_speed = float(value("escape_speed"))
        self.escape_distance = float(value("escape_distance"))
        self.escape_timeout_sec = float(value("escape_timeout_sec"))
        self.rear_angle = math.radians(float(value("rear_angle_deg")))
        self.rear_stop_distance = float(value("rear_stop_distance"))
        self.resume_angle = math.radians(float(value("resume_angle_deg")))
        self.resume_clear_distance = float(value("resume_clear_distance"))
        self.clear_match_distance = float(value("clear_match_distance"))
        self.clear_stable_sec = float(value("clear_stable_sec"))
        self.clear_wait_timeout_sec = float(value("clear_wait_timeout_sec"))
        self.resume_timeout_sec = float(value("resume_timeout_sec"))
        self.resume_cooldown_sec = float(value("resume_cooldown_sec"))

        distances = (
            self.hard_stop_distance,
            self.hard_release_distance,
            self.sudden_trigger_distance,
            self.reference_range,
        )
        if not (0.0 < distances[0] < distances[1]
                <= distances[2] < distances[3]):
            raise ValueError(
                "거리 설정은 0 < hard_stop < hard_release <= "
                "sudden_trigger < reference_range 순서여야 합니다.")
        if (self.min_obstacle_points < 1
                or self.min_surprise_points < self.min_obstacle_points
                or self.consecutive_scan_frames < 2):
            raise ValueError(
                "min_surprise_points >= min_obstacle_points >= 1, "
                "consecutive_scan_frames >= 2여야 합니다.")
        if self.escape_speed <= 0.0 or self.escape_distance <= 0.0:
            raise ValueError("escape_speed와 escape_distance는 양수여야 합니다.")
        if self.forward_intent_hold_sec <= 0.0:
            raise ValueError("forward_intent_hold_sec는 양수여야 합니다.")
        if self.static_map_margin < 0.0:
            raise ValueError("static_map_margin은 0 이상이어야 합니다.")
        if not self.map_frame:
            raise ValueError("map_frame은 비어 있을 수 없습니다.")
        if self.scan_tf_wait_sec < 0.0:
            raise ValueError("scan_tf_wait_sec는 0 이상이어야 합니다.")
        if not 0.0 < self.front_angle <= self.protective_angle <= math.pi:
            raise ValueError(
                "각도 설정은 0 < front_angle_deg <= "
                "protective_angle_deg <= 180 순서여야 합니다.")
        if not 0.0 < self.resume_angle <= self.front_angle:
            raise ValueError(
                "resume_angle_deg는 0보다 크고 front_angle_deg 이하여야 합니다.")
        if self.clear_match_distance <= 0.0:
            raise ValueError("clear_match_distance는 양수여야 합니다.")
        if not 0 <= self.occupied_threshold <= 100:
            raise ValueError("occupied_threshold는 0~100 범위여야 합니다.")

    def _phase_callback(self, message):
        previous_phase = self._mission_phase
        self._mission_phase = message.data.strip()
        self._last_phase_sec = time.monotonic()
        self._ever_received_phase = True
        if self._mission_phase not in AUTO_PHASES:
            self._disable_for_manual_phase()
        elif previous_phase not in AUTO_PHASES:
            # 수동 주행 scan을 갑툭튀 기준선으로 가져오지 않는다.
            self._reset_detection_history()

    def _reset_detection_history(self):
        self._reset_candidate()
        self._have_scan_baseline = False
        self._previous_front_distance = self.reference_range
        self._previous_dynamic_points_map = []
        self._previous_dynamic_close_point_count = 0
        self._recovery_obstacle_points_map = []

    def _disable_for_manual_phase(self):
        """Remove every Safety velocity intervention outside autonomous legs."""
        self._reset_detection_history()
        self._protective_clear_frames = 0
        self._front_clear_started_sec = None
        self._recovery_origin = None
        self._recovery_obstacle_points_map = []
        if self.state != SafetyState.MONITORING:
            self._set_state(
                SafetyState.MONITORING,
                "수동 구간 진입 — Safety 속도 개입 비활성")
        self._publish_active(False)

    def _nav_status_callback(self, message):
        status = message.data.strip()
        self._nav_status = status
        if self.state == SafetyState.WAIT_NAV_CANCEL:
            if status == "canceled":
                self._begin_reverse()
            elif status in ("cancel_failed", "failed"):
                self._halt(f"Nav2 취소 실패 응답: {status}")
        elif self.state == SafetyState.WAIT_NAV_RESUME:
            if status == "resumed":
                self._set_state(SafetyState.COOLDOWN, "Nav2 목표 재전송 확인")
                self._publish_active(False)
            elif status in ("resume_failed", "failed"):
                self._halt(f"Nav2 재개 실패 응답: {status}")

    def _odom_callback(self, message):
        pose = message.pose.pose
        self._odom_x = float(pose.position.x)
        self._odom_y = float(pose.position.y)
        self._odom_yaw = yaw_from_quaternion(pose.orientation)
        self._forward_speed = float(message.twist.twist.linear.x)
        self._angular_speed = float(message.twist.twist.angular.z)
        self._last_odom_sec = time.monotonic()

    def _nav_cmd_callback(self, message):
        """Latch recent straight-ahead Nav2 intent across its emergency brake.

        Nav2 can publish a zero command immediately after a new obstacle reaches
        its local costmap.  The LiDAR callback often runs just after that zero,
        so using only measured odometry speed makes the surprise detector miss
        the exact event it is meant to handle. A non-forward command clears the
        latch; a zero command merely lets it expire shortly afterward.
        """
        linear_x = float(message.linear.x)
        angular_z = float(message.angular.z)
        if (linear_x >= self.minimum_forward_speed
                and abs(angular_z) <= self.maximum_angular_speed):
            self._last_forward_intent_sec = time.monotonic()
            self._last_forward_intent_speed = linear_x
        elif (linear_x < -self.minimum_forward_speed
              or abs(angular_z) > self.maximum_angular_speed):
            self._last_forward_intent_sec = None
            self._last_forward_intent_speed = 0.0

    def _effective_forward_intent(self, now):
        intent = max(0.0, self._forward_speed)
        if (self._last_forward_intent_sec is not None
                and now - self._last_forward_intent_sec
                <= self.forward_intent_hold_sec):
            intent = max(intent, self._last_forward_intent_speed)
        return intent

    def _map_callback(self, message):
        self._occupancy_grid = message

    def _lookup_scan_pose_in_map(self, message):
        """Return the exact map->LaserScan-frame planar transform, or None."""
        scan_frame = message.header.frame_id.strip()
        if not scan_frame:
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                self.map_frame,
                scan_frame,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=self.scan_tf_wait_sec),
            )
        except TransformException as exc:
            now = time.monotonic()
            if (now - self._last_tf_diagnostic_sec
                    >= self.diagnostic_log_interval_sec):
                self._last_tf_diagnostic_sec = now
                self.get_logger().warning(
                    "[SAFETY 진단] scan 시각의 지도 TF가 없어 자동 후진 "
                    f"판정을 건너뜁니다({self.map_frame}<-{scan_frame}): {exc}")
            return None

        translation = transform.transform.translation
        return (
            float(translation.x),
            float(translation.y),
            yaw_from_quaternion(transform.transform.rotation),
        )

    def _scan_callback(self, message):
        now = time.monotonic()
        self._last_scan_sec = now
        # 제스처·조이스틱·장갑 구간은 운전자 책임의 수동 주행이다.
        # LiDAR 진단을 포함해 Safety 판단 자체를 수행하지 않는다.
        if self._mission_phase not in AUTO_PHASES:
            return
        if self._odom_x is None or self._odom_yaw is None:
            return

        # 충돌 직전 보호와 후진 중 후방 감지는 TF 대기보다 먼저 처리한다.
        # 장애물 제거 확인만 감지 당시 map 군집과 비교하기 위해 TF를 사용한다.
        protective_points = points_in_sector(
            message, 0.0, self.protective_angle, self.hard_release_distance)
        hard_points = [
            point for point in protective_points
            if point[1] <= self.hard_stop_distance]
        release_points = [
            point for point in protective_points
            if point[1] <= self.hard_release_distance]
        hard_detected = len(hard_points) >= self.min_obstacle_points
        if hard_detected and self.state in (
                SafetyState.MONITORING,
                SafetyState.CANDIDATE,
                SafetyState.PROTECTIVE_STOP):
            self._protective_clear_frames = 0
            if self.state != SafetyState.PROTECTIVE_STOP:
                self._set_state(
                    SafetyState.PROTECTIVE_STOP,
                    f"{self.hard_stop_distance:.2f}m 절대 보호정지")
            self._publish_active(True)
            self._publish_stop()
        elif self.state == SafetyState.PROTECTIVE_STOP:
            if len(release_points) == 0:
                self._protective_clear_frames += 1
            else:
                self._protective_clear_frames = 0

        rear_points = points_in_sector(
            message, math.pi, self.rear_angle, self.rear_stop_distance)
        self._rear_blocked = len(rear_points) >= self.min_obstacle_points
        resume_points = points_in_sector(
            message, 0.0, self.resume_angle, self.resume_clear_distance)
        reference_points = points_in_sector(
            message, 0.0, self.front_angle, self.reference_range)
        current_front_distance = min(
            (distance for _, distance in reference_points),
            default=self.reference_range)
        close_points = [
            point for point in reference_points
            if point[1] <= self.sudden_trigger_distance]
        scan_pose_map = None
        map_filter_context = (
            self.state in (
                SafetyState.MONITORING,
                SafetyState.CANDIDATE,
                SafetyState.PROTECTIVE_STOP,
            )
            and self._mission_phase in AUTO_PHASES
            and self._nav_status in ("navigating", "resumed")
            and self._phase_is_fresh(now)
        )
        clear_wait_map_context = (
            self.state == SafetyState.CLEAR_WAIT
            and bool(self._recovery_obstacle_points_map)
        )
        if (self._occupancy_grid is not None
                and ((close_points and map_filter_context)
                     or (resume_points and clear_wait_map_context))):
            scan_pose_map = self._lookup_scan_pose_in_map(message)

        if self.state == SafetyState.CLEAR_WAIT:
            # 후진 뒤에는 정면 0.75m 전체가 비기를 요구하지 않는다. 좁은
            # 지도에서는 원래 벽이 그 범위 안에 있어 영원히 CLEAR_WAIT에
            # 갇힐 수 있다. 감지 순간 저장한 동적 군집과 같은 map 위치의
            # 점들만 추적해, 그 임시 장애물이 사라졌는지를 확인한다.
            if not resume_points:
                self._front_clear = True
            elif not self._recovery_obstacle_points_map:
                self._front_clear = False
            elif scan_pose_map is None or self._occupancy_grid is None:
                self._front_clear = False
            else:
                dynamic_resume_points = dynamic_scan_points(
                    resume_points,
                    scan_pose_map[0],
                    scan_pose_map[1],
                    scan_pose_map[2],
                    self._occupancy_grid,
                    self.static_map_margin,
                    self.occupied_threshold,
                )
                dynamic_resume_points_map = points_in_odom(
                    dynamic_resume_points,
                    scan_pose_map[0],
                    scan_pose_map[1],
                    scan_pose_map[2],
                )
                matching_points = count_matching_points(
                    dynamic_resume_points_map,
                    self._recovery_obstacle_points_map,
                    self.clear_match_distance,
                )
                self._front_clear = (
                    matching_points < self.min_obstacle_points)

            if self._front_clear:
                if self._front_clear_started_sec is None:
                    self._front_clear_started_sec = now
            else:
                self._front_clear_started_sec = None

        map_comparison_available = bool(
            self._occupancy_grid is not None
            and (not close_points or scan_pose_map is not None)
            and map_filter_context)
        if scan_pose_map is not None and map_filter_context:
            dynamic_close_points = dynamic_scan_points(
                close_points,
                scan_pose_map[0],
                scan_pose_map[1],
                scan_pose_map[2],
                self._occupancy_grid,
                self.static_map_margin,
                self.occupied_threshold,
            )
            dynamic_close_points_map = points_in_odom(
                dynamic_close_points,
                scan_pose_map[0],
                scan_pose_map[1],
                scan_pose_map[2],
            )
        else:
            # 지도 또는 scan 시각 TF가 없으면 새 장애물이라고 추측하지 않는다.
            # 특수 자동후진만 금지하며 15 cm 절대 보호정지는 계속 동작한다.
            dynamic_close_points = []
            dynamic_close_points_map = []

        dynamic_novel_count = count_novel_points(
            dynamic_close_points_map,
            self._previous_dynamic_points_map,
            self.novelty_match_distance)
        candidate_close_points = dynamic_close_points
        candidate_novel_count = dynamic_novel_count
        candidate_source = "scan-time-map-filtered"
        candidate_front_distance = min(
            (distance for _, distance in candidate_close_points),
            default=current_front_distance)
        special_context = (
            self.state in (
                SafetyState.MONITORING,
                SafetyState.CANDIDATE,
                SafetyState.PROTECTIVE_STOP,
            )
            and self._mission_phase in AUTO_PHASES
            and self._nav_status in ("navigating", "resumed")
            and self._phase_is_fresh(now)
            and map_comparison_available
        )
        forward_intent = self._effective_forward_intent(now)
        temporal_seed = is_sudden_obstacle_seed(
            self._have_scan_baseline,
            self._previous_front_distance,
            candidate_front_distance,
            len(candidate_close_points),
            candidate_novel_count,
            forward_intent,
            self._angular_speed,
            self.sudden_trigger_distance,
            self.sudden_distance_drop,
            self.min_surprise_points,
            self.minimum_forward_speed,
            self.maximum_angular_speed,
        )
        dynamic_onset_seed = is_new_dynamic_cluster_seed(
            self._have_scan_baseline,
            self._previous_dynamic_close_point_count,
            len(dynamic_close_points),
            candidate_front_distance,
            forward_intent,
            self._angular_speed,
            self.sudden_trigger_distance,
            self.min_surprise_points,
            self.minimum_forward_speed,
            self.maximum_angular_speed,
        )
        seed = special_context and (temporal_seed or dynamic_onset_seed)
        if dynamic_onset_seed and not temporal_seed:
            candidate_source = "scan-time-map-filtered-onset"

        if seed:
            self._candidate_frames = 1
            self._candidate_started_sec = now
            if self.state == SafetyState.MONITORING:
                self._set_state(
                    SafetyState.CANDIDATE,
                    "갑작스러운 전방 장애물 후보 "
                    f"({candidate_front_distance:.2f}m, "
                    f"{len(candidate_close_points)} points, {candidate_source})")
        elif self._candidate_frames > 0:
            candidate_fresh = (
                self._candidate_started_sec is not None
                and now - self._candidate_started_sec <= self.candidate_timeout_sec)
            if (special_context and candidate_fresh
                    and len(candidate_close_points) >= self.min_surprise_points):
                self._candidate_frames += 1
            else:
                self._reset_candidate()
                if self.state == SafetyState.CANDIDATE:
                    self._set_state(SafetyState.MONITORING, "후보가 연속 확인되지 않음")

        if (not seed and self._candidate_frames == 0
                and len(close_points) >= self.min_obstacle_points
                and self.state in (
                    SafetyState.MONITORING,
                    SafetyState.CANDIDATE,
                    SafetyState.PROTECTIVE_STOP,
                )):
            self._log_candidate_rejection(
                now,
                special_context,
                current_front_distance,
                candidate_front_distance,
                len(close_points),
                len(dynamic_close_points),
                candidate_novel_count,
                forward_intent,
                candidate_source,
                map_comparison_available,
            )

        if (self._candidate_frames >= self.consecutive_scan_frames
                and special_context):
            self._begin_recovery(dynamic_close_points_map)

        if self.state == SafetyState.PROTECTIVE_STOP:
            if (self._protective_clear_frames
                    >= self.required_protective_clear_frames
                    and self._candidate_frames == 0):
                self._set_state(SafetyState.MONITORING, "보호정지 구역 해제")
                self._publish_active(False)

        self._previous_front_distance = current_front_distance
        if map_comparison_available:
            self._previous_dynamic_points_map = dynamic_close_points_map
            self._previous_dynamic_close_point_count = len(
                dynamic_close_points)
            self._have_scan_baseline = True

    def _log_candidate_rejection(
            self, now, special_context, current_front_distance,
            candidate_front_distance, raw_point_count, dynamic_point_count,
            novel_count, forward_intent, candidate_source,
            map_comparison_available):
        if (now - self._last_candidate_diagnostic_sec
                < self.diagnostic_log_interval_sec):
            return

        reasons = []
        if not special_context:
            context = []
            if self._mission_phase not in AUTO_PHASES:
                context.append(f"phase={self._mission_phase}")
            if self._nav_status not in ("navigating", "resumed"):
                context.append(f"nav={self._nav_status}")
            if not self._phase_is_fresh(now):
                context.append("phase heartbeat stale")
            if self._occupancy_grid is None:
                context.append("map missing")
            elif not map_comparison_available:
                context.append("scan-time map TF missing")
            reasons.append("자동복구 context 불충족(" + ", ".join(context) + ")")
        if not self._have_scan_baseline:
            reasons.append("이전 scan baseline 없음")
        distance_drop = self._previous_front_distance - candidate_front_distance
        if distance_drop < self.sudden_distance_drop:
            reasons.append(
                f"거리 급감 {distance_drop:.2f}m < "
                f"{self.sudden_distance_drop:.2f}m")
        candidate_point_count = dynamic_point_count
        if candidate_point_count < self.min_surprise_points:
            reasons.append(
                f"후보 points {candidate_point_count} < "
                f"{self.min_surprise_points}")
        if novel_count < self.min_surprise_points:
            reasons.append(
                f"신규 points {novel_count} < {self.min_surprise_points}")
        if forward_intent < self.minimum_forward_speed:
            reasons.append(
                f"최근 전진의도 {forward_intent:.3f}m/s < "
                f"{self.minimum_forward_speed:.3f}m/s")
        if abs(self._angular_speed) > self.maximum_angular_speed:
            reasons.append(
                f"회전속도 {self._angular_speed:.2f}rad/s > "
                f"{self.maximum_angular_speed:.2f}rad/s")
        if not reasons:
            reasons.append("연속 scan 확인 대기")

        self._last_candidate_diagnostic_sec = now
        self.get_logger().warning(
            "[SAFETY 진단] 정면 장애물은 보이지만 자동복구 후보에서 제외: "
            + "; ".join(reasons)
            + f" (raw={raw_point_count}, dynamic={dynamic_point_count}, "
              f"source={candidate_source}, nearest={current_front_distance:.2f}m)")

    def _phase_is_fresh(self, now):
        return bool(
            self._last_phase_sec is not None
            and now - self._last_phase_sec <= self.mission_context_timeout_sec)

    def _begin_recovery(self, obstacle_points_map):
        if self.state not in (
                SafetyState.MONITORING,
                SafetyState.CANDIDATE,
                SafetyState.PROTECTIVE_STOP):
            return
        if (self._odom_x is None or self._odom_y is None
                or self._odom_yaw is None):
            self._halt("복구 기준 위치를 저장할 odometry가 없습니다")
            return
        # Measure the straight escape from the live obstacle-event pose so the
        # requested distance is independent of odometry's absolute origin.
        self._recovery_origin = (
            self._odom_x,
            self._odom_y,
            self._odom_yaw,
        )
        self._recovery_obstacle_points_map = list(obstacle_points_map)
        if (len(self._recovery_obstacle_points_map)
                < self.min_surprise_points):
            self._halt("복구 대상 장애물의 지도상 군집을 저장할 수 없습니다")
            return
        self._reset_candidate()
        self._publish_active(True)
        self._publish_stop()
        self._publish_request("cancel")
        self._set_state(
            SafetyState.WAIT_NAV_CANCEL,
            "갑작스러운 장애물 확정, Nav2 목표 취소 요청")

    def _control_tick(self):
        now = time.monotonic()
        if self._mission_phase not in AUTO_PHASES:
            self._disable_for_manual_phase()
            return
        data_fresh = self._sensor_data_is_fresh(now)
        if not data_fresh:
            self._publish_active(True)
            self._publish_stop()
            if self.state in (
                    SafetyState.WAIT_NAV_CANCEL,
                    SafetyState.REVERSE,
                    SafetyState.CLEAR_WAIT,
                    SafetyState.WAIT_NAV_RESUME):
                self._halt("복구 중 LiDAR 또는 odometry 데이터가 끊겼습니다")
            return

        if (self._ever_received_phase
                and not self._phase_is_fresh(now)
                and self._nav_status in ("navigating", "resumed")):
            self._halt("자율주행 중 Safety 미션 상태 heartbeat가 끊겼습니다")
            return

        if self.state == SafetyState.MONITORING:
            self._publish_active(False)
        elif self.state == SafetyState.CANDIDATE:
            self._publish_active(False)
            if (self._candidate_started_sec is not None
                    and now - self._candidate_started_sec > self.candidate_timeout_sec):
                self._reset_candidate()
                self._set_state(SafetyState.MONITORING, "장애물 후보 시간 만료")
        elif self.state == SafetyState.PROTECTIVE_STOP:
            self._publish_active(True)
            self._publish_stop()
        elif self.state == SafetyState.WAIT_NAV_CANCEL:
            self._publish_active(True)
            self._publish_stop()
            if self._elapsed() > self.cancel_timeout_sec:
                self._halt("Nav2 목표 취소 확인 시간 초과")
        elif self.state == SafetyState.REVERSE:
            self._tick_reverse()
        elif self.state == SafetyState.CLEAR_WAIT:
            self._publish_active(True)
            self._publish_stop()
            # A non-positive timeout intentionally means "wait until the
            # operator removes the temporary obstacle".  The robot remains
            # stopped with Safety ownership while waiting; sensor/mission
            # heartbeat failures are still handled by the fail-safe checks
            # above.  This avoids latching HALT merely because a human took
            # longer than expected to clear the demonstration obstacle.
            if (self.clear_wait_timeout_sec > 0.0
                    and self._elapsed() > self.clear_wait_timeout_sec):
                self._halt("복구 후 전방 안전거리 확인 시간 초과")
            elif (self._front_clear_started_sec is not None
                  and now - self._front_clear_started_sec >= self.clear_stable_sec):
                self._publish_request("resume")
                self._set_state(
                    SafetyState.WAIT_NAV_RESUME, "중단된 Nav2 목표 재전송 요청")
        elif self.state == SafetyState.WAIT_NAV_RESUME:
            self._publish_active(True)
            self._publish_stop()
            if self._elapsed() > self.resume_timeout_sec:
                self._halt("Nav2 목표 재전송 확인 시간 초과")
        elif self.state == SafetyState.COOLDOWN:
            self._publish_active(False)
            if self._elapsed() >= self.resume_cooldown_sec:
                self._have_scan_baseline = False
                self._recovery_obstacle_points_map = []
                self._set_state(SafetyState.MONITORING, "Safety 복구 완료")
        elif self.state == SafetyState.HALT:
            self._publish_active(True)
            self._publish_stop()

    def _sensor_data_is_fresh(self, now):
        return bool(
            self._last_scan_sec is not None
            and self._last_odom_sec is not None
            and now - self._last_scan_sec <= self.sensor_timeout_sec
            and now - self._last_odom_sec <= self.sensor_timeout_sec)

    def _begin_reverse(self):
        if self._rear_blocked:
            self._halt("후진 시작 전 후방 장애물이 감지되었습니다")
            return
        if self._recovery_origin is None:
            self._halt("장애물 감지 순간의 복구 기준 위치가 없습니다")
            return
        self._set_state(
            SafetyState.REVERSE,
            "Nav2 취소 및 후방 확인 완료, 직선 안전 후진")

    def _tick_reverse(self):
        self._publish_active(True)
        if self._elapsed() > self.escape_timeout_sec:
            self._halt("후진 거리 완료 전 제한시간 초과")
            return
        if self._rear_blocked:
            self._halt("후진 중 후방 장애물이 감지되었습니다")
            return
        if (self._recovery_origin is None
                or self._odom_x is None or self._odom_y is None):
            self._halt("후진 거리를 계산할 odometry가 없습니다")
            return
        start_x, start_y, start_yaw = self._recovery_origin
        net_retreat = backward_progress(
            start_x,
            start_y,
            start_yaw,
            self._odom_x,
            self._odom_y,
        )
        if net_retreat >= self.escape_distance:
            self._publish_stop()
            self._front_clear_started_sec = None
            self._set_state(
                SafetyState.CLEAR_WAIT,
                f"감지 위치 기준 {net_retreat:.2f}m 후진 완료, 전방 안정 확인")
            return
        self._publish_velocity(linear_x=-self.escape_speed)

    def _reset_candidate(self):
        self._candidate_frames = 0
        self._candidate_started_sec = None

    def _elapsed(self):
        return time.monotonic() - self._state_started_sec

    def _set_state(self, new_state, detail=""):
        previous = self.state
        self.state = new_state
        self._state_started_sec = time.monotonic()
        self._publish_state()
        suffix = f" — {detail}" if detail else ""
        self.get_logger().info(
            f"[SAFETY] {previous.name} -> {new_state.name}{suffix}")

    def _halt(self, reason):
        if self.state != SafetyState.HALT:
            self._set_state(SafetyState.HALT, reason)
            self.get_logger().error(f"[SAFETY HALT] {reason}")
        self._publish_active(True)
        self._publish_stop()

    def _publish_velocity(self, linear_x=0.0):
        message = Twist()
        message.linear.x = float(linear_x)
        self._cmd_publisher.publish(message)

    def _publish_stop(self):
        self._publish_velocity()

    def _publish_active(self, active):
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        message = Bool()
        message.data = active
        self._active_publisher.publish(message)
        self.get_logger().info(
            f"Safety 속도 우선권: {'ON' if active else 'OFF'}")

    def _publish_heartbeat(self):
        message = Bool()
        message.data = True
        self._heartbeat_publisher.publish(message)

    def _publish_request(self, request):
        message = String()
        message.data = request
        self._request_publisher.publish(message)
        self.get_logger().warning(f"Safety 미션 요청: {request}")

    def _publish_state(self):
        message = String()
        message.data = self.state.name.lower()
        self._state_publisher.publish(message)


def main():
    rclpy.init()
    node = SafetyMissionManager()
    # scan callback이 정확한 시각의 TF를 잠깐 기다리는 동안에도 TF listener
    # callback은 다른 실행 스레드에서 Buffer를 계속 채워야 한다.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # ros2 launch가 SIGINT로 context를 먼저 종료한 경우 publisher를 다시
        # 호출하면 종료 시점에 불필요한 RCLError traceback이 발생한다.
        if rclpy.ok():
            node._publish_active(True)
            node._publish_stop()
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
