#!/usr/bin/env python3
"""Safety 연동 A-B-C-D-A 미션.

기존 WaypointHandoffMission을 상속해 제스처·조이스틱·장갑 전환은 그대로
사용한다. 이 파생 노드만 Safety 요청에 따라 자신이 소유한 Nav2 목표를
취소하고, 복구 완료 뒤 같은 목표를 재전송한다.
"""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Path
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.executors import ExternalShutdownException
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from ros2_bridge.waypoint_handoff_mission import (
        ARRIVAL_RADIUS_M,
        NAV_GOAL_ACCEPT_TIMEOUT_SEC,
        NAV_GOAL_RETRY_SEC,
        WAYPOINTS,
        WaypointHandoffMission,
        claim_goal_response,
    )
except ImportError:
    from waypoint_handoff_mission import (
        ARRIVAL_RADIUS_M,
        NAV_GOAL_ACCEPT_TIMEOUT_SEC,
        NAV_GOAL_RETRY_SEC,
        WAYPOINTS,
        WaypointHandoffMission,
        claim_goal_response,
    )


AUTO_PHASES = frozenset(("auto_to_b", "auto_to_a"))
PHASE_HEARTBEAT_SEC = 0.2
NAV_ARRIVAL_MAX_POSE_AGE_SEC = 1.0
NAV_ARRIVAL_MAX_SCAN_AGE_SEC = 1.0
NAV_ARRIVAL_MAX_COVARIANCE_M2 = 0.16
NAV_ARRIVAL_DISTANCE_EPSILON_M = 0.02
RESUME_LIVE_POSE_TIMEOUT_SEC = 6.0
RECOVERY_PLAN_GOAL_TOLERANCE_M = 0.25
RECOVERY_PLAN_START_TOLERANCE_M = 0.25
RECOVERY_COSTMAP_CLEAR_TIMEOUT_SEC = 3.0
INVALID_NAV_SUCCESS_LIMIT = 3


def _yaw_from_quaternion(orientation):
    return math.atan2(
        2.0 * (orientation.w * orientation.z
               + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y
                     + orientation.z * orientation.z),
    )


def path_reaches_waypoint(path, waypoint_name, expected_start=None):
    """Accept a path only when both its live start and requested end are valid."""
    if waypoint_name not in WAYPOINTS or len(path.poses) < 2:
        return False
    if expected_start is not None:
        start = path.poses[0].pose.position
        if math.hypot(
                float(start.x) - expected_start[0],
                float(start.y) - expected_start[1],
        ) > RECOVERY_PLAN_START_TOLERANCE_M:
            return False
    endpoint = path.poses[-1].pose.position
    target_x, target_y, _ = WAYPOINTS[waypoint_name]
    return math.hypot(
        float(endpoint.x) - target_x,
        float(endpoint.y) - target_y,
    ) <= RECOVERY_PLAN_GOAL_TOLERANCE_M


def nav_arrival_validation_reason(
        waypoint_name, position, pose_age_sec, covariance_xy,
        safety_state, scan_age_sec):
    """Return a fail-closed reason when Nav2 success is not physically credible."""
    if waypoint_name not in WAYPOINTS:
        return f"알 수 없는 waypoint={waypoint_name}"
    if position is None:
        return "현재 위치가 없습니다"
    if (len(position) < 2
            or not math.isfinite(position[0])
            or not math.isfinite(position[1])):
        return f"현재 위치가 유효하지 않습니다(position={position})"
    if pose_age_sec is None or pose_age_sec > NAV_ARRIVAL_MAX_POSE_AGE_SEC:
        return f"현재 위치 정보가 오래되었습니다(age={pose_age_sec})"
    if (covariance_xy is None
            or not math.isfinite(covariance_xy)
            or covariance_xy > NAV_ARRIVAL_MAX_COVARIANCE_M2):
        return f"AMCL 위치 불확실성이 큽니다(cov={covariance_xy})"
    target_x, target_y, _ = WAYPOINTS[waypoint_name]
    distance = math.hypot(position[0] - target_x, position[1] - target_y)
    if distance > ARRIVAL_RADIUS_M + NAV_ARRIVAL_DISTANCE_EPSILON_M:
        return (
            f"현재 위치 기준 목표 거리 {distance:.2f}m가 "
            f"허용 반경 {ARRIVAL_RADIUS_M:.2f}m 밖입니다")
    if safety_state not in ("monitoring", "cooldown"):
        return f"Safety 상태가 도착 허용 상태가 아닙니다(state={safety_state})"
    if scan_age_sec is None or scan_age_sec > NAV_ARRIVAL_MAX_SCAN_AGE_SEC:
        return f"LiDAR scan이 오래되었습니다(age={scan_age_sec})"
    return None


class SafetyWaypointHandoffMission(WaypointHandoffMission):
    """기존 미션에 Nav2 목표 취소/재전송 handshake만 추가한다."""

    def __init__(self):
        super().__init__()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._clear_global_costmap_client = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely_global_costmap",
        )
        self._clear_local_costmap_client = self.create_client(
            ClearEntireCostmap,
            "/local_costmap/clear_entirely_local_costmap",
        )
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._safety_phase_publisher = self.create_publisher(
            String, "/safety/mission_phase", latched_qos)
        self._safety_nav_status_publisher = self.create_publisher(
            String, "/safety/nav_status", 10)
        self.create_subscription(
            String, "/safety/request", self._on_safety_request, 10)
        self.create_subscription(
            String, "/safety/state", self._on_safety_state, latched_qos)
        self.create_subscription(
            LaserScan, "/scan", self._on_arrival_scan,
            qos_profile_sensor_data)
        self.create_subscription(Path, "/plan", self._on_global_plan, 10)

        self._active_goal_handle = None
        self._safety_cancel_pending = False
        self._safety_cancel_future = None
        self._safety_paused = False
        self._safety_resuming = False
        self._interrupted_goal_name = None
        self._last_phase_publish_sec = 0.0
        self._safety_state = "unknown"
        self._last_arrival_scan_sec = None
        self._resume_waiting_for_live_pose = False
        self._resume_live_pose_deadline_sec = None
        self._resume_last_live_pose_diagnostic_sec = 0.0
        self._resume_clearing_costmaps = False
        self._resume_costmap_clear_deadline_sec = None
        self._resume_global_clear_future = None
        self._resume_local_clear_future = None
        self._resume_expected_pose = None
        self._resume_target_name = None
        self._resume_plan_not_before_sec = None
        self._resume_goal_accepted = False
        self._last_valid_resume_plan_sec = None
        self._invalid_nav_success_waypoint = None
        self._invalid_nav_success_count = 0
        self._publish_safety_phase(force=True)

    def run(self):
        self.get_logger().info(
            "Safety 연동 미션: 기존 A-B-C-D-A 제어권 전환 + "
            "자율주행 구간 Nav2 취소/복구/재전송")
        super().run()

    def _tick(self):
        super()._tick()
        self._tick_resume_live_pose_wait()
        self._tick_resume_costmap_clear()
        self._publish_safety_phase()

    def _set_phase(self, phase, label, target):
        if (hasattr(self, "_invalid_nav_success_count")
                and phase != self._phase):
            self._invalid_nav_success_waypoint = None
            self._invalid_nav_success_count = 0
        super()._set_phase(phase, label, target)
        self._publish_safety_phase(force=True)

    def _halt(self, reason):
        super()._halt(reason)
        self._publish_nav_status("failed")
        self._publish_safety_phase(force=True)

    def _on_safety_request(self, message):
        request = message.data.strip().lower()
        if request == "cancel":
            self._handle_safety_cancel()
        elif request == "resume":
            self._handle_safety_resume()
        else:
            self.get_logger().warning(f"알 수 없는 Safety 요청 무시: {request}")

    def _on_safety_state(self, message):
        self._safety_state = message.data.strip().lower()

    def _on_arrival_scan(self, message):
        # 도착 검증은 scan freshness만 확인한다. B/A 주변의 지도상 정적
        # 구조물을 raw 거리만으로 "남은 임시 장애물"이라 판단하면, 실제
        # 목표 안에 도착한 뒤에도 목표 재전송과 불필요한 Safety 후진이
        # 연쇄될 수 있다. 장애물 판정은 지도 비교를 수행하는 supervisor가
        # 자율주행 중 별도로 담당한다.
        self._last_arrival_scan_sec = time.monotonic()

    def _on_global_plan(self, message):
        if (self._safety_resuming
                and self._resume_target_name is not None
                and path_reaches_waypoint(
                    message,
                    self._resume_target_name,
                    self._resume_expected_pose,
                )):
            self._last_valid_resume_plan_sec = time.monotonic()
            self._maybe_complete_safety_resume()

    def _handle_safety_cancel(self):
        if self._phase not in AUTO_PHASES:
            self.get_logger().error(
                f"Safety 자동복구 요청이 수동 단계({self._phase})에서 들어왔습니다")
            self._publish_nav_status("cancel_failed")
            return
        if self._safety_cancel_pending:
            self._publish_nav_status("canceling")
            return
        if self._safety_paused:
            self._publish_nav_status("canceled")
            return

        target = self._active_goal_name or self._pending_goal_name
        if target is None:
            self.get_logger().error("취소할 Nav2 목표가 없습니다")
            self._publish_nav_status("cancel_failed")
            return

        self._interrupted_goal_name = target
        self._safety_cancel_pending = True
        self._label = f"Safety 정지 — {target} Nav2 목표 취소 확인 중"
        self._post_status(force=True)
        self._publish_nav_status("canceling")

        if self._active_goal_handle is not None:
            self._cancel_active_goal()
        elif self._active_goal_name is None and self._pending_goal_name is not None:
            # Nav2 lifecycle 준비 전 목표가 아직 전송되지 않았다면 취소할 실제
            # action goal이 없다. 대기 목표만 제거해 확정 정지 상태로 만든다.
            self._clear_pending_goal()
            self._safety_cancel_pending = False
            self._safety_paused = True
            self._publish_nav_status("canceled")
        else:
            # send_goal_async 응답 대기 중이다. 목표 handle을 받는 즉시 취소한다.
            self.get_logger().warning("Nav2 목표 수락 응답 직후 취소하도록 대기합니다")

    def _cancel_active_goal(self):
        if self._active_goal_handle is None:
            return
        self._safety_cancel_future = self._active_goal_handle.cancel_goal_async()
        self._safety_cancel_future.add_done_callback(self._on_cancel_response)

    def _on_cancel_response(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self._safety_cancel_pending = False
            self._publish_nav_status("cancel_failed")
            self._halt(f"Safety Nav2 취소 요청 전송 실패: {exc}")
            return
        if not response.goals_canceling:
            self._safety_cancel_pending = False
            self._publish_nav_status("cancel_failed")
            self._halt("Nav2가 Safety 목표 취소 요청을 수락하지 않았습니다")
            return
        self.get_logger().warning("Nav2 목표 취소 수락 — 최종 CANCELED 결과 대기")

    def _handle_safety_resume(self):
        if not self._safety_paused or self._interrupted_goal_name is None:
            self.get_logger().error("재개할 Safety 중단 목표가 없습니다")
            self._publish_nav_status("resume_failed")
            return
        if self._active_goal_name is not None or self._pending_goal_name is not None:
            self.get_logger().error("다른 Nav2 목표가 있어 Safety 목표를 재개할 수 없습니다")
            self._publish_nav_status("resume_failed")
            return

        target = self._interrupted_goal_name
        self._safety_resuming = True
        self._label = f"Safety 직선 후진 완료 — {target} 재개 전 최신 TF 확인 중"
        self._post_status(force=True)
        self._begin_resume_live_pose_wait(target)

    def _begin_resume_live_pose_wait(self, target):
        now = time.monotonic()
        self._resume_target_name = target
        self._resume_waiting_for_live_pose = True
        self._resume_live_pose_deadline_sec = (
            now + RESUME_LIVE_POSE_TIMEOUT_SEC)
        self._resume_last_live_pose_diagnostic_sec = 0.0
        self._resume_clearing_costmaps = False
        self._resume_costmap_clear_deadline_sec = None
        self._resume_global_clear_future = None
        self._resume_local_clear_future = None
        self._resume_expected_pose = None
        self._resume_plan_not_before_sec = None
        self._resume_goal_accepted = False
        self._last_valid_resume_plan_sec = None
        self._publish_nav_status("validating_live_pose")
        self.get_logger().info(
            "Safety 직선 후진 후 최신 map->base_link TF 확인 시작")

    def _tick_resume_live_pose_wait(self):
        if not self._resume_waiting_for_live_pose:
            return
        now = time.monotonic()
        if now > self._resume_live_pose_deadline_sec:
            self._resume_waiting_for_live_pose = False
            self._safety_resuming = False
            self._publish_nav_status("resume_failed")
            self._halt("Safety 재개 전 최신 map->base_link TF 확인 시간 초과")
            return

        map_tf_pose = self._lookup_current_map_pose()
        if map_tf_pose is None:
            if now - self._resume_last_live_pose_diagnostic_sec >= 1.0:
                self.get_logger().warning(
                    "Safety 재개 대기: map->base_link TF가 없습니다")
                self._resume_last_live_pose_diagnostic_sec = now
            return

        self._resume_waiting_for_live_pose = False
        self._resume_expected_pose = map_tf_pose
        target = self._resume_target_name
        self._label = (
            f"최신 TF 확인 완료 — {target} 재개 전 costmap 정리 중")
        self._post_status(force=True)
        self.get_logger().info(
            "후진 후 최신 map->base_link TF 확인: "
            f"x={map_tf_pose[0]:.3f}, y={map_tf_pose[1]:.3f}, "
            f"yaw={math.degrees(map_tf_pose[2]):.1f}deg")
        self._begin_resume_costmap_clear()

    def _begin_resume_costmap_clear(self):
        self._resume_clearing_costmaps = True
        self._resume_costmap_clear_deadline_sec = (
            time.monotonic() + RECOVERY_COSTMAP_CLEAR_TIMEOUT_SEC)
        self._resume_global_clear_future = None
        self._resume_local_clear_future = None
        self._publish_nav_status("clearing_costmaps")

    def _tick_resume_costmap_clear(self):
        if not self._resume_clearing_costmaps:
            return
        now = time.monotonic()
        if now > self._resume_costmap_clear_deadline_sec:
            self._resume_clearing_costmaps = False
            self._safety_resuming = False
            self._publish_nav_status("resume_failed")
            self._halt("Safety 재개 전 Nav2 costmap 정리 시간 초과")
            return

        if (self._resume_global_clear_future is None
                or self._resume_local_clear_future is None):
            if (not self._clear_global_costmap_client.service_is_ready()
                    or not self._clear_local_costmap_client.service_is_ready()):
                return
            self._resume_global_clear_future = (
                self._clear_global_costmap_client.call_async(
                    ClearEntireCostmap.Request()))
            self._resume_local_clear_future = (
                self._clear_local_costmap_client.call_async(
                    ClearEntireCostmap.Request()))
            return

        if (not self._resume_global_clear_future.done()
                or not self._resume_local_clear_future.done()):
            return
        try:
            self._resume_global_clear_future.result()
            self._resume_local_clear_future.result()
        except Exception as exc:
            self._resume_clearing_costmaps = False
            self._safety_resuming = False
            self._publish_nav_status("resume_failed")
            self._halt(f"Safety 재개 전 Nav2 costmap 정리 실패: {exc}")
            return

        self._resume_clearing_costmaps = False
        target = self._resume_target_name
        self._resume_plan_not_before_sec = now
        self._label = f"costmap 정리 완료 — {target} 유효 경로 생성 확인 중"
        self._post_status(force=True)
        self.get_logger().info(
            "Safety 재개 전 global/local costmap 정리 완료")
        self._send_nav_goal(target)

    def _lookup_current_map_pose(self):
        try:
            transform = self._tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time())
        except TransformException:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            float(translation.x),
            float(translation.y),
            _yaw_from_quaternion(rotation),
        )

    def _maybe_complete_safety_resume(self):
        if (not self._safety_resuming
                or not self._resume_goal_accepted
                or self._resume_plan_not_before_sec is None
                or self._last_valid_resume_plan_sec is None
                or self._last_valid_resume_plan_sec
                < self._resume_plan_not_before_sec):
            return
        waypoint_name = self._interrupted_goal_name
        self._safety_resuming = False
        self._safety_paused = False
        self._interrupted_goal_name = None
        self._resume_target_name = None
        self._resume_expected_pose = None
        self._label = (
            "Nav2 자율주행으로 A에서 B로 이동 중"
            if waypoint_name == "B"
            else "Nav2 자율주행으로 D에서 A로 복귀 중")
        self._post_status(force=True)
        self.get_logger().info(
            f"{waypoint_name} 재개 경로 생성 확인 — Safety 제어권 해제")
        self._publish_nav_status("resumed")

    def _on_goal_response(self, waypoint_name, future):
        if not claim_goal_response(self, future):
            return
        try:
            handle = future.result()
        except Exception as exc:
            self._active_goal_name = None
            self._active_goal_handle = None
            self._next_goal_attempt_sec = time.monotonic() + NAV_GOAL_RETRY_SEC
            self.get_logger().warning(
                f"{waypoint_name} Nav2 목표 전송 실패 — 재시도 대기: {exc}")
            return
        if not handle.accepted:
            self._active_goal_name = None
            self._active_goal_handle = None
            self._next_goal_attempt_sec = time.monotonic() + NAV_GOAL_RETRY_SEC
            self.get_logger().info(
                f"Nav2가 {waypoint_name} 목표를 아직 수락하지 않았습니다 — "
                f"{NAV_GOAL_RETRY_SEC:.1f}초 뒤 재시도")
            return

        self._active_goal_handle = handle
        self._clear_pending_goal()
        self.get_logger().info(f"Nav2 목표 수락: {waypoint_name}")
        handle.get_result_async().add_done_callback(
            lambda result, name=waypoint_name: self._on_nav_result(name, result))

        if self._safety_cancel_pending:
            self._cancel_active_goal()
            return
        if self._safety_resuming and waypoint_name == self._interrupted_goal_name:
            self._resume_goal_accepted = True
            self._label = (
                f"{waypoint_name} Nav2 목표 수락 — 유효 경로 생성 확인 중")
            self._post_status(force=True)
            self._publish_nav_status("resume_planning")
            self._maybe_complete_safety_resume()
        else:
            self._publish_nav_status("navigating")

    def _queue_invalid_nav_success_retry(self, waypoint_name, reason):
        """Retry a false success, but never create an unbounded resend loop."""
        now = time.monotonic()
        self._active_goal_name = None
        self._active_goal_handle = None
        previous_waypoint = getattr(
            self, "_invalid_nav_success_waypoint", None)
        previous_count = getattr(self, "_invalid_nav_success_count", 0)
        if previous_waypoint == waypoint_name:
            self._invalid_nav_success_count = previous_count + 1
        else:
            self._invalid_nav_success_waypoint = waypoint_name
            self._invalid_nav_success_count = 1

        if self._invalid_nav_success_count >= INVALID_NAV_SUCCESS_LIMIT:
            self._pending_goal_name = None
            self._goal_accept_deadline_sec = None
            self._next_goal_attempt_sec = None
            self._halt(
                f"Nav2가 {waypoint_name} 성공을 연속 "
                f"{self._invalid_nav_success_count}회 반환했지만 실제 도착 "
                f"검증에 실패했습니다: {reason}")
            return

        self._pending_goal_name = waypoint_name
        self._goal_accept_deadline_sec = now + NAV_GOAL_ACCEPT_TIMEOUT_SEC
        self._next_goal_attempt_sec = now + NAV_GOAL_RETRY_SEC
        self._label = (
            f"Nav2의 {waypoint_name} 도착 판정 불일치 — 같은 목표 재시도 중")
        self._post_status(force=True)
        self.get_logger().warning(
            f"Nav2가 {waypoint_name} 성공을 반환했지만 실제 도착 검증 실패: "
            f"{reason}. 미션을 정지하지 않고 {NAV_GOAL_RETRY_SEC:.1f}초 뒤 "
            "같은 목표를 재전송합니다.")
        self._publish_nav_status("navigating")

    def _on_nav_result(self, waypoint_name, future):
        self._active_goal_handle = None
        if self._safety_cancel_pending:
            self._active_goal_name = None
            try:
                result = future.result()
            except Exception as exc:
                self._safety_cancel_pending = False
                self._publish_nav_status("cancel_failed")
                self._halt(f"Safety Nav2 취소 결과 수신 실패: {exc}")
                return
            if result.status != GoalStatus.STATUS_CANCELED:
                self._safety_cancel_pending = False
                self._publish_nav_status("cancel_failed")
                self._halt(
                    f"Safety 취소 중 Nav2가 CANCELED가 아닌 결과를 반환했습니다 "
                    f"(status={result.status})")
                return
            self._safety_cancel_pending = False
            self._safety_paused = True
            self._label = (
                f"Safety 복구 동작 중 — {waypoint_name} 목표는 안전하게 취소됨")
            self._post_status(force=True)
            self.get_logger().warning(
                f"Safety Nav2 취소 완료: {waypoint_name}")
            self._publish_nav_status("canceled")
            return

        try:
            result = future.result()
        except Exception:
            # The base class owns the standard transport-error handling.
            return super()._on_nav_result(waypoint_name, future)
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            # 결과 검증과 phase 전환 사이의 짧은 구간에도 supervisor가 새
            # 자동후진을 시작하지 않도록 주행 상태를 먼저 닫는다.
            self._publish_nav_status("validating_arrival")
            now = time.monotonic()
            pose_age_sec = (
                None if self._last_position_sec is None
                else now - self._last_position_sec)
            scan_age_sec = (
                None if self._last_arrival_scan_sec is None
                else now - self._last_arrival_scan_sec)
            # Nav2와 같은 map->base_link TF를 우선 사용한다. /amcl_pose 메시지가
            # 잠시 뜸하더라도 최신 TF가 목표 반경 안이면 정상 도착으로 인정하고,
            # TF가 실제 목표 밖이면 Nav2의 잘못된 성공을 그대로 통과시키지 않는다.
            map_tf_pose = self._lookup_current_map_pose()
            arrival_position = self._position
            arrival_pose_age_sec = pose_age_sec
            arrival_source = "AMCL"
            if map_tf_pose is not None:
                arrival_position = map_tf_pose[:2]
                arrival_pose_age_sec = 0.0
                arrival_source = "map->base_link TF"
            reason = nav_arrival_validation_reason(
                waypoint_name,
                arrival_position,
                arrival_pose_age_sec,
                self._position_covariance_xy,
                self._safety_state,
                scan_age_sec,
            )
            if reason is not None:
                self._queue_invalid_nav_success_retry(waypoint_name, reason)
                return
            target_x, target_y, _ = WAYPOINTS[waypoint_name]
            distance = math.hypot(
                arrival_position[0] - target_x,
                arrival_position[1] - target_y,
            )
            self.get_logger().info(
                f"{waypoint_name} 실제 도착 검증 통과 — "
                f"{arrival_source} 거리={distance:.3f}m, "
                f"cov={self._position_covariance_xy:.4f}, "
                f"Safety={self._safety_state}")
            self._invalid_nav_success_waypoint = None
            self._invalid_nav_success_count = 0

        super()._on_nav_result(waypoint_name, future)
        if self._phase in AUTO_PHASES:
            self._publish_nav_status("navigating")

    def _publish_safety_phase(self, force=False):
        now = time.monotonic()
        if (not force
                and now - self._last_phase_publish_sec < PHASE_HEARTBEAT_SEC):
            return
        message = String()
        message.data = self._phase
        self._safety_phase_publisher.publish(message)
        self._last_phase_publish_sec = now

    def _publish_nav_status(self, status):
        message = String()
        message.data = status
        self._safety_nav_status_publisher.publish(message)
        self.get_logger().info(f"Safety Nav2 상태: {status}")


def main():
    rclpy.init()
    mission = SafetyWaypointHandoffMission()
    try:
        mission.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        mission.get_logger().info("사용자 요청으로 Safety 연동 미션을 종료합니다.")
    finally:
        mission.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
