#!/usr/bin/env python3
"""Nav2와 Safety만 사용하는 A-B-C-D 무한 순환 시연 미션.

기존 waypoint_handoff_mission.py와 waypoint_handoff_mission_safety.py는
수정하지 않는다. 시연 좌표도 이 파일 안에 복사해 두어 기존 전체 실행과
서로 영향을 주지 않는다.
"""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.executors import ExternalShutdownException

try:
    from ros2_bridge.waypoint_handoff_mission import (
        INITIAL_POSE_PUBLISH_INTERVAL_SEC,
        INITIAL_POSE_WAIT_TIMEOUT_SEC,
        NAV_GOAL_RESPONSE_TIMEOUT_SEC,
    )
    from ros2_bridge.waypoint_handoff_mission_safety import (
        ARRIVAL_RADIUS_M,
        NAV_ARRIVAL_DISTANCE_EPSILON_M,
        NAV_ARRIVAL_MAX_COVARIANCE_M2,
        NAV_ARRIVAL_MAX_POSE_AGE_SEC,
        NAV_ARRIVAL_MAX_SCAN_AGE_SEC,
        RECOVERY_PLAN_GOAL_TOLERANCE_M,
        RECOVERY_PLAN_START_TOLERANCE_M,
        SafetyWaypointHandoffMission,
    )
except ImportError:
    from waypoint_handoff_mission import (
        INITIAL_POSE_PUBLISH_INTERVAL_SEC,
        INITIAL_POSE_WAIT_TIMEOUT_SEC,
        NAV_GOAL_RESPONSE_TIMEOUT_SEC,
    )
    from waypoint_handoff_mission_safety import (
        ARRIVAL_RADIUS_M,
        NAV_ARRIVAL_DISTANCE_EPSILON_M,
        NAV_ARRIVAL_MAX_COVARIANCE_M2,
        NAV_ARRIVAL_MAX_POSE_AGE_SEC,
        NAV_ARRIVAL_MAX_SCAN_AGE_SEC,
        RECOVERY_PLAN_GOAL_TOLERANCE_M,
        RECOVERY_PLAN_START_TOLERANCE_M,
        SafetyWaypointHandoffMission,
    )


# 기존 미션 좌표의 시연 전용 복사본. 이후 시연 동선만 바꿀 때는 이 값만
# 수정하며, TurtleBot3_전체실행_Safety의 waypoint 원본은 건드리지 않는다.
DEMO_WAYPOINTS = {
    "A": (0.044, -0.115, 0.000),
    "B": (1.429, -0.213, -1.579),
    "C": (1.376, -1.451, -1.559),
    "D": (0.124, -1.482, -3.113),
}
DEMO_ROUTE = ("B", "C", "D", "A")
DEMO_AUTO_PHASE = "auto_to_b"
PREVIOUS_WAYPOINT = {"B": "A", "C": "B", "D": "C", "A": "D"}
NAV_STATUS_HEARTBEAT_SEC = 0.2


def _yaw_to_quaternion(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def demo_path_reaches_waypoint(path, waypoint_name, expected_start=None):
    """Safety 복구 뒤 생성된 경로의 현재 시작점과 목표 끝점을 검증한다."""
    if waypoint_name not in DEMO_WAYPOINTS or len(path.poses) < 2:
        return False
    if expected_start is not None:
        start = path.poses[0].pose.position
        if math.hypot(
                float(start.x) - expected_start[0],
                float(start.y) - expected_start[1],
        ) > RECOVERY_PLAN_START_TOLERANCE_M:
            return False
    endpoint = path.poses[-1].pose.position
    target_x, target_y, _ = DEMO_WAYPOINTS[waypoint_name]
    return math.hypot(
        float(endpoint.x) - target_x,
        float(endpoint.y) - target_y,
    ) <= RECOVERY_PLAN_GOAL_TOLERANCE_M


def demo_arrival_validation_reason(
        waypoint_name, position, pose_age_sec, covariance_xy,
        safety_state, scan_age_sec):
    """Nav2 성공을 실제 위치·AMCL 신뢰도·센서 상태로 다시 검증한다."""
    if waypoint_name not in DEMO_WAYPOINTS:
        return f"알 수 없는 waypoint={waypoint_name}"
    if position is None:
        return "현재 위치가 없습니다"
    if (len(position) < 2
            or not math.isfinite(position[0])
            or not math.isfinite(position[1])):
        return f"현재 위치가 유효하지 않습니다(position={position})"
    if (pose_age_sec is None
            or pose_age_sec > NAV_ARRIVAL_MAX_POSE_AGE_SEC):
        return f"현재 위치 정보가 오래되었습니다(age={pose_age_sec})"
    if (covariance_xy is None
            or not math.isfinite(covariance_xy)
            or covariance_xy > NAV_ARRIVAL_MAX_COVARIANCE_M2):
        return f"AMCL 위치 불확실성이 큽니다(cov={covariance_xy})"
    target_x, target_y, _ = DEMO_WAYPOINTS[waypoint_name]
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


class DemoSafetyPatrolMission(SafetyWaypointHandoffMission):
    """모든 구간을 Nav2로 주행하고 A 도착 뒤 B부터 다시 반복한다."""

    def __init__(self):
        super().__init__()
        self._route_index = 0
        self._cycle_count = 0
        self._demo_halted = False
        self._last_nav_status_heartbeat_sec = 0.0
        self._checkpoints = {
            "A": "출발",
            "B": "이동 예정",
            "C": "대기",
            "D": "대기",
        }

    def run(self):
        self.get_logger().info(
            "시연용 무한 순찰: A->B->C->D->A, 전 구간 Nav2 + Safety")
        self._publish_nav_status("initializing")

        if not self._nav_client.wait_for_server(timeout_sec=45.0):
            self._halt("Nav2 navigate_to_pose 서버를 찾지 못했습니다")
            rclpy.spin(self)
            return

        # 로봇을 A에 놓고 실행한다는 기존 Safety 시연 조건을 그대로 사용한다.
        # AMCL이 초기 위치를 실제로 받을 때까지 A를 반복 발행한다.
        deadline = time.monotonic() + INITIAL_POSE_WAIT_TIMEOUT_SEC
        next_pose_publish_sec = 0.0
        while (rclpy.ok() and self._position is None
               and time.monotonic() < deadline):
            now = time.monotonic()
            if now >= next_pose_publish_sec:
                self._publish_initial_pose()
                next_pose_publish_sec = (
                    now + INITIAL_POSE_PUBLISH_INTERVAL_SEC)
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._position is None:
            self._halt(
                f"AMCL 현재 좌표를 {INITIAL_POSE_WAIT_TIMEOUT_SEC:.0f}초 안에 "
                "받지 못해 순찰을 시작하지 않았습니다")
            rclpy.spin(self)
            return
        if not self._wait_for_nav2_active():
            self._halt(
                "bt_navigator가 ACTIVE 상태가 되지 않아 순찰을 시작하지 않았습니다")
            rclpy.spin(self)
            return

        self._start_current_leg()
        rclpy.spin(self)

    def _post_status(self, force=False):
        # 시연용은 제스처 웹 서버를 실행하지 않으므로 HTTP 상태 보고도 생략한다.
        del force

    def _tick(self):
        super()._tick()
        if self._demo_halted:
            self._publish_nav_status_heartbeat("failed")
        elif (self._active_goal_handle is not None
              and not self._safety_cancel_pending
              and not self._safety_resuming):
            self._publish_nav_status_heartbeat("navigating")

    def _publish_nav_status_heartbeat(self, status):
        now = time.monotonic()
        if now - self._last_nav_status_heartbeat_sec < NAV_STATUS_HEARTBEAT_SEC:
            return
        self._publish_nav_status(status)
        self._last_nav_status_heartbeat_sec = now

    def _publish_initial_pose(self):
        x, y, yaw = DEMO_WAYPOINTS["A"]
        message = PoseWithCovarianceStamped()
        message.header.frame_id = "map"
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        (message.pose.pose.orientation.z,
         message.pose.pose.orientation.w) = _yaw_to_quaternion(yaw)
        message.pose.covariance[0] = 0.04
        message.pose.covariance[7] = 0.04
        message.pose.covariance[35] = math.radians(15.0) ** 2
        self._initial_pose_publisher.publish(message)

    def _attempt_pending_nav_goal(self):
        waypoint_name = self._pending_goal_name
        if waypoint_name is None or self._active_goal_name is not None:
            return
        x, y, yaw = DEMO_WAYPOINTS[waypoint_name]
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z, pose.pose.orientation.w = (
            _yaw_to_quaternion(yaw))
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._active_goal_name = waypoint_name
        future = self._nav_client.send_goal_async(goal)
        self._goal_response_future = future
        self._goal_response_deadline_sec = (
            time.monotonic() + NAV_GOAL_RESPONSE_TIMEOUT_SEC)
        future.add_done_callback(
            lambda response, name=waypoint_name:
                self._on_goal_response(name, response))

    def _on_global_plan(self, message):
        if (self._safety_resuming
                and self._resume_target_name is not None
                and demo_path_reaches_waypoint(
                    message,
                    self._resume_target_name,
                    self._resume_expected_pose,
                )):
            self._last_valid_resume_plan_sec = time.monotonic()
            self._maybe_complete_safety_resume()

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
        self._label = f"Safety 복구 완료 — {waypoint_name} 자율주행 재개"
        self.get_logger().info(
            f"{waypoint_name} 재개 경로 생성 확인 — Safety 제어권 해제")
        self._publish_nav_status("resumed")

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
                    "Safety 취소 중 Nav2가 CANCELED가 아닌 결과를 반환했습니다 "
                    f"(status={result.status})")
                return
            self._safety_cancel_pending = False
            self._safety_paused = True
            self._label = (
                f"Safety 복구 동작 중 — {waypoint_name} 목표 취소 완료")
            self.get_logger().warning(
                f"Safety Nav2 취소 완료: {waypoint_name}")
            self._publish_nav_status("canceled")
            return

        try:
            result = future.result()
        except Exception as exc:
            self._active_goal_name = None
            self._halt(f"{waypoint_name} Nav2 결과 수신 실패: {exc}")
            return
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self._active_goal_name = None
            self._halt(
                f"Nav2가 {waypoint_name} 도착에 실패 또는 취소되었습니다 "
                f"(status={result.status})")
            return

        self._publish_nav_status("validating_arrival")
        now = time.monotonic()
        pose_age_sec = (
            None if self._last_position_sec is None
            else now - self._last_position_sec)
        scan_age_sec = (
            None if self._last_arrival_scan_sec is None
            else now - self._last_arrival_scan_sec)
        map_tf_pose = self._lookup_current_map_pose()
        arrival_position = self._position
        arrival_pose_age_sec = pose_age_sec
        arrival_source = "AMCL"
        if map_tf_pose is not None:
            arrival_position = map_tf_pose[:2]
            arrival_pose_age_sec = 0.0
            arrival_source = "map->base_link TF"
        reason = demo_arrival_validation_reason(
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

        expected_waypoint = DEMO_ROUTE[self._route_index]
        if waypoint_name != expected_waypoint:
            self._active_goal_name = None
            self._halt(
                f"현재 순찰 목표 {expected_waypoint}와 완료 결과 "
                f"{waypoint_name}가 다릅니다")
            return

        target_x, target_y, _ = DEMO_WAYPOINTS[waypoint_name]
        distance = math.hypot(
            arrival_position[0] - target_x,
            arrival_position[1] - target_y,
        )
        self.get_logger().info(
            f"{waypoint_name} 실제 도착 검증 통과 — "
            f"{arrival_source} 거리={distance:.3f}m, "
            f"cov={self._position_covariance_xy:.4f}, "
            f"Safety={self._safety_state}")

        self._active_goal_name = None
        self._invalid_nav_success_waypoint = None
        self._invalid_nav_success_count = 0
        self._checkpoints[waypoint_name] = "도착 완료"
        if waypoint_name == "A":
            self._cycle_count += 1
            self.get_logger().info(
                f"A-B-C-D-A {self._cycle_count}바퀴 완료 — 다음 바퀴 시작")
            self._checkpoints = {
                "A": f"{self._cycle_count}바퀴 완료",
                "B": "이동 예정",
                "C": "대기",
                "D": "대기",
            }
        self._route_index = (self._route_index + 1) % len(DEMO_ROUTE)
        self._start_current_leg()

    def _start_current_leg(self):
        if self._demo_halted:
            return
        waypoint_name = DEMO_ROUTE[self._route_index]
        start_name = PREVIOUS_WAYPOINT[waypoint_name]
        self._checkpoints[waypoint_name] = "Nav2 이동 중"
        self._set_phase(
            DEMO_AUTO_PHASE,
            f"시연용 Nav2+Safety: {start_name}에서 {waypoint_name}로 이동 중",
            waypoint_name,
        )
        self._send_nav_goal(waypoint_name)

    def _halt(self, reason):
        if self._demo_halted:
            return
        self._demo_halted = True
        self._label = f"시연용 순찰 안전정지: {reason}"
        self._target = "-"
        self._clear_pending_goal()
        self.get_logger().error(self._label)
        # phase heartbeat는 유지하고 nav_status를 failed로 고정한다. 시연 전용
        # mux가 이후의 Nav2 속도를 fail-closed 방식으로 계속 차단한다.
        self._phase = DEMO_AUTO_PHASE
        self._publish_nav_status("failed")
        self._publish_safety_phase(force=True)


def main():
    rclpy.init()
    mission = DemoSafetyPatrolMission()
    try:
        mission.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        mission.get_logger().info("사용자 요청으로 시연용 무한 순찰을 종료합니다")
    finally:
        mission.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
