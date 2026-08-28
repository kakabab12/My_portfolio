#!/usr/bin/env python3
"""Safety 구성 없이 Nav2만으로 A-B-C-D를 무한 순찰하는 시연 미션."""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


WAYPOINTS = {
    "A": (0.044, -0.115, 0.000),
    "B": (1.429, -0.213, -1.579),
    "C": (1.376, -1.451, -1.559),
    "D": (0.124, -1.482, -3.113),
}
ROUTE = ("B", "C", "D", "A")
NAV_SERVER_WAIT_TIMEOUT_SEC = 60.0
GOAL_RETRY_DELAY_SEC = 2.0


class DemoPatrolMission(Node):
    """A를 초기 위치로 등록한 뒤 B-C-D-A를 계속 Nav2로 요청한다."""

    def __init__(self):
        super().__init__("waypoint_patrol_demo")
        self._nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self._initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)
        self._route_index = 0
        self._cycle_count = 0
        self._active_waypoint = None
        self._next_goal_timer = None

    def run(self):
        self.get_logger().info(
            "시연용 무한 순찰 시작: A->B->C->D->A (Safety 로직 없음)")
        deadline = time.monotonic() + NAV_SERVER_WAIT_TIMEOUT_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            self._publish_initial_pose()
            if self._nav_client.wait_for_server(timeout_sec=0.5):
                self.get_logger().info("Nav2 준비 완료 — B 목표부터 순찰을 시작합니다")
                self._schedule_next_goal(0.5)
                rclpy.spin(self)
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(
            f"Nav2 navigate_to_pose 서버가 {NAV_SERVER_WAIT_TIMEOUT_SEC:.0f}초 안에 준비되지 않았습니다")

    def _publish_initial_pose(self):
        x, y, yaw = WAYPOINTS["A"]
        message = PoseWithCovarianceStamped()
        # AMCL TF가 현재 시각보다 늦게 올라오는 경우도 허용하도록 stamp는 0으로 둔다.
        message.header.frame_id = "map"
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        message.pose.pose.orientation.z = math.sin(yaw / 2.0)
        message.pose.pose.orientation.w = math.cos(yaw / 2.0)
        message.pose.covariance[0] = 0.04
        message.pose.covariance[7] = 0.04
        message.pose.covariance[35] = math.radians(15.0) ** 2
        self._initial_pose_publisher.publish(message)

    def _schedule_next_goal(self, delay_sec):
        if self._next_goal_timer is not None:
            self._next_goal_timer.cancel()
            self.destroy_timer(self._next_goal_timer)
        self._next_goal_timer = self.create_timer(delay_sec, self._start_next_goal)

    def _start_next_goal(self):
        if self._next_goal_timer is not None:
            self._next_goal_timer.cancel()
            self.destroy_timer(self._next_goal_timer)
            self._next_goal_timer = None
        if self._active_waypoint is not None:
            return

        waypoint_name = ROUTE[self._route_index]
        x, y, yaw = WAYPOINTS[waypoint_name]
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._active_waypoint = waypoint_name
        self.get_logger().info(f"Nav2 목표 전송: {waypoint_name} ({x:.3f}, {y:.3f}, {yaw:.3f})")
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(
            lambda response, name=waypoint_name: self._on_goal_response(name, response))

    def _on_goal_response(self, waypoint_name, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._retry_current_goal(waypoint_name, f"목표 전송 실패: {exc}")
            return
        if not goal_handle.accepted:
            self._retry_current_goal(waypoint_name, "Nav2가 목표를 거부했습니다")
            return
        self.get_logger().info(f"Nav2 목표 수락: {waypoint_name}")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, name=waypoint_name: self._on_goal_result(name, result))

    def _on_goal_result(self, waypoint_name, future):
        self._active_waypoint = None
        try:
            status = future.result().status
        except Exception as exc:
            self._retry_current_goal(waypoint_name, f"결과 수신 실패: {exc}")
            return
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._retry_current_goal(waypoint_name, f"Nav2 결과 status={status}")
            return

        self.get_logger().info(f"{waypoint_name} 도착 완료")
        if waypoint_name == "A":
            self._cycle_count += 1
            self.get_logger().info(f"A-B-C-D-A {self._cycle_count}바퀴 완료")
        self._route_index = (self._route_index + 1) % len(ROUTE)
        self._schedule_next_goal(0.5)

    def _retry_current_goal(self, waypoint_name, reason):
        self._active_waypoint = None
        self.get_logger().warning(
            f"{waypoint_name} 목표를 {GOAL_RETRY_DELAY_SEC:.0f}초 후 재시도합니다: {reason}")
        self._schedule_next_goal(GOAL_RETRY_DELAY_SEC)


def main():
    rclpy.init()
    mission = DemoPatrolMission()
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
