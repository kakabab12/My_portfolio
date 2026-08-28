#!/usr/bin/env python3

import math

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy


# factory_map_final map-frame measurements.
#
# The robot begins at A.  A is published only as AMCL's initial pose, then
# FollowWaypoints receives B -> C -> D -> A.  This avoids treating the already
# occupied starting point as a separate navigation goal.
INITIAL_POSE = ('A', 0.044, -0.115, 0.000)
WAYPOINTS = (
    ('B', 1.429, -0.213, -1.579),
    ('C', 1.376, -1.451, -1.559),
    ('D', 0.124, -1.482, -3.113),
    ('A', 0.044, -0.115, 0.000),
)


def make_pose(navigator, x, y, yaw):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def main(args=None):
    rclpy.init(args=args)
    navigator = BasicNavigator(node_name='turtlebot3_waypoint_patrol')

    try:
        initial_pose = make_pose(navigator, *INITIAL_POSE[1:])
        poses = [make_pose(navigator, x, y, yaw) for _, x, y, yaw in WAYPOINTS]

        navigator.get_logger().info(
            '지도 factory_map_final에서 로봇이 실제 A 지점과 지정 방향에 '
            '놓여 있다고 가정합니다.')
        navigator.get_logger().info(
            'A 위치를 AMCL 초기 위치로 자동 설정하고 Nav2 활성화를 기다립니다.')
        navigator.setInitialPose(initial_pose)
        navigator.waitUntilNav2Active()

        navigator.get_logger().info('A -> B -> C -> D -> A 한 바퀴 순찰을 시작합니다.')
        accepted = navigator.followWaypoints(poses)
        if not accepted:
            navigator.get_logger().error('Nav2가 순찰 요청을 거부했습니다.')
            return

        last_waypoint = None
        while not navigator.isTaskComplete():
            feedback = navigator.getFeedback()
            if feedback is not None and feedback.current_waypoint != last_waypoint:
                last_waypoint = feedback.current_waypoint
                name = WAYPOINTS[last_waypoint][0]
                navigator.get_logger().info(
                    f'{name} 지점으로 이동 중 ({last_waypoint + 1}/{len(WAYPOINTS)})')

        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            navigator.get_logger().info('A -> B -> C -> D -> A 순찰을 완료했습니다.')
        elif result == TaskResult.CANCELED:
            navigator.get_logger().warn('순찰이 취소되었습니다.')
        else:
            navigator.get_logger().error('순찰에 실패했습니다. Nav2/RViz 상태를 확인하세요.')
    except KeyboardInterrupt:
        navigator.get_logger().warn('사용자 요청으로 순찰을 취소합니다.')
        navigator.cancelTask()
    finally:
        navigator.destroyNode()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
