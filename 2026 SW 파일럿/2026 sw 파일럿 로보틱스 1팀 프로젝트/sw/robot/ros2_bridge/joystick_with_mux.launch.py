"""turtlebot3_ws의 조이스틱을 mux 전용 /cmd_vel_joy로 실행한다."""
import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "configs", "tgz_850m_mux.yaml"))
    return LaunchDescription([
        Node(
            package="joy", executable="joy_node", name="joy_node",
            parameters=[{
                "device_id": 0,
                "deadzone": 0.12,
                "autorepeat_rate": 20.0,
                "coalesce_interval_ms": 1,
            }],
            output="screen",
        ),
        Node(
            package="teleop_twist_joy", executable="teleop_node",
            name="teleop_twist_joy_node", parameters=[config],
            remappings=[("cmd_vel", "/cmd_vel_joy")], output="screen",
        ),
    ])
