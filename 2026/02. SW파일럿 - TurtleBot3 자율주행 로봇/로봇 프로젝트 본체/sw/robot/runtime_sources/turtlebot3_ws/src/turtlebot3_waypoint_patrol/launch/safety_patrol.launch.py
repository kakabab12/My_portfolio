"""Launch the Safety supervisor used by the separate integrated launcher.

TurtleBot3 bringup, Nav2, velocity mux, and waypoint mission remain owned by
the integrated shell launcher. This launch starts only the Safety supervisor,
so it cannot create a second Nav2 goal owner or base velocity publisher.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('turtlebot3_waypoint_patrol')
    default_params = os.path.join(
        package_share, 'config', 'safety_mission.yaml')

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Safety supervisor parameter YAML file.'),
        Node(
            package='turtlebot3_waypoint_patrol',
            executable='safety_mission_manager',
            name='turtlebot3_safety_mission_manager',
            output='screen',
            parameters=[params_file],
        ),
    ])
