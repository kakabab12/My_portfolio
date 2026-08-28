"""Launch the ordinary A -> B -> C -> D -> A waypoint patrol.

Run TurtleBot3 bringup and Nav2 separately before this launch.  Nav2 must be
started with maps/factory_map_final.yaml.  This launch starts only
``patrol_node`` and has no Safety Stop / Scan / Escape behaviour.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot3_waypoint_patrol',
            executable='patrol_node',
            name='turtlebot3_waypoint_patrol',
            output='screen',
        ),
    ])
