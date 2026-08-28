from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='led_status_node',
            executable='led_status',
            name='led_status_node',
            output='screen',
            parameters=[{
                'green_pin': 31,
                'red_pin': 33,
                'blink_period': 0.5,
                'active_high': True,
                'initial_state': 'idle',
            }],
        )
    ])
