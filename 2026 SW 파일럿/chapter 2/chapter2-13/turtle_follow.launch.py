from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Turtlesim 시뮬레이터 노드
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='turtlesim'
        ),
        # 2. 추적 제어 및 스폰/킬 서비스 관리 노드
        Node(
            package='my_robot_controller',
            executable='turtle_follow',
            name='turtle_follow_control'
        )
    ])