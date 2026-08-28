from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. turtlesim 시뮬레이터 노드 실행
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='turtlesim'
        ),
        # 2. 직전 문제에서 만든 자율 주행 및 비동기 서비스 제어 노드 실행
        Node(
            package='my_robot_controller',
            executable='turtle_move_control',
            name='turtle_move_control'
        )
    ])