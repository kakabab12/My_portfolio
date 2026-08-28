"""TurtleBot3 하드웨어가 mux의 단일 출력만 받도록 실행하는 launch."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetRemap


def generate_launch_description():
    robot_launch = os.path.join(
        get_package_share_directory("turtlebot3_bringup"),
        "launch", "robot.launch.py")
    return LaunchDescription([
        DeclareLaunchArgument("usb_port", default_value="/dev/ttyACM0"),
        GroupAction([
            SetRemap(src="/cmd_vel", dst="/cmd_vel_muxed"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(robot_launch),
                launch_arguments={
                    "usb_port": LaunchConfiguration("usb_port"),
                }.items(),
            ),
        ]),
    ])
