from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from pathlib import Path


def generate_launch_description():
    package_share = Path(get_package_share_directory("sound_anomaly"))
    # Dependencies are kept outside the system Python to protect other Jetson projects.
    dependency_path = str(Path.home() / "ros2_ws" / "python_deps")
    python_path = dependency_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    return LaunchDescription(
        [
            Node(
                package="sound_anomaly",
                executable="sound_anomaly_node",
                name="sound_anomaly_node",
                output="screen",
                parameters=[str(package_share / "config" / "sound_anomaly.yaml")],
                additional_env={"PYTHONPATH": python_path},
                respawn=True,
                respawn_delay=2.0,
            )
        ]
    )
