from glob import glob
from setuptools import setup


package_name = 'turtlebot3_waypoint_patrol'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@localhost.localdomain',
    description='One-lap Nav2 waypoint patrol for TurtleBot3.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'patrol_node = turtlebot3_waypoint_patrol.patrol_node:main',
            'safety_mission_manager = turtlebot3_waypoint_patrol.safety_mission_manager:main',
        ],
    },
)
