from glob import glob
from setuptools import setup

package_name = 'turtlebot3_joystick'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@localhost.localdomain',
    description='Safe joystick teleoperation for the TurtleBot3 Burger.',
    license='Apache-2.0',
)
