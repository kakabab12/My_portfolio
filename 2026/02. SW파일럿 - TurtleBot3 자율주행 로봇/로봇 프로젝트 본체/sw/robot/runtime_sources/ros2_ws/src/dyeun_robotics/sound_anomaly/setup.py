from glob import glob
from setuptools import find_packages, setup


package_name = "sound_anomaly"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/models", glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="robotics",
    maintainer_email="robotics@example.com",
    description="Live gearbox sound anomaly detection with OpenCR LED output.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sound_anomaly_node = sound_anomaly.sound_anomaly_node:main",
        ],
    },
)
