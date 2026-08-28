#!/usr/bin/env python3
"""Keep a recent, averaged AMCL pose available for waypoint measurements.

The node deliberately does not command motion or set an initial pose.  Once an
operator has placed the robot and set AMCL's initial pose in RViz, this node
records the most recent stationary-pose window in a small state file.  The
operator can then ask the coordinator for A/B/C/D without opening another
terminal.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


STATE_PATH = Path("/tmp/turtlebot3_waypoint_coordinate.json")
WINDOW_SECONDS = 3.0
MAX_SAMPLES = 150


def quaternion_to_yaw(quaternion) -> float:
    """Return the planar yaw angle represented by a geometry quaternion."""
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny_cosp, cosy_cosp)


def circular_mean(angles: list[float]) -> float:
    """Mean heading without a discontinuity at -pi/pi."""
    return math.atan2(
        sum(math.sin(angle) for angle in angles),
        sum(math.cos(angle) for angle in angles),
    )


class WaypointCoordinateSampler(Node):
    """Write the current AMCL pose and its short-window variation to JSON."""

    def __init__(self) -> None:
        super().__init__("waypoint_coordinate_sampler")
        self._samples: deque[tuple[float, float, float, float]] = deque(
            maxlen=MAX_SAMPLES)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._on_pose,
            30,
        )
        self.create_timer(0.5, self._write_state)
        self._write_state()
        self.get_logger().info(
            "Waypoint coordinate sampler ready; waiting for /amcl_pose")

    def _on_pose(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        self._samples.append((
            time.monotonic(),
            float(pose.position.x),
            float(pose.position.y),
            quaternion_to_yaw(pose.orientation),
        ))

    def _write_state(self) -> None:
        now = time.monotonic()
        recent = [sample for sample in self._samples
                  if now - sample[0] <= WINDOW_SECONDS]
        payload: dict[str, object] = {
            "ready": False,
            "sample_count": len(recent),
            "window_seconds": WINDOW_SECONDS,
            "updated_monotonic": now,
        }
        if recent:
            xs = [sample[1] for sample in recent]
            ys = [sample[2] for sample in recent]
            yaws = [sample[3] for sample in recent]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            mean_yaw = circular_mean(yaws)
            xy_spread = max(
                math.hypot(x - mean_x, y - mean_y)
                for x, y in zip(xs, ys)
            )
            yaw_spread = max(
                abs(math.atan2(math.sin(yaw - mean_yaw),
                               math.cos(yaw - mean_yaw)))
                for yaw in yaws
            )
            payload.update({
                "ready": True,
                "x": mean_x,
                "y": mean_y,
                "yaw": mean_yaw,
                "xy_spread_m": xy_spread,
                "yaw_spread_rad": yaw_spread,
            })
        self._atomic_write(payload)

    @staticmethod
    def _atomic_write(payload: dict[str, object]) -> None:
        state_directory = STATE_PATH.parent
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{STATE_PATH.name}.", dir=state_directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
                stream.write("\n")
            os.replace(temporary_name, STATE_PATH)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def main() -> None:
    rclpy.init()
    node = WaypointCoordinateSampler()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
