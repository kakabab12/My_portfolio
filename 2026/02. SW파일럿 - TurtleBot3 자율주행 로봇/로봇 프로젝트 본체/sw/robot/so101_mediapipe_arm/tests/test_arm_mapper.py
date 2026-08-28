import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_mapper import (  # noqa: E402
    ALL_JOINTS,
    HumanArmMeasurements,
    JointLimit,
    RelativeArmMapper,
    extract_human_arm,
)


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0


@dataclass
class Landmarks:
    landmark: list[Point]


def make_pose() -> Landmarks:
    points = [Point(0.5, 0.5, 0.0) for _ in range(33)]
    # 화면 오른팔: 위팔은 약간 오른쪽 아래, 아래팔은 더 아래로 뻗은 기본 자세.
    points[12] = Point(0.50, 0.40, 0.05)
    points[14] = Point(0.62, 0.56, -0.02)
    points[16] = Point(0.68, 0.72, -0.08)
    return Landmarks(points)


def make_hand(aperture: float = 0.9) -> Landmarks:
    points = [Point(0.68, 0.72, -0.08) for _ in range(21)]
    points[0] = Point(0.68, 0.72, -0.08)
    points[5] = Point(0.63, 0.65, -0.08)
    points[9] = Point(0.68, 0.62, -0.08)
    points[17] = Point(0.73, 0.65, -0.08)
    points[8] = Point(0.64, 0.56, -0.08)
    points[12] = Point(0.68, 0.54, -0.08)
    points[16] = Point(0.72, 0.56, -0.08)
    points[20] = Point(0.75, 0.60, -0.08)
    # palm 폭은 0.10이므로 thumb와 fingertip 평균 거리를 aperture*0.10으로 설정.
    points[4] = Point(0.697, 0.58 + aperture * 0.10, -0.08)
    return Landmarks(points)


class ArmMapperTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "smoothing": 1.0,
            "gripper": {"closed_aperture": 0.32, "open_aperture": 1.05, "direction": 1},
            "joints": {
                joint: {"direction": 1, "ticks_per_radian": 300, "max_offset_ticks": 400}
                for joint in ALL_JOINTS[:-1]
            },
        }
        self.limits = {joint: JointLimit(1000, 3000) for joint in ALL_JOINTS}

    def test_extract_returns_finite_six_dof_measurement(self):
        value = extract_human_arm(make_pose(), [make_hand()], None, "right")
        self.assertIsNotNone(value)
        self.assertTrue(all(math.isfinite(getattr(value, field)) for field in value.__dataclass_fields__))
        self.assertGreater(value.gripper_aperture, 0.0)

    def test_mapper_starts_at_robot_zero_and_clamps(self):
        human = HumanArmMeasurements(0.1, 0.2, 0.3, -0.2, 0.4, 0.7)
        mapper = RelativeArmMapper(self.limits, self.config)
        arm_zero = {joint: 2000 for joint in ALL_JOINTS}
        mapper.calibrate(human, arm_zero)
        target = mapper.targets(human)
        for joint in ALL_JOINTS[:-1]:
            self.assertEqual(target[joint], 2000)

        moved = HumanArmMeasurements(3.0, -3.0, 3.0, 3.0, -3.0, 9.0)
        target = mapper.targets(moved)
        self.assertTrue(all(1000 <= value <= 3000 for value in target.values()))
        self.assertEqual(target["shoulder_pan"], 2400)  # max_offset_ticks=400
        self.assertEqual(target["gripper"], 3000)

    def test_gripper_aperture_changes_output(self):
        mapper = RelativeArmMapper(self.limits, self.config)
        zero = HumanArmMeasurements(0, 0, 0, 0, 0, 0.5)
        mapper.calibrate(zero, {joint: 2000 for joint in ALL_JOINTS})
        closed = mapper.targets(HumanArmMeasurements(0, 0, 0, 0, 0, 0.2))["gripper"]
        opened = mapper.targets(HumanArmMeasurements(0, 0, 0, 0, 0, 1.2))["gripper"]
        self.assertLess(closed, opened)


if __name__ == "__main__":
    unittest.main()
