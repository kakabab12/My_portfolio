"""사람 팔 랜드마크를 SO-101의 안전한 상대 관절 목표값으로 변환한다.

이 파일은 MediaPipe나 USB 장치에 직접 의존하지 않는다. 따라서 단위 테스트와
방향/감도 튜닝은 로봇팔 없이 수행할 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi
from typing import Mapping, Sequence

import numpy as np


BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ALL_JOINTS = (*BODY_JOINTS, "gripper")


@dataclass(frozen=True)
class HumanArmMeasurements:
    """카메라 좌표계에서 계산한 사람 팔의 관절값(라디안)과 집게 벌어짐."""

    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper_aperture: float

    def body_values(self) -> dict[str, float]:
        return {joint: float(getattr(self, joint)) for joint in BODY_JOINTS}


@dataclass(frozen=True)
class JointLimit:
    range_min: int
    range_max: int


def _vector(point) -> np.ndarray:
    """MediaPipe landmark를 aspect-ratio 보정한 3D numpy 벡터로 바꾼다."""
    return np.asarray((float(point.x), float(point.y) * 0.75, float(point.z)), dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length < 1e-7:
        return None
    return vector / length


def _angle(first: np.ndarray, second: np.ndarray) -> float | None:
    first_u = _unit(first)
    second_u = _unit(second)
    if first_u is None or second_u is None:
        return None
    return float(np.arccos(np.clip(np.dot(first_u, second_u), -1.0, 1.0)))


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray | None:
    axis_u = _unit(axis)
    if axis_u is None:
        return None
    return _unit(vector - np.dot(vector, axis_u) * axis_u)


def _signed_angle_about_axis(first: np.ndarray, second: np.ndarray, axis: np.ndarray) -> float | None:
    """axis 주위 first -> second의 부호 있는 각도를 계산한다."""
    axis_u = _unit(axis)
    first_u = _project_perpendicular(first, axis)
    second_u = _project_perpendicular(second, axis)
    if axis_u is None or first_u is None or second_u is None:
        return None
    sine = float(np.dot(np.cross(first_u, second_u), axis_u))
    cosine = float(np.dot(first_u, second_u))
    return float(atan2(sine, cosine))


def _hand_for_side(hands, handedness, side: str):
    if not hands:
        return None
    wanted = side.lower()
    if handedness:
        for landmarks, categories in zip(hands, handedness):
            if categories and str(categories[0].classification[0].label).lower() == wanted:
                return landmarks.landmark
    # Hands의 handedness가 한 프레임 유실돼도 첫 손을 쓰는 편이 조작 연속성에 낫다.
    return hands[0].landmark


def extract_human_arm(pose_landmarks, hand_landmarks, handedness, side: str = "right") -> HumanArmMeasurements | None:
    """MediaPipe Pose/Hands 결과에서 한쪽 팔의 관절값을 추출한다.

    Pose landmark index: left=11/13/15, right=12/14/16.  손목·손바닥은 Hands
    랜드마크를 사용해 손목 굽힘, 회전, 엄지-네손가락 집게 거리를 얻는다.
    """
    if pose_landmarks is None or not getattr(pose_landmarks, "landmark", None):
        return None
    hand = _hand_for_side(hand_landmarks, handedness, side)
    if hand is None:
        return None

    if side.lower() == "left":
        shoulder_i, elbow_i, wrist_i = 11, 13, 15
    else:
        shoulder_i, elbow_i, wrist_i = 12, 14, 16

    pose = pose_landmarks.landmark
    shoulder = _vector(pose[shoulder_i])
    elbow = _vector(pose[elbow_i])
    wrist = _vector(pose[wrist_i])
    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    upper_u = _unit(upper_arm)
    forearm_u = _unit(forearm)
    if upper_u is None or forearm_u is None:
        return None

    # MediaPipe 카메라 좌표: x=화면 오른쪽, y=아래쪽, z=카메라에서 멀어지는 방향.
    shoulder_pan = atan2(float(upper_u[0]), float(-upper_u[2]))
    shoulder_lift = atan2(float(-upper_u[1]), float(np.hypot(upper_u[0], upper_u[2])))
    elbow_angle = _angle(-upper_arm, forearm)
    if elbow_angle is None:
        return None
    elbow_flex = pi - elbow_angle

    hand_wrist = _vector(hand[0])
    middle_mcp = _vector(hand[9])
    index_mcp = _vector(hand[5])
    pinky_mcp = _vector(hand[17])
    palm_axis = _unit(middle_mcp - hand_wrist)
    lateral_axis = _unit(index_mcp - pinky_mcp)
    if palm_axis is None or lateral_axis is None:
        return None

    camera_forward = np.asarray((0.0, 0.0, 1.0))
    camera_up = np.asarray((0.0, -1.0, 0.0))
    bend_reference = _project_perpendicular(camera_forward, forearm_u)
    if bend_reference is None:
        bend_reference = _project_perpendicular(camera_up, forearm_u)
    roll_reference = _project_perpendicular(camera_up, forearm_u)
    if bend_reference is None or roll_reference is None:
        return None

    wrist_flex = _signed_angle_about_axis(bend_reference, palm_axis, forearm_u)
    wrist_roll = _signed_angle_about_axis(roll_reference, lateral_axis, forearm_u)
    if wrist_flex is None or wrist_roll is None:
        return None

    fingertips = [_vector(hand[index]) for index in (8, 12, 16, 20)]
    thumb_tip = _vector(hand[4])
    palm_width = float(np.linalg.norm(index_mcp - pinky_mcp))
    if palm_width < 1e-7:
        return None
    fingertip_center = np.mean(fingertips, axis=0)
    gripper_aperture = float(np.linalg.norm(thumb_tip - fingertip_center) / palm_width)

    return HumanArmMeasurements(
        shoulder_pan=float(shoulder_pan),
        shoulder_lift=float(shoulder_lift),
        elbow_flex=float(elbow_flex),
        wrist_flex=float(wrist_flex),
        wrist_roll=float(wrist_roll),
        gripper_aperture=gripper_aperture,
    )


class RelativeArmMapper:
    """기준 사람 자세와 기준 로봇 위치 사이의 상대 관절 제어기.

    `calibrate()`를 호출한 직후에는 계산 목표가 `arm_zero`와 동일하다. 각 관절은
    보정 파일 범위를 넘지 않고, EMA로 부드럽게 이동한다.
    """

    def __init__(self, limits: Mapping[str, JointLimit], config: Mapping):
        if set(limits) != set(ALL_JOINTS):
            raise ValueError("6개 SO-101 관절의 limit가 모두 필요합니다.")
        self.limits = dict(limits)
        self.config = config
        self.human_zero: HumanArmMeasurements | None = None
        self.arm_zero: dict[str, int] | None = None
        self._filtered: dict[str, float] | None = None

    @property
    def calibrated(self) -> bool:
        return self.human_zero is not None and self.arm_zero is not None

    def calibrate(self, human: HumanArmMeasurements, arm_positions: Mapping[str, int]) -> None:
        missing = set(ALL_JOINTS) - set(arm_positions)
        if missing:
            raise ValueError(f"로봇 시작 위치가 없습니다: {sorted(missing)}")
        self.human_zero = human
        self.arm_zero = {
            joint: self._clamp(joint, float(arm_positions[joint])) for joint in ALL_JOINTS
        }
        self._filtered = dict(self.arm_zero)

    def reset_filter(self) -> None:
        self._filtered = dict(self.arm_zero) if self.arm_zero else None

    def targets(self, human: HumanArmMeasurements) -> dict[str, int] | None:
        if not self.calibrated or self.human_zero is None or self.arm_zero is None:
            return None

        raw_targets: dict[str, float] = {}
        zero_values = self.human_zero.body_values()
        now_values = human.body_values()
        joints_cfg = self.config["joints"]
        for joint in BODY_JOINTS:
            cfg = joints_cfg[joint]
            delta = self._wrapped_delta(now_values[joint], zero_values[joint])
            offset = float(cfg["direction"]) * float(cfg["ticks_per_radian"]) * delta
            max_offset = float(cfg["max_offset_ticks"])
            offset = float(np.clip(offset, -max_offset, max_offset))
            raw_targets[joint] = self._clamp(joint, self.arm_zero[joint] + offset)

        grip_cfg = self.config["gripper"]
        span = float(grip_cfg["open_aperture"]) - float(grip_cfg["closed_aperture"])
        if span <= 0:
            raise ValueError("gripper open_aperture는 closed_aperture보다 커야 합니다.")
        openness = (human.gripper_aperture - float(grip_cfg["closed_aperture"])) / span
        openness = float(np.clip(openness, 0.0, 1.0))
        if float(grip_cfg["direction"]) < 0:
            openness = 1.0 - openness
        grip_limit = self.limits["gripper"]
        raw_targets["gripper"] = grip_limit.range_min + openness * (grip_limit.range_max - grip_limit.range_min)

        smoothing = float(self.config.get("smoothing", 0.28))
        smoothing = float(np.clip(smoothing, 0.0, 1.0))
        if self._filtered is None:
            self._filtered = dict(raw_targets)
        else:
            self._filtered = {
                joint: (1.0 - smoothing) * self._filtered[joint] + smoothing * raw_targets[joint]
                for joint in ALL_JOINTS
            }
        return {joint: int(round(self._clamp(joint, value))) for joint, value in self._filtered.items()}

    def _clamp(self, joint: str, value: float) -> float:
        limit = self.limits[joint]
        return float(np.clip(value, limit.range_min, limit.range_max))

    @staticmethod
    def _wrapped_delta(value: float, baseline: float) -> float:
        """손목 roll처럼 -pi/pi 경계를 넘는 값도 가까운 변화량으로 만든다."""
        return float((value - baseline + pi) % (2.0 * pi) - pi)
