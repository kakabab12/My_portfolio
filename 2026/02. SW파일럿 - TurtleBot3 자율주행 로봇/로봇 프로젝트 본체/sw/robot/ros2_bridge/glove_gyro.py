"""자이로 장갑의 한 줄 직렬 데이터를 TurtleBot3 속도로 바꾸는 순수 로직.

ROS나 직렬 장치 없이도 파서와 기울기 매핑을 검증할 수 있도록 분리했다.
장갑 펌웨어는 아래 중 하나를 개행(\n) 단위로 전송하면 된다.

* ``{\"pitch\": 12.5, \"roll\": -8.0}`` (권장 JSON)
* ``pitch:12.5, roll:-8.0``
* ``ROLL: -8.0 | CTRL_ROLL: 8.0 | PITCH: 12.5 | STATE:FORWARD``
* ``12.5,-8.0`` (기본 순서는 pitch,roll)
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Mapping, Optional, Sequence


_KEY_VALUE_RE = re.compile(
    r"(?i)\b(pitch|roll|ctrl_roll|yaw|enabled|button|deadman)\b\s*[:=]\s*"
    r"(true|false|on|off|[-+]?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Orientation:
    """장갑의 자세. 각도 단위는 도(degree)다."""

    pitch: float
    roll: float
    yaw: Optional[float] = None
    enabled: Optional[bool] = None


@dataclass(frozen=True)
class Velocity:
    """ROS ``Twist``에 넣을 전진 속도와 회전 속도."""

    linear_x: float
    angular_z: float


def _as_finite_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "on", "1"}:
            return True
        if normalized in {"false", "off", "0"}:
            return False
    return None


def _orientation_from_mapping(values: Mapping[str, object]) -> Optional[Orientation]:
    normalized = {str(key).lower(): value for key, value in values.items()}
    # The supplied ESP-IDF source sends physical ROLL and sign-corrected
    # CTRL_ROLL. Prefer CTRL_ROLL because it is the intended control axis.
    if "ctrl_roll" in normalized:
        normalized["roll"] = normalized["ctrl_roll"]
    pitch = _as_finite_float(normalized.get("pitch"))
    roll = _as_finite_float(normalized.get("roll"))
    if pitch is None or roll is None:
        return None
    yaw = _as_finite_float(normalized.get("yaw"))
    enabled = None
    for key in ("enabled", "button", "deadman"):
        if key in normalized:
            enabled = _as_bool(normalized[key])
            break
    return Orientation(pitch=pitch, roll=roll, yaw=yaw, enabled=enabled)


def parse_orientation(
    line: str, csv_fields: Sequence[str] = ("pitch", "roll", "yaw"),
) -> Optional[Orientation]:
    """직렬 한 줄에서 자세를 읽는다. 모르는/불완전한 데이터는 ``None``이다."""
    text = line.strip()
    if not text:
        return None

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        parsed = _orientation_from_mapping(decoded)
        if parsed is not None:
            return parsed

    pairs = {
        key.lower(): value
        for key, value in _KEY_VALUE_RE.findall(text)
    }
    parsed = _orientation_from_mapping(pairs)
    if parsed is not None:
        return parsed

    parts = [part.strip() for part in text.split(",")]
    if len(parts) < 2:
        return None
    values = {}
    for field, part in zip(csv_fields, parts):
        value = _as_finite_float(part)
        if value is None:
            return None
        values[field] = value
    return _orientation_from_mapping(values)


def shortest_angle_delta(current_deg: float, neutral_deg: float) -> float:
    """-180~180 경계를 넘더라도 가장 작은 각도 차이를 반환한다."""
    return (current_deg - neutral_deg + 180.0) % 360.0 - 180.0


def average_orientation(samples: Sequence[Orientation]) -> Optional[Orientation]:
    """중립 자세 보정용 평균. yaw는 쓰지 않으므로 평균에서 제외한다."""
    if not samples:
        return None
    count = len(samples)
    return Orientation(
        pitch=sum(sample.pitch for sample in samples) / count,
        roll=sum(sample.roll for sample in samples) / count,
    )


def _apply_deadzone(value: float, deadzone_deg: float) -> float:
    if abs(value) <= deadzone_deg:
        return 0.0
    # 데드존 바깥에서는 경계부터 부드럽게 속도가 증가하도록 한다.
    return math.copysign(abs(value) - deadzone_deg, value)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def orientation_to_velocity(
    orientation: Orientation,
    neutral: Orientation,
    *,
    deadzone_deg: float = 10.0,
    linear_per_degree: float = 0.012,
    angular_per_degree: float = 0.035,
    max_linear: float = 0.12,
    max_angular: float = 0.7,
    invert_pitch: bool = False,
    invert_roll: bool = False,
) -> Velocity:
    """중립 대비 기울기를 안전한 TurtleBot3 속도로 변환한다.

    기본 매핑은 손목을 앞으로 기울이면 전진(+linear.x), 오른쪽으로 기울이면
    우회전(-angular.z)이다. 장갑 축 정의가 반대라면 CLI의 invert 옵션을 쓴다.
    """
    if deadzone_deg < 0 or any(limit <= 0 for limit in (max_linear, max_angular)):
        raise ValueError("deadzone은 0 이상, 최대 속도는 0보다 커야 합니다.")
    pitch_delta = _apply_deadzone(
        shortest_angle_delta(orientation.pitch, neutral.pitch), deadzone_deg)
    roll_delta = _apply_deadzone(
        shortest_angle_delta(orientation.roll, neutral.roll), deadzone_deg)
    if invert_pitch:
        pitch_delta = -pitch_delta
    if invert_roll:
        roll_delta = -roll_delta
    return Velocity(
        linear_x=_clamp(pitch_delta * linear_per_degree, max_linear),
        angular_z=_clamp(-roll_delta * angular_per_degree, max_angular),
    )
