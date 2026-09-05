"""얼굴이 향한 곳에 커서가 있는가 (2026-09-05 신설).

이 제품의 핵심 요구는 하나다. **사람이 아이콘을 바라보면 커서도 그 아이콘에
있어야 한다.** 그래야 사용자가 커서를 쳐다보며 조이스틱처럼 몰지 않고,
가고 싶은 곳을 그냥 보면 된다.

그런데 지금까지 커서는 `tan(회전각) / tan(15도)`로 정해졌다. 이 15도에는
근거가 없다. 기하학이 정하는 값은 따로 있다 — 사용자가 화면에서 Z만큼
떨어져 고개를 θ 돌리면 시선이 화면에서 Z·tan(θ)만큼 옮겨가므로

    반폭 = atan((화면 가로 / 2) / 거리)

이고, 531mm 화면이면 500mm에서 28.0도, 1300mm에서 11.5도다. 고정 15도는
989mm 한 지점에서만 맞는다.

여기서 보증하는 것
------------------
  1) 설치 치수를 주면 반폭을 그 치수로 계산한다 (설정 각도를 무시한다)
  2) 안 주면 예전과 **완전히 같다** — 거동이 바뀌지 않는다
  3) 치수가 이상하면(0, 음수) 무시하고 예전 값을 쓴다
  4) 캘리브레이션 뒤에 사용자가 멀어지면 반폭이 그만큼 좁아진다
  5) 그 거리 추정이 회전에 안 흔들린다 (고개를 돌린 것을 멀어진 것으로
     읽으면 커서 감도가 회전할 때마다 출렁인다)
"""
import copy
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_orientation import (               # noqa: E402
    RIGID_LANDMARKS, HeadOrientation, _orthonormal_frame,
)
from src.postprocess.head_tracker import HeadTracker          # noqa: E402
from src.utils.config_loader import load_config               # noqa: E402

from tests.test_head_orientation import (                     # noqa: E402
    _apply, _rot_about, _synthetic_head,
)
from tests.test_orientation_mapping import (                  # noqa: E402
    _Clock, _Face, _calibrate,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "config.yaml",
)

# 24인치 16:9 모니터의 실제 표시 영역과, 그 앞에 사용자가 서는 거리
SCREEN_W_MM = 531.0
SCREEN_H_MM = 299.0
REF_DIST_MM = 700.0
GEOMETRIC_SPAN_X_DEG = math.degrees(math.atan((SCREEN_W_MM * 0.5) / REF_DIST_MM))
GEOMETRIC_SPAN_Y_DEG = math.degrees(math.atan((SCREEN_H_MM * 0.5) / REF_DIST_MM))


def _tracker(clock, *, geometry=True, width=SCREEN_W_MM, height=SCREEN_H_MM,
             distance=REF_DIST_MM, distance_scaling=True):
    config = copy.deepcopy(load_config(CONFIG_PATH))
    pointer = config["head_tracker"]["pointer"]
    pointer["orientation_mapping"] = True
    pointer["orientation_half_span_x_deg"] = 15.0
    pointer["orientation_half_span_y_deg"] = 10.0
    pointer["one_euro_enabled"] = False
    pointer["smoothing_alpha"] = 1.0            # 평활 없음 — 매핑만 본다
    pointer["orientation_lens_calibration"] = False   # 렌즈는 이 시험의 대상이 아니다
    pointer["orientation_distance_scaling"] = distance_scaling
    pointer["screen_width_mm"] = width if geometry else None
    pointer["screen_height_mm"] = height if geometry else None
    pointer["reference_distance_mm"] = distance if geometry else None
    return HeadTracker(config, clock=clock)


def _mapper(tracker):
    """커서 매핑을 들고 있는 객체 — 반폭을 직접 확인할 때 쓴다."""
    for name in ("_cursor", "_cursor_mapper", "_mapper"):
        obj = getattr(tracker, name, None)
        if obj is not None and hasattr(obj, "_orientation_tan_x"):
            return obj
    for value in vars(tracker).values():
        if hasattr(value, "_orientation_tan_x"):
            return value
    raise AssertionError("커서 매퍼를 못 찾았다")


def _turn(neutral, deg):
    _x, y_axis, _z = _orthonormal_frame(neutral[list(RIGID_LANDMARKS)])
    return _apply(_rot_about(y_axis, deg), neutral)


def _scaled(points, factor):
    """얼굴이 factor배 크게 보이게 한다 = 1/factor 거리로 다가온 것."""
    arr = np.asarray(points, dtype=np.float64)
    center = arr[list(RIGID_LANDMARKS)].mean(axis=0)
    return (arr - center) * factor + center


# ── 1. 설치 치수가 반폭을 정한다 ────────────────────────────────────────────

def test_screen_geometry_sets_half_span():
    """화면 치수를 주면 설정 각도(15도) 대신 기하학이 정한 각도를 쓴다."""
    mapper = _mapper(_tracker(_Clock()))
    got = math.degrees(math.atan(mapper._orientation_tan_x))
    assert got == pytest.approx(GEOMETRIC_SPAN_X_DEG, abs=1e-6)
    assert got == pytest.approx(20.78, abs=0.05)      # 531mm / 700mm
    assert got != pytest.approx(15.0, abs=0.5)        # 설정값이 아니다
    got_y = math.degrees(math.atan(mapper._orientation_tan_y))
    assert got_y == pytest.approx(GEOMETRIC_SPAN_Y_DEG, abs=1e-6)


def test_no_geometry_keeps_old_behaviour():
    """치수를 안 주면 예전 그대로 — 이 변경으로 기존 설치가 흔들리면 안 된다."""
    mapper = _mapper(_tracker(_Clock(), geometry=False))
    assert math.degrees(math.atan(mapper._orientation_tan_x)) == pytest.approx(15.0, abs=1e-6)
    assert math.degrees(math.atan(mapper._orientation_tan_y)) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize("width,height,distance", [
    (0.0, SCREEN_H_MM, REF_DIST_MM),
    (SCREEN_W_MM, 0.0, REF_DIST_MM),
    (SCREEN_W_MM, SCREEN_H_MM, 0.0),
    (-531.0, SCREEN_H_MM, REF_DIST_MM),
    (SCREEN_W_MM, SCREEN_H_MM, -700.0),
])
def test_nonsense_geometry_is_ignored(width, height, distance, monkeypatch):
    """0이나 음수를 받으면 무시하고 설정 각도로 돌아간다 — 0으로 나누지 않는다.

    자동 인식도 막아 둔다. 이 시험은 **막는 장치가 있는가**를 보는 것이라,
    돌리는 기계에 어떤 모니터가 붙어 있느냐에 결과가 달라지면 안 된다.
    """
    from src.utils import display_size as DS
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", lambda: None)
    monkeypatch.setattr(DS, "_from_device_caps", lambda: None)
    try:
        mapper = _mapper(_tracker(_Clock(), width=width, height=height,
                                  distance=distance))
        assert math.degrees(math.atan(mapper._orientation_tan_x)) == pytest.approx(
            15.0, abs=1e-6)
    finally:
        DS.detect_screen_size_mm.cache_clear()


def test_geometric_span_puts_cursor_at_edge():
    """그 각도만큼 돌리면 커서가 화면 끝에 닿는다 — 매핑이 실제로 이어진다."""
    clock = _Clock()
    tracker = _tracker(clock)
    neutral = _synthetic_head()
    _calibrate(tracker, clock, _Face(neutral))
    clock.now += 0.1
    result = tracker.update(_Face(_turn(neutral, GEOMETRIC_SPAN_X_DEG)))
    limit = min(0.5, load_config(CONFIG_PATH)["head_tracker"]["pointer"]["max_offset_ratio"])
    assert abs(result.cursor_x_ratio - 0.5) == pytest.approx(limit, abs=0.02)


def test_geometry_needs_less_turning_than_fixed_span_at_close_range():
    """가까이 있는 설치에서는 고정 15도보다 **더** 돌려야 한다.

    500mm 앞의 531mm 화면은 28도를 봐야 끝에 닿는다. 15도로 두면 절반만
    돌려도 커서가 끝까지 날아가 버려, 얼굴이 향한 곳과 커서가 어긋난다.
    """
    clock = _Clock()
    close = _mapper(_tracker(clock, distance=500.0))
    far = _mapper(_tracker(_Clock(), distance=1300.0))
    assert math.degrees(math.atan(close._orientation_tan_x)) > 15.0
    assert math.degrees(math.atan(far._orientation_tan_x)) < 15.0


# ── 2. 거리에 따라 반폭이 따라간다 ──────────────────────────────────────────

def test_distance_ratio_tracks_apparent_size():
    """얼굴이 절반 크기로 보이면 두 배 멀어진 것으로 읽는다."""
    ho = HeadOrientation(rotation_source="landmarks")
    neutral = _synthetic_head()
    for _ in range(20):
        ho.add_calibration_sample(_Face(neutral))
    assert ho.finalize_neutral()
    assert ho.distance_ratio == pytest.approx(1.0, abs=1e-9)
    far = _Face(_scaled(neutral, 0.5))
    for _ in range(400):                      # EMA가 수렴할 만큼
        ho.pointing_offset(far)
    assert ho.distance_ratio == pytest.approx(2.0, rel=0.02)


def test_distance_ratio_is_not_fooled_by_rotation():
    """고개를 돌린 것을 '멀어졌다'로 읽으면 안 된다.

    그렇게 읽으면 고개를 돌릴 때마다 감도가 출렁여 커서가 가속하는 것처럼
    느껴진다. 크기 척도를 회전 불변인 프로베니우스 노름으로 잡은 이유다.
    """
    ho = HeadOrientation(rotation_source="landmarks")
    neutral = _synthetic_head()
    for _ in range(20):
        ho.add_calibration_sample(_Face(neutral))
    assert ho.finalize_neutral()
    for deg in (-20.0, -10.0, 10.0, 20.0):
        turned = _Face(_turn(neutral, deg))
        for _ in range(200):
            ho.pointing_offset(turned)
        assert ho.distance_ratio == pytest.approx(1.0, abs=0.03), f"{deg}도에서 흔들렸다"


def test_moving_away_makes_cursor_more_sensitive():
    """멀어지면 같은 각도로 커서가 더 많이 움직인다 — 화면이 작게 보이니까."""
    clock = _Clock()
    tracker = _tracker(clock)
    neutral = _synthetic_head()
    _calibrate(tracker, clock, _Face(neutral))

    clock.now += 0.1
    near = tracker.update(_Face(_turn(neutral, 8.0)))
    near_off = abs(near.cursor_x_ratio - 0.5)

    # 두 배 멀어진다 (얼굴이 절반 크기로 보인다).
    # 간격을 짧게 두는 이유: 같은 프레임을 오래 먹이면 트래커가 '아무도
    # 없다'고 보고 잠금을 푼다 — 이 시험이 보려는 것과 무관한 경로다
    far_neutral = _Face(_scaled(neutral, 0.5))
    for _ in range(90):
        clock.now += 0.02
        tracker.update(far_neutral)
    far = tracker.update(_Face(_scaled(_turn(neutral, 8.0), 0.5)))
    far_off = abs(far.cursor_x_ratio - 0.5)

    assert far_off > near_off * 1.5, (
        f"멀어졌는데 감도가 안 올랐다: {near_off:.4f} -> {far_off:.4f}")


def test_distance_scaling_can_be_turned_off():
    """스위치를 끄면 거리가 바뀌어도 반폭이 그대로다 — 되돌릴 길을 막지 않는다."""
    clock = _Clock()
    tracker = _tracker(clock, distance_scaling=False)
    neutral = _synthetic_head()
    _calibrate(tracker, clock, _Face(neutral))
    clock.now += 0.1
    near = abs(tracker.update(_Face(_turn(neutral, 8.0))).cursor_x_ratio - 0.5)
    far_neutral = _Face(_scaled(neutral, 0.5))
    for _ in range(90):
        clock.now += 0.02
        tracker.update(far_neutral)
    far = abs(tracker.update(_Face(_scaled(_turn(neutral, 8.0), 0.5))).cursor_x_ratio - 0.5)
    assert far == pytest.approx(near, rel=0.05)


def test_distance_ratio_is_clamped():
    """말이 안 되는 크기 변화는 잘라낸다 — 검출이 무너져도 커서가 안 날아간다."""
    from src.postprocess import head_orientation as HO
    ho = HeadOrientation(rotation_source="landmarks")
    neutral = _synthetic_head()
    for _ in range(20):
        ho.add_calibration_sample(_Face(neutral))
    assert ho.finalize_neutral()
    tiny = _Face(_scaled(neutral, 0.02))       # 50배 멀어진 것처럼
    for _ in range(600):
        ho.pointing_offset(tiny)
    assert ho.distance_ratio <= HO.MAX_DISTANCE_RATIO + 1e-9
    huge = _Face(_scaled(neutral, 40.0))
    for _ in range(600):
        ho.pointing_offset(huge)
    assert ho.distance_ratio >= HO.MIN_DISTANCE_RATIO - 1e-9


# ── 3. 데스크탑·키오스크 공용 — 화면 크기는 운영체제에서 읽는다 ────────────

def test_screen_size_is_detected_when_not_configured(monkeypatch):
    """거리만 적어 두면 화면 크기는 알아서 읽어 온다.

    이게 있어야 한 빌드로 데스크탑과 키오스크를 같이 쓴다 — 14인치 노트북과
    32인치 키오스크는 화면 절반 폭이 2.3배 차이라, 같은 각도 설정을 쓰면
    한쪽은 반드시 틀린다.
    """
    from src.utils import display_size as DS
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", lambda: (700.0, 393.0))   # 32인치
    monkeypatch.setattr(DS, "_from_device_caps", lambda: None)
    try:
        mapper = _mapper(_tracker(_Clock(), width=None, height=None, distance=800.0))
        got = math.degrees(math.atan(mapper._orientation_tan_x))
        assert got == pytest.approx(math.degrees(math.atan(350.0 / 800.0)), abs=1e-6)
        assert got == pytest.approx(23.62, abs=0.05)
    finally:
        DS.detect_screen_size_mm.cache_clear()


def test_no_reference_distance_means_no_detection_and_no_change(monkeypatch):
    """거리를 안 주면 화면 크기를 알아봐야 소용없다 — 아예 안 알아본다.

    거리를 모르는 채로 각도를 지어내지 않는다. 시작이 느려지지도 않는다.
    """
    from src.utils import display_size as DS
    calls = []
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", lambda: calls.append(1) or (700.0, 393.0))
    try:
        mapper = _mapper(_tracker(_Clock(), geometry=False))
        assert math.degrees(math.atan(mapper._orientation_tan_x)) == pytest.approx(15.0, abs=1e-6)
        assert calls == []
    finally:
        DS.detect_screen_size_mm.cache_clear()


def test_configured_size_wins_over_detection(monkeypatch):
    """직접 적어 놓은 값이 자동 인식보다 우선한다 — 인식이 틀리는 화면이 있다."""
    from src.utils import display_size as DS
    DS.detect_screen_size_mm.cache_clear()
    monkeypatch.setattr(DS, "_from_edid", lambda: (700.0, 393.0))
    try:
        mapper = _mapper(_tracker(_Clock()))     # 531 x 299 를 적어 둔 상태
        assert math.degrees(math.atan(mapper._orientation_tan_x)) == pytest.approx(
            GEOMETRIC_SPAN_X_DEG, abs=1e-6)
    finally:
        DS.detect_screen_size_mm.cache_clear()
