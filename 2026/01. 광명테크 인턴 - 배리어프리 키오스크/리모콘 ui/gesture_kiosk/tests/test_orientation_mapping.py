"""HeadTracker에 붙인 상대 회전 매핑 통합 검증 (2026-08-31 신설).

head_orientation 단위 테스트가 알고리즘 자체를 보증한다면, 여기서는 그것이
**커서까지 실제로 이어지는지**를 본다. 설정 키 하나만 켜면 되고, 카메라를
어떻게 달아도 같은 커서가 나와야 한다.
"""
import copy
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_orientation import RIGID_LANDMARKS   # noqa: E402
from src.postprocess.head_tracker import HeadTracker           # noqa: E402
from src.utils.config_loader import load_config                # noqa: E402

from tests.test_head_orientation import (                      # noqa: E402
    _apply, _rot_about, _synthetic_head,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml"
)
LMK_LEFT_EYE_OUTER = 33
LMK_RIGHT_EYE_OUTER = 263


class _Face:
    """HeadTracker.update가 읽는 것만 갖춘 얼굴."""

    def __init__(self, points_3d):
        self.landmarks_3d = points_3d
        self.landmarks_px = np.asarray(points_3d, dtype=np.float32)[:, :2]
        self.blendshapes = {}

    def landmark_px(self, index):
        x, y = self.landmarks_px[index]
        return float(x), float(y)

    def landmarks_mean_px(self, indices):
        pts = self.landmarks_px[list(indices)]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def blendshape(self, name, default=0.0):
        return self.blendshapes.get(name, default)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _make_tracker(clock, half_span_x=15.0, half_span_y=10.0):
    config = copy.deepcopy(load_config(CONFIG_PATH))
    pointer = config["head_tracker"]["pointer"]
    pointer["orientation_mapping"] = True
    pointer["orientation_half_span_x_deg"] = half_span_x
    pointer["orientation_half_span_y_deg"] = half_span_y
    pointer["one_euro_enabled"] = False
    pointer["smoothing_alpha"] = 1.0          # 평활 없음 — 매핑만 본다
    return HeadTracker(config, clock=clock)


def _calibrate(tracker, clock, face, window_sec=3.0):
    """캘리브레이션 구간을 지나 커서가 확정될 때까지 먹인다."""
    for _ in range(40):
        tracker.update(face)
        clock.now += window_sec / 20.0
    return tracker.update(face)


def test_cursor_starts_at_center_after_calibration():
    clock = _Clock()
    tracker = _make_tracker(clock)
    result = _calibrate(tracker, clock, _Face(_synthetic_head()))
    assert result.is_tracking
    assert result.cursor_x_ratio == pytest.approx(0.5, abs=1e-6)
    assert result.cursor_y_ratio == pytest.approx(0.5, abs=1e-6)


def test_half_span_angle_puts_cursor_at_screen_edge():
    """설계값 그대로 — 정한 각도만큼 돌리면 커서가 정확히 화면 끝에 닿는다.

    이것이 "재지 않아도 되는" 이유다. 감도가 추상적인 배율이 아니라
    '몇 도 돌리면 끝'이라는 사람이 정하는 숫자다.
    """
    clock = _Clock()
    neutral = _synthetic_head()
    half_span = 15.0
    tracker = _make_tracker(clock, half_span_x=half_span)
    _calibrate(tracker, clock, _Face(neutral))

    from src.postprocess.head_orientation import _orthonormal_frame
    _x, y_axis, _z = _orthonormal_frame(neutral[list(RIGID_LANDMARKS)])

    clock.now += 0.1
    turned = _apply(_rot_about(y_axis, half_span), neutral)
    result = tracker.update(_Face(turned))
    # 중앙에서 half_span 만큼 돌리면 오프셋이 0.5 = 화면 끝.
    # 어느 쪽 끝인지(부호)는 회전 방향에 달렸고 그건 head_orientation의
    # 단위 테스트가 코 위치로 이미 검증한다 — 여기서는 크기만 본다.
    # max_offset_ratio에 먼저 걸릴 수 있으므로 그 한계와 비교한다
    limit = min(0.5, load_config(CONFIG_PATH)["head_tracker"]["pointer"]["max_offset_ratio"])
    assert abs(result.cursor_x_ratio - 0.5) == pytest.approx(limit, rel=1e-6)
    assert result.cursor_y_ratio == pytest.approx(0.5, abs=1e-6)   # 세로는 안 움직인다


def test_camera_mount_does_not_change_the_cursor():
    """★카메라를 어떻게 달아도 커서가 같아야 한다 (이 작업의 목표).

    카메라 배치를 바꾸는 것 = 보이는 모든 점에 같은 회전을 더 거는 것.
    중립도 그 카메라로 잡히므로 상쇄돼야 한다.
    """
    neutral = _synthetic_head()
    turn = _rot_about((0.0, 1.0, 0.0), 11.0)
    turned = _apply(turn, neutral)

    def cursor_under(mount):
        clock = _Clock()
        tracker = _make_tracker(clock)
        _calibrate(tracker, clock, _Face(_apply(mount, neutral)))
        clock.now += 0.1
        r = tracker.update(_Face(_apply(mount, turned)))
        return r.cursor_x_ratio, r.cursor_y_ratio

    baseline = cursor_under(np.eye(3))
    mounts = {
        "밑에서 올려봄": _rot_about((1.0, 0.0, 0.0), 25.0),
        "위에서 내려봄": _rot_about((1.0, 0.0, 0.0), -28.0),
        "옆으로 기울어짐": _rot_about((0.0, 0.0, 1.0), 15.0),
        "비스듬히": _rot_about((0.4, -0.6, 0.35), 20.0),
    }
    for label, mount in mounts.items():
        got = cursor_under(mount)
        assert got[0] == pytest.approx(baseline[0], abs=1e-6), label
        assert got[1] == pytest.approx(baseline[1], abs=1e-6), label


def test_arc_compensation_is_not_used_on_this_path():
    """곡률 보정 값을 넣어도 이 경로에서는 커서가 달라지지 않아야 한다.

    투영을 안 거치니 보정할 왜곡이 없다 — 카메라를 옮길 때마다 이 값을
    다시 재던 일이 사라진다는 뜻이다.
    """
    neutral = _synthetic_head()
    turn = _rot_about((0.0, 1.0, 0.0), 12.0)

    def cursor_with(arc):
        clock = _Clock()
        config = copy.deepcopy(load_config(CONFIG_PATH))
        pointer = config["head_tracker"]["pointer"]
        pointer["orientation_mapping"] = True
        pointer["arc_compensation"] = arc
        pointer["one_euro_enabled"] = False
        pointer["smoothing_alpha"] = 1.0
        tracker = HeadTracker(config, clock=clock)
        _calibrate(tracker, clock, _Face(neutral))
        clock.now += 0.1
        r = tracker.update(_Face(_apply(turn, neutral)))
        return r.cursor_x_ratio, r.cursor_y_ratio

    assert cursor_with(0.0) == cursor_with(-0.9)


def test_falls_back_when_3d_landmarks_missing():
    """3차원 좌표가 안 오면 조용히 기존 방식으로 되돌아가야 한다.

    모델이나 mediapipe 버전이 바뀌어 z가 안 오는 상황에서 커서가 죽으면 안 된다.
    """
    class NoDepthFace(_Face):
        def __init__(self, pts):
            super().__init__(pts)
            self.landmarks_3d = None

    clock = _Clock()
    tracker = _make_tracker(clock)
    result = _calibrate(tracker, clock, NoDepthFace(_synthetic_head()))
    # 기존(2D) 경로로 넘어가 커서가 나온다 — 죽지 않는다
    assert result is not None


def test_larger_half_span_needs_more_turning():
    """화면 끝까지 필요한 각도를 키우면 같은 회전에서 커서가 덜 움직인다."""
    neutral = _synthetic_head()
    turn = _rot_about((0.0, 1.0, 0.0), 10.0)

    def offset_for(half_span):
        clock = _Clock()
        tracker = _make_tracker(clock, half_span_x=half_span)
        _calibrate(tracker, clock, _Face(neutral))
        clock.now += 0.1
        return abs(tracker.update(_Face(_apply(turn, neutral))).cursor_x_ratio - 0.5)

    assert offset_for(25.0) < offset_for(12.0)


def test_brief_loss_keeps_neutral():
    """★잠깐 놓친 것으로 중립을 버리면 안 된다 (2026-08-31 실기 보고 대응).

    기울여 단 카메라처럼 검출이 간헐적으로 끊기는 배치에서, 한 프레임 놓칠
    때마다 중립을 버리면 캘리브레이션이 끝나지 않아 커서가 영영 안 나온다.
    """
    clock = _Clock()
    tracker = _make_tracker(clock)
    _calibrate(tracker, clock, _Face(_synthetic_head()))
    assert tracker.update(None).is_tracking is False
    clock.now += 0.1
    # 곧바로 다시 잡히면 중립을 그대로 쓴다 — 커서가 바로 나온다
    assert tracker.update(_Face(_synthetic_head())).cursor_x_ratio is not None


def test_sustained_loss_clears_neutral():
    """오래 안 보이면 새 사용자일 수 있으니 중립을 다시 잡는다."""
    from src.postprocess.head_tracker import FACE_LOST_RESET_SEC

    clock = _Clock()
    tracker = _make_tracker(clock)
    _calibrate(tracker, clock, _Face(_synthetic_head()))
    tracker.update(None)                       # 여기서 미검출 시계가 시작된다
    clock.now += FACE_LOST_RESET_SEC + 0.1
    assert tracker.update(None).is_tracking is False
    clock.now += 0.1
    assert tracker.update(_Face(_synthetic_head())).cursor_x_ratio is None


def test_intermittent_detection_still_produces_a_moving_cursor():
    """★실기 증상 재현 — 검출이 띄엄띄엄이어도 커서가 움직여야 한다.

    3프레임마다 한 번씩 얼굴을 놓치는 상황을 만든다. 예전 동작(즉시 리셋)
    이라면 캘리브레이션이 끝나지 않아 커서가 계속 None이었다.
    """
    neutral = _synthetic_head()
    clock = _Clock()
    tracker = _make_tracker(clock)
    _calibrate(tracker, clock, _Face(neutral))

    seen = []
    for step in range(60):
        clock.now += 1.0 / 30.0
        if step % 3 == 2:
            tracker.update(None)              # 놓친 프레임
            continue
        turned = _apply(_rot_about((0.0, 1.0, 0.0), 8.0 * math.sin(step * 0.25)), neutral)
        result = tracker.update(_Face(turned))
        if result.cursor_x_ratio is not None:
            seen.append(result.cursor_x_ratio)

    assert len(seen) > 30, "커서가 거의 안 나왔습니다"
    assert max(seen) - min(seen) > 0.05, "커서가 움직이지 않았습니다"


# ------------------------------------------- 세 트래커에 빠짐없이 붙어 있는지

TRACKER_MODULES = ("eyebrow", "forehead", "head")


@pytest.mark.parametrize("module_name", TRACKER_MODULES)
def test_every_tracker_enables_orientation_mapping(module_name):
    """eyebrow·forehead·head 세 트래커 모두 상대 회전 매핑을 쓴다.

    셋 다 같은 문제(카메라 배치마다 다시 재기)를 겪었으므로 처방도 같아야 한다.
    새 트래커를 만들면서 이 배선을 빠뜨리면 여기서 걸린다.
    """
    import importlib

    module = importlib.import_module(module_name)
    assert module.ORIENTATION_MAPPING is True
    assert 1.0 <= module.ORIENTATION_HALF_SPAN_X_DEG <= 60.0
    assert 1.0 <= module.ORIENTATION_HALF_SPAN_Y_DEG <= 60.0
    # 세로가 가로보다 좁아야 한다 — 고개는 위아래보다 좌우로 넓게 돌아간다
    assert module.ORIENTATION_HALF_SPAN_Y_DEG <= module.ORIENTATION_HALF_SPAN_X_DEG


@pytest.mark.parametrize("module_name", TRACKER_MODULES)
def test_every_tracker_passes_orientation_settings_to_config(module_name):
    """상수만 있고 config로 안 넘기면 아무 일도 안 일어난다 — 그 배선을 확인한다.

    main() 안에서 대입하므로 실행 없이 확인하려면 소스를 읽는 수밖에 없다.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        module_name + ".py")
    source = open(path, encoding="utf-8").read()
    for key, const in (
        ("orientation_mapping", "ORIENTATION_MAPPING"),
        ("orientation_half_span_x_deg", "ORIENTATION_HALF_SPAN_X_DEG"),
        ("orientation_half_span_y_deg", "ORIENTATION_HALF_SPAN_Y_DEG"),
    ):
        line = 'config["head_tracker"]["pointer"]["%s"] = %s' % (key, const)
        assert line in source, "%s 에 %s 배선이 없습니다" % (module_name, key)


# --------------------------------- 실시간 조절 UI가 이 경로에서도 실제로 먹는가

def test_live_tuning_changes_the_cursor_on_orientation_path():
    """★조절 UI 슬라이더가 상대 회전 경로에서도 실제로 커서를 바꿔야 한다.

    이 경로에서는 예전 감도(sensitivity_x/y)와 곡률 보정이 안 쓰인다. 그래서
    조절 UI가 예전 값만 넘기면 슬라이더를 움직여도 아무 일이 안 일어나
    "고장 난 것처럼" 보인다 — 각도 손잡이를 함께 넘기도록 고친 뒤 그것이
    끝까지 이어지는지 여기서 지킨다.
    """
    neutral = _synthetic_head()
    turn = _rot_about((0.0, 1.0, 0.0), 10.0)
    clock = _Clock()
    tracker = _make_tracker(clock, half_span_x=15.0)
    _calibrate(tracker, clock, _Face(neutral))

    clock.now += 0.1
    before = abs(tracker.update(_Face(_apply(turn, neutral))).cursor_x_ratio - 0.5)

    # 화면 끝까지 필요한 각도를 두 배로 -> 같은 회전에서 커서가 덜 움직여야 한다
    tracker.set_pointer_tuning(half_span_x_deg=30.0)
    clock.now += 0.1
    after = abs(tracker.update(_Face(_apply(turn, neutral))).cursor_x_ratio - 0.5)
    assert after < before


def test_live_tuning_ignores_none_and_clamps_range():
    """None은 건드리지 않고, 말도 안 되는 각도는 안전 범위로 묶어야 한다."""
    clock = _Clock()
    tracker = _make_tracker(clock)
    _calibrate(tracker, clock, _Face(_synthetic_head()))
    tracker.set_pointer_tuning()                       # 전부 None — 죽지 않아야
    tracker.set_pointer_tuning(half_span_x_deg=0.0)    # 0도는 나눗셈 폭발
    tracker.set_pointer_tuning(half_span_y_deg=1e9)
    clock.now += 0.1
    result = tracker.update(_Face(_synthetic_head()))
    assert result.cursor_x_ratio is not None
    assert 0.0 <= result.cursor_x_ratio <= 1.0


def test_tracker_reports_which_mapping_is_active():
    """조절 UI가 어떤 손잡이를 보여줄지 알 수 있어야 한다."""
    clock = _Clock()
    assert _make_tracker(clock).is_orientation_mapping is True

    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["head_tracker"]["pointer"]["orientation_mapping"] = False
    assert HeadTracker(config, clock=clock).is_orientation_mapping is False


def test_first_frame_without_face_does_not_crash():
    """★첫 프레임에 얼굴이 없어도 죽으면 안 된다 (2026-08-31).

    얼굴 미검출 유예를 넣으면서 그 상태를 reset()에서만 만들었더니,
    __init__이 reset()을 부르지 않아 **첫 프레임이 미검출이면 AttributeError**로
    죽었다. 카메라를 켜자마자 사람이 없는 것은 키오스크의 정상 상태다.
    """
    clock = _Clock()
    tracker = _make_tracker(clock)
    result = tracker.update(None)
    assert result.is_tracking is False
    clock.now += 0.1
    assert tracker.update(None).is_tracking is False
