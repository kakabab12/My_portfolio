"""가동범위 측정 검증 (2026-09-05 신설).

목이 조금밖에 안 돌아가는 사람은 화면 가장자리에 못 닿는다 — 좌우 7도면
화면 폭의 46%까지다. 이 제품이 겨냥하는 사용자가 정확히 그런 경우라
정확도가 아니라 "쓸 수 있냐"의 문제다.

고치려면 도달 배율을 올리면 되는데 그 값을 아무도 모른다. reach_measure는
**측정해서 알려 준다.** 자동으로 적용하지는 않는다 — 관측만으로는 "못 돌리는
사람"과 "가운데만 쓴 사람"이 안 갈리기 때문이다(모듈 독스트링 참고).

여기서 보증하는 것
------------------
  1) 실제로 돌린 각도를 맞게 측정한다
  2) 표본이 모자라면 **모른다고 한다** (지어내지 않는다)
  3) 이미 끝까지 닿는 사람에게는 1.0을 권한다
  4) 떨림이 크면 덜 권한다 — 닿는 범위를 얻자고 못 쓸 만큼 떨게 하지 않는다
  5) 떨림을 측정 못 했으면 권하지 않는다
  6) 상한을 넘겨 권하지 않고, 1.0 아래로도 안 내려간다
  7) 가로·세로를 따로 측정한다 (가로는 넓고 세로는 좁은 사람이 흔하다)
  8) 천장 지표를 보고하되, 그것만으로 판정하지 않는다는 사실이 코드에 남아 있다
"""
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess import reach_measure as RM                 # noqa: E402
from src.postprocess.reach_measure import ReachMeasure, _Axis   # noqa: E402

SPAN_X = math.tan(math.radians(15.0))
SPAN_Y = math.tan(math.radians(10.0))


def _excursions(axis, peak_degs, noise=0.0004, still_frames=20, seed=7):
    """중심에서 나갔다 돌아오기를 되풀이한다 (사이에 가만히 있는 구간 포함)."""
    rng = random.Random(seed)
    for deg in peak_degs:
        peak = math.tan(math.radians(deg))
        for frac in (0.35, 0.7, 1.0, 1.0, 0.7, 0.35, 0.05):
            axis.add(peak * frac + rng.gauss(0.0, noise))
        for _ in range(still_frames):
            axis.add(rng.gauss(0.0, noise))


def _ceiling(deg, n=40, spread=0.6, seed=3):
    """물리적 한계에 부딪히는 사람 — 늘 거의 같은 데서 멈춘다."""
    rng = random.Random(seed)
    return [deg + rng.gauss(0.0, spread) for _ in range(n)]


# ── 1. 실제로 돌린 각도를 맞게 측정한다 ────────────────────────────────────────

def test_measures_the_angle_actually_turned():
    axis = _Axis(SPAN_X)
    _excursions(axis, _ceiling(7.0, n=40))
    assert axis.reach_deg() == pytest.approx(7.0, abs=1.2)
    assert axis.reach_ratio() == pytest.approx(0.458, abs=0.09)


def test_reach_ratio_matches_the_table_in_the_docstring():
    """독스트링의 표(5도=32.7%, 7도=45.8%, 10도=65.8%)가 실제 계산과 맞는가."""
    for deg, expected in ((5.0, 0.327), (7.0, 0.458), (10.0, 0.658)):
        assert math.tan(math.radians(deg)) / SPAN_X == pytest.approx(expected, abs=0.002)


# ── 2. 모르면 모른다고 한다 ────────────────────────────────────────────────

def test_says_it_cannot_tell_without_enough_excursions():
    axis = _Axis(SPAN_X)
    _excursions(axis, _ceiling(7.0, n=RM.MIN_PEAKS - 3))
    assert axis.reach_deg() is None
    gain, why = axis.recommended_gain()
    assert gain is None
    assert "왕복" in why


def test_says_it_cannot_tell_without_stillness():
    """떨림을 측정 못 했으면 권하지 않는다 — 예산을 모르는 채로 늘리지 않는다."""
    axis = _Axis(SPAN_X)
    _excursions(axis, _ceiling(7.0, n=40), still_frames=0)
    gain, why = axis.recommended_gain()
    assert gain is None
    assert "떨림" in why


def test_empty_input_does_not_crash():
    axis = _Axis(SPAN_X)
    for bad in (float("nan"), float("inf"), -float("inf")):
        axis.add(bad)
    assert axis.reach_deg() is None
    assert axis.jitter_ratio() is None
    assert axis.ceiling_tightness() is None
    assert axis.recommended_gain()[0] is None


# ── 3~5. 권장값이 말이 되는가 ──────────────────────────────────────────────

def test_full_range_user_is_told_to_change_nothing():
    axis = _Axis(SPAN_X)
    _excursions(axis, _ceiling(14.5, n=40))
    gain, why = axis.recommended_gain()
    assert gain == 1.0
    assert "이미" in why


def test_limited_user_gets_a_gain_that_reaches_the_edge():
    axis = _Axis(SPAN_X)
    _excursions(axis, _ceiling(7.0, n=60))
    gain, _why = axis.recommended_gain()
    assert gain is not None
    reached = min(1.0, axis.reach_ratio() * gain)
    assert reached == pytest.approx(1.0, abs=0.02), (
        f"권장값 {gain:.2f}를 넣어도 화면의 {reached:.0%}까지밖에 안 간다")


def test_noise_shrinks_the_recommendation():
    """떨림이 큰 상황에서는 덜 권한다."""
    quiet = _Axis(SPAN_X)
    _excursions(quiet, _ceiling(6.0, n=70), noise=0.0004)
    noisy = _Axis(SPAN_X)
    _excursions(noisy, _ceiling(6.0, n=70), noise=0.003)
    gq, _ = quiet.recommended_gain()
    gn, why = noisy.recommended_gain()
    assert gn < gq, f"떨림이 7.5배인데 같은 값을 권했다: {gq:.2f} vs {gn:.2f}"
    assert "떨림 예산" in why


# ── 6. 경계 ───────────────────────────────────────────────────────────────

def test_recommendation_stays_within_bounds():
    axis = _Axis(SPAN_X)
    _excursions(axis, _ceiling(4.0, n=90, spread=0.2), noise=0.00002)
    gain, _ = axis.recommended_gain()
    assert 1.0 <= gain <= RM.MAX_GAIN


def test_never_recommends_below_one():
    """감도를 낮추는 방향으로는 권하지 않는다."""
    for deg in (4.0, 9.0, 14.0, 14.9):
        axis = _Axis(SPAN_X)
        _excursions(axis, _ceiling(deg, n=50))
        gain, _ = axis.recommended_gain()
        if gain is not None:
            assert gain >= 1.0


# ── 7. 축을 따로 본다 ─────────────────────────────────────────────────────

def test_axes_are_measured_separately():
    """가로는 넓고 세로는 좁은 사람 — 축마다 다른 값이 나와야 한다."""
    m = ReachMeasure(SPAN_X, SPAN_Y)
    rng = random.Random(5)
    for i in range(60):
        wide = math.tan(math.radians(14.0 + rng.gauss(0, 0.5)))
        narrow = math.tan(math.radians(3.5 + rng.gauss(0, 0.3)))
        for frac in (0.4, 0.8, 1.0, 1.0, 0.6, 0.2, 0.02):
            m.add(wide * frac + rng.gauss(0, 3e-4),
                  narrow * frac + rng.gauss(0, 3e-4))
        for _ in range(18):
            m.add(rng.gauss(0, 3e-4), rng.gauss(0, 3e-4))
    report = m.report()
    assert report["x"]["recommended_gain"] == 1.0          # 가로는 이미 닿는다
    assert report["y"]["recommended_gain"] > 1.4           # 세로는 막혔다
    assert report["y"]["reach_deg"] == pytest.approx(3.5, abs=0.8)


# ── 8. 천장 지표는 보고만 한다 ─────────────────────────────────────────────

def test_ceiling_tightness_is_reported_but_does_not_decide():
    """천장 지표가 판정에 안 쓰인다는 것이 이 프로젝트의 결론이다.

    두 사람의 지표가 겹치기 때문이다 — 그 사실을 여기서 다시 확인한다.
    겹치지 않게 되는 날이 오면 이 시험이 먼저 깨져서 알려 줄 것이다.
    """
    blocked = _Axis(SPAN_X)
    _excursions(blocked, _ceiling(7.0, n=60, spread=0.6))
    rng = random.Random(9)
    just_lazy = _Axis(SPAN_X)
    _excursions(just_lazy, [rng.uniform(4.0, 11.0) for _ in range(60)])

    t_blocked = blocked.ceiling_tightness()
    t_lazy = just_lazy.ceiling_tightness()
    assert t_blocked is not None and t_lazy is not None
    # 방향은 맞다 — 막힌 쪽이 더 몰려 있다
    assert t_blocked > t_lazy
    # 그런데 그 차이가 작아서 문턱 하나로 못 가른다. 이것이 자동 판정을
    # 넣지 않은 이유다 (reach_measure.py 독스트링의 측정)
    assert t_blocked - t_lazy < 0.12, (
        f"이제 잘 갈리는가? 막힘 {t_blocked:.3f} 게으름 {t_lazy:.3f} — "
        "그렇다면 자동 판정을 다시 검토할 것")
    # 그럼에도 **권장값은 두 사람에게 다르게 나오면 안 된다** — 지표를
    # 판정에 안 쓰므로, 둘 다 "닿는 데까지"를 기준으로 권하게 된다.
    # 그래서 사람이 보고 정하라고 이유 문구를 함께 준다
    assert blocked.recommended_gain()[1]
    assert just_lazy.recommended_gain()[1]
