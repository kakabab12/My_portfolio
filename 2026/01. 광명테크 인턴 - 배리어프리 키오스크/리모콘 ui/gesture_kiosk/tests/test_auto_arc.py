"""잔여 곡률 자동 소거(auto_arc) 검증 (2026-08-31 신설).

핵심 주장 두 가지를 지킨다.
  1) 계통 오차(포물선)는 시간이 지나면 스스로 사라진다.
  2) 의도한 움직임(대각선·원·세로)은 건드리지 않는다.

둘 다 카메라 없이 합성 궤적으로 확인한다 — 곡률을 정확히 아는 값으로 넣고
빠지는지 보면 된다.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.auto_arc import (   # noqa: E402
    MAX_COEF, MIN_SAMPLES, OnlineArcCompensator,
)


def _sweep(step):
    """좌우 왕복 x 좌표 — 실사용처럼 사인파로 훑는다 (탄젠트 단위, ±0.25)."""
    return 0.25 * math.sin(step * 0.13)


def test_systematic_parabola_is_learned_and_removed():
    """★계통 곡률 y = c·x² 이 몇 창 안에 소거돼야 한다."""
    true_c = 0.8
    comp = OnlineArcCompensator()
    residuals = []
    for step in range(MIN_SAMPLES * 6):
        x = _sweep(step)
        y_raw = true_c * x * x                    # 순수한 계통 오차
        residuals.append(abs(comp.update(x, y_raw)))
    # 수렴 후: 계수가 참값 근처, 마지막 구간의 잔여 오차가 원래의 1/5 이하
    assert comp.coef == pytest.approx(true_c, rel=0.15)
    tail = residuals[-MIN_SAMPLES:]
    worst_raw = true_c * 0.25 * 0.25
    assert max(tail) < worst_raw * 0.2


def test_negative_curvature_also_removed():
    """∪가 아니라 ∩로 휘어도 (계수 음수) 똑같이 배운다."""
    comp = OnlineArcCompensator()
    for step in range(MIN_SAMPLES * 6):
        x = _sweep(step)
        comp.update(x, -0.6 * x * x)
    assert comp.coef == pytest.approx(-0.6, rel=0.15)


def test_diagonal_motion_is_not_treated_as_curvature():
    """대각선 이동(1차)은 c에 실리면 안 된다 — 1차항이 흡수한다."""
    comp = OnlineArcCompensator()
    for step in range(MIN_SAMPLES * 6):
        x = _sweep(step)
        comp.update(x, 0.7 * x)                   # 순수 대각선
    assert abs(comp.coef) < 0.05


def test_circles_are_not_treated_as_curvature():
    """원 그리기(의도)는 위아래가 대칭이라 2차 적합이 상쇄돼야 한다."""
    comp = OnlineArcCompensator()
    for step in range(MIN_SAMPLES * 6):
        t = step * 0.11
        comp.update(0.22 * math.cos(t), 0.22 * math.sin(t))
    assert abs(comp.coef) < 0.05


def test_vertical_only_motion_never_updates():
    """세로로만 움직이면 x 폭이 없어 갱신 자체가 없어야 한다."""
    comp = OnlineArcCompensator()
    for step in range(MIN_SAMPLES * 4):
        y = 0.3 * math.sin(step * 0.1)
        out = comp.update(0.0, y)
        assert out == y                           # 보정 0 그대로
    assert comp.coef == 0.0


def test_noise_only_does_not_grow_a_coefficient():
    """포물선이 아닌 잡음(낮은 R²)으로는 계수를 만들지 않는다 — 안전장치 ⑤."""
    import random
    rng = random.Random(7)
    comp = OnlineArcCompensator()
    for step in range(MIN_SAMPLES * 6):
        comp.update(_sweep(step), rng.uniform(-0.05, 0.05))
    assert abs(comp.coef) < 0.05


def test_coefficient_is_clamped():
    """병적인 표본이 와도 계수가 상한을 넘지 않는다 — 안전장치 ④."""
    comp = OnlineArcCompensator()
    for step in range(MIN_SAMPLES * 8):
        x = _sweep(step)
        comp.update(x, 50.0 * x * x)              # 말도 안 되는 곡률
    assert abs(comp.coef) <= MAX_COEF + 1e-9


def test_no_feedback_runaway():
    """보정된 값이 아니라 원본으로 배우므로, 계수가 계속 자라면 안 된다.

    보정 후 값으로 학습하면 (이미 뺀 것을 또 없다고 판단해) 계수가 0으로
    되돌아가거나 진동한다 — 그 되먹임이 없는지 오래 돌려 확인한다.
    """
    true_c = 0.5
    comp = OnlineArcCompensator()
    coefs = []
    for step in range(MIN_SAMPLES * 12):
        x = _sweep(step)
        comp.update(x, true_c * x * x)
        coefs.append(comp.coef)
    # 마지막 두 구간의 계수가 사실상 같아야 한다 (수렴 후 안정)
    assert coefs[-1] == pytest.approx(coefs[-MIN_SAMPLES], abs=0.02)
    assert coefs[-1] == pytest.approx(true_c, rel=0.15)


def test_reset_clears_everything():
    comp = OnlineArcCompensator()
    for step in range(MIN_SAMPLES * 4):
        x = _sweep(step)
        comp.update(x, 0.8 * x * x)
    assert comp.coef != 0.0
    comp.reset()
    assert comp.coef == 0.0
    assert comp.update(0.1, 0.05) == 0.05         # 보정 없이 그대로


def test_mixed_intent_plus_systematic_learns_only_systematic():
    """실사용 조합 — 의도(대각선+세로 사인) 위에 계통 곡률이 얹힌 경우,
    계통 곡률만 배워야 한다."""
    true_c = 0.6
    comp = OnlineArcCompensator()
    for step in range(MIN_SAMPLES * 8):
        x = _sweep(step)
        intent = 0.4 * x + 0.1 * math.sin(step * 0.031)   # 대각선 + 느린 세로
        comp.update(x, intent + true_c * x * x)
    assert comp.coef == pytest.approx(true_c, rel=0.3)
