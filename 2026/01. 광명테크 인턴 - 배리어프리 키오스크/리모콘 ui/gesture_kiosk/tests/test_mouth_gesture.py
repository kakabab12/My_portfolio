# -*- coding: utf-8 -*-
"""입 벌리기 클릭·드래그 판정 (2026-09-09~10 신설).

여기 있는 시험 대부분은 **실기 보고에서 왔다.** 2026-09-09에 사용자가 두 번
보고했고, 두 번째는 첫 번째 수정이 만든 문제였다.

    1차: "입 벌리면 커서 움직이고, 커서가 파란색이 아닌데 드래그가 되네"
    2차: "입 벌릴 때 클릭 노란색으로 안 뜨더라. 한 번 클릭 하려고 하는데
          바로 파란색으로 변하고 드래그 되더라"

두 번 다 코드만 읽고 원인을 짐작해 고쳤다. 그래서 판정을 모듈로 꺼내고
(src/postprocess/mouth_gesture.py) 가상 사용자로 **직접 써 보는** 방식으로
바꿨다(tests/virtual_user.py, tests/test_virtual_user_click.py).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.mouth_gesture import (                     # noqa: E402
    CLICK, HOLD_END, HOLD_START, PRESS, ClickFreeze, MouthGesture)

# forehead.py에 들어 있는 값 그대로
OPEN, CLOSE, HOLD = 0.12, 0.05, 0.7
CONFIRM, REL_MARGIN, REL_CONFIRM = 0.08, 0.03, 0.20
FPS = 30.0
BASE = 0.05


def _gesture(**kwargs):
    opts = dict(close_fraction=0.55, hold_release_fraction=0.75,
                rest_window_sec=6.0, stuck_open_sec=8.0)
    opts.update(kwargs)
    return MouthGesture(OPEN, CLOSE, HOLD, CONFIRM, REL_MARGIN, REL_CONFIRM,
                        **opts)


def _play(gesture, profile, start_sec=0.0):
    """(초, 턱벌림) 구간들을 프레임으로 펼쳐 먹인다 -> [(시각, 이름표)]."""
    seen = []
    now = start_sec
    for seconds, jaw in profile:
        for _ in range(max(1, int(round(seconds * FPS)))):
            now += 1.0 / FPS
            for event in gesture.update(jaw, BASE, now):
                seen.append((round(now, 3), event))
    return seen


def _names(seen):
    return [name for _, name in seen]


# ── 기본 동작 ──────────────────────────────────────────────────────────────

def test_short_open_is_a_click():
    """짧게 벌렸다 다물면 클릭 — 드래그가 아니다."""
    seen = _play(_gesture(), [(0.5, BASE), (0.3, 0.45), (0.5, BASE)])
    assert _names(seen) == [PRESS, CLICK]


def test_long_open_becomes_a_drag():
    """계속 벌리고 있으면 드래그로 넘어가고, 다물면 끝난다."""
    seen = _play(_gesture(), [(0.5, BASE), (1.5, 0.45), (0.5, BASE)])
    assert _names(seen) == [PRESS, HOLD_START, HOLD_END]


def test_drag_starts_after_hold_sec():
    """드래그 전환은 hold_sec에 일어난다 — 그 전에는 아직 클릭 후보다."""
    seen = _play(_gesture(), [(0.2, BASE), (1.2, 0.45), (0.4, BASE)])
    press_sec = next(t for t, n in seen if n == PRESS)
    hold_sec = next(t for t, n in seen if n == HOLD_START)
    assert hold_sec - press_sec == pytest.approx(HOLD, abs=1.5 / FPS)


def test_jitter_at_the_threshold_does_not_fire_twice():
    """임계값 근처에서 턱이 떨어도 열림/닫힘이 연달아 나오면 안 된다."""
    profile = [(0.3, BASE)]
    for _ in range(10):
        profile += [(1.0 / FPS, BASE + OPEN - 0.005),
                    (1.0 / FPS, BASE + OPEN + 0.005)]
    seen = _play(_gesture(), profile + [(0.5, BASE)])
    assert _names(seen).count(PRESS) == 1


def test_open_margin_must_exceed_close_margin():
    """히스테리시스가 뒤집힌 설정은 조용히 이상하게 돌기보다 죽는 게 낫다."""
    with pytest.raises(ValueError):
        MouthGesture(0.05, 0.05, HOLD, CONFIRM, REL_MARGIN, REL_CONFIRM)


# ── 2026-09-09 실기 보고: 다물어도 클릭이 안 되고 드래그로 갇힘 ────────────
#
# 사람은 클릭한 뒤 입을 완전히 원래대로 안 다문다. 다묾 선(기준선+0.05=0.10)
# 위에 턱이 머무르면
#   클릭 확정 안 됨(노란 깜빡임 없음) -> 0.7초 뒤 드래그(파란색)
#   -> 드래그 해제는 더 빡빡해서 역시 안 걸림 -> 버튼이 눌린 채로 남는다.
# 고치기 전에는 잔여 0.11부터 전부 이랬다.

@pytest.mark.parametrize("residual", [0.06, 0.08, 0.10, 0.11, 0.12, 0.15, 0.16])
def test_click_works_even_if_mouth_does_not_fully_close(residual):
    """잔여 턱이 **벌림 임계값 아래**면 곧바로 깨끗한 클릭이어야 한다.

    임계값은 기준선 0.05 + 0.12 = 0.17이다. 고치기 전에는 0.11부터 전부
    갇혔다 — 다묾 선 0.10을 안 넘어서.
    """
    seen = _play(_gesture(), [(0.5, BASE), (0.3, 0.45), (2.0, residual)])
    assert _names(seen) == [PRESS, CLICK], (
        "다문 뒤 턱이 %.2f 남으면 클릭이 안 되고 갇힌다" % residual)


@pytest.mark.parametrize("residual", [0.18, 0.22, 0.30])
def test_resting_above_the_open_threshold_settles_down(residual):
    """다문 턱이 벌림 임계값보다 **높은** 사람.

    이건 곧바로는 못 고친다 — 다물고 있는 것이 정말 다문 것인지 판단할
    근거가 아직 없기 때문이다. 처음 몇 초는 헛발화가 나올 수 있다.
    중요한 것은 **결국 가라앉고, 버튼이 눌린 채로 남지 않는다**는 것이다.
    """
    gesture = _gesture()
    seen = _play(gesture, [(0.5, BASE), (0.3, 0.45), (15.0, residual)])
    assert not gesture.is_open, "%.2f에서 계속 벌린 것으로 본다" % residual
    assert not gesture.is_holding
    # 눌린 채로 끝나지 않는다 — 누른 횟수만큼 놓았다
    downs = _names(seen).count(PRESS)
    ups = _names(seen).count(CLICK) + _names(seen).count(HOLD_END)
    assert downs == ups, "누른 횟수(%d)와 놓은 횟수(%d)가 다르다" % (downs, ups)


def test_slightly_open_mouth_still_needs_a_real_open_to_click():
    """잔여 턱이 남았다고 저절로 클릭이 나오면 안 된다(반대쪽 오류)."""
    seen = _play(_gesture(), [(3.0, 0.12)])
    assert seen == []


def test_relative_close_does_not_fire_on_a_shallow_open():
    """살짝만 벌린 경우 — 상대 조건이 절대 조건보다 느슨해지면 안 된다.

    최대치가 임계값 언저리면 '최대치의 55%'는 기준선 바로 위라서, 그것만
    쓰면 벌리자마자 닫힌다. 둘 중 덜 관대한 쪽을 써야 한다.
    """
    gesture = _gesture()
    seen = _play(gesture, [(0.3, BASE), (0.5, BASE + OPEN + 0.005)])
    assert _names(seen) == [PRESS], "살짝 벌린 채로 있는데 다묾이 나왔다"


# ── 눌린 채 갇히지 않는다 ──────────────────────────────────────────────────

def test_stuck_open_is_released():
    """다문 턱이 벌림 임계값보다 높은 사람 — 버튼을 눌린 채 두면 안 된다.

    사용자가 할 수 있는 일이 아무것도 없어지기 때문이다. 이보다 긴 드래그를
    포기하더라도 놓는다.
    """
    seen = _play(_gesture(stuck_open_sec=2.0),
                 [(0.3, BASE), (5.0, 0.45)])
    assert HOLD_END in _names(seen)
    hold_end_sec = next(t for t, n in seen if n == HOLD_END)
    press_sec = next(t for t, n in seen if n == PRESS)
    assert hold_end_sec - press_sec == pytest.approx(2.0, abs=2.0 / FPS)


def test_long_drag_is_not_cut_off_early():
    """안전장치가 정상적인 드래그를 끊으면 안 된다."""
    seen = _play(_gesture(stuck_open_sec=8.0),
                 [(0.3, BASE), (5.0, 0.45), (0.5, BASE)])
    assert _names(seen) == [PRESS, HOLD_START, HOLD_END]


def test_rest_level_follows_a_higher_resting_jaw():
    """다문 값이 통째로 올라간 사람도 결국 정상으로 돌아와야 한다.

    기준선은 캘리브레이션 때 한 번 정해지면 안 바뀐다(head_tracker의
    _MedianCalibrator). 그 뒤에 사람의 다문 턱이 올라가면, 다물고 있는데도
    계속 벌린 것으로 판정된다.
    """
    gesture = _gesture()
    # 벌림 임계(0.17)보다 높은 0.20에 머무른다 — 창(6초)이 채워지면 잡힌다
    _play(gesture, [(0.3, BASE), (0.3, 0.45), (10.0, 0.20)])
    assert not gesture.is_open, "다문 값이 올라간 뒤에도 계속 벌린 것으로 본다"
    # 이제 이 사람 기준으로 다시 클릭이 되어야 한다
    seen = _play(gesture, [(0.3, 0.55), (1.0, 0.20)], start_sec=11.0)
    assert _names(seen) == [PRESS, CLICK]


def test_reset_clears_everything():
    gesture = _gesture()
    _play(gesture, [(0.3, BASE), (1.2, 0.45)])
    assert gesture.is_holding
    gesture.reset()
    assert not gesture.is_open and not gesture.is_holding
    assert gesture.held_sec is None


def test_no_judgement_before_baseline_is_ready():
    """기준선을 아직 못 잡았으면 아무 판정도 하지 않는다."""
    gesture = _gesture()
    assert gesture.update(0.9, None, 1.0) == ()
    assert not gesture.is_open


# ── 커서 붙잡기 ────────────────────────────────────────────────────────────

def test_cursor_is_pinned_while_clicking():
    """누르는 동안 커서가 움직이면 그건 의도치 않은 드래그다."""
    freeze = ClickFreeze(lookback_sec=0.08)
    for i in range(20):                       # 겨누는 동안의 기록
        freeze.apply(i / FPS, 0.5, 0.5)
    freeze.begin(20 / FPS)
    seen = {freeze.apply((20 + i) / FPS, 0.5 + 0.01 * i, 0.5) for i in range(10)}
    assert seen == {(0.5, 0.5)}, "붙잡는 동안 커서가 움직였다"


def test_pinned_point_is_from_before_the_mouth_opened():
    """붙잡는 지점은 **되짚어** 잡는다 — 판정 시점엔 이미 밀린 뒤다."""
    freeze = ClickFreeze(lookback_sec=0.10)
    for i in range(30):
        freeze.apply(i / FPS, i / 100.0, 0.5)   # 커서가 꾸준히 오른쪽으로
    freeze.begin(30 / FPS)
    x, _ = freeze.apply(30 / FPS, 0.30, 0.5)
    # 0.10초 = 3프레임 전 = 0.27 근처. 지금 값(0.30)이면 되짚기가 안 된 것
    assert x == pytest.approx(0.27, abs=0.02)


def test_drag_release_is_blended_not_jumped():
    """드래그로 넘어갈 때 커서가 튀면 안 된다."""
    freeze = ClickFreeze(lookback_sec=0.0, unfreeze_sec=0.25)
    freeze.apply(0.0, 0.5, 0.5)
    freeze.begin(0.0)
    freeze.end(0.0, blend=True)
    first, _ = freeze.apply(1.0 / FPS, 0.9, 0.5)
    assert 0.5 < first < 0.62, "붙잡던 곳에서 곧바로 튀었다"
    last, _ = freeze.apply(0.30, 0.9, 0.5)
    assert last == pytest.approx(0.9), "이어 주기가 안 끝났다"


def test_click_release_does_not_blend():
    """클릭은 커서를 옮기지 않는다 — 이어 줄 것도 없다."""
    freeze = ClickFreeze(lookback_sec=0.0)
    freeze.apply(0.0, 0.5, 0.5)
    freeze.begin(0.0)
    freeze.end(0.0, blend=False)
    assert freeze.apply(1.0 / FPS, 0.9, 0.5) == (0.9, 0.5)


def test_freeze_can_be_turned_off():
    freeze = ClickFreeze(enabled=False)
    freeze.apply(0.0, 0.5, 0.5)
    freeze.begin(0.0)
    assert freeze.apply(1.0 / FPS, 0.9, 0.7) == (0.9, 0.7)


def test_freeze_with_no_history_does_not_crash():
    """첫 프레임에 바로 입을 벌린 경우 — 되짚을 기록이 없다."""
    freeze = ClickFreeze()
    freeze.begin(0.0)
    assert freeze.apply(0.0, 0.4, 0.6) == (0.4, 0.6)
