# -*- coding: utf-8 -*-
"""가상 사용자가 실제 파이프라인으로 클릭·드래그를 해 본다 (2026-09-10 신설).

test_mouth_gesture.py는 판정만 따로 본다. 여기서는 **가상 카메라 -> 얼굴
랜드마크 -> HeadTracker -> 커서 좌표 -> 입 판정**을 통째로 통과시킨다.
중간을 흉내 내지 않으므로, 여기서 나오는 결과는 실제로 그렇게 돈다는 뜻이다.

사람이 진짜 쓸 때는 **가만히 있지 않는다.** 클릭하는 도중에도 고개가
움직이고, 입을 매번 같은 깊이로 벌리지도 다물지도 않는다. 그래서 모든
시나리오에 고개 움직임과 미세한 떨림이 들어 있다(tests/virtual_user.py).
"""
import copy
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_tracker import HeadTracker                # noqa: E402
from src.postprocess.mouth_gesture import (                         # noqa: E402
    CLICK, HOLD_END, HOLD_START, PRESS, ClickFreeze, MouthGesture)
from src.utils.config_loader import load_config                     # noqa: E402
from tests.virtual_camera import VirtualCamera, rotation            # noqa: E402
from tests.virtual_user import VirtualUser, aim, mouth_open, rest   # noqa: E402

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "config.yaml")

SCREEN_W_PX, SCREEN_H_PX = 1920, 1080
SCREEN_W_MM, SCREEN_H_MM, REF_DIST_MM = 531.0, 299.0, 700.0

# forehead.py의 값 그대로
OPEN, CLOSE, HOLD = 0.12, 0.05, 0.7
CONFIRM, REL_MARGIN, REL_CONFIRM = 0.08, 0.03, 0.20
LOOKBACK = 0.0        # 0 = 누른 그 자리에 붙잡는다 (트래커 상수 설명 참고)

OPEN_RAMP, CLOSE_RAMP = 0.12, 0.15     # 입은 순간이동하지 않는다
JAW_SHUT, JAW_WIDE = 0.05, 0.45


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _click_actions(plateau_sec, yaw_from, yaw_to, residual=JAW_SHUT):
    """벌렸다 다무는 한 동작. 고개는 그 사이에도 움직인다(사람이니까)."""
    mid = yaw_from + (yaw_to - yaw_from) * 0.5
    return [
        mouth_open(OPEN_RAMP, jaw_open=JAW_WIDE, yaw_deg=mid, label="클릭"),
        mouth_open(plateau_sec, jaw_open=JAW_WIDE, yaw_deg=yaw_to, label="클릭"),
        mouth_open(CLOSE_RAMP, jaw_open=residual, yaw_deg=yaw_to, label="다뭄"),
    ]


def _use(actions, freeze_enabled=True, lookback=LOOKBACK):
    """가상 사용자가 한 번 쓴다 -> 무슨 일이 있었는지."""
    clock = _Clock()
    config = copy.deepcopy(load_config(CONFIG_PATH))
    pointer = config["head_tracker"]["pointer"]
    pointer["orientation_mapping"] = True
    pointer["screen_width_mm"] = SCREEN_W_MM
    pointer["screen_height_mm"] = SCREEN_H_MM
    pointer["reference_distance_mm"] = REF_DIST_MM
    pointer["orientation_lens_calibration"] = False
    tracker = HeadTracker(config, clock=clock)
    camera = VirtualCamera(distance_mm=REF_DIST_MM, lens="일반 65도", noise_px=0.35)
    gesture = MouthGesture(OPEN, CLOSE, HOLD, CONFIRM, REL_MARGIN, REL_CONFIRM)
    freeze = ClickFreeze(lookback_sec=lookback, enabled=freeze_enabled)

    events = []
    button_down = False
    holding_color = False
    travel_while_green = 0.0     # 눌렸는데 파란색이 아닌 동안 커서가 간 거리
    aim_px = None
    click_px = None
    last_px = None

    for now, yaw, pitch, jaw, _label in VirtualUser(actions).frames():
        clock.now = now
        rot = rotation((0, 1, 0), yaw) @ rotation((1, 0, 0), pitch)
        # 입 벌림이 랜드마크를 실제로 밀도록 넣는다(가상 카메라의 expression)
        expression = max(0.0, min(1.0, (jaw - JAW_SHUT) / 0.40))
        face = camera.observe(rot, expression=expression)
        face.blendshapes["jawOpen"] = jaw
        result = tracker.update(face)
        if result is None or not getattr(result, "is_tracking", True):
            continue
        x, y = freeze.apply(now, result.cursor_x_ratio, result.cursor_y_ratio)
        px = (x * SCREEN_W_PX, y * SCREEN_H_PX)
        if not gesture.is_open and click_px is None and jaw <= JAW_SHUT + 0.01:
            aim_px = px          # 입을 벌리기 전에 겨누고 있던 곳

        for event in gesture.update(tracker.debug.get("jaw_open"),
                                    tracker.debug.get("jaw_base"), now):
            events.append(event)
            if event == PRESS:
                freeze.begin(now)
                button_down = True
                # 버튼이 내려간 **뒤부터** 센다 — 그 전 움직임은 그냥 겨누기다
                last_px = px
            elif event == CLICK:
                freeze.end(now, False)
                button_down = False
                holding_color = False
                if click_px is None:
                    click_px = px
            elif event == HOLD_START:
                freeze.end(now, True)
                holding_color = True
            elif event == HOLD_END:
                freeze.end(now, False)
                button_down = False
                holding_color = False

        if button_down and not holding_color and last_px is not None:
            travel_while_green += math.hypot(px[0] - last_px[0], px[1] - last_px[1])
        last_px = px

    return {
        "events": events,
        "travel_while_green_px": travel_while_green,
        "aim_px": aim_px,
        "click_px": click_px,
    }


WARMUP = [rest(3.0, jaw_open=JAW_SHUT, label="기준선 잡는 중")]


# ── 사용자가 보고한 두 가지가 다시 나오지 않는다 ───────────────────────────

@pytest.mark.parametrize("plateau,yaw_to", [
    (0.15, 8.0),      # 빠른 클릭, 고개 정지
    (0.30, 10.0),     # 보통 클릭, 고개 2도
    (0.30, 13.0),     # 보통 클릭, 고개 5도 — 사람은 클릭 중에도 움직인다
    (0.50, 9.0),      # 느긋한 클릭
])
def test_a_click_is_a_click_not_a_drag(plateau, yaw_to):
    """"한 번 클릭 하려고 하는데 바로 파란색으로 변하고 드래그 되더라"."""
    out = _use(WARMUP + [aim(1.0, yaw_deg=8.0)]
               + _click_actions(plateau, 8.0, yaw_to)
               + [rest(1.0, yaw_deg=yaw_to, label="대기")])
    assert out["events"] == [PRESS, CLICK], (
        "클릭이 드래그가 됐다 — %s" % (out["events"],))


@pytest.mark.parametrize("residual", [0.05, 0.09, 0.12, 0.15])
def test_click_survives_a_mouth_that_does_not_fully_close(residual):
    """사람은 클릭한 뒤 입을 완전히 원래대로 안 다문다."""
    out = _use(WARMUP + [aim(1.0, yaw_deg=8.0)]
               + _click_actions(0.3, 8.0, 10.0, residual=residual)
               + [rest(2.0, yaw_deg=13.0, jaw_open=residual, label="덜 다문 채")])
    assert out["events"] == [PRESS, CLICK], (
        "잔여 턱 %.2f에서 클릭이 안 됐다 — %s" % (residual, out["events"]))


def test_no_unintended_drag_while_the_cursor_is_green():
    """"커서가 파란색이 아닌데 드래그가 되는 현상".

    파란색은 드래그일 때만 켠다. 그러니 누른 채 초록색인 동안 커서가
    움직이면 그게 곧 보이지 않는 드래그다 — 붙잡기가 그걸 막는다.
    """
    out = _use(WARMUP + [aim(1.0, yaw_deg=8.0)]
               + _click_actions(0.3, 8.0, 14.0)     # 클릭 중에 고개 6도
               + [rest(1.0, yaw_deg=14.0, label="대기")])
    assert out["travel_while_green_px"] == pytest.approx(0.0, abs=1.0)


def test_click_lands_where_the_user_was_aiming():
    """겨눈 곳에 찍혀야 한다 — 입 벌리느라 얼굴이 밀린 만큼 흘러가면 안 된다."""
    out = _use(WARMUP + [aim(1.0, yaw_deg=8.0)]
               + _click_actions(0.3, 8.0, 12.0)
               + [rest(1.0, yaw_deg=12.0, label="대기")])
    off_px = math.hypot(out["click_px"][0] - out["aim_px"][0],
                        out["click_px"][1] - out["aim_px"][1])
    # 측정값은 고개 속도에 따라 12~25 px (트래커의 CLICK_FREEZE_LOOKBACK_SEC
    # 설명에 표가 있다). 붙잡기를 끄면 같은 조건에서 104 px다.
    # 키오스크 아이콘 한 변보다 훨씬 작으면 된다.
    assert off_px < 30.0, "클릭이 겨눈 곳에서 %.0f px 벗어났다" % off_px


def test_without_the_freeze_the_click_drifts_away():
    """붙잡기가 실제로 일하고 있다는 확인 — 끄면 눈에 띄게 흘러간다.

    이게 실패한다면 붙잡기가 아니라 다른 것이 좋아진 것이므로, 그때는
    붙잡기를 계속 둘 이유를 다시 따져야 한다.
    """
    actions = (WARMUP + [aim(1.0, yaw_deg=8.0)]
               + _click_actions(0.3, 8.0, 12.0)
               + [rest(1.0, yaw_deg=12.0, label="대기")])
    off = []
    for freeze_on in (True, False):
        out = _use(actions, freeze_enabled=freeze_on)
        off.append(math.hypot(out["click_px"][0] - out["aim_px"][0],
                              out["click_px"][1] - out["aim_px"][1]))
    assert off[1] > off[0] * 3, (
        "붙잡기 켬 %.0f px / 끔 %.0f px — 차이가 없다" % (off[0], off[1]))


# ── 드래그는 여전히 드래그다 ───────────────────────────────────────────────

def test_deliberate_drag_still_works():
    """오래 벌리고 고개로 끌면 드래그. 파란색이 켜지고, 다물면 끝난다."""
    out = _use(WARMUP + [aim(1.0, yaw_deg=-10.0)]
               + _click_actions(1.5, -10.0, 10.0)
               + [rest(1.0, yaw_deg=10.0, label="대기")])
    assert out["events"] == [PRESS, HOLD_START, HOLD_END]


def test_two_clicks_in_a_row():
    """더블클릭 시도 — 두 번 다 클릭이어야 한다(합성은 OS에 맡긴다)."""
    out = _use(WARMUP + [aim(1.0, yaw_deg=8.0)]
               + _click_actions(0.2, 8.0, 8.0)
               + [rest(0.25, yaw_deg=8.0, label="사이")]
               + _click_actions(0.2, 8.0, 8.0)
               + [rest(1.0, yaw_deg=8.0, label="대기")])
    assert out["events"] == [PRESS, CLICK, PRESS, CLICK]
