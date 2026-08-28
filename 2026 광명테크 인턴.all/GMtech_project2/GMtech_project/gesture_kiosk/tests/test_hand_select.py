"""hand_select 단위 테스트 — 단일 손 추적·거리 자·머리 앵커 검증 (몸통판).

2026-07-31 단일 손 추적(라벨 제거) 스펙: 좌/우 슬롯·재라벨 소멸 — 획득(이동+
모양) / 이음(연속성·재등장 반경) / 해제(release_sec)가 검증 대상이다. 카메라·
모델 없이 HandDetection 대역(hand_fixtures.make_hand)과 HeadDetection 대역만으로
검증한다. 손 실측 자(가상 어깨너비)·머리 앵커 게이트(2026-07-31 — 포즈 머리)는
유지.

픽스처 기하: 손 폭 80px · 월드 0.08m → 가상 어깨 400px — 획득 임계
0.25×400=100px, 재등장 반경 4×80=320px, 연속 반경 1.5×80=120px.
"""
import logging
import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.head_detector import HeadDetection
from src.postprocess.hand_select import (
    REJECT_STREAK_LIMIT, REJECT_STREAK_LIMIT_WHILE_TRACKED, STANDARD_SHOULDER_M,
    TRACKED_DRIFT_ALPHA, HandSelector, hand_span_px, hand_span_world_m,
    stable_span_px,
)
from tests.hand_fixtures import make_hand

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720
FRAME_DT_SEC = 1.0 / 30.0


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config():
    return {
        "hand_select": {
            "release_sec": 2.0,
            "acquire": {"move_dist_shoulder": 0.25, "window_sec": 0.5},
            "hand_shape": {
                "extend_ratio": 1.35,
                "min_valid_fingers": 3,
                "curl_confirm_ratio": 0.9,
            },
        },
    }


def make_selector(config=None):
    clock = FakeClock()
    selector = HandSelector(config or make_config(), FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
                            clock=clock)
    return selector, clock


def scaled_hand(user_side, shape, root_xy, px_factor):
    """화면 크기만 px_factor배인 손 — 더 가까운(크게 보이는) 손 재현.

    월드 랜드마크는 그대로 둔다 — 실제 손 크기는 같고 거리만 다른 상황.
    """
    hand = make_hand(user_side, shape, root_xy)
    center_x = hand.landmarks[:, 0].mean()
    center_y = hand.landmarks[:, 1].mean()
    hand.landmarks[:, 0] = center_x + (hand.landmarks[:, 0] - center_x) * px_factor
    hand.landmarks[:, 1] = center_y + (hand.landmarks[:, 1] - center_y) * px_factor
    return hand


def feed(selector, clock, frames, heads=None):
    """프레임 목록 공급 — 각 프레임 = HandDetection 목록. 마지막 신호를 돌려준다."""
    signal = None
    for hands in frames:
        selector.update(hands, heads)
        signal = selector.user_hand_signal()
        clock.tick(FRAME_DT_SEC)
    return signal


def moving_hand_frames(start_x, step_px, count, y_px=400, shape="finger", side="right"):
    """이동하는 손의 프레임 목록 — 획득(이동+모양) 재현용."""
    return [[make_hand(side, shape, (start_x + step_px * i, y_px))]
            for i in range(count)]


class AcquireTest(unittest.TestCase):
    """획득 — 모양이 보이는 손이 실제로 움직여야 잡힌다 (구 지시 손 v2의 계승)."""

    def test_moving_hand_with_shape_is_acquired(self):
        # 5프레임 × 30px = 120px ≥ 임계 100px — 획득
        selector, clock = make_selector()
        signal = feed(selector, clock, moving_hand_frames(400, 30, 6))
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], "finger")

    def test_stationary_hand_is_never_acquired(self):
        # 가만히 떠 있는 손 — 영원히 안 잡힌다 (쉬는 손·구경꾼 손 방어)
        selector, clock = make_selector()
        frames = [[make_hand("right", "finger", (500, 400))]] * 30
        self.assertIsNone(feed(selector, clock, frames))

    def test_moving_shapeless_hand_is_not_acquired(self):
        # 모양 불명(블러 잔상)인 이동 — 지시가 아니다
        selector, clock = make_selector()
        frames = [[make_hand("right", "open", (400 + 30 * i, 400))] for i in range(8)]
        for frame in frames:
            frame[0].world_landmarks = frame[0].world_landmarks * 0.0   # 판별 불능화
        self.assertIsNone(feed(selector, clock, frames))

    def test_moving_hand_beats_resting_hand(self):
        # 실기 보고 계승(2026-07-30·31 — 가만히 있는 손 독점·배구 토스): 쉬는 손이
        # 아무리 먼저·크게 보여도, 움직이는 손이 잡힌다
        selector, clock = make_selector()
        rest = make_hand("left", "finger", (200, 300))
        frames = [[rest, make_hand("right", "finger", (700 + 30 * i, 450))]
                  for i in range(8)]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertGreater(signal[1][0], 600)   # 움직인 손 위치

    def test_higher_hand_wins_when_both_qualify(self):
        # ★2026-08-04 사용자 제안("배 위로 있는 손이 잡히도록"): 같은 프레임에 둘 다
        # 획득 요건(모양+이동)을 넘기면 **위에 있는 손**이 이긴다.
        # ※보장 범위: 이 규칙은 **동시 경쟁**만 가른다. 아래 손이 더 빨리 움직여
        # 임계를 **먼저** 넘으면 그 손이 잡히고, 그 뒤 교체는 인계 규칙
        # (takeover_idle_sec)이 담당한다 — 획을 그리는 중인 손을 높이만으로
        # 뺏게 하면 낮게 제스처하는 사용자를 도중에 끊게 된다
        selector, clock = make_selector()
        frames = [[make_hand("left", "finger", (300 + 30 * i, 600)),   # 배 앞
                   make_hand("right", "finger", (800 + 30 * i, 250))]  # 위 — 같은 속도
                  for i in range(8)]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertLess(signal[1][1], 400)   # 위쪽 손이 잡혔다

    def test_low_hand_alone_is_still_acquired(self):
        # ★높이는 **우선순위일 뿐 자르는 기준이 아니다**: 실측에서 낮은 제스처
        # (어깨 아래 0.55~0.82)와 배 앞 손이 겹쳐, 높이로 자르면 팔을 높이 못 드는
        # 사용자가 못 쓴다(배리어프리 위배). 손이 하나면 낮아도 잡혀야 한다
        selector, clock = make_selector()
        signal = feed(selector, clock, moving_hand_frames(300, 30, 6, y_px=650))
        self.assertIsNotNone(signal)
        self.assertGreater(signal[1][1], 500)   # 낮은 손이지만 잡혔다

    def test_blur_frames_within_window_still_acquire(self):
        # 이동 중 일부 프레임 판별 실패(모양 None) — 창 안에 모양이 한 번이라도
        # 보였으면 획득된다 (블러 내성)
        selector, clock = make_selector()
        frames = []
        for i in range(8):
            hand = make_hand("right", "finger", (400 + 30 * i, 400))
            if i % 2 == 1:
                hand.world_landmarks = hand.world_landmarks * 0.0   # 격프레임 블러
            frames.append([hand])
        self.assertIsNotNone(feed(selector, clock, frames))


class TrackContinuityTest(unittest.TestCase):
    """이음 — 추적 손은 연속성으로 따라가고, 소실 후에도 근처 재등장이면 같은 손."""

    def _acquire(self):
        selector, clock = make_selector()
        feed(selector, clock, moving_hand_frames(400, 30, 6))
        self.assertIsNotNone(selector.user_hand_signal())
        return selector, clock

    def test_tracked_hand_follows_fast_move(self):
        # 획득 후 빠른 이동(프레임당 100px < 재등장 반경 320px) — 계속 같은 손
        selector, clock = self._acquire()
        signal = feed(selector, clock,
                      [[make_hand("right", "finger", (550 + 100 * i, 400))]
                       for i in range(1, 5)])
        self.assertIsNotNone(signal)
        self.assertGreater(signal[1][0], 800)

    def test_intruder_beyond_reentry_is_ignored(self):
        # 추적 중 반경(320px) 밖에 나타난 손(다른 사람) — 정체성을 못 뺏는다
        selector, clock = self._acquire()
        signal = feed(selector, clock, [[make_hand("left", "fist", (1150, 300))]] * 3)
        self.assertIsNone(signal)   # 추적 손 미관측 — 난입 손은 신호가 아니다

    def test_dropout_reappear_resumes_identity(self):
        # 소실(release_sec 안) 후 근처 재등장 — 이동 없이도 같은 손으로 승계
        # (화면 가리킴의 수 초 소실 — 구 래치 유예·rejoin의 계승)
        selector, clock = self._acquire()
        feed(selector, clock, [[]] * 20)                       # 0.67초 소실
        signal = feed(selector, clock,
                      [[make_hand("right", "finger", (600, 420))]] * 2)
        self.assertIsNotNone(signal)

    def test_release_requires_reacquisition(self):
        # release_sec(2초) 초과 소실 — 정체성 해제: 그 자리 정지 손은 새로 획득해야
        selector, clock = self._acquire()
        feed(selector, clock, [[]] * 5)
        clock.tick(2.5)
        signal = feed(selector, clock,
                      [[make_hand("right", "finger", (560, 400))]] * 10)
        self.assertIsNone(signal)   # 정지 재등장 — 획득 요건(이동) 미달

    def test_crossing_center_with_resting_hand_keeps_identity(self):
        # 획 교차(2026-07-31 실기 계승): 획 손이 쉬는 손 쪽으로 관통해도 연속성이
        # 정체성을 지킨다 — 라벨 시절 같은 라벨 충돌로 씹히던 시나리오
        selector, clock = make_selector()
        rest = make_hand("left", "fist", (350, 550))
        frames = [[rest, make_hand("right", "finger", (900 - 90 * i, 400))]
                  for i in range(7)]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertLess(signal[1][0], 500)   # 관통한 획 손을 끝까지 따라감

    def test_resting_hand_does_not_hijack_during_blur(self):
        # ★2026-08-03 실기(사용자 보고 — "제스처 시전 중인 손보다 가만히 있는 손이
        # 인식이 더 잘돼서 포커스가 이동된다"): 제스처 손이 빨라 모션 블러로 몇
        # 프레임 끊기면, 그 프레임엔 **가만히 있는 손이 마지막 추적점에 더 가깝다**
        # — 재등장 반경(320px)이 그 손을 이어 정체성이 통째로 넘어갔다.
        # 내 손이 보이던 동안 함께 보이던 손은 내 손이 아니다 (_other_tracks)
        selector, clock = make_selector()
        rest = make_hand("left", "fist", (700, 480))   # 쉬는 손 — 추적점에서 171px:
                                              #   재등장 반경(248px) **안**이라
                                              #   방어가 없으면 승계된다.
                                              #   연속 반경(93px) 밖 — 획득 궤적 분리
        frames = [[rest, make_hand("right", "finger", (400 + 30 * i, 400))]
                  for i in range(6)]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], "finger")          # 움직인 손을 잡았다(전제)
        # 제스처 손만 블러로 소실 — 쉬는 손은 계속 보인다
        signal = feed(selector, clock, [[rest]] * 4)
        self.assertIsNone(signal)                      # 쉬는 손이 승계하면 안 된다
        # 제스처 손이 돌아오면 다시 그 손을 잇는다
        signal = feed(selector, clock,
                      [[rest, make_hand("right", "finger", (610, 400))]] * 2)
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], "finger")

    def test_new_hand_appearing_after_loss_still_resumes(self):
        # 방어가 과하면 안 된다 — 내 손이 끊긴 사이 **없던** 손이 근처에 나타나면
        # 그건 내 손의 재등장이다 (화면 가리킴 소실 후 복귀 — 기존 동작 유지)
        selector, clock = make_selector()
        rest = make_hand("left", "fist", (700, 480))
        feed(selector, clock,
             [[rest, make_hand("right", "finger", (400 + 30 * i, 400))] for i in range(6)])
        feed(selector, clock, [[rest]] * 3)            # 내 손만 소실
        signal = feed(selector, clock,
                      [[rest, make_hand("right", "finger", (600, 410))]] * 2)
        self.assertIsNotNone(signal)
        self.assertEqual(signal[0], "finger")


class EngagementTest(unittest.TestCase):
    def test_engaged_while_hands_recent(self):
        selector, clock = make_selector()
        selector.update([make_hand("right", "finger", (500, 400))])
        self.assertTrue(selector.is_engaged())
        clock.tick(1.0)
        selector.update([])                                        # 잠깐 소실 — 유예 안
        self.assertTrue(selector.is_engaged())

    def test_release_clears_selection_state(self):
        # 유예(2초) 초과 소실 — 사용 종료: 다음 사용자에 상태를 승계하지 않는다
        selector, clock = make_selector()
        feed(selector, clock, moving_hand_frames(400, 30, 6))
        clock.tick(2.5)
        self.assertFalse(selector.update([]))
        self.assertIsNone(selector.locked_box)


class HandScaleTest(unittest.TestCase):
    """손 실측 자 — 가상 어깨너비 비율 (기존 임계 체계 유지의 핵심)."""

    def test_scale_formula(self):
        # 비율 = (화면 폭 px / 실제 폭 m × 표준 어깨 0.4m) / 프레임 폭
        selector, _ = make_selector()
        hand = make_hand("right", "fist", (500, 400))
        selector.update([hand])
        expected = (hand_span_px(hand.landmarks) / hand_span_world_m(hand.world_landmarks)
                    * STANDARD_SHOULDER_M) / FRAME_WIDTH_PX
        self.assertAlmostEqual(selector.hand_scale_ratio(), expected, places=6)

    def test_closer_hand_gives_bigger_scale(self):
        # 가까울수록(화면에 크게) 자가 커진다 — 거리 불변 임계의 원리
        selector, _ = make_selector()
        far_hand = make_hand("right", "fist", (500, 400))
        near_hand = scaled_hand("right", "fist", (500, 400), px_factor=2.0)
        selector.update([far_hand])
        far_scale = selector.hand_scale_ratio()
        selector.update([near_hand])
        self.assertAlmostEqual(selector.hand_scale_ratio(), far_scale * 2.0, places=6)

    def test_no_hands_returns_none(self):
        selector, _ = make_selector()
        selector.update([])
        self.assertIsNone(selector.hand_scale_ratio())   # 필터가 마지막 값·폴백 사용

    def test_shoulder_line_is_none_without_anchor(self):
        # 앵커 없음 — 어깨선 없음: 들어올리기 게이트는 하단 띠 폴백이 담당
        selector, _ = make_selector()
        self.assertIsNone(selector.shoulder_line_y_ratio())


def make_head(center_x_px, center_y_px, width_px, wrists=(), shoulders=None,
              wrist_visibility=1.0, segmentation_mask=None):
    """사람 1명 관측 대역.

    shoulders 미지정 = **머리폭 × 3.16을 어깨 폭으로** 자동 부여하고 y는 머리와
    같게 둔다(2026-08-04 앵커 기준 교체). 그러면 앵커 = (머리 중심, 머리폭×3.16)이
    되어, 어깨 단위 임계(reach 1.58 · wrist 0.95)가 구 머리 단위(5.0 · 3.0)와
    **같은 픽셀 거리**를 낸다 — 기존 테스트의 거리 가정이 그대로 성립한다.
    좌표는 실물과 같은 (x, y, visibility) 3-튜플.
    segmentation_mask(2026-08-07 신설) — 미지정 시 None(신호 비활성).
    """
    if shoulders is None:
        half = width_px * ANCHOR_SPAN_RATIO / 2.0
        shoulders = ((center_x_px - half, center_y_px),
                     (center_x_px + half, center_y_px))
    return HeadDetection(
        center_x_px=center_x_px, center_y_px=center_y_px, width_px=width_px, conf=1.0,
        shoulders=tuple((x, y, 1.0) for x, y in shoulders),
        wrists=tuple((x, y, wrist_visibility) for x, y in wrists),
        segmentation_mask=segmentation_mask,
    )


ANCHOR_SPAN_RATIO = 3.16   # 픽스처 어깨 폭 = 머리폭 × N (실측 정면 비율과 동일)

# 손목 심사 테스트의 공통 몸: 어깨 폭 316px(머리 100 기준) → 팔 길이 허용 1.6×316 = 506px
GATE_SHOULDERS = ((482, 200), (798, 200))
GATE_WRISTS = ((500, 540), (780, 540))      # 어깨에서 340px — 팔 길이(506) 안


class HeadAnchorTest(unittest.TestCase):
    """머리 앵커(2026-07-31 몸통판 — 포즈 머리) — 최대(가까운) 머리 고정 + 도달 반경 게이트."""

    def setUp(self):
        config = make_config()
        config["head_anchor"] = {
            "reach_shoulder_widths": 1.58,   # 어깨 316px → 반경 499px (구 5.0×머리폭과 동일)
            "anchor_grace_sec": 1.0,
        }
        self.clock = FakeClock()
        self.selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
                                     clock=self.clock)

    def _feed(self, frames, heads):
        return feed(self.selector, self.clock, frames, heads=heads)

    def test_far_moving_hand_blocked_while_anchor_alive(self):
        # 경성 게이트: 앵커가 살아 있으면 반경(500px) 밖 손은 움직여도(획득 요건
        # 충족) 후보조차 못 된다 — 옆 사람 손 차단
        head = make_head(640, 200, 100)
        frames = [[make_hand("right", "finger", (1200, 600 + 30 * i))]
                  for i in range(8)]
        self.assertIsNone(self._feed(frames, [head]))

    def test_near_moving_hand_is_acquired(self):
        # 반경 안에서 움직이는 손 — 정상 획득
        head = make_head(640, 200, 100)
        frames = moving_hand_frames(500, 30, 6)
        self.assertIsNotNone(self._feed(frames, [head]))

    def test_tracked_hand_exempt_beyond_reach(self):
        # 추적 면제: 반경 안에서 획득된 손은 크게 뻗어 반경 밖으로 나가도 안 잘린다
        head = make_head(640, 200, 100)
        frames = moving_hand_frames(500, 30, 6)          # 반경 안 획득
        frames += [[make_hand("right", "finger", (700 + 90 * i, 480))]
                   for i in range(1, 7)]                 # 끝은 반경 밖
        signal = self._feed(frames, [head])
        self.assertIsNotNone(signal)
        self.assertGreater(signal[1][0], 1100)

    def test_stale_exemption_not_inherited_by_new_hand(self):
        # 면제 신선도(0.5초): 반경 밖으로 뻗었던 손을 내리고 0.5초가 지나면 그
        # 자리 추적점은 만료 — 거기 나타난 새 손(옆 사람)은 게이트에 걸린다
        head = make_head(640, 200, 100)
        frames = moving_hand_frames(500, 30, 6)
        frames += [[make_hand("right", "finger", (700 + 90 * i, 480))]
                   for i in range(1, 7)]                 # 반경 밖(끝 ~1240)까지 추적
        self._feed(frames, [head])
        self.clock.tick(0.7)                             # 신선도(0.5초) 만료
        signal = self._feed([[make_hand("right", "finger", (1240, 480))]] * 3, [head])
        self.assertIsNone(signal)

    def test_dropout_reappear_far_still_exempt(self):
        # 소실 재등장 이음: 빠른 획 중 잠깐 끊겼다 반경 밖·재등장 반경(320px) 안에
        # 재등장 — 게이트 면제 + 정체성 승계로 획이 이어진다
        head = make_head(640, 200, 100)
        frames = moving_hand_frames(700, 60, 5)          # 반경 안 획득 (끝 ~990)
        self._feed(frames, [head])
        self.clock.tick(0.3)                             # 모션 블러 소실 (신선도 안)
        signal = self._feed([[make_hand("right", "finger", (1230, 480))]] * 2, [head])
        self.assertIsNotNone(signal)                     # 머리에서 ~640px — 반경 밖인데 통과

    def test_biggest_head_becomes_anchor(self):
        # 큰 머리(가까운 사람) 기준 게이트 — 작은(뒷) 머리 옆에서 움직이는 손 제외
        heads = [make_head(400, 200, 120), make_head(1000, 180, 50)]
        frames = [[make_hand("left", "finger", (1150 + 20 * i, 300))] for i in range(8)]
        self.assertIsNone(self._feed(frames, heads))

    def test_bigger_head_cannot_steal_live_anchor(self):
        # sticky: 앵커가 살아 있는 동안 다른 머리는 크기와 무관하게 무시
        self.selector.update([], [make_head(640, 200, 100)])
        self.selector.update([], [make_head(640, 200, 100), make_head(200, 220, 140)])
        x1, _, x2, _ = self.selector.anchor_head_box
        self.assertLess(abs((x1 + x2) / 2 - 640), 50)   # 원래 사용자 머리 유지

    def test_other_head_alone_does_not_hijack_anchor(self):
        # 앵커 머리가 한 프레임 안 잡히고 다른 머리만 잡혀도 — 즉시 점프 금지
        self.selector.update([], [make_head(640, 200, 100)])
        self.selector.update([], [make_head(200, 220, 140)])   # 앵커 머리 미관측
        x1, _, x2, _ = self.selector.anchor_head_box
        self.assertLess(abs((x1 + x2) / 2 - 640), 50)   # 앵커 그대로 (유예가 수명 관리)

    def test_new_head_anchors_after_grace_expiry(self):
        # 교체는 사용자가 떠나 유예(1초)가 앵커를 푼 뒤에만 — 다음 사용자 정상 인수
        self.selector.update([], [make_head(640, 200, 100)])
        self.clock.tick(1.5)
        self.selector.update([], [make_head(200, 220, 90)])    # 유예 만료 후 새 머리
        x1, _, x2, _ = self.selector.anchor_head_box
        self.assertLess(abs((x1 + x2) / 2 - 200), 50)

    def test_anchor_grace_then_gate_off(self):
        # 머리 소실 — 유예(1초) 안엔 게이트 유지(밖 손 차단), 초과하면 해제
        # (모든 손 통과): 머리 미검출로 방어가 인식을 해치지 않게
        head = make_head(640, 200, 100)
        far_frames = [[make_hand("left", "finger", (60 + 20 * i, 600))] for i in range(8)]
        self.assertIsNone(self._feed(far_frames, [head]))      # 반경 밖 — 차단
        self.clock.tick(1.5)                                   # 유예 초과 — 앵커 해제
        signal = self._feed(
            [[make_hand("left", "finger", (60 + 20 * i, 600))] for i in range(8)], [])
        self.assertIsNotNone(signal)                           # 게이트 꺼짐 — 획득 가능

    def test_closer_passerby_does_not_steal_anchor(self):
        # ★2026-08-04 실기(사용자 보고 — "고정이 안 되고 이전되는 현상"): 종전엔
        # 반경 안 후보 중 **가장 큰 머리**를 골라서, 옆으로 지나가는 사람이 카메라에
        # 더 가까우면(=더 크면) 앵커가 풀리지도 않은 채 그쪽으로 끌려갔다.
        # 이제 **위치가 가까운 쪽**을 고른다 — 지나가는 사람이 더 커도 안 뺏긴다
        self._feed([[]], [make_head(640, 200, 100)])
        passerby = make_head(700, 210, 180)          # 반경 안 + 훨씬 큼(더 가까움)
        self._feed([[]] * 3, [make_head(645, 202, 100), passerby])
        x1, _, x2, _ = self.selector.anchor_head_box
        self.assertLess(abs((x1 + x2) // 2 - 645), 30)   # 내 머리를 계속 따라간다

    def test_anchor_is_not_dropped_while_hand_is_tracked(self):
        # ★2026-08-04 앵커 고정(사용자 결정 — "선은 인식되면 고정으로 박아야"):
        # 사람이 여럿이면 내가 포즈 관측에서 잠깐 빠지는 일이 생기는데, 그때
        # 유예가 끝나 앵커가 풀리면 **가장 큰 사람**에게 다시 붙어 내 손이 통째로
        # 거부됐다(사용자 보고 "인식 손이 꺼짐"). 쓰고 있는 동안엔 안 푼다
        head = make_head(640, 200, 100)
        self._feed(moving_hand_frames(500, 30, 6), [head])
        self.assertIsNotNone(self.selector.user_hand_signal())   # 추적 중(전제)
        # 손은 계속 잡힌 채(세션 유지) 머리 관측만 2초간 없음 — 유예(1초) 초과
        self._feed([[make_hand("right", "finger", (680, 400))]] * 60, [])
        self.assertIsNotNone(self.selector.user_hand_signal())   # 여전히 추적 중
        self.assertIsNotNone(self.selector.anchor_head_box)      # 앵커 유지

    def test_anchor_drops_after_session_ends(self):
        # 다음 사용자 인수는 막지 않는다 — 추적 정체성이 풀린 **뒤에는** 앵커도 풀린다
        head = make_head(640, 200, 100)
        self._feed(moving_hand_frames(500, 30, 6), [head])
        self.assertIsNotNone(self.selector.user_hand_signal())
        self._feed([[]] * 5, [])
        self.clock.tick(3.0)                     # 손·머리 모두 소실 (release 2초 초과)
        self._feed([[]] * 3, [])
        self.assertIsNone(self.selector.anchor_head_box)

    def test_shoulder_line_comes_from_anchor_center(self):
        # ★2026-08-04 추정 제거: 앵커 중심이 곧 어깨선이다(구 "귀 중점 + 폭×1.6"
        # 추정과 config 항목 shoulder_below_head_widths가 함께 사라졌다).
        # 픽스처 어깨는 머리와 같은 y라 앵커 y = 200
        self.assertIsNone(self.selector.shoulder_line_y_ratio())
        self.selector.update([], [make_head(640, 200, 100)])
        self.assertAlmostEqual(self.selector.shoulder_line_y_ratio(),
                               200 / FRAME_WIDTH_PX, places=4)


class DepthGateTest(unittest.TestCase):
    """깊이(거리) 관문 — 뒷사람 손 차단 (2026-08-07 신설, _is_far_person_hand).

    픽스처 기하: 손 폭 80px · 월드 0.08m → 깊이값 80/0.08×0.4 = 400px.
    make_head(640, 200, 100) → 앵커 어깨너비 100×3.16 = 316px.
    기준 비 = 400/316 = 1.27 (정상 사용 — 관문 통과).
    scaled_hand의 px_factor가 곧 거리 배율의 역수다 — px_factor 0.5인 손은
    **2배 먼 곳의 손**(= 뒷사람)과 같은 관측을 낸다.
    """

    def _selector(self, depth_reject_ratio=None, depth_rel_reject_ratio=None,
                  min_operating_depth_px=None):
        config = make_config()
        config["head_anchor"] = {
            "reach_shoulder_widths": 1.58,
            "anchor_grace_sec": 1.0,
        }
        if depth_reject_ratio is not None:
            config["head_anchor"]["depth_reject_ratio"] = depth_reject_ratio
        if depth_rel_reject_ratio is not None:
            config["head_anchor"]["depth_rel_reject_ratio"] = depth_rel_reject_ratio
        if min_operating_depth_px is not None:
            config["head_anchor"]["min_operating_depth_px"] = min_operating_depth_px
        clock = FakeClock()
        return HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=clock), clock

    def test_gate_off_by_default(self):
        # 키가 없으면 종전 동작 — 먼 손도 위치만 맞으면 통과(회귀 방지)
        selector, clock = self._selector()
        head = make_head(640, 200, 100)
        far_frames = [[scaled_hand("right", "finger", (500 + 30 * i, 400), 0.4)]
                      for i in range(6)]
        self.assertIsNotNone(feed(selector, clock, far_frames, heads=[head]))

    def test_far_hand_rejected_by_absolute_gate(self):
        # ★핵심: 뒷사람이 **혼자** 손을 든 순간 — 비교할 다른 손이 없어 상대
        # 관문은 못 쓰고, 절대 관문만이 이걸 잡는다(사용자 보고 13회차의 상황)
        selector, clock = self._selector(depth_reject_ratio=0.70)
        head = make_head(640, 200, 100)
        # 깊이 400×0.4 = 160px < 0.70×316 = 221px -> 거부
        far_frames = [[scaled_hand("right", "finger", (500 + 30 * i, 400), 0.4)]
                      for i in range(6)]
        self.assertIsNone(feed(selector, clock, far_frames, heads=[head]))

    def test_own_hand_survives_absolute_gate(self):
        # 같은 거리의 내 손(깊이 400px = 앵커의 1.27배)은 그대로 통과해야 한다
        selector, clock = self._selector(depth_reject_ratio=0.70)
        head = make_head(640, 200, 100)
        self.assertIsNotNone(
            feed(selector, clock, moving_hand_frames(500, 30, 6), heads=[head]))

    def test_reaching_forward_hand_survives(self):
        # 팔을 앞으로 뻗으면 손이 **가까워져** 깊이값이 커진다 — 안전한 방향.
        # 내 손이 이 관문에 잘릴 이유가 없다는 것을 고정한다.
        # ※걸음 폭이 큰 이유: 획득 임계도 손 크기에 비례해 커진다(hand_shoulder_px가
        #   자를 겸하므로) — 1.8배 손은 임계도 1.8배다. 관문과 무관한 전제 조건
        selector, clock = self._selector(depth_reject_ratio=0.70)
        head = make_head(640, 200, 100)
        near_frames = [[scaled_hand("right", "finger", (500 + 50 * i, 400), 1.8)]
                       for i in range(8)]
        self.assertIsNotNone(feed(selector, clock, near_frames, heads=[head]))

    def test_far_hand_rejected_by_relative_gate_only(self):
        # 상대 관문 단독 — 절대 관문 없이도, 더 가까운 손이 함께 보이면 갈린다.
        # 체형 상수가 약분돼 사라지는 경로라 절대 관문보다 이론적으로 깨끗하다
        selector, clock = self._selector(depth_rel_reject_ratio=0.70)
        head = make_head(640, 200, 100)
        # 내 손(깊이 400) + 뒷사람 손(깊이 400×0.5=200) — 200/400 = 0.5 < 0.7
        frames = [[make_hand("right", "finger", (600 + 30 * i, 400)),
                   scaled_hand("left", "finger", (700 + 30 * i, 300), 0.5)]
                  for i in range(6)]
        for hands in frames:
            selector.update(hands, [head])
            clock.tick(FRAME_DT_SEC)
        self.assertEqual(len(selector._hands), 1)   # 먼 손 하나가 잘렸다

    def test_nearer_intruder_cannot_cut_my_hand(self):
        # ★2026-08-07 코드 리뷰 지적 — 상대 관문의 기준점 보호.
        # 나보다 **앞으로** 끼어든 사람의 손이 최근접이 되면, 기준점 가정이
        # 뒤집혀 내 손이 잘린다. 내 손(깊이 400, 절대비 1.27)은 침입자 손
        # (깊이 720) 대비 0.56으로 상대 임계(0.70) 아래지만, 절대비가 1.0
        # 이상이라 "뒤에 있는 사람"일 수 없으므로 면제돼야 한다
        selector, clock = self._selector(depth_rel_reject_ratio=0.70)
        head = make_head(640, 200, 100)
        frames = [[make_hand("right", "finger", (600 + 30 * i, 400)),
                   scaled_hand("left", "finger", (700 + 30 * i, 300), 1.8)]
                  for i in range(6)]
        for hands in frames:
            selector.update(hands, [head])
            clock.tick(FRAME_DT_SEC)
        self.assertEqual(len(selector._hands), 2)   # 내 손이 살아남았다

    def test_similar_depth_hands_both_survive_relative_gate(self):
        # 같은 사람의 두 손(깊이가 비슷)은 상대 관문에 안 걸린다 — 과잉 차단 방지
        selector, clock = self._selector(depth_rel_reject_ratio=0.70)
        head = make_head(640, 200, 100)
        frames = [[make_hand("right", "finger", (600 + 30 * i, 400)),
                   scaled_hand("left", "finger", (700 + 30 * i, 300), 0.9)]
                  for i in range(6)]
        for hands in frames:
            selector.update(hands, [head])
            clock.tick(FRAME_DT_SEC)
        self.assertEqual(len(selector._hands), 2)

    def test_far_hand_cannot_hide_behind_tracking_exemption(self):
        # ★오늘 하루 반복된 "한번 뺏기면 안 돌아온다"의 구조적 뒷문 봉쇄:
        # 추적 면제(위치 심사 면제)는 깊이 관문보다 **뒤에** 있어야 한다.
        # 먼 손이 추적에 들어간 뒤에도 깊이로는 계속 걸러져야 한다
        selector, clock = self._selector(depth_reject_ratio=0.70)
        head = make_head(640, 200, 100)
        # 관문 없이 먼저 추적을 만든다(관문을 끈 선택기로 획득시킨 뒤 상태 이식은
        # 불가하므로, 여기서는 먼 손이 매 프레임 거부되는지로 확인한다)
        far_frames = [[scaled_hand("right", "finger", (500 + 30 * i, 400), 0.4)]
                      for i in range(20)]
        feed(selector, clock, far_frames, heads=[head])
        self.assertEqual(selector._hands, [])          # 20프레임 내내 한 번도 못 들어옴
        self.assertIsNone(selector.user_hand_signal())

    def test_extended_arm_hand_is_never_rejected(self):
        # ★2026-08-07 실기 회귀 방지(학습 기준선 제거의 근거). 팔을 뻗으면
        # 손이 카메라에 가까워져 깊이비가 2~3까지 오른다. 그 값이 **어떤
        # 경로로도** 임계를 끌어올려 본인 손을 자르는 일이 없어야 한다 —
        # 실기 로그에서 비 1.48·1.60·1.99·2.16인 본인 손이 거부된 적이 있다
        selector, _ = self._selector(depth_reject_ratio=0.80)
        selector._head_anchor = (640.0, 200.0, 400.0)
        for px_factor in (1.5, 2.0, 2.5, 3.0):
            hand = scaled_hand("right", "finger", (640, 400), px_factor)
            self.assertFalse(
                selector._is_far_person_hand(hand, 400.0, None),
                "팔 뻗음 배율 %.1f에서 본인 손이 거부됐다" % px_factor)


    def test_operating_zone_survives_anchor_collapse(self):
        # ★작동 거리 창의 존재 이유(2026-08-07 — PDA 스캐너 발상): 앵커 어깨
        # 관측이 붕괴하면(실측 12~21px) 비율 기반 관문은 임계까지 함께 붕괴해
        # (0.80 × 12 = 9.6px) **아무도 못 자른다** — 하필 겹칠 때 잦은 붕괴다.
        # 카메라 기준 절대 하한은 그 순간에도 살아 있어야 한다
        selector, clock = self._selector(depth_reject_ratio=0.80,
                                         min_operating_depth_px=300)
        collapsed = make_head(640, 200, 100, shoulders=((634, 200), (646, 200)))
        # 붕괴한 앵커(어깨 12px)라 비율 관문 임계는 9.6px — 깊이 160px도 통과시킨다
        far_frames = [[scaled_hand("right", "finger", (600 + 20 * i, 300), 0.4)]
                      for i in range(6)]
        self.assertIsNone(feed(selector, clock, far_frames, heads=[collapsed]))

    def test_operating_zone_keeps_near_hand(self):
        # 작동 거리 안(깊이 400 > 300)이면 그대로 통과 — 과잉 차단 방지
        selector, clock = self._selector(min_operating_depth_px=300)
        head = make_head(640, 200, 100)
        self.assertIsNotNone(
            feed(selector, clock, moving_hand_frames(500, 30, 6), heads=[head]))

    def test_unmeasurable_depth_does_not_cut(self):
        # 깊이를 못 재면(월드 랜드마크 이상) 아무도 안 자른다 —
        # "못 믿을 관측으로는 탈락시키지 않는다"(reliable_wrists와 같은 원칙).
        # ※신호가 아니라 **게이트 통과 목록**(_hands)으로 확인한다: 깊이를 못
        #   재면 획득 임계의 자(hand_shoulder_px)도 못 세워 획득 자체가 별개
        #   이유로 실패한다 — 이 관문이 자르지 않는다는 것만 여기서 고정한다
        selector, clock = self._selector(depth_reject_ratio=0.70)
        head = make_head(640, 200, 100)
        hand = make_hand("right", "finger", (600, 400))
        hand.world_landmarks = np.zeros_like(hand.world_landmarks)
        selector.update([hand], [head])
        self.assertEqual(len(selector._hands), 1)


class QueueContinuityTest(unittest.TestCase):
    """대기줄 방어 — 이음 후보에서 '더 먼 사람'을 뺀다 (2026-08-07 신설).

    일렬로 서면 두 사람의 2D 위치가 사실상 같아 위치 기반 이음이 구분하지
    못한다(실측: 중심 이동 4~13px인데 어깨폭은 49~167% 요동). 거리로 가른다.
    """

    def _selector(self, continuity_depth_ratio=None):
        config = make_config()
        config["head_anchor"] = {
            "reach_shoulder_widths": 1.58,
            "anchor_grace_sec": 1.0,
            "continuity_depth_ratio": continuity_depth_ratio,
        }
        if continuity_depth_ratio is None:
            del config["head_anchor"]["continuity_depth_ratio"]
        clock = FakeClock()
        return HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=clock), clock

    def test_queue_person_does_not_steal_chosen(self):
        # ★핵심 재현: 앞사람(어깨 316px)과 뒷사람(어깨 190px)이 **같은 자리**에
        # 겹쳐 보인다. 위치로는 구분 불가 — 뒷사람이 앵커 중심에 더 가까워도
        # 거리(어깨폭)로 걸러져 마스크가 그쪽으로 넘어가면 안 된다
        selector, clock = self._selector(continuity_depth_ratio=0.80)
        front_mask = np.full((4, 4), 0.9, dtype=np.float32)
        back_mask = np.full((4, 4), 0.1, dtype=np.float32)
        selector.update([], [make_head(640, 200, 100, segmentation_mask=front_mask)])
        clock.tick(0.1)
        for _ in range(5):
            selector.update([], [
                make_head(645, 202, 100, segmentation_mask=front_mask),
                # 뒷사람: 거의 같은 자리(중심 3px 차)인데 어깨는 60%
                make_head(642, 201, 60, segmentation_mask=back_mask),
            ])
            clock.tick(0.1)
        self.assertIs(selector.anchor_segmentation_mask(), front_mask)

    def test_queue_person_takes_over_without_the_gate(self):
        # 관문이 없으면(종전) 뒷사람이 더 가까우므로 chosen을 뺏는다 —
        # 위 테스트가 진짜로 무언가를 막고 있음을 보증한다
        selector, clock = self._selector()
        front_mask = np.full((4, 4), 0.9, dtype=np.float32)
        back_mask = np.full((4, 4), 0.1, dtype=np.float32)
        selector.update([], [make_head(640, 200, 100, segmentation_mask=front_mask)])
        clock.tick(0.1)
        for _ in range(5):
            selector.update([], [
                make_head(645, 202, 100, segmentation_mask=front_mask),
                make_head(642, 201, 60, segmentation_mask=back_mask),
            ])
            clock.tick(0.1)
        self.assertIs(selector.anchor_segmentation_mask(), back_mask)

    def test_nearer_candidate_is_never_dropped(self):
        # 비대칭 보장 — 더 **가까운** 후보는 안 뺀다. 앵커가 어쩌다 뒷사람에게
        # 가 있어도 앞사람으로 되돌아올 길이 남아야 한다(가두면 안 된다)
        selector, _ = self._selector(continuity_depth_ratio=0.80)
        far = (make_head(640, 200, 60), (640.0, 200.0, 190.0))
        near = (make_head(640, 200, 100), (640.0, 200.0, 316.0))
        kept = selector._drop_farther_candidates([far, near], 190.0)
        self.assertIn(near, kept)

    def test_all_candidates_far_falls_open(self):
        # 앵커 폭이 붕괴해 모든 후보가 '멀다'로 나오면 아무도 안 자른다
        selector, _ = self._selector(continuity_depth_ratio=0.80)
        candidates = [(make_head(640, 200, 60), (640.0, 200.0, 190.0))]
        self.assertEqual(
            selector._drop_farther_candidates(candidates, 1000.0), candidates)


class StableSpanTest(unittest.TestCase):
    """정체성 반경의 자는 **손 모양에 안 흔들려야** 한다 (2026-08-07 신설).

    사용자 보고: "특정 손짓을 하면 초록색 관절값이 안 보이거나 다른 사람한테
    이동한다". 주먹을 쥐면 화면상 손이 작아져 이음 반경이 함께 줄고, 그 순간
    추적 손이 제 반경 밖으로 밀려나 정체성을 잃던 것이 원인이었다.
    """

    def test_span_is_same_across_hand_shapes(self):
        spans = {shape: stable_span_px(make_hand("right", shape, (640, 400)))
                 for shape in ("open", "fist", "finger", "ok")}
        self.assertAlmostEqual(min(spans.values()), max(spans.values()), places=3,
                               msg="손 모양별 자가 흔들린다: %s" % spans)

    def test_raw_span_does_shake(self):
        # 종전 방식이 실제로 흔들렸음을 고정 — 이 테스트가 깨지면 위 수정의
        # 전제(모양에 따라 화면 크기가 변한다)가 사라진 것이다
        raw = {shape: hand_span_px(make_hand("right", shape, (640, 400)).landmarks)
               for shape in ("open", "fist")}
        self.assertNotAlmostEqual(raw["open"], raw["fist"], places=1)

    def test_scale_matches_open_hand_so_thresholds_keep_meaning(self):
        # 기존 임계(REENTRY 4.0 등)를 재보정하지 않아도 되도록, 펼친 손에서는
        # 종전 값과 같아야 한다
        hand = make_hand("right", "open", (640, 400))
        self.assertAlmostEqual(stable_span_px(hand),
                               hand_span_px(hand.landmarks), delta=1.0)

    def test_falls_back_when_world_landmarks_unusable(self):
        hand = make_hand("right", "open", (640, 400))
        hand.world_landmarks = np.zeros_like(hand.world_landmarks)
        self.assertAlmostEqual(stable_span_px(hand),
                               hand_span_px(hand.landmarks), places=3)

    def test_candidate_hands_hands_arbiter_a_stable_span(self):
        # ★중재기는 candidate_hands()의 네 번째 값을 **슬롯 이음 반경**으로
        # 쓴다(hand_arbiter._sync_slots: dist <= match_ratio × span_px).
        # 그 값이 모양에 흔들리면 주먹을 쥐는 순간 확정 슬롯이 손을 놓치고
        # 경쟁이 재시작돼 옆 사람이 이긴다 — 셀렉터 쪽과 완전히 같은 버그였다
        selector, _ = make_selector()
        spans = {}
        for shape in ("open", "fist"):
            selector.update([make_hand("right", shape, (640, 400))])
            spans[shape] = selector.candidate_hands()[0][3]
        self.assertAlmostEqual(spans["open"], spans["fist"], places=3,
                               msg="중재기에 넘어가는 자가 모양에 흔들린다: %s" % spans)


class SilhouetteOwnerTest(unittest.TestCase):
    """실루엣 소유자 판정 — 옆 사람 손 차단 (2026-08-07 신설).

    옆 사람은 거리가 같아 깊이로 못 가르고, 손목 기반 방어는 손목을 못 믿으면
    통째로 꺼진다. 손 중심 화소가 누구 몸 위인지 직접 묻는다.
    """

    def _selector(self):
        config = make_config()
        config["head_anchor"] = {
            "reach_shoulder_widths": 1.58,
            "anchor_grace_sec": 1.0,
        }
        clock = FakeClock()
        selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=clock)
        anchor = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        anchor[300:500, 100:400] = 1.0          # 내 몸
        other = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32)
        other[300:500, 800:1100] = 1.0          # 옆 사람 몸
        selector._anchor_segmentation_mask = anchor
        selector._other_segmentation_mask = other
        return selector

    def test_hand_on_other_body_is_rejected(self):
        self.assertTrue(self._selector()._is_on_other_silhouette((900, 400)))

    def test_hand_on_my_body_is_kept(self):
        self.assertFalse(self._selector()._is_on_other_silhouette((200, 400)))

    def test_hand_in_empty_space_is_kept(self):
        # ★뻗은 팔은 세그멘테이션이 놓칠 수 있다 — 양쪽 다 밖이면 자르지 않는다
        # (자르면 내 손이 사라진다. 오늘 세 번 겪은 실패 유형)
        self.assertFalse(self._selector()._is_on_other_silhouette((600, 400)))

    def test_overlap_favours_keeping(self):
        # 두 실루엣이 겹치는 화소 — 내 몸이기도 하므로 자르지 않는다
        selector = self._selector()
        selector._other_segmentation_mask[300:500, 100:400] = 1.0
        self.assertFalse(selector._is_on_other_silhouette((200, 400)))

    def test_no_masks_disables_the_check(self):
        selector = self._selector()
        selector._other_segmentation_mask = None
        self.assertFalse(selector._is_on_other_silhouette((900, 400)))

    def test_out_of_bounds_is_kept(self):
        self.assertFalse(self._selector()._is_on_other_silhouette((-5, 400)))


class ObservationMedianTest(unittest.TestCase):
    """관측 중앙값 필터 — 단발 붕괴를 입력 단계에서 제거 (2026-08-07 신설).

    실측된 붕괴는 단발 스파이크 형태다(어깨폭 209→18→200px). 종전 4겹 방어는
    "들어온 뒤 처리"라 서로 부딪혔고, 중앙값은 아예 안 들어오게 한다.
    """

    def _selector(self, median_count=None):
        config = make_config()
        config["head_anchor"] = {
            "reach_shoulder_widths": 1.58,
            "anchor_grace_sec": 1.0,
            "jump_reject_ratio": 0.25,
        }
        if median_count is not None:
            config["head_anchor"]["observation_median_count"] = median_count
        clock = FakeClock()
        return HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=clock), clock

    def test_single_frame_collapse_never_reaches_anchor(self):
        # 실측 붕괴 패턴 재현: 한 프레임만 어깨가 무너진다(316→30→316px).
        # 중앙값이면 그 값은 **선택되지 않는다** — 앵커 폭이 거의 안 흔들려야 한다
        selector, clock = self._selector(median_count=3)
        for _ in range(4):
            selector.update([], [make_head(640, 200, 100)])   # 어깨 316px
            clock.tick(0.1)
        width_before = selector._head_anchor[2]
        selector.update([], [make_head(640, 200, 100, shoulders=((625, 200), (655, 200)))])
        clock.tick(0.1)
        selector.update([], [make_head(640, 200, 100)])
        clock.tick(0.1)
        self.assertAlmostEqual(selector._head_anchor[2], width_before, delta=5.0)

    def test_sustained_change_still_passes_through(self):
        # 진짜로 다가오면(연속해서 커지면) 중앙값도 따라간다 — 굳어버리면 안 된다
        selector, clock = self._selector(median_count=3)
        for width_px in (100, 100, 100):
            selector.update([], [make_head(640, 200, width_px)])
            clock.tick(0.1)
        start_width = selector._head_anchor[2]
        for width_px in (108, 116, 124, 132, 140):
            selector.update([], [make_head(640, 200, width_px)])
            clock.tick(0.1)
        self.assertGreater(selector._head_anchor[2], start_width * 1.1)

    def test_filter_off_by_default(self):
        # 키가 없으면 관측이 그대로 들어간다(종전 동작) — 회귀 방지
        selector, _ = self._selector()
        self.assertIsNone(selector._median_count)
        self.assertEqual(selector._median_observed_frame((1.0, 2.0, 3.0)), (1.0, 2.0, 3.0))

    def test_median_is_componentwise(self):
        selector, _ = self._selector(median_count=3)
        selector._median_observed_frame((0.0, 0.0, 100.0))
        selector._median_observed_frame((500.0, 500.0, 10.0))    # 스파이크
        result = selector._median_observed_frame((10.0, 10.0, 110.0))
        self.assertEqual(result, (10.0, 10.0, 100.0))            # 성분별 중앙값


class SegmentationMaskTest(unittest.TestCase):
    """앵커 실루엣 마스크 배선(2026-08-07 2차 — "블러를 사람 누끼 딴거 제외하고").

    HandSelector는 마스크 내용을 해석하지 않는다(불투명 전달) — chosen.
    segmentation_mask가 anchor_segmentation_mask()로 그대로 나오는지, 최신
    관측으로 갱신되는지만 확인한다(hand_select._accept_anchor_frame).
    """

    def setUp(self):
        config = make_config()
        config["head_anchor"] = {
            "reach_shoulder_widths": 1.58,
            "anchor_grace_sec": 1.0,
        }
        self.clock = FakeClock()
        self.selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
                                     clock=self.clock)

    def test_none_before_any_anchor(self):
        self.assertIsNone(self.selector.anchor_segmentation_mask())

    def test_initial_acquisition_wires_mask(self):
        mask = np.full((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), 0.9, dtype=np.float32)
        self.selector.update([], [make_head(640, 200, 100, segmentation_mask=mask)])
        self.assertIs(self.selector.anchor_segmentation_mask(), mask)

    def test_continuity_update_replaces_mask_with_latest(self):
        first_mask = np.full((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), 0.9, dtype=np.float32)
        self.selector.update([], [make_head(640, 200, 100, segmentation_mask=first_mask)])
        second_mask = np.full((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), 0.1, dtype=np.float32)
        self.selector.update([], [make_head(645, 202, 101, segmentation_mask=second_mask)])
        self.assertIs(self.selector.anchor_segmentation_mask(), second_mask)

    def test_missing_mask_on_update_clears_stale_mask(self):
        # 이번 관측에서 마스크 추출이 실패하면 낡은 마스크를 들고 있는 것보다
        # 미관측(None)이 안전하다
        mask = np.full((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), 0.9, dtype=np.float32)
        self.selector.update([], [make_head(640, 200, 100, segmentation_mask=mask)])
        self.selector.update([], [make_head(645, 202, 101, segmentation_mask=None)])
        self.assertIsNone(self.selector.anchor_segmentation_mask())

    def test_mask_is_held_across_short_anchor_gap(self):
        # ★2026-08-07(사용자 보고 — 뒷사람이 "보였다가 사라졌다가"): 앵커가 잠깐
        # 풀렸다고 마스크를 즉시 버리면 블러가 통째로 꺼져(폴백 상자도 앵커가
        # 없으면 None) 대기줄이 그대로 노출된다. 짧은 공백은 붙들어야 한다
        mask = np.full((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), 0.9, dtype=np.float32)
        self.selector.update([], [make_head(640, 200, 100, segmentation_mask=mask)])
        self.clock.tick(1.2)                     # 앵커 유예(1초) 초과 — 앵커 해제
        self.selector.update([], [])             #   유지 시간(1.5초) 안이라 붙들려 있어야
        self.assertIsNotNone(self.selector.anchor_segmentation_mask())

    def test_mask_expires_after_hold_period(self):
        # 무한정 붙들면 ①낡은 실루엣이 다음 사람에게 승계되고 ②움직이는 본인의
        # **지금 손 위치가 블러**된다. 유지 시간이 지나면 버린다
        mask = np.full((FRAME_HEIGHT_PX, FRAME_WIDTH_PX), 0.9, dtype=np.float32)
        self.selector.update([], [make_head(640, 200, 100, segmentation_mask=mask)])
        self.clock.tick(3.0)                     # 유지 시간(1.5초) 초과
        self.selector.update([], [])
        self.assertIsNone(self.selector.anchor_segmentation_mask())

    def test_hold_must_outlast_the_anchor_grace(self):
        # ★설계 제약을 코드로 고정(2026-08-07 — 처음에 1.0으로 뒀다가 잡힌 실수):
        # 둘 다 "마지막 관측"부터 재므로, 유지 시간이 앵커 유예 이하면 앵커가
        # 풀리는 바로 그 순간 마스크도 함께 만료돼 공백을 하나도 못 메운다
        self.assertGreater(self.selector._mask_hold_sec,
                           self.selector._head_anchor_grace_sec)


class AnchorCenterTest(unittest.TestCase):
    """앵커 중심 = **어깨 중점**(2026-08-04 사용자 결정 — 머리 기준점 제거).

    공용 픽스처는 어깨 y를 머리와 같게 두므로 이 변경을 **구분하지 못한다** —
    그래서 어깨를 머리보다 아래에 둔 전용 픽스처로 잰다. 귀 좌표가 판정에
    남아 있으면 여기서 걸린다.
    """

    def _anchor(self, head):
        config = make_config()
        config["head_anchor"] = {"reach_shoulder_widths": 2.09, "anchor_grace_sec": 1.0}
        selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=FakeClock())
        selector.update([], [head])
        return selector._head_anchor

    def test_center_is_shoulder_midpoint_not_ears(self):
        # 귀는 y=200, 어깨는 y=360 — 중심이 360이어야 머리를 안 보는 것이다
        head = make_head(640, 200, 100, shoulders=((480, 360), (800, 360)))
        center_x, center_y, span_px = self._anchor(head)
        self.assertAlmostEqual(center_x, 640.0, places=3)
        self.assertAlmostEqual(center_y, 360.0, places=3)   # ★귀(200)가 아니다
        self.assertAlmostEqual(span_px, 320.0, places=3)    # 실측 어깨 폭

    def test_falls_back_to_ears_when_shoulders_missing(self):
        # 어깨 관측이 빠져도 앵커를 버리면 게이트가 꺼져 옆 사람 손이 샌다 —
        # 귀에서 환산해 잇는다(중심 = 귀 y + 어깨너비 × 0.51)
        head = make_head(640, 200, 100, shoulders=())
        center_x, center_y, span_px = self._anchor(head)
        expected_span = 100 * 3.16
        self.assertAlmostEqual(center_x, 640.0, places=3)
        self.assertAlmostEqual(span_px, expected_span, places=3)
        self.assertAlmostEqual(center_y, 200.0 + 0.51 * expected_span, places=3)

    def test_display_circle_follows_anchor_not_ears(self):
        # 화면과 판정이 같은 것을 가리켜야 한다(사용자 보고 — "얼굴 쪽이 흔들려")
        head = make_head(640, 200, 100, shoulders=((480, 360), (800, 360)))
        config = make_config()
        config["head_anchor"] = {"reach_shoulder_widths": 2.09, "anchor_grace_sec": 1.0}
        selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=FakeClock())
        selector.update([], [head])
        x1, y1, x2, y2 = selector.anchor_head_box
        self.assertAlmostEqual((y1 + y2) / 2, 360.0, delta=1)   # ★귀(200)가 아니다
        self.assertAlmostEqual((x1 + x2) / 2, 640.0, delta=1)


class ImplausibleObservationTest(unittest.TestCase):
    """붕괴한 어깨 관측을 버린다 (2026-08-04 실기 로그로 원인 특정).

    로그의 점프가 거의 전부 `관측 1명 · 반경 안 후보 1`이었다 — 남에게 끌려간 게
    아니라 같은 사람의 어깨 관측이 무너진 것이다(209→18px 등). 반경이 어깨폭
    × 2.09라 어깨폭이 무너지면 내 손이 통째로 거부된다.
    """

    def _selector(self, reject_ratio=0.35, move_reject_ratio=None,
                  ):
        config = make_config()
        config["head_anchor"] = {"reach_shoulder_widths": 2.09, "anchor_grace_sec": 1.0}
        if reject_ratio is not None:
            config["head_anchor"]["jump_reject_ratio"] = reject_ratio
        if move_reject_ratio is not None:
            config["head_anchor"]["move_reject_ratio"] = move_reject_ratio
        self.clock = FakeClock()
        return HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=self.clock)

    def test_collapsed_shoulder_observation_is_dropped(self):
        selector = self._selector()
        selector.update([], [make_head(640, 300, 100)])       # 어깨 316px
        self.assertAlmostEqual(selector._head_anchor[2], 316.0, places=1)
        selector.update([], [make_head(640, 300, 6)])         # 어깨 19px — 붕괴
        # 앵커가 그대로여야 한다 — EMA로 끌려가면 반경이 쪼그라들어 손이 잘린다
        self.assertAlmostEqual(selector._head_anchor[2], 316.0, places=1)

    def test_normal_change_is_accepted(self):
        # 걸어서 다가오는 정도(10%)는 받아들여야 한다 — 실측 정상 95%가 0.10이다
        selector = self._selector()
        selector.update([], [make_head(640, 300, 100)])
        selector.update([], [make_head(640, 300, 110)])       # 어깨 316 → 347.6
        self.assertGreater(selector._head_anchor[2], 316.0)

    def test_anchor_does_not_freeze_on_repeated_rejects(self):
        # 계속 거부만 하면 사람이 정말 멀어졌을 때 앵커가 낡은 채 굳는다 —
        # 안전판(REJECT_STREAK_LIMIT)이 받아들여야 한다
        selector = self._selector()
        selector.update([], [make_head(640, 300, 100)])
        for _ in range(REJECT_STREAK_LIMIT + 1):
            selector.update([], [make_head(640, 300, 30)])    # 어깨 95px — 계속 거부 대상
        self.assertLess(selector._head_anchor[2], 316.0)      # 결국 받아들였다

    def test_no_key_no_gate(self):
        # 키를 지우면 종전 동작 — 붕괴가 그대로 들어온다
        selector = self._selector(reject_ratio=None)
        selector.update([], [make_head(640, 300, 100)])
        selector.update([], [make_head(640, 300, 6)])
        self.assertLess(selector._head_anchor[2], 316.0)

    def test_position_jump_with_stable_width_is_dropped(self):
        # 2026-08-07 실기 로그 — 비슷한 크기의 옆/뒷사람 사이 전환: 폭은
        # 거의 안 변하고(여기선 완전히 동일) 중심만 어깨너비의 20% 넘게 튄다.
        # 폭 관문(reject_ratio)은 안 걸려도 위치 관문(move_reject_ratio)이 잡아야 한다
        selector = self._selector(reject_ratio=0.35, move_reject_ratio=0.20)
        selector.update([], [make_head(640, 300, 100)])            # 어깨 316px
        self.assertAlmostEqual(selector._head_anchor[0], 640.0, places=1)
        selector.update([], [make_head(740, 300, 100)])            # 폭 동일, 중심만 100px(31.6%) 이동
        # 앵커가 그대로여야 한다 — 위치 관문이 이 관측을 버렸다
        self.assertAlmostEqual(selector._head_anchor[0], 640.0, places=1)

    def test_two_consecutive_agreeing_jumps_are_accepted_fast(self):
        # 2026-08-07 — 거부된 관측끼리 서로 합의(가까움)하면 REJECT_STREAK_LIMIT
        # (5회)까지 기다리지 않고 2번째에 바로 받아들인다: 노이즈라면 매번 다른
        # 방향으로 튀지, 같은 위치로 두 번 연속 합의할 리 없다 — 진짜 변화(또는
        # 진짜 인수인계)로 본다
        selector = self._selector(reject_ratio=0.35, move_reject_ratio=0.20)
        selector.update([], [make_head(640, 300, 100)])
        selector.update([], [make_head(740, 300, 100)])   # 1차 거부 — 후보로만 기록
        self.assertAlmostEqual(selector._head_anchor[0], 640.0, places=1)
        selector.update([], [make_head(740, 300, 100)])   # 2차 — 후보와 합의 → 수용
        self.assertAlmostEqual(selector._head_anchor[0], 640.0 + 0.4 * 100.0, places=1)

    def test_disagreeing_rejects_do_not_fast_accept(self):
        # 거부된 관측끼리 서로 다른 방향이면(합의 아님) 후보만 교체될 뿐 계속
        # 거부된다 — 진짜 노이즈(방향이 매번 다름)를 빠른 수용으로 잘못 받지 않는다
        selector = self._selector(reject_ratio=0.35, move_reject_ratio=0.20)
        selector.update([], [make_head(640, 300, 100)])
        selector.update([], [make_head(740, 300, 100)])   # 1차 거부 — 후보 (740,300)
        selector.update([], [make_head(550, 380, 100)])   # 2차 거부 — 후보와도 안 맞음
        self.assertAlmostEqual(selector._head_anchor[0], 640.0, places=1)
        self.assertAlmostEqual(selector._head_anchor[1], 300.0, places=1)

    def test_fast_accept_replaced_by_gentle_drift_while_tracked(self):
        # 2026-08-07 4차 — 3차("얼려서 보호", 0% 반영)는 역설이 있었다: 얼어붙은
        # 앵커가 실제 손 위치와 어긋나 반경 밖으로 밀려나면, 그 손의 추적이
        # release_sec 뒤 풀리면서 "추적 없음" 상태가 되어 보호 자체가 사라졌다.
        # 지금은 완전히 얼리지 않고 TRACKED_DRIFT_ALPHA(0.1)만큼만 살살 반영한다
        # — 완전히 튄 값(740)에 스냅되지는 않되, 조금씩은 따라간다
        selector = self._selector(reject_ratio=0.35, move_reject_ratio=0.20)
        head = make_head(640, 300, 100)
        selector.update([], [head])                        # 앵커 (640,300,316px)
        feed(selector, self.clock, moving_hand_frames(500, 30, 6), heads=[head])
        self.assertIsNotNone(selector._tracked_center)      # 활성 추적 확인
        expected_x = 640.0
        for _ in range(2):
            selector.update([], [make_head(740, 300, 100)])
            expected_x += TRACKED_DRIFT_ALPHA * (740.0 - expected_x)
        self.assertAlmostEqual(selector._head_anchor[0], expected_x, places=1)
        self.assertLess(selector._head_anchor[0], 700.0)    # 740으로 스냅되지 않음

    def test_reject_safety_valve_also_slower_while_tracked(self):
        # 2026-08-07 3차 실기 로그 — 2연속 합의만 막아서는 부족했다: 활성 추적
        # 중에도 REJECT_STREAK_LIMIT(5회)이 무조건 강제 수용해 지속 드리프트가
        # 그 뒷문으로 들어왔다. 안전판 자체도 활성 추적 중엔 훨씬 느려야 한다
        # (REJECT_STREAK_LIMIT_WHILE_TRACKED). 그 사이에는 완전히 얼지 않고
        # 살살만 반영되므로(4차) 원점과 정확히 같지는 않지만, 두 후보 지점
        # 근처로 스냅되지는 않아야 한다
        selector = self._selector(reject_ratio=0.35, move_reject_ratio=0.20)
        head = make_head(640, 300, 100)
        selector.update([], [head])
        feed(selector, self.clock, moving_hand_frames(500, 30, 6), heads=[head])
        self.assertIsNotNone(selector._tracked_center)
        # 서로 합의하지 않는(매번 다른 곳) 관측을 종전 한도(5)보다 많이 먹인다 —
        # 예전 한도라면 이미 강제 수용됐을 횟수인데 여전히 스냅되지 않아야 한다
        points = [(740, 300), (550, 390)]
        for i in range(REJECT_STREAK_LIMIT + 1):
            selector.update([], [make_head(*points[i % 2], 100)])
        self.assertLess(math.dist(selector._head_anchor[:2], (640.0, 300.0)), 40.0)
        self.assertGreater(math.dist(selector._head_anchor[:2], (740.0, 300.0)), 80.0)
        self.assertGreater(math.dist(selector._head_anchor[:2], (550.0, 390.0)), 80.0)
        self.assertLess(REJECT_STREAK_LIMIT, REJECT_STREAK_LIMIT_WHILE_TRACKED)

    def test_small_position_change_is_accepted_by_move_gate(self):
        # 정상 사용의 미세한 위치 변화(어깨너비의 10% 이내)는 받아들여야 한다
        selector = self._selector(reject_ratio=0.35, move_reject_ratio=0.20)
        selector.update([], [make_head(640, 300, 100)])
        selector.update([], [make_head(670, 300, 100)])            # 30px(9.5%) 이동 — 정상
        self.assertAlmostEqual(selector._head_anchor[0], 640.0 + 0.4 * 30.0, places=1)

    def test_move_gate_off_by_default(self):
        # 키(move_reject_ratio) 없으면 위치 관문 없음 — 폭 관문(reject_ratio)만 동작
        selector = self._selector(reject_ratio=0.35, move_reject_ratio=None)
        selector.update([], [make_head(640, 300, 100)])
        selector.update([], [make_head(740, 300, 100)])            # 폭 동일, 큰 이동
        # 위치 관문이 꺼져 있으니 받아들여진다(EMA로 이동)
        self.assertGreater(selector._head_anchor[0], 640.0)


class AnchorJumpLogTest(unittest.TestCase):
    """앵커 점프 로깅(2026-08-04 진단) — 실사용 중 이전 순간을 받아 적는 도구.

    겹침을 실측으로 재현하지 못해(뒷사람이 겹치면 포즈가 아예 안 잡힌다 —
    215관측 중 동시 관측 0회) 실사용 로그로 방향을 틀었다. 임계가 너무 높으면
    증상이 나도 안 찍히고, 낮으면 로그가 넘쳐 정작 그 순간이 묻힌다 — 그래서
    "점프는 찍고 정상은 조용하다"를 테스트로 박는다.
    """

    def setUp(self):
        # 앵커 획득 로그가 테스트 출력으로 새지 않게 — assertLogs는 해당 로거에
        # 직접 핸들러를 달아 잡으므로 전파를 꺼도 검증에는 지장이 없다
        self._logger = logging.getLogger("postprocess")
        self._propagate = self._logger.propagate
        self._logger.propagate = False

    def tearDown(self):
        self._logger.propagate = self._propagate

    def _selector(self, jump_log_ratio=0.20):
        config = make_config()
        config["head_anchor"] = {"reach_shoulder_widths": 1.58, "anchor_grace_sec": 1.0}
        if jump_log_ratio is not None:
            config["head_anchor"]["jump_log_ratio"] = jump_log_ratio
        return HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=FakeClock())

    # ※진단 로그는 INFO다(2026-08-05 사용자 요청 "cmd 창이 너무 난잡" — 검은 옷
    #   붕괴가 상시라 WARNING이면 콘솔이 진단으로 뒤덮인다). 파일(level INFO)에는
    #   그대로 남는다. 콘솔에 안 뜨는 것도 여기서 같이 박는다(WARNING 없음 검증).

    def test_logs_when_anchor_jumps_to_another_person(self):
        # 겹쳐서 나를 놓친 프레임 — 반경 안에 뒷사람만 남아 앵커가 넘어간다
        selector = self._selector()
        selector.update([], [make_head(640, 300, 100)])       # 나: 어깨 316px
        with self.assertLogs("postprocess", level="INFO") as captured:
            selector.update([], [make_head(730, 260, 70)])    # 뒷사람: 어깨 221px
        message = "\n".join(captured.output)
        self.assertIn("앵커 점프", message)
        self.assertIn("관측 1명", message)   # ★이 조합이 버그의 지문이다
        # 콘솔(WARNING 이상)에는 안 떠야 한다 — 전부 INFO
        self.assertNotIn("WARNING", message)

    def test_quiet_on_normal_jitter(self):
        # 실측 기준선(2026-08-04): 정상 사용의 어깨폭 변화는 중앙 0.02 · 95% 0.10.
        # 그 수준의 잔떨림에 로그가 찍히면 실사용에서 파일이 넘쳐 못 쓴다
        selector = self._selector()
        selector.update([], [make_head(640, 300, 100)])
        with self.assertNoLogs("postprocess", level="INFO"):
            for offset in range(1, 12):
                selector.update([], [make_head(640, 300, 100 + offset % 3)])

    def test_lost_anchor_logged_once(self):
        # 소실은 프레임마다 찍으면 정작 이전 순간이 묻힌다 — 시작할 때 한 번만
        selector = self._selector()
        selector.update([], [make_head(640, 300, 100)])
        with self.assertLogs("postprocess", level="INFO") as captured:
            for _ in range(4):
                selector.update([], [make_head(100, 100, 100)])   # 이음 반경 밖
        lost_lines = [line for line in captured.output if "앵커 머리 소실" in line]
        self.assertEqual(len(lost_lines), 1)

    def test_no_key_no_logging(self):
        # 진단은 꺼둘 수 있어야 한다 — 키를 지우면 동작도 로그도 종전 그대로
        selector = self._selector(jump_log_ratio=None)
        selector.update([], [make_head(640, 300, 100)])
        with self.assertNoLogs("postprocess", level="INFO"):
            selector.update([], [make_head(730, 260, 70)])


def takeover_config(idle_sec=0.3):
    """인계 켠 config — 테스트는 짧은 지속 시간으로 (실 config는 1.5초)."""
    config = make_config()
    config["hand_select"]["takeover_idle_sec"] = idle_sec
    return config


class IdleTakeoverTest(unittest.TestCase):
    """인계 — 추적 손이 놀고 있을 때만 제스처하는 손에 넘긴다 (2026-08-03 2차).

    사용자 보고: "아예 배에 있는 손에 고정돼 버린다." 종전엔 한번 잡힌 손을
    2초 소실 전에는 절대 안 놓아서, 배 앞의 쉬는 손이 먼저 잡히면 다른 손으로
    아무리 크게 제스처해도 못 들어왔다. 무작정 뺏게 하면 구 "배구 토스"가
    돌아오므로 **추적 손이 유휴일 때만** 도전을 받는다.
    """

    def test_gesturing_hand_takes_over_idle_hand(self):
        # 배 앞 손이 먼저 잡힌 뒤(살짝 움직여 획득) 계속 정지 — 그 사이 다른 손이
        # 제대로 지시하면 넘어와야 한다
        selector, clock = make_selector(takeover_config())
        belly = make_hand("left", "finger", (500, 550))
        feed(selector, clock, moving_hand_frames(400, 30, 6, y_px=550, side="left"))
        held = selector.user_hand_signal()
        self.assertIsNotNone(held)                       # 배 앞 손이 잡힌 상태(전제)
        self.assertGreater(held[1][1], 500)
        # 창(0.5초=15프레임)이 차고 → 유휴가 takeover_idle_sec(0.3초=9프레임)
        # 이어지고 → 도전자가 획득 임계(100px)를 넘겨야 한다. 넉넉히 40프레임
        frames = [[belly, make_hand("right", "finger", (800 + 12 * i, 300))]
                  for i in range(40)]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertLess(signal[1][1], 400)               # 위쪽에서 제스처한 손으로 인계

    def test_no_takeover_while_tracked_hand_is_moving(self):
        # ★안전 조건: 획을 그리는 중에는 어떤 손도 못 뺏는다 (구 배구 토스 방지)
        selector, clock = make_selector(takeover_config())
        feed(selector, clock, moving_hand_frames(300, 30, 6, y_px=300))
        self.assertIsNotNone(selector.user_hand_signal())
        frames = [[make_hand("right", "finger", (480 + 12 * i, 300)),      # 계속 이동
                   make_hand("left", "finger", (900 + 12 * i, 550))]       # 다른 손도 이동
                  for i in range(40)]
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertLess(signal[1][1], 450)               # 원래 손(위쪽)을 유지

    def test_brief_pause_between_strokes_does_not_hand_over(self):
        # ★2026-08-03 실측으로 추가: 유휴 조건만 걸었더니 **쓸기 사이의 짧은 멈춤**에
        # 인계가 나 추적이 두 손을 오갔다(10초에 4회·위 손 추적 64%). 지속 시간
        # (takeover_idle_sec)을 요구해 "잠깐 멈춤"과 "손을 놓아둔 상태"를 가른다
        selector, clock = make_selector(takeover_config(idle_sec=1.0))
        feed(selector, clock, moving_hand_frames(300, 30, 6, y_px=300))
        self.assertIsNotNone(selector.user_hand_signal())
        other = make_hand("left", "finger", (900, 560))
        frames = []
        for stroke_idx in range(3):                       # 쓸기 → 짧은 멈춤 → 쓸기
            base_x = 480 + 30 * stroke_idx
            frames += [[make_hand("right", "finger", (base_x + 20 * i, 300)),
                        make_hand("left", "finger", (900 + 20 * i, 560))]
                       for i in range(6)]                 # 둘 다 이동
            frames += [[make_hand("right", "finger", (base_x + 100, 300)), other]] * 9
        signal = feed(selector, clock, frames)             # 멈춤 0.3초 — 인계 금지
        self.assertIsNotNone(signal)
        self.assertLess(signal[1][1], 450)                 # 원래 손(위쪽) 유지

    def test_resting_other_hand_never_takes_over(self):
        # 쉬는 손은 인계 못 받는다 — 획득 규칙(이동+모양)을 그대로 쓰기 때문
        selector, clock = make_selector(takeover_config())
        feed(selector, clock, moving_hand_frames(400, 30, 6, y_px=300))
        tracked = selector.user_hand_signal()
        self.assertIsNotNone(tracked)
        rest = make_hand("left", "finger", (900, 560))
        frames = [[make_hand("right", "finger", (550, 300)), rest]] * 30   # 둘 다 정지
        signal = feed(selector, clock, frames)
        self.assertIsNotNone(signal)
        self.assertLess(signal[1][1], 450)               # 원래 손 유지


class WristGateTest(unittest.TestCase):
    """손목 심사(2026-08-03) — "내 팔 끝에 붙은 손인가".

    사용자 보고: "내 몸에서 뻗어나가는 손이 아닌데도 남의 몸 손이 잡힌다."
    머리 반경 하나로는 못 가른다 — 팔을 옆으로 뻗으면 내 머리~내 손 거리가
    옆 사람 손까지의 거리와 같다. 손목 기준이면 판정 영역이 내 팔을 따라간다.
    머리 폭 100px · wrist_reach 3.0 → 손목 반경 300px · 머리 반경 500px.
    """

    def setUp(self):
        config = make_config()
        config["head_anchor"] = {
            "reach_shoulder_widths": 1.58,          # 도달 반경 499px
            "wrist_reach_shoulder_widths": 0.95,    # 손목 반경 300px
            "anchor_grace_sec": 1.0,
        }
        self.clock = FakeClock()
        self.selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
                                     clock=self.clock)

    def _feed(self, frames, heads):
        return feed(self.selector, self.clock, frames, heads=heads)

    def _head(self, **kwargs):
        options = {"wrists": GATE_WRISTS, "shoulders": GATE_SHOULDERS}
        options.update(kwargs)
        return make_head(640, 200, 100, **options)

    def _intruder_frames(self):
        """머리 반경(500px)은 통과하지만 손목 반경(300px) 밖인 손 — 옆 사람 손."""
        return [[make_hand("right", "finger", (1080, 200 + 30 * step_idx))]
                for step_idx in range(6)]

    def test_hand_near_own_wrist_is_acquired(self):
        # 내 손목 옆에서 움직이는 손 — 정상 획득
        self.assertIsNotNone(self._feed(moving_hand_frames(500, 30, 6), [self._head()]))

    def test_hand_far_from_wrists_is_blocked(self):
        # ★핵심: 머리 반경은 통과하지만 어느 손목에서도 멀다 — 종전엔 이게 통과했다
        self.assertIsNone(self._feed(self._intruder_frames(), [self._head()]))

    def test_same_hand_passes_without_wrist_gate(self):
        # 대조군: 손목 심사가 꺼져 있으면(키 없음·구 동작) 같은 손이 통과한다
        config = make_config()
        config["head_anchor"] = {"reach_shoulder_widths": 1.58, "anchor_grace_sec": 1.0}
        clock = FakeClock()
        selector = HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX, clock=clock)
        self.assertIsNotNone(
            feed(selector, clock, self._intruder_frames(), heads=[self._head()]))

    def test_invisible_wrist_is_not_used(self):
        # ★2026-08-04 사용자 관찰("카메라에 안 보여도 추론이 손을 만들어 낸다"):
        # 안 보이는 손목의 추측 좌표로 판정 영역을 만들면 안 된다 — 심사를
        # 건너뛰고 머리 반경만 쓴다(인식 우선). 그래서 이 손은 통과한다
        head = self._head(wrist_visibility=0.1)
        self.assertIsNotNone(self._feed(self._intruder_frames(), [head]))

    def test_wrist_beyond_arm_length_is_not_used(self):
        # ★2026-08-04 사용자 관찰("남의 손 난입 시 내 팔이 내려가 있어도 그 손을
        # 내 손목으로 잡아 올린다"): 끌려간 손목은 어깨에서 팔 길이(256px)를
        # 넘는다 — 그 추정은 버리고 심사를 건너뛴다
        head = self._head(wrists=((1080, 260), (1100, 280)))   # 어깨에서 520px+
        self.assertIsNotNone(self._feed(self._intruder_frames(), [head]))

    def test_tracked_hand_exempt_from_wrist_gate(self):
        # 추적 면제 — 손목 추정이 뒤처져도(포즈 10 FPS) 진행 중인 획이 안 끊긴다
        self.assertIsNotNone(self._feed(moving_hand_frames(500, 30, 6), [self._head()]))
        stale = self._head(wrists=((560, 400), (620, 420)))
        signal = self._feed([[make_hand("right", "finger", (680 + 40 * i, 400))]
                             for i in range(1, 4)], [stale])
        self.assertIsNotNone(signal)

    def test_other_persons_hand_is_rejected_by_ownership(self):
        # ★2026-08-04 실기(사용자 보고 — "다른 사람 난입 시 노이즈"): 반경만으로는
        # 원리적으로 못 가른다(팔 뻗은 내 손까지 ≈ 옆 사람까지). 두 사람의 골격을
        # 다 보면 **누구 손목이 더 가까운가**로 임계값 없이 갈린다.
        # 난입자 손목이 그 손 바로 옆에 있다 — 내 손목보다 가까우니 남의 손이다
        intruder = make_head(1000, 220, 100, wrists=((1075, 205), (1150, 230)),
                             shoulders=((950, 330), (1090, 330)))
        signal = self._feed(self._intruder_frames(), [self._head(), intruder])
        self.assertIsNone(signal)

    def test_own_hand_survives_when_intruder_is_farther(self):
        # ★반대 방향 안전: 난입자가 있어도 **내 손목이 더 가까우면** 내 손이다.
        # 소유자 판정이 내 손을 자르면 인식이 죽으므로 이쪽이 더 중요하다
        intruder = make_head(1000, 220, 100, wrists=((1075, 205), (1150, 230)),
                             shoulders=((950, 330), (1090, 330)))
        signal = self._feed(moving_hand_frames(500, 30, 6), [self._head(), intruder])
        self.assertIsNotNone(signal)

    def test_unreliable_intruder_wrists_reject_nobody(self):
        # 난입자 손목이 안 보이면(추측 좌표) 소유자 판정에서 뺀다 — 못 믿을
        # 관측으로 내 손을 자르지 않는다(인식 우선). 그러면 종전처럼 반경만 남는다
        intruder = make_head(1000, 220, 100, wrists=((1075, 205), (1150, 230)),
                             shoulders=((950, 330), (1090, 330)), wrist_visibility=0.1)
        signal = self._feed(moving_hand_frames(500, 30, 6), [self._head(), intruder])
        self.assertIsNotNone(signal)

    def test_no_wrists_falls_back_to_head_reach(self):
        # 손목 관측이 비면(포즈가 손목을 못 냄) 심사를 건너뛴다 — 인식 우선 폴백
        head = make_head(640, 200, 100)          # wrists=() — 기본값
        self.assertIsNotNone(self._feed(moving_hand_frames(500, 30, 6), [head]))


if __name__ == "__main__":
    unittest.main()
