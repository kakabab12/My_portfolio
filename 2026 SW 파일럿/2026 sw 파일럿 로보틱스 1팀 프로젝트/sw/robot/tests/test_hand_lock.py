"""hand_lock 단위 테스트 — 배경 인물 손 무시(크기 필터 + 연속성 잠금 추적)."""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.hand_lock import PrimaryHandTracker

FRAME_WIDTH_PX = 1280
CFG = {"min_span_ratio": 0.05, "continuity_span_ratio": 3.5, "release_sec": 1.5}


def _fake_hand(center_px, span_px, conf):
    """PrimaryHandTracker가 쓰는 hand_center_point/hand_span_px용 21점 가짜 손.

    hand_center_point는 21점(HAND_KPT_COUNT) 미만이면 무조건 None을 돌려주므로
    (실제 MediaPipe 출력은 항상 21점), 점 2개짜리로 만들면 안 된다. 양 끝 2점으로
    span_px를, 나머지 19점을 정확히 center에 둬 평균(mean)이 center_px와
    정확히 일치하게 한다.
    """
    x, y = center_px
    half = span_px / 2.0
    points = [[x - half, y, 0.0], [x + half, y, 0.0]] + [[x, y, 0.0]] * 19
    landmarks = np.array(points, dtype=np.float32)
    return SimpleNamespace(landmarks=landmarks, conf=conf)


class _FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


class PrimaryHandTrackerTest(unittest.TestCase):
    def test_no_hands_returns_none(self):
        tracker = PrimaryHandTracker(CFG, clock=_FakeClock())
        self.assertIsNone(tracker.select([], FRAME_WIDTH_PX))

    def test_too_small_hand_is_ignored(self):
        # min_span_ratio 0.05 * 1280 = 64px — 그보다 작은 손(배경의 먼 사람)은 후보 제외
        tracker = PrimaryHandTracker(CFG, clock=_FakeClock())
        far_hand = _fake_hand((900, 400), span_px=40.0, conf=0.95)
        self.assertIsNone(tracker.select([far_hand], FRAME_WIDTH_PX))

    def test_no_lock_yet_picks_largest_candidate(self):
        tracker = PrimaryHandTracker(CFG, clock=_FakeClock())
        near_operator = _fake_hand((300, 300), span_px=150.0, conf=0.7)
        background_person = _fake_hand((900, 200), span_px=80.0, conf=0.95)   # 신뢰도는 더 높음
        chosen = tracker.select([background_person, near_operator], FRAME_WIDTH_PX)
        self.assertIs(chosen, near_operator)   # 더 큰(가까운) 손이 우선

    def test_new_hand_requires_configured_consecutive_frames(self):
        config = dict(CFG, acquire_frames=3, acquire_center_span_ratio=1.25)
        tracker = PrimaryHandTracker(config, clock=_FakeClock())
        hand = _fake_hand((300, 300), span_px=150.0, conf=0.9)

        self.assertIsNone(tracker.select([hand], FRAME_WIDTH_PX))
        self.assertIsNone(tracker.select([hand], FRAME_WIDTH_PX))
        self.assertIs(tracker.select([hand], FRAME_WIDTH_PX), hand)

    def test_single_frame_false_positive_does_not_accumulate(self):
        config = dict(CFG, acquire_frames=3, acquire_center_span_ratio=1.25)
        tracker = PrimaryHandTracker(config, clock=_FakeClock())
        false_positive = _fake_hand((300, 300), span_px=150.0, conf=0.9)

        self.assertIsNone(tracker.select([false_positive], FRAME_WIDTH_PX))
        self.assertIsNone(tracker.select([], FRAME_WIDTH_PX))
        self.assertIsNone(tracker.select([false_positive], FRAME_WIDTH_PX))

    def test_continuity_beats_higher_confidence_newcomer(self):
        clock = _FakeClock()
        tracker = PrimaryHandTracker(CFG, clock=clock)
        operator_frame1 = _fake_hand((300, 300), span_px=150.0, conf=0.7)
        tracker.select([operator_frame1], FRAME_WIDTH_PX)   # 조작자 손을 획득(잠금)

        clock.advance(0.05)
        operator_frame2 = _fake_hand((320, 310), span_px=150.0, conf=0.6)   # 살짝 움직임
        background_person = _fake_hand((950, 150), span_px=140.0, conf=0.99)   # 더 크고 신뢰도도 높음
        chosen = tracker.select([background_person, operator_frame2], FRAME_WIDTH_PX)
        self.assertIs(chosen, operator_frame2)   # 배경 사람이 더 확실해 보여도 연속성 우선

    def test_brief_loss_returns_none_without_switching(self):
        clock = _FakeClock()
        tracker = PrimaryHandTracker(CFG, clock=clock)
        operator_hand = _fake_hand((300, 300), span_px=150.0, conf=0.7)
        tracker.select([operator_hand], FRAME_WIDTH_PX)

        clock.advance(0.5)   # release_sec(1.5) 이내 — 유예 중
        background_person = _fake_hand((950, 150), span_px=140.0, conf=0.99)
        chosen = tracker.select([background_person], FRAME_WIDTH_PX)
        self.assertIsNone(chosen)   # 배경 사람으로 넘어가지 않고 소실로 처리

    def test_reacquires_new_hand_after_release_timeout(self):
        clock = _FakeClock()
        tracker = PrimaryHandTracker(CFG, clock=clock)
        operator_hand = _fake_hand((300, 300), span_px=150.0, conf=0.7)
        tracker.select([operator_hand], FRAME_WIDTH_PX)

        clock.advance(2.0)   # release_sec(1.5) 초과 — 정체성 해제
        new_person = _fake_hand((950, 150), span_px=140.0, conf=0.99)
        chosen = tracker.select([new_person], FRAME_WIDTH_PX)
        self.assertIs(chosen, new_person)   # 이제는 새로 획득 가능

    def test_continuation_must_be_within_radius(self):
        clock = _FakeClock()
        tracker = PrimaryHandTracker(CFG, clock=clock)
        operator_hand = _fake_hand((300, 300), span_px=150.0, conf=0.7)
        tracker.select([operator_hand], FRAME_WIDTH_PX)

        clock.advance(0.05)
        # continuity_span_ratio(3.5) * span(150) = 525px 반경 — 훨씬 멀리 있는 손은
        # "같은 사람"으로 이어지지 않는다(다른 사람 손이 우연히 반경 안 들어오는 것 방지)
        far_unrelated = _fake_hand((1200, 700), span_px=150.0, conf=0.9)
        chosen = tracker.select([far_unrelated], FRAME_WIDTH_PX)
        self.assertIsNone(chosen)


if __name__ == "__main__":
    unittest.main()
