"""dual_camera 단위 테스트 — 손 품질 점수·활성 카메라 히스테리시스 전환.

카메라·모델 없이 순수 로직만 검증한다.
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.capture.dual_camera import CameraArbiter, score_hand_quality


def _fake_hand(span_px, conf):
    landmarks = np.array([[0.0, 0.0, 0.0], [span_px, 0.0, 0.0]], dtype=np.float32)
    return SimpleNamespace(landmarks=landmarks, conf=conf)


class ScoreHandQualityTest(unittest.TestCase):
    def test_no_hands_scores_zero(self):
        self.assertEqual(score_hand_quality([], 1280, 0.10), 0.0)

    def test_span_at_good_ratio_scores_confidence(self):
        hand = _fake_hand(span_px=128, conf=0.8)   # 128/1280 = 0.10 = good_span_ratio
        self.assertAlmostEqual(score_hand_quality([hand], 1280, 0.10), 0.8, places=6)

    def test_smaller_span_scores_proportionally_lower(self):
        hand = _fake_hand(span_px=64, conf=1.0)   # 64/1280 = 0.05 -> 절반
        self.assertAlmostEqual(score_hand_quality([hand], 1280, 0.10), 0.5, places=6)

    def test_span_above_good_ratio_caps_at_confidence(self):
        hand = _fake_hand(span_px=256, conf=0.9)   # 0.20 비율이어도 1.0으로 캡
        self.assertAlmostEqual(score_hand_quality([hand], 1280, 0.10), 0.9, places=6)

    def test_uses_best_of_multiple_hands(self):
        hands = [_fake_hand(64, 1.0), _fake_hand(128, 0.8)]
        self.assertAlmostEqual(score_hand_quality(hands, 1280, 0.10), 0.8, places=6)


class CameraArbiterTest(unittest.TestCase):
    def test_stays_on_active_when_lead_below_margin(self):
        arbiter = CameraArbiter(switch_margin=0.1, switch_frames=3)
        for _ in range(10):
            self.assertEqual(arbiter.update([0.5, 0.55]), 0)   # 차이 0.05 < margin

    def test_switches_only_after_sustained_lead(self):
        arbiter = CameraArbiter(switch_margin=0.1, switch_frames=3)
        self.assertEqual(arbiter.update([0.3, 0.5]), 0)   # 1회차 — 아직 전환 안 함
        self.assertEqual(arbiter.update([0.3, 0.5]), 0)   # 2회차
        self.assertEqual(arbiter.update([0.3, 0.5]), 1)   # 3회 연속 — 전환

    def test_interrupted_lead_resets_streak(self):
        arbiter = CameraArbiter(switch_margin=0.1, switch_frames=3)
        arbiter.update([0.3, 0.5])
        arbiter.update([0.3, 0.5])
        arbiter.update([0.5, 0.5])   # 우위 끊김 — 카운트 리셋
        self.assertEqual(arbiter.update([0.3, 0.5]), 0)   # streak 1(재시작)
        self.assertEqual(arbiter.update([0.3, 0.5]), 0)   # streak 2
        self.assertEqual(arbiter.update([0.3, 0.5]), 1)   # streak 3 -> 전환

    def test_can_switch_back(self):
        arbiter = CameraArbiter(switch_margin=0.1, switch_frames=2, initial_active=1)
        self.assertEqual(arbiter.update([0.9, 0.1]), 1)
        self.assertEqual(arbiter.update([0.9, 0.1]), 0)

    def test_selects_best_of_three_cameras_after_sustained_lead(self):
        arbiter = CameraArbiter(switch_margin=0.1, switch_frames=2)
        self.assertEqual(arbiter.update([0.2, 0.3, 0.8]), 0)
        self.assertEqual(arbiter.update([0.2, 0.3, 0.8]), 2)

    def test_sticky_owner_does_not_switch_when_same_hand_is_visible_in_both(self):
        arbiter = CameraArbiter(0.1, 2, sticky_ownership=True, release_frames=3)
        for _ in range(20):
            self.assertEqual(arbiter.update([0.4, 0.95], [True, True]), 0)

    def test_sticky_owner_switches_only_after_loss_and_stable_candidate(self):
        arbiter = CameraArbiter(0.1, 2, sticky_ownership=True, release_frames=3,
                                cooldown_frames=4)
        self.assertEqual(arbiter.update([0.0, 0.9], [False, True]), 0)
        self.assertEqual(arbiter.update([0.0, 0.9], [False, True]), 0)
        self.assertEqual(arbiter.update([0.0, 0.9], [False, True]), 0)
        self.assertEqual(arbiter.update([0.0, 0.9], [False, True]), 1)

    def test_sticky_owner_candidate_streak_resets_when_candidate_disappears(self):
        arbiter = CameraArbiter(0.1, 2, sticky_ownership=True, release_frames=1)
        self.assertEqual(arbiter.update([0.0, 0.9], [False, True]), 0)
        self.assertEqual(arbiter.update([0.0, 0.0], [False, False]), 0)
        self.assertEqual(arbiter.update([0.0, 0.9], [False, True]), 0)
        self.assertEqual(arbiter.update([0.0, 0.9], [False, True]), 1)


if __name__ == "__main__":
    unittest.main()
