"""head_shake 단위 테스트 — 카메라·mediapipe 없이 왕복(반전) 판정 로직만 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_shake import HeadShakeDetector

EYE_DIST_PX = 40.0   # 정규화 자 — 실제 값은 무관(비율만 중요)


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config(min_amplitude=0.3, min_reversals=3, window_sec=1.5,
                axis_dominance=1.5, cooldown_sec=1.0, min_duration_sec=0.0):
    return {
        "mode_switch": {
            "head_shake": {
                "min_amplitude": min_amplitude,
                "min_reversals": min_reversals,
                "window_sec": window_sec,
                "axis_dominance": axis_dominance,
                "cooldown_sec": cooldown_sec,
                "min_duration_sec": min_duration_sec,
            }
        }
    }


class HeadShakeDetectorTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.detector = HeadShakeDetector(make_config(), clock=self.clock)

    def _feed(self, xs, y=0.0):
        """정규화 x좌표 목록을 순서대로 먹인다(같은 y) -> 마지막 프레임 결과."""
        result = False
        for x in xs:
            result = self.detector.update((x * EYE_DIST_PX, y * EYE_DIST_PX), EYE_DIST_PX)
            self.clock.tick(0.03)
        return result

    def test_no_movement_never_fires(self):
        result = self._feed([0.0] * 20)
        self.assertFalse(result)

    def test_single_direction_sweep_does_not_fire(self):
        # 한쪽으로만 계속 이동 — 반전이 없어 절대 안 걸린다
        result = self._feed([i * 0.05 for i in range(20)])
        self.assertFalse(result)

    def test_enough_reversals_within_window_fires(self):
        # 0 -> +0.4(1획, 기준만 세움) -> -0.4(반전1) -> +0.4(반전2) -> -0.4(반전3, 확정)
        xs = [0.0, 0.4, -0.4, 0.4, -0.4]
        result = self._feed(xs)
        self.assertTrue(result)

    def test_reversals_below_minimum_do_not_fire(self):
        xs = [0.0, 0.4, -0.4]   # 반전 1회뿐(min_reversals=3 미달)
        result = self._feed(xs)
        self.assertFalse(result)

    def test_reversals_outside_window_are_forgotten(self):
        detector = HeadShakeDetector(make_config(window_sec=0.5), clock=self.clock)
        detector.update((0.0, 0.0), EYE_DIST_PX)
        self.clock.tick(0.03)
        detector.update((0.4 * EYE_DIST_PX, 0.0), EYE_DIST_PX)   # 첫 획
        self.clock.tick(0.03)
        detector.update((0.0, 0.0), EYE_DIST_PX)                  # 반전 1
        self.clock.tick(1.0)   # window_sec(0.5) 밖으로 밀려남
        detector.update((0.4 * EYE_DIST_PX, 0.0), EYE_DIST_PX)   # 반전 2 — 창 안엔 이것뿐
        self.clock.tick(0.03)
        result = detector.update((0.0, 0.0), EYE_DIST_PX)         # 반전 3
        self.assertFalse(result)   # 창 안엔 2회뿐(min_reversals=3 미달)

    def test_vertical_dominant_movement_does_not_fire(self):
        # 수직(끄덕임) 성분이 더 크면 반전으로 세지 않는다
        detector = HeadShakeDetector(make_config(axis_dominance=1.5), clock=self.clock)
        xs_ys = [
            (0.0, 0.0), (0.3, 0.5), (-0.3, -0.5), (0.3, 0.5), (-0.3, -0.5),
        ]
        result = False
        for x, y in xs_ys:
            result = detector.update((x * EYE_DIST_PX, y * EYE_DIST_PX), EYE_DIST_PX)
            self.clock.tick(0.03)
        self.assertFalse(result)

    def test_face_loss_resets_progress(self):
        self.detector.update((0.0, 0.0), EYE_DIST_PX)
        self.clock.tick(0.03)
        self.detector.update((0.4 * EYE_DIST_PX, 0.0), EYE_DIST_PX)
        self.clock.tick(0.03)
        self.detector.update((0.0, 0.0), EYE_DIST_PX)   # 반전 1 — 진행 중
        self.assertGreater(self.detector.progress_ratio, 0.0)
        self.detector.update(None, 0.0)                  # 얼굴 소실
        self.assertEqual(self.detector.progress_ratio, 0.0)

    def test_cooldown_blocks_immediate_refire(self):
        xs = [0.0, 0.4, -0.4, 0.4, -0.4]   # 1차 확정
        self.assertTrue(self._feed(xs))
        # 곧바로 같은 패턴 반복 — cooldown_sec(1.0) 안이라 재발화 없음
        result = self._feed([0.0, 0.4, -0.4, 0.4, -0.4])
        self.assertFalse(result)

    def test_fires_again_after_cooldown(self):
        xs = [0.0, 0.4, -0.4, 0.4, -0.4]
        self.assertTrue(self._feed(xs))
        self.clock.tick(1.2)   # cooldown_sec(1.0) 경과
        result = self._feed([0.0, 0.4, -0.4, 0.4, -0.4])
        self.assertTrue(result)


class MinDurationTest(unittest.TestCase):
    """빠른 좌우 까딱임 차단(2026-08-04) — 진폭·횟수는 채워도 너무 빨리 몰리면 보류."""

    def _feed_with_gap(self, detector, clock, xs, gap_sec):
        result = False
        for x in xs:
            result = detector.update((x * EYE_DIST_PX, 0.0), EYE_DIST_PX)
            clock.tick(gap_sec)
        return result

    def test_fast_reversals_below_min_duration_do_not_fire(self):
        # 프레임 간격 0.03초 — 반전 3회가 ~0.09초에 다 몰림 < min_duration_sec(0.6)
        clock = FakeClock()
        detector = HeadShakeDetector(make_config(min_duration_sec=0.6), clock=clock)
        result = self._feed_with_gap(detector, clock, [0.0, 0.4, -0.4, 0.4, -0.4], 0.03)
        self.assertFalse(result)   # 까딱임 — 진폭·횟수는 채웠지만 폭이 너무 좁다

    def test_slow_natural_shake_still_fires(self):
        # 반전 간격을 넉넉히(0.5초) 둬 span(~1.0초)이 min_duration_sec(0.6)을
        # 여유 있게 넘긴다(0.3초 간격은 부동소수점 누적 오차로 경계값 0.6과
        # 맞닿아 불안정 — 여유를 확실히 뒀다)
        clock = FakeClock()
        detector = HeadShakeDetector(make_config(min_duration_sec=0.6), clock=clock)
        result = self._feed_with_gap(detector, clock, [0.0, 0.4, -0.4, 0.4, -0.4], 0.5)
        self.assertTrue(result)   # 의도적 흔들기 — 회귀 방지

    def test_no_duration_floor_configured_keeps_old_behavior(self):
        # min_duration_sec 키 없으면(기본 0.0) 빠른 반전도 종전대로 확정된다
        clock = FakeClock()
        detector = HeadShakeDetector(make_config(), clock=clock)   # 기본 0.0
        result = self._feed_with_gap(detector, clock, [0.0, 0.4, -0.4, 0.4, -0.4], 0.03)
        self.assertTrue(result)

    def test_burst_that_keeps_repeating_never_fires(self):
        # 빠르게 계속 까딱거려도(반전이 계속 빨리 몰림) 끝내 확정되지 않는다
        clock = FakeClock()
        detector = HeadShakeDetector(make_config(min_duration_sec=0.6), clock=clock)
        xs = [0.0] + [0.4, -0.4] * 10   # 20회 왕복, 매번 0.03초 간격
        result = self._feed_with_gap(detector, clock, xs, 0.03)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
