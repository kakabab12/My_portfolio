"""gesture_filter 단위 테스트 — 카메라·모델 없이 판정 로직만 검증한다.

동작 하나만 남긴 단순화 버전: 손등이 카메라를 향한 채(class_name="손등팔등") N프레임
안정 유지되면 next_item 이벤트 확정.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.person_lock import HandObservation


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config():
    return {
        "detect": {"cooldown_sec": 1.0},
        "gestures": {
            "next_item": {"stable_frame_count": 5, "max_static_move_ratio": 0.08},
        },
    }


def obs(side, gesture, conf=0.9, cx_ratio=0.5):
    return HandObservation(side=side, gesture=gesture, conf=conf, cx_ratio=cx_ratio)


FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정


class GestureFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), clock=self.clock)

    def _feed(self, observations, frame_count=1, dt_sec=FRAME_DT_SEC):
        """frame_count 프레임 공급 — 첫 확정 이벤트를 즉시 돌려준다 (없으면 None)."""
        for _ in range(frame_count):
            event = self.filter.filter_observations(observations)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return None


class NextItemTest(GestureFilterTestBase):
    def test_fist_stable_fires_next_item(self):
        event = self._feed([obs("right", "손등팔등")], frame_count=5)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "next_item")
        self.assertEqual(event.hand_side, "right")

    def test_either_hand_fires(self):
        event = self._feed([obs("left", "손등팔등")], frame_count=5)
        self.assertEqual(event.class_name, "next_item")
        self.assertEqual(event.hand_side, "left")

    def test_short_hold_does_not_fire(self):
        event = self._feed([obs("right", "손등팔등")], frame_count=3)
        self.assertIsNone(event)

    def test_moving_fist_does_not_fire(self):
        event = None
        for i in range(10):
            event = self._feed([obs("right", "손등팔등", cx_ratio=0.1 + i * 0.1)])
        self.assertIsNone(event)

    def test_no_observations_does_not_fire(self):
        event = self._feed([], frame_count=10)
        self.assertIsNone(event)

    def test_unmapped_gesture_does_not_fire(self):
        event = self._feed([obs("right", "palm")], frame_count=10)
        self.assertIsNone(event)


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        self._feed([obs("right", "손등팔등")], frame_count=5)
        event = self._feed([obs("right", "손등팔등")], frame_count=5)  # 쿨다운(1초) 내
        self.assertIsNone(event)
        self.clock.tick(1.0)
        event = self._feed([obs("right", "손등팔등")], frame_count=5)
        self.assertIsNotNone(event)


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
