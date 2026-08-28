"""person_lock 단위 테스트 — 카메라·얼굴 모델 없이 잠금 로직만 검증한다.

얼굴 결과는 FaceLandmarks와 같은 필드를 가진 대역(FakeFace)으로 만들고,
초점 선명도는 sharpness_fn 주입으로 고정해 결정적으로 테스트한다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.postprocess.person_lock import PersonLock

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


class FakeFace:
    """FaceLandmarks와 같은 필드를 가진 테스트 대역 (mediapipe 임포트 회피)."""

    def __init__(self, center_x, center_y, size_px=200.0, conf=0.9):
        half = size_px / 2.0
        self.bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
        self.conf = conf


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config(enabled=True):
    return {
        "person_lock": {
            "enabled": enabled,
            "lock_frame_count": 3,
            "follow_radius_ratio": 0.25,
            "release_sec": 2.0,
            "sharpness_weight": 0.5,
        },
    }


def make_lock(config=None, sharpness_by_x=None):
    """sharpness_by_x: 얼굴 박스 중심 x -> 선명도. 미지정 시 모두 같은 값."""

    def sharpness_fn(frame, face_box):
        if sharpness_by_x is None:
            return 100.0
        center_x = (face_box[0] + face_box[2]) / 2.0
        for x_range, value in sharpness_by_x.items():
            if x_range[0] <= center_x <= x_range[1]:
                return value
        return 10.0

    clock = FakeClock()
    lock = PersonLock(
        config or make_config(), FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
        clock=clock, sharpness_fn=sharpness_fn,
    )
    return lock, clock


FRAME = np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=np.uint8)


def lock_face(lock, clock, face):
    """lock_frame_count(3) 프레임 연속 공급해 face에게 잠근다."""
    for _ in range(3):
        lock.update(FRAME, [face])
        clock.tick(1 / 30)


class LockSelectionTest(unittest.TestCase):
    def test_locks_after_consecutive_frames(self):
        lock, clock = make_lock()
        face = FakeFace(640, 360)
        for _ in range(2):
            lock.update(FRAME, [face])
            clock.tick(1 / 30)
        self.assertIsNone(lock.locked_face)   # lock_frame_count(3) 미만
        lock.update(FRAME, [face])
        self.assertIsNotNone(lock.locked_face)

    def test_sharpest_face_wins_over_blurry(self):
        # 같은 크기 두 얼굴 — 왼쪽(x<600)이 흐릿, 오른쪽이 선명(초점 맞음)
        lock, clock = make_lock(sharpness_by_x={(0, 600): 5.0, (601, 1280): 500.0})
        blurry = FakeFace(300, 360)
        sharp = FakeFace(900, 360)
        for _ in range(3):
            lock.update(FRAME, [blurry, sharp])
            clock.tick(1 / 30)
        self.assertIsNotNone(lock.locked_face)
        locked_cx = (lock.locked_face.bbox[0] + lock.locked_face.bbox[2]) / 2.0
        self.assertGreater(locked_cx, 600)      # 선명한 쪽이 잠겼다

    def test_no_sharpness_computed_while_locked(self):
        # 잠금 중 추적은 위치만 쓴다 — 선명도(라플라시안) 계산이 매 프레임 돌면
        # 추론 FPS를 갉아먹는 순수 낭비 (2026-07-20 최적화의 회귀 방지)
        call_count = [0]

        def counting_sharpness_fn(frame, face_box):
            call_count[0] += 1
            return 100.0

        clock = FakeClock()
        lock = PersonLock(make_config(), FRAME_WIDTH_PX, FRAME_HEIGHT_PX,
                          clock=clock, sharpness_fn=counting_sharpness_fn)
        face = FakeFace(640, 360)
        lock_face(lock, clock, face)
        self.assertIsNotNone(lock.locked_face)
        calls_at_lock = call_count[0]

        for _ in range(10):   # 잠금 유지 구간 — 선명도 호출이 늘면 안 된다
            lock.update(FRAME, [face])
            clock.tick(1 / 30)
        self.assertEqual(call_count[0], calls_at_lock)

    def test_release_after_absence(self):
        lock, clock = make_lock()
        face = FakeFace(640, 360)
        lock_face(lock, clock, face)
        self.assertIsNotNone(lock.locked_face)
        clock.tick(2.5)                          # release_sec(2.0) 초과 공백
        lock.update(FRAME, [])
        self.assertIsNone(lock.locked_face)

    def test_disabled_lock_tracks_highest_conf_face(self):
        # 잠금 비활성 — head_tracker 신호용으로 최고 신뢰도 얼굴을 추적한다
        lock, _ = make_lock(make_config(enabled=False))
        low = FakeFace(300, 360, conf=0.5)
        high = FakeFace(900, 360, conf=0.95)
        lock.update(FRAME, [low, high])
        self.assertIsNotNone(lock.locked_face)
        self.assertEqual(lock.locked_face.conf, 0.95)


class LockGenerationTest(unittest.TestCase):
    """lock_generation — head_tracker가 캘리브레이션 재실행 여부를 판단하는 신호 (2026-07-18)."""

    def test_increments_only_on_new_lock(self):
        lock, clock = make_lock()
        face = FakeFace(640, 360)
        self.assertEqual(lock.lock_generation, 0)
        lock_face(lock, clock, face)
        self.assertEqual(lock.lock_generation, 1)
        lock.update(FRAME, [face])   # 같은 사람 계속 추적 — 세대 변화 없음
        self.assertEqual(lock.lock_generation, 1)

    def test_increments_again_after_release_and_relock(self):
        lock, clock = make_lock()
        face = FakeFace(640, 360)
        lock_face(lock, clock, face)
        self.assertEqual(lock.lock_generation, 1)
        clock.tick(2.5)
        lock.update(FRAME, [])
        self.assertIsNone(lock.locked_face)
        lock_face(lock, clock, face)
        self.assertEqual(lock.lock_generation, 2)


if __name__ == "__main__":
    unittest.main()
