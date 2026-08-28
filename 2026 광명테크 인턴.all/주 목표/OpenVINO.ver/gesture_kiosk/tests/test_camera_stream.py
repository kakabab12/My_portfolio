"""카메라 새 프레임 동기화(2026-07-20) 테스트 — 실제 카메라 없이 게시/대기 로직만 검증.

capture_new_frame은 같은 프레임의 중복 추론(카메라 30 FPS < 추론 40+ FPS 낭비)을
막는 장치다: 새 일련번호가 나올 때까지 재우고, 카메라 멈칫 땐 기존 프레임으로 진행.
"""
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import src.capture.camera_stream as camera_stream_module
from src.capture.camera_stream import CameraStream


def _make_stream():
    return CameraStream({"camera": {"device_id": 0}})   # start() 안 함 — 장치 미사용


def _frame(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


class CaptureNewFrameTest(unittest.TestCase):
    def setUp(self):
        # 대기 한도를 짧게 — 멈칫 폴백 테스트가 느려지지 않게
        self._saved_timeout = camera_stream_module.NEW_FRAME_TIMEOUT_SEC
        camera_stream_module.NEW_FRAME_TIMEOUT_SEC = 0.05

    def tearDown(self):
        camera_stream_module.NEW_FRAME_TIMEOUT_SEC = self._saved_timeout

    def test_returns_immediately_when_newer_frame_exists(self):
        stream = _make_stream()
        stream._publish_frame(_frame(10))
        frame, seq = stream.capture_new_frame(last_seq=0)
        self.assertEqual(seq, 1)
        self.assertEqual(frame[0, 0, 0], 10)

    def test_same_seq_waits_then_returns_stale(self):
        # 새 프레임이 안 오면(카메라 멈칫) 한도 후 기존 프레임으로 진행 — seq 불변
        stream = _make_stream()
        stream._publish_frame(_frame(10))
        frame, seq = stream.capture_new_frame(last_seq=1)   # 이미 본 프레임
        self.assertEqual(seq, 1)                            # 그대로 — 다음 호출도 새것 대기
        self.assertEqual(frame[0, 0, 0], 10)

    def test_wakes_up_when_frame_arrives_during_wait(self):
        # 대기 중 캡처 스레드가 게시하면 즉시 깨어난다 (조건변수 통지)
        stream = _make_stream()
        stream._publish_frame(_frame(10))
        timer = threading.Timer(0.01, lambda: stream._publish_frame(_frame(20)))
        timer.start()
        try:
            frame, seq = stream.capture_new_frame(last_seq=1)
        finally:
            timer.cancel()
        self.assertEqual(seq, 2)
        self.assertEqual(frame[0, 0, 0], 20)

    def test_no_frame_ever_raises(self):
        stream = _make_stream()
        with self.assertRaises(RuntimeError):
            stream.capture_new_frame(last_seq=0)


class _FakeCapture:
    """cv2.VideoCapture 대역 — switch_device()가 실제 카메라 없이 돌게 한다."""

    def __init__(self, value=99):
        self._value = value
        self.released = False

    def read(self):
        return True, _frame(self._value)

    def release(self):
        self.released = True


class SwitchDeviceTest(unittest.TestCase):
    """카메라 전환(2026-07-27 신설 — 키오스크 위/아래 2대 대응) 단위 테스트."""

    def test_switch_updates_device_id_and_resumes_capture(self):
        stream = _make_stream()
        stream._publish_frame(_frame(10))   # 전환 전 프레임 — 전환 후엔 안 쓰여야 한다
        fake_cap = _FakeCapture(value=42)
        original_init_camera = camera_stream_module.init_camera
        camera_stream_module.init_camera = lambda config: fake_cap
        try:
            stream.switch_device(1)
            frame, seq = stream.capture_new_frame(last_seq=0)
        finally:
            stream.stop()
            camera_stream_module.init_camera = original_init_camera

        self.assertEqual(stream._config["camera"]["device_id"], 1)
        self.assertEqual(frame[0, 0, 0], 42)   # 새 장치(fake_cap)에서 온 프레임
        self.assertGreaterEqual(seq, 1)

    def test_switch_releases_previous_capture(self):
        stream = _make_stream()
        old_cap = _FakeCapture(value=1)
        stream._cap = old_cap
        original_init_camera = camera_stream_module.init_camera
        camera_stream_module.init_camera = lambda config: _FakeCapture(value=2)
        try:
            stream.switch_device(1)
        finally:
            stream.stop()
            camera_stream_module.init_camera = original_init_camera
        self.assertTrue(old_cap.released)

    def test_failed_switch_keeps_previous_camera_alive(self):
        # 새 장치가 안 열리면(케이블 빠짐 등) 기존 스트림을 절대 내리면 안 된다 —
        # 순서를 반대로 하면 카메라가 하나도 안 열린 채 파이프라인이 죽는다
        stream = _make_stream()
        old_cap = _FakeCapture(value=1)
        stream._cap = old_cap
        original_init_camera = camera_stream_module.init_camera

        def _fail(config):
            raise RuntimeError("장치 없음")

        camera_stream_module.init_camera = _fail
        try:
            with self.assertRaises(RuntimeError):
                stream.switch_device(1)
        finally:
            camera_stream_module.init_camera = original_init_camera
        self.assertFalse(old_cap.released)                 # 기존 캡처는 안 내려감
        self.assertEqual(stream._config["camera"]["device_id"], 0)   # device_id도 원복


if __name__ == "__main__":
    unittest.main()
