"""성능 측정 — FPS."""
import time


class FpsMeter:
    """1초 단위로 평균 FPS를 갱신하는 측정기."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._frame_count = 0
        self._window_start_sec = clock()
        self.avg_fps = 0.0

    def update(self):
        self._frame_count += 1
        now_sec = self._clock()
        elapsed_sec = now_sec - self._window_start_sec
        if elapsed_sec >= 1.0:
            self.avg_fps = self._frame_count / elapsed_sec
            self._frame_count = 0
            self._window_start_sec = now_sec
