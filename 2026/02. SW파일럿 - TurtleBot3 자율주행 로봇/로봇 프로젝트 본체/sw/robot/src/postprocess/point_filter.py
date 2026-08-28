"""추적점 스무딩 — One Euro 필터.

포즈 키포인트는 정지 상태에서도 프레임마다 수 픽셀씩 떨린다(관측 잡음).
One Euro 필터(Casiez·Roussel·Vogel, CHI 2012 공개 알고리즘 — 수식만 참조해
직접 구현)는 속도 적응형 저역통과 필터다: 느리게 움직일 때는 컷오프를 낮춰
떨림을 강하게 걸러내고, 빠르게 움직일 때는 컷오프를 올려 지연을 줄인다.
"""
import math


class OneEuroFilter:
    """스칼라 1개(x 또는 y)의 One Euro 필터. filter(value, ts_sec) -> 평활값."""

    def __init__(self, min_cutoff_hz, beta, d_cutoff_hz):
        self._min_cutoff_hz = min_cutoff_hz
        self._beta = beta
        self._d_cutoff_hz = d_cutoff_hz
        self.reset()

    def reset(self):
        self._prev_value = None
        self._prev_derivative = 0.0
        self._prev_ts_sec = None

    @staticmethod
    def _alpha(cutoff_hz, dt_sec):
        tau_sec = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau_sec / dt_sec)

    def filter(self, value, ts_sec):
        if self._prev_value is None:
            self._prev_value = value
            self._prev_ts_sec = ts_sec
            return value
        dt_sec = ts_sec - self._prev_ts_sec
        if dt_sec <= 0.0:
            return self._prev_value

        derivative = (value - self._prev_value) / dt_sec
        alpha_d = self._alpha(self._d_cutoff_hz, dt_sec)
        smoothed_derivative = (
            self._prev_derivative + alpha_d * (derivative - self._prev_derivative)
        )

        cutoff_hz = self._min_cutoff_hz + self._beta * abs(smoothed_derivative)
        alpha = self._alpha(cutoff_hz, dt_sec)
        smoothed_value = self._prev_value + alpha * (value - self._prev_value)

        self._prev_value = smoothed_value
        self._prev_derivative = smoothed_derivative
        self._prev_ts_sec = ts_sec
        return smoothed_value


class PointFilter:
    """(x, y) 좌표쌍용 One Euro 필터 묶음."""

    def __init__(self, min_cutoff_hz, beta, d_cutoff_hz):
        self._filter_x = OneEuroFilter(min_cutoff_hz, beta, d_cutoff_hz)
        self._filter_y = OneEuroFilter(min_cutoff_hz, beta, d_cutoff_hz)

    def reset(self):
        self._filter_x.reset()
        self._filter_y.reset()

    def filter(self, point, ts_sec):
        return (
            self._filter_x.filter(point[0], ts_sec),
            self._filter_y.filter(point[1], ts_sec),
        )
