"""tests 공용 픽스처 — 카메라·모델 없이 순수 로직만 검증하는 테스트에서 공유한다."""


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def path(start, end, step_count, y_ratio=None, x_ratio=None):
    """직선 궤적 점 목록 — y_ratio 지정 시 수평 이동, x_ratio 지정 시 수직 이동."""
    points = []
    for step_idx in range(step_count + 1):
        value = start + (end - start) * step_idx / step_count
        points.append((value, y_ratio) if y_ratio is not None else (x_ratio, value))
    return points
