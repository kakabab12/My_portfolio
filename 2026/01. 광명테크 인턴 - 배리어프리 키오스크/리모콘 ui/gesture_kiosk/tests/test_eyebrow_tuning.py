"""eyebrow.py 실시간 조절 UI 연동 테스트 — 2026-08-28 신설.

scripts/tuning_ui.py가 쓰는 JSON 파일을 트래커가 주기적으로 확인해서
head_tracker에 반영하는 _TuningReloader와, 그 파일을 읽는
_load_tuning_overrides를 검증한다. 진짜 파일 I/O(임시 디렉터리)만 쓰고
카메라·mediapipe는 전혀 건드리지 않는다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eyebrow import _load_tuning_overrides, _TuningReloader   # noqa: E402


class _FakeClock:
    def __init__(self):
        self.now_sec = 0.0

    def __call__(self):
        return self.now_sec

    def advance(self, delta_sec):
        self.now_sec += delta_sec


class _FakeHeadTracker:
    def __init__(self):
        self.calls = []

    def set_pointer_tuning(self, sensitivity_x=None, sensitivity_y=None,
                           arc_compensation=None,
                           half_span_x_deg=None, half_span_y_deg=None):
        # ★2026-08-31 — 상대 회전 매핑에서는 앞의 세 값이 안 쓰이고 아래 두
        # 각도가 감도 손잡이다. 조절 UI가 실제로 먹는지 이 가짜가 지켜본다
        self.calls.append({"sensitivity_x": sensitivity_x,
                           "sensitivity_y": sensitivity_y,
                           "arc_compensation": arc_compensation,
                           "half_span_x_deg": half_span_x_deg,
                           "half_span_y_deg": half_span_y_deg})


class LoadTuningOverridesTest(unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(_load_tuning_overrides("존재하지_않는_파일.json"))

    def test_valid_file_returns_dict(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sensitivity_x": 4.2}, f)
            self.assertEqual(_load_tuning_overrides(path), {"sensitivity_x": 4.2})

    def test_corrupted_file_returns_none_not_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("이건 json이 아니다{{{")
            self.assertIsNone(_load_tuning_overrides(path))


class TuningReloaderTest(unittest.TestCase):
    def test_does_nothing_when_file_never_existed(self):
        """UI를 한 번도 안 켰으면(파일 없음) 아무 일도 안 일어나야 한다."""
        clock = _FakeClock()
        reloader = _TuningReloader("존재하지_않는_파일.json", poll_interval_sec=0.1,
                                   clock=clock)
        tracker = _FakeHeadTracker()
        for _ in range(5):
            reloader.maybe_reload(tracker)
            clock.advance(0.2)
        self.assertEqual(tracker.calls, [])

    def test_reloads_when_file_changes(self):
        """파일이 새로 생기거나 바뀌면 head_tracker에 반영해야 한다."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            clock = _FakeClock()
            reloader = _TuningReloader(path, poll_interval_sec=0.1, clock=clock)
            tracker = _FakeHeadTracker()

            reloader.maybe_reload(tracker)   # 파일 없음 — 아무 일 없음
            self.assertEqual(tracker.calls, [])

            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sensitivity_x": 3.3, "arc_compensation": -0.5}, f)
            clock.advance(0.2)   # 확인 주기를 지나야 본다
            reloader.maybe_reload(tracker)
            self.assertEqual(len(tracker.calls), 1)
            self.assertEqual(tracker.calls[0]["sensitivity_x"], 3.3)
            self.assertEqual(tracker.calls[0]["arc_compensation"], -0.5)
            self.assertIsNone(tracker.calls[0]["sensitivity_y"])   # 파일에 없던 키

    def test_does_not_reload_within_poll_interval(self):
        """확인 주기가 지나기 전에는(매 프레임 파일을 열어보지 않는다) 파일이
        바뀌어도 아직 반영하면 안 된다."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sensitivity_x": 1.0}, f)
            clock = _FakeClock()
            reloader = _TuningReloader(path, poll_interval_sec=0.5, clock=clock)
            tracker = _FakeHeadTracker()
            reloader.maybe_reload(tracker)
            self.assertEqual(len(tracker.calls), 1)   # 첫 확인은 항상 반영

            clock.advance(0.1)   # 주기(0.5)보다 짧게
            reloader.maybe_reload(tracker)
            self.assertEqual(len(tracker.calls), 1, "확인 주기 전인데 또 읽었다")

    def test_does_not_reload_when_file_unchanged(self):
        """확인 주기는 지났지만 파일 내용(수정 시각)이 그대로면 다시 반영할
        필요가 없다 — 매번 head_tracker를 건드리면 불필요한 부담이다."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sensitivity_x": 1.0}, f)
            clock = _FakeClock()
            reloader = _TuningReloader(path, poll_interval_sec=0.1, clock=clock)
            tracker = _FakeHeadTracker()
            reloader.maybe_reload(tracker)
            self.assertEqual(len(tracker.calls), 1)

            clock.advance(0.2)
            reloader.maybe_reload(tracker)   # 파일은 그대로
            self.assertEqual(len(tracker.calls), 1, "파일이 안 바뀌었는데 또 반영했다")


if __name__ == "__main__":
    unittest.main()
