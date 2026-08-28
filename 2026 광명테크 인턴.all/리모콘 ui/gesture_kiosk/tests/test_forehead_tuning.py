"""forehead.py 실시간 조절 UI 연동 테스트 — 2026-08-28 신설.

eyebrow.py의 test_eyebrow_tuning.py와 완전히 같은 구조·이유다(그 파일
설명 참고) — forehead.py에도 동일한 _TuningReloader/_load_tuning_overrides가
있어 똑같이 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forehead import _load_tuning_overrides, _TuningReloader   # noqa: E402


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
                           arc_compensation=None):
        self.calls.append({"sensitivity_x": sensitivity_x,
                           "sensitivity_y": sensitivity_y,
                           "arc_compensation": arc_compensation})


class LoadTuningOverridesTest(unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(_load_tuning_overrides("존재하지_않는_파일.json"))

    def test_valid_file_returns_dict(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"arc_compensation": -0.9}, f)
            self.assertEqual(_load_tuning_overrides(path), {"arc_compensation": -0.9})

    def test_corrupted_file_returns_none_not_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{{{안깨진json아님")
            self.assertIsNone(_load_tuning_overrides(path))


class TuningReloaderTest(unittest.TestCase):
    def test_first_check_is_not_skipped(self):
        """★이 값 자체가 회귀 검사다 — _last_checked_sec을 0.0으로 초기화하면
        가짜 시계가 0.0에서 시작할 때 첫 확인이 통째로 건너뛰어졌었다
        (eyebrow.py에서 처음 발견된 버그, forehead.py도 같은 클래스 구조라
        똑같이 재발할 수 있다)."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sensitivity_x": 2.9}, f)
            clock = _FakeClock()
            reloader = _TuningReloader(path, poll_interval_sec=0.5, clock=clock)
            tracker = _FakeHeadTracker()
            reloader.maybe_reload(tracker)
            self.assertEqual(len(tracker.calls), 1, "첫 확인이 건너뛰어졌다")

    def test_reloads_when_file_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            clock = _FakeClock()
            reloader = _TuningReloader(path, poll_interval_sec=0.1, clock=clock)
            tracker = _FakeHeadTracker()

            reloader.maybe_reload(tracker)
            self.assertEqual(tracker.calls, [])

            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sensitivity_y": 4.4}, f)
            clock.advance(0.2)
            reloader.maybe_reload(tracker)
            self.assertEqual(len(tracker.calls), 1)
            self.assertEqual(tracker.calls[0]["sensitivity_y"], 4.4)

    def test_does_not_reload_within_poll_interval(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sensitivity_x": 1.0}, f)
            clock = _FakeClock()
            reloader = _TuningReloader(path, poll_interval_sec=0.5, clock=clock)
            tracker = _FakeHeadTracker()
            reloader.maybe_reload(tracker)
            self.assertEqual(len(tracker.calls), 1)

            clock.advance(0.1)
            reloader.maybe_reload(tracker)
            self.assertEqual(len(tracker.calls), 1, "확인 주기 전인데 또 읽었다")


if __name__ == "__main__":
    unittest.main()
