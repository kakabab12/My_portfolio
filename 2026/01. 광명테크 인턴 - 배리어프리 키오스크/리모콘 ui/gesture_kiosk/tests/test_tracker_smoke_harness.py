# -*- coding: utf-8 -*-
"""스모크 실행기(tests/tracker_smoke.py) 자체의 시험 (2026-09-25 신설).

실행기가 틀리면 트래커가 멀쩡해도 실패가 나거나, 반대로 고장을 못 본다.
2026-09-25 30분 장시간 실행에서 forehead만 위치 확인 셋이 값 없이(-) 실패했다.
원인은 실행기였다 — 메모리를 아끼려고 이동 기록을 맨 앞부터 지웠는데, 커서를
자주 옮기는 forehead만 30분에 2만 번을 넘어 **확인에 쓰는 첫 회차 기록**이
사라졌다. 여기서는 트래커 없이 이동 기록만 흘려 넣어 그 부분을 본다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.tracker_smoke as ts  # noqa: E402

FOREHEAD_RATE_HZ = 13.5    # 30분 장시간 실행에서 forehead가 커서를 옮긴 빈도(약 24,400번)


def _good_x(yaw_deg):
    """맞게 도는 트래커 — 오른쪽을 보면 커서가 반대쪽(거울)으로."""
    return 0.5 - 0.03 * yaw_deg


def _feed(minutes, rate_hz=FOREHEAD_RATE_HZ, broken_after_sec=None):
    """가상 시계로 장시간 실행의 이동 기록만 흘려 넣는다.

    broken_after_sec가 있으면 그 뒤로는 커서가 가운데에 멈춘 트래커를 흉내 낸다."""
    scenario = ts.Scenario(frame_w_px=405)
    scenario.loop, scenario.run_sec, scenario.t0 = True, minutes * 60.0, 0.0
    clock = [0.0]
    scenario.now = lambda: clock[0]
    ts.FakeMouse.scenario = scenario
    mouse = ts.FakeMouse()
    for i in range(int(rate_hz * minutes * 60.0)):
        t = i / rate_hz
        clock[0] = t
        idx = min(len(scenario.frames) - 1, int((t % scenario.duration) * ts.FPS))
        yaw = scenario.frames[idx][1]
        broken = broken_after_sec is not None and t >= broken_after_sec
        mouse.move(0.5 if broken else _good_x(yaw), 0.5)
    return scenario, mouse


class LongRunRecordsTest(unittest.TestCase):
    def tearDown(self):
        ts.FakeMouse.scenario = None

    def test_first_pass_survives_trimming(self):
        """9/25 실패 그대로 — 2만 번을 넘겨도 첫 회차 위치를 읽을 수 있어야 한다."""
        scenario, mouse = _feed(30)
        self.assertGreater(mouse.move_count, ts.MOVES_KEEP_MAX)      # 실제로 지우는 상황
        self.assertLessEqual(len(mouse.moves), ts.MOVES_KEEP_MAX)    # 메모리는 여전히 아낀다
        self.assertEqual(mouse.moves[0][0], 0.0)                     # 첫 기록이 남아 있다
        for ok, text in ts._position_checks(mouse.moves, scenario, 0.0):
            self.assertTrue(ok, text)

    def test_last_pass_is_checked_too(self):
        scenario, mouse = _feed(30)
        passes = ts._check_passes(scenario)
        self.assertEqual(len(passes), 2)
        _name, start = passes[1]
        self.assertGreater(start, 0.0)
        self.assertLessEqual(start + scenario.duration, scenario.run_sec)   # 온전히 돈 회차
        for ok, text in ts._position_checks(mouse.moves, scenario, start):
            self.assertTrue(ok, text)

    def test_tracker_that_breaks_later_is_caught(self):
        """첫 회차만 보면 못 잡는 고장 — 10분 뒤부터 커서가 멈춘 트래커."""
        scenario, mouse = _feed(30, broken_after_sec=600.0)
        first = ts._position_checks(mouse.moves, scenario, 0.0)
        last = ts._position_checks(mouse.moves, scenario, ts._check_passes(scenario)[1][1])
        self.assertTrue(first[0][0], first[0][1])
        self.assertFalse(last[0][0], last[0][1])

    def test_short_run_checks_one_pass(self):
        scenario = ts.Scenario(frame_w_px=405)
        self.assertEqual(ts._check_passes(scenario), [("", 0.0)])


class MemoryTrendTest(unittest.TestCase):
    @staticmethod
    def _samples(mb_at):
        return [(i * 0.5, mb_at(i), 6) for i in range(60)]     # 30분, 30초 간격

    def test_one_low_sample_does_not_look_like_growth(self):
        """9/25 forehead처럼 — 끝점 하나가 낮으면 두 끝점의 차는 크게 늘어난 것처럼 보인다."""
        def mb_at(i):
            if i == 2:                       # 1분째 표본 하나가 낮다
                return 95.0
            return 110.0 + (5.0 if i % 2 else -5.0)
        samples = self._samples(mb_at)
        _first, _last, low, high, per_hour, warm = ts._memory_trend(samples, 30.0)
        endpoints = (warm[-1][1] - warm[0][1]) / (warm[-1][0] - warm[0][0]) * 60.0
        self.assertGreater(endpoints, 30.0)          # 예전 방식이면 경고 수준
        self.assertLess(abs(per_hour), 10.0)         # 직선 추세는 거의 평평
        self.assertEqual((low, high), (95.0, 115.0))

    def test_real_growth_is_still_caught(self):
        samples = self._samples(lambda i: 100.0 + 1.0 * (i * 0.5))    # 분당 1MB = 시간당 60MB
        per_hour = ts._memory_trend(samples, 30.0)[4]
        self.assertGreater(per_hour, 50.0)


if __name__ == "__main__":
    unittest.main()
