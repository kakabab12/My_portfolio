# -*- coding: utf-8 -*-
"""실시간 조절 UI(scripts/tuning_ui.py) 시험 (2026-09-24 신설).

두 가지를 지킨다.
  1. 슬라이더는 실제로 커서를 바꾸는 값(화각 두 개)만 다룬다. 예전에는 상대 회전
     매핑이 쓰지 않는 감도·곡률 슬라이더가 셋 있었다 — 움직여도 아무 일도 없었다.
  2. 위젯끼리 같은 칸에 겹치지 않는다. 예전에는 안내 문구가 row=3에 있어 "가로 각도"
     슬라이더와 같은 줄에 겹쳐 그려졌다.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.tuning_ui as tuning_ui  # noqa: E402

REAL_KNOBS = {"orientation_half_span_x_deg", "orientation_half_span_y_deg"}


class TuningDefaultsTest(unittest.TestCase):
    def test_only_knobs_that_move_the_cursor(self):
        for tracker, values in tuning_ui._DEFAULTS.items():
            self.assertEqual(set(values), REAL_KNOBS, tracker)

    def test_old_saved_keys_are_ignored(self):
        """연구실 파일처럼 옛 키(arc_compensation 등)가 남아 있어도 불러오지 않는다."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"arc_compensation": -1.008, "orientation_half_span_x_deg": 12.0}')
            values = tuning_ui.load_tuning(path, "eyebrow")
        self.assertEqual(set(values), REAL_KNOBS)
        self.assertEqual(values["orientation_half_span_x_deg"], 12.0)


class TuningLayoutTest(unittest.TestCase):
    def test_no_two_widgets_share_a_grid_cell(self):
        try:
            import tkinter as tk
            root = tk.Tk()
        except Exception as exc:   # noqa: 화면이 없는 환경이면 건너뛴다
            self.skipTest("Tk를 열 수 없음: %s" % exc)
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as d:
                tuning_ui.TuningWindow(root, os.path.join(d, "t.json"), "head")
                cells = {}
                for widget in root.grid_slaves():
                    info = widget.grid_info()
                    row, col = int(info["row"]), int(info["column"])
                    span = int(info.get("columnspan", 1))
                    for c in range(col, col + span):
                        self.assertNotIn((row, c), cells,
                                         "%s와 %s가 %d행 %d열에 겹친다"
                                         % (cells.get((row, c)), widget, row, c))
                        cells[(row, c)] = widget
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
