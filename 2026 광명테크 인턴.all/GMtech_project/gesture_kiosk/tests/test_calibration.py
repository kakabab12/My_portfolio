"""calibration 단위 테스트 — 보정 계산·config 반영 (카메라 없이 순수 함수만).

임계값을 자동으로 바꾸는 코드라 잘못 계산하면 현장이 통째로 망가진다 —
안전 범위 클램프·제약(손목 임계 < 위 쓸기 임계)·주석 보존을 특히 본다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.calibration import (
    LIMITS, apply_to_config_text, format_report, recommend_thresholds,
)

CURRENT = {
    "min_dist_x_shoulder": 0.55,
    "min_dist_y_shoulder": 0.16,
    "flick_min_dist_shoulder": 0.25,
    "extend_ratio": 1.05,
    "dip_drop_ratio": 0.15,
    "move_dip_shoulder": 0.10,
}


class RecommendTest(unittest.TestCase):
    def test_swipe_thresholds_sit_below_smallest_gesture(self):
        # 핵심 규칙: "가장 작게 한 동작"보다 살짝 아래 — 그래야 작은 동작도 잡힌다
        picks = recommend_thresholds(
            {"swipe_x_small_min": 0.40, "swipe_y_small_min": 0.30}, CURRENT)
        self.assertAlmostEqual(picks["min_dist_x_shoulder"][0], 0.36)
        self.assertAlmostEqual(picks["min_dist_y_shoulder"][0], 0.27)

    def test_flick_stays_below_main_threshold(self):
        # 플릭이 본 임계보다 크면 플릭 경로가 무의미해진다 (2026-08-03 실기 교훈)
        picks = recommend_thresholds(
            {"swipe_x_small_min": 0.40, "swipe_y_small_min": 0.30}, CURRENT)
        self.assertLess(picks["flick_min_dist_shoulder"][0],
                        picks["min_dist_y_shoulder"][0])

    def test_extend_ratio_lands_in_the_gap(self):
        # 굽힘 0.74 / 폄 1.37 사이의 빈 구간 한가운데 — 어느 쪽으로도 여유가 크다
        picks = recommend_thresholds(
            {"shape_curl_p90": 0.74, "shape_extend_p10": 1.37}, CURRENT)
        value = picks["extend_ratio"][0]
        self.assertGreater(value, 0.74)
        self.assertLess(value, 1.37)

    def test_extend_ratio_skipped_when_no_gap(self):
        # 분포가 겹치면(측정 불량) 손대지 않는다 — 억지로 정하면 판별이 무너진다
        picks = recommend_thresholds(
            {"shape_curl_p90": 1.20, "shape_extend_p10": 1.10}, CURRENT)
        self.assertNotIn("extend_ratio", picks)

    def test_wrist_dip_capped_below_up_swipe(self):
        # ★제약: 손목 까딱 임계가 위 쓸기 임계보다 크면 까딱이 select로 먼저
        # 확정된다 — 큰 값이 측정돼도 위 임계의 80%로 잘려야 한다
        picks = recommend_thresholds(
            {"swipe_y_small_min": 0.30, "tap_wrist_drop": 0.50}, CURRENT)
        up = picks["min_dist_y_shoulder"][0]
        self.assertLessEqual(picks["move_dip_shoulder"][0], up * 0.8 + 1e-9)

    def test_values_are_clamped_to_safe_range(self):
        # 측정이 이상해도(0에 가깝거나 과대) config가 망가지지 않는다
        picks = recommend_thresholds(
            {"swipe_x_small_min": 0.001, "swipe_y_small_min": 9.0}, CURRENT)
        self.assertGreaterEqual(picks["min_dist_x_shoulder"][0],
                                LIMITS["min_dist_x_shoulder"][0])
        self.assertLessEqual(picks["min_dist_y_shoulder"][0],
                             LIMITS["min_dist_y_shoulder"][1])

    def test_no_change_when_already_matching(self):
        # 같은 값이면 항목을 내지 않는다 — 불필요한 config 수정·주석 오염 방지
        picks = recommend_thresholds({"swipe_x_small_min": 0.55 / 0.9}, CURRENT)
        self.assertNotIn("min_dist_x_shoulder", picks)

    def test_missing_measurements_touch_nothing(self):
        self.assertEqual(recommend_thresholds({}, CURRENT), {})


CONFIG_SAMPLE = """gestures:
  swipe:
    min_dist_x_shoulder: 0.55 # 좌/우 쓸기 최소 이동 — 2026-07-16 실측 근거
                              #   여러 줄 주석도 그대로 남아야 한다
    min_dist_y_shoulder: 0.16 # 위 쓸기
"""


class ApplyToConfigTest(unittest.TestCase):
    def test_value_replaced_and_comment_preserved(self):
        # config 주석에는 실측 근거·날짜가 쌓여 있다(기획서 4.7) — 반드시 보존
        text, missed = apply_to_config_text(
            CONFIG_SAMPLE, {"min_dist_x_shoulder": (0.36, "작게 0.40 × 0.9")}, "2026-08-03")
        self.assertIn("min_dist_x_shoulder: 0.36", text)
        self.assertIn("2026-07-16 실측 근거", text)
        self.assertIn("여러 줄 주석도 그대로 남아야 한다", text)
        self.assertIn("[자동보정 2026-08-03]", text)
        self.assertEqual(missed, [])

    def test_other_keys_untouched(self):
        text, _ = apply_to_config_text(
            CONFIG_SAMPLE, {"min_dist_x_shoulder": (0.36, "r")}, "2026-08-03")
        self.assertIn("min_dist_y_shoulder: 0.16", text)

    def test_rerun_does_not_stack_markers(self):
        # 두 번 돌려도 표시가 겹겹이 붙지 않는다
        text, _ = apply_to_config_text(
            CONFIG_SAMPLE, {"min_dist_x_shoulder": (0.36, "첫 번째")}, "2026-08-03")
        text, _ = apply_to_config_text(
            text, {"min_dist_x_shoulder": (0.30, "두 번째")}, "2026-08-04")
        self.assertEqual(text.count("[자동보정"), 1)
        self.assertIn("min_dist_x_shoulder: 0.3", text)
        self.assertIn("두 번째", text)

    def test_unknown_key_reported_not_written(self):
        text, missed = apply_to_config_text(
            CONFIG_SAMPLE, {"없는키_shoulder": (0.1, "r")}, "2026-08-03")
        self.assertEqual(missed, ["없는키_shoulder"])
        self.assertEqual(text, CONFIG_SAMPLE)


class ReportTest(unittest.TestCase):
    def test_report_lists_changes(self):
        report = format_report({"min_dist_x_shoulder": (0.36, "사유")}, [],
                               {"swipe_x_small_min": 0.40})
        self.assertIn("min_dist_x_shoulder: 0.36", report)
        self.assertIn("사유", report)

    def test_report_when_nothing_changed(self):
        self.assertIn("변경 없음", format_report({}, [], {}))


if __name__ == "__main__":
    unittest.main()
