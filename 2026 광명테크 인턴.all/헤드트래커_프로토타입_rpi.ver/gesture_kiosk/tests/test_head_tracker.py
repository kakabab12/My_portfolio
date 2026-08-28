"""head_tracker 단위 테스트 — 카메라·mediapipe 없이 판정 로직만 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.inference.face_estimator import (
    LMK_LEFT_EYE_OUTER, LMK_NOSE_TIP, LMK_RIGHT_EYE_OUTER, FaceLandmarks,
)
from src.postprocess.head_tracker import HeadTracker
from tests.conftest import FakeClock

NUM_LANDMARKS = 478
FRAME_DT_SEC = 1.0 / 30.0


def make_face(nose_px=(100.0, 100.0), eye_dist_px=40.0, jaw_open=0.0,
              eye_close=0.0, eye_close_left=None, eye_close_right=None, mouth_pucker=0.0):
    """FaceLandmarks 테스트 대역 — mediapipe 임포트 없이 필요한 신호만 채운다.

    eye_close는 양쪽에 같은 값을 채우는 편의 인자 — 좌우를 다르게(윙크) 테스트할
    때만 eye_close_left/right를 개별로 지정한다.
    """
    landmarks_px = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
    landmarks_px[LMK_NOSE_TIP] = nose_px
    landmarks_px[LMK_LEFT_EYE_OUTER] = (nose_px[0] - eye_dist_px / 2.0, nose_px[1])
    landmarks_px[LMK_RIGHT_EYE_OUTER] = (nose_px[0] + eye_dist_px / 2.0, nose_px[1])
    blendshapes = {
        "jawOpen": jaw_open,
        "eyeBlinkLeft": eye_close if eye_close_left is None else eye_close_left,
        "eyeBlinkRight": eye_close if eye_close_right is None else eye_close_right,
        "mouthPucker": mouth_pucker,
    }
    return FaceLandmarks(bbox=(0, 0, 200, 200), conf=1.0, landmarks_px=landmarks_px, blendshapes=blendshapes)


def make_config(smoothing_alpha=1.0, distance_smoothing_alpha=1.0, dwell_enabled=True,
                 recenter_enabled=True):
    return {
        "head_tracker": {
            "calibration_window_sec": 0.1,
            "pointer": {
                "sensitivity_x": 2.0,
                "sensitivity_y": 2.0,
                "smoothing_alpha": smoothing_alpha,
                "distance_smoothing_alpha": distance_smoothing_alpha,
                "max_offset_ratio": 0.5,
            },
            "click": {"min_interval_sec": 0.2},
            "mouth_click": {"enabled": True, "open_margin": 0.5, "close_margin": 0.3},
            "eye_close_cancel": {"close_margin": 0.5, "hold_sec": 0.3},
            "recenter_gesture": {
                "enabled": recenter_enabled, "open_margin": 0.5, "close_margin": 0.3,
                "cooldown_sec": 0.2,
            },
            "dwell_click": {
                "enabled": dwell_enabled, "radius_ratio": 0.05, "dwell_sec": 0.3,
                "require_release_to_rearm": True,
            },
        }
    }


class HeadTrackerTestBase(unittest.TestCase):
    def setUp(self, **config_kwargs):
        self.clock = FakeClock()
        self.tracker = HeadTracker(make_config(**config_kwargs), clock=self.clock)

    def _settle_calibration(self, nose_px=(100.0, 100.0), jaw_open=0.0, eye_close=0.0,
                             generation=1, max_frames=20):
        """커서가 확정될 때까지만 같은 위치를 공급하고 즉시 돌려준다.

        코 캘리브레이션과 같은 구간에서 jaw_open/eye_close도 기준선으로 잡힌다(2026-07-20) —
        기본값 0.0으로 주면 옛 고정 임계값과 수치적으로 동일하게 동작하고, 값을 주면
        "평상시 표정이 다른 사용자" 시나리오를 재현할 수 있다(AdaptiveBaselineTest 참고).

        완료 직후 추가로 프레임을 더 태우면 dwell 타이머가 그만큼 미리 누적돼(커서가
        캘리브레이션 완료 순간부터 바로 dwell 추적을 시작하므로) 이후 타이밍 assert가
        어긋난다 — 그래서 캘리브레이션이 끝나는 정확한 프레임에서 멈춘다.
        """
        for _ in range(max_frames):
            face = make_face(nose_px=nose_px, jaw_open=jaw_open, eye_close=eye_close)
            result = self.tracker.update(face, generation)
            if result.cursor_x_ratio is not None:
                return result
            self.clock.tick(FRAME_DT_SEC)
        raise AssertionError("캘리브레이션이 max_frames 안에 끝나지 않았다")


class CursorMappingTest(HeadTrackerTestBase):
    def test_calibration_centers_cursor(self):
        result = self._settle_calibration()
        self.assertAlmostEqual(result.cursor_x_ratio, 0.5, places=2)
        self.assertAlmostEqual(result.cursor_y_ratio, 0.5, places=2)
        self.assertTrue(result.is_tracking)

    def test_cursor_moves_with_nose_offset(self):
        self._settle_calibration(nose_px=(100.0, 100.0))
        # eye_dist=40px, dx=20px -> offset = 20/40 * sensitivity(2.0) = 1.0 -> max_offset_ratio(0.5)로 클램프
        result = self.tracker.update(make_face(nose_px=(120.0, 100.0)), 1)
        self.assertAlmostEqual(result.cursor_x_ratio, 1.0, places=2)
        self.assertAlmostEqual(result.cursor_y_ratio, 0.5, places=2)

    def test_smoothing_lags_step_change(self):
        self.setUp(smoothing_alpha=0.5)
        self._settle_calibration(nose_px=(100.0, 100.0))
        result = self.tracker.update(make_face(nose_px=(120.0, 100.0)), 1)
        # alpha=0.5 — 목표(1.0)까지 한 프레임 만에 못 간다 (0.5와 1.0 사이)
        self.assertGreater(result.cursor_x_ratio, 0.5)
        self.assertLess(result.cursor_x_ratio, 1.0)

    def test_calibration_median_resists_single_outlier_frame(self):
        # calibration_window_sec(0.1s) 동안 프레임 여러 장이 들어간다 — 그중 1장만
        # 크게 아래로 쏠린 값(과도기 프레임 모사)이어도 중앙값 기준점은 흔들리지 않아야 한다
        # (2026-07-18 실기: 평균이었을 때 이런 프레임 하나로 커서가 상단에 눌러붙는 현상 있었음)
        nose_ys = [100.0, 100.0, 100.0, 100.0, 300.0]
        result = None
        for nose_y in nose_ys:
            result = self.tracker.update(make_face(nose_px=(100.0, nose_y)), 1)
            if result.cursor_x_ratio is not None:
                break
            self.clock.tick(FRAME_DT_SEC)
        self.assertIsNotNone(result.cursor_x_ratio)

        # 캘리브레이션 후 원래 위치(100,100)로 돌아오면 중앙값 기준점(100)과 거의 일치 —
        # 평균(150)이었다면 여기서 위로 크게 벗어난 것처럼 오판정됐을 것
        result = self.tracker.update(make_face(nose_px=(100.0, 100.0)), 1)
        self.assertAlmostEqual(result.cursor_y_ratio, 0.5, delta=0.05)

    def test_new_lock_generation_recalibrates(self):
        self._settle_calibration(nose_px=(100.0, 100.0), generation=1)
        # 새 세대 — 다른 위치에서 다시 캘리브레이션 구간에 들어가 커서가 확정되지 않는다
        result = self.tracker.update(make_face(nose_px=(300.0, 300.0)), 2)
        self.assertIsNone(result.cursor_x_ratio)
        self.assertFalse(result.is_tracking)

    def test_tracking_loss_resets_cursor_and_calibration(self):
        self._settle_calibration(nose_px=(100.0, 100.0), generation=1)
        result = self.tracker.update(None, 1)
        self.assertIsNone(result.cursor_x_ratio)
        self.assertFalse(result.is_tracking)
        self.assertEqual(result.events, [])
        # 재잠금(같은 generation 값이라도) — 추적 손실을 거쳤으니 다시 캘리브레이션
        result = self.tracker.update(make_face(nose_px=(100.0, 100.0)), 1)
        self.assertIsNone(result.cursor_x_ratio)


class MouthClickTest(HeadTrackerTestBase):
    def setUp(self):
        # dwell 비활성 — 코 위치가 고정된 이 테스트들에서 dwell도 같이 진행돼
        # 공용 클릭 쿨다운을 오염시키는 것을 막는다 (mouth 신호만 격리해서 본다)
        super().setUp(dwell_enabled=False)

    def test_mouth_open_fires_once_per_episode(self):
        self._settle_calibration()
        result = self.tracker.update(make_face(jaw_open=0.8), 1)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].class_name, "select")
        self.assertEqual(result.events[0].data["trigger"], "mouth")

        # 입을 계속 벌리고 있어도 재발화 없음
        result = self.tracker.update(make_face(jaw_open=0.8), 1)
        self.assertEqual(len(result.events), 0)

    def test_mouth_click_respects_shared_cooldown(self):
        self._settle_calibration()
        self.tracker.update(make_face(jaw_open=0.8), 1)          # 1차 확정
        self.tracker.update(make_face(jaw_open=0.1), 1)          # 닫음 — 재장전
        self.clock.tick(0.05)
        result = self.tracker.update(make_face(jaw_open=0.8), 1)  # min_interval_sec(0.2) 이내
        self.assertEqual(len(result.events), 0)

    def test_mouth_click_fires_again_after_cooldown(self):
        self._settle_calibration()
        self.tracker.update(make_face(jaw_open=0.8), 1)
        self.tracker.update(make_face(jaw_open=0.1), 1)
        self.clock.tick(0.3)
        self.tracker.update(make_face(jaw_open=0.1), 1)
        result = self.tracker.update(make_face(jaw_open=0.8), 1)
        self.assertEqual(len(result.events), 1)


class EyeCloseCancelTest(HeadTrackerTestBase):
    def setUp(self):
        super().setUp(dwell_enabled=False)   # 눈 감김 신호만 격리 (MouthClickTest와 같은 이유)

    def test_no_event_before_hold_duration(self):
        self._settle_calibration()
        self.tracker.update(make_face(eye_close=0.8), 1)   # 감기 시작
        self.clock.tick(0.29)                                # hold_sec(0.3) 미만
        result = self.tracker.update(make_face(eye_close=0.8), 1)
        self.assertEqual(len(result.events), 0)

    def test_fires_once_after_hold_duration(self):
        self._settle_calibration()
        self.tracker.update(make_face(eye_close=0.8), 1)
        self.clock.tick(0.31)
        result = self.tracker.update(make_face(eye_close=0.8), 1)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].class_name, "go_back")
        self.assertEqual(result.events[0].data["trigger"], "eye_close")

    def test_does_not_refire_while_sustained(self):
        self._settle_calibration()
        self.tracker.update(make_face(eye_close=0.8), 1)
        self.clock.tick(0.31)
        self.tracker.update(make_face(eye_close=0.8), 1)          # 1차 확정
        result = self.tracker.update(make_face(eye_close=0.8), 1)  # 계속 감은 채 — 재발화 없음
        self.assertEqual(len(result.events), 0)

    def test_refires_after_opening_and_reholding(self):
        self._settle_calibration()
        self.tracker.update(make_face(eye_close=0.8), 1)
        self.clock.tick(0.31)
        self.tracker.update(make_face(eye_close=0.8), 1)   # 1차 확정

        self.tracker.update(make_face(eye_close=0.0), 1)    # 눈 뜸 — 재장전
        self.tracker.update(make_face(eye_close=0.8), 1)    # 다시 감기 시작
        self.clock.tick(0.31)
        result = self.tracker.update(make_face(eye_close=0.8), 1)
        self.assertEqual(len(result.events), 1)

    def test_one_eye_wink_does_not_fire(self):
        # 한쪽만 감으면(윙크) 안 걸린다 — 양쪽 다 감아야 인정(오발화 방지 + 윙크
        # 못 하는 사용자도 조작 가능하게)
        self._settle_calibration()
        for _ in range(20):
            result = self.tracker.update(
                make_face(eye_close_left=0.9, eye_close_right=0.1), 1
            )
            self.clock.tick(FRAME_DT_SEC)
        self.assertEqual(len(result.events), 0)


class RecenterGestureTest(HeadTrackerTestBase):
    def setUp(self):
        super().setUp(dwell_enabled=False)   # 입 오므림 신호만 격리 (MouthClickTest와 같은 이유)

    def test_mouth_pucker_fires_recenter_and_hides_cursor(self):
        self._settle_calibration()
        result = self.tracker.update(make_face(mouth_pucker=0.9), 1)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].class_name, "recenter")
        self.assertEqual(result.events[0].data["trigger"], "mouth_pucker")
        # 리셋은 다음 프레임부터 반영된다(이번 프레임 커서는 리셋 전에 이미 계산됨) —
        # 캘리브레이션이 다시 시작돼 다음 프레임은 커서가 미확정 상태다
        result = self.tracker.update(make_face(mouth_pucker=0.1), 1)
        self.assertIsNone(result.cursor_x_ratio)

    def test_recenter_does_not_reset_mouth_baseline(self):
        self._settle_calibration()
        self.tracker.update(make_face(mouth_pucker=0.9), 1)   # 재정렬 발화
        # jaw 기준선은 그대로라 재캘리브레이션 대기 없이 바로 입 벌리기가 먹힌다
        result = self.tracker.update(make_face(jaw_open=0.8), 1)
        select_events = [e for e in result.events if e.class_name == "select"]
        self.assertEqual(len(select_events), 1)

    def test_recenter_respects_cooldown(self):
        self._settle_calibration()
        self.tracker.update(make_face(mouth_pucker=0.9), 1)
        self.tracker.update(make_face(mouth_pucker=0.1), 1)   # 재장전(히스테리시스 close_margin 아래)
        self.clock.tick(0.05)                                 # cooldown_sec(0.2) 이내
        result = self.tracker.update(make_face(mouth_pucker=0.9), 1)
        recenter_events = [e for e in result.events if e.class_name == "recenter"]
        self.assertEqual(len(recenter_events), 0)

    def test_disabled_recenter_never_fires(self):
        super().setUp(dwell_enabled=False, recenter_enabled=False)
        self._settle_calibration()
        result = self.tracker.update(make_face(mouth_pucker=0.9), 1)
        self.assertEqual(result.events, [])


class DwellClickTest(HeadTrackerTestBase):
    def test_no_event_before_dwell_duration(self):
        self._settle_calibration()
        self.tracker.update(make_face(), 1)   # 응시 시작(앵커 설정)
        self.clock.tick(0.29)
        result = self.tracker.update(make_face(), 1)
        self.assertEqual(len(result.events), 0)

    def test_fires_once_after_dwell_duration(self):
        self._settle_calibration()
        self.tracker.update(make_face(), 1)
        self.clock.tick(0.31)
        result = self.tracker.update(make_face(), 1)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].data["trigger"], "dwell")

    def test_does_not_refire_without_leaving_radius(self):
        self._settle_calibration()
        self.tracker.update(make_face(), 1)
        self.clock.tick(0.31)
        self.tracker.update(make_face(), 1)          # 1차 확정
        self.clock.tick(0.31)
        result = self.tracker.update(make_face(), 1)  # 같은 자리 유지 — 재발화 없음
        self.assertEqual(len(result.events), 0)

    def test_refires_after_leaving_and_returning(self):
        self._settle_calibration()
        self.tracker.update(make_face(), 1)
        self.clock.tick(0.31)
        self.tracker.update(make_face(), 1)   # 1차 확정

        moved = make_face(nose_px=(400.0, 100.0))   # 반경 이탈 — 새 앵커
        result = self.tracker.update(moved, 1)
        self.assertEqual(len(result.events), 0)

        self.clock.tick(0.31)
        result = self.tracker.update(moved, 1)   # 새 위치에서 dwell_sec 재충족
        self.assertEqual(len(result.events), 1)


class CrossDetectorCooldownTest(HeadTrackerTestBase):
    def test_mouth_and_dwell_in_same_tick_fire_once(self):
        self._settle_calibration()
        self.tracker.update(make_face(), 1)
        self.clock.tick(0.31)   # dwell_sec 충족 시점에 입도 벌림
        result = self.tracker.update(make_face(jaw_open=0.8), 1)
        select_events = [e for e in result.events if e.class_name == "select"]
        self.assertEqual(len(select_events), 1)


class AdaptiveBaselineTest(HeadTrackerTestBase):
    """2026-07-20 정확도 개선 — 고정 임계값 대신 잠금 직후 캡처한 평상시 기준선 +
    여유값으로 판정한다. 사람마다 평상시 블렌드셰이프 값이 다른 상황을 재현해
    임계값이 실제로 기준선을 따라 이동하는지 검증한다."""

    def setUp(self):
        super().setUp(dwell_enabled=False)

    def test_mouth_threshold_shifts_with_elevated_baseline(self):
        # 평상시 입이 약간 벌어져 있는 사용자(기준선 0.3) 모사
        self._settle_calibration(jaw_open=0.3)
        # 기준선(0.3) 그대로면 open_margin(0.5)을 못 넘는다 — 옛 고정 임계(0.5)였다면
        # 여기서 오탐(0.3만으로는 안 넘지만, 다른 사용자의 절대값 0.5 자체가 이 사용자
        # 기준선보다 낮을 수도 있었다는 문제의식) — 기준선 상대 판정이므로 안전
        result = self.tracker.update(make_face(jaw_open=0.3), 1)
        self.assertEqual(len(result.events), 0)
        # 기준선 + margin을 확실히 넘겨야 확정
        result = self.tracker.update(make_face(jaw_open=0.9), 1)
        self.assertEqual(len(result.events), 1)

    def test_eye_close_baseline_prevents_resting_false_trigger(self):
        # 평상시 eyeBlink가 높게 잡히는 사용자(기준선 0.5) 모사 — 2026-07-18 실기에서
        # 실제로 관찰된 패턴(사람마다 0.1~0.6까지 편차) — 옛 고정 임계(0.6)였다면
        # 이런 사용자는 평상시에도 문턱에 걸쳐 있어 오탐 위험이 컸다
        self._settle_calibration(eye_close=0.5)
        result = None
        for _ in range(20):   # 기준선 그대로 오래 유지 — 확정되면 안 된다
            result = self.tracker.update(make_face(eye_close=0.5), 1)
            self.clock.tick(FRAME_DT_SEC)
        self.assertEqual(len(result.events), 0)
        # 기준선보다 확실히 더(거의 최대) 감아야 확정
        self.tracker.update(make_face(eye_close=1.0), 1)
        self.clock.tick(0.31)
        result = self.tracker.update(make_face(eye_close=1.0), 1)
        self.assertEqual(len(result.events), 1)

    def test_no_detection_fires_before_baseline_ready(self):
        # 캘리브레이션 구간 중에는(코 캘리브레이션도 아직) 입/눈 판정 자체가 보류된다 —
        # 기준선 없이 판정하면 안 된다는 계약을 명시적으로 검증
        result = self.tracker.update(make_face(jaw_open=0.9, eye_close=0.9), 1)
        self.assertIsNone(result.cursor_x_ratio)
        self.assertEqual(result.events, [])


if __name__ == "__main__":
    unittest.main()
