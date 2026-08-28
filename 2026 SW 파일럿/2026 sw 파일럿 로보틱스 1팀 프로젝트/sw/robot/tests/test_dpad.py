"""dpad 단위 테스트 — 존 판정, 존->속도 매핑, 최초 캘리브레이션, 커서 증폭,
잠금 토글, 재중심, 인게이지 게이트.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.dpad import (
    ZONE_CENTER, ZONE_DOWN, ZONE_LEFT, ZONE_RIGHT, ZONE_UP,
    DpadController, classify_zone, zone_velocity,
)

FRAME_W, FRAME_H = 640, 480
NEUTRAL = (320.0, 240.0)

BASE_CFG = {
    "engage_shape": "open",
    # 증폭 배율은 기본 1.0으로 둬서(main_dpad.py 기본값은 3.0) 이 파일 대부분의
    # 테스트가 raw px 오프셋으로 단순하게 검증되게 한다 — 증폭 자체의 동작은
    # CursorSensitivityAndCalibrationTest에서 따로 검증한다.
    "cursor_sensitivity": 1.0,
    "center_radius_ratio": 0.10,
    "axis_dominance": 1.3,
    "linear_mps": 0.12,
    "angular_radps": 0.8,
    "invert_x": False,
    "invert_y": False,
    "recenter": {"enabled": False},
    # 이 파일의 대부분 테스트는 잠금 토글 자체가 아니라 존 판정/속도를 다루므로
    # 미리 잠금 해제된 상태로 시작한다 — 잠금 토글 자체는 LockToggleTest에서 검증.
    "lock_toggle": {"enabled": True, "start_armed": True, "require_shape": "open",
                    "hold_sec": 1.2},
}
PF_CFG = {"min_cutoff_hz": 1.0, "beta": 0.3, "d_cutoff_hz": 1.0}


def _prime(controller, neutral_point=NEUTRAL, t=0.0):
    """최초 등장 캘리브레이션으로 neutral_point를 기준(0점)으로 확정시킨다.

    DpadController는 main_dpad.py와 동일하게 손이 처음 나타난 자리를 곧바로
    기준으로 삼는다 — 이후 그 기준에서 얼마나 벗어났는지로 존을 검증하려면
    먼저 원하는 자리에서 "등장"시켜 둬야 한다.
    """
    controller.update(neutral_point, "fist", FRAME_W, FRAME_H, t)
    return t + 0.05


def _settle(controller, point_px, shape, start_t, iterations=10, step=0.1):
    """같은 위치를 여러 번 넣어 One Euro 필터가 그 값에 충분히 수렴하게 한다."""
    t = start_t
    result = None
    for _ in range(iterations):
        result = controller.update(point_px, shape, FRAME_W, FRAME_H, t)
        t += step
    return result, t


class ClassifyZoneTest(unittest.TestCase):
    def test_within_radius_is_center(self):
        self.assertEqual(classify_zone(10.0, 10.0, 48.0, 1.3), ZONE_CENTER)
        self.assertEqual(classify_zone(0.0, 0.0, 48.0, 1.3), ZONE_CENTER)

    def test_up_and_down(self):
        self.assertEqual(classify_zone(0.0, -100.0, 48.0, 1.3), ZONE_UP)
        self.assertEqual(classify_zone(0.0, 100.0, 48.0, 1.3), ZONE_DOWN)

    def test_left_and_right(self):
        self.assertEqual(classify_zone(-100.0, 0.0, 48.0, 1.3), ZONE_LEFT)
        self.assertEqual(classify_zone(100.0, 0.0, 48.0, 1.3), ZONE_RIGHT)

    def test_diagonal_is_ambiguous(self):
        self.assertIsNone(classify_zone(80.0, 80.0, 48.0, 1.3))
        self.assertIsNone(classify_zone(-80.0, 80.0, 48.0, 1.3))


class ZoneVelocityTest(unittest.TestCase):
    def test_up_is_forward_only(self):
        linear_x, angular_z = zone_velocity(ZONE_UP, BASE_CFG)
        self.assertEqual(linear_x, BASE_CFG["linear_mps"])
        self.assertEqual(angular_z, 0.0)

    def test_down_is_backward_only(self):
        linear_x, angular_z = zone_velocity(ZONE_DOWN, BASE_CFG)
        self.assertEqual(linear_x, -BASE_CFG["linear_mps"])
        self.assertEqual(angular_z, 0.0)

    def test_left_turns_left_positive_angular(self):
        # ROS REP103: angular_z 양수 = 좌회전
        linear_x, angular_z = zone_velocity(ZONE_LEFT, BASE_CFG)
        self.assertEqual(linear_x, 0.0)
        self.assertEqual(angular_z, BASE_CFG["angular_radps"])

    def test_right_turns_right_negative_angular(self):
        linear_x, angular_z = zone_velocity(ZONE_RIGHT, BASE_CFG)
        self.assertEqual(linear_x, 0.0)
        self.assertEqual(angular_z, -BASE_CFG["angular_radps"])

    def test_center_and_none_are_stopped(self):
        self.assertEqual(zone_velocity(ZONE_CENTER, BASE_CFG), (0.0, 0.0))
        self.assertEqual(zone_velocity(None, BASE_CFG), (0.0, 0.0))


class DpadControllerTest(unittest.TestCase):
    def test_no_hand_is_stopped(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        linear_x, angular_z, engaged = controller.update(None, None, FRAME_W, FRAME_H, 0.0)
        self.assertEqual((linear_x, angular_z, engaged), (0.0, 0.0, False))

    def test_first_appearance_calibrates_neutral_to_itself(self):
        # main_dpad.py와 동일 — 손이 처음 나타난 자리가 곧 기준(0점)이라
        # 그 프레임은 오프셋 0 -> center 존 -> 정지
        controller = DpadController(BASE_CFG, PF_CFG)
        linear_x, angular_z, engaged = controller.update(
            (500.0, 100.0), "open", FRAME_W, FRAME_H, 0.0)
        self.assertEqual((linear_x, angular_z, engaged), (0.0, 0.0, False))
        self.assertEqual(controller.neutral_px, (500.0, 100.0))

    def test_wrong_shape_is_stopped_even_in_zone(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        t = _prime(controller)
        linear_x, angular_z, engaged = controller.update(
            (320.0, 140.0), "fist", FRAME_W, FRAME_H, t)   # 위 존이지만 주먹
        self.assertEqual((linear_x, angular_z, engaged), (0.0, 0.0, False))

    def test_open_hand_in_up_zone_drives_forward(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        t = _prime(controller)
        linear_x, angular_z, engaged = controller.update(
            (320.0, 140.0), "open", FRAME_W, FRAME_H, t)
        self.assertTrue(engaged)
        self.assertEqual(linear_x, BASE_CFG["linear_mps"])
        self.assertEqual(angular_z, 0.0)

    def test_center_zone_is_not_engaged(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        linear_x, angular_z, engaged = controller.update(
            NEUTRAL, "open", FRAME_W, FRAME_H, 0.0)   # 첫 등장 = 그 자리가 center
        self.assertFalse(engaged)
        self.assertEqual((linear_x, angular_z), (0.0, 0.0))

    def test_ambiguous_diagonal_is_not_engaged(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        t = _prime(controller)
        fingertip_px = (320.0 + 80.0, 240.0 - 80.0)   # 45도 대각선 — 애매
        linear_x, angular_z, engaged = controller.update(
            fingertip_px, "open", FRAME_W, FRAME_H, t)
        self.assertFalse(engaged)
        self.assertEqual((linear_x, angular_z), (0.0, 0.0))

    def test_leaving_zone_stops_immediately(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        t = _prime(controller)
        controller.update((320.0, 140.0), "open", FRAME_W, FRAME_H, t)   # 위 존 — 주행
        linear_x, angular_z, engaged = controller.update(
            NEUTRAL, "open", FRAME_W, FRAME_H, t + 0.05)   # 중심으로 복귀
        self.assertFalse(engaged)
        self.assertEqual((linear_x, angular_z), (0.0, 0.0))

    def test_hand_loss_stops_and_recalibrates_on_reappear(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        t = _prime(controller)
        controller.update((320.0, 140.0), "open", FRAME_W, FRAME_H, t)   # 위 존 확인
        linear_x, angular_z, engaged = controller.update(None, "open", FRAME_W, FRAME_H, t + 0.1)
        self.assertEqual((linear_x, angular_z, engaged), (0.0, 0.0, False))
        # 재등장 — 예외 없이 동작해야 하고, 그 자리가 다시 새 기준(0점)이 된다
        linear_x, angular_z, engaged = controller.update(
            (500.0, 400.0), "open", FRAME_W, FRAME_H, t + 0.2)
        self.assertEqual(controller.neutral_px, (500.0, 400.0))
        self.assertFalse(engaged)   # 재등장 직후는 그 자리가 곧 center

    def test_camera_switch_reset_recalibrates_without_cursor_jump(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        t = _prime(controller)
        controller.update((500.0, 100.0), "open", FRAME_W, FRAME_H, t)
        controller.reset_tracking()
        linear_x, angular_z, engaged = controller.update(
            (80.0, 420.0), "open", FRAME_W, FRAME_H, t + 0.1)
        self.assertEqual(controller.neutral_px, (80.0, 420.0))
        self.assertEqual(controller.offset_px, (0.0, 0.0))
        self.assertEqual((linear_x, angular_z, engaged), (0.0, 0.0, False))

    def test_v_sign_recenters_at_current_point_once(self):
        cfg = dict(BASE_CFG, v_sign_recenter={"enabled": True})
        controller = DpadController(cfg, PF_CFG)
        t = _prime(controller)
        target = (500.0, 100.0)
        linear_x, angular_z, engaged = controller.update(
            target, "v", FRAME_W, FRAME_H, t)
        self.assertEqual(controller.offset_px, (0.0, 0.0))
        self.assertEqual((linear_x, angular_z, engaged), (0.0, 0.0, False))
        controller.update((450.0, 150.0), "v", FRAME_W, FRAME_H, t + 0.1)
        self.assertNotEqual(controller.offset_px, (0.0, 0.0))

    def test_recenter_after_holding_still_away_from_neutral(self):
        cfg = dict(BASE_CFG, recenter={"enabled": True, "still_radius_ratio": 0.05,
                                       "still_sec": 0.5})
        controller = DpadController(cfg, PF_CFG)
        t = _prime(controller)   # neutral = (320, 240)
        still_point = (400.0, 300.0)   # neutral과 다른 자리에서 가만히 있기
        _, t = _settle(controller, still_point, "fist", t, iterations=20)
        linear_x, angular_z, engaged = controller.update(
            still_point, "open", FRAME_W, FRAME_H, t)
        self.assertFalse(engaged)   # 재중심된 자리 = 이제 그 자리가 center 존
        self.assertEqual((linear_x, angular_z), (0.0, 0.0))


class CursorSensitivityAndCalibrationTest(unittest.TestCase):
    def test_sensitivity_amplifies_reach(self):
        # center_radius_px = 0.10 * 480 = 48px. 원래(raw) 40px 이동은 못 미치지만
        # 3배 증폭되면 120px로 커져 UP 존에 닿는다.
        cfg = dict(BASE_CFG, cursor_sensitivity=3.0)
        controller = DpadController(cfg, PF_CFG)
        t = _prime(controller)
        small_move_px = (320.0, 240.0 - 40.0)
        (linear_x, angular_z, engaged), _ = _settle(controller, small_move_px, "open", t)
        self.assertTrue(engaged)
        self.assertEqual(linear_x, cfg["linear_mps"])
        # 화면에 그리는 커서 오프셋도 같은 배율로 증폭돼 있어야 한다(표시=판정 일치):
        # raw dy=-40px * 3배 = -120px
        self.assertAlmostEqual(controller.offset_px[1], -120.0, places=1)

    def test_same_small_move_without_amplification_stays_center(self):
        cfg = dict(BASE_CFG, cursor_sensitivity=1.0)
        controller = DpadController(cfg, PF_CFG)
        t = _prime(controller)
        small_move_px = (320.0, 240.0 - 40.0)
        (linear_x, angular_z, engaged), _ = _settle(controller, small_move_px, "open", t)
        self.assertFalse(engaged)   # 40px < center_radius(48px) — 증폭 없인 아직 center

    def test_neutral_calibrates_on_first_appearance_not_frame_center(self):
        controller = DpadController(BASE_CFG, PF_CFG)
        off_center_point = (100.0, 500.0)   # 화면 중앙이 아닌 곳
        controller.update(off_center_point, "open", FRAME_W, FRAME_H, 0.0)
        self.assertEqual(controller.neutral_px, off_center_point)


class LockToggleTest(unittest.TestCase):
    """center 존에서 open 손을 잠시 멈추면 armed가 반전되는 주행 잠금 토글."""

    def _cfg(self, **overrides):
        cfg = dict(BASE_CFG)
        cfg["recenter"] = {"enabled": False}
        cfg["lock_toggle"] = dict(
            {"enabled": True, "start_armed": False, "require_shape": "open", "hold_sec": 0.5},
            **overrides,
        )
        return cfg

    def _hold(self, controller, point_px, shape, start_t, seconds, step=0.1):
        t = start_t
        while t < start_t + seconds:
            controller.update(point_px, shape, FRAME_W, FRAME_H, t)
            t += step
        return t

    def test_starts_locked_by_default(self):
        controller = DpadController(self._cfg(), PF_CFG)
        self.assertFalse(controller.armed)
        linear_x, angular_z, engaged = controller.update(
            (320.0, 140.0), "open", FRAME_W, FRAME_H, 0.0)
        self.assertEqual((linear_x, angular_z, engaged), (0.0, 0.0, False))

    def test_holding_open_at_center_unlocks(self):
        controller = DpadController(self._cfg(), PF_CFG)
        t = self._hold(controller, NEUTRAL, "open", 0.0, 1.0)
        self.assertTrue(controller.armed)
        linear_x, angular_z, engaged = controller.update((320.0, 140.0), "open", FRAME_W, FRAME_H, t)
        self.assertTrue(engaged)
        self.assertEqual(linear_x, self._cfg()["linear_mps"])

    def test_wrong_shape_does_not_unlock(self):
        controller = DpadController(self._cfg(), PF_CFG)
        self._hold(controller, NEUTRAL, "fist", 0.0, 1.0)
        self.assertFalse(controller.armed)

    def test_toggle_relocks_on_second_hold(self):
        controller = DpadController(self._cfg(), PF_CFG)
        t = self._hold(controller, NEUTRAL, "open", 0.0, 1.0)
        self.assertTrue(controller.armed)
        t = self._hold(controller, (320.0, 140.0), "open", t, 0.2)   # center 존 이탈 — 재장전
        self._hold(controller, NEUTRAL, "open", t, 1.0)
        self.assertFalse(controller.armed)

    def test_continuing_to_hold_does_not_retoggle(self):
        controller = DpadController(self._cfg(), PF_CFG)
        t = self._hold(controller, NEUTRAL, "open", 0.0, 1.0)
        self.assertTrue(controller.armed)
        self._hold(controller, NEUTRAL, "open", t, 1.0)   # center를 안 벗어나고 계속 유지
        self.assertTrue(controller.armed)   # 재발화되어 다시 잠기면 안 된다

    def test_disabled_lock_toggle_is_always_armed(self):
        controller = DpadController(self._cfg(enabled=False, start_armed=False), PF_CFG)
        self.assertTrue(controller.armed)
        t = _prime(controller)
        linear_x, angular_z, engaged = controller.update(
            (320.0, 140.0), "open", FRAME_W, FRAME_H, t)
        self.assertTrue(engaged)


if __name__ == "__main__":
    unittest.main()
