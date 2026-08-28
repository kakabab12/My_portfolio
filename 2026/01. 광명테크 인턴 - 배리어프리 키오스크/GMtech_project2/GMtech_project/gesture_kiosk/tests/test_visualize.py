"""visualize 단위 테스트 — 그리기 경로가 현재 신호 형식과 맞는지 (2026-08-03 신설).

★신설 이유(키오스크 실기 사고): 탭 판정에 검지 비율을 추가하며 손 신호가
3-튜플 → 4-튜플이 됐는데 draw_user_hands가 3개로 언팩한 채 남아 있었다.
판정 테스트는 전부 통과했지만 **그리기 경로를 아무도 안 봐서** 놓쳤고,
현장에서 `cam on` 하는 순간 추론 스레드가 ValueError로 죽어 화면이 멈췄다.
카메라·모델 없이 대역(fake) 셀렉터로 그리기 계약만 검증한다 — 이런 종류의
불일치는 다시는 실기에서 발견되면 안 된다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.inference.head_detector import HeadDetection
from src.postprocess.hand_select import HandSelector
from src.utils.visualize import draw_debug_panel, draw_status, draw_user_hands
from tests.hand_fixtures import make_hand

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720


def make_frame():
    return np.zeros((FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3), dtype=np.uint8)


def make_selector(with_anchor=False):
    """실물 HandSelector — 신호 형식이 바뀌면 이 테스트가 먼저 깨지도록 대역이 아닌 실물.

    with_anchor: 머리 앵커 절을 넣는다 — 앵커가 있어야 몸 골격(어깨·팔) 좌표가 산다.
    """
    config = {
        "hand_select": {
            "release_sec": 2.0,
            "acquire": {"move_dist_shoulder": 0.25, "window_sec": 0.5},
            "hand_shape": {"extend_ratio": 1.05, "min_valid_fingers": 3,
                           "curl_confirm_ratio": 0.85},
        },
    }
    if with_anchor:
        config["head_anchor"] = {"reach_shoulder_widths": 1.58, "anchor_grace_sec": 1.0}
    return HandSelector(config, FRAME_WIDTH_PX, FRAME_HEIGHT_PX)


class DrawUserHandsTest(unittest.TestCase):
    def test_draws_with_tracked_hand(self):
        # 추적 손이 있는 상태 — user_hand_signal의 **현재 형식**으로 그려져야 한다
        selector = make_selector()
        for step_idx in range(6):
            selector.update([make_hand("right", "finger", (400 + 30 * step_idx, 400))])
        self.assertIsNotNone(selector.user_hand_signal())   # 획득 확인(전제)
        frame = draw_user_hands(make_frame(), selector)
        self.assertEqual(frame.shape, (FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3))
        self.assertGreater(int(frame.sum()), 0)             # 무언가 그려졌다

    def test_draws_without_hand(self):
        # 손이 없어도 예외 없이 지나가야 한다 (유휴 상태의 매 프레임 경로)
        selector = make_selector()
        selector.update([])
        draw_user_hands(make_frame(), selector)

    def test_signal_arity_matches_drawing(self):
        # 계약 고정: 신호는 (모양, 좌표, 라벨, 검지비율) — 항목 수가 바뀌면
        # 그리기·판정 양쪽을 함께 고쳐야 한다는 것을 이 테스트가 강제한다
        selector = make_selector()
        for step_idx in range(6):
            selector.update([make_hand("right", "finger", (400 + 30 * step_idx, 400))])
        signal = selector.user_hand_signal()
        self.assertEqual(len(signal), 4)


class DrawSkeletonTest(unittest.TestCase):
    """골격 선 그리기(2026-08-04 사용자 요청 — 박스 대신 선)의 계약 고정.

    ★이 파일이 생긴 이유와 같다: 그리기 경로는 판정 테스트가 안 덮어서, 셀렉터가
    주는 자료 형식이 바뀌면 현장에서 `cam on` 하는 순간 죽는다.
    """

    def test_body_lines_drawn_with_pose(self):
        # 포즈(머리·어깨·팔꿈치·손목)가 있으면 몸 선이 그려진다
        selector = make_selector(with_anchor=True)
        head = HeadDetection(center_x_px=640, center_y_px=200, width_px=100, conf=1.0,
                             shoulders=((560, 320), (720, 320)),
                             elbows=((520, 430), (760, 430)),
                             wrists=((500, 540), (780, 540)))
        for step_idx in range(6):
            selector.update([make_hand("right", "finger", (600 + 20 * step_idx, 400))],
                            [head])
        self.assertTrue(selector.body_lines())          # 선이 나온다(전제)
        frame = draw_user_hands(make_frame(), selector)
        self.assertGreater(int(frame.sum()), 0)

    def test_body_lines_empty_without_pose(self):
        # 포즈 관측이 없으면 몸 선은 없다 — 그려도 예외 없이 지나가야 한다
        selector = make_selector(with_anchor=True)
        selector.update([make_hand("right", "finger", (600, 400))])
        self.assertEqual(selector.body_lines(), [])
        draw_user_hands(make_frame(), selector)

    def test_tracked_hand_landmarks_shape(self):
        # 손 골격은 추적 손의 21점을 그대로 쓴다 — 형식이 바뀌면 여기서 걸린다
        selector = make_selector()
        for step_idx in range(6):
            selector.update([make_hand("right", "finger", (400 + 30 * step_idx, 400))])
        landmarks = selector.tracked_hand_landmarks()
        self.assertIsNotNone(landmarks)
        self.assertEqual(len(landmarks), 21)

    def test_tracked_hand_landmarks_none_without_hand(self):
        selector = make_selector()
        selector.update([])
        self.assertIsNone(selector.tracked_hand_landmarks())


class DrawPanelTest(unittest.TestCase):
    def test_panel_with_full_debug(self):
        # 계기판 — 탭 진단 필드(2026-08-03 추가) 포함해 그려진다
        debug = {"body_scale": 0.25, "active_side": "right", "hand_shape": "finger",
                 "latched_shape": "finger", "latch_candidate": None, "swallow": None,
                 "swipe_progress_x": 0.4, "swipe_progress_y": -0.1,
                 "tap_index_ratio": 1.34, "tap_baseline": 1.35, "tap_low": 1.15,
                 "tap_drop_y": 0.02, "tap_move_dip": 0.10, "tap_dips": 1,
                 "grab_count": 1, "grab_repeat": 2}
        draw_debug_panel(make_frame(), debug)

    def test_panel_without_tap_fields(self):
        # 탭 기능이 꺼진 config — 탭 필드가 없어도 계기판이 죽지 않는다
        draw_debug_panel(make_frame(), {"body_scale": 0.25, "swipe_progress_x": 0.0,
                                        "swipe_progress_y": 0.0})

    def test_panel_with_empty_debug(self):
        draw_debug_panel(make_frame(), {})

    def test_status_without_event(self):
        draw_status(make_frame(), 29.5, None)


if __name__ == "__main__":
    unittest.main()
