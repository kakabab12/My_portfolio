"""rotor 단위 테스트 — 카메라·모델 없이 로터(리모컨) 다이얼 판정만 검증한다.

2026-08-05 신설 (feat/rotor_remote 판) → 같은 날 사용자 확인 8차(펼친 손 래칫
다이얼 — 기아 자동차 다이얼 방식)로 재작성: 조작 = 펼친 손을 제자리에서
비틀기(손목→중지 뿌리 축 각도), 문턱 이상 비틀면 포커스 1칸(오른쪽=다음·
왼쪽=이전) + 중립 복귀로 재장전(찰칵), 확정 = 주먹 쥐기(연속 판별 확인).
머무름(차징)은 7차에서 이미 제거 — 없다. 검지 조준 방식은 완전 교체(폐기).
입력은 추적 손 1개만(후보 폴백 없음). 버튼은 config 목록(4→9개 확장 대비).

각도 규약: 0도=12시·시계방향+ (화면 y는 아래로 증가 — clock_angle_deg 독스트링).
각 테스트는 _open(0)으로 중립을 12시에 놓고 시작한다(중립 = 첫 관측 방향).

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.gesture_filter import GestureEvent
from src.postprocess.hand_shape import HAND_FINGERS
from src.postprocess.rotor import RotorController, clock_angle_deg

FRAME_DT_SEC = 1.0 / 30.0
FRAME_W_PX, FRAME_H_PX = 1280, 720
WRIST_PX = (500.0, 300.0)
HAND_AXIS_LEN_PX = 100.0   # 손목 -> 중지 뿌리 축 길이 (화면 픽셀 — 각도만 쓴다)


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_world(pose):
    """월드 랜드마크 (21,3) — 자세별 손가락 상태 (finger_states 임계 대비 여유).

    open:  전부 폄(1.8) -> SHAPE_OPEN (조작 자세)
    fist:  전부 굽힘(0.4) -> SHAPE_FIST (확정)
    point: 검지만 폄 -> SHAPE_FINGER (조작도 확정도 아님 — 계수 리셋)
    uncertain: 검지 기권(0.9·비접힘) + 나머지 굽힘 -> 판별 None (블러 재현)
    """
    world = np.zeros((21, 3))
    for finger_idx, (mcp, pip, dip, tip) in enumerate(HAND_FINGERS):
        world[mcp] = (0.0, 0.03, 0.0)
        world[pip] = (0.0, 0.05, 0.0)
        is_extended = (pose == "open") or (pose == "point" and finger_idx == 0)
        if is_extended:
            world[dip] = (0.0, 0.07, 0.0)
            world[tip] = (0.0, 0.09, 0.0)
        elif pose == "uncertain" and finger_idx == 0:
            world[dip] = (0.0, 0.04, 0.0)
            world[tip] = (0.0, 0.045, 0.0)   # 비율 0.9 — 폄도 굽힘 확인도 아님(기권)
        else:
            world[dip] = (0.0, 0.04, 0.0)
            world[tip] = (0.0, 0.02, 0.0)
    return world


class FakeHand:
    def __init__(self, landmarks, pose="open", conf=0.9):
        self.landmarks = landmarks
        self.world_landmarks = make_world(pose)
        self.conf = conf


class FakeSelector:
    """rotor가 쓰는 hand_selector 표면만 흉내 낸다 — 추적 손 + 후보 목록."""

    def __init__(self):
        self.hand = None
        self.candidates = []

    def tracked_hand(self):
        return self.hand

    def tracked_hand_landmarks(self):
        return None if self.hand is None else self.hand.landmarks

    def candidate_hands(self):
        return self.candidates


def make_landmarks(mcp_px, wrist=WRIST_PX):
    """손목(0)·중지 뿌리(9)만 의미 있는 21점 화면 좌표 목록 — 나머지는 원점."""
    return ([wrist] + [(0.0, 0.0)] * 8 + [mcp_px] + [(0.0, 0.0)] * 11)


def hand_at(angle_deg, wrist=WRIST_PX, length_px=HAND_AXIS_LEN_PX):
    """손목에서 시계 각도(0=12시) 방향으로 뻗은 중지 뿌리 좌표 — 손의 비틀림 자세."""
    angle_rad = math.radians(angle_deg)
    return (wrist[0] + length_px * math.sin(angle_rad),
            wrist[1] - length_px * math.cos(angle_rad))


BUTTONS_4 = [
    {"name": "home", "event": "home", "label": "HOME"},
    {"name": "ok", "event": "confirm", "label": "OK"},
    {"name": "back", "event": "back", "label": "BACK"},
    {"name": "select", "event": "select", "label": "SELECT"},
]


def make_config(start_grace_sec=0.0, buttons=None, recenter_alpha=0.0):
    """recenter 기본 0(재중심 없음) — 재중심은 RotorRecenterTest만 켠다."""
    return {
        "rotor": {
            "toggle_event": "up",
            "step_threshold_deg": 25.0,
            "rearm_ratio": 0.5,
            "recenter_alpha": recenter_alpha,
            "fist_confirm_frames": 3,
            "flash_sec": 0.3,
            "start_grace_sec": start_grace_sec,
            "buttons": buttons if buttons is not None else BUTTONS_4,
        },
        # 조작(펼친 손)·확정(주먹) 판별 임계 — 본 엔진과 같은 키 재사용 (rotor.py)
        "hand_select": {"hand_shape": {"extend_ratio": 1.05,
                                       "min_valid_fingers": 3,
                                       "curl_confirm_ratio": 0.85}},
    }


class ClockAngleTest(unittest.TestCase):
    """각도 규약 — 0=12시·시계방향+ (4방위 실좌표 검증)."""

    def test_four_directions(self):
        origin = (0.0, 0.0)
        self.assertAlmostEqual(clock_angle_deg(origin, (0.0, -1.0)), 0.0)     # 위
        self.assertAlmostEqual(clock_angle_deg(origin, (1.0, 0.0)), 90.0)     # 오른쪽
        self.assertAlmostEqual(clock_angle_deg(origin, (0.0, 1.0)), 180.0)    # 아래
        self.assertAlmostEqual(clock_angle_deg(origin, (-1.0, 0.0)), -90.0)   # 왼쪽


class RotorTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.selector = FakeSelector()
        self.rotor = RotorController(make_config(), FRAME_W_PX, FRAME_H_PX,
                                     clock=self.clock)

    def _event(self, class_name):
        return GestureEvent(class_name=class_name, conf=1.0, ts_sec=self.clock.now_sec)

    def _feed(self, mcp_px, frame_count=1, gesture_event=None, pose="open"):
        """프레임 공급 — 로터가 발화한 이벤트 목록을 돌려준다."""
        fired = []
        for _ in range(frame_count):
            self.selector.hand = (None if mcp_px is None
                                  else FakeHand(make_landmarks(mcp_px), pose))
            event = self.rotor.update(gesture_event, self.selector)
            gesture_event = None   # 쓸기 이벤트는 첫 프레임에만
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                fired.append(event)
        return fired

    def _turn_on(self):
        self.assertEqual(self._feed(None, gesture_event=self._event("up")), [])
        self.assertTrue(self.rotor.is_rotor_on)

    def _open(self, twist_deg, frame_count=3):
        """펼친 손을 해당 비틀림 자세로 공급 (중립은 첫 관측 방향)."""
        return self._feed(hand_at(twist_deg), frame_count=frame_count, pose="open")

    def _fist(self, frame_count=5):
        """주먹 쥐기(확정 동작) — 손 방향은 무의미(비틀림 판정은 펼친 손에서만)."""
        return self._feed(hand_at(0), frame_count=frame_count, pose="fist")

    def _focus(self):
        return self.rotor.snapshot()["focus_name"]


class RotorToggleTest(RotorTestBase):
    """위 쓸기(up) = 설정 토글 — 파이프로 안 나가고 모드만 바꾼다."""

    def test_up_toggles_on_and_is_consumed(self):
        fired = self._feed(None, gesture_event=self._event("up"))
        self.assertEqual(fired, [])                  # up은 파이프로 안 나간다
        self.assertTrue(self.rotor.is_rotor_on)

    def test_up_again_toggles_off(self):
        self._turn_on()
        self._feed(None, gesture_event=self._event("up"))
        self.assertFalse(self.rotor.is_rotor_on)

    def test_left_right_pass_through_when_off(self):
        fired = self._feed(None, gesture_event=self._event("left"))
        self.assertEqual([e.class_name for e in fired], ["left"])

    def test_swipes_ignored_while_on(self):
        # 로터 켜짐 중엔 토글(up) 외 쓸기 전부 무시 (사용자 결정 2026-08-05)
        self._turn_on()
        self.assertEqual(self._feed(None, gesture_event=self._event("left")), [])
        self.assertEqual(self._feed(None, gesture_event=self._event("right")), [])

    def test_off_state_reports_snapshot(self):
        self._feed(None, gesture_event=self._event("left"))
        self.assertFalse(self.rotor.snapshot()["is_on"])


class RotorRatchetTest(RotorTestBase):
    """래칫 스텝 — 문턱 이상 비틀면 1칸, 중립 복귀로 재장전 (찰칵 감각)."""

    def test_right_twist_steps_to_next(self):
        # 오른쪽 30도 비틀기(문턱 25도 이상) — home에서 다음(ok)으로 1칸
        self._turn_on()
        self._open(0)
        self.assertEqual(self._focus(), "home")
        self._open(30)
        self.assertEqual(self._focus(), "ok")

    def test_left_twist_steps_to_previous(self):
        # 왼쪽 비틀기 — 이전(목록 끝 select)으로 1칸
        self._turn_on()
        self._open(0)
        self._open(-30)
        self.assertEqual(self._focus(), "select")

    def test_below_threshold_does_not_step(self):
        # 문턱(25도) 미달 비틀기 — 포커스 유지
        self._turn_on()
        self._open(0)
        self._open(20)
        self.assertEqual(self._focus(), "home")

    def test_holding_twist_steps_once(self):
        # 비튼 채 유지 — 1칸에서 멈춘다(찰칵). 되돌아와야 다음 칸
        self._turn_on()
        self._open(0)
        self._open(30, frame_count=40)
        self.assertEqual(self._focus(), "ok")

    def test_return_to_neutral_rearms_next_step(self):
        # 비틀기 → 중립 복귀(재장전 반경 12.5도 미만) → 다시 비틀기 = 두 칸
        self._turn_on()
        self._open(0)
        self._open(30)                                    # home -> ok
        self._open(5)                                     # 중립 복귀 — 재장전
        self._open(30)                                    # ok -> back
        self.assertEqual(self._focus(), "back")

    def test_partial_return_does_not_rearm(self):
        # 재장전 반경(12.5도)까지 안 돌아오면 장전 안 됨 — 다시 비틀어도 그대로
        self._turn_on()
        self._open(0)
        self._open(30)                                    # home -> ok
        self._open(15)                                    # 12.5도 밖 — 미장전
        self._open(30)
        self.assertEqual(self._focus(), "ok")

    def test_twist_reading_matches_hand(self):
        # 비틀림 표시값 = 중립 대비 손 각도 그대로 (UI 노브 마커 근거)
        self._turn_on()
        self._open(0)
        self._open(18)
        self.assertAlmostEqual(self.rotor.snapshot()["twist_deg"], 18.0, places=5)

    def test_reopen_recaptures_neutral(self):
        # 펼친 손이 끊겼다 다시 나타나면 그 자세가 새 중립 — 이전 비틀림 승계 금지
        self._turn_on()
        self._open(0)
        self._open(30)                                    # home -> ok
        self._feed(None, frame_count=3)                   # 소실
        self._open(30)                                    # 같은 자세로 재등장 = 새 중립
        self.assertEqual(self._focus(), "ok")             # 추가 스텝 없음
        self._open(60)                                    # 새 중립 대비 +30도
        self.assertEqual(self._focus(), "back")


class RotorFistFireTest(RotorTestBase):
    """주먹 확정 — 펼친 손을 본 뒤 주먹을 쥐면 현재 포커스가 발화한다."""

    def test_fist_fires_focused_button(self):
        self._turn_on()
        self._open(0)
        self._open(30)                                    # 포커스 ok
        fired = self._fist()
        self.assertEqual([e.class_name for e in fired], ["confirm"])

    def test_fist_without_open_does_not_fire(self):
        # 펼친 손 없이 처음부터 주먹(휴식 손) — 발화 금지
        self._turn_on()
        self.assertEqual(self._fist(frame_count=30), [])

    def test_holding_fist_fires_once(self):
        # 쥔 채 유지해도 1회 — 다시 펼쳐야 재발화(중립도 재캡처된다)
        self._turn_on()
        self._open(0)
        fired = self._fist(frame_count=60)
        self.assertEqual([e.class_name for e in fired], ["home"])
        self._open(0)                                     # 다시 폄 — 재장전
        fired = self._fist()
        self.assertEqual([e.class_name for e in fired], ["home"])

    def test_short_fist_below_confirm_frames_does_not_fire(self):
        # 주먹 판별이 연속 3프레임 미달(오판 한두 프레임) — 발화 금지
        self._turn_on()
        self._open(0)
        fired = self._fist(frame_count=2)
        fired += self._open(0, frame_count=2)             # 폄 복귀 — 계수 리셋
        fired += self._fist(frame_count=2)
        self.assertEqual(fired, [])

    def test_blur_frames_keep_fist_streak(self):
        # 쥐는 도중 판별 불가(블러) 프레임은 계수를 끊지 않는다 — 확정이 씹히지 않게
        self._turn_on()
        self._open(0)
        fired = self._fist(frame_count=2)
        fired += self._feed(hand_at(0), frame_count=2, pose="uncertain")
        fired += self._fist(frame_count=1)                # 누적 3프레임 — 발화
        self.assertEqual([e.class_name for e in fired], ["home"])

    def test_point_pose_resets_fist_streak(self):
        # 한 손가락 등 다른 모양은 계수를 리셋한다 — 토막 주먹 합산 금지
        self._turn_on()
        self._open(0)
        fired = self._fist(frame_count=2)
        fired += self._feed(hand_at(0), frame_count=2, pose="point")
        fired += self._fist(frame_count=2)
        self.assertEqual(fired, [])

    def test_hand_loss_resets_fist_streak(self):
        # 손 소실도 계수를 끊는다 — 소실을 사이에 둔 두 토막 주먹은 무발화
        self._turn_on()
        self._open(0)
        fired = self._fist(frame_count=2)
        fired += self._feed(None, frame_count=2)
        fired += self._fist(frame_count=2)
        self.assertEqual(fired, [])
        self.assertEqual(self._focus(), "home")           # 표시는 유지(대기)

    def test_candidates_are_ignored(self):
        # ★입력 손 1개 제한(2026-08-05 사용자 확인): 추적 손이 없으면 후보가
        # 있어도 **대기**다 — 신뢰도 폴백이 노이즈 손·반대쪽 손을 태우던 구멍 봉쇄
        self._turn_on()
        self.selector.candidates = [
            ((0.0, 0.0), FakeHand(make_landmarks(hand_at(30))), 0.2, 100.0),
        ]
        fired = self._feed(None, frame_count=60)
        self.assertEqual(fired, [])
        self.assertIsNone(self.rotor.snapshot()["twist_deg"])   # 관측 자체가 없다


class RotorRecenterTest(RotorTestBase):
    """중립 지속 재중심(사용자 제안 캘리브레이션) — 손 자세가 흘러도 기준이 따라간다."""

    def setUp(self):
        super().setUp()
        self.rotor = RotorController(make_config(recenter_alpha=0.05),
                                     FRAME_W_PX, FRAME_H_PX, clock=self.clock)

    def test_neutral_follows_drifted_pose(self):
        # 재장전 반경 안(10도)에서 오래 머물면 중립이 그 자세로 수렴 — 비틀림 ≈ 0
        self._turn_on()
        self._open(0)
        self._open(10, frame_count=60)                    # 흘러간 자세 유지 (2초)
        self.assertLess(abs(self.rotor.snapshot()["twist_deg"]), 2.0)
        self._open(40)                                    # 새 기준 대비 +30도 — 스텝
        self.assertEqual(self._focus(), "ok")


class RotorNineButtonTest(RotorTestBase):
    """버튼 확장(사용자 요구 — 4개→9개) — config 목록만 늘리면 순환으로 동작한다."""

    def setUp(self):
        super().setUp()
        buttons = [{"name": f"b{idx}", "event": f"event_{idx}", "label": f"B{idx}"}
                   for idx in range(9)]
        self.rotor = RotorController(make_config(buttons=buttons),
                                     FRAME_W_PX, FRAME_H_PX, clock=self.clock)

    def test_two_ratchet_steps_reach_b2(self):
        self._turn_on()
        self._open(0)
        self._open(30)                                    # b0 -> b1
        self._open(0)                                     # 재장전
        self._open(30)                                    # b1 -> b2
        fired = self._fist()
        self.assertEqual([e.class_name for e in fired], ["event_2"])

    def test_left_twist_wraps_to_last_button(self):
        self._turn_on()
        self._open(0)
        self._open(-30)                                   # b0 -> b8 (역순 순환)
        fired = self._fist()
        self.assertEqual([e.class_name for e in fired], ["event_8"])


class RotorStartGraceTest(RotorTestBase):
    """시작 유예(2026-08-05 실기) — 토글 직후에는 스텝·확정하지 않는다."""

    def setUp(self):
        super().setUp()
        self.rotor = RotorController(make_config(start_grace_sec=1.2),
                                     FRAME_W_PX, FRAME_H_PX, clock=self.clock)

    def test_twist_and_fist_during_grace_do_nothing(self):
        # 유예(1.2초) 안의 비틀기·주먹 — 스텝도 발화도 없다 (복귀 팔 오발 방지)
        self._turn_on()
        fired = self._open(30, frame_count=5)
        self.assertEqual(self._focus(), "home")           # 스텝 없음 (중립이 따라감)
        fired += self._fist(frame_count=10)               # 0.5초 시점 — 유예 안
        self.assertEqual(fired, [])

    def test_grace_end_pose_becomes_neutral(self):
        # 유예 동안 유지한 자세가 중립이 된다 — 유예 뒤 거기서 +30도 비틀면 1칸
        self._turn_on()
        self._open(30, frame_count=45)                    # 1.5초 — 유예 경과(중립=30도)
        self.assertEqual(self._focus(), "home")
        self._open(60)                                    # 중립 대비 +30도
        self.assertEqual(self._focus(), "ok")


if __name__ == "__main__":
    unittest.main()
