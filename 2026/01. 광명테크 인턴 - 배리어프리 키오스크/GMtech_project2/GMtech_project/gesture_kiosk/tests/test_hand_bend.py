"""hand_bend 단위 테스트 — 카메라·모델 없이 손 꺾임(포스처) 판정만 검증한다.

2026-08-05 신설 → 08-06 9차 확정(사용자 지정 매핑):
- 세 모양(펼침·검지 주먹·주먹) 전부 같은 4방위 섹터(±45·±135) 각도 판정
  (7차 "판정 똑같게" — 높이·어깨 앵커·z 판정 미사용, 계측 표시 전용.
  중립(open_flat)·히스테리시스 소멸).
- 펼친 손: 좌/우/위/아래 = left/right/home/back (8차 매핑)
- 검지 주먹(포인팅): 위 = **select** · 하/좌/우 = temp_* (9차)
- 주먹: 평평(±22도) = temp(무방향) · 꺾임 4방위 = temp_* (9차 — 8차
  "주먹 = select" 폐기. 뒤집기 폐기 유지. 평평 존은 주먹에만)
- temp 계열은 발화해도 쿨다운을 안 건다 — 예비 자세 직후 본 제스처 보호
- OK 사인(엄지·검지 고리 + 중지~새끼 폄) = confirm (8차 신설)
- 쓸기 이벤트는 종류 무관 전부 삼켜진다 — 파이프 출력은 postures 매핑뿐.

좌표 규약: 화면 y는 아래로 증가, 각도 0=12시·시계방향+ (rotor.clock_angle_deg).
기준은 화면 수직(12시) 고정(2026-08-06 사용자 결정 — 포즈 팔 추적점 전면
제거): 손 방향(손목→중지 뿌리)의 화면 각도가 곧 꺾임각이다.
테스트 hold 값은 배포값과 무관하게 고정(0.35/0.9) — 판정 로직만 검증한다.

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
from src.postprocess.hand_bend import (
    HandBendController, compass_posture, wrap180_deg,
)
from src.postprocess.hand_shape import HAND_FINGERS, INDEX_TIP_IDX, THUMB_TIP_IDX, is_ok_sign

FRAME_DT_SEC = 1.0 / 30.0
WRIST_PX = (500.0, 300.0)
KNUCKLE_LEN_PX = 80.0

SHOULDER_RAISED = (600.0, 200.0)   # 계측(up= 표시) 전용 — 7차부터 높이는 판정
SHOULDER_LOW = (350.0, 200.0)      #   미사용: 두 값 어느 쪽이든 판정이 같아야 한다
HOLD_FRAMES = 15        # hold_sec 0.35 (11프레임) 여유 포함
FLAT_HOLD_FRAMES = 32   # flat_hold_sec 0.9 (27프레임) 여유 포함

# 손가락 뿌리(MCP)의 월드 x 오프셋 (검지·중지·약지·새끼) — 손바닥 평면의 외적
# 부호가 손등/손바닥에서 반대가 되게 하는 배치. 부호 규약은 합성 좌표 기준 —
# 실기 부호는 config postures 매핑으로 흡수한다 (hand_bend.py 독스트링)
FIST_BACK_XS = (-0.03, -0.01, 0.01, 0.03)
FIST_PALM_XS = (0.03, 0.01, -0.01, -0.03)


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_world(pose, knuckle_z=0.0):
    """월드 랜드마크 (21,3) — 자세별 손가락 상태 (finger_states 임계 대비 여유).

    open: 전부 폄 -> SHAPE_OPEN / fist: 전부 굽힘(손등) · fist_palm: 굽힘+뒤집힘
    (8차부터 둘 다 같은 "주먹" — 뒤집어도 select가 같음을 검증하는 데 쓴다)
    point: 검지만 폄 -> SHAPE_FINGER (자세 아님) / uncertain: 판별 None (블러 재현)
    ok: 검지 굽힘 + 엄지 끝을 검지 끝에 붙임(고리 닫힘) + 중지~새끼 폄 -> OK 사인
    knuckle_z: 중지 뿌리(9)의 월드 z — 손목 젖힘(z 기울기) 재현 (계측 표시 검증용)
    """
    xs = FIST_PALM_XS if pose == "fist_palm" else FIST_BACK_XS
    world = np.zeros((21, 3))
    for finger_idx, (mcp, pip, dip, tip) in enumerate(HAND_FINGERS):
        x = xs[finger_idx]
        world[mcp] = (x, 0.03, 0.0)
        world[pip] = (x, 0.05, 0.0)
        is_extended = ((pose == "open") or (pose == "point" and finger_idx == 0)
                       or (pose == "ok" and finger_idx > 0))
        if is_extended:
            world[dip] = (x, 0.07, 0.0)
            world[tip] = (x, 0.09, 0.0)
        elif pose == "uncertain" and finger_idx == 0:
            world[dip] = (x, 0.04, 0.0)
            world[tip] = (x, 0.045, 0.0)   # 비율 ~0.93 — 폄도 굽힘 확인도 아님(기권)
        else:
            world[dip] = (x, 0.04, 0.0)
            world[tip] = (x, 0.02, 0.0)
    if pose == "ok":
        world[THUMB_TIP_IDX] = world[INDEX_TIP_IDX]   # 고리 닫힘 — 끝끼리 맞닿음
    world[9][2] = knuckle_z
    return world


def make_landmarks(knuckle_px, wrist=WRIST_PX):
    """손목(0)·중지 뿌리(9)만 의미 있는 21점 화면 좌표 — 나머지는 손목과 겹침.

    (0,0) 고정이 아니라 손목에 두는 이유: 손 중심(21점 평균 —
    hand_center_point)이 실제 손 위치로 나와야 계측(up= 높이 표시)이 성립한다.
    np.array인 이유: 실제 HandDetection.landmarks와 같은 형(넘파이 슬라이싱).
    """
    points = [wrist] * 21
    points[9] = knuckle_px
    return np.array(points, dtype=float)


def knuckle_at(angle_deg, wrist=WRIST_PX, length_px=KNUCKLE_LEN_PX):
    """손목에서 시계 각도(0=12시) 방향으로 뻗은 중지 뿌리 좌표 — 손 방향 지정."""
    angle_rad = math.radians(angle_deg)
    return (wrist[0] + length_px * math.sin(angle_rad),
            wrist[1] - length_px * math.cos(angle_rad))


class FakeHand:
    def __init__(self, landmarks, pose="open", user_side="right", conf=0.9,
                 knuckle_z=0.0):
        self.landmarks = landmarks
        self.world_landmarks = make_world(pose, knuckle_z)
        self.user_side = user_side
        self.conf = conf


class FakeSelector:
    """hand_bend가 쓰는 hand_selector 표면만 흉내 — 추적 손 + 어깨선.

    shoulder_frame: (어깨선 y_px, 어깨너비 px) | None — 7차부터 계측(up= 표시)
    전용이다. 판정은 어느 값이든(None 포함) 같아야 한다 — 그 불변을 테스트가
    직접 검증한다(test_up_needs_no_shoulder_anchor).
    """

    def __init__(self):
        self.hand = None
        self.shoulder_frame = SHOULDER_LOW

    def tracked_hand(self):
        return self.hand

    def anchor_shoulder_frame(self):
        return self.shoulder_frame


def make_config():
    return {
        "hand_bend": {
            # ※구 판정 관문 키 3종(flat_max_deg·bend_min_deg·raise_above_shoulder)
            #   은 7차 소멸 — 섹터 경계(±45·±135)는 코드의 기하 상수다
            "hold_sec": 0.35,
            "flat_hold_sec": 0.9,
            "cooldown_sec": 0.6,
            "ok_touch_palm_ratio": 0.45,
            "ok_touch_screen_ratio": 0.45,
            "fist_flat_max_deg": 22,
            "postures": {
                # 9차 사용자 지정 매핑 — 배포 config와 같은 구조
                "bend_left": "left", "bend_right": "right",
                "bend_up": "home", "bend_down": "back",
                "finger_up": "select", "finger_down": "temp_down",
                "finger_left": "temp_left", "finger_right": "temp_right",
                "fist_flat": "temp",
                "fist_up": "temp_up", "fist_down": "temp_down",
                "fist_left": "temp_left", "fist_right": "temp_right",
                "ok_sign": "confirm",
            },
        },
        # 모양 판별 임계 — 본 엔진 판별과 같은 키를 재사용한다 (hand_bend.py)
        "hand_select": {"hand_shape": {"extend_ratio": 1.05,
                                       "min_valid_fingers": 3,
                                       "curl_confirm_ratio": 0.85}},
    }


class PureFunctionTest(unittest.TestCase):
    """순수 함수 — 각도 접기·4방위·주먹 방향."""

    def test_wrap180(self):
        self.assertAlmostEqual(wrap180_deg(0.0), 0.0)
        self.assertAlmostEqual(wrap180_deg(190.0), -170.0)
        self.assertAlmostEqual(wrap180_deg(-190.0), 170.0)
        self.assertAlmostEqual(wrap180_deg(360.0), 0.0)

    def test_compass_four_directions(self):
        self.assertEqual(compass_posture(0.0), "bend_up")
        self.assertEqual(compass_posture(90.0), "bend_right")
        self.assertEqual(compass_posture(180.0), "bend_down")
        self.assertEqual(compass_posture(270.0), "bend_left")

    def test_ok_sign_pure(self):
        # OK 픽스처(고리 닫힘 + 중지~새끼 폄)만 참 — 다른 자세는 전부 거짓:
        # open은 고리가 열려 있고(엄지 끝이 멀다), fist는 배경 손가락이 굽었다
        self.assertTrue(is_ok_sign(make_world("ok"), 1.05, 0.85, 0.45))
        self.assertFalse(is_ok_sign(make_world("open"), 1.05, 0.85, 0.45))
        self.assertFalse(is_ok_sign(make_world("fist"), 1.05, 0.85, 0.45))
        self.assertFalse(is_ok_sign(make_world("point"), 1.05, 0.85, 0.45))


class BendTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.selector = FakeSelector()
        self.bend = HandBendController(make_config(), clock=self.clock)

    def _feed(self, angle_deg, frame_count=1, pose="open", gesture_event=None,
              knuckle_z=0.0):
        """프레임 공급 — 발화 이벤트 목록. angle_deg=None이면 손 소실."""
        fired = []
        for _ in range(frame_count):
            self.selector.hand = (None if angle_deg is None
                                  else FakeHand(make_landmarks(knuckle_at(angle_deg)),
                                                pose, knuckle_z=knuckle_z))
            event = self.bend.update(gesture_event, self.selector)
            gesture_event = None   # 쓸기 이벤트는 첫 프레임에만
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                fired.append(event)
        return fired

    def _names(self, fired):
        return [event.class_name for event in fired]


class BendPostureTest(BendTestBase):
    """자세 6종 — 유지 시간 뒤 정확한 이벤트 1건 (사용자 지정 매핑, 중립 없음)."""

    def test_bend_left_fires_left(self):
        fired = self._feed(-90, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["left"])

    def test_bend_right_fires_right(self):
        fired = self._feed(90, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["right"])

    def test_bend_down_fires_back(self):
        # 8차 매핑 — 아래 꺾음 = back
        fired = self._feed(180, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["back"])

    def test_bend_up_fires_home(self):
        # ★위 섹터 = 좌/우/하와 동일한 4방위 각도 판정(7차 "판정 똑같게") —
        # 높이·어깨 앵커 무관. 8차 매핑으로 이벤트는 home
        fired = self._feed(0, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["home"])

    def test_up_sector_tilt_also_home(self):
        # 위 섹터는 ±45도 전부 — 구 히스테리시스(22~40 보류)·2D 띠 구간도
        # 이제 같은 위 섹터다 (7차: 펼친 손은 항상 4방위 중 하나)
        fired = self._feed(42, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["home"])
        self.bend = HandBendController(make_config(), clock=self.clock)
        fired = self._feed(-30, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["home"])

    def test_up_needs_no_shoulder_anchor(self):
        # 어깨(앵커) 미관측이어도 위 섹터 발화 — 높이 판정 폐기(7차): 어깨선은
        # 계측(up= 표시) 전용이라 판정에 어떤 영향도 없어야 한다
        self.selector.shoulder_frame = None
        fired = self._feed(0, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["home"])

    def test_raised_bent_hand_is_direction(self):
        # 높이와 무관하게 꺾인 방향이 곧 자세 — 좌로 꺾이면 left다
        self.selector.shoulder_frame = SHOULDER_RAISED
        fired = self._feed(-90, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["left"])

    def test_flat_fist_fires_temp(self):
        # ★9차 보강 — 그냥 쥔 평평 주먹(수직 ±22도) = 무방향 temp
        fired = self._feed(0, frame_count=HOLD_FRAMES, pose="fist")
        self.assertEqual(self._names(fired), ["temp"])

    def test_fist_directions_map_to_temp(self):
        # 주먹도 펼친 손과 같은 섹터 판정 — 평평 존(22도) 밖은 방향별 temp_*
        fired = self._feed(90, frame_count=HOLD_FRAMES, pose="fist")
        self.assertEqual(self._names(fired), ["temp_right"])
        self.bend = HandBendController(make_config(), clock=self.clock)
        fired = self._feed(180, frame_count=HOLD_FRAMES, pose="fist")
        self.assertEqual(self._names(fired), ["temp_down"])
        self.bend = HandBendController(make_config(), clock=self.clock)
        fired = self._feed(30, frame_count=HOLD_FRAMES, pose="fist")
        self.assertEqual(self._names(fired), ["temp_up"])   # 평평 밖·위 섹터 안

    def test_flipped_fist_same_temp(self):
        # 뒤집기 폐기(8차 유지) — 뒤집은 주먹도 그냥 주먹: 같은 temp
        fired = self._feed(0, frame_count=HOLD_FRAMES, pose="fist_palm")
        self.assertEqual(self._names(fired), ["temp"])

    def test_finger_up_fires_select(self):
        # ★9차 — 검지 주먹(포인팅) 위 = select: 사용자 지정 본 제스처
        fired = self._feed(0, frame_count=HOLD_FRAMES, pose="point")
        self.assertEqual(self._names(fired), ["select"])

    def test_finger_sides_fire_temp(self):
        # 검지 주먹 하/좌/우 = temp_* — select는 위 방향 전용 (9차)
        fired = self._feed(-90, frame_count=HOLD_FRAMES, pose="point")
        self.assertEqual(self._names(fired), ["temp_left"])
        self.bend = HandBendController(make_config(), clock=self.clock)
        fired = self._feed(180, frame_count=HOLD_FRAMES, pose="point")
        self.assertEqual(self._names(fired), ["temp_down"])

    def test_fist_ignores_handedness_flicker(self):
        # 주먹 판정은 handedness와 무관(8차부터 분리 소멸) — 출렁여도 한 번만
        fired = []
        for frame_idx in range(HOLD_FRAMES):
            side = "left" if frame_idx % 3 == 2 else "right"
            self.selector.hand = FakeHand(make_landmarks(knuckle_at(0)), "fist",
                                          user_side=side)
            event = self.bend.update(None, self.selector)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                fired.append(event)
        self.assertEqual(self._names(fired), ["temp"])

    def test_side_on_fist_same_temp(self):
        # 옆면 주먹(손바닥 평면 퇴화)도 그냥 주먹 — 구 옆면 보류 소멸(8차 유지)
        fired = []
        for _ in range(60):
            hand = FakeHand(make_landmarks(knuckle_at(0)), "fist")
            hand.world_landmarks[:, 0] = 0.0   # 뿌리들이 한 줄로 — 외적 퇴화
            self.selector.hand = hand
            event = self.bend.update(None, self.selector)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                fired.append(event)
        self.assertEqual(self._names(fired), ["temp"])

    def test_temp_does_not_block_next_event(self):
        # ★temp는 쿨다운을 안 건다(9차) — temp_left 직후 검지 위(select)가
        # 쿨다운(0.6) 대기 없이 hold만 채우면 바로 나간다
        fired = self._feed(-90, frame_count=HOLD_FRAMES, pose="point")   # temp_left
        fired += self._feed(0, frame_count=HOLD_FRAMES, pose="point")    # 0.5초 안 select
        self.assertEqual(self._names(fired), ["temp_left", "select"])

    def test_ok_sign_fires_confirm(self):
        # ★8차 신설 — OK 사인 = confirm: 모양 자세라 flat_hold_sec(0.9초) 유지
        fired = self._feed(0, frame_count=FLAT_HOLD_FRAMES, pose="ok")
        self.assertEqual(self._names(fired), ["confirm"])

    def test_tilted_ok_still_confirms(self):
        # OK도 방향 무관 — 기울인 채 고리를 쥐어도 4방위로 새지 않는다
        # (OK 판별이 섹터보다 먼저 — hand_bend._classify_posture 순서)
        fired = self._feed(90, frame_count=FLAT_HOLD_FRAMES, pose="ok")
        self.assertEqual(self._names(fired), ["confirm"])

    def _screen_ring_hand(self, pose):
        """화면(2D) 고리가 닫힌 손 — 손바닥쪽 OK 환각 재현(월드는 pose 그대로).

        손바닥 폭 80px(5↔17), 엄지 끝↔검지 끝 5.4px = 비율 0.07 (임계 0.45 안).
        """
        hand = FakeHand(make_landmarks(knuckle_at(0)), pose)
        hand.landmarks = hand.landmarks.copy()
        hand.landmarks[5] = (460.0, 300.0)
        hand.landmarks[17] = (540.0, 300.0)
        hand.landmarks[4] = (500.0, 340.0)
        hand.landmarks[8] = (505.0, 342.0)
        return hand

    def test_palm_side_ok_rescued_by_screen_ring(self):
        # ★손바닥쪽 OK 구제(사용자 제안 — "엄지와 검지가 맞닿았는지 로직만
        # 추가"): 월드가 검지를 폄으로 환각해 open으로 읽혀도(실기 16:09 재현)
        # 화면 고리가 닫혀 있으면 OK = confirm — home으로 새지 않는다
        fired = []
        for _ in range(FLAT_HOLD_FRAMES):
            self.selector.hand = self._screen_ring_hand("open")
            event = self.bend.update(None, self.selector)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                fired.append(event)
        self.assertEqual(self._names(fired), ["confirm"])

    def test_screen_ring_on_fist_stays_temp(self):
        # 주먹도 엄지가 화면상 검지에 붙는다 — 배경 손가락(중지~새끼) 폄
        # 조건이 주먹의 OK 오발을 막는다: 화면 고리가 닫혀도 주먹은 temp
        fired = []
        for _ in range(HOLD_FRAMES):
            self.selector.hand = self._screen_ring_hand("fist")
            event = self.bend.update(None, self.selector)
            self.clock.tick(FRAME_DT_SEC)
            if event is not None:
                fired.append(event)
        self.assertEqual(self._names(fired), ["temp"])

class BendRepeatTest(BendTestBase):
    """발화 1회·재장전·경유 오발 — 자세 상태기의 핵심 규칙."""

    def test_holding_bend_fires_once(self):
        fired = self._feed(-90, frame_count=90)   # 3초 유지 — 1회만
        self.assertEqual(self._names(fired), ["left"])

    def test_pump_left_twice_via_up_passthrough(self):
        # 좌→위(0.1초 경유)→좌 = left 2회. 경유 위 섹터는 hold(0.35) 미달이라
        # home으로 안 새고 재장전만 한다 — "지나가는 자세" 방어는 유지 시간 담당
        fired = self._feed(-90, frame_count=HOLD_FRAMES)
        fired += self._feed(0, frame_count=3)              # 0.1초 경유 — 재장전만
        fired += self._feed(-90, frame_count=30)           # 쿨다운(0.6) 지나 재발화
        self.assertEqual(self._names(fired), ["left", "left"])

    def test_slight_release_rearms_without_up_event(self):
        # ★7차 행동 변화: 살짝 풀린 손(-30도)은 위 섹터 = 다른 자세 경유라
        # left가 재장전된다(중립·히스테리시스 소멸 — "판정 똑같게"의 대가).
        # 경유 위 자세(home) 오발은 hold 미달·쿨다운이 삼킨다 — 미포함 확인
        fired = self._feed(-90, frame_count=HOLD_FRAMES)
        fired += self._feed(-30, frame_count=10)           # 0.33초 — hold 미달
        fired += self._feed(-90, frame_count=30)
        self.assertEqual(self._names(fired), ["left", "left"])

    def test_direction_change_fires_new_event(self):
        fired = self._feed(-90, frame_count=HOLD_FRAMES)
        fired += self._feed(90, frame_count=30)            # 쿨다운 포함 여유
        self.assertEqual(self._names(fired), ["left", "right"])

    def test_blur_frames_keep_candidate(self):
        # 판별 불가(블러) 프레임이 유예(0.4초) 안이면 유지 시간이 이어진다
        fired = self._feed(-90, frame_count=5)
        fired += self._feed(-90, frame_count=5, pose="uncertain")
        fired += self._feed(-90, frame_count=5)
        self.assertEqual(self._names(fired), ["left"])

    def test_long_hand_loss_resets_and_refires(self):
        # 유예(0.4초) 초과 소실 — 재등장 자세는 처음부터, 같은 자세도 재발화된다
        fired = self._feed(-90, frame_count=HOLD_FRAMES)
        fired += self._feed(None, frame_count=20)          # 0.67초 소실
        fired += self._feed(-90, frame_count=HOLD_FRAMES)
        self.assertEqual(self._names(fired), ["left", "left"])


class BendPipelineTest(BendTestBase):
    """파이프라인 접점 — 쓸기 삼킴·팔 폴백·계기판."""

    def test_swipe_events_are_swallowed(self):
        # gesture_filter의 이벤트는 종류 무관 파이프로 안 나간다
        for name in ("left", "right", "up", "select", "click"):
            event = GestureEvent(class_name=name, conf=1.0, ts_sec=self.clock.now_sec)
            fired = self._feed(None, gesture_event=event)
            self.assertEqual(fired, [], name)

    def test_debug_reports_bend(self):
        self._feed(-90, frame_count=3)
        self.assertEqual(self.bend.debug["bend_posture"], "bend_left")
        self.assertAlmostEqual(self.bend.debug["bend_deg"], -90.0, places=3)


if __name__ == "__main__":
    unittest.main()
