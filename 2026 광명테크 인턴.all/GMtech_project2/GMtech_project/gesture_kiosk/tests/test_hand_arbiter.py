"""hand_arbiter 단위 테스트 — 슬롯 하나만 판정: 먼저 잡힌 손 외엔 후보조차 안 된다.

2026-08-04 신설(사용자 결정 — "두 개 다 보고 제스처하는 게 입력되면 나머지 무시") 뒤
★2026-08-07 MAX_SLOTS 2→1 되돌림(사용자 결정 — "한쪽 손이 인식되면 다른쪽 손은
인식 안 되게, 불필요한 인식들은 다 없애줘"): 두 손 병렬 경쟁(FirstGestureWinsTest)
계약은 폐기 — 슬롯이 하나뿐이라 구조적으로 경쟁이 성립하지 않는다(옛 테스트는
git 이력에 남아있다). 카메라·모델 없이 hand_select 대역(candidate_signals가 주는
튜플)만으로 검증한다. 판정값은 configs/config.yaml **실물**로 읽는다 — 임계가
바뀌면 여기서도 걸린다(test_swipe_scenarios와 같은 이유).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.hand_arbiter import HandArbiter
from src.utils.config_loader import load_config

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")

FPS = 30
FRAME_WIDTH_PX = 1280
SHOULDER_RATIO = 0.22       # 키오스크 표준 거리 — test_swipe_scenarios와 같은 자
SHOULDER_LINE_Y = 0.20
SPAN_PX = 80.0              # 손 폭 — 슬롯 이음 반경 1.5×80 = 120px
AMP_X_PX = 0.25 * FRAME_WIDTH_PX   # 임계의 2배쯤 (시나리오 하네스와 같은 진폭)

LEFT_REST_PX = (0.30 * FRAME_WIDTH_PX, 0.31 * FRAME_WIDTH_PX)
RIGHT_REST_PX = (0.70 * FRAME_WIDTH_PX, 0.31 * FRAME_WIDTH_PX)


class FakeSelector:
    """hand_select 대역 — 손들의 중심을 직접 조종한다.

    손모양 계산은 **지연**이다(2026-08-04 FPS 수리): candidate_hands는 손을 주기만
    하고, 중재기가 판정할 슬롯에 대해서만 signal_for를 부른다. signal_calls로
    그 호출 횟수를 세어 "판정 안 하는 손에는 비용을 안 쓴다"를 검증한다.
    """

    def __init__(self):
        self.hands = {}          # 이름 -> [x_px, y_px]
        self.shapes = {}         # 이름 -> 손모양
        self.adopted_center = None
        self.signal_calls = 0

    def candidate_hands(self):
        entries = []
        for name, center in self.hands.items():
            center_px = (center[0], center[1])
            entries.append((center_px, name, SHOULDER_RATIO, SPAN_PX))
        return entries

    def signal_for(self, hand, center_px):
        self.signal_calls += 1
        return (self.shapes.get(hand, "open"), center_px, "left", None)

    def shoulder_line_y_ratio(self):
        return SHOULDER_LINE_Y

    def adopt_tracked_hand(self, center_px):
        self.adopted_center = center_px
        return True


class ArbiterHarness:
    """가짜 시계로 중재기를 돌린다 — 프레임 단위 주입."""

    def __init__(self):
        self.now_sec = 0.0
        self.selector = FakeSelector()
        self.arbiter = HandArbiter(load_config(CONFIG_PATH), clock=lambda: self.now_sec)
        self.events = []

    def step(self):
        self.now_sec += 1.0 / FPS
        event = self.arbiter.update(self.selector, FRAME_WIDTH_PX)
        if event is not None:
            self.events.append(event.class_name)

    def hold(self, duration_sec):
        for _ in range(round(duration_sec * FPS)):
            self.step()

    def swipe(self, name, dx_px, duration_sec):
        """그 손만 등속으로 옮긴다 — 나머지 손은 제자리."""
        steps = max(1, round(duration_sec * FPS))
        x0, y0 = self.selector.hands[name]
        for i in range(1, steps + 1):
            self.selector.hands[name] = [x0 + dx_px * i / steps, y0]
            self.step()


class SingleSlotTest(unittest.TestCase):
    """★핵심 계약(2026-08-07) — 슬롯은 하나뿐: 먼저 잡힌 손 외엔 후보조차 안 된다."""

    def setUp(self):
        self.harness = ArbiterHarness()
        self.harness.selector.hands = {
            "left_hand": list(LEFT_REST_PX),
            "right_hand": list(RIGHT_REST_PX),
        }

    def test_first_seen_hand_claims_the_only_slot(self):
        # 두 손이 동시에 보여도 슬롯은 하나 — 먼저 관측된(딕셔너리 순서상 첫)
        # 손만 후보가 되고, 나머지는 그 손이 확정·해제될 때까지 아예 안 보인다
        self.harness.hold(0.4)
        self.harness.swipe("right_hand", -AMP_X_PX, 0.3)   # 슬롯을 못 받아 무반응
        self.harness.hold(0.3)
        self.assertEqual(self.harness.events, [])
        self.harness.swipe("left_hand", -AMP_X_PX, 0.3)    # 슬롯을 가진 손만 발화
        self.harness.hold(0.3)
        self.assertEqual(self.harness.events, ["left"])
        self.assertIsNotNone(self.harness.selector.adopted_center)
        self.assertLess(self.harness.selector.adopted_center[0], LEFT_REST_PX[0])

    def test_other_hand_ignored_after_lock(self):
        # 확정 뒤에는 나머지 손이 무엇을 하든 무시 — 애초에 슬롯이 없으므로
        # 구조적으로 판정 대상이 아니다(사용자 결정, 종전 경쟁 폐기 뒤에도 유지)
        self.harness.hold(0.4)
        self.harness.swipe("left_hand", -AMP_X_PX, 0.3)
        self.harness.hold(0.6)          # 쿨다운(0.5초) 통과
        before = list(self.harness.events)
        self.harness.swipe("right_hand", AMP_X_PX, 0.3)
        self.harness.hold(0.3)
        self.assertEqual(self.harness.events, before)   # ★슬롯 없는 손은 이벤트를 못 낸다

    def test_locked_hand_keeps_firing(self):
        # 확정된 손은 계속 판정된다 — 무시는 슬롯 없는 손에만 적용.
        # ※반대 방향으로 되돌리는 획은 복귀 삼킴(return_suppress_sec 1.4)이 1회
        #   먹으므로 그보다 길게 기다린다 — 중재기가 아니라 필터의 정상 동작이다
        self.harness.hold(0.4)
        self.harness.swipe("left_hand", -AMP_X_PX, 0.3)
        self.harness.hold(1.6)
        self.harness.swipe("left_hand", AMP_X_PX, 0.3)
        self.harness.hold(0.3)
        self.assertEqual(self.harness.events, ["left", "right"])

    def test_new_hand_can_acquire_after_release(self):
        # 확정 손이 떠나 소실 유예(release_sec 2.0)를 넘기면 슬롯이 풀리고,
        # 그제서야 다른 손이 새로 슬롯을 받을 수 있다 — 안 그러면 다음
        # 사용자가 영원히 막힌다
        self.harness.hold(0.4)
        self.harness.swipe("left_hand", -AMP_X_PX, 0.3)
        self.harness.hold(0.6)
        del self.harness.selector.hands["left_hand"]
        self.harness.hold(2.3)                          # 유예(2.0) 초과 — 확정 해제
        self.harness.swipe("right_hand", -AMP_X_PX, 0.3)
        self.harness.hold(0.3)
        self.assertEqual(self.harness.events, ["left", "left"])


class SlotGraceTest(unittest.TestCase):
    """슬롯 소실 유예 — 검출 끊김에서 필터(궤적·래치)를 지킨다 (2026-08-04).

    종전엔 한 프레임만 끊겨도 슬롯을 버려서, GestureFilter의 소실 기계(궤적 유예
    0.35초 · 래치 유예 1.5초)가 아예 못 돌았다 — 빠른 쓸기의 모션 블러 끊김에서
    획이 통째로 리셋된다(07-31 "획 씹힘"의 재림 구조).
    """

    def _remove(self, harness, name, duration_sec):
        """손 하나를 duration_sec 동안 관측에서 뺀다 — 모션 블러·가림 재현."""
        center = harness.selector.hands.pop(name)
        harness.hold(duration_sec)
        return center

    def test_stroke_survives_mid_swipe_dropout(self):
        # 획 중간 0.2초 끊김(궤적 유예 0.35 안) — 슬롯이 살아 있어야 궤적이 이어져
        # 한 획으로 발화한다. 종전 코드는 여기서 필터가 리셋돼 획이 죽었다
        harness = ArbiterHarness()
        harness.selector.hands = {"a": list(LEFT_REST_PX)}
        harness.hold(0.4)
        harness.swipe("a", -0.4 * AMP_X_PX, 0.12)          # 획 전반부
        center = self._remove(harness, "a", 0.2)           # 모션 블러 — 관측 끊김
        # 끊긴 동안에도 손은 움직였다 — 재등장은 이어진 위치(재등장 반경 4×80 안)
        harness.selector.hands["a"] = [center[0] - 0.3 * AMP_X_PX, center[1]]
        harness.swipe("a", -0.3 * AMP_X_PX, 0.1)           # 획 후반부
        harness.hold(0.3)
        self.assertEqual(harness.events, ["left"])

    def test_locked_slot_survives_dropout_and_keeps_ignoring_newcomer(self):
        # 확정 손이 잠깐(1초 < release_sec 2.0) 사라져도 확정은 유지된다 — 그
        # 사이 새로 나타난 손이 쓸어도 무시(슬롯이 이미 하나 차 있어 후보조차
        # 안 된다). 슬롯이 죽어야만 새 손이 잡힐 수 있다
        harness = ArbiterHarness()
        harness.selector.hands = {"a": list(LEFT_REST_PX)}
        harness.hold(0.4)
        harness.swipe("a", -AMP_X_PX, 0.3)
        harness.hold(0.3)
        self.assertEqual(harness.events, ["left"])
        center = self._remove(harness, "a", 1.0)           # 화면 가리킴 — 수 초 소실의 축소판
        harness.selector.hands["b"] = list(RIGHT_REST_PX)  # 그 틈에 새 손이 나타나 쓸어도
        harness.swipe("b", AMP_X_PX, 0.3)
        harness.hold(0.3)
        self.assertEqual(harness.events, ["left"])          # ★이벤트가 안 는다 — 여전히 무시
        del harness.selector.hands["b"]
        harness.selector.hands["a"] = list(center)
        harness.hold(0.5)
        harness.swipe("a", -AMP_X_PX, 0.3)                 # 확정 손이 돌아와 계속 쓴다
        harness.hold(0.3)
        self.assertEqual(harness.events, ["left", "left"])

    def test_lock_releases_after_grace_then_new_hand_acquires(self):
        # 확정 손이 유예(release_sec 2.0)를 넘겨 사라지면 확정을 풀어야 한다 —
        # 안 풀면 다음 사용자가 영원히 막힌다
        harness = ArbiterHarness()
        harness.selector.hands = {"a": list(LEFT_REST_PX)}
        harness.hold(0.4)
        harness.swipe("a", -AMP_X_PX, 0.3)
        harness.hold(0.3)
        self.assertEqual(harness.events, ["left"])
        self._remove(harness, "a", 2.5)                     # 유예 초과 — 확정 손이 떠났다
        harness.selector.hands["b"] = list(RIGHT_REST_PX)   # 이제 다음 사용자가 슬롯을 받는다
        harness.hold(0.4)
        harness.swipe("b", -AMP_X_PX, 0.3)
        harness.hold(0.3)
        self.assertEqual(harness.events, ["left", "left"])


class LazySignalCostTest(unittest.TestCase):
    """판정하지 않는 손에는 손모양 계산 비용을 쓰지 않는다.

    ★2026-08-04 실기(사용자 보고 "추론 FPS는 확실히 낮아졌다"): 두 손 판정을
    넣으면서 게이트 통과 손 **전부**에 finger_states + classify_hand_shape를 매
    프레임 돌렸다. 판정 손 하나만 계산하던 것이 max_num_hands(4)까지 늘어난 것이다.
    """

    def test_locked_hand_costs_one_signal_per_frame(self):
        harness = ArbiterHarness()
        harness.selector.hands = {
            "a": list(LEFT_REST_PX), "b": list(RIGHT_REST_PX),
            "c": [0.5 * FRAME_WIDTH_PX, 0.50 * FRAME_WIDTH_PX],
        }
        harness.hold(0.3)
        harness.swipe("a", -AMP_X_PX, 0.3)     # a가 이긴다
        harness.hold(0.2)
        self.assertEqual(harness.events, ["left"])

        harness.selector.signal_calls = 0
        frames = 30
        harness.hold(frames / FPS)
        # 확정 뒤에는 손이 셋 보여도 **확정 손 하나만** 계산해야 한다
        self.assertEqual(harness.selector.signal_calls, frames)

    def test_uncommitted_state_costs_at_most_one_signal_per_frame(self):
        # ★2026-08-07 MAX_SLOTS 2→1 — 슬롯 상한이 1이라 손이 4개 보여도
        # 계산은 프레임당 최대 1건(구 "경쟁 중 2건"은 폐기)
        harness = ArbiterHarness()
        harness.selector.hands = {
            "a": list(LEFT_REST_PX), "b": list(RIGHT_REST_PX),
            "c": [0.45 * FRAME_WIDTH_PX, 0.50 * FRAME_WIDTH_PX],
            "d": [0.55 * FRAME_WIDTH_PX, 0.55 * FRAME_WIDTH_PX],
        }
        frames = 30
        harness.hold(frames / FPS)
        self.assertEqual(harness.events, [])
        self.assertLessEqual(harness.selector.signal_calls, frames)


class SlotIdentityTest(unittest.TestCase):
    """슬롯 정체성은 **위치 연속성**으로만 잇는다 — 라벨은 안 쓴다.

    07-31에 좌/우 라벨을 걷어낸 이유(handedness 불안정 → 배구 토스)를 여기서
    다시 들이지 않기 위해. 라벨이 뒤집혀도 슬롯이 안 섞여야 한다.
    """

    def test_label_flip_does_not_swap_slots(self):
        harness = ArbiterHarness()
        harness.selector.hands = {"a": list(LEFT_REST_PX), "b": list(RIGHT_REST_PX)}
        harness.hold(0.3)
        # 라벨(user_side)은 FakeSelector가 항상 "left"로 준다 — 즉 두 손이 같은
        # 라벨이어도 위치로 갈려야 한다. 한쪽만 쓸면 그쪽 이벤트만 나온다
        harness.swipe("a", -AMP_X_PX, 0.3)
        harness.hold(0.3)
        self.assertEqual(harness.events, ["left"])
        self.assertLess(harness.selector.adopted_center[0], LEFT_REST_PX[0])

    def test_third_hand_does_not_displace_existing_slots(self):
        # 슬롯 상한 1(2026-08-07) — 나중에 나타난 손은 앵커 게이트가 이미
        # 걸렀어야 하고, 여기서도 기존 슬롯을 밀어내지 않는다(쓰던 손이
        # 밀리면 판정이 끊긴다)
        harness = ArbiterHarness()
        harness.selector.hands = {"a": list(LEFT_REST_PX)}
        harness.hold(0.3)
        harness.selector.hands["c"] = [0.5 * FRAME_WIDTH_PX, 0.45 * FRAME_WIDTH_PX]
        harness.hold(0.3)
        harness.swipe("a", -AMP_X_PX, 0.3)
        harness.hold(0.3)
        self.assertEqual(harness.events, ["left"])


class BendFakeSelector:
    """hand_select 대역(자세 경쟁 — 10차) — 후보 손을 FakeHand 실물 구조로 준다."""

    def __init__(self):
        self.hands = {}              # 이름 -> (center_px, FakeHand)
        self.shoulder_frame = None   # (어깨선 y_px, 어깨너비 px) | None(앵커 없음)
        self.adopted_center = None

    def candidate_hands(self):
        return [(center, hand, SHOULDER_RATIO, SPAN_PX)
                for center, hand in self.hands.values()]

    def shoulder_line_y_ratio(self):
        return SHOULDER_LINE_Y

    def anchor_shoulder_frame(self):
        return self.shoulder_frame

    def adopt_tracked_hand(self, center_px):
        self.adopted_center = center_px
        return True


class BendHarness:
    """자세 경쟁 모드 중재기를 가짜 시계로 돌린다 — 실물 config."""

    def __init__(self):
        self.now_sec = 0.0
        self.selector = BendFakeSelector()
        self.arbiter = HandArbiter(load_config(CONFIG_PATH),
                                   clock=lambda: self.now_sec, judge_kind="bend")
        self.events = []

    def set_hand(self, name, angle_deg, pose, center_px):
        from tests.test_hand_bend import FakeHand, knuckle_at, make_landmarks
        hand = FakeHand(make_landmarks(knuckle_at(angle_deg, wrist=center_px),
                                       wrist=center_px), pose)
        self.selector.hands[name] = (center_px, hand)

    def hold(self, duration_sec):
        for _ in range(round(duration_sec * FPS)):
            self.now_sec += 1.0 / FPS
            event = self.arbiter.update(self.selector, FRAME_WIDTH_PX)
            if event is not None:
                self.events.append(event.class_name)


class PostureCompetitionTest(unittest.TestCase):
    """★10차 계약(사용자 확정 "잡히고 고정") — 자세 경쟁: 정지한 손이
    이동 없이 자세만으로 잡히고, 먼저 발화한 손이 잠긴다."""

    def setUp(self):
        self.h = BendHarness()

    def test_still_hand_locks_by_posture(self):
        # 이동 0 — 가만히 든 왼쪽 꺾임 손이 hold(0.2초)만으로 발화·잠금된다.
        # 종전(쓸기 경쟁 + 이동 획득 0.25 어깨너비)에서는 영원히 못 잡히던 손
        self.h.set_hand("a", -90, "open", LEFT_REST_PX)
        self.h.set_hand("b", 0, "uncertain", RIGHT_REST_PX)
        self.h.hold(0.5)
        self.assertEqual(self.h.events, ["left"])
        self.assertEqual(self.h.selector.adopted_center, LEFT_REST_PX)

    def test_loser_posture_ignored_after_lock(self):
        # 확정 뒤 진 손은 어떤 자세를 취해도 무시 (기존 잠금 계약 유지)
        self.h.set_hand("a", -90, "open", LEFT_REST_PX)
        self.h.set_hand("b", 0, "uncertain", RIGHT_REST_PX)
        self.h.hold(0.5)
        self.h.set_hand("b", 90, "open", RIGHT_REST_PX)
        self.h.hold(1.0)
        self.assertEqual(self.h.events, ["left"])

    def test_rest_zone_hand_cannot_lock(self):
        # ★휴식 존 게이트(compete_below_shoulder_max 1.6) — 늘어뜨린 펼친 손
        # (bend_down -> back 오발 후보)이 어깨선 아래 깊숙이 있는 동안은
        # 경쟁에서 발화 보류. 제스처 존으로 올라오면 정상 발화한다
        self.h.selector.shoulder_frame = (200.0, 100.0)    # 휴식 존: y > 360
        self.h.set_hand("a", 180, "open", (400.0, 500.0))  # 깊이 3.0 어깨너비
        self.h.hold(1.0)
        self.assertEqual(self.h.events, [])
        self.h.selector.hands.clear()
        self.h.set_hand("a", 180, "open", (400.0, 300.0))  # 깊이 1.0 — 허용 존
        self.h.hold(0.5)
        self.assertEqual(self.h.events, ["back"])

    def test_no_anchor_no_rest_gate(self):
        # 앵커(어깨) 미관측이면 게이트 없음 — 인식 우선 폴백(이 판의 관행)
        self.h.selector.shoulder_frame = None
        self.h.set_hand("a", 180, "open", (400.0, 500.0))
        self.h.hold(0.5)
        self.assertEqual(self.h.events, ["back"])

    def test_locked_hand_fires_in_rest_zone(self):
        # 잠금 뒤에는 게이트 없음 — 확정 손이 아래로 내려가 꺾어도 안 잘린다
        # (hand_select 추적 면제와 같은 정신). 슬롯 이음 반경(1.5x80=120px)
        # 안에서 조금씩 내린다 — 같은 슬롯(같은 판정기)이 이어져야 한다
        self.h.selector.shoulder_frame = (200.0, 100.0)
        self.h.set_hand("a", -90, "open", (400.0, 300.0))  # 허용 존에서 잠금
        self.h.hold(0.5)
        self.assertEqual(self.h.events, ["left"])
        for y_px in (380.0, 460.0, 520.0):                 # 휴식 존 깊이로 하강
            self.h.set_hand("a", 90, "open", (400.0, y_px))
            self.h.hold(0.3)
        self.assertEqual(self.h.events, ["left", "right"])


if __name__ == "__main__":
    unittest.main()
