"""eyebrow.py 미간 기준점의 눈 깜빡임 흔들림 완화 테스트 — 2026-08-28 신설.

연구실 키오스크에서 수직 민감도를 6.0까지 올린 뒤 "눈 깜빡이면 커서가 아래로
튄다"는 실기 보고로 _RollingMedianPoint(중앙값 기반 이상치 제거)를 추가했다.
_median()의 원칙(head_tracker.py 설명 참고 — "눈 깜빡임 같은 순간적 이상치는
평균이 아니라 중앙값으로 거른다")을 캘리브레이션이 아니라 매 프레임 흐르는
신호에 적용한 것이라, 그 핵심 성질(이상치 1프레임은 완전히 무시된다)만
검증한다.

★그런데 중앙값 필터만으로는 부족하다는 재보고가 왔다 — "여전히 눈감을때
커서가 내려가더라". 두 가지가 겹친 문제였다:

  ① 중앙값은 창의 절반 이상이 오염되면 못 거른다 — 눈을 감고 있는 동안
     내내(수백 ms) 낮게 잡히는 건 "이상치"가 아니라 "다수"가 되기 때문이다.
  ② (더 근본적) 점 하나만 얼려도 소용없었다 — HeadTracker.update()가
     안구간거리(_smoothed_dist_px)를 cursor_point_fn과는 별도로 face에서
     직접 다시 읽는데, eyebrow.py는 이 분모로 나눠서 dx/dy를 구하므로
     분자(미간 좌표)를 얼려도 분모가 흔들리면 결과가 같이 흔들린다.

그래서 _BlinkGate로 재설계했다 — 눈을 감은 동안은 head_tracker.update()
호출 자체를 건너뛴다(_process_one_frame). 이 테스트는 _BlinkGate가 실제
깜빡임(지속형)만 잡아내고, 사람마다 다른 평상시 eyeBlink 기준선에도
오탐하지 않는지를 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eyebrow import (   # noqa: E402
    EYE_BLINK_FREEZE_MARGIN, EYE_BLINK_MIN_HOLD_SEC, EYE_BLINK_RELEASE_MARGIN,
    LMK_LEFT_EYE_OUTER, LMK_RIGHT_EYE_OUTER, _BlinkGate, _RollingMedianPoint,
)


class _FakeClock:
    """테스트에서 시간을 직접 조작하기 위한 가짜 시계 — time.sleep 없이
    EYE_BLINK_MIN_HOLD_SEC 같은 시간 조건을 검증한다."""

    def __init__(self):
        self.now_sec = 0.0

    def __call__(self):
        return self.now_sec

    def advance(self, delta_sec):
        self.now_sec += delta_sec


def _default_gate(clock=None):
    return _BlinkGate(EYE_BLINK_FREEZE_MARGIN, EYE_BLINK_RELEASE_MARGIN,
                      EYE_BLINK_MIN_HOLD_SEC, clock=clock or time.monotonic)


class _FakeFace:
    """_glabella_point가 요구하는 최소 인터페이스만 흉내 낸 가짜 얼굴 결과."""

    def __init__(self, left_px, right_px, jaw=0.0, eye_blink_left=0.0, eye_blink_right=0.0):
        self._points = {LMK_LEFT_EYE_OUTER: left_px, LMK_RIGHT_EYE_OUTER: right_px}
        self._blend = {"jawOpen": jaw, "eyeBlinkLeft": eye_blink_left,
                       "eyeBlinkRight": eye_blink_right}

    def landmark_px(self, index):
        return self._points[index]

    def blendshape(self, name, default=0.0):
        return self._blend.get(name, default)


class RollingMedianPointTest(unittest.TestCase):
    def test_single_outlier_frame_is_fully_rejected(self):
        """눈 깜빡임처럼 1프레임만 튀는 잡음은 중앙값에 전혀 안 섞여야 한다."""
        f = _RollingMedianPoint(window=5)
        steady = (100.0, 200.0)
        for _ in range(4):
            out = f.update(steady)
        # 이 시점에 창은 [steady, steady, steady, steady] — 이상치 삽입 전
        self.assertEqual(out, steady)

        # 눈 깜빡임 한 프레임 — 세로로 크게 튄 값
        blink_outlier = (100.0, 260.0)
        out = f.update(blink_outlier)
        # 창은 이제 [steady x4, outlier] — 중앙값은 여전히 steady여야 한다
        self.assertEqual(out, steady, "이상치 1프레임이 중앙값에 섞여 나왔다")

    def test_sustained_movement_is_tracked(self):
        """실제로 계속 이동하면(이상치가 아니라 진짜 이동) 결국 따라가야 한다.

        지연 없이 즉시 반영되길 기대하는 게 아니라 — 창(5프레임) 만큼
        머문 뒤에는 새 위치의 중앙값이 나와야 한다는 뜻이다.
        """
        f = _RollingMedianPoint(window=5)
        for _ in range(10):
            out = f.update((300.0, 400.0))
        self.assertEqual(out, (300.0, 400.0))

    def test_window_size_is_respected(self):
        """창보다 오래된 표본은 더 이상 중앙값에 영향을 주지 않는다."""
        f = _RollingMedianPoint(window=3)
        f.update((0.0, 0.0))
        f.update((0.0, 0.0))
        out = f.update((0.0, 0.0))
        self.assertEqual(out, (0.0, 0.0))
        # 창(3)을 다 새 값으로 밀어내면 옛 값의 흔적이 없어야 한다
        for _ in range(3):
            out = f.update((10.0, 20.0))
        self.assertEqual(out, (10.0, 20.0))


class BlinkGateTest(unittest.TestCase):
    """_BlinkGate — head_tracker.update() 호출 여부를 결정하는 판단부."""

    def test_starts_open_never_blinking_before_baseline(self):
        """기준선을 아직 못 잡았으면(첫 프레임) 무조건 '깜빡임 아님'이어야
        한다 — 안 그러면 시작하자마자 얼어붙을 수 있다."""
        gate = _default_gate()
        face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.9, eye_blink_right=0.9)
        self.assertFalse(gate.is_blinking(face))

    def test_sustained_high_score_is_detected_as_blinking(self):
        """평상시 낮다가 확 튀어서 지속되면(진짜 깜빡임) 감지해야 한다."""
        clock = _FakeClock()
        gate = _default_gate(clock)
        open_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.1, eye_blink_right=0.1)
        for _ in range(10):
            self.assertFalse(gate.is_blinking(open_face))
            clock.advance(0.033)

        closed_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.9, eye_blink_right=0.9)
        for _ in range(8):   # 여러 프레임 지속 — 중앙값이 못 거르던 바로 그 상황
            self.assertTrue(gate.is_blinking(closed_face))
            clock.advance(0.033)

    def test_resumes_normal_well_after_eyes_reopen(self):
        """깜빡임이 완전히 끝나고 최소 유지 시간도 지나면 '아님'으로
        돌아와야 한다."""
        clock = _FakeClock()
        gate = _default_gate(clock)
        open_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.1, eye_blink_right=0.1)
        closed_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.9, eye_blink_right=0.9)
        for _ in range(10):
            gate.is_blinking(open_face)
            clock.advance(0.033)
        for _ in range(8):
            gate.is_blinking(closed_face)
            clock.advance(0.033)
        clock.advance(EYE_BLINK_MIN_HOLD_SEC + 0.5)   # 최소 유지 시간을 확실히 지난다
        self.assertFalse(gate.is_blinking(open_face))

    def test_high_baseline_user_does_not_false_trigger(self):
        """평상시 eyeBlink 점수가 원래 높은 사람이라도(고정 임계값이었다면
        오탐 위험) 자기 기준선 대비 마진을 안 넘으면 깜빡임으로 안 잡혀야
        한다 — head_tracker.py의 home 판정과 같은 이유(사람마다 0.1~0.6로
        편차가 크다)."""
        clock = _FakeClock()
        gate = _default_gate(clock)
        high_baseline_face = _FakeFace((0, 0), (0, 0),
                                       eye_blink_left=0.5, eye_blink_right=0.5)
        for _ in range(10):
            self.assertFalse(gate.is_blinking(high_baseline_face))
            clock.advance(0.033)

        slightly_higher_face = _FakeFace((0, 0), (0, 0),
                                         eye_blink_left=0.52, eye_blink_right=0.52)
        for _ in range(5):
            self.assertFalse(gate.is_blinking(slightly_higher_face),
                             "평상시보다 살짝 높을 뿐인데 깜빡임으로 오탐했다")
            clock.advance(0.033)

    def test_release_ramp_stays_frozen_until_fully_open(self):
        """★2026-08-28 재보고로 추가 — "눈감고 뜨면 커서 움직이는건 여전해".

        진입 문턱 하나만으론 눈이 다시 뜨이는 램프 구간(점수가 서서히
        내려가지만 아직 진입 문턱보다는 낮은)에서 너무 일찍 풀려 흔들림이
        샜다. 해제는 더 낮은 EYE_BLINK_RELEASE_MARGIN까지 내려와야 풀려야
        한다 — 그 중간 어중간한 점수에서는 계속 얼어 있어야 한다.
        """
        clock = _FakeClock()
        gate = _default_gate(clock)
        open_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.1, eye_blink_right=0.1)
        for _ in range(10):
            gate.is_blinking(open_face)
            clock.advance(0.033)

        closed_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.9, eye_blink_right=0.9)
        gate.is_blinking(closed_face)
        clock.advance(EYE_BLINK_MIN_HOLD_SEC + 0.05)   # 최소 유지 시간은 지나게 한다

        # 눈이 완전히 안 뜨이고 절반쯤 뜬 상태 — 진입 문턱(0.1+0.15=0.25)보다는
        # 낮지만 해제 문턱(0.1+0.05=0.15)보다는 아직 높다
        half_open_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.2, eye_blink_right=0.2)
        self.assertTrue(gate.is_blinking(half_open_face),
                        "덜 뜬 상태(램프 구간)에서 너무 일찍 풀렸다")

        # 이제 해제 문턱 아래로 확실히 내려온다
        fully_open_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.11, eye_blink_right=0.11)
        self.assertFalse(gate.is_blinking(fully_open_face),
                         "완전히 떴는데도 계속 얼어 있다")

    def test_brief_flicker_below_entry_still_holds_minimum(self):
        """최소 유지 시간 안에는(램프 상승 구간에서 점수가 잠깐 흔들려도)
        깜빡임 판정이 풀리면 안 된다."""
        clock = _FakeClock()
        gate = _default_gate(clock)
        open_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.1, eye_blink_right=0.1)
        for _ in range(10):
            gate.is_blinking(open_face)
            clock.advance(0.033)

        closed_face = _FakeFace((0, 0), (0, 0), eye_blink_left=0.9, eye_blink_right=0.9)
        gate.is_blinking(closed_face)
        clock.advance(EYE_BLINK_MIN_HOLD_SEC / 2)   # 최소 유지 시간의 절반만 지남

        self.assertTrue(gate.is_blinking(open_face),
                        "최소 유지 시간이 끝나기 전인데 풀렸다")


if __name__ == "__main__":
    unittest.main()
