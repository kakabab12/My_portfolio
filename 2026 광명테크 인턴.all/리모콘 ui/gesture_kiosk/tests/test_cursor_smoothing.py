"""커서 평활(1€ 필터·거리 적응) 단위 테스트 — 2026-08-27 신설.

forehead.py가 새로 쓰는 두 가지를 검증한다:
  · 1€ 필터 (Casiez et al., ACM CHI 2012) — 속도 적응형 저역통과
  · 거리 적응 평활 — 멀수록 자동으로 더 세게 평활

★가장 중요한 검사는 "head.py·eyebrow.py가 영향을 안 받는가"다. 둘 다 이
기능을 안 켜므로(설정 키 기본값 False) 예전과 완전히 같은 EMA 경로를 타야
한다 — 그게 깨지면 검증이 끝난 두 실행기가 조용히 바뀌는 셈이라 가장 위험하다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import math
import os
import random
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_tracker import (  # noqa: E402
    ONE_EURO_ADAPT_MAX_SCALE, ONE_EURO_ADAPT_MIN_SCALE, OneEuroFilter, _CursorMapper,
)

FRAME_DT_SEC = 1.0 / 30.0


def _mapper(**overrides):
    """실사용에 가까운 기본값으로 _CursorMapper를 만든다 (forehead.py 값 기준)."""
    kwargs = dict(
        calibration_window_sec=0.3, sensitivity_x=2.2, sensitivity_y=2.0,
        smoothing_alpha=0.26, distance_smoothing_alpha=0.08, max_offset_ratio=0.6,
        face_local=True, face_local_gain=2.0,
    )
    kwargs.update(overrides)
    return _CursorMapper(**kwargs)


def _run_still(mapper, interocular_px, frames=400, noise_px=0.6, seed=7):
    """가만히 있는 사용자 + 일정한 랜드마크 잡음 -> 커서 흔들림(화면 px).

    두 눈 위치는 고정하고 기준점에만 잡음을 준다. 커서 좌표는 0~1 비율이라
    1920x1080 화면에 곱해 사람이 읽을 수 있는 px로 바꾼다.
    """
    rng = random.Random(seed)
    half = interocular_px / 2.0
    samples = []
    now_sec = 0.0
    for i in range(frames):
        now_sec += FRAME_DT_SEC
        point = (320 + rng.gauss(0, noise_px), 250 + rng.gauss(0, noise_px))
        x_ratio, y_ratio = mapper.update(
            point, (320 - half, 240), (320 + half, 240), now_sec)
        if x_ratio is not None and i > frames // 2:   # 캘리브레이션·수렴 구간 제외
            samples.append((x_ratio, y_ratio))
    if len(samples) < 20:
        return None
    sd_x_px = statistics.pstdev([s[0] for s in samples]) * 1920
    sd_y_px = statistics.pstdev([s[1] for s in samples]) * 1080
    return math.hypot(sd_x_px, sd_y_px)


class OneEuroFilterTest(unittest.TestCase):
    def test_constant_input_stays_constant(self):
        """상수를 넣으면 상수가 나온다 — 필터가 값을 끌고 다니면 커서가 흐른다."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        now_sec = 0.0
        for _ in range(50):
            now_sec += FRAME_DT_SEC
            self.assertAlmostEqual(f(0.5, now_sec), 0.5, places=9)

    def test_reduces_stationary_noise(self):
        """정지 상태 잡음을 눈에 띄게 줄인다 (1€ 필터의 존재 이유)."""
        rng = random.Random(0)
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        raw, filtered = [], []
        now_sec = 0.0
        for _ in range(200):
            now_sec += FRAME_DT_SEC
            value = 0.5 + rng.gauss(0, 0.01)
            raw.append(value)
            filtered.append(f(value, now_sec))
        self.assertLess(statistics.pstdev(filtered[50:]),
                        statistics.pstdev(raw[50:]) * 0.6)

    def test_tracks_fast_ramp_without_large_lag(self):
        """빠르게 움직이면 평활이 풀려 따라온다 — 안 그러면 커서가 질질 끌린다."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        now_sec, out = 0.0, None
        for i in range(60):
            now_sec += FRAME_DT_SEC
            out = f(i * 0.02, now_sec)
        self.assertLess(abs(out - 59 * 0.02), 0.10)   # 램프 끝값과 큰 차이 없어야

    def test_zero_dt_does_not_crash(self):
        """같은 시각에 두 번 불려도 0으로 나누지 않는다 (방어)."""
        f = OneEuroFilter()
        f(0.5, 1.0)
        self.assertTrue(math.isfinite(f(0.6, 1.0)))

    def test_prime_sets_state_without_jump(self):
        """prime 직후 같은 값을 넣으면 그 값이 그대로 나온다 — 캘리브레이션
        직후 커서가 순간이동하지 않게 하는 장치."""
        f = OneEuroFilter()
        f.prime(0.5, 10.0)
        self.assertAlmostEqual(f(0.5, 10.0 + FRAME_DT_SEC), 0.5, places=9)


class DistanceAdaptiveSmoothingTest(unittest.TestCase):
    """거리 적응 평활 — 근거는 8/26 실측 "커서 흔들림 x 안구간거리 = 일정"."""

    def test_noise_is_inversely_proportional_to_distance(self):
        """전제 확인 — 적응을 끄면 흔들림이 1/거리에 비례해야 한다.

        이 전제가 성립해야 거리로 보정한다는 발상 자체가 성립한다.
        (실측: 90px->2.0px, 60px->3.0px, 40px->4.5px — 곱하면 전부 180)
        """
        products = []
        for dist_px in (90, 60, 40):
            shake = _run_still(_mapper(one_euro_enabled=True), dist_px)
            self.assertIsNotNone(shake)
            products.append(shake * dist_px)
        # 곱이 일정한가 — 최대·최소 차이가 평균의 5% 안
        self.assertLess((max(products) - min(products)) / statistics.fmean(products), 0.05)

    def test_far_user_gets_less_shake(self):
        """멀리 선 사용자는 적응을 켜면 확실히 덜 떨린다."""
        for dist_px in (40, 30):
            off = _run_still(_mapper(one_euro_enabled=True), dist_px)
            on = _run_still(
                _mapper(one_euro_enabled=True, one_euro_distance_adaptive=True,
                        one_euro_reference_dist_px=60.0), dist_px)
            self.assertLess(on, off * 0.9, f"안구간거리 {dist_px}px에서 개선 없음")

    def test_near_user_is_not_made_worse(self):
        """★기준 거리보다 가까우면 지금과 완전히 같아야 한다 (상한 1.0의 목적).

        상한을 1.0으로 묶은 이유가 이것이다 — 배율이 1을 넘으면 평활이 풀려
        가까운 사용자가 오히려 더 떨리게 된다(시뮬레이션에서 +21% 악화 확인).
        """
        for dist_px in (60, 90, 120):
            off = _run_still(_mapper(one_euro_enabled=True), dist_px)
            on = _run_still(
                _mapper(one_euro_enabled=True, one_euro_distance_adaptive=True,
                        one_euro_reference_dist_px=60.0), dist_px)
            self.assertAlmostEqual(on, off, delta=off * 0.02,
                                   msg=f"안구간거리 {dist_px}px에서 동작이 바뀌었다")

    def test_scale_is_clamped(self):
        """극단적인 거리에서도 배율이 상·하한 안에 묶인다."""
        mapper = _mapper(one_euro_enabled=True, one_euro_distance_adaptive=True,
                         one_euro_min_cutoff=1.0, one_euro_reference_dist_px=60.0)
        for dist_px, expected in ((5.0, ONE_EURO_ADAPT_MIN_SCALE),
                                  (10000.0, ONE_EURO_ADAPT_MAX_SCALE)):
            mapper._smoothed_dist_px = dist_px
            mapper._apply_distance_adaptive_cutoff()
            self.assertAlmostEqual(mapper._one_euro_x.min_cutoff, expected, places=9)


class ArcCompensationTest(unittest.TestCase):
    """가로 이동 시 세로가 활처럼 휘는 것(뒤집힌 U) 보정 — 원인은 코의 원근 왜곡."""

    # 훑는 폭(px) — 작게 잡는다. 크게 잡으면 커서가 화면 밖으로 포화(클램프)돼
    # 곡률이 그 지점부터 평평해져서 보정 효과를 잴 수가 없다.
    # 안구간거리 60px·gain 2.0·감도 2.2 기준으로 ±4px면 커서가 화면 안에 머문다
    SWEEP_HALF_PX = 4.0

    @classmethod
    def _sweep(cls, mapper, curvature, steps=41):
        """고개를 좌우로만 훑는다. curvature>0이면 기준점 자체가 2차로 휘게 만든다
        (실기에서 관측되는 원근 왜곡을 흉내 낸 것).

        ★훑기 전에 중앙에서 충분히 머문다 — 안 그러면 캘리브레이션(중앙값)이
        훑는 도중의 좌표로 잡혀 중심이 한쪽으로 밀린다.
        """
        half = 30.0
        eyes = ((320 - half, 240), (320 + half, 240))
        now_sec = 0.0
        for _ in range(30):   # 캘리브레이션 창(0.3초)보다 넉넉히
            now_sec += FRAME_DT_SEC
            mapper.update((320.0, 250.0), *eyes, now_sec)

        out = []
        for i in range(steps):
            offset_px = -cls.SWEEP_HALF_PX + 2 * cls.SWEEP_HALF_PX * i / (steps - 1)
            bow_px = curvature * (offset_px / cls.SWEEP_HALF_PX) ** 2
            # ★한 위치에서 EMA가 수렴할 때까지 머문다. 움직이면서 재면 커서가
            # 목표에 아직 못 미친 값이 잡혀(평활 지연), 그 값으로 계수를 유도하면
            # 분모가 작아져 보정이 과해진다 — 실제로 3.6배 과보정이 났었다
            for _ in range(20):
                now_sec += FRAME_DT_SEC
                x_ratio, y_ratio = mapper.update(
                    (320 + offset_px, 250 + bow_px), *eyes, now_sec)
            if x_ratio is not None:
                out.append((x_ratio, y_ratio))
        return out

    @staticmethod
    def _curvature_of(points):
        """궤적에서 2차항 크기를 뽑는다 — 양끝 평균과 가운데의 차이로 근사."""
        if len(points) < 5:
            return None
        mid = points[len(points) // 2][1]
        edge = (points[0][1] + points[-1][1]) / 2.0
        return edge - mid

    def test_uncompensated_trajectory_bows(self):
        """보정을 끄면 실제로 휜다 — 이 전제가 있어야 보정이 의미를 갖는다."""
        points = self._sweep(_mapper(one_euro_enabled=False), curvature=6.0)
        self.assertIsNotNone(points)
        self.assertGreater(abs(self._curvature_of(points)), 0.01)

    def test_compensation_flattens_trajectory(self):
        """관측된 휨에서 계수를 유도해 넣으면 휨이 거의 사라진다.

        보정은 `세로 += 계수 x 가로offset^2` 이므로, 양끝(가로offset = ox)에서
        생기는 변화량은 계수 x ox^2 이고 가운데(offset 0)에서는 0이다.
        따라서 관측된 휨 bow를 상쇄하려면  계수 = -bow / ox^2  이면 된다 —
        추측이 아니라 유도된 값이다.
        """
        bowed = self._sweep(_mapper(one_euro_enabled=False), curvature=6.0)
        bow = self._curvature_of(bowed)
        edge_offset_x = bowed[-1][0] - 0.5          # 훑은 양끝의 가로 offset
        self.assertGreater(abs(edge_offset_x), 0.05, "가로로 충분히 훑지 못했다")
        coefficient = -bow / (edge_offset_x ** 2)

        fixed = self._sweep(
            _mapper(one_euro_enabled=False, arc_compensation=coefficient),
            curvature=6.0)
        self.assertLess(abs(self._curvature_of(fixed)), abs(bow) * 0.1)

    def test_zero_compensation_changes_nothing(self):
        """기본값 0이면 예전과 완전히 같아야 한다 (head/eyebrow 보호)."""
        a = self._sweep(_mapper(one_euro_enabled=False), curvature=6.0)
        b = self._sweep(_mapper(one_euro_enabled=False, arc_compensation=0.0),
                        curvature=6.0)
        self.assertEqual(a, b)

    def test_compensation_does_not_run_away_past_the_clamp(self):
        """★2026-08-28 버그 재현 — 클램프 지점을 넘어 고개를 더 돌려도 보정이
        계속 커지면 안 된다.

        예전엔 클램프 전(raw) offset_x로 보정을 계산해서, 커서가 이미 화면
        가장자리에 닿은 뒤에도 고개를 더 돌리면 offset_x²이 한없이 커져
        offset_y가 보정만으로 자기 클램프에 부딪혔다 — 실사용에서 "화면
        양쪽 끝에서 커서가 위로 확 올라간다"로 나타난 원인. 클램프된 값으로
        고치면, 클램프 지점을 넘어서는 어떤 dx를 넣어도 결과가 완전히 같아야
        한다(더 이상 커질 자리가 없으므로).
        """
        eyes = ((320 - 30, 240), (320 + 30, 240))

        def _settle(mapper, cursor_point, frames=200):
            now_sec = 0.0
            x_ratio = y_ratio = None
            for _ in range(frames):
                now_sec += FRAME_DT_SEC
                x_ratio, y_ratio = mapper.update(cursor_point, *eyes, now_sec)
            return x_ratio, y_ratio

        def _result_at(dx_px):
            mapper = _mapper(one_euro_enabled=False, sensitivity_x=2.2,
                             max_offset_ratio=0.5, arc_compensation=-0.3)
            _settle(mapper, (320.0, 250.0))   # 캘리브레이션 중앙 고정
            return _settle(mapper, (320.0 + dx_px, 250.0))

        # dx=10px, 30px 둘 다 raw offset_x가 이미 클램프(0.5)를 넘어(각각
        # 0.73, 2.2) 가로 위치(x_ratio)는 똑같이 화면 끝에 붙는다 — 그러니
        # 세로(y_ratio)도 완전히 같아야 한다. 수정 전 버그라면 raw offset_x를
        # 그대로 제곱해 old_offset_y가 -0.161 -> -0.5로 계속 커졌다(위 사전
        # 계산 참고) — 클램프된 값을 쓰면 둘 다 -0.075로 고정된다.
        at_10px = _result_at(10.0)
        at_30px = _result_at(30.0)
        self.assertEqual(at_10px, at_30px,
                         "클램프를 넘는 구간에서 보정이 계속 커지고 있다 — 회귀")


class SetTuningTest(unittest.TestCase):
    """_CursorMapper.set_tuning — 실시간 조절 UI(scripts/tuning_ui.py) 대응."""

    def test_updates_take_effect_on_next_update(self):
        """감도를 실행 중에 바꾸면 바로 다음 프레임부터 새 감도로 움직여야 한다."""
        mapper = _mapper(one_euro_enabled=False, sensitivity_x=1.0)
        eyes = ((320 - 30, 240), (320 + 30, 240))
        now_sec = 0.0
        for _ in range(20):   # 캘리브레이션
            now_sec += FRAME_DT_SEC
            mapper.update((320.0, 250.0), *eyes, now_sec)

        def _settle(dx_px):
            x = None
            nonlocal now_sec
            for _ in range(20):
                now_sec += FRAME_DT_SEC
                x, _y = mapper.update((320.0 + dx_px, 250.0), *eyes, now_sec)
            return x

        before = _settle(10.0)
        mapper.set_tuning(sensitivity_x=3.0)
        after = _settle(10.0)
        self.assertGreater(abs(after - 0.5), abs(before - 0.5),
                           "감도를 올렸는데 같은 입력에 대한 커서 이동량이 안 커졌다")

    def test_none_leaves_other_values_untouched(self):
        """일부 값만 넘기면 나머지는 그대로여야 한다 — UI가 슬라이더 하나만
        움직였을 때 다른 슬라이더 값까지 덩달아 바뀌면 안 된다."""
        mapper = _mapper(one_euro_enabled=False, sensitivity_x=1.0, sensitivity_y=2.0)
        mapper.set_tuning(sensitivity_x=5.0)
        self.assertEqual(mapper._sensitivity_x, 5.0)
        self.assertEqual(mapper._sensitivity_y, 2.0, "안 건드린 값이 바뀌었다")


class LerpReferenceTest(unittest.TestCase):
    """★렌더 속도를 올려도 커서 손맛이 안 바뀌는지 — 60fps 상향의 안전장치.

    보간 계수의 시간 기준을 렌더 주기에 묶어 두면, 렌더를 30->60으로 올릴 때
    같은 alpha가 두 배 자주 적용돼 커서가 두 배 빨리 붙는다. 실기로 맞춰 온
    감각이 통째로 어긋나므로 기준을 분리했다 — 그게 유지되는지 고정한다.
    """

    @staticmethod
    def _traj(render_fps, fixed_ref, duration=0.4):
        from forehead import (   # 지연 임포트 — 이 테스트에서만 필요
            LERP_REFERENCE_FPS, RENDER_LERP_ALPHA, _dt_adjusted_alpha)
        dt = 1.0 / render_fps
        ref = (1.0 / LERP_REFERENCE_FPS) if fixed_ref else dt
        pos, now_sec, out = 0.0, 0.0, {}
        while now_sec < duration - 1e-9:
            now_sec += dt
            pos += _dt_adjusted_alpha(RENDER_LERP_ALPHA, dt, ref) * (1.0 - pos)
            out[round(now_sec, 6)] = pos
        return out

    def test_trajectory_identical_across_render_rates(self):
        """30fps와 60fps가 같은 시각에 같은 위치에 있어야 한다."""
        t30 = self._traj(30, fixed_ref=True)
        t60 = self._traj(60, fixed_ref=True)
        common = [k for k in t30 if k in t60]
        self.assertGreater(len(common), 5)
        for k in common:
            self.assertAlmostEqual(t30[k], t60[k], places=12)

    def test_would_differ_without_decoupling(self):
        """기준을 분리하지 않으면 실제로 달라진다 — 위 검사가 헛돌지 않음을 보인다."""
        t30 = self._traj(30, fixed_ref=True)
        t60_bad = self._traj(60, fixed_ref=False)
        common = [k for k in t30 if k in t60_bad]
        worst = max(abs(t30[k] - t60_bad[k]) for k in common)
        self.assertGreater(worst, 0.1)


class DefaultsUnchangedTest(unittest.TestCase):
    """★head.py·eyebrow.py 보호 — 이 셋이 깨지면 검증 끝난 실행기가 조용히 바뀐다."""

    def test_one_euro_off_by_default(self):
        self.assertFalse(_mapper()._one_euro_enabled)

    def test_distance_adaptive_off_by_default(self):
        self.assertFalse(_mapper(one_euro_enabled=True)._one_euro_distance_adaptive)

    def test_ema_path_is_bit_identical_when_disabled(self):
        """기능을 끄면 예전 EMA 경로와 결과가 완전히 같아야 한다."""
        baseline = _run_still(_mapper(), 60)
        with_flags_off = _run_still(
            _mapper(one_euro_enabled=False, one_euro_distance_adaptive=False), 60)
        self.assertEqual(baseline, with_flags_off)


if __name__ == "__main__":
    unittest.main()
