"""실 config 제스처 시나리오 시뮬레이션 게이트 (작업내역서 §4.3 — 2026-07-20 영구화).

2026-08-03 개편 스펙: 손 모양(손 펼침/주먹/한 손가락) × 방향 — 이벤트
left/right/select/back/home/confirm (상하(top/bottom) 제거·ok→confirm,
아래 방향은 두 모양 다 정의 없음 = 무시).

규칙(§4.3·§5): ① 좌표는 물리적으로 연속 ② 진폭은 임계의 2배쯤(플릭만 1.2배)
③ 동작 전 정지 프레임 공급(콜드 스타트 — 없으면 궤적 시작점이 이동 중간이 돼
가짜 실패) ④ 판정값은 configs/config.yaml **실물**로 읽는다 (튜닝 회귀 감지)
⑤ y는 프레임 폭 정규화 — 720p 최대 0.5625, 범위 밖 좌표는 가짜 검증.

시나리오는 point_filter(One Euro) 켠 상태(현행 config)와 끈 상태 양쪽에서
전부 통과해야 한다 — 필터 지연이 판정을 깨지 않는 증명.
"""
import copy
import os
import sys
import unittest

SKIP_ROTOR_BRANCH = ("2026-08-05 로터 판(feat/rotor_remote): 주먹·한손가락 계층, "
                     "펼친손 아래, 쥐기 select 비활성 — 사용자 결정 "
                     "(gesture_filter.EVENT_BY_SHAPE 주석 참고)")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.gesture_filter import GestureFilter
from src.utils.config_loader import load_config

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml"
)

FPS = 30
SHOULDER_RATIO = 0.22   # 키오스크 표준 거리(fallback_ratio와 동일) — 임계 환산 자(尺)
FRAME_BOTTOM_Y = 0.5625
SHOULDER_LINE_Y = 0.20  # 어깨선 높이 — 휴식 존 상단 = 0.20 + 1.2×0.22 = 0.464 (화면 안)
REST = (0.5, 0.31)      # 손의 기본 위치: 가슴께 든 상태
HANG = (0.5, 0.52)      # 손을 낮게 내린 위치 — 휴식 존 안(0.464 아래), 화면 안

# 진폭 (§4.3: 임계의 2배쯤이 현실적) — 임계 x=0.55·0.22≈0.121, y=0.35·0.22≈0.077
AMP_X = 0.25
AMP_Y = 0.16
FLICK_X = 0.145         # 손목 플릭 — 임계의 1.2배 (콜드 스타트 필수 케이스)


class _Sim:
    """연속 좌표 시뮬레이터 — 가짜 시계로 GestureFilter에 합성 궤적을 주입한다.

    shape 속성이 현재 손 모양("open"/"fist"/"finger"/None=판별 불명)으로 매 프레임 전달된다.
    """

    def __init__(self, config):
        self._now_sec = 0.0
        self._dt_sec = 1.0 / FPS
        self._filter = GestureFilter(config, clock=lambda: self._now_sec)
        self.position = REST
        self.shape = "open"
        self.side = "left"      # 손 라벨 — 정보용(이벤트 hand_side)일 뿐 판정 무관
        self.events = []

    def _step(self, hand_signal):
        self._now_sec += self._dt_sec
        event = self._filter.filter_signals(hand_signal, SHOULDER_RATIO, SHOULDER_LINE_Y)
        if event is not None:
            self.events.append(event.class_name)

    def feed(self, x, y):
        self.position = (x, y)
        self._step((self.shape, (x, y), self.side))

    def hold(self, duration_sec):
        for _ in range(round(duration_sec * FPS)):
            self.feed(*self.position)

    def move_by(self, dx, dy, duration_sec):
        """현재 위치에서 (dx, dy)만큼 등속 이동 — 프레임마다 연속 좌표."""
        steps = max(1, round(duration_sec * FPS))
        x0, y0 = self.position
        for i in range(1, steps + 1):
            self.feed(x0 + dx * i / steps, y0 + dy * i / steps)

    def drop(self, dx, dy, duration_sec):
        """추적점 소실 구간(모션 블러·화면 밖) — 손은 움직이지만 점은 전달 안 됨."""
        steps = max(1, round(duration_sec * FPS))
        x0, y0 = self.position
        for i in range(1, steps + 1):
            self.position = (x0 + dx * i / steps, y0 + dy * i / steps)
            self._step(None)


def _sims():
    """필터 켠 실물 config + 끈 config 두 시뮬레이터 — 양쪽 다 통과해야 한다."""
    config = load_config(CONFIG_PATH)
    config_no_filter = copy.deepcopy(config)
    config_no_filter["gestures"]["swipe"].pop("point_filter", None)
    return [("filter_on", _Sim(config)), ("filter_off", _Sim(config_no_filter))]


class SwipeScenarioTest(unittest.TestCase):
    def _run(self, scenario, expected_events):
        for label, sim in _sims():
            scenario(sim)
            self.assertEqual(sim.events, expected_events, f"[{label}] 이벤트 불일치")

    # ----- 손 펼침(탐색 계층 — 2026-08-03 개편) 4방향 -----

    def test_01_open_right(self):
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
        self._run(scenario, ["right"])

    def test_02_open_left(self):
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(-AMP_X, 0, 0.3)
        self._run(scenario, ["left"])

    def test_03_open_up_fires_up(self):
        # ★2026-08-04 손 펼침 4방향화 — 08-03에 비웠던 위 칸이 up으로 채워졌다.
        # 여기서는 손이 이미 가슴께(REST)에 안정돼 있다 — 들어올리기가 아니다
        def scenario(sim):
            sim.hold(0.8)
            sim.move_by(0, -AMP_Y, 0.3)
        self._run(scenario, ["up"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_03b_grab_once_and_hold_is_select(self):
        # ★2026-08-04 사용자 결정으로 2회 → 1회. 대신 쥔 뒤 **그 자리에 머물러야**
        # 한다(dwell_sec) — 쥐자마자 쓸면 명령이지 select가 아니다(시나리오 03c)
        def scenario(sim):
            sim.hold(0.4)
            sim.shape = "open"
            sim.hold(0.4)
            sim.shape = "fist"
            sim.hold(0.9)       # 쥐고 정지 — 유예(0.4초)를 넘긴다
        self._run(scenario, ["select"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_03c_grab_then_swipe_loses_command_to_select(self):
        # ⚠2026-08-04 사용자 결정으로 **뒤집힌 시나리오** — dwell_sec 0.4 → 0
        # ("주먹 접히는거 인식되면 바로 보내면 된다"). 실기에서 select가 굼떴다:
        # 래치 전환 0.27초 + 유예 0.4 ≈ 0.67초라 체감 1초였고, 쥐었다 바로 펴면
        # 유예를 못 채워 통째로 씹혔다.
        #
        # 유예가 이 경로의 **유일한 방어**였다. 없애면 손바닥에서 주먹으로 바꾸는
        # 순간 select가 먼저 나가고, cooldown_sec(0.5)이 뒤이은 쓸기를 삼켜
        # **명령이 사라진다** — 붙어 나오는 게 아니라 대체된다. 감수한 결정이다.
        # ※한 손가락 → 주먹으로 명령을 내면 select가 안 나간다(_has_seen_open) —
        #   실사용에서 거슬리면 dwell_sec 0.2~0.25가 첫 후보다 (docs/TODO.md)
        # "왜 back이 안 나가지?"를 다시 파기 전에 이 주석을 볼 것
        def scenario(sim):
            sim.hold(0.4)
            sim.shape = "open"
            sim.hold(0.4)
            sim.shape = "fist"
            # 래치가 주먹으로 넘어가는 데 8프레임(≈0.27초)이 필요하다 — 그보다
            # 짧게 잡으면 모양이 아직 펼침이라 left가 나가고 이 시나리오가 무의미해진다
            sim.hold(0.32)
            sim.move_by(-AMP_X, 0, 0.3)     # 왼쪽 = back — 쿨다운에 삼켜진다
            sim.hold(0.5)
        self._run(scenario, ["select"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_04_open_down_fires_down(self):
        # ★2026-08-04 아래 방향 신설 — 시나리오 22(팔 내리기)와 대조쌍.
        # 손을 든 위치에서 시작해 **가드 경계 위에서 끝나는** 아래 획만 down이다.
        # 가슴께에서 시작하면 끝점이 팔 처진 위치와 겹쳐 구분이 불가능하다 —
        # down의 쓸 수 있는 띠가 위쪽에 있다는 뜻이고, 실기에서 확인할 항목이다
        def scenario(sim):
            sim.position = (0.5, 0.22)      # 얼굴께로 든 손
            sim.hold(0.6)
            sim.move_by(0, AMP_Y, 0.3)      # -> 0.38 (가드 경계 0.464 위)
            sim.hold(0.4)                   # 확정 유예(0.25초) — down은 끝점을 본다
        self._run(scenario, ["down"])

    # ----- 주먹(명령 계층) -----

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_05_fist_left_is_back(self):
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)
            sim.move_by(-AMP_X, 0, 0.3)
        self._run(scenario, ["back"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_06_fist_right_is_confirm(self):
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
        self._run(scenario, ["confirm"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_07_fist_up_is_home(self):
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.8)                   # 등장 유예 경과 (test_03 주석 참고)
            sim.move_by(0, -AMP_Y, 0.3)
        self._run(scenario, ["home"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_08_fist_down_then_return_does_not_fire_home(self):
        # ★2026-08-04: 주먹+아래가 temp_fist_bottom으로 채워졌다. 바뀐 뒤에도
        # **복귀(위)가 home으로 오발되지 않는 것**이 이 시나리오의 본론이다 —
        # 실제로 움직인 팔은 반드시 돌아오기 때문 (삼킴 + 휴식 존 이중 방어)
        def scenario(sim):
            sim.shape = "fist"
            sim.position = (0.5, 0.22)      # 든 손에서 시작 (test_04 주석 참고)
            sim.hold(0.6)
            sim.move_by(0, AMP_Y, 0.3)      # 아래 — temp_fist_bottom
            sim.hold(0.4)                   # 확정 유예
            sim.move_by(0, -AMP_Y, 0.3)     # 복귀(위) — home 금지
            sim.hold(0.5)
        self._run(scenario, ["temp_fist_bottom"])

    # ----- 손 모양 다수결 · 전환 -----

    def test_09_shape_blur_gaps_still_fire(self):
        # 빠른 이동 중 블러로 판별이 절반 끊겨도(None = 기권) 유효 판별 다수결로 확정
        def scenario(sim):
            sim.hold(0.5)
            x0, y0 = sim.position
            steps = 9
            for i in range(1, steps + 1):
                sim.shape = "open" if i % 2 else None
                sim.feed(x0 + AMP_X * i / steps, y0)
            sim.shape = "open"
            sim.hold(0.3)
        self._run(scenario, ["right"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_10_shape_change_mid_stroke_uses_majority(self):
        # 이동 중 주먹→한 손가락 전환(모양 흔들림) — 궤적은 연속, 다수 모양이 정한다
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)                   # 정지 중 주먹 표가 쌓인다 (창 0.8초 내 유효)
            sim.move_by(AMP_X * 0.6, 0, 0.2)
            sim.shape = "open"            # 끝부분 잠깐 판별 흔들림
            sim.move_by(AMP_X * 0.4, 0, 0.1)
            sim.hold(0.2)
        self._run(scenario, ["confirm"])

    # ----- 복귀 삼킴 · 의도적 반대 동작 -----

    def test_11_return_stroke_is_swallowed(self):
        # 우 쓸기 후 손 되돌리기 — 직전 획 끝을 경유하는 반대 방향은 복귀로 삼킴
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
            sim.hold(1.0)                   # 쿨다운 경과 (손은 획 끝에 머무름)
            sim.move_by(-AMP_X, 0, 0.3)     # 복귀 — 획 끝에서 시작하므로 삼킴
            sim.hold(0.3)
        self._run(scenario, ["right"])

    def test_11b_return_during_cooldown_is_swallowed(self):
        # ★2026-08-04 쿨다운 인하(0.9→0.5)의 전제: 복귀가 쿨다운 **안에 다 들어가지
        # 않는** 상황. 종전엔 쿨다운 중 궤적을 안 쌓아, 쿨다운이 끝나는 순간 궤적이
        # 복귀 도중부터 시작해 "직전 획 끝을 지나왔는가"라는 삼킴 근거가 사라졌고
        # 되돌아오는 팔이 좌 이벤트로 튀었다. 이제 궤적을 쌓아 증거를 남긴다 —
        # 쉬지 않고 곧바로 되돌리는, 실제로 가장 흔한 동작이다.
        # (쿨다운의 다른 역할 — 한 획이 두 번 확정되지 않게 — 은 시나리오 15가
        #  지킨다: 0.3으로 내리면 거기서 'right'가 두 번 나온다)
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)      # 우 — 확정
            sim.move_by(-AMP_X, 0, 0.8)     # 쉼 없이 천천히 복귀 (쿨다운 0.5를 넘긴다)
            sim.hold(0.3)
        self._run(scenario, ["right"])

    def test_12_intentional_left_after_right_passes(self):
        # 우 다음 의도적 좌 — 쿨다운 중 중앙 복귀 후의 좌 쓸기는 획 끝을 경유하지
        # 않으므로(위치 조건) 삼킴 창 안이어도 정상 발화
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
            sim.move_by(-AMP_X, 0, 0.4)     # 쿨다운 중 중앙 복귀 (판정 없음)
            sim.hold(0.5)                   # 쿨다운 마저 경과
            sim.move_by(-AMP_X, 0, 0.3)     # 중앙에서 시작한 의도적 좌
        self._run(scenario, ["right", "left"])

    # ----- 플릭 · 블러 (개인별 동작 크기·빠른 동작) -----

    def test_13_forearm_flick_cold_start(self):
        # 손목 플릭 — 임계 1.2배 진폭. 정지 프레임 선공급 필수(§5 콜드 스타트 함정)
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(FLICK_X, 0, 0.25)
            sim.hold(0.3)                   # 필터 수렴·창 내 판정 여유
        self._run(scenario, ["right"])

    def test_14_fast_flick_recognized(self):
        # 아주 빠른 플릭(0.10초 = 3프레임) — 플릭 후 정지 프레임이 궤적을 채워 인식된다
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.10)
            sim.hold(0.4)
        self._run(scenario, ["right"])

    def test_15_blur_dropout_mid_swipe_survives(self):
        # 빠른 쓸기 중 모션 블러로 2프레임 소실 — 소실 유예(dropout_grace_sec)가
        # 궤적을 유지해 인식된다 (유예 없인 리셋 → 인식 실패, 2026-07-20 실증)
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(AMP_X * 0.4, 0, 0.12)
            sim.drop(AMP_X * 0.2, 0, 0.07)   # 블러 구간 — 손은 전진, 점은 소실
            sim.move_by(AMP_X * 0.4, 0, 0.12)
            sim.hold(0.4)
        self._run(scenario, ["right"])

    # ----- 들어올리기 게이트 (select·home 오발 방지) -----

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_16_arm_raise_itself_is_not_an_event(self):
        # 손을 내리고 있다가 들어올리는 동작이 select로 오발되면 안 된다 — 휴식 존
        # 이력 게이트 (2026-07-20). 이어지는 아래 쓸기도 07-29 개편으로 정의 없음
        def scenario(sim):
            sim.position = HANG             # 손 축 처진 상태(휴식 존 안)에서 시작
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.4)   # 들어올리기 — select 금지
            sim.hold(0.3)
            sim.move_by(0, AMP_Y, 0.3)               # 아래 — 정상 down (가드 없음)
            sim.hold(0.4)
        self._run(scenario, ["down"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_17_up_fires_after_settling_above_rest_zone(self):
        # 들어올린 뒤 유예(0.6초)를 넘겨 자세가 안정되면 위 스냅은 정상 발화.
        # 2026-08-03 개편으로 위 방향은 주먹(home)에만 남았다 — 게이트 검증은 동일
        def scenario(sim):
            sim.shape = "fist"
            sim.position = HANG
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.4)   # 들어올리기 — 무시
            sim.hold(0.8)                            # 유예(0.6) 경과 — 손 든 채 안정
            sim.move_by(0, -AMP_Y, 0.3)              # 위 스냅 = 의도적 home
        self._run(scenario, ["home"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_18_raise_then_immediate_down_fires_only_down(self):
        # 들어올리기 직후 곧바로 아래로 내리는 동작 — 들어올리기(select 금지)도
        # 아래(정의 없음)도 이벤트가 없어야 한다. 상승 꼬리 트림(RAISE_TRIM)은
        # 좌/우 쓸기 지연 방지용으로 유지 (test_19가 검증)
        def scenario(sim):
            sim.position = HANG
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.35)  # 들어올리기
            sim.move_by(0, AMP_Y, 0.3)               # 쉼 없이 바로 아래 — 정상 down
            sim.hold(0.4)
        self._run(scenario, ["down"])

    def test_19_diagonal_raise_then_left_swipe(self):
        # 우측으로 호를 그리는 들어올리기 직후 좌 쓸기 — 호의 수평 꼬리가 좌 이동을
        # 상쇄해 포커스가 의도대로 안 가던 실기 증상 (2026-07-20)
        def scenario(sim):
            sim.position = (0.62, HANG[1])
            sim.hold(0.5)
            sim.move_by(0.10, REST[1] - HANG[1], 0.35)  # 대각 들어올리기(우로 호)
            sim.move_by(-AMP_X, 0, 0.3)                 # 즉시 좌 쓸기
            sim.hold(0.4)
        self._run(scenario, ["left"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_20_close_range_appear_then_lower_fires_down(self):
        # 근거리 실기 정정(2026-07-21): 내린 손은 화면 밖(휴식 존이 프레임 아래) —
        # 손이 어깨선 아래에서 "등장"해 올라오는 것 자체가 들어올리기 신호다.
        # 등장→상승은 이벤트가 아니어야 하고(게이트), 이어지는 아래 획은 down이다.
        # ★2026-08-04 한계 명시: 손이 가슴~허리께까지만 내려가면 내리는 팔 가드
        # (골반선)에 안 걸려 down이 나간다. 팔을 끝까지 내리는 경우만 가드가 잡는다
        # (시나리오 22) — 그 사이 구간을 어떻게 가를지는 실측 대기 (docs/TODO.md)
        def scenario(sim):
            sim.drop(0, 0, 0.6)                  # 손 부재(화면 밖 — 추적점 없음)
            sim.position = (0.5, 0.40)           # 화면 하단(존 밖·어깨선 아래)에서 등장
            sim.move_by(0, -0.16, 0.25)          # 등장하며 올라옴 — 이벤트 금지
            sim.hold(0.2)
            sim.move_by(0, AMP_Y, 0.3)           # 아래 — down (끝점 0.40, 경계 위)
            sim.hold(0.4)                        # 확정 유예
        self._run(scenario, ["down"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_22_full_arm_drop_fires_down_by_decision(self):
        # ★2026-08-04 **감수하기로 한 오발**을 명시적으로 남긴다(사용자 결정).
        # 팔을 축 내리는 동작은 아래 쓸기와 구분되지 않는다 — 같은 날 두 번
        # 독립적으로 쟀고 둘 다 안 갈렸다:
        #   위치(골반선 기준) 진짜 쓸기 126건 중 60건이 팔 내리기 구간 안
        #   간 거리(유예 0.25초) 쓸기 중앙 0.50 / 내리기 중앙 0.49 — 사실상 동일
        #   게다가 같은 아래 쓸기가 서서 0.50 · 앉아서 0.11로 5배 차이라
        #   자세 하나를 맞추면 다른 자세가 무너진다
        # 가드를 넣으면 진짜 아래 쓸기를 87% 삼켰다(5회 시도 -> 1회 발화). 그래서
        # 가드를 빼고 오발을 받아들였다 — 이 테스트가 그 결정을 눈에 보이게 둔다.
        # 나중에 "왜 팔 내릴 때 down이 나가지?"를 다시 파기 전에 위 숫자를 볼 것
        def scenario(sim):
            sim.hold(0.5)
            sim.move_by(0, HANG[1] - REST[1], 0.45)   # 가슴 -> 축 처진 위치까지
            sim.hold(0.5)
        self._run(scenario, ["down"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_21_close_range_appear_pause_then_up(self):
        # 근거리 정정 2차(2026-07-31 키오스크 실기 — 근거리에서 위 쓸기 무반응):
        # 근거리에선 내린 손이 화면 밖이라 손 등장마다 휴식 존이 스탬프되는데,
        # 구 로직은 유예(0.6초)를 다 기다려야 위 플릭이 살았다 — 등장 직후의
        # 자연스러운 "멈췄다 위로 스냅"이 전부 삼켜졌다. 존 밖 정지(재장전
        # went_still)가 확인되면 들어올리기는 끝난 것 — 스탬프를 지워 짧게 멈춘
        # 뒤의 위 플릭이 나가야 한다 (쉼 없는 등장→상승은 test_20이
        # 여전히 차단을 검증)
        def scenario(sim):
            sim.shape = "fist"                   # 위 방향은 주먹(home)에만 남았다
            sim.drop(0, 0, 0.6)                  # 손 부재(화면 밖)
            sim.position = (0.5, 0.40)           # 하단(존 밖·어깨선 아래)에서 등장
            sim.move_by(0, -0.05, 0.15)          # 가슴께로 마저 올림 — 들어올리기 꼬리
            sim.hold(0.25)                       # 잠깐 정지 — 게이트 해제 (유예 0.6초 미만)
            sim.move_by(0, -AMP_Y, 0.25)         # 위 플릭 = 의도적 home
        self._run(scenario, ["home"])

    def test_22_pointing_at_screen_navigates_via_memory(self):
        # v2 모양 기억(실기 사진 실증): 손을 펴 보인 뒤 화면을 가리키며
        # (손이 카메라 쪽으로 누움 — 판별 기권) 쓸어도 항법이 유지된다
        def scenario(sim):
            sim.hold(0.5)                    # 손 펼침 각인 (분명한 판별 구간)
            sim.shape = None                 # 화면을 가리킴 — 이후 판별 전부 기권
            sim.hold(1.0)                    # 표가 만료돼 창이 비는 구간까지 재현
            sim.move_by(AMP_X, 0, 0.3)       # 가리킨 채 우로 쓸기
            sim.hold(0.2)
        self._run(scenario, ["right"])

    def test_23_pointing_without_prior_shape_is_safe(self):
        # 처음부터 끝까지 판별 불가(기억 없음) — 오발(confirm) 대신 무시가 정답
        def scenario(sim):
            sim.shape = None
            sim.hold(0.5)
            sim.move_by(AMP_X, 0, 0.3)
            sim.hold(0.3)
        self._run(scenario, [])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_24_handedness_flap_mid_stroke_keeps_back(self):
        # 좌/우 라벨 플랩(2026-07-28 실기 — MediaPipe handedness가 주먹에서 불안정):
        # 2026-07-31 라벨 제거로 라벨은 판정과 무관해졌다 — 획 중간에 라벨이 튀어도
        # 신호는 같은 손(hand_select 연속성)이라 back이 그대로 발화한다.
        # 구 플랩 보정(side_flap_jump)의 회귀 검증을 라벨 무관성 검증으로 계승
        def scenario(sim):
            sim.shape = "fist"
            sim.hold(0.5)
            sim.move_by(-AMP_X / 2, 0, 0.15)
            sim.side = "right"              # 라벨 플랩 — 판정에 아무 영향 없다
            sim.move_by(-AMP_X / 2, 0, 0.15)
            sim.hold(0.3)
        self._run(scenario, ["back"])

    @unittest.skip(SKIP_ROTOR_BRANCH)
    def test_21_fist_raise_is_not_home(self):
        # 주먹 쥔 채 들어올리기 — home(처음으로)으로 오발되면 안 된다:
        # 화면 이탈 사고는 사용자 신뢰를 즉시 깎는 최악의 오발이다
        def scenario(sim):
            sim.shape = "fist"
            sim.position = HANG
            sim.hold(0.5)
            sim.move_by(0, REST[1] - HANG[1], 0.4)   # 들어올리기 — home 금지
            sim.hold(0.8)                            # 유예 경과 후
            sim.move_by(0, -AMP_Y, 0.3)              # 의도적 위 스냅 = home
        self._run(scenario, ["home"])


if __name__ == "__main__":
    unittest.main()
