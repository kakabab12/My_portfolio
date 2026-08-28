"""postprocess 모듈 — 머리 좌우 흔들기 감지 (입력 모드 전환 트리거).

2026-07-30 신설 (사용자 결정): 손 제스처 모드 ↔ 헤드트래커 모드를 **머리를 좌우로
흔드는 동작 하나**로 오간다(토글). 손을 쓸 수 없는 상황에서도 모드를 바꿀 수 있어야
하므로 트리거 자체가 손에 의존하면 안 된다 — 그래서 얼굴 신호(코끝)로 판정한다.

판정 원리 — "왕복(방향 반전) 횟수":
코끝 x가 한 방향으로 min_amplitude 이상 움직인 뒤 **반대 방향으로** 다시 그만큼
움직이면 반전 1회로 센다. window_sec 안에 min_reversals회가 쌓이면 흔들기 확정.
단순 이동량 합계가 아니라 반전을 세는 이유: 고개를 한쪽으로 천천히 돌리는 것(=
평상시 동작)은 반전이 없어 절대 걸리지 않는다.

거리 정규화: 코끝 이동량을 안구간 거리로 나눈다 — 카메라와 사용자 거리가 달라져도
같은 크기의 실제 동작이 같은 판정을 받는다 (head_tracker 커서와 같은 원리).

세로 흔들기(끄덕임) 차단: 같은 구간의 y 이동이 x보다 크면 반전으로 세지 않는다 —
"좌우로 흔들기"만 트리거이므로 끄덕임(수직)은 통과시키지 않는다.

빠른 좌우 까딱임 차단(2026-08-04 사용자 보고 — 의도적 흔들기가 아닌 잦은 좌우
까딱임에도 전환됨): 진폭·반전 횟수(min_amplitude·min_reversals)만으로는 "크지만
빠르게 반복되는 까딱임"과 "느긋한 의도적 흔들기"를 구분 못 한다 — 확정 직전
min_reversals개 반전의 첫~끝 시간 폭(span)이 min_duration_sec 미만이면 확정을
보류한다. 진짜 흔들기는 자연스러운 왕복 속도라 window_sec(1.5초)에 가깝게
걸리지만, 까딱임은 훨씬 짧은 폭에 반전 4회가 몰린다. 키 없으면 종전(폭 제한 없음).
"""
import time

from src.utils.logger import get_logger

logger = get_logger("postprocess")

MIN_INTEROCULAR_DIST_PX = 10.0   # 이보다 좁으면(검출 불량) 정규화 자로 못 쓴다 — 이 프레임은 건너뛴다


class HeadShakeDetector:
    """코끝 궤적 -> 좌우 흔들기 확정(bool). 순수 기하 — 단위 테스트로 검증된다."""

    def __init__(self, config, clock=time.monotonic):
        shake = config["mode_switch"]["head_shake"]
        self._min_amplitude = shake["min_amplitude"]         # 안구간 거리 배수
        self._min_reversals = shake["min_reversals"]
        self._window_sec = shake["window_sec"]
        self._axis_dominance = shake["axis_dominance"]
        self._cooldown_sec = shake["cooldown_sec"]
        self._min_duration_sec = shake.get("min_duration_sec", 0.0)
        self._clock = clock
        self._last_fire_sec = None
        self.reset()

    def reset(self):
        """얼굴 소실·모드 전환 직후 — 진행 중이던 왕복 누적을 버린다."""
        self._extreme_x = None          # 마지막 반전 지점(정규화 x) — 다음 진폭의 기준
        self._extreme_y = None
        self._direction = 0             # 마지막으로 확정된 진행 방향 (+1/-1, 0=미정)
        self._reversal_sec_list = []    # 최근 반전 시각들 — window_sec 밖은 버린다
        self.progress_ratio = 0.0       # 계기판용 — 확정까지 얼마나 왔는지(0~1)

    def update(self, nose_px, interocular_dist_px):
        """코끝 1프레임 반영 -> 이번 프레임에 흔들기가 확정되면 True.

        nose_px가 None(얼굴 미검출)이면 누적을 버린다 — 끊긴 궤적을 이어 붙이면
        얼굴이 사라졌다 나타나는 것만으로 반전이 잡힐 수 있다.
        """
        if nose_px is None or interocular_dist_px < MIN_INTEROCULAR_DIST_PX:
            self.reset()
            return False

        x_norm = nose_px[0] / interocular_dist_px
        y_norm = nose_px[1] / interocular_dist_px
        if self._extreme_x is None:
            self._extreme_x, self._extreme_y = x_norm, y_norm
            return False

        dx = x_norm - self._extreme_x
        if abs(dx) < self._min_amplitude:
            return False   # 아직 이번 방향으로 충분히 안 움직였다

        # 좌우 우세 확인 — 끄덕임(수직 이동 우세)은 반전으로 세지 않는다
        dy = y_norm - self._extreme_y
        if abs(dx) < self._axis_dominance * abs(dy):
            self._extreme_x, self._extreme_y = x_norm, y_norm   # 기준만 옮기고 무시
            return False

        direction = 1 if dx > 0 else -1
        self._extreme_x, self._extreme_y = x_norm, y_norm
        if direction == self._direction:
            return False   # 같은 방향 연속 진행 — 반전이 아니다(한쪽으로 크게 돌리는 중)

        is_first_stroke = self._direction == 0
        self._direction = direction
        if is_first_stroke:
            return False   # 첫 획은 방향 기준을 세울 뿐 — 반전은 두 번째 획부터

        now_sec = self._clock()
        self._reversal_sec_list.append(now_sec)
        self._reversal_sec_list = [
            sec for sec in self._reversal_sec_list if now_sec - sec <= self._window_sec
        ]
        self.progress_ratio = min(1.0, len(self._reversal_sec_list) / self._min_reversals)
        if len(self._reversal_sec_list) < self._min_reversals:
            return False
        # 최근 min_reversals개만의 폭을 잰다(전체 목록이 아니라) — window_sec이
        # 넉넉하면 오래 지속되는 빠른 까딱임도 목록이 계속 불어나 폭이 결국
        # min_duration_sec을 넘겨버린다. 최근 N개로 재면 반전 속도가 그대로
        # 유지되는 한 폭도 일정해 아무리 오래 지속돼도 계속 걸러진다
        recent = self._reversal_sec_list[-self._min_reversals:]
        span_sec = recent[-1] - recent[0]
        if span_sec < self._min_duration_sec:
            return False   # 반전은 다 모였지만 너무 빨리 몰림 — 까딱임으로 보고 보류
        if (self._last_fire_sec is not None
                and now_sec - self._last_fire_sec < self._cooldown_sec):
            return False   # 한 번의 긴 흔들기가 연속 전환을 일으키지 않게

        self._last_fire_sec = now_sec
        self.reset()
        logger.info("머리 흔들기 감지 — 입력 모드 전환")
        return True
