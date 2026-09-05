"""postprocess 모듈 — 얼굴 신호를 커서 위치와 동작 이벤트로 변환한다 (헤드트래커 모드).

2026-07-30 병합 이식 (구 헤드트래커_프로토타입_win.ver → 손 제스처판에 합류):
손 제스처 모드는 그대로 두고, 머리 흔들기(head_shake.py)로 전환되는 **두 번째 입력
모드**로 들어왔다. 손을 쓰기 어려운 사용자를 위한 대체 경로다.

동작 체계 (2026-07-30 사용자 확정 — 이벤트명은 회사 UI 계약을 따른다.
같은 날 재확정: ok→select로 개명 — 손 모드의 select와 이벤트명을 맞춘다):
- cursor      : 코끝(랜드마크 1) 위치를 화면 커서로 매핑. 잠금 직후 짧은 구간
                중앙값으로 중심을 캘리브레이션하고, 안구간 거리로 매 프레임 정규화
                (카메라 거리 무관), EMA로 떨림 평활
- select      : 입 벌리기(jawOpen) 즉시, 또는 커서가 반경 안에 1.5초 머무르는
                응시(dwell) — 어느 쪽이든 같은 이벤트, 공용 쿨다운으로 1건만 확정
- home        : 양 눈 감고 0.6초 버티기(eyeBlinkLeft/Right 둘 다 기준선 이상 유지).
                자연스러운 깜빡임(보통 0.4초 이하)은 hold_sec보다 짧아 걸러진다
- calibration : 커서가 반경 안에 일정 시간(config recenter_dwell.dwell_sec) 머무르는
                응시(dwell) — 커서 중심을 지금 고개 위치로 다시 잡는다. 입 오므리기
                (mouthPucker) 판정은 오탐이 너무 잦아 2026-07-30 사용자 결정으로
                기본 비활성(recenter_gesture.enabled: false)

⚠ 아이트래커(시선) 모드는 이번 이식에서 **제외**했다(사용자 결정 2026-07-30) —
프로토타입의 _GazeCursorMapper·홍채 랜드마크·cursor_mode 전환 경로 전부 미이식.
커서 입력은 코끝 하나뿐이라 모드 분기 자체가 없다.

기준선(baseline) 설계 — 프로토타입에서 그대로 가져온 핵심: 입벌림·눈감김 판정은
고정 임계값이 아니라 **잠금 직후 캡처한 평상시 점수 + 여유값**이다. 사람마다 평상시
eyeBlink 블렌드셰이프가 0.1~0.6까지 편차가 커(얼굴 형태·카메라 각도·조명) 고정
임계 하나로는 어떤 사용자는 오탐이, 어떤 사용자는 미탐이 났다. 코끝 커서
캘리브레이션과 같은 구간에서 함께 수집해 추가 대기 시간이 없다.

블렌드셰이프는 MediaPipe FaceLandmarker가 이미 0~1로 정규화·학습해 내보내는 신호라
랜드마크 거리비를 직접 계산할 필요가 없다 — 얼굴 형태·거리 편차에 더 강건하다.

모든 수치는 config head_tracker에서 읽는다 (기획서 4.7).
"""
import math
import time
from dataclasses import dataclass, field

from src.inference.face_estimator import (
    LMK_LEFT_EYE_OUTER, LMK_NOSE_TIP, LMK_RIGHT_EYE_OUTER,
)
from src.postprocess.gesture_filter import GestureEvent
from src.postprocess.auto_arc import OnlineArcCompensator
from src.postprocess.head_orientation import HeadOrientation
from src.postprocess.lens_calibration import LensSelfCalibrator
from src.utils.display_size import detect_screen_size_mm
from src.utils.logger import get_logger


logger = get_logger("postprocess")

MIN_INTEROCULAR_DIST_PX = 10.0  # 이보다 좁으면(검출 불량) 정규화 자로 못 쓴다 — 마지막 값 유지

# 거리 적응 평활의 배율 상·하한 — _CursorMapper._apply_distance_adaptive_cutoff 참고.
#
# 하한 0.5 (기준 60px 대비 30px = 아주 멀리 물러선 경우): 그보다 더 멀면 얼굴
# 인식 자체가 불안정해지는 구간이라, 평활을 더 밀어봐야 얻는 게 없다.
#
# ★상한 1.0인 이유 (2026-08-27 시뮬레이션으로 결정) — 처음엔 2.0으로 뒀는데,
# 그러면 기준보다 가까이 선 사용자는 평활이 풀려 **지금보다 더 떨리게** 된다:
#
#     안구간거리 90px   적응 OFF 17.4px  ->  적응 ON(상한 2.0) 21.0px   +21% 악화
#
# 반응성은 좋아지지만, 지금 손으로 맞춰 검증해 둔 60px 기준 감각을 가까운
# 거리에서 깨뜨리는 건 이득보다 손해다. 상한을 1.0으로 두면 배율이 1을 넘지
# 못하므로 **기준 거리보다 가까울 때는 지금과 완전히 동일하게 동작**하고,
# 멀어질 때만 평활이 세진다 — "같거나 나아지기만 한다"가 보장된다
ONE_EURO_ADAPT_MIN_SCALE = 0.5
ONE_EURO_ADAPT_MAX_SCALE = 1.0

# 각도 기반 매핑에서 회전각을 자르는 한도 (도).
# tan(θ)는 90°에서 발산한다 — 키오스크에서 고개를 그만큼 돌릴 일은 없지만,
# 한 프레임이라도 튀면 커서가 화면 밖으로 순간이동한다. 사람이 화면을 보며
# 조작하는 범위를 넉넉히 감싸는 값으로 자른다
HEAD_POSE_MAX_ANGLE_DEG = 60.0

EVENT_SELECT = "select"            # 입 벌리기 / 1.5초 응시 — 선택·확인 (구 ok, 2026-07-30 개명)
EVENT_HOME = "home"               # 양 눈 감고 버티기 — 처음으로
EVENT_CALIBRATION = "calibration" # 응시(또는 입 오므리기, 기본 비활성) — 커서 중심 재정렬


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _median(samples):
    """스칼라 또는 (x, y) 튜플 표본 목록 -> 중앙값(튜플이면 요소별로).

    평균 대신 중앙값인 이유: 캘리브레이션 구간 중 순간적인 과도기 프레임(눈 깜빡임·
    고개 숙임 등) 1~2장이 섞여도 평균처럼 기준 전체가 끌려가지 않는다
    (프로토타입 2026-07-18 실기 — 커서가 화면 상단에 눌러붙던 문제의 원인이었다).
    """
    if isinstance(samples[0], tuple):
        return tuple(_median(list(dim_samples)) for dim_samples in zip(*samples))
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def _clamp(value, limit):
    return max(-limit, min(limit, value))



# 얼굴이 이만큼 이어서 안 보여야 "새 사용자"로 보고 처음부터 다시 잡는다.
# 그보다 짧은 끊김은 검출이 잠깐 놓친 것으로 보고 중립을 그대로 들고 있는다
# (HeadTracker.update의 설명 참고 — 이걸 0으로 두면 한 프레임만 놓쳐도
# 캘리브레이션이 처음으로 돌아가, 검출이 불안정한 배치에서 커서가 영영
# 안 나온다). 1초면 30fps에서 30프레임 연속 미검출이라, 사람이 실제로
# 자리를 뜬 경우와 한두 프레임 놓친 경우가 확실히 갈린다.
FACE_LOST_RESET_SEC = 1.0

class _MedianCalibrator:
    """calibration_window_sec 동안 표본을 모아 중앙값을 낸다 — 캘리브레이션 공용 로직.

    코 위치(화면 중심)·입 벌림/눈 감김 평상시 점수(판정 기준선)가 전부 이 패턴을
    쓴다 — "잠금 직후 짧은 구간을 관찰해 그걸 기준으로 삼는다"는 셋의 본질이 같다.
    한 번 확정(value가 설정)되면 이후 표본은 무시한다.
    """

    def __init__(self, window_sec):
        self._window_sec = window_sec
        self.reset()

    @property
    def window_sec(self):
        """표본을 모으는 구간 길이 — 상대 회전 매핑이 중립을 잡을 때 같은 구간을
        쓰려고 읽는다(_update_from_orientation 참고). 두 캘리브레이션이 같은 시간에
        끝나야 커서가 확정되는 시점이 하나로 맞는다."""
        return self._window_sec

    def reset(self):
        self._start_sec = None
        self._samples = []
        self.value = None   # 캘리브레이션 완료 전 None, 완료 후 중앙값으로 고정

    def update(self, sample, now_sec):
        """표본 1건 반영 -> 캘리브레이션 완료 후 확정된 중앙값(미완료 시 None)."""
        if self.value is not None:
            return self.value
        if self._start_sec is None:
            self._start_sec = now_sec
        self._samples.append(sample)
        if now_sec - self._start_sec < self._window_sec:
            return None
        self.value = _median(self._samples)
        return self.value

    def force_set(self, value):
        """중앙값 수집 창을 건너뛰고 즉시 이 값으로 확정한다 (2026-08-18 신설 —
        HeadTracker.recenter_cursor 전용). reset()과 달리 표본을 새로 모으는
        동안 미확정(None) 구간이 생기지 않는다 — 이미 안정된 값을 알고 있을 때
        (예: 정지 유예가 끝나는 시점의 현재 위치) 곧바로 그 값을 기준으로 쓴다."""
        self.value = value
        self._samples = []
        self._start_sec = None


class OneEuroFilter:
    """1€ 필터 — Casiez, Roussel, Vogel, "1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems", ACM CHI 2012.
    마우스·터치 대체 인터페이스(눈짓·머리 추적 포함) 잡음 제거의 사실상
    표준으로 쓰이는, 논문으로 발표되고 널리 검증된 알고리즘이다
    (2026-08-27 forehead.py 대응 신설 — _CursorMapper의 단순 EMA를 대체할
    선택지로 추가. head.py·eyebrow.py는 기본값(비활성)이라 동작이 그대로다).

    EMA 하나만 쓰면 "잘 안 떨리게"(느리게 반응)와 "잘 따라오게"(떨림 남음)를
    동시에 만족할 수 없다 — 이 프로젝트가 그동안 여러 상수(POINTER_SMOOTHING_
    ALPHA_OVERRIDE 등)를 손으로 절충해 온 문제 그 자체다. 1€ 필터는 **입력이
    빠르게 움직일 때는 평활을 약하게(지연 최소화), 느리거나 멈췄을 때는
    평활을 강하게(떨림 억제)** 자동으로 조절해 이 절충을 알고리즘 차원에서
    해결한다 — 속도 dx를 추정해 매 프레임 차단 주파수(cutoff)를 새로 계산하는
    적응형 저역통과 필터다.

    min_cutoff: 정지 시 얼마나 부드러운가(작을수록 떨림이 덜 보이지만 느려짐).
    beta: 속도가 오를 때 얼마나 빨리 평활을 풀어주는가(클수록 빠른 움직임을
    더 지연 없이 따라간다). 논문 권장 시작값(min_cutoff=1.0, beta=0.0)을
    기본으로 둔다 — 실기로 두 값을 조정해 정지 시 떨림과 빠른 반응 사이의
    균형을 잡을 것.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = max(t - self._t_prev, 1e-6)   # 0 나눗셈 방지 — 같은 틱에 두 번 불려도 안전
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def prime(self, value, t):
        """캘리브레이션이 막 끝난 시점처럼 "이 값에서 시작한다"고 강제로
        맞춰 둔다 — 안 그러면 캘리브레이션 직후 첫 실제 신호와 필터 내부
        상태(0.5 근방)가 어긋나 있어 첫 몇 프레임이 순간이동처럼 보인다."""
        self._x_prev = value
        self._dx_prev = 0.0
        self._t_prev = t

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


@dataclass
class HeadTrackerResult:
    """head_tracker.update() 1회 호출의 결과 — 연속 커서 상태 + 이산 이벤트 묶음.

    gesture_filter는 "이벤트 1개 또는 None"을 반환하지만, 헤드트래커는 매 프레임
    연속 커서 상태를 UI가 그려야 해서 이 결과 구조로 확장한다.
    """

    cursor_x_ratio: float = None
    cursor_y_ratio: float = None
    is_tracking: bool = False
    events: list = field(default_factory=list)


class _CursorMapper:
    """얼굴 기준점 -> 화면 커서 비율. 캘리브레이션 1회 + 얼굴 기준 정규화 + EMA.

    ★2026-08-20 재설계 — "내가 바라보는 지점에 커서가 정확히 가 있게" (사용자 요청).

    [무엇이 문제였나]
    예전에는 기준점(코끝)의 **화면상 절대 픽셀 위치**를 캘리브레이션 때 기억해
    두고, 매 프레임 그 위치에서 얼마나 벗어났는지로 커서를 움직였다. 문제는 이
    방식이 전혀 다른 두 가지를 구분하지 못한다는 것이다:

      ① 고개를 돌렸다      -> 커서가 움직여야 한다 (맞음)
      ② 몸이 움직였다      -> 커서는 그대로여야 한다 (틀림)
         (자세를 고쳐 앉거나, 키오스크 앞에서 한 발 옆으로 서거나, 몸을 기울이거나)

    ②도 코끝의 절대 픽셀 위치를 바꾸므로 커서가 따라 움직였다. 게다가 기준점은
    캘리브레이션 때 고정된 값이라 **되돌아오지 않는다** — 이게 처음부터 보고됐던
    "고개를 가만히 있어도 커서가 조금씩 밀려간다"의 정체이고, 주기적 자동
    재정렬이라는 반창고가 필요했던 이유다.

    [어떻게 고쳤나]
    기준점을 화면이 아니라 **얼굴 자신을 기준으로** 잰다. 양쪽 눈 바깥쪽 끝
    두 점으로 얼굴의 좌표계를 만들고(원점 = 두 눈의 중점, 가로축 = 두 눈을 잇는
    방향, 길이 단위 = 두 눈 사이 거리), 그 안에서 기준점이 어디 있는지를 본다.

    그러면 세 가지가 자동으로 해결된다:
      · 몸이 움직이면 눈과 코가 **함께** 움직이므로 얼굴 좌표계 안의 값은 그대로
        -> 커서가 안 움직인다 (밀려감이 원리적으로 사라진다)
      · 카메라에 가까워지거나 멀어지면 두 눈 사이 거리로 나누므로 상쇄된다
      · 고개를 갸웃하면(roll) 얼굴 가로축도 같이 기울므로 그만큼 되돌려 잰다

    남는 건 순수한 고개 회전뿐이다 — 즉 "얼굴이 향한 방향" 하나만 커서를
    움직인다. 이게 곧 "바라보는 지점에 커서가 간다"의 의미다.

    [왜 gain 보정이 필요한가]
    고개를 각도 θ만큼 돌리면, 얼굴 좌표계에서 코끝은 (코가 얼굴에서 튀어나온
    길이) x sinθ 만큼 옮겨간다. 반면 예전 방식에서는 (회전 중심에서 코끝까지의
    거리) x sinθ 만큼 옮겨갔다 — 회전 중심은 목 부근이라 훨씬 멀다. 그래서 같은
    각도라도 새 방식의 값이 **몇 배 작다**. 그 차이를 되돌려 예전과 같은 손맛을
    유지하는 게 face_local_gain이다. 사람 머리 치수로 어림하면 2.3~4.0 범위이고
    기본값은 2.0이다 — 이 값은 신호와 잡음을 **똑같이** 키우므로 올린다고 떨림
    대비 이득이 없고, 순전히 "정밀도 vs 고개를 덜 돌려도 됨"의 손잡이다.
    선정 근거와 값별 효과는 head.py의 FACE_LOCAL_GAIN 상수 설명에 표로 정리했다.
    """

    def __init__(self, calibration_window_sec, sensitivity_x, sensitivity_y, smoothing_alpha,
                 distance_smoothing_alpha, max_offset_ratio,
                 face_local=True, face_local_gain=2.0,
                 one_euro_enabled=False, one_euro_min_cutoff=1.0, one_euro_beta=0.0,
                 one_euro_distance_adaptive=False, one_euro_reference_dist_px=60.0,
                 arc_compensation=0.0, head_pose_mapping=False,
                 orientation_mapping=False,
                 orientation_half_span_x_deg=15.0, orientation_half_span_y_deg=10.0,
                 orientation_rotation_source="auto", orientation_auto_arc=True,
                 orientation_lens_calibration=True,
                 orientation_lens_distortion=False,
                 orientation_distance_scaling=True,
                 orientation_reach_gain=1.0,
                 screen_width_mm=None, screen_height_mm=None,
                 reference_distance_mm=None):
        """one_euro_enabled(2026-08-27 forehead.py 대응 신설, OneEuroFilter 독스트링
        참고) — 기본 False라 head.py·eyebrow.py는 예전과 완전히 동일한 단순 EMA
        평활을 그대로 쓴다(동작 변화 없음). True면 EMA 대신 1€ 필터로 최종 커서
        떨림을 줄인다 — 속도에 따라 평활 강도를 자동 조절해 정지 시 떨림과
        빠른 반응 사이 절충을 알고리즘이 대신 해 준다.

        one_euro_distance_adaptive(2026-08-27 신설) — 사용자가 카메라에서 멀수록
        평활을 자동으로 더 세게 건다. 근거는 이 프로젝트가 8/26에 직접 실측한
        값이다 (head.py DRAG_DEAD_ZONE_SCALE 주석의 표):

            안구간거리 90px (가까움)  커서 흔들림 2.0px
            안구간거리 60px (키오스크) 커서 흔들림 3.0px
            안구간거리 40px (물러섬)  커서 흔들림 4.5px

        곱해 보면 90x2.0 = 60x3.0 = 40x4.5 = 180 으로 **정확히 1/거리 비례**다.
        당연한 결과다 — 커서 위치를 (랜드마크가 움직인 px / 안구간거리)로 정하는데
        랜드마크 잡음 자체는 거리와 무관하게 일정하니, 나누는 값이 작아질수록
        (멀수록) 같은 잡음이 그대로 증폭된다.

        그래서 평활 강도를 거리에 비례시켜 이 증폭을 상쇄한다 — 1€ 필터의
        min_cutoff를 (지금 안구간거리 / 기준 안구간거리)만큼 조절한다. 멀면
        cutoff가 내려가 더 세게 평활되고, 가까우면 올라가 더 민첩해진다.
        기준값(one_euro_reference_dist_px)은 키오스크 실사용 거리에 맞춘 60px —
        그 거리에서는 배율이 1.0이라 손으로 맞춘 감각이 그대로 유지된다."""
        self._sensitivity_x = sensitivity_x
        self._sensitivity_y = sensitivity_y
        self._smoothing_alpha = smoothing_alpha
        self._distance_smoothing_alpha = distance_smoothing_alpha
        self._max_offset_ratio = max_offset_ratio
        self._face_local = face_local
        self._face_local_gain = face_local_gain
        self._one_euro_enabled = one_euro_enabled
        # ★가로 이동 시 세로가 휘는 것(활 모양) 보정 계수 — 2026-08-27 신설.
        #
        # 증상: 고개를 좌우로만 돌렸는데 커서가 수평이 아니라 뒤집힌 U(∩) 모양
        # 포물선을 그린다. 실기 보고로 확인됐다.
        #
        # 원인: 커서 기준점에 섞인 **코**는 얼굴 밖으로 튀어나온 3차원 점이다.
        # 고개를 좌우로 돌리면 코끝은 화면상에서 단순히 옆으로만 가는 게 아니라
        # 원근 때문에 **세로 위치까지 같이 밀린다**(카메라에서 멀어졌다 가까워지며
        # 투영 배율이 바뀐다). 그 밀림은 가로 회전량의 **제곱에 비례**한다 —
        # 즉 궤적이 정확히 2차 곡선이다.
        #
        # 대응: 같은 형태(2차)를 빼주면 원리적으로 상쇄된다.
        #     보정 후 세로 = 세로 + 계수 x (가로offset)^2
        # 양수면 화면 양끝에서 커서를 아래로 내려 위로 휜 활을 편다(∩ -> ―).
        # 반대로 U자로 휘면(∪) 음수를 넣는다. 0이면 보정 없음(기존 동작).
        #
        # 계수는 커서 좌표계 기준이라 감각적으로 읽힌다 — 예를 들어 0.2면
        # 화면 좌우 끝(가로offset 0.5)에서 화면 높이의 0.2 x 0.25 = 5%만큼
        # 내려준다는 뜻이다. 실측으로 맞추는 게 정확하다:
        # (measure_arc.py는 2026-08-31 정리로 삭제 — auto_arc.py가 자동 대체)
        self._arc_compensation = arc_compensation
        # ★각도 기반 매핑 — HEAD_POSE_MAPPING 설명 참고. 기본 False라
        # 이 값을 안 넘기는 기존 호출부는 동작이 전혀 바뀌지 않는다
        self._head_pose_mapping = head_pose_mapping
        # ★상대 회전 매핑 (2026-08-31) — head_orientation.py 참고.
        # 카메라를 어떻게 달아도 잴 것이 없다. 감도는 배율이 아니라
        # "고개를 몇 도 돌리면 화면 끝인가"라는 사람 기준 각도로 준다
        self._orientation_mapping = orientation_mapping
        # 회전 재료(auto=변환행렬 우선) — head_orientation.py "회전의 재료 두 가지" 참고
        self._orientation = (HeadOrientation(rotation_source=orientation_rotation_source)
                             if orientation_mapping else None)
        # 잔여 곡률 자동 소거 — auto_arc.py 참고. 어떤 카메라 배치에서 어떤
        # 곡률이 남든, 쓰는 동안 스스로 2차항을 추정해 빼 준다.
        # 사람이 재서 넣던 ARC_COMPENSATION의 자동판이다
        self._auto_arc = (OnlineArcCompensator()
                          if (orientation_mapping and orientation_auto_arc) else None)
        # ★렌즈 자가 보정 (lens_calibration.py) — 광각 렌즈의 배럴 왜곡과
        # 원근 단축을 되돌린다. 사용자의 얼굴이 곧 보정판이라 현장에서
        # 아무것도 재지 않는다. 첫 얼굴을 볼 때 화면 크기를 알고 만든다
        self._lens_calibration_enabled = bool(
            orientation_mapping and orientation_lens_calibration)
        # 왜곡 되돌리기는 기본 끔 (lens_calibration.py 독스트링 참고) —
        # 얼굴이 정규 모형과 다르면 해로운데 가려낼 방법을 못 찾았다
        self._lens_distortion_enabled = bool(orientation_lens_distortion)
        self._lens_calibrator = None
        self._lens_applied = False
        # tan으로 미리 바꿔 둔다 — 매 프레임 삼각함수를 다시 부르지 않게
        # ★반폭 — 설치 치수를 알면 계산하고, 모르면 설정값을 쓴다 (2026-09-05).
        #
        # 사용자가 화면에서 Z만큼 떨어져 고개를 θ 돌리면 시선이 화면에서
        # Z·tan(θ)만큼 옮겨간다. 그러니 화면 절반 폭 W/2에 닿는 각도는
        #
        #     반폭 = atan((W/2) / Z)
        #
        # 로 **기하학이 정해 준다.** 이 값을 쓰면 "얼굴이 향한 곳에 커서가
        # 있다"가 성립한다 — 사용자가 아이콘을 보면 커서가 거기 있으므로
        # 커서를 보며 조이스틱처럼 몰 필요가 없다.
        #
        # 15도라는 기존 상수에는 근거가 없었다. 530mm 화면 기준으로 맞는
        # 거리는 989mm 하나뿐이고, 600mm에서는 23.8도여야 한다.
        #
        # 화면 크기는 운영체제가 알려 준다(display_size.py — 모니터 EDID).
        # 거리는 사람이 한 번 정한다: 그 화면 앞에 사용자가 앉거나 서는 거리.
        #
        # ★거리를 카메라로 추정해 보려다 접었다. Z = f x 얼굴크기 / 화면크기로
        # 구할 수 있는데, f가 얼굴 생김새에 따라 중앙값 15.7%, 최악 69.3%까지
        # 틀린다. 그 추정을 섞어 봤더니 **오히려 나빠졌다** — 실제로 생기는
        # 거리 편차(기준 대비 +-20%) 안에서 재 보면:
        #
        #     방식                    평균   중앙값  90분위   최악
        #     15도 고정               6.4%   5.0%   15.5%  20.0%
        #     화면 치수만 (채택)        3.1%   2.7%    7.0%  11.0%
        #     + f로 거리 추정 (w=0.4)  3.6%   2.8%    7.6%  13.8%
        #     + f로 거리 추정 (w=1.0)  5.9%   5.0%   14.3%  20.4%
        #
        # 사람이 한 번 준 거리가 f보다 정확하다. 그래서 f는 원근 되돌리기에만
        # 쓰고 거리에는 안 쓴다. 사용자가 그 자리에서 앞뒤로 움직이는 것은
        # **거리 변화 비율**이 맡는다 — 그쪽은 f가 필요 없어 0.7% 안쪽이다.
        #
        # 안 주면 예전 그대로 설정 상수를 쓴다 — 거동이 안 바뀐다.
        span_x_deg = orientation_half_span_x_deg
        span_y_deg = orientation_half_span_y_deg
        self._screen_geometry = None
        if (screen_width_mm and screen_height_mm and reference_distance_mm
                and screen_width_mm > 0 and screen_height_mm > 0
                and reference_distance_mm > 0):
            span_x_deg = math.degrees(math.atan(
                (screen_width_mm * 0.5) / reference_distance_mm))
            span_y_deg = math.degrees(math.atan(
                (screen_height_mm * 0.5) / reference_distance_mm))
            self._screen_geometry = (float(screen_width_mm),
                                     float(screen_height_mm),
                                     float(reference_distance_mm))
            logger.info(
                "화면 치수로 반폭을 계산했습니다: %.0fx%.0fmm 화면, %.0fmm 거리 "
                "-> 가로 %.1f도 세로 %.1f도 (설정값 %.1f/%.1f 대신)",
                screen_width_mm, screen_height_mm, reference_distance_mm,
                span_x_deg, span_y_deg,
                orientation_half_span_x_deg, orientation_half_span_y_deg)
        self._orientation_tan_x = math.tan(math.radians(
            max(1.0, min(60.0, span_x_deg))))
        self._orientation_tan_y = math.tan(math.radians(
            max(1.0, min(60.0, span_y_deg))))
        self._orientation_distance_scaling = bool(orientation_distance_scaling)
        # ★도달 배율 (2026-09-05 신설) — 고개가 조금밖에 안 돌아가는 사람용.
        #
        # 좌우 7도까지만 돌아가는 사람은 화면 폭의 46%에서 멈춘다. 가장자리
        # 버튼을 아예 못 누르므로 정확도가 아니라 "쓸 수 있냐"의 문제다.
        # 이 값을 올리면 그만큼 적게 돌려도 끝까지 간다.
        #
        # **자동으로 정하지 않는다.** 관측만으로는 "못 돌리는 사람"과 "그냥
        # 가운데만 쓴 사람"이 구별되지 않는다 — 왕복 최대치의 상위 75%/95%
        # 비가 전자 0.93, 후자 0.87로 겹친다(measure_reach.py). 잘못 올리면
        # 목이 멀쩡한 사람은 아무 이득 없이 떨림만 배율만큼 커진다.
        #
        # 그래서 **재서 정한다.** scripts/measure_reach.py가 그 사람의 실제
        # 가동범위를 재고 권장값을 알려 준다. 기본 1.0은 손대지 않음이다.
        self._reach_gain = max(0.5, min(3.0, float(orientation_reach_gain or 1.0)))
        self._orientation_calibrating = True
        self._orientation_started_sec = None
        self._one_euro_base_cutoff = one_euro_min_cutoff
        self._one_euro_distance_adaptive = one_euro_distance_adaptive
        self._one_euro_reference_dist_px = max(1.0, one_euro_reference_dist_px)
        self._one_euro_x = OneEuroFilter(one_euro_min_cutoff, one_euro_beta)
        self._one_euro_y = OneEuroFilter(one_euro_min_cutoff, one_euro_beta)
        self._center_calibrator = _MedianCalibrator(calibration_window_sec)
        self.reset()

    def _apply_distance_adaptive_cutoff(self):
        """지금 거리에 맞춰 1€ 필터의 min_cutoff를 다시 잡는다 — __init__ 독스트링
        참고. 배율에 상·하한을 두는 이유: 얼굴이 아주 크게(가깝게) 잡히면 평활이
        거의 풀려 떨림이 그대로 나가고, 아주 작게 잡히면 평활이 과해져 커서가
        끌려오듯 느려진다. 실사용 거리 범위를 넘어서면 그냥 그 끝값으로 묶는다."""
        if not self._one_euro_distance_adaptive or not self._smoothed_dist_px:
            return
        scale = self._smoothed_dist_px / self._one_euro_reference_dist_px
        scale = max(ONE_EURO_ADAPT_MIN_SCALE, min(ONE_EURO_ADAPT_MAX_SCALE, scale))
        cutoff = self._one_euro_base_cutoff * scale
        self._one_euro_x.min_cutoff = cutoff
        self._one_euro_y.min_cutoff = cutoff

    def reset(self):
        """새 사용자·추적 끊김 — 캘리브레이션부터 다시 한다."""
        self._center_calibrator.reset()
        # 상대 회전 매핑의 중립도 함께 버린다 — 사람이 바뀌면 얼굴 모양이
        # 달라져 이전 중립에 맞춘 회전이 통째로 틀어진다
        if self._orientation is not None:
            self._orientation.reset()
        if self._auto_arc is not None:
            self._auto_arc.reset()      # 사람이 바뀌면 곡률도 그 사람 것이 아니다
        if self._lens_calibrator is not None:
            # 렌즈는 사람이 바뀌어도 그대로다 — 모아 둔 뷰만 비운다
            self._lens_calibrator.reset()
        self._orientation_calibrating = True
        self._orientation_started_sec = None
        self._smoothed_dist_px = None
        self.cursor_x_ratio = None
        self.cursor_y_ratio = None
        self._one_euro_x.reset()
        self._one_euro_y.reset()

    def set_tuning(self, sensitivity_x=None, sensitivity_y=None, arc_compensation=None,
                   half_span_x_deg=None, half_span_y_deg=None):
        """감도·곡률 보정을 실행 중에 바꾼다 (2026-08-28 신설 — 볼륨 조절
        같은 실시간 조절 UI, scripts/tuning_ui.py 대응).

        None인 항목은 건드리지 않는다 — 호출부가 매번 세 값을 다 알 필요
        없이 바뀐 것만 넘기면 된다. 캘리브레이션 상태(center_calibrator,
        one_euro 내부 상태 등)는 전혀 안 건드린다 — 감도만 바뀌어도 커서가
        순간이동하듯 튀면 안 되므로, 다음 프레임부터 새 감도로 자연스럽게
        이어가는 것만 목표로 한다.
        """
        if sensitivity_x is not None:
            self._sensitivity_x = sensitivity_x
        if sensitivity_y is not None:
            self._sensitivity_y = sensitivity_y
        if arc_compensation is not None:
            self._arc_compensation = arc_compensation
        # ★상대 회전 매핑의 감도 (2026-08-31).
        #
        # 이 경로에서는 위의 sensitivity_x/y·arc_compensation이 쓰이지 않는다.
        # 그대로 두면 실시간 조절 UI의 슬라이더를 움직여도 아무 일이 안 일어나
        # "고장 난 것처럼" 보인다 — 그래서 각도 손잡이도 같이 받는다.
        if half_span_x_deg is not None:
            self._orientation_tan_x = math.tan(math.radians(
                max(1.0, min(60.0, half_span_x_deg))))
        if half_span_y_deg is not None:
            self._orientation_tan_y = math.tan(math.radians(
                max(1.0, min(60.0, half_span_y_deg))))

    @property
    def is_orientation_mapping(self):
        """지금 상대 회전 경로로 도는가 — 조절 UI가 어떤 손잡이를 보여줄지 정할 때."""
        return bool(self._orientation_mapping)

    def _measure(self, cursor_px, eye_left_px, eye_right_px):
        """이번 프레임의 기준점 위치를 재서 (가로, 세로) 한 쌍으로 돌려준다.

        face_local이면 얼굴 좌표계 기준(위 독스트링 참고), 아니면 예전처럼 화면
        픽셀 기준 그대로. 두 경우 모두 뒤쪽 계산(중앙값 캘리브레이션 -> 차이 ->
        민감도)은 완전히 동일하게 흘러가므로 분기는 여기 한 곳뿐이다.
        """
        if not self._face_local:
            return cursor_px
        ex = eye_right_px[0] - eye_left_px[0]
        ey = eye_right_px[1] - eye_left_px[1]
        dist = math.hypot(ex, ey)
        if dist < MIN_INTEROCULAR_DIST_PX:
            return None
        # 두 눈의 중점을 원점으로
        vx = cursor_px[0] - (eye_left_px[0] + eye_right_px[0]) * 0.5
        vy = cursor_px[1] - (eye_left_px[1] + eye_right_px[1]) * 0.5
        # 얼굴이 갸웃한 만큼(roll) 되돌려 얼굴 자신의 가로/세로축으로 바꾼다.
        # 회전각을 따로 구할 필요 없이 두 눈 방향 단위벡터로 사영하면 된다
        ux, uy = ex / dist, ey / dist
        local_x = (vx * ux + vy * uy) / dist      # 얼굴 가로축 성분
        local_y = (-vx * uy + vy * ux) / dist     # 얼굴 세로축 성분
        return (local_x * self._face_local_gain, local_y * self._face_local_gain)

    def recenter_instant(self, cursor_px, eye_left_px=None, eye_right_px=None, face=None):
        """지금 이 자세를 새 중심으로 즉시 확정한다 (HeadTracker.recenter_cursor 전용).
        reset()과 달리 표본을 다시 모으지 않아 is_tracking이 끊기지 않는다."""
        if self._orientation_mapping and face is not None:
            # 상대 회전 경로는 중립이 곧 중심이다 — 지금 얼굴을 새 중립으로 삼는다
            if self._orientation.set_neutral(face):
                self._orientation_calibrating = False
                self.cursor_x_ratio, self.cursor_y_ratio = 0.5, 0.5
                now_sec = time.monotonic()
                self._one_euro_x.prime(0.5, now_sec)
                self._one_euro_y.prime(0.5, now_sec)
                return True
            return False
        measured = self._measure(cursor_px, eye_left_px, eye_right_px)
        if measured is None:
            return False   # 눈 사이 거리가 너무 짧다(검출 불량) — 호출자가 다시 시도하게
        self._center_calibrator.force_set(measured)
        self.cursor_x_ratio, self.cursor_y_ratio = 0.5, 0.5
        # 1€ 필터도 같은 값·같은 시각으로 맞춰 둔다 — 안 그러면 다음 update()에서
        # 내부 상태가 어긋나 첫 프레임이 순간이동처럼 보인다(OneEuroFilter.prime 참고)
        now_sec = time.monotonic()
        self._one_euro_x.prime(0.5, now_sec)
        self._one_euro_y.prime(0.5, now_sec)
        return True

    def _measure_head_pose(self, head_pose):
        """머리 회전각을 커서 좌표 재료로 바꾼다 (head_pose_mapping 전용).

        ★2026-08-28 신설 — HEAD_POSE_MAPPING 설명 참고.

        각도를 그대로 쓰지 않고 **탄젠트**를 쓴다. 고개를 각도 θ만큼 돌렸을 때
        사용자가 바라보는 화면 위의 지점은 tan(θ)에 비례해 움직이지 θ에 비례하지
        않는다 — 눈에서 화면까지가 직선 거리이기 때문이다. θ를 그대로 쓰면 화면
        가장자리로 갈수록 커서가 실제 시선보다 뒤처진다.

        _measure()가 돌려주는 값과 단위를 맞춰 둔다 — 뒤쪽 계산(중앙값
        캘리브레이션 -> 차이 -> 민감도 -> 클램프)이 두 방식에서 완전히 같게
        흘러가야 하기 때문이다. 그래서 여기서도 부호 없는 무차원 값을 낸다.
        """
        if head_pose is None:
            return None
        yaw = math.radians(head_pose.yaw_deg)
        pitch = math.radians(head_pose.pitch_deg)
        # 고개를 90°에 가깝게 돌리면 tan이 발산한다 — 키오스크에서 그럴 일은
        # 없지만, 한 프레임이라도 튀면 커서가 화면 밖으로 순간이동한다
        limit = math.radians(HEAD_POSE_MAX_ANGLE_DEG)
        yaw = max(-limit, min(limit, yaw))
        pitch = max(-limit, min(limit, pitch))
        # pitch 부호를 뒤집는다 — 위를 보면 pitch가 양수인데, 화면 좌표는
        # 위쪽이 0이라 커서도 위(작은 값)로 가야 한다
        return (math.tan(yaw), -math.tan(pitch))

    def update(self, cursor_px, eye_left_px, eye_right_px, now_sec, head_pose=None,
               face=None):
        # ★상대 회전 매핑 (2026-08-31) — 켜져 있고 3차원 랜드마크가 실제로 올 때만
        # 이 경로를 탄다. 안 오면 조용히 기존 방식으로 되돌아간다
        if self._orientation_mapping and face is not None:
            result = self._update_from_orientation(face, now_sec)
            if result is not None:
                return result
        # ★각도 기반 매핑 (HEAD_POSE_MAPPING) — 켜져 있고 자세 정보가 실제로
        # 올 때만 이 경로를 탄다. 자세가 안 오면(옵션 꺼짐·모델 미지원) 조용히
        # 기존 랜드마크 방식으로 되돌아간다 — 갑자기 커서가 멈추면 안 된다
        if self._head_pose_mapping and head_pose is not None:
            return self._update_from_head_pose(head_pose, now_sec)

        interocular_dist_px = _dist(eye_left_px, eye_right_px)
        if interocular_dist_px < MIN_INTEROCULAR_DIST_PX:
            return self.cursor_x_ratio, self.cursor_y_ratio   # 검출 불량 — 마지막 값 유지

        if self._smoothed_dist_px is None:
            self._smoothed_dist_px = interocular_dist_px
        else:
            self._smoothed_dist_px += self._distance_smoothing_alpha * (
                interocular_dist_px - self._smoothed_dist_px
            )

        measured = self._measure(cursor_px, eye_left_px, eye_right_px)
        if measured is None:
            return self.cursor_x_ratio, self.cursor_y_ratio

        center = self._center_calibrator.update(measured, now_sec)
        if center is None:
            return None, None   # 캘리브레이션 중 — 커서 미확정

        if self.cursor_x_ratio is None:
            # 캘리브레이션이 막 끝난 프레임 — 정의상 화면 중앙에서 시작한다
            logger.info("커서 중심 캘리브레이션 완료 (%s 기준)",
                        "얼굴 좌표계" if self._face_local else "화면 픽셀")
            self.cursor_x_ratio, self.cursor_y_ratio = 0.5, 0.5
            self._one_euro_x.prime(0.5, now_sec)   # recenter_instant와 동일 이유
            self._one_euro_y.prime(0.5, now_sec)
            return self.cursor_x_ratio, self.cursor_y_ratio

        # face_local이면 measured가 이미 두 눈 사이 거리로 나뉜 무차원 값이라
        # 여기서 또 나누지 않는다. 아니면(예전 방식) 픽셀 차이를 여기서 정규화한다
        if self._face_local:
            dx = measured[0] - center[0]
            dy = measured[1] - center[1]
        else:
            dx = (measured[0] - center[0]) / self._smoothed_dist_px
            dy = (measured[1] - center[1]) / self._smoothed_dist_px
        # y는 x보다 낮은 민감도를 쓴다 — 고개는 좌우 회전 범위가 상하보다 훨씬 넓어
        # 같은 민감도면 상하가 쉽게 끝까지 튄다 (프로토타입 실기: 상단에 눌러붙는 현상)
        offset_x = dx * self._sensitivity_x
        offset_y = dy * self._sensitivity_y
        # ★가로 이동 시 세로가 활처럼 휘는 것을 상쇄한다 (_arc_compensation 설명 참고).
        #
        # ★2026-08-28 버그 수정 — 예전엔 "클램프 전에 해야 한다"며 클램프 안 된
        # offset_x를 그대로 제곱해 썼는데, 이게 틀렸다. 화면 가장자리에 이미
        # 닿은 뒤에도 고개를 더 돌리면 offset_x는(클램프되기 전 값이라) 계속
        # 커지고, 보정량(offset_x²에 비례)도 한계 없이 계속 커진다 — 결국
        # offset_y가 이 보정만으로 자기 클램프(max_offset_ratio)에 부딪혀
        # "화면 양쪽 끝에서 커서가 위로 확 올라가는" 현상, 그리고 중간 구간의
        # 잔여 곡률과 겹쳐 "누운 S자"로 보이는 현상이 실사용에서 나왔다
        # (2026-08-28 사용자 실기 보고로 발견).
        #
        # 화면에서 가장 많이 휘는 지점은 정확히 커서가 가장자리에 닿는
        # 지점(offset_x가 클램프에 닿는 지점)이다 — 그 이상 고개를 돌려도 커서
        # 표시 위치는 어차피 안 바뀌므로 보정도 그 이상 커질 이유가 없다.
        # 보정 계산에는 **클램프된** offset_x를 쓴다.
        if self._arc_compensation:
            clamped_offset_x = _clamp(offset_x, self._max_offset_ratio)
            offset_y += self._arc_compensation * clamped_offset_x * clamped_offset_x
        offset_x = _clamp(offset_x, self._max_offset_ratio)
        offset_y = _clamp(offset_y, self._max_offset_ratio)
        raw_x, raw_y = 0.5 + offset_x, 0.5 + offset_y

        if self._one_euro_enabled:
            # OneEuroFilter 독스트링 참고 — head.py·eyebrow.py는 이 분기를 타지
            # 않는다(one_euro_enabled 기본 False)
            self._apply_distance_adaptive_cutoff()   # 멀수록 더 세게 평활
            self.cursor_x_ratio = self._one_euro_x(raw_x, now_sec)
            self.cursor_y_ratio = self._one_euro_y(raw_y, now_sec)
        else:
            self.cursor_x_ratio += self._smoothing_alpha * (raw_x - self.cursor_x_ratio)
            self.cursor_y_ratio += self._smoothing_alpha * (raw_y - self.cursor_y_ratio)
        return self.cursor_x_ratio, self.cursor_y_ratio

    def _update_from_orientation(self, face, now_sec):
        """중립 대비 상대 회전으로 커서를 정한다 (orientation_mapping 전용).

        이 경로가 다른 두 경로와 근본적으로 다른 점은 **잴 것이 없다**는 것이다.

          · 카메라 배치 — 중립과 현재를 같은 카메라로 본 것끼리 비교하므로
            상대 회전에서 소거된다 (head_orientation.py 참고)
          · 부호 규약 — 방향을 중립 얼굴 자신의 축으로 읽으므로 정의상 정해진다
          · 곡률 보정 — 투영을 안 거치니 휠 것이 없다. ARC_COMPENSATION이 없다
          · 감도 — "몇 도 돌리면 화면 끝"이라는 사람 기준 값이라 카메라와 무관하다

        돌려줄 값을 못 만들면 None을 돌려준다 — 호출자가 기존 경로로 되돌아간다.
        """
        if self._orientation_calibrating:
            if self._orientation_started_sec is None:
                self._orientation_started_sec = now_sec
            self._orientation.add_calibration_sample(face)
            if now_sec - self._orientation_started_sec < self._center_calibrator.window_sec:
                return (None, None)      # 캘리브레이션 중 — 커서 미확정
            if not self._orientation.finalize_neutral():
                # 3차원 좌표가 안 들어온다 -> 이 경로를 포기하고 기존 방식으로
                self._orientation_mapping = False
                logger.info("3차원 랜드마크가 없어 상대 회전 매핑을 끕니다 - 기존 방식으로 진행합니다")
                return None
            self._orientation_calibrating = False
            logger.info("커서 중심 캘리브레이션 완료 (중립 자세 대비 상대 회전 기준)")
            self.cursor_x_ratio, self.cursor_y_ratio = 0.5, 0.5
            self._one_euro_x.prime(0.5, now_sec)
            self._one_euro_y.prime(0.5, now_sec)
            return (self.cursor_x_ratio, self.cursor_y_ratio)

        self._update_lens(face)
        offset = self._orientation.pointing_offset(face)
        if offset is None:
            # 이 프레임만 실패(너무 많이 돌렸거나 검출 불량) — 마지막 값을 유지한다.
            # 여기서 기존 경로로 넘기면 두 방식이 프레임마다 번갈아 나와 커서가 튄다
            return (self.cursor_x_ratio, self.cursor_y_ratio)

        raw_tan_x, raw_tan_y = offset
        # 잔여 곡률 자동 소거 (auto_arc.py) — 탄젠트 단계에서 뺀다.
        # 화면 비율로 바꾼 뒤에 빼면 half_span을 조절할 때마다 계수의 의미가
        # 달라지지만, 탄젠트끼리는 감도와 무관한 순수 기하 관계라 그대로 남는다
        if self._auto_arc is not None:
            raw_tan_y = self._auto_arc.update(raw_tan_x, raw_tan_y)

        # ★거리 보정 (2026-09-05) — head_orientation.py의 DISTANCE_* 설명 참고.
        #
        # 화면 끝까지 닿는 데 필요한 고개 각도는 고정이 아니다. 화면에서 Z만큼
        # 떨어져 고개를 θ 돌리면 시선이 화면에서 Z·tan(θ)만큼 움직이므로,
        # 화면 절반 폭 W/2에 닿는 각도는 atan((W/2)/Z) — **멀수록 작다.**
        # 530mm 화면이면 600mm에서 23.8도, 1300mm에서 11.5도다.
        #
        # 그런데 반폭은 15도로 고정돼 있었다. 그래서 뒤로 물러난 사람은 필요
        # 이상으로 크게 돌려야 했고(가장자리에 못 닿는다), 가까이 붙은 사람은
        # 조금만 돌려도 커서가 날아갔다. 캘리브레이션 때 대비 거리 비율을
        # 곱해 이걸 없앤다 — 비율은 얼굴 크기 비라서 초점거리도 실제 얼굴
        # 치수도 필요 없고, 가상 카메라 측정에서 오차가 0.7% 안쪽이다.
        span_scale = 1.0
        if self._orientation_distance_scaling:
            span_scale = getattr(self._orientation, "distance_ratio", 1.0) or 1.0


        # 반쪽 화면을 채우는 각도로 나눈다 -> 그 각도에서 정확히 화면 끝
        scale = span_scale * self._reach_gain
        offset_x = _clamp(raw_tan_x * scale / self._orientation_tan_x * 0.5,
                          self._max_offset_ratio)
        offset_y = _clamp(raw_tan_y * scale / self._orientation_tan_y * 0.5,
                          self._max_offset_ratio)
        raw_x, raw_y = 0.5 + offset_x, 0.5 + offset_y

        if self.cursor_x_ratio is None:
            self.cursor_x_ratio, self.cursor_y_ratio = raw_x, raw_y
            self._one_euro_x.prime(raw_x, now_sec)
            self._one_euro_y.prime(raw_y, now_sec)
            return (self.cursor_x_ratio, self.cursor_y_ratio)

        if self._one_euro_enabled:
            self.cursor_x_ratio = self._one_euro_x(raw_x, now_sec)
            self.cursor_y_ratio = self._one_euro_y(raw_y, now_sec)
        else:
            self.cursor_x_ratio += self._smoothing_alpha * (raw_x - self.cursor_x_ratio)
            self.cursor_y_ratio += self._smoothing_alpha * (raw_y - self.cursor_y_ratio)
        return (self.cursor_x_ratio, self.cursor_y_ratio)

    def _update_lens(self, face):
        """렌즈 자가 보정을 한 프레임 진행한다 (lens_calibration.py 참고).

        보정 계산 자체는 그 모듈이 딴 스레드로 돌린다 — 여기서는 랜드마크를
        넘기고, 결과가 나왔으면 한 번만 받아 적용한다.
        """
        if not self._lens_calibration_enabled or self._lens_applied:
            return
        landmarks = getattr(face, "landmarks_3d", None)
        if landmarks is None:
            return
        if self._lens_calibrator is None:
            size = getattr(face, "frame_size", None)
            if not size or len(size) != 2 or not all(size):
                return                      # 화면 크기를 모르면 중심도 모른다
            self._lens_calibrator = LensSelfCalibrator(
                size[0], size[1], distortion=self._lens_distortion_enabled)
        self._lens_calibrator.add(landmarks)
        model = self._lens_calibrator.model
        if model is not None and self._orientation is not None:
            if self._orientation.set_lens(model):
                self._lens_applied = True
                logger.info("렌즈 자가 보정 완료 — %s", model)
            else:
                # 중립을 다시 못 만들었다 — 렌즈를 물리고 예전 경로로 남는다
                self._orientation.set_lens(None)
                self._lens_applied = True

    def _update_from_head_pose(self, head_pose, now_sec):
        """머리 회전각으로 커서를 정한다 (HEAD_POSE_MAPPING 전용).

        위 update()의 랜드마크 경로와 **뒤쪽 계산이 완전히 같다** — 중앙값
        캘리브레이션 -> 중심 대비 차이 -> 민감도 -> 클램프 -> 평활. 재료를
        "화면에 투영된 좌표"에서 "회전각의 탄젠트"로 바꿨을 뿐이다.

        곡률 보정(_arc_compensation)이 여기엔 없다. 그 보정은 3차원 점이 2차원
        화면에 투영될 때 생기는 왜곡을 2차식으로 되돌리는 장치인데, 회전각은
        애초에 투영을 거치지 않아 왜곡될 것이 없다. **보정 상수를 카메라 배치마다
        다시 재야 하는 문제 자체가 사라지는 것**이 이 방식의 핵심이다.
        """
        measured = self._measure_head_pose(head_pose)
        if measured is None:
            return self.cursor_x_ratio, self.cursor_y_ratio

        center = self._center_calibrator.update(measured, now_sec)
        if center is None:
            return None, None   # 캘리브레이션 중 — 커서 미확정

        if self.cursor_x_ratio is None:
            logger.info("커서 중심 캘리브레이션 완료 (머리 회전각 기준)")
            self.cursor_x_ratio, self.cursor_y_ratio = 0.5, 0.5
            self._one_euro_x.prime(0.5, now_sec)
            self._one_euro_y.prime(0.5, now_sec)
            return self.cursor_x_ratio, self.cursor_y_ratio

        offset_x = (measured[0] - center[0]) * self._sensitivity_x
        offset_y = (measured[1] - center[1]) * self._sensitivity_y
        offset_x = _clamp(offset_x, self._max_offset_ratio)
        offset_y = _clamp(offset_y, self._max_offset_ratio)
        raw_x, raw_y = 0.5 + offset_x, 0.5 + offset_y

        if self._one_euro_enabled:
            # 거리 적응 평활은 안구간거리를 쓰는데 이 경로에선 그 값을 갱신하지
            # 않는다. 대신 자세의 tz(실제 거리)가 있으니 그걸 쓸 수 있지만,
            # 단위가 달라 그대로 넣으면 배율이 어긋난다 — 먼저 실측한 뒤에
            # 붙일 일이라 지금은 고정 평활만 쓴다
            self.cursor_x_ratio = self._one_euro_x(raw_x, now_sec)
            self.cursor_y_ratio = self._one_euro_y(raw_y, now_sec)
        else:
            self.cursor_x_ratio += self._smoothing_alpha * (raw_x - self.cursor_x_ratio)
            self.cursor_y_ratio += self._smoothing_alpha * (raw_y - self.cursor_y_ratio)
        return self.cursor_x_ratio, self.cursor_y_ratio


class _ThresholdGate:
    """블렌드셰이프 점수 1개 -> 히스테리시스 + 쿨다운으로 걸러진 확정 신호.

    open_threshold를 넘으면 확정하고 잠그며(armed=False), close_threshold 아래로
    내려와야 재장전한다 — 임계 경계에서 값이 떨릴 때 연속 확정되는 것을 막는다.

    임계값은 고정하지 않고 매 update마다 인자로 받는다 — 호출 쪽이 평상시 기준선 +
    여유값으로 계산해 넘기므로, 게이트 자체는 임계가 어디서 왔는지 몰라도 된다.
    """

    def __init__(self, cooldown_sec, clock):
        self._cooldown_sec = cooldown_sec
        self._clock = clock
        self.reset()

    def reset(self):
        self._is_armed = True
        self._last_fire_sec = None

    def update(self, score, open_threshold, close_threshold):
        """score 1건 반영 -> 이번 프레임에 확정되면 True."""
        if not self._is_armed:
            if score <= close_threshold:
                self._is_armed = True
            return False
        if score < open_threshold:
            return False
        now_sec = self._clock()
        if self._last_fire_sec is not None and now_sec - self._last_fire_sec < self._cooldown_sec:
            return False
        self._is_armed = False
        self._last_fire_sec = now_sec
        return True


class _HoldGate:
    """조건(예: 양 눈 감음)이 hold_sec 이상 연속 유지되면 확정 — "버티기" 원리.

    자연스러운 눈 깜빡임은 보통 0.4초 이하라 hold_sec(기본 0.6초)보다 짧아 걸러진다.
    조건이 풀려야(눈을 다시 뜸) 재장전한다 — 감은 채 유지해도 반복 확정되지 않는다.
    """

    def __init__(self, hold_sec, clock):
        self._hold_sec = hold_sec
        self._clock = clock
        self.reset()

    def reset(self):
        self._condition_start_sec = None
        self._is_armed = True
        self.progress_ratio = 0.0

    def update(self, is_condition_met):
        """condition 1건 반영 -> 이번 프레임에 확정되면 True."""
        if not is_condition_met:
            self._condition_start_sec = None
            self._is_armed = True
            self.progress_ratio = 0.0
            return False
        now_sec = self._clock()
        if self._condition_start_sec is None:
            self._condition_start_sec = now_sec
        elapsed_sec = now_sec - self._condition_start_sec
        self.progress_ratio = min(1.0, elapsed_sec / self._hold_sec)
        if not self._is_armed or elapsed_sec < self._hold_sec:
            return False
        self._is_armed = False
        return True


class _DwellDetector:
    """스무딩된 커서가 radius_ratio 안에 dwell_sec 머물면 확정 (응시 클릭)."""

    def __init__(self, radius_ratio, dwell_sec, require_release_to_rearm, clock):
        self._radius_ratio = radius_ratio
        self._dwell_sec = dwell_sec
        self._require_release_to_rearm = require_release_to_rearm
        self._clock = clock
        self.reset()

    def reset(self):
        self._anchor = None
        self._anchor_start_sec = None
        self._is_armed = True
        self.progress_ratio = 0.0

    def update(self, cursor_x_ratio, cursor_y_ratio):
        if cursor_x_ratio is None:
            self.reset()
            return False
        point = (cursor_x_ratio, cursor_y_ratio)

        if self._anchor is None or _dist(point, self._anchor) > self._radius_ratio:
            self._anchor = point
            self._anchor_start_sec = self._clock()
            if self._require_release_to_rearm:
                self._is_armed = True   # 반경을 벗어났다 — 재장전
            self.progress_ratio = 0.0
            return False

        elapsed_sec = self._clock() - self._anchor_start_sec
        self.progress_ratio = min(1.0, elapsed_sec / self._dwell_sec)
        if not self._is_armed or elapsed_sec < self._dwell_sec:
            return False
        if self._require_release_to_rearm:
            self._is_armed = False   # 반경 이탈 전까지 재발화 금지
        else:
            self._anchor_start_sec = self._clock()   # 즉시 재시작(연속 재발화 허용 모드)
        return True


class HeadTracker:
    """얼굴 랜드마크 1프레임 -> HeadTrackerResult (커서 + select/home/calibration 이벤트)."""

    def __init__(self, config, clock=time.monotonic, cursor_point_fn=None):
        """cursor_point_fn(face) -> (x_px, y_px) — 커서 위치의 기준점을 얼굴에서
        어떻게 뽑을지 바꿀 수 있다(2026-08-13 신설, eyebrow.py 대응 — head.py는
        기존 그대로 코끝, eyebrow.py는 미간/양쪽 눈 사이). 생략하면(None) 기존과
        완전히 동일하게 코끝(LMK_NOSE_TIP)을 쓴다 — 이 매개변수를 안 넘기는
        기존 호출부(main.py, main_dpad.py)는 동작이 전혀 바뀌지 않는다. 나머지
        판정 로직(EMA·안구간거리 정규화·select/home/calibration)은 기준점이
        무엇이든 완전히 동일하게 재사용된다 — 코 대신 다른 점을 쓴다고 해서
        이 클래스를 통째로 새로 만들 이유가 없다.
        """
        ht = config["head_tracker"]
        self._clock = clock
        self._cursor_point_fn = cursor_point_fn or (lambda face: face.landmark_px(LMK_NOSE_TIP))
        calibration_window_sec = ht["calibration_window_sec"]

        pointer = ht["pointer"]
        # 화면의 실제 크기 — 설정에 없으면 운영체제(모니터 EDID)에서 읽는다.
        # 픽셀 해상도로는 알 수 없다: 1920x1080이 14인치일 수도 55인치일
        # 수도 있고 물리 크기는 4배 차이다 (display_size.py 참고)
        screen_w = pointer.get("screen_width_mm")
        screen_h = pointer.get("screen_height_mm")
        if pointer.get("reference_distance_mm") and not (screen_w and screen_h):
            detected = detect_screen_size_mm()
            if detected:
                screen_w, screen_h = detected
        self._cursor_mapper = _CursorMapper(
            calibration_window_sec, pointer["sensitivity_x"], pointer["sensitivity_y"],
            pointer["smoothing_alpha"], pointer["distance_smoothing_alpha"],
            pointer["max_offset_ratio"],
            # 얼굴 좌표계 기준 매핑 — _CursorMapper 독스트링 참고. 키가 없는
            # 기존 설정 파일도 그대로 돌아가도록 기본값을 둔다
            face_local=pointer.get("face_local", True),
            face_local_gain=pointer.get("face_local_gain", 2.0),
            # 1€ 필터 — OneEuroFilter·_CursorMapper.__init__ 독스트링 참고.
            # 기본 False — 키가 없는 기존 설정(head.py·eyebrow.py)은 동작 변화 없다
            one_euro_enabled=pointer.get("one_euro_enabled", False),
            one_euro_min_cutoff=pointer.get("one_euro_min_cutoff", 1.0),
            one_euro_beta=pointer.get("one_euro_beta", 0.0),
            one_euro_distance_adaptive=pointer.get("one_euro_distance_adaptive", False),
            one_euro_reference_dist_px=pointer.get("one_euro_reference_dist_px", 60.0),
            arc_compensation=pointer.get("arc_compensation", 0.0),
            # ★각도 기반 매핑 — HEAD_POSE_MAPPING 설명 참고. 기본 False라
            # 이 키가 없는 기존 설정은 동작이 그대로다
            head_pose_mapping=pointer.get("head_pose_mapping", False),
            # ★상대 회전 매핑 (2026-08-31) — head_orientation.py 참고.
            # 기본 False라 이 키가 없는 기존 설정은 동작이 그대로다.
            # 켜면 감도(sensitivity_x/y)와 곡률 보정(arc_compensation)이 함께
            # 무시된다 — 이 경로는 각도로 직접 매핑해서 둘 다 필요 없다
            orientation_mapping=pointer.get("orientation_mapping", False),
            orientation_distance_scaling=pointer.get("orientation_distance_scaling", True),
            orientation_reach_gain=pointer.get("orientation_reach_gain", 1.0),
            # 설치 치수 — 있으면 반폭을 기하학으로 계산한다 (위 설명 참고).
            # 화면 크기를 안 적어 놨으면 운영체제에서 알아본다 —
            # 이게 있어야 데스크탑과 키오스크가 같은 빌드로 돈다
            screen_width_mm=screen_w,
            screen_height_mm=screen_h,
            reference_distance_mm=pointer.get("reference_distance_mm"),
            orientation_half_span_x_deg=pointer.get("orientation_half_span_x_deg", 15.0),
            orientation_half_span_y_deg=pointer.get("orientation_half_span_y_deg", 10.0),
            orientation_rotation_source=pointer.get("orientation_rotation_source", "auto"),
            orientation_auto_arc=pointer.get("orientation_auto_arc", True),
            # 렌즈 자가 보정 (lens_calibration.py) — 광각 렌즈의 배럴 왜곡과
            # 원근 단축을 사용자의 얼굴만으로 되돌린다. 기본 켬이고,
            # 못 믿을 상황에서는 스스로 아무것도 하지 않는다
            orientation_lens_calibration=pointer.get(
                "orientation_lens_calibration", True),
            # 왜곡 되돌리기는 기본 끔 — 카메라를 아는 배포처에서만 켠다
            orientation_lens_distortion=pointer.get(
                "orientation_lens_distortion", False),
        )

        mouth = ht["mouth_click"]
        self._mouth_enabled = mouth["enabled"]
        self._mouth_open_margin = mouth["open_margin"]
        self._mouth_close_margin = mouth["close_margin"]
        self._jaw_baseline = _MedianCalibrator(calibration_window_sec)
        # 자체 쿨다운 없음(0.0) — 재발화 방지는 히스테리시스 + 아래 공용 클릭 쿨다운이 담당
        self._mouth_gate = _ThresholdGate(0.0, clock)

        eye_close = ht["eye_close_home"]
        self._eye_close_margin = eye_close["close_margin"]
        self._eye_baseline = _MedianCalibrator(calibration_window_sec)
        self._eye_close_gate = _HoldGate(eye_close["hold_sec"], clock)

        recenter = ht["recenter_gesture"]
        self._recenter_enabled = recenter["enabled"]
        self._recenter_open_margin = recenter["open_margin"]
        self._recenter_close_margin = recenter["close_margin"]
        self._mouth_pucker_baseline = _MedianCalibrator(calibration_window_sec)
        self._recenter_gate = _ThresholdGate(recenter["cooldown_sec"], clock)

        dwell = ht["dwell_click"]
        self._dwell_enabled = dwell["enabled"]
        self._dwell_detector = _DwellDetector(
            dwell["radius_ratio"], dwell["dwell_sec"], dwell["require_release_to_rearm"], clock,
        )

        # 응시 -> calibration(2026-07-30 신설 — 입 오므리기 오탐 대체). select용
        # dwell_detector와 별개 인스턴스라 서로 독립적으로 진행·확정된다(모듈 독스트링 참고)
        recenter_dwell = ht["recenter_dwell"]
        self._recenter_dwell_enabled = recenter_dwell["enabled"]
        self._recenter_dwell_detector = _DwellDetector(
            recenter_dwell["radius_ratio"], recenter_dwell["dwell_sec"],
            recenter_dwell["require_release_to_rearm"], clock,
        )

        self._click_min_interval_sec = ht["click"]["min_interval_sec"]
        self._last_click_sec = None
        # 얼굴이 언제부터 안 보이는지 (FACE_LOST_RESET_SEC 설명 참고).
        # __init__은 reset()을 부르지 않으므로 여기서 직접 만들어 둔다 —
        # 안 그러면 **첫 프레임에 얼굴이 없을 때** 속성이 없어 죽는다
        # (2026-08-31: 실제로 그 경로를 밟아 AttributeError를 봤다)
        self._face_lost_since_sec = None
        self.debug = {}   # 실기 튜닝 계기판 — 디버그 창에 노출 (판정에 미사용)

    def update(self, face):
        """얼굴 신호 1프레임 -> HeadTrackerResult (기획서 4.6 계약).

        face가 None(미검출)이면 **잠깐은 기다렸다가** 리셋한다 — 아래 설명 참고.
        """
        if face is None:
            now_sec = self._clock()
            if self._face_lost_since_sec is None:
                self._face_lost_since_sec = now_sec
            # ★2026-08-31 — 한 프레임 놓쳤다고 캘리브레이션을 버리지 않는다.
            #
            # 예전에는 face가 None인 프레임마다 곧바로 reset()을 불렀다. 얼굴이
            # 잠깐씩 끊기는 조건(카메라를 기울여 단 배치, 역광, 사용자가 살짝
            # 벗어남)에서는 이것이 **리셋 -> 재캘리브레이션 -> 또 리셋**의
            # 되풀이가 되어, 캘리브레이션이 끝나질 않고 커서가 영영 안 나온다.
            # 밖에서 보면 "커서가 아예 안 움직인다"로만 보인다 — 원인을 찾기
            # 어려운 부류다(2026-08-31 실기 보고: "옆으로 살짝 기운 카메라는
            # 커서가 아예 안 움직인다").
            #
            # 한 프레임 빠진 것은 **사람이 바뀐 것이 아니다.** 이만큼 이어서
            # 없을 때만 새 사용자로 보고 처음부터 다시 잡는다. 그 사이에는
            # 지금까지 잡아 둔 중립·기준선을 그대로 들고 기다린다.
            if now_sec - self._face_lost_since_sec < FACE_LOST_RESET_SEC:
                self._update_debug(None, None)
                return HeadTrackerResult(is_tracking=False, events=[])
            self.reset()
            self._update_debug(None, None)
            return HeadTrackerResult(is_tracking=False, events=[])
        self._face_lost_since_sec = None

        now_sec = self._clock()
        jaw_open_score = face.blendshape("jawOpen")
        # 둘 다 감아야 인정 — 한쪽 윙크(의도적 신호 아님)로 오발화하지 않게, 윙크를
        # 못 하는 사용자도 조작 가능하게 (프로토타입 2026-07-18 3차 설계)
        eye_close_score = min(
            face.blendshape("eyeBlinkLeft"), face.blendshape("eyeBlinkRight")
        )
        mouth_pucker_score = face.blendshape("mouthPucker")

        cursor_source_px = self._cursor_point_fn(face)   # 기본: 코끝 — __init__ 참고
        # 양쪽 눈 바깥쪽 끝 두 점 = 얼굴 좌표계의 가로축이자 길이 자(尺).
        # _CursorMapper가 이 두 점으로 "몸이 움직인 것"과 "고개를 돌린 것"을
        # 구분한다 — 그 독스트링 참고
        eye_left_px = face.landmark_px(LMK_LEFT_EYE_OUTER)
        eye_right_px = face.landmark_px(LMK_RIGHT_EYE_OUTER)
        cursor_x, cursor_y = self._cursor_mapper.update(
            cursor_source_px, eye_left_px, eye_right_px, now_sec,
            head_pose=getattr(face, "head_pose", None), face=face)
        # 코 캘리브레이션과 같은 구간에서 입/눈/오므림 평상시 기준선도 함께 잡는다
        jaw_baseline = self._jaw_baseline.update(jaw_open_score, now_sec)
        eye_baseline = self._eye_baseline.update(eye_close_score, now_sec)
        mouth_pucker_baseline = self._mouth_pucker_baseline.update(mouth_pucker_score, now_sec)

        events = self._detect_events(
            cursor_x, cursor_y, jaw_open_score, jaw_baseline, eye_close_score, eye_baseline,
            mouth_pucker_score, mouth_pucker_baseline, now_sec,
        )

        self._update_debug(
            cursor_x, cursor_y, jaw_open_score, jaw_baseline, eye_close_score, eye_baseline,
            mouth_pucker_score, mouth_pucker_baseline,
        )
        return HeadTrackerResult(
            cursor_x_ratio=cursor_x, cursor_y_ratio=cursor_y,
            is_tracking=cursor_x is not None, events=events,
        )

    def _detect_events(self, cursor_x, cursor_y, jaw_open_score, jaw_baseline,
                       eye_close_score, eye_baseline, mouth_pucker_score,
                       mouth_pucker_baseline, now_sec):
        events = []
        # 기준선이 아직 안 잡혔으면(캘리브레이션 중) 입/눈 판정은 보류 — 커서와 동일한 전제
        if self._mouth_enabled and jaw_baseline is not None:
            if self._mouth_gate.update(jaw_open_score,
                                       jaw_baseline + self._mouth_open_margin,
                                       jaw_baseline + self._mouth_close_margin):
                event = self._try_confirm_select(jaw_open_score, "mouth", now_sec)
                if event is not None:
                    events.append(event)
        if self._dwell_enabled and self._dwell_detector.update(cursor_x, cursor_y):
            event = self._try_confirm_select(1.0, "dwell", now_sec)
            if event is not None:
                events.append(event)
        if eye_baseline is not None:
            is_eye_closed = eye_close_score >= eye_baseline + self._eye_close_margin
            if self._eye_close_gate.update(is_eye_closed):
                logger.info("gesture_event: %s (trigger=eye_close, conf=%.2f)",
                            EVENT_HOME, eye_close_score)
                events.append(GestureEvent(
                    class_name=EVENT_HOME, conf=eye_close_score, ts_sec=now_sec,
                    data={"trigger": "eye_close"},
                ))
        if self._recenter_enabled and mouth_pucker_baseline is not None:
            if self._recenter_gate.update(mouth_pucker_score,
                                          mouth_pucker_baseline + self._recenter_open_margin,
                                          mouth_pucker_baseline + self._recenter_close_margin):
                # 커서 중심만 다시 잡는다 — 입/눈 기준선까지 건드리면 재정렬 직후
                # 잠깐 select/home이 먹통이 되는 불필요한 부작용이 생긴다
                self._cursor_mapper.reset()
                logger.info("gesture_event: %s (trigger=mouth_pucker, conf=%.2f)",
                            EVENT_CALIBRATION, mouth_pucker_score)
                events.append(GestureEvent(
                    class_name=EVENT_CALIBRATION, conf=mouth_pucker_score, ts_sec=now_sec,
                    data={"trigger": "mouth_pucker"},
                ))
        if self._recenter_dwell_enabled and self._recenter_dwell_detector.update(cursor_x, cursor_y):
            self._cursor_mapper.reset()
            logger.info("gesture_event: %s (trigger=dwell, conf=%.2f)", EVENT_CALIBRATION, 1.0)
            events.append(GestureEvent(
                class_name=EVENT_CALIBRATION, conf=1.0, ts_sec=now_sec,
                data={"trigger": "dwell"},
            ))
        return events

    def _try_confirm_select(self, conf, trigger, now_sec):
        """입 벌리기·응시 공용 쿨다운 — 같은 프레임에 둘 다 충족해도 select는 1개만."""
        if (self._last_click_sec is not None
                and now_sec - self._last_click_sec < self._click_min_interval_sec):
            return None
        self._last_click_sec = now_sec
        logger.info("gesture_event: %s (trigger=%s, conf=%.2f)", EVENT_SELECT, trigger, conf)
        return GestureEvent(class_name=EVENT_SELECT, conf=conf, ts_sec=now_sec,
                            data={"trigger": trigger})

    def set_pointer_tuning(self, sensitivity_x=None, sensitivity_y=None, arc_compensation=None,
                           half_span_x_deg=None, half_span_y_deg=None):
        """_CursorMapper.set_tuning() 그대로 전달 — 실시간 조절 UI가 이
        메서드 하나만 알면 되게 하려고 얇게 감쌌다(그 메서드 설명 참고)."""
        self._cursor_mapper.set_tuning(
            sensitivity_x=sensitivity_x, sensitivity_y=sensitivity_y,
            arc_compensation=arc_compensation,
            half_span_x_deg=half_span_x_deg, half_span_y_deg=half_span_y_deg)

    @property
    def is_orientation_mapping(self):
        """상대 회전 경로로 도는가 — 조절 UI가 손잡이를 고를 때 본다."""
        return self._cursor_mapper.is_orientation_mapping

    def reset(self):
        """추적 끊김·모드 전환 — 커서 캘리브레이션·기준선·클릭 상태 전부 리셋."""
        self._face_lost_since_sec = None
        self._cursor_mapper.reset()
        self._jaw_baseline.reset()
        self._eye_baseline.reset()
        self._mouth_pucker_baseline.reset()
        self._mouth_gate.reset()
        self._eye_close_gate.reset()
        self._recenter_gate.reset()
        self._dwell_detector.reset()
        self._recenter_dwell_detector.reset()
        self._last_click_sec = None

    def reset_event_gates(self):
        """select/home/calibration 판정 게이트만 리셋 — 커서 캘리브레이션·
        기준선은 그대로 둔다(2026-08-14 신설, head.py/eyebrow.py 대응 — 그
        실행기들의 "캘리브레이션 직후 몇 초는 조작을 안 받는다" 정지 유예
        기능이 이 메서드로 매 프레임 게이트를 계속 비워서 만들어진다).

        reset()과의 차이: reset()은 커서 위치까지 통째로 지워 처음부터 다시
        캘리브레이션하게 만들지만, 이건 커서 추적은 그대로 진행시키면서
        "지금까지 응시했던·입 벌렸던 시간"만 매번 지운다 — 그래서 이 메서드를
        연속 호출하는 동안은 어떤 판정도 완성될 수 없다(매 프레임 그 진행이
        지워지므로). 호출을 멈추면 다음 프레임부터 게이트가 정상적으로 다시
        진행을 쌓기 시작한다 — 정지 유예가 끝나는 순간 밀린 판정이 한꺼번에
        발화하는 일 없이 깨끗하게 새로 시작한다.
        """
        self._mouth_gate.reset()
        self._eye_close_gate.reset()
        self._recenter_gate.reset()
        self._dwell_detector.reset()
        self._recenter_dwell_detector.reset()

    def reset_recenter_dwell(self):
        """자동 재정렬(캘리브레이션) 대기 시계 **하나만** 0으로 되돌린다
        (2026-08-20 신설 — head.py/eyebrow.py의 _reset_recenter_timer 대응).

        reset_event_gates()와의 차이가 이 메서드의 존재 이유다. 처음엔 클릭이
        일어날 때 reset_event_gates()를 불렀는데, 그건 응시 클릭 게이트까지
        같이 지워서 두 가지가 한꺼번에 망가졌다:

          ① 재정렬이 영영 안 된다 — 가만히 있으면 1.5초마다 응시 클릭이 나가고,
             그 클릭이 다시 재정렬 시계를 0으로 되돌려서 10초를 채울 수가 없다.
          ② 응시 클릭이 무한 반복된다 — require_release_to_rearm(확정 후 반경을
             벗어나야 재장전)이 걸려 있는데도 reset()이 재장전 상태로 되돌려
             버려서, 가만히 있기만 하면 클릭이 계속 나갔다(실측: 27초에 13번).

        그래서 "조작 중이니 재정렬은 미루자"는 원래 의도에 필요한 것만 — 재정렬
        시계만 — 되돌린다. 나머지 판정은 건드리지 않는다."""
        self._recenter_dwell_detector.reset()

    def recenter_cursor(self, face):
        """커서 중심을 지금 이 순간의 기준점(코끝, 또는 cursor_point_fn 지정 지점)
        위치로 즉시 재설정한다 (2026-08-18 신설, 사용자 요청 — "캘리브레이션
        되면 백그라운드 커서 값도 무조건 사용자가 어떤 자세를 취하던 커서가
        중앙에 가있게 고정해줘").

        reset()과 다르다 — reset()은 calibration_window_sec만큼 다시 표본을
        모아야 해서 그동안 is_tracking이 끊긴다(이 프로젝트에서는 그게 곧
        "얼굴 새로 잡힘" 이벤트라 head.py/eyebrow.py의 정지 유예가 통째로
        다시 시작돼버린다 — 이 메서드를 정지 유예 "끝" 시점에 부르면서 그런
        재귀적 재유예를 피하려고 만들어졌다). recenter_cursor는 재수집 없이
        바로 확정하므로 is_tracking이 끊기지 않고, 입/눈/입 오므림 기준선도
        건드리지 않는다(select/home 판정에 영향 없음).

        head.py/eyebrow.py는 정지 유예(SETTLE_DELAY_SEC)가 끝나는 정확히 그
        순간 이걸 한 번 호출한다 — "화면 정면을 보고 편한 자세로 있어달라"는
        안내대로 그 5초 동안 자리 잡은 실제 자세를, 원래 캘리브레이션 순간의
        자세 대신 새 중심으로 확정해서 유예 종료 시점에 커서가 정확히 화면
        중앙에서 시작하도록 보장한다.

        성공하면 True, 얼굴이 없거나 검출이 불량해 재중심을 못 잡았으면 False를
        돌려준다 — 호출자가 "했다"고 표시해 버리면 그 유예 사이클 내내 다시
        시도하지 않아, 사용자가 요청한 "유예가 끝나면 커서는 무조건 중앙"이
        조용히 깨진다(2026-08-20 추가).
        """
        if face is None:
            return False
        return self._cursor_mapper.recenter_instant(
            self._cursor_point_fn(face),
            face.landmark_px(LMK_LEFT_EYE_OUTER), face.landmark_px(LMK_RIGHT_EYE_OUTER),
            face=face)

    def _update_debug(self, cursor_x, cursor_y, jaw_open_score=0.0, jaw_baseline=None,
                      eye_close_score=0.0, eye_baseline=None,
                      mouth_pucker_score=0.0, mouth_pucker_baseline=None):
        self.debug = {
            "mode": "head",
            "cursor_x": None if cursor_x is None else round(cursor_x, 3),
            "cursor_y": None if cursor_y is None else round(cursor_y, 3),
            "jaw_open": round(jaw_open_score, 2),
            "jaw_base": None if jaw_baseline is None else round(jaw_baseline, 2),
            "eye_close": round(eye_close_score, 2),
            "eye_base": None if eye_baseline is None else round(eye_baseline, 2),
            "eye_progress": round(self._eye_close_gate.progress_ratio, 2),
            "dwell_progress": round(self._dwell_detector.progress_ratio, 2),
            "recenter_progress": round(self._recenter_dwell_detector.progress_ratio, 2),
            "pucker": round(mouth_pucker_score, 2),
            "pucker_base": None if mouth_pucker_baseline is None else round(mouth_pucker_baseline, 2),
        }
