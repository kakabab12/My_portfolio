"""postprocess 모듈 — 얼굴 랜드마크 신호를 커서 위치와 동작 이벤트로 변환한다.

동작 체계(2026-07-18 헤드트래커 전환 — head_tracker_manual.md 기반, 팔 쓸기 폐기):
- 커서 이동   : 코끝(랜드마크 1) 위치를 화면 커서로 매핑. 잠금 직후 짧은 구간
                중앙값으로 중심을 캘리브레이션하고, 안구간 거리로 매 프레임 정규화
                (카메라 거리 무관 — 기존 어깨너비 정규화와 같은 원리), EMA로 떨림 평활
- select     : 입 벌리기(jawOpen 블렌드셰이프) 즉시, 또는 커서가 반경 안에 1.5초
                머무르는 응시(dwell) — 어느 쪽이든 같은 이벤트, 공용 쿨다운으로
                동시 발화 시 1건만 확정
- go_back    : 양 눈 감고 0.6초 버티기(eyeBlinkLeft/Right 둘 다 기준선 이상 유지) —
                "처음부터"도 겸함(별도 go_home 없음 — 기존 UI가 go_back을 모든 화면에서
                홈으로 처리). 자연스러운 깜빡임(보통 0.4초 이하)은 hold_sec보다 짧아 걸러진다.
- recenter   : 입 오므리기(mouthPucker 블렌드셰이프) — 커서 중심을 지금 고개
                위치로 다시 잡는다(2026-07-21 신설). person_lock 재잠금(자리 이탈)이
                유일한 재캘리브레이션 경로였던 게 접근성 문제였다(docs/TODO.md №1-신).
                ⚠ 제스처 변천(2026-07-21, 같은 날 2차례 교체): 원안 볼 부풀리기
                (cheekPuff) → 코 찡그리기(noseSneerLeft/Right, max) 모두 실기에서
                인식이 거의 안 됐다(사용자 실측) — MediaPipe 블렌드셰이프 모델이 이런
                희소 표정 신호를 약하게 낸다고 추정. mouthPucker는 원래 go_back 후보
                였다가 select(jawOpen)와 같은 입 근육이라 **연속 전환 중** 신호가
                겹쳐 폐기된 이력이 있지만(설치가이드.md E절 참고), 그건 select와
                빠르게 번갈아 쓰는 동작이라 문제였던 것 — recenter는 드물고 독립적인
                동작이라 같은 위험이 훨씬 작고, 무엇보다 그 폐기 이력 자체가
                MediaPipe가 이 신호는 강하게 잡는다는 실증이라 지금 상황(미인식)에
                필요한 신호 강도가 있다.

2026-07-20 정확도 개선: 입벌림·눈감김 판정을 **고정 임계값**에서 **잠금 직후 캡처한
평상시(기준선) + 여유값**으로 바꿨다. 실기 관찰 결과 사람마다 평상시 eyeBlink
블렌드셰이프가 0.1~0.6까지 편차가 커서(얼굴 형태·카메라 각도·조명 영향), 고정
임계값 하나로는 어떤 사용자는 오탐(평상시에도 넘음)이, 어떤 사용자는 미탐(감아도
못 넘음)이 났다. 코끝 커서 캘리브레이션과 완전히 같은 원리(같은 구간에서 함께
수집)라 추가 대기 시간이 들지 않는다. 잠금 직후 표정을 찡그리고 있으면 기준선이
왜곡되지만, 코 캘리브레이션도 같은 전제를 이미 깔고 있어(§_MedianCalibrator) 새로
생긴 약점은 아니다.

블렌드셰이프는 MediaPipe FaceLandmarker가 이미 0~1로 정규화·학습해 내보내는 신호라
매뉴얼의 랜드마크 거리비(입술 13/14, 눈썹 105/334) 계산을 직접 짤 필요가 없다 —
얼굴 형태·거리 편차에 더 강건하다 (2026-07-18 설계 결정).

2026-07-22 아이트래킹 모드 추가: 코끝 기준 헤드트래커(head)는 그대로 두고, 양쪽
눈 홍채가 눈꺼풀 경계 안에서 차지하는 상대 위치를 커서로 쓰는 시선 모드(eye)를
나란히 추가했다 — 데모 UI 버튼(POST /set_cursor_mode)으로 실행 중 전환한다.
두 모드는 "원시 신호 -> 캘리브레이션 -> EMA 평활 -> 커서 비율"이라는 같은 골격을
쓰지만 정규화 방식이 달라(코는 안구간 거리로 나눈 픽셀 오프셋, 시선은 눈 코너/
눈꺼풀 기준 비율이라 이미 스케일 불변) 별도 클래스(_GazeCursorMapper)로 분리했다.
입 벌리기/눈 감기/입 오므리기 판정은 모드와 무관하게 항상 동작한다(커서 이동
방식만 바뀐다). ⚠ 시선 비율은 완전한 3D 시선벡터가 아니라 2D 근사라 고개를 크게
돌리면 원근이 약간 섞인다 — 정면을 향한 채 눈만 움직이는 사용을 전제로 한다.
이 세션은 실기 카메라로 검증할 수 없었다 — gaze_pointer 민감도는 미검증 시작값.

모든 수치는 config head_tracker에서 읽는다 (기획서 4.7).
"""
import time
from dataclasses import dataclass, field

from src.inference.face_estimator import (
    LMK_LEFT_EYE_BOTTOM, LMK_LEFT_EYE_INNER, LMK_LEFT_EYE_OUTER, LMK_LEFT_EYE_TOP,
    LMK_LEFT_IRIS_CENTER, LMK_NOSE_TIP, LMK_RIGHT_EYE_BOTTOM, LMK_RIGHT_EYE_INNER,
    LMK_RIGHT_EYE_OUTER, LMK_RIGHT_EYE_TOP, LMK_RIGHT_IRIS_CENTER,
)
from src.postprocess.gesture_event import GestureEvent
from src.utils.logger import get_logger

logger = get_logger("postprocess")

MIN_INTEROCULAR_DIST_PX = 10.0  # 이보다 좁으면(검출 불량) 정규화 자로 못 쓴다 — 마지막 값 유지
MIN_EYE_SPAN_PX = 4.0  # 이보다 좁으면(검출 불량) 그 눈은 시선 비율 계산에서 제외 — 다른 쪽만 사용

CURSOR_MODE_HEAD = "head"   # 코끝 위치 기준(기존)
CURSOR_MODE_EYE = "eye"     # 양쪽 눈 시선 비율 기준(2026-07-22 신설)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _median(samples):
    """스칼라 또는 (x, y) 튜플 표본 목록 -> 중앙값(튜플이면 요소별로).

    평균 대신 중앙값인 이유: 캘리브레이션 구간 중 순간적인 과도기 프레임(눈 깜빡임·
    고개 숙임 등) 1~2장이 섞여도 평균처럼 기준 전체가 끌려가지 않는다
    (2026-07-18 실기 — 커서가 화면 상단에 눌러붙던 문제의 원인이었다).
    """
    if isinstance(samples[0], tuple):
        return tuple(_median(list(dim_samples)) for dim_samples in zip(*samples))
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def _iris_gaze_ratio(iris_px, corner_a_px, corner_b_px, top_px, bottom_px):
    """홍채 중심의 눈꺼풀 경계 내 상대 위치(0~1 근방) — 아이트래킹 모드의 원시 시선 신호.

    corner_a/b는 어느 쪽이 안쪽/바깥쪽 코너인지 몰라도 되게 min/max로 처리한다 —
    좌우 눈이 코너 순서가 반대라 부호를 직접 계산하면 실수하기 쉽다(2026-07-22).
    검출 불량(폭·높이가 거의 0)이면 None — 다른 쪽 눈으로 대체하라는 신호.
    """
    x_min, x_max = sorted((corner_a_px[0], corner_b_px[0]))
    y_min, y_max = sorted((top_px[1], bottom_px[1]))
    span_x_px, span_y_px = x_max - x_min, y_max - y_min
    if span_x_px < MIN_EYE_SPAN_PX or span_y_px < MIN_EYE_SPAN_PX:
        return None
    return (iris_px[0] - x_min) / span_x_px, (iris_px[1] - y_min) / span_y_px


def _gaze_point(locked_face):
    """양쪽 눈 시선 비율의 평균 -> 시선 좌표 1점 (한쪽만 불량이면 다른 쪽만, 둘 다면 None)."""
    left = _iris_gaze_ratio(
        locked_face.landmark_px(LMK_LEFT_IRIS_CENTER),
        locked_face.landmark_px(LMK_LEFT_EYE_OUTER), locked_face.landmark_px(LMK_LEFT_EYE_INNER),
        locked_face.landmark_px(LMK_LEFT_EYE_TOP), locked_face.landmark_px(LMK_LEFT_EYE_BOTTOM),
    )
    right = _iris_gaze_ratio(
        locked_face.landmark_px(LMK_RIGHT_IRIS_CENTER),
        locked_face.landmark_px(LMK_RIGHT_EYE_OUTER), locked_face.landmark_px(LMK_RIGHT_EYE_INNER),
        locked_face.landmark_px(LMK_RIGHT_EYE_TOP), locked_face.landmark_px(LMK_RIGHT_EYE_BOTTOM),
    )
    if left is None:
        return right
    if right is None:
        return left
    return (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0


class _MedianCalibrator:
    """calibration_window_sec 동안 표본을 모아 중앙값을 낸다 — 캘리브레이션 공용 로직.

    코 위치(화면 중심)·입 벌림/눈 감김 평상시 점수(판정 기준선)가 전부 이 패턴을
    쓴다(2026-07-20 통합) — "잠금 직후 짧은 구간을 관찰해 그걸 기준으로 삼는다"는
    셋의 본질이 같기 때문. 한 번 확정(value가 설정)되면 이후 표본은 무시한다.
    """

    def __init__(self, window_sec):
        self._window_sec = window_sec
        self.reset()

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


@dataclass
class HeadTrackerResult:
    """head_tracker.update() 1회 호출의 결과 — 연속 커서 상태 + 이산 이벤트 묶음.

    기존 gesture_filter는 "이벤트 1개 또는 None"을 반환했지만, 헤드트래커는 매 프레임
    연속 커서 상태를 UI가 그려야 해서 이 결과 구조로 확장한다.
    """

    cursor_x_ratio: float = None
    cursor_y_ratio: float = None
    is_tracking: bool = False
    events: list = field(default_factory=list)


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class _CursorMapper:
    """코끝 위치 -> 화면 커서 비율. 잠금 직후 1회 캘리브레이션 + 안구간거리 정규화 + EMA."""

    def __init__(self, calibration_window_sec, sensitivity_x, sensitivity_y, smoothing_alpha,
                 distance_smoothing_alpha, max_offset_ratio):
        self._sensitivity_x = sensitivity_x
        self._sensitivity_y = sensitivity_y
        self._smoothing_alpha = smoothing_alpha
        self._distance_smoothing_alpha = distance_smoothing_alpha
        self._max_offset_ratio = max_offset_ratio
        self._center_calibrator = _MedianCalibrator(calibration_window_sec)
        self.reset()

    def reset(self):
        """새 잠금(lock_generation 변경)·추적 끊김 — 캘리브레이션부터 다시 한다."""
        self._center_calibrator.reset()
        self._smoothed_dist_px = None
        self.cursor_x_ratio = None
        self.cursor_y_ratio = None

    def update(self, nose_px, interocular_dist_px, now_sec):
        if interocular_dist_px < MIN_INTEROCULAR_DIST_PX:
            return self.cursor_x_ratio, self.cursor_y_ratio   # 검출 불량 — 마지막 값 유지

        if self._smoothed_dist_px is None:
            self._smoothed_dist_px = interocular_dist_px
        else:
            self._smoothed_dist_px += self._distance_smoothing_alpha * (
                interocular_dist_px - self._smoothed_dist_px
            )

        center_px = self._center_calibrator.update(nose_px, now_sec)
        if center_px is None:
            return None, None   # 캘리브레이션 중 — 커서 미확정

        if self.cursor_x_ratio is None:
            # 캘리브레이션이 막 끝난 프레임 — 정의상 화면 중앙에서 시작한다
            logger.info("커서 중심 캘리브레이션 완료")
            self.cursor_x_ratio, self.cursor_y_ratio = 0.5, 0.5
            return self.cursor_x_ratio, self.cursor_y_ratio

        dx = (nose_px[0] - center_px[0]) / self._smoothed_dist_px
        dy = (nose_px[1] - center_px[1]) / self._smoothed_dist_px
        # y는 x보다 낮은 민감도를 쓴다 — 고개는 좌우 회전 범위가 상하보다 훨씬 넓어
        # 같은 민감도면 상하가 쉽게 끝까지 튄다 (2026-07-18 실기: 상단에 눌러붙는 현상)
        offset_x = _clamp(dx * self._sensitivity_x, self._max_offset_ratio)
        offset_y = _clamp(dy * self._sensitivity_y, self._max_offset_ratio)
        raw_x, raw_y = 0.5 + offset_x, 0.5 + offset_y

        self.cursor_x_ratio += self._smoothing_alpha * (raw_x - self.cursor_x_ratio)
        self.cursor_y_ratio += self._smoothing_alpha * (raw_y - self.cursor_y_ratio)
        return self.cursor_x_ratio, self.cursor_y_ratio


class _GazeCursorMapper:
    """시선 비율(양쪽 눈 평균) -> 화면 커서 비율. _CursorMapper와 같은 캘리브레이션+EMA
    골격이지만, 입력이 이미 눈 코너/눈꺼풀 기준 비율이라 안구간거리 같은 별도
    정규화가 필요 없다(스스로 이미 스케일 불변) — 그래서 별도 클래스로 분리했다
    (모듈 docstring 2026-07-22 항목 참고).

    2026-07-23 x/y 평활을 분리했다(x/y 각각 다른 smoothing_alpha) — 실기에서 상하만
    떨림이 심하다고 보고됐다. 상하 시선 비율의 분모(눈꺼풀 뜬 높이, _iris_gaze_ratio
    참고)가 좌우 분모(눈 코너 간격)보다 훨씬 좁아 같은 크기의 랜드마크 검출 잡음이라도
    상하 비율에서 훨씬 크게 증폭된다고 추정 — sensitivity_y를 올리는 시도(5.0→9.0)는
    "진짜 신호"와 "잡음"을 같이 키울 뿐이라 떨림만 심해지고 반응성은 안 나아졌다
    (사용자 실측). 그래서 이번엔 배율이 아니라 y축 평활을 더 강하게 걸어 잡음을
    누르는 쪽으로 방향을 바꿨다 — 반응이 그만큼 느려지는 대가가 있다.
    """

    def __init__(self, calibration_window_sec, sensitivity_x, sensitivity_y,
                 smoothing_alpha_x, smoothing_alpha_y, max_offset_ratio):
        self._sensitivity_x = sensitivity_x
        self._sensitivity_y = sensitivity_y
        self._smoothing_alpha_x = smoothing_alpha_x
        self._smoothing_alpha_y = smoothing_alpha_y
        self._max_offset_ratio = max_offset_ratio
        self._center_calibrator = _MedianCalibrator(calibration_window_sec)
        self.reset()

    def reset(self):
        """새 잠금·모드 전환·추적 끊김 — 캘리브레이션부터 다시 한다(_CursorMapper와 동일 계약)."""
        self._center_calibrator.reset()
        self.cursor_x_ratio = None
        self.cursor_y_ratio = None

    def update(self, gaze_point, now_sec):
        if gaze_point is None:
            return self.cursor_x_ratio, self.cursor_y_ratio   # 검출 불량 — 마지막 값 유지

        center = self._center_calibrator.update(gaze_point, now_sec)
        if center is None:
            return None, None   # 캘리브레이션 중 — 커서 미확정

        if self.cursor_x_ratio is None:
            logger.info("시선 커서 중심 캘리브레이션 완료")
            self.cursor_x_ratio, self.cursor_y_ratio = 0.5, 0.5
            return self.cursor_x_ratio, self.cursor_y_ratio

        offset_x = _clamp((gaze_point[0] - center[0]) * self._sensitivity_x, self._max_offset_ratio)
        offset_y = _clamp((gaze_point[1] - center[1]) * self._sensitivity_y, self._max_offset_ratio)
        raw_x, raw_y = 0.5 + offset_x, 0.5 + offset_y

        self.cursor_x_ratio += self._smoothing_alpha_x * (raw_x - self.cursor_x_ratio)
        self.cursor_y_ratio += self._smoothing_alpha_y * (raw_y - self.cursor_y_ratio)
        return self.cursor_x_ratio, self.cursor_y_ratio


class _ThresholdGate:
    """블렌드셰이프 점수 1개 -> 히스테리시스 + 쿨다운으로 걸러진 확정 신호.

    open_threshold를 넘으면 확정하고 잠그며(armed=False), close_threshold 아래로
    내려와야 재장전(rearm)한다 — 임계 경계에서 값이 떨릴 때 연속 확정되는 것을 막는다.
    cooldown_sec은 재장전과 별개로 확정 자체의 최소 간격 (0이면 히스테리시스만 적용).

    임계값은 고정하지 않고 매 update 호출마다 인자로 받는다(2026-07-20) — 호출 쪽이
    평상시 기준선(baseline) + 여유값으로 매번 계산해서 넘기므로, 게이트 자체는
    임계가 어디서 왔는지 몰라도 된다 (관심사 분리).
    """

    def __init__(self, cooldown_sec, clock):
        self._cooldown_sec = cooldown_sec
        self._clock = clock
        self._is_armed = True
        self._last_fire_sec = None

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
    """조건(예: 양 눈 감음)이 hold_sec 이상 연속 유지되면 확정 — 응시 클릭과 같은 "버티기" 원리.

    자연스러운 눈 깜빡임은 보통 0.4초 이하라 hold_sec(기본 0.6초)보다 짧아 걸러진다 —
    의도적으로 오래 감아야만 확정된다. 조건이 풀려야(눈을 다시 뜸) 재장전한다 — 감은
    채 유지해도 반복 확정되지 않는다. 재발화에는 어차피 매번 hold_sec 전체가 다시
    필요해 별도 쿨다운은 불필요(_DwellDetector와 같은 설계).
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
    def __init__(self, config, clock=time.monotonic):
        ht = config["head_tracker"]
        self._clock = clock
        calibration_window_sec = ht["calibration_window_sec"]

        pointer = ht["pointer"]
        self._cursor_mapper = _CursorMapper(
            calibration_window_sec, pointer["sensitivity_x"], pointer["sensitivity_y"],
            pointer["smoothing_alpha"], pointer["distance_smoothing_alpha"],
            pointer["max_offset_ratio"],
        )

        gaze_pointer = ht["gaze_pointer"]
        self._gaze_cursor_mapper = _GazeCursorMapper(
            calibration_window_sec, gaze_pointer["sensitivity_x"], gaze_pointer["sensitivity_y"],
            gaze_pointer["smoothing_alpha_x"], gaze_pointer["smoothing_alpha_y"],
            gaze_pointer["max_offset_ratio"],
        )
        self.cursor_mode = ht["cursor_mode"]   # 실행 중 set_cursor_mode()로 전환 (공개 속성 — /data status)
        self._pending_cursor_mode = None       # 교차 스레드 전환 요청 대기칸 (set_cursor_mode 참고)

        mouth = ht["mouth_click"]
        self._mouth_enabled = mouth["enabled"]
        self._mouth_open_margin = mouth["open_margin"]
        self._mouth_close_margin = mouth["close_margin"]
        self._jaw_baseline = _MedianCalibrator(calibration_window_sec)
        # 자체 쿨다운 없음(0.0) — 재발화 방지는 히스테리시스 + 아래 공용 클릭 쿨다운이 담당
        self._mouth_gate = _ThresholdGate(0.0, clock)

        eye_close = ht["eye_close_cancel"]
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

        self._click_min_interval_sec = ht["click"]["min_interval_sec"]
        self._last_click_sec = None
        self._last_lock_generation = None
        self.debug = {}   # 실기 튜닝 계기판 — /data·화면 오버레이로 노출 (판정에 미사용)

    def set_cursor_mode(self, mode):
        """커서 입력 모드를 코(head)/시선(eye)으로 전환한다 — demo_server.py의
        POST /set_cursor_mode(데모 UI 버튼 클릭)가 호출한다.

        실제 전환은 다음 update() 프레임에서 반영한다(즉시 반영 안 함) — 이 메서드는
        FastAPI 스레드에서, update()는 추론 스레드에서 돌아 서로 다른 스레드다. 매퍼
        상태(reset 등)를 교차 스레드로 직접 건드리는 대신 문자열 하나만 건네고(단순
        속성 대입은 GIL 하에 원자적) 실제 reset은 추론 스레드 자신이 처리하게 한다.
        """
        if mode not in (CURSOR_MODE_HEAD, CURSOR_MODE_EYE):
            raise ValueError(f"알 수 없는 커서 모드: {mode}")
        self._pending_cursor_mode = mode

    @property
    def _active_mapper(self):
        return self._cursor_mapper if self.cursor_mode == CURSOR_MODE_HEAD else self._gaze_cursor_mapper

    def _apply_pending_cursor_mode(self):
        if self._pending_cursor_mode != self.cursor_mode:
            self.cursor_mode = self._pending_cursor_mode
            self._active_mapper.reset()   # 꺼져있던 동안 쌓인 기준이 없어 새로 캘리브레이션
            logger.info("커서 모드 전환: %s", self.cursor_mode)
        self._pending_cursor_mode = None

    def _update_cursor(self, locked_face, now_sec):
        """활성 모드(코/시선)에 맞는 매퍼만 갱신한다 — 비활성 매퍼는 건드리지 않는다."""
        if self.cursor_mode == CURSOR_MODE_HEAD:
            nose_px = locked_face.landmark_px(LMK_NOSE_TIP)
            interocular_dist_px = _dist(
                locked_face.landmark_px(LMK_LEFT_EYE_OUTER), locked_face.landmark_px(LMK_RIGHT_EYE_OUTER)
            )
            return self._cursor_mapper.update(nose_px, interocular_dist_px, now_sec)
        return self._gaze_cursor_mapper.update(_gaze_point(locked_face), now_sec)

    def update(self, locked_face, lock_generation):
        """얼굴 신호 1프레임 -> HeadTrackerResult (기획서 4.6 계약)."""
        if self._pending_cursor_mode is not None:
            self._apply_pending_cursor_mode()

        if locked_face is None:
            self._reset_all()
            self._last_lock_generation = None
            self._update_debug(None, None)
            return HeadTrackerResult(is_tracking=False, events=[])

        if lock_generation != self._last_lock_generation:
            self._reset_all()
            self._last_lock_generation = lock_generation

        now_sec = self._clock()
        jaw_open_score = locked_face.blendshape("jawOpen")
        # 둘 다 감아야 인정 — 한쪽 윙크(의도적 신호 아님)로 오발화하지 않게, 윙크를
        # 못 하는 사용자도 조작 가능하게(2026-07-18 3차 설계)
        eye_close_score = min(
            locked_face.blendshape("eyeBlinkLeft"), locked_face.blendshape("eyeBlinkRight")
        )
        mouth_pucker_score = locked_face.blendshape("mouthPucker")

        cursor_x, cursor_y = self._update_cursor(locked_face, now_sec)
        # 코 캘리브레이션과 같은 구간에서 입/눈/오므림 평상시 기준선도 함께 잡는다 (모듈 docstring 참고)
        jaw_baseline = self._jaw_baseline.update(jaw_open_score, now_sec)
        eye_baseline = self._eye_baseline.update(eye_close_score, now_sec)
        mouth_pucker_baseline = self._mouth_pucker_baseline.update(mouth_pucker_score, now_sec)

        events = self._detect_events(
            cursor_x, cursor_y, jaw_open_score, jaw_baseline, eye_close_score, eye_baseline,
            mouth_pucker_score, mouth_pucker_baseline, now_sec,
        )

        self._update_debug(
            cursor_x, cursor_y, jaw_open_score, jaw_baseline, eye_close_score, eye_baseline,
            mouth_pucker_score, mouth_pucker_baseline, locked_face.blendshapes,
        )
        return HeadTrackerResult(
            cursor_x_ratio=cursor_x, cursor_y_ratio=cursor_y,
            is_tracking=cursor_x is not None, events=events,
        )

    def _detect_events(self, cursor_x, cursor_y, jaw_open_score, jaw_baseline,
                        eye_close_score, eye_baseline, mouth_pucker_score, mouth_pucker_baseline, now_sec):
        events = []
        # 기준선이 아직 안 잡혔으면(캘리브레이션 중) 입/눈/코 판정은 보류 — 커서와 동일한 전제
        if self._mouth_enabled and jaw_baseline is not None:
            open_threshold = jaw_baseline + self._mouth_open_margin
            close_threshold = jaw_baseline + self._mouth_close_margin
            if self._mouth_gate.update(jaw_open_score, open_threshold, close_threshold):
                event = self._try_confirm_click(jaw_open_score, "mouth", now_sec)
                if event is not None:
                    events.append(event)
        if self._dwell_enabled and self._dwell_detector.update(cursor_x, cursor_y):
            event = self._try_confirm_click(1.0, "dwell", now_sec)
            if event is not None:
                events.append(event)
        if eye_baseline is not None:
            is_eye_closed = eye_close_score >= eye_baseline + self._eye_close_margin
            if self._eye_close_gate.update(is_eye_closed):
                events.append(GestureEvent(
                    class_name="go_back", conf=eye_close_score, ts_sec=now_sec,
                    data={"trigger": "eye_close"},
                ))
        if self._recenter_enabled and mouth_pucker_baseline is not None:
            open_threshold = mouth_pucker_baseline + self._recenter_open_margin
            close_threshold = mouth_pucker_baseline + self._recenter_close_margin
            if self._recenter_gate.update(mouth_pucker_score, open_threshold, close_threshold):
                # 커서 중심만 다시 잡는다 — 입/눈 기준선까지 건드리면 재정렬 직후
                # 잠깐 select/go_back이 먹통이 되는 불필요한 부작용이 생긴다.
                # 활성 모드(코/시선)의 매퍼만 리셋 — 비활성 쪽은 건드릴 필요 없다
                self._active_mapper.reset()
                logger.info("gesture_event: recenter (trigger=mouth_pucker, conf=%.2f)", mouth_pucker_score)
                events.append(GestureEvent(
                    class_name="recenter", conf=mouth_pucker_score, ts_sec=now_sec,
                    data={"trigger": "mouth_pucker"},
                ))
        return events

    def _try_confirm_click(self, conf, trigger, now_sec):
        """입 벌리기·dwell 공용 쿨다운 — 같은 프레임에 둘 다 충족해도 select는 1개만."""
        if (self._last_click_sec is not None
                and now_sec - self._last_click_sec < self._click_min_interval_sec):
            return None
        self._last_click_sec = now_sec
        logger.info("gesture_event: select (trigger=%s, conf=%.2f)", trigger, conf)
        return GestureEvent(class_name="select", conf=conf, ts_sec=now_sec, data={"trigger": trigger})

    def _reset_all(self):
        """추적 끊김·새 사용자 잠금 — 커서 캘리브레이션·기준선·클릭 상태 전부 리셋.

        모드 전환과 달리 여기서는 두 매퍼(코·시선) 다 리셋한다 — 완전히 새 추적
        세션이라 나중에 어느 모드로 전환되든 캘리브레이션이 낡아있으면 안 된다.
        """
        self._cursor_mapper.reset()
        self._gaze_cursor_mapper.reset()
        self._jaw_baseline.reset()
        self._eye_baseline.reset()
        self._mouth_pucker_baseline.reset()
        self._mouth_gate.reset()
        self._eye_close_gate.reset()
        self._recenter_gate.reset()
        self._dwell_detector.reset()
        self._last_click_sec = None

    def _update_debug(self, cursor_x, cursor_y, jaw_open_score=0.0, jaw_baseline=None,
                       eye_close_score=0.0, eye_baseline=None,
                       mouth_pucker_score=0.0, mouth_pucker_baseline=None, all_blendshapes=None):
        self.debug = {
            "cursor_mode": self.cursor_mode,
            "cursor_x_ratio": None if cursor_x is None else round(cursor_x, 3),
            "cursor_y_ratio": None if cursor_y is None else round(cursor_y, 3),
            "jaw_open_score": round(jaw_open_score, 2),
            "jaw_baseline": None if jaw_baseline is None else round(jaw_baseline, 2),
            "eye_close_score": round(eye_close_score, 2),
            "eye_baseline": None if eye_baseline is None else round(eye_baseline, 2),
            "eye_close_progress_ratio": round(self._eye_close_gate.progress_ratio, 2),
            "dwell_progress_ratio": round(self._dwell_detector.progress_ratio, 2),
            "mouth_pucker_score": round(mouth_pucker_score, 2),
            "mouth_pucker_baseline": None if mouth_pucker_baseline is None else round(mouth_pucker_baseline, 2),
            # 2026-07-21 2차: 제스처 후보를 두 번 연속 잘못 골라(cheekPuff·noseSneer
            # 모두 실기 미인식) 추가 — 전체 블렌드셰이프 원점수를 노출해 다음부터는
            # 추측이 아니라 실측으로 후보를 고른다 (판정에는 미사용, 진단 전용)
            "all_blendshapes": {k: round(v, 2) for k, v in (all_blendshapes or {}).items()},
        }
