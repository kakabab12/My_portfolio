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
import time
from dataclasses import dataclass, field

from src.inference.face_estimator import (
    LMK_LEFT_EYE_OUTER, LMK_NOSE_TIP, LMK_RIGHT_EYE_OUTER,
)
from src.postprocess.gesture_filter import GestureEvent
from src.utils.logger import get_logger

logger = get_logger("postprocess")

MIN_INTEROCULAR_DIST_PX = 10.0  # 이보다 좁으면(검출 불량) 정규화 자로 못 쓴다 — 마지막 값 유지

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


class _MedianCalibrator:
    """calibration_window_sec 동안 표본을 모아 중앙값을 낸다 — 캘리브레이션 공용 로직.

    코 위치(화면 중심)·입 벌림/눈 감김 평상시 점수(판정 기준선)가 전부 이 패턴을
    쓴다 — "잠금 직후 짧은 구간을 관찰해 그걸 기준으로 삼는다"는 셋의 본질이 같다.
    한 번 확정(value가 설정)되면 이후 표본은 무시한다.
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

    gesture_filter는 "이벤트 1개 또는 None"을 반환하지만, 헤드트래커는 매 프레임
    연속 커서 상태를 UI가 그려야 해서 이 결과 구조로 확장한다.
    """

    cursor_x_ratio: float = None
    cursor_y_ratio: float = None
    is_tracking: bool = False
    events: list = field(default_factory=list)


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
        """새 사용자·추적 끊김·재정렬 — 캘리브레이션부터 다시 한다."""
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
        # 같은 민감도면 상하가 쉽게 끝까지 튄다 (프로토타입 실기: 상단에 눌러붙는 현상)
        offset_x = _clamp(dx * self._sensitivity_x, self._max_offset_ratio)
        offset_y = _clamp(dy * self._sensitivity_y, self._max_offset_ratio)
        raw_x, raw_y = 0.5 + offset_x, 0.5 + offset_y

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
        self.debug = {}   # 실기 튜닝 계기판 — 디버그 창에 노출 (판정에 미사용)

    def update(self, face):
        """얼굴 신호 1프레임 -> HeadTrackerResult (기획서 4.6 계약).

        face가 None(미검출)이면 전체를 리셋한다 — 다시 나타나면 캘리브레이션부터.
        """
        if face is None:
            self.reset()
            self._update_debug(None, None)
            return HeadTrackerResult(is_tracking=False, events=[])

        now_sec = self._clock()
        jaw_open_score = face.blendshape("jawOpen")
        # 둘 다 감아야 인정 — 한쪽 윙크(의도적 신호 아님)로 오발화하지 않게, 윙크를
        # 못 하는 사용자도 조작 가능하게 (프로토타입 2026-07-18 3차 설계)
        eye_close_score = min(
            face.blendshape("eyeBlinkLeft"), face.blendshape("eyeBlinkRight")
        )
        mouth_pucker_score = face.blendshape("mouthPucker")

        nose_px = face.landmark_px(LMK_NOSE_TIP)
        interocular_dist_px = _dist(
            face.landmark_px(LMK_LEFT_EYE_OUTER), face.landmark_px(LMK_RIGHT_EYE_OUTER)
        )
        cursor_x, cursor_y = self._cursor_mapper.update(nose_px, interocular_dist_px, now_sec)
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

    def reset(self):
        """추적 끊김·모드 전환 — 커서 캘리브레이션·기준선·클릭 상태 전부 리셋."""
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
