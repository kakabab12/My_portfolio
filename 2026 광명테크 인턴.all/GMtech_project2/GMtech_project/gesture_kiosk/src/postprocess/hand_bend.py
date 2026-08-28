"""postprocess 모듈 — 손 꺾임(포스처) 판: 정지한 손의 자세가 곧 명령이다.

2026-08-05 신설 → 08-06 실기 세션(계측 주도) 반복 개정 — **8차 확정**. 사용자
요구사항 그대로 — 이 목록에 없는 판정 경로는 이 모듈에 존재하지 않는다:
- 펼친 손 = 화면 4방위 섹터(±45·±135) **그대로**: 좌 left · 우 right ·
  **위 home · 아래 back** (8차 매핑 — 매핑 자체는 config postures가 정한다).
  ★7차(사용자 확정 "왼 오 하도 어차피 침범하는데 상만 다르니 안 되는 것 —
  판정 똑같게 만들라"): 위 방향만 높이(어깨선)로 판정하던 비대칭 폐기 —
  네 방향 전부 같은 각도 판정. 구 위 방향 특례 계보 전부 소멸(1차 평평 →
  2D 띠 → z 정적 → z 스냅 → 높이 0.9→0.55) — z·높이는 계측 표시로만 남는다.
  중립(open_flat)·히스테리시스(22~40도 보류)도 소멸 — 펼친 손은 항상
  4방위 중 하나다.
- **주먹·검지 주먹(포인팅)도 같은 4방위** (9차 — 사용자 지정 "손가락있는
  주먹 상은 select, 나머지와 주먹은 temp"): 모양이 계열(접두)을, 섹터가
  방향(접미)을 정한다 — 자세 이름 = finger_up/…/fist_up/… .
  검지 주먹 위 = select · 검지 주먹 하/좌/우와 주먹 4방위 = temp_*
  (예비 동작 표시 — 델파이가 받는 실제 이벤트). 8차 "주먹 = select 한
  자세"는 폐기. ★평평 주먹(수직 ±fist_flat_max_deg — 그냥 쥔 주먹) =
  **temp**(무방향, 9차 보강 — 사용자 지정 "그냥 주먹 평평도 temp로"):
  평평 존은 주먹에만 있다 — 검지·펼침은 위 방향이 본 제스처(select·
  home)라 평평 존이 그걸 잘라먹으면 안 된다.
  ★temp 계열은 발화해도 쿨다운을 걸지 않는다 — 예비 자세
  직후의 본 제스처(예: temp → select)가 쿨다운에 씹히면 안 된다.
  뒤집기 폐기(8차)는 유지 — 손등/손바닥 분리·다수결 없음, 외적 부호는
  계측(F=)만.
- **OK 사인 = confirm** (8차 신설 — 사용자 지정 "ok 손표시 confirm"):
  엄지·검지 끝 고리 + 중지~새끼 폄 (hand_shape.is_ok_sign — 판별 근거는
  그쪽 독스트링). 방향 무관. ※섹터 판정보다 먼저 물어야 한다 — OK는
  모양 판별(classify)에 SHAPE_OPEN으로 읽히기 때문이다.
  ★손바닥쪽 구제(같은 날 — 사용자 제안): 월드 3D 고리가 검지 환각으로
  부풀 때 화면(2D) 고리 거리로 한 번 더 판별(_is_ok_posture).
- 이동(쓸기)·로터를 전부 대체: gesture_filter가 낸 이벤트는 여기서 전부
  삼킨다 — 이 판의 파이프 출력은 위 6종뿐이다.
- 재발화 방어: 발화 1회 래치(다른 자세 경유 후 재장전) + 유지 시간
  (hold_sec, 주먹·OK는 flat_hold_sec) + 쿨다운.

판정 원리 (손 21점 단독 — 포즈 팔목·손목·팔꿈치는 어디에도 안 쓴다):
- 손 방향 = 손목(0) → 중지 뿌리(9)의 화면 각도, 기준은 화면 수직(12시) 고정.
- 펼친 손 = compass_posture 4방위 섹터가 곧 자세 — 네 방향 전부 같은 판정.
- 자세는 유지 시간(hold_sec, 주먹은 flat_hold_sec)을 채워야 발화. 같은
  자세는 1회만 — 재발화는 다른 자세를 거친 뒤에만.

수치는 전부 config hand_bend 섹션 (기획서 4.7 — 숫자 하드코딩 금지).
판별 임계(펼침/주먹)는 본 엔진 판별(hand_select.hand_shape)과 같은 값 재사용.
"""
import math
import time

from src.postprocess.gesture_filter import GestureEvent
from src.postprocess.hand_shape import (
    SHAPE_FINGER, SHAPE_FIST, SHAPE_OPEN, STATE_EXTENDED, classify_hand_shape,
    finger_states, hand_center_point, is_ok_sign, ok_touch_ratio,
    ok_touch_ratio_2d,
)
from src.postprocess.rotor import clock_angle_deg
from src.utils.logger import get_logger

logger = get_logger("postprocess")

WRIST_LMK = 0        # MediaPipe 21점 중 손목 (hand_tracker.py 독스트링)
INDEX_MCP_LMK = 5    # 검지 뿌리 — 손바닥 평면의 자(尺): 주먹 손등/손바닥 판별
MIDDLE_MCP_LMK = 9   # 중지 뿌리(MCP) — 손등 축의 끝점 (모양 무관 존재)
PINKY_MCP_LMK = 17   # 새끼 뿌리 — 손바닥 평면의 자

NONE_GRACE_SEC = 0.4   # 판별 불가(블러)·손 소실을 자세 유지로 봐주는 시간 —
                       #   swipe.dropout_grace_sec(0.35)와 같은 취지: 짧은 끊김이
                       #   유지 시간 계수를 리셋해 발화가 씹히는 것 방지. 초과하면
                       #   자세를 처음부터 다시 잡는다(재발화 허용)
# 자세 이름 — config hand_bend.postures의 키와 계약. 매핑에 없는 자세는
# 중립(발화 없음). ※open_flat은 2026-08-06 7차에서, fist_back/fist_palm 분리는
# 8차(주먹 뒤집기 폐기)에서, 방향 무관 fist 단일 자세는 9차에서 소멸.
# 방향 자세 이름 = 계열_방향: bend_*(펼친 손) · finger_*(검지 주먹) ·
# fist_*(주먹) — 방향은 전부 같은 compass 섹터에서 나온다 (9차)
POSTURE_BEND_LEFT = "bend_left"
POSTURE_BEND_RIGHT = "bend_right"
POSTURE_BEND_UP = "bend_up"
POSTURE_BEND_DOWN = "bend_down"
POSTURE_OK = "ok_sign"      # OK 사인 — 엄지·검지 고리 + 중지~새끼 폄 (8차)
POSTURE_FIST_FLAT = "fist_flat"   # 그냥 쥔 평평 주먹(수직 근방) — temp (9차 보강)
DIRECTION_SUFFIXES = {POSTURE_BEND_UP: "up", POSTURE_BEND_RIGHT: "right",
                      POSTURE_BEND_LEFT: "left", POSTURE_BEND_DOWN: "down"}
FLAT_POSTURES = (POSTURE_OK,)   # 모양 자세(방향 무관) — 긴 유지(flat_hold_sec)
                            #   대상: 전환 중 지나가기 쉬운 자세라 방향 자세
                            #   (hold_sec)보다 길게 잡는다. 9차부터 OK뿐 — 주먹은
                            #   방향 자세(temp 계열)가 되어 hold_sec을 쓴다


def wrap180_deg(angle_deg):
    """각도를 [-180, 180) 범위로 접는다 — 꺾임각의 부호 있는 표현."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def compass_posture(hand_deg):
    """손이 가리키는 화면 방향의 4방위 자세 — 90도 섹터 (0=12시·시계+)."""
    folded_deg = wrap180_deg(hand_deg)
    if -45.0 <= folded_deg < 45.0:
        return POSTURE_BEND_UP
    if 45.0 <= folded_deg < 135.0:
        return POSTURE_BEND_RIGHT
    if -135.0 <= folded_deg < -45.0:
        return POSTURE_BEND_LEFT
    return POSTURE_BEND_DOWN


def fist_palm_sign(world_landmarks):
    """손바닥 평면 외적의 정규화 값 -> float(-1~1) | None(퇴화) — 판별·계기판 공용.

    두 벡터(손목→검지 뿌리, 손목→새끼 뿌리)의 화면(xy) 외적을 두 벡터 크기로
    나눈 값(사잇각의 sin). 부호가 손등/손바닥을 가르고, 크기가 판별 신뢰다 —
    계기판(FIST 줄)이 이 값을 그대로 보여 실기에서 임계·부호를 검증한다.
    """
    wrist = world_landmarks[WRIST_LMK]
    index_vec = (float(world_landmarks[INDEX_MCP_LMK][0] - wrist[0]),
                 float(world_landmarks[INDEX_MCP_LMK][1] - wrist[1]))
    pinky_vec = (float(world_landmarks[PINKY_MCP_LMK][0] - wrist[0]),
                 float(world_landmarks[PINKY_MCP_LMK][1] - wrist[1]))
    norm_product = math.hypot(*index_vec) * math.hypot(*pinky_vec)
    if norm_product <= 0.0:
        return None
    return (index_vec[0] * pinky_vec[1] - index_vec[1] * pinky_vec[0]) / norm_product


def z_tilt_sin(world_landmarks):
    """손목→중지 뿌리 선의 카메라 축 기울기 -> sin(-1~1) | None(퇴화).

    select(손목 위로 젖히기)의 판정 신호 — 젖히면 이 선이 카메라 축으로
    기울어 |z|가 커진다(중립 |z| ≤ ~0.4). ★부호는 못 믿는다(2026-08-06
    14:10 계측: 같은 동작이 -0.88 ↔ +0.87 — MediaPipe 월드 재구성이 손잡이
    해석에 따라 거울로 뒤집힘) — 그래서 판정은 양쪽 부호 비대칭 임계를 쓴다
    (config z_select_fire_neg/pos, 실측 근거 포함).
    """
    wrist = world_landmarks[WRIST_LMK]
    knuckle = world_landmarks[MIDDLE_MCP_LMK]
    span_m = math.dist((float(wrist[0]), float(wrist[1]), float(wrist[2])),
                       (float(knuckle[0]), float(knuckle[1]), float(knuckle[2])))
    if span_m <= 0.0:
        return None
    return float(knuckle[2] - wrist[2]) / span_m


# ※is_fist_palm_facing(주먹 손등/손바닥 분리)·PALM_SIGN_MIN_SIN·손잡이 다수결
#   (SIDE_VOTE_WINDOW)은 2026-08-06 8차 **소멸**(사용자 지정 — "주먹은 select,
#   기존 주먹 뒤집기 폐기"): 주먹이 한 자세가 되면서 분리할 것이 없다.
#   외적 부호(fist_palm_sign)는 계측(F=)으로만 남긴다. 구현·부호 규약 이력은
#   git(2차 4a98f97 · 6차 d606c18 직전)에 보존.


class HandBendController:
    """손 꺾임 판정기 — 쓸기 이벤트를 삼키고 자세 확정 이벤트를 돌려준다."""

    def __init__(self, config, clock=time.monotonic):
        bend_cfg = config["hand_bend"]
        # ※구 판정 관문 키 3종(flat_max_deg·bend_min_deg·raise_above_shoulder)은
        #   2026-08-06 7차 소멸 — 섹터 경계(±45·±135)는 compass_posture의 기하
        #   상수다 (config hand_bend 섹션의 키 소멸 주석 참고)
        self._hold_sec = bend_cfg["hold_sec"]
        self._flat_hold_sec = bend_cfg["flat_hold_sec"]
        self._cooldown_sec = bend_cfg["cooldown_sec"]
        self._events = dict(bend_cfg["postures"])
        self._ok_touch_palm_ratio = bend_cfg["ok_touch_palm_ratio"]
        # 손바닥쪽 OK 구제 임계(화면 2D) — 키 없으면 구제 경로 없음(3D 단독)
        self._ok_touch_screen_ratio = bend_cfg.get("ok_touch_screen_ratio")
        self._fist_flat_max_deg = bend_cfg["fist_flat_max_deg"]
        # 손 모양 판별 임계 — 본 엔진 판별(hand_select.hand_shape)과 같은 값 재사용
        shape_cfg = config["hand_select"]["hand_shape"]
        self._extend_ratio = shape_cfg["extend_ratio"]
        self._min_valid_fingers = shape_cfg["min_valid_fingers"]
        self._curl_confirm_ratio = shape_cfg["curl_confirm_ratio"]
        self._clock = clock

        self._candidate = None          # 유지 시간을 재는 중인 자세 이름 | None
        self._candidate_since_sec = None
        self._last_match_sec = None     # 후보 자세가 마지막으로 관측된 시각 — 블러 유예
        self._fired_posture = None      # 발화한 자세 — 다른 자세를 거쳐야 재장전
        self._cooldown_until_sec = 0.0
        self._debug_fist_sign = None    # 계기판(FIST) — 최근 주먹 프레임의 외적 부호값
        self._debug_ok_ratio = None     # 계측 — 엄지·검지 고리 닫힘 비율 (OK 임계 튜닝)
        self._debug_ok2_ratio = None    # 계측 — 고리 닫힘의 화면(2D) 비율: 손바닥 OK
                                        #   진단(ok_touch_ratio_2d 독스트링) — 판정 미사용
        self._debug_index_ratio = None  # 계측 — 검지 폄 비율: 손바닥 OK에서 검지가
                                        #   진짜 폄(1.37+)인지 환각(중간값)인지 실측
        self._debug_shape = None        # 계측 — 이번 프레임 모양 판별 결과
        self._debug_z = None            # 계측 — z 젖힘 sin (표시 전용)
        self._debug_raise = None        # 계측 — 손 높이(어깨선 위 어깨너비 배수)
        self._last_measure_sec = 0.0    # 계측 로그 스로틀 — 0.5초당 1줄(파일 전용)
        self.debug = {}                 # 계기판(visualize) — bend_deg·자세·유지 진행도

    # ----- 파이프라인 접점 -----

    def update(self, gesture_event, hand_selector):
        """쓸기 이벤트를 삼키고 추적 손 하나를 판정 — 종전 단일 경로.

        gesture_filter의 이벤트는 종류 무관 전부 소비된다 — 이 판의 입력은
        자세뿐이다(모듈 독스트링). ★2026-08-06 10차부터 실배포 경로는
        중재기(hand_arbiter)가 judge()를 직접 부른다 — 이 래퍼는 단독 사용·
        기존 테스트 호환용으로 남는다.
        """
        if gesture_event is not None:
            logger.info("쓸기 이벤트 무시(꺾임 판): %s", gesture_event.class_name)
        return self.judge(hand_selector.tracked_hand(), hand_selector)

    def judge(self, hand, hand_selector):
        """손 하나(None=미관측)를 자세 상태기로 판정 -> 이벤트 | None.

        ★2026-08-06 10차(자세 획득 — 사용자 확정 "중재기 교체"): 중재기가
        경쟁 슬롯마다 판정기 인스턴스를 하나씩 쥐고 이 진입점을 부른다 —
        잠금 전 두 손이 각자의 상태기로 나란히 판정되고, **먼저 발화한 손이
        곧 획득(잠금)**이다. 손이 움직여야 잡히던 이동 획득 전제(쓸기 시대
        유산)가 이것으로 끝난다: 정지한 손도 자세만 보이면 잡힌다.
        hand=None은 관측 없음 — 소실 기계(NONE_GRACE_SEC)가 돈다.
        hand_selector는 어깨선(anchor_shoulder_frame — 계측 up=) 공급용.
        """
        now_sec = self._clock()
        posture, bend_deg = (None, None) if hand is None else (
            self._classify_posture(hand, hand_selector))
        if hand is not None and now_sec - self._last_measure_sec >= 0.5:
            # ★계측(2026-08-06 실기 세션): 판별 내부값을 0.5초당 1줄 파일 로그로
            # 남긴다 — 임계(ok_touch_palm_ratio 등)를 감이 아니라 데이터로 정한다.
            # up·z·F는 폐기된 구 판정 경로의 잔여 계측 — 실기 재검증용 표시 전용.
            # ok2(화면 2D 고리)·idx(검지 폄 비율)는 손바닥 OK 진단(같은 날 실기 —
            # 손바닥쪽 OK가 open으로 환각돼 home 발화): 손바닥 OK ↔ 진짜 home을
            # 가르는 특징이 있는지 이 두 값의 분포로 판단한다
            self._last_measure_sec = now_sec
            logger.info(
                "계측: shape=%s deg=%s up=%s z=%s F=%s ok=%s ok2=%s idx=%s -> %s",
                self._debug_shape or "-",
                "-" if bend_deg is None else f"{bend_deg:.0f}",
                "-" if self._debug_raise is None else f"{self._debug_raise:+.2f}",
                "-" if self._debug_z is None else f"{self._debug_z:+.2f}",
                "-" if self._debug_fist_sign is None else f"{self._debug_fist_sign:+.2f}",
                "-" if self._debug_ok_ratio is None else f"{self._debug_ok_ratio:.2f}",
                "-" if self._debug_ok2_ratio is None else f"{self._debug_ok2_ratio:.2f}",
                "-" if self._debug_index_ratio is None else f"{self._debug_index_ratio:.2f}",
                posture or "-")
        if posture is None:
            # 판별 불가·손 소실 — 유예 안이면 후보 유지(발화는 안 한다),
            # 넘기면 처음부터 (재등장 자세는 새 자세 — 재발화 허용)
            if (self._candidate is not None and self._last_match_sec is not None
                    and now_sec - self._last_match_sec > NONE_GRACE_SEC):
                self._reset_posture()
            self._write_debug(bend_deg, now_sec)
            return None

        if posture != self._candidate:
            if self._fired_posture is not None and posture != self._fired_posture:
                self._fired_posture = None   # 다른 자세 확인 — 재장전
            self._candidate = posture
            self._candidate_since_sec = now_sec
        self._last_match_sec = now_sec
        self._write_debug(bend_deg, now_sec)

        hold_sec = (self._flat_hold_sec if posture in FLAT_POSTURES
                    else self._hold_sec)
        if (posture == self._fired_posture
                or now_sec - self._candidate_since_sec < hold_sec
                or now_sec < self._cooldown_until_sec):
            return None
        event_name = self._events.get(posture)
        if event_name is None:
            # 매핑 없는 자세 = 중립(발화 없음) — 계약 유지: postures에서 줄을
            # 빼면 그 자세를 끌 수 있다 (기본 매핑은 6자세 전부 매핑돼 있다)
            return None
        self._fired_posture = posture
        if not event_name.startswith("temp"):
            # ★temp 계열(예비 동작 표시 — 9차)은 쿨다운을 걸지 않는다: 예비
            # 자세 직후의 본 제스처(temp_up → select 등)가 쿨다운에 씹히면
            # 안 된다. temp도 진행 중인 쿨다운은 존중한다(위 발화 조건)
            self._cooldown_until_sec = now_sec + self._cooldown_sec
        logger.info("꺾임 발화: %s -> %s (꺾임각 %.0f도)", posture, event_name,
                    0.0 if bend_deg is None else bend_deg)
        return GestureEvent(class_name=event_name, conf=1.0, ts_sec=now_sec)

    # ----- 자세 판정 -----

    def _classify_posture(self, hand, hand_selector):
        """손 1프레임의 자세 -> (자세 이름 | None, 꺾임각 | None).

        None = 단정하지 않는다: 모양 불명(블러)·한 손가락·옆면 주먹 —
        어느 쪽도 이벤트 근거가 아니다.
        hand_selector: 어깨선(anchor_shoulder_frame) — 계측(up= 표시) 전용:
        2026-08-06 7차부터 판정은 포즈 관측을 전혀 안 쓴다.
        """
        self._debug_fist_sign = None   # 계기판용 — 주먹 프레임에서만 채워진다
        # 계측: OK 고리 닫힘 비율 — 임계(ok_touch_palm_ratio) 실기 튜닝의 근거
        self._debug_ok_ratio = ok_touch_ratio(hand.world_landmarks)
        # 계측: 손바닥 OK 진단 2종 — 화면(2D) 고리 비율 + 검지 폄 비율
        # (손바닥쪽 OK의 open 환각을 가를 특징 탐색 — ok_touch_ratio_2d 독스트링)
        self._debug_ok2_ratio = ok_touch_ratio_2d(hand.landmarks)
        states = finger_states(hand.world_landmarks, self._extend_ratio,
                               self._curl_confirm_ratio)
        self._debug_index_ratio = float(states[0][0]) if states else None
        # 계측: 손 높이(어깨선 위 어깨너비 배수) — 표시 전용(7차부터 판정 미사용)
        self._debug_raise = None
        shoulder_frame = hand_selector.anchor_shoulder_frame()
        center_px = hand_center_point(hand.landmarks)
        if (shoulder_frame is not None and center_px is not None
                and shoulder_frame[1] > 0.0):
            self._debug_raise = (shoulder_frame[0] - center_px[1]) / shoulder_frame[1]
        shape = classify_hand_shape(hand.world_landmarks, self._extend_ratio,
                                    self._min_valid_fingers, self._curl_confirm_ratio)
        self._debug_shape = shape      # 계측 — 뒤집은 주먹이 무엇으로 읽히는지 추적
        # deg·z는 **모양 판별과 무관하게** 계산한다(랜드마크 21점은 항상 있다).
        # z는 계측 표시 전용 — 구 z select 경로가 왜 폐기됐는지(깊은 젖힘 시
        # shape=None 연속 — 14:13 계측) 실기에서 재확인할 수 있게 남겨 둔다
        self._debug_z = z_tilt_sin(hand.world_landmarks)
        if shape == SHAPE_FIST:
            # 계기판(F= 표시)용 — 주먹 프레임의 외적 부호. 8차부터 판정 미사용
            # (뒤집기 폐기)이지만 실기 재검증을 위해 계측은 남긴다
            self._debug_fist_sign = fist_palm_sign(hand.world_landmarks)
        landmarks = hand.landmarks
        wrist_px = (float(landmarks[WRIST_LMK][0]), float(landmarks[WRIST_LMK][1]))
        knuckle_px = (float(landmarks[MIDDLE_MCP_LMK][0]),
                      float(landmarks[MIDDLE_MCP_LMK][1]))
        if wrist_px == knuckle_px:
            return None, None
        # 기준 = 화면 수직(12시) 고정 — 포즈 팔 좌표는 판정에 안 들어간다
        # (2026-08-06 사용자 결정 "팔목 손목 추적점 없애라" — 모듈 독스트링)
        bend_deg = wrap180_deg(clock_angle_deg(wrist_px, knuckle_px))

        # ★OK 사인 = 방향 무관 한 자세 (2026-08-06 8차 — "ok 손표시 confirm").
        # 섹터 판정보다 **먼저**: OK는 폄 3·굽힘 확인 0이라 classify가
        # SHAPE_OPEN으로 읽는다 — 순서를 바꾸면 OK가 4방위로 새어 나간다
        if self._is_ok_posture(hand, states):
            return POSTURE_OK, bend_deg
        if shape in (SHAPE_FIST, SHAPE_FINGER):
            # ★9차(사용자 지정): 주먹·검지 주먹도 펼친 손과 같은 4방위 각도
            # 판정 — 모양이 계열(접두)을, 섹터가 방향(접미)을 정한다.
            # 검지 주먹 위 = select · 나머지 = temp_* (매핑은 config postures).
            # 옆면·뒤집힘 무관(8차 유지) — 손등/손바닥 분리 없음
            if shape == SHAPE_FIST and abs(bend_deg) <= self._fist_flat_max_deg:
                # 그냥 쥔 평평 주먹 = 무방향 temp (9차 보강 — 사용자 지정).
                # 평평 존은 주먹에만 — 검지·펼침은 위가 본 제스처(모듈 독스트링)
                return POSTURE_FIST_FLAT, bend_deg
            family = "fist" if shape == SHAPE_FIST else "finger"
            direction = DIRECTION_SUFFIXES[compass_posture(bend_deg)]
            return f"{family}_{direction}", bend_deg
        if shape != SHAPE_OPEN:
            return None, bend_deg   # 모양 불명(블러) — 단정 안 함
        # ★4방위 그대로 (2026-08-06 7차 — 사용자 확정 "판정 똑같게 만들라"):
        # 위 섹터도 좌/우/하와 동일하게 자세가 된다(위 = bend_up — 이벤트
        # 매핑은 config postures). 구 위 방향 특례(평평 중립 → 2D 띠 → z →
        # 높이)로 상만 조건이 다르던 비대칭이 "상만 안 잡힌다"의 뿌리였다.
        # 중립·히스테리시스 소멸 — 쉬는 펼친 손의 위 자세 남발은 발화 1회
        # 래치·hold_sec·쿨다운이 막는다
        return compass_posture(bend_deg), bend_deg

    def _is_ok_posture(self, hand, states):
        """OK 사인인가 — 월드(3D) 경로 + 화면(2D) 구제 경로.

        ★2D 구제(2026-08-06 — 사용자 제안 "엄지와 검지가 맞닿았는지 로직만
        추가"): 손바닥쪽 OK는 월드 재구성이 검지를 폄으로 환각해 3D 고리
        거리가 부풀어(실기 16:09 세션 ok=0.57~0.91) 진짜 펼친 손과 겹쳤다 —
        3D 경로(is_ok_sign)가 못 잡는다. 화면(2D) 랜드마크는 이미지에서 직접
        회귀돼 고리를 제대로 그리므로(카메라 창 골격 관찰) 화면 거리로 한 번
        더 묻는다. 배경 손가락(중지~새끼) 폄 조건은 유지 — **주먹도 엄지가
        화면상 검지에 붙기 때문에**, 이 조건이 없으면 모든 주먹이 OK가 된다.
        states: 판별용 손가락 상태(검지~새끼) — _classify_posture가 계측용으로
        이미 계산한 것을 재사용한다(중복 계산 방지).
        """
        if is_ok_sign(hand.world_landmarks, self._extend_ratio,
                      self._curl_confirm_ratio, self._ok_touch_palm_ratio):
            return True
        if self._ok_touch_screen_ratio is None:
            return False   # 구제 경로 꺼짐 — 3D 단독(키 삭제 시 종전)
        if len(states) != 4 or any(
                state != STATE_EXTENDED for _, state in states[1:]):
            return False   # 중지~새끼 미폄 — 주먹·포인팅의 OK 오발 차단
        return (self._debug_ok2_ratio is not None
                and self._debug_ok2_ratio <= self._ok_touch_screen_ratio)

    # ----- 상태 -----

    def _reset_posture(self):
        """자세 추적 초기화 — 손이 떠났다: 다음 자세는 처음부터(재발화 허용)."""
        self._candidate = None
        self._candidate_since_sec = None
        self._last_match_sec = None
        self._fired_posture = None

    def _write_debug(self, bend_deg, now_sec):
        """계기판 값 — visualize.draw_debug_panel의 BEND 줄이 읽는다."""
        hold_ratio = 0.0
        if self._candidate is not None and self._candidate_since_sec is not None:
            hold_sec = (self._flat_hold_sec if self._candidate in FLAT_POSTURES
                        else self._hold_sec)
            if hold_sec > 0.0:
                hold_ratio = min(1.0, (now_sec - self._candidate_since_sec) / hold_sec)
            else:
                hold_ratio = 1.0
        self.debug = {
            "bend_deg": bend_deg,
            "bend_z": self._debug_z,                   # z 젖힘 sin | None (계측)
            "bend_fist_sign": self._debug_fist_sign,   # 주먹 외적 부호값 | None
            "bend_ok_ratio": self._debug_ok_ratio,     # OK 고리 닫힘 비율 | None
            "bend_posture": self._candidate,
            "bend_hold_ratio": hold_ratio,
        }
