"""postprocess 모듈 — 손 모양 판별: 주먹 / 한 손가락 / 손바닥.

손가락(검지~새끼) 3단계:
- **폄**: 손끝-손목뿌리 3D 거리가 둘째 관절(PIP)-뿌리 3D 거리의 extend_ratio배 이상
- **굽힘(확인된 것만)**: ①관절 진행 방향 반전(뿌리→PIP vs DIP→손끝 3D 내적 음수) 또는
  ②손끝이 PIP보다 안쪽(curl_confirm_ratio 이하)
- **판단 불가(기권)**: 짧지만 접힘이 확인 안 됨 — 단정하지 않는다

손 판정: 한 손가락 = 폄 1개 + 굽힘 확인 min_valid_fingers-1개 이상 ·
주먹 = 폄 0 + 기권 0 + 굽힘 확인 min_valid_fingers개 이상 ·
손바닥 = 폄 min_valid_fingers개 이상 + 굽힘 확인 0 · 그 외 = None.
엄지는 세지 않는다 — 주먹을 쥐어도 엄지는 밖으로 삐져나와 오판이 잦다.
"""
import math

HAND_ROOT_IDX = 0
HAND_FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
HAND_KPT_COUNT = 21

SHAPE_FIST = "fist"
SHAPE_FINGER = "finger"
SHAPE_OPEN = "open"
SHAPE_V = "v"

STATE_EXTENDED = "extend"
STATE_CURLED = "curl"
STATE_UNCERTAIN = "uncertain"


class HandShapeStabilizer:
    """짧은 오판은 거르고, 정지 제스처와 손 소실은 즉시 반영한다.

    open/finger/V는 같은 판정이 ``confirm_frames``회 연속 들어와야 확정한다.
    단발성 랜드마크 튐이 주행이나 재중심 명령으로 이어지는 것을 막기 위함이다.
    fist는 안전 정지 제스처이므로 확인을 기다리지 않고 즉시 확정한다.
    """

    def __init__(self, hold_open_on_unknown_sec=0.3, confirm_frames=2,
                 confirm_frames_by_shape=None):
        self._hold_open_sec = hold_open_on_unknown_sec
        self._confirm_frames = max(1, int(confirm_frames))
        self._confirm_frames_by_shape = confirm_frames_by_shape or {}
        self.reset()

    def reset(self):
        self._last_open_sec = None
        self._stable_shape = None
        self._candidate_shape = None
        self._candidate_count = 0

    def update(self, raw_shape, hand_detected, now_sec):
        if not hand_detected:
            self.reset()
            return None
        if raw_shape == SHAPE_FIST:
            # 주먹은 주행을 멈추는 자세이므로 지연시키지 않는다.
            self._stable_shape = SHAPE_FIST
            self._candidate_shape = None
            self._candidate_count = 0
            self._last_open_sec = None
            return SHAPE_FIST
        if raw_shape is not None:
            if raw_shape == self._stable_shape:
                self._candidate_shape = None
                self._candidate_count = 0
                if raw_shape == SHAPE_OPEN:
                    self._last_open_sec = now_sec
                return raw_shape
            if raw_shape == self._candidate_shape:
                self._candidate_count += 1
            else:
                self._candidate_shape = raw_shape
                self._candidate_count = 1
            required_frames = max(1, int(
                self._confirm_frames_by_shape.get(raw_shape, self._confirm_frames)))
            if self._candidate_count < required_frames:
                return None
            self._stable_shape = raw_shape
            self._candidate_shape = None
            self._candidate_count = 0
            self._last_open_sec = now_sec if raw_shape == SHAPE_OPEN else None
            return raw_shape
        if (self._last_open_sec is not None
                and self._stable_shape == SHAPE_OPEN
                and now_sec - self._last_open_sec <= self._hold_open_sec):
            return SHAPE_OPEN
        self._stable_shape = None
        self._candidate_shape = None
        self._candidate_count = 0
        self._last_open_sec = None
        return None


def _dist3d(point_a, point_b):
    return math.dist(
        (float(point_a[0]), float(point_a[1]), float(point_a[2])),
        (float(point_b[0]), float(point_b[1]), float(point_b[2])),
    )


def finger_states(landmarks, extend_ratio, curl_confirm_ratio):
    """손가락(검지~새끼)별 (3D 비율, 판정 상태) 목록."""
    if landmarks is None or len(landmarks) < HAND_KPT_COUNT:
        return []
    root = landmarks[HAND_ROOT_IDX]
    states = []
    for mcp, pip, dip, tip in HAND_FINGERS:
        pip_dist = _dist3d(landmarks[pip], root)
        if pip_dist <= 0.0:
            continue
        ratio = _dist3d(landmarks[tip], root) / pip_dist
        if ratio >= extend_ratio:
            states.append((ratio, STATE_EXTENDED))
            continue
        base = landmarks[pip] - landmarks[mcp]
        tip_dir = landmarks[tip] - landmarks[dip]
        is_folded = float(
            base[0] * tip_dir[0] + base[1] * tip_dir[1] + base[2] * tip_dir[2]
        ) < 0.0
        if is_folded or ratio <= curl_confirm_ratio:
            states.append((ratio, STATE_CURLED))
        else:
            states.append((ratio, STATE_UNCERTAIN))
    return states


def classify_hand_shape(landmarks, extend_ratio, min_valid_fingers, curl_confirm_ratio):
    """손 모양 판별 -> "fist" | "finger" | "open" | None (모양 불명).

    landmarks: HandDetection.world_landmarks(미터 단위 월드 좌표) 권장 — 화면
    좌표의 z는 노이즈가 커서 방향 반전 판정이 튄다. 판별은 비율·방향만 쓰므로
    스케일 무관.
    """
    states = finger_states(landmarks, extend_ratio, curl_confirm_ratio)
    if not states:
        return None
    extended_count = sum(1 for _, state in states if state == STATE_EXTENDED)
    curled_count = sum(1 for _, state in states if state == STATE_CURLED)
    uncertain_count = sum(1 for _, state in states if state == STATE_UNCERTAIN)

    if (len(states) >= 4
            and states[0][1] == STATE_EXTENDED
            and states[1][1] == STATE_EXTENDED
            and states[2][1] == STATE_CURLED
            and states[3][1] == STATE_CURLED):
        return SHAPE_V
    if extended_count == 1 and curled_count >= min_valid_fingers - 1:
        return SHAPE_FINGER
    if extended_count == 0 and uncertain_count == 0 and curled_count >= min_valid_fingers:
        return SHAPE_FIST
    if extended_count >= min_valid_fingers and curled_count == 0:
        return SHAPE_OPEN
    return None


def hand_center_point(landmarks):
    """손 중심 추적점 — 21점 화면 좌표의 평균 (x_px, y_px) | None."""
    if landmarks is None or len(landmarks) < HAND_KPT_COUNT:
        return None
    return (
        float(sum(point[0] for point in landmarks) / len(landmarks)),
        float(sum(point[1] for point in landmarks) / len(landmarks)),
    )


def hand_span_px(landmarks):
    """손 화면 크기 — 랜드마크 묶음의 최대 폭(px)."""
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    return max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def hand_bbox_px(landmarks, pad_px=0.0):
    """손 바운딩 박스 (x1, y1, x2, y2) — roi_zoom·시각화 공용."""
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    return (float(xs.min() - pad_px), float(ys.min() - pad_px),
            float(xs.max() + pad_px), float(ys.max() + pad_px))
