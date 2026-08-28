"""D-pad 없는 손가락 개수 명령과 좌우 3회 흔들기 인식."""
import math

from src.postprocess.hand_shape import (
    STATE_CURLED, STATE_EXTENDED, STATE_UNCERTAIN, finger_states,
    hand_center_point,
)

COUNT_SHAPES = {0: "fist", 1: "finger", 2: "two", 3: "three", 4: "four"}


def is_thumb_extended(landmarks, extend_ratio=1.08):
    """화면 투영에서 엄지 끝이 엄지 IP보다 손목에서 충분히 먼지 판정."""
    if landmarks is None or len(landmarks) < 21:
        return False
    wrist, thumb_ip, thumb_tip = landmarks[0], landmarks[3], landmarks[4]
    ip_dist = math.dist((float(wrist[0]), float(wrist[1])),
                        (float(thumb_ip[0]), float(thumb_ip[1])))
    tip_dist = math.dist((float(wrist[0]), float(wrist[1])),
                         (float(thumb_tip[0]), float(thumb_tip[1])))
    return ip_dist > 0.0 and tip_dist / ip_dist >= extend_ratio


def is_ok_sign(screen_landmarks, finger_state_list, touch_ratio=0.16):
    """엄지·검지 끝을 붙이고 중지·약지·새끼를 편 OK 사인."""
    if (screen_landmarks is None or len(screen_landmarks) < 21
            or len(finger_state_list) != 4):
        return False
    xs = screen_landmarks[:, 0]
    ys = screen_landmarks[:, 1]
    hand_span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
    if hand_span <= 0.0:
        return False
    thumb_tip, index_tip = screen_landmarks[4], screen_landmarks[8]
    tip_gap = math.dist((float(thumb_tip[0]), float(thumb_tip[1])),
                        (float(index_tip[0]), float(index_tip[1])))
    other_three_extended = all(
        finger_state_list[index][1] == STATE_EXTENDED for index in (1, 2, 3))
    return tip_gap <= hand_span * touch_ratio and other_three_extended


def is_back_of_hand(landmarks, user_side):
    """손목-검지뿌리-새끼뿌리의 화면 방향과 손잡이로 손등 여부를 판정."""
    if landmarks is None or len(landmarks) < 21 or user_side not in ("left", "right"):
        return False
    wrist, index_mcp, pinky_mcp = landmarks[0], landmarks[5], landmarks[17]
    cross = ((index_mcp[0] - wrist[0]) * (pinky_mcp[1] - wrist[1])
             - (index_mcp[1] - wrist[1]) * (pinky_mcp[0] - wrist[0]))
    # 입력 영상은 사용자 조작 방향에 맞춰 mirror 처리된 좌표다.
    return cross < 0.0 if user_side == "right" else cross > 0.0


def classify_finger_command(world_landmarks, screen_landmarks, user_side,
                            extend_ratio, curl_confirm_ratio):
    # 실물 카메라에서 손을 정면으로 펼쳤을 때 MediaPipe world_landmarks의 깊이
    # 추정이 뒤집혀 네 손가락을 모두 curled로 내는 경우가 있다. 사용자가 실제로
    # 보는 투영 좌표를 우선해 개수를 세고, 화면 좌표가 없을 때만 월드 좌표를 쓴다.
    pose_landmarks = (screen_landmarks if screen_landmarks is not None
                      and len(screen_landmarks) >= 21 else world_landmarks)
    screen_states = finger_states(
        pose_landmarks, extend_ratio, curl_confirm_ratio)
    world_states = finger_states(
        world_landmarks, extend_ratio, curl_confirm_ratio)
    if len(screen_states) != 4:
        return None
    # 따봉 자세에서는 3D 깊이 오차로 검지가 순간적으로 펴진 것처럼 보일 수
    # 있다. 화면에서 나머지 네 손가락 중 3개 이상이 명확히 접혔다면 엄지
    # ON/OFF를 개수 판정보다 먼저 처리해 finger 명령과 중복되지 않게 한다.
    # 두 손가락만 확실히 접혀 보여도 따봉 후보로 허용하되, 화면상 일반
    # 손가락이 하나라도 명확히 펴졌다면 1/2/3/4 명령을 우선한다. 이렇게 해야
    # 검지·중지 포즈의 엄지가 우연히 곧게 보여도 모드 토글로 가로채지 않는다.
    screen_curled_count = sum(
        state == STATE_CURLED for _, state in screen_states)
    screen_extended_count = sum(
        state == STATE_EXTENDED for _, state in screen_states)
    if screen_curled_count >= 2 and screen_extended_count == 0:
        thumb_toggle = classify_thumb_toggle(screen_landmarks)
        if thumb_toggle is not None:
            return thumb_toggle
    # 실물 테스트에서는 3D 깊이의 거짓 펴짐이 다른 손가락 개수까지 바꾸었다.
    # 화면 판정이 명확하면 그대로 사용하고, 화면이 uncertain일 때만 3D로
    # 보완한다. 이렇게 해야 방향 보정과 실제 포즈 안정성을 함께 유지한다.
    states = []
    for index, (screen_ratio, screen_state) in enumerate(screen_states):
        if len(world_states) == 4:
            world_ratio, world_state = world_states[index]
        else:
            world_ratio, world_state = screen_ratio, STATE_UNCERTAIN
        if screen_state in (STATE_EXTENDED, STATE_CURLED):
            states.append((screen_ratio, screen_state))
        elif world_state in (STATE_EXTENDED, STATE_CURLED):
            states.append((world_ratio, world_state))
        else:
            states.append((screen_ratio, STATE_UNCERTAIN))
    # 비스듬한 손가락 하나가 경계에 걸렸다는 이유로 포즈 전체를 버리지 않는다.
    # 비율 1.0을 기준으로 가까운 상태에 귀속하며 최종 명령의 프레임 확인은 유지한다.
    states = [
        (ratio, STATE_EXTENDED if ratio >= 1.0 else STATE_CURLED)
        if state == STATE_UNCERTAIN else (ratio, state)
        for ratio, state in states
    ]
    if is_ok_sign(screen_landmarks, states):
        return "joystick_toggle"
    count = sum(state == STATE_EXTENDED for _, state in states)
    if count == 4 and is_back_of_hand(screen_landmarks, user_side):
        return "back"
    # 네 손가락+접힌 엄지는 좌회전(four), 엄지까지 모두 펴면 흔들기(open)로
    # 분리한다. 그래야 3 -> 4 전환을 흔들기 판정 대기 없이 즉시 처리할 수 있다.
    if count == 4 and is_thumb_extended(screen_landmarks):
        return "open"
    return COUNT_SHAPES[count]


def classify_thumb_toggle(landmarks):
    """손의 화면 회전과 무관한 엄지 위/아래 모드 토글 판정."""
    if landmarks is None or len(landmarks) < 21:
        return None
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    hand_span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
    wrist, middle_mcp = landmarks[0], landmarks[9]
    thumb_mcp, thumb_ip, thumb_tip = landmarks[2], landmarks[3], landmarks[4]
    palm_x = float(middle_mcp[0] - wrist[0])
    palm_y = float(middle_mcp[1] - wrist[1])
    thumb_x = float(thumb_tip[0] - thumb_mcp[0])
    thumb_y = float(thumb_tip[1] - thumb_mcp[1])
    palm_len = math.hypot(palm_x, palm_y)
    thumb_len = math.hypot(thumb_x, thumb_y)
    thumb_segments = (
        math.dist((float(thumb_mcp[0]), float(thumb_mcp[1])),
                  (float(thumb_ip[0]), float(thumb_ip[1])))
        + math.dist((float(thumb_ip[0]), float(thumb_ip[1])),
                    (float(thumb_tip[0]), float(thumb_tip[1])))
    )
    if (hand_span <= 0.0 or palm_len <= 0.0
            or thumb_len < hand_span * 0.15 or thumb_segments <= 0.0
            or thumb_len / thumb_segments < 0.82):
        return None
    alignment = (palm_x * thumb_x + palm_y * thumb_y) / (palm_len * thumb_len)
    if alignment >= 0.35:
        return "mode_on"
    if alignment <= -0.35:
        return "mode_off"
    return None


class WaveDetector:
    """손 중심이 좌우 극점을 6번 번갈아 찍으면 3회 흔들기로 발화."""

    def __init__(self, cfg):
        self._amplitude_ratio = float(cfg.get("amplitude_ratio", 0.08))
        self._timeout_sec = float(cfg.get("timeout_sec", 3.0))
        self.reset()

    def reset(self):
        self._anchor_x = None
        self._last_side = 0
        self._half_swings = 0
        self._last_extreme_sec = None

    @property
    def in_progress(self):
        # 손을 화면 안으로 가져오는 최초 이동 하나는 흔들기가 아니다. 반대 방향
        # 이동까지 확인된 뒤에만 four/back 명령을 억제한다.
        return self._half_swings >= 2

    def update(self, landmarks, frame_width_px, now_sec):
        center = hand_center_point(landmarks)
        if center is None or frame_width_px <= 0:
            self.reset()
            return False
        if (self._last_extreme_sec is not None
                and now_sec - self._last_extreme_sec > self._timeout_sec):
            self.reset()
        if self._anchor_x is None:
            self._anchor_x = center[0]
            return False
        threshold = self._amplitude_ratio * frame_width_px
        delta = center[0] - self._anchor_x
        side = 1 if delta >= threshold else (-1 if delta <= -threshold else 0)
        if side and side != self._last_side:
            self._last_side = side
            self._half_swings += 1
            self._last_extreme_sec = now_sec
            # 방금 도달한 극점을 다음 이동 기준으로 삼아 자연스러운 왕복을 센다.
            self._anchor_x = center[0]
            if self._half_swings >= 6:
                self.reset()
                return True
        return False
