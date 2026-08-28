"""MediaPipe Tasks(x, y) 기반 사람 팔 -> SO-ARM-101 관절값 변환.

이미지 평면의 x, y 좌표만 사용한다 (z / world landmark 미사용).

Pose:  어깨 -> 팔꿈치 -> 손목
Hand:  엄지 끝 vs 나머지 4손가락 끝 -> 그리퍼 개폐
"""

import math
import os
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
POSE_MODEL = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
HAND_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")

# SO-ARM-101 6축. (min, max) 는 degree 기준 안전 범위.
# 사람 관절의 실제 가동범위를 1:1 로 담을 수 있어야 해서 넉넉히 잡는다.
# (이전의 ±100 은 팔꿈치가 최대 150도까지 굽는 것을 잘라내 "절반만 펴지는" 원인이었다)
JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-110.0, 110.0),
    "elbow_flex": (-110.0, 160.0),  # 0=폄, 150=최대로 접음 (음수는 0점 보정 여유)
    "wrist_flex": (-110.0, 110.0),
    "wrist_roll": (-110.0, 110.0),
    "gripper": (0.0, 100.0),
}

# Pose landmark 인덱스 (BlazePose 33점): shoulder, elbow, wrist, 반대쪽 어깨
# 몸통(엉덩이)은 쓰지 않는다 -> 카메라에 상체만 잡혀도 동작한다.
SIDES = {
    "left": (11, 13, 15, 12),
    "right": (12, 14, 16, 11),
}
ARM_CHAIN = {
    "left": [(11, 13), (13, 15)],
    "right": [(12, 14), (14, 16)],
}

# Hand landmark 인덱스
THUMB_TIP = 4
OTHER_TIPS = (8, 12, 16, 20)  # 검지, 중지, 약지, 소지 끝
HAND_WRIST = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
PINKY_MCP = 17
NOSE = 0  # pose landmark: 몸이 향한 방향 판별용

# 전체 배율(gain)이 적용되는 각도 관절. 그리퍼는 0~100 정규화 값이라 제외한다.
GAIN_JOINTS = frozenset({"shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"})
HAND_LINKS = [
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def remap(v, in_lo, in_hi, out_lo, out_hi):
    if in_hi - in_lo < 1e-9:
        return out_lo
    t = clamp((v - in_lo) / (in_hi - in_lo), 0.0, 1.0)
    return out_lo + t * (out_hi - out_lo)


def angle_deg(ax, ay, bx, by):
    """벡터 a 와 b 의 사잇각 (0~180도), xy 평면."""
    da, db = math.hypot(ax, ay), math.hypot(bx, by)
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return math.degrees(math.acos(clamp((ax * bx + ay * by) / (da * db), -1.0, 1.0)))


def signed_angle_between(ax, ay, bx, by):
    """벡터 a 에서 b 로 회전하는 부호있는 각도 (-180~180도), xy 평면."""
    cross = ax * by - ay * bx
    dot = ax * bx + ay * by
    return math.degrees(math.atan2(cross, dot))


def wrap180(deg):
    """각도를 -180~180 범위로 감싼다."""
    return (deg + 180.0) % 360.0 - 180.0


class OneEuroFilter:
    """1€ Filter — Casiez, Roussel & Vogel, CHI 2012.

    속도에 따라 컷오프 주파수를 바꾸는 적응형 저역통과 필터.
    손이 멈춰 있을 때는 min_cutoff 로 강하게 눌러 떨림을 없애고,
    빠르게 움직일 때는 beta*speed 만큼 컷오프를 올려 지연을 줄인다.
    포인팅/제스처 트래킹의 사실상 표준 필터.
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

    def __call__(self, t, x):
        if self._t_prev is None:
            self._t_prev, self._x_prev = t, x
            return x
        dt = max(t - self._t_prev, 1e-6)

        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self._x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


@dataclass
class OneEuroBank:
    """관절 이름별로 독립된 1€ 필터를 관리한다."""

    min_cutoff: float = 1.0
    beta: float = 0.4
    d_cutoff: float = 1.0
    _filters: dict = field(default_factory=dict)

    def __call__(self, key, t, value):
        f = self._filters.get(key)
        if f is None:
            f = self._filters[key] = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
        return f(t, value)

    def reset(self, key):
        """해당 관절의 필터 이력을 지운다 (반전 등으로 값이 급변할 때)."""
        self._filters.pop(key, None)


class ArmTracker:
    """웹캠 프레임 -> {관절명: degree} 딕셔너리."""

    def __init__(self, side="left", beta=0.4, min_cutoff=1.0, camera_index=0, flip=True,
                 view="front", pan_hold=0.0, lock_pan=False, invert_joints=(),
                 wrist_gain=1.0, roll_neutral=90.0, lift_gain=1.0,
                 elbow_gain=1.0, elbow_offset=0.0,
                 lock_roll=True, roll_hold=0.0, gain=1.0):
        if side not in SIDES:
            raise ValueError("side must be 'left' or 'right'")
        if view not in ("front", "side"):
            raise ValueError("view must be 'front' or 'side'")
        self.view = view
        self.pan_hold = pan_hold
        self.lock_pan = lock_pan
        self.wrist_gain = wrist_gain
        self.lift_gain = lift_gain
        self.elbow_gain = elbow_gain
        self.elbow_offset = elbow_offset
        self.lock_roll = lock_roll
        self.roll_hold = roll_hold
        self.gain = gain
        # 실행 중 키로 조절하는 관절별 0점 보정(도).
        self.offsets = {k: 0.0 for k in JOINT_LIMITS}
        self.roll_neutral = roll_neutral
        self.invert_joints = set(invert_joints)
        unknown = self.invert_joints - set(JOINT_LIMITS)
        if unknown:
            raise ValueError(f"invert_joints 에 알 수 없는 관절: {unknown}")
        for path in (POSE_MODEL, HAND_MODEL):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"모델 파일이 없습니다: {path}\n"
                    "  python download_models.py 를 먼저 실행하세요."
                )
        self.side = side
        self.flip = flip
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {camera_index} 를 열 수 없습니다.")

        # 경로에 한글이 있으면 MediaPipe 의 C++ 로더가 파일을 못 여므로 바이트로 넘긴다.
        with open(POSE_MODEL, "rb") as f:
            pose_bytes = f.read()
        with open(HAND_MODEL, "rb") as f:
            hand_bytes = f.read()

        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=pose_bytes),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=hand_bytes),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        )
        self.filters = OneEuroBank(min_cutoff=min_cutoff, beta=beta)
        self.last_action = {k: 0.0 for k in JOINT_LIMITS}
        self.last_action["gripper"] = 50.0
        self._t0 = None

    # ---------- landmark -> 관절 ----------

    def _pose_joints(self, lms, w, h):
        s_i, e_i, w_i, o_i = SIDES[self.side]
        s, e, wr, opp = lms[s_i], lms[e_i], lms[w_i], lms[o_i]
        if min(getattr(p, "visibility", 1.0) for p in (s, e, wr)) < 0.5:
            return None

        # 픽셀 좌표 (x, y 만 사용)
        sx, sy = s.x * w, s.y * h
        ex, ey = e.x * w, e.y * h
        wx, wy = wr.x * w, wr.y * h

        # 위팔 길이로 정규화 -> 몸통(엉덩이)이 프레임 밖이어도 동작하고,
        # 카메라 거리 변화에도 둔감해진다.
        upper_arm = max(math.hypot(ex - sx, ey - sy), 1e-6)

        if self.view == "side":
            # 측면 촬영: 팔이 움직이는 시상면이 카메라와 나란하므로 각도가 정확하다.
            # 대신 좌우(pan) 정보는 화면에 존재하지 않는다.
            # 몸이 향한 방향 = 코가 어깨보다 어느 쪽에 있는가.
            facing = 1.0 if lms[NOSE].x * w >= sx else -1.0

            # 1) shoulder_pan: 측면에서는 관측 불가 -> 고정값 유지
            shoulder_pan = self.pan_hold

            # 2) shoulder_lift: 아래 방향 기준 위팔이 앞/위로 든 각도
            #    (0=차렷, 90=앞으로 수평, 180=만세)
            flex = math.degrees(math.atan2(facing * (ex - sx), ey - sy))
            shoulder_lift = remap(flex, -20.0, 180.0, -100.0, 100.0)
        else:
            # 정면 촬영: 좌우는 x, 상하는 y 로 분리해서 읽는다.
            # 몸 바깥쪽(+x) 방향. 반대쪽 어깨로 판별하므로 좌/우, 미러 여부에 무관.
            away = 1.0 if sx >= opp.x * w else -1.0

            # 1) shoulder_pan: 손목의 좌우 오프셋 (x 성분만). lock_pan 이면 고정값 유지.
            if self.lock_pan:
                shoulder_pan = self.pan_hold
            else:
                shoulder_pan = remap(away * (wx - sx) / upper_arm, -2.0, 2.0, -100.0, 100.0)

            # 2) shoulder_lift: 위팔이 수평 기준 얼마나 들렸는지의 "실제 각도".
            #    (sy-ey)/upper_arm 은 sin 값이므로 asin 을 씌워 각도로 되돌린다.
            #    비율을 그대로 쓰면 팔을 높이 들수록 감도가 뭉개져서 부자연스러웠다.
            #    -90(팔 아래) ~ 0(수평) ~ +90(팔 위) 을 1:1 로 반영한다.
            sin_lift = clamp((sy - ey) / upper_arm, -1.0, 1.0)
            shoulder_lift = math.degrees(math.asin(sin_lift)) * self.lift_gain

        # 3) elbow_flex: 사람 팔꿈치 안쪽각(180=폄, 30=최대로 접음)을 "굽힌 각도"로 바꿔
        #    1:1 로 반영한다. 폄=0도, 90도 접으면 90도.
        elbow_inner = angle_deg(sx - ex, sy - ey, wx - ex, wy - ey)
        elbow_flex = (180.0 - elbow_inner) * self.elbow_gain

        return {
            "shoulder_pan": shoulder_pan,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
        }, (ex, ey, wx, wy)

    def _hand_joints(self, lm, w, h, forearm):
        hw_x, hw_y = lm[HAND_WRIST].x * w, lm[HAND_WRIST].y * h
        mm_x, mm_y = lm[MIDDLE_MCP].x * w, lm[MIDDLE_MCP].y * h
        palm = max(math.hypot(mm_x - hw_x, mm_y - hw_y), 1e-6)

        out = {}

        # 4) wrist_flex: 아래팔 벡터 대비 손바닥 벡터의 부호있는 각도.
        #    angle_deg(부호없음)을 쓰면 위로 꺾으나 아래로 꺾으나 "구부러진 정도"만
        #    나와서 방향을 구분 못했다 -> signed_angle_between 으로 교체.
        #    로봇이 use_degrees 모드이므로 각도를 그대로 넘겨 1:1 로 반영한다.
        if forearm is not None:
            ex, ey, wx, wy = forearm
            bend = signed_angle_between(wx - ex, wy - ey, mm_x - hw_x, mm_y - hw_y)
            out["wrist_flex"] = bend * self.wrist_gain

        # 5) wrist_roll: 진짜 회전축(전완 longitudinal axis)은 xy 만으로는 원리상
        #    구할 수 없어서 손 폭 벡터의 회전각으로 근사할 수밖에 없다.
        #    실사용에서 쓸모가 없어 기본적으로 고정한다 (--roll-hold 로 각도 지정).
        if self.lock_roll:
            out["wrist_roll"] = self.roll_hold
        else:
            ix, iy = lm[INDEX_MCP].x * w, lm[INDEX_MCP].y * h
            px, py = lm[PINKY_MCP].x * w, lm[PINKY_MCP].y * h
            if forearm is not None:
                ex, ey, wx, wy = forearm
                roll = signed_angle_between(wx - ex, wy - ey, px - ix, py - iy)
                out["wrist_roll"] = wrap180(roll - self.roll_neutral) * self.wrist_gain

        # 6) gripper: 엄지 끝 <-> 나머지 4손가락 끝 평균거리 (손 크기로 정규화)
        #    손바닥 펴기(손끝 벌어짐) -> 100, 주먹 쥐기(손끝 모임) -> 0
        tx, ty = lm[THUMB_TIP].x * w, lm[THUMB_TIP].y * h
        pinch = sum(
            math.hypot(lm[i].x * w - tx, lm[i].y * h - ty) / palm for i in OTHER_TIPS
        ) / len(OTHER_TIPS)
        out["gripper"] = remap(pinch, 0.35, 1.6, 0.0, 100.0)
        return out

    # ---------- 그리기 ----------

    @staticmethod
    def _draw(frame, lms, links, color, w, h):
        pts = [(int(p.x * w), int(p.y * h)) for p in lms]
        for a, b in links:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, 2)
                cv2.circle(frame, pts[a], 4, (0, 0, 255), -1)
                cv2.circle(frame, pts[b], 4, (0, 0, 255), -1)

    # ---------- 메인 루프용 ----------

    def read(self):
        """(action, annotated_frame) 반환. 인식 실패 시 (None, frame)."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if self.flip:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if self._t0 is None:
            self._t0 = now
        elapsed = now - self._t0
        ts = max(int(elapsed * 1000), 0)

        pose_res = self.pose.detect_for_video(mp_img, ts)
        hand_res = self.hands.detect_for_video(mp_img, ts)

        raw, forearm = {}, None
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]
            got = self._pose_joints(lms, w, h)
            if got:
                joints, forearm = got
                raw.update(joints)
            self._draw(frame, lms, ARM_CHAIN[self.side], (0, 200, 255), w, h)
        if hand_res.hand_landmarks:
            lms = hand_res.hand_landmarks[0]
            raw.update(self._hand_joints(lms, w, h, forearm))
            self._draw(frame, lms, HAND_LINKS, (255, 200, 0), w, h)

        if not raw:
            cv2.putText(frame, "no arm detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, frame

        action = dict(self.last_action)
        for name, value in raw.items():
            lo, hi = JOINT_LIMITS[name]
            # 각도 관절만 배율 적용. 그리퍼(0~100)와 고정된 축은 제외한다.
            if name in GAIN_JOINTS and not (name == "wrist_roll" and self.lock_roll):
                value *= self.gain
            if name in self.invert_joints:
                value = lo + hi - value  # 대칭 범위면 부호반전, 0..100 이면 100-value
            value += self.offsets[name]
            action[name] = clamp(self.filters(name, elapsed, value), lo, hi)
        self.last_action = action

        for i, (name, val) in enumerate(action.items()):
            cv2.putText(frame, f"{name:14s}{val:7.1f}", (10, 26 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return action, frame

    def close(self):
        self.cap.release()
        self.pose.close()
        self.hands.close()
