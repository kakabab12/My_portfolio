"""MediaPipe Tasks 기반 사람 팔 -> SO-ARM-101 관절값 변환.

기본은 3D world landmark(미터 단위)를 쓴다. 몸통(어깨선+척추)으로 직교 기저를
세우고 팔 벡터를 거기에 투영하므로, 팔을 카메라 쪽으로 뻗어도 각도가 뭉개지지
않는다. 2D 이미지 좌표만 쓰는 예전 방식은 --no-world 로 남겨뒀다.

  Pose : 어깨 -> 팔꿈치 -> 손목  (shoulder_pan/lift, elbow_flex, wrist_flex)
  Hand : 손가락 마디 관절각      (gripper — 주먹/폄)

조도 대응으로 LAB 색공간의 L 채널에 CLAHE + 적응형 감마를 적용한다.
"""

import math
import os
import statistics
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")

# 정확도 등급: lite < full < heavy. 느려지는 대신 랜드마크가 정확해진다.
POSE_MODELS = {
    "lite": os.path.join(MODEL_DIR, "pose_landmarker_lite.task"),
    "full": os.path.join(MODEL_DIR, "pose_landmarker_full.task"),
    "heavy": os.path.join(MODEL_DIR, "pose_landmarker_heavy.task"),
}
POSE_MODEL = POSE_MODELS["lite"]   # 하위호환용 기본값

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
    # 입력 범위가 내림차순(in_lo > in_hi)인 경우도 지원해야 하므로 abs 로 판정한다.
    span = in_hi - in_lo
    if abs(span) < 1e-9:
        return out_lo
    t = clamp((v - in_lo) / span, 0.0, 1.0)
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


# ---------- 3D 벡터 유틸 (world landmark 용) ----------

def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def v_norm(a):
    return math.sqrt(v_dot(a, a))


def v_unit(a):
    n = v_norm(a)
    return (0.0, 0.0, 0.0) if n < 1e-9 else (a[0] / n, a[1] / n, a[2] / n)


def v_angle(a, b):
    """두 3D 벡터의 사잇각(0~180도). 투영 왜곡이 없는 진짜 각도."""
    na, nb = v_norm(a), v_norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(clamp(v_dot(a, b) / (na * nb), -1.0, 1.0)))


def lm_xyz(lm):
    """world landmark -> (x, y, z) 튜플. MediaPipe 는 y 가 아래로 증가한다."""
    return (lm.x, lm.y, lm.z)


class LightingNormalizer:
    """조도 변화에 강인하게 만드는 전처리.

    CLAHE (Contrast Limited Adaptive Histogram Equalization, Zuiderveld 1994)
    를 LAB 색공간의 L(밝기) 채널에만 적용한다. 색은 건드리지 않으므로
    피부색 기반 검출을 망가뜨리지 않으면서, 역광·그림자·어두운 방처럼
    화면 안에서 밝기가 불균일한 상황을 국소적으로 보정한다.

    전역 히스토그램 평활화와 달리 타일 단위로 처리하고 대비를 제한(clip)해서
    노이즈가 증폭되는 문제가 없다.
    """

    def __init__(self, clip_limit=2.0, tile=8, target_mean=125.0, gamma_strength=0.7):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        self.target_mean = target_mean
        self.gamma_strength = gamma_strength
        self._lut_cache = {}

    def _gamma_lut(self, exponent):
        """out = (in/255)^exponent * 255.  exponent < 1 이면 밝아진다."""
        key = round(exponent, 2)
        lut = self._lut_cache.get(key)
        if lut is None:
            lut = np.array([((i / 255.0) ** key) * 255
                            for i in range(256)], dtype=np.uint8)
            if len(self._lut_cache) > 64:
                self._lut_cache.clear()
            self._lut_cache[key] = lut
        return lut

    def __call__(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 1) 국소 대비 향상 — 그림자/역광 대응
        l = self._clahe.apply(l)

        # 2) 전역 밝기를 목표치로 끌어오는 감마 보정 — 너무 어둡거나 밝은 방 대응
        mean = float(np.mean(l))
        if 1.0 < mean < 254.0:
            # 평균이 target 이 되게 하는 지수. mean < target 이면 1보다 작아져 밝아진다.
            exponent = (math.log(self.target_mean / 255.0)
                        / math.log(mean / 255.0))
            exponent = clamp(exponent, 0.25, 3.0)
            # 과보정 방지: 강도만큼만 적용 (1.0 = 변화 없음)
            exponent = 1.0 + (exponent - 1.0) * self.gamma_strength
            l = cv2.LUT(l, self._gamma_lut(exponent))

        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ---------- 손가락 굽힘 (그리퍼) ----------

# 각 손가락의 (MCP, PIP, DIP, TIP) 랜드마크 인덱스
FINGER_CHAINS = {
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}


def finger_curl(wl, chain):
    """손가락이 굽은 정도를 0(완전히 폄) ~ 1(완전히 주먹) 로 반환.

    마디 사이 각도의 합을 쓰므로 손의 크기·거리·회전과 무관하다.
    끝점 사이 거리를 재던 방식은 손을 옆에서 보면 값이 무너졌는데,
    관절각은 3D world landmark 에서 그런 왜곡이 없다.
    """
    mcp, pip, dip, tip = (lm_xyz(wl[i]) for i in chain)
    total = (v_angle(v_sub(mcp, pip), v_sub(dip, pip))
             + v_angle(v_sub(pip, dip), v_sub(tip, dip)))
    # 완전히 편 손가락은 두 각이 모두 180도(합 360), 주먹은 대략 합 160도.
    return clamp(remap(total, 360.0, 160.0, 0.0, 1.0), 0.0, 1.0)


class AnatomicalSolver:
    """뼈 길이 제약으로 깊이(z)를 다시 푸는 단안 3D 보정기.

    문제: MediaPipe 의 world landmark 는 z(깊이)가 특히 부정확하다. 실측해보니
    가만히 있어도 위팔 길이가 7.7%, 팔꿈치각이 ±7.5도씩 흔들렸다.

    관찰: 이미지 평면의 x, y 는 정확하다. 부정확한 건 z 뿐이다.
          그리고 뼈 길이는 물리적으로 변하지 않는다.

    해법: 뼈 길이 L 을 미리 재두면, 이미지에서 읽은 (dx, dy) 로부터
          dz = ±sqrt(L² - dx² - dy²) 로 깊이를 **역산**할 수 있다.
          부호(앞/뒤)만 MediaPipe 의 z 추정에서 가져온다. 크기는 못 믿어도
          부호는 신뢰할 만하기 때문이다.

    이미지 픽셀 -> 미터 변환은 어깨너비를 자로 삼는다
    (world 어깨너비 / 픽셀 어깨너비).
    """

    def __init__(self, calib_frames=45):
        self.calib_frames = calib_frames
        self._samples = {}      # 뼈 이름 -> 길이 표본
        self.lengths = {}       # 확정된 뼈 길이(m)
        self.ready = False

    def reset(self):
        self._samples.clear()
        self.lengths.clear()
        self.ready = False

    def observe(self, bones):
        """{뼈이름: 길이} 를 모아 중앙값으로 확정한다. 중앙값이라 이상치에 강하다."""
        if self.ready:
            return
        for name, length in bones.items():
            self._samples.setdefault(name, []).append(length)
        n = min(len(v) for v in self._samples.values())
        if n >= self.calib_frames:
            for name, vals in self._samples.items():
                self.lengths[name] = statistics.median(vals)
            self.ready = True

    def solve_bone(self, name, dx_px, dy_px, z_hint, px_per_m):
        """이미지 변위 + 뼈 길이로 3D 벡터를 재구성한다.

        반환: (dx_m, dy_m, dz_m) — 길이가 정확히 뼈 길이인 3D 벡터.
        보정할 수 없으면 None.
        """
        L = self.lengths.get(name)
        if not L or px_per_m <= 1e-6:
            return None
        dx = dx_px / px_per_m
        dy = dy_px / px_per_m
        planar_sq = dx * dx + dy * dy
        rem = L * L - planar_sq
        if rem <= 0.0:
            # 이미지상 길이가 뼈 길이보다 길다 = 스케일 오차.
            # 이 경우 깊이는 0으로 보고 평면 성분만 뼈 길이에 맞춰 정규화한다.
            scale = L / max(math.sqrt(planar_sq), 1e-9)
            return (dx * scale, dy * scale, 0.0)
        dz = math.sqrt(rem)
        return (dx, dy, math.copysign(dz, z_hint if z_hint else 1.0))


class CameraStream:
    """카메라를 별도 스레드에서 계속 읽어 항상 최신 프레임만 들고 있는다.

    cap.read() 는 이 웹캠에서 66ms(15FPS 상한) 가 걸린다. 메인 루프에서
    직접 호출하면 그 시간만큼 추론이 멈춰 기다리므로, 읽기를 분리해
    카메라 대기와 추론이 겹치게 한다. 큐를 쌓지 않고 최신 것만 유지해서
    지연이 누적되지 않는다.
    """

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {index} 를 열 수 없습니다.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = f

    def read(self):
        with self._lock:
            f = self._frame
        return (f is not None), (None if f is None else f.copy())

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


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
    """관절 이름별로 독립된 1€ 필터를 관리한다.

    per_joint 에 {관절: (min_cutoff, beta)} 를 주면 그 관절만 다른 세기로 거른다.
    depth 기반 shoulder_pan 처럼 유독 노이즈가 큰 축을 따로 눌러줄 때 쓴다.
    """

    min_cutoff: float = 1.0
    beta: float = 0.4
    d_cutoff: float = 1.0
    per_joint: dict = field(default_factory=dict)
    _filters: dict = field(default_factory=dict)

    def __call__(self, key, t, value):
        f = self._filters.get(key)
        if f is None:
            mc, b = self.per_joint.get(key, (self.min_cutoff, self.beta))
            f = self._filters[key] = OneEuroFilter(mc, b, self.d_cutoff)
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
                 lock_roll=True, roll_hold=0.0, gain=1.0,
                 pan_mode="depth", depth_scale=1.0, pan_gain=1.0,
                 pan_deadzone=0.35, depth_span=4.0, depth_calib_frames=30,
                 pan_cutoff=0.5, pan_beta=0.0, use_world=True,
                 model='full', enhance=True, hand_interval=2,
                 use_anatomy=True, anatomy_frames=45):
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
        self.use_world = use_world
        if model not in POSE_MODELS:
            raise ValueError(f"model must be one of {sorted(POSE_MODELS)}")
        pose_model_path = POSE_MODELS[model]
        self.model_name = model
        self.enhancer = LightingNormalizer() if enhance else None
        self.hand_interval = max(1, int(hand_interval))
        self.use_anatomy = use_anatomy
        self.anatomy = AnatomicalSolver(anatomy_frames)
        if pan_mode not in ("depth", "azimuth", "lateral"):
            raise ValueError("pan_mode must be 'depth', 'azimuth' or 'lateral'")
        self.pan_mode = pan_mode
        self.depth_scale = depth_scale
        self.pan_gain = pan_gain
        self.pan_deadzone = pan_deadzone
        self.depth_span = depth_span
        self.depth_calib_frames = depth_calib_frames
        self._last_pan = 0.0
        self._depth_ref = None       # 깊이 중립 기준 (처음 몇 프레임 평균)
        self._depth_samples = []
        # 관절별 개별 배율. 전체 배율(gain)에 곱해진다.
        self.joint_gains = {k: 1.0 for k in JOINT_LIMITS}
        # 실행 중 키로 조절하는 관절별 0점 보정(도).
        self.offsets = {k: 0.0 for k in JOINT_LIMITS}
        self.roll_neutral = roll_neutral
        self.invert_joints = set(invert_joints)
        unknown = self.invert_joints - set(JOINT_LIMITS)
        if unknown:
            raise ValueError(f"invert_joints 에 알 수 없는 관절: {unknown}")
        for path in (pose_model_path, HAND_MODEL):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"모델 파일이 없습니다: {path}\n"
                    "  python download_models.py 를 먼저 실행하세요."
                )
        self.side = side
        self.flip = flip
        self.cap = CameraStream(camera_index)

        # 경로에 한글이 있으면 MediaPipe 의 C++ 로더가 파일을 못 여므로 바이트로 넘긴다.
        with open(pose_model_path, "rb") as f:
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
        # shoulder_pan 은 MediaPipe z(depth) 기반이라 노이즈가 다른 축의 10배 수준이다.
        # (정지 상태 측정: pan 표준편차 14.7 vs 나머지 1~3)
        # 컷오프를 크게 낮추고 beta 도 줄여 강하게 눌러준다.
        per_joint = {}
        if self.pan_mode == "depth" or self.use_world:
            per_joint["shoulder_pan"] = (pan_cutoff, pan_beta)
        self.filters = OneEuroBank(min_cutoff=min_cutoff, beta=beta,
                                   per_joint=per_joint)
        self.last_action = {k: 0.0 for k in JOINT_LIMITS}
        self.last_action["gripper"] = 50.0
        self._t0 = None
        self._frame_no = 0
        self._hand_res = None      # 손 추론을 건너뛴 프레임에서 재사용

    def reset_depth_reference(self):
        """현재 자세를 깊이 중립(pan 0도)으로 다시 잡는다."""
        self._depth_ref = None
        self._depth_samples = []

    # ---------- landmark -> 관절 ----------

    # ---------- 3D world landmark 기반 (투영 왜곡 없음) ----------

    def _refine_with_anatomy(self, wl, lms, w, h):
        """뼈 길이 제약으로 위팔·아래팔 벡터를 다시 푼다.

        반환: (upper, fore) 3D 벡터 쌍. 보정 불가하면 None.
        """
        s_i, e_i, w_i, o_i = SIDES[self.side]
        # world 좌표(추정) — 뼈 길이 관측과 z 부호에만 쓴다
        sw, ew, ww = lm_xyz(wl[s_i]), lm_xyz(wl[e_i]), lm_xyz(wl[w_i])
        upper_w, fore_w = v_sub(ew, sw), v_sub(ww, ew)

        self.anatomy.observe({"upper": v_norm(upper_w), "fore": v_norm(fore_w)})
        if not self.anatomy.ready:
            return None

        # 이미지 픽셀 -> 미터 스케일 자: 어깨너비
        shoulder_px = math.hypot((lms[s_i].x - lms[o_i].x) * w,
                                 (lms[s_i].y - lms[o_i].y) * h)
        shoulder_m = v_norm(v_sub(lm_xyz(wl[s_i]), lm_xyz(wl[o_i])))
        if shoulder_px < 1e-6 or shoulder_m < 1e-6:
            return None
        px_per_m = shoulder_px / shoulder_m

        upper = self.anatomy.solve_bone(
            "upper", (lms[e_i].x - lms[s_i].x) * w, (lms[e_i].y - lms[s_i].y) * h,
            upper_w[2], px_per_m)
        fore = self.anatomy.solve_bone(
            "fore", (lms[w_i].x - lms[e_i].x) * w, (lms[w_i].y - lms[e_i].y) * h,
            fore_w[2], px_per_m)
        if upper is None or fore is None:
            return None
        return upper, fore

    def _pose_joints_3d(self, wl, lms=None, w=None, h=None):
        """pose_world_landmarks(미터 단위 3D)로 관절각을 푼다.

        핵심은 **몸통 기준 좌표계**를 만드는 것이다. 카메라 좌표 그대로 쓰면
        사람이 몸을 틀거나 팔을 카메라 쪽으로 뻗을 때 각도가 뭉개진다(foreshortening).
        어깨선과 척추로 직교 기저를 세우고 팔 벡터를 거기에 투영하면,
        "팔을 어떻게 보여주든" 몸 기준의 동일한 각도가 나온다.

          u_lat  : 몸의 좌우축 (반대쪽 어깨 -> 추적하는 어깨 방향이 +)
          u_up   : 몸의 위쪽축 (엉덩이중심 -> 어깨중심)
          u_fwd  : 몸의 정면축 (u_lat x u_up)
        """
        s_i, e_i, w_i, o_i = SIDES[self.side]
        hip_i = 23 if self.side == "left" else 24
        opp_hip_i = 24 if self.side == "left" else 23

        s = lm_xyz(wl[s_i])
        e = lm_xyz(wl[e_i])
        wrist = lm_xyz(wl[w_i])
        opp_s = lm_xyz(wl[o_i])
        hip = lm_xyz(wl[hip_i])
        opp_hip = lm_xyz(wl[opp_hip_i])

        # --- 몸통 직교 기저 ---
        lat = v_sub(s, opp_s)                       # 어깨선 (바깥쪽이 +)
        shoulder_mid = tuple((s[k] + opp_s[k]) / 2 for k in range(3))
        hip_mid = tuple((hip[k] + opp_hip[k]) / 2 for k in range(3))
        up = v_sub(shoulder_mid, hip_mid)           # 척추 (위쪽이 +)

        if v_norm(lat) < 1e-4 or v_norm(up) < 1e-4:
            return None

        u_lat = v_unit(lat)
        # up 에서 lat 성분을 제거해 직교화 (Gram-Schmidt)
        up_orth = v_sub(up, tuple(v_dot(up, u_lat) * c for c in u_lat))
        u_up = v_unit(up_orth)
        u_fwd = v_unit(v_cross(u_lat, u_up))

        # --- 위팔 벡터를 몸통 기저에 투영 ---
        # 뼈 길이 제약으로 다시 푼 벡터가 있으면 그쪽을 쓴다 (z 잡음 제거됨).
        refined = None
        if self.use_anatomy and lms is not None:
            refined = self._refine_with_anatomy(wl, lms, w, h)

        if refined is not None:
            upper, fore_vec = refined
        else:
            upper, fore_vec = v_sub(e, s), v_sub(wrist, e)
        if v_norm(upper) < 1e-4:
            return None
        a_lat = v_dot(upper, u_lat)
        a_up = v_dot(upper, u_up)
        a_fwd = v_dot(upper, u_fwd)
        arm_len = v_norm(upper)

        # shoulder_lift: 수평면 기준 위팔의 올라간 각도 (-90 아래 ~ +90 위)
        shoulder_lift = math.degrees(math.asin(clamp(a_up / arm_len, -1.0, 1.0)))
        shoulder_lift *= self.lift_gain

        # shoulder_pan: 수평면상의 방위각. 0=정면, +90=몸 바깥쪽 옆.
        # 팔을 앞/옆 어디로 뻗든 몸 기준으로 일관된 값이 나온다.
        horiz = math.hypot(a_lat, a_fwd)
        if horiz < arm_len * 0.15:
            # 팔이 거의 수직(위/아래)이면 방위각이 불안정 -> 직전 값 유지
            shoulder_pan = self._last_pan
        else:
            shoulder_pan = math.degrees(math.atan2(a_lat, a_fwd)) * self.pan_gain
            self._last_pan = shoulder_pan

        # elbow_flex: 3D 사잇각이라 투영 왜곡이 원천적으로 없다.
        # 위팔의 반대방향과 아래팔 사이각 = 팔꿈치 안쪽각.
        fore = fore_vec
        elbow_inner = v_angle(tuple(-c for c in upper), fore)
        elbow_flex = (180.0 - elbow_inner) * self.elbow_gain

        return {
            "shoulder_pan": shoulder_pan,
            "shoulder_lift": shoulder_lift,
            "elbow_flex": elbow_flex,
        }, (fore, u_up, u_lat)

    def _wrist_flex_3d(self, wl, fore, u_up, u_lat):
        """손목 굽힘을 3D 로 푼다. pose 의 손 랜드마크(검지/새끼 MCP)를 쓴다."""
        w_i = SIDES[self.side][2]
        idx_i = 19 if self.side == "left" else 20     # index MCP
        pky_i = 17 if self.side == "left" else 18     # pinky MCP
        wrist = lm_xyz(wl[w_i])
        hand = v_sub(lm_xyz(wl[idx_i]), wrist)
        if v_norm(hand) < 1e-4 or v_norm(fore) < 1e-4:
            return None

        # 아래팔 대비 손 방향의 사잇각에, 굽힘 방향 부호를 붙인다.
        bend = v_angle(fore, hand)
        # 손등/손바닥 축(아래팔에 수직인 평면의 법선)으로 부호 결정
        axis = v_unit(v_cross(fore, v_sub(lm_xyz(wl[pky_i]), wrist)))
        sign = 1.0 if v_dot(v_cross(fore, hand), axis) >= 0 else -1.0
        return bend * sign * self.wrist_gain

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

            # 1) shoulder_pan: 베이스 회전. pan_mode 로 계산 방식을 고른다.
            #    MediaPipe z 는 x 와 같은 스케일이고, 카메라에 가까울수록 음수다.
            dz = (wr.z - s.z) * w * self.depth_scale

            if self.lock_pan:
                shoulder_pan = self.pan_hold

            elif self.pan_mode == "depth":
                # 깊이만으로 좌우 결정: 기준보다 가까우면 왼쪽(-), 멀면 오른쪽(+).
                #
                # 주의: 팔을 뻗으면 손목은 늘 어깨보다 카메라 쪽(음수)이라
                # dz 절대값을 그대로 쓰면 한쪽으로만 쏠린다. 그래서 처음 몇 초의
                # 평균 깊이를 중립(0도)으로 잡고, 거기서의 "변화량"으로 좌우를 정한다.
                depth_ratio = dz / upper_arm
                if self._depth_ref is None:
                    self._depth_samples.append(depth_ratio)
                    if len(self._depth_samples) >= self.depth_calib_frames:
                        self._depth_ref = (sum(self._depth_samples)
                                           / len(self._depth_samples))
                    shoulder_pan = self.pan_hold
                else:
                    delta = depth_ratio - self._depth_ref
                    shoulder_pan = remap(delta, -self.depth_span, self.depth_span,
                                         -90.0, 90.0) * self.pan_gain

            elif self.pan_mode == "azimuth":
                # 수평면상의 진짜 방위각. 0도=정면, +90도=몸 바깥쪽 옆.
                dx = away * (wx - sx)
                # 팔을 아래로 내리면 수평 성분이 0 에 가까워져 방위각이 마구 튄다.
                # 수평으로 충분히 뻗었을 때만 갱신하고, 아니면 직전 값을 유지한다.
                if math.hypot(dx, dz) < upper_arm * self.pan_deadzone:
                    shoulder_pan = self._last_pan
                else:
                    shoulder_pan = math.degrees(math.atan2(dx, -dz)) * self.pan_gain
                    self._last_pan = shoulder_pan

            else:  # "lateral" — 화면상 좌우 오프셋만 사용 (depth 미사용)
                shoulder_pan = remap(away * (wx - sx) / upper_arm,
                                     -2.0, 2.0, -100.0, 100.0) * self.pan_gain

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

        # 6) gripper: 2D 폴백 — 엄지 끝 <-> 나머지 4손가락 끝 평균거리
        tx, ty = lm[THUMB_TIP].x * w, lm[THUMB_TIP].y * h
        pinch = sum(
            math.hypot(lm[i].x * w - tx, lm[i].y * h - ty) / palm for i in OTHER_TIPS
        ) / len(OTHER_TIPS)
        out["gripper"] = remap(pinch, 0.35, 1.6, 0.0, 100.0)
        return out

    def _gripper_3d(self, hwl):
        """3D world landmark 의 손가락 관절각으로 주먹/폄을 판정한다.

        기존 방식(손끝 사이 거리)은 손을 옆에서 보거나 카메라 쪽으로 향하면
        거리가 투영으로 줄어들어 주먹으로 오인했다. 관절각 합은 손의
        방향·거리·크기와 무관해서 그런 오인이 없다.

        네 손가락의 굽힘 정도를 중앙값으로 모아 튀는 손가락 하나에
        휘둘리지 않게 한다.
        """
        curls = sorted(finger_curl(hwl, c) for c in FINGER_CHAINS.values())
        curl = (curls[1] + curls[2]) / 2.0        # 중앙 2개 평균 = 이상치에 강함
        return clamp((1.0 - curl) * 100.0, 0.0, 100.0)

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

        # 조도 보정은 추론용 이미지에만 적용한다. 화면에 보여주는 frame 은
        # 원본 그대로 둬야 사용자가 실제 조명 상태를 판단할 수 있다.
        infer_bgr = self.enhancer(frame) if self.enhancer else frame

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB),
        )
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if self._t0 is None:
            self._t0 = now
        elapsed = now - self._t0
        ts = max(int(elapsed * 1000), 0)

        pose_res = self.pose.detect_for_video(mp_img, ts)

        # 손 추론은 pose 만큼 자주 할 필요가 없다 (그리퍼는 천천히 변한다).
        # hand_interval 프레임마다 한 번만 돌리고 사이에는 직전 결과를 재사용해
        # 전체 프레임레이트를 끌어올린다.
        self._frame_no += 1
        if self._frame_no % self.hand_interval == 0 or self._hand_res is None:
            self._hand_res = self.hands.detect_for_video(mp_img, ts)
        hand_res = self._hand_res

        raw, forearm = {}, None
        solved_3d = False
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]

            if self.use_world and pose_res.pose_world_landmarks:
                # 3D world landmark 로 푼다 (투영 왜곡 없음).
                wl = pose_res.pose_world_landmarks[0]
                got = self._pose_joints_3d(wl, lms, w, h)
                if got:
                    joints, (fore3d, u_up, u_lat) = got
                    raw.update(joints)
                    wf = self._wrist_flex_3d(wl, fore3d, u_up, u_lat)
                    if wf is not None:
                        raw["wrist_flex"] = wf
                    solved_3d = True

            if not solved_3d:
                got = self._pose_joints(lms, w, h)
                if got:
                    joints, forearm = got
                    raw.update(joints)

            self._draw(frame, lms, ARM_CHAIN[self.side], (0, 200, 255), w, h)
        if hand_res.hand_landmarks:
            lms = hand_res.hand_landmarks[0]
            hand_joints = self._hand_joints(lms, w, h, forearm)
            # 그리퍼는 3D 손가락 관절각이 훨씬 정확하다 (투영 왜곡 없음).
            if self.use_world and hand_res.hand_world_landmarks:
                hand_joints["gripper"] = self._gripper_3d(
                    hand_res.hand_world_landmarks[0])
            if solved_3d:
                # 3D 로 이미 푼 축은 2D 결과로 덮어쓰지 않는다. 그리퍼만 손 모델에서.
                hand_joints = {k: v for k, v in hand_joints.items()
                               if k not in ("wrist_flex", "wrist_roll")}
            raw.update(hand_joints)
            self._draw(frame, lms, HAND_LINKS, (255, 200, 0), w, h)

        if solved_3d and self.lock_roll:
            raw["wrist_roll"] = self.roll_hold

        if not raw:
            cv2.putText(frame, "no arm detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, frame

        action = dict(self.last_action)
        for name, value in raw.items():
            lo, hi = JOINT_LIMITS[name]
            # 각도 관절만 배율 적용. 그리퍼(0~100)와 고정된 축은 제외한다.
            # 관절별 배율(joint_gains)이 있으면 전체 배율에 곱해진다.
            if name in GAIN_JOINTS and not (name == "wrist_roll" and self.lock_roll):
                value *= self.gain * self.joint_gains.get(name, 1.0)
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
