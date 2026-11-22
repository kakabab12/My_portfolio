"""MediaPipe 손 찾기 + 떨림 거르기 (2026-11-15).

원래 웹캠 조종(My_portfolio 2026/05)은 팔 관절 각도를 썼지만, 여기서는 손 하나만 본다.
손 21점의 화면 위치와 손 기준 3D 좌표(m)를 함께 받아 geometry.HandMeasure 로 책상 좌표를 구한다.
"""
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(HERE, "models", "hand_landmarker.task")
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


class OneEuro:
    """1€ 필터 (Casiez 2012) — 원래 웹캠 조종 코드와 같은 식. 멈춰 있으면 강하게, 빨리 움직이면 약하게 거른다."""

    def __init__(self, min_cutoff=1.5, beta=0.5, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        x = np.asarray(x, np.float64)
        if self.t is None:
            self.t, self.x, self.dx = t, x, np.zeros_like(x)
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x

    def reset(self):
        self.x = self.dx = self.t = None


class Median3:
    """앞 3장면의 가운데 값 (축마다, 11/20). 한 장면만 튀는 값을 버린다.
    움직일 때 쓰면 한 장면(~40 ms) 늦으므로 Smoother 는 손이 멈춰 있을 때만 쓴다."""

    def __init__(self):
        self.buf = []

    def __call__(self, x):
        self.buf = (self.buf + [np.asarray(x, np.float64)])[-3:]
        return np.median(np.array(self.buf), axis=0)

    def reset(self):
        self.buf = []


class StillZone:
    """멈춤 구역 (11/20, 손떨림·지터 없애기). 값이 반지름 r 안에서만 떨리면 내보내는 값을 그대로 두고
    (tau 초에 걸쳐 천천히 가운데로만 맞춤), r 밖으로 나가면 넘은 만큼만 끌려간다 (줄에 묶인 것처럼).
    움직이는 동안은 r 만큼 뒤에서 따라오고, 멈추면 tau 안에 따라잡는다. r = 0 이면 그대로 내보낸다.
    가상 손(손떨림 9 Hz 1 mm, 느린 흔들림 1.5 mm, 잡음 xy 2·z 5 mm, 23 fps)에서 r 3 mm 가 멈춤 흔들림을 가장 줄였다."""

    def __init__(self, r=0.003, tau=0.4):
        self.r, self.tau = float(r), float(tau)
        self.y = self.t = None

    def __call__(self, t, x):
        x = np.asarray(x, np.float64)
        if self.r <= 0:
            return x
        if self.y is None:
            self.y, self.t = x.copy(), t
            return self.y.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        d = x - self.y
        n = float(np.linalg.norm(d))
        if n > self.r:
            self.y = self.y + d * ((n - self.r) / n)
            d = x - self.y
        self.y = self.y + d * (1.0 - math.exp(-dt / self.tau))
        return self.y.copy()

    def reset(self):
        self.y = self.t = None


class HandTracker:
    """웹캠 한 장 → 원하는 손(오른손/왼손)의 화면 21점(px)과 손 3D 21점(m)."""

    def __init__(self, hand="right", model=HAND_MODEL, min_conf=0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        if not os.path.exists(model):
            raise FileNotFoundError(f"손 모델이 없음: {model}\n  python download_models.py 를 먼저 실행하세요")
        self.mp = mp
        self.want = hand
        # 두 손이 다 보일 때 고를 MediaPipe 표시. 처음에는 '원본 사진이면 오른손이 Left' 로 어림하고,
        # 보정 때 손 하나만 보일 때 실제로 나온 표시로 바꾼다 (teleop.py, calib.json 의 label). 11/16
        self.want_label = "Left" if hand == "right" else "Right"
        self.det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf))
        self._last_ts = -1

    def detect(self, frame_bgr, t_ms):
        """돌려줌: dict(img_px 21×2, world 21×3, label, score) 또는 None.
        손이 하나만 보이면 그 손을 쓴다. 둘이면 표시가 want_label 인 손을 쓴다.
        MediaPipe 문서는 표시가 거울(셀카) 사진을 가정한다고 하지만, 11/16 에 예시 사진으로 보니 그대로가 아닐 수 있어서
        어느 쪽이라고 정해 두지 않고 보정 때 잰 표시를 쓴다."""
        h, w = frame_bgr.shape[:2]
        ts = max(int(t_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.hand_landmarks:
            return None
        pick = 0
        if len(res.hand_landmarks) > 1:
            for i, hd in enumerate(res.handedness):
                if hd and hd[0].category_name == self.want_label:
                    pick = i
                    break
        hands = []
        for i in range(len(res.hand_landmarks)):
            hd = res.handedness[i][0] if res.handedness and res.handedness[i] else None
            hands.append({"img_px": np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[i]]),
                          "world": np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]]),
                          "label": hd.category_name if hd else "?", "score": hd.score if hd else 0.0})
        out = dict(hands[pick])
        out["all"] = hands                     # 11/18: 카메라 두 대일 때 두 화면에서 같은 손끼리 짝지으려고 다 넘김
        return out

    def close(self):
        self.det.close()


def draw_hand(img, img_px, color=(0, 255, 0), tips=(4, 8)):
    """tips: 빨간 점으로 표시할 손가락 끝 (지금 쓰는 손 모양을 화면에서 바로 알 수 있게)."""
    for a, b in HAND_LINKS:
        cv2.line(img, tuple(int(v) for v in img_px[a]), tuple(int(v) for v in img_px[b]), color, 2)
    for i in tips:
        cv2.circle(img, tuple(int(v) for v in img_px[i]), 6, (0, 0, 255), -1)
