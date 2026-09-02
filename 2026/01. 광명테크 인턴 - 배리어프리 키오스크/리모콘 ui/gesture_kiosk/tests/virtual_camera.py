"""가상 카메라 — 임의의 배치·거리·잡음에서 얼굴 관측을 합성한다 (2026-08-31 신설).

왜 필요한가
-----------
지금까지 검증은 **개발용 노트북 카메라 한 대**로만 했다. 그 카메라 한 대에서
잘 도는 것과, 현장의 어떤 카메라에서도 잘 도는 것은 다른 이야기다. 실제로
연구실 키오스크는 밑에서 올려보는 배치라 개발 노트북과 조건이 완전히 다르다.

카메라를 여러 대 구해 오는 대신, **관측을 합성**한다. 카메라가 하는 일은
결국 "3차원 얼굴을 어떤 자세에서 보고 2차원에 투영하는 것"이므로, 그 과정을
그대로 계산하면 어떤 배치의 카메라든 만들어 낼 수 있다. 합성이라서 **정답을
알고 있다**는 것이 핵심이다 — 커서가 어디에 있어야 하는지 알고 비교할 수 있다.

무엇을 재현하나
---------------
  · 카메라 배치 — 밑에서 올려봄·위에서 내려봄·옆에서·기울어짐(임의 회전)
  · 거리 — 가까이/멀리 (원근 세기와 얼굴 크기가 함께 바뀐다)
  · 화각 — 광각(왜곡 심함) / 망원(거의 평행투영)
  · 랜드마크 잡음 — 실측 기반(안구간거리 60px에서 커서 3.0px, 8/26 측정)
  · 거울 반전 — 프레임 좌우 반전 여부

무엇을 재현하지 못하나 (정직하게)
---------------------------------
MediaPipe가 **그 각도에서 얼굴을 얼마나 잘 검출하는지**는 재현할 수 없다.
밑에서 보면 콧구멍이 보이고 눈이 눌려 보여 검출 정확도가 달라질 수 있는데,
그것은 실제 영상이 있어야 알 수 있다. 이 도구가 보장하는 것은

    "매핑 수식이 어떤 배치에서도 옳은 커서를 낸다"        <- 확인 가능
    "그 배치에서 얼굴이 잘 잡힌다"                        <- 확인 불가

전자가 우리 코드의 몫이고, 후자는 MediaPipe와 조명의 몫이다.

핀홀 카메라 모형
----------------
표준 핀홀 모형을 쓴다 (Hartley & Zisserman, "Multiple View Geometry in
Computer Vision", 2nd ed., 2004, §6.1):

    x_화면 = f · X_카메라 / Z_카메라 + c

여기서 f는 초점거리(화각), c는 화면 중심이다. 얼굴 점을 카메라 좌표로 옮긴
뒤 이 식을 적용하면, 원근 왜곡(가까운 점이 더 크게 움직이는 것)이 자연히
생긴다 — ARC_COMPENSATION이 상수로 보정하려던 바로 그 현상이다.
"""
import math

import numpy as np

# MediaPipe가 돌려주는 정규화 좌표의 척도를 흉내 내기 위한 기준 프레임 크기.
# 실제 파이프라인의 9:16 크롭 결과(405x720)와 같게 둔다
FRAME_W_PX = 405
FRAME_H_PX = 720

# 랜드마크 잡음 표준편차(픽셀). 8/26 실측(안구간거리 60px에서 커서 3.0px 흔들림)
# 에서 역산한 값 — 점 하나당 이 정도가 흔들린다고 보면 그 커서 흔들림이 나온다
LANDMARK_NOISE_PX = 0.35

# 얼굴 모형 — tests/test_head_orientation.py의 합성 머리와 같은 치수(mm 어림).
# 두 곳이 같은 얼굴을 써야 시험 결과를 나란히 읽을 수 있다
FACE_MODEL = {
    33: (-32.0, 0.0, 4.0), 263: (32.0, 0.0, 4.0),
    133: (-12.0, 1.0, 0.0), 362: (12.0, 1.0, 0.0),
    168: (0.0, -6.0, -6.0), 6: (0.0, -1.0, -10.0),
    197: (0.0, 3.0, -13.0), 195: (0.0, 8.0, -17.0),
    5: (0.0, 13.0, -21.0), 4: (0.0, 18.0, -25.0),
    8: (0.0, -11.0, -3.0),
    234: (-46.0, 2.0, 22.0), 454: (46.0, 2.0, 22.0),
    10: (0.0, -58.0, 2.0), 151: (0.0, -46.0, -2.0),
    9: (0.0, -34.0, -5.0),
    107: (-14.0, -30.0, -4.0), 336: (14.0, -30.0, -4.0),
    117: (-34.0, 20.0, 8.0), 346: (34.0, 20.0, 8.0),
    50: (-38.0, 12.0, 10.0), 280: (38.0, 12.0, 10.0),
}
LANDMARK_COUNT = 478


def rotation(axis, degrees):
    """축 둘레 회전행렬 — 로드리게스 공식."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = math.radians(degrees)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


class VirtualFace:
    """가상 카메라가 만들어 낸 한 프레임의 얼굴 — FaceLandmarks 대역."""

    def __init__(self, landmarks_3d, head_rotation):
        self.landmarks_3d = landmarks_3d
        self.head_rotation = head_rotation
        self.landmarks_px = np.asarray(landmarks_3d, dtype=np.float32)[:, :2]
        self.blendshapes = {}

    def landmark_px(self, index):
        x, y = self.landmarks_px[index]
        return float(x), float(y)

    def blendshape(self, name, default=0.0):
        return self.blendshapes.get(name, default)


class VirtualCamera:
    """임의 배치의 카메라. observe(머리회전) -> VirtualFace.

    mount        카메라를 어떻게 달았나 (3x3 회전; 세계 -> 카메라)
    distance_mm  얼굴까지 거리. 작을수록 원근 왜곡이 세다
    focal_px     초점거리. 크면 망원(평행투영에 가깝다), 작으면 광각
    mirror       프레임을 좌우 반전하는가 (실제 파이프라인 기본값 True)
    noise_px     랜드마크 잡음 표준편차
    """

    def __init__(self, mount=None, distance_mm=600.0, focal_px=700.0,
                 mirror=True, noise_px=LANDMARK_NOISE_PX, seed=0):
        self.mount = np.eye(3) if mount is None else np.asarray(mount, dtype=np.float64)
        self.distance_mm = distance_mm
        self.focal_px = focal_px
        self.mirror = mirror
        self.noise_px = noise_px
        self._rng = np.random.default_rng(seed)

    def observe(self, head_rotation=None, offset_mm=(0.0, 0.0, 0.0)):
        """머리를 그만큼 돌린 상태를 이 카메라로 본다 -> VirtualFace.

        offset_mm은 몸이 움직인 것(평행이동) — 커서가 따라가면 안 되는 성분이라
        시험에서 자주 쓴다.
        """
        head_rotation = np.eye(3) if head_rotation is None else np.asarray(head_rotation)
        pts = np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)

        for idx, model_xyz in FACE_MODEL.items():
            world = head_rotation @ np.asarray(model_xyz, dtype=np.float64)
            world = world + np.asarray(offset_mm, dtype=np.float64)
            # 카메라 좌표로: 카메라 배치를 걸고 거리만큼 앞에 둔다
            cam = self.mount @ world
            cam = cam + np.array([0.0, 0.0, self.distance_mm])
            depth = max(1.0, cam[2])
            # 핀홀 투영 (독스트링의 Hartley & Zisserman §6.1)
            x_px = self.focal_px * cam[0] / depth + FRAME_W_PX * 0.5
            y_px = self.focal_px * cam[1] / depth + FRAME_H_PX * 0.5
            # MediaPipe의 z 규약: x와 같은 척도, 가까울수록 작다(음수 쪽)
            z_px = self.focal_px * (depth - self.distance_mm) / self.distance_mm
            if self.mirror:
                x_px = FRAME_W_PX - x_px
            pts[idx] = (x_px, y_px, z_px)

        if self.noise_px > 0.0:
            idx = list(FACE_MODEL)
            pts[idx] += self._rng.normal(0.0, self.noise_px, (len(idx), 3))

        # MediaPipe의 변환행렬에 해당하는 값 — 카메라가 보는 머리의 자세.
        # 거울이면 실제 장치처럼 반사 켤레로 돌려준다(head_orientation의
        # 부호 자가 학습이 이것을 다루는지 시험하기 위해)
        observed = self.mount @ head_rotation
        if self.mirror:
            flip = np.diag([-1.0, 1.0, 1.0])
            observed = flip @ observed @ flip
        return VirtualFace(pts, observed)


# 현장에서 실제로 나올 법한 배치 모음 — 시험이 공통으로 쓴다
MOUNTS = {
    "정면": np.eye(3),
    "밑에서 20도": rotation((1.0, 0.0, 0.0), 20.0),
    "밑에서 35도": rotation((1.0, 0.0, 0.0), 35.0),
    "위에서 25도": rotation((1.0, 0.0, 0.0), -25.0),
    "왼쪽에서 25도": rotation((0.0, 1.0, 0.0), 25.0),
    "오른쪽에서 25도": rotation((0.0, 1.0, 0.0), -25.0),
    "옆으로 기울어짐 20도": rotation((0.0, 0.0, 1.0), 20.0),
    "비스듬히": rotation((0.5, -0.4, 0.3), 28.0),
    "심하게 비스듬히": rotation((0.6, 0.5, -0.4), 40.0),
}
