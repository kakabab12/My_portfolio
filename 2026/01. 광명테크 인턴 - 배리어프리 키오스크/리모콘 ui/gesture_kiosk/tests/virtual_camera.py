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
  · 화각 — 광각 / 망원(거의 평행투영)
  · **렌즈 왜곡** — 광각 렌즈의 배럴 왜곡(2026-09-03 추가)
  · 랜드마크 잡음 — 실측 기반(안구간거리 60px에서 커서 3.0px, 8/26 측정)
  · 거울 반전 — 프레임 좌우 반전 여부
  · **사람마다 다른 얼굴** — 안구간거리·얼굴 폭·코 높이 (2026-09-03 추가)

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

렌즈 왜곡 — 광각의 진짜 문제 (2026-09-03 추가)
----------------------------------------------
"광각 카메라에서도 되나"를 초점거리(focal_px)만 줄여서 시험하던 것은 **틀린
시험**이었다. 초점거리는 배율일 뿐이고, 상대 회전 매핑은 배율에 원리적으로
면역이라 무엇을 넣어도 통과한다. 광각 렌즈의 진짜 문제는 **배럴 왜곡**이다 —
직선이 바깥으로 휘고, 그 휨은 화면 중심에서 멀수록 세다.

Brown-Conrady 방사왜곡 모형을 쓴다:

    x_d = x_u (1 + k1 r^2 + k2 r^4),    r^2 = x_u^2 + y_u^2

  · **Conrady, A.E. (1919).** "Decentred Lens-Systems." *Monthly Notices of the
    Royal Astronomical Society*, 79(5), 384-390.
  · **Brown, D.C. (1966).** "Decentering Distortion of Lenses."
    *Photogrammetric Engineering*, 32(3), 444-462. — k1·k2 방사 계수의 출처.
  · **Zhang, Z. (2000).** "A Flexible New Technique for Camera Calibration."
    *IEEE TPAMI* 22(11), 1330-1334. — 이 모형이 표준 보정에 쓰이는 형태.

x_u는 **정규화 좌표(X/Z)** 다. 왜곡은 초점거리를 곱하기 **전**에 걸어야
물리적으로 옳다 — 렌즈가 빛을 휘게 한 뒤에 센서가 배율을 먹이는 순서다.

주의 — LENS_PROFILES의 계수는 그 화각대 렌즈에서 **나올 법한 대표값**이지
특정 제품의 실측값이 아니다. 이 도구가 답하는 질문은 "이 정도 왜곡에서
매핑이 무너지는가"이지 "이 카메라의 k1이 얼마인가"가 아니다.
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


def varied_face(interocular=1.0, width=1.0, height=1.0, nose=1.0, scale=1.0):
    """사람마다 다른 얼굴을 만든다 -> FACE_MODEL과 같은 모양의 딕셔너리.

    왜 필요한가 (2026-09-03)
    ------------------------
    렌즈 자가 보정은 **정규 얼굴 모형을 보정판으로 삼는다.** 그런데 실제
    사람 얼굴은 제각각이라, 보정판의 치수가 틀린 셈이 된다. 그것이 얼마나
    해로운지 재려면 정규 모형과 **다른** 얼굴을 만들 수 있어야 한다.

    지금까지는 가상 카메라와 렌즈 보정이 같은 FACE_MODEL을 써서, 내가 만든
    얼굴을 내가 맞히는 순환 시험이었다.

    인체 계측 범위 (성인 기준의 어림):
      안구간거리  55~70mm  (평균 63) -> 0.87 ~ 1.11
      얼굴 폭     +-10%
      코 높이     15~30mm  -> 0.6 ~ 1.2
      전체 크기   어린이~큰 어른  0.85 ~ 1.15
    """
    out = {}
    for idx, (x, y, z) in FACE_MODEL.items():
        # 눈 관련 점만 안구간거리로 따로 늘린다 (다른 폭과 독립적으로 변한다)
        gain_x = interocular if idx in (33, 263, 133, 362) else width
        # 코는 앞으로 튀어나온 양(-z 방향)이 사람마다 특히 다르다
        gain_z = nose if idx in (4, 5, 195, 197, 6, 168) else 1.0
        out[idx] = (x * gain_x * scale, y * height * scale, z * gain_z * scale)
    return out


# 현장에서 만날 법한 얼굴들 — 정규 모형과 얼마나 다른지가 요점이다
FACE_VARIANTS = {
    "정규 모형": FACE_MODEL,
    "눈 좁은 얼굴": varied_face(interocular=0.87),
    "눈 넓은 얼굴": varied_face(interocular=1.11),
    "코 낮은 얼굴": varied_face(nose=0.60),
    "코 높은 얼굴": varied_face(nose=1.20),
    "갸름한 얼굴": varied_face(width=0.90, height=1.08),
    "넓은 얼굴": varied_face(width=1.10, height=0.94),
    "작은 얼굴(어린이)": varied_face(scale=0.85),
    "큰 얼굴": varied_face(scale=1.15),
    "겹친 최악": varied_face(interocular=0.87, width=1.10, height=0.94,
                             nose=0.60, scale=1.15),
}

# 화각대별 대표 렌즈 (k1, k2, focal_px). 위 독스트링의 주의 참고 —
# 실측값이 아니라 "그 화각대에서 나올 법한" 값이다
LENS_PROFILES = {
    "왜곡없음":      (0.00, 0.00, 700.0),   # 이상적 핀홀 — 대조군
    "일반 65도":     (-0.05, 0.00, 700.0),  # 보통 노트북 웹캠
    "준광각 78도":   (-0.10, 0.01, 560.0),
    "광각 90도":     (-0.15, 0.02, 430.0),  # 회의용/키오스크 웹캠
    "초광각 120도":  (-0.30, 0.10, 290.0),  # 넓은 대기공간을 담는 키오스크
}


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
    k1, k2       Brown-Conrady 방사왜곡 계수 (음수면 배럴 — 광각 렌즈)
    lens         LENS_PROFILES의 이름. 주면 k1·k2·focal_px를 덮어쓴다
    face         얼굴 모형. FACE_VARIANTS의 이름이나 딕셔너리. 기본은 정규 모형
    mirror       프레임을 좌우 반전하는가 (실제 파이프라인 기본값 True)
    noise_px     랜드마크 잡음 표준편차
    """

    def __init__(self, mount=None, distance_mm=600.0, focal_px=700.0,
                 mirror=True, noise_px=LANDMARK_NOISE_PX, seed=0,
                 k1=0.0, k2=0.0, lens=None, face=None):
        self.mount = np.eye(3) if mount is None else np.asarray(mount, dtype=np.float64)
        self.distance_mm = distance_mm
        self.focal_px = focal_px
        # lens="광각 90도"처럼 이름으로 고르면 k1·k2·초점거리를 한꺼번에 받는다
        if lens is not None:
            k1, k2, self.focal_px = LENS_PROFILES[lens]
        self.k1 = k1
        self.k2 = k2
        if face is None:
            self.face = FACE_MODEL
        elif isinstance(face, str):
            self.face = FACE_VARIANTS[face]
        else:
            self.face = face
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

        for idx, model_xyz in self.face.items():
            world = head_rotation @ np.asarray(model_xyz, dtype=np.float64)
            world = world + np.asarray(offset_mm, dtype=np.float64)
            # 카메라 좌표로: 카메라 배치를 걸고 거리만큼 앞에 둔다
            cam = self.mount @ world
            cam = cam + np.array([0.0, 0.0, self.distance_mm])
            depth = max(1.0, cam[2])
            # 핀홀 투영 (독스트링의 Hartley & Zisserman §6.1)
            x_u = cam[0] / depth
            y_u = cam[1] / depth
            # ★렌즈 왜곡은 배율(초점거리)을 먹이기 **전**에 — 렌즈가 먼저 휘고
            # 센서가 나중에 확대한다 (독스트링 Brown 1966)
            if self.k1 or self.k2:
                r2 = x_u * x_u + y_u * y_u
                gain = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
                x_u *= gain
                y_u *= gain
            x_px = self.focal_px * x_u + FRAME_W_PX * 0.5
            y_px = self.focal_px * y_u + FRAME_H_PX * 0.5
            # MediaPipe의 z 규약: x와 같은 척도, 가까울수록 작다(음수 쪽)
            z_px = self.focal_px * (depth - self.distance_mm) / self.distance_mm
            if self.mirror:
                x_px = FRAME_W_PX - x_px
            pts[idx] = (x_px, y_px, z_px)

        if self.noise_px > 0.0:
            idx = list(self.face)
            pts[idx] += self._rng.normal(0.0, self.noise_px, (len(idx), 3))

        # MediaPipe의 변환행렬에 해당하는 값 — 카메라가 보는 머리의 자세.
        # 거울이면 실제 장치처럼 반사 켤레로 돌려준다(head_orientation의
        # 부호 자가 학습이 이것을 다루는지 시험하기 위해)
        if self.k1 or self.k2:
            # ★왜곡이 있으면 변환행렬도 오염된다. MediaPipe는 정규 얼굴 모형을
            # **관측된(= 이미 휘어 버린)** 랜드마크에 맞춰 자세를 푸므로, 그
            # 편향을 그대로 재현하려면 여기서도 같은 정합을 해야 한다.
            # 진짜 회전을 그대로 돌려주면 왜곡을 공짜로 없애 주는 셈이라
            # 시험이 거짓 통과한다
            observed = _fit_model_to_observation(pts)
            if observed is None:
                observed = self.mount @ head_rotation
        else:
            observed = self.mount @ head_rotation
            if self.mirror:
                flip = np.diag([-1.0, 1.0, 1.0])
                observed = flip @ observed @ flip
        return VirtualFace(pts, observed)


def _fit_model_to_observation(points):
    """정규 얼굴 모형을 관측 랜드마크에 맞춘 회전 (MediaPipe 변환행렬 대역).

    Kabsch-Umeyama. 왜곡이 걸린 관측에 맞추면 회전도 그만큼 편향되는데,
    그 편향이 바로 실제 장치에서 일어나는 일이다.
    """
    idx = list(FACE_MODEL)
    p = np.array([FACE_MODEL[i] for i in idx], dtype=np.float64)
    q = np.array([points[i] for i in idx], dtype=np.float64)
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    sp = math.sqrt(float((p ** 2).sum()))
    sq = math.sqrt(float((q ** 2).sum()))
    if sp < 1e-9 or sq < 1e-9:
        return None
    p /= sp
    q /= sq
    try:
        u, _s, vt = np.linalg.svd(p.T @ q)
    except np.linalg.LinAlgError:
        return None
    d = 1.0 if np.linalg.det(vt.T @ u.T) > 0 else -1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


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
