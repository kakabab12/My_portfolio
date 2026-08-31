"""중립 자세 대비 머리의 상대 회전을 랜드마크에서 직접 추정한다 (2026-08-31 신설).

무엇을 푸는가
-------------
"카메라를 어디에 어떤 각도로 달든, 프로그램을 켜면 알아서 맞아야 한다."

지금까지는 그게 안 됐다. 두 가지가 걸림돌이었다.

  1) **곡률 보정(ARC_COMPENSATION)** — 커서를 2차원 화면 좌표로 만들다 보니
     코·미간처럼 튀어나온 점이 원근 때문에 활처럼 휘었고, 그 휘어짐은 카메라
     위치에 따라 달라서 자리를 옮길 때마다 다시 재야 했다.
  2) **오일러 각의 부호 규약** — 회전각(yaw/pitch/roll)을 쓰면 투영 왜곡은
     사라지지만, "위를 보면 pitch가 양수인가 음수인가"가 MediaPipe 내부 축
     규약에 달려 있어서 결국 재 봐야만 알 수 있었다.

이 파일은 **오일러 각을 아예 거치지 않는다**. 시작할 때 얼굴 랜드마크를 한 벌
저장해 두고("중립"), 매 프레임 지금 랜드마크를 그 중립에 겹치는 **최적 회전
행렬**을 직접 구한다. 그러면

  · 카메라가 밑에서 올려보든, 위에서 내려보든, 옆으로 기울어져 있든
    -> 중립도 같은 카메라로 찍힌 것이라 그 배치가 **양쪽에 똑같이 들어 있고**,
       상대 회전을 구할 때 소거된다. 카메라 배치가 수식에서 사라진다.
  · 부호 규약을 알 필요가 없다 — 방향을 중립 얼굴 자신의 축으로 읽기 때문에
    "화면 오른쪽"이 어느 쪽인지가 정의상 정해진다.
  · 잴 것이 남지 않는다. 감도는 "고개를 몇 도 돌리면 화면 끝인가"라는
    **사람 기준 설계값**이 되고, 이건 카메라와 무관하다.

어떻게 구하는가 — Kabsch–Umeyama 정합
--------------------------------------
두 점집합을 가장 잘 겹치는 회전은 닫힌 해가 있다. 각 집합을 무게중심으로
옮기고, 교차공분산 H = P^T Q 를 특이값분해(SVD)한 뒤

    R = V * diag(1, 1, d) * U^T,      d = sign(det(V U^T))

가 최소제곱 최적해다. 가운데 d 가 반사(거울상)를 막는다 — 이걸 빼먹으면
잡음이 클 때 얼굴이 뒤집힌 해가 나와 커서가 순간이동한다.

★회전의 재료 두 가지 (2026-08-31 확장)
--------------------------------------
같은 "중립 대비 상대 회전"이라도 회전을 어디서 얻느냐가 갈린다.

  1) 랜드마크 정합(Kabsch) — 이미지 좌표(+의사 z)의 강체 점 22개를 정합.
     원근 투영을 거친 좌표라 좌우 회전에 세로가 딸려오는 **잔여 곡률**이
     남는다 (실기 곡률 +1.88).
  2) MediaPipe 변환행렬 — 카메라 모델을 넣고 표준 얼굴 모형을 정식 3D
     정합한 결과의 회전 부분. 원근을 이미 소화한 값이라 훨씬 곧다
     (같은 조건 실기 곡률 -0.65).

그래서 기본은 **행렬을 우선**하고, 행렬이 안 오는 프레임은 랜드마크 정합으로
자동 폴백한다. 중요한 것은 두 경로 모두 "중립 대비 상대"라는 것 —
행렬의 축 규약(어느 회전이 양수인가)을 알 필요가 없다. 카메라를 M만큼
돌려 달면 중립도 현재도 M이 곱해져 R_now @ R_neutral^T 에서 소거된다.

★반사 켤레와 부호 자가 학습 (2026-08-31 저녁 — 실기 보고 "좌우가 반대")
------------------------------------------------------------------------
위 소거 논리에는 빈틈이 하나 있었다. 프레임이 거울 반전(mirror)돼 있으면
행렬은 반사 X = diag(-1,1,1)로 **켤레**가 되어 온다: R_관측 = X·R·X.
상대 회전을 구해도 켤레는 안 사라진다 — X·R_rel·X 가 남고, 이것은
**yaw만 뒤집고 pitch는 그대로 둔다** (X R_y(θ) X = R_y(-θ),
X R_x(θ) X = R_x(θ)). 정확히 "좌우만 반대"라는 실기 보고 그대로다.
랜드마크 정합은 반전된 좌표끼리 정합하므로 이 문제가 없다.

처방은 여기서도 "재지 않는다"를 지킨다 — **부호를 실행 중에 스스로 배운다**.
랜드마크 정합의 방향은 정의상 옳다(코가 화면에서 가는 쪽 — 단위 테스트로
증명됨). 그래서 두 값이 동시에 나올 때 축마다 부호가 같은지 투표하고,
충분히 쌓이면(LOCK_VOTES) 그 부호를 확정해 행렬 값에 적용한다. 확정
전에는 방향이 보장된 랜드마크 값을 쓰므로 **커서는 첫 프레임부터 옳은
방향**으로 움직인다. 거울 설정이 바뀌든 MediaPipe 규약이 바뀌든, 코드는
고칠 것이 없다.

여러 장으로 중립 행렬을 잡을 때는 성분 평균을 SVD로 다시 회전에 사영하는
코달 평균을 쓴다 (Hartley, Trumpf, Dai, Li (2013). "Rotation Averaging."
International Journal of Computer Vision 103, 267-305 — L2 chordal mean).

근거 (원문 확인)
  · Kabsch, W. (1976). "A solution for the best rotation to relate two sets of
    vectors." Acta Crystallographica A32, 922-923.
  · Umeyama, S. (1991). "Least-squares estimation of transformation parameters
    between two point patterns." IEEE TPAMI 13(4), 376-380.
    -> 반사 처리(위의 d)를 정식화한 논문. 여기 구현이 따르는 형태다.
  · Horn, B.K.P. (1987). "Closed-form solution of absolute orientation using
    unit quaternions." JOSA A 4(4), 629-642.
    -> 같은 문제의 사원수 해법. 결과는 같고, 여기서는 SVD 쪽을 썼다
       (numpy만으로 끝나고 반사 처리가 명시적이라 읽기 쉽다).

왜 점 3개가 아니라 수십 개인가
------------------------------
눈 두 점과 코 한 점으로도 좌표계는 만들어진다. 하지만 랜드마크 하나하나는
프레임마다 몇 px씩 떨린다 — 이 프로젝트가 8/26에 실측한 값이 안구간거리 60px
기준 커서 3.0px 흔들림이었다. 점 3개면 그 떨림이 그대로 회전에 실린다.

강체(표정으로 안 움직이는) 부위의 점을 수십 개 함께 넣으면 최소제곱이
평균을 내 준다. 잡음이 독립이라면 점 N개에서 대략 sqrt(N)배로 줄어든다.
그래서 아래 RIGID_LANDMARKS는 **표정에 안 움직이는 곳만** 고른다 —
입·턱·눈꺼풀·눈썹은 뺀다(각각 말하기·씹기·깜빡임·찡그림으로 움직인다).
"""
import math

import numpy as np

# 표정에 거의 안 움직이는 랜드마크만 (MediaPipe 478점 기준).
#
# 넣은 곳
#   콧대       6, 197, 195, 5, 4, 168, 8   — 얼굴에서 가장 단단한 축
#   눈 구석    33, 133, 263, 362           — 눈꺼풀이 아니라 '구석'이라 깜빡임에 안 움직인다
#   관자놀이   234, 454                    — 얼굴 폭을 잡아 회전 추정을 안정시킨다
#   이마       10, 151, 9, 107, 336        — 눈썹보다 위라 찡그림의 영향이 적다
#   광대       117, 346, 50, 280           — 좌우 대칭 쌍
#
# 뺀 곳과 이유
#   입·턱(0,17,152,...)  말하기·씹기·입벌림 판정 동작으로 크게 움직인다.
#                        이 프로젝트는 입 벌림을 클릭에 쓰므로 특히 위험하다.
#   눈꺼풀(159,145,...)  깜빡임마다 움직인다. 커서가 깜빡임에 흔들리던
#                        문제(_BlinkGate 참고)를 다시 불러들이게 된다.
#   눈썹(70,105,...)     이 프로젝트는 미간 찌푸림을 신호로 쓴다 — 정의상 움직인다.
RIGID_LANDMARKS = (
    6, 197, 195, 5, 4, 168, 8,
    33, 133, 263, 362,
    234, 454,
    10, 151, 9, 107, 336,
    117, 346, 50, 280,
)

# 정합에 쓸 최소 점 개수. 3점이면 수학적으로는 회전이 정해지지만 잡음에
# 무방비다. 넉넉히 요구한다 (RIGID_LANDMARKS는 22개다)
MIN_POINTS = 8

# 중립 대비 회전이 이보다 크면 탄젠트가 발산한다. 키오스크에서 고개를
# 이만큼 돌릴 일은 없고, 한 프레임이라도 튀면 커서가 화면 밖으로 날아간다
MAX_ANGLE_DEG = 60.0

# 부호 투표에 쓸 최소 오프셋(탄젠트 단위, 약 1.7도) — 이보다 작은 움직임은
# 잡음이 부호를 지배해서 증거로 안 친다
SIGN_EVIDENCE_MIN = 0.03

# 이만큼 투표가 한쪽으로 쌓이면 그 축의 부호를 확정한다. 8이면 고개를
# 그 축으로 한 번만 크게 왕복해도 잠긴다(30fps에서 1초 미만)
LOCK_VOTES = 8

# 얼굴이 이보다 작게 잡히면 z 좌표의 상대 오차가 커져 회전이 불안정하다.
# head_tracker.MIN_INTEROCULAR_DIST_PX와 같은 취지
MIN_SCALE = 1e-6


def _orthonormal_frame(points):
    """중립 랜드마크에서 얼굴 자신의 좌표축을 만든다 -> (x축, y축, z축).

    x축 = 왼눈->오른눈 (이미지에서 오른쪽)
    y축 = 이마->코밑   (이미지에서 아래쪽)
    z축 = x cross y    (오른손 좌표계를 완성 — 얼굴이 향하는 쪽의 반대/정방향은
                        아래 설명 참고)

    **부호를 왜 잴 필요가 없는가**: 이 축들은 랜드마크에서 곧바로 나오고,
    랜드마크는 이미 화면 좌표계(오른쪽 +x, 아래쪽 +y)로 들어온다. 거울 반전도
    랜드마크를 뽑기 전에 이미 적용돼 있다. 그래서 "x축이 화면 오른쪽"이
    측정 결과가 아니라 정의로 성립한다.

    y축은 그람-슈미트로 x축 성분을 빼서 직교화한다 — 이마-코밑 벡터가 x축과
    정확히 수직이라는 보장이 없기 때문이다(사람마다 다르고 고개를 갸웃하면 더).
    """
    idx = {v: i for i, v in enumerate(RIGID_LANDMARKS)}
    x_axis = points[idx[263]] - points[idx[33]]        # 왼눈 바깥 -> 오른눈 바깥
    down = points[idx[4]] - points[idx[10]]            # 이마 위 -> 코밑
    nx = np.linalg.norm(x_axis)
    if nx < MIN_SCALE:
        return None
    x_axis = x_axis / nx
    # 그람-슈미트 — down에서 x 성분을 빼고 남은 것이 진짜 아래쪽
    down = down - np.dot(down, x_axis) * x_axis
    nd = np.linalg.norm(down)
    if nd < MIN_SCALE:
        return None
    y_axis = down / nd
    # ★z축은 **얼굴이 향하는 쪽**(카메라 쪽)으로 잡는다.
    #
    # cross(x, y)로 잡으면 화면 안쪽(뒤통수 방향)이 나온다 — x가 화면 오른쪽,
    # y가 화면 아래쪽인 오른손 좌표계에서 cross(x, y)는 +z이고, MediaPipe는
    # z가 작을수록(음수) 카메라에 가깝다고 정의하므로 +z는 카메라 반대편이다.
    # 그대로 쓰면 고개를 오른쪽으로 돌렸을 때 뒤통수는 왼쪽을 향하므로 커서가
    # 반대로 간다. cross(y, x)로 뒤집어 얼굴 정면을 잡는다.
    # (2026-08-31: 처음에 cross(x, y)로 썼다가 단위 테스트가 잡아냈다 —
    #  tests/test_head_orientation.py의 부호 검증 두 개)
    z_axis = np.cross(y_axis, x_axis)
    nz = np.linalg.norm(z_axis)
    if nz < MIN_SCALE:
        return None
    return x_axis, y_axis, z_axis / nz


def estimate_rotation(neutral, current):
    """중립 -> 현재 의 최적 회전 행렬 R (3x3). 못 구하면 None.

    Kabsch-Umeyama (위 독스트링의 논문 참고). R @ neutral ~= current 가 되는
    R을 최소제곱으로 구한다.

    두 점집합을 각자의 무게중심으로 옮기고 **각자의 크기로 나눈다**. 크기를
    맞추는 이유: 사용자가 카메라에 다가가거나 물러나면 얼굴이 통째로 커지고
    작아지는데, 그건 회전이 아니다. 정규화하면 거리 변화가 회전에 안 섞인다.
    """
    if neutral is None or current is None:
        return None
    if len(neutral) < MIN_POINTS or len(neutral) != len(current):
        return None

    p = np.asarray(neutral, dtype=np.float64)
    q = np.asarray(current, dtype=np.float64)
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    # 크기 정규화 — 거리 변화가 회전으로 새지 않도록
    sp = math.sqrt(float((p ** 2).sum()))
    sq = math.sqrt(float((q ** 2).sum()))
    if sp < MIN_SCALE or sq < MIN_SCALE:
        return None
    p /= sp
    q /= sq

    try:
        u, _s, vt = np.linalg.svd(p.T @ q)
    except np.linalg.LinAlgError:
        return None      # 특이값분해가 수렴하지 않는 병적인 입력 — 이 프레임은 버린다
    # ★반사 방지 (Umeyama 1991). det가 음수면 거울상 해가 나온다 —
    # 잡음이 크거나 점들이 거의 한 평면에 있을 때 실제로 발생하고,
    # 그대로 두면 커서가 반대편으로 순간이동한다
    d = 1.0 if np.linalg.det(vt.T @ u.T) > 0 else -1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


class HeadOrientation:
    """중립 자세를 잡아 두고, 매 프레임 '중립 대비 어디를 향하는가'를 낸다.

    쓰는 법
        ho = HeadOrientation()
        ho.set_neutral(face)                 # 시작 캘리브레이션 때 한 번
        offset = ho.pointing_offset(face)    # 매 프레임 -> (가로, 세로) 또는 None

    돌려주는 값은 **탄젠트**다. 고개를 각도 θ 돌렸을 때 사용자가 보는 화면 위의
    지점은 tan(θ)에 비례해 움직인다(눈에서 화면까지가 직선 거리라서). θ를 그대로
    쓰면 화면 가장자리로 갈수록 커서가 실제 시선보다 뒤처진다.

    카메라를 어떻게 달아도 되는 이유는 이 클래스가 **중립과 현재를 같은 카메라로
    찍은 것끼리만 비교**하기 때문이다. 카메라의 위치·기울기는 두 쪽에 똑같이
    들어 있으므로 상대 회전에서 사라진다.
    """

    def __init__(self, rotation_source="auto"):
        """rotation_source — "auto"(행렬 우선, 없으면 랜드마크) | "matrix" | "landmarks".

        기본이 auto인 이유는 위 독스트링 "회전의 재료 두 가지" 참고 — 행렬이
        실측으로 더 곧고, 안 오는 프레임에서도 커서가 죽지 않아야 한다.
        """
        self._rotation_source = rotation_source
        # 행렬 부호 자가 학습 상태 (위 "반사 켤레" 설명 참고). None = 미확정.
        # reset()에서도 지우지 않는다 — 부호는 카메라·거울 설정의 성질이라
        # 사용자가 바뀌어도 그대로다 (다시 배우게 하면 그동안 랜드마크 경로로
        # 돌 뿐 틀리지는 않지만, 유지하는 쪽이 그 시간을 아낀다)
        self._sign_h = None
        self._sign_v = None
        self._vote_h = 0
        self._vote_v = 0
        self._neutral_points = None
        self._neutral_rotation = None      # 중립의 변환행렬 회전 (3,3) — 코달 평균
        self._axes = None
        self._samples = []
        self._rotation_samples = []

    def reset(self):
        self._neutral_points = None
        self._neutral_rotation = None
        self._axes = None
        self._samples = []
        self._rotation_samples = []

    def add_calibration_sample(self, face):
        """중립을 잡기 위해 표본을 모은다 (시작 캘리브레이션 구간에서 매 프레임).

        한 프레임만으로 중립을 잡으면 그 프레임의 랜드마크 떨림이 중립에 그대로
        박히고, 이후 모든 커서 위치가 그만큼 치우친다. 여러 장 모아 두었다가
        좌표별 **중앙값**으로 확정한다 — 이 프로젝트가 커서 중심 캘리브레이션에
        이미 쓰는 방식과 같다(_MedianCalibrator). 평균이 아니라 중앙값인 이유도
        같다: 검출이 한 번 크게 튄 프레임이 섞여도 결과가 끌려가지 않는다.
        """
        points = extract_rigid_points(face)
        if points is None:
            return False
        self._samples.append(points)
        rot = getattr(face, "head_rotation", None)
        if rot is not None:
            self._rotation_samples.append(np.asarray(rot, dtype=np.float64))
        return True

    def finalize_neutral(self):
        """모아 둔 표본의 좌표별 중앙값을 중립으로 확정한다. 성공하면 True."""
        if len(self._samples) < 1:
            return False
        median = np.median(np.stack(self._samples, axis=0), axis=0)
        axes = _orthonormal_frame(median)
        if axes is None:
            return False
        self._neutral_points = median
        self._axes = axes
        # 중립 회전 행렬 — 표본 절반 이상에서 행렬이 왔을 때만 확정한다.
        # 평균은 코달 평균(성분 평균 -> SVD로 회전에 사영, 위 독스트링의
        # Hartley et al. 2013): 회전들의 "중간"으로 수렴하고 반사가 안 생긴다
        if len(self._rotation_samples) * 2 >= len(self._samples):
            self._neutral_rotation = _chordal_mean(self._rotation_samples)
        else:
            self._neutral_rotation = None
        self._samples = []
        self._rotation_samples = []
        return True

    @property
    def sample_count(self):
        return len(self._samples)

    @property
    def is_ready(self):
        return self._neutral_points is not None and self._axes is not None

    def set_neutral(self, face):
        """지금 자세를 중립으로 삼는다. 성공하면 True."""
        points = extract_rigid_points(face)
        if points is None:
            return False
        axes = _orthonormal_frame(points)
        if axes is None:
            return False
        self._neutral_points = points
        self._axes = axes
        rot = getattr(face, "head_rotation", None)
        self._neutral_rotation = (np.asarray(rot, dtype=np.float64)
                                  if rot is not None else None)
        return True

    def pointing_offset(self, face):
        """중립 대비 머리가 향하는 방향 -> (tan 가로, tan 세로). 못 구하면 None.

        중립일 때 (0, 0)이고, 고개를 오른쪽으로 돌리면 가로가 커지고(화면
        오른쪽), 아래로 숙이면 세로가 커진다(화면 아래쪽) — 화면 좌표와 같은
        방향이다. 이 대응이 랜드마크 좌표계에서 곧바로 나오므로 부호를 따로
        정할 일이 없다.
        """
        if not self.is_ready:
            return None
        # 1순위: 변환행렬 상대 회전 (독스트링 "회전의 재료 두 가지" 참고)
        mat_rot = None
        if self._rotation_source in ("auto", "matrix") and self._neutral_rotation is not None:
            cur = getattr(face, "head_rotation", None)
            if cur is not None:
                # R_rel = R_now @ R_neutral^T — 회전이라 전치 = 역행렬.
                # 카메라를 M만큼 돌려 달면 양쪽에 M이 곱해져 여기서 소거된다
                mat_rot = np.asarray(cur, dtype=np.float64) @ self._neutral_rotation.T
        if mat_rot is None and self._rotation_source == "matrix":
            return None      # 행렬 강제인데 이 프레임엔 행렬이 없다

        # 랜드마크 정합 — 행렬이 없거나, 행렬의 부호가 아직 미확정일 때 필요
        # (부호가 다 잠기면 이 계산은 건너뛰어 프레임 비용을 아낀다)
        landmark = None
        signs_locked = self._sign_h is not None and self._sign_v is not None
        if mat_rot is None or not signs_locked:
            points = extract_rigid_points(face)
            if points is not None:
                lrot = estimate_rotation(self._neutral_points, points)
                if lrot is not None:
                    landmark = self._project(lrot)

        if mat_rot is None:
            return landmark                       # 폴백 (기존 동작)
        matrix = self._project(mat_rot)
        if matrix is None:
            return landmark                       # 행렬상 과회전 — 랜드마크로

        # ★부호 자가 학습 (독스트링 "반사 켤레" 참고) — 두 값이 함께 있고
        # 움직임이 충분할 때만 축마다 투표한다
        if landmark is not None:
            self._learn_signs(matrix, landmark)
            signs_locked = self._sign_h is not None and self._sign_v is not None

        if not signs_locked:
            # 아직 미확정 — 방향이 정의상 옳은 랜드마크 값으로 움직인다.
            # 커서가 첫 프레임부터 옳은 방향인 이유가 이 줄이다
            return landmark
        return (self._sign_h * matrix[0], self._sign_v * matrix[1])

    def _project(self, rot):
        """상대 회전 -> (가로, 세로) 탄젠트. 과회전이면 None."""
        x_axis, y_axis, z_axis = self._axes
        # 중립일 때 얼굴이 향하던 방향을, 지금 회전만큼 돌린 것
        facing = rot @ z_axis
        forward = float(np.dot(facing, z_axis))
        if abs(forward) < math.cos(math.radians(MAX_ANGLE_DEG)):
            return None      # 너무 많이 돌아 탄젠트가 발산하는 영역 — 이 프레임은 버린다
        horizontal = float(np.dot(facing, x_axis))
        vertical = float(np.dot(facing, y_axis))
        # forward의 부호는 z축을 어느 쪽으로 잡았느냐에 따라 갈리는데, 나누는
        # 쪽과 곱해지는 쪽이 같은 축에서 나오므로 비율에서 상쇄된다.
        # abs를 쓰면 뒤통수를 보이는 각도에서도 부호가 안 뒤집힌다
        return (horizontal / abs(forward), vertical / abs(forward))

    def _learn_signs(self, matrix, landmark):
        """축마다 행렬·랜드마크의 부호 일치를 투표 -> 쌓이면 확정."""
        if self._sign_h is None and (abs(matrix[0]) >= SIGN_EVIDENCE_MIN
                                     and abs(landmark[0]) >= SIGN_EVIDENCE_MIN):
            self._vote_h += 1 if matrix[0] * landmark[0] > 0 else -1
            if abs(self._vote_h) >= LOCK_VOTES:
                self._sign_h = 1.0 if self._vote_h > 0 else -1.0
        if self._sign_v is None and (abs(matrix[1]) >= SIGN_EVIDENCE_MIN
                                     and abs(landmark[1]) >= SIGN_EVIDENCE_MIN):
            self._vote_v += 1 if matrix[1] * landmark[1] > 0 else -1
            if abs(self._vote_v) >= LOCK_VOTES:
                self._sign_v = 1.0 if self._vote_v > 0 else -1.0


def _chordal_mean(rotations):
    """회전행렬들의 코달 평균 — 성분 평균을 SVD로 가장 가까운 회전에 사영.

    Hartley et al. (2013) IJCV 103의 L2 chordal mean. 표본 몇 개가 튀어도
    반사 없는 순수 회전으로 수렴한다.
    """
    mean = np.mean(np.stack(rotations, axis=0), axis=0)
    u, _s, vt = np.linalg.svd(mean)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] = -u[:, -1]
        rot = u @ vt
    return rot


def extract_rigid_points(face):
    """얼굴에서 강체 랜드마크만 (N, 3)으로 뽑는다. 3차원 좌표가 없으면 None."""
    points = getattr(face, "landmarks_3d", None)
    if points is None:
        return None
    try:
        picked = np.asarray(points, dtype=np.float64)[list(RIGID_LANDMARKS)]
    except (IndexError, ValueError, TypeError):
        return None
    if picked.shape != (len(RIGID_LANDMARKS), 3):
        return None
    if not np.all(np.isfinite(picked)):
        return None
    return picked
