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

    def __init__(self):
        self._neutral_points = None
        self._axes = None
        self._samples = []

    def reset(self):
        self._neutral_points = None
        self._axes = None
        self._samples = []

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
        self._samples = []
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
        points = extract_rigid_points(face)
        if points is None:
            return None
        rot = estimate_rotation(self._neutral_points, points)
        if rot is None:
            return None

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
