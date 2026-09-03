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
문제는 "무엇을 진실로 삼느냐"였다.

처음에는 랜드마크 정합의 방향을 진실로 삼았다. 그런데 그것이 틀렸다 —
얼굴 축(x축 = 왼눈->오른눈)은 **거울 반전 여부에 따라 화면에서 가리키는
쪽이 뒤집힌다.** 가상 카메라로 거울을 끄고 재 보니 커서 오차가 44.5%로
치솟았다(2026-08-31, tests/virtual_camera.py). 배포 설정은 거울 켬 고정이라
실기에서는 안 드러났지만, 설정 한 줄로 깨지는 구조였다.

진짜 기준은 **중립 얼굴의 축이 화면에서 어느 쪽을 향하는가**다.

처음에는 "기준점(코)이 화면에서 어느 쪽으로 갔는가"를 관측해 투표했는데,
그것도 부족했다. 카메라를 밑에서 50도로 올려보는 배치에서 세로 부호가
확정되지 않았다(2026-09-02, 가상 카메라). 그 각도에서는 고개를 위로 들든
아래로 숙이든 코의 화면 세로 이동이 -1.35 ~ -0.17px로 **전부 같은 방향**이라
증거가 되지 못한다 — 카메라 기울기가 세로 성분을 압축해 버리기 때문이다.

관측으로 부호를 배우는 방법(코의 화면 이동, 눈 중점 대비 코의 위치)도
차례로 시도했지만 둘 다 **카메라를 50도 이상 기울이면 증거가 사라졌다.**
가상 카메라로 잰 "고개 pitch 대 코의 세로 상대이동"이 그것을 보여 준다:

    카메라  0도   -0.099  -0.044  +0.058  +0.088   (단조 — 증거 뚜렷)
    카메라 30도   -0.056  -0.022  +0.033  +0.037   (단조)
    카메라 50도   -0.018  -0.003  +0.014  -0.002   (부호가 섞임)
    카메라 60도   +0.001  +0.007  +0.004  -0.021   (무의미)

관측에 기대는 한 이 구간을 넘을 수 없다. 그래서 **관측을 버리고 대수로
푼다.**

거울 반전은 화면 x좌표만 뒤집는다. 그러면 중립 프레임에서
  · x축(왼눈->오른눈)의 화면 x성분이 뒤집히고,
  · z축 = cross(y축, x축) 도 함께 뒤집힌다.
투영은 horizontal = dot(R·z, x), vertical = dot(R·z, y) 이므로

    가로 — z와 x가 **둘 다** 뒤집혀 서로 상쇄된다  -> 부호 불변
    세로 — z만 뒤집힌다                          -> 부호 반전

즉 **가로 부호는 항상 +1이고, 세로 부호는 거울 여부 하나로 정해진다.**
거울 여부는 x축의 화면 x성분 부호가 그대로 알려준다 — 눈은 좌우로 놓이므로
카메라를 아무리 기울여도 이 부호는 흔들리지 않는다(60도에서도 ±1.000).

    가로 부호 = +1
    세로 부호 = sign(x축의 화면 x성분)

가상 카메라로 배치 9종 × 거울 2종에서 이 규칙이 관측 투표와 일치하는 것을
확인했고, 어긋난 두 건은 모두 증거가 무의미해진 60도 구간이었다(관측 쪽이
틀린 것). 투표를 기다릴 필요가 없어 **중립을 잡는 순간 확정**되므로 커서는
첫 프레임부터 옳은 방향이고, 카메라 각도의 제약도 사라졌다.

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

# 부호 기준점 — 진단·시험에서 "코가 어디로 갔나"를 볼 때 쓴다.
# 부호 결정 자체는 대수로 하므로 여기에 의존하지 않는다(독스트링 참고)
SIGN_REFERENCE_LANDMARK = 4

# 얼굴이 이보다 작게 잡히면 z 좌표의 상대 오차가 커져 회전이 불안정하다.
# head_tracker.MIN_INTEROCULAR_DIST_PX와 같은 취지
MIN_SCALE = 1e-6


def _rotation_vector(rot):
    """회전행렬 -> 회전벡터 (로드리게스 역변환). 각이 0에 가까워도 안정적.

    반환값의 크기가 회전각(라디안), 방향이 회전축이다.
    """
    trace = float(np.trace(rot))
    cos_t = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)                      # 회전이 없다
    sin_t = math.sin(theta)
    if abs(sin_t) < 1e-9:
        return np.zeros(3)                      # 180도 — 키오스크에선 올 일이 없다
    skew = np.array([rot[2, 1] - rot[1, 2],
                     rot[0, 2] - rot[2, 0],
                     rot[1, 0] - rot[0, 1]])
    return skew * (theta / (2.0 * sin_t))


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

    def __init__(self, rotation_source="auto", lens=None):
        """rotation_source — "auto"(랜드마크 우선) | "matrix" | "landmarks".

        ★2026-09-03: auto의 뜻을 **뒤집었다.** 예전에는 변환행렬을 먼저 썼는데,
        가상 카메라에 렌즈 왜곡을 넣어 보니 그 경로가 훨씬 약했다.

        이유는 구조적이다. 변환행렬은 **정규 얼굴 모형을 관측에 맞춘 절대
        자세**라, 렌즈가 상을 휘게 하면 그 편향이 고스란히 들어온다. 랜드마크
        정합은 **같은 사람의 중립 배치와 지금 배치를 맞대는 상대 정합**이라,
        얼굴 근처의 공통된 휨이 양쪽에 똑같이 들어 있어 상쇄된다.

        가상 카메라 측정 (2026-09-03, 얼굴이 화면 옆+위, 가로/세로 오차%):

            렌즈           행렬 경로        랜드마크 경로
            왜곡없음      10.40/12.21%      3.85/ 5.41%
            일반 65도     10.74/ 9.78%      3.83/ 4.80%
            광각 90도     12.44/16.80%      4.95/ 9.28%
            초광각 120도  14.79/33.73%      7.64/15.53%

        배치 9종으로도 재 봤는데 **전 배치에서 랜드마크가 이겼다** (최악은
        "위에서 25도 + 광각"으로 행렬 72.55% 대 랜드마크 23.85%).

        정직한 단서: 이 비교의 행렬은 가상 카메라가 **흉내 낸** 것이다(정규
        모형을 관측에 Kabsch 정합). 실제 MediaPipe는 더 정교할 수 있다.
        다만 "절대 자세는 왜곡을 그대로 받고 상대 정합은 상쇄한다"는 구조는
        흉내와 무관하게 성립한다.

        ★그래서 순서를 **증거에 따라** 정한다. 랜드마크 경로에는 약점이 하나
        있는데, 화면 좌표를 그대로 3차원 점으로 보고 정합하는 탓에 사람이 옆으로
        걸어가면 원근 단축을 회전으로 오해한다(몸 평행이동 끌림 0.023~0.045,
        한도 0.020). 이 약점은 **초점거리 f를 알면 사라진다** — 원근을 되돌리면
        0.004~0.008로 떨어진다 (lens_calibration.py 참고).

        따라서 lens(초점거리와 왜곡을 알아낸 결과)가 있을 때만 랜드마크를
        1순위로 올린다. 없으면 지금까지 검증된 행렬 우선 그대로다 — **아는 것이
        늘기 전에는 바꾸지 않는다.**

        lens — LensModel 또는 None. 나중에 set_lens()로 줘도 된다
        """
        self._rotation_source = rotation_source
        # 행렬 부호 자가 학습 상태 (위 "반사 켤레" 설명 참고). None = 미확정.
        # reset()에서도 지우지 않는다 — 부호는 카메라·거울 설정의 성질이라
        # 사용자가 바뀌어도 그대로다 (다시 배우게 하면 그동안 랜드마크 경로로
        # 돌 뿐 틀리지는 않지만, 유지하는 쪽이 그 시간을 아낀다)
        # 중립을 잡는 순간 대수로 확정된다 (위 "반사 켤레" 설명 참고)
        self._sign_h = 1.0
        self._sign_v = 1.0
        self._lens = lens
        self._neutral_points = None
        self._neutral_raw = None           # 렌즈 보정 **전**의 중립 — 렌즈를 나중에
                                           # 알게 되면 여기서 다시 만든다
        self._neutral_rotation = None      # 중립의 변환행렬 회전 (3,3) — 코달 평균
        self._axes = None
        self._samples = []
        self._rotation_samples = []

    def reset(self):
        self._neutral_points = None
        self._neutral_raw = None
        self._neutral_rotation = None
        self._axes = None
        self._samples = []
        self._rotation_samples = []

    # ------------------------------------------------------------------ 렌즈
    @property
    def lens(self):
        return self._lens

    def set_lens(self, lens):
        """렌즈를 알게 됐다 -> 지금부터 적용하고 **중립도 다시 만든다**.

        중립은 보정 전 좌표로 잡혀 있으므로, 그대로 두면 중립과 현재 프레임이
        서로 다른 좌표계가 되어 커서가 통째로 치우친다. 원본을 들고 있다가
        여기서 다시 만드는 이유다. 성공하면 True.
        """
        self._lens = lens
        if self._neutral_raw is None:
            return True                    # 아직 중립을 안 잡았다 — 잡을 때 반영된다
        points = self._rectify(self._neutral_raw)
        axes = _orthonormal_frame(points)
        if axes is None:
            return False
        self._neutral_points = points
        self._axes = axes
        self._decide_signs(axes)
        return True

    def _rectify(self, points):
        """렌즈를 알면 왜곡과 원근을 되돌린다. 모르면 그대로."""
        if self._lens is None or points is None:
            return points
        try:
            fixed = self._lens.rectify(points)
        except Exception:
            return points                  # 보정이 삐끗해도 커서는 살아야 한다
        if fixed is None or np.shape(fixed) != np.shape(points):
            return points
        if not np.all(np.isfinite(fixed)):
            return points
        return fixed

    def _prefers_landmarks(self):
        """랜드마크를 1순위로 둘 것인가 (위 생성자 독스트링의 근거 참고)."""
        if self._rotation_source == "matrix":
            return False
        if self._rotation_source in ("landmark", "landmarks"):
            return True
        return self._lens is not None      # auto — 렌즈를 알 때만 올린다

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
        rectified = self._rectify(median)
        axes = _orthonormal_frame(rectified)
        if axes is None:
            return False
        self._neutral_raw = median         # 렌즈를 나중에 알면 여기서 다시 만든다
        self._neutral_points = rectified
        self._axes = axes
        self._decide_signs(axes)
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
        rectified = self._rectify(points)
        axes = _orthonormal_frame(rectified)
        if axes is None:
            return False
        self._neutral_raw = points
        self._neutral_points = rectified
        self._axes = axes
        self._decide_signs(axes)
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
        points = self._rectify(extract_rigid_points(face))
        prefer_landmarks = self._prefers_landmarks()

        # ★1순위 — 렌즈를 알면 랜드마크, 모르면 행렬 (생성자 독스트링의 근거 참고)
        raw = None
        if prefer_landmarks and points is not None:
            lrot = estimate_rotation(self._neutral_points, points)
            if lrot is not None:
                raw = self._project(lrot)
        # 2순위: 변환행렬 상대 회전. 랜드마크가 안 나온 프레임에서
        # 커서가 죽지 않게 한다
        if (raw is None and self._neutral_rotation is not None
                and self._rotation_source in ("auto", "matrix")):
            cur = getattr(face, "head_rotation", None)
            if cur is not None:
                # R_rel = R_now @ R_neutral^T — 회전이라 전치 = 역행렬.
                # 카메라를 M만큼 돌려 달면 양쪽에 M이 곱해져 여기서 소거된다
                raw = self._project(np.asarray(cur, dtype=np.float64)
                                    @ self._neutral_rotation.T)
        # 3순위(안전망): 행렬을 먼저 쓰는 설정인데 행렬이 안 온 프레임
        if (raw is None and not prefer_landmarks and points is not None
                and self._rotation_source != "matrix"):
            lrot = estimate_rotation(self._neutral_points, points)
            if lrot is not None:
                raw = self._project(lrot)
        if raw is None:
            return None

        # ★부호 — 두 경로 공통. 중립을 잡을 때 대수로 확정해 둔 값이다
        # (독스트링 "반사 켤레와 부호 자가 학습" 참고)
        return (self._sign_h * raw[0], self._sign_v * raw[1])

    def _decide_signs(self, axes):
        """중립 축에서 부호를 대수로 확정한다 (독스트링 "반사 켤레" 참고).

        가로는 항상 +1 — 거울이 x축과 z축을 함께 뒤집어 투영에서 상쇄된다.
        세로는 z축만 뒤집히므로 거울 여부만큼 반전되고, 거울 여부는 x축의
        화면 x성분 부호가 알려준다.
        """
        x_axis = axes[0]
        self._sign_h = 1.0
        self._sign_v = 1.0 if x_axis[0] >= 0.0 else -1.0

    def _project(self, rot):
        """상대 회전 -> (가로, 세로) 탄젠트. 과회전이면 None.

        ★2026-09-03: 얼굴이 향하는 벡터를 화면에 사영하던 것(그노몬 투영)을
        **회전벡터 분해**로 바꿨다. 세로가 활처럼 휘던 원인이 여기 있었다.

        왜 휘었나 — 얼굴 좌표축은 코가 튀어나온 만큼 기울어져 있다. 이 파일의
        축 정의에서 "아래쪽"은 이마(10)에서 코끝(4)으로 잡는데, 코가 25mm쯤
        앞으로 나와 있어 그 벡터가 얼굴 평면에서 **19.5도 앞으로 기운다.**
        기울어진 축에 대고 향하는 벡터를 사영하면, 고개를 좌우로 돌릴 때
        그 궤적이 원뿔을 그리고 화면에서는 원뿔곡선 — 즉 활 모양이 된다.

        회전벡터로 풀면 이 문제가 원리적으로 사라진다. 상대 회전을 회전벡터
        w로 바꾸면(로드리게스 역변환), 1차 근사에서

            dot(향하는벡터, x축) ~ -dot(w, y축)      (가로 = 세로축 둘레 회전)
            dot(향하는벡터, y축) ~ +dot(w, x축)      (세로 = 가로축 둘레 회전)

        이고, **축이 얼마나 기울었든 축 둘레의 회전량은 그대로 나온다.**
        세로축 둘레로만 돌린 회전은 x축 성분이 정확히 0이므로 세로가 0이다.

        가상 카메라 측정 (2026-09-03, 세로 휨 = 세로 반폭 대비 %):

            렌즈           그노몬 투영    회전벡터
            왜곡없음         3.90%    ->   1.58%
            일반 65도        4.27%    ->   0.92%
            광각 90도        8.20%    ->   4.45%
            초광각 120도    13.80%    ->   9.47%

        부호 규약은 바뀌지 않는다. 거울 반전은 w -> (w_x, -w_y, -w_z)로,
        축은 x축 -> (-x, y, z)로 바뀌는데, 두 경로 모두 가로·세로가 함께
        뒤집혀 그노몬 때와 같은 부호 규칙이 그대로 성립한다(위 "반사 켤레").
        """
        x_axis, y_axis, _z_axis = self._axes
        w = _rotation_vector(np.asarray(rot, dtype=np.float64))
        horizontal = -float(np.dot(w, y_axis))       # 세로축 둘레 회전 = 좌우
        vertical = float(np.dot(w, x_axis))          # 가로축 둘레 회전 = 상하
        limit = math.radians(MAX_ANGLE_DEG)
        if abs(horizontal) > limit or abs(vertical) > limit:
            return None      # 너무 많이 돌았다 — 이 프레임은 버린다
        return (math.tan(horizontal), math.tan(vertical))



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
