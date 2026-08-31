"""head_orientation — 중립 대비 상대 회전 추정 검증 (2026-08-31 신설).

카메라 없이 전부 검증한다. 합성 점구름을 **정확히 아는 각도만큼** 돌려 넣고
그 각도가 되돌아 나오는지 보면 되므로, 실기 측정이 필요 없다.

가장 중요한 것은 test_camera_pose_invariance 다 — 카메라를 어떻게 달아도
결과가 같다는 이 설계의 핵심 주장을 그대로 시험한다.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_orientation import (   # noqa: E402
    _orthonormal_frame,
    MIN_POINTS, RIGID_LANDMARKS, HeadOrientation, estimate_rotation,
    extract_rigid_points,
)

LANDMARK_COUNT = 478


class _FakeFace:
    """landmarks_3d만 있으면 되는 최소 얼굴 객체."""

    def __init__(self, points_3d):
        self.landmarks_3d = points_3d


def _synthetic_head():
    """사람 머리를 닮은 합성 랜드마크 (478, 3).

    좌표계는 화면과 같다 — x 오른쪽, y 아래쪽, z는 카메라에 가까울수록 작다
    (MediaPipe 규약). 강체 인덱스만 실제 얼굴 비율에 맞춰 놓고 나머지는
    0으로 둔다(이 모듈은 강체 인덱스만 읽는다).
    """
    pts = np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)
    # (인덱스, x, y, z) — 단위는 px 어림. 코가 가장 앞(z 음수), 관자놀이가 뒤
    layout = [
        (33, -32.0, 0.0, 4.0), (263, 32.0, 0.0, 4.0),        # 눈 바깥 구석
        (133, -12.0, 1.0, 0.0), (362, 12.0, 1.0, 0.0),        # 눈 안쪽 구석
        (168, 0.0, -6.0, -6.0), (6, 0.0, -1.0, -10.0),        # 콧대 위
        (197, 0.0, 3.0, -13.0), (195, 0.0, 8.0, -17.0),
        (5, 0.0, 13.0, -21.0), (4, 0.0, 18.0, -25.0),         # 코밑(가장 앞)
        (8, 0.0, -11.0, -3.0),
        (234, -46.0, 2.0, 22.0), (454, 46.0, 2.0, 22.0),      # 관자놀이(가장 뒤)
        (10, 0.0, -58.0, 2.0), (151, 0.0, -46.0, -2.0),       # 이마
        (9, 0.0, -34.0, -5.0),
        (107, -14.0, -30.0, -4.0), (336, 14.0, -30.0, -4.0),
        (117, -34.0, 20.0, 8.0), (346, 34.0, 20.0, 8.0),      # 광대
        (50, -38.0, 12.0, 10.0), (280, 38.0, 12.0, 10.0),
    ]
    for idx, x, y, z in layout:
        pts[idx] = (x, y, z)
    return pts


def _rot_about(axis, degrees):
    """축(단위벡터) 둘레로 각도만큼 도는 회전행렬 — 로드리게스 공식."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = math.radians(degrees)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


def _apply(rot, pts):
    """강체 점만 회전시킨다(나머지 0은 그대로 둬도 이 모듈이 안 읽는다)."""
    out = pts.copy()
    idx = list(RIGID_LANDMARKS)
    out[idx] = pts[idx] @ rot.T
    return out


# --------------------------------------------------------------- 회전 추정 자체

def test_recovers_known_rotation():
    """정확히 아는 회전을 넣으면 그대로 돌아와야 한다."""
    neutral = _synthetic_head()
    rot_true = _rot_about((0.0, 1.0, 0.0), 17.0)
    current = _apply(rot_true, neutral)

    got = estimate_rotation(neutral[list(RIGID_LANDMARKS)],
                            current[list(RIGID_LANDMARKS)])
    assert got is not None
    assert np.allclose(got, rot_true, atol=1e-6)


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                                  (1.0, 1.0, 0.0), (0.3, -0.7, 0.5)])
@pytest.mark.parametrize("degrees", [-40.0, -12.0, -3.0, 3.0, 12.0, 40.0])
def test_recovers_rotation_on_many_axes(axis, degrees):
    neutral = _synthetic_head()
    rot_true = _rot_about(axis, degrees)
    got = estimate_rotation(neutral[list(RIGID_LANDMARKS)],
                            _apply(rot_true, neutral)[list(RIGID_LANDMARKS)])
    assert got is not None
    assert np.allclose(got, rot_true, atol=1e-6)


def test_result_is_a_proper_rotation_not_a_reflection():
    """행렬식이 항상 +1 — 거울상 해(Umeyama 1991의 반사 문제)가 나오면 안 된다."""
    neutral = _synthetic_head()
    rng = np.random.default_rng(7)
    for _ in range(30):
        rot_true = _rot_about(rng.normal(size=3), rng.uniform(-45, 45))
        noisy = _apply(rot_true, neutral)
        noisy[list(RIGID_LANDMARKS)] += rng.normal(0.0, 3.0, (len(RIGID_LANDMARKS), 3))
        got = estimate_rotation(neutral[list(RIGID_LANDMARKS)],
                                noisy[list(RIGID_LANDMARKS)])
        assert got is not None
        assert np.linalg.det(got) == pytest.approx(1.0, abs=1e-6)


def test_ignores_translation_and_scale():
    """사용자가 옆으로 서거나 카메라에 다가가도 회전은 그대로여야 한다."""
    neutral = _synthetic_head()
    rot_true = _rot_about((0.0, 1.0, 0.0), 21.0)
    moved = _apply(rot_true, neutral)
    idx = list(RIGID_LANDMARKS)
    moved[idx] = moved[idx] * 1.7 + np.array([120.0, -45.0, 33.0])   # 확대 + 평행이동

    got = estimate_rotation(neutral[idx], moved[idx])
    assert got is not None
    assert np.allclose(got, rot_true, atol=1e-6)


def test_noise_is_averaged_down_by_using_many_points():
    """점을 많이 쓰면 잡음이 평균돼 회전 오차가 작아진다 (이 설계의 근거).

    같은 잡음을 주고 강체 점 전부를 쓸 때와 최소 개수만 쓸 때를 비교한다.
    """
    neutral = _synthetic_head()
    rot_true = _rot_about((0.0, 1.0, 0.0), 15.0)
    rng = np.random.default_rng(11)
    idx_all = list(RIGID_LANDMARKS)
    idx_few = idx_all[:MIN_POINTS]

    def angle_error(indices, seed):
        gen = np.random.default_rng(seed)
        cur = _apply(rot_true, neutral)
        cur[idx_all] += gen.normal(0.0, 2.0, (len(idx_all), 3))
        got = estimate_rotation(neutral[indices], cur[indices])
        if got is None:
            return float("inf")
        # 두 회전 사이의 각도 = trace 로부터
        cos_t = (np.trace(got.T @ rot_true) - 1.0) / 2.0
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))

    errs_all = [angle_error(idx_all, s) for s in range(40)]
    errs_few = [angle_error(idx_few, s) for s in range(40)]
    assert np.mean(errs_all) < np.mean(errs_few)


# --------------------------------------------------- 카메라 배치 무관성 (핵심)

def test_camera_pose_invariance():
    """★이 설계의 핵심 주장 — 카메라를 어떤 각도로 달아도 결과가 같다.

    카메라를 옮겨 다는 것은, 보이는 모든 점에 **같은 회전을 한 번 더 거는**
    것과 같다. 중립과 현재 양쪽에 똑같이 걸리므로 상대 회전에서 소거돼야 한다.

    밑에서 올려보는 배치, 위에서 내려보는 배치, 옆으로 기울어진 배치를
    모두 시험한다 — 실기에서 실제로 겪은 배치들이다(키오스크는 정면,
    연구실은 밑에서 위로).
    """
    neutral = _synthetic_head()
    head_turn = _rot_about((0.0, 1.0, 0.0), 13.0)
    current = _apply(head_turn, neutral)

    base = HeadOrientation()
    assert base.set_neutral(_FakeFace(neutral))
    expected = base.pointing_offset(_FakeFace(current))
    assert expected is not None

    camera_mounts = [
        ("밑에서 올려봄", _rot_about((1.0, 0.0, 0.0), 25.0)),
        ("위에서 내려봄", _rot_about((1.0, 0.0, 0.0), -30.0)),
        ("옆으로 기울어짐", _rot_about((0.0, 0.0, 1.0), 18.0)),
        ("비스듬히", _rot_about((0.6, -0.5, 0.3), 22.0)),
    ]
    for label, mount in camera_mounts:
        ho = HeadOrientation()
        assert ho.set_neutral(_FakeFace(_apply(mount, neutral))), label
        got = ho.pointing_offset(_FakeFace(_apply(mount, current)))
        assert got is not None, label
        assert got[0] == pytest.approx(expected[0], abs=1e-6), label
        assert got[1] == pytest.approx(expected[1], abs=1e-6), label


def test_neutral_pose_maps_to_zero():
    """중립 그대로면 커서는 정확히 중앙(오프셋 0)이어야 한다."""
    neutral = _synthetic_head()
    ho = HeadOrientation()
    assert ho.set_neutral(_FakeFace(neutral))
    got = ho.pointing_offset(_FakeFace(neutral))
    assert got is not None
    assert got[0] == pytest.approx(0.0, abs=1e-9)
    assert got[1] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------- 부호 방향

def _nose_screen_x(points):
    return points[4][0]


def _nose_screen_y(points):
    return points[4][1]


def test_turning_head_toward_screen_right_moves_cursor_right():
    """부호 검증 — 기준을 '코가 화면에서 어느 쪽으로 갔나'로 잡는다.

    이 방식이면 오일러 각의 부호 규약을 알 필요가 없다. 코가 화면 오른쪽으로
    가는 회전을 만들고, 그때 커서 가로 오프셋이 양수인지만 보면 된다.
    """
    neutral = _synthetic_head()
    for degrees in (10.0, 25.0):
        for direction in (1.0, -1.0):
            rot = _rot_about((0.0, 1.0, 0.0), degrees * direction)
            current = _apply(rot, neutral)
            ho = HeadOrientation()
            assert ho.set_neutral(_FakeFace(neutral))
            got = ho.pointing_offset(_FakeFace(current))
            assert got is not None
            moved_right = _nose_screen_x(current) > _nose_screen_x(neutral)
            assert (got[0] > 0.0) == moved_right, (degrees, direction)


def test_lowering_head_moves_cursor_down():
    """고개를 숙여 코가 화면 아래로 가면 커서도 아래(세로 오프셋 양수)."""
    neutral = _synthetic_head()
    for degrees in (8.0, 20.0):
        for direction in (1.0, -1.0):
            rot = _rot_about((1.0, 0.0, 0.0), degrees * direction)
            current = _apply(rot, neutral)
            ho = HeadOrientation()
            assert ho.set_neutral(_FakeFace(neutral))
            got = ho.pointing_offset(_FakeFace(current))
            assert got is not None
            moved_down = _nose_screen_y(current) > _nose_screen_y(neutral)
            assert (got[1] > 0.0) == moved_down, (degrees, direction)


def test_offset_grows_with_angle():
    """더 많이 돌릴수록 커서가 더 멀리 가야 한다 (단조성)."""
    neutral = _synthetic_head()
    ho = HeadOrientation()
    assert ho.set_neutral(_FakeFace(neutral))
    previous = 0.0
    for degrees in (5.0, 10.0, 20.0, 30.0):
        rot = _rot_about((0.0, 1.0, 0.0), degrees)
        got = ho.pointing_offset(_FakeFace(_apply(rot, neutral)))
        assert got is not None
        assert abs(got[0]) > previous
        previous = abs(got[0])


def test_offset_matches_tangent_of_angle():
    """오프셋이 각도의 탄젠트와 일치해야 한다 — 화면 위 지점은 tan에 비례한다.

    돌리는 축은 **머리 자신의 세로축**이어야 한다. 전역 y축으로 돌리면
    머리의 세로축과 조금 어긋나(합성 얼굴의 이마-코밑 벡터가 z 성분을 가진다)
    그만큼 탄젠트가 안 맞는다 — 이 테스트를 처음 그렇게 썼다가 5도에서
    0.0824 vs 0.0875로 어긋나 알아챘다.
    """
    neutral = _synthetic_head()
    axes = _orthonormal_frame(neutral[list(RIGID_LANDMARKS)])
    assert axes is not None
    x_axis, y_axis, _z_axis = axes

    ho = HeadOrientation()
    assert ho.set_neutral(_FakeFace(neutral))
    for degrees in (5.0, 12.0, 25.0, 35.0):
        # 머리 세로축 둘레 회전 = 순수한 좌우 돌리기
        rot = _rot_about(y_axis, degrees)
        got = ho.pointing_offset(_FakeFace(_apply(rot, neutral)))
        assert got is not None
        assert abs(got[0]) == pytest.approx(math.tan(math.radians(degrees)), rel=1e-6)
        assert got[1] == pytest.approx(0.0, abs=1e-9)   # 세로는 안 움직여야 한다

    for degrees in (5.0, 12.0, 25.0):
        # 머리 가로축 둘레 회전 = 순수한 위아래 끄덕임
        rot = _rot_about(x_axis, degrees)
        got = ho.pointing_offset(_FakeFace(_apply(rot, neutral)))
        assert got is not None
        assert abs(got[1]) == pytest.approx(math.tan(math.radians(degrees)), rel=1e-6)
        assert got[0] == pytest.approx(0.0, abs=1e-9)   # 가로는 안 움직여야 한다


# ----------------------------------------------------------------- 방어적 처리

def test_missing_3d_landmarks_returns_none():
    class NoDepth:
        landmarks_3d = None

    assert extract_rigid_points(NoDepth()) is None
    ho = HeadOrientation()
    assert ho.set_neutral(NoDepth()) is False
    assert ho.pointing_offset(NoDepth()) is None


def test_offset_before_neutral_is_none():
    ho = HeadOrientation()
    assert ho.is_ready is False
    assert ho.pointing_offset(_FakeFace(_synthetic_head())) is None


def test_non_finite_landmarks_rejected():
    pts = _synthetic_head()
    pts[RIGID_LANDMARKS[0]] = (float("nan"), 0.0, 0.0)
    assert extract_rigid_points(_FakeFace(pts)) is None


def test_extreme_rotation_is_rejected():
    """탄젠트가 발산하는 각도는 커서를 화면 밖으로 날리므로 버려야 한다."""
    neutral = _synthetic_head()
    ho = HeadOrientation()
    assert ho.set_neutral(_FakeFace(neutral))
    rot = _rot_about((0.0, 1.0, 0.0), 85.0)
    assert ho.pointing_offset(_FakeFace(_apply(rot, neutral))) is None


def test_reset_clears_neutral():
    ho = HeadOrientation()
    assert ho.set_neutral(_FakeFace(_synthetic_head()))
    ho.reset()
    assert ho.is_ready is False


def test_degenerate_point_cloud_returns_none():
    """모든 점이 한 자리에 겹치면 회전을 정할 수 없다."""
    flat = np.zeros((len(RIGID_LANDMARKS), 3), dtype=np.float64)
    assert estimate_rotation(flat, flat) is None


def test_mismatched_point_counts_returns_none():
    neutral = _synthetic_head()[list(RIGID_LANDMARKS)]
    assert estimate_rotation(neutral, neutral[:-1]) is None


# --------------------------------------------------- 중립을 여러 장으로 확정하기

def test_median_neutral_is_robust_to_a_spiking_frame():
    """검출이 한 번 크게 튄 프레임이 섞여도 중립이 끌려가면 안 된다.

    중앙값을 쓰는 이유가 이것이다 — 평균이면 튄 프레임이 그대로 반영된다.
    """
    neutral = _synthetic_head()
    ho = HeadOrientation()
    for _ in range(20):
        assert ho.add_calibration_sample(_FakeFace(neutral))
    spike = neutral.copy()
    spike[list(RIGID_LANDMARKS)] += 250.0        # 검출이 통째로 튄 프레임
    assert ho.add_calibration_sample(_FakeFace(spike))
    assert ho.finalize_neutral()

    got = ho.pointing_offset(_FakeFace(neutral))
    assert got is not None
    assert got[0] == pytest.approx(0.0, abs=1e-6)
    assert got[1] == pytest.approx(0.0, abs=1e-6)


def test_median_neutral_averages_down_noise():
    """표본을 많이 모을수록 중립이 참값에 가까워진다."""
    neutral = _synthetic_head()
    rng = np.random.default_rng(3)

    def offset_error(sample_count):
        ho = HeadOrientation()
        for _ in range(sample_count):
            noisy = neutral.copy()
            noisy[list(RIGID_LANDMARKS)] += rng.normal(0.0, 2.5,
                                                       (len(RIGID_LANDMARKS), 3))
            ho.add_calibration_sample(_FakeFace(noisy))
        assert ho.finalize_neutral()
        got = ho.pointing_offset(_FakeFace(neutral))
        return math.hypot(got[0], got[1])

    few = statistics_mean([offset_error(2) for _ in range(15)])
    many = statistics_mean([offset_error(40) for _ in range(15)])
    assert many < few


def statistics_mean(values):
    return sum(values) / len(values)


def test_calibration_samples_cleared_after_finalize():
    ho = HeadOrientation()
    ho.add_calibration_sample(_FakeFace(_synthetic_head()))
    assert ho.sample_count == 1
    assert ho.finalize_neutral()
    assert ho.sample_count == 0
    assert ho.is_ready


def test_finalize_without_samples_fails():
    ho = HeadOrientation()
    assert ho.finalize_neutral() is False
    assert ho.is_ready is False


def test_calibration_sample_rejects_face_without_depth():
    class NoDepth:
        landmarks_3d = None

    ho = HeadOrientation()
    assert ho.add_calibration_sample(NoDepth()) is False
    assert ho.sample_count == 0
