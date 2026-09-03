"""렌즈 자가 보정 검증 (2026-09-03 신설).

무엇을 지키는 시험인가
----------------------
광각 카메라의 배럴 왜곡은 커서를 크게 망가뜨린다(초광각에서 세로 오차 32%).
그것을 **현장에서 아무것도 재지 않고** 사용자의 얼굴만으로 되돌리는 것이
src/postprocess/lens_calibration.py다.

자가 보정은 잘못 쓰면 위험하다 — 증거가 부족할 때 나오는 값은 **부호까지
틀리기** 때문이다(움직임 ±60mm에서 k1이 +0.130으로 나왔다. 참값은 -0.30).
그래서 이 파일의 절반은 "언제 채택하나"가 아니라 **"언제 거부하나"** 를
지킨다. 틀린 값을 쓰느니 안 쓰는 쪽이 낫다.
"""
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_orientation import HeadOrientation   # noqa: E402
from src.postprocess.lens_calibration import (                 # noqa: E402
    CROSS_TOL, MAX_ABS_K1, MIN_RADIUS_SPAN_PX, LensModel, LensSelfCalibrator,
)
from tests.virtual_camera import (                             # noqa: E402
    FRAME_H_PX, FRAME_W_PX, LENS_PROFILES, MOUNTS, VirtualCamera, rotation,
)

TAN_HALF_X = math.tan(math.radians(15.0))
TAN_HALF_Y = math.tan(math.radians(10.0))


def _stream(camera, seconds=60.0, fps=30.0, walk_mm=200.0, seed=5):
    """키오스크 앞 사람 흉내 — 서 있는 자리도 바뀌고 고개도 돌린다."""
    rng = np.random.default_rng(seed)
    for i in range(int(seconds * fps)):
        t = i / fps
        offset = (walk_mm * math.sin(t * 0.32) + rng.normal(0.0, 8.0),
                  walk_mm * 0.55 * math.sin(t * 0.21 + 1.0) + rng.normal(0.0, 6.0),
                  90.0 * math.sin(t * 0.14))
        head = (rotation((0.0, 1.0, 0.0), 13.0 * math.sin(t * 1.1))
                @ rotation((1.0, 0.0, 0.0), 8.0 * math.sin(t * 0.8 + 0.5)))
        yield camera.observe(head, offset_mm=offset)


def _run(camera, distortion=False, **kwargs):
    """보정이 끝날 때까지(또는 흐름이 끝날 때까지) 돌린 보정기."""
    cal = LensSelfCalibrator(FRAME_W_PX, FRAME_H_PX, distortion=distortion)
    for face in _stream(camera, **kwargs):
        cal.add(face.landmarks_3d)
        if cal.finished:
            break
    deadline = time.time() + 20.0
    while (cal._thread is not None and cal._thread.is_alive()
           and time.time() < deadline):
        time.sleep(0.02)
    return cal


# ------------------------------------------------------------------- 되찾기

@pytest.mark.parametrize("lens", list(LENS_PROFILES))
def test_recovers_the_focal_length_from_the_users_face(lens):
    """★얼굴만 보고 초점거리를 알아낸다 — 현장 측정 없이.

    초점거리는 **기본으로 채택하는** 값이다. 원근 되돌리기에 쓰이고,
    40% 틀려도 안 고친 것보다 나을 만큼 둔감하다.
    """
    _k1_true, _k2, focal_true = LENS_PROFILES[lens]
    cal = _run(VirtualCamera(lens=lens, seed=3))
    model = cal.model
    assert model is not None, cal.reject_reason
    assert abs(model.focal_px - focal_true) / focal_true < 0.20, model


def test_distortion_is_not_used_unless_asked():
    """★왜곡 되돌리기는 기본으로 쓰지 않는다.

    정규 얼굴이면 세로 휨을 절반으로 줄여 주지만, 얼굴이 다르면 다섯 배로
    악화시킨다(15.10% -> 83.89%). 그 둘을 가려낼 잡음-독립적 지표를 찾지
    못했으므로, 가려낼 수 없으면 안 쓴다.
    """
    cal = _run(VirtualCamera(lens="초광각 120도", seed=3))
    assert cal.model is not None, cal.reject_reason
    assert cal.model.k1 == 0.0
    assert cal.k1_adopted is False


@pytest.mark.parametrize("lens", ["광각 90도", "초광각 120도"])
def test_recovers_distortion_when_explicitly_enabled(lens):
    """켜 달라고 하면 왜곡 계수도 제대로 찾아낸다 (카메라를 아는 배포처용)."""
    k1_true, _k2, _f = LENS_PROFILES[lens]
    cal = _run(VirtualCamera(lens=lens, seed=3), distortion=True)
    assert cal.model is not None, cal.reject_reason
    assert cal.k1_adopted is True
    assert abs(cal.model.k1 - k1_true) < 0.08, (cal.model, k1_true)


def test_undistorted_camera_is_reported_as_undistorted():
    """왜곡 없는 렌즈에 없는 왜곡을 지어내면 안 된다."""
    cal = _run(VirtualCamera(lens="왜곡없음", seed=3), distortion=True)
    assert cal.model is not None, cal.reject_reason
    assert abs(cal.model.k1) < 0.05, cal.model


# ------------------------------------------------------------------- 거부

@pytest.mark.parametrize("lens", ["광각 90도", "초광각 120도"])
def test_refuses_when_the_user_barely_moves(lens):
    """★증거가 없으면 거부한다 — 이때 나오는 값은 부호까지 틀린다."""
    cal = _run(VirtualCamera(lens=lens, seed=3), walk_mm=25.0)
    assert cal.model is None
    assert cal.reject_reason == "움직임 부족"


def test_refuses_before_enough_views():
    """뷰가 몇 장 없을 때 성급히 결론 내지 않는다."""
    cal = LensSelfCalibrator(FRAME_W_PX, FRAME_H_PX)
    camera = VirtualCamera(lens="광각 90도", seed=3)
    for i in range(10):
        cal.add(camera.observe(offset_mm=(30.0 * i, 0.0, 0.0)).landmarks_3d)
    assert cal.model is None


def test_garbage_landmarks_do_not_crash_or_get_adopted():
    """말이 안 되는 입력에도 죽지 않고, 그것으로 결론 내지도 않는다."""
    cal = LensSelfCalibrator(FRAME_W_PX, FRAME_H_PX)
    rng = np.random.default_rng(0)
    for _ in range(200):
        cal.add(rng.normal(200.0, 400.0, (478, 3)))
    cal.add(None)
    cal.add(np.full((478, 3), np.nan))
    cal.add(np.zeros((5, 3)))                       # 점이 모자란다
    deadline = time.time() + 20.0
    while (cal._thread is not None and cal._thread.is_alive()
           and time.time() < deadline):
        time.sleep(0.02)
    if cal.model is not None:                       # 우연히 통과했더라도
        assert abs(cal.model.k1) <= MAX_ABS_K1      # 범위 밖 값은 절대 안 나온다


def test_gives_up_instead_of_burning_cpu_forever():
    """못 믿을 상황이 이어져도 무한정 시도하지 않는다."""
    cal = LensSelfCalibrator(FRAME_W_PX, FRAME_H_PX)
    camera = VirtualCamera(lens="광각 90도", seed=3)
    for face in _stream(camera, seconds=20.0, walk_mm=10.0):
        cal.add(face.landmarks_3d)
    # 움직임이 없으니 비싼 계산은 시작조차 안 했어야 한다
    assert cal._attempts == 0
    assert cal.reject_reason == "움직임 부족"


# ------------------------------------------------------------------- 거울

@pytest.mark.parametrize("mirror", [True, False])
def test_detects_mirroring_from_the_landmarks_alone(mirror):
    """★거울 여부를 설정이 아니라 랜드마크에서 읽는다.

    설정에 기대면 설정 한 줄로 깨진다 — 2026-08-31에 실제로 그랬다.
    """
    cal = _run(VirtualCamera(lens="광각 90도", mirror=mirror, seed=3))
    assert cal.model is not None, cal.reject_reason
    assert cal.mirrored is mirror
    focal_true = LENS_PROFILES["광각 90도"][2]
    assert abs(cal.model.focal_px - focal_true) / focal_true < 0.20, cal.model


# ------------------------------------------------------------------- 쓰임새

def _accuracy(camera, lens_model, offset=(220.0, -140.0, 0.0)):
    """가로 선형성 이탈 / 세로 휨 / 몸 평행이동 끌림."""
    ho = HeadOrientation(lens=lens_model)
    for _ in range(20):
        ho.add_calibration_sample(camera.observe(offset_mm=offset))
    assert ho.finalize_neutral()
    xs, ys, truth = [], [], []
    for deg in np.arange(-14.0, 14.1, 0.5):
        got = ho.pointing_offset(
            camera.observe(rotation((0.0, 1.0, 0.0), float(deg)), offset_mm=offset))
        if got is None:
            continue
        xs.append(got[0])
        ys.append(got[1])
        truth.append(math.tan(math.radians(float(deg))))
    xs, ys, truth = np.array(xs), np.array(ys), np.array(truth)
    slope, intercept = np.polyfit(truth, xs, 1)
    linearity = 100.0 * np.abs(xs - (slope * truth + intercept)).max() / TAN_HALF_X
    bow = 100.0 * np.abs(ys - ys.mean()).max() / TAN_HALF_Y

    still_ho = HeadOrientation(lens=lens_model)
    for _ in range(20):
        still_ho.add_calibration_sample(camera.observe())
    assert still_ho.finalize_neutral()
    still = still_ho.pointing_offset(camera.observe())
    drift = 0.0
    for shift in ((60.0, 0.0, 0.0), (-60.0, 0.0, 0.0),
                  (0.0, 50.0, 0.0), (0.0, 0.0, 120.0)):
        moved = still_ho.pointing_offset(camera.observe(offset_mm=shift))
        if moved is not None:
            drift = max(drift, math.hypot(moved[0] - still[0], moved[1] - still[1]))
    return linearity, bow, drift


@pytest.mark.parametrize("lens", ["광각 90도", "초광각 120도"])
def test_calibration_actually_improves_the_cursor(lens):
    """★알아낸 값을 쓰면 커서가 실제로 좋아진다 (아는 것이 늘어야 의미가 있다)."""
    camera = VirtualCamera(lens=lens, seed=3)
    model = _run(VirtualCamera(lens=lens, seed=3)).model
    assert model is not None
    before = _accuracy(camera, None)
    after = _accuracy(camera, model)
    assert after[0] < before[0], ("가로", before, after)
    assert after[1] < before[1], ("세로", before, after)
    assert after[2] < before[2], ("끌림", before, after)


@pytest.mark.parametrize("mount_name", list(MOUNTS))
def test_body_translation_stays_separated_on_every_mount(mount_name):
    """★어떤 배치에서도 몸이 움직인 것이 커서를 끌고 가면 안 된다.

    이 프로젝트가 8주 내내 지킨 "커서 분리"다. 광각 렌즈에서 보정 없이는
    0.04~0.16까지 벌어졌는데, 보정 뒤에는 한도 0.020 안으로 들어온다.
    """
    camera = VirtualCamera(mount=MOUNTS[mount_name], lens="광각 90도", seed=3)
    model = _run(VirtualCamera(mount=MOUNTS[mount_name], lens="광각 90도",
                               seed=3)).model
    assert model is not None
    assert _accuracy(camera, model)[2] < 0.020


# ------------------------------------------------------------------- 모델 자체

def test_rectify_is_a_no_op_without_distortion_or_depth():
    """왜곡이 0이고 깊이가 0이면 좌표를 건드리지 않는다."""
    model = LensModel(700.0, 0.0, FRAME_W_PX, FRAME_H_PX, mirrored=True)
    pts = np.array([[100.0, 200.0, 0.0], [300.0, 500.0, 0.0]])
    assert np.allclose(model.rectify(pts), pts)


def test_rectify_survives_absurd_input():
    """터무니없는 값이 들어와도 유한한 값을 돌려준다 — 커서가 날아가면 안 된다."""
    model = LensModel(300.0, -0.55, FRAME_W_PX, FRAME_H_PX, mirrored=True)
    pts = np.array([[1e6, -1e6, -1e6], [0.0, 0.0, 1e6], [np.pi, np.e, 0.0]])
    out = model.rectify(pts)
    assert np.all(np.isfinite(out))


def test_rectify_leaves_the_input_untouched():
    """입력 배열을 제자리에서 고치면 부르는 쪽이 모르게 값이 바뀐다."""
    model = LensModel(430.0, -0.15, FRAME_W_PX, FRAME_H_PX, mirrored=True)
    pts = np.array([[100.0, 200.0, 5.0], [300.0, 500.0, -5.0]])
    before = pts.copy()
    model.rectify(pts)
    assert np.array_equal(pts, before)


def test_thresholds_are_ordered_sensibly():
    """상수가 서로 모순되면 게이트가 무의미해진다."""
    assert 0.0 < CROSS_TOL < MAX_ABS_K1
    assert MIN_RADIUS_SPAN_PX > 0.0
