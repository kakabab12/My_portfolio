"""가상 카메라 전수 검증 — 어떤 배치·거리·화각에서도 커서가 옳은가 (2026-08-31 신설).

왜 이 파일이 필요한가
---------------------
그동안 검증은 **개발 노트북 카메라 한 대**로만 했다. 한 대에서 잘 도는 것과
어떤 카메라에서도 잘 도는 것은 다른 이야기다. 실제로 이 파일을 만들자마자
거울 설정 하나로 커서 방향이 뒤집히는 구멍을 찾았다(2026-08-31).

가상 카메라(tests/virtual_camera.py)는 핀홀 모형으로 관측을 합성하므로
**정답을 알고 있다**. 그래서 "커서가 어디에 있어야 하는가"와 비교할 수 있다.

무엇이 정답인가 — 커서는 화면 이동을 따른다
-------------------------------------------
검증 기준을 처음에 "머리를 오른쪽으로 돌리면 커서도 오른쪽"으로 잡았다가
거울을 끈 조건에서 오차 44%가 나왔다. 그런데 그것은 코드가 아니라 **기준이
틀린 것**이었다 — 거울을 끄면 화면 좌표 자체가 뒤집히므로, 커서도 화면
기준으로 반대쪽에 가는 것이 옳다.

그래서 정답은 하나로 정리된다:

    커서 오프셋의 부호 == 기준점이 화면에서 움직인 방향의 부호

이 기준은 거울 설정·카메라 배치·MediaPipe 축 규약과 무관하게 성립한다.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_orientation import (   # noqa: E402
    SIGN_REFERENCE_LANDMARK, HeadOrientation,
)
from tests.virtual_camera import (               # noqa: E402
    MOUNTS, VirtualCamera, rotation,
)

HALF_X_DEG = 15.0
TAN_HALF_X = math.tan(math.radians(HALF_X_DEG))

# 부호 학습이 잠기기까지 왕복시키는 횟수 — 실사용에서 1~2초에 해당
WARMUP_SWEEPS = 30


def _prepared(camera, source="auto"):
    """중립을 잡고 부호가 잠길 때까지 왕복시킨 HeadOrientation."""
    ho = HeadOrientation(rotation_source=source)
    for _ in range(20):
        ho.add_calibration_sample(camera.observe())
    assert ho.finalize_neutral()
    for i in range(WARMUP_SWEEPS):
        swing = 1.0 if i % 2 else -1.0
        ho.pointing_offset(camera.observe(rotation((0.0, 1.0, 0.0), 9.0 * swing)))
        ho.pointing_offset(camera.observe(rotation((1.0, 0.0, 0.0), 7.0 * swing)))
    return ho


def _screen_move(camera, head_rotation):
    """기준점이 중립 대비 화면에서 얼마나 움직였나 -> (dx, dy)."""
    base = camera.observe().landmarks_3d[SIGN_REFERENCE_LANDMARK]
    now = camera.observe(head_rotation).landmarks_3d[SIGN_REFERENCE_LANDMARK]
    return float(now[0] - base[0]), float(now[1] - base[1])


# ------------------------------------------------- 방향 (전 배치 × 거울 ON/OFF)

@pytest.mark.parametrize("mount_name", list(MOUNTS))
@pytest.mark.parametrize("mirror", [True, False])
def test_cursor_follows_screen_motion_everywhere(mount_name, mirror):
    """★어떤 배치·거울 설정에서도 커서가 화면 이동 방향을 따라야 한다.

    이 시험이 2026-08-31에 실제로 구멍을 잡았다 — 부호 판정의 진실 기준을
    얼굴 축(거울에 따라 뒤집힘)으로 삼았던 것을 화면 이동으로 바꿨다.
    """
    camera = VirtualCamera(mount=MOUNTS[mount_name], mirror=mirror, seed=1)
    ho = _prepared(camera)

    for axis, index in (((0.0, 1.0, 0.0), 0), ((1.0, 0.0, 0.0), 1)):
        for degrees in (-14.0, -9.0, -5.0, 5.0, 9.0, 14.0):
            rot = rotation(axis, degrees)
            offset = ho.pointing_offset(camera.observe(rot))
            assert offset is not None, (mount_name, mirror, degrees)
            move = _screen_move(camera, rot)[index]
            assert offset[index] * move > 0, (mount_name, mirror, axis, degrees)


@pytest.mark.parametrize("mirror", [True, False])
def test_landmark_path_also_follows_screen_motion(mirror):
    """행렬 없이 랜드마크 정합만 쓸 때도 같은 기준을 지켜야 한다."""
    camera = VirtualCamera(mirror=mirror, seed=2)
    ho = _prepared(camera, source="landmarks")
    for degrees in (-12.0, -6.0, 6.0, 12.0):
        rot = rotation((0.0, 1.0, 0.0), degrees)
        offset = ho.pointing_offset(camera.observe(rot))
        assert offset is not None
        assert offset[0] * _screen_move(camera, rot)[0] > 0, degrees


def test_mirror_flips_the_learned_sign():
    """거울을 켜고 끄면 배운 부호가 서로 반대여야 한다 — 그래야 화면 기준이
    양쪽에서 모두 성립한다."""
    on = _prepared(VirtualCamera(mirror=True, seed=3))
    off = _prepared(VirtualCamera(mirror=False, seed=3))
    assert on._sign_v is not None and off._sign_v is not None
    assert on._sign_v == -off._sign_v


# --------------------------------------------------------- 크기 (배치 무관성)

@pytest.mark.parametrize("mount_name", list(MOUNTS))
def test_cursor_magnitude_is_mount_independent(mount_name):
    """배치를 바꿔도 같은 회전에는 같은 크기의 오프셋이 나와야 한다.

    카메라 배치는 상대 회전에서 소거되므로, 정면과의 차이가 잡음 수준
    (여기서는 5%)을 넘으면 안 된다.
    """
    front = _prepared(VirtualCamera(mount=MOUNTS["정면"], seed=4))
    other = _prepared(VirtualCamera(mount=MOUNTS[mount_name], seed=4))
    front_cam = VirtualCamera(mount=MOUNTS["정면"], seed=4)
    other_cam = VirtualCamera(mount=MOUNTS[mount_name], seed=4)

    for degrees in (-12.0, -6.0, 6.0, 12.0):
        rot = rotation((0.0, 1.0, 0.0), degrees)
        a = front.pointing_offset(front_cam.observe(rot))
        b = other.pointing_offset(other_cam.observe(rot))
        assert a is not None and b is not None
        assert abs(abs(b[0]) - abs(a[0])) < abs(a[0]) * 0.05 + 0.01, (mount_name, degrees)


@pytest.mark.parametrize("distance_mm", [350.0, 600.0, 1200.0])
def test_distance_does_not_change_the_cursor(distance_mm):
    """가까이·멀리 서도 같은 회전에는 같은 커서 — 원근 세기가 달라져도."""
    ref_cam = VirtualCamera(distance_mm=600.0, seed=5)
    ref = _prepared(ref_cam)
    cam = VirtualCamera(distance_mm=distance_mm, seed=5)
    ho = _prepared(cam)
    for degrees in (-10.0, 10.0):
        rot = rotation((0.0, 1.0, 0.0), degrees)
        a = ref.pointing_offset(ref_cam.observe(rot))
        b = ho.pointing_offset(cam.observe(rot))
        assert abs(abs(b[0]) - abs(a[0])) < abs(a[0]) * 0.10 + 0.01, distance_mm


@pytest.mark.parametrize("focal_px", [400.0, 700.0, 1400.0])
def test_field_of_view_does_not_change_the_cursor(focal_px):
    """광각이든 망원이든 같은 회전에는 같은 커서."""
    ref_cam = VirtualCamera(focal_px=700.0, seed=6)
    ref = _prepared(ref_cam)
    cam = VirtualCamera(focal_px=focal_px, seed=6)
    ho = _prepared(cam)
    for degrees in (-10.0, 10.0):
        rot = rotation((0.0, 1.0, 0.0), degrees)
        a = ref.pointing_offset(ref_cam.observe(rot))
        b = ho.pointing_offset(cam.observe(rot))
        assert abs(abs(b[0]) - abs(a[0])) < abs(a[0]) * 0.10 + 0.01, focal_px


# ------------------------------------------------------------ 몸 이동 무관성

@pytest.mark.parametrize("mount_name", ["정면", "밑에서 35도", "비스듬히"])
def test_body_translation_does_not_move_the_cursor(mount_name):
    """★몸이 옆으로 움직여도 커서는 제자리여야 한다.

    이 프로젝트가 8월 내내 싸운 "커서 밀림"의 핵심 조건이다. 상대 회전은
    평행이동에 반응하지 않으므로 원리적으로 0이어야 한다.
    """
    camera = VirtualCamera(mount=MOUNTS[mount_name], seed=7)
    ho = _prepared(camera)
    still = ho.pointing_offset(camera.observe())
    assert still is not None
    for shift in ((60.0, 0.0, 0.0), (-60.0, 0.0, 0.0),
                  (0.0, 50.0, 0.0), (0.0, 0.0, 120.0)):
        moved = ho.pointing_offset(camera.observe(offset_mm=shift))
        assert moved is not None
        drift = math.hypot(moved[0] - still[0], moved[1] - still[1])
        # 화면 절반이 tan(15도)=0.268이므로 0.02는 화면의 약 3.7%
        assert drift < 0.02, (mount_name, shift, drift)


# ------------------------------------------------------------------- 곡률

@pytest.mark.parametrize("mount_name", list(MOUNTS))
def test_horizontal_sweep_stays_horizontal(mount_name):
    """★좌우로만 돌릴 때 세로가 크게 휘면 안 된다 (포물선 재발 방지).

    가로 진폭 대비 세로 변동으로 잰다. 배치가 달라져도 이 값이 유지되는
    것이 "곡률 보정을 다시 잴 필요가 없다"의 근거다.
    """
    camera = VirtualCamera(mount=MOUNTS[mount_name], seed=8)
    ho = _prepared(camera)
    xs, ys = [], []
    for degrees in np.linspace(-14.0, 14.0, 15):
        offset = ho.pointing_offset(camera.observe(rotation((0.0, 1.0, 0.0), degrees)))
        assert offset is not None
        xs.append(offset[0])
        ys.append(offset[1])
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    assert span_x > 0.2, "좌우로 충분히 움직이지 않았다면 시험이 무의미하다"
    assert span_y < span_x * 0.15, (mount_name, span_y / span_x)


# ---------------------------------------------------------------- 잡음 내성

@pytest.mark.parametrize("noise_px", [0.0, 0.35, 1.0])
def test_survives_landmark_noise(noise_px):
    """랜드마크 잡음이 커져도 방향은 흔들리면 안 된다."""
    camera = VirtualCamera(noise_px=noise_px, seed=9)
    ho = _prepared(camera)
    for degrees in (-12.0, 12.0):
        rot = rotation((0.0, 1.0, 0.0), degrees)
        offset = ho.pointing_offset(camera.observe(rot))
        assert offset is not None
        assert offset[0] * _screen_move(camera, rot)[0] > 0, (noise_px, degrees)
