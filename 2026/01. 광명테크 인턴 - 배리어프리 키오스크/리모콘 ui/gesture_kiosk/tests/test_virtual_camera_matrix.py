"""가상 카메라 전수 검증 — 어떤 배치·거리·화각에서도 커서가 옳은가 (2026-08-31 신설).

왜 이 파일이 필요한가
---------------------
그동안 검증은 **개발 노트북 카메라 한 대**로만 했다. 한 대에서 잘 도는 것과
어떤 카메라에서도 잘 도는 것은 다른 이야기다. 실제로 이 파일을 만들자마자
거울 설정 하나로 커서 방향이 뒤집히는 구멍을 찾았다(2026-08-31).

가상 카메라(tests/virtual_camera.py)는 핀홀 모형으로 관측을 합성하므로
**정답을 알고 있다**. 그래서 "커서가 어디에 있어야 하는가"와 비교할 수 있다.

무엇이 정답인가 — 가상 카메라는 실제 회전을 알고 있다
-----------------------------------------------------
정답 기준을 두 번 갈아엎었다. 그 과정 자체가 기록할 값어치가 있다.

  1차: "머리를 오른쪽으로 돌리면 커서도 오른쪽" -> 거울을 끄면 화면 좌표가
       뒤집히므로 틀렸다.
  2차: "커서 부호 == 기준점이 화면에서 움직인 방향" -> 카메라를 50도 이상
       기울이면 그 화면 이동이 단조성을 잃어 채점이 무의미해졌다. 코드가
       멀쩡한데 점수가 떨어져 한동안 코드를 의심했다.
  3차(지금): **가상 카메라가 만들어 낸 실제 머리 회전**을 기준으로 삼는다.
       합성이므로 정답을 알고 있다 — 관측을 거치지 않으니 카메라 각도에
       흔들리지 않는다.

물리 정의는 이렇다:

    가로 — R_y(+θ)는 코를 사용자 기준 왼쪽으로 보낸다. 거울을 켜면 화면
           오른쪽으로 보이므로 커서는 +, 거울을 끄면 -.
    세로 — R_x(+θ)는 코를 아래로 보낸다. 거울은 세로에 영향이 없으므로 +.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_orientation import HeadOrientation   # noqa: E402
from tests.virtual_camera import (               # noqa: E402
    LENS_PROFILES, MOUNTS, VirtualCamera, rotation,
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


def _expected_sign(axis_index, degrees, mirror):
    """이 회전에서 커서 오프셋이 가져야 할 부호 (모듈 독스트링의 물리 정의)."""
    turn = 1.0 if degrees > 0 else -1.0
    if axis_index == 0:                       # 가로 — 거울이 화면 좌우를 뒤집는다
        return turn * (1.0 if mirror else -1.0)
    return turn                               # 세로 — 거울과 무관


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
        for degrees in (-14.0, -9.0, -5.0, -3.0, 3.0, 5.0, 9.0, 14.0):
            offset = ho.pointing_offset(camera.observe(rotation(axis, degrees)))
            assert offset is not None, (mount_name, mirror, degrees)
            want = _expected_sign(index, degrees, mirror)
            assert offset[index] * want > 0, (mount_name, mirror, axis, degrees)


@pytest.mark.parametrize("mirror", [True, False])
def test_landmark_path_also_follows_screen_motion(mirror):
    """행렬 없이 랜드마크 정합만 쓸 때도 같은 기준을 지켜야 한다."""
    camera = VirtualCamera(mirror=mirror, seed=2)
    ho = _prepared(camera, source="landmarks")
    for degrees in (-12.0, -6.0, 6.0, 12.0):
        offset = ho.pointing_offset(camera.observe(rotation((0.0, 1.0, 0.0), degrees)))
        assert offset is not None
        assert offset[0] * _expected_sign(0, degrees, mirror) > 0, degrees


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
        offset = ho.pointing_offset(camera.observe(rotation((0.0, 1.0, 0.0), degrees)))
        assert offset is not None
        assert offset[0] * _expected_sign(0, degrees, True) > 0, (noise_px, degrees)


# ------------------------------- 연구실 배치 (밑에서 올려봄) 집중 + 워밍업 0

@pytest.mark.parametrize("degrees_up", [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
@pytest.mark.parametrize("mirror", [True, False])
def test_bottom_mounted_camera_any_angle(degrees_up, mirror):
    """★연구실처럼 카메라를 밑에 달아도 모든 각도에서 방향이 옳아야 한다.

    부호를 대수로 정하게 바꾼 뒤로는 80도까지 성립한다. 관측으로 배우던
    때에는 50도부터 증거가 사라져 세로가 흔들렸다(2026-09-02).
    """
    camera = VirtualCamera(mount=rotation((1.0, 0.0, 0.0), degrees_up),
                           mirror=mirror, seed=11)
    ho = _prepared(camera)
    for axis, index in (((0.0, 1.0, 0.0), 0), ((1.0, 0.0, 0.0), 1)):
        for degrees in (-14.0, -6.0, 6.0, 14.0):
            offset = ho.pointing_offset(camera.observe(rotation(axis, degrees)))
            assert offset is not None, (degrees_up, mirror, degrees)
            want = _expected_sign(index, degrees, mirror)
            assert offset[index] * want > 0, (degrees_up, mirror, axis, degrees)


@pytest.mark.parametrize("mount_name", list(MOUNTS))
def test_correct_direction_without_any_warmup(mount_name):
    """★중립만 잡으면 **첫 프레임부터** 방향이 옳아야 한다.

    부호를 관측으로 배우던 때에는 투표가 쌓일 때까지 기다려야 했다.
    대수로 정하면서 그 대기가 사라졌다 — 이 시험이 그것을 지킨다.
    """
    camera = VirtualCamera(mount=MOUNTS[mount_name], seed=12)
    ho = HeadOrientation(rotation_source="auto")
    for _ in range(20):
        ho.add_calibration_sample(camera.observe())
    assert ho.finalize_neutral()          # 워밍업 없음

    for axis, index in (((0.0, 1.0, 0.0), 0), ((1.0, 0.0, 0.0), 1)):
        for degrees in (-12.0, 12.0):
            offset = ho.pointing_offset(camera.observe(rotation(axis, degrees)))
            assert offset is not None, mount_name
            assert offset[index] * _expected_sign(index, degrees, True) > 0, mount_name


# --------------------------------------------------- 렌즈 왜곡 (2026-09-03)
#
# 그동안 "광각에서도 되나"를 초점거리만 줄여서 시험했는데 그것은 틀린
# 시험이었다 — 상대 회전 매핑은 배율에 원리적으로 면역이라 무엇을 넣어도
# 통과한다. 광각의 진짜 문제는 **배럴 왜곡**이고, 아래가 그것을 건다.


@pytest.mark.parametrize("lens", list(LENS_PROFILES))
@pytest.mark.parametrize("mirror", [True, False])
def test_cursor_direction_survives_lens_distortion(lens, mirror):
    """★배럴 왜곡이 걸려도 커서 방향은 옳아야 한다.

    방향이 틀리면 정확도 이전의 문제다 — 커서가 반대로 간다.
    """
    camera = VirtualCamera(lens=lens, mirror=mirror, seed=11)
    ho = _prepared(camera)
    for axis, index in (((0.0, 1.0, 0.0), 0), ((1.0, 0.0, 0.0), 1)):
        for degrees in (-14.0, -9.0, -5.0, 5.0, 9.0, 14.0):
            offset = ho.pointing_offset(camera.observe(rotation(axis, degrees)))
            assert offset is not None, (lens, mirror, degrees)
            want = _expected_sign(index, degrees, mirror)
            assert offset[index] * want > 0, (lens, mirror, axis, degrees)


@pytest.mark.parametrize("lens", list(LENS_PROFILES))
@pytest.mark.parametrize("mount_name", list(MOUNTS))
def test_direction_survives_distortion_on_every_mount(lens, mount_name):
    """★배치와 왜곡이 겹쳐도 방향은 옳아야 한다 (현장은 둘이 함께 온다)."""
    camera = VirtualCamera(mount=MOUNTS[mount_name], lens=lens, seed=12)
    ho = _prepared(camera)
    for axis, index in (((0.0, 1.0, 0.0), 0), ((1.0, 0.0, 0.0), 1)):
        for degrees in (-12.0, -6.0, 6.0, 12.0):
            offset = ho.pointing_offset(camera.observe(rotation(axis, degrees)))
            assert offset is not None, (lens, mount_name, degrees)
            want = _expected_sign(index, degrees, True)
            assert offset[index] * want > 0, (lens, mount_name, axis, degrees)


def test_distortion_actually_changes_the_observation():
    """왜곡을 넣었는데 관측이 그대로면, 위 시험들은 아무것도 안 건 것이다.

    시험이 스스로를 검사한다 — 있으나 마나 한 시험을 만들지 않기 위해.
    """
    clean = VirtualCamera(lens="왜곡없음", seed=13).observe()
    wide = VirtualCamera(lens="초광각 120도", seed=13).observe()
    moved = np.abs(clean.landmarks_px - wide.landmarks_px).max()
    assert moved > 2.0, moved


# --------------------------------------- 회전벡터 분해 (그노몬 투영 대체)


@pytest.mark.parametrize("mount_name", list(MOUNTS))
def test_pure_horizontal_turn_does_not_move_the_cursor_vertically(mount_name):
    """★가로로만 돌리면 세로는 (거의) 가만히 있어야 한다.

    2026-09-03 이전에는 얼굴이 향하는 벡터를 화면에 사영했는데(그노몬 투영),
    얼굴 좌표축이 코가 튀어나온 만큼 19.5도 기울어 있어 좌우로 돌릴 때
    세로가 활처럼 휘었다. 회전벡터로 분해하면 축이 기울었든 축 둘레의
    회전량이 그대로 나오므로 이 휨이 원리적으로 사라진다.
    """
    camera = VirtualCamera(mount=MOUNTS[mount_name], noise_px=0.0, seed=14)
    ho = _prepared(camera)
    verticals = []
    for degrees in np.arange(-14.0, 14.1, 2.0):
        offset = ho.pointing_offset(
            camera.observe(rotation((0.0, 1.0, 0.0), float(degrees))))
        assert offset is not None
        verticals.append(offset[1])
    verticals = np.array(verticals)
    tan_half_y = math.tan(math.radians(10.0))
    bow = float(np.abs(verticals - verticals.mean()).max()) / tan_half_y
    assert bow < 0.05, (mount_name, bow)      # 세로 반폭의 5% 미만
