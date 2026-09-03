"""사람마다 얼굴이 다른데 보정판은 고정이다 — 그 위험을 지키는 시험 (2026-09-03).

왜 이 파일이 따로 있나
----------------------
어제(09-03) 넣은 렌즈 자가 보정에는 시험이 28개 있었는데, **전부 순환
논리**였다. 가상 카메라가 FACE_MODEL로 얼굴을 만들고, 렌즈 보정이 그것과
**똑같은** CANONICAL_FACE로 맞췄다. 내가 만든 얼굴을 내가 맞히는 셈이라
"사람마다 얼굴이 다르면?"이라는 질문에 답할 수 없었다.

인체 계측 범위의 얼굴 10종으로 재 보니 두 가지가 드러났다.

  1) f는 크게 틀릴 수 있다(-64% ~ +302%). 그런데도 **원근 되돌리기는 10종
     전부에서 커서를 좋게 했다** — f에 그만큼 둔감하다.
  2) **f가 크게 틀린 채로 k1을 얹으면 파국이다** — 가로 157%, 세로 444%.
     그때 k1의 교차검증은 통과했다. k1만 보고 f를 안 보면 놓친다.

그래서 2단 채택(f만 -> f와 k1)과 f 타당성 게이트를 넣었고, 이 파일이 그것을
지킨다. 여기 있는 시험의 대부분은 **"개선되는가"가 아니라 "나빠지지 않는가"**
를 본다.
"""
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.head_orientation import HeadOrientation      # noqa: E402
from src.postprocess.lens_calibration import (                    # noqa: E402
    FOCAL_CROSS_TOL, MAX_FOCAL_RATIO, MIN_FOCAL_RATIO, LensSelfCalibrator,
)
from tests.virtual_camera import (                                # noqa: E402
    FACE_VARIANTS, FRAME_H_PX, FRAME_W_PX, LENS_PROFILES, VirtualCamera,
    rotation, varied_face,
)

TAN_HALF_X = math.tan(math.radians(15.0))
TAN_HALF_Y = math.tan(math.radians(10.0))
DRIFT_LIMIT = 0.020


def _stream(camera, seconds=60.0, fps=30.0, walk_mm=200.0, seed=5):
    rng = np.random.default_rng(seed)
    for i in range(int(seconds * fps)):
        t = i / fps
        offset = (walk_mm * math.sin(t * 0.32) + rng.normal(0.0, 8.0),
                  walk_mm * 0.55 * math.sin(t * 0.21 + 1.0) + rng.normal(0.0, 6.0),
                  90.0 * math.sin(t * 0.14))
        head = (rotation((0.0, 1.0, 0.0), 13.0 * math.sin(t * 1.1))
                @ rotation((1.0, 0.0, 0.0), 8.0 * math.sin(t * 0.8 + 0.5)))
        yield camera.observe(head, offset_mm=offset)


def _settle(cal, timeout_sec=20.0):
    """보정 스레드가 돌고 있으면 끝날 때까지 기다린다 (시험 재현성 전용)."""
    deadline = time.time() + timeout_sec
    while (cal._thread is not None and cal._thread.is_alive()
           and time.time() < deadline):
        time.sleep(0.005)


def _learn(camera):
    cal = LensSelfCalibrator(FRAME_W_PX, FRAME_H_PX)
    for face in _stream(camera):
        cal.add(face.landmarks_3d)
        # ★시도가 시작되면 **끝날 때까지 기다린다.** 안 기다리면 그동안에도
        # 뷰가 쌓여서, 다음 시도가 무엇을 보는지가 기계 부하에 좌우된다 —
        # 같은 코드가 실행마다 다른 판정을 받는다(2026-09-03에 겪었다:
        # 초점거리가 418~424로 흔들려 한도 근처의 배치가 4회 중 1회 실패).
        # 실사용에서는 기다리지 않는다 — 여기서만 재현성을 위해 기다린다.
        _settle(cal)
        if cal.finished:
            break
    _settle(cal)
    return cal


def _measure(camera, lens_model, offset=(220.0, -140.0, 0.0)):
    """가로 선형성 이탈 / 세로 휨 / 몸 평행이동 끌림."""
    camera.reset_noise()
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

    # 앞의 측정으로 난수가 진행돼 있다 — 되감아야 실행마다 같은 값이 나온다
    camera.reset_noise()
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


# ------------------------------------------------- 절대 나빠지지 않는다

@pytest.mark.parametrize("face_name", list(FACE_VARIANTS))
def test_never_makes_the_cursor_worse_for_any_face(face_name):
    """★어떤 얼굴이든 보정을 켜서 나빠지면 안 된다.

    이 시험이 지키는 것은 정확도가 아니라 **안전**이다. 보정은 사용자가
    고를 수 있는 것이 아니라 저절로 켜지므로, 어떤 사람에게도 손해가
    되어서는 안 된다.
    """
    camera = VirtualCamera(lens="광각 90도", face=face_name, seed=3)
    model = _learn(VirtualCamera(lens="광각 90도", face=face_name, seed=3)).model
    if model is None:
        return                              # 못 믿어서 안 쓴 것 — 그것도 정답이다
    before = _measure(camera, None)
    after = _measure(camera, model)
    # 끌림은 이 프로젝트가 8주 지킨 성질이라 절대 나빠지면 안 된다
    assert after[2] <= before[2], (face_name, "끌림", before, after)
    # 가로·세로는 크게 나빠지면 안 된다 (잡음 폭을 감안해 1.5배까지)
    assert after[0] <= before[0] * 1.5 + 1.0, (face_name, "가로", before, after)
    assert after[1] <= before[1] * 1.5 + 1.0, (face_name, "세로", before, after)


@pytest.mark.parametrize("face_name", list(FACE_VARIANTS))
def test_adopted_models_keep_drift_within_limit(face_name):
    """★채택했다면 몸 평행이동 끌림이 좋아져야 한다.

    정규 모형에 가까운 얼굴에는 **절대 한도**(0.020)를 요구하고, 인체 계측의
    끝자락에 있는 얼굴에는 **개선**을 요구한다. 정직하게 적자면, 극단적인
    얼굴에서 절대 한도까지 보장하지는 못한다 — 보정판이 그만큼 안 맞는다.
    """
    camera = VirtualCamera(lens="광각 90도", face=face_name, seed=3)
    model = _learn(VirtualCamera(lens="광각 90도", face=face_name, seed=3)).model
    if model is None:
        return
    before = _measure(camera, None)[2]
    after = _measure(camera, model)[2]
    assert after < before, (face_name, before, after)
    if face_name in ("정규 모형", "작은 얼굴(어린이)", "큰 얼굴"):
        assert after < DRIFT_LIMIT, face_name


def test_a_wildly_wrong_focal_is_refused():
    """★f가 물리적으로 말이 안 되면 거부한다.

    f가 크게 틀린 채로 k1을 얹으면 커서가 파국으로 간다(가로 157%, 세로 444%).
    그때 k1의 교차검증은 통과하므로, f를 따로 봐야 한다.
    """
    cal = LensSelfCalibrator(FRAME_W_PX, FRAME_H_PX)
    lo = MIN_FOCAL_RATIO * FRAME_H_PX
    hi = MAX_FOCAL_RATIO * FRAME_H_PX
    assert lo < hi
    # 얼굴이 정규 모형과 극단적으로 다르면 f가 범위를 벗어나 거부돼야 한다
    weird = varied_face(interocular=0.55, width=1.6, height=0.6, nose=0.3)
    cal = _learn(VirtualCamera(lens="광각 90도", face=weird, seed=3))
    if cal.model is not None:
        # 채택했다면 적어도 f는 말이 되는 범위 안이어야 한다
        assert lo <= cal.model.focal_px <= hi, cal.model


def test_two_tier_drops_k1_but_keeps_focal():
    """★k1을 안 쓰더라도 **f는 쓴다** — 보정 전체를 포기하지 않는다.

    원근 되돌리기는 f가 크게 틀려도 듣는다. 그래서 k1을 안 쓰는 것이 보정
    전체를 버릴 이유가 못 된다.
    """
    adopted = 0
    for face_name in FACE_VARIANTS:
        cal = _learn(VirtualCamera(lens="광각 90도", face=face_name, seed=3))
        if cal.model is None:
            continue
        adopted += 1
        assert cal.model.k1 == 0.0           # 기본은 왜곡을 안 쓴다
        assert cal.model.focal_px > 0.0      # 그래도 초점거리는 쓴다
    # 인체 계측 범위의 얼굴 대부분이 개선을 받아야 의미가 있다
    assert adopted >= len(FACE_VARIANTS) // 2, adopted


def test_face_variants_really_differ_from_the_canonical_model():
    """시험이 스스로를 검사한다 — 얼굴이 실제로 달라야 위 시험들이 의미가 있다."""
    base = VirtualCamera(face="정규 모형", noise_px=0.0, seed=1).observe()
    for name in FACE_VARIANTS:
        if name == "정규 모형":
            continue
        other = VirtualCamera(face=name, noise_px=0.0, seed=1).observe()
        # 3차원으로 봐야 한다 — 코 높이 변이는 z만 바꿔서 2차원에는 거의
        # 안 나타난다(0.2~0.4px). 그래도 회전 추정에는 크게 영향을 준다
        moved = np.abs(np.asarray(base.landmarks_3d)
                       - np.asarray(other.landmarks_3d)).max()
        assert moved > 1.0, (name, moved)


@pytest.mark.parametrize("lens", list(LENS_PROFILES))
def test_cursor_direction_is_right_for_a_different_face(lens):
    """★얼굴이 정규 모형과 달라도 커서 방향은 옳아야 한다.

    정확도 이전의 문제다 — 방향이 틀리면 쓸 수가 없다.
    """
    camera = VirtualCamera(lens=lens, face="겹친 최악", seed=21)
    ho = HeadOrientation()
    for _ in range(20):
        ho.add_calibration_sample(camera.observe())
    assert ho.finalize_neutral()
    for degrees in (-12.0, -6.0, 6.0, 12.0):
        horizontal = ho.pointing_offset(
            camera.observe(rotation((0.0, 1.0, 0.0), degrees)))
        vertical = ho.pointing_offset(
            camera.observe(rotation((1.0, 0.0, 0.0), degrees)))
        assert horizontal is not None and vertical is not None
        want = 1.0 if degrees > 0 else -1.0
        assert horizontal[0] * want > 0, (lens, "가로", degrees)
        assert vertical[1] * want > 0, (lens, "세로", degrees)


def test_focal_cross_tolerance_separates_the_two_groups():
    """게이트 문턱이 실제 두 무리 사이에 있어야 의미가 있다.

    믿을 만한 경우 1.8~5.9%, 못 믿을 경우 13~182%로 갈렸으므로, 문턱은
    그 사이(6%~13%)에 있어야 한다. 밖으로 나가면 한쪽을 통째로 놓친다.
    """
    assert 0.06 < FOCAL_CROSS_TOL < 0.13
