"""카메라를 다른 각도에 달아도 커서가 같은지 **실제 얼굴 데이터로** 확인한다 (2026-08-31 신설).

무엇을 확인하나
---------------
상대 회전 매핑(src/postprocess/head_orientation.py)은 "카메라를 어떻게 달아도
잴 것이 없다"고 주장한다. 단위 테스트는 그것을 **합성 점구름**으로 증명했지만,
합성 데이터는 잡음이 없고 회전이 정확하다. 실기 보고는 정면 배치에서만
확인됐고, 연구실 키오스크처럼 **밑에서 올려보는 배치**는 아직 미확인이다.

이 도구는 그 사이를 메운다. 카메라를 실제로 옮기지 않고도 확인할 수 있는
이유는 기하학이다 — **카메라를 옮겨 다는 것은, 관측되는 3차원 점구름 전체에
같은 회전을 한 번 더 거는 것과 같다.** 그래서 정면에서 녹화한 진짜 랜드마크
(진짜 잡음, 진짜 사람 움직임)에 그 회전을 걸면 다른 배치에서 본 것과 같은
입력이 된다.

무엇을 못 보나 (솔직히)
-----------------------
점구름을 돌리는 것은 **기하만** 재현한다. 실제로 카메라를 밑에 달면 화면에
보이는 얼굴 자체가 달라진다 — 콧구멍이 보이고 눈이 눌려 보인다. 그러면
MediaPipe가 랜드마크를 **얼마나 정확히** 잡는지가 달라질 수 있는데, 그건
이 방법으로 알 수 없다. 즉 여기서 통과한다는 것은

    "매핑 수식은 배치가 바뀌어도 같은 커서를 낸다"   <- 확인됨
    "밑에서 본 얼굴도 똑같이 잘 검출된다"             <- 확인 안 됨

두 번째는 실제로 그 자리에서 돌려 봐야 한다. 다만 그건 이 매핑의 문제가
아니라 얼굴 검출 자체의 문제이고, 기존 2D 방식도 똑같이 겪는 조건이다.

같은 자리에서 진짜로 확인하는 법
--------------------------------
카메라를 밑에 다는 것과, 그 자리에서 **고개를 숙인 채로 쓰는 것**은 머리와
카메라의 상대 자세가 같다. 그래서 정면 카메라 앞에서 고개를 숙이고
좌우로 훑으면 밑 배치를 몸으로 재현할 수 있다 — 아래 --tilt-guide 참고.

어떻게 쓰나
-----------
    py scripts/verify_camera_mount.py
    py scripts/verify_camera_mount.py --seconds 20
    py scripts/verify_camera_mount.py --tilt-guide     # 고개 숙인 채 재기

측정 중에는 **고개를 좌우로만** 천천히 크게 왕복한다(3~4회). 포물선이
있는지 보는 것이 목적이라 세로로 같이 움직이면 값이 오염된다.
"""
import argparse
import math
import os
import statistics
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2   # noqa: E402

from src.capture.camera_stream import CameraStream                        # noqa: E402
from src.inference.face_estimator import FaceEstimator, select_user_face  # noqa: E402
from src.inference.preprocessor import Preprocessor                       # noqa: E402
from src.postprocess.head_orientation import (                            # noqa: E402
    RIGID_LANDMARKS, HeadOrientation,
)
from src.utils.config_loader import load_config                           # noqa: E402
from src.utils.console import enable_utf8_output                          # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
FIRST_FRAME_TIMEOUT_SEC = 45.0

# 중립을 잡는 데 쓰는 앞쪽 표본 수 — 트래커와 같은 취지(좌표별 중앙값)
NEUTRAL_SAMPLES = 20

# 시험할 카메라 배치. 회전축은 화면 좌표계 기준(x 오른쪽, y 아래, z 안쪽).
#   x축 둘레 = 위아래로 기울여 단 것 (연구실처럼 밑에서 올려보는 배치)
#   z축 둘레 = 옆으로 돌아가게 단 것
MOUNTS = [
    ("정면 (기준)", (1.0, 0.0, 0.0), 0.0),
    ("밑에서 15도", (1.0, 0.0, 0.0), 15.0),
    ("밑에서 25도", (1.0, 0.0, 0.0), 25.0),
    ("밑에서 35도", (1.0, 0.0, 0.0), 35.0),
    ("밑에서 45도", (1.0, 0.0, 0.0), 45.0),
    ("위에서 25도", (1.0, 0.0, 0.0), -25.0),
    ("옆으로 20도", (0.0, 0.0, 1.0), 20.0),
    ("비스듬히", (0.5, -0.4, 0.3), 25.0),
]


class _Face:
    """HeadOrientation이 읽는 것만 갖춘 얼굴."""

    def __init__(self, points_3d):
        self.landmarks_3d = points_3d


def _rot_about(axis, degrees):
    """축 둘레 회전행렬 — 로드리게스 공식."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = math.radians(degrees)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


def _fit_quadratic(xs, ys):
    """y = a + b*x + c*x^2 -> (c, R^2). measure_arc.py와 같은 방식."""
    n = len(xs)
    if n < 6:
        return None
    s = [sum(x ** k for x in xs) for k in range(5)]
    t = [sum(y * (x ** k) for x, y in zip(xs, ys)) for k in range(3)]
    mat = [[s[0], s[1], s[2], t[0]],
           [s[1], s[2], s[3], t[1]],
           [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        pivot_row = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot_row][col]) < 1e-15:
            return None
        mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        pivot = mat[col][col]
        for j in range(col, 4):
            mat[col][j] /= pivot
        for r in range(3):
            if r == col:
                continue
            factor = mat[r][col]
            for j in range(col, 4):
                mat[r][j] -= factor * mat[col][j]
    a, b, c = mat[0][3], mat[1][3], mat[2][3]
    my = sum(ys) / n
    ss_res = sum((y - (a + b * x + c * x * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return c, (1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan"))


def _normalize_span(xs, ys):
    """가로 진폭을 1로 맞춘다(두 축 같은 배율) — 곡률을 같은 단위로 읽으려고."""
    span = max(xs) - min(xs)
    if span < 1e-9:
        return None, None
    cx, cy = statistics.median(xs), statistics.median(ys)
    return [(x - cx) / span for x in xs], [(y - cy) / span for y in ys]


def _record(config, config_path, seconds, tilt_guide):
    """3차원 랜드마크를 프레임마다 통째로 모은다 -> [ (478, 3) 배열 ].

    강체 점만 골라 저장하지 않는 이유: HeadOrientation이 전체 배열을 받아
    자기가 RIGID_LANDMARKS로 고르는 계약이라, 미리 잘라 주면 인덱스가 어긋난다
    (처음에 그렇게 만들었다가 모든 배치가 "표본 부족"으로 나왔다).
    """
    preprocessor = Preprocessor(config)
    estimator = FaceEstimator(config)
    camera = CameraStream(config, config_path=config_path).start()
    frames = []
    seq = -1
    print("   카메라 준비 중...", end="", flush=True)
    import time
    deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
    while True:
        try:
            _raw, seq = camera.capture_new_frame(seq)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                camera.stop()
                print("\n 카메라가 프레임을 주지 않았습니다.")
                return None
            print(".", end="", flush=True)
    print(" 준비됨")
    print()
    if tilt_guide:
        print(" ★고개를 숙인 채로 유지하세요 (밑에 달린 카메라를 몸으로 재현합니다).")
    print(" 지금부터 %.0f초 - **고개를 좌우로만** 천천히 크게 왕복하세요 (3~4회)." % seconds)
    print(" 세로로 같이 움직이면 곡률 측정이 오염됩니다.")
    print()
    started = time.monotonic()
    last = 0.0
    try:
        while time.monotonic() - started < seconds:
            try:
                raw, seq = camera.capture_new_frame(seq)
            except RuntimeError:
                continue
            frame = preprocessor.preprocess_frame(raw, apply_crop=True)
            face = select_user_face(estimator.infer(frame))
            if face is not None and face.landmarks_3d is not None:
                frames.append(np.asarray(face.landmarks_3d, dtype=np.float64))
            elapsed = time.monotonic() - started
            if elapsed - last >= 1.0:
                last = elapsed
                print("\r   %3.0f초 / %.0f초   표본 %d개"
                      % (elapsed, seconds, len(frames)), end="", flush=True)
    finally:
        camera.stop()
    print("\r   표본 %d개 모았습니다.                    " % len(frames))
    return frames


def _track_2d(frames, mount_rot):
    """같은 프레임을 **기존 2D 방식**으로 처리한 궤적 -> [(가로, 세로)].

    비교 대상이 있어야 곡률 숫자가 의미를 가진다. 배치를 바꿨을 때 2D 쪽은
    휘어짐이 달라지고(그래서 ARC_COMPENSATION을 다시 재야 했다) 상대 회전
    쪽은 안 달라진다는 것이 이 작업의 주장이므로, 나란히 놓고 본다.

    카메라를 옮겨 단 뒤 화면에 보이는 좌표는 회전된 3차원 점의 x, y다
    (원근은 무시한 정사영 근사 — 얼굴 크기에 비해 거리가 멀어 차이가 작다).
    eyebrow.py와 같은 방식으로 미간을 안구간거리로 나눈다.
    """
    left_eye, right_eye = 33, 263
    out = []
    for points in frames:
        rotated = points @ mount_rot.T
        lx, ly = rotated[left_eye][0], rotated[left_eye][1]
        rx, ry = rotated[right_eye][0], rotated[right_eye][1]
        dist = math.hypot(rx - lx, ry - ly)
        if dist < 20.0:
            out.append((None, None))
            continue
        out.append(((lx + rx) * 0.5 / dist, (ly + ry) * 0.5 / dist))
    return out


def _cursor_track(frames, mount_rot):
    """이 배치에서의 커서 오프셋 궤적 -> [(가로, 세로)]."""
    orientation = HeadOrientation()
    out = []
    rigid = list(RIGID_LANDMARKS)
    for i, points in enumerate(frames):
        # 카메라를 옮겨 단 효과 — 보이는 점 전체에 같은 회전을 건다.
        # 강체 점만 돌려도 결과는 같지만(뒤에서 그것만 쓴다) 전체를 돌려
        # "카메라를 옮긴 것"이라는 의미를 그대로 유지한다
        rotated = points.copy()
        rotated[rigid] = points[rigid] @ mount_rot.T
        face = _Face(rotated)
        if i < NEUTRAL_SAMPLES:
            orientation.add_calibration_sample(face)
            if i == NEUTRAL_SAMPLES - 1:
                orientation.finalize_neutral()
            continue
        offset = orientation.pointing_offset(face)
        out.append(offset if offset else (None, None))
    return out


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="카메라 배치 무관성 실측 검증")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--tilt-guide", action="store_true",
                        help="고개를 숙인 채 재도록 안내 (밑 배치를 몸으로 재현)")
    args = parser.parse_args()

    config = load_config(args.config)
    print("=" * 70)
    print(" 카메라를 다른 각도에 달아도 커서가 같은가 — 실제 얼굴 데이터로 확인")
    print("=" * 70)

    frames = _record(config, args.config, args.seconds, args.tilt_guide)
    if not frames:
        return 2
    if len(frames) < NEUTRAL_SAMPLES + 40:
        print(" 표본이 %d개뿐이라 분석하지 않습니다." % len(frames))
        return 2

    print()
    print("=" * 70)
    print(" 결과 — 정면에서 녹화한 같은 움직임에, 배치만 바꿔 걸었습니다")
    print("=" * 70)
    print()
    print(f"{'카메라 배치':>14} {'상대회전 곡률':>13} {'2D 곡률':>10} "
          f"{'정면과 최대차':>13} {'실패':>6}")

    def curvature(track):
        good = [(x, y) for x, y in track if x is not None]
        if len(good) < 30:
            return None
        nx, ny = _normalize_span([p[0] for p in good], [p[1] for p in good])
        return _fit_quadratic(nx, ny) if nx else None

    base_track = None
    rows = []
    for label, axis, degrees in MOUNTS:
        rot = _rot_about(axis, degrees)
        track = _cursor_track(frames, rot)
        failed = sum(1 for x, _y in track if x is None)
        new_fit = curvature(track)
        old_fit = curvature(_track_2d(frames, rot))
        if new_fit is None:
            print(f"{label:>14}   표본 부족")
            continue
        if base_track is None:
            base_track = track
            diff = 0.0
        else:
            pairs = [(a, b) for a, b in zip(base_track, track)
                     if a[0] is not None and b[0] is not None]
            diff = max((math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in pairs),
                       default=float("nan"))
        old_curv = old_fit[0] if old_fit else float("nan")
        rows.append((label, new_fit[0], old_curv))
        print(f"{label:>14} {new_fit[0]:+13.4f} {old_curv:+10.4f} "
              f"{diff:13.2e} {failed:6}")

    if len(rows) > 1:
        new_spread = max(r[1] for r in rows) - min(r[1] for r in rows)
        old_spread = max(r[2] for r in rows) - min(r[2] for r in rows)
        print()
        print(f"   배치에 따른 곡률 흔들림   상대회전 {new_spread:.4f}   2D {old_spread:.4f}")
        if abs(new_spread) > 1e-12:
            print(f"   -> 2D 쪽이 {old_spread / new_spread:.0f}배 더 흔들립니다"
                  " (= 배치마다 보정을 다시 재야 하는 정도)")

    print()
    print(" 읽는 법")
    print("   · '정면과 최대차'가 0에 가까우면 배치를 바꿔도 커서가 같다는 뜻입니다.")
    print("     (1e-9 수준이면 부동소수점 오차일 뿐 실제 차이가 없는 것입니다)")
    print("   · 곡률은 절대값이 작을수록 곧게 움직인다는 뜻입니다. 배치가 달라져도")
    print("     곡률이 그대로면, 포물선이 배치에 따라 되살아나지 않는다는 근거입니다.")
    print("   · '실패'는 그 프레임에서 회전을 못 구한 횟수입니다. 배치마다 크게")
    print("     다르면 그 각도에서는 얼굴이 잘 안 잡힌다는 뜻입니다.")
    print()
    print(" ※ 이 시험은 **기하**만 재현합니다. 실제로 카메라를 밑에 달면 보이는")
    print("    얼굴 모양 자체가 달라져(콧구멍이 보이는 등) 랜드마크 검출 정확도가")
    print("    달라질 수 있는데, 그건 이 방법으로 알 수 없습니다. 그 부분은")
    print("    연구실에서 직접 돌려 봐야 하고, 기존 2D 방식도 똑같이 겪는 조건입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
