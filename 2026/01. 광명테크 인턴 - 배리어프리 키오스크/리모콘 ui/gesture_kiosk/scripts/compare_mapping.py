"""2D 랜드마크 매핑과 각도 매핑을 **같은 프레임에서 동시에** 재서 비교한다 (2026-08-31 신설).

무엇을 풀려고 만들었나
----------------------
forehead.py의 HEAD_POSE_MAPPING(각도 기반 커서)은 코드가 다 들어가 있는데도
기본이 꺼져 있다. 그 상수 주석에 이유가 적혀 있다:

    "감도(SENSITIVITY_*)는 지금 2D 좌표 기준으로 맞춰 둔 값이라, 각도 기준으로
     바꾸면 배율이 완전히 달라진다. 켜면 감도를 처음부터 다시 맞춰야 한다."

즉 막고 있던 건 알고리즘이 아니라 **두 신호 사이의 환산 배율을 모른다**는 것
하나였다. 그건 추측할 값이 아니라 재면 나오는 값이다. 이 도구가 그걸 잰다.

두 가지를 한 번에 낸다
----------------------
1) **환산 배율** — 같은 고개 움직임을 두 방식이 각각 얼마로 재는지 회귀해서
   기울기를 구한다. 이 값이 나오면 팀장님이 실기로 확정한 감도(eyebrow 기준
   x=2.05, y=6.0)를 각도 기준으로 **그대로 옮길 수 있다**:

       각도용 감도 = 2D용 감도 / 기울기

   처음부터 다시 맞출 필요가 없어진다.

2) **곡률** — 좌우로만 고개를 돌릴 때 세로가 얼마나 휘는지를 두 방식 각각에
   대해 2차 회귀로 잰다. ARC_COMPENSATION은 2D 방식의 이 휘어짐을 사후에
   빼주는 반창고이고, 카메라 배치가 바뀌면 다시 재야 하는 값이다. 각도 방식은
   투영을 거치지 않으니 원리적으로 휘지 않아야 한다 — 정말 그런지 숫자로
   확인한다.

   비교가 공정하도록 두 방식 모두 **가로 진폭을 1로 정규화한 뒤** 잰다.
   그래서 곡률 값은 "가로로 화면 폭만큼 훑는 동안 세로가 폭의 몇 배만큼
   휘는가"로 같은 단위에서 읽힌다.

어떻게 쓰나
-----------
    py scripts/compare_mapping.py                 # eyebrow 기준 (기본)
    py scripts/compare_mapping.py --mode forehead
    py scripts/compare_mapping.py --seconds 25

키오스크 앞에 평소 쓰는 자세로 서서, **고개를 좌우로만** 천천히 크게
왕복한다(3~4회). 세로로 같이 움직이면 곡률 측정이 오염되므로 좌우만.
q 를 누르면 언제든 끝난다.

왜 이 순서인가
--------------
이 프로젝트는 8월 내내 "재기 전에 값을 넣지 않는다"를 지켜 왔다. 어림값을
먼저 넣었다가 두 번 되돌린 기록이 forehead.py ARC_COMPENSATION 주석에 남아
있고, 클램프 버그도 낮은 R²를 잡음으로 넘겼다가 실기 보고로 뒤늦게 찾았다.
그래서 각도 매핑도 "켜 보고 감으로 맞추기" 대신 **먼저 환산비를 재고** 옮긴다.
"""
import argparse
import math
import os
import statistics
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2   # noqa: E402

from src.capture.camera_stream import CameraStream                        # noqa: E402
from src.inference.face_estimator import (                                # noqa: E402
    FaceEstimator, LMK_LEFT_EYE_OUTER, LMK_RIGHT_EYE_OUTER, select_user_face,
)
from src.inference.preprocessor import Preprocessor                       # noqa: E402
from src.postprocess.head_orientation import HeadOrientation              # noqa: E402
from src.utils.config_loader import load_config                           # noqa: E402
from src.utils.console import enable_utf8_output                          # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

# measure_head_pose.py와 같은 이유로 넉넉히 잡는다 (그 파일 상수 설명 참고)
FIRST_FRAME_TIMEOUT_SEC = 45.0

# 얼굴이 너무 작게 잡히면 정규화 분모가 작아져 값이 통째로 튄다.
# head_tracker.MIN_INTEROCULAR_DIST_PX와 같은 취지의 하한
MIN_DIST_PX = 20.0

# 각도가 이보다 작게만 움직였으면 표본이 부족하다고 본다. 좌우 왕복을
# 제대로 하면 40°는 가볍게 넘는다(8/31 실측 50.5°)
MIN_YAW_SWING_DEG = 20.0

# forehead.py FACE_LOCAL_GAIN과 같은 값 — face_local 모드의 재현을 위해
FACE_LOCAL_GAIN = 2.0


def _fit_linear(xs, ys):
    """y = a + b*x 최소제곱 -> (b, R^2). 표본이 모자라면 None."""
    n = len(xs)
    if n < 6:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-15:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    return b, r2


def _fit_quadratic(xs, ys):
    """y = a + b*x + c*x^2 최소제곱 -> (c, R^2).

    measure_arc.py의 같은 이름 함수와 같은 방식(3x3 정규방정식 + 가우스 소거).
    거기서는 보정 계수를 뽑는 게 목적이라 세 계수를 다 돌려주지만, 여기서는
    "얼마나 휘었나"만 보면 되므로 2차항과 적합도만 돌려준다.
    """
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
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    return c, r2


def _normalize_span(xs, ys):
    """가로 진폭이 1이 되도록 두 축을 **같은 배율로** 줄인다.

    가로만 1로 맞추고 세로를 따로 늘리면 곡률이 왜곡된다 — 두 축에 같은 배율을
    써야 "가로로 이만큼 훑는 동안 세로가 이만큼 휘었다"는 모양이 보존된다.
    """
    if not xs:
        return [], []
    span = max(xs) - min(xs)
    if span < 1e-9:
        return [], []
    cx = statistics.median(xs)
    cy = statistics.median(ys)
    return ([(x - cx) / span for x in xs], [(y - cy) / span for y in ys])


def _measure_2d(face, mode):
    """이번 프레임의 2D 재료를 트래커와 같은 방식으로 낸다 -> (x, y, 안구간거리, 화면세로).

    마지막 값(화면세로)은 매핑에 안 쓰이고 부호 진단에만 쓴다 — 화면 세로는
    위가 0이라 "값이 작다 = 고개를 들었다"가 확실하게 성립하는 유일한 기준이다.

    mode="eyebrow"  — eyebrow.py 그대로. FACE_LOCAL_MAPPING=False라 기준점의
                      화면 픽셀 좌표를 쓰고, 안구간거리로 나누는 건 뒤에서 한다.
                      기준점은 미간(양쪽 눈 바깥쪽 끝의 중점).
    mode="forehead" — forehead.py 그대로. FACE_LOCAL_MAPPING=True라 얼굴
                      좌표계로 바꾼다(head_tracker._measure와 같은 계산).
    """
    lx, ly = face.landmark_px(LMK_LEFT_EYE_OUTER)
    rx, ry = face.landmark_px(LMK_RIGHT_EYE_OUTER)
    ex, ey = rx - lx, ry - ly
    dist = math.hypot(ex, ey)
    if dist < MIN_DIST_PX:
        return None
    px, py = (lx + rx) * 0.5, (ly + ry) * 0.5     # 미간 = 두 눈 중점

    if mode == "eyebrow":
        # 화면 픽셀 그대로. 중심 빼기·거리 나누기는 분석 단계에서 한 번에 한다
        return px, py, dist, py

    # face_local — 두 눈 중점을 원점으로, 두 눈 방향을 가로축으로 (roll 상쇄)
    vx, vy = px - (lx + rx) * 0.5, py - (ly + ry) * 0.5
    ux, uy = ex / dist, ey / dist
    local_x = (vx * ux + vy * uy) / dist
    local_y = (-vx * uy + vy * ux) / dist
    return local_x * FACE_LOCAL_GAIN, local_y * FACE_LOCAL_GAIN, dist, py


def main():
    enable_utf8_output()

    parser = argparse.ArgumentParser(description="2D 매핑 vs 각도 매핑 동시 실측")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=("eyebrow", "forehead"), default="eyebrow",
                        help="어느 트래커의 2D 방식과 비교할지 (기본 eyebrow)")
    parser.add_argument("--seconds", type=float, default=15.0,
                        help="구간당 측정 시간 (좌우/상하 각각, 기본 15초)")
    parser.add_argument("--sens-x", type=float, default=2.05,
                        help="지금 쓰는 2D 기준 가로 감도 (기본 eyebrow의 2.05)")
    parser.add_argument("--sens-y", type=float, default=6.0,
                        help="지금 쓰는 2D 기준 세로 감도 (기본 eyebrow의 6.0)")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    preprocessor = Preprocessor(config)
    face_estimator = FaceEstimator(config)
    camera = CameraStream(config, config_path=args.config).start()

    print("=" * 68)
    print(" 2D 매핑 vs 각도 매핑 — 같은 프레임에서 동시 실측 (%s 기준)" % args.mode)
    print("=" * 68)
    print(" 평소 쓰는 자세로 서세요. 좌우 -> 상하 순서로 두 번 잽니다.")
    print(" 각 구간마다 안내가 나오고 3초 뒤에 시작합니다.")
    print(" 한 구간에서는 그 방향으로만 움직이세요 - 섞이면 값이 오염됩니다.")
    print()
    print(" q 를 누르면 종료. 구간당 최대 %.0f초." % args.seconds)
    print("=" * 68)

    print("   카메라 준비 중...", end="", flush=True)
    frame_seq = -1
    deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
    while True:
        try:
            _raw, frame_seq = camera.capture_new_frame(frame_seq)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                camera.stop()
                print()
                print(" 카메라가 %.0f초 안에 프레임을 주지 않았습니다." % FIRST_FRAME_TIMEOUT_SEC)
                return 2
            print(".", end="", flush=True)
    print(" 준비됨")
    print()

    # ★두 국면으로 나눠 잰다 (2026-08-31).
    #
    # 처음엔 한 번에 몰아 재고 가로·세로 환산비를 같은 표본에서 뽑았는데, 그건
    # 틀린 설계였다. 세로 환산비를 **가로로 움직인 표본**에서 재면, 실제로 재는
    # 건 세로 축의 배율이 아니라 가로 움직임에 딸려온 세로 성분(곡률·축 섞임)이다.
    # 실측에서도 R^2가 0.967로 높게 나와 그럴듯해 보였지만, 그 높은 적합도는
    # "가로에 딸려오는 세로가 일정하게 딸려온다"는 뜻일 뿐 세로 감도와는 무관하다.
    #
    # 그래서 축마다 그 축으로만 움직인 표본에서 잰다.
    phases = [
        ("horizontal", "고개를 **좌우로만** 천천히 크게 왕복하세요 (3~4회)"),
        ("vertical",   "고개를 **위아래로만** 천천히 크게 왕복하세요 (3~4회)"),
    ]
    collected = {}
    aborted = False
    try:
        for name, guide in phases:
            if aborted:
                break
            print("-" * 68)
            print(" [%s] %s" % ("좌우" if name == "horizontal" else "상하", guide))
            for count in (3, 2, 1):
                print("\r   %d초 뒤 시작..." % count, end="", flush=True)
                if _sleep_or_quit(1.0, args.no_window):
                    aborted = True
                    break
            if aborted:
                break
            print("\r   측정 중          ")

            rows = []
            # ★상대 회전 매핑(2026-08-31)도 같은 프레임에서 함께 잰다 —
            # "진짜 사람 움직임에서도 정말 안 휘는가"를 확인하려면 합성 데이터가
            # 아니라 실기에서 재야 한다
            orientation = HeadOrientation()
            started = time.monotonic()
            last_note = 0.0
            while time.monotonic() - started < args.seconds:
                try:
                    raw, frame_seq = camera.capture_new_frame(frame_seq)
                except RuntimeError:
                    continue
                # measure_head_pose.py와 같은 호출 — 트래커 본체와 같은 화면을 봐야
                # 재는 값이 실제 동작과 일치한다 (세로 크롭 포함)
                frame = preprocessor.preprocess_frame(raw, apply_crop=True)
                faces = face_estimator.infer(frame)
                face = select_user_face(faces)
                if face is not None and face.head_pose is not None:
                    m = _measure_2d(face, args.mode)
                    if m is not None:
                        pose = face.head_pose
                        # 처음 몇 장으로 중립을 잡고, 그 뒤부터 오프셋을 낸다
                        if not orientation.is_ready:
                            orientation.add_calibration_sample(face)
                            if orientation.sample_count >= 15:
                                orientation.finalize_neutral()
                            ox = oy = None
                        else:
                            off = orientation.pointing_offset(face)
                            ox, oy = (off if off else (None, None))
                        rows.append((m[0], m[1], m[2],
                                     math.tan(math.radians(pose.yaw_deg)),
                                     -math.tan(math.radians(pose.pitch_deg)),
                                     pose.yaw_deg, pose.pitch_deg, m[3], ox, oy))

                elapsed = time.monotonic() - started
                if elapsed - last_note >= 1.0:
                    last_note = elapsed
                    print("\r   %4.0f초 / %.0f초   표본 %d개"
                          % (elapsed, args.seconds, len(rows)), end="", flush=True)

                if not args.no_window:
                    cv2.imshow("compare_mapping (q=종료)", frame)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        aborted = True
                        break
            collected[name] = rows
            print("\r   %s 구간 끝 - 표본 %d개          "
                  % ("좌우" if name == "horizontal" else "상하", len(rows)))
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        if not args.no_window:
            cv2.destroyAllWindows()

    print()
    _report(collected, args)
    return 0


def _sleep_or_quit(seconds, no_window):
    """카운트다운 대기. 창이 있으면 그 사이에도 q를 받는다 -> True면 중단."""
    if no_window:
        time.sleep(seconds)
        return False
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
            return True
    return False


def _centered(rows, mode):
    """표본을 트래커가 실제로 쓰는 형태로 바꾸고 중앙값을 뺀다.

    -> (2D가로, 2D세로, 각도가로, 각도세로)
    """
    dist_med = statistics.median(r[2] for r in rows)
    cx = statistics.median(r[0] for r in rows)
    cy = statistics.median(r[1] for r in rows)
    if mode == "eyebrow":
        # update()가 (좌표차이) / 안구간거리 로 만든다 (FACE_LOCAL_MAPPING=False)
        d2x = [(r[0] - cx) / dist_med for r in rows]
        d2y = [(r[1] - cy) / dist_med for r in rows]
    else:
        # face_local이면 _measure가 이미 나눠 놓았다
        d2x = [r[0] - cx for r in rows]
        d2y = [r[1] - cy for r in rows]
    cax = statistics.median(r[3] for r in rows)
    cay = statistics.median(r[4] for r in rows)
    return d2x, d2y, [r[3] - cax for r in rows], [r[4] - cay for r in rows]


def _report(collected, args):
    print("=" * 68)
    print(" 결과")
    print("=" * 68)

    horiz = collected.get("horizontal", [])
    vert = collected.get("vertical", [])
    for label, rows in (("좌우", horiz), ("상하", vert)):
        print(" %s 구간 표본 %d개" % (label, len(rows)))
    print()

    # --- 1. 환산 배율 — 축마다 그 축으로 움직인 구간에서만 잰다 ---
    print(" [1] 환산 배율 — 같은 움직임을 두 방식이 각각 얼마로 재는가")
    print()
    results = []
    axes = (("가로", horiz, 0, args.sens_x, 5, "yaw"),
            ("세로", vert, 1, args.sens_y, 6, "pitch"))
    for label, rows, axis_idx, sens, deg_idx, deg_name in axes:
        if len(rows) < 30:
            print("   %s  표본 부족 (%d개, 최소 30개)" % (label, len(rows)))
            continue
        degs = [r[deg_idx] for r in rows]
        swing = max(degs) - min(degs)
        d2 = _centered(rows, args.mode)
        a2d = d2[axis_idx]
        aang = d2[2 + axis_idx]
        fit = _fit_linear(aang, a2d)      # 2D = 기울기 x 각도
        if fit is None:
            print("   %s  적합 실패" % label)
            continue
        slope, r2 = fit
        note = ""
        if swing < MIN_YAW_SWING_DEG:
            note = "   ⚠ %s를 %.0f도밖에 안 움직였습니다" % (deg_name, swing)
        elif r2 < 0.8:
            note = "   ⚠ R^2가 낮아 신뢰도 부족"
        print("   %s  2D = %+.4f x 각도   (R^2 %.3f, %s 폭 %.1f도)%s"
              % (label, slope, r2, deg_name, swing, note))
        results.append((label, slope, r2, sens))
    print()
    print("   -> 각도 기준으로 옮긴 감도 (지금 감도 / 기울기):")
    if not results:
        print("      표본이 모자라 계산하지 못했습니다.")
    for label, slope, r2, sens in results:
        if abs(slope) < 1e-9:
            continue
        conv = sens / slope
        flip = ""
        if conv < 0:
            # 기울기가 음수면 두 방식의 축 방향이 반대다. 감도에 음수를 넣는
            # 대신 부호를 어디서 뒤집을지는 코드에서 정해야 한다 - 그냥 옮기면
            # 커서가 반대로 움직인다
            flip = "   ★부호 반대 - 축 방향을 뒤집어야 합니다"
        print("      %s  %.2f / %+.4f = %+.3f%s" % (label, sens, slope, conv, flip))
    print()

    # --- 1-2. pitch 부호 진단 ---
    #
    # head_tracker._measure_head_pose는 "위를 보면 pitch가 양수"라고 가정하고
    # -tan(pitch)를 쓴다. 그 가정이 맞는지 여기서 직접 확인한다. 기준은 화면
    # 세로다 - 위가 0이므로 "값이 작다 = 고개를 들었다"가 확실히 성립한다.
    print(" [1-2] pitch 부호 — 코드의 가정('위를 보면 pitch 양수')이 맞는가")
    print()
    if len(vert) < 30:
        print("   상하 구간 표본이 부족해 확인하지 못했습니다.")
    else:
        screen_y = [r[7] for r in vert]
        pitches = [r[6] for r in vert]
        swing = max(pitches) - min(pitches)
        fit = _fit_linear(screen_y, pitches)
        if fit is None:
            print("   적합 실패")
        else:
            slope, r2 = fit
            # 화면 세로가 커진다 = 고개를 내렸다.
            #   그때 pitch가 작아지면(기울기 음수) -> 들었을 때 pitch가 크다 = 가정 맞음
            ok = slope < 0
            print("   pitch = %+.4f x 화면세로   (R^2 %.3f, pitch 폭 %.1f도)"
                  % (slope, r2, swing))
            if swing < 10.0:
                print("   ⚠ 상하로 %.0f도밖에 안 움직여 판정이 불안정합니다." % swing)
            elif r2 < 0.5:
                print("   ⚠ 적합도가 낮아 판정이 불안정합니다.")
            elif ok:
                print("   -> 가정이 맞습니다. -tan(pitch) 그대로 쓰면 됩니다.")
            else:
                print("   -> ★가정이 틀렸습니다. 위를 볼 때 pitch가 **음수**입니다.")
                print("      지금 코드대로 HEAD_POSE_MAPPING을 켜면 상하가 뒤집힙니다.")
    print()

    # --- 2. 곡률 — 좌우 구간에서만 의미가 있다 ---
    print(" [2] 곡률 — 좌우로만 훑을 때 세로가 얼마나 휘는가")
    print("     (가로 진폭을 1로 맞춘 뒤라 두 방식을 같은 단위로 비교합니다)")
    print()
    if len(horiz) < 30:
        print("   좌우 구간 표본이 부족해 계산하지 못했습니다.")
    else:
        d2x, d2y, dax, day = _centered(horiz, args.mode)
        pairs = [("2D 랜드마크", d2x, d2y), ("각도(head pose)", dax, day)]
        ori = [(r[8], r[9]) for r in horiz if r[8] is not None]
        if len(ori) >= 30:
            pairs.append(("상대회전(신규)", [o[0] for o in ori], [o[1] for o in ori]))
        else:
            print("   상대회전(신규)  표본 부족 - 중립을 잡는 동안은 값이 안 나옵니다")
        for label, ax, ay in pairs:
            nx, ny = _normalize_span(ax, ay)
            fit = _fit_quadratic(nx, ny) if nx else None
            if fit is None:
                print("   %-16s 적합 실패" % label)
                continue
            curv, r2 = fit
            print("   %-16s 곡률 %+.4f   (2차 적합 R^2 %.3f)" % (label, curv, r2))
    print()
    print(" 읽는 법")
    print("   · 곡률의 절대값이 작을수록 곧게 움직인다는 뜻입니다.")
    print("   · 각도 쪽 곡률이 2D보다 뚜렷이 작으면, ARC_COMPENSATION이라는")
    print("     사후 보정 자체가 필요 없어진다는 근거가 됩니다.")
    print("   · 2차 적합 R^2가 낮으면 애초에 포물선이 아니라는 뜻이라,")
    print("     그 방식에는 곡률 보정이 맞지 않는 처방입니다.")


if __name__ == "__main__":
    sys.exit(main())
