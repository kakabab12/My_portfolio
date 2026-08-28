"""가로 이동 시 세로가 휘는 양(활 모양)을 실측해 보정 계수를 알려준다 (2026-08-27 신설).

무엇을 재나
-----------
고개를 좌우로만 돌렸는데 커서가 수평이 아니라 뒤집힌 U(∩) 모양으로 휘는 현상.
원인은 커서 기준점에 섞인 코가 얼굴 밖으로 튀어나온 3차원 점이라, 고개를 돌리면
원근 때문에 세로 위치까지 같이 밀리기 때문이다 — 그 밀림이 **가로 이동량의
제곱에 비례**해서 궤적이 2차 곡선이 된다
(head_tracker.py `_arc_compensation` 설명 참고).

이 프로그램은 실제 커서가 그리는 궤적을 받아 2차 곡선을 맞춰(최소제곱 회귀),
**그 곡률을 상쇄할 보정 계수**를 계산해 준다. 추측으로 값을 넣지 않기 위한 도구다.

어떻게 쓰나
-----------
1) 평소처럼 트래커를 켠다 (forehead.py)
2) 캘리브레이션이 끝나 커서가 화면 중앙에 자리잡을 때까지 기다린다
3) 이 프로그램을 켠다
4) **고개를 좌우로만** 천천히 크게 왕복한다 (위아래로는 움직이지 않도록 주의 —
   일부러 올린 세로 움직임까지 "휨"으로 잘못 재게 된다)
5) 화면 좌우 폭을 충분히 훑었으면 q를 눌러 끝낸다
6) 나온 계수를 forehead.py의 ARC_COMPENSATION에 넣는다

왜 커서 좌표를 그대로 쓰나
--------------------------
트래커가 이미 실제 OS 마우스를 움직이고 있으므로, 그 좌표가 곧 사용자가 겪는
최종 결과다. 내부 중간값을 캐내지 않고 결과를 직접 재는 편이 정확하다 —
중간 단계의 어떤 처리가 곡률에 기여하든 전부 포함해서 잡힌다.
"""
import argparse
import ctypes
import math
import os
import sys
import time
from ctypes import wintypes

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from forehead import _cursor_y_to_screen  # noqa: E402
from src.utils.console import enable_utf8_output

SAMPLE_INTERVAL_SEC = 1.0 / 60
MIN_SAMPLES = 120
# 가로로 이만큼(화면 폭 비율)은 훑어야 곡률을 믿을 만하게 잡을 수 있다.
# 좁게만 움직이면 2차항이 잡음에 묻힌다
MIN_X_SPAN_RATIO = 0.30


def _screen_size_px():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _cursor_pos():
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def _key_pressed(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def _fit_quadratic(xs, ys):
    """y = a + b*x + c*x^2 최소제곱 회귀 -> (a, b, c, R^2).

    정규방정식을 3x3 그대로 푼다(가우스 소거) — 표본이 수백 개뿐이라
    수치 안정성 문제가 없고, numpy 없이도 읽기 쉽다.
    """
    n = len(xs)
    if n < 6:
        return None
    s = [sum(x ** k for x in xs) for k in range(5)]      # sum x^0 .. x^4
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

    mean_y = sum(ys) / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x + c * x * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    return a, b, c, r2


def main():
    enable_utf8_output()   # cp949 콘솔에서 줄표(—) 등으로 죽는 것 방지
    parser = argparse.ArgumentParser(
        description="가로 이동 시 세로가 휘는 양을 실측해 보정 계수를 계산한다")
    parser.add_argument("--seconds", type=float, default=25.0,
                        help="최대 측정 시간(초). q를 누르면 그 전에 끝낼 수 있다")
    args = parser.parse_args()

    screen_w_px, screen_h_px = _screen_size_px()
    top_px = _cursor_y_to_screen(0.0) * screen_h_px
    usable_h_px = _cursor_y_to_screen(1.0) * screen_h_px - top_px

    print()
    print("=" * 60)
    print(" 가로 이동 시 세로 휨 측정")
    print("=" * 60)
    print(" 고개를 **좌우로만** 천천히 크게 왕복하세요.")
    print(" 위아래로는 움직이지 마세요 — 그 움직임까지 휨으로 잘못 잽니다.")
    print(" 최대 %.0f초, q를 누르면 즉시 종료." % args.seconds)
    print()

    xs, ys = [], []
    start_sec = time.monotonic()
    next_sample_sec = start_sec
    last_report_sec = start_sec
    while True:
        now_sec = time.monotonic()
        if now_sec - start_sec > args.seconds:
            break
        if _key_pressed(0x51):   # 'Q'
            break
        if now_sec < next_sample_sec:
            time.sleep(0.002)
            continue
        next_sample_sec = now_sec + SAMPLE_INTERVAL_SEC

        px, py = _cursor_pos()
        # 화면 px -> 트래커 내부 커서 비율로 되돌린다
        offset_x = px / max(1, screen_w_px - 1) - 0.5
        offset_y = ((py - top_px) / max(1.0, usable_h_px - 1)) - 0.5
        xs.append(offset_x)
        ys.append(offset_y)

        if now_sec - last_report_sec >= 1.0:
            last_report_sec = now_sec
            span = (max(xs) - min(xs)) if xs else 0.0
            print("  표본 %4d개 · 가로 훑은 폭 %.2f (%.2f 이상 필요)"
                  % (len(xs), span, MIN_X_SPAN_RATIO))

    print()
    if len(xs) < MIN_SAMPLES:
        print(" [부족] 표본이 %d개뿐입니다 (최소 %d개). 더 오래 움직여 주세요."
              % (len(xs), MIN_SAMPLES))
        return 2
    x_span = max(xs) - min(xs)
    if x_span < MIN_X_SPAN_RATIO:
        print(" [부족] 가로로 %.2f밖에 안 움직였습니다 (최소 %.2f)."
              % (x_span, MIN_X_SPAN_RATIO))
        print("        화면 좌우를 더 넓게 훑어야 곡률이 잡음에 안 묻힙니다.")
        return 2

    fit = _fit_quadratic(xs, ys)
    if fit is None:
        print(" [실패] 회귀를 풀지 못했습니다.")
        return 2
    a, b, c, r2 = fit

    print("=" * 60)
    print(" 결과")
    print("=" * 60)
    print(" 표본 %d개 · 가로 훑은 폭 %.2f" % (len(xs), x_span))
    print(" 맞춘 곡선:  세로 = %+.4f %+.4f·가로 %+.4f·가로²   (R² = %.3f)"
          % (a, b, c, r2))
    print()

    # 화면 좌우 끝(가로 offset ±0.5)에서 생기는 세로 오차
    edge_error = c * 0.25
    print(" 화면 좌우 끝에서 세로로 %+.1f%% 밀립니다 (화면 높이 대비)"
          % (edge_error * 100))
    if c > 0:
        print("   -> 아래로 처지는 U(∪) 모양")
    else:
        print("   -> 위로 솟는 뒤집힌 U(∩) 모양")
    print()

    if r2 < 0.3:
        print(" ⚠ R²가 낮습니다(%.3f) — 세로로도 같이 움직였거나 표본이 부족합니다." % r2)
        print("   고개를 좌우로만 움직여 다시 재보세요. 이 값은 믿기 어렵습니다.")
    if abs(edge_error) < 0.01:
        print(" 휨이 화면 높이의 1%% 미만입니다 — 보정이 필요 없는 수준입니다.")
        print(" ARC_COMPENSATION = 0.0  (그대로 두세요)")
    else:
        print(" ▶ forehead.py 에 아래 값을 넣으세요:")
        print()
        print("     ARC_COMPENSATION = %.4f" % (-c))
        print()
        print("   (맞춘 곡률 %+.4f를 상쇄하도록 부호를 뒤집은 값입니다)" % c)
        print("   넣은 뒤 이 프로그램을 다시 돌려 남은 휨이 1%% 아래로")
        print("   떨어졌는지 확인하면 확실합니다.")
    return 0


if __name__ == "__main__":
    # ★2026-08-28 신설 — 창을 더블클릭 등으로 띄우면 결과가 뜨자마자 콘솔
    # 창까지 같이 닫혀 버려 값을 읽을 새가 없었다(사용자 보고). main()이
    # 어느 경로로 끝나든(정상 결과·오류 메시지 전부) 여기서 한 번 멈춰서
    # 사용자가 직접 Enter를 눌러야 닫히게 한다.
    exit_code = main()
    input("\n계속하려면 Enter를 누르세요...")
    sys.exit(exit_code)
