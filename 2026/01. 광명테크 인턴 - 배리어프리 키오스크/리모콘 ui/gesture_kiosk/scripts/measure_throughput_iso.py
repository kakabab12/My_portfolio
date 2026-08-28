"""처리량(Throughput) 측정 — ISO 9241-411 표준 다방향 탭 시험 (2026-08-27 신설).

왜 이걸 따로 만드나
-------------------
measure_accuracy.py는 "성공률·걸린 시간·최소 버튼 크기"를 잰다. 실사용 설계에는
그게 맞지만, **다른 입력장치와 견줄 수 있는 공통 잣대**가 아니다. "우리 건
성공률 90%"라고 해도 심사위원은 "마우스보다 나은가?"를 알 수 없다.

ISO 9241-411은 마우스·트랙볼·터치·시선추적을 **같은 숫자 하나로** 비교하도록
국제적으로 합의된 시험 절차다. 그 숫자가 처리량(Throughput, 단위 bit/s)이다.
값이 나오면 이렇게 말할 수 있다:

    "이 헤드트래커의 처리량은 X bit/s로, 같은 방식으로 측정된 공개 문헌의
     머리 조작 2.04 bit/s·시선 조작 1.85 bit/s와 견줄 수 있다."

근거 (전부 원문 확인함)
-----------------------
· ISO 9241-411:2012 (2022 재확인) — 비키보드 입력장치 성능 평가 시험 절차.
  6가지 과제 중 **다방향 탭 시험(multi-directional tapping)** 을 쓴다.
· 처리량 TP = IDe / MT      (bit/s)
    IDe = log2(Ae / We + 1)   유효 난이도 (bit)
    Ae  = 실제 이동 거리 평균 (과녁 중심까지의 명목 거리 d + 평균 편차)
    We  = 4.133 x SDx        유효 과녁 폭
  We의 4.133은 정규분포 +-2.066 표준편차 = 전체의 96%에 해당하는 값이다.
  즉 **실제로 얼마나 정확히 찍었는지**를 과녁 폭에 반영한다 — 크게 빗나가면
  과녁이 실제로는 더 컸던 셈으로 쳐서 난이도를 깎는다. 그래서 "대충 빨리
  찍기"로 점수를 올릴 수 없다(속도-정확도 상충이 자동으로 상쇄된다).
· 편차 dx 계산은 MacKenzie의 표준 사영식을 쓴다 (아래 _projected_deviation).

Fitts 법칙 자체는 Fitts(1954), 지금 쓰는 Shannon 형태는 MacKenzie(1992).

어떻게 쓰나
-----------
1) 평소처럼 트래커를 켠다 (forehead.py 권장 — head.py/eyebrow.py도 됨)
2) 이 프로그램을 따로 켠다
3) 원을 따라 배치된 과녁 중 **분홍색으로 강조된 것**을 고개로 겨냥해
   입을 벌려 누른다. 강조가 원 반대편으로 건너뛰며 이어진다
4) 끝나면 처리량과 Fitts 회귀 결과가 보고서용으로 정리돼 나온다

트래커 오버레이가 투명해 이 창의 과녁이 그대로 비친다 — measure_accuracy.py와
같은 이유로 이 창을 맨 앞으로 올리지 않는다(올리면 커서 점이 가려진다).

측정 조건
---------
거리(A) x 과녁크기(W) 조합을 여러 개 돌린다. 조합마다 난이도(ID)가 달라야
Fitts 회귀선이 의미를 갖는다 — ID가 한 점에 몰리면 기울기를 못 구한다.
"""
import argparse
import csv
import ctypes
import math
import os
import statistics
import sys
import time
from ctypes import wintypes

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from forehead import CURSOR_Y_SPAN, _cursor_y_to_screen, put_korean_text  # noqa: E402

WINDOW_NAME = "ISO 9241-411 throughput"

# 원 위에 놓는 과녁 개수 — ISO 표준 예시가 쓰는 홀수. 홀수여야 강조가 원
# 반대편으로 건너뛰며 한 바퀴에 모든 과녁을 정확히 한 번씩 지난다
TARGET_COUNT = 13

# (원 지름 비율, 과녁 지름 cm) 조합. 원 지름 비율은 "커서가 쓸 수 있는 영역의
# 짧은 변" 대비. 난이도(ID)가 서로 충분히 벌어지게 골랐다 — 아래 표 참고
CONDITIONS = (
    (0.75, 4.0),
    (0.75, 2.0),
    (0.45, 2.0),
    (0.75, 1.2),
    (0.45, 1.2),
)

TIMEOUT_SEC = 20.0            # 한 과녁에 이만큼 넘게 걸리면 중단(측정 불가로 본다)
SETTLE_BETWEEN_SEC = 0.6      # 직전 클릭이 다음 시도로 새지 않게
WARMUP_TAPS = 1               # 조합마다 버리는 연습 탭 수(첫 이동은 조건이 다르다)


def _screen_size_px():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _screen_width_mm():
    """모니터 물리 폭(mm) 추정 — measure_accuracy.py와 동일 방식."""
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        if not hdc:
            return None
        try:
            HORZSIZE = 4
            mm = ctypes.windll.gdi32.GetDeviceCaps(hdc, HORZSIZE)
        finally:
            ctypes.windll.user32.ReleaseDC(0, hdc)
        return mm if mm and mm > 50 else None
    except Exception:   # noqa: 방어적 — 못 구하면 사용자가 직접 넣는다
        return None


def _cursor_pos():
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def _left_button_down():
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)


def _tap_order(count):
    """ISO 다방향 탭 순서 — 원 반대편으로 건너뛰며 한 바퀴.

    step = (count+1)//2 로 돌면 홀수 count에서 모든 과녁을 정확히 한 번씩
    지나고 출발점으로 돌아온다. 이렇게 해야 매 이동 거리가 원 지름에
    가깝게 일정해진다(표준이 요구하는 조건).
    """
    step = (count + 1) // 2
    return [(i * step) % count for i in range(count)]


def _target_centers(center_px, radius_px, count):
    """원 위에 count개를 균등 배치. 12시부터 시계방향."""
    centers = []
    for i in range(count):
        angle = -math.pi / 2 + (2 * math.pi * i / count)
        centers.append((center_px[0] + radius_px * math.cos(angle),
                        center_px[1] + radius_px * math.sin(angle)))
    return centers


def _projected_deviation(from_px, to_px, select_px):
    """과녁 중심에서 얼마나 빗나갔는지를 **이동 축에 사영**해 돌려준다 (px).

    MacKenzie의 표준식. 세 변의 길이만으로 사영 길이를 구한다(삼각형 제2코사인
    법칙 변형) — 각도를 따로 구할 필요가 없다.

        d = |to - from|          이번 이동의 명목 거리
        a = |select - from|
        b = |select - to|
        dx = (a^2 - b^2 + d^2) / (2d) - d

    dx > 0 이면 지나쳤고(overshoot), dx < 0 이면 못 미쳤다(undershoot).
    이 dx들의 표준편차가 곧 실제 조준 산포이고, We = 4.133 x SD(dx)가 된다.
    """
    d = math.dist(from_px, to_px)
    if d < 1e-6:
        return None, 0.0
    a = math.dist(select_px, from_px)
    b = math.dist(select_px, to_px)
    dx = (a * a - b * b + d * d) / (2.0 * d) - d
    return dx, d


def _draw(canvas, centers, radius_px, active_index, next_index, headline, subline):
    canvas[:] = (18, 18, 18)
    for i, c in enumerate(centers):
        cx, cy = int(c[0]), int(c[1])
        if i == active_index:
            cv2.circle(canvas, (cx, cy), radius_px, (200, 90, 230), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), radius_px, (255, 255, 255), 2, cv2.LINE_AA)
        elif i == next_index:
            cv2.circle(canvas, (cx, cy), radius_px, (70, 70, 70), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), radius_px, (140, 140, 140), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy), radius_px, (48, 48, 48), -1, cv2.LINE_AA)
    put_korean_text(canvas, headline, (30, 30), 26, (255, 255, 255))
    put_korean_text(canvas, subline, (30, 66), 20, (170, 170, 170))
    return canvas


def _run_condition(canvas, centers, radius_px, order, label, sub):
    """한 조합을 끝까지 돌고 시도 기록 목록을 돌려준다. 중단이면 None."""
    trials = []
    prev_center = centers[order[0]]
    # 첫 과녁으로 일단 이동시켜 시작점을 맞춘다(이 이동은 기록하지 않는다)
    for seq in range(1, len(order) + 1):
        active = order[seq % len(order)]
        target_center = centers[active]
        upcoming = order[(seq + 1) % len(order)]

        # 직전 클릭이 새지 않도록 버튼이 떨어질 때까지 + 잠깐 대기
        settle_until = time.monotonic() + SETTLE_BETWEEN_SEC
        while time.monotonic() < settle_until or _left_button_down():
            _draw(canvas, centers, radius_px, active, upcoming, label, sub)
            cv2.imshow(WINDOW_NAME, canvas)
            if (cv2.waitKey(16) & 0xFF) in (27, ord('q')):
                return None

        start_sec = time.monotonic()
        select_px = None
        while True:
            _draw(canvas, centers, radius_px, active, upcoming, label, sub)
            cv2.imshow(WINDOW_NAME, canvas)
            if (cv2.waitKey(16) & 0xFF) in (27, ord('q')):
                return None
            if _left_button_down():
                select_px = _cursor_pos()
                break
            if time.monotonic() - start_sec > TIMEOUT_SEC:
                print("  [중단] %.0f초 안에 못 눌렀습니다 — 조준이 안 되는 조건입니다."
                      % TIMEOUT_SEC)
                return None
        move_time_sec = time.monotonic() - start_sec

        dx, d = _projected_deviation(prev_center, target_center, select_px)
        if dx is not None and seq > WARMUP_TAPS:
            trials.append({
                "순번": seq,
                "이동거리px": d,
                "이동시간초": move_time_sec,
                "편차px": dx,
                "적중": math.dist(select_px, target_center) <= radius_px,
            })
        prev_center = target_center
    return trials


def _analyze(trials):
    """ISO 9241-411 처리량 계산 — 위 독스트링의 식 그대로."""
    if len(trials) < 3:
        return None
    deviations = [t["편차px"] for t in trials]
    times = [t["이동시간초"] for t in trials]
    nominal_a = statistics.fmean([t["이동거리px"] for t in trials])

    sd_x = statistics.stdev(deviations)          # 표본 표준편차
    effective_w = 4.133 * sd_x                   # We
    effective_a = nominal_a + statistics.fmean(deviations)   # Ae
    if effective_w <= 1e-9 or effective_a <= 0:
        return None
    id_e = math.log2(effective_a / effective_w + 1.0)        # IDe (bit)
    mean_mt = statistics.fmean(times)
    return {
        "시도수": len(trials),
        "명목거리px": nominal_a,
        "유효거리px": effective_a,
        "유효폭px": effective_w,
        "산포SDpx": sd_x,
        "IDe": id_e,
        "평균시간초": mean_mt,
        "처리량bps": id_e / mean_mt,
        "적중률": sum(1 for t in trials if t["적중"]) / len(trials),
    }


def _fitts_regression(points):
    """MT = a + b*IDe 최소제곱 회귀 -> (a, b, R^2). 점이 3개 미만이면 None."""
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx < 1e-12:
        return None   # ID가 한 점에 몰렸다 — 기울기를 못 구한다
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return intercept, slope, r2


def main():
    parser = argparse.ArgumentParser(
        description="ISO 9241-411 다방향 탭 시험으로 처리량(bit/s)을 측정한다")
    parser.add_argument("--label", default="forehead",
                        help="측정 대상 이름 (결과 파일·표에 그대로 들어간다)")
    parser.add_argument("--screen-width-cm", type=float, default=None,
                        help="모니터 화면 실제 가로 길이(cm). 안 넣으면 자동 추정")
    parser.add_argument("--out", default=None, help="시도별 CSV 저장 경로")
    args = parser.parse_args()

    screen_w_px, screen_h_px = _screen_size_px()
    if args.screen_width_cm:
        width_cm, source = args.screen_width_cm, "직접 입력"
    else:
        mm = _screen_width_mm()
        if mm is None:
            print("[중단] 화면 실제 크기를 알 수 없습니다. --screen-width-cm 으로 넣어주세요.")
            return 2
        width_cm, source = mm / 10.0, "자동 추정"
    px_per_cm = screen_w_px / width_cm

    # 커서가 닿는 영역 안에만 원을 놓는다 (트래커가 화면 일부만 쓴다).
    # 위/아래 어디에 붙었든 따라가도록 매핑 함수로 구간을 구한다
    top_px = int(_cursor_y_to_screen(0.0) * screen_h_px)
    bottom_px = int(_cursor_y_to_screen(1.0) * screen_h_px)
    usable_h_px = max(1, bottom_px - top_px)
    center_px = (screen_w_px / 2.0, usable_h_px / 2.0)
    short_side_px = min(screen_w_px, usable_h_px)

    print()
    print("=" * 62)
    print(" ISO 9241-411 다방향 탭 시험 — 처리량 측정")
    print("=" * 62)
    print(" 대상: %s" % args.label)
    print(" 화면 %d x %d, 가로 %.1fcm (%s) -> 1cm = %.0f점"
          % (screen_w_px, screen_h_px, width_cm, source, px_per_cm))
    print(" 커서 사용 영역: 세로 %d~%dpx (%dpx, 화면의 %.0f%%)"
          % (top_px, bottom_px, usable_h_px, CURSOR_Y_SPAN * 100))
    print(" 과녁 %d개, 조합 %d가지" % (TARGET_COUNT, len(CONDITIONS)))
    print()
    print(" 분홍색 과녁을 겨냥해 입을 벌려 누르세요. q/ESC로 중단.")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, screen_w_px, usable_h_px)
    cv2.moveWindow(WINDOW_NAME, 0, top_px)   # 커서가 닿는 구간에 창을 겹친다
    canvas = np.zeros((usable_h_px, screen_w_px, 3), dtype=np.uint8)
    order = _tap_order(TARGET_COUNT)

    all_rows = []
    summaries = []
    aborted = False
    try:
        for cond_index, (circle_ratio, target_cm) in enumerate(CONDITIONS, 1):
            circle_radius_px = short_side_px * circle_ratio / 2.0
            radius_px = max(6, int(px_per_cm * target_cm / 2))
            # 과녁이 화면 밖으로 나가지 않게
            circle_radius_px = min(circle_radius_px,
                                   min(center_px[0], center_px[1]) - radius_px - 10)
            if circle_radius_px <= radius_px:
                print(" [건너뜀] 조합 %d — 화면이 좁아 배치 불가" % cond_index)
                continue
            centers = _target_centers(center_px, circle_radius_px, TARGET_COUNT)

            label = "조합 %d/%d" % (cond_index, len(CONDITIONS))
            sub = ("과녁 지름 %.1fcm · 이동 거리 약 %.1fcm"
                   % (target_cm, 2 * circle_radius_px / px_per_cm))
            print(" %s  %s" % (label, sub))

            trials = _run_condition(canvas, centers, radius_px, order, label, sub)
            if trials is None:
                aborted = True
                break
            summary = _analyze(trials)
            if summary is None:
                print("   -> 유효 시도가 부족해 이 조합은 제외합니다.")
                continue
            summary["과녁cm"] = target_cm
            summary["조합"] = cond_index
            summaries.append(summary)
            for t in trials:
                row = dict(t)
                row["조합"] = cond_index
                row["과녁cm"] = target_cm
                all_rows.append(row)
            print("   -> IDe %.2f bit · 평균 %.2f초 · 처리량 %.2f bit/s · 적중 %.0f%%"
                  % (summary["IDe"], summary["평균시간초"], summary["처리량bps"],
                     summary["적중률"] * 100))
    finally:
        cv2.destroyAllWindows()

    _report(args.label, summaries, all_rows, args.out, aborted, px_per_cm)
    return 0


def _report(label, summaries, rows, out_path, aborted, px_per_cm):
    print()
    print("=" * 62)
    print(" 결과 — %s%s" % (label, " (중단됨)" if aborted else ""))
    print("=" * 62)
    if not summaries:
        print(" 완료된 조합이 없습니다.")
        return

    print()
    print(" 조합  과녁cm  거리cm   IDe    시간     처리량      적중률")
    print(" " + "-" * 56)
    for s in summaries:
        print("  %2d   %5.1f  %6.1f  %5.2f  %5.2f초  %5.2f bit/s   %3.0f%%"
              % (s["조합"], s["과녁cm"], s["유효거리px"] / px_per_cm, s["IDe"],
                 s["평균시간초"], s["처리량bps"], s["적중률"] * 100))

    # ISO 권장: 조건별 처리량의 평균을 대표값으로 쓴다
    mean_tp = statistics.fmean([s["처리량bps"] for s in summaries])
    print()
    print(" ▶ 처리량(Throughput) = %.2f bit/s" % mean_tp)
    print("   ISO 9241-411 다방향 탭 시험, TP = IDe/MT, We = 4.133 x SD(dx)")

    reg = _fitts_regression([(s["IDe"], s["평균시간초"]) for s in summaries])
    if reg:
        a, b, r2 = reg
        print()
        print(" ▶ Fitts 법칙 회귀:  MT = %.3f + %.3f x IDe   (R² = %.3f)" % (a, b, r2))
        if r2 >= 0.9:
            fit_note = "매우 잘 들어맞음 — 예측 가능한 조작 특성"
        elif r2 >= 0.7:
            fit_note = "잘 들어맞음"
        else:
            fit_note = "설명력이 낮음 — 시도 수를 늘리거나 조건을 넓혀 재측정 권장"
        print("   R² 해석: %s" % fit_note)

    print()
    print(" ▶ 비교 기준 (동일 방식으로 측정된 공개 문헌값)")
    print("   마우스      2.75 bit/s")
    print("   머리 조작   2.04 bit/s   ← 이 프로젝트와 같은 범주")
    print("   시선 조작   1.85 bit/s")
    print("   * 출처: HMD 환경 Fitts 법칙 비교 연구. 장비·과제가 달라 절대 비교는")
    print("     조심해야 하지만, 같은 표준 절차라 자릿수 비교는 유효하다.")

    if out_path is None:
        os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
        out_path = os.path.join(ROOT_DIR, "logs",
                                "throughput_iso_%s_%s.csv"
                                % (label, time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(" 시도별 기록: %s" % out_path)


if __name__ == "__main__":
    sys.exit(main())
