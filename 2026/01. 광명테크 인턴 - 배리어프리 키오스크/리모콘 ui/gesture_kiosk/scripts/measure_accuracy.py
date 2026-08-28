"""정확도 측정 — "마우스를 대체할 만큼 정확한가"에 숫자로 답한다 (2026-08-26 신설).

왜 필요한가
-----------
최종보고서의 핵심 주장이 "미간 방식이 마우스를 대체할 만큼 정확하다"인데,
지금 그 근거는 시연 영상뿐이다. 영상은 "되네"는 보여주지만 "얼마나"는 못
보여준다. 숫자 세 개면 충분하다.

    성공률              쓸 수 있는 물건인가
    목표당 걸린 시간      답답하지 않은가
    눌리는 최소 버튼 크기  ★키오스크 화면을 어떻게 설계해야 하는가

세 번째가 특히 중요하다. 앞의 둘은 성능 이야기지만, 이건 광명테크가 실제로
화면을 만들 때 바로 쓰는 설계 수치다.

어떻게 쓰나
-----------
1) 평소처럼 트래커를 켠다 (head.py 또는 eyebrow.py)
2) 이 프로그램을 따로 켠다
3) 화면에 뜨는 동그라미를 고개로 겨냥해 입을 벌려 누른다
4) 끝나면 결과가 보고서에 넣을 형태로 정리돼 나온다

트래커 오버레이는 투명해서 이 창의 과녁이 그대로 비쳐 보인다. 그래서 이
프로그램은 창을 맨 앞으로 올리지 않는다 — 올리면 초록 커서 점을 가려버린다.

측정 방식
---------
과녁 크기를 여러 단계로 바꿔가며 각 크기마다 여러 번 시도한다. 위치는 매번
다르되 순서는 고정한다(무작위 씨앗 고정) — 코 방식과 미간 방식이 **완전히
같은 문제**를 풀어야 비교가 성립하기 때문이다.

커서가 화면 위쪽 절반만 쓰므로(트래커의 설계) 과녁도 그 안에만 놓는다.
"""
import argparse
import csv
import ctypes
import math
import os
import random
import sys
import time
from ctypes import wintypes

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from head import _cursor_y_to_screen, put_korean_text, _korean_text_width_px  # noqa: E402

WINDOW_NAME = "accuracy target"

# 재볼 과녁 지름(cm) — 큰 것부터. 키오스크 버튼으로 흔히 쓰는 크기 언저리를 훑는다
TARGET_SIZES_CM = (4.0, 3.0, 2.0, 1.5, 1.0)
TRIALS_PER_SIZE = 5            # 크기마다 몇 번 시도할지
TIMEOUT_SEC = 20.0             # 이 시간을 넘기면 실패로 센다
SETTLE_BETWEEN_SEC = 0.8       # 과녁이 바뀐 뒤 잠깐 — 직전 클릭이 다음 시도로 새지 않게
RANDOM_SEED = 20260903         # 두 방식에 같은 문제를 내기 위해 고정

BG_COLOR = (250, 249, 247)
TARGET_COLOR = (124, 184, 0)
TARGET_RING = (97, 39, 30)
TEXT_COLOR = (144, 117, 107)
TITLE_COLOR = (97, 39, 30)
HIT_COLOR = (124, 184, 0)
MISS_COLOR = (60, 60, 220)


def _screen_size_px():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _screen_width_mm():
    """모니터의 실제 가로 길이(mm). 드라이버가 엉뚱한 값을 주는 일이 흔해서
    --screen-width-cm 으로 직접 넣을 수 있게 해두고, 여기서는 참고값만 준다."""
    try:
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        hdc = user32.GetDC(0)
        if not hdc:
            return None
        try:
            mm = gdi32.GetDeviceCaps(hdc, 4)   # HORZSIZE
        finally:
            user32.ReleaseDC(0, hdc)
        return mm if mm and mm > 50 else None
    except Exception:   # noqa: 방어적 — 참고값일 뿐이라 실패해도 진행
        return None


def _cursor_pos():
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def _left_button_down():
    """지금 왼쪽 버튼이 눌려 있는가. 트래커가 프로그램으로 내는 클릭도 잡힌다."""
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)


def _make_trials():
    """(지름cm, x비율, y비율) 목록. 씨앗을 고정해 두 방식에 같은 문제를 낸다."""
    rng = random.Random(RANDOM_SEED)
    trials = []
    for size_cm in TARGET_SIZES_CM:
        for _ in range(TRIALS_PER_SIZE):
            # 가장자리에 너무 붙지 않게 안쪽으로 여유를 둔다
            trials.append((size_cm, rng.uniform(0.10, 0.90), rng.uniform(0.12, 0.88)))
    return trials


def _draw(canvas, target_px, radius_px, headline, subline, flash=None):
    canvas[:] = BG_COLOR
    if flash is not None:
        cv2.circle(canvas, target_px, radius_px + 14, flash, 4, cv2.LINE_AA)
    cv2.circle(canvas, target_px, radius_px, TARGET_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, target_px, radius_px, TARGET_RING, 2, cv2.LINE_AA)
    # 한가운데 점 — 어디를 겨냥해야 하는지 분명하게
    cv2.circle(canvas, target_px, max(2, radius_px // 12), (255, 255, 255), -1, cv2.LINE_AA)
    w_px = canvas.shape[1]
    for text, y_px, size_px, color in ((headline, 60, 34, TITLE_COLOR),
                                       (subline, 104, 20, TEXT_COLOR)):
        text_w = _korean_text_width_px(text, size_px)
        put_korean_text(canvas, text, (int(w_px / 2 - text_w / 2), y_px), size_px, color)


def main():
    parser = argparse.ArgumentParser(
        description="헤드트래커 정확도 측정 — 성공률·걸린 시간·눌리는 최소 버튼 크기")
    parser.add_argument("--label", default="미간 방식",
                        help="결과에 적을 이름 (예: 코 방식 / 미간 방식)")
    parser.add_argument("--screen-width-cm", type=float, default=None,
                        help="모니터 화면의 실제 가로 길이(cm). 안 넣으면 자동 추정하는데, "
                             "드라이버가 틀린 값을 주는 일이 흔하니 자로 재서 넣는 게 정확하다")
    parser.add_argument("--out", default=None, help="결과를 저장할 csv 경로")
    args = parser.parse_args()

    screen_w_px, screen_h_px = _screen_size_px()
    if args.screen_width_cm:
        width_cm, source = args.screen_width_cm, "직접 입력"
    else:
        mm = _screen_width_mm()
        if mm is None:
            print("[중단] 화면 실제 크기를 알 수 없습니다. --screen-width-cm 으로 넣어주세요.")
            print("       모니터 화면의 가로 길이를 자로 재면 됩니다. 예: --screen-width-cm 34.5")
            return 2
        width_cm, source = mm / 10.0, "자동 추정"
    px_per_cm = screen_w_px / width_cm

    print("=" * 62)
    print(" 헤드트래커 정확도 측정 - %s" % args.label)
    print("=" * 62)
    print(" 화면 %d x %d, 가로 %.1fcm (%s) -> 1cm = %.0f점" %
          (screen_w_px, screen_h_px, width_cm, source, px_per_cm))
    print(" 과녁 %d개 (크기 %d단계 x %d번)" %
          (len(TARGET_SIZES_CM) * TRIALS_PER_SIZE, len(TARGET_SIZES_CM), TRIALS_PER_SIZE))
    print()
    print(" 트래커(head.py 또는 eyebrow.py)를 먼저 켜두세요.")
    print(" 동그라미를 고개로 겨냥해 입을 벌려 누르면 됩니다. ESC = 중단")
    print("=" * 62)

    canvas = np.empty((screen_h_px, screen_w_px, 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # ★맨 앞으로 올리지 않는다 — 트래커 오버레이(맨 앞·투명)가 위에 있어야
    # 초록 커서 점이 보인다. 이 창은 그 아래에서 과녁만 비춰주면 된다

    trials = _make_trials()
    results = []
    aborted = False
    try:
        for index, (size_cm, x_ratio, y_ratio) in enumerate(trials, 1):
            radius_px = max(6, int(px_per_cm * size_cm / 2))
            # 커서는 화면 일부 구간만 쓴다 — 과녁도 그 안에 놓는다.
            # ★캔버스가 화면 전체 크기이고 창도 전체화면이라 캔버스 y = 화면 y다.
            # 그래서 구간 시작점(usable_top_px)을 반드시 더해야 한다 — 안 더하면
            # 커서가 닿지 않는 자리에 과녁이 그려져 측정 자체가 불가능해진다
            # (2026-08-27 커서 구간을 화면 하단으로 옮기며 함께 고침)
            usable_top_px = int(_cursor_y_to_screen(0.0) * screen_h_px)
            usable_bottom_px = int(_cursor_y_to_screen(1.0) * screen_h_px)
            usable_h_px = usable_bottom_px - usable_top_px
            target_px = (int(x_ratio * screen_w_px),
                         usable_top_px + int(y_ratio * usable_h_px))
            target_px = (min(max(target_px[0], radius_px + 4), screen_w_px - radius_px - 4),
                         min(max(target_px[1], usable_top_px + radius_px + 4),
                             usable_bottom_px - radius_px - 4))

            headline = "%d / %d" % (index, len(trials))
            subline = "동그라미를 보고 입을 벌려 누르세요  (지름 %.1fcm)" % size_cm
            _draw(canvas, target_px, radius_px, headline, subline)
            cv2.imshow(WINDOW_NAME, canvas)

            # 직전 클릭이 새어들지 않게 버튼이 떨어질 때까지 기다린다
            settle_end = time.monotonic() + SETTLE_BETWEEN_SEC
            while time.monotonic() < settle_end or _left_button_down():
                if (cv2.waitKey(16) & 0xFF) == 27:
                    aborted = True
                    break
            if aborted:
                break

            start_sec = time.monotonic()
            hit, elapsed, click_px = False, TIMEOUT_SEC, None
            while True:
                if (cv2.waitKey(16) & 0xFF) == 27:
                    aborted = True
                    break
                if _left_button_down():
                    click_px = _cursor_pos()
                    elapsed = time.monotonic() - start_sec
                    hit = math.dist(click_px, target_px) <= radius_px
                    break
                if time.monotonic() - start_sec > TIMEOUT_SEC:
                    break
            if aborted:
                break

            results.append({"순번": index, "지름cm": size_cm, "성공": hit,
                            "걸린시간초": round(elapsed, 2),
                            "빗나간거리점": (round(math.dist(click_px, target_px), 1)
                                             if click_px else ""),
                            "과녁x": target_px[0], "과녁y": target_px[1]})
            _draw(canvas, target_px, radius_px, headline, subline,
                  flash=HIT_COLOR if hit else MISS_COLOR)
            cv2.imshow(WINDOW_NAME, canvas)
            cv2.waitKey(220)
            print("  %2d/%d  지름 %.1fcm  %s  %.1f초" %
                  (index, len(trials), size_cm, "성공" if hit else "실패", elapsed))
    finally:
        cv2.destroyAllWindows()

    if not results:
        print()
        print("측정된 것이 없습니다.")
        return 1
    _report(args.label, results, args.out, aborted)
    return 0


def _report(label, results, out_path, aborted):
    print()
    print("=" * 62)
    print(" 결과 - %s%s" % (label, "  (중간에 멈춤)" if aborted else ""))
    print("=" * 62)
    print(" %-10s %-8s %-12s %s" % ("과녁 지름", "성공률", "평균 걸린 시간", "시도"))
    print(" " + "-" * 52)
    by_size = {}
    for row in results:
        by_size.setdefault(row["지름cm"], []).append(row)
    smallest_ok = None
    for size_cm in sorted(by_size, reverse=True):
        rows = by_size[size_cm]
        hits = [r for r in rows if r["성공"]]
        rate = 100.0 * len(hits) / len(rows)
        mean_text = ("%.1f 초" % (sum(r["걸린시간초"] for r in hits) / len(hits))
                     if hits else "-")
        print(" %6.1f cm  %5.0f %%  %8s    %d번" % (size_cm, rate, mean_text, len(rows)))
        # "눌리는 최소 크기" = 모두 성공한 가장 작은 크기
        if rate >= 100.0:
            smallest_ok = size_cm

    # ★대표 성공률은 "권장 크기 이상"에서만 낸다.
    # 전체를 뭉뚱그리면 아무도 못 누르는 작은 과녁까지 섞여, **어떤 크기를
    # 시험했느냐에 따라 숫자가 달라진다** — 보고서에 넣기엔 의미가 없다.
    # 실제 운영 조건은 "권장 크기 이상의 버튼을 쓴다"이므로 그 조건에서 잰다.
    usable = [r for r in results if smallest_ok and r["지름cm"] >= smallest_ok]
    scope = usable if usable else results
    hits = [r for r in scope if r["성공"]]
    overall_rate = 100.0 * len(hits) / len(scope)
    overall_sec = (sum(r["걸린시간초"] for r in hits) / len(hits)) if hits else float("nan")

    print()
    print(" 보고서 표에 그대로 넣을 값  (지름 %s 이상 %d번 기준)"
          % (("%.1fcm" % smallest_ok) if smallest_ok else "전체", len(scope)))
    print(" " + "-" * 52)
    print("   %-20s %s" % ("구분", label))
    print("   %-20s %.0f %%" % ("성공률", overall_rate))
    print("   %-20s %.1f 초" % ("목표당 걸린 시간", overall_sec))
    print("   %-20s %s" % ("눌리는 최소 버튼 크기",
                           ("%.1f cm" % smallest_ok) if smallest_ok
                           else "모든 크기에서 실패 있음"))
    print()
    if smallest_ok:
        print(" -> 키오스크 버튼은 지름 %.1fcm 이상으로 만들면 됩니다." % smallest_ok)
    else:
        print(" -> 가장 큰 과녁에서도 놓친 적이 있습니다. 조명·거리·자세를 확인하세요.")

    if out_path is None:
        stamp = time.strftime("%Y%m%d_%H%M")
        out_path = os.path.join(ROOT_DIR, "logs",
                                "accuracy_%s_%s.csv" % (label.replace(" ", ""), stamp))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(" 시도별 기록: %s" % out_path)


if __name__ == "__main__":
    sys.exit(main())
