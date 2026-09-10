# -*- coding: utf-8 -*-
"""측정값을 그림으로 (2026-09-10 신설).

논문과 발표에 쓸 그림을 CSV와 측정 문서에서 만들어 `그림/`에 넣는다.
숫자를 손으로 옮겨 적지 않는다 — CSV가 있는 것은 CSV에서 읽고, 없는 것은
어느 문서 몇 절에서 온 값인지 이 파일 안에 출처를 적어 둔다.

    py -3 "데이터 수치/그림생성.py"

원칙
----
- **막대에 값을 적는다.** 발표장에서 축을 눈으로 재게 하지 않는다.
- **좋아진 쪽을 색으로 구분한다.** 회색=이전, 파랑=이후, 빨강=나빠진 것.
- 흑백으로 인쇄해도 구분되게 밝기를 다르게 둔다(학회 논문집은 흑백이 많다).
- 제목에 **무엇을 재는지와 단위**를 적는다.
"""
import csv
import io
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "그림")

# 색 — 흑백 인쇄를 생각해 밝기 차이를 크게 둔다
BEFORE = "#B0B0B0"      # 이전 (밝은 회색)
AFTER = "#1F4E79"       # 이후 (진한 파랑)
WORSE = "#C0504D"       # 나빠진 것 (빨강)
NEUTRAL = "#7F7F7F"

for name in ("Malgun Gothic", "NanumGothic", "Gulim"):
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False      # 마이너스가 네모로 깨지는 것 방지
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.axisbelow"] = True


def read_csv(name):
    with open(os.path.join(HERE, name), encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def save(fig, filename, caption):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    print("  %-34s %s" % (filename, caption))


def label_bars(ax, bars, fmt="%.2f", offset=0.01):
    """막대 끝에 값을 적는다 — 발표장에서 축을 눈으로 재게 하지 않는다."""
    top = max((b.get_height() for b in bars), default=0.0)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * offset,
                fmt % b.get_height(), ha="center", va="bottom", fontsize=9)


# ─────────────────────────────────────────────────────────────────────────
# 1. 기하학적 반폭 — 고정 15도가 왜 틀렸나
# ─────────────────────────────────────────────────────────────────────────
def fig_geometric_span():
    """화면비 16:9(데스크탑)와 9:16(키오스크)에서 맞는 반폭.

    기준을 **같은 패널을 가로로 달았을 때와 세로로 달았을 때**로 잡았다.
    그래야 화면비만 변수가 되고, 이 제품이 실제로 쓰이는 두 배치와도
    그대로 겹친다(--aspect desktop / kiosk).

    32인치(대각 813 mm) 하나를 돌려 단 것으로 본다. 제품 두 배치가 대개
    비슷한 크기의 패널을 쓰고, 크기를 바꾸면 화면비의 효과와 섞여 버린다.
    """
    diagonal_mm = 32 * 25.4                      # 32인치
    ratio = math.hypot(16.0, 9.0)
    long_mm = diagonal_mm * 16.0 / ratio         # 708 mm
    short_mm = diagonal_mm * 9.0 / ratio         # 398 mm
    dists = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]

    def span(size_mm, dist_mm):
        return math.degrees(math.atan((size_mm * 0.5) / dist_mm))

    # 그림에 쓰는 값을 CSV로도 남긴다 — 표로 확인할 수 있게
    rows = []
    for label, w_mm, h_mm in (("16:9 가로 (데스크탑)", long_mm, short_mm),
                              ("9:16 세로 (키오스크)", short_mm, long_mm)):
        for d in dists:
            rows.append([label, round(w_mm, 1), round(h_mm, 1), d,
                         round(span(w_mm, d), 2), round(span(h_mm, d), 2)])
    with open(os.path.join(HERE, "화면비별_기하반폭.csv"), "w",
              encoding="utf-8-sig", newline="") as fp:
        wr = csv.writer(fp)
        wr.writerow(["배치", "가로(mm)", "세로(mm)", "사용자 거리(mm)",
                     "가로 반폭(도)", "세로 반폭(도)"])
        wr.writerows(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    panels = ((ax1, "16:9 가로 — 데스크탑", long_mm, short_mm, (500, 700)),
              (ax2, "9:16 세로 — 키오스크", short_mm, long_mm, (700, 1000)))
    for ax, title, w_mm, h_mm, band in panels:
        ax.plot(dists, [span(w_mm, d) for d in dists], "-o", color=AFTER,
                markersize=4, linewidth=1.8, label="좌우 끝을 보는 각도")
        ax.plot(dists, [span(h_mm, d) for d in dists], "-s", color=NEUTRAL,
                markersize=4, linewidth=1.8, label="위아래 끝을 보는 각도")
        ax.axhline(15.0, color=WORSE, linestyle="--", linewidth=1.6)
        ax.axvspan(band[0], band[1], color="#F0F0F0", zorder=0)
        ax.text((band[0] + band[1]) / 2, 3.0, "실제로 쓰는 거리",
                ha="center", fontsize=8, color="#606060")
        ax.set_title("%s\n%.0f x %.0f mm" % (title, w_mm, h_mm), fontsize=10)
        ax.set_xlabel("사용자와 화면 사이 거리 (mm)")
        ax.legend(fontsize=9)
    ax1.set_ylabel("화면 끝을 보려면 고개를\n얼마나 돌려야 하는가 (도)")
    ax1.set_ylim(0, 45)
    for ax in (ax1, ax2):
        ax.annotate("예전에 쓰던 고정 15도", xy=(1150, 15.0),
                    xytext=(1030, 8.0), color=WORSE, fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color=WORSE, lw=1.2))
    fig.suptitle("화면 끝을 보는 고개 각도는 고를 값이 아니다\n"
                 "화면 크기와 사용자 거리가 이미 정해 놓았다 — 돌려 달면 좌우·위아래가 서로 바뀐다",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))      # 큰 제목과 소제목이 안 겹치게
    save(fig, "01_기하반폭.png", "16:9 / 9:16 배치별 맞는 반폭")


# ─────────────────────────────────────────────────────────────────────────
# 2. 기법별 전/후 — 정규화해야 진짜가 보인다
# ─────────────────────────────────────────────────────────────────────────
def fig_ablation():
    # 출처: 데이터 수치/04_기법별_전후비교.md §① (정규화 = 커서 이동폭으로 나눈 값)
    metrics = ["고개를 안 돌렸는데\n커서가 따라감",
               "고개를 돌린 만큼\n커서가 안 감",
               "좌우로 움직이는데\n위아래로 휨",
               "가만히 있는데\n커서가 흔들림"]
    before = [0.2075, 0.0950, 0.1587, 0.0403]      # 2차원 투영점
    after = [0.0468, 0.0163, 0.0371, 0.0077]       # 상대 회전
    x = range(len(metrics))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], before, w, label="예전 — 얼굴의 한 점이 화면에서 움직인 거리로",
                color=BEFORE, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([i + w / 2 for i in x], after, w, label="지금 — 고개가 돌아간 각도로",
                color=AFTER, edgecolor="black", linewidth=0.6)
    label_bars(ax, b1, "%.3f")
    label_bars(ax, b2, "%.3f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("커서가 실제로 움직인 거리로 나눈 값\n(낮을수록 좋다)")
    ax.set_ylim(0, max(before) * 1.45)
    # 배율은 막대 값과 겹치지 않게 한 줄로 맞춰 적는다
    row_y = ax.get_ylim()[1] * 0.93
    for i, (a, b) in enumerate(zip(before, after)):
        ax.text(i, row_y, "%.1f배" % (a / b), ha="center", va="top",
                fontsize=11, fontweight="bold", color=AFTER)
    ax.set_title("고개 회전으로 커서를 정하니 네 가지가 모두 4~6배 좋아졌다\n"
                 "커서가 움직인 거리로 나눠, 단순히 덜 움직여서 좋아 보이는 것을 걷어낸 값")
    # 범례가 막대 값을 가리지 않게 그림 밖(아래)에 둔다
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "02_상대회전_전후.png", "상대 회전 vs 2차원 투영 (정규화)")


def fig_gain_trap():
    """정규화 안 하면 왜 속는가 — 같은 기법이 좋아 보였다가 무효가 된다."""
    # 출처: 04_기법별_전후비교.md §⑤ 화면 기하 반폭
    raw_before = [1.0, 1.0]          # 고정 15도를 100%로 둔 상대값
    raw_after = [0.71, 0.71]         # 원값만 보면 29% 좋아 보인다
    norm_before = [1.0, 1.0]
    norm_after = [1.0, 1.0]          # 정규화하면 완전히 같다
    labels = ["고개를 돌린 만큼\n커서가 안 감", "가만히 있는데\n커서가 흔들림"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, before, after, title in (
            (ax1, raw_before, raw_after, "잰 값 그대로 보면\n「29% 좋아졌다」"),
            (ax2, norm_before, norm_after, "커서가 움직인 거리로 나누면\n「달라진 것 없음」")):
        x = range(len(labels))
        w = 0.36
        ax.bar([i - w / 2 for i in x], before, w, color=BEFORE,
               edgecolor="black", linewidth=0.6, label="예전 — 고정 15도")
        ax.bar([i + w / 2 for i in x], after, w, color=AFTER,
               edgecolor="black", linewidth=0.6, label="지금 — 화면 크기로 계산")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10)
    ax1.set_ylabel("고정 15도 = 1.0")
    ax1.legend(fontsize=8)
    fig.suptitle("커서를 덜 움직이게만 해도 모든 값이 좋아 보인다 — 그래서 나눠서 본다",
                 fontsize=11)
    save(fig, "03_정규화의_필요.png", "정규화 전/후로 결론이 뒤집히는 예")


# ─────────────────────────────────────────────────────────────────────────
# 3. 렌즈 자가 보정
# ─────────────────────────────────────────────────────────────────────────
def fig_lens():
    """렌즈 자가 보정 — auto_arc를 끈 조건(⑪)만 쓴다.

    ⑩은 auto_arc가 함께 켜져 있어 세로 휨에 프로토콜 인공물이 섞였다
    (05_렌즈보정_전후비교.md §2). 기여를 분리한 ⑪이 논문에 쓸 값이다.
    """
    rows = [r for r in read_csv("렌즈보정_전후비교.csv")
            if str(r["조건"]).startswith("⑪")]
    if not rows:
        print("  [건너뜀] 렌즈보정 — ⑪ 조건 행이 없음")
        return
    lenses, off, on = [], [], []
    for r in rows:
        lens, state = [t.strip() for t in r["설정"].split("/")]
        val = float(r["몸평행이동 끌림(%)"])
        if lens not in lenses:
            lenses.append(lens); off.append(None); on.append(None)
        i = lenses.index(lens)
        if "켬" in state:
            on[i] = val
        else:
            off[i] = val
    x = range(len(lenses)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], off, w, label="보정 없음",
                color=BEFORE, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([i + w / 2 for i in x], on, w, label="렌즈 자가 보정",
                color=AFTER, edgecolor="black", linewidth=0.6)
    label_bars(ax, b1, "%.2f%%"); label_bars(ax, b2, "%.2f%%")
    for i, (a, b) in enumerate(zip(off, on)):
        ax.text(i, max(a, b) * 1.16, "-%.0f%%" % ((a - b) / a * 100),
                ha="center", fontsize=10, fontweight="bold", color=AFTER)
    ax.set_xticks(list(x)); ax.set_xticklabels(lenses)
    ax.set_ylabel("고개를 안 돌렸는데 커서가\n따라간 거리 (화면 폭의 %)")
    ax.set_ylim(0, max(off) * 1.3)
    ax.set_title("사용자 얼굴만으로 렌즈를 보정한다 — 보정판이 필요 없다\n"
                 "커서가 움직인 거리는 87.6→87.3%로 그대로다 — 덜 움직여서 좋아진 게 아니다")
    ax.legend(fontsize=9)
    save(fig, "04_렌즈보정.png", "렌즈별 끌림 감소 (auto_arc 끈 조건)")


# ─────────────────────────────────────────────────────────────────────────
# 4. 거리 추정 — 절대(f) vs 상대(크기비)
# ─────────────────────────────────────────────────────────────────────────
def fig_distance():
    # 출처: 데이터 수치/02_시뮬레이션_요약.md — 절대 15.7%(중앙값)/69.3%(최악),
    #       상대 0.7%
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = ["거리를 직접 계산\n(렌즈 초점거리 사용)",
              "거리 변화만 따라감\n(얼굴 크기 비율 사용)"]
    median = [15.7, 0.7]
    worst = [69.3, 0.7]
    x = range(len(labels)); w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], median, w, label="보통일 때",
                color=BEFORE, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([i + w / 2 for i in x], worst, w, label="가장 나쁠 때",
                color=WORSE, edgecolor="black", linewidth=0.6)
    label_bars(ax, b1, "%.1f%%"); label_bars(ax, b2, "%.1f%%")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("실제 거리와 추정 거리의 차이 (%)")
    ax.set_title("거리를 직접 계산하면 얼굴 생김새에 따라 크게 틀린다\n"
                 "얼굴 6종으로 확인. 크기 비율만 쓰면 모르는 값 두 개가 서로 지워져 정확해진다")
    ax.legend(fontsize=9)
    save(fig, "05_거리추정.png", "절대 vs 상대 거리 추정 오차")


# ─────────────────────────────────────────────────────────────────────────
# 5. U자 휨 — 렌즈 왜곡이 원인
# ─────────────────────────────────────────────────────────────────────────
def fig_bow():
    # 출처: 데이터 수치/06_떨림과_U자휨.md
    lenses = ["왜곡없음", "일반 65도", "광각 90도", "초광각 120도"]
    depth = [0.0, 7.0, 10.3, 14.5]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = [AFTER] + [NEUTRAL] * 3
    bars = ax.bar(lenses, depth, color=colors, edgecolor="black", linewidth=0.6)
    label_bars(ax, bars, "%.1f px")
    ax.set_ylabel("좌우로만 움직였을 때 커서가\n위아래로 휜 깊이 (px)")
    ax.set_ylim(0, max(depth) * 1.25)
    ax.set_title("좌우로만 움직였는데 커서가 위아래로 휜다 — 원인은 렌즈 왜곡\n"
                 "왜곡이 없는 렌즈에서는 0.0 px. 얼굴 생김새를 바꿔도 6.9~7.6 px로 같다")
    save(fig, "06_U자휨_렌즈.png", "렌즈별 U자 깊이")


def fig_rejected():
    """재 보고 버린 것들 — 음성 결과."""
    # 출처: 06_떨림과_U자휨.md
    labels = ["지금 방식\n(고개가 돌아간 양)", "버린 방법\n(코가 향한 방향)"]
    v65 = [4.8, 9.3]
    v120 = [11.3, 16.0]
    x = range(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    b1 = ax.bar([i - w / 2 for i in x], v65, w, label="일반 65도",
                color=AFTER, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([i + w / 2 for i in x], v120, w, label="초광각 120도",
                color=BEFORE, edgecolor="black", linewidth=0.6)
    label_bars(ax, b1, "%.1f"); label_bars(ax, b2, "%.1f")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("커서가 위아래로 휜 깊이 (px)")
    ax.set_ylim(0, max(v120) * 1.25)
    ax.set_title("재 보고 버린 방법 — 「코가 향한 방향」을 쓰면 두 배 나빠진다\n"
                 "방향을 직접 쓰는 쪽이 렌즈 왜곡에 더 약하기 때문")
    ax.legend(fontsize=9)
    save(fig, "07_버린대안_방향분해.png", "방향 분해 대안의 음성 결과")


# ─────────────────────────────────────────────────────────────────────────
# 6. 1유로 필터 — 한 조건만 보면 반대로 고른다
# ─────────────────────────────────────────────────────────────────────────
def fig_one_euro():
    # 출처: 데이터 수치/06_떨림과_U자휨.md (떨림 / 지연, px)
    conds = ["정면", "오른쪽\n25도", "광각\n90도", "멀리\n1100mm"]
    now_j = [1.29, 1.21, 2.16, 2.04]
    cand_j = [0.89, 1.01, 1.74, 1.69]
    now_l = [75.8, 49.7, 54.3, 53.4]
    cand_l = [59.4, 64.2, 67.8, 67.5]
    x = range(len(conds)); w = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    b1 = ax1.bar([i - w / 2 for i in x], now_j, w, label="예전 설정",
                 color=BEFORE, edgecolor="black", linewidth=0.6)
    b2 = ax1.bar([i + w / 2 for i in x], cand_j, w, label="새 설정 후보",
                 color=AFTER, edgecolor="black", linewidth=0.6)
    label_bars(ax1, b1, "%.2f"); label_bars(ax1, b2, "%.2f")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(conds, fontsize=9)
    ax1.set_ylabel("가만히 있을 때 커서가\n흔들리는 거리 (px)")
    ax1.set_title("흔들림 — 새 설정이 모든 조건에서 낫다", fontsize=10)
    ax1.legend(fontsize=8)

    c1 = ax2.bar([i - w / 2 for i in x], now_l, w, color=BEFORE,
                 edgecolor="black", linewidth=0.6)
    c2 = ax2.bar([i + w / 2 for i in x], cand_l, w, color=AFTER,
                 edgecolor="black", linewidth=0.6)
    for i in (1, 2, 3):                       # 뒤집힌 조건을 빨갛게
        c2[i].set_color(WORSE); c2[i].set_edgecolor("black")
    label_bars(ax2, c1, "%.0f"); label_bars(ax2, c2, "%.0f")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(conds, fontsize=9)
    ax2.set_ylabel("빠르게 움직일 때 커서가\n뒤처지는 거리 (px)")
    ax2.set_title("뒤처짐 — 정면 말고는 전부 나빠진다 (빨강)", fontsize=10)
    fig.suptitle("카메라 정면에서만 재 보면 잘못된 설정을 고르게 된다", fontsize=12)
    save(fig, "08_1유로_조건별.png", "1유로 필터 조건별 역전")


def fig_one_euro_final():
    # 출처: 06_떨림과_U자휨.md — 최악 조건 기준 상대값
    labels = ["가만히 있을 때\n커서 흔들림", "빠르게 움직일 때\n커서 뒤처짐"]
    before = [100.0, 100.0]
    after = [69.0, 103.0]
    x = range(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    b1 = ax.bar([i - w / 2 for i in x], before, w, label="예전 설정",
                color=BEFORE, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([i + w / 2 for i in x], after, w, label="지금 설정",
                color=[AFTER, WORSE], edgecolor="black", linewidth=0.6)
    label_bars(ax, b1, "%.0f%%"); label_bars(ax, b2, "%.0f%%")
    ax.axhline(100, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("이전 설정을 100%로 둔 상대값\n(가장 나쁜 조건 기준)")
    ax.set_ylim(0, 125)
    ax.set_title("고른 설정 — 흔들림 31% 줄고, 뒤처짐 3% 늘었다\n"
                 "다섯 조건을 모두 이기는 설정이 없어서, 한쪽만 좋게 하고 다른 쪽은 조금 내줬다")
    ax.legend(fontsize=9)
    save(fig, "09_1유로_채택.png", "1유로 최종 선택의 맞바꿈")


# ─────────────────────────────────────────────────────────────────────────
# 7. 카메라 위치와 거리
# ─────────────────────────────────────────────────────────────────────────
def fig_camera_place():
    # 출처: 데이터 수치/07_카메라_위치와_거리.md §1
    places = ["중앙\n아래", "중앙\n위", "왼쪽\n280mm", "오른쪽\n280mm",
              "모서리", "왼쪽\n450mm"]
    mean = [2.6, 4.5, 8.4, 8.4, 12.4, 20.8]
    worst = [5.1, 7.2, 16.4, 16.9, 23.8, 32.7]
    x = range(len(places)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], mean, w, label="평균",
                color=AFTER, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([i + w / 2 for i in x], worst, w, label="최악",
                color=BEFORE, edgecolor="black", linewidth=0.6)
    label_bars(ax, b1, "%.1f"); label_bars(ax, b2, "%.1f")
    ax.set_xticks(list(x)); ax.set_xticklabels(places, fontsize=9)
    ax.set_ylabel("얼굴이 향한 곳과 커서 사이 거리\n(화면 대각선의 %)")
    ax.set_ylim(0, max(worst) * 1.2)
    ax.set_title("카메라가 화면 중앙에서 옆으로 멀어질수록 커서가 부정확해진다\n"
                 "왼쪽 8.4, 오른쪽 8.4로 같다 — 어느 쪽이냐가 아니라 얼마나 멀어졌느냐의 문제")
    ax.legend(fontsize=9)
    save(fig, "10_카메라위치.png", "카메라 위치별 겨냥 오차")


def fig_mouth_cliff():
    """입을 완전히 안 다물면 버튼이 눌린 채 남던 절벽."""
    # 출처: 개발일지 2026-09-10 §2
    residual = [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.15, 0.20]
    fixed_ok = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    before_ok = [1, 1, 1, 1, 1, 0, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.step(residual, before_ok, where="mid", color=WORSE, linewidth=2.0,
            label="고치기 전 — 0.11부터 마우스 버튼이 눌린 채 안 풀림")
    ax.step(residual, fixed_ok, where="mid", color=AFTER, linewidth=2.0,
            linestyle="--", label="고친 뒤 — 전부 정상으로 클릭됨")
    ax.axvline(0.105, color="black", linestyle=":", linewidth=1.0)
    ax.text(0.107, 0.5, "절벽", fontsize=10)
    ax.set_ylim(-0.15, 1.35)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["실패", "클릭 성공"])
    ax.set_xlabel("클릭한 뒤 입에 남아 있는 벌림 정도\n(사람은 매번 입을 완전히 다물지 않는다)")
    ax.set_title("입을 완전히 안 다물면 클릭이 아예 안 되던 문제\n"
                 "「원래 얼마나 벌렸었나」와 견주도록 고쳐서 없앴다")
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "11_입다묾_절벽.png", "잔여 턱 벌림에 따른 클릭 성공")


def fig_click_freeze():
    """되짚기가 곧 드래그였다 — 일대일 맞바꿈."""
    # 출처: 개발일지 2026-09-10 §4
    labels = ["누른 그 자리에 고정\n(지금)", "0.04초 전 자리로",
              "0.08초 전 자리로", "고정 안 함"]
    aim = [18.5, 0.0, 14.0, 144.5]        # 겨냥 오차 (범위 중앙값)
    drag = [0.0, 18.5, 31.0, 152.5]       # 눌린 채 끌린 거리
    x = range(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], aim, w, label="겨눈 곳에서 벗어난 거리",
                color=NEUTRAL, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([i + w / 2 for i in x], drag, w, label="의도하지 않은 드래그 거리",
                color=WORSE, edgecolor="black", linewidth=0.6)
    b2[0].set_color(AFTER)
    label_bars(ax, b1, "%.0f"); label_bars(ax, b2, "%.0f")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("픽셀 (1920×1080)")
    ax.set_title("커서를 이전 자리로 되돌리면 그만큼 그대로 드래그가 된다\n"
                 "되돌리는 일이 마우스 버튼이 눌린 뒤에 일어나기 때문")
    ax.legend(fontsize=9)
    save(fig, "12_클릭중_커서붙잡기.png", "되짚기 길이별 정확도와 드래그")


def main():
    print("그림을 만듭니다 -> %s" % OUT)
    print()
    for fn in (fig_geometric_span, fig_ablation, fig_gain_trap, fig_lens,
               fig_distance, fig_bow, fig_rejected, fig_one_euro,
               fig_one_euro_final, fig_camera_place, fig_mouth_cliff,
               fig_click_freeze):
        try:
            fn()
        except Exception as exc:               # noqa: 한 장이 실패해도 나머지는 만든다
            print("  [실패] %s -> %s" % (fn.__name__, exc))
    print()
    made = sorted(f for f in os.listdir(OUT)) if os.path.isdir(OUT) else []
    print("%d장 생성" % len(made))


if __name__ == "__main__":
    main()
