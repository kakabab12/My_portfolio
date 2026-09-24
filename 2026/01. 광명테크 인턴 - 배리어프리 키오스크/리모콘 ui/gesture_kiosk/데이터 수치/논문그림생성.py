# -*- coding: utf-8 -*-
"""논문 2단 조판용 그림 (2026-09-10 신설).

    py -3 "데이터 수치/논문그림생성.py"

왜 따로 만드나
--------------
`그림생성.py`가 만드는 그림은 **화면에서 보고 발표에 쓰는 것**이라 가로가
넓고 제목이 두 줄이다. 그걸 논문 2단(단 폭 8 cm)에 넣으면 글자가 겹쳐
읽을 수 없다 — 실제로 겹쳤다.

논문용은 규칙이 다르다.

- **그림 안에 제목을 넣지 않는다.** 설명은 그림 아래 캡션이 한다.
  같은 말을 두 번 쓰면 자리만 먹는다.
- **세로로 쌓는다.** 단 폭이 좁으니 가로로 늘어놓으면 글자가 뭉갠다.
- 글자를 키운다(축 8pt, 값 8pt). 인쇄하면 화면보다 작아 보인다.
- 흑백 인쇄를 생각해 밝기 차이를 크게 둔다.

논문이 쓰는 세 장만 만든다. 출력은 `그림/논문/`.
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
OUT = os.path.join(HERE, "그림", "논문")

BEFORE = "#B0B0B0"
AFTER = "#1F4E79"
WORSE = "#C0504D"

for name in ("Malgun Gothic", "NanumGothic", "Gulim"):
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200          # 인쇄용 — 화면용보다 높다
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.size"] = 8

# 단 폭 8 cm = 3.15 인치. 조금 크게 그려 넣을 때 줄인다
W_IN = 3.4


def save(fig, name, note):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("  %-30s %s" % (name, note))


def label_bars(ax, bars, fmt="%.2f"):
    top = max((b.get_height() for b in bars), default=0.0)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.02,
                fmt % b.get_height(), ha="center", va="bottom", fontsize=7)


def fig_normalization():
    """정규화 전후 — 세로로 쌓는다(가로로 놓으면 단 폭에서 뭉갠다)."""
    labels = ["고개를 돌린 만큼\n커서가 안 감", "가만히 있는데\n커서가 흔들림"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_IN, 2.2), sharex=True)
    for ax, after, sub in ((ax1, [0.71, 0.71], "측정한 값 그대로 — 29% 개선"),
                           (ax2, [1.0, 1.0], "커서가 움직인 거리로 나누면 — 그대로")):
        x = range(len(labels)); w = 0.34
        b1 = ax.bar([i - w / 2 for i in x], [1.0, 1.0], w, color=BEFORE,
                    edgecolor="black", linewidth=0.5, label="예전")
        b2 = ax.bar([i + w / 2 for i in x], after, w, color=AFTER,
                    edgecolor="black", linewidth=0.5, label="지금")
        label_bars(ax, b1, "%.2f"); label_bars(ax, b2, "%.2f")
        ax.set_ylim(0, 1.35)
        ax.set_ylabel("예전 = 1.0", fontsize=7)
        ax.set_title(sub, fontsize=8)
    ax2.set_xticks(list(range(len(labels))))
    ax2.set_xticklabels(labels, fontsize=7)
    # 범례를 축 안에 두면 막대 위 값(1.00)을 가린다 — 그림 아래로 뺀다
    handles, names = ax1.get_legend_handles_labels()
    fig.legend(handles, names, fontsize=6.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(h_pad=0.8, rect=(0, 0.06, 1, 1))
    save(fig, "논문_그림1_정규화.png", "정규화 전후 (세로 2단)")


def fig_relative_rotation():
    # 한 칸이 2 cm뿐이다. 한 줄을 6자 안으로 끊지 않으면 옆 라벨과 맞닿아
    # "…돌렸는데고개 돌린…"으로 붙어 읽힌다.
    labels = ["고개 안\n돌렸는데\n커서 따라감", "고개 돌린\n만큼\n안 감",
              "좌우인데\n위아래로\n휨", "가만히\n있는데\n흔들림"]
    before = [0.2075, 0.0950, 0.1587, 0.0403]
    after = [0.0468, 0.0163, 0.0371, 0.0077]
    x = range(len(labels)); w = 0.34
    fig, ax = plt.subplots(figsize=(W_IN, 2.5))
    b1 = ax.bar([i - w / 2 for i in x], before, w, color=BEFORE,
                edgecolor="black", linewidth=0.5, label="예전 방식")
    b2 = ax.bar([i + w / 2 for i in x], after, w, color=AFTER,
                edgecolor="black", linewidth=0.5, label="지금 방식")
    ax.set_ylim(0, max(before) * 1.45)
    row_y = ax.get_ylim()[1] * 0.95
    for i, (a, b) in enumerate(zip(before, after)):
        ax.text(i, row_y, "%.1f배" % (a / b), ha="center", va="top",
                fontsize=8, fontweight="bold", color=AFTER)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("커서가 움직인 거리로\n나눈 값 (낮을수록 좋다)", fontsize=7)
    ax.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    save(fig, "논문_그림2_상대회전.png", "설계 결정의 개선 예")


def fig_mouth_cliff():
    """클릭 뒤 남은 벌림에 따른 클릭 성공 — mouth_cliff.py가 실제 판정 코드로 낸 값.

    예전에는 규칙을 손으로 옮긴 도식이었고 테스트 범위 밖(0.20)까지 그렸다.
    이제 벌림 임계(0.17) 아래에서 측정한 점을 그대로 찍는다.
    """
    with open(os.path.join(HERE, "입다묾_절벽.csv"), encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    residual = [float(r["잔여 벌림"]) for r in rows]
    before = [int(r["고치기 전 성공"]) for r in rows]
    after = [int(r["고친 뒤 성공"]) for r in rows]
    last_ok = max(x for x, ok in zip(residual, before) if ok)
    first_bad = min(x for x, ok in zip(residual, before) if not ok)
    cliff = (last_ok + first_bad) / 2
    fig, ax = plt.subplots(figsize=(W_IN, 1.5))
    ax.step(residual, before, where="mid", color=WORSE, linewidth=1.6,
            label="고치기 전")
    ax.step(residual, after, where="mid", color=AFTER, linewidth=1.6,
            linestyle="--", label="고친 뒤")
    ax.plot(residual, before, "o", color=WORSE, markersize=2.2)
    ax.plot(residual, after, "o", color=AFTER, markersize=2.2)
    ax.axvline(cliff, color="black", linestyle=":", linewidth=0.9)
    ax.text(cliff + 0.002, 0.45, "절벽", fontsize=7.5)
    ax.set_xlim(residual[0] - 0.004, residual[-1] + 0.004)
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["실패", "성공"], fontsize=7)
    ax.set_xlabel("클릭한 뒤 입에 남은 벌림", fontsize=7)
    # 왼쪽 아래는 "실패" 선이 지나간다 — 가운데 오른쪽 빈 곳에 둔다
    ax.legend(fontsize=6.5, loc="center right", framealpha=0.95)
    fig.tight_layout()
    save(fig, "논문_그림3_입다묾절벽.png", "다묾 판정의 절벽 (측정값)")


def fig_t_gate():
    """t값 판정이 없는 곡률을 만든다 — arc_t_gate.py가 낸 곡률_t값판정_시드별.csv.

    위: 시드마다 추정한 곡률 계수와, 사용자가 의도한 세로 움직임과 가로²의 상관.
    점이 1·3사분면에만 있으면 계수가 그 우연한 상관을 따라간다는 뜻이다.
    회색 띠는 좌우로만 움직여 구한 참 곡률의 범위다.
    아래: t 판정을 통과한 시드에서 추정값을 걸었을 때의 휨 배수(보정 없음 = 1).
    """
    with open(os.path.join(HERE, "곡률_t값판정_시드별.csv"), encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    nl = chr(10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_IN, 2.8),
                                   gridspec_kw={"height_ratios": [1.3, 1.0]})
    band = max(abs(float(r["c_참"])) for r in rows)
    ax1.axhspan(-band, band, color="#D9D9D9", zorder=0, label="참 곡률 범위")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.5)
    for lens, marker in (("광각 90도", "o"), ("초광각 120도", "^")):
        for passed in (1, 0):
            pick = [r for r in rows if r["렌즈"] == lens and int(r["t판정"]) == passed]
            if not pick:
                continue
            ax1.scatter([float(r["의도세로_가로제곱_상관"]) for r in pick],
                        [float(r["c_추정"]) for r in pick],
                        marker=marker, s=13, linewidths=0.7, edgecolors=WORSE,
                        facecolors=WORSE if passed else "white",
                        label="%s · t 판정 %s" % (lens.split()[0], "통과" if passed else "미통과"))
    ax1.set_xlabel("의도한 세로 움직임과 가로²의 상관", fontsize=7)
    ax1.set_ylabel("추정한 곡률 계수", fontsize=7)
    ax1.legend(fontsize=5.3, loc="upper left", framealpha=0.95)

    passed_rows = sorted([r for r in rows if int(r["t판정"])],
                         key=lambda r: float(r["추정보정/보정없음"]))
    ratios = [float(r["추정보정/보정없음"]) for r in passed_rows]
    colors = [AFTER if r["렌즈"] == "광각 90도" else "#8DB4E2" for r in passed_rows]
    ax2.bar(range(len(ratios)), ratios, color=colors, edgecolor="black", linewidth=0.3)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ordered = sorted(ratios)
    half = len(ordered) // 2
    median = ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2
    ax2.axhline(median, color=WORSE, linestyle=":", linewidth=0.9)
    # 막대가 오름차순이라 왼쪽 위가 비어 있다 — 글자를 거기 둔다
    ax2.text(-0.4, median + 0.08, "중앙값 %.1f배" % median, fontsize=6,
             ha="left", va="bottom", color=WORSE)
    ax2.set_xticks([])
    ax2.set_xlabel("t 판정을 통과한 시드 %d개 (진한 막대 광각 · 옅은 막대 초광각)" % len(ratios),
                   fontsize=6.5)
    ax2.set_ylabel("추정값을 건 뒤 휨" + nl + "(보정 없음 = 1)", fontsize=7)
    fig.tight_layout(h_pad=0.6)
    save(fig, "논문_t값판정_가짜곡률.png", "t값 판정이 없는 곡률을 만든다")

def main():
    print("논문 2단용 그림을 만듭니다 (단 폭 8 cm, 그림 안 제목 없음)")
    print("")
    for fn in (fig_normalization, fig_relative_rotation, fig_mouth_cliff, fig_t_gate):
        try:
            fn()
        except Exception as exc:              # noqa: 한 장 실패해도 나머지는 만든다
            print("  [실패] %s -> %s" % (fn.__name__, exc))
    print("")
    print("설명은 그림 안이 아니라 **원고의 캡션**이 합니다.")


if __name__ == "__main__":
    main()
