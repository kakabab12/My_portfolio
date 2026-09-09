# -*- coding: utf-8 -*-
"""렌즈 자가 보정 전/후 — 공정하게 다시 (2026-09-09).

앞선 측정은 공정하지 않았다. "보정 켬"에만 70초 예열을 줬는데, 그 시간 동안
**auto_arc(온라인 곡률 보정)도 함께 학습**한다. 그래서 휨이 나빠진 것이
렌즈 보정 탓인지 auto_arc 탓인지 구분되지 않았다.

여기서는 두 가지를 바로잡는다.

  1) **양쪽 모두** 같은 70초 예열을 준다 — 오직 렌즈 보정 플래그만 다르다
  2) auto_arc를 끈 조건도 따로 잰다 — 두 기법의 기여를 분리한다

돌리는 법 (프로젝트 루트에서):
    py "데이터 수치/lens_ablation.py"
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

# ablation 이 import 시점에 stdout 을 UTF-8 로 감싼다 — 여기서 또 감싸면
# 앞 래퍼가 닫히면서 "I/O operation on closed file" 이 난다. 그래서 안 감싼다
from ablation import measure          # 같은 계측 함수를 그대로 쓴다

ROWS = []
LENSES = ("일반 65도", "광각 90도", "초광각 120도")
MOUNT = "오른쪽에서 25도"              # 가장 빠듯한 배치


def run(title, auto_arc):
    print(f"\n■ {title}")
    print(f"   배치 {MOUNT} · 양쪽 모두 70초 예열 (플래그만 다름)")
    print(f"   {'설정':<26s} {'끌림':>8s} {'선형성':>8s} {'휨':>8s} {'떨림':>8s} {'이동폭':>8s}")
    for lens in LENSES:
        got = {}
        for on in (False, True):
            m = measure({"orientation_lens_calibration": on,
                         "orientation_auto_arc": auto_arc},
                        lens=lens, mount=MOUNT, do_warmup=True)
            label = f"{lens} / 보정 {'켬' if on else '끔'}"
            if not m or m.get("측정불가"):
                print(f"   {label:<26s}  측정 불가")
                continue
            print(f"   {label:<26s} {m['끌림']:7.3f}% {m['선형성']:7.3f}%"
                  f" {m['휨']:7.3f}% {m['떨림']:7.3f}% {m['이동폭']:7.2f}%")
            ROWS.append([title, label, round(m["끌림"], 4), round(m["선형성"], 4),
                         round(m["휨"], 4), round(m["떨림"], 4), round(m["이동폭"], 3)])
            got[on] = m
        if False in got and True in got:
            a, b = got[False], got[True]
            for k in ("끌림", "선형성", "휨"):
                if a[k] > 1e-6:
                    chg = (b[k] - a[k]) / a[k] * 100.0
                    if abs(chg) >= 5.0:
                        print(f"      {k}: {a[k]:.3f}% -> {b[k]:.3f}%  ({chg:+.0f}%)")


if __name__ == "__main__":
    print("=" * 78)
    print(" 렌즈 자가 보정 전/후 — 예열 조건을 맞춘 공정 비교")
    print("=" * 78)
    run("⑩ 렌즈 자가 보정 (auto_arc 켬 — 배포 기본)", True)
    run("⑪ 렌즈 자가 보정 (auto_arc 끔 — 기여 분리)", False)

    path = os.path.join(HERE, "렌즈보정_전후비교.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["조건", "설정", "몸평행이동 끌림(%)", "가로 선형성 이탈(%)",
                    "세로 휨(%)", "정지 떨림(%)", "커서 이동폭(%)"])
        w.writerows(ROWS)
    print(f"\n저장: 렌즈보정_전후비교.csv ({len(ROWS)}행)")
