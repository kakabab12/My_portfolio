# -*- coding: utf-8 -*-
"""클릭할 때 커서를 어느 시점 위치에 붙잡나 — 되짚기와 드래그의 맞바꿈 (2026-09-11).

왜 측정하나
-----------
head.py의 CLICK_FREEZE_LOOKBACK_SEC 주석에 표가 있다(되짚기 0초 → 겨눈 곳에서 12~25 px,
끌림 0 / 0.04초 → 0, 끌림 12~25 ...). 논문이 이 "되짚은 만큼 그대로 끌린다"를 쓰는데,
그 표를 낸 스크립트와 원자료가 저장소에 없다. 그래서 가상 사용자 시험
(tests/test_virtual_user_click.py)의 실행 도구를 그대로 가져와 다시 측정하고 CSV로 남긴다.

무엇을 측정하나
---------------
고개를 8도 돌려 겨눈 채 입을 벌려 클릭하고, 벌리고 있는 동안 고개를 0·2·4·6도 더 돌린다
(사람은 클릭 중에도 움직인다). 붙잡기 설정 넷을 견준다 — 되짚기 0·0.04·0.08초, 붙잡기 끔.

  · 겨눈 곳에서 벗어난 거리 — 클릭이 찍힌 자리와, 입을 벌리기 전 겨누던 자리 사이 (px, 1920×1080)
  · 눌린 채 끌린 거리 — 버튼이 눌렸는데 드래그 색이 아닌 동안 커서가 간 거리 (px)
  · 판정 — 이름표가 [PRESS, CLICK]이면 클릭

    py -3 "데이터 수치/click_freeze_lookback.py"
"""
import csv
import io
import math
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src.postprocess.mouth_gesture import CLICK, PRESS  # noqa: E402
from tests.test_virtual_user_click import WARMUP, _click_actions, _use  # noqa: E402
from tests.virtual_user import aim, rest  # noqa: E402

YAW_FROM = 8.0
TURNS = (0.0, 2.0, 4.0, 6.0)
SETTINGS = (
    ("되짚기 0초", True, 0.0),
    ("되짚기 0.04초", True, 0.04),
    ("되짚기 0.08초", True, 0.08),
    ("붙잡기 끔", False, 0.0),
)


def main():
    rows = []
    for label, freeze_on, lookback in SETTINGS:
        for turn in TURNS:
            yaw_to = YAW_FROM + turn
            actions = (WARMUP + [aim(1.0, yaw_deg=YAW_FROM)]
                       + _click_actions(0.3, YAW_FROM, yaw_to)
                       + [rest(1.0, yaw_deg=yaw_to, label="대기")])
            out = _use(actions, freeze_enabled=freeze_on, lookback=lookback)
            if out["click_px"] is not None and out["aim_px"] is not None:
                off = math.hypot(out["click_px"][0] - out["aim_px"][0],
                                 out["click_px"][1] - out["aim_px"][1])
            else:
                off = float("nan")
            rows.append([label, turn, round(off, 1),
                         round(out["travel_while_green_px"], 1),
                         int(out["events"] == [PRESS, CLICK]),
                         "+".join(out["events"]) or "(없음)"])

    path = os.path.join(HERE, "클릭붙잡기_되짚기.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["설정", "클릭 중 고개를 더 돌린 각도(도)", "겨눈 곳에서 벗어난 거리(px)",
                         "눌린 채 끌린 거리(px)", "클릭 판정", "이름표"])
        writer.writerows(rows)

    print("클릭 중 고개를 0·2·4·6도 더 돌릴 때 (1920×1080 px)")
    for label, _on, _lb in SETTINGS:
        picked = [r for r in rows if r[0] == label]
        offs = [r[2] for r in picked if not math.isnan(r[2])]
        drags = [r[3] for r in picked]
        clicks = sum(r[4] for r in picked)
        print("  %-12s 겨눈 곳에서 %s px · 끌림 %.1f~%.1f px · 클릭 %d/%d"
              % (label,
                 ("%.1f~%.1f" % (min(offs), max(offs))) if offs else "-",
                 min(drags), max(drags), clicks, len(picked)))
    print("저장:", path)


if __name__ == "__main__":
    main()
