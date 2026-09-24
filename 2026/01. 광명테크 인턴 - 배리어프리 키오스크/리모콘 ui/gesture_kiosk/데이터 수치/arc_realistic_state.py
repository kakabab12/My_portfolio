# -*- coding: utf-8 -*-
"""실사용 움직임을 2분 겪은 뒤 auto_arc는 어떤 상태이고, 그때 휨은 얼마인가 (2026-09-11).

왜 측정하나
-----------
arc_protocol_trap.py로 적합 기록을 남겨 보니, 렌즈 비교 측정에서 auto_arc를 깨운 것은
순수 좌우 훑기가 아니라 그 앞의 70초 예열(여러 방향이 섞인 움직임)이었다. 그런데 같은
배치(오른쪽에서 25도)에서 실사용 움직임 2분도 판정을 28번 중 10번 통과했고, 끝난 뒤
계수(-1.78)가 예열 뒤(-1.72)와 비슷했다.

그렇다면 렌즈 비교의 17%대 휨은 "측정 절차가 만든 값"이 아니라, 실사용을 거친 뒤의
상태에서도 나오는 값일 수 있다. 05번 문서 §3은 그 값을 "프로토콜 인공물이니 쓰지 말라"고
적었는데, 한 시드·한 배치로는 어느 쪽인지 말할 수 없다. 그래서 넓혀 측정한다.

무엇을 측정하나
---------------
배치 2종(정면, 오른쪽에서 25도) × 렌즈 3종 × 렌즈 보정 켬/끔 × 시드 5개, 600 mm.
설정은 응시 기능을 포함한 배포 설정 그대로다. 조합마다 auto_arc를 켜고 한 번, 끄고 한 번:

  중립 70프레임 → 실사용 움직임 2분(arc_protocol_trap.py와 같은 생성기, 시드만 바꿈)
  → 좌우 ±14° 훑기(도마다 3프레임)

  · 실사용 2분 동안 auto_arc의 적합 시도와 판정 통과 횟수
  · 훑기 직전 계수
  · 훑기의 세로 휨(%) — lens_ablation.py와 같은 정의(평균에서 가장 먼 세로 위치, 화면 높이 대비)

배포 설정은 렌즈 보정 켬이다. 렌즈 보정 끔은 렌즈 비교 측정의 '보정 끔' 줄과 잇기 위해 넣었다.

    py -3 "데이터 수치/arc_realistic_state.py"
"""
import csv
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# arc_protocol_trap을 불러오면 ablation(표준출력 UTF-8 래핑)과 적합 기록 장치가 함께 들어온다
import arc_protocol_trap as trap  # noqa: E402
from ablation import _Face, build  # noqa: E402
import numpy as np  # noqa: E402
from tests.virtual_camera import MOUNTS, VirtualCamera, rotation  # noqa: E402

MOUNT_NAMES = ("정면", "오른쪽에서 25도")
LENSES = ("일반 65도", "광각 90도", "초광각 120도")
LENS_CAL = (True, False)
SEEDS = (1, 2, 3, 4, 5)
DISTANCE_MM = 600.0
FPS = 30.0
USE_SEC = 120.0


def run(mount, lens, lens_cal, seed, arc_on):
    tracker, clock = build({"orientation_lens_calibration": lens_cal,
                            "orientation_auto_arc": arc_on})
    cam = VirtualCamera(lens=lens, distance_mm=DISTANCE_MM, seed=3, mount=MOUNTS[mount])
    cam.reset_noise()
    tag = "%s|%s|%d|%d|%d" % (mount, lens, lens_cal, seed, arc_on)
    trap.STATE.update(lens=lens, input=tag, phase="실사용 2분", clock=clock)

    def step(R=None):
        clock.now += 1.0 / FPS
        return tracker.update(_Face(cam.observe(R).landmarks_3d))

    for _ in range(70):
        step()
    for yaw, pitch in trap.kiosk_user(USE_SEC, seed=seed):
        step(rotation((0, 1, 0), yaw) @ rotation((1, 0, 0), pitch))

    arc = getattr(tracker._cursor_mapper, "_auto_arc", None)
    coef = arc.coef if (arc_on and arc is not None) else float("nan")

    cam.reset_noise()
    trap.STATE["phase"] = "좌우 ±14° 훑기"
    ys = []
    for deg in np.arange(-14.0, 14.01, 1.0):
        r = None
        for _ in range(3):
            r = step(rotation((0, 1, 0), float(deg)))
        if r is not None and r.cursor_x_ratio is not None:
            ys.append(r.cursor_y_ratio)
    ys = np.array(ys)
    bow = 100.0 * float(np.abs(ys - ys.mean()).max()) if len(ys) else float("nan")

    logs = [e for e in trap.LOG if e["입력"] == tag and e["단계"] == "실사용 2분"]
    return {"시도": len(logs), "통과": sum(e["통과"] for e in logs),
            "계수": coef, "휨": bow, "훑기 표본": len(ys)}


def _med(values):
    values = [v for v in values if not math.isnan(v)]
    return statistics.median(values) if values else float("nan")


def main():
    started = time.time()
    rows = []
    for mount in MOUNT_NAMES:
        for lens in LENSES:
            for lens_cal in LENS_CAL:
                for seed in SEEDS:
                    off = run(mount, lens, lens_cal, seed, False)
                    on = run(mount, lens, lens_cal, seed, True)
                    rows.append({
                        "배치": mount, "렌즈": lens, "렌즈 보정": "켬" if lens_cal else "끔",
                        "시드": seed,
                        "적합 시도": on["시도"], "판정 통과": on["통과"],
                        "훑기 직전 계수": on["계수"],
                        "휨 auto_arc 끔(%)": off["휨"], "휨 auto_arc 켬(%)": on["휨"],
                        "켬/끔": on["휨"] / off["휨"] if off["휨"] > 0 else float("nan"),
                        "훑기 표본 끔": off["훑기 표본"], "훑기 표본 켬": on["훑기 표본"],
                    })
                print("  %s · %s 끝 (%.0f초)" % (mount, lens, time.time() - started))
                sys.stdout.flush()

    path = os.path.join(HERE, "곡률보정_실사용뒤_휨.csv")
    columns = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([round(row[k], 4) if isinstance(row[k], float) else row[k]
                             for k in columns])

    print("\n■ 실사용 2분 뒤 좌우 훑기의 세로 휨 (시드 %d개, 중앙값과 범위)" % len(SEEDS))
    for mount in MOUNT_NAMES:
        for lens_cal in ("켬", "끔"):
            for lens in LENSES:
                pick = [r for r in rows if r["배치"] == mount and r["렌즈"] == lens
                        and r["렌즈 보정"] == lens_cal]
                ratios = [r["켬/끔"] for r in pick]
                coefs = [r["훑기 직전 계수"] for r in pick]
                passes = [r["판정 통과"] for r in pick]
                worse = sum(1 for r in pick if r["휨 auto_arc 켬(%)"] > r["휨 auto_arc 끔(%)"])
                print("  %-9s 렌즈보정 %s %-8s 휨 끔 %5.2f%% · 켬 %5.2f%% (%.2f~%.2f) · 켬/끔 %.2f배 (%.2f~%.2f)"
                      " · 나빠짐 %d/%d · 통과 %s회 (%d~%d) · 계수 %+.2f (%+.2f~%+.2f)"
                      % (mount, lens_cal, lens,
                         _med([r["휨 auto_arc 끔(%)"] for r in pick]),
                         _med([r["휨 auto_arc 켬(%)"] for r in pick]),
                         min(r["휨 auto_arc 켬(%)"] for r in pick),
                         max(r["휨 auto_arc 켬(%)"] for r in pick),
                         _med(ratios), min(ratios), max(ratios), worse, len(pick),
                         statistics.median(passes), min(passes), max(passes),
                         _med(coefs), min(coefs), max(coefs)))
    print("\n저장: %s (%d행, %.0f초)" % (path, len(rows), time.time() - started))


if __name__ == "__main__":
    main()
