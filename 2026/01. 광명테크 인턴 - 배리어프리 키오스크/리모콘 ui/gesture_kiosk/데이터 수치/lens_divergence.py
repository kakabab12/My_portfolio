# -*- coding: utf-8 -*-
"""렌즈 자가 보정이 얼굴마다 몇 번 시도하고, 한 번에 얼마나 걸리나 (2026-09-25).

출시 점검 중 시험 하나(tests/test_lens_face_variation.py, "눈 좁은 얼굴")가 유난히
오래 걸려 들여다보니, 정규 모형과 다른 얼굴에서는 보정이 **발산**했다 — 초점거리가
수백만 px로 가고, 한 번에 0.1초대가 아니라 수 초씩 걸린 뒤 범위 밖으로 거부된다.
얼굴마다 그 비용을 적는다.

흐름은 시험과 같다(가상 카메라 광각 90도, 60초 동안 걷고 고개를 돌리는 사용자,
시도가 시작되면 끝날 때까지 기다림). 보정 호출 하나마다 벽시계 시간과 **그
스레드가 실제로 쓴 CPU 시간**을 함께 적는다 — 보정 스레드는 우선순위가 가장
낮아서, PC가 바쁘면 벽시계만 늘어난다.

돌리는 법 (프로젝트 루트에서):
    py -3.11 "데이터 수치/lens_divergence.py"
결과: 렌즈보정_발산.csv (덮어씀)
"""
import csv
import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from src.postprocess import lens_calibration as lc  # noqa: E402
from tests.virtual_camera import FACE_VARIANTS, VirtualCamera  # noqa: E402
import tests.test_lens_face_variation as flow  # noqa: E402  (시험과 같은 흐름을 쓴다)

OUT = os.path.join(HERE, "렌즈보정_발산.csv")


def main():
    env = (sys.version.split()[0], np.__version__, cv2.__version__)
    print("■ 파이썬 %s · numpy %s · OpenCV %s" % env, flush=True)
    calls = []
    original = lc._calibrate

    def timed(views, width, height):
        wall0, cpu0 = time.perf_counter(), time.thread_time()
        out = original(views, width, height)
        calls.append((len(views), time.perf_counter() - wall0, time.thread_time() - cpu0, out))
        return out

    lc._calibrate = timed
    rows = []
    try:
        for face in FACE_VARIANTS:
            calls.clear()
            cal = flow._learn(VirtualCamera(lens="광각 90도", face=face, seed=3))
            adopted = cal.model is not None
            cpu_total = sum(c[2] for c in calls)
            print("  %-10s 시도 %d번 · 보정 호출 %2d번 · CPU 합 %5.1f초 · %s"
                  % (face, cal._attempts, len(calls), cpu_total,
                     "채택 f=%.1f" % cal.model.focal_px if adopted else "거부(%s)" % cal.reject_reason),
                  flush=True)
            for i, (n, wall, cpu, out) in enumerate(calls, start=1):
                rows.append([*env, face, i, n, round(wall, 3), round(cpu, 3),
                             round(out[0], 1) if out else "", round(out[1], 4) if out else "",
                             round(out[2], 4) if out else "", "채택" if adopted else "거부",
                             cal.reject_reason or ""])
    finally:
        lc._calibrate = original
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["파이썬", "numpy", "OpenCV", "얼굴", "호출 순서", "뷰 수", "벽시계(초)",
                    "스레드 CPU(초)", "초점거리(px)", "k1", "재투영오차(px)", "최종", "거부 사유"])
        w.writerows(rows)
    print("→ " + OUT)


if __name__ == "__main__":
    main()
