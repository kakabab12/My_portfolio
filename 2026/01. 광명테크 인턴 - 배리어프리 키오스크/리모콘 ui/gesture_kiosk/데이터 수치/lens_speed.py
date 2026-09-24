# -*- coding: utf-8 -*-
"""렌즈 자가 보정 한 번(cv2.calibrateCamera)이 설치 조합마다 몇 초인가 (2026-09-25).

출시 점검에서 키오스크와 같은 조합(requirements.txt 옛 고정: numpy 1.26.4 +
OpenCV 4.10.0.84)으로 스모크 실행을 돌렸더니 렌즈 보정이 끝나지 않았다.
같은 계산이 개발 PC(OpenCV 5.0)에서는 0.1초대였다. 입력을 똑같이 두고
설치 조합만 바꿔 한 번에 걸리는 시간과 결과를 적는다.

입력: 가상 카메라(광각 90도) 얼굴 60뷰, 강체 22점 — 실제 보정기의 _pick으로
뽑고 실제 _calibrate를 부른다. 세 번 되풀이한다.

돌리는 법 (프로젝트 루트에서, 조합마다 그 파이썬으로):
    py -3.11 "데이터 수치/lens_speed.py"
결과는 렌즈보정_속도.csv에 한 줄씩 **덧붙인다**(조합별 비교용).
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
from tests.virtual_camera import FRAME_H_PX, FRAME_W_PX, VirtualCamera, rotation  # noqa: E402

VIEWS = 60
REPEATS = 3
OUT = os.path.join(HERE, "렌즈보정_속도.csv")


def make_views():
    cam = VirtualCamera(lens="광각 90도", seed=3)
    cal = lc.LensSelfCalibrator(FRAME_W_PX, FRAME_H_PX, enabled=False)
    rng = np.random.default_rng(1)
    views = []
    while len(views) < VIEWS:
        off = (rng.uniform(-200, 200), rng.uniform(-110, 110), rng.uniform(-90, 90))
        rot = (rotation((0, 1, 0), rng.uniform(-13, 13))
               @ rotation((1, 0, 0), rng.uniform(-8, 8)))
        pts = cal._pick(cam.observe(rot, offset_mm=off).landmarks_3d)
        if pts is not None:
            views.append(pts)
    return views


def main():
    env = "파이썬 %s · numpy %s · OpenCV %s" % (
        sys.version.split()[0], np.__version__, cv2.__version__)
    print("■ " + env, flush=True)
    views = make_views()
    rows = []
    for rep in range(1, REPEATS + 1):
        t0 = time.perf_counter()
        out = lc._calibrate(views, FRAME_W_PX, FRAME_H_PX)
        sec = time.perf_counter() - t0
        if out is None:
            print("   %d번째: %.2f초 · 실패" % (rep, sec), flush=True)
            rows.append([sys.version.split()[0], np.__version__, cv2.__version__,
                         VIEWS, rep, round(sec, 3), "", "", ""])
            continue
        focal, k1, rms = out
        print("   %d번째: %.2f초 · f=%.1f k1=%.3f rms=%.3f" % (rep, sec, focal, k1, rms),
              flush=True)
        rows.append([sys.version.split()[0], np.__version__, cv2.__version__,
                     VIEWS, rep, round(sec, 3), round(focal, 2), round(k1, 4), round(rms, 4)])
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["파이썬", "numpy", "OpenCV", "뷰 수", "회차", "걸린 시간(초)",
                        "초점거리(px)", "k1", "재투영오차(px)"])
        w.writerows(rows)
    print("→ " + OUT)


if __name__ == "__main__":
    main()
