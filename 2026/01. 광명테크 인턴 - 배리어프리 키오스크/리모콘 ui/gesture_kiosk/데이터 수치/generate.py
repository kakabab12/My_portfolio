# -*- coding: utf-8 -*-
"""논문용 수치를 한 번에 뽑는다 (2026-09-09).

여기서 나오는 값은 전부 **가상 카메라 시뮬레이션**이다. 실제 장치에서 잰
값은 `01_실기_측정치.md`에 따로 모아 뒀다. 둘을 섞어 쓰면 안 된다.

돌리는 법:
    py "데이터 수치/generate.py"

결과물은 같은 폴더에 CSV로 떨어지고, 요약은 화면에 찍힌다.
"""
import csv
import io
import math
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from src.postprocess.head_orientation import HeadOrientation
from src.postprocess.head_tracker import OneEuroFilter
from tests.virtual_camera import (FACE_VARIANTS, MOUNTS, VirtualCamera, rotation)

LMK_EYE_L, LMK_EYE_R, LMK_NOSE = 33, 263, 4
FPS = 30.0


def write_csv(name, header, rows):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  저장: {name}  ({len(rows)}행)")


# ─────────────────────────────────────────────────────────────────────────
# 1. 혼합비 r 스윕 — forehead.py의 기준점 방식
#
#     기준점 p_r = 눈중점 + r x (코끝 - 눈중점)
#     얼굴 기준 신호 u = (p_r - 눈중점) / 안구간거리
#
# r을 훑으면서 네 가지를 잰다. 논문 초안에서 "훑지 않았다"고 한계로 적었던
# 바로 그 곡선이다.
# ─────────────────────────────────────────────────────────────────────────
def blend_signal(face, r):
    """그 프레임의 얼굴 기준 신호 (u_x, u_y). 못 구하면 None."""
    pts = np.asarray(face.landmarks_3d, dtype=np.float64)
    eL, eR, nose = pts[LMK_EYE_L][:2], pts[LMK_EYE_R][:2], pts[LMK_NOSE][:2]
    mid = 0.5 * (eL + eR)
    dist = float(np.hypot(*(eR - eL)))
    if dist < 1e-6:
        return None
    p = mid + r * (nose - mid)
    return tuple((p - mid) / dist)


def sweep_blend_ratio():
    print("\n=== 1. 혼합비 r 스윕 (가상 카메라, 광각 90도, 600mm) ===")
    print(f"  {'r':>5s} {'회전이득':>9s} {'곡률(휨)':>10s} {'정지떨림':>9s} {'평행이동끌림':>12s}")
    rows = []
    for r in [round(x, 2) for x in np.arange(0.05, 1.001, 0.05)]:
        cam = VirtualCamera(lens="광각 90도", seed=3)

        # (a) 회전 이득 — yaw ±12도에서 u_x의 기울기 (단위: 1/도)
        cam.reset_noise()
        xs, degs = [], []
        for d in np.arange(-12.0, 12.01, 1.0):
            s = blend_signal(cam.observe(rotation((0, 1, 0), float(d))), r)
            if s:
                xs.append(s[0]); degs.append(float(d))
        gain = float(np.polyfit(degs, xs, 1)[0]) if len(xs) > 4 else float("nan")

        # (b) 곡률 — 순수 yaw 회전인데 u_y가 얼마나 흔들리나 (원근 휨)
        cam.reset_noise()
        ys = []
        for d in np.arange(-12.0, 12.01, 1.0):
            s = blend_signal(cam.observe(rotation((0, 1, 0), float(d))), r)
            if s:
                ys.append(s[1])
        bow = float(max(ys) - min(ys)) if ys else float("nan")

        # (c) 정지 떨림 — 가만히 있을 때 u_x의 표준편차
        cam.reset_noise()
        still = [blend_signal(cam.observe(), r) for _ in range(120)]
        still = [s for s in still if s]
        jitter = float(np.std([s[0] for s in still])) if still else float("nan")

        # (d) 몸 평행이동 끌림 — 얼굴 기준 좌표계가 상쇄해 주는 부분
        cam.reset_noise()
        base = blend_signal(cam.observe(), r)
        drift = 0.0
        for sh in ((60, 0, 0), (-60, 0, 0), (0, 50, 0), (0, 0, 120)):
            s = blend_signal(cam.observe(offset_mm=sh), r)
            if s and base:
                drift = max(drift, math.hypot(s[0] - base[0], s[1] - base[1]))

        rows.append([r, round(gain, 6), round(bow, 6), round(jitter, 6), round(drift, 6)])
        print(f"  {r:5.2f} {gain:9.5f} {bow:10.5f} {jitter:9.5f} {drift:12.5f}")

    write_csv("혼합비_스윕.csv",
              ["혼합비 r", "회전이득(1/도)", "곡률 휨(신호폭)", "정지떨림(표준편차)",
               "몸평행이동 끌림"], rows)
    print("  ※ r=0 이면 신호가 항상 0이 되어(원점과 축퇴) 측정 자체가 불가능하다.")
    print("  ※ 입 벌림 간섭은 가상 카메라가 코 변형을 모형에 안 넣어 여기서 못 잰다 —")
    print("     실기 측정이 필요하다.")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# 2. 거리에 따른 기하학적 반폭 — 화면 크기별
# ─────────────────────────────────────────────────────────────────────────
def geometric_span():
    print("\n=== 2. 거리별 기하학적 반폭 = atan((화면 절반 폭)/거리) ===")
    screens = [("14인치 노트북", 310.0, 174.0),
               ("24인치 모니터", 531.0, 299.0),
               ("32인치 키오스크", 700.0, 393.0),
               ("55인치 사이니지", 1210.0, 680.0)]
    dists = [400, 500, 600, 700, 800, 1000, 1300]
    rows = []
    print("  " + "화면".ljust(16) + "".join(f"{d}mm".rjust(9) for d in dists))
    for name, w, _h in screens:
        line = []
        for d in dists:
            deg = math.degrees(math.atan((w * 0.5) / d))
            line.append(deg)
            rows.append([name, w, d, round(deg, 2)])
        print("  " + name.ljust(16) + "".join(f"{v:8.1f}도" for v in line))
    write_csv("거리별_기하반폭.csv",
              ["화면", "가로(mm)", "사용자 거리(mm)", "맞는 반폭(도)"], rows)
    print("  ※ 코드의 기존 고정값은 15.0도였다. 위 표에서 15도와 맞는 조합은 드물다.")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# 3. 목 가동범위별 화면 도달률
# ─────────────────────────────────────────────────────────────────────────
def reach_by_rom():
    print("\n=== 3. 목 가동범위별 화면 도달률 (반폭 15도 기준) ===")
    span = math.tan(math.radians(15.0))
    rows = []
    for deg in [3, 4, 5, 6, 7, 8, 10, 12, 15, 18]:
        ratio = min(1.0, math.tan(math.radians(deg)) / span)
        need = 1.0 / ratio if ratio > 0 else float("inf")
        rows.append([deg, round(ratio * 100, 1), round(min(need, 99), 2)])
        print(f"  좌우 {deg:2d}도  ->  화면의 {ratio*100:5.1f}%  "
              f"(전체를 쓰려면 배율 {min(need,99):.2f})")
    write_csv("가동범위별_도달률.csv",
              ["좌우 가동범위(도)", "화면 도달률(%)", "필요한 도달 배율"], rows)
    return rows


# ─────────────────────────────────────────────────────────────────────────
# 4. 커서 평활 비교 — EMA 대 1유로 필터
# ─────────────────────────────────────────────────────────────────────────
def smoothing_compare():
    print("\n=== 4. 커서 평활 비교 (씨앗 3개 평균) ===")
    TANH = math.tan(math.radians(15.0))

    def series(dist, seconds=8.0, swing=0.0, seed=3):
        cam = VirtualCamera(lens="광각 90도", distance_mm=dist, seed=seed)
        ho = HeadOrientation(rotation_source="landmarks")
        for _ in range(20):
            ho.add_calibration_sample(cam.observe())
        ho.finalize_neutral()
        out, n = [], int(seconds * FPS)
        for i in range(n):
            deg = swing if i >= n // 2 else 0.0
            v = ho.pointing_offset(cam.observe(rotation((0, 1, 0), deg)))
            out.append(v[0] if v else (out[-1] if out else 0.0))
        return np.array(out)

    def ema(x, a):
        y = np.empty_like(x); y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = y[i - 1] + a * (x[i] - y[i - 1])
        return y

    def one_euro(x, mc, beta):
        f = OneEuroFilter(min_cutoff=mc, beta=beta)
        return np.array([f(v, i / FPS) for i, v in enumerate(x)])

    seeds = (3, 11, 29)
    methods = [("평활 없음", lambda x: x),
               ("EMA a=0.20", lambda x: ema(x, 0.20)),
               ("EMA a=0.32", lambda x: ema(x, 0.32)),
               ("1유로 0.25/1.5", lambda x: one_euro(x, 0.25, 1.5)),
               ("1유로 0.20/2.5", lambda x: one_euro(x, 0.20, 2.5))]
    rows = []
    print(f"  {'방식':<16s} {'정지떨림(화면%)':>14s} {'지연(프레임)':>12s}")
    for name, fn in methods:
        js = [np.std(fn(series(d, seed=s))) / TANH * 100
              for d in (600.0, 1000.0, 1300.0) for s in seeds]
        lags = []
        for s in seeds:
            x = series(600.0, swing=10.0, seed=s); n = len(x) // 2
            y = fn(x); tgt = abs(np.mean(x[-20:])) * 0.9
            got = next((i - n for i in range(n, len(y)) if abs(y[i]) >= tgt), len(y) - n)
            lags.append(got)
        rows.append([name, round(float(np.mean(js)), 3), round(float(np.mean(lags)), 1)])
        print(f"  {name:<16s} {np.mean(js):13.3f}% {np.mean(lags):11.1f}")
    write_csv("평활_비교.csv", ["방식", "정지떨림(화면 %)", "10도 계단 90% 도달(프레임)"], rows)
    return rows


# ─────────────────────────────────────────────────────────────────────────
# 5. 카메라 배치 × 렌즈별 몸 평행이동 끌림 (상대 회전 경로)
# ─────────────────────────────────────────────────────────────────────────
def drift_by_mount_lens():
    print("\n=== 5. 배치 x 렌즈별 몸 평행이동 끌림 (상대 회전 경로, 렌즈보정 없음) ===")
    rows = []
    lenses = ("왜곡없음", "일반 65도", "광각 90도", "초광각 120도")
    mounts = ("정면", "밑에서 35도", "위에서 25도", "오른쪽에서 25도", "심하게 비스듬히")
    print("  " + "배치".ljust(16) + "".join(l.rjust(12) for l in lenses))
    for mn in mounts:
        line = []
        for lens in lenses:
            cam = VirtualCamera(mount=MOUNTS[mn], lens=lens, seed=3)
            cam.reset_noise()
            ho = HeadOrientation(rotation_source="landmarks")
            for _ in range(20):
                ho.add_calibration_sample(cam.observe())
            if not ho.finalize_neutral():
                line.append(float("nan")); continue
            base = ho.pointing_offset(cam.observe())
            d = 0.0
            for sh in ((60, 0, 0), (-60, 0, 0), (0, 50, 0), (0, 0, 120)):
                m = ho.pointing_offset(cam.observe(offset_mm=sh))
                if m and base:
                    d = max(d, math.hypot(m[0] - base[0], m[1] - base[1]))
            line.append(d)
            rows.append([mn, lens, round(d, 5)])
        print("  " + mn.ljust(16) + "".join(f"{v:12.4f}" for v in line))
    write_csv("배치렌즈별_끌림.csv", ["카메라 배치", "렌즈", "몸 평행이동 끌림"], rows)
    print("  ※ 프로젝트가 정한 허용 한도는 0.020이다.")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# 6. 얼굴 개인차에 따른 거리 추정 오차 — 절대 대 상대
# ─────────────────────────────────────────────────────────────────────────
def distance_estimation():
    print("\n=== 6. 상대 거리 추정 정확도 (얼굴 크기 비, 초점거리 불필요) ===")
    from src.postprocess.head_orientation import RIGID_LANDMARKS
    from src.postprocess.lens_calibration import CANONICAL_FACE
    idx = [i for i in RIGID_LANDMARKS if i in CANONICAL_FACE]

    def size_px(face):
        p = np.asarray(face.landmarks_3d, dtype=np.float64)[idx]
        p = p - p.mean(axis=0)
        return float(np.sqrt((p ** 2).sum()))

    rows = []
    print(f"  {'얼굴':<12s} {'참 거리':>8s} {'추정':>8s} {'오차':>8s}")
    for name, face in list(FACE_VARIANTS.items())[:6]:
        ref = VirtualCamera(lens="광각 90도", distance_mm=600.0, seed=3, face=face)
        ref.reset_noise()
        s_ref = size_px(ref.observe())
        for d in (400.0, 600.0, 800.0, 1000.0, 1200.0):
            cam = VirtualCamera(lens="광각 90도", distance_mm=d, seed=3, face=face)
            cam.reset_noise()
            est = 600.0 * s_ref / size_px(cam.observe())
            err = (est - d) / d
            rows.append([name, d, round(est, 1), round(err * 100, 2)])
            print(f"  {name:<12s} {d:7.0f}mm {est:7.0f}mm {err*100:+7.2f}%")
    write_csv("상대거리_추정오차.csv",
              ["얼굴", "참 거리(mm)", "추정 거리(mm)", "오차(%)"], rows)
    return rows


if __name__ == "__main__":
    print("=" * 72)
    print(" 논문용 수치 생성 — 전부 가상 카메라 시뮬레이션")
    print("=" * 72)
    sweep_blend_ratio()
    geometric_span()
    reach_by_rom()
    smoothing_compare()
    drift_by_mount_lens()
    distance_estimation()
    print("\n완료. CSV는 '데이터 수치' 폴더에 있습니다.")
