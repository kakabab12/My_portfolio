"""기록에 남은 손 21점 원본으로 손 위치를 여러 방식으로 다시 풀어 비교한다 (2026-11-17).

  python replay_raw.py data/날짜_시각/ep_000.npz            # 표만
  python replay_raw.py data/날짜_시각/ep_000.npz --png 그림.png

방식: pnp (회전+이동, 11/15~17 기본), oat (OAT 논문: 이동만, 11/18 기본), oat+벌점 (이동만 + 앞 장면과의 차이 벌점),
palm (손바닥 다섯 점으로 거리, 11/18 — 노트북 보정에 손바닥 자리(flat)가 있을 때만).
모두 노트북과 같은 Smoother(1€ 필터, 튀는 값 버리기)를 거친 뒤 비교한다. 노트북 보정 값은 같은 이름의 .json 에서 읽는다.
손 크기 배율은 노트북이 보정한 값을 그대로 쓴다 (방식마다 1~2% 다를 수 있음).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geometry import GRIPS, HandMeasure, TableCalib   # noqa: E402
from teleop import Smoother                            # noqa: E402


def load(path):
    d = np.load(path, allow_pickle=True)
    with open(path[:-4] + ".json", encoding="utf-8") as f:
        cal = json.load(f)["laptop_calib"]
    calib = TableCalib(cal["img_pts"], cal["w"], cal["h"],
                       hfov=None if cal.get("focal_source") != "hfov" else math.degrees(2 * math.atan(cal["w"] / 2 / cal["f"])))
    # 연구실 PC 는 박자마다 줄을 남기므로 같은 패킷(seq)이 여러 줄일 수 있다 → 패킷마다 한 줄
    _, first = np.unique(d["seq"], return_index=True)
    keep = np.sort(first)
    return d, cal, calib, keep


def solve(d, keep, calib, cal, solver, beta):
    flat = np.array(cal["flat"], float).reshape(21, 3) if cal.get("flat") else None
    meas = HandMeasure(calib, 1.0 if solver == "palm" else cal["scale"], cal.get("grip", "whole"), cal.get("mirror", False),
                       solver, beta, flat=flat)
    sm, raw_out, out = Smoother(), [], []
    for i in keep:
        l2, l3 = d["lm2d"][i], d["lm3d"][i]
        if "grip_mode" in d.files and d["grip_mode"][i] in (0, 1):      # 기록 중에 TAB 으로 손 모양을 바꿨으면 따라 바꿈
            meas.tips = list(GRIPS["pinch" if d["grip_mode"][i] == 1 else "whole"])
        r = meas(l3, l2) if np.isfinite(l2).all() else None
        if r is None:
            meas.reset()
        o = sm(float(d["t"][i]), r)
        raw_out.append(r["pinch"] if r else np.full(3, np.nan))
        out.append(o["pinch"] if o else np.full(3, np.nan))
    return np.array(raw_out), np.array(out)


def stats(p, t):
    ok = np.isfinite(p).all(1)
    q, tt = p[ok], t[ok]
    if len(q) < 10:
        return None
    k = 15
    det = []
    for i in range(3):
        pad = np.pad(q[:, i], (k // 2, k // 2), mode="edge")
        det.append(q[:, i] - np.convolve(pad, np.ones(k) / k, mode="valid"))
    det = np.array(det).T
    jump = np.linalg.norm(np.diff(q, axis=0), axis=1)
    return {"found": ok.mean(), "below": np.mean(q[:, 2] < -0.03), "jit": det.std(0), "jump99": np.percentile(jump, 99),
            "jumpmax": jump.max(), "z_med": np.median(q[:, 2]), "z5": np.percentile(q[:, 2], 5), "z95": np.percentile(q[:, 2], 95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--pen", type=float, default=0.5, help="oat+벌점 줄의 벌점")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    d, cal, calib, keep = load(a.episode)
    t = d["t"][keep]
    has = np.isfinite(d["lm2d"][keep]).all((1, 2))
    print(f"{os.path.basename(a.episode)}: 패킷 {len(keep)}개, 손 21점 있는 것 {has.mean() * 100:.0f}%, 노트북 방식 {cal.get('solver')}, "
          f"배율 {cal['scale']:.3f}, 초점 {cal['f']:.0f}px ({cal['focal_source']}), 카메라 자리 "
          f"{np.round(np.array(cal['cam_pos_table_m']) * 100)} cm")
    rows = [("노트북이 보낸 값", d["hand"][keep])]
    ways = [("pnp", "pnp", 0.0), ("oat 이동만", "oat", 0.0), (f"oat+벌점{a.pen}", "oat", a.pen)]
    if cal.get("flat"):
        ways.append(("palm 손바닥", "palm", 0.0))
    for name, solver, beta in ways:
        _, out = solve(d, keep, calib, cal, solver, beta)
        rows.append((name, out))
    print("방식              | 손 찾음 | 책상 3cm 아래 | 흔들림 std 앞뒤/좌우/높이 cm | 장면간 이동 99%/최대 cm | 높이 가운데 (5~95%) cm")
    for name, p in rows:
        s = stats(p, t)
        if s is None:
            print(f"{name:16s} | 값 없음")
            continue
        j = s["jit"] * 100
        print(f"{name:16s} | {s['found'] * 100:5.0f}% | {s['below'] * 100:9.0f}%  | {j[0]:.2f}/{j[1]:.2f}/{j[2]:.2f}"
              f"                | {s['jump99'] * 100:.1f}/{s['jumpmax'] * 100:.1f}            | {s['z_med'] * 100:.1f}"
              f" ({s['z5'] * 100:.1f}~{s['z95'] * 100:.1f})")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        tt = t - t[0]
        for name, p in rows:
            for i in range(3):
                ax[i].plot(tt, p[:, i] * 100, lw=0.8, label=name)
        for i, n in enumerate(["x (forward) cm", "y (left) cm", "z (up) cm"]):
            ax[i].set_ylabel(n)
            ax[i].grid(alpha=0.3)
        ax[0].legend(fontsize=8, ncol=4)
        ax[2].set_xlabel("s")
        plt.tight_layout()
        plt.savefig(a.png, dpi=80)
        print("그림:", a.png)


if __name__ == "__main__":
    main()
