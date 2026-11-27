"""녹화 영상 → 두 화면의 손 21점 (MediaPipe, 논문 3장 B 와 같음, 2026-11-27).

논문도 손은 MediaPipe 로 엄지 끝·검지 끝을 찾는다. 여기서는 21점을 다 남긴다 (손 회전은 손바닥 다섯 점으로 잼).
두 화면에 손이 여럿이면 삼각측량이 가장 잘 맞는 짝을 고른다 (webcam_piper teleop 과 같은 생각).
이미 hands2d.npz 가 있으면 건너뛴다 (--redo 로 다시).

  ~/smolVLA/webcam_piper/.venv/bin/python point_policy/hand_points.py point_policy/data/cube_to_pad/real
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import common  # noqa: F401
from episode_io import list_episodes, load_meta, proj_matrices
from geometry import triangulate
from hand import HandTracker


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--max-err", type=float, default=25.0)
    a = ap.parse_args()
    for ep in [e for f in a.folders for e in list_episodes(f)]:
        if (ep / "hands2d.npz").exists() and not a.redo:
            continue
        meta = load_meta(ep)
        if meta.get("source") != "real":
            continue
        Ps = proj_matrices(meta)
        fps = float(meta["fps"])
        t0 = time.time()
        per_cam = []
        for c in range(2):
            tr = HandTracker("right")
            hands = []
            for k, f in enumerate(read_video(ep / f"cam{c}.avi")):
                d = tr.detect(f, k * 1000.0 / fps)
                hands.append([] if d is None else [h["img_px"] for h in d["all"]])
            tr.close()
            per_cam.append(hands)
        T = min(len(per_cam[0]), len(per_cam[1]))
        out = [np.full((T, 21, 2), np.nan), np.full((T, 21, 2), np.nan)]
        found = 0
        for k in range(T):
            best = None
            for h0 in per_cam[0][k]:
                for h1 in per_cam[1][k]:
                    _, e = triangulate(Ps, [h0, h1])
                    m = float(np.median(e))
                    if np.isfinite(m) and (best is None or m < best[0]):
                        best = (m, h0, h1)
            if best is not None and best[0] <= a.max_err:
                out[0][k], out[1][k] = best[1], best[2]
                found += 1
        np.savez_compressed(ep / "hands2d.npz", cam0=out[0], cam1=out[1])
        print(f"{ep.name}: {T}장 중 두 화면 모두 손 {found}장 ({found / max(1, T):.0%}), {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
