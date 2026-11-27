"""물체 점 따라가기(object_points.py) 시험용: 가짜 시범을 두 '진짜 카메라 자리'에서 본 영상으로 그린다 (2026-11-27).

시뮬에서 손 대신 PiPER 집게가 같은 길로 움직이고 정육면체가 따라 움직인다 (손은 MediaPipe 가 못 찾으니 물체 점 시험에만 씀).
카메라는 webcam_piper/calib.json 의 두 카메라 자리·초점거리 (시뮬 카메라로 옮김, sim_scene.cv_camera).
기준 시범의 첫 장면 점은 참값 + 1.5 px 잡음을 '사람이 찍은 것'으로 넣는다 (annotations.json).
참값 화면 위치는 objects2d_true.npz 로 남겨, object_points.py 가 낸 objects2d.npz 와 비교한다.

  ~/smolVLA/.venv/bin/python point_policy/render_synth_video.py point_policy/data/cube_to_pad/synth --n 6 --out point_policy/data/cube_to_pad/synth_video
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import TABLE_Z, load_task
from episode_io import calibs, list_episodes, load_meta, save_meta
from piper_kin import DOWN, HOME_Q
from sim_scene import PiperSim, cv_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--task", default=None)
    ap.add_argument("--click-px", type=float, default=1.5)
    ap.add_argument("--click-all", action="store_true", help="모든 시범 첫 장면을 '직접 찍은 것'으로 (참값 + 잡음)")
    a = ap.parse_args()
    task = load_task(a.task)
    eps = list_episodes(a.src)[:a.n]
    meta0 = load_meta(eps[0])
    cals = calibs(meta0)
    cams = {f"cam{c}": cv_camera(cals[c], task) for c in range(2)}
    sim = PiperSim(task, cams=cams)
    W, H = cals[0].w, cals[0].h
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rng = np.random.default_rng(5)
    origin = np.asarray(task["desk_origin_robot"])
    h = task["cube_size"] / 2
    import mujoco
    for ep in eps:
        meta = load_meta(ep)
        gt = np.load(ep / "truth.npy")
        dst = out / ep.name
        dst.mkdir()
        for f in ("hands2d.npz", "times.npy", "truth.npy"):
            shutil.copy(ep / f, dst / f)
        meta["source"] = "synthetic_render"
        save_meta(dst, meta)
        sim.reset(meta["cube_xy"], meta["pad_xy"], settle=0.0)
        writers = [cv2.VideoWriter(str(dst / f"cam{c}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), meta["fps"], (W, H))
                   for c in range(2)]
        q = HOME_Q.copy()
        true2d = [[], []]
        d, m = sim.data, sim.model
        for k in range(len(gt)):
            pos, gap, cube = gt[k, :3], gt[k, 3], gt[k, 4:7]
            q = sim.kin.ik(pos, DOWN, q, iters=40)
            d.qpos[sim.jq] = q
            wdt = float(np.clip(gap - 0.012, 0.0, 0.07)) / 2
            d.qpos[sim.fq] = [wdt, -wdt]
            a_ = sim.cube_qadr
            d.qpos[a_:a_ + 3] = cube
            d.qpos[a_ + 3:a_ + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(m, d)
            O = np.array([cube + [0, 0, h], [meta["pad_xy"][0], meta["pad_xy"][1], TABLE_Z]])
            for c in range(2):
                img = sim.render(f"cam{c}", W, H)
                writers[c].write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                true2d[c].append(cals[c].project(O - origin))
        for w_ in writers:
            w_.release()
        np.savez_compressed(dst / "objects2d_true.npz", cam0=np.array(true2d[0]), cam1=np.array(true2d[1]))
        print(dst.name, len(gt), "장")
    ref = eps[0].name
    t0 = np.load(out / ref / "objects2d_true.npz")
    clicked = [e.name for e in eps] if a.click_all else [ref]
    ann = {"names": [op["name"] for op in task["object_points"]], "reference": ref, "episodes": {},
           "note": "가짜: 참값 + 잡음을 사람이 찍은 점으로"}
    for name in clicked:
        tt = np.load(out / name / "objects2d_true.npz")
        ann["episodes"][name] = {f"cam{c}": (tt[f"cam{c}"][0] + rng.normal(0, a.click_px, (2, 2))).tolist() for c in range(2)}
    (out / "annotations.json").write_text(json.dumps(ann, indent=1), encoding="utf-8")
    # 첫 장면에 참값 점을 그린 그림 (카메라 자리가 맞는지 눈으로)
    for c in range(2):
        cap = cv2.VideoCapture(str(out / ref / f"cam{c}.avi"))
        _, f = cap.read()
        cap.release()
        for p in t0[f"cam{c}"][0]:
            cv2.drawMarker(f, (int(p[0]), int(p[1])), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
        cv2.imwrite(str(out / f"check_cam{c}.jpg"), f)


if __name__ == "__main__":
    main()
