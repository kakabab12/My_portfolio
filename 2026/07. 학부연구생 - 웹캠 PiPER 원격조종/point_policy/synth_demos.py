"""가짜 사람 시범 만들기 (카메라 없이 전체 길을 시험하려고, 2026-11-27).

진짜 시범과 같은 꼴(episode_io)로 낸다: 손 21점과 물체 점을 '두 카메라 화면 px' 로 만들어 잡음을 얹는다.
그래서 뒤의 삼각측량 → 로봇 점 → 학습 → 시뮬 평가가 진짜 녹화 때와 똑같은 코드로 돈다.

사람 손 움직임 (각본):
  시작 자리(+ 위 10 cm) → 정육면체 위 7 cm → 내려가 손가락 끝 가운데가 정육면체 가운데 높이 → 손가락 오므림
  → 7 cm 들어 올림 → 판 위로 → 내려놓음 → 손가락 벌림 → 조금 들어 올림
  구간마다 빠르기는 초당 12~22 cm 사이에서 뽑고 최소 저크(처음·끝 속도 0)로 잇는다. 손목은 천천히 ±6° 흔들린다.
정육면체는 손가락을 다 오므린 뒤부터 놓을 때까지 손에 붙어 움직인다 (물리 없이, 사람 시범이라).
카메라는 webcam_piper/calib.json 의 진짜 두 카메라 자리 (11/18 ABKO 두 대, 1280x960).
잡음: 손 점 화면 3 px (11/18 연구실 기록에서 두 화면 어긋남 가운데 6.5 px), 손을 못 찾는 장면 3%, 물체 점 1 px.

  .venv/bin/python point_policy/synth_demos.py --n 40 --out point_policy/data/cube_to_pad/synth
"""
import argparse
import json
import math
import shutil

import numpy as np

import common
from common import TABLE_Z, load_task
from episode_io import new_episode_dir, save_meta
from geometry import TableCalib
from piper_kin import DOWN
from sim_scene import sample_layout

# 손 21점 (집게 끝 사이트 좌표와 같은 축: x = 엄지 ↔ 검지 줄, z = 손목 → 손끝 (아래), m). 손끝 둘은 따로 놓는다.
_HAND_LOCAL = {
    0: (0.000, -0.060, -0.095),                                                  # 손목
    1: (0.020, -0.045, -0.080), 2: (0.032, -0.030, -0.060), 3: (0.035, -0.015, -0.035),  # 엄지 뿌리 → 마디
    5: (-0.022, -0.012, -0.070), 6: (-0.030, -0.005, -0.045), 7: (-0.030, 0.000, -0.022),  # 검지
    9: (-0.006, -0.016, -0.075), 10: (-0.012, -0.008, -0.050), 11: (-0.014, -0.004, -0.032), 12: (-0.015, -0.002, -0.020),
    13: (0.010, -0.020, -0.073), 14: (0.005, -0.014, -0.052), 15: (0.003, -0.010, -0.036), 16: (0.002, -0.008, -0.026),
    17: (0.024, -0.026, -0.068), 18: (0.020, -0.022, -0.052), 19: (0.018, -0.019, -0.042), 20: (0.017, -0.017, -0.034),
}


def hand21(center, R, gap):
    """집기점·손 방향·손가락 사이 → 손 21점 (로봇 좌표)."""
    out = np.zeros((21, 3))
    for i, p in _HAND_LOCAL.items():
        out[i] = center + R @ np.asarray(p)
    out[4] = center + R @ np.array([gap / 2, 0.0, 0.0])        # 엄지 끝
    out[8] = center + R @ np.array([-gap / 2, 0.0, 0.0])       # 검지 끝
    return out


def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def script(task, cube_xy, pad_xy, rng, fps):
    """장마다 (집기점, 손 방향, 손가락 사이, 정육면체 가운데)."""
    h = task["cube_size"] / 2
    z_grasp = TABLE_Z + h
    above = 0.07
    start = np.asarray(task["start_pos"]) + np.r_[rng.normal(0, 0.015, 2), rng.normal(0, 0.01)]
    cube = np.r_[cube_xy, z_grasp]
    pad = np.r_[pad_xy, z_grasp + 0.004]
    g_open = rng.uniform(0.09, 0.11)
    g_closed = task["cube_size"] + rng.uniform(0.008, 0.016)
    # (목표 집기점, 손가락 사이, 머무는 초)
    grasp_xy = cube_xy + rng.normal(0, 0.004, 2)
    place_xy = pad_xy + rng.normal(0, 0.008, 2)
    keys = [
        (np.r_[grasp_xy, z_grasp + above], g_open, rng.uniform(0.0, 0.2)),
        (np.r_[grasp_xy, z_grasp], g_open, rng.uniform(0.05, 0.2)),
        ("close", g_closed, rng.uniform(0.15, 0.35)),
        (np.r_[grasp_xy, z_grasp + above], g_closed, 0.0),
        (np.r_[place_xy, z_grasp + above], g_closed, rng.uniform(0.0, 0.15)),
        (np.r_[place_xy, pad[2]], g_closed, rng.uniform(0.05, 0.2)),
        ("open", g_open, rng.uniform(0.15, 0.3)),
        (np.r_[place_xy, pad[2] + 0.06], g_open, rng.uniform(0.2, 0.5)),
    ]
    frames = []
    pos, gap = start.copy(), g_open
    held, cube_c, grip_off = False, cube.copy(), None
    # 처음 조금 가만히 (사람이 자리 잡는 때)
    for _ in range(int(rng.uniform(0.3, 0.6) * fps)):
        frames.append((pos.copy(), gap, held))
    for target, g_target, dwell in keys:
        if isinstance(target, str):
            n = int(rng.uniform(0.25, 0.4) * fps)
            g0 = gap
            for i in range(n):
                gap = g0 + min_jerk((i + 1) / n) * (g_target - g0)
                frames.append((pos.copy(), gap, held))
            held = target == "close"
        else:
            d = float(np.linalg.norm(target - pos))
            v = rng.uniform(0.12, 0.22)
            T = max(0.35, d / v * 1.875)               # 최소 저크의 가장 빠른 때 = 평균의 1.875배
            n = max(2, int(T * fps))
            p0 = pos.copy()
            for i in range(n):
                pos = p0 + min_jerk((i + 1) / n) * (target - p0)
                frames.append((pos.copy(), gap, held))
        for _ in range(int(dwell * fps)):
            frames.append((pos.copy(), gap, held))
    # 손목 흔들림 (천천히) 과 정육면체
    T = len(frames)
    t = np.arange(T) / fps
    yaw0 = rng.normal(0, math.radians(4))
    wob = np.zeros(T)
    for k in range(3):
        f = rng.uniform(0.1, 0.5)
        wob += rng.normal(0, math.radians(2.5)) * np.sin(2 * math.pi * f * t + rng.uniform(0, 6.3))
    out = []
    prev_held = False
    for k, (p, g, hd) in enumerate(frames):
        R = rot_z(yaw0 + wob[k]) @ DOWN
        if hd and not prev_held:
            grip_off = cube_c - p
        if hd:
            cube_c = p + grip_off
        elif prev_held:
            cube_c = np.r_[cube_c[:2], TABLE_Z + h]              # 놓으면 책상(판) 위로
        prev_held = hd
        out.append((p, R, g, cube_c.copy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib", default=str(common.WEBCAM / "calib.json"))
    ap.add_argument("--hand-px", type=float, default=3.0)
    ap.add_argument("--obj-px", type=float, default=1.0)
    ap.add_argument("--drop", type=float, default=0.03)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()
    task = load_task(a.task)
    fps = a.fps or task["record_fps"]
    out = a.out or str(common.DATA / task["name"] / "synth")
    if a.clear:
        shutil.rmtree(out, ignore_errors=True)
    saved = json.loads(open(a.calib, encoding="utf-8").read())
    cal_d = [saved["table"], saved["table2"]]
    cals = [TableCalib.from_dict(c) for c in cal_d]
    origin = np.asarray(task["desk_origin_robot"])
    rng = np.random.default_rng(a.seed)

    def project(cal, pts_robot, sigma):
        px = cal.project(np.asarray(pts_robot) - origin)
        return px + rng.normal(0, sigma, px.shape)

    for e in range(a.n):
        cube_xy, pad_xy = sample_layout(task, rng)
        fr = script(task, cube_xy, pad_xy, rng, fps)
        T = len(fr)
        hands = [np.zeros((T, 21, 2)), np.zeros((T, 21, 2))]
        objs = [np.zeros((T, 2, 2)), np.zeros((T, 2, 2))]
        pad_pt = np.r_[pad_xy, TABLE_Z]
        for k, (p, R, g, cube_c) in enumerate(fr):
            H = hand21(p, R, g)
            O = np.array([cube_c + [0, 0, task["cube_size"] / 2], pad_pt])
            for c in range(2):
                hands[c][k] = project(cals[c], H, a.hand_px)
                if rng.random() < a.drop:
                    hands[c][k] = np.nan
                objs[c][k] = project(cals[c], O, a.obj_px)
        ep = new_episode_dir(out)
        np.savez_compressed(ep / "hands2d.npz", cam0=hands[0], cam1=hands[1])
        np.savez_compressed(ep / "objects2d.npz", cam0=objs[0], cam1=objs[1],
                            vis0=np.ones((T, 2), bool), vis1=np.ones((T, 2), bool))
        np.save(ep / "times.npy", np.arange(T) / fps)
        gt = np.array([np.r_[p, g, c] for p, R, g, c in fr])
        np.save(ep / "truth.npy", gt)                     # 가짜만: 참 집기점(3)·손가락 사이(1)·정육면체 가운데(3)
        save_meta(ep, {"source": "synthetic", "task": task["name"], "fps": fps, "n_frames": T,
                       "calib": cal_d, "cube_xy": cube_xy.tolist(), "pad_xy": pad_xy.tolist(),
                       "object_points": [op["name"] for op in task["object_points"]],
                       "noise": {"hand_px": a.hand_px, "obj_px": a.obj_px, "drop": a.drop}})
    print(f"가짜 시범 {a.n}개 → {out}")


if __name__ == "__main__":
    main()
