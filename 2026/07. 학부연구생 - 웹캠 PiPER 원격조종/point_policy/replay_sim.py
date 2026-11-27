"""사람 시범에서 만든 로봇 궤적을 시뮬 PiPER 로 그대로 재생 (학습 없이, 2026-11-27).

시범이 쓸 만한지 보는 검사: 첫 장면의 물체 점(삼각측량)으로 시뮬에 정육면체·판을 같은 자리에 놓고,
build_dataset 이 푼 관절(q)·집게를 초당 policy_hz 로 보낸다. 정육면체가 판 위에 놓이면 성공.
사람 손으로는 잘 집었는데 여기서 실패하면, 손 → 로봇 바꾸기(높이·집게 7 cm 기준·방향)가 그 시범에 안 맞는 것.

  .venv/bin/python point_policy/replay_sim.py point_policy/data/cube_to_pad/synth/dataset.npz [--gif out.gif --ep 0]
"""
import argparse
import json

import numpy as np

from common import load_task
from sim_scene import PiperSim


def episodes(ds):
    ends = ds["episode_ends"]
    starts = np.r_[0, ends[:-1]]
    return list(zip(starts, ends))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--task", default=None)
    ap.add_argument("--ik", choices=["numpy", "placo"], default="numpy", help="placo 는 point_policy/.venv 로 실행")
    ap.add_argument("--ep", type=int, nargs="*", default=None)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--hold", type=float, default=1.0, help="끝난 뒤 더 기다리는 초")
    a = ap.parse_args()
    task = load_task(a.task)
    ds = np.load(a.dataset)
    sim = PiperSim(task, ik=a.ik)
    dt = 1.0 / task["policy_hz"]
    h = task["cube_size"] / 2
    names = [op["name"] for op in task["object_points"]]
    ic, ip = names.index("cube_top"), names.index("pad_center")
    res, frames = [], []
    sel = a.ep if a.ep else range(len(episodes(ds)))
    for e in sel:
        s, t = episodes(ds)[e]
        q, g, O = ds["q"][s:t], ds["grip"][s:t], ds["obj_pts"][s:t]
        sim.reset(O[0, ic, :2], O[0, ip, :2], q0=q[0])
        lifted = False
        for k in range(len(q)):
            sim.move(q[k], g[k] > 0.5, dt)
            lifted |= sim.cube_lifted()
            if a.gif and len(sel) == 1:
                frames.append(sim.render("front", 480, 360))
        sim.closed = False
        sim.run(a.hold)
        ok = sim.success()
        c, p = sim.cube_pos(), sim.pad_pos()
        res.append({"ep": int(e), "success": ok, "lifted": bool(lifted),
                    "cube_to_pad_cm": round(100 * float(np.linalg.norm(c[:2] - p[:2])), 1)})
        print(json.dumps(res[-1]))
    n = sum(r["success"] for r in res)
    print(f"재생 성공 {n}/{len(res)}, 들어 올림 {sum(r['lifted'] for r in res)}/{len(res)}")
    if a.gif and frames:
        import imageio
        imageio.mimsave(a.gif, frames[::1], duration=dt, loop=0)
        print("→", a.gif)


if __name__ == "__main__":
    main()
