"""학습한 Point Policy 를 시뮬 PiPER 로 평가 (2026-11-27).

박자마다 (초당 policy_hz = 6, 논문과 같음):
  관측 = 시뮬 집게 자세로 만든 로봇 점 5개 + 시뮬 물체 점 (정육면체 윗면 가운데·판 가운데) + 집게 상태, 지난 H 걸음
  정책 → 앞으로 C 걸음의 로봇 점·집게 → 지난 박자들이 같은 때를 두고 낸 값과 지수 가중 평균 (논문: exponential temporal averaging)
  → 다음 걸음 로봇 점에 강체를 맞춰 자세를 되찾음 (Kabsch) → 역기구학 → 1/6 초 동안 관절을 곧게 옮김
정육면체·판 자리는 시범과 같은 범위에서 새로 뽑는다 (--seed 로 가짜 시범과 다른 자리). 정책 계산은 1080 Ti 에서 수 ms 라 지연은 넣지 않았다.

  .venv/bin/python point_policy/sim_eval.py point_policy/runs/synth40/policy.pt --n 50 [--gif-dir 그림폴더 --gifs 3]
"""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from common import load_task
from model import PointPolicy
from points import pose_from_points, robot_points
from sim_scene import PiperSim, sample_layout


class Runner:
    def __init__(self, ckpt, dev=None, m=0.1):
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        c = d["cfg"]
        self.cfg = c
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PointPolicy(c["n_robot"], c["n_obj"], c["hist"], c["chunk"], c["d"], c["layers"]).to(self.dev)
        self.net.load_state_dict(d["state"])
        self.net.eval()
        self.mean, self.std = np.array(c["mean"], np.float32), np.array(c["std"], np.float32)
        self.m = m

    def reset(self, pts0, grip0=0.0):
        H = self.cfg["hist"]
        self.hist = deque([pts0.copy()] * H, maxlen=H)
        self.ghist = deque([grip0] * H, maxlen=H)
        self.preds = []                                       # (만든 걸음, 점 (C,5,3), 집게 확률 (C,))
        self.k = 0

    @torch.no_grad()
    def step(self, pts, grip):
        """pts (5+K, 3) 지금 관측, grip 지금 집게 → 다음 걸음 로봇 점 (5,3), 집게 닫힘 확률."""
        self.hist.append(pts.copy())
        self.ghist.append(float(grip))
        P = (np.stack(self.hist, axis=1) - self.mean) / self.std           # (n, H, 3)
        G = np.array(self.ghist, np.float32)
        fut, gl = self.net(torch.from_numpy(P[None].astype(np.float32)).to(self.dev),
                           torch.from_numpy(G[None]).to(self.dev))
        fut = fut[0].cpu().numpy().transpose(1, 0, 2) * self.std + self.mean  # (C, 5, 3)
        gp = torch.sigmoid(gl[0]).cpu().numpy()
        self.preds.append((self.k, fut, gp))
        C = self.cfg["chunk"]
        self.preds = [p for p in self.preds if self.k - p[0] < C]
        # 다음 걸음 (k+1) 을 두고 낸 예측들: 걸음 j 에서 낸 것의 (k − j) 번째. ACT 처럼 오래된 것에 w = exp(−m·i), i = 0 이 가장 오래됨
        items = sorted(self.preds, key=lambda p: p[0])
        w = np.exp(-self.m * np.arange(len(items)))
        w /= w.sum()
        pt = sum(wi * p[1][self.k - p[0]] for wi, p in zip(w, items))
        g = float(sum(wi * p[2][self.k - p[0]] for wi, p in zip(w, items)))
        self.k += 1
        return pt, g


def run_episode(sim, run, task, cube_xy, pad_xy, max_s=20.0, frames=None, wh=(480, 360), cam="front"):
    dt = 1.0 / task["policy_hz"]
    sim.reset(cube_xy, pad_xy, q0=sim.start_q())
    p, R = sim.ee_pose()
    obs = np.concatenate([robot_points(p, R), sim.object_points()])
    run.reset(obs, 0.0)
    lifted, done_k, ik_bad, res_max = False, 0, 0, 0.0
    for k in range(int(max_s / dt)):
        p, R = sim.ee_pose()
        obs = np.concatenate([robot_points(p, R), sim.object_points()])
        pt, g = run.step(obs, float(sim.closed))
        pos, Rt, res = pose_from_points(pt)
        res_max = max(res_max, res)
        q, err = sim.ik(pos, Rt)
        ik_bad += err > 0.01
        sim.move(q, g > 0.5, dt)
        lifted |= sim.cube_lifted()
        if frames is not None:
            frames.append(sim.render(cam, *wh))
        # 끝: 놓았고, 성공 자리이고, 집게가 정육면체보다 4 cm 넘게 위에 1초
        ee = sim.ee_pose()[0]
        if lifted and sim.success() and ee[2] > sim.cube_pos()[2] + 0.04:
            done_k += 1
            if done_k >= task["policy_hz"]:
                break
        else:
            done_k = 0
    sim.closed = False
    sim.run(0.5)
    c, pp = sim.cube_pos(), sim.pad_pos()
    return {"success": sim.success(), "lifted": bool(lifted), "seconds": round(sim.t, 1),
            "cube_to_pad_cm": round(100 * float(np.linalg.norm(c[:2] - pp[:2])), 1),
            "ik_far": int(ik_bad), "rigid_res_mm": round(1000 * res_max, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--task", default=None)
    ap.add_argument("--ik", choices=["numpy", "placo"], default="numpy", help="placo 는 point_policy/.venv 로 실행")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--m", type=float, default=0.1, help="시간 평균 지수 (클수록 새 예측보다 옛 예측을 덜 씀… ACT 와 같은 꼴)")
    ap.add_argument("--gif-dir", default=None)
    ap.add_argument("--gifs", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Runner(a.ckpt, m=a.m)
    task = load_task(a.task) if a.task else run.cfg["task"]
    sim = PiperSim(task, ik=a.ik)
    rng = np.random.default_rng(a.seed)
    results = []
    for e in range(a.n):
        cube_xy, pad_xy = sample_layout(task, rng)
        frames = [] if (a.gif_dir and e < a.gifs) else None
        r = run_episode(sim, run, task, cube_xy, pad_xy, a.max_s, frames)
        r.update(ep=e, cube_xy=np.round(cube_xy, 3).tolist(), pad_xy=np.round(pad_xy, 3).tolist())
        results.append(r)
        print(json.dumps(r), flush=True)
        if frames:
            import imageio
            Path(a.gif_dir).mkdir(parents=True, exist_ok=True)
            gp = Path(a.gif_dir) / f"eval_{e:02d}_{'ok' if r['success'] else 'fail'}.gif"
            imageio.mimsave(gp, frames, duration=1.0 / task["policy_hz"], loop=0)
    n_ok = sum(r["success"] for r in results)
    summ = {"ckpt": a.ckpt, "n": a.n, "success": n_ok, "rate": round(n_ok / a.n, 3),
            "lifted": sum(r["lifted"] for r in results), "seed": a.seed}
    print(json.dumps(summ))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summ, "episodes": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
