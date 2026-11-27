"""Point Policy 학습 (2026-11-27).

자료: build_dataset.py 의 dataset.npz (초당 6 걸음, 로봇 점 5개·물체 점 K개·집게).
걸음 i 의 입력 = i−H+1 … i (앞이 모자라면 첫 값으로 채움), 정답 = i+1 … i+C 의 로봇 점·집게 (뒤가 모자라면 끝 값).
위치는 자료 전체 평균·표준편차로 고르게 한다 (축마다). 논문 표 VII: Adam 1e-4, 묶음 64.

  .venv/bin/python point_policy/train.py point_policy/data/cube_to_pad/synth/dataset.npz --out point_policy/runs/synth40
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import PointPolicy
from points import N_ROBOT


class Windows:
    def __init__(self, ds, hist, chunk):
        self.R = ds["robot_pts"].astype(np.float32)          # (N, 5, 3)
        self.O = ds["obj_pts"].astype(np.float32)            # (N, K, 3)
        self.G = ds["grip"].astype(np.float32)
        ends = ds["episode_ends"]
        starts = np.r_[0, ends[:-1]]
        hi, fi = [], []
        for s, e in zip(starts, ends):
            for i in range(s, e):
                hi.append(np.clip(np.arange(i - hist + 1, i + 1), s, e - 1))
                fi.append(np.clip(np.arange(i + 1, i + chunk + 1), s, e - 1))
        self.hi, self.fi = np.array(hi), np.array(fi)
        allp = np.concatenate([self.R.reshape(-1, 3), self.O.reshape(-1, 3)])
        self.mean, self.std = allp.mean(0), allp.std(0) + 1e-6

    def __len__(self):
        return len(self.hi)

    def batch(self, idx, rng=None, noise=0.0, shift=0.0):
        h, f = self.hi[idx], self.fi[idx]
        P = np.concatenate([self.R[h], self.O[h]], axis=2)    # (B, H, n, 3)
        Y = self.R[f]                                         # (B, C, 5, 3)
        if shift and rng is not None:                         # 장면 전체를 책상 위에서 조금 옮기기 (선택)
            d = rng.uniform(-shift, shift, (len(idx), 1, 1, 3)).astype(np.float32)
            d[..., 2] = 0
            P, Y = P + d, Y + d
        if noise and rng is not None:
            P = P + rng.normal(0, noise, P.shape).astype(np.float32)
        P = (P - self.mean) / self.std
        Y = (Y - self.mean) / self.std
        return (torch.from_numpy(P.transpose(0, 2, 1, 3).copy()), torch.from_numpy(self.G[h]),
                torch.from_numpy(Y.transpose(0, 2, 1, 3).copy()), torch.from_numpy(self.G[f]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.003, help="입력 점에 얹는 잡음 (m)")
    ap.add_argument("--shift", type=float, default=0.0, help="장면 전체를 옮기는 폭 (m, 0 이면 안 함)")
    ap.add_argument("--grip-w", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ds = np.load(a.dataset)
    W = Windows(ds, a.hist, a.chunk)
    K = ds["obj_pts"].shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = PointPolicy(N_ROBOT, K, a.hist, a.chunk, a.d, a.layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bce = torch.nn.BCEWithLogitsLoss()
    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, len(W), a.batch)
        P, G, Y, GF = (x.to(dev) for x in W.batch(idx, rng, a.noise, a.shift))
        fut, gl = net(P, G)
        l_pt = ((fut - Y) ** 2).mean()
        l_g = bce(gl, GF)
        loss = l_pt + a.grip_w * l_g
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            # 고른 값 → cm (축 평균 표준편차로 어림)
            cm = float(torch.sqrt(l_pt).item() * W.std.mean() * 100)
            log.append({"step": step, "loss_pt": round(l_pt.item(), 5), "loss_grip": round(l_g.item(), 4),
                        "rmse_cm": round(cm, 2), "sec": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    cfg = {k: getattr(a, k) for k in ("hist", "chunk", "d", "layers")}
    cfg.update(n_robot=N_ROBOT, n_obj=int(K), mean=W.mean.tolist(), std=W.std.tolist(), dataset=a.dataset,
               task=json.loads(str(ds["task"])), train=vars(a), samples=len(W))
    torch.save({"state": net.state_dict(), "cfg": cfg}, out / "policy.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("→", out / "policy.pt")


if __name__ == "__main__":
    main()
