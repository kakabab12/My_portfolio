"""녹화 영상 → 두 화면의 물체 점 궤적 (논문 3장 B: 한 번 찍은 점 → 다른 시범 첫 장면으로 옮김 → CoTracker 로 따라감, 2026-11-27).

  1) annotations.json: 사람이 기준 시범 첫 장면에 찍은 점 (annotate.py). 직접 찍은 시범은 그 점을 그대로 쓴다.
  2) 다른 시범 첫 장면으로 옮기기: 논문은 DIFT(Stable Diffusion 2.1). 공식 SD 2.1 이 내려가서 DINOv2-base(facebook/dinov2-base).
     첫 장면에는 물체가 책상에 놓여 있으므로 기준 시범에서 잰 점 높이에서 책상 위 격자를 두 카메라에 비춰,
     두 화면 유사도 합이 가장 큰 곳 둘레의 무게중심을 고른다 (plane_sweep, 11/27: 화면마다 따로 찾으면 무늬 없는 판에서 30 cm 넘게 빗나감).
  3) CoTracker3 (facebook/cotracker3 scaled_offline, 논문과 같은 모델 계열) 로 첫 장면 점을 영상 끝까지 따라간다.
     손에 가려도 이어서 낸다 (보이는지 vis 도 남김).
결과는 objects2d.npz (episode_io 꼴). 이미 있으면 건너뛴다 (--redo).

  ~/smolVLA/.venv/bin/python point_policy/object_points.py point_policy/data/cube_to_pad/real
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import common  # noqa: F401
from common import load_task
from episode_io import list_episodes, load_meta, proj_matrices
from geometry import triangulate


def read_frames(path, stride=1):
    cap = cv2.VideoCapture(str(path))
    out, k = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % stride == 0:
            out.append(f)
        k += 1
    cap.release()
    return out


class Dino:
    def __init__(self, dev, width=896):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("facebook/dinov2-base", use_safetensors=True).to(dev).eval()
        self.dev, self.width = dev, width
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(self, bgr):
        h, w = bgr.shape[:2]
        W = self.width
        H = int(round(h * W / w / 14)) * 14
        img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H))
        x = torch.from_numpy(img).to(self.dev).permute(2, 0, 1)[None].float() / 255.0
        x = (x - self.mean) / self.std
        tok = self.m(pixel_values=x).last_hidden_state[0, 1:]           # (n, 768)
        f = tok.reshape(H // 14, W // 14, -1).permute(2, 0, 1)            # (768, h, w)
        return torch.nn.functional.normalize(f, dim=0), (w / W, h / H)

    def sim_maps(self, ref_img, ref_pts, tgt_img):
        """기준 점마다 대상 장면 전체 화면(px)의 코사인 유사도 지도 (K, h, w), float16 GPU."""
        fr, (sx, sy) = self.feats(ref_img)
        ft, _ = self.feats(tgt_img)
        C, h, w = fr.shape
        H, W = tgt_img.shape[:2]
        maps = []
        for x, y in ref_pts:
            gx = (x / sx) / 14 - 0.5
            gy = (y / sy) / 14 - 0.5
            grid = torch.tensor([[[[2 * gx / (w - 1) - 1, 2 * gy / (h - 1) - 1]]]], device=fr.device, dtype=fr.dtype)
            v = torch.nn.functional.normalize(
                torch.nn.functional.grid_sample(fr[None], grid, align_corners=True)[0, :, 0, 0], dim=0)
            sim = torch.einsum("c,chw->hw", v, ft)
            maps.append(torch.nn.functional.interpolate(sim[None, None], size=(H, W), mode="bilinear",
                                                        align_corners=False)[0, 0])
        return torch.stack(maps)

    def transfer(self, ref_img, ref_pts, tgt_img):
        fr, (sx, sy) = self.feats(ref_img)
        ft, _ = self.feats(tgt_img)
        C, h, w = fr.shape
        out, conf = [], []
        for x, y in ref_pts:
            # 기준 점 자리의 특징 (조각 격자에서 이중선형)
            gx = (x / sx) / 14 - 0.5
            gy = (y / sy) / 14 - 0.5
            grid = torch.tensor([[[[2 * gx / (w - 1) - 1, 2 * gy / (h - 1) - 1]]]], device=fr.device, dtype=fr.dtype)
            v = torch.nn.functional.grid_sample(fr[None], grid, align_corners=True)[0, :, 0, 0]
            v = torch.nn.functional.normalize(v, dim=0)
            sim = torch.einsum("c,chw->hw", v, ft)
            up = torch.nn.functional.interpolate(sim[None, None], scale_factor=14, mode="bilinear", align_corners=False)[0, 0]
            i = int(torch.argmax(up))
            yy, xx = divmod(i, up.shape[1])
            out.append(((xx + 0.5) * sx, (yy + 0.5) * sy))
            conf.append(float(up.max()))
        return np.array(out), np.array(conf)


def plane_sweep(maps2, Ps, z, area=((-0.30, 0.35), (-0.35, 0.35)), step=0.003, radius=0.04, delta=0.05):
    """두 카메라 유사도 지도를 높이 z 의 책상 위 격자에 비춰 더한다 → 가장 비슷한 곳 둘레의 무게중심 (책상 좌표 3D).
    첫 장면에는 물체가 책상에 놓여 있어 기준 시범에서 잰 높이 z 를 그대로 쓸 수 있다 (11/27).
    두 화면을 한 3D 점으로 묶으니 두 화면 점이 처음부터 서로 맞는다."""
    xs = np.arange(area[0][0], area[0][1], step)
    ys = np.arange(area[1][0], area[1][1], step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P3 = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z), np.ones(X.size)]
    score = np.zeros(X.size)
    for m, P in zip(maps2, Ps):
        H, W = m.shape
        q = P3 @ P.T
        u, v = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
        ok = (q[:, 2] > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        mm = m.float().cpu().numpy()
        val = np.full(X.size, -2.0)
        ui, vi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(v).astype(int), 0, H - 1)
        val[ok] = mm[vi[ok], ui[ok]]
        score += val
    i = int(np.argmax(score))
    best = P3[i, :3]
    # 가장 비슷한 곳 둘레 radius 안에서, 두 화면 평균 유사도가 꼭대기보다 delta 안쪽인 자리들의 가운데.
    # 무늬 없는 판은 판 전체가 비슷하게 나오므로 그 가운데가 판 가운데 (점마다 작업 설정 transfer_radius).
    near = (np.linalg.norm(P3[:, :2] - best[:2], axis=1) < radius) & (score >= score[i] - 2 * delta)
    c = P3[near, :3].mean(0)
    return c, float(score[i] / 2)


@torch.no_grad()
def track(cot, frames, pts, dev, stride=2, chunk=16, support=6):
    """CoTracker3: 첫 장면 점 (K,2) → (T,K,2) px, (T,K) 보임.
    오프라인 모델은 영상 전체를 한 번에 넣는다. 1080 Ti(11 GB)에서 219장이 메모리를 넘어서 (11/27)
      - 모델 안 크기(384x512)로 줄여 넣고, 특징 계산은 chunk 장씩 (fmaps_chunk_size), fp16
      - stride 장마다 하나만 따라간 뒤 사이는 곧게 메운다 (학습 자료는 초당 6장만 씀)
    공식 예제처럼 둘레에 support×support 보조 점 격자를 같이 넣는다 (따라가기가 더 안정됨)."""
    model = cot.model
    Hm, Wm = model.model_resolution
    T = len(frames)
    h, w = frames[0].shape[:2]
    idx = np.arange(0, T, stride)
    if idx[-1] != T - 1:
        idx = np.r_[idx, T - 1]
    vid = np.stack([cv2.cvtColor(cv2.resize(frames[i], (Wm, Hm), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    for i in idx])
    v = torch.from_numpy(vid).to(dev).permute(0, 3, 1, 2)[None].float()
    kx, ky = (Wm - 1) / (w - 1), (Hm - 1) / (h - 1)
    qs = [[0.0, float(x) * kx, float(y) * ky] for x, y in pts]
    if support:
        gx, gy = np.meshgrid(np.linspace(8, Wm - 8, support), np.linspace(8, Hm - 8, support))
        qs += [[0.0, float(a_), float(b_)] for a_, b_ in zip(gx.ravel(), gy.ravel())]
    q = torch.tensor(qs, device=dev, dtype=torch.float32)[None]
    with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
        tr, vis, *_ = model(video=v, queries=q, iters=6, fmaps_chunk_size=chunk)
    K = len(pts)
    tr = tr[0, :, :K].float().cpu().numpy()
    vis = vis[0, :, :K].float().cpu().numpy()
    tr[0] = np.array([[float(x) * kx, float(y) * ky] for x, y in pts])      # 첫 장면은 찍은 점 그대로 (공식 예측기와 같음)
    tr[..., 0] /= kx
    tr[..., 1] /= ky
    full = np.zeros((T, K, 2))
    fv = np.zeros((T, K))
    t = np.arange(T)
    for k in range(K):
        for d in range(2):
            full[:, k, d] = np.interp(t, idx, tr[:, k, d])
        fv[:, k] = np.interp(t, idx, vis[:, k])
    del v
    if dev == "cuda":
        torch.cuda.empty_cache()
    return full, fv > 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--task", default=None)
    ap.add_argument("--max-err", type=float, default=15.0, help="옮긴 첫 점의 두 화면 어긋남 기준 px")
    ap.add_argument("--out-name", default="objects2d.npz")
    ap.add_argument("--stride", type=int, default=2, help="CoTracker 에 몇 장마다 하나씩 넣나 (메모리)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = load_task(a.task)
    radii = [float(op.get("transfer_radius", 0.04)) for op in task["object_points"]]
    dino, cot = None, None
    for folder in a.folders:
        folder = Path(folder)
        ann = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
        # 기준 = 사람이 직접 찍은 시범들 (하나여도 되고, 2~3개면 점마다 가장 비슷한 기준을 고른다: 11/27 가짜 영상에서
        # 시작 자세 그림자를 무늬 없는 판으로 잘못 고른 적이 있어서)
        refs = []
        for name, cl in ann["episodes"].items():
            imgs = [read_frames(folder / name / f"cam{c}.avi")[0] for c in range(2)]
            pts = [np.array(cl[f"cam{c}"], float) for c in range(2)]
            X, _ = triangulate(proj_matrices(load_meta(folder / name)), pts)
            refs.append((name, imgs, pts, X[:, 2]))       # 첫 장면 물체 점 높이 (책상 좌표 m): 다른 시범 첫 장면에도 같다고 봄
        report = []
        for ep in list_episodes(folder):
            if (ep / a.out_name).exists() and not a.redo:
                continue
            t0 = time.time()
            meta = load_meta(ep)
            frames = [read_frames(ep / f"cam{c}.avi") for c in range(2)]
            T = min(len(frames[0]), len(frames[1]))
            info = {"ep": ep.name, "frames": T}
            if ep.name in ann["episodes"]:
                first = [np.array(ann["episodes"][ep.name][f"cam{c}"], float) for c in range(2)]
                info["first"] = "clicked"
            else:
                if dino is None:
                    dino = Dino(dev)
                Ps = proj_matrices(meta)
                K = len(refs[0][2][0])
                best = [None] * K                                # (유사도, 3D 점, 기준 이름)
                for name, imgs, pts, z in refs:
                    maps = [dino.sim_maps(imgs[c], pts[c], frames[c][0]) for c in range(2)]
                    for k in range(K):
                        c3, sc = plane_sweep([maps[0][k], maps[1][k]], Ps, z[k], radius=radii[k])
                        if best[k] is None or sc > best[k][0]:
                            best[k] = (sc, c3, name)
                    del maps
                pts3 = np.array([b[1] for b in best])
                sims = [round(b[0], 3) for b in best]
                info["ref"] = [b[2] for b in best]
                first = [np.array([(P @ np.r_[p, 1])[:2] / (P @ np.r_[p, 1])[2] for p in pts3]) for P in Ps]
                info["first"] = "dinov2-plane"
                info["dino_sim"] = sims
                if min(sims) < 0.7:
                    info["warn"] = f"옮긴 점 유사도 {min(sims):.2f} < 0.7: 첫 장면을 확인하고, 틀리면 annotate.py --ep {ep.name}"
            _, err = triangulate(proj_matrices(meta), [first[0], first[1]])
            info["first_stereo_px"] = np.round(err, 1).tolist()
            if np.any(err > a.max_err):
                info["warn"] = f"첫 점 두 화면 어긋남 {np.max(err):.0f}px > {a.max_err}: annotate.py --ep {ep.name} 로 직접 찍어 주세요"
            if cot is None:
                cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True).to(dev).eval()
            res = [track(cot, frames[c][:T], first[c], dev, stride=a.stride) for c in range(2)]
            np.savez_compressed(ep / a.out_name, cam0=res[0][0], cam1=res[1][0], vis0=res[0][1], vis1=res[1][1],
                                first0=first[0], first1=first[1])
            info["vis_frac"] = [round(float(r[1].mean()), 3) for r in res]
            info["sec"] = round(time.time() - t0, 1)
            if dev == "cuda":
                info["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            report.append(info)
            print(json.dumps(info, ensure_ascii=False), flush=True)
        (folder / "object_points_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
