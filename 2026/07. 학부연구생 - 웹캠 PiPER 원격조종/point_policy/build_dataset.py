"""시범들 → 학습 자료 (논문 3장, 2026-11-27).

시범마다:
  1) 두 화면의 손 21점·물체 점을 삼각측량 (webcam_piper geometry.triangulate, 논문 부록 B1 의 DLT)
     두 화면 어긋남이 max_err_px 보다 크면 버리고, 빈 장면은 앞뒤로 곧게 메운다. 시간으로 조금 고른다 (가운데 평균 5장).
  2) 책상 좌표 → 시뮬 로봇 좌표 (원점만 옮김)
  3) 논문 식: 위치 = 엄지·검지 끝 가운데, 방향 = 첫 장면 손바닥 → 지금 손바닥 회전 × 첫 로봇 방향(아래), 집게 = 7 cm 보다 가까우면 닫힘
  4) PiPER 에 맞춘 것 (논문에 없음): 그 자세를 역기구학으로 풀어 PiPER 가 실제로 할 수 있는 자세로 바꾼다
     (위치 1순위, 방향 2순위. PiPER 는 6축이라 집게를 아래로 세우면 책상 위 11 cm 까지만 닿음)
  5) 로봇 점 5개 (points.ROBOT_OFFSETS), 초당 policy_hz 장으로 줄이기 (논문: 20 Hz → 6 Hz)

  .venv/bin/python point_policy/build_dataset.py point_policy/data/cube_to_pad/synth
"""
import argparse
import json
from pathlib import Path

import numpy as np

import common
from common import load_task
from episode_io import list_episodes, load_meta, proj_matrices
from geometry import triangulate
from piper_kin import HOME_Q, PiperKin
from points import HandToRobot, desk_to_robot, robot_points


def fill_gaps(X, max_gap=None):
    """(T, ...) 에서 NaN 인 장면을 시간으로 곧게 메움. 앞뒤 끝은 가장 가까운 값."""
    X = np.array(X, np.float64)
    T = len(X)
    flat = X.reshape(T, -1)
    ok = ~np.isnan(flat).any(axis=1)
    if ok.sum() == 0:
        return X, ok
    t = np.arange(T)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, t[ok], flat[ok, j])
    return flat.reshape(X.shape), ok


def smooth(X, k=5):
    if k <= 1:
        return X
    pad = k // 2
    Xp = np.concatenate([np.repeat(X[:1], pad, 0), X, np.repeat(X[-1:], pad, 0)])
    c = np.cumsum(Xp, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c])
    return (c[k:] - c[:-k]) / k


def tri_track(Ps, a, b, max_err):
    """두 화면 점 (T, n, 2) → (T, n, 3) 책상 좌표, 어긋남 큰 점은 NaN."""
    T, n = a.shape[:2]
    X = np.full((T, n, 3), np.nan)
    errs = []
    for k in range(T):
        good = ~(np.isnan(a[k]).any(axis=1) | np.isnan(b[k]).any(axis=1))
        if not good.any():
            continue
        Xk, e = triangulate(Ps, [a[k][good], b[k][good]])
        Xk[e > max_err] = np.nan
        X[k][good] = Xk
        errs.extend(e[np.isfinite(e)].tolist())
    return X, np.array(errs)


def process(ep, task, kin, max_err=25.0, smooth_k=5, solver=None):
    meta = load_meta(ep)
    Ps = proj_matrices(meta)
    fps = float(meta["fps"])
    hz = float(task["policy_hz"])
    hands = np.load(ep / "hands2d.npz")
    objs = np.load(ep / "objects2d.npz")
    H, herr = tri_track(Ps, hands["cam0"], hands["cam1"], max_err)
    O, oerr = tri_track(Ps, objs["cam0"], objs["cam1"], max_err)
    # 손을 처음·마지막으로 찾은 장면 사이만 쓴다
    hand_ok = ~np.isnan(H).any(axis=(1, 2))
    if hand_ok.sum() < 10:
        return None, {"ep": ep.name, "skip": "손을 찾은 장면이 10장 안됨"}
    i0, i1 = np.argmax(hand_ok), len(hand_ok) - np.argmax(hand_ok[::-1])
    H, O = H[i0:i1], O[i0:i1]
    H, ok_h = fill_gaps(H)
    O, ok_o = fill_gaps(O)
    H = smooth(H.reshape(len(H), -1), smooth_k).reshape(H.shape)
    O = smooth(O.reshape(len(O), -1), smooth_k).reshape(O.shape)
    H = desk_to_robot(H, task)
    O = desk_to_robot(O, task)
    step = max(1, int(round(fps / hz)))
    idx = np.arange(0, len(H), step)
    h2r = HandToRobot(task)
    h2r(np.mean(H[:min(5, len(H))], axis=0))                       # 첫 손바닥 = 처음 5장 평균
    q = HOME_Q.copy()
    rp, grip, pos_l, ik_err, rot_err, qs = [], [], [], [], [], []
    for k in idx:
        pos, R, closed, gap = h2r(H[k])
        q = (solver or kin).ik(pos, R, q, iters=60)
        p_r, R_r = kin.fk(q)
        ik_err.append(float(np.linalg.norm(p_r - pos)))
        rot_err.append(float(np.degrees(np.arccos(np.clip((np.trace(R_r.T @ R) - 1) / 2, -1, 1)))))
        rp.append(robot_points(p_r, R_r))
        grip.append(float(closed))
        pos_l.append(pos)
        qs.append(q.copy())
    info = {"ep": ep.name, "frames": int(len(H)), "steps": int(len(idx)), "seconds": round(len(H) / fps, 2),
            "hand_found": round(float(ok_h.mean()), 3),
            "tri_err_px_median": round(float(np.median(herr)), 2) if len(herr) else None,
            "ik_err_mm_max": round(1000 * max(ik_err), 1), "rot_err_deg_max": round(max(rot_err), 1),
            "closed_frac": round(float(np.mean(grip)), 3)}
    if (ep / "truth.npy").exists():                                  # 가짜 시범: 참값과 비교
        gt = np.load(ep / "truth.npy")[i0:i1][idx]
        info["pinch_err_mm_median"] = round(1000 * float(np.median(np.linalg.norm(np.array(pos_l) - gt[:, :3], axis=1))), 1)
        info["grip_agree"] = round(float(np.mean((gt[:, 3] < task["grip_close_m"]) == np.array(grip, bool))), 3)
    out = {"robot_pts": np.array(rp), "grip": np.array(grip), "obj_pts": O[idx], "q": np.array(qs)}
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--task", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-err", type=float, default=25.0)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--ik", choices=["numpy", "placo"], default="numpy", help="placo 는 point_policy/.venv 로 실행")
    a = ap.parse_args()
    task = load_task(a.task)
    kin = PiperKin()
    if a.ik == "placo":
        from placo_ik import PlacoIK
        solver = PlacoIK()
    else:
        solver = kin
    eps = [e for f in a.folders for e in list_episodes(f)]
    parts, infos = [], []
    for ep in eps:
        d, info = process(ep, task, kin, a.max_err, a.smooth, solver)
        infos.append(info)
        if d is not None:
            parts.append(d)
        print(json.dumps(info, ensure_ascii=False))
    if not parts:
        raise SystemExit("쓸 수 있는 시범이 없음")
    ends = np.cumsum([len(p["grip"]) for p in parts])
    out = Path(a.out) if a.out else Path(a.folders[0]) / "dataset.npz"
    np.savez_compressed(out, robot_pts=np.concatenate([p["robot_pts"] for p in parts]),
                        grip=np.concatenate([p["grip"] for p in parts]),
                        obj_pts=np.concatenate([p["obj_pts"] for p in parts]),
                        q=np.concatenate([p["q"] for p in parts]), episode_ends=ends,
                        task=json.dumps({k: v for k, v in task.items() if not k.startswith("_")}, ensure_ascii=False),
                        ik=a.ik,
                        sources=json.dumps([str(e) for e in eps]))
    (out.parent / (out.stem + "_info.json")).write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}: 시범 {len(parts)}개, {int(ends[-1])} 걸음 (초당 {task['policy_hz']})")


if __name__ == "__main__":
    main()
