"""카메라 두 대 보정 다듬기 (2026-11-18).

종이 모서리 네 점만으로는 초점거리·렌즈 왜곡을 잘 못 잡아서, 두 화면의 손 점이 삼각측량에서 서로 어긋난다 (연구실 첫 시험 15~29 px).
그래서 손을 이리저리 움직이며 모은 '두 화면 손 21점 짝' (teleop.py --status 폴더의 pairs.jsonl) 으로
두 카메라의 초점거리, 렌즈 왜곡(k1, k2), 자세를 함께 맞춘다 (번들 조정. 3D 점은 매번 삼각측량으로 없앰).
종이 모서리 네 점(책상 좌표를 알고 있음)이 크기와 방향을 붙잡는다.

  ~/smolVLA/chanhyuk_piper-teleop/.venv/bin/python refine_stereo.py 폴더/pairs.jsonl            # 숫자만
  ~/smolVLA/chanhyuk_piper-teleop/.venv/bin/python refine_stereo.py 폴더/pairs.jsonl --save     # calib.json 에 씀
scipy 가 필요하다 (찬혁님 .venv 에 있음).
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

from geometry import CORNERS_TABLE

HERE = os.path.dirname(os.path.abspath(__file__))


def load_pairs(path, max_n=600, min_gap_px=4.0):
    """pairs.jsonl → (n,21,2) 두 화면. 거의 같은 자리는 빼서 고르게 (손을 멈춘 동안 같은 장면이 쌓이지 않게)."""
    A, B, last = [], [], None
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        a = np.array(d["l1"], float).reshape(21, 2)
        b = np.array(d["l2"], float).reshape(21, 2)
        if last is not None and np.median(np.linalg.norm(a - last, axis=1)) < min_gap_px:
            continue
        last = a
        A.append(a)
        B.append(b)
    A, B = np.array(A), np.array(B)
    if len(A) > max_n:
        k = np.linspace(0, len(A) - 1, max_n).astype(int)
        A, B = A[k], B[k]
    return A, B


def kmat(f, w, h):
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def unpack(p, w, h):
    cams = []
    for i in range(2):
        q = p[i * 9:(i + 1) * 9]
        cams.append((kmat(q[0], w, h), np.array([q[1], q[2], 0, 0, 0.0]), q[3:6].copy(), q[6:9].copy()))
    return cams


def undist(K, dist, pts):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(-1, 2)


def tri(Ps, uv1, uv2):
    """DLT 를 한꺼번에 (n 점): 정규화한 4×4 → SVD."""
    rows = np.stack([uv1[:, :1] * Ps[0][2] - Ps[0][0], uv1[:, 1:] * Ps[0][2] - Ps[0][1],
                     uv2[:, :1] * Ps[1][2] - Ps[1][0], uv2[:, 1:] * Ps[1][2] - Ps[1][1]], axis=1)
    rows /= np.linalg.norm(rows, axis=2, keepdims=True)
    X = np.linalg.svd(rows)[2][:, -1]
    return X[:, :3] / X[:, 3:4]


def residuals(p, A, B, corners, w, h, wc=5.0):
    cams = unpack(p, w, h)
    res = []
    # 종이 모서리 (책상 좌표를 앎)
    for (K, dist, rv, tv), c in zip(cams, corners):
        pr, _ = cv2.projectPoints(CORNERS_TABLE, rv, tv, K, dist)
        res.append(wc * (pr.reshape(-1, 2) - c).ravel())
    # 손 점: 왜곡을 편 점으로 삼각측량 → 각 카메라로 다시 비춤
    Ps = [K @ np.c_[cv2.Rodrigues(rv)[0], tv] for K, _, rv, tv in cams]
    uv1 = undist(cams[0][0], cams[0][1], A.reshape(-1, 2))
    uv2 = undist(cams[1][0], cams[1][1], B.reshape(-1, 2))
    X = tri(Ps, uv1, uv2)
    for (K, dist, rv, tv), obs in zip(cams, (A, B)):
        pr, _ = cv2.projectPoints(X, rv, tv, K, dist)
        res.append((pr.reshape(-1, 2) - obs.reshape(-1, 2)).ravel())
    return np.concatenate(res)


def hand_err(p, A, B, w, h):
    """손 점 하나마다 두 카메라 중 큰 재투영 어긋남 (px) — teleop 의 StereoHand 와 같은 잣대."""
    cams = unpack(p, w, h)
    Ps = [K @ np.c_[cv2.Rodrigues(rv)[0], tv] for K, _, rv, tv in cams]
    uv1 = undist(cams[0][0], cams[0][1], A.reshape(-1, 2))
    uv2 = undist(cams[1][0], cams[1][1], B.reshape(-1, 2))
    X = tri(Ps, uv1, uv2)
    e = []
    for (K, dist, rv, tv), obs in zip(cams, (A, B)):
        pr, _ = cv2.projectPoints(X, rv, tv, K, dist)
        e.append(np.linalg.norm(pr.reshape(-1, 2) - obs.reshape(-1, 2), axis=1))
    return np.maximum(*e).reshape(-1, 21), X.reshape(-1, 21, 3)


def start_params(cal, w, h):
    K = kmat(cal["f"], w, h)
    pts = np.array(cal["img_pts"], float)
    ok, rv, tv = cv2.solvePnP(CORNERS_TABLE, pts, K, None, flags=cv2.SOLVEPNP_IPPE)
    return np.r_[cal["f"], 0.0, 0.0, rv.ravel(), tv.ravel()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--calib", default=os.path.join(HERE, "calib.json"))
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    c = json.load(open(a.calib))
    w, h = c["table"]["w"], c["table"]["h"]
    A, B = load_pairs(a.pairs)
    print(f"손 짝 {len(A)}장 (21점씩)")
    corners = [np.array(c["table"]["img_pts"], float), np.array(c["table2"]["img_pts"], float)]
    p0 = np.r_[start_params(c["table"], w, h), start_params(c["table2"], w, h)]
    e0, X0 = hand_err(p0, A, B, w, h)
    lo = np.tile(np.r_[300.0, -1.0, -1.0, [-np.inf] * 6], 2)
    hi = np.tile(np.r_[3000.0, 1.0, 1.0, [np.inf] * 6], 2)
    r = least_squares(residuals, p0, args=(A, B, corners, w, h), loss="soft_l1", f_scale=3.0,
                      bounds=(lo, hi), x_scale="jac", max_nfev=400)
    e1, X1 = hand_err(r.x, A, B, w, h)
    names = ["손목", "엄지1", "엄지2", "엄지3", "엄지끝", "검지뿌리", "검지2", "검지3", "검지끝", "중지뿌리", "중지2", "중지3", "중지끝",
             "약지뿌리", "약지2", "약지3", "약지끝", "새끼뿌리", "새끼2", "새끼3", "새끼끝"]
    print(f"두 화면 어긋남 (장면마다 21점 중간값의 중간값): {np.median(np.median(e0, 1)):.1f} → {np.median(np.median(e1, 1)):.1f} px, "
          f"장면 90%: {np.percentile(np.median(e0, 1), 90):.1f} → {np.percentile(np.median(e1, 1), 90):.1f} px")
    per = np.median(e1, 0)
    print("점마다 (다듬은 뒤 중간값 px): " + ", ".join(f"{n} {v:.1f}" for n, v in zip(names, per)))
    for i, nm in enumerate(("카메라 1", "카메라 2")):
        q = r.x[i * 9:(i + 1) * 9]
        R = cv2.Rodrigues(q[3:6])[0]
        pos = -R.T @ q[6:9]
        print(f"{nm}: 초점 {p0[i * 9]:.0f} → {q[0]:.0f} px, 왜곡 k1 {q[1]:+.3f} k2 {q[2]:+.3f}, 카메라 자리 {np.round(pos * 100, 1).tolist()} cm")
    cams = unpack(r.x, w, h)
    for (K, dist, rv, tv), cc, nm in zip(cams, corners, ("카메라 1", "카메라 2")):
        pr, _ = cv2.projectPoints(CORNERS_TABLE, rv, tv, K, dist)
        print(f"{nm} 종이 모서리 어긋남 최대 {np.max(np.linalg.norm(pr.reshape(-1, 2) - cc, axis=1)):.1f} px")
    z = X1[..., 2] * 100
    print(f"손 높이 범위 (다듬은 뒤, 손목~손끝 전체): {np.percentile(z, 1):.1f} ~ {np.percentile(z, 99):.1f} cm")
    if a.save:
        for key, i in (("table", 0), ("table2", 1)):
            q = r.x[i * 9:(i + 1) * 9]
            c[key].update({"f": float(q[0]), "focal_source": "refined", "dist": [float(q[1]), float(q[2]), 0.0, 0.0, 0.0],
                           "rvec": q[3:6].tolist(), "tvec": q[6:9].tolist()})
        c["refined"] = {"pairs": len(A), "err_before_px": float(np.median(np.median(e0, 1))),
                        "err_after_px": float(np.median(np.median(e1, 1)))}
        json.dump(c, open(a.calib, "w"), indent=1)
        print("저장:", a.calib)


if __name__ == "__main__":
    main()
