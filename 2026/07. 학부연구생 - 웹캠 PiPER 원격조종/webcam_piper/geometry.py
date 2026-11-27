"""웹캠 화면 ↔ 사람 책상 좌표 (2026-11-15).

사람 책상 좌표 (m): 원점 = A4 종이 한가운데(시작 표시), x = 앞 (사람에게서 멀어지는 쪽, 노트북 쪽),
y = 사람의 왼쪽, z = 위. A4 는 가로로 놓는다 (사람 기준 좌우 297 mm, 앞뒤 210 mm).

1) 종이 네 모서리를 화면에서 누르면 → 초점거리를 추정하고(모르면) 카메라가 책상을 어디서 보는지(자세)를 구한다.
2) MediaPipe 손 랜드마크 21점은 화면 위치(px)와 "손 기준 3D 좌표(m)" 두 가지가 나온다. 둘을 맞추면(solvePnP)
   카메라에서 손까지의 3D 위치가 나온다. 다만 MediaPipe 는 평균 손 크기를 가정해서 거리가 몇 % 틀릴 수 있다.
3) 그래서 시작할 때 손을 종이 위에 펴서 올려놓고 한 번 재서, 손 21점이 책상 바로 위에 오도록 거리 배율 s 를 맞춘다.

쥐는 손 (11/16 사용자: "손가락 전체를 사용하는 게 더 편할 것 같다")
  기본(whole): 엄지 끝 ↔ 나머지 네 손가락 끝의 가운데. 컵을 쥐듯이 손 전체로 쥔다.
  집기점 = 두 점의 가운데, 손가락 사이 = 두 점 거리, 집게 방향 = 엄지 → 네 손가락 줄과 같은 줄로 집게가 닫히게.
  pinch: 엄지 끝 ↔ 검지 끝 (예전 방식, 작은 물체용).
"""
import math

import cv2
import numpy as np

A4_W, A4_D = 0.297, 0.210            # 사람 기준 좌우, 앞뒤 (m)
# 모서리 누르는 순서 (사람 자리에서 본 방향): 1 내 쪽-왼쪽, 2 내 쪽-오른쪽, 3 먼 쪽-오른쪽, 4 먼 쪽-왼쪽
CORNERS_TABLE = np.array([[-A4_D / 2, +A4_W / 2, 0.0],
                          [-A4_D / 2, -A4_W / 2, 0.0],
                          [+A4_D / 2, -A4_W / 2, 0.0],
                          [+A4_D / 2, +A4_W / 2, 0.0]])
CORNER_NAMES = ["1 near-left", "2 near-right", "3 far-right", "4 far-left"]
PINCH_H0 = 0.012                     # 사람 시작점(종이 + 위 집기점) 높이 = 로봇 시작점의 집게 끝 높이 (m)

# MediaPipe 손 랜드마크 번호
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
GRIPS = {"whole": (8, 12, 16, 20), "pinch": (8,)}       # 엄지 맞은편 손가락 끝들
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, 13, PINKY_MCP)      # 손목·네 손가락 뿌리: 쥐어도 거의 안 변하는 손바닥 점 (11/18)
# 손을 종이에 펴서 올려놓았을 때 21점의 높이 어림 (m). MediaPipe 점은 관절 가운데라 그 자리 살 두께의 절반쯤.
# 손목 1.8, 손가락 뿌리 1.3, 가운데 마디 1.0, 끝 마디 0.8, 손끝 0.6 cm (어림값, 실제 손으로 잰 것 아님)
FLAT_H = np.array([0.018,
                   0.018, 0.014, 0.010, 0.008,
                   0.013, 0.010, 0.008, 0.006,
                   0.013, 0.010, 0.008, 0.006,
                   0.013, 0.010, 0.008, 0.006,
                   0.012, 0.009, 0.007, 0.006])


def intrinsics(f, w, h):
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def focal_from_hfov(hfov_deg, w):
    return (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def focal_candidates(img_pts, w, h):
    """종이 한 장(평면) 한 장면에서 초점거리 후보 (Zhang 방식, 주점 = 화면 가운데, 정사각 화소 가정).
    호모그래피 H = K [r1 r2 t] 에서 r1·r2 = 0, |r1| = |r2| 두 식을 f 에 대해 따로 푼 값 (풀린 것만, 0.4w~3w).
    모서리 순서가 A4 가로 비율과 맞으면 두 값이 거의 같고, 틀리면 어긋난다 (paper.py 가 순서 고를 때 씀, 11/18)."""
    src = CORNERS_TABLE[:, :2].astype(np.float64)
    H, _ = cv2.findHomography(src, np.asarray(img_pts, np.float64))
    if H is None:
        return []
    cx, cy = w / 2.0, h / 2.0
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    Hc = T @ H                                    # 주점을 원점으로 옮김 → K = diag(f, f, 1)
    h1, h2 = Hc[:, 0], Hc[:, 1]
    cands = []
    # r1·r2 = 0 → (h11 h12 + h21 h22)/f² + h31 h32 = 0
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
        if f2 > 0:
            cands.append(math.sqrt(f2))
    # |r1|² = |r2|² → (h11² + h21² − h12² − h22²)/f² + h31² − h32² = 0
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-12:
        f2 = -((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / den
        if f2 > 0:
            cands.append(math.sqrt(f2))
    return [f for f in cands if 0.4 * w < f < 3.0 * w]


def corner_reproj_px(img_pts, f, w, h):
    """초점거리 f 로 종이 자세를 풀었을 때 네 모서리가 화면 점과 가장 많이 어긋난 거리 (px)."""
    K = intrinsics(f, w, h)
    pts = np.asarray(img_pts, np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(CORNERS_TABLE, pts, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return float("inf")
    p, _ = cv2.projectPoints(CORNERS_TABLE, rvec, tvec, K, None)
    return float(np.max(np.linalg.norm(p.reshape(-1, 2) - pts, axis=1)))


def estimate_focal(img_pts, w, h):
    """후보(와 두 후보의 평균) 중 네 모서리에 가장 잘 맞는 것. 못 풀면 None.
    11/18: 예전에는 두 후보를 그냥 평균했는데, 카메라가 종이를 거의 정면에서 보면 r1·r2 = 0 쪽 식이 잘 안 풀려
    (가짜 화면: 참 914 px 에 후보 908·1028) 평균이 6% 틀렸다. 어긋남이 가장 작은 것을 고른다."""
    cands = focal_candidates(img_pts, w, h)
    if not cands:
        return None
    if len(cands) == 2:
        cands.append(float(np.mean(cands)))
    return float(min(cands, key=lambda f: corner_reproj_px(img_pts, f, w, h)))


class TableCalib:
    """종이 네 모서리 화면 좌표 → 카메라 내부값 K 와 카메라 자세 (책상 → 카메라)."""

    def __init__(self, img_pts, w, h, hfov=None):
        self.img_pts = np.asarray(img_pts, np.float64).reshape(4, 2)
        self.w, self.h = int(w), int(h)
        f = focal_from_hfov(hfov, w) if hfov else estimate_focal(self.img_pts, w, h)
        self.focal_source = "hfov" if hfov else ("estimate" if f else "default 70deg")
        if not f:
            f = focal_from_hfov(70.0, w)
        self.f = float(f)
        self.K = intrinsics(self.f, w, h)
        ok, rvec, tvec = cv2.solvePnP(CORNERS_TABLE, self.img_pts, self.K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise RuntimeError("카메라 자세를 못 구함 — 모서리를 다시 눌러 주세요")
        self.R = cv2.Rodrigues(rvec)[0]           # 책상 → 카메라 회전
        self.t = tvec.reshape(3)
        proj = self.project(CORNERS_TABLE)
        self.reproj_px = float(np.max(np.linalg.norm(proj - self.img_pts, axis=1)))
        self.cam_pos = -self.R.T @ self.t         # 카메라 위치 (책상 좌표)

    def project(self, pts_table):
        p, _ = cv2.projectPoints(np.asarray(pts_table, np.float64).reshape(-1, 3),
                                 cv2.Rodrigues(self.R)[0], self.t, self.K, None)
        return p.reshape(-1, 2)

    def cam_to_table(self, Xc):
        return self.R.T @ (np.asarray(Xc, np.float64) - self.t)

    def to_dict(self):
        return {"img_pts": self.img_pts.tolist(), "w": self.w, "h": self.h, "f": self.f,
                "focal_source": self.focal_source, "reproj_px": self.reproj_px,
                "cam_pos_table_m": self.cam_pos.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["img_pts"], d["w"], d["h"], hfov=None if d.get("focal_source") != "hfov"
                   else math.degrees(2 * math.atan((d["w"] / 2) / d["f"])))


def hand_pose_cam(world_lm, img_px, K):
    """MediaPipe 손 3D(손 기준, m) 21점 + 화면 21점 → 손 기준 → 카메라 (R, t). 실패하면 None."""
    obj = np.asarray(world_lm, np.float64).reshape(-1, 3)
    img = np.asarray(img_px, np.float64).reshape(-1, 2)
    flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        return None
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


def hand_trans_cam(world_lm, img_px, K, prev=None, beta=0.0):
    """OAT (Chen et al., ICRA 2026) 식 (2): MediaPipe 손 3D 가 카메라와 같은 방향이라고 보고 이동 p 만 푼다.
    화면 정규 좌표 x_i = (X_i + px)/(Z_i + pz) 를 x_i (Z_i + pz) = X_i + px 로 바꾸면 p 에 대해 선형이라 한 번에 풀린다.
    prev 가 있으면 beta·(p − prev) 줄을 더해 앞 장면에서 크게 벗어나지 않게 한다 (화면 점이 잘 못 잡는 깊이 쪽이 주로 눌림).
    11/17 MediaPipe 1.2.3 예시 사진에서 3D 와 화면의 손 방향이 1~5° 안으로 맞는 것을 확인. 돌려줌: (21점 카메라 좌표, p) 또는 None."""
    world = np.asarray(world_lm, np.float64).reshape(-1, 3)
    n = np.c_[np.asarray(img_px, np.float64).reshape(-1, 2), np.ones(len(world))] @ np.linalg.inv(K).T
    A = np.zeros((2 * len(world), 3))
    A[0::2, 0], A[1::2, 1] = 1.0, 1.0
    A[0::2, 2], A[1::2, 2] = -n[:, 0], -n[:, 1]
    b = np.empty(2 * len(world))
    b[0::2] = n[:, 0] * world[:, 2] - world[:, 0]
    b[1::2] = n[:, 1] * world[:, 2] - world[:, 1]
    if prev is not None and beta > 0:
        A = np.vstack([A, beta * np.eye(3)])
        b = np.r_[b, beta * np.asarray(prev, np.float64)]
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    if p[2] + world[:, 2].min() <= 0.05:                    # 카메라 뒤나 코앞이면 실패
        return None
    return world + p, p


def backproject_height(calib, img_px, heights):
    """화면 점 → 책상 좌표. 점 i 가 책상 위 heights[i] 높이에 있다고 보고, 카메라에서 그 점으로 뻗은 광선이 그 높이와 만나는 곳.
    손을 종이에 펴서 올려놓았을 때 21점의 진짜 3D 자리를 MediaPipe 손 크기 가정 없이 구한다 (11/18)."""
    uv1 = np.c_[np.asarray(img_px, np.float64).reshape(-1, 2), np.ones(len(heights))]
    d = (uv1 @ np.linalg.inv(calib.K).T) @ calib.R            # 광선 방향 (책상 좌표) = Rᵀ K⁻¹ [u v 1]
    dz = np.where(np.abs(d[:, 2]) < 1e-9, np.nan, d[:, 2])
    lam = (np.asarray(heights, np.float64) - calib.cam_pos[2]) / dz
    return calib.cam_pos + lam[:, None] * d


def kabsch(A, B):
    """가운데를 뺀 점 A, B (n×3) 에서 R @ A_i ≈ B_i 인 회전 R (거울 없음)."""
    U, _, Vt = np.linalg.svd(np.asarray(A).T @ np.asarray(B))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0)])
    return Vt.T @ D @ U.T


def palm_size(pts21):
    """손바닥 크기 (m): 검지 뿌리 ↔ 새끼 뿌리 폭, 손목 ↔ 가운데 손가락 뿌리 길이. 보정이 그럴듯한지 보는 데 쓴다."""
    p = np.asarray(pts21, np.float64).reshape(-1, 3)
    return float(np.linalg.norm(p[INDEX_MCP] - p[PINKY_MCP])), float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]))


def hand_palm_cam(world_lm, img_px, K, tpl):
    """손바닥 다섯 점으로만 거리를 잰다 (11/18, 11/16 실물에서 손 높이가 책상 아래 5~20 cm 로 나온 것을 고치려고).
    손가락은 쥘 때마다 모양이 바뀌어 MediaPipe 손 3D 가 틀리기 쉽고, 21점 전체로 풀면 그 틀림이 카메라 방향 거리로 커진다.
    손바닥(손목, 네 손가락 뿌리)은 쥐어도 거의 안 변하므로, 보정 때 종이 위에서 잰 진짜 손바닥 tpl (5×3, 가운데 뺀 것, m) 을 쓴다.
      1) MediaPipe 손바닥 다섯 점에 tpl 을 겹치는 회전 → 손바닥이 카메라에 어떻게 놓였는지 (MediaPipe 3D 는 카메라와 같은 방향)
      2) 그렇게 돌린 진짜 손바닥이 화면 다섯 점에 맞는 위치 p (OAT 식 (2) 와 같은 직선 풀이, 이동만)
      3) 손가락은 MediaPipe 3D 를 진짜 손바닥 크기에 맞춰 늘이거나 줄여 그 손바닥에 붙인다 (손가락 모양 틀림은 그만큼만 남음)
    돌려줌: (21점 카메라 좌표, p) 또는 None."""
    world = np.asarray(world_lm, np.float64).reshape(-1, 3)
    W = world[list(PALM)]
    wc = W.mean(axis=0)
    X = (kabsch(tpl, W - wc) @ np.asarray(tpl).T).T                  # 진짜 손바닥, 카메라 방향, 가운데가 원점
    n = np.c_[np.asarray(img_px, np.float64).reshape(-1, 2)[list(PALM)], np.ones(len(PALM))] @ np.linalg.inv(K).T
    A = np.zeros((2 * len(PALM), 3))
    A[0::2, 0], A[1::2, 1] = 1.0, 1.0
    A[0::2, 2], A[1::2, 2] = -n[:, 0], -n[:, 1]
    b = np.empty(2 * len(PALM))
    b[0::2] = n[:, 0] * X[:, 2] - X[:, 0]
    b[1::2] = n[:, 1] * X[:, 2] - X[:, 1]
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    if p[2] + X[:, 2].min() <= 0.05:                              # 카메라 뒤나 코앞이면 실패
        return None
    k = math.sqrt(float((X ** 2).sum()) / max(float(((W - wc) ** 2).sum()), 1e-12))   # 진짜 손바닥 / MediaPipe 손바닥 크기
    return p + k * (world - wc), p


def scale_for_start(calib, pinch_cam, h0=PINCH_H0):
    """시작 자세에서 잰 집기점(카메라 좌표)이 책상 위 높이 h0 에 오도록 하는 거리 배율 s.
    카메라 원점에서 뻗은 같은 직선 위를 s 배 늘이거나 줄인다: z_table(s·p) = h0."""
    r3 = calib.R[:, 2]                           # 책상 z 축의 카메라 좌표 방향
    den = float(r3 @ pinch_cam)
    if abs(den) < 1e-9:
        return None
    s = (h0 + float(r3 @ calib.t)) / den
    return s if 0.5 < s < 2.0 else None


def flat_fit(calib, lm_cam, heights=FLAT_H):
    """손을 종이에 펴서 올려놓은 자세: 21점(카메라 좌표, 배율 1)의 책상 높이가 heights 에 가장 가깝게 하는 배율 s 와,
    그때 높이가 어림에서 벗어난 정도(RMS, m). z_table(s·p) = s·(r3·p) − r3·t 이므로 최소제곱 s = Σ a(h + b) / Σ a²."""
    r3 = calib.R[:, 2]
    a = np.asarray(lm_cam, np.float64).reshape(-1, 3) @ r3
    b = float(r3 @ calib.t)
    den = float(a @ a)
    if den < 1e-12:
        return None, float("inf")
    s = float(a @ (np.asarray(heights) + b)) / den
    rms = float(np.sqrt(np.mean((s * a - b - heights) ** 2)))
    return s, rms


def scale_for_flat(calib, lm_cam, heights=FLAT_H):
    s, _ = flat_fit(calib, lm_cam, heights)
    return s if s is not None and 0.5 < s < 2.0 else None


def proj_matrix(calib):
    """책상 좌표 → 화면 px 투영 행렬 P = K [R | t] (3×4)."""
    return calib.K @ np.c_[calib.R, calib.t]


def triangulate(Ps, pts):
    """카메라 여러 대의 같은 점들 → 책상 좌표 3D (DLT, 교수님 논문 Point Policy 부록 B1 과 같은 식, 11/18).
    카메라마다 x(p3·X) − p1·X = 0, y(p3·X) − p2·X = 0 두 줄을 쌓은 AX = 0 을 SVD 로 푼다 (가장 작은 특이값의 벡터).
    Ps: [3×4 투영 행렬], pts: [(n,2) 화면 점] → ((n,3) 책상 좌표, (n,) 카메라 중 가장 크게 어긋난 재투영 거리 px)."""
    pts = [np.asarray(p, np.float64).reshape(-1, 2) for p in pts]
    n = len(pts[0])
    X = np.full((n, 3), np.nan)
    err = np.full(n, np.inf)
    for i in range(n):
        A = []
        for P, p in zip(Ps, pts):
            A.append(p[i, 0] * P[2] - P[0])
            A.append(p[i, 1] * P[2] - P[1])
        A = np.array(A)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        Xh = np.linalg.svd(A)[2][-1]
        if abs(Xh[3]) < 1e-12:
            continue
        X[i] = Xh[:3] / Xh[3]
        e = 0.0
        for P, p in zip(Ps, pts):
            q = P @ np.r_[X[i], 1.0]
            e = max(e, float(np.linalg.norm(q[:2] / q[2] - p[i])) if q[2] > 0 else float("inf"))
        err[i] = e
    return X, err


# 11/23 손바닥이 향한 쪽: teleop 이 --hand 에 맞춰 바꾼다 (오른손 +1, 왼손 −1). 오른손 손바닥을 아래로 펴면 (0, 0, −1)
HAND_SIGN = 1.0


def palm_normal(tab, sign=None):
    """손 21점 (책상 좌표) → 손바닥이 향한 쪽 단위 벡터 (11/23). 손등이 보이면(손바닥이 아래) z < 0, 손바닥이 위면 z > 0.
    손목·검지 뿌리·새끼 뿌리 세 점의 면 (쥐어도 거의 안 변하는 점). 11/18 연구실 기록에서 손바닥을 아래로 했을 때 z −0.8~−0.9 로 확인."""
    sign = HAND_SIGN if sign is None else sign
    n = np.cross(tab[INDEX_MCP] - tab[WRIST], tab[PINKY_MCP] - tab[WRIST]) * sign
    k = float(np.linalg.norm(n))
    return n / k if k > 1e-9 else np.array([0.0, 0.0, -1.0])


def hand_summary(tab, tips):
    """손 21점 (책상 좌표) → 집기점·집게 방향(yaw)·손가락 사이·손바닥 가운데 (HandMeasure 와 같은 식).
    11/23: 손바닥이 향한 쪽(normal)과 엄지 → 손가락 줄 단위 벡터(axis)도 낸다 (집게를 위로도 돌리기)."""
    thumb, other = tab[THUMB_TIP], tab[list(tips)].mean(axis=0)
    axis = other - thumb                                                        # 엄지 → 맞은편 손가락
    # 로봇 집게는 yaw 0 에서 로봇 y 축(좌우)으로 닫힌다. 엄지 → 손가락 줄과 같은 줄로 닫히게 하는 yaw (180° 마다 같음)
    yaw = math.atan2(axis[1], axis[0]) - math.pi / 2
    gap = float(np.linalg.norm(axis))
    return {"pinch": 0.5 * (thumb + other), "yaw": yaw, "gap": gap,
            "palm": 0.5 * (tab[WRIST] + tab[MIDDLE_MCP]),
            "normal": palm_normal(tab), "axis": axis / gap if gap > 1e-9 else np.array([0.0, 1.0, 0.0])}


class StereoHand:
    """카메라 두 대의 손 21점 → 삼각측량한 책상 좌표 손 (11/18, ABKO APC930 두 대).
    MediaPipe 손 크기 가정도, 손 펴서 재는 배율도 필요 없다 (두 광선이 만나는 곳이 곧 거리).
    두 화면의 손이 같은 손이 아니거나(다른 손을 고름) 크게 틀리면 재투영이 커지므로 max_err_px 로 거른다.
    11/18 연구실 첫 기록 1098장면: MediaPipe 가 두 화면에서 같은 마디를 조금씩 다른 데 잡아 어긋남 가운데 6.5 px, 90% 16 px
    (보정을 다듬어도 6.8 → 5.9 px 로만 줄어 카메라 보정 탓이 아님). 이 거리(약 70 cm)에서 7 px ≈ 5 mm, 25 px ≈ 2 cm.
    기준 12 px 이면 84%, 25 px 이면 96% 를 받는다 → 25."""

    def __init__(self, calib1, calib2, grip="whole", max_err_px=25.0):
        self.calibs = (calib1, calib2)
        self.Ps = [proj_matrix(calib1), proj_matrix(calib2)]
        self.tips = list(GRIPS[grip])
        self.max_err = max_err_px
        self.last_err = None

    def __call__(self, img1, img2):
        X, err = triangulate(self.Ps, [img1, img2])
        self.last_err = float(np.median(err))
        if not np.isfinite(self.last_err) or self.last_err > self.max_err:
            return None
        out = hand_summary(X, self.tips)
        out["tab21"] = X
        out["err_px"] = self.last_err
        return out


class FlatHandGate:
    """손 보정 저절로 시작 (11/18): 손이 종이 + 위에 펴진 채 가만히 있으면 True.
      - 손바닥 가운데(손목과 가운데 손가락 뿌리 사이)를 종이 높이로 내린 자리가 + 에서 center_m 안
      - 손목 → 가운데 손가락 뿌리가 노트북 쪽(책상 +x) 을 향함 (±angle_deg)
      - 편 손: MediaPipe 3D 21점이 '종이에 편 손' 높이(FLAT_H)에 rms_m 안으로 맞음 (거울상 두 경우 중 잘 맞는 쪽)
      - 가만히: hold 장면 동안 손바닥 다섯 점의 평균 자리가 still_px 안에서만 움직임
        (점 하나하나로 보면 MediaPipe 의 1 px 흔들림에도 걸려서 11/18 가짜 카메라 시험에서 끝내 시작하지 않았음)
    부를 때마다 (됐는지, 화면에 띄울 짧은 영어 안내) 를 돌려준다. 손이 들려 있으면 종이 높이로 내린 자리가
    카메라 반대쪽으로 크게 밀려나 + 에서 벗어나므로 '종이 위에 올려놓음' 도 같이 걸러진다."""

    def __init__(self, calib, solver="oat", hold=12, still_px=4.0, center_m=0.04, angle_deg=40.0, rms_m=0.012):
        self.calib, self.solver = calib, solver
        self.hold, self.still, self.center, self.angle, self.rms_max = hold, still_px, center_m, angle_deg, rms_m
        self.hist = []
        self.last_rms = None

    def reset(self):
        self.hist = []

    def __call__(self, world_lm, img_px):
        img = np.asarray(img_px, np.float64).reshape(21, 2)
        flat = backproject_height(self.calib, img, FLAT_H)
        if not np.all(np.isfinite(flat)):
            self.reset()
            return False, "show your hand over the paper"
        palm = 0.5 * (flat[WRIST] + flat[MIDDLE_MCP])
        off = float(np.linalg.norm(palm[:2]))
        if off > self.center:
            self.reset()
            return False, f"palm center onto the cross ({off * 100:.0f} cm off)"
        d = flat[MIDDLE_MCP] - flat[WRIST]
        if abs(math.degrees(math.atan2(d[1], d[0]))) > self.angle:
            self.reset()
            return False, "point your fingers toward the laptop"
        rms = float("inf")
        for mirror in (False, True):
            r = HandMeasure(self.calib, 1.0, "whole", mirror, self.solver)(world_lm, img)
            if r is not None:
                s, e = flat_fit(self.calib, r["lm_cam_raw"])
                if s is not None and 0.6 < s < 1.6:
                    rms = min(rms, e)
        self.last_rms = rms
        if rms > self.rms_max:
            self.reset()
            return False, "lay the hand FLAT, fingers straight" + (f" ({rms * 1000:.0f} mm)" if np.isfinite(rms) else "")
        c = img[list(PALM)].mean(axis=0)
        self.hist = [h for h in self.hist if np.linalg.norm(h - c) <= self.still] + [c]
        if len(self.hist) < self.hold:
            return False, f"hold still {len(self.hist)}/{self.hold}"
        return True, "ok"


class HandMeasure:
    """한 장면의 손 → 사람 책상 좌표의 집기점, 집게 방향(yaw), 손가락 사이 거리, 손바닥 가운데."""

    def __init__(self, calib, scale=1.0, grip="whole", mirror=False, solver="pnp", beta=0.0, flat=None):
        self.calib = calib
        self.scale = scale
        self.tips = list(GRIPS[grip])
        self.mirror = mirror        # MediaPipe 손 3D 가 화면과 거울상이면 z 를 뒤집어 쓴다 (보정 때 판별)
        # "pnp": 회전+이동 (11/15~), "oat": 이동만 + 앞 장면 벌점 beta (11/17, OAT 논문),
        # "palm": 손바닥 다섯 점으로 거리 (11/18 기본). palm 은 flat (종이에 편 손 21점의 책상 좌표, 보정 때 잼) 이 있어야 하고 배율은 1
        self.solver = solver
        self.beta = beta
        self.prev_p = None
        self.flat = None if flat is None else np.asarray(flat, np.float64).reshape(21, 3)
        if solver == "palm":
            if self.flat is None or not np.all(np.isfinite(self.flat)):
                raise ValueError("palm 방식은 손바닥 보정(flat)이 있어야 함 — 손을 종이에 펴서 S")
            tpl = self.flat[list(PALM)]
            self.palm_tpl = tpl - tpl.mean(axis=0)

    def reset(self):
        """손을 놓쳤을 때: 앞 장면 벌점을 끊는다."""
        self.prev_p = None

    def __call__(self, world_lm, img_px):
        world = np.asarray(world_lm, np.float64).reshape(-1, 3)
        if self.mirror:
            world = world * np.array([1.0, 1.0, -1.0])
        if self.solver == "oat":
            r = hand_trans_cam(world, img_px, self.calib.K, self.prev_p, self.beta)
            if r is None:
                return None
            cam, self.prev_p = r
        elif self.solver == "palm":
            r = hand_palm_cam(world, img_px, self.calib.K, self.palm_tpl)
            if r is None:
                return None
            cam = r[0]
        else:
            pose = hand_pose_cam(world, img_px, self.calib.K)
            if pose is None:
                return None
            Rh, th = pose
            cam = (Rh @ world.T).T + th                                             # 21점 카메라 좌표 (배율 1)
        tab = (self.calib.R.T @ (self.scale * cam - self.calib.t).T).T              # 책상 좌표
        out = hand_summary(tab, self.tips)
        out["lm_cam_raw"] = cam
        return out
