"""A4 종이 저절로 찾기 (2026-11-18, 사용자: "보정 자동으로 하는 거 없나").

화면에서 책상보다 밝고 색이 옅은 네모를 찾고, A4 를 가로로 놓은 것으로 보아 말이 되는 것만 받는다.
  1) 밝기 문턱 몇 개로 밝은 덩어리를 나누고, 바깥 윤곽이 네모(꼭짓점 4개, 볼록)인 것을 고른다.
  2) 네 변마다 윤곽 점에 직선을 맞춰 그 교점을 모서리로 쓴다 (꼭짓점을 바로 쓰는 것보다 1~2 px 정확).
  3) 모서리 순서 여덟 가지(돌림 4 × 뒤집힘 2) 중에서
     - A4 가로 비율(297 × 210 mm)로 풀었을 때 초점거리가 웹캠다운 값(가로 시야각 47~87°)이고
       (순서가 틀리면 1.5~3배 큰 값이 나온다: 가짜 화면에서 참 914 px 에 1870~2700 px),
     - 카메라가 책상 위, 사람 맞은편(노트북 쪽, 책상 x > 0)에 오는 것 중 모서리가 가장 잘 맞는 것을 고른다.
     사람 자리에서 본 1 내 쪽-왼쪽, 2 내 쪽-오른쪽, 3 먼 쪽-오른쪽, 4 먼 쪽-왼쪽 (geometry.CORNERS_TABLE) 순서로 돌려준다.
한 장면만 보고 정하지 않는다: PaperFinder 가 여러 장면에서 같은 자리에 보일 때만 내준다.
"""
import itertools

import cv2
import numpy as np

from geometry import TableCalib

K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)


CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _quads(bgr, min_frac=0.003, max_frac=0.5):
    """밝고 색이 옅은 덩어리 중 바깥 윤곽이 볼록 네모인 것: [(꼭짓점 4×2, 윤곽 점 n×2)].
    밝기는 두 가지로 본다: 원래 밝기, 그리고 CLAHE (작은 구역마다 대비를 늘림, 11/18 사용자 제안 — 연구실 흰 책상에서
    종이 254 / 책상 231 이던 차이가 253 / 213 으로 벌어짐)."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    out = []
    for v in (cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0), cv2.GaussianBlur(CLAHE.apply(hsv[:, :, 2]), (5, 5), 0)):
        otsu, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hi = float(np.percentile(v, 99))
        # 11/18 연구실: 책상이 거의 흰색이라 (밝기 226~231, 종이 251~254) 낮은 문턱에서는 종이와 책상이 한 덩어리가 된다.
        # 밝은 쪽을 촘촘히 훑는다 (가장 밝은 것보다 6~45 아래).
        thrs = sorted({int(x) for x in (otsu, (otsu + hi) / 2, otsu - 15, hi - 45, hi - 30, hi - 20, hi - 12, hi - 6)
                       if 30 < x < 252})
        for thr in thrs:
            mask = ((v > thr) & (s < 90)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K5)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                a = cv2.contourArea(c)
                if not (min_frac * w * h < a < max_frac * w * h):
                    continue
                hull = cv2.convexHull(c)
                peri = cv2.arcLength(hull, True)
                ap = None
                for eps in (0.015, 0.025, 0.035, 0.05):
                    ap = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(ap) <= 4:
                        break
                if ap is None or len(ap) != 4 or not cv2.isContourConvex(ap):
                    continue
                if a < 0.85 * cv2.contourArea(ap):                 # 네모를 거의 꽉 채워야 함 (종이 가장자리를 손이 가리면 탈락)
                    continue
                q = ap.reshape(4, 2).astype(np.float64)
                if np.any(q[:, 0] < 3) or np.any(q[:, 0] > w - 4) or np.any(q[:, 1] < 3) or np.any(q[:, 1] > h - 4):
                    continue                                         # 화면 밖으로 잘린 것
                out.append((q, c.reshape(-1, 2).astype(np.float64)))
    return out


def _refine(q, pts):
    """네 변마다 윤곽 점에 직선을 맞추고 이웃한 두 직선의 교점을 모서리로. 못 하면 None."""
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 20:
            return None
        u = d / L
        rel = pts - a
        along = rel @ u
        off = np.abs(rel @ np.array([-u[1], u[0]]))
        sel = pts[(off < 4.0) & (along > 0.12 * L) & (along < 0.88 * L)]
        if len(sel) < 10:
            return None
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    return out if np.max(np.linalg.norm(out - q, axis=1)) < 15 else None


def _edge_support(q, pts, tol=2.5, bins=40):
    """네모의 네 변이 윤곽 점으로 얼마나 덮이는지 (가장 덜 덮인 변의 비율). 손이 모서리를 가리면 그 두 변이 비어서 작아진다."""
    worst = 1.0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        u = d / L
        rel = pts - a
        along = rel @ u
        off = np.abs(rel @ np.array([-u[1], u[0]]))
        sel = along[(off < tol) & (along >= 0) & (along <= L)]
        hit = np.zeros(bins, bool)
        hit[np.clip((sel / L * bins).astype(int), 0, bins - 1)] = True
        worst = min(worst, float(hit.mean()))
    return worst


def order_paper(q, w, h, hfov=None, f_range=(0.53, 1.15)):
    """네모 꼭짓점 (어떤 순서든) → CORNERS_TABLE 순서의 모서리, TableCalib, 점수 (작을수록 좋음). 말이 안 되면 None.
    f_range: 초점거리 / 화면 가로 폭 (0.53~1.15 = 가로 시야각 87~47°, 노트북 웹캠·C920 은 60~80°).
    정면에서 본 네모 하나로는 '가까운 A4' 와 '넓은 화각으로 본 다른 네모' 를 가르기 어렵다 (15 cm 정사각 카드가
    시야각 92° 카메라로 본 A4 로 맞춰짐). 화각을 웹캠다운 범위로 묶어서 가른다. 넓은 화각 카메라면 --hfov 를 주거나 M 으로 누른다."""
    best = None
    for rev, k in itertools.product((False, True), range(4)):
        cand = q[::-1] if rev else q
        cand = np.roll(cand, -k, axis=0)
        try:
            c = TableCalib(cand, w, h, hfov)
        except (RuntimeError, cv2.error):
            continue
        if hfov is None and (c.focal_source != "estimate" or not f_range[0] * w < c.f < f_range[1] * w):
            continue                                             # A4 가로 비율로 맞지 않는 순서 (초점거리가 엉뚱함)
        cp = c.cam_pos
        dist = float(np.linalg.norm(cp))
        if not (0.05 < cp[2] < 1.5 and 0.15 < dist < 2.0 and cp[0] > 0.15 and cp[0] > 0.5 * abs(cp[1])):
            continue                                             # 카메라가 책상 위, 사람 맞은편(옆으로 63° 안)이어야 함
        score = c.reproj_px
        if best is None or score < best[2]:
            best = (cand, c, score)
    return best


def find_paper(bgr, hfov=None):
    """한 장면에서 A4 종이 → (모서리 4×2 원본 px, CORNERS_TABLE 순서), TableCalib) 또는 None."""
    h, w = bgr.shape[:2]
    best = None
    for q, pts in _quads(bgr):
        r = _refine(q, pts)
        r = q if r is None else r
        if _edge_support(r, pts) < 0.9:
            continue                                             # 변이 끝까지 안 보임 (손·물건이 가장자리를 가림)
        o = order_paper(r, w, h, hfov)
        if o is None or o[1].reproj_px > 3.0:
            continue
        area = cv2.contourArea(r.astype(np.float32))
        if best is None or area > best[2]:
            best = (o[0], o[1], area)
    return None if best is None else (best[0], best[1])


class PaperFinder:
    """장면마다 넣으면, 같은 자리(모서리 tol_px 안)에 need 번 연달아 보였을 때 그 가운데값 모서리를 내준다."""

    def __init__(self, hfov=None, need=8, tol_px=3.0, every=2):
        self.hfov, self.need, self.tol, self.every = hfov, need, tol_px, every
        self.n, self.hist, self.last = 0, [], None

    def reset(self):
        self.n, self.hist, self.last = 0, [], None

    def feed(self, bgr):
        self.n += 1
        if self.n % self.every:
            return None
        r = find_paper(bgr, self.hfov)
        if r is None:
            self.hist = []
            self.last = None
            return None
        self.last = r[0]
        if self.hist and np.max(np.linalg.norm(self.hist[-1] - r[0], axis=1)) > self.tol:
            self.hist = []
        self.hist.append(r[0])
        if len(self.hist) >= self.need:
            return np.median(self.hist[-self.need:], axis=0)
        return None
