"""시험용 가짜 책상 사진: A4 종이(가운데 +)를 정해진 카메라에서 본 모습을 그린다 (2026-11-18, paper.py 시험).
test_geometry.py 시험 10 과 자동 보정 흐름 시험에서 쓴다. 실물 사진이 아니므로 숫자는 '그린 종이' 기준이다."""
import cv2
import numpy as np

from geometry import A4_D, A4_W, CORNERS_TABLE


def project(R, t, K, pts):
    p, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 3), cv2.Rodrigues(R)[0], t, K, None)
    return p.reshape(-1, 2)


def render_desk(R, t, K, w, h, rng, desk=(60, 85, 115), paper=(228, 230, 232), noise=5.0, cross=True,
                blob=None, other=None):
    """책상(나뭇결 비슷한 무늬) 위 A4. blob: (책상 x, y, 반지름 m) 어두운 동그라미 (손 흉내). other: 다른 흰 네모 [(x,y)...] 책상 좌표."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array(desk, np.float32)[None, None, :]
    grain = (np.sin(xx / 7.0 + np.sin(yy / 40.0) * 3.0) * 8.0)[..., None]
    shade = (1.0 - 0.25 * yy / h)[..., None]                       # 위쪽(먼 곳)이 조금 어둡게
    img = np.clip((base + grain) * shade, 0, 255)
    img = img.astype(np.uint8)

    def poly(pts_table, color):
        p = project(R, t, K, pts_table)
        cv2.fillPoly(img, [np.round(p * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)

    if other is not None:
        poly([[x, y, 0.0] for x, y in other], (215, 218, 220))
    poly(CORNERS_TABLE, paper)
    if cross:                                                       # 가운데 + (검은 펜, 3 mm 굵기)
        for a, b in (([-0.04, 0, 0], [0.04, 0, 0]), ([0, -0.04, 0], [0, 0.04, 0])):
            p = project(R, t, K, [a, b])
            cv2.line(img, tuple(np.round(p[0] * 16).astype(int)), tuple(np.round(p[1] * 16).astype(int)), (30, 30, 30),
                     3, cv2.LINE_AA, shift=4)
    for k in range(3):                                              # 종이 위 글씨 흉내
        y0 = -A4_W / 2 + 0.03
        p = project(R, t, K, [[A4_D / 2 - 0.03 - 0.012 * k, y0, 0], [A4_D / 2 - 0.03 - 0.012 * k, y0 + 0.08, 0]])
        cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (120, 120, 120), 1, cv2.LINE_AA)
    if blob is not None:
        x, y, r = blob
        ring = [[x + r * np.cos(a), y + r * np.sin(a), 0.02] for a in np.linspace(0, 2 * np.pi, 40)]
        p = project(R, t, K, ring)
        cv2.fillPoly(img, [np.round(p).astype(np.int32)], (70, 100, 150), cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img
