"""Logitech C920 스탠드 — 나사 없음 (2026-11-17).

C920 에 원래 달린 모니터 집게를 그대로 쓴다. 스탠드 꼭대기는 기울어진 "가짜 모니터 모서리"(두께 12 mm 판)라서,
모니터에 걸 때처럼 걸치기만 하면 카메라가 판이 기운 만큼 아래를 본다 (기본 55°, 45°·65° 도 만듦).
기둥은 24 mm 네모 막대를 끼워 잇고, 출력한 핀으로 빠지지 않게 한다. 나사·볼트·접착제 없음.

  python make_stand.py                        → stl/ 에 부품, preview.png 에 조립 그림 (카메라 높이 600 mm, 55°)
  python make_stand.py --cam-height 650 --angle 50   → 높이·각도 바꾸기 (막대 길이가 저절로 맞춰짐)
  python make_stand.py --segments 4            → 프린터가 작으면 막대를 4개로
  python make_stand.py --panel 15              → C920 집게가 헐거우면 판을 두껍게
  (pip install manifold3d numpy matplotlib)

자리 C (웹캠 자리 3D 페이지): 카메라를 종이 + 에서 노트북 쪽 40 cm, 높이 60 cm 에 두고 55° 아래를 보게.
→ 기둥 중심을 종이 + 에서 약 49 cm 뒤에 둔다 (카메라는 기둥보다 9 cm 앞, 9 cm 위에 걸린다).
"""
import math
import os
import struct

import numpy as np
from manifold3d import CrossSection as CS
from manifold3d import Manifold as M
from manifold3d import set_circular_segments

set_circular_segments(64)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stl")

BAR, CH = 24.0, 1.5                  # 기둥 네모 막대, 모서리 모따기 (mm)
TEN, SOCK = 16.0, 16.4               # 끼우는 촉 / 구멍 (한쪽 0.2 mm 틈)
TEN_L, SOCK_D = 25.0, 26.0           # 촉 길이 / 구멍 깊이
PIN_D, PIN_HOLE, PIN_AT = 3.0, 3.4, 12.5   # 핀 굵기 / 핀 구멍 / 이음면에서 핀까지
LS = 155.0                           # 막대 하나 길이 (촉 빼고)
N_SEG = 3
BASE, BASE_T, BOSS, BOSS_H = 180.0, 10.0, 40.0, 32.0
PANEL_T, PANEL_W, RAMP_L = 12.0, 60.0, 100.0   # 가짜 모니터 판: 두께, 폭, 길이
HEAD_Y0, HEAD_Y1 = -30.0, -6.0       # 머리의 기둥 쪽 블록 (판 한쪽 옆면과 맞춰 눕혀서 받침 없이 출력)
A0 = (-4.0, 26.0)                    # 판 가운데 줄이 시작하는 점 (x 앞, z 위, 기둥 꼭대기 기준)


def pcs(pts):
    pts = np.asarray(pts, float)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    return CS([pts if area > 0 else pts[::-1]])


def box(x0, x1, y0, y1, z0, z1):
    return M.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def sq(w, c):
    h = w / 2
    return pcs([(-h + c, -h), (h - c, -h), (h, -h + c), (h, h - c), (h - c, h), (-h + c, h), (-h, h - c), (-h, -h + c)])


def along_y(cs2d, y0, y1):
    """(x, z) 평면 단면을 y0~y1 로 밀어 낸 것."""
    return M.extrude(cs2d, y1 - y0).rotate((90, 0, 0)).translate((0, y1, 0))


def tenon_down(z_top):
    """z_top 에서 아래로 뻗은 16 mm 촉 (끝 1.5 mm 모따기)."""
    body = M.extrude(sq(TEN, 0.8), TEN_L - 1.5).translate((0, 0, z_top - TEN_L + 1.5))
    tip = M.extrude(sq(TEN - 2.0, 0.6), 1.5, scale_top=(TEN / (TEN - 2.0), TEN / (TEN - 2.0))).translate((0, 0, z_top - TEN_L))
    return body + tip


def socket_up(z_top):
    """z_top 에서 아래로 파인 구멍 (입구 모따기)."""
    hole = box(-SOCK / 2, SOCK / 2, -SOCK / 2, SOCK / 2, z_top - SOCK_D, z_top + 1)
    lip = M.extrude(sq(SOCK, 0.0), 1.01, scale_top=((SOCK + 2) / SOCK, (SOCK + 2) / SOCK)).translate((0, 0, z_top - 1.0))
    return hole + lip


def pin_y(z, x=0.0, y=0.0, teardrop=False):
    """y 방향 핀 구멍. teardrop: 세워서 출력하는 부품용 (위가 뾰족해 처지지 않음)."""
    r = PIN_HOLE / 2
    cs = CS.batch_hull([CS.circle(r), CS.circle(0.05).translate((0, r * 1.42))]) if teardrop else CS.circle(r)
    return along_y(cs.translate((x, z)), y - 40, y + 40)


def segment():
    s = M.extrude(sq(BAR, CH), LS) + tenon_down(0.0)
    return s - socket_up(LS) - pin_y(LS - PIN_AT) - pin_y(-PIN_AT)


def coupler():
    """기둥 맨 위와 머리를 잇는 양쪽 촉."""
    t = tenon_down(0.0)
    c = t + t.mirror((0, 0, 1))
    return c - pin_y(-PIN_AT) - pin_y(PIN_AT)


def base():
    r, h = 16.0, BASE / 2 - 16.0
    plate = M.extrude(CS.batch_hull([CS.circle(r).translate((sx * h, sy * h)) for sx in (-1, 1) for sy in (-1, 1)]), BASE_T)
    boss = M.extrude(sq(BOSS, 3.0), BOSS_H).translate((0, 0, BASE_T))
    rib = along_y(pcs([(BOSS / 2 - 1, 0), (BOSS / 2 + 26, 0), (BOSS / 2 - 1, 24)]), -2.5, 2.5).translate((0, 0, BASE_T))
    ribs = rib + rib.rotate((0, 0, 90)) + rib.rotate((0, 0, 180)) + rib.rotate((0, 0, 270))
    pockets = None
    for sx in (-1, 1):
        for sy in (-1, 1):
            xs, ys = sorted((sx * 29, sx * 81)), sorted((sy * 29, sy * 81))
            pk = box(xs[0], xs[1], ys[0], ys[1], BASE_T - 6, BASE_T + 1)
            pockets = pk if pockets is None else pockets + pk
    top = BASE_T + BOSS_H
    return plate + boss + ribs - pockets - socket_up(top) - pin_y(top - PIN_AT, teardrop=True)


def head(alpha):
    """alpha: 카메라가 수평에서 아래로 보는 각도 (도). 판은 수평에서 (90 − alpha)° 로 사람 쪽으로 올라간다."""
    al = math.radians(alpha)
    a = np.array([math.sin(al), math.cos(al)])            # 판 방향 (아래 → 걸치는 끝)
    n = np.array([math.cos(al), -math.sin(al)])           # 판 앞면 바깥 = 카메라가 보는 방향
    # 판: 지역 좌표 X 판 길이, Y 폭, Z 두께 (+Z = 윗면 = 집게 뒷다리가 놓이는 면)
    ramp = box(0, RAMP_L, -PANEL_W / 2, PANEL_W / 2, -PANEL_T / 2, PANEL_T / 2)
    teeth = CS([np.array([(s, PANEL_T / 2 - 0.05), (s + 4, PANEL_T / 2 - 0.05), (s + 4, PANEL_T / 2 + 1.2)])
                for s in np.arange(RAMP_L - 80, RAMP_L - 20, 5.0)])
    ramp = ramp + along_y(teeth, -PANEL_W / 2, PANEL_W / 2)            # 뒷다리가 미끄러지지 않게 톱니
    for sy in (-1, 1):                                                  # 고무줄 홈 (선택)
        y0, y1 = sorted((sy * (PANEL_W / 2 - 4), sy * (PANEL_W / 2 + 1)))
        ramp = ramp - box(RAMP_L - 22, RAMP_L - 14, y0, y1, -PANEL_T, PANEL_T)
    for sz in (-1, 1):                                                  # 걸치는 끝 모서리 둥글게 (모따기)
        ramp = ramp - box(-2, 2, -PANEL_W, PANEL_W, -2, 2).rotate((0, 45, 0)).translate((RAMP_L, 0, sz * PANEL_T / 2))
    ramp = ramp.rotate((0, alpha - 90, 0)).translate((A0[0], 0, A0[1]))
    blk = box(-BAR / 2, BAR / 2, HEAD_Y0, HEAD_Y1, 0, 36)              # 36: 45° 에서도 판 아랫면까지 메움 (가는 틈 없게)
    # 판 아래 버팀대
    def under(X):                                                       # 판 아랫면보다 1 mm 안쪽 (딱 닿기만 하면 틈이 생김)
        p = np.array(A0) + X * a + (PANEL_T / 2 - 1.0) * n
        return tuple(p)
    X12 = (BAR / 2 - A0[0] - (PANEL_T / 2 - 1.0) * n[0]) / a[0]
    gus = along_y(pcs([(BAR / 2 - 0.5, 2), under(45), under(X12)]), HEAD_Y0, HEAD_Y0 + 12)
    yc = (HEAD_Y0 + HEAD_Y1) / 2
    h = blk + gus + ramp
    sock = box(-SOCK / 2, SOCK / 2, yc - SOCK / 2, yc + SOCK / 2, -1, SOCK_D) + \
        M.extrude(sq(SOCK + 2, 0.0), 1.01, scale_top=(SOCK / (SOCK + 2), SOCK / (SOCK + 2))).translate((0, yc, -0.01))
    return h - sock - pin_y(PIN_AT, 0.0, yc), {"a": a, "n": n, "yc": yc}


def pins(k=6):
    one = (M.cylinder(27.0, PIN_D / 2) + M.cylinder(1.5, 3.0).translate((0, 0, -1.5))).rotate((0, 90, 0))
    one = one ^ box(-5, 40, -5, 5, -1.1, 5)                              # 바닥을 평평하게 잘라 눕혀 출력
    out = None
    for i in range(k):
        p = one.translate((0, i * 9.0, 0))
        out = p if out is None else out + p
    return out


def to_bed(m, lay_on_side=False):
    if lay_on_side:
        m = m.rotate((90, 0, 0))                                         # 쓸 때 y → 출력 z (핀 구멍이 세로)
    b = m.bounding_box()
    return m.translate((-(b[0] + b[3]) / 2, -(b[1] + b[4]) / 2, -b[2]))


def save(m, name):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    t = np.asarray(mesh.tri_verts)
    tri = v[t].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    data = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data["n"], data["v"] = nrm, tri
    path = os.path.join(OUT, name)
    with open(path, "wb") as f:
        f.write(b"C920 stand, no screws (webcam_piper 2026-11-17)".ljust(80, b" "))
        f.write(struct.pack("<I", len(t)))
        f.write(data.tobytes())
    b = m.bounding_box()
    print(f"{name:28s} {b[3] - b[0]:6.1f} x {b[4] - b[1]:6.1f} x {b[5] - b[2]:6.1f} mm, 부피 {m.volume() / 1000:6.1f} cm³, "
          f"삼각형 {len(t)}, 구멍 수(genus) {m.genus()}, 상태 {m.status()}")
    return path


def cam_offset(alpha):
    """머리에 건 카메라 가운데가 기둥 꼭대기에서 (사람 쪽, 위로) 몇 mm 인지 (대강)."""
    al = math.radians(alpha)
    a = np.array([math.sin(al), math.cos(al)])
    n = np.array([math.cos(al), -math.sin(al)])
    return np.array(A0) + (RAMP_L + 14) * a + 6 * n


def main():
    import argparse
    global LS, N_SEG, PANEL_T
    ap = argparse.ArgumentParser(description="C920 스탠드 STL (나사 없음)")
    ap.add_argument("--cam-height", type=float, default=600, help="책상에서 카메라 가운데까지 (mm). 기둥 막대 길이가 여기에 맞춰짐")
    ap.add_argument("--angle", type=int, default=55, help="카메라가 수평에서 아래로 보는 각도 (도). 이 머리로 높이를 맞춤")
    ap.add_argument("--extra-angles", default="45,65", help="같이 만들 머리 각도 (쉼표로)")
    ap.add_argument("--segments", type=int, default=3, help="기둥 막대 개수 (프린터가 작으면 늘림)")
    ap.add_argument("--panel", type=float, default=12.0, help="가짜 모니터 판 두께 (mm). C920 집게가 헐거우면 늘림")
    a = ap.parse_args()
    N_SEG, PANEL_T = a.segments, a.panel
    LS = round((a.cam_height - cam_offset(a.angle)[1] - (BASE_T + BOSS_H)) / N_SEG, 1)
    print(f"카메라 높이 {a.cam_height:.0f} mm, 각도 {a.angle}° → 막대 {N_SEG}개 × {LS} mm (촉 포함 {LS + TEN_L:.0f} mm)")
    if LS + TEN_L > 220:
        print("  막대가 220 mm 를 넘음: 프린터 판에 안 들어가면 --segments 를 늘리세요")
    angles = sorted({a.angle, *[int(x) for x in a.extra_angles.split(",") if x.strip()]})
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".stl"):
            os.remove(os.path.join(OUT, f))
    parts = {}
    parts["base"] = base()
    parts["segment"] = segment()
    parts["coupler"] = coupler()
    heads = {al: head(al) for al in angles}
    save(to_bed(parts["base"]), "1_base_x1.stl")
    save(to_bed(parts["segment"], True), f"2_pole_segment_x{N_SEG}.stl")
    save(to_bed(parts["coupler"], True), "3_coupler_x1.stl")
    for al, (hm, _) in heads.items():
        save(to_bed(hm, True), f"4_head_{al}deg_x1.stl")
    save(to_bed(pins()), "5_pins_x6.stl")

    # 조립 (쓸 때 자세): 기둥 축 = z, 사람 쪽 = +x
    top = BASE_T + BOSS_H + N_SEG * LS
    asm = [("base", parts["base"], "#8f9a96")]
    for i in range(N_SEG):
        asm.append((f"seg{i}", parts["segment"].translate((0, 0, BASE_T + BOSS_H + i * LS)), "#6f8f84" if i % 2 else "#7fa395"))
    hm, info = heads[a.angle]
    asm.append(("head", hm.translate((0, -info["yc"], top)), "#c98b3c"))
    # C920 (대강의 모양): 걸치는 끝 위에 몸통, 앞면 아래 앞판, 윗면 위 뒷다리
    cam = (box(RAMP_L, RAMP_L + 29, -47, 47, -20, 5) + box(RAMP_L - 16, RAMP_L, -14, 14, -PANEL_T / 2 - 3, -PANEL_T / 2) +
           box(RAMP_L - 48, RAMP_L, -12, 12, PANEL_T / 2 + 1.2, PANEL_T / 2 + 4.2))
    cam = cam.rotate((0, a.angle - 90, 0)).translate((A0[0], -info["yc"], top + A0[1]))
    asm.append(("c920", cam, "#2b2f31"))
    for al in angles:
        c = cam_offset(al)
        print(f"카메라 가운데 ({al}° 머리): 기둥 축에서 사람 쪽 {c[0]:.0f} mm, 높이 {top + c[1]:.0f} mm (책상 위), "
              f"종이 + 는 기둥에서 {c[0] + (top + c[1] - 30) / math.tan(math.radians(al)):.0f} mm 앞")
    preview(asm, top, cam_offset(a.angle), info, a.angle)


def preview(asm, top, c_local, info, angle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 7.5))
    for k, (elev, azim, title) in enumerate(((18, -60, f"assembly ({angle} deg head)"), (0, -90, "side view"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for name, m, col in asm:
            mesh = m.to_mesh()
            v = np.asarray(mesh.vert_properties)[:, :3]
            t = np.asarray(mesh.tri_verts)
            pc = Poly3DCollection(v[t], facecolor=col, edgecolor="none", alpha=1.0)
            nrm = np.cross(v[t][:, 1] - v[t][:, 0], v[t][:, 2] - v[t][:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            shade = 0.55 + 0.45 * np.clip(nrm @ np.array([0.3, -0.5, 0.8]), 0, 1)
            base = np.array(matplotlib.colors.to_rgb(col))
            pc.set_facecolor(np.clip(base[None, :] * shade[:, None], 0, 1))
            ax.add_collection3d(pc)
        # 카메라가 보는 줄 (55°)과 종이 (+ 가 카메라 앞 400 mm)
        cx, cz = c_local[0], top + c_local[1]
        d = info["n"]
        L = (cz - 30) / -d[1]
        ax.plot([cx, cx + d[0] * L], [0, 0], [cz, cz + d[1] * L], color="#2f8f74", lw=2)
        px = cx + (cz - 30) / math.tan(math.radians(angle))
        ax.plot([px - 105, px + 105, px + 105, px - 105, px - 105], [-148, -148, 148, 148, -148], [0] * 5, color="#555", lw=1)
        ax.text(px, 0, 5, "A4 +", fontsize=8)
        ax.set_xlim(-150, 650); ax.set_ylim(-400, 400); ax.set_zlim(0, 800)
        ax.set_box_aspect((800, 800, 800))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("toward person (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("height (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "preview.png"), dpi=90)


if __name__ == "__main__":
    main()
