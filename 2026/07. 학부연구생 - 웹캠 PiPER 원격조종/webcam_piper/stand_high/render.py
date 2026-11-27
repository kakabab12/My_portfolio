"""stl/ 의 부품을 읽어 그림 한 장으로 그린다 (2026-11-25, 주간보고용).
왼쪽: 조립한 기둥 (카메라 높이 800 mm, 55° 머리) 과 카메라가 보는 줄, 오른쪽: 출력할 부품.

  python render.py                → render.png
  (make_stand.py 를 먼저 돌려 stl/ 을 만들어 둘 것. numpy, matplotlib 만 씀)
"""
import math
import os
import struct

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
STL = os.path.join(HERE, "stl")
INK, MUTED = "#1f2328", "#6e7781"
LIGHT = np.array([0.35, -0.55, 0.75]) / np.linalg.norm([0.35, -0.55, 0.75])


def read_stl(name):
    with open(os.path.join(STL, name), "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        d = np.frombuffer(f.read(50 * n), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return d["v"].astype(float)


def draw(ax, tris, color):
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    shade = 0.5 + 0.5 * np.clip(nrm @ LIGHT, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(color))
    pc = Poly3DCollection(tris, facecolor=np.clip(base[None] * shade[:, None], 0, 1), edgecolor="none")
    ax.add_collection3d(pc)


def draw_many(ax, items):
    """여러 부품을 한 묶음으로 그린다 (따로 그리면 matplotlib 가 부품 단위로 앞뒤를 정해 작은 부품이 가려짐)."""
    tris, cols = [], []
    for t, color in items:
        nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        shade = 0.5 + 0.5 * np.clip(nrm @ LIGHT, 0, 1)
        tris.append(t)
        cols.append(np.clip(np.array(matplotlib.colors.to_rgb(color))[None] * shade[:, None], 0, 1))
    ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolor=np.concatenate(cols), edgecolor="none"))


def tri_of(m):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    return v[np.asarray(mesh.tri_verts)]


def tidy(ax, lim, elev, azim):
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main():
    import make_stand as S
    # 조립은 make_stand.py 와 같은 계산으로 (STL 은 출력 자세로 눕혀 있어서 조립 그림은 같은 모양을 다시 세운다)
    S.N_SEG, angle, cam_h = 4, 55, 800
    S.LS = round((cam_h - S.cam_offset_tripod(angle)[1] - (S.BASE_T + S.BOSS_H)) / S.N_SEG, 1)
    top = S.BASE_T + S.BOSS_H + S.N_SEG * S.LS
    asm = [(S.base(), "#8f9a96")]
    for i in range(S.N_SEG):
        asm.append((S.segment().translate((0, 0, S.BASE_T + S.BOSS_H + i * S.LS)), "#6f8f84" if i % 2 else "#7fa395"))
    # 11/25 APC930: 삼각대 머리 + 나사 기둥에 카메라를 올린 모습 (카메라 모양은 제조사 그림을 보고 대강, 치수는 어림)
    hm, info = S.head_tripod(angle)
    asm.append((hm.translate((0, 0, top)), "#c98b3c"))
    stud = S.tripod_stud().translate((0, 0, -S.PEG_L)).rotate((0, angle, 0)).translate(tuple(info["P"] + np.array([0, 0, top])))
    asm.append((stud, "#57606a"))
    from manifold3d import Manifold as Mf
    leg = S.box(-22, 18, -14, 14, 2, 8)                                    # 접은 뒷다리 (삼각대 구멍이 있는 쪽)
    plate = S.box(-22, 18, -14, 14, 8, 13)                                 # 윗판
    body = Mf.batch_hull([Mf.cylinder(30, 16).rotate((0, 90, 0)).translate((-14, sy * 34, 29)) for sy in (-1, 1)])
    lens = Mf.cylinder(2, 7).rotate((0, 90, 0)).translate((16, 0, 29))
    cam_local = leg + plate + body
    R = lambda m: m.rotate((0, angle, 0)).translate(tuple(info["P"] + np.array([0, 0, top])))  # noqa: E731
    asm.append((R(cam_local), "#e9ecef"))
    asm.append((R(lens), "#24292f"))
    u = info["u"]
    wdir = np.array([math.cos(math.radians(angle)), 0.0, -math.sin(math.radians(angle))])
    cam_c = info["P"] + np.array([0, 0, top]) + 29 * u + 16 * wdir

    fig = plt.figure(figsize=(10.0, 3.6), dpi=220)
    ax = fig.add_axes([0.0, 0.0, 0.42, 0.92], projection="3d")
    draw_many(ax, [(tri_of(m), col) for m, col in asm])
    cx, cz = cam_c[0], cam_c[2]
    px = cx + (cz - 30) / math.tan(math.radians(angle))
    ax.plot([cx, px], [0, 0], [cz, 0], color="#2f6fdc", lw=1.4, ls="--")
    ax.plot([px - 105, px + 105, px + 105, px - 105, px - 105], [-148, -148, 148, 148, -148], [0] * 5, color=MUTED, lw=0.9)
    ax.text(px + 60, 170, 0, "A4 paper", fontsize=6.5, color=MUTED, ha="center")
    ax.plot([-120, -120], [0, 0], [0, cz], color=INK, lw=0.8)
    for z in (0, cz):
        ax.plot([-135, -105], [0, 0], [z, z], color=INK, lw=0.8)
    ax.text(-150, 0, cz / 2, f"{cz / 10:.0f} cm", fontsize=8, color=INK, ha="right", va="center")
    ax.text(cx + 40, 0, cz + 60, "APC930, 55° down", fontsize=7, color=INK)
    tidy(ax, ((-200, 750), (-300, 300), (0, 900)), 14, -62)
    zx = fig.add_axes([0.27, 0.36, 0.18, 0.46], projection="3d")          # 머리·나사·카메라 확대
    draw_many(zx, [(tri_of(m), col) for m, col in asm[-4:-2]])          # 머리와 나사 기둥만 (카메라는 빼서 나사가 보이게)
    zc = info["P"] + np.array([0, 0, top])
    tidy(zx, ((zc[0] - 32, zc[0] + 32), (-32, 32), (zc[2] - 45, zc[2] + 19)), 14, -62)
    zx.set_title("tripod head + printed 1/4\" stud\ncamera screws onto the stud", fontsize=6.5, color=MUTED, y=0.95)
    fig.text(0.02, 0.93, "Camera pole for ABKO APC930, 3D-printed, no metal screws", fontsize=9, color=INK)

    # 오른쪽: 한 벌 (카메라 하나 몫) 을 실제 크기 비율 그대로 판 위에 늘어놓음
    fig.text(0.50, 0.93, "Parts for one camera, true scale, print two sets", fontsize=9, color=INK)
    fig.text(0.50, 0.86, "base 180 mm, 4 poles 202 mm, coupler, tripod head 55°, printed 1/4\" stud, 8 pins", fontsize=7.5, color=MUTED)
    bed = fig.add_axes([0.44, 0.0, 0.56, 0.90], projection="3d")
    place = [("1_base_x1.stl", (0, 0), "#8f9a96", "base", "below"),
             ("2_pole_segment_x4.stl", (200, -6), "#7fa395", "", ""), ("2_pole_segment_x4.stl", (232, -6), "#6f8f84", "", ""),
             ("2_pole_segment_x4.stl", (264, -6), "#7fa395", "", ""),
             ("2_pole_segment_x4.stl", (296, -6), "#6f8f84", "4 poles, 193 mm", "above"),
             ("3_coupler_x1.stl", (345, 0), "#6f8f84", "coupler", "below"), ("5_pins_x8.stl", (385, 0), "#57606a", "pins", "below"),
             ("4b_head_tripod_55deg_x1.stl", (345, 85), "#c98b3c", "tripod head (55°)", "above"),
             ("4c_tripod_stud_x1.stl", (405, 95), "#57606a", "stud", "above")]
    items = []
    for fn, (ox, oy), col, label, where in place:
        tris = read_stl(fn)
        items.append((tris - tris.reshape(-1, 3).min(0) + np.array([ox, oy, 0.0]), col))
    draw_many(bed, items)
    bed.plot([-10, 450, 450, -10, -10], [-45, -45, 215, 215, -45], [0] * 5, color="#d0d7de", lw=0.8)
    tidy(bed, ((-20, 460), (-60, 240), (0, 120)), 42, -75)
    out = os.path.join(HERE, "render.png")
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
