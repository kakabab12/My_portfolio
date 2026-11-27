"""시뮬 평가 장면을 GIF 로 (일지용, 4K 까지, 2026-11-27).

정책은 sim_eval 과 똑같이 돌리고, 그림만 크게 다시 그린다 (정책 입력은 점이라 그림 크기와 상관없음).
GitHub 한 파일 100 MB 한도 안에 들게 장 수·색 수를 줄인다.

  .venv/bin/python point_policy/make_gif.py point_policy/runs/synth40/policy.pt --ep 0 --res 3840x2160 --out 그림.gif
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_eval import Runner, run_episode
from sim_scene import PiperSim, sample_layout


def font(size):
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--ep", type=int, default=0, help="sim_eval 의 몇 번째 장면 (seed 1000 부터 같은 자리)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--res", default="3840x2160")
    ap.add_argument("--every", type=int, default=1, help="몇 박자마다 한 장")
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--ik", default="numpy")
    ap.add_argument("--title", default="각본으로 만든 가짜 손 시범 40개로 배운 Point Policy · 시뮬 PiPER · 처음 보는 자리")
    ap.add_argument("--cam", default="close")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.lower().split("x"))
    run = Runner(a.ckpt)
    task = run.cfg["task"]
    sim = PiperSim(task, ik=a.ik, offscreen=(W, H))
    rng = np.random.default_rng(a.seed)
    for _ in range(a.ep + 1):
        cube_xy, pad_xy = sample_layout(task, rng)
    frames = []
    r = run_episode(sim, run, task, cube_xy, pad_xy, frames=frames, wh=(W, H), cam=a.cam)
    hz = task["policy_hz"]
    f1, f2 = font(int(H * 0.032)), font(int(H * 0.024))
    imgs = []
    sel = frames[::a.every]
    for k, fr in enumerate(sel):
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        t = k * a.every / hz
        d.rectangle([0, 0, W, int(H * 0.1)], fill=(20, 20, 24))
        d.text((int(W * 0.02), int(H * 0.012)), a.title, font=f1, fill=(240, 240, 240))
        d.text((int(W * 0.02), int(H * 0.058)),
               f"정육면체(빨강)를 집어 판(파랑) 위에 놓기 · 정책 초당 {hz}번 · {t:4.1f}초"
               + (f" · 결과: {'성공' if r['success'] else '실패'}" if k == len(sel) - 1 else ""),
               font=f2, fill=(200, 200, 200))
        imgs.append(im.quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT))
    imgs += [imgs[-1]] * int(hz * 1.5 / a.every)            # 끝 장면 잠깐 멈춤
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=int(1000 * a.every / hz), loop=0, optimize=True)
    print(f"{a.out}: {len(imgs)}장 {W}x{H}, {os.path.getsize(a.out) / 1e6:.1f} MB, {r}")


if __name__ == "__main__":
    main()
