"""녹화하는 동안 시뮬 PiPER 가 손을 따라 움직이는 창 (2026-11-27).

record.py --mirror 127.0.0.1:5011 이 두 카메라로 삼각측량한 손 21점(책상 좌표)을 UDP 로 보내면,
학습 자료를 만들 때와 똑같은 식(build_dataset: 엄지·검지 가운데, 손바닥 회전, 7 cm 기준, 역기구학)으로 시뮬 팔을 놓는다.
물리는 돌리지 않고 자세만 보여 준다. 손 → 로봇 바꾸기가 그럴듯한지 녹화하면서 눈으로 보는 용도.
첫 손바닥(회전 기준)은 녹화를 시작할 때, 또는 손이 1초 넘게 안 보였다가 다시 보일 때 잡는다.

  ~/smolVLA/.venv/bin/python point_policy/mirror.py [--port 5011]
"""
import argparse
import json
import socket
import time

import mujoco
import mujoco.viewer
import numpy as np

from common import load_task
from piper_kin import HOME_Q
from points import HandToRobot, desk_to_robot
from sim_scene import GRIP_OPEN, PiperSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5011)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--task", default=None)
    ap.add_argument("--ik", choices=["numpy", "placo"], default="numpy")
    a = ap.parse_args()
    task = load_task(a.task)
    sim = PiperSim(task, ik=a.ik)
    m, d = sim.model, sim.data
    q = sim.start_q()
    d.qpos[sim.jq] = q
    d.qpos[sim.fq] = [GRIP_OPEN, -GRIP_OPEN]
    a_ = sim.cube_qadr
    d.qpos[a_:a_ + 3] = [0.0, 0.0, -1.0]                # 물체는 안 보이게 (진짜 물체는 손 쪽에 있음)
    m.body_pos[sim.pad_body] = [0.0, 0.0, -1.0]
    mujoco.mj_forward(m, d)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((a.host, a.port))
    sock.setblocking(False)
    h2r, last_seen, last_rec, n, t_print = None, 0.0, False, 0, 0.0
    print(f"[mirror] {a.host}:{a.port} 에서 손을 기다림 (record.py --mirror {a.host}:{a.port})")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.28, 0.0, 0.95]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.0, 200, -25
        while v.is_running():
            pkt = None
            while True:                                   # 쌓인 것 중 가장 새것만
                try:
                    pkt = json.loads(sock.recv(65536))
                except BlockingIOError:
                    break
            if pkt is not None:
                now = time.monotonic()
                rec = bool(pkt.get("rec"))
                if pkt.get("hand21") is not None:
                    H = desk_to_robot(np.array(pkt["hand21"]), task)
                    if h2r is None or (rec and not last_rec) or now - last_seen > 1.0:
                        h2r = HandToRobot(task)           # 회전 기준을 새로 잡음
                    last_seen = now
                    pos, R, closed, gap = h2r(H)
                    q = sim.solver.ik(pos, R, q, iters=30)
                    p_r, _ = sim.kin.fk(q)
                    d.qpos[sim.jq] = q
                    g = 0.0 if closed else GRIP_OPEN
                    d.qpos[sim.fq] = [g, -g]
                    mujoco.mj_forward(m, d)
                    n += 1
                    if now - t_print > 1.0:
                        print(f"[mirror] 집게 끝 x{p_r[0]*100:.0f} y{p_r[1]*100:+.0f} z{(p_r[2]-0.9)*100:.0f} cm "
                              f"(손과 {np.linalg.norm(p_r-pos)*1000:.0f} mm), 손가락 {gap*100:.1f} cm "
                              f"{'닫힘' if closed else '열림'}{'  [녹화 중]' if rec else ''}", flush=True)
                        t_print = now
                last_rec = rec
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
