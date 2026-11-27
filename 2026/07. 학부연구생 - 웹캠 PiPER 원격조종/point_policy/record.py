"""사람 손 시범 녹화 — 카메라 두 대, 로봇 없이 (Point Policy 처럼, 2026-11-27).

webcam_piper 의 MediaPipe 실시간 조종과 다른 점: 여기서는 로봇을 움직이지 않는다. 사람이 진짜 물체(정육면체·판)를 손으로 옮기는
영상만 두 카메라로 찍고, 손·물체 점은 나중에 영상에서 뽑는다 (hand_points.py, object_points.py). 로봇은 시뮬에서만 쓴다.
보고 싶으면 --mirror 로 시뮬 PiPER(mirror.py)가 손을 따라 움직이는 것을 옆에서 볼 수 있다 (논문 식 손 → 로봇 바꾸기 그대로).

카메라 보정은 webcam_piper 와 같다: A4 종이를 책상에 가로로 놓고, 두 카메라 모두 종이를 비스듬히 내려다보게.
webcam_piper/calib.json 의 두 카메라 보정(table, table2)을 불러오고, 크기가 다르거나 C 를 누르면 종이를 저절로 다시 찾는다.

시범 하나 (작업 설정 task_cube.json: 정육면체를 판 위에):
  1) 정육면체·판을 종이 둘레에 놓는다 (서로 13 cm 넘게)
  2) 오른손을 종이 가운데 + 위 10 cm 에, 손등이 위, 엄지·검지를 벌린 채 (집게가 아래를 보고 벌린 자세)
  3) SPACE (왼손) → 엄지·검지로 정육면체 옆면을 집어 판 위에 놓고 손을 조금 든다 → SPACE
  시범마다 정육면체·판 자리를 바꾼다. D 두 번 = 방금 시범 지우기, C = 종이 다시 찾기, H = 손 점 보이기 켜고 끄기, Q = 끝.
녹화 파일에는 사람이 찍힐 수 있다. 연구실 PC 안에만 두고 GitHub 에 올리지 않는다 (data/ 는 .gitignore).

  ~/smolVLA/webcam_piper/.venv/bin/python point_policy/record.py --cam 0 --cam2 2 [--mirror 127.0.0.1:5011]
"""
import argparse
import json
import shutil
import socket
import time
from pathlib import Path

import cv2
import numpy as np

import common
from common import load_task
from episode_io import new_episode_dir, save_meta
from geometry import CORNERS_TABLE, TableCalib, proj_matrix, triangulate
from paper import PaperFinder
from recorder import CamGrabber

DISP_H = 400


def pick_pair(Ps, hands0, hands1, max_err):
    """두 화면의 손들 중 삼각측량이 가장 잘 맞는 짝 → (21×3 책상 좌표, 어긋남 px) 또는 None."""
    best = None
    for h0 in hands0:
        for h1 in hands1:
            X, e = triangulate(Ps, [h0["img_px"], h1["img_px"]])
            m = float(np.median(e))
            if np.isfinite(m) and (best is None or m < best[1]):
                best = (X, m, h0, h1)
    if best is None or best[1] > max_err:
        return None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--cam2", type=int, default=2)
    ap.add_argument("--size", default="1280x960", help="11/18 보정과 같은 크기")
    ap.add_argument("--task", default=None)
    ap.add_argument("--out", default=None, help="기본 point_policy/data/<작업>/real")
    ap.add_argument("--calib", default=str(common.WEBCAM / "calib.json"))
    ap.add_argument("--hfov", type=float, default=None)
    ap.add_argument("--mirror", default=None, help="mirror.py 주소 host:port (시뮬 PiPER 가 손을 따라감)")
    ap.add_argument("--no-hand", action="store_true", help="녹화 중 손 찾기를 끔 (CPU 아끼기)")
    ap.add_argument("--max-err", type=float, default=25.0)
    a = ap.parse_args()
    task = load_task(a.task)
    W, H = (int(v) for v in a.size.lower().split("x"))
    out = Path(a.out) if a.out else common.DATA / task["name"] / "real"
    grabs = [CamGrabber(i, W, H, mjpg=True, fps=30, lock_focus_after=3.0) for i in (a.cam, a.cam2)]
    t_wait = time.monotonic()
    while any(g.ok and g.latest() is None for g in grabs) and time.monotonic() - t_wait < 4.0:
        time.sleep(0.05)
    for g, i in zip(grabs, (a.cam, a.cam2)):
        if not g.ok or g.latest() is None:
            raise SystemExit(f"카메라 {i} 를 못 엶 (리눅스는 보통 0 과 2)")
        f = g.latest()
        if f.shape[:2] != (H, W):
            raise SystemExit(f"카메라 {i} 크기가 {f.shape[1]}x{f.shape[0]} (바란 것 {W}x{H}) — --size 를 맞추세요")

    calibs = [None, None]
    try:
        saved = json.loads(Path(a.calib).read_text(encoding="utf-8"))
        for k, key in enumerate(("table", "table2")):
            d = saved.get(key)
            if d and d["w"] == W and d["h"] == H:
                calibs[k] = TableCalib.from_dict(d)
        if all(calibs):
            print(f"[보정] {a.calib} 의 두 카메라 보정을 불러옴 (어긋남 {calibs[0].reproj_px:.1f}, {calibs[1].reproj_px:.1f} px)")
    except (OSError, ValueError, KeyError):
        pass
    finders = [PaperFinder(a.hfov), PaperFinder(a.hfov)]
    finding = [c is None for c in calibs]

    trackers = None
    if not a.no_hand:
        from hand import HandTracker
        trackers = [HandTracker("right"), HandTracker("right")]
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=2)
    sock, mirror_addr = None, None
    if a.mirror:
        host, port = a.mirror.rsplit(":", 1)
        mirror_addr = (host, int(port))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    rec, writers, times, ep_dir, ep_t0 = False, None, [], None, None
    last_ep, del_armed, show_hand, n_ep = None, 0.0, True, 0
    msg, msg_t = "", 0.0
    last_n0 = -1
    futs, last_dets = None, [None, None]
    t_start = time.monotonic()
    win = "point_policy record"
    cv2.namedWindow(win)
    while True:
        f0, t0, n0 = grabs[0].latest_with_time()
        if n0 == last_n0:
            time.sleep(0.003)
            continue
        last_n0 = n0
        f1, t1, _ = grabs[1].latest_with_time()
        frames = [f0.copy(), f1.copy()]           # 일꾼이 같은 사진을 다시 줄 수 있어 그림을 덧그리기 전에 복사
        # 종이 찾기 (보정)
        for k in range(2):
            if finding[k]:
                q = finders[k].feed(frames[k])
                if q is not None:
                    calibs[k] = TableCalib(q, W, H, hfov=a.hfov)
                    finding[k] = False
                    msg, msg_t = f"cam{k} paper found (reproj {calibs[k].reproj_px:.1f}px)", time.monotonic()
        ready = all(c is not None for c in calibs) and not any(finding)
        if rec:
            for k in range(2):
                writers[k].write(frames[k])
            times.append((t0 - ep_t0, t1 - ep_t0))
        # 손 (보기·mirror 용)
        # 손 찾기는 기다리지 않는다 (녹화가 초당 30장에서 밀리지 않게): 앞의 것이 끝났으면 결과를 쓰고 새로 맡긴다
        hand3d, err = None, None
        if trackers and ready and (show_hand or sock):
            if futs is None or all(fu.done() for fu in futs):
                dets = [fu.result() for fu in futs] if futs is not None else [None, None]
                t_ms = (time.monotonic() - t_start) * 1000
                futs = [pool.submit(trackers[k].detect, frames[k].copy(), t_ms) for k in range(2)]
                last_dets = dets
            dets = last_dets
            if dets and all(dets):
                Ps = [proj_matrix(c) for c in calibs]
                best = pick_pair(Ps, dets[0]["all"], dets[1]["all"], a.max_err)
                if best is not None:
                    hand3d, err = best[0], best[1]
                    if show_hand:
                        for k, hd in enumerate((best[2], best[3])):
                            for i in (4, 8):
                                cv2.circle(frames[k], tuple(int(v) for v in hd["img_px"][i]), 9, (0, 255, 0), -1)
            if sock:
                pkt = {"t": time.time(), "rec": rec, "ep": n_ep,
                       "hand21": None if hand3d is None else np.round(hand3d, 4).tolist()}
                sock.sendto(json.dumps(pkt).encode(), mirror_addr)
        # 그리기
        disp = []
        for k in range(2):
            img = frames[k] if not rec else frames[k].copy()
            if calibs[k] is not None:
                c = calibs[k].project(CORNERS_TABLE).astype(int)
                cv2.polylines(img, [c.reshape(-1, 1, 2)], True, (255, 160, 0), 3)
                o = calibs[k].project(np.array([[0, 0, 0.0], [0.03, 0, 0], [-0.03, 0, 0], [0, 0.03, 0], [0, -0.03, 0],
                                                [0, 0, 0.10]])).astype(int)
                cv2.line(img, tuple(o[1]), tuple(o[2]), (0, 0, 255), 3)
                cv2.line(img, tuple(o[3]), tuple(o[4]), (0, 0, 255), 3)
                cv2.circle(img, tuple(o[5]), 12, (0, 200, 255), 2)                 # 시작 자리 (+ 위 10 cm)
            s = DISP_H / H
            disp.append(cv2.resize(img, (int(W * s), DISP_H)))
        view = np.hstack(disp)
        lines = []
        if not ready:
            lines.append("looking for the A4 paper in both cameras... (C = find again)")
        elif rec:
            lines.append(f"REC ep {n_ep}  {len(times) / 30.0:.1f}s   SPACE = stop")
        else:
            lines.append(f"ready. hand at the orange circle (10 cm above +), SPACE = start   saved: {n_ep}")
        if hand3d is not None:
            th, ix = hand3d[4], hand3d[8]
            gap = float(np.linalg.norm(th - ix))
            mid = 0.5 * (th + ix)
            closed = gap < task["grip_close_m"]
            lines.append(f"pinch x{mid[0]*100:+.0f} y{mid[1]*100:+.0f} z{mid[2]*100:+.0f} cm  gap {gap*100:.1f} cm "
                         f"-> {'CLOSED' if closed else 'open'}  (stereo {err:.0f}px)")
        if msg and time.monotonic() - msg_t < 3:
            lines.append(msg)
        for i, t in enumerate(lines):
            cv2.putText(view, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if rec else (40, 220, 40), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("h"):
            show_hand = not show_hand
        if key == ord("c"):
            for k in range(2):
                finders[k].reset()
                finding[k] = True
            calibs = [None, None]
        if key == ord(" ") and ready:
            if not rec:
                ep_dir = new_episode_dir(out)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writers = [cv2.VideoWriter(str(ep_dir / f"cam{k}.avi"), fourcc, 30.0, (W, H)) for k in range(2)]
                times, ep_t0, rec = [], t0, True
            else:
                rec = False
                for w_ in writers:
                    w_.release()
                tt = np.array(times)
                np.save(ep_dir / "times.npy", tt.mean(axis=1))
                fps = (len(tt) - 1) / max(1e-6, tt[-1, 0] - tt[0, 0]) if len(tt) > 1 else 30.0
                save_meta(ep_dir, {"source": "real", "task": task["name"], "fps": round(float(fps), 2),
                                   "n_frames": int(len(tt)), "size": [W, H], "cams": [a.cam, a.cam2],
                                   "calib": [calibs[0].to_dict(), calibs[1].to_dict()],
                                   "cam_skew_ms_median": round(1000 * float(np.median(np.abs(tt[:, 0] - tt[:, 1]))), 1),
                                   "object_points": [op["name"] for op in task["object_points"]],
                                   "recorded": time.strftime("%Y-%m-%d %H:%M:%S")})
                last_ep, n_ep = ep_dir, n_ep + 1
                msg, msg_t = f"saved {ep_dir.name} ({len(tt)} frames, {fps:.1f} fps)", time.monotonic()
        if key == ord("d") and last_ep is not None and not rec:
            if time.monotonic() - del_armed < 2.0:
                shutil.rmtree(last_ep, ignore_errors=True)
                msg, msg_t, last_ep, n_ep = f"deleted {last_ep.name}", time.monotonic(), None, max(0, n_ep - 1)
            else:
                del_armed = time.monotonic()
                msg, msg_t = "press D again within 2 s to delete the last demo", time.monotonic()
    if rec:
        for w_ in writers:
            w_.release()
    for g in grabs:
        g.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
