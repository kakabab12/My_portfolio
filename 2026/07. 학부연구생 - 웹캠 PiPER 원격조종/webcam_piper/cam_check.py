"""카메라 자리 맞추기 미리보기 (2026-11-18, ABKO APC930 두 대).

연결된 카메라를 나란히 띄우고, A4 종이를 찾으면 파란 테두리와 카메라 자리(종이 + 기준 앞·왼쪽·위 cm)를 보여 준다.
화면을 보면서 카메라를 숙여 종이가 화면 가운데 아래쪽에 오게 맞추면 된다. 끝은 Q.

저절로 못 찾을 때 (흰 책상·반사·케이블, 11/18 연구실): 그 카메라 화면에서 종이 모서리 네 개를 아무 순서로 클릭한다.
  순서는 paper.order_paper 가 맞춘다 (A4 가로 비율, 카메라는 사람 맞은편). R = 클릭 지우기.
  S = 두 카메라 보정을 calib.json 에 저장 → teleop.py --cam 0 --cam2 2 가 그대로 쓴다 (손 보정은 teleop 에서 손 펴서 1초).

  .venv/bin/python cam_check.py              # 연결된 카메라를 모두 (리눅스: /dev/video0, 2, ...)
  .venv/bin/python cam_check.py --cams 0 2
"""
import argparse
import glob
import os
import threading
import time

import cv2
import numpy as np

import json

from geometry import CORNERS_TABLE, TableCalib
from paper import find_paper, order_paper

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_W = 640


def capture_nodes():
    """영상을 내는 /dev/video 번호만 (카메라 하나에 노드가 둘씩 생기는데 둘째는 메타데이터)."""
    out = []
    for d in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            if open(os.path.join(d, "index")).read().strip() == "0":
                out.append(int(os.path.basename(d)[5:]))
        except OSError:
            pass
    return out


class Cam:
    def __init__(self, idx, w, h):
        self.idx = idx
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if os.name != "nt" else cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame, self.n, self.fps = None, 0, 0.0
        self.paper, self.lock, self.run = None, threading.Lock(), self.cap.isOpened()
        self.clicks, self.clicked, self.size = [], None, (w, h)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._find, daemon=True).start()

    def _read(self):
        t0, k = time.monotonic(), 0
        while self.run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame, self.n = f, self.n + 1
            k += 1
            if time.monotonic() - t0 > 1.0:
                self.fps, t0, k = k / (time.monotonic() - t0), time.monotonic(), 0

    def _find(self):
        while self.run:
            with self.lock:
                f = None if self.frame is None else self.frame.copy()
            self.paper = None if f is None else find_paper(f)
            time.sleep(0.2)

    def close(self):
        self.run = False
        time.sleep(0.1)
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs="*", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--calib", default=os.path.join(HERE, "calib.json"))
    a = ap.parse_args()
    idxs = a.cams if a.cams else capture_nodes()
    if not idxs:
        raise SystemExit("카메라가 없습니다 (USB 를 확인하세요)")
    cams = [Cam(i, a.width, a.height) for i in idxs]
    print("카메라:", ", ".join(f"{c.idx}{'' if c.run else ' (못 엶)'}" for c in cams), " — 창에서 Q 로 끝")
    cv2.namedWindow("cam_check")
    last, t_scan = {}, time.monotonic()
    note = {"msg": "", "t": 0.0}

    def on_mouse(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        k = x // TILE_W
        if k >= len(cams):
            return
        c = cams[k]
        w, h = c.size
        sc = w / TILE_W
        if len(c.clicks) >= 4:
            c.clicks, c.clicked = [], None
        c.clicks.append(((x - k * TILE_W) * sc, y * sc))
        if len(c.clicks) == 4:
            q = np.array(c.clicks, np.float64)
            ctr = q.mean(axis=0)
            q = q[np.argsort(np.arctan2(q[:, 1] - ctr[1], q[:, 0] - ctr[0]))]      # 볼록 네모 순서로
            o = order_paper(q, w, h)
            if o is None:
                note["msg"], note["t"] = f"cam {c.idx}: those 4 clicks do not look like the A4 from the front - R and click again", time.monotonic()
                print(f"[cam {c.idx}] 클릭한 네 점이 A4 로 맞지 않음 (카메라가 사람 맞은편이어야 함) — R 로 지우고 다시")
            else:
                c.clicked = (o[0], o[1])
                pos = np.round(o[1].cam_pos * 100).astype(int)
                print(f"[cam {c.idx}] 클릭으로 종이 잡음: 카메라 자리 앞 {pos[0]} 왼쪽 {pos[1]} 위 {pos[2]} cm, 초점 {o[1].f:.0f}px")
    cv2.setMouseCallback("cam_check", on_mouse)
    while True:
        if not a.cams and time.monotonic() - t_scan > 2.0:          # 새로 꽂은 카메라를 2초마다 찾아 더함
            t_scan = time.monotonic()
            have = {c.idx for c in cams}
            for i in capture_nodes():
                if i not in have:
                    cams.append(Cam(i, a.width, a.height))
                    print(f"[cam {i}] 새로 연결됨")
            for c in [c for c in cams if not os.path.exists(f"/dev/video{c.idx}")]:
                print(f"[cam {c.idx}] 연결 끊김")
                c.close()
                cams.remove(c)
            if not cams:
                time.sleep(0.5)
                continue
        tiles = []
        for c in cams:
            with c.lock:
                f = None if c.frame is None else c.frame.copy()
            if f is None:
                f = np.zeros((a.height, a.width, 3), np.uint8)
            h, w = f.shape[:2]
            cv2.line(f, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
            cv2.line(f, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
            p = c.clicked if c.clicked is not None else c.paper
            for q in c.clicks:
                cv2.circle(f, (int(q[0]), int(q[1])), 12, (0, 0, 255), -1)
            if p is not None:
                corners, cal = p
                cv2.polylines(f, [np.round(corners).astype(np.int32)], True, (255, 200, 0), 5)
                cv2.drawMarker(f, tuple(cal.project([[0, 0, 0]])[0].astype(int)), (0, 200, 255), cv2.MARKER_CROSS, 50, 5)
                pos = np.round(cal.cam_pos * 100).astype(int)
                txt = [f"cam {c.idx}: PAPER OK" + (" (clicked)" if c.clicked is not None else ""),
                       f"camera at fwd {pos[0]} left {pos[1]} up {pos[2]} cm, f {cal.f:.0f}px"]
                col = (0, 255, 0)
                if last.get(c.idx) is None:
                    print(f"[cam {c.idx}] 종이 찾음: 카메라 자리 앞 {pos[0]} 왼쪽 {pos[1]} 위 {pos[2]} cm, 초점 {cal.f:.0f}px")
            else:
                txt = [f"cam {c.idx}: no paper - tilt so the whole A4 is in view,",
                       "or click its 4 corners (any order)"]
                col = (0, 0, 255)
                if last.get(c.idx) is not None:
                    print(f"[cam {c.idx}] 종이를 놓침")
            last[c.idx] = p
            tile = cv2.resize(f, (TILE_W, int(TILE_W * h / w)))
            for i, s in enumerate(txt + [f"{c.fps:.0f} fps, {w}x{h}"]):
                cv2.putText(tile, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(tile, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col if i == 0 else (255, 255, 255), 1)
            tiles.append(tile)
        hh = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.zeros((hh - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
        board = np.hstack(tiles)
        bar = np.zeros((44, board.shape[1], 3), np.uint8)
        hint = note["msg"] if time.monotonic() - note["t"] < 6 else "S = save both for teleop   R = clear clicks   Q = quit"
        cv2.putText(bar, hint, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("cam_check", np.vstack([board, bar]))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            for c in cams:
                c.clicks, c.clicked = [], None
        if key == ord("s"):
            got = [(c.idx, c.clicked if c.clicked is not None else c.paper) for c in sorted(cams, key=lambda c: c.idx)]
            missing = [i for i, g in got if g is None]
            if missing or len(got) < 2:
                note["msg"], note["t"] = f"not saved: paper missing on cam {missing}" if missing else "need two cameras", time.monotonic()
                print(f"[저장 안 함] 종이를 못 잡은 카메라: {missing}" if missing else "[저장 안 함] 카메라가 두 대여야 함")
            else:
                d = {}
                if os.path.exists(a.calib):
                    try:
                        d = json.load(open(a.calib, encoding="utf-8"))
                    except (OSError, ValueError):
                        d = {}
                for k_ in ("scale", "flat_hand_m", "mirror", "mirror_votes", "palm_off_m", "palm_w_l_m", "pose"):
                    d.pop(k_, None)                                   # 손 보정은 teleop 에서 다시 (손 펴서 1초)
                (i1, g1), (i2, g2) = got[0], got[1]
                d.update({"table": g1[1].to_dict(), "table2": g2[1].to_dict(), "cam": i1, "cam2": i2,
                          "paper_source": "cam_check"})
                json.dump(d, open(a.calib, "w", encoding="utf-8"), indent=1)
                note["msg"], note["t"] = f"saved: cam {i1} + cam {i2} -> {os.path.basename(a.calib)}. Now quit (Q) and start teleop", time.monotonic()
                print(f"[저장] 카메라 {i1}, {i2} 종이 보정 → {a.calib}. Q 로 끄고 teleop.py --cam {i1} --cam2 {i2} --send 127.0.0.1")
    for c in cams:
        c.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
