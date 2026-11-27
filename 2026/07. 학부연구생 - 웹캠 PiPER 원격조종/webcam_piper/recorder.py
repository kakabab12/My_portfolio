"""시범 기록 + 로봇 쪽 카메라 읽기 (2026-11-15).

에피소드 하나 = 파일 하나 (ep_000.npz). 박자마다:
  t (초), q (실제 관절 6, rad), grip (실제 집게 폭, m), q_cmd (보낸 관절 목표 6), grip_cmd (보낸 집게 폭),
  hand (사람 집기점, 사람 책상 좌표 m), hand_yaw (rad), hand_gap (엄지·검지 사이, m), target (로봇 집게 끝 목표, 로봇 좌표 m),
  tracked (손을 찾았는지), human_jpg (웹캠 사진 JPEG, 가로 480), 그리고 로봇 쪽 카메라가 있으면 wrist_jpg / front_jpg.
11/17 부터: lm2d (손 21점 화면 px), lm3d (MediaPipe 손 3D, m), hand_raw (거르기 전 집기점 x y z, yaw, gap) — 손을 못 찾은 장면은 nan.
연구실 PC 기록은 노트북 보정 값을 ep_XXX.json 에 같이 남긴다. replay_raw.py 로 다른 방식(PnP / OAT)으로 다시 풀어 볼 수 있다.
찬혁님 VR 조종 기록(zarr: obs/qpos = 관절 6 + 집게 폭, action/qpos, obs/wrist_img·front_img)과 같은 값이 다 들어 있어서
나중에 같은 형식으로 바꿀 수 있다. 사진은 JPEG 바이트라 np.load(..., allow_pickle=True) 로 읽는다.
"""
import json
import os
import time

import cv2
import numpy as np


def jpeg(img, width=None, quality=85):
    if img is None:
        return b""
    if width and img.shape[1] != width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class Recorder:
    def __init__(self, out_dir, meta):
        self.dir = os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S"))
        self.meta = meta                                     # 폴더는 첫 기록을 저장할 때 만든다
        self.rows = None
        self.saved = []
        self.extra = {}                                      # 있으면 ep_XXX.json 으로 같이 저장 (노트북 보정 등, 11/17)

    @property
    def on(self):
        return self.rows is not None

    def start(self):
        self.rows = []

    def add(self, **row):
        if self.rows is not None:
            self.rows.append(row)

    def stop(self, keep=True):
        rows, self.rows = self.rows, None
        if not keep or not rows:
            return None
        out = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            if isinstance(vals[0], (bytes, bytearray)):
                arr = np.empty(len(vals), dtype=object)
                arr[:] = vals
                out[k] = arr
            else:
                out[k] = np.asarray(vals)
        if not os.path.exists(os.path.join(self.dir, "meta.json")):
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=1)
        path = os.path.join(self.dir, f"ep_{len(self.saved):03d}.npz")
        np.savez(path, **out)
        if self.extra:
            with open(path[:-4] + ".json", "w", encoding="utf-8") as f:
                json.dump(self.extra, f, ensure_ascii=False, indent=1)
        self.saved.append(path)
        return path, len(rows)

    def discard_last(self):
        if self.saved:
            path = self.saved.pop()
            os.remove(path)
            return path
        return None


def open_camera(index, width=None, height=None):
    """웹캠을 열고 첫 장을 읽는다. 윈도우 노트북 중에는 기본 방식으로 안 열리는 것이 있어 DirectShow 로 한 번 더 해 본다.
    돌려줌: (cap, 첫 장) 또는 (None, None)."""
    apis = [cv2.CAP_ANY] + ([cv2.CAP_DSHOW] if os.name == "nt" else [])
    # 11/18: 먼저 MJPG·초당 30장을 청해 본다 (노트북 기본 방식은 1280x720 에서 초당 18장쯤이었음). 안 되면 예전처럼
    for mjpg in (True, False):
        for api in apis:
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                if mjpg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if mjpg:
                    cap.set(cv2.CAP_PROP_FPS, 30)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, frame
            cap.release()
    return None, None


class CamGrabber:
    """카메라를 따로 읽는 일꾼. 기다리지 않고 가장 새 사진을 꺼낸다 (로봇 쪽 기록 카메라, 11/18 부터 두 번째 손 카메라).
    mjpg: MJPG·초당 fps 장을 청함 (ABKO APC930 은 압축 안 하면 1280x720 에서 초당 10장).
    lock_focus_after: 켠 뒤 이 초가 지나면 자동 초점을 끈다 (초점이 움직이면 초점거리도 조금 바뀌어 삼각측량이 틀어짐)."""

    def __init__(self, index, width=640, height=480, mjpg=False, fps=None, lock_focus_after=None):
        import threading
        self.cap = cv2.VideoCapture(index)
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.ok = self.cap.isOpened()
        self.frame, self.t, self.n = None, None, 0
        self.lock_focus_after = lock_focus_after
        self.lock = threading.Lock()
        self.running = self.ok
        self.thread = threading.Thread(target=self._run, daemon=True)
        if self.ok:
            self.thread.start()

    def _run(self):
        t0 = time.monotonic()
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.t, self.n = f, time.monotonic(), self.n + 1
            else:
                time.sleep(0.01)
            if self.lock_focus_after is not None and time.monotonic() - t0 > self.lock_focus_after:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)            # 읽는 쓰레드에서 바꿔야 안전
                self.lock_focus_after = None

    def latest(self):
        with self.lock:
            return self.frame

    def latest_with_time(self):
        """(사진, 찍힌 때 monotonic, 몇 번째 사진) — 같은 사진을 두 번 쓰지 않게."""
        with self.lock:
            return self.frame, self.t, self.n

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


class FakeCam:
    """시험용 가짜 로봇 카메라 (11/26, robot_server --fake-cams). 초당 fps 장, 640x480, 장 번호가 찍힌 그림.
    CamGrabber 와 같은 꼴 (ok, latest, latest_with_time, close). 실물 카메라 없이 기록·tinybc 변환 길을 시험할 때만."""

    def __init__(self, name, width=640, height=480, fps=30.0):
        self.ok = True
        self.name, self.w, self.h, self.fps = name, width, height, fps
        self.t0 = time.monotonic()
        self.n, self.frame, self.t = -1, None, None

    def latest_with_time(self):
        now = time.monotonic()
        n = int((now - self.t0) * self.fps)
        if n != self.n:
            img = np.zeros((self.h, self.w, 3), np.uint8)
            img[:, :, 0] = np.linspace(0, 255, self.w, dtype=np.uint8)[None, :]          # 파랑: 가로로 밝아짐
            img[:, :, 1] = np.linspace(0, 255, self.h, dtype=np.uint8)[:, None]          # 초록: 세로로 밝아짐
            x = int((n * 7) % (self.w - 80))
            img[200:280, x:x + 80, 2] = 255                                               # 빨간 네모가 옆으로 감
            cv2.putText(img, f"{self.name} {n}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            self.n, self.frame, self.t = n, img, now
        return self.frame, self.t, self.n

    def latest(self):
        return self.latest_with_time()[0]

    def close(self):
        pass
