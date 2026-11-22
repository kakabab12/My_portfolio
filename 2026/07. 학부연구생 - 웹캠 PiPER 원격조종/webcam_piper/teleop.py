"""웹캠 하나로 PiPER 원격조종 + 시범 기록 (2026-11-15).

My_portfolio 2026/05 웹캠 조종(SO-ARM101, 팔 관절 각도를 그대로 옮김)을 PiPER 용으로 바꾼 것.
PiPER 는 팔 길이가 사람과 달라 관절 각도를 옮기면 집게가 엉뚱한 곳에 간다. 그래서 여기서는
**사람 엄지·검지 끝 가운데(집기점)가 움직인 만큼 집게 끝을 똑같이 움직인다** (mapping.py). 관절은 역기구학이 푼다.

실행
  python teleop.py                     # 팔 없이 화면 미리보기 (처음엔 이것부터)
  python teleop.py --robot agxbus      # 실물 PiPER 가 이 컴퓨터에 붙어 있을 때 (agx-bus, CAN can0)
  python teleop.py --send 192.168.0.10 # 팔은 연구실 PC 에: 손 측정만 보내고, 연구실 PC 에서 robot_server.py 가 팔을 움직임
화면 글자는 영어다 (OpenCV 가 한글을 못 그림). 키:
  SPACE 따라가기 켜기/끄기   R 기록 시작/끝   X 마지막 기록 지우기   G 집게 따라가기   Y 손 돌림 따라가기
  [ 집게 약하게 ] 세게 (5 mm)   --grip pinch 면 엄지·검지로 집음 (기본은 손 전체)   S 손 보정 다시   Q 또는 ESC 끝
  C 종이 다시 찾기 (저절로)   M 종이 모서리를 손으로 누르기 (저절로 못 찾을 때)
  Z 손 닿는 높이 맞추기 (11/19): 엄지·손가락 끝을 종이에 대고 누르면 그 높이를 0 으로 → 손끝이 책상에 닿을 때 그리퍼 끝도 책상 높이로
    (서버는 --lift 없이 켜야 함. 책상을 누르지 않게 서버 --min-z 바닥은 남음)
보정 (11/18 부터 저절로): 종이가 보이면 모서리를 스스로 찾고, 손을 + 위에 펴서 1초쯤 가만히 두면 손 보정이 된다.
  저장된 보정으로 켜면 처음 몇 초 동안 종이를 다시 찾아, 노트북이나 종이가 움직였으면 다시 보정한다. --manual-calib 이면 예전처럼.
  P 위치 확인: 다섯 손가락 끝을 모아 종이 모서리나 + 에 대고 누르면, 가장 가까운 점과 몇 cm 다른지 띄우고 checks.jsonl 에 남김
  TAB (또는 H) 손 모양 바꾸기: 손 전체 ↔ 엄지·검지 (11/17). 따라가기 중이면 멈추므로 SPACE 로 다시. 쥐는 세기는 모양마다 따로 기억
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from geometry import (CORNER_NAMES, CORNERS_TABLE, A4_D, A4_W, FLAT_H, GRIPS, PALM, FlatHandGate, HandMeasure, StereoHand,
                      TableCalib, backproject_height, flat_fit, palm_size, scale_for_flat)
from hand import HandTracker, Median3, OneEuro, StillZone, draw_hand
from mapping import Mapper
from net import PORT, Link, code_version
from paper import PaperFinder
from panel import DISP_H, robot_panel
from piper_kin import HOME_Q, PiperKin
from recorder import CamGrabber, Recorder, jpeg, open_camera
from robots import make_arm

HERE = os.path.dirname(os.path.abspath(__file__))
BACKSPACE = (8, 127)                                       # 127 = 맥의 delete 키
GRIP_KEYS = (9, ord("h"))                                  # TAB 은 한글 입력 상태에서도 먹힘
# 손 모양마다: 집게를 더 닫는 양 기본값, 손가락 사이가 이보다 좁으면 집게 방향을 새로 읽지 않음, 화면 이름
GRIP_SQUEEZE = {"whole": 0.025, "pinch": 0.02}
GRIP_YGM = {"whole": 0.05, "pinch": 0.025}
GRIP_NAME = {"whole": "WHOLE HAND", "pinch": "THUMB+INDEX"}
# P 로 위치를 확인할 점: 종이 네 모서리와 + (사람 책상 좌표)
CHECK_PTS = [(n.split(" ", 1)[1], CORNERS_TABLE[i]) for i, n in enumerate(CORNER_NAMES)] + [("center", np.zeros(3))]


def check_point(meas, scale, path):
    """P (11/16): 다섯 손가락 끝을 모아 종이 모서리나 + 에 댄 채 누른다. 가장 가까운 점과의 차이를 돌려주고 path 에 한 줄 더한다.
    손가락 끝을 모으면 집기점이 그 끝에 오므로, 차이가 곧 위치 오차다 (높이는 손가락 끝 살 두께만큼, 1 cm 안팎이 맞음)."""
    if not meas:
        return "P: hand not found"
    p = np.asarray(meas["pinch"], float)
    name, ref = min(CHECK_PTS, key=lambda c: np.linalg.norm(p[:2] - c[1][:2]))
    d = (p - ref) * 100
    row = {"t": round(time.time(), 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "point": name, "scale": round(scale, 4),
           "hand_cm": np.round(p * 100, 2).tolist(), "off_cm": np.round(d, 2).tolist(), "gap_mm": round(meas["gap"] * 1000, 1)}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[확인] {name}: x {d[0]:+.1f} y {d[1]:+.1f} cm 차이, 높이 {p[2] * 100:.1f} cm, 손가락 사이 {meas['gap'] * 1000:.0f} mm")
    return f"P: {name} off by x {d[0]:+.1f} y {d[1]:+.1f} cm, z {p[2] * 100:.1f} cm (saved)"


def _load_guide(width=400):
    """보정 2/2 화면에 띄울 시작 자세 그림 (img/guide_start.png). 없으면 글만."""
    g = cv2.imread(os.path.join(HERE, "img", "guide_start.png"))
    if g is None:
        return None
    return cv2.resize(g, (width, int(g.shape[0] * width / g.shape[1])), interpolation=cv2.INTER_AREA)


GUIDE = _load_guide()


def parse():
    ap = argparse.ArgumentParser(description="웹캠 PiPER 원격조종")
    ap.add_argument("--cam", type=int, default=0, help="웹캠 번호 (노트북 내장 = 0)")
    ap.add_argument("--cam2", type=int, default=None,
                    help="두 번째 카메라 번호: 두 화면 삼각측량 (11/18, Point Policy 처럼). 리눅스는 카메라 하나에 /dev/video 가 둘씩이라 보통 2")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960, help="ABKO APC930 은 4:3 센서라 1280x960 이 화면 전체 (11/17 까지 720)")
    ap.add_argument("--af", action="store_true", help="자동 초점을 켜 둠 (기본: 켠 뒤 3초에 끔 — 초점이 움직이면 초점거리도 바뀜)")
    ap.add_argument("--hfov", type=float, default=None, help="카메라 가로 시야각(도). 모르면 비워 두면 종이로 추정")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--robot", choices=["preview", "agxbus", "sdk"], default="preview")
    ap.add_argument("--send", default=None, metavar="주소[:포트]",
                    help=f"팔이 붙은 연구실 PC 주소. 주면 손 측정만 보내고 팔은 그쪽 robot_server.py 가 움직임 (포트 기본 {PORT})")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--speed", type=int, default=30, help="실물 팔 속도 %%")
    ap.add_argument("--x0", type=float, default=0.30, help="로봇 시작 표시: 밑동에서 앞으로 (m)")
    ap.add_argument("--y0", type=float, default=0.0, help="로봇 시작 표시: 밑동에서 왼쪽으로 (m)")
    ap.add_argument("--gain", type=float, default=1.0, help="사람 1 cm → 로봇 몇 cm (같은 물체면 1.0)")
    ap.add_argument("--max-speed", type=float, default=0.12, help="집게 끝 최고 속도 (m/s)")
    ap.add_argument("--grip", choices=["whole", "pinch"], default="whole",
                    help="whole: 엄지 ↔ 네 손가락으로 손 전체로 쥠 (기본), pinch: 엄지 ↔ 검지로 집음")
    ap.add_argument("--squeeze", type=float, default=None,
                    help="집게 폭 = 엄지와 맞은편 손가락 사이 − 이 값 (m). 기본 whole 0.025, pinch 0.02")
    ap.add_argument("--calib", default=os.path.join(HERE, "calib.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--checks", default=os.path.join(HERE, "checks.jsonl"), help="P 로 잰 위치 확인을 남길 파일")
    ap.add_argument("--solver", choices=["oat", "pnp", "palm"], default="oat",
                    help="손 위치 푸는 방식. oat: 손 방향은 MediaPipe 것, 위치만 풂 (OAT 논문, 11/18 기본) / "
                         "pnp: 21점으로 방향·위치 다 풂 (11/15~17, 손가락 모양이 틀리면 높이가 크게 틀림) / palm: 손바닥 다섯 점으로 거리")
    ap.add_argument("--pen", type=float, default=0.0,
                    help="oat 의 앞 장면 벌점 (클수록 덜 흔들리고 더 늦게 따라감). 11/18 부터 0: 흔들림은 1€ 필터가 거름")
    ap.add_argument("--no-steady", action="store_true",
                    help="11/20 손떨림·지터 거르기(멈춰 있을 때 가운데 값 + 멈춤 구역)를 끔 — 11/18 처럼 1€ 필터만")
    ap.add_argument("--still-mm", type=float, default=3.0,
                    help="멈춤 구역 반지름 (mm). 손이 이 안에서 떨리면 로봇 목표를 그대로 둠. 클수록 덜 떨고 움직일 때 이만큼 뒤처짐")
    ap.add_argument("--status", default=None, metavar="폴더",
                    help="이 폴더에 상태(status.json, 0.25초마다)와 화면(view.jpg, 1초마다)을 남김 — 다른 프로그램이 실시간으로 보게 (11/18)")
    ap.add_argument("--manual-calib", action="store_true",
                    help="보정을 손으로만: 종이 모서리 네 번 누르기, 손 펴고 S (11/18 부터 기본은 저절로)")
    ap.add_argument("--wrist-cam", type=int, default=None, help="기록할 로봇 손목 카메라 번호 (선택)")
    ap.add_argument("--front-cam", type=int, default=None, help="기록할 로봇 앞/위 카메라 번호 (선택)")
    return ap.parse_args()


class Smoother:
    """손 측정 거르기. 11/16 검토에서 고침: 손을 다시 찾은 첫 장면이 엉뚱하면 그 값에 계속 묶여 있던 문제.
    - 새로 잡을 때: 3장이 서로 3 cm 안에서 맞아야 받아들인다 (그 전에는 '손 없음').
    - 잡은 뒤 한 장면에 15 cm 넘게 튀면 버린다. 다만 튄 쪽 값이 5장 연달아 서로 3 cm 안이면 손이 정말 옮겨 간 것으로 보고 다시 잡는다."""
    ACQUIRE_N, REJUMP_N, AGREE = 3, 5, 0.03
    STILL_V = 0.05                     # 손이 이보다 느리면(m/s) 멈춘 것으로 보고 앞 3장면 가운데 값을 쓴다

    def __init__(self, steady=True, still_m=0.003, still_deg=3.0):
        # 11/18: beta 를 m·rad 단위에 맞춤. 예전 (1.5, 0.5) 는 라디안용이라 손을 빨리 움직이면 70 ms 늦고 3 cm 뒤처졌다
        # (20 fps·잡음 3 mm 흉내: 멈춤 흔들림 4.8 → 1.6 mm, 움직일 때 늦음 70 → 15 ms)
        self.pinch, self.yaw, self.gap = OneEuro(1.0, 20.0), OneEuro(1.0, 5.0), OneEuro(1.0, 20.0)
        # 11/20 지터·손떨림: 1€ 앞에 '멈춰 있을 때만 가운데 값'(한 장면 튐 버림), 뒤에 멈춤 구역(반지름 안 떨림은 그대로 둠).
        # 가상 손에서 멈춤 흔들림 3.4 → 1.3 mm (가장 크게 튈 때 19.5 → 4.7 mm), 늦음 +13 ms. steady=False 면 11/18 그대로
        self.steady = steady
        self.med = Median3()
        self.zone_p = StillZone(still_m if steady else 0.0)
        self.zone_y = StillZone(math.radians(still_deg) if steady else 0.0)
        self.last = None
        self.yaw_unwrapped = None
        self.lost_since = None
        self.pending = []

    def reset(self):
        for f in (self.pinch, self.yaw, self.gap, self.med, self.zone_p, self.zone_y):
            f.reset()
        self.last = self.yaw_unwrapped = None
        self.pending = []

    def _agree(self, n):
        if len(self.pending) < n:
            return False
        ref = self.pending[-1]["pinch"]
        return all(np.linalg.norm(p["pinch"] - ref) < self.AGREE for p in self.pending[-n:])

    def __call__(self, t, m):
        if m is None:
            if self.lost_since is None:
                self.lost_since = t
            if t - self.lost_since > 0.5:
                self.reset()
            self.pending = []
            return None
        self.lost_since = None
        if self.last is None:
            self.pending = (self.pending + [m])[-self.ACQUIRE_N:]
            if not self._agree(self.ACQUIRE_N):
                return None
            self.pending = []
        elif np.linalg.norm(m["pinch"] - self.last["pinch"]) > 0.15:
            self.pending = (self.pending + [m])[-self.REJUMP_N:]
            if not self._agree(self.REJUMP_N):
                return self.last                               # 한 장면에 15 cm 넘게 튀면 잘못 찾은 것으로 본다
            self.reset()                                       # 손이 정말 옮겨 감: 새로 잡는다
        else:
            self.pending = []
        y = m["yaw"]
        if self.yaw_unwrapped is not None:
            y = self.yaw_unwrapped + math.remainder(y - self.yaw_unwrapped, 2 * math.pi)
        self.yaw_unwrapped = y
        p = np.asarray(m["pinch"], np.float64)
        if self.steady:
            med = self.med(p)
            slow = self.pinch.dx is None or float(np.linalg.norm(self.pinch.dx)) < self.STILL_V
            p = med if slow else p
        out = {"pinch": self.zone_p(t, self.pinch(t, p)), "yaw": float(self.zone_y(t, self.yaw(t, y))),
               "gap": float(self.gap(t, m["gap"]))}
        self.last = out
        return out


def main():
    a = parse()
    kin = PiperKin()
    link = Link(a.send) if a.send else None
    arm = make_arm("preview" if link else a.robot, a.can, a.speed)
    if link:
        print(f"[연결] 손 측정을 {link.addr[0]}:{link.addr[1]} 로 보냄. 연구실 PC 에서 robot_server.py 를 켜 두세요.")
    if arm.is_real:
        print("\n[안전] 실물 팔입니다. 팔 둘레를 비우고, 전원 스위치 옆에 사람이 있어야 합니다.")
        print("       SPACE 를 눌러야 따라가기 시작하고, 손을 1초 넘게 놓치면 멈춥니다. 속도 %d%%" % a.speed)
        input("       준비됐으면 Enter (취소 Ctrl+C) ")
    grip_mode = a.grip                                     # TAB 으로 프로그램 켠 채 바꿈
    cal_solver = "oat" if a.solver == "palm" else a.solver  # 보정 중 거울상 판별·배율은 이 방식으로 (palm 은 보정이 끝나야 쓸 수 있음)
    squeeze_by = dict(GRIP_SQUEEZE)
    if a.squeeze is not None:
        squeeze_by[grip_mode] = a.squeeze
    mapper = Mapper(kin, a.x0, a.y0, a.gain, max_speed=a.max_speed, squeeze=squeeze_by[grip_mode],
                    yaw_gap_min=GRIP_YGM[grip_mode],
                    yaw_pref=math.radians(-60 if a.hand == "right" else 60))
    mapper.sync_to(arm.joints())

    cap, frame = open_camera(a.cam, a.width, a.height)
    if cap is None:
        raise SystemExit(f"웹캠 {a.cam} 을 못 엶 (다른 프로그램이 쓰는 중인지, 카메라 권한, --cam 1 을 보세요)")
    H, W = frame.shape[:2]
    t_open = time.monotonic()
    focus_locked = a.af
    grab2, tracker2 = None, None
    if a.cam2 is not None:
        grab2 = CamGrabber(a.cam2, W, H, mjpg=True, fps=30, lock_focus_after=None if a.af else 3.0)
        t_wait = time.monotonic()
        while grab2.ok and grab2.latest() is None and time.monotonic() - t_wait < 3.0:
            time.sleep(0.05)
        if not grab2.ok or grab2.latest() is None:
            raise SystemExit(f"두 번째 카메라 {a.cam2} 을 못 엶 (번호 확인: 리눅스는 보통 0 과 2)")
        f2 = grab2.latest()
        if f2.shape[:2] != (H, W):
            raise SystemExit(f"두 카메라 크기가 다름: {W}x{H} / {f2.shape[1]}x{f2.shape[0]}")
        print(f"[카메라] 두 대: {a.cam} 과 {a.cam2}, {W}x{H}. 둘 다 손을 보면 삼각측량, 한 대만 보면 그 한 대로")
    extra = {} if link else {k: CamGrabber(i) for k, i in (("wrist", a.wrist_cam), ("front", a.front_cam))
                             if i is not None}

    tracker = HandTracker(a.hand)
    tracker2 = HandTracker(a.hand) if grab2 else None
    pool2 = None
    if grab2:
        from concurrent.futures import ThreadPoolExecutor
        pool2 = ThreadPoolExecutor(max_workers=1)
    smooth = Smoother(steady=not a.no_steady, still_m=a.still_mm / 1000.0)
    calib, measure, mode, clicks = None, None, "corners", []
    calib2, stereo, saved = None, None, {}
    if os.path.exists(a.calib):
        with open(a.calib, encoding="utf-8") as f:
            saved = json.load(f)
        t2 = saved.get("table2")
        if grab2 and t2 and t2["w"] == W and t2["h"] == H and saved.get("cam2") == a.cam2:
            calib2 = TableCalib.from_dict(t2)
            print(f"[보정] 두 번째 카메라 보정 불러옴 (초점 {calib2.f:.0f}px)")
        if "table" in saved and saved["table"]["w"] == W and saved["table"]["h"] == H:
            calib = TableCalib.from_dict(saved["table"])
            if saved.get("label") and saved.get("hand", "right") == a.hand:
                tracker.want_label = saved["label"]
            if "scale" not in saved:
                # cam_check.py 에서 종이만 맞추고 저장한 것: 손 보정만 하면 됨 (손 펴서 1초, 11/18)
                measure, mode = HandMeasure(calib, 1.0, grip_mode, solver=cal_solver), "hand"
                print("[보정] 종이 보정을 불러옴 (cam_check). 손을 종이 + 위에 펴서 1초 두면 손 보정")
            elif a.solver == "palm" and not saved.get("flat_hand_m"):
                # 종이 모서리는 그대로 쓰고 손 펴서 올려놓기만 다시 (손바닥 다섯 점의 진짜 크기를 재야 함)
                measure, mode = HandMeasure(calib, 1.0, grip_mode, solver="oat"), "hand"
                print("[보정] --solver palm 은 손바닥 보정이 있어야 합니다: 손을 종이에 펴서 S")
            else:
                measure = HandMeasure(calib, 1.0 if a.solver == "palm" else saved["scale"], grip_mode,
                                      saved.get("mirror", False), a.solver, a.pen, flat=saved.get("flat_hand_m"))
                if saved.get("solver", "pnp") != a.solver:
                    print(f"[보정] 저장된 보정은 --solver {saved.get('solver', 'pnp')} 로 한 것 (지금 {a.solver}). "
                          "손 크기 배율이 조금 다를 수 있으니 S 로 손 보정만 다시 하세요")
                mode = "run"
                print(f"[보정] {a.calib} 불러옴 (거리 배율 {saved['scale']:.3f}, 방식 {a.solver})")
    # 11/18 저절로 보정: 종이가 보이면 모서리를 스스로 찾고, 손을 + 위에 펴서 가만히 두면 손 보정을 시작한다.
    # 저장된 보정을 불러왔으면 켠 뒤 몇 초 동안 종이를 찾아 보고, 노트북이나 종이가 움직였으면 다시 보정한다.
    auto = not a.manual_calib
    finder, gate = PaperFinder(a.hfov), None
    # cam_check.py 에서 사람이 확인(클릭)하고 저장한 종이는 다시 찾지 않는다 (흰 책상에서 엉뚱한 네모로 덮어쓸 수 있음, 11/18)
    trusted = os.path.exists(a.calib) and saved.get("paper_source") == "cam_check"
    recheck_until = time.monotonic() + 4.0 if (auto and mode == "run" and not trusted) else None
    # 두 번째 카메라는 언제나 저절로 (종이 모서리를 누르는 화면이 하나뿐이라). 불러온 보정은 켠 뒤 4초 동안 다시 확인
    finder2 = PaperFinder(a.hfov)
    recheck2_until = time.monotonic() + 4.0 if (grab2 and calib2 is not None and not trusted) else None
    det2, frame2, n2_seen, t_stereo, src, last_pinch = None, None, -1, -1e9, 0, None
    hand_z0 = float(saved.get("hand_touch_z", 0.0) or 0.0)   # 손끝이 종이에 닿았을 때의 집기점 높이 (Z 로 잼, 11/19)
    z_collect = None
    if hand_z0:
        print(f"[보정] 손 닿는 높이 {hand_z0 * 100:.1f} cm 를 0 으로 씀 (Z 로 다시 잼)")

    def save_table2():
        d = {}
        if os.path.exists(a.calib):
            try:
                with open(a.calib, encoding="utf-8") as f_:
                    d = json.load(f_)
            except (OSError, ValueError):
                d = {}
        d["table2"], d["cam2"] = calib2.to_dict(), a.cam2
        with open(a.calib, "w", encoding="utf-8") as f_:
            json.dump(d, f_, indent=1)
    rec = None if link else Recorder(a.out, {
        "args": vars(a), "robot_start": mapper.robot_start.tolist(),
        "note": "사람 책상 좌표: 원점 A4 가운데, x 앞(노트북 쪽), y 사람 왼쪽, z 위. 로봇 좌표: 밑동 (0,0,0.90)"})
    # 연구실 PC 로 보낼 때: 기록은 그쪽에서 한다. 여기서는 바라는 상태만 보내고, 그쪽이 알려 준 실제 상태를 그린다.
    rec_req, disc_n, seen_msg_n = False, 0, None
    home_n = 0                                               # 11/20: B 를 누른 횟수 (연구실 PC 가 늘 때마다 처음 자세로 감)
    my_ver, ver_warned = code_version(), False
    mismatch = {"follow": None, "rec": None}
    scale_disp = DISP_H / H
    win = "webcam_piper"
    cv2.namedWindow(win)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and mode == "corners" and len(clicks) < 4 and x < W * scale_disp:
            clicks.append((W - 1 - x / scale_disp, y / scale_disp))      # 화면은 거울로 보여 주므로 되돌린다
    cv2.setMouseCallback(win, on_mouse)

    follow, collecting, samples, msg, msg_t = False, False, [], "", 0.0
    status_t = snap_t = 0.0
    last_gate = ""
    if a.status:
        os.makedirs(a.status, exist_ok=True)
    rec_on = (lambda: rec.on) if rec else (lambda: bool(link.status and link.status.get("rec")))
    t_prev = time.monotonic()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic()
        dt, t_prev = t - t_prev, t
        fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
        # 두 카메라 손 찾기를 함께 돌린다 (MediaPipe 는 계산 중에 파이썬 잠금을 풀어서, 따로 돌리면 초당 15장 → 더 빨라짐, 11/18)
        fut2 = None
        if grab2:
            f2_, t2_now, n2_now = grab2.latest_with_time()
            if n2_now != n2_seen and f2_ is not None:
                fut2 = pool2.submit(tracker2.detect, f2_, t2_now * 1000)
        det = tracker.detect(frame, t * 1000)
        if not focus_locked and t - t_open > 3.0:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)                  # 초점을 그 자리에 고정 (11/18)
            focus_locked = True
        if grab2:
            if fut2 is not None:                                # 함께 돌린 것: 그때 넘긴 사진·시각과 짝을 맞춘다
                f2, t2_, n2 = f2_, t2_now, n2_now
            else:
                f2, t2_, n2 = grab2.latest_with_time()
            if t2_ is None or time.monotonic() - t2_ > 0.3:
                det2 = None                                     # 두 번째 카메라 사진이 0.3초 넘게 안 옴 (선이 빠짐 등): 옛 손 점을 쓰지 않음
            if n2 != n2_seen and f2 is not None:
                n2_seen, frame2 = n2, f2
                det2 = fut2.result() if fut2 is not None else tracker2.detect(frame2, t2_ * 1000)
                if calib2 is None or recheck2_until is not None:
                    found2 = finder2.feed(frame2)
                    if recheck2_until is not None and t > recheck2_until:
                        recheck2_until = None
                    elif found2 is not None:
                        moved2 = None if calib2 is None else float(np.max(np.linalg.norm(found2 - calib2.img_pts, axis=1)))
                        recheck2_until = None
                        if moved2 is None or moved2 > 8.0:
                            calib2 = TableCalib(found2, W, H, a.hfov)
                            save_table2()
                            msg, msg_t = "camera 2: paper found" + (f" (moved {moved2:.0f} px)" if moved2 else ""), t
                            print(f"[보정] 두 번째 카메라가 종이를 찾음: 초점 {calib2.f:.0f}px, 카메라 자리 "
                                  f"{np.round(calib2.cam_pos * 100).astype(int).tolist()} cm")
        if calib is not None and calib2 is not None:
            if stereo is None or stereo.calibs[0] is not calib or stereo.calibs[1] is not calib2:
                stereo = StereoHand(calib, calib2, grip_mode)
                if calib.focal_source == "estimate" and calib2.focal_source == "estimate":
                    print(f"[삼각측량] 준비됨. 두 카메라 초점 {calib.f:.0f} / {calib2.f:.0f} px "
                          f"(같은 기종이라 비슷해야 함, 차이 {abs(calib.f - calib2.f) / max(calib.f, calib2.f) * 100:.0f}%)")
        else:
            stereo = None
        raw, src = None, 0
        if stereo is not None and det and det2:
            # 화면마다 손이 둘 보이면 '오른손/왼손' 표시는 옆에서 보는 카메라에서 자주 틀린다 (11/18 실물: 카메라 2 가 몸 쪽 다른 손을 잡아 63 px).
            # 두 화면 손의 모든 짝을 삼각측량해서 가장 잘 맞는 짝을 쓴다
            # 두 손이 다 맞으면 프레임마다 왼손·오른손을 오갈 수 있다 → 방금(0.5초 안) 따라가던 손에 가까운 것을 고른다 (실물 안전)
            best, cands = None, []
            for h1 in det.get("all", [det]):
                for h2 in det2.get("all", [det2]):
                    r_ = stereo(h1["img_px"], h2["img_px"])
                    cands.append((stereo.last_err, r_, h1, h2))
            ok_ = [c for c in cands if c[1] is not None]
            if ok_ and last_pinch is not None and t - t_stereo < 0.5:
                best = min(ok_, key=lambda c: np.linalg.norm(c[1]["pinch"] - last_pinch))
                if np.linalg.norm(best[1]["pinch"] - last_pinch) > 0.15:     # 0.5초 안에 15 cm 넘게 옮긴 손은 다른 손으로 본다
                    best = None
            elif ok_:
                best = min(ok_, key=lambda c: c[0])
            if best is None:
                stereo.last_err = min(c[0] for c in cands)
            else:
                stereo.last_err = best[0]
                raw = best[1]
                det, det2 = best[2], best[3]
                src, t_stereo, last_pinch = 2, t, raw["pinch"].copy()
        # 카메라 두 대가 준비됐으면 한 대 계산으로 넘어가지 않고 기다린다 (11/18: 한 대 계산은 배율이 따로 틀려 값이 튐,
        # Point Policy 도 두 화면이 다 볼 때만 씀). 한 대만 있을 때만 그 한 대로
        if raw is None and measure and det and stereo is None:
            raw = measure(det["world"], det["img_px"])
            src = 1 if raw is not None else 0
        if measure and not det:
            measure.reset()
        meas = smooth(t, raw)
        if z_collect is not None and raw is not None:
            z_collect.append(float(raw["pinch"][2]))
            if len(z_collect) >= 15:
                z_sorted = sorted(z_collect)
                if z_sorted[-3] - z_sorted[2] > 0.015:                  # 재는 동안 손이 1.5 cm 넘게 오르내림: 다시
                    z_collect, msg, msg_t = [], "hand moved while measuring - hold still on the paper", t
                else:
                    hand_z0 = float(np.median(z_collect))
                    z_collect = None
                    try:
                        with open(a.calib, encoding="utf-8") as f_:
                            d_ = json.load(f_)
                    except (OSError, ValueError):
                        d_ = {}
                    d_["hand_touch_z"] = round(hand_z0, 4)
                    with open(a.calib, "w", encoding="utf-8") as f_:
                        json.dump(d_, f_, indent=1)
                    msg, msg_t = f"touch zero set: fingertips on the paper = {hand_z0 * 100:.1f} cm -> 0", t
                    print(f"[보정] 손끝이 종이에 닿은 집기점 높이 {hand_z0 * 100:.1f} cm 를 0 으로 (calib.json 에 저장)")
        if meas is not None and hand_z0:
            meas = dict(meas)
            meas["pinch"] = meas["pinch"] - np.array([0.0, 0.0, hand_z0])
        view = frame.copy()

        # 화면에 그릴 것 (원본 좌표에 먼저 그린 뒤 거울로 뒤집는다)
        if det:
            draw_hand(view, det["img_px"], tips=(4,) + GRIPS[grip_mode])
        if calib:
            grid = calib.project(np.array([[x, y, 0.0] for x in np.linspace(-A4_D / 2, A4_D / 2, 4)
                                           for y in np.linspace(-A4_W / 2, A4_W / 2, 5)]))
            for p in grid:
                cv2.circle(view, (int(p[0]), int(p[1])), 3, (255, 200, 0), -1)
            c = calib.project(CORNERS_TABLE).astype(int)
            cv2.polylines(view, [c], True, (255, 200, 0), 2)
            cv2.drawMarker(view, tuple(calib.project([[0, 0, 0]])[0].astype(int)), (0, 200, 255),
                           cv2.MARKER_CROSS, 24, 2)
        for i, p in enumerate(clicks):
            cv2.circle(view, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)
        if mode == "corners" and auto and finder.last is not None:          # 찾는 중인 종이 (빨강)
            cv2.polylines(view, [np.round(finder.last).astype(np.int32)], True, (0, 0, 255), 2)
        disp = cv2.resize(cv2.flip(view, 1), (int(W * scale_disp), DISP_H))
        if grab2 and frame2 is not None:                                   # 두 번째 카메라: 오른쪽 아래 작은 화면
            v2 = frame2.copy()
            if det2:
                draw_hand(v2, det2["img_px"], tips=(4,) + GRIPS[grip_mode])
            if calib2 is not None:
                cv2.polylines(v2, [calib2.project(CORNERS_TABLE).astype(np.int32)], True, (255, 200, 0), 4)
                cv2.drawMarker(v2, tuple(calib2.project([[0, 0, 0]])[0].astype(int)), (0, 200, 255),
                               cv2.MARKER_CROSS, 40, 4)
            elif finder2.last is not None:
                cv2.polylines(v2, [np.round(finder2.last).astype(np.int32)], True, (0, 0, 255), 4)
            iw = disp.shape[1] // 3
            ih = int(iw * H / W)
            small = cv2.resize(cv2.flip(v2, 1), (iw, ih))
            y0_, x0_ = DISP_H - ih - 6, disp.shape[1] - iw - 6
            disp[y0_:y0_ + ih, x0_:x0_ + iw] = small
            cv2.rectangle(disp, (x0_ - 1, y0_ - 1), (x0_ + iw, y0_ + ih), (255, 255, 255), 1)
            cv2.putText(disp, "cam2" + ("" if calib2 is not None else ": looking for the paper"), (x0_ + 6, y0_ + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        lines = []
        if mode == "corners" and auto and not clicks:
            found = finder.feed(frame)
            if found is not None:
                try:
                    calib = TableCalib(found, W, H, a.hfov)
                    measure, gate = HandMeasure(calib, 1.0, grip_mode, solver=cal_solver), None
                    mode, collecting, samples = "hand", False, []
                    msg, msg_t = "paper found automatically (blue dots should sit on the paper; M = click by hand)", t
                    print(f"[보정] 종이를 저절로 찾음: 초점 {calib.f:.0f}px, 모서리 어긋남 {calib.reproj_px:.1f}px, "
                          f"카메라 자리 {np.round(calib.cam_pos * 100).astype(int).tolist()} cm (종이 + 기준 앞·왼쪽·위)")
                except RuntimeError:
                    finder.reset()
        if mode == "corners":
            if len(clicks) < 4:
                lines = ["CALIBRATION 1/2: " + ("looking for the paper... keep hands and things off it"
                                                if auto and not clicks else "click the paper corners (seen from YOUR seat)"),
                         (f"or click the corners by hand: next {CORNER_NAMES[len(clicks)]}" if auto and not clicks
                          else f"next: {CORNER_NAMES[len(clicks)]}   (BACKSPACE: undo)")]
            else:
                try:
                    calib = TableCalib(clicks, W, H, a.hfov)
                    mode = "corners_check"
                except RuntimeError as e:
                    print(e)
                    clicks.clear()
        if mode == "corners_check":
            lines = ["Check: blue dots must sit on the paper.  ENTER = ok,  BACKSPACE = redo",
                     f"focal {calib.f:.0f}px ({calib.focal_source}), corner error {calib.reproj_px:.1f}px"]
        if mode == "hand":
            lines = [f"CALIBRATION 2/2: lay your {a.hand} hand FLAT on the paper,",
                     "palm center on the cross, fingers toward the laptop. Hold still" + (" (starts by itself)" if auto
                                                                                         else ", press S"),
                     "(keep the other hand out of view)   C: find paper again  M: click corners"]
            if auto and not collecting:
                if gate is None or gate.calib is not calib:
                    gate = FlatHandGate(calib, cal_solver)
                ok_flat, why = gate(det["world"], det["img_px"]) if det else (False, "show your hand")
                last_gate = why
                lines.append("auto: " + why)
                if ok_flat:
                    collecting, samples = True, []
                    gate.reset()
            if collecting:
                lines.append(f"measuring... {len(samples)}/20  (keep still)")
                if det is not None:
                    if samples and np.linalg.norm(np.asarray(det["img_px"])[list(PALM)].mean(axis=0)
                                                  - np.asarray(samples[0]["img_px"])[list(PALM)].mean(axis=0)) > 6.0:
                        samples = []                                  # 재는 동안 손이 움직이면 처음부터 (11/18)
                        msg, msg_t = "hand moved - measuring again", t
                    samples.append(det)
                if len(samples) >= 20:
                    collecting = False
                    # MediaPipe 손 3D 가 화면과 같은 손인지 거울상인지: 펴서 올려놓은 21점이 종이 높이에 더 잘 맞는 쪽 (다수결)
                    votes = []
                    for d in samples:
                        r0, r1 = HandMeasure(calib, 1.0, grip_mode, False, cal_solver)(d["world"], d["img_px"]), \
                            HandMeasure(calib, 1.0, grip_mode, True, cal_solver)(d["world"], d["img_px"])
                        if r0 and r1:
                            votes.append(flat_fit(calib, r1["lm_cam_raw"])[1] < flat_fit(calib, r0["lm_cam_raw"])[1])
                    mirror = bool(votes) and sum(votes) > 0.7 * len(votes)
                    # 이 손을 MediaPipe 가 뭐라고 부르는지 (두 손이 다 보일 때 이 표시로 고름)
                    labels = [d["label"] for d in samples]
                    label = max(set(labels), key=labels.count)
                    if labels.count(label) >= 0.8 * len(labels):
                        tracker.want_label = label
                    m1 = HandMeasure(calib, 1.0, grip_mode, mirror, cal_solver)
                    raws = [m1(d["world"], d["img_px"]) for d in samples]
                    raws = [r for r in raws if r]
                    s = scale_for_flat(calib, np.median([r["lm_cam_raw"] for r in raws], axis=0)) if raws else None
                    # 11/18: 종이에 편 손 21점의 진짜 자리 (화면 점을 어림 높이 FLAT_H 평면에 내림) → palm 방식과 기록 다시 풀기에 씀
                    flat21 = np.median([backproject_height(calib, d["img_px"], FLAT_H) for d in samples], axis=0)
                    pw, pl = palm_size(flat21)
                    if s is None or not np.all(np.isfinite(flat21)):
                        msg, msg_t = "hand calibration failed - try again (S)", t
                    else:
                        measure = HandMeasure(calib, 1.0 if a.solver == "palm" else s, grip_mode, mirror, a.solver, a.pen,
                                              flat=flat21)
                        rs = [r for r in (measure(d["world"], d["img_px"]) for d in samples) if r]
                        palm = np.median([r["palm"] for r in rs], axis=0)       # 손바닥 가운데가 + 에서 얼마나
                        with open(a.calib, "w", encoding="utf-8") as f:
                            json.dump({"table": calib.to_dict(), "scale": s, "pose": "flat hand", "mirror": mirror,
                                       "mirror_votes": [int(sum(votes)), len(votes)],
                                       "palm_off_m": palm.tolist(), "hand": a.hand, "label": tracker.want_label,
                                       "label_votes": [labels.count(label), len(labels), label],
                                       "solver": a.solver, "flat_hand_m": np.round(flat21, 5).tolist(),
                                       "palm_w_l_m": [round(pw, 4), round(pl, 4)],
                                       "table2": calib2.to_dict() if calib2 is not None else None,
                                       "cam2": a.cam2, "paper_source": saved.get("paper_source"),
                                       "hand_touch_z": round(hand_z0, 4) if hand_z0 else None}, f, indent=1)
                        msg = (f"saved. scale {s:.3f}, palm {pw*100:.1f} x {pl*100:.1f} cm, off the cross x {palm[0]*100:+.1f}"
                               f" y {palm[1]*100:+.1f} cm (near 0 is good), 3D {'MIRRORED' if mirror else 'ok'}"
                               f" {len(votes) - sum(votes)}/{len(votes)}")
                        print(f"[보정] 방식 {a.solver}, 배율 {s:.3f}, 손바닥 폭 {pw*100:.1f} cm·길이 {pl*100:.1f} cm (어른 보통 7~9, 9~11),"
                              f" 손 3D {'거울상으로 판단해 뒤집어 씀' if mirror else '화면과 같은 손'}"
                              f" (거울상 표 {sum(votes)}/{len(votes)}), MediaPipe 표시 '{label}' {labels.count(label)}/{len(labels)}")
                        msg_t = t
                        smooth.reset()
                        mode = "run"
        if mode == "run" and recheck_until is not None:
            # 켠 뒤 몇 초: 저장된 종이 자리가 지금도 맞는지 (따라가기를 켜면 그만 봄)
            found = None if follow else finder.feed(frame)
            if follow or t > recheck_until:
                recheck_until = None
                if not follow:
                    print("[보정] 종이를 못 찾아 저장된 보정을 그대로 씀 (종이 위에 물건이 있거나 가려졌을 수 있음)")
            elif found is not None:
                recheck_until = None
                moved = float(np.max(np.linalg.norm(found - calib.img_pts, axis=1)))
                if moved > 8.0:
                    calib = TableCalib(found, W, H, a.hfov)
                    measure, gate = HandMeasure(calib, 1.0, grip_mode, solver=cal_solver), None
                    mode, collecting, samples = "hand", False, []
                    msg, msg_t = f"camera or paper moved ({moved:.0f} px): paper re-found, now lay your hand flat", t
                    print(f"[보정] 노트북이나 종이가 움직였음 (모서리 {moved:.0f} px): 종이를 다시 잡았고 손 보정을 다시 합니다")
                else:
                    msg, msg_t = f"paper check ok ({moved:.1f} px)", t
                    print(f"[보정] 종이 자리 그대로 (모서리 {moved:.1f} px 차)")
        if mode == "run":
            if not link:
                q_cmd, width = mapper.step(meas if follow else None, dt, follow)
            if follow:
                if smooth.lost_since is not None and t - smooth.lost_since > 1.0:
                    follow = False
                    arm.hold()
                    msg, msg_t = "hand lost > 1 s: follow stopped (SPACE to resume)", t
                elif not link:
                    arm.send(q_cmd, width)
            if meas:
                p = meas["pinch"]
                lines = [f"hand x {p[0]*100:+5.1f}  y {p[1]*100:+5.1f}  z {(p[2])*100:4.1f} cm   "
                         f"grip {meas['gap']*1000:3.0f} mm   turn {math.degrees(math.remainder(meas['yaw'] + math.pi / 2, math.pi)):+4.0f} deg"]
            else:
                lines = ["hand not found"]
            lines.append(f"{fps:4.1f} fps   SPACE follow  R rec  B start pose  TAB hand: {GRIP_NAME[grip_mode]}")
            if grab2:
                lines.append("3D: " + ("2 cameras (triangulated, err %.1f px)" % stereo.last_err if src == 2 else
                                       "camera 1 only" if src == 1 else
                                       "waiting for camera 2 paper" if calib2 is None else
                                       "hand not seen by both cameras"
                                       + (" (err %.0f px)" % stereo.last_err if stereo and stereo.last_err else "")))
            lines.append(("Z touch zero: measuring... keep fingertips on the paper" if z_collect is not None else
                          f"Z touch zero: {hand_z0 * 100:.1f} cm" if hand_z0 else "Z: fingertips on paper -> touch zero"))
            lines.append(f"G grip:{'on' if mapper.follow_grip else 'off'}  Y yaw:{'on' if mapper.follow_yaw else 'off'}"
                         f"  [ ] squeeze {mapper.squeeze*1000:.0f}mm")
        if mode == "hand" and GUIDE is not None and GUIDE.shape[0] + 16 <= disp.shape[0] \
                and GUIDE.shape[1] + 16 <= disp.shape[1]:          # 시작 자세 그림을 왼쪽 아래에 (화면이 작으면 생략)
            gh, gw = GUIDE.shape[:2]
            disp[DISP_H - gh - 8:DISP_H - 8, 8:8 + gw] = GUIDE
        if msg and t - msg_t < 6:
            lines.append(msg)
        for i, s_ in enumerate(lines):
            cv2.putText(disp, s_, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(disp, s_, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if link:
            # 연구실 PC 로 손 측정 보내기 (보정 중에는 손 없음·따라가기 꺼짐으로 보냄)
            run = mode == "run"
            hand_out = ([round(float(v), 5) for v in meas["pinch"]] + [round(meas["yaw"], 4),
                                                                       round(meas["gap"], 4)]) if (run and meas) else None
            # 11/17: 기록 중이면 손 21점 원본(화면 px, MediaPipe 3D m)과 거르기 전 측정도 보낸다 → 연구실에서 다른 방식으로 다시 풀어 볼 수 있게
            raw_out = {}
            if run and rec_req and det is not None:
                raw_out = {"l2": [round(float(v), 1) for v in np.ravel(det["img_px"])],
                           "l3": [round(float(v), 4) for v in np.ravel(det["world"])], "lb": det["label"]}
                if raw:
                    raw_out["hr"] = [round(float(v), 5) for v in raw["pinch"]] + [round(raw["yaw"], 4), round(raw["gap"], 4)]
                if det2 is not None:                               # 11/18 두 번째 카메라 화면 21점, 삼각측량 21점 (책상 좌표)
                    raw_out["l2b"] = [round(float(v), 1) for v in np.ravel(det2["img_px"])]
                if raw is not None and src == 2:
                    raw_out["t3"] = [round(float(v), 4) for v in np.ravel(raw["tab21"])]
            if run and link.seq % 30 == 0:                        # 1초에 한 번 노트북 보정 값
                raw_out["cal"] = {**calib.to_dict(), "scale": measure.scale, "mirror": measure.mirror,
                                  "label": tracker.want_label, "solver": a.solver, "pen": a.pen, "grip": grip_mode,
                                  "flat": (None if measure.flat is None else np.round(measure.flat, 4).ravel().tolist()),
                                  "table2": calib2.to_dict() if calib2 is not None else None}
            link.send(hand=hand_out, follow=int(run and follow), rec=int(run and rec_req),
                      grip=int(mapper.follow_grip), yawf=int(mapper.follow_yaw), sq=round(mapper.squeeze, 4),
                      ygm=mapper.yaw_gap_min, ypref=round(mapper.yaw_pref, 4), disc=disc_n, home=home_n, bye=0,
                      gm=int(grip_mode == "pinch"), src=src, **raw_out)
            st = link.poll()
            if st and st.get("msg_n") != seen_msg_n:
                if seen_msg_n is not None and st.get("msg"):
                    msg, msg_t = "robot PC: " + st["msg"], t
                seen_msg_n = st.get("msg_n")
            # 연구실 PC 가 따라가기·기록을 껐는데 여기는 켜져 있으면 (통신 끊김·멈춤) 여기도 맞춰 끈다
            if link.connected and st:
                for k, want in (("follow", follow), ("rec", rec_req)):
                    if bool(st.get(k)) != want:
                        mismatch[k] = mismatch[k] or t
                        if t - mismatch[k] > (0.7 if k == "follow" else 1.5):
                            if k == "follow":
                                follow = False
                            else:
                                rec_req = False
                            mismatch[k] = None
                    else:
                        mismatch[k] = None
            if st:
                q_now = np.array(st["q"], float)
                w_now = float(st["w"]) if st.get("w") is not None else float("nan")
                tgt, fol, x0, y0 = np.array(st["tgt"], float), bool(st["follow"]), st["x0"], st["y0"]
            else:
                q_now, w_now, tgt, fol, x0, y0 = HOME_Q, float("nan"), None, False, a.x0, a.y0
            if link.connected:
                extra_txt = [f"robot PC {link.addr[0]}", f"ping {link.rtt*1000:.0f} ms" if link.rtt else "",
                             ("!REAL ARM" if st.get("real") else "preview arm") + (" PAUSED" if st.get("paused") else "")]
                if st.get("ver") and st["ver"] != my_ver:               # 연구실 PC 와 코드가 다름 (11/17)
                    extra_txt.append("!code differs from robot PC")
                    if not ver_warned:
                        print(f"[버전] 노트북 코드 {my_ver}, 연구실 PC 코드 {st['ver']} — 다릅니다. "
                              "노트북은 다시 켜면 GitHub 에서 새 코드를 받고, 연구실 PC 는 연구실에서 맞춰야 합니다")
                        ver_warned = True
            else:
                extra_txt = [f"!no answer from", f"!{link.addr[0]}:{link.addr[1]}", "!start robot_server.py"]
            panel = robot_panel(kin, q_now, tgt, x0, y0, w_now, fol, rec_on(), extra_txt)
        else:
            q_now = arm.joints()
            w_now = arm.gripper()
            panel = robot_panel(kin, q_now, mapper.cur, a.x0, a.y0, w_now if np.isfinite(w_now) else mapper.width,
                                follow, rec.on)
        board = np.hstack([disp, panel])
        cv2.imshow(win, board)
        if a.status and t - status_t > 0.25:
            status_t = t
            st_ = {"t": round(t, 2), "mode": mode, "fps": round(fps, 1), "follow": bool(follow), "src": src,
                   "err_px": (round(stereo.last_err, 2) if stereo is not None and stereo.last_err is not None else None),
                   "det1": det is not None, "det2": det2 is not None, "gate": last_gate if mode == "hand" else "",
                   "collecting": (len(samples) if collecting else None),
                   "pinch_cm": (np.round(meas["pinch"] * 100, 1).tolist() if meas else None),
                   "gap_mm": (round(meas["gap"] * 1000) if meas else None),
                   "raw_cm": (np.round(raw["pinch"] * 100, 1).tolist() if raw else None),
                   "msg": msg if t - msg_t < 6 else "", "rec": bool(rec_on())}
            if raw is not None and src == 2:
                st_["palm_cm"] = np.round(raw["palm"] * 100, 1).tolist()
                st_["tab21_z_cm"] = np.round(raw["tab21"][:, 2] * 100, 1).tolist()
            with open(os.path.join(a.status, "status.tmp"), "w") as f_:
                json.dump(st_, f_)
            os.replace(os.path.join(a.status, "status.tmp"), os.path.join(a.status, "status.json"))
        if a.status and det is not None and det2 is not None:
            # 두 화면 손 21점 짝 (보정 다듬기용, 11/18): 초점·렌즈 왜곡을 두 화면이 맞도록 고치는 데 씀
            with open(os.path.join(a.status, "pairs.jsonl"), "a") as f_:
                f_.write(json.dumps({"t": round(t, 3), "t2": round(float(t2_), 3) if grab2 and t2_ else None,
                                     "l1": np.round(det["img_px"], 1).ravel().tolist(),
                                     "l2": np.round(det2["img_px"], 1).ravel().tolist()}) + "\n")
        if a.status and t - snap_t > 1.0:
            snap_t = t
            cv2.imwrite(os.path.join(a.status, "view.tmp.jpg"), cv2.resize(board, (board.shape[1] // 2, board.shape[0] // 2)))
            os.replace(os.path.join(a.status, "view.tmp.jpg"), os.path.join(a.status, "view.jpg"))

        if rec and rec.on:
            imgs = {k: c.latest() for k, c in extra.items()}
            rec.add(t=t, q=q_now, grip=w_now, q_cmd=mapper.q_cmd.copy(), grip_cmd=mapper.width,
                    hand=(meas["pinch"] if meas else np.full(3, np.nan)),
                    hand_yaw=(meas["yaw"] if meas else np.nan),
                    hand_gap=(meas["gap"] if meas else np.nan), target=mapper.cur.copy(), tracked=meas is not None,
                    follow=follow, human_jpg=jpeg(frame, 480),
                    lm2d=(det["img_px"] if det else np.full((21, 2), np.nan)),
                    lm3d=(det["world"] if det else np.full((21, 3), np.nan)),
                    hand_raw=(np.r_[raw["pinch"], raw["yaw"], raw["gap"]] if raw else np.full(5, np.nan)),
                    grip_mode=int(grip_mode == "pinch"),
                    **{f"{k}_jpg": jpeg(v, 640) for k, v in imgs.items()})

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("b"):                                    # 11/20 처음 자세로 (save_home.py 로 저장한 자세)
            if link:
                if follow:
                    follow = False
                home_n += 1
                msg, msg_t = "robot PC: going to start pose (SPACE stops)", t
            else:
                msg, msg_t = "B (start pose) works with robot_server.py: run teleop with --send", t
        if mode == "corners" and key in BACKSPACE and clicks:
            clicks.pop()
        elif mode == "corners_check":
            if key in (13, 10):
                mode, collecting, samples = "hand", False, []
                measure = HandMeasure(calib, 1.0, grip_mode, solver=cal_solver)
            elif key in BACKSPACE:
                clicks.clear()
                calib, mode = None, "corners"
        elif mode == "hand" and key == ord("s") and not collecting:
            collecting, samples = True, []
        elif mode == "hand" and key in (ord("c"), ord("m")):
            auto = key == ord("c") and not a.manual_calib       # C: 종이 다시 찾기, M: 모서리를 손으로 누르기
            clicks.clear()
            finder.reset()
            if key == ord("c") and grab2:
                calib2 = None
                finder2.reset()
            calib, measure, mode, collecting = None, None, "corners", False
        elif mode == "run":
            if key == ord(" "):
                follow = not follow
                if follow:
                    if not link:
                        mapper.sync_to(arm.joints())
                    smooth.reset()
                else:
                    arm.hold()
            elif key == ord("r"):
                if link:
                    rec_req = not rec_req                    # 연구실 PC 가 기록 시작/저장
                elif rec.on:
                    r = rec.stop()
                    msg, msg_t = (f"saved {os.path.basename(r[0])} ({r[1]} frames)" if r else "nothing recorded"), t
                else:
                    rec.start()
            elif key == ord("x"):
                if link:
                    disc_n += 1
                else:
                    p = rec.discard_last()
                    msg, msg_t = (f"deleted {os.path.basename(p)}" if p else "no episode to delete"), t
            elif key in GRIP_KEYS:                       # 손 모양 바꾸기: 쥐는 점이 몇 cm 옮겨 가므로 따라가기는 멈춘다
                squeeze_by[grip_mode] = mapper.squeeze
                grip_mode = "pinch" if grip_mode == "whole" else "whole"
                measure.tips = list(GRIPS[grip_mode])
                if stereo is not None:
                    stereo.tips = list(GRIPS[grip_mode])
                mapper.squeeze, mapper.yaw_gap_min = squeeze_by[grip_mode], GRIP_YGM[grip_mode]
                smooth.reset()
                paused_now = follow
                if follow:
                    follow = False
                    arm.hold()
                msg = f"hand: {GRIP_NAME[grip_mode]}" + ("  (follow paused - SPACE to resume)" if paused_now else "")
                msg_t = t
                print(f"[손 모양] {'엄지·검지' if grip_mode == 'pinch' else '손 전체'}" + (" — 따라가기 멈춤, SPACE 로 다시" if paused_now else ""))
            elif key == ord("g"):
                mapper.follow_grip = not mapper.follow_grip
            elif key == ord("y"):
                mapper.follow_yaw = not mapper.follow_yaw
            elif key == ord("["):
                mapper.squeeze = max(0.0, mapper.squeeze - 0.005)
            elif key == ord("]"):
                mapper.squeeze = min(0.05, mapper.squeeze + 0.005)
            elif key in (ord("c"), ord("m")):
                follow = False
                arm.hold()
                auto = key == ord("c") and not a.manual_calib
                clicks.clear()
                finder.reset()
                recheck_until = None
                if key == ord("c") and grab2:
                    calib2 = None
                    finder2.reset()
                calib, measure, mode = None, None, "corners"
            elif key == ord("s"):
                follow = False
                arm.hold()
                mode, collecting, samples = "hand", False, []
                measure = HandMeasure(calib, 1.0, grip_mode, solver=cal_solver)
            elif key == ord("z"):                              # 손 닿는 높이 재기 (따라가기는 멈춤: 높이 기준이 바뀌므로)
                if follow:
                    follow = False
                    arm.hold()
                z_collect = []
                msg, msg_t = "touch zero: put thumb and fingertips on the paper and hold still", t
            elif key == ord("p"):
                msg, msg_t = check_point(meas, measure.scale, a.checks), t

    if link:
        for _ in range(5):                                   # 연구실 PC 에 "끝" 을 알림 (팔 멈춤·기록 저장)
            link.send(hand=None, follow=0, rec=0, grip=int(mapper.follow_grip), yawf=int(mapper.follow_yaw),
                      sq=round(mapper.squeeze, 4), disc=disc_n, home=home_n, bye=1)
            time.sleep(0.02)
        link.close()
    if rec and rec.on:
        rec.stop()
    arm.hold()
    arm.close()
    tracker.close()
    cap.release()
    if grab2:
        tracker2.close()
        grab2.close()
    for c in extra.values():
        c.close()
    cv2.destroyAllWindows()
    if rec:
        print(f"[끝] 기록 {len(rec.saved)}개" + (f" → {rec.dir}" if rec.saved else ""))
    else:
        print("[끝] 기록은 연구실 PC 의 data/ 폴더에 있습니다")


if __name__ == "__main__":
    from autoupdate import update_and_restart
    update_and_restart()                                    # 켤 때 GitHub 에서 새 코드를 받아 오고, 받았으면 다시 켬 (11/17)
    main()
