"""연구실 PC 쪽: 노트북이 보낸 손 위치로 PiPER 를 움직이고 기록한다 (2026-11-15).

노트북은 카메라로 손만 재서 숫자(집기점 위치·손 돌림·손가락 사이)를 보낸다 (teleop.py --send).
이 프로그램은 그 숫자를 mapping.py 로 집게 끝 목표로 바꾸고, 역기구학으로 관절을 풀어 팔에 보낸다.
mediapipe·mujoco 가 필요 없다 (numpy, opencv + 실물이면 agx-bus 나 piper_sdk). 찬혁님 piper-teleop .venv 로 돈다.

실행
  python robot_server.py                          # 팔 없이 가상 팔로 연결만 시험
  python robot_server.py --robot agxbus --speed 20  # 실물 PiPER

안전
  - 실물이면 시작할 때 터미널에서 Enter 를 받는다.
  - 노트북에서 따라가기를 새로 켜야 움직인다 (이 프로그램을 켜기 전에 켜져 있던 것은 무시, 노트북 쪽이 저절로 꺼짐).
  - 노트북 패킷이 0.5초 넘게 안 오면 그 자리에 멈추고 따라가기를 끈다. 다시 움직이려면 노트북에서 SPACE.
  - 집게 끝 속도·작업 상자·관절 속도 제한은 노트북 혼자 쓸 때와 같다 (mapping.py).
  - 창이 있으면 이 PC 에서도 SPACE 로 멈춤/풀기, Q 로 끝.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

from home import HOME_CLEAR, HOME_FILE, HomeMove, load_home, path_ok, safe_target
from mapping import Mapper
from net import PORT, Server, code_version, my_ips
from panel import robot_panel
from piper_kin import TABLE_Z, PiperKin
from recorder import CamGrabber, FakeCam, Recorder, jpeg
from robots import GRIP_MAX, make_arm
from stream import Follower, LogArm

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    ap = argparse.ArgumentParser(description="노트북 손 측정 → PiPER")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--robot", choices=["preview", "agxbus", "sdk"], default="preview")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--speed", type=int, default=30, help="실물 팔 속도 %%")
    # 11/20 사용자 "손 + 가운데, 집게도 + 가운데 뒀는데 SPACE 누르면 앞으로 가": 처음 자세 집게 끝은 밑동 앞 14.8 cm 인데
    # 종이 + 를 앞 30 cm 로 잡아 15 cm 앞으로 갔다. 이제 처음 자세(home_pose.json)가 있으면 + = 처음 자세 집게 끝 (앞·옆·높이).
    # --x0/--y0 를 주면 예전처럼 그 자리 (높이는 책상 + 1.2 cm)
    ap.add_argument("--x0", type=float, default=None, help="로봇 종이 가운데: 밑동에서 앞으로 (m). 안 주면 처음 자세 집게 끝 (없으면 0.30)")
    ap.add_argument("--y0", type=float, default=None, help="로봇 종이 가운데: 밑동에서 왼쪽으로 (m). 안 주면 처음 자세 집게 끝 (없으면 0)")
    ap.add_argument("--gain", type=float, default=1.0, help="사람 1 cm → 로봇 몇 cm")
    # 11/18: 사람 손 속도로 따라가게 기본값을 올림 (11/16 실물 첫 시험은 6 cm/s·관절 1 rad/s 라 손보다 한참 느렸다).
    # 노트북은 초당 18~30번 보내므로, 100 Hz 로 돌며 이 속도 제한 안에서 다음 값까지 이어 준다 (뚝뚝 끊기지 않게)
    ap.add_argument("--max-speed", type=float, default=1.0, help="집게 끝 최고 속도 (m/s). 사람 손 빠르기 ~1")
    ap.add_argument("--joint-rate", type=float, default=3.0, help="관절 최고 속도 (rad/s). 11/17 까지 1.0")
    ap.add_argument("--rot-rate", type=float, default=180.0, help="집게가 도는 최고 속도 (도/s). 11/17 까지 60")
    # 11/19 사용자 "종이 밖에서도 움직일 수 있게": 작업 상자를 넓힘 (11/18 까지 앞 14~50 cm, 좌우 ±25 cm, 높이 25 cm).
    # PiPER 는 책상 높이에서 밑동 6~58 cm 까지 닿고, 20 cm 넘게 들면 집게가 비스듬해진다 (역기구학 계산 11/19)
    ap.add_argument("--box-x", type=float, nargs=2, default=[0.12, 0.50], metavar=("앞최소", "앞최대"),
                    help="집게 끝이 갈 수 있는 앞뒤 범위 (밑동에서 m)")
    ap.add_argument("--box-y", type=float, default=0.40, help="좌우 ±이만큼 (m)")
    ap.add_argument("--max-z", type=float, default=0.30, help="책상 위 이 높이(m)까지")
    ap.add_argument("--lift", type=float, default=0.0,
                    help="집게를 사람 손보다 이만큼 높이 따라가게 (m). 실물 첫 시험은 0.10 (높이 바닥 --min-z 대신)")
    ap.add_argument("--min-z", type=float, default=0.006,
                    help="집게 끝을 책상 위 이 높이(m) 아래로 내리지 않음. 실물 첫 시험은 0.10 처럼 높게 (11/16)")
    ap.add_argument("--no-interp", action="store_true",
                    help="11/20 패킷 사이 잇기를 끔 (새 손 값이 올 때마다 최고 속도로 갔다 서던 11/19 방식)")
    ap.add_argument("--home-grip", choices=["close", "keep"], default="close",
                    help="B 로 처음 자세에 갈 때 그리퍼: close 닫음 (11/23 기본, 사용자 요청) / keep 그대로 (11/20)")
    ap.add_argument("--home-clear", type=float, default=HOME_CLEAR,
                    help="11/23 B 로 갈 자세의 집게 끝을 가상 팔 모델로 책상 위 이만큼(m) 아래로는 안 둠 (저장한 자세가 더 낮으면 같은 방향으로 들어 올림)")
    ap.add_argument("--start", choices=["keep", "paper"], default="keep",
                    help="11/23 SPACE 를 누를 때: keep = 팔은 그 자세 그대로, 그때 손 자리를 지금 집게 끝으로 삼고 손이 움직인 만큼 따라감 (기본, 사용자 요청) "
                         "/ paper = 손이 종이 + 에서 떨어진 만큼 집게도 처음 자세(종이 +)에서 떨어진 자리로 감 (11/22 까지)")
    ap.add_argument("--down-only", action="store_true",
                    help="11/23 손바닥 방향 따라가기를 끔: 집게는 늘 아래를 보고 돌림만 따라감 (11/22 까지와 같음)")
    ap.add_argument("--home-rate", type=float, default=0.6,
                    help="처음 자세로 돌아갈 때 가장 많이 도는 관절의 최고 속도 (rad/s). 11/20")
    ap.add_argument("--hz", type=float, default=100.0, help="팔에 명령 보내고 기록하는 박자 (11/17 까지 30)")
    ap.add_argument("--timeout", type=float, default=0.5, help="노트북 패킷이 이만큼(초) 안 오면 멈춤")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--wrist-cam", type=int, default=None, help="기록할 손목 카메라 번호 (선택)")
    ap.add_argument("--front-cam", "--top-cam", dest="front_cam", type=int, default=None,
                    help="기록할 앞/위 카메라 번호 (선택). 11/26 tinybc 형식에서는 'top' 카메라")
    ap.add_argument("--fake-cams", action="store_true", help="11/26 시험용: 손목·위 카메라 대신 가짜 그림 (실물 카메라 없이 기록 길 시험)")
    ap.add_argument("--headless", action="store_true", help="창 없이 (끝낼 때 Ctrl+C)")
    # 11/26 석사형 메모: 팔에 보내는 일만 하는 follower (줄에서 꺼내 200 Hz), 3차 중간 점, MIT 모드 (kp 5, kd 0.8).
    # 모두 켜는 옵션이고, 안 주면 11/23 까지와 똑같다. 실물 MIT 는 아직 해 보지 않았다
    ap.add_argument("--follower-hz", type=float, default=0.0,
                    help="11/26 팔에 보내는 follower 박자 (예 200). 0 이면 끔 (예전처럼 --hz 박자에 바로 보냄). --mode mit 면 안 줘도 200")
    ap.add_argument("--mode", choices=["position", "mit"], default="position",
                    help="11/26 position = 관절 각도만 보냄 (팔 안 제어기가 길을 정함, 지금까지) / mit = 각도·속도·kp·kd 를 보내 모터가 바로 당김")
    ap.add_argument("--kp", type=float, default=5.0, help="11/26 MIT 모드 당기는 세기 (석사형 메모 5)")
    ap.add_argument("--kd", type=float, default=0.8, help="11/26 MIT 모드 흔들림 잡는 세기 (석사형 메모 0.8)")
    ap.add_argument("--profile", choices=["linear", "cubic"], default="linear",
                    help="11/26 손 값 사이 중간 점: linear 곧게 같은 빠르기 (11/20) / cubic s = 3t² − 2t³ (석사형 메모, 처음·끝 속도 0)")
    ap.add_argument("--cmd-log", default=None, help="11/26 팔에 보낸 명령을 이 npz 로 남김 (시험·분석용)")
    ap.add_argument("--run-seconds", type=float, default=None, help=argparse.SUPPRESS)   # 시험용
    return ap.parse_args()


VER = None


def main():
    global VER
    a = parse()
    VER = code_version()
    kin = PiperKin()
    arm = make_arm(a.robot, a.can, a.speed)
    if a.mode == "mit":
        if a.robot == "sdk":
            raise SystemExit("--mode mit 은 --robot agxbus 나 preview 만 (piper_sdk 쪽은 아직 안 만듦)")
        if a.follower_hz <= 0:
            a.follower_hz = 200.0                       # MIT 는 팔 안에서 길을 안 짜므로 촘촘히 보내야 함
    if arm.is_real:
        print("\n[안전] 실물 팔입니다. 팔 둘레를 비우고, 전원 스위치 옆에 사람이 있어야 합니다.")
        print("       노트북에서 SPACE 를 눌러야 따라가고, 통신이 %.1f초 끊기면 멈춥니다. 속도 %d%%" % (a.timeout, a.speed))
        print("       집게 끝 최고 속도 %.0f cm/s, 관절 %.1f rad/s, 책상 위 %.0f cm 아래로는 내려가지 않음, 손보다 %.0f cm 위에서 따라감"
              % (a.max_speed * 100, a.joint_rate, a.min_z * 100, a.lift * 100))
        print("       움직이는 범위: 밑동에서 앞 %.0f~%.0f cm, 좌우 ±%.0f cm, 책상 위 %.0f cm 까지 (팔 둘레를 이만큼 비울 것)"
              % (a.box_x[0] * 100, a.box_x[1] * 100, a.box_y * 100, a.max_z * 100))
        if a.follower_hz > 0:
            print("       follower %.0f Hz 로 보냄 (%s 모드%s), 중간 점 %s"
                  % (a.follower_hz, a.mode, ", kp %.1f kd %.2f" % (a.kp, a.kd) if a.mode == "mit" else "", a.profile))
        if a.mode == "mit":
            print("       [MIT] 팔 안의 속도 제한이 없습니다. 관절 속도 %.1f rad/s 제한은 이 PC 에서만 겁니다." % a.joint_rate)
            print("       [MIT] 실물에서 처음이면 --speed 와 상관없이 세게 움직일 수 있으니, 손을 비상정지 옆에 두세요.")
        input("       준비됐으면 Enter (취소 Ctrl+C) ")
    if a.follower_hz > 0:                               # 11/26: Enter 뒤에 켬 (MIT 는 켜자마자 지금 자세를 붙잡기 시작)
        arm = Follower(arm, hz=a.follower_hz, mode=a.mode, kp=a.kp, kd=a.kd, push_hz=a.hz,
                       max_rate=max(a.joint_rate, a.home_rate), lo=kin.lo, hi=kin.hi, log=bool(a.cmd_log))
        print("[follower] %.0f Hz, %s 모드%s, 중간 점 %s" % (a.follower_hz, a.mode,
              ", kp %.1f kd %.2f" % (a.kp, a.kd) if a.mode == "mit" else "", a.profile))
    elif a.cmd_log:
        arm = LogArm(arm)
    home_q = load_home()                                # 11/20: save_home.py 로 저장한 처음 자세 (B 로 돌아감)
    home_lift = 0.0
    if home_q is not None:
        home_q = np.clip(home_q, kin.lo, kin.hi)
        # 11/23 오후 사용자 "B 를 누르면 로봇팔이 책상에 박고 힘이 빠지네": 저장한 자세가 모델로 책상 −0.8 cm 였다
        home_q, home_lift = safe_target(kin, home_q, a.home_clear)
    origin_home = home_q is not None and a.x0 is None and a.y0 is None
    if origin_home:                                     # 11/20: 종이 + = 처음 자세 집게 끝
        hp, _ = kin.fk(home_q)
        a.x0, a.y0 = float(hp[0]), float(hp[1])
    a.x0 = 0.30 if a.x0 is None else a.x0
    a.y0 = 0.0 if a.y0 is None else a.y0
    mapper = Mapper(kin, a.x0, a.y0, a.gain, max_speed=a.max_speed, min_z=a.min_z, max_joint_rate=a.joint_rate,
                    max_yaw_rate=math.radians(a.rot_rate), max_z=a.max_z, box_x=tuple(a.box_x), box_y=a.box_y)
    mapper.follow_tilt = not a.down_only                # 11/23 손바닥 방향 따라 집게를 옆·위로
    if origin_home:                                     # 손을 종이에 대면 (높이 PINCH_H0) 처음 자세 높이
        mapper.robot_start[2] = max(float(hp[2]), TABLE_Z + a.min_z)
    mapper.robot_start[2] += a.lift                     # 11/18: 손 높이 + lift 로 따라감
    mapper.hold_start = a.start == "keep"               # 11/23 사용자 "SPACE 누르면 집게가 뜨는데? 자세가 그대로여야지"
    if mapper.hold_start:
        print("[시작] SPACE 를 누르면 팔은 그 자세 그대로이고, 그때 손 자리가 지금 집게 끝이 됨. 그 뒤로 손이 움직인 만큼 따라감"
              " (그리퍼는 손가락을 1 cm 넘게 움직이면 따라감. 종이 + 기준은 --start paper)")
    else:
        print("[종이 +] 손을 + 에 대면 집게 끝: 밑동에서 앞 %.1f cm, 왼쪽 %.1f cm, 책상 위 %.1f cm%s"
              % (mapper.robot_start[0] * 100, mapper.robot_start[1] * 100, (mapper.robot_start[2] - TABLE_Z) * 100,
                 " (처음 자세 집게 끝)" if origin_home else ""))
    if mapper.robot_start[0] - a.box_x[0] < 0.05:
        print("        밑동 쪽으로는 %.1f cm 만 갈 수 있음 (작업 상자 앞최소 %.0f cm)"
              % ((mapper.robot_start[0] - a.box_x[0]) * 100, a.box_x[0] * 100))
    mapper.interp = not a.no_interp                     # 11/20: 손 값 사이를 곧게 이어 끊어 가는 지터를 없앰
    mapper.profile = a.profile                          # 11/26 석사형 메모: cubic 이면 s = 3t² − 2t³
    ik_iters = 10 if a.hz >= 60 else 40                 # 100 Hz 면 한 박자 10번이면 됨 (40번은 7~12 ms 라 박자를 못 지킴)
    mapper.yaw0 = 0.0                                   # 노트북이 집게 방향(yaw)을 그대로 보냄
    mapper.sync_to(arm.joints())
    if home_lift > 0:
        print("[처음 자세] 저장한 자세는 집게 끝이 책상에 너무 가까워서 (모델로 %.1f cm) 같은 방향으로 %.1f cm 들어 올린 자세로 감 (--home-clear)"
              % ((a.home_clear - home_lift) * 100, home_lift * 100))
    if home_q is not None:
        print("[처음 자세] %s 의 관절(도) %s — 노트북 B 로 돌아감" % (os.path.basename(HOME_FILE),
                                                     ", ".join(f"{math.degrees(v):.1f}" for v in home_q)))
    else:
        print("[처음 자세] 저장된 것 없음 — 팔을 손으로 옮겨 두고 python save_home.py")
    srv = Server(a.port)
    cams = {k: CamGrabber(i) for k, i in (("wrist", a.wrist_cam), ("front", a.front_cam)) if i is not None}
    if a.fake_cams:                                     # 11/26 시험용
        cams = {"wrist": FakeCam("wrist"), "front": FakeCam("top")}
    cam_cache = {}                                      # 11/26 카메라 장 번호가 같으면 JPEG 를 다시 만들지 않고 같은 bytes 를 씀 (메모리·파일)
    q_prev_rec, t_prev_rec = None, None                 # 11/26 관절 속도를 못 읽는 팔은 차분으로
    for k, c in cams.items():
        if not c.ok:
            print(f"[카메라] {k} 카메라를 못 엶 — 사진 없이 기록")
    rec = Recorder(a.out, {"args": vars(a), "robot_start": mapper.robot_start.tolist(), "source": "laptop --send",
                           "note": "사람 책상 좌표: 원점 A4 가운데, x 앞(노트북 쪽), y 사람 왼쪽, z 위. 로봇 좌표: 밑동 (0,0,0.90)"})
    ips = ", ".join(my_ips()) or "?"
    print(f"[대기] 노트북에서:  python teleop.py --send <이 PC 주소>:{a.port}    이 PC 주소: {ips}")
    win = None if a.headless else "robot_server"
    if win:
        try:
            cv2.namedWindow(win)
        except cv2.error:                                # 찬혁님 .venv 의 opencv 는 창을 못 띄우는 판 (11/16 실물 첫 시험에서 걸림)
            print("[창] 이 opencv 는 창을 못 띄움 — 창 없이 돕니다. 멈춤은 노트북 SPACE, 끝은 여기서 Ctrl+C")
            win = None

    follow, paused = False, False
    was_blocked = False
    homing, home_prev = None, None     # 11/20 처음 자세로 가는 중 (HomeMove), 노트북 B 누른 횟수
    prev_req = None                    # 노트북이 바로 전에 보낸 follow (None = 아직 못 받음 → 껐다 켜야 움직임)
    rec_req_prev, disc_prev = False, None
    stop_n, msg, msg_n = 0, "", 0
    hand, last = None, None
    period = 1.0 / a.hz
    t_start = t_prev = time.monotonic()
    seen_peer = None

    def say(text):
        nonlocal msg, msg_n
        msg, msg_n = text, msg_n + 1
        print("[알림]", text)

    def force_stop(reason):
        nonlocal follow, stop_n
        if follow:
            follow = False
            arm.hold()
            stop_n += 1
            say(reason)

    def start_home():
        """노트북 B (또는 이 창 B): 따라가기를 끄고 처음 자세로 부드럽게 간다."""
        nonlocal homing, follow, stop_n
        if home_q is None:
            say("no start pose saved: run save_home.py")
            return
        if paused:
            say("paused on robot PC (press SPACE there)")
            return
        if follow:
            follow = False
            stop_n += 1
        q0 = arm.joints()
        ok, low = path_ok(kin, q0, home_q)
        if not ok:
            arm.hold()
            say(f"path to start pose goes down to {low * 100:.1f} cm over the table: not moving")
            return
        homing = HomeMove(q0, home_q, max_rate=a.home_rate)
        if a.home_grip == "close":                      # 11/23 사용자: B 로 처음 자세를 잡으면 그리퍼도 닫히게
            mapper.width = 0.0
        say(f"going to start pose ({homing.T:.1f} s, SPACE stops)")

    def stop_home(reason):
        nonlocal homing
        if homing is not None:
            homing = None
            arm.hold()
            mapper.sync_to(arm.joints())
            say(reason)

    try:
        while True:
            t = time.monotonic()
            dt, t_prev = t - t_prev, t
            m = srv.poll()
            if srv.new_peer:
                srv.new_peer = False
                force_stop("new laptop connected: follow stopped")
                stop_home("new laptop connected: start pose move stopped")
                prev_req, rec_req_prev, disc_prev, home_prev = None, False, None, None
                seen_peer = srv.peer[0]
                say(f"laptop {seen_peer[0]} connected")
            if m is not None:
                last = m
                if m.get("cal"):
                    rec.extra["laptop_calib"] = m["cal"]          # 에피소드마다 ep_XXX.json 으로 (11/17)
                h = m.get("hand")
                hand = {"pinch": np.array(h[:3], float), "yaw": float(h[3]), "gap": float(h[4])} if h else None
                if hand is not None and len(h) >= 11:            # 11/23 손바닥이 향한 쪽, 엄지 → 손가락 줄
                    hand["normal"], hand["axis"] = np.array(h[5:8], float), np.array(h[8:11], float)
                mapper.follow_grip = bool(m.get("grip", 1))
                mapper.follow_yaw = bool(m.get("yawf", 1))
                mapper.squeeze = float(np.clip(m.get("sq", 0.025), 0.0, 0.05))
                mapper.yaw_gap_min = float(np.clip(m.get("ygm", 0.05), 0.0, 0.1))
                mapper.yaw_pref = float(np.clip(m.get("ypref", -1.047), -3.2, 3.2))
                req = bool(m.get("follow", 0))
                if m.get("bye"):
                    force_stop("laptop closed")
                    stop_home("laptop closed: start pose move stopped")
                    req = False
                if prev_req is None:
                    if req:
                        say("follow was ON before connecting: press SPACE again")
                elif req and not prev_req:
                    if homing is not None:                       # 가는 중 SPACE = 그 자리에 멈춤 (따라가기는 안 켬)
                        stop_home("start pose move stopped (SPACE again to follow)")
                    elif paused:
                        say("paused on robot PC (press SPACE there)")
                    else:
                        mapper.sync_to(arm.joints())
                        w_arm = arm.gripper()                    # 11/23 그리퍼도 지금 폭에서 시작 (hold_start 면 손가락을 움직이기 전까지 그대로)
                        if np.isfinite(w_arm):
                            mapper.width = float(np.clip(w_arm, 0.0, GRIP_MAX))
                        follow = True
                        say("follow started (arm stays as it is)" if mapper.hold_start else "follow started")
                elif not req and prev_req:
                    if follow:
                        follow = False
                        arm.hold()
                prev_req = req
                # 기록: 노트북 R 이 켜면 시작, 끄면 저장
                rq = bool(m.get("rec", 0)) and not m.get("bye")
                if rq and not rec_req_prev and not rec.on:
                    rec.start()
                    say(f"recording ep_{len(rec.saved):03d}")
                elif not rq and rec_req_prev and rec.on:
                    r = rec.stop()
                    say(f"saved {os.path.basename(r[0])} ({r[1]} frames)" if r else "nothing recorded")
                rec_req_prev = rq
                d = int(m.get("disc", 0))
                if disc_prev is not None and d != disc_prev:
                    p = rec.discard_last()
                    say(f"deleted {os.path.basename(p)}" if p else "no episode to delete")
                disc_prev = d
                hn = int(m.get("home", 0))                       # 노트북 B 를 누를 때마다 1 씩 늚
                if home_prev is not None and hn != home_prev and not m.get("bye"):
                    start_home()
                home_prev = hn

            age = srv.age()
            if follow and age > a.timeout:
                force_stop(f"no packet for {age:.1f} s: follow stopped (SPACE on laptop)")
            if rec.on and age > 3.0:
                r = rec.stop()
                rec_req_prev = False
                say(f"link lost, saved {os.path.basename(r[0])} ({r[1]} frames)" if r else "link lost")
            fresh = age < 0.25
            if homing is not None:
                q_home, done = homing.step(dt)
                if a.home_grip == "close":
                    arm.send(q_home, 0.0)                       # 관절과 함께 그리퍼 닫기 (2 mm 넘게 바뀔 때만 명령이 감)
                else:
                    arm.move_joints(q_home)
                if done:
                    homing = None
                    mapper.sync_to(home_q)
                    say("at start pose (SPACE to follow)")
            q_cmd, width = mapper.step(hand if (follow and fresh) else None, dt, follow, ik_iters)
            if mapper.blocked and not was_blocked:        # 11/23 팔 마디가 책상에 닿을 자세: 그 박자는 안 움직임
                say("arm would touch the table: holding (turn the palm back down)")
            was_blocked = mapper.blocked
            if follow:
                arm.send(q_cmd, width)
            q_now = arm.joints()
            w_now = arm.gripper()

            if rec.on:
                imgs = {}
                for k, c in cams.items():                       # 11/26 로봇 카메라: tinybc 로 바꿀 때 다시 압축하므로 품질 95
                    f, _, n = c.latest_with_time()
                    if k not in cam_cache or cam_cache[k][0] != n:
                        cam_cache[k] = (n, jpeg(f, 640, quality=95))
                    imgs[k] = cam_cache[k][1]
                qd = arm.joint_vel() if hasattr(arm, "joint_vel") else None
                if qd is None or not np.all(np.isfinite(qd)):
                    qd = ((q_now - q_prev_rec) / max(t - t_prev_rec, 1e-3)) if q_prev_rec is not None else np.zeros(6)
                q_prev_rec, t_prev_rec = q_now.copy(), t
                g_force = arm.gripper_force() if hasattr(arm, "gripper_force") else float("nan")
                lh = last.get("hand") if last else None
                rec.add(t=t, seq=int(last["seq"]) if last else -1, q=q_now, grip=w_now, q_cmd=mapper.q_cmd.copy(),
                        grip_cmd=mapper.width, hand=(np.array(lh[:3], float) if lh else np.full(3, np.nan)),
                        hand_yaw=(float(lh[3]) if lh else np.nan), hand_gap=(float(lh[4]) if lh else np.nan),
                        hand_normal=(np.array(lh[5:8], float) if lh and len(lh) >= 11 else np.full(3, np.nan)),
                        blocked=mapper.blocked,
                        target=mapper.cur.copy(), tracked=lh is not None and fresh, follow=follow, link_age=age,
                        lm2d=(np.array(last["l2"], float).reshape(21, 2) if last and last.get("l2") else np.full((21, 2), np.nan)),
                        lm3d=(np.array(last["l3"], float).reshape(21, 3) if last and last.get("l3") else np.full((21, 3), np.nan)),
                        hand_raw=(np.array(last["hr"], float) if last and last.get("hr") else np.full(5, np.nan)),
                        grip_mode=(int(last.get("gm", -1)) if last else -1),
                        # 11/18 카메라 두 대: 손 측정이 어디서 왔나 (2 삼각측량, 1 한 대, 0 없음), 두 번째 화면 21점, 삼각측량 21점(책상 좌표)
                        src=(int(last.get("src", -1)) if last else -1),
                        lm2d_b=(np.array(last["l2b"], float).reshape(21, 2) if last and last.get("l2b") else np.full((21, 2), np.nan)),
                        tri3d=(np.array(last["t3"], float).reshape(21, 3) if last and last.get("t3") else np.full((21, 3), np.nan)),
                        qvel=np.asarray(qd, np.float64), grip_force=float(g_force),   # 11/26 tinybc obs/qvel
                        **{f"{k}_jpg": v for k, v in imgs.items()})

            srv.reply(q=[round(float(v), 5) for v in q_now], w=(round(float(w_now), 4) if np.isfinite(w_now) else None),
                      tgt=[round(float(v), 4) for v in mapper.cur], follow=int(follow), rec=int(rec.on),
                      real=int(arm.is_real), x0=a.x0, y0=a.y0, stop=stop_n, msg=msg, msg_n=msg_n,
                      paused=int(paused), eps=len(rec.saved), ver=VER, homing=int(homing is not None))

            if win:
                link = (f"laptop {seen_peer[0]}" if seen_peer else f"waiting :{a.port}")
                extra = [link, (f"!no packet {age:.1f}s" if age > a.timeout and seen_peer else
                                (f"packet {age*1000:.0f} ms ago" if seen_peer else "")),
                         "!PAUSED (SPACE)" if paused else ("REAL ARM" if arm.is_real else "preview arm")]
                panel = robot_panel(kin, q_now, mapper.cur, a.x0, a.y0, w_now if np.isfinite(w_now) else mapper.width,
                                    follow, rec.on, extra)
                if msg:
                    cv2.putText(panel, msg[:48], (8, 24 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
                cv2.imshow(win, panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("b"):
                    start_home()
                if key == ord(" "):
                    paused = not paused
                    if paused:
                        force_stop("paused on robot PC")
                        stop_home("paused on robot PC: start pose move stopped")
                        if not follow:
                            arm.hold()
                    else:
                        say("unpaused: press SPACE on laptop to follow")
                        prev_req = None
            if a.run_seconds is not None and t - t_start > a.run_seconds:
                break
            spare = period - (time.monotonic() - t)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        if rec.on:
            r = rec.stop()
            if r:
                print(f"[기록] {r[0]} ({r[1]} 장면)")
        arm.hold()
        if hasattr(arm, "report"):
            print(arm.report())
        if a.cmd_log and hasattr(arm, "save_log"):
            p = arm.save_log(a.cmd_log)
            if p:
                print(f"[명령 기록] {p}")
        arm.close()
        for c in cams.values():
            c.close()
        srv.close()
        if win:
            cv2.destroyAllWindows()
        print(f"[끝] 기록 {len(rec.saved)}개" + (f" → {rec.dir}" if rec.saved else ""))


if __name__ == "__main__":
    main()
