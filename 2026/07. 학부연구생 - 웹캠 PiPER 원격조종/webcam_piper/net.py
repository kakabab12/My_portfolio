"""노트북(손 재기) ↔ 연구실 PC(팔 움직이기) 통신 (2026-11-15).

같은 공유기·같은 망 안에서 UDP 로 짧은 JSON 을 주고받는다. 사진은 보내지 않는다 (숫자 몇 개뿐이라 와이파이로 충분).

노트북 → 연구실 PC (장면마다, 초당 15~30번)
  sid   노트북 프로그램 한 번 켤 때마다 바뀌는 번호 (다른 노트북·옛 프로그램 것과 섞이지 않게)
  seq   보낸 순서 번호
  hand  [x, y, z, yaw, gap] 사람 집기점 (사람 책상 좌표 m), 집게 방향 yaw (엄지 → 맞은편 손가락 줄에 맞춘 것, rad),
        엄지와 맞은편 손가락 사이 (m). 손을 못 찾으면 null
  follow, rec, grip, yawf   따라가기 / 기록 / 집게 따라가기 / 손 돌림 따라가기 를 켰는지 (0·1)
  sq    집게 폭 = gap − sq     ygm  손가락 사이가 이보다 좁으면 집게 방향을 새로 읽지 않음 (m)
  ypref 따라가기를 켤 때 고를 집게 yaw 쪽 (오른손 −60°, 왼손 +60°, rad)
  disc  "마지막 기록 지우기" 를 누른 횟수 (패킷이 하나 빠져도 놓치지 않게 횟수로 보냄)
  gm    손 모양 0 손 전체, 1 엄지·검지 (11/17, TAB 으로 바꿈. 연구실 기록에 grip_mode 로 남음)
  l2, l3, lb, hr, cal   기록 중일 때만: 손 21점 원본(화면 px, MediaPipe 3D), 손 이름, 거르기 전 측정, 1초마다 노트북 보정 값
  bye   노트북 프로그램이 끝남
연구실 PC → 노트북 (받을 때마다 답장)
  ack 받은 seq, q 실제 관절 6, w 실제 집게 폭, tgt 집게 끝 목표, follow·rec 실제로 켜져 있는지, real 실물 팔인지,
  x0·y0 로봇 종이 가운데, stop 연구실 PC 가 따라가기를 강제로 끈 횟수, msg·msg_n 알릴 말,
  ver 연구실 PC 코드 버전 (code_version, 노트북 것과 다르면 노트북 화면에 알림)
"""
import hashlib
import json
import os
import random
import socket
import time

PORT = 5005
HERE = os.path.dirname(os.path.abspath(__file__))
# 노트북과 연구실 PC 가 같은 코드인지 보는 파일들 (줄바꿈 차이는 무시)
RUNTIME_FILES = ("net.py", "geometry.py", "mapping.py", "piper_kin.py", "robots.py", "recorder.py", "panel.py",
                 "hand.py", "teleop.py", "robot_server.py")


def code_version():
    h = hashlib.sha1()
    for f in RUNTIME_FILES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:7]


def _sock():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setblocking(False)
    return s


class Link:
    """노트북 쪽: 연구실 PC 로 손 측정을 보내고 답장(팔 상태)을 받는다."""

    def __init__(self, target):
        host, _, port = target.partition(":")
        self.addr = (socket.gethostbyname(host), int(port or PORT))
        self.sock = _sock()
        self.sid = random.randrange(1 << 30)
        self.seq = 0
        self.sent_t = {}
        self.status = None
        self.last_rx = None
        self.rtt = None

    def send(self, **fields):
        self.seq += 1
        msg = {"sid": self.sid, "seq": self.seq, **fields}
        self.sent_t[self.seq] = time.monotonic()
        if len(self.sent_t) > 200:
            for k in sorted(self.sent_t)[:100]:
                del self.sent_t[k]
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
        except OSError:
            pass                                            # 망이 잠깐 끊김 — 다음 장면에 다시 보냄

    def poll(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:                                 # 윈도우: 상대 포트가 닫혀 있으면 ConnectionResetError 가 남
                break
            try:
                st = json.loads(data)
            except ValueError:
                continue
            now = time.monotonic()
            t0 = self.sent_t.get(st.get("ack"))
            if t0 is not None:
                self.rtt = now - t0
            self.status, self.last_rx = st, now
        return self.status

    @property
    def connected(self):
        return self.last_rx is not None and time.monotonic() - self.last_rx < 1.0

    def close(self):
        self.sock.close()


class Server:
    """연구실 PC 쪽: 손 측정을 받는다. 처음 말을 건 노트북 하나만 듣고, 그 노트북이 3초 넘게 조용하면 다른 노트북을 받는다."""

    def __init__(self, port=PORT, takeover=3.0):
        self.sock = _sock()
        self.sock.bind(("0.0.0.0", port))
        self.port = port
        self.takeover = takeover
        self.peer = None            # (주소, sid)
        self.last_seq = 0
        self.last_rx = None
        self.new_peer = False

    def poll(self):
        """쌓인 패킷을 다 읽고 가장 새것만 돌려준다 (없으면 None)."""
        latest = None
        now = time.monotonic()
        while True:
            try:
                data, addr = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            try:
                m = json.loads(data)
                key = (addr, int(m["sid"]))
                seq = int(m["seq"])
            except (ValueError, KeyError, TypeError):
                continue
            if self.peer != key:
                if self.peer is not None and self.last_rx is not None and now - self.last_rx < self.takeover:
                    continue                                 # 다른 노트북이 쓰는 중
                self.peer, self.last_seq, self.new_peer = key, 0, True
            if seq <= self.last_seq:
                continue                                     # 늦게 도착한 옛 패킷
            self.last_seq, self.last_rx = seq, now
            latest = m
        return latest

    def age(self):
        return float("inf") if self.last_rx is None else time.monotonic() - self.last_rx

    def reply(self, **fields):
        if self.peer is None:
            return
        try:
            self.sock.sendto(json.dumps({"ack": self.last_seq, **fields}, separators=(",", ":")).encode(),
                             self.peer[0])
        except OSError:
            pass

    def close(self):
        self.sock.close()


def my_ips():
    """이 컴퓨터의 망 주소 (노트북에 알려 줄 것)."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))                    # 실제로 보내지는 않음, 바깥으로 나가는 주소만 알아냄
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))
