"""follower: 팔에 보내는 일만 따로 하는 일꾼 (2026-11-26, 석사형 메모).

석사형 메모 (찬혁님 VR 조종 구조):
  손 자세 받기 → joint = ik(frame) → follower.append(joint)
  follower:  while 200 Hz:  joint = q.popleft()  → 팔
  중간 점:   θ_t = θ0 + s(θ1 − θ0),  s = 3t² − 2t³  (mapping.cubic_s, robot_server --profile cubic)
  팔:        MIT 모드 (모터가 a = kp·e + kd·ė 로 당김), kp 5, kd 0.8

여기서는
  - robot_server 의 100 Hz 박자가 mapper 로 관절 목표를 풀어 push() 한다 (작업 상자·속도·책상 높이 제한은 mapper 그대로).
  - push 할 때 바로 전 목표에서 새 목표까지를 follower 박자 몇 개로 나눠 줄에 넣는다 (100 Hz → 200 Hz 면 2개).
  - follower 쓰레드는 박자마다 줄에서 하나 꺼내 팔에 보낸다. 줄이 비면 마지막 점을 속도 0 으로 계속 보낸다 (그 자리).
  - 줄은 짧게 묶는다 (기본 20 ms 어치). 가득 차면 가장 옛 점을 버린다 → 늦음이 쌓이지 않는다.
    (석사형 구조는 WebSocket 이라 순서대로 다 오지만, 여기 손 값은 UDP 라 빠지거나 몰려 온다.)
  - 박자가 밀리면 따라잡으려고 몰아 보내지 않는다 (다음 박자부터).
  - 점과 점 사이 관절 변화를 max_rate × 박자 로 한 번 더 자르고 관절 한계 안에 둔다
    (mapper 가 이미 자르지만, MIT 는 팔 안에서 속도를 자르지 않아서).

팔 연결과 같은 꼴 (joints, gripper, send, move_joints, hold, close, is_real) 이라 robot_server 는 거의 그대로 쓴다.
실물 PiPER 에서 MIT 모드로 움직여 본 적은 아직 없다 (11/26 가상 팔로만).
"""
import threading
import time
from collections import deque

import numpy as np


class Follower:
    def __init__(self, arm, hz=200.0, mode="position", kp=5.0, kd=0.8, push_hz=100.0, max_rate=3.0,
                 lo=None, hi=None, queue_s=0.02, log=False, start=True):
        if mode not in ("position", "mit"):
            raise ValueError(mode)
        if mode == "mit" and not hasattr(arm, "move_mit"):
            raise ValueError("이 팔 연결은 MIT 모드를 못 함 (agxbus 나 preview 만)")
        self.arm, self.hz, self.mode, self.kp, self.kd = arm, float(hz), mode, float(kp), float(kd)
        self.is_real = arm.is_real
        self.dt = 1.0 / self.hz
        self.n_sub = max(1, int(round(self.hz / push_hz)))            # push 하나를 박자 몇 개로 나누나
        self.max_rate = float(max_rate)
        self.max_step = self.max_rate * self.dt
        self.lo = None if lo is None else np.asarray(lo, np.float64)
        self.hi = None if hi is None else np.asarray(hi, np.float64)
        self.lock = threading.RLock()                                 # 팔 부르기 (follower 쓰레드와 robot_server 가 같이 씀)
        self.qlock = threading.Lock()                                 # 줄
        self.q = deque(maxlen=max(self.n_sub, int(round(queue_s * self.hz))))
        q0 = np.asarray(arm.joints(), np.float64)
        self.last_push = q0.copy()
        self.last_sent = q0.copy()
        self.zero = np.zeros_like(q0)
        self.n_sent = self.n_hold = self.n_drop = self.n_late = self.n_clip = self.n_err = 0
        self.gaps = deque(maxlen=20000)                               # 보낸 간격 (초), 통계용
        self.log = [] if log else None                                # (시각, 보낸 관절, 보낸 속도, 줄이 비었나)
        self.running = start
        self.thread = threading.Thread(target=self._run, daemon=True)
        if start:                                                     # 시험은 start=False 로 tick() 을 직접 부름
            self.thread.start()

    # ---- robot_server 가 부름 (팔 연결과 같은 꼴) ----
    def push(self, q):
        """관절 목표 하나 (100 Hz 박자). 바로 전 목표에서 여기까지를 follower 박자 n_sub 개로 나눠 줄에 넣는다."""
        q = np.asarray(q, np.float64)
        with self.qlock:
            q0 = self.last_push
            v = (q - q0) / (self.n_sub * self.dt)
            for k in range(1, self.n_sub + 1):
                if len(self.q) == self.q.maxlen:
                    self.n_drop += 1                                  # 가장 옛 점이 밀려남 (늦음이 쌓이지 않게)
                self.q.append((q0 + (q - q0) * (k / self.n_sub), v))
            self.last_push = q.copy()

    def send(self, q, width):
        self.push(q)
        with self.lock:
            self.arm.send_gripper(width)

    def move_joints(self, q):
        self.push(q)

    def hold(self):
        """지금 실제 관절에서 멈춤: 줄을 비우고 그 자리를 속도 0 으로."""
        with self.lock:
            qn = np.asarray(self.arm.joints(), np.float64)
        with self.qlock:
            self.q.clear()
            self.last_push = qn.copy()
            self.q.append((qn, self.zero.copy()))

    def joints(self):
        with self.lock:
            return self.arm.joints()

    def gripper(self):
        with self.lock:
            return self.arm.gripper()

    def joint_vel(self):
        with self.lock:
            f = getattr(self.arm, "joint_vel", None)
            return f() if f else None

    def gripper_force(self):
        with self.lock:
            f = getattr(self.arm, "gripper_force", None)
            return f() if f else float("nan")

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            self.arm.close()

    # ---- follower 쓰레드 ----
    def tick(self, now):
        """박자 하나: 줄에서 꺼내 (비면 그 자리) 자르고 보낸다. 시험에서 바로 부를 수 있게 쓰레드와 떼어 둠."""
        with self.qlock:
            item = self.q.popleft() if self.q else None
        if item is None:
            p, v = self.last_sent.copy(), self.zero.copy()
            self.n_hold += 1
        else:
            p, v = item
        step = p - self.last_sent
        lim = np.clip(step, -self.max_step, self.max_step)
        if np.any(np.abs(step - lim) > 1e-6):                         # 한도와 같은 값(매퍼가 이미 자른 것)은 세지 않음
            self.n_clip += 1
            v = lim / self.dt
        p = self.last_sent + lim
        if self.lo is not None:
            p = np.clip(p, self.lo, self.hi)
        v = np.clip(v, -self.max_rate, self.max_rate)
        try:
            with self.lock:
                if self.mode == "mit":
                    self.arm.move_mit(p, v, self.kp, self.kd)
                else:
                    self.arm.move_joints(p)
        except Exception as e:  # noqa: BLE001 — 한 번 실패해도 다음 박자에 다시
            self.n_err += 1
            if self.n_err == 1:
                print(f"[follower] 팔에 보내기 실패: {e}")
            return
        self.last_sent = p
        self.n_sent += 1
        if self.log is not None:
            self.log.append((now, p.copy(), v.copy(), item is None))

    def _run(self):
        nxt = time.perf_counter()
        prev = None
        while self.running:
            now = time.perf_counter()
            if now < nxt:
                time.sleep(nxt - now)
                now = time.perf_counter()
            if now - nxt > self.dt:                                   # 한 박자 넘게 밀림: 몰아 보내지 않고 지금부터 다시
                self.n_late += 1
                nxt = now
            nxt += self.dt
            self.tick(now)
            if prev is not None:
                self.gaps.append(now - prev)
            prev = now

    def report(self):
        g = np.asarray(self.gaps, np.float64)
        if len(g) == 0:
            return "[follower] 보낸 것 없음"
        n = max(self.n_sent + self.n_err, 1)
        return (f"[follower] {self.mode}, 목표 {self.hz:.0f} Hz → 실제 {1.0 / g.mean():.0f} Hz, 간격 가운데 {np.median(g) * 1000:.2f} ms · "
                f"99% {np.percentile(g, 99) * 1000:.2f} ms · 최대 {g.max() * 1000:.1f} ms, 밀린 박자 {self.n_late}, "
                f"줄이 빈 박자 {self.n_hold / n * 100:.0f}%, 버린 점 {self.n_drop}, 한 번 더 자른 점 {self.n_clip}, 실패 {self.n_err}")

    def save_log(self, path):
        if not self.log:
            return None
        t = np.array([r[0] for r in self.log])
        np.savez(path, t=t - t[0], q=np.array([r[1] for r in self.log]), v=np.array([r[2] for r in self.log]),
                 hold=np.array([r[3] for r in self.log]), mode=self.mode, hz=self.hz, kp=self.kp, kd=self.kd)
        return path


class LogArm:
    """follower 없이 (예전 방식) 보낸 명령만 남기는 껍데기 — 비교 시험용."""

    def __init__(self, arm):
        self.arm = arm
        self.log = []

    def send(self, q, width):
        self.log.append((time.perf_counter(), np.asarray(q, np.float64).copy()))
        self.arm.send(q, width)

    def move_joints(self, q):
        self.log.append((time.perf_counter(), np.asarray(q, np.float64).copy()))
        self.arm.move_joints(q)

    def __getattr__(self, name):
        return getattr(self.arm, name)

    def report(self):
        return f"[명령 기록] 예전 방식, 보낸 명령 {len(self.log)}개"

    def save_log(self, path):
        if not self.log:
            return None
        t = np.array([r[0] for r in self.log])
        np.savez(path, t=t - t[0], q=np.array([r[1] for r in self.log]), mode="old")
        return path
