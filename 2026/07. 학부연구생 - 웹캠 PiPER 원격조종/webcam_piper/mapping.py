"""사람 손 집기점 → 로봇 집게 끝 목표 (2026-11-15).

규칙은 하나: **사람 손이 시작 표시에서 움직인 만큼, 로봇 집게 끝도 로봇 시작 표시에서 똑같이 움직인다.**
  로봇 목표 = 로봇 시작점 + (사람 집기점 − 사람 시작점) × gain (기본 1.0, 같은 거리)
사람 시작점 = A4 한가운데 (+), 높이 PINCH_H0. 로봇 시작점 = 밑동에서 앞으로 x0 (기본 0.30 m), 옆 y0, 높이 = 책상 + PINCH_H0.
그래서 두 책상의 물체를 각자 시작 표시에서 같은 거리·같은 방향에 두면, 사람이 물체를 집을 때 집게도 같은 물체를 집는다.

집게 방향 (11/16): 사람 엄지 → 나머지 손가락 줄과 같은 줄로 집게가 닫히게 돌린다 (두 책상의 앞·왼쪽이 같으므로 그대로).
  집게는 앞뒤가 같아서 180° 마다 같은 모양이다. 따라가기를 켠 첫 장면은 오른손이 잘 도는 쪽(yaw_pref, 기본 −60°)에 가까운 것을,
  그다음부터는 바로 전 목표에 가까운 것을 고른다 (갑자기 반 바퀴 돌지 않게).
  6번 관절은 ±120° 까지라, 집게 yaw 는 팔이 향한 방향(1번 관절) ±112° 안에 둔다 (6번 ≈ 1번 − yaw, 11/16 역기구학으로 확인).
  그 끝에 닿으면 집게는 거기서 멈추고, 반 바퀴 돌아 반대쪽으로 넘어가지 않는다 (물체를 든 채 휙 도는 것이 더 위험).
  손을 오므리는 중(손가락 사이 < yaw_gap_min)에는 방향을 새로 읽지 않는다. 쥐는 순간 엄지·손가락 줄이 흔들리기 때문.

안전: 목표를 작업 상자 안으로 자르고, 목표가 움직이는 속도와 관절이 한 박자에 바뀌는 양을 제한한다.

VR 손잡이 (11/17, vr_handle.py): step_robot 은 로봇 좌표 목표를 바로 받는다 (사람 손 → 로봇 바꾸기는 vr_handle 이 함).
  작업 상자·속도·관절 제한·역기구학은 같고, 집게 방향은 한 박자에 도는 양을 제한해 따라가기를 켤 때 손목이 휙 돌지 않게 한다.
"""
import math

import numpy as np

from geometry import PINCH_H0
from piper_kin import HOME_Q, TABLE_Z, grip_orientation
from robots import GRIP_MAX


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotate_toward(cur, goal, max_angle):
    """회전 행렬 cur 를 goal 쪽으로 최대 max_angle (rad) 만큼 돌린 것 (축-각, numpy 만)."""
    err = goal @ cur.T
    c = float(np.clip((np.trace(err) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(c)
    if ang <= max_angle or ang < 1e-9:
        return goal.copy()
    w = np.array([err[2, 1] - err[1, 2], err[0, 2] - err[2, 0], err[1, 0] - err[0, 1]])
    if np.linalg.norm(w) < 1e-6:                      # 거의 반 바퀴: 대칭 부분에서 축을 고름
        sym = (err + np.eye(3)) / 2.0
        w = sym[:, int(np.argmax(np.diag(sym)))]
    k = w / np.linalg.norm(w)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    step = np.eye(3) + math.sin(max_angle) * K + (1 - math.cos(max_angle)) * (K @ K)
    out = step @ cur
    u, _, vt = np.linalg.svd(out)                     # 반올림이 쌓이지 않게 다시 직교로
    return u @ vt


class Mapper:
    def __init__(self, kin, x0=0.30, y0=0.0, gain=1.0, max_speed=0.12, max_yaw_rate=math.radians(60),
                 max_joint_rate=1.0, squeeze=0.025, yaw_limit=math.radians(112), yaw_gap_min=0.05,
                 yaw_pref=math.radians(-60), min_z=0.006, max_z=0.25, box_x=(0.14, 0.50), box_y=0.25):
        self.kin = kin
        self.robot_start = np.array([x0, y0, TABLE_Z + PINCH_H0])
        self.gain = gain
        self.max_speed, self.max_yaw_rate, self.max_joint_rate = max_speed, max_yaw_rate, max_joint_rate
        self.squeeze, self.yaw_limit, self.yaw_gap_min, self.yaw_pref = squeeze, yaw_limit, yaw_gap_min, yaw_pref
        self.yaw_fresh = True
        # 집게 끝이 책상에 박히지 않게 min_z 위까지 (기본 6 mm). 실물 첫 시험은 robot_server --min-z 0.10 처럼 높여서
        # 작업 상자: 밑동에서 앞 box_x, 좌우 ±box_y (11/19 robot_server 에서 넓힐 수 있게. 기본은 11/15 값)
        self.box_lo = np.array([box_x[0], -box_y, TABLE_Z + min_z])
        self.box_hi = np.array([box_x[1], box_y, TABLE_Z + max_z])
        self.reach = 0.55                                            # 밑동 어깨(높이 0.12 m) 에서 이 거리까지
        self.yaw0 = 0.0                                              # 예전 방식(손등 가로줄 기준)의 흔적. 지금은 0
        self.follow_yaw = True
        self.follow_grip = True
        self.q_cmd = HOME_Q.copy()
        pos, _ = kin.fk(self.q_cmd)
        self.cur = pos.copy()                                        # 지금 보내는 목표 (속도 제한을 거친 것)
        self.cur_yaw = self.yaw_of(self.q_cmd)
        self.yaw_goal = self.cur_yaw
        self.cur_mat = kin.fk(self.q_cmd)[1]                         # step_robot 이 역기구학에 주는 집게 방향 (11/17)
        self.width = GRIP_MAX
        self.desired = self.cur.copy()
        # 11/20 패킷 사이 잇기 (robot_server 가 켬): 손 값은 ~23 Hz 로 오는데 팔 명령은 100 Hz 라, 새 값이 올 때마다
        # 최고 속도로 갔다 서기를 되풀이했다 (끊어 가며 움직이는 지터). 켜면 새 값이 올 때의 목표에서 새 값까지
        # 평균 패킷 간격 동안 곧게 간다. 가상 손에서 목표 가속도 거칠기 38 → 12 m/s², 늦음 +12 ms.
        # 같은 손 dict 가 다시 오면 같은 패킷 (robot_server 는 패킷마다 dict 를 새로 만듦). teleop 혼자 돌 때는 끔 (장면마다 한 번 부름)
        self.interp = False
        self._pkt, self._since, self._pkt_dt = None, 0.0, 1.0 / 23.0
        self._seg_age, self._seg_p0, self._seg_y0 = 0.0, self.cur.copy(), self.cur_yaw

    def human_to_robot(self, pinch):
        p = self.robot_start + self.gain * (np.asarray(pinch) - np.array([0.0, 0.0, PINCH_H0]))
        return self.clip_robot(p)

    def inside(self, p, tol=0.005):
        """p 가 작업 상자(와 팔 닿는 거리) 안인지."""
        return bool(np.linalg.norm(self.clip_robot(p) - np.asarray(p)) < tol)

    def clip_robot(self, p):
        """로봇 좌표 목표를 작업 상자와 팔이 닿는 거리 안으로 자른다."""
        p = np.clip(np.asarray(p, np.float64), self.box_lo, self.box_hi)
        shoulder = np.array([0.0, 0.0, TABLE_Z + 0.12])
        v = p - shoulder
        n = np.linalg.norm(v)
        if n > self.reach:
            p = shoulder + v * (self.reach / n)
        return p

    def step(self, meas, dt, follow, ik_iters=40):
        """meas = 거른 손 측정 dict(pinch, yaw, gap) 또는 None(손 못 찾음). 돌려줌 (관절 목표, 집게 폭).
        ik_iters: 한 박자에 역기구학 반복 수. 박자가 빠르면(연구실 PC 100 Hz) 10 이면 된다 (앞 박자 답에서 이어 풀기 때문)."""
        dt = float(np.clip(dt, 1e-3, 0.2))
        if follow and meas is not None:
            goal = self.human_to_robot(meas["pinch"])
            if self.follow_yaw and meas["gap"] > self.yaw_gap_min:
                y = meas["yaw"] - self.yaw0
                ref = self.yaw_pref if self.yaw_fresh else self.yaw_goal
                c = ref + math.remainder(y - ref, math.pi)                         # 180° 다른 두 방향 중 ref 에 가까운 쪽
                self.yaw_fresh = False
                j1 = math.atan2(goal[1], goal[0])                                  # 팔이 향한 방향
                self.yaw_goal = float(np.clip(c, j1 - self.yaw_limit, j1 + self.yaw_limit))   # 끝에서는 멈춤 (반 바퀴 돌지 않음)
            if self.follow_grip:
                self.width = float(np.clip(meas["gap"] - self.squeeze, 0.0, GRIP_MAX))
            if self.interp:
                if meas is not self._pkt:                                          # 새 패킷: 지금 목표에서 새 구간 시작
                    if self._pkt is not None:
                        self._pkt_dt = float(np.clip(0.8 * self._pkt_dt + 0.2 * self._since, 0.01, 0.15))
                    self._pkt, self._since, self._seg_age = meas, 0.0, 0.0
                    self._seg_p0, self._seg_y0 = self.cur.copy(), self.cur_yaw
                self._since += dt
                self._seg_age += dt
                s = min(self._seg_age / self._pkt_dt, 1.0)
                self.desired = self._seg_p0 + (goal - self._seg_p0) * s
                yaw_d = self._seg_y0 + (self.yaw_goal - self._seg_y0) * s
            else:
                self.desired = goal
                yaw_d = self.yaw_goal
        else:
            yaw_d = self.cur_yaw                                     # 손을 놓치거나 멈춤이면 그 자리
            self.desired = self.cur.copy()
            self._pkt = None                                         # 다시 잡으면 그 자리에서 새로 시작 (끊긴 시간은 간격에 안 넣음)
        step = self.desired - self.cur
        n = np.linalg.norm(step)
        lim = self.max_speed * dt
        if n > lim:
            step *= lim / n
        self.cur = self.cur + step
        self.cur_yaw += float(np.clip(yaw_d - self.cur_yaw, -self.max_yaw_rate * dt, self.max_yaw_rate * dt))
        q = self.kin.ik(self.cur, grip_orientation(self.cur_yaw), self.q_cmd, iters=ik_iters)
        dq = np.clip(q - self.q_cmd, -self.max_joint_rate * dt, self.max_joint_rate * dt)
        self.q_cmd = self.q_cmd + dq
        return self.q_cmd.copy(), self.width

    def step_robot(self, target, dt, yaw=None, width=None, max_speed=None, max_rot_rate=None, ik_iters=10):
        """VR 손잡이 (11/17): 로봇 좌표의 집게 끝 목표를 바로 받는다. target None 이면 그 자리에 멈춘다.
        yaw: 집게를 아래로 세운 채 이 방향 (팔이 향한 방향 ±yaw_limit 안으로 자름, None 이면 지금 방향 그대로).
        max_speed, max_rot_rate: 이번 박자만 다른 끝 속도(m/s)·방향 도는 속도(rad/s) — 따라가기 시작할 때 천천히 다가가기.
        집게 방향은 지금 방향에서 목표 방향으로 한 박자에 max_rot_rate·dt 까지만 돈다 (켤 때 집게가 옆을 보고 있어도 휙 돌지 않음).
        돌려줌 (관절 목표, 집게 폭)."""
        dt = float(np.clip(dt, 1e-3, 0.2))
        if target is None:                                   # 그 자리: 위치·방향을 그대로 둔다 (역기구학도 안 풂)
            self.desired = self.cur.copy()
            self.yaw_goal = self.cur_yaw
            return self.q_cmd.copy(), self.width
        else:
            self.desired = self.clip_robot(target)
            if width is not None:
                self.width = float(np.clip(width, 0.0, GRIP_MAX))
            if yaw is not None:
                j1 = math.atan2(self.desired[1], self.desired[0])
                self.yaw_goal = float(np.clip(yaw, j1 - self.yaw_limit, j1 + self.yaw_limit))
        step = self.desired - self.cur
        n = np.linalg.norm(step)
        lim = (self.max_speed if max_speed is None else max_speed) * dt
        if n > lim:
            step *= lim / n
        self.cur = self.cur + step
        rate = self.max_yaw_rate if max_rot_rate is None else max_rot_rate
        self.cur_yaw += float(np.clip(self.yaw_goal - self.cur_yaw, -rate * dt, rate * dt))
        self.cur_mat = rotate_toward(self.cur_mat, grip_orientation(self.cur_yaw), rate * dt)
        # 박자가 빨라(100 Hz) 한 박자에 조금씩만 풀어도 된다 (웹캠 30 Hz 는 40번). 한 박자 7 ms → 2 ms 쯤 (11/17)
        q = self.kin.ik(self.cur, self.cur_mat, self.q_cmd, iters=ik_iters)
        dq = np.clip(q - self.q_cmd, -self.max_joint_rate * dt, self.max_joint_rate * dt)
        self.q_cmd = self.q_cmd + dq
        return self.q_cmd.copy(), self.width

    def yaw_of(self, q):
        """관절 → 집게 yaw (grip_orientation 의 yaw, ±90° 로 접음)."""
        _, m = self.kin.fk(q)
        return math.atan2(m[1, 0], m[0, 0]) - math.pi / 2                  # 사이트 x 축이 yaw 0 에서 로봇 y 를 향함

    def sync_to(self, q):
        """실물 팔의 지금 관절에서 출발하도록 맞춘다 (시작할 때)."""
        self.q_cmd = np.asarray(q, np.float64).copy()
        self.cur, self.cur_mat = self.kin.fk(self.q_cmd)
        self.desired = self.cur.copy()
        self.cur_yaw = self.yaw_goal = self.yaw_of(self.q_cmd)
        self.yaw_fresh = True
        self._pkt = None
