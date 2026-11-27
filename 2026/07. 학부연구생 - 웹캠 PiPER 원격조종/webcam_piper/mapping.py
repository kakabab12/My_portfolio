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

손바닥 방향 (11/23, 사용자: "손등이 보이면 집게가 아래, 손바닥이 보이면 집게도 위로"):
  집게가 향하는 쪽 = 손바닥이 향한 쪽, 집게가 닫히는 줄 = 엄지 → 손가락 줄. 손바닥이 아래에서 TILT_DEAD(20°) 안으로만 기울면
  예전처럼 똑바로 아래 (위에서 쥘 때 손바닥이 조금 기운 것까지 따라 하지 않게). 그 밖에서는 남은 각을 0~180° 로 펴서 따라간다.
  PiPER 는 6번 관절 ±120°, 손목 꺾기(5번) ±70° 라 책상 가까이(높이 16 cm 아래)에서는 집게가 옆(90°) 안팎까지만 돈다.
  똑바로 위는 팔을 높이 들었을 때만 된다 (11/23 관절 6만 개 뽑아 봄). 닿지 않으면 역기구학이 위치를 먼저 맞추고 방향은 닿는 데까지.
  집게를 옆이나 위로 세우면 집게 몸통·손가락이 집게 끝보다 아래로 내려가므로, 그만큼 집게 끝 목표를 들어 올린다 (min_height_for).
  그래도 팔 마디가 책상 1 cm 안으로 내려가는 자세가 나오면 그 박자는 움직이지 않는다 (blocked). 끄기: follow_tilt = False (--down-only).

시작할 때 자세 그대로 (11/23 오후, 사용자: "스페이스바 누르고 시작하면 집게가 뜨는데? 자세가 그대로여야지"):
  hold_start = True (robot_server·teleop 기본, --start keep) 면 SPACE 를 누른 뒤 처음 받은 손을 '지금 집게 자세' 로 삼는다.
  그 뒤로는 손이 움직인 만큼 (위치), 손이 돈 만큼 (방향) 집게도 움직인다. 그래서 손이 종이 + 에서 조금 벗어나 있어도,
  손바닥이 살짝 기울어 있어도 SPACE 순간에 팔은 움직이지 않는다. 손가락 벌림도 SPACE 때보다 1 cm 넘게 바뀌기 전에는 그리퍼를 그대로 둔다.
  처음 자세가 가상 팔 모델로는 책상 높이 하한(min_z)보다 낮게 나오면 (11/23 저장 자세: −0.8 cm) 모델 책상 높이가 그만큼 틀린 것으로 보고
  하한을 3 cm 까지 내려 준다. 종이 + 기준 (예전): --start paper.

VR 손잡이 (11/17, vr_handle.py): step_robot 은 로봇 좌표 목표를 바로 받는다 (사람 손 → 로봇 바꾸기는 vr_handle 이 함).
  작업 상자·속도·관절 제한·역기구학은 같고, 집게 방향은 한 박자에 도는 양을 제한해 따라가기를 켤 때 손목이 휙 돌지 않게 한다.
"""
import math

import numpy as np

from geometry import PINCH_H0
from piper_kin import HOME_Q, TABLE_Z, grip_orientation
from robots import GRIP_MAX


def cubic_s(t):
    """11/26 석사형 메모: 3차 시간 맞추기 s(t) = 3t² − 2t³ (t 는 0~1). s(0)=0, s(1)=1, 처음과 끝 속도 ṡ(0)=ṡ(1)=0."""
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


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


TILT_DEAD = math.radians(20)        # 손바닥이 아래에서 이만큼 안으로 기울면 집게는 똑바로 아래
GRIP_BODY = 0.12                    # 집게 끝 → 6번 마디 (m, 11/23 정기구학)
JAW_SIDE = 0.05                     # 집게 끝에서 손가락이 옆으로 나온 거리 (열었을 때 반 폭 + 두께, 어림)
FLIP = np.diag([-1.0, -1.0, 1.0])   # 집게는 향하는 축으로 180° 돌려도 같은 모양
TILT_LEAD = math.radians(15)        # 집게 방향 목표가 팔이 실제로 낸 방향보다 앞설 수 있는 각
GRIP_TAKEOVER = 0.01                # hold_start: SPACE 때 손가락 벌림에서 이만큼 넘게 바뀌어야 그리퍼가 따라가기 시작 (m)
FLOOR_DROP_MAX = 0.03               # hold_start: 처음 자세가 모델 높이 하한보다 낮을 때 하한을 내려 주는 최대 (m)


def rot_angle(a, b):
    """두 회전 행렬 사이 각 (rad)."""
    return math.acos(float(np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0)))


def approach_from_palm(normal, dead=TILT_DEAD):
    """손바닥이 향한 쪽 → 집게가 향할 쪽 (단위 벡터). 아래에서 dead 안이면 똑바로 아래, 밖이면 남은 각을 0~180° 로 편다."""
    n = np.asarray(normal, np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    down = np.array([0.0, 0.0, -1.0])
    tilt = math.acos(float(np.clip(n @ down, -1.0, 1.0)))
    if tilt <= dead:
        return down
    h = n - n[2] * np.array([0.0, 0.0, 1.0])
    k = float(np.linalg.norm(h))
    h = h / k if k > 1e-6 else np.array([1.0, 0.0, 0.0])
    t = (tilt - dead) * math.pi / (math.pi - dead)
    return math.cos(t) * down + math.sin(t) * h


def orient_from(approach, close_dir, prev_close=None):
    """집게가 향하는 쪽(사이트 z)과 닫히는 줄(사이트 x) → 회전 행렬 (열 = 사이트 x, y, z). 닫히는 줄이 향하는 쪽과 거의 같으면
    prev_close 를 쓴다 (손을 세워 엄지가 손가락 위에 오면 줄을 못 읽음)."""
    a = np.asarray(approach, np.float64)
    a = a / np.linalg.norm(a)
    for c in (close_dir, prev_close, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
        if c is None:
            continue
        c = np.asarray(c, np.float64) - float(np.dot(c, a)) * a
        k = float(np.linalg.norm(c))
        if k > 0.3:
            c = c / k
            return np.column_stack([c, np.cross(a, c), a])
    raise ValueError("집게 방향을 못 만듦")


def min_height_for(R, table_z, min_z):
    """집게 방향 R 일 때 집게 끝이 있어야 할 가장 낮은 높이 (몸통·손가락이 책상에 닿지 않게)."""
    a, c = R[:, 2], R[:, 0]
    return table_z + min_z + GRIP_BODY * max(0.0, float(a[2])) + JAW_SIDE * abs(float(c[2]))


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
        self.follow_tilt = True                                      # 11/23 손바닥 방향 따라 집게를 옆·위로 (--down-only 면 끔)
        self.tilt_fresh = True
        self.blocked = False                                         # 11/23 팔 마디가 책상에 닿을 자세라 이 박자는 안 움직임
        self.min_link_z = 0.01                                       # 팔 마디가 책상 위 이만큼은 떠 있어야 (m)
        # 11/23 오후 시작할 때 자세 그대로 (hold_start). sync_to 가 기준을 지우고, 다음에 받은 손이 기준이 된다
        self.hold_start = False
        self._anchor = self._hand0 = self._hand_prev = self._gap0 = None
        self._keep_mat = False
        self.grip_hold = False
        self.floor_drop = 0.0
        self._lo_ext, self._hi_ext = np.zeros(3), np.zeros(3)       # 작업 상자를 처음 자세까지 넓힌 양 (hold_start)
        self.q_cmd = HOME_Q.copy()
        pos, _ = kin.fk(self.q_cmd)
        self.cur = pos.copy()                                        # 지금 보내는 목표 (속도 제한을 거친 것)
        self.cur_yaw = self.yaw_of(self.q_cmd)
        self.yaw_goal = self.cur_yaw
        self.cur_mat = kin.fk(self.q_cmd)[1]                         # step_robot 이 역기구학에 주는 집게 방향 (11/17)
        self.goal_mat = self.cur_mat.copy()                          # 11/23 손바닥 방향 목표
        self._start_tip, self._start_mat = self.cur.copy(), self.cur_mat.copy()   # hold_start 기준 (sync_to 가 다시 잡음)
        self.width = GRIP_MAX
        self.desired = self.cur.copy()
        # 11/20 패킷 사이 잇기 (robot_server 가 켬): 손 값은 ~23 Hz 로 오는데 팔 명령은 100 Hz 라, 새 값이 올 때마다
        # 최고 속도로 갔다 서기를 되풀이했다 (끊어 가며 움직이는 지터). 켜면 새 값이 올 때의 목표에서 새 값까지
        # 평균 패킷 간격 동안 곧게 간다. 가상 손에서 목표 가속도 거칠기 38 → 12 m/s², 늦음 +12 ms.
        # 같은 손 dict 가 다시 오면 같은 패킷 (robot_server 는 패킷마다 dict 를 새로 만듦). teleop 혼자 돌 때는 끔 (장면마다 한 번 부름)
        self.interp = False
        # 11/26 석사형 메모: 중간 점을 θ0 + s(θ1 − θ0), s = 3t² − 2t³ (3차, 처음과 끝에서 속도 0) 로 만든다.
        # "linear" 는 11/20 그대로 (곧게, 같은 빠르기). robot_server --profile cubic 으로 켬
        self.profile = "linear"
        self._pkt, self._since, self._pkt_dt = None, 0.0, 1.0 / 23.0
        self._seg_age, self._seg_p0, self._seg_y0 = 0.0, self.cur.copy(), self.cur_yaw

    def human_to_robot(self, pinch):
        p = self.robot_start + self.gain * (np.asarray(pinch) - np.array([0.0, 0.0, PINCH_H0]))
        return self.clip_robot(p)

    def inside(self, p, tol=0.005):
        """p 가 작업 상자(와 팔 닿는 거리) 안인지."""
        return bool(np.linalg.norm(self.clip_robot(p) - np.asarray(p)) < tol)

    def clip_robot(self, p):
        """로봇 좌표 목표를 작업 상자와 팔이 닿는 거리 안으로 자른다 (hold_start 면 처음 자세까지 넓힌 상자)."""
        p = np.clip(np.asarray(p, np.float64), self.box_lo + self._lo_ext, self.box_hi + self._hi_ext)
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
        tilt = self.follow_tilt and meas is not None and meas.get("normal") is not None
        if follow and meas is not None:
            if self.hold_start:                                                    # SPACE 뒤 처음 받은 손 = 지금 집게 끝
                if self._anchor is None:
                    self._anchor = np.asarray(meas["pinch"], np.float64).copy()
                goal = self.clip_robot(self._start_tip + self.gain * (np.asarray(meas["pinch"], np.float64) - self._anchor))
            else:
                goal = self.human_to_robot(meas["pinch"])
            if tilt and self.follow_yaw:
                self.goal_mat = self.tilt_goal(meas, goal)
            elif self.follow_yaw and meas["gap"] > self.yaw_gap_min:
                y = meas["yaw"] - self.yaw0
                ref = self.yaw_pref if self.yaw_fresh else self.yaw_goal
                c = ref + math.remainder(y - ref, math.pi)                         # 180° 다른 두 방향 중 ref 에 가까운 쪽
                self.yaw_fresh = False
                j1 = math.atan2(goal[1], goal[0])                                  # 팔이 향한 방향
                self.yaw_goal = float(np.clip(c, j1 - self.yaw_limit, j1 + self.yaw_limit))   # 끝에서는 멈춤 (반 바퀴 돌지 않음)
            if self.follow_grip:
                if self.grip_hold:                                                 # SPACE 때 그리퍼 폭 그대로, 손가락이 1 cm 넘게 움직이면 따라감
                    if self._gap0 is None:
                        self._gap0 = float(meas["gap"])
                    if abs(float(meas["gap"]) - self._gap0) > GRIP_TAKEOVER:
                        self.grip_hold = False
                if not self.grip_hold:
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
                if self.profile == "cubic":
                    s = cubic_s(s)
                self.desired = self._seg_p0 + (goal - self._seg_p0) * s
                yaw_d = self._seg_y0 + (self.yaw_goal - self._seg_y0) * s
            else:
                self.desired = goal
                yaw_d = self.yaw_goal
        else:
            yaw_d = self.cur_yaw                                     # 손을 놓치거나 멈춤이면 그 자리
            self.desired = self.cur.copy()
            self._pkt = None                                         # 다시 잡으면 그 자리에서 새로 시작 (끊긴 시간은 간격에 안 넣음)
        if tilt or not self.follow_tilt_off_path():
            # 11/23 손바닥 방향: 집게 방향은 행렬로 돌리고, 옆·위로 세운 만큼 집게 끝 목표를 들어 올린다
            mz = self.box_lo[2] - self.floor_drop - TABLE_Z
            if follow and meas is not None and tilt:
                nxt = rotate_toward(self.cur_mat, self.goal_mat, self.max_yaw_rate * dt)
                # 먼저 올라가고 나서 돈다: 다음 방향이 지금 높이로는 책상에 닿으면 이 박자는 돌지 않는다
                need_nxt, need_now = min_height_for(nxt, TABLE_Z, mz), min_height_for(self.cur_mat, TABLE_Z, mz)
                if need_nxt <= max(self.cur[2], need_now) + 1e-4:
                    self.cur_mat = nxt
            lo = min_height_for(self.cur_mat, TABLE_Z, mz)
            self.desired = self.desired.copy()
            self.desired[2] = min(max(self.desired[2], lo), self.box_hi[2] + self._hi_ext[2])
        step = self.desired - self.cur
        n = np.linalg.norm(step)
        lim = self.max_speed * dt
        if n > lim:
            step *= lim / n
        self.cur = self.cur + step
        if tilt or not self.follow_tilt_off_path():
            q = self.kin.ik(self.cur, self.cur_mat, self.q_cmd, iters=ik_iters)
        else:
            self.cur_yaw += float(np.clip(yaw_d - self.cur_yaw, -self.max_yaw_rate * dt, self.max_yaw_rate * dt))
            if follow and meas is not None:
                self._keep_mat = False
            if not self._keep_mat:                                   # hold_start: 손을 받기 전에는 처음 자세 방향을 그대로 (똑바로 아래로 펴지 않음)
                self.cur_mat = grip_orientation(self.cur_yaw)
                self.goal_mat = self.cur_mat.copy()
            q = self.kin.ik(self.cur, self.cur_mat, self.q_cmd, iters=ik_iters)
        dq = np.clip(q - self.q_cmd, -self.max_joint_rate * dt, self.max_joint_rate * dt)
        q_new = self.q_cmd + dq
        if not self.follow_tilt_off_path():
            # 관절이 닿지 않는 방향으로 목표만 계속 돌아가지 않게: 팔이 실제로 낸 방향에서 TILT_LEAD 안으로 붙잡아 둔다
            # (그래야 집게 끝을 들어 올리는 높이도 실제 방향에 맞고, 손을 되돌리면 바로 따라 돌아온다)
            m_ach = self.kin.fk(q_new)[1]
            if rot_angle(m_ach, self.cur_mat) > TILT_LEAD:
                self.cur_mat = rotate_toward(m_ach, self.cur_mat, TILT_LEAD)
        low = float(np.asarray(self.kin.link_points(q_new))[1:-1, 2].min()) - TABLE_Z
        self.blocked = low < self.min_link_z and low < (float(np.asarray(self.kin.link_points(self.q_cmd))[1:-1, 2].min()) - TABLE_Z)
        if not self.blocked:
            self.q_cmd = q_new
        return self.q_cmd.copy(), self.width

    def follow_tilt_off_path(self):
        """예전 길(집게는 늘 아래, yaw 만)로 가는지: 손바닥 따라가기를 껐거나, 아직 손바닥 방향을 한 번도 받지 못했을 때."""
        return (not self.follow_tilt) or self.tilt_fresh

    def tilt_goal(self, meas, goal_pos):
        """손바닥 방향·엄지 → 손가락 줄 → 집게 목표 방향 (11/23). 180° 같은 모양 둘 중 바로 전 목표에 가까운 쪽 (처음은 지금 집게 방향).
        손을 오므리는 중(손가락 사이 < yaw_gap_min)에는 닫히는 줄을 새로 읽지 않는다. 집게가 아래를 볼 때는 예전처럼
        팔이 향한 방향 ±yaw_limit 안으로 돌림을 자른다 (6번 관절 끝에서 반 바퀴 돌지 않게)."""
        ref = self.cur_mat if self.tilt_fresh else self.goal_mat
        a = approach_from_palm(meas["normal"])
        close = meas.get("axis") if meas["gap"] > self.yaw_gap_min else None
        if self.hold_start:
            # 손이 SPACE 뒤 처음 모양(H0)에서 돈 만큼 처음 집게 방향을 돌린다: 목표 = H · H0ᵀ · 처음 방향
            prev = self._hand_prev
            H = orient_from(a, close, prev_close=None if prev is None else prev[:, 0])
            if prev is not None and rot_angle(H @ FLIP, prev) < rot_angle(H, prev):
                H = H @ FLIP                                      # 180° 같은 모양 둘 중 바로 전 손 모양에 이어지는 쪽
            self._hand_prev = H
            if self._hand0 is None:
                self._hand0 = H.copy()
            R = H @ self._hand0.T @ self._start_mat
        else:
            R = orient_from(a, close, prev_close=ref[:, 0])
            R2 = R @ FLIP
            R = R if rot_angle(R, ref) <= rot_angle(R2, ref) else R2
        if R[2, 2] < -0.7:                                        # 거의 아래: 돌림(yaw) 을 팔 방향 ±yaw_limit 안으로
            yaw = math.atan2(R[1, 0], R[0, 0]) - math.pi / 2
            j1 = math.atan2(goal_pos[1], goal_pos[0])
            if self.hold_start:                                   # 팔 방향에 가장 가까운 표현으로 (처음 자세 그대로면 안 자름)
                ref_yaw = j1
            else:
                ref_yaw = self.yaw_pref if self.tilt_fresh else math.atan2(self.goal_mat[1, 0], self.goal_mat[0, 0]) - math.pi / 2
            yaw_c = ref_yaw + math.remainder(yaw - ref_yaw, 2 * math.pi)
            yaw_cl = float(np.clip(yaw_c, j1 - self.yaw_limit, j1 + self.yaw_limit))
            if abs(yaw_cl - yaw_c) > 1e-9:
                from piper_kin import rot_z
                R = rot_z(yaw_cl - yaw_c) @ R
        self.tilt_fresh = False
        return R

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
        self.goal_mat = self.cur_mat.copy()
        self.yaw_fresh = True
        self.tilt_fresh = True
        self.blocked = False
        self._pkt = None
        # 11/23 오후 hold_start: 다음에 받은 손이 기준. 처음 자세가 작업 상자·높이 하한 밖이어도 그 자리에서 움직이지 않게 넓힌다
        self._anchor = self._hand0 = self._hand_prev = self._gap0 = None
        self._start_tip, self._start_mat = self.cur.copy(), self.cur_mat.copy()
        self._keep_mat = self.grip_hold = self.hold_start
        self.floor_drop = 0.0
        self._lo_ext, self._hi_ext = np.zeros(3), np.zeros(3)
        if self.hold_start:
            need = min_height_for(self.cur_mat, TABLE_Z, self.box_lo[2] - TABLE_Z)
            self.floor_drop = float(np.clip(need - self.cur[2], 0.0, FLOOR_DROP_MAX))
            self._lo_ext[:2] = np.minimum(0.0, self.cur[:2] - self.box_lo[:2])
            self._lo_ext[2] = -self.floor_drop
            self._hi_ext = np.maximum(0.0, self.cur - self.box_hi)
