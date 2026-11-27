"""placo 역기구학 (Rhoban PlaCo, github.com/Rhoban/placo, 2026-11-27).

찬혁님 VR 조종(piper-teleop)도 placo 를 쓴다 (DynamicsSolver, 위치 무게 1 · 방향 0.01, 관절·속도 한계). 여기서는 같은 생각을
KinematicsSolver 로 쓴다: 한 번 풀 때마다 '관절을 얼마나 움직일지'를 이차 계획(QP)으로 구하는데,
  - 관절 한계·속도 한계를 '꼭 지킬 조건'으로 넣고 (numpy 판은 한계에 닿으면 그 관절을 빼고 다시 풂)
  - 집게 끝 위치·방향은 '무게를 준 바람'으로 넣는다 (위치 1, 방향 rot_w)
    11/27 목표 240개 (책상 위 0.5~16 cm, 아래·25° 기울임): 찬혁님 값 0.01 은 방향을 맞추느라 위치가 95% 16.5 mm·최대 32 cm 틀림,
    0.001 은 위치 최대 3.1 mm·방향 가운데 11.5°·1 cm 넘게 틀린 것 0개 (numpy 판: 위치 가운데 0.03 mm 지만 2개가 18 cm 까지 틀림,
    방향 가운데 21°, 한 번 21 ms → placo 1.1 ms). 그래서 기본 0.001
  - 처음 자세 쪽으로 아주 약하게 끌어 (posture), 같은 목표면 같은 자세가 나오게 한다.

로봇 모델: 찬혁님 URDF(piper/robot.urdf, AgileX)는 우리 시뮬 모델과 같은 관절 각도에서 집게 끝이 가운데값 7 mm, 최대 15 mm 다르다
(11/27, 두 모델 사이 고정 변환을 최소제곱으로 맞춘 뒤). 그래서 시뮬과 똑같아야 하는 여기서는 시뮬 모델(webcam_piper/kin/piper_kin.xml,
real_piper 와 같음)의 관절 사슬을 URDF 로 옮겨 쓴다 (write_urdf). 충돌 모양은 넣지 않았다 (자기 충돌 피하기는 아직).

  point_policy/.venv/bin/python point_policy/placo_ik.py   # 시험: 모델 일치·정확도·속도를 numpy 판과 비교
"""
import time
from pathlib import Path

import numpy as np

import common
from piper_kin import HOME_Q, JOINTS, SITE, PiperKin, _chain

URDF = common.HERE / "placo_model" / "piper_sim.urdf"
VEL_LIMIT = 3.0          # rad/s (webcam_piper joint-rate 와 같음)


def _rpy(R):
    """회전 행렬 → URDF rpy (고정축 x, y, z 차례 = R = Rz(y) Ry(p) Rx(r))."""
    sy = -R[2, 0]
    p = np.arcsin(np.clip(sy, -1, 1))
    if abs(np.cos(p)) > 1e-9:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:
        r = np.arctan2(-R[1, 2], R[1, 1])
        y = 0.0
    return r, p, y


def write_urdf(path=URDF, xml=common.WEBCAM / "kin" / "piper_kin.xml"):
    """시뮬 모델의 몸통 사슬(위치·방향·관절 축·범위) → URDF. 관절 없는 몸통은 고정 관절."""
    steps, (sp, sR) = _chain(str(xml), SITE)
    L = ['<?xml version="1.0"?>', '<robot name="piper_sim">', '  <link name="world"/>']
    inertial = ('<inertial><origin xyz="0 0 0"/><mass value="0.1"/>'
                '<inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial>')
    parent = "world"
    for name, pos, R, jn, axis, rng in steps:
        r, p, y = _rpy(R)
        L.append(f'  <link name="{name}">{inertial}</link>')
        origin = f'<origin xyz="{pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}" rpy="{r:.12f} {p:.12f} {y:.12f}"/>'
        if jn:
            ax = axis / np.linalg.norm(axis)
            L.append(f'  <joint name="{jn}" type="revolute"><parent link="{parent}"/><child link="{name}"/>{origin}'
                     f'<axis xyz="{ax[0]:.9f} {ax[1]:.9f} {ax[2]:.9f}"/>'
                     f'<limit lower="{rng[0]:.6f}" upper="{rng[1]:.6f}" effort="40" velocity="{VEL_LIMIT}"/></joint>')
        else:
            L.append(f'  <joint name="fixed_{name}" type="fixed"><parent link="{parent}"/><child link="{name}"/>{origin}</joint>')
        parent = name
    r, p, y = _rpy(sR)
    L.append(f'  <link name="{SITE}">{inertial}</link>')
    L.append(f'  <joint name="fixed_{SITE}" type="fixed"><parent link="{parent}"/><child link="{SITE}"/>'
             f'<origin xyz="{sp[0]:.9f} {sp[1]:.9f} {sp[2]:.9f}" rpy="{r:.12f} {p:.12f} {y:.12f}"/></joint>')
    L.append('</robot>')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


class PlacoIK:
    """PiperKin.ik 와 같은 꼴: ik(pos, R, q0) → q. step() 은 속도 한계를 지키며 dt 만큼만 다가감 (실시간·정책 박자용)."""

    def __init__(self, rot_w=1e-3, posture_w=1e-4, dt=0.01):
        import placo
        if not URDF.exists():
            write_urdf()
        self.placo = placo
        self.robot = placo.RobotWrapper(str(URDF), placo.Flags.ignore_collisions)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.dt = dt
        self.task = self.solver.add_frame_task(SITE, np.eye(4))
        self.task.configure("grip", "soft", 1.0, rot_w)
        self.posture = self.solver.add_joints_task()
        self.posture.set_joints({j: float(v) for j, v in zip(JOINTS, HOME_Q)})
        self.posture.configure("posture", "soft", posture_w)
        self.set_q(HOME_Q)

    def set_q(self, q):
        for j, v in zip(JOINTS, q):
            self.robot.set_joint(j, float(v))
        self.robot.update_kinematics()

    def q(self):
        return np.array([self.robot.get_joint(j) for j in JOINTS])

    def fk(self, q=None):
        if q is not None:
            self.set_q(q)
        T = self.robot.get_T_world_frame(SITE)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def _target(self, pos, R):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pos
        self.task.T_world_frame = T

    def ik(self, pos, R, q0, iters=40, tol=2e-4):
        """멀리 있는 목표까지 다 푼다 (속도 한계 끔). 자료 만들기·평가에서 PiperKin.ik 대신."""
        self.set_q(q0)
        self._target(pos, R)
        self.solver.enable_velocity_limits(False)
        for _ in range(iters):
            q_prev = self.q()
            self.solver.solve(True)
            self.robot.update_kinematics()
            if np.max(np.abs(self.q() - q_prev)) < tol * 0.1:
                break
        return self.q()

    def step(self, pos, R, dt):
        """지금 자세에서 dt 초 동안 갈 수 있는 만큼 (관절 속도 VEL_LIMIT 안)."""
        self._target(pos, R)
        self.solver.dt = dt
        self.solver.enable_velocity_limits(True)
        self.solver.solve(True)
        self.robot.update_kinematics()
        return self.q()


def _selftest():
    from piper_kin import DOWN
    kin = PiperKin()
    ik = PlacoIK()
    rng = np.random.default_rng(0)
    # 1) 모델 일치 (같은 관절 → 같은 집게 끝)
    e = []
    for _ in range(200):
        q = rng.uniform(kin.lo, kin.hi)
        p1, R1 = kin.fk(q)
        p2, R2 = ik.fk(q)
        e.append((np.linalg.norm(p1 - p2), np.abs(R1 - R2).max()))
    e = np.array(e)
    print(f"1) 모델 일치: 집게 끝 위치 차 최대 {e[:, 0].max():.1e} m, 방향 차 최대 {e[:, 1].max():.1e}")

    # 2) 역기구학: 아래로 세운 집게 + 기울인 집게, 책상 위 높이 여러 곳
    def tilt(deg):
        a = np.radians(deg)
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ DOWN
    rows = []
    for name, solve in (("numpy (위치 먼저)", lambda p, R, q0: kin.ik(p, R, q0, iters=60)),
                        ("placo (QP)", lambda p, R, q0: ik.ik(p, R, q0))):
        pe, ae, t0, n = [], [], time.perf_counter(), 0
        for x in np.linspace(0.18, 0.40, 6):
            for y in np.linspace(-0.18, 0.18, 5):
                for z in (0.925, 0.97, 1.01, 1.06):
                    for deg in (0, 25):
                        R = tilt(deg)
                        q = solve(np.array([x, y, z]), R, HOME_Q)
                        p, Rq = kin.fk(q)
                        pe.append(np.linalg.norm(p - [x, y, z]) * 1000)
                        ae.append(np.degrees(np.arccos(np.clip((np.trace(Rq.T @ R) - 1) / 2, -1, 1))))
                        n += 1
        ms = (time.perf_counter() - t0) / n * 1000
        pe, ae = np.array(pe), np.array(ae)
        rows.append(name)
        print(f"2) {name}: 목표 {n}개, 위치 오차 가운데 {np.median(pe):.2f} mm · 95% {np.percentile(pe, 95):.1f} mm · "
              f"최대 {pe.max():.1f} mm, 방향 오차 가운데 {np.median(ae):.1f}° · 최대 {ae.max():.1f}°, 한 번 {ms:.1f} ms")
    # 3) 속도 한계: 먼 목표를 1/6 초 한 박자에
    ik.set_q(HOME_Q)
    q0 = ik.q()
    q1 = ik.step(np.array([0.35, 0.15, 0.93]), DOWN, 1 / 6)
    print(f"3) 한 박자(1/6 초) 관절 변화 최대 {np.abs(q1 - q0).max():.3f} rad (한계 {VEL_LIMIT / 6:.3f})")


if __name__ == "__main__":
    _selftest()
