"""PiPER 시뮬 장면: 책상·정육면체·놓을 판 (MuJoCo 만, 2026-11-27).

로봇 모델은 real_piper/kin/robot_full.xml (piper_sim 시뮬 로봇과 같은 모양, 받침대 없음, 밑동 (0, 0, 0.90)).
관절은 파이썬에서 PD + 중력 보상으로 목표 관절을 따라가게 하고 (모델의 관절 모터는 힘 명령),
그리퍼 두 손가락은 모델에 있는 위치 모터 (한쪽 0 ~ 3.5 cm, 둘이면 7 cm) 를 쓴다.
역기구학은 webcam_piper/piper_kin.py (numpy, mujoco 계산과 1e-15 m 까지 같음, check_kin.py) 를 그대로 쓴다.

로봇 바깥 세상(책상·물체)은 MjSpec 으로 붙인다. 판은 얇은 색 종이처럼 보이기만 하고 부딪히지 않는다 (높이 1 mm).
"""
import math
import os

import numpy as np

import common
from common import TABLE_Z
from piper_kin import HOME_Q, PiperKin

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

ROBOT_XML = common.ROOT / "real_piper" / "kin" / "robot_full.xml"
JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
FINGERS = ["gripper0_finger_joint1", "gripper0_finger_joint2"]
SITE = "gripper0_grip_site"
KP = np.array([120.0, 120.0, 120.0, 30.0, 30.0, 20.0])
KD = np.array([8.0, 8.0, 8.0, 2.0, 2.0, 1.0])
GRIP_OPEN = 0.035


def _mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(9))
    return q


def look_at(pos, target, up=(0, 0, 1)):
    """MuJoCo 카메라 방향 (사원수): -z 가 target 을 보고 y 가 위."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return _mat_to_quat(np.c_[x, y, z])


def cv_camera(calib, task):
    """webcam_piper TableCalib (책상 → 카메라, OpenCV) → 시뮬 카메라 자리 (로봇 좌표 위치, 사원수, 세로 시야각, 크기)."""
    R, t = np.asarray(calib.R), np.asarray(calib.t)
    pos = -R.T @ t + np.asarray(task["desk_origin_robot"])
    axes_cv = R.T                                   # 열 = 카메라 x(오른쪽)·y(아래)·z(앞) 의 세계 방향
    mj = np.c_[axes_cv[:, 0], -axes_cv[:, 1], -axes_cv[:, 2]]
    fovy = math.degrees(2 * math.atan(calib.h / (2 * calib.f)))
    return pos, _mat_to_quat(mj), fovy, (calib.w, calib.h)


class PiperSim:
    def __init__(self, task, cams=None, physics_dt=0.002, ik="numpy", offscreen=(1920, 1080)):
        """cams: {이름: (위치, 사원수, 세로 시야각)} 더 붙일 카메라. 'front' 는 늘 있다.
        ik: "numpy" (webcam_piper piper_kin, 위치 먼저) 또는 "placo" (placo_ik.PlacoIK, QP, point_policy/.venv 필요)."""
        self.task = task
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
        # robot_full.xml 에는 각도 단위가 없어 MuJoCo 가 '도'로 읽는다 (관절 한계 ±2.6 → ±2.6°). 라디안으로 (11/27)
        spec.compiler.degree = False
        spec.option.timestep = physics_dt
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.visual.global_.offwidth = int(offscreen[0])
        spec.visual.global_.offheight = int(offscreen[1])
        wb = spec.worldbody
        wb.add_light(pos=[0.3, 0.0, 2.2], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
        wb.add_light(pos=[1.2, -0.8, 1.8], dir=[-0.6, 0.4, -0.6], diffuse=[0.35, 0.35, 0.35], castshadow=False)
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.55, 0.55, 0.58, 1], group=2)
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.25, 0.0, TABLE_Z / 2],
                    size=[0.45, 0.55, TABLE_Z / 2], rgba=[0.80, 0.76, 0.70, 1], friction=[1.0, 0.005, 0.0001], group=2)
        # A4 종이 (사람 책상 좌표 원점) — 보이기만
        o = np.asarray(task["desk_origin_robot"])
        wb.add_geom(name="a4", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[o[0], o[1], TABLE_Z + 0.0002],
                    size=[0.105, 0.1485, 0.0002], rgba=[0.97, 0.97, 0.97, 1], contype=0, conaffinity=0, group=2)
        h = task["cube_size"] / 2
        self.cube_h = h
        cube = wb.add_body(name="cube", pos=[0.3, 0.1, TABLE_Z + h])
        cube.add_freejoint(name="cube_free")
        cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h], rgba=[0.85, 0.2, 0.15, 1],
                      mass=0.05, friction=[1.2, 0.005, 0.0001], condim=4, group=2)
        ps = task["pad_size"] / 2
        pad = wb.add_body(name="pad", pos=[0.3, -0.1, TABLE_Z])
        pad.add_geom(name="pad_geom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.0005], size=[ps, ps, 0.0005],
                     rgba=[0.15, 0.35, 0.85, 1], contype=0, conaffinity=0, group=2)
        wb.add_camera(name="front", pos=[0.95, -0.45, 1.42], quat=look_at([0.95, -0.45, 1.42], [0.27, 0.0, 0.93]),
                      fovy=50)
        wb.add_camera(name="close", pos=[0.74, -0.42, 1.36], quat=look_at([0.74, -0.42, 1.36], [0.27, 0.0, 0.98]),
                      fovy=46)
        wb.add_camera(name="top", pos=[0.28, 0.0, 1.75], quat=look_at([0.28, 0.0, 1.75], [0.28, 0.0, 0.9], up=(1, 0, 0)),
                      fovy=45)
        self.cam_sizes = {}
        for name, (pos, quat, fovy, *size) in (cams or {}).items():
            wb.add_camera(name=name, pos=list(pos), quat=list(quat), fovy=float(fovy))
            if size:
                self.cam_sizes[name] = tuple(size[0])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jq = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.jd = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        self.fq = np.array([m.joint(j).qposadr[0] for j in FINGERS])
        self.act_j = np.array([m.actuator(f"robot0_torq_j{i}").id for i in range(1, 7)])
        self.act_f = np.array([m.actuator(f"gripper0_gripper_finger_joint{i}").id for i in (1, 2)])
        self.lo = np.array([m.actuator_ctrlrange[a][0] for a in self.act_j])
        self.hi = np.array([m.actuator_ctrlrange[a][1] for a in self.act_j])
        self.site = m.site(SITE).id
        self.cube_body = m.body("cube").id
        self.pad_body = m.body("pad").id
        self.cube_qadr = m.joint("cube_free").qposadr[0]
        self.kin = PiperKin()
        self.ik_name = ik
        if ik == "placo":
            from placo_ik import PlacoIK
            self.solver = PlacoIK()
        else:
            self.solver = self.kin
        self.q_target = HOME_Q.copy()
        self.closed = False
        self.renderers = {}
        self.t = 0.0

    # ───────── 장면 ─────────
    def reset(self, cube_xy, pad_xy, q0=None, settle=0.3):
        mujoco.mj_resetData(self.model, self.data)
        q0 = HOME_Q if q0 is None else np.asarray(q0)
        self.data.qpos[self.jq] = q0
        self.data.qpos[self.fq] = [GRIP_OPEN, -GRIP_OPEN]
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = [cube_xy[0], cube_xy[1], TABLE_Z + self.cube_h]
        self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.model.body_pos[self.pad_body] = [pad_xy[0], pad_xy[1], TABLE_Z]
        self.q_target = q0.copy()
        self.closed = False
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        self.run(settle)

    def ik(self, pos, R, q_seed=None):
        q = self.solver.ik(np.asarray(pos), np.asarray(R), self.q_target if q_seed is None else q_seed, iters=60)
        p, Rm = self.kin.fk(q)
        return q, float(np.linalg.norm(p - pos))

    def start_q(self, pos=None, R=None):
        from piper_kin import DOWN
        pos = self.task["start_pos"] if pos is None else pos
        q, err = self.ik(np.asarray(pos), DOWN if R is None else R, HOME_Q)
        return q

    # ───────── 움직이기 ─────────
    def _ctrl(self):
        d = self.data
        q, qd = d.qpos[self.jq], d.qvel[self.jd]
        tau = KP * (self.q_cmd - q) - KD * qd + d.qfrc_bias[self.jd]
        d.ctrl[self.act_j] = np.clip(tau, self.lo, self.hi)
        g = 0.0 if self.closed else GRIP_OPEN
        d.ctrl[self.act_f] = [g, -g]

    def run(self, duration):
        """지금 목표 그대로 duration 초."""
        self.q_cmd = self.q_target.copy()
        for _ in range(max(1, int(round(duration / self.model.opt.timestep)))):
            self._ctrl()
            mujoco.mj_step(self.model, self.data)
        self.t += duration

    def move(self, q_goal, closed, duration):
        """관절 목표를 duration 초 동안 곧게 옮기며 따라간다 (정책 한 박자 = 1/6 초)."""
        q0 = self.q_target.copy()
        q_goal = np.asarray(q_goal, np.float64)
        n = max(1, int(round(duration / self.model.opt.timestep)))
        self.closed = bool(closed)
        for i in range(n):
            s = (i + 1) / n
            self.q_cmd = q0 + s * (q_goal - q0)
            self._ctrl()
            mujoco.mj_step(self.model, self.data)
        self.q_target = q_goal.copy()
        self.t += duration

    # ───────── 읽기 ─────────
    def ee_pose(self):
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def joints(self):
        return self.data.qpos[self.jq].copy()

    def grip_width(self):
        return float(self.data.qpos[self.fq[0]] - self.data.qpos[self.fq[1]])

    def object_points(self):
        """설정의 물체 점 (K×3, 로봇 좌표): 몸통 위치 + 몸통 방향 × 붙은 자리."""
        out = []
        for op in self.task["object_points"]:
            b = self.model.body(op["body"]).id
            out.append(self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ np.asarray(op["offset"]))
        return np.array(out)

    def cube_pos(self):
        return self.data.xpos[self.cube_body].copy()

    def pad_pos(self):
        return self.model.body_pos[self.pad_body].copy()

    def success(self, margin=0.0):
        """정육면체 가운데가 판 위(가장자리 margin 안쪽)에 있고, 책상에 내려앉았고, 집게가 놓았으면."""
        c, p = self.cube_pos(), self.pad_pos()
        half = self.task["pad_size"] / 2 - margin
        on_pad = abs(c[0] - p[0]) < half and abs(c[1] - p[1]) < half
        down = c[2] < TABLE_Z + self.cube_h + 0.01
        return bool(on_pad and down and not self.closed)

    def cube_lifted(self):
        return bool(self.cube_pos()[2] > TABLE_Z + self.cube_h + 0.02)

    def render(self, cam="front", w=640, h=480):
        key = (w, h)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.model, h, w)
        r = self.renderers[key]
        if not hasattr(self, "_vopt"):
            # 로봇 충돌 모양(무리 0)과 사이트(집게 끝 표시 등)는 그림에서 숨긴다
            self._vopt = mujoco.MjvOption()
            self._vopt.geomgroup[0] = 0
            self._vopt.sitegroup[:] = 0
        r.update_scene(self.data, camera=cam, scene_option=self._vopt)
        return r.render()

    def close(self):
        for r in self.renderers.values():
            r.close()
        self.renderers = {}


def sample_layout(task, rng):
    """정육면체·판 자리 (로봇 좌표 xy). 둘 사이 min_sep 이상."""
    for _ in range(1000):
        c = np.array([rng.uniform(*task["region_x"]), rng.uniform(*task["region_y"])])
        p = np.array([rng.uniform(*task["region_x"]), rng.uniform(*task["region_y"])])
        if np.linalg.norm(c - p) >= task["min_sep"]:
            return c, p
    raise RuntimeError("자리를 못 뽑음")
