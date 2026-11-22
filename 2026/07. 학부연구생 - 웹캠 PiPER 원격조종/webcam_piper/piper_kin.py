"""PiPER 운동학 — 웹캠 원격조종용 (2026-11-15).

모델은 real_piper/kin.py 와 같은 kin/piper_kin.xml (시뮬레이터 로봇 + 그리퍼에서 메시만 뺀 것) 이다.
좌표도 시뮬과 같다: 로봇 밑동 (0, 0, 0.90), 책상 윗면 z = 0.90, x = 앞, y = 왼쪽, z = 위. 단위 m, rad.

역기구학은 "위치 먼저, 방향은 남는 만큼" 이다.
  1순위: 집게 끝(gripper0_grip_site)을 목표 위치에 정확히 보낸다.
  2순위: 남는 자유도로 집게를 아래로 세우고, 사람 손 방향(yaw)에 맞춘다.
방향까지 꼭 맞추게 하면 밑동 가까이·높은 곳에서 손이 닿지 않는다 (2026-11-15 측정: 방향 고정이면 A4 범위 588점 중 408점,
이 방식이면 588점 모두 5 mm 안, 집게 기울기 대부분 0°, 끝자락에서 최대 35°).

계산은 numpy 만 쓴다 (mujoco 필요 없음). xml 의 몸통 위치·사원수와 관절 축을 읽어 차례로 곱한다.
연구실 PC 의 agx-bus 환경(찬혁님 .venv)에 mujoco 가 없어서 11/15 에 바꿨고, mujoco 계산과 1e-15 m 까지 같다 (check_kin.py).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLE_Z = 0.90
SITE = "gripper0_grip_site"
JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
LINKS = ["robot0_base", "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4",
         "robot0_link5", "robot0_link6"]
# real_piper/config_real.json 의 home_qpos (시뮬 시작 자세와 같음)
HOME_Q = np.array([-0.0372, 0.6342, -0.551, 0.0339, 0.8454, -0.0491])
# 집게가 똑바로 아래를 보고, 집게 끝 사이트의 x 축이 로봇 왼쪽(y)을 향하는 방향 (열 = 사이트 x, y, z 축의 세계 방향)
DOWN = np.array([[0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def grip_orientation(yaw):
    """집게를 아래로 세운 채 세계 z 축으로 yaw (rad) 만큼 돌린 방향."""
    return rot_z(yaw) @ DOWN


def orientation_error(desired, current):
    """robosuite control_utils.orientation_error 와 같은 식 (real_piper/kin.py 와 같음)."""
    return 0.5 * (np.cross(current[:, 0], desired[:, 0]) + np.cross(current[:, 1], desired[:, 1])
                  + np.cross(current[:, 2], desired[:, 2]))


def _dpinv(J, lam):
    return J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(J.shape[0]))


def _vec(el, key, default):
    return np.array([float(v) for v in el.get(key, default).split()])


def _quat_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _axis_angle(axis, a):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _chain(xml, site):
    """worldbody 에서 site 가 달린 몸통까지 내려가는 길: [(몸통 이름, pos, 회전, 관절 이름|None, 축, 범위)], site (pos, 회전)."""
    root = ET.parse(xml).getroot()
    if root.find("compiler") is not None and root.find("compiler").get("angle", "degree") != "radian":
        raise ValueError("compiler angle 은 radian 이어야 함")

    def walk(el, path):
        for s_ in el.findall("site"):
            if s_.get("name") == site:
                return path, s_
        for b in el.findall("body"):
            r = walk(b, path + [b])
            if r:
                return r
        return None

    path, s_el = walk(root.find("worldbody"), [])
    steps = []
    for b in path:
        if any(b.get(k) for k in ("euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError(f"{b.get('name')}: pos·quat 말고 다른 방향 표기는 지원 안 함")
        js = [j for j in b.findall("joint") if j.get("type", "hinge") == "hinge"]
        if len(b.findall("joint")) > len(js) or len(js) > 1 or (js and np.any(_vec(js[0], "pos", "0 0 0"))):
            raise ValueError(f"{b.get('name')}: 원점에 붙은 회전 관절 하나만 지원")
        j = js[0] if js else None
        steps.append((b.get("name"), _vec(b, "pos", "0 0 0"), _quat_mat(_vec(b, "quat", "1 0 0 0")),
                      j.get("name") if j is not None else None,
                      _vec(j, "axis", "0 0 1") if j is not None else None,
                      _vec(j, "range", "0 0") if j is not None else None))
    return steps, (_vec(s_el, "pos", "0 0 0"), _quat_mat(_vec(s_el, "quat", "1 0 0 0")))


class PiperKin:
    def __init__(self, xml=HERE / "kin" / "piper_kin.xml", margin=0.01):
        self.steps, (self.site_pos, self.site_rot) = _chain(str(xml), SITE)
        names = [st[3] for st in self.steps]
        self.jidx = {n: i for i, n in enumerate(JOINTS)}
        if sorted(n for n in names if n) != sorted(JOINTS):
            raise ValueError(f"관절이 {JOINTS} 와 다름: {names}")
        rng = np.array([dict((st[3], st[5]) for st in self.steps if st[3])[n] for n in JOINTS])
        self.lo, self.hi = rng[:, 0] + margin, rng[:, 1] - margin

    def _forward(self, q):
        """(집게 끝 위치, 방향, 링크 원점들 {이름: 위치}, 관절 원점 6×3, 관절 축 6×3)."""
        q = np.asarray(q, dtype=np.float64)
        p, R = np.zeros(3), np.eye(3)
        bodies, jo, ja = {}, np.zeros((6, 3)), np.zeros((6, 3))
        for name, pos, rot, jn, axis, _ in self.steps:
            p = p + R @ pos
            R = R @ rot
            if jn:
                i = self.jidx[jn]
                jo[i], ja[i] = p, R @ (axis / np.linalg.norm(axis))
                R = R @ _axis_angle(axis, q[i])
            bodies[name] = p
        return p + R @ self.site_pos, R @ self.site_rot, bodies, jo, ja

    def fk(self, q):
        """관절 6개 → (집게 끝 위치, 집게 끝 방향 3×3)."""
        pos, mat, *_ = self._forward(q)
        return pos, mat

    def jacobian(self, q):
        """집게 끝의 (위치 자코비안 3×6, 회전 자코비안 3×6) 과 위치·방향."""
        pos, mat, _, jo, ja = self._forward(q)
        Jp = np.cross(ja, pos - jo).T
        return Jp, ja.T.copy(), pos, mat

    def link_points(self, q):
        """그림 그리기용: 밑동 → 각 링크 → 집게 끝 위치들 (N×3)."""
        pos, _, bodies, _, _ = self._forward(q)
        return np.array([bodies[n] for n in LINKS] + [pos])

    def ik(self, pos, mat, q0, iters=40):
        """위치 1순위·방향 2순위 역기구학. 관절 한계에 닿은 관절은 그 반복에서 빼고 다시 푼다."""
        q = np.clip(np.asarray(q0, dtype=np.float64), self.lo, self.hi)
        for _ in range(iters):
            Jp, Jo, cur_p, cur_m = self.jacobian(q)
            ep = pos - cur_p
            eo = orientation_error(mat, cur_m)
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(eo) < 2e-3:
                break
            free = np.ones(6, bool)
            dq = np.zeros(6)
            for _round in range(3):
                Jpf, Jof = Jp * free, Jo * free
                Jpi = _dpinv(Jpf, 0.01)
                dq1 = Jpi @ ep
                N = (np.eye(6) - Jpi @ Jpf) * free
                dq2 = _dpinv(Jof @ N, 0.05) @ (eo - Jof @ dq1)
                dq = (dq1 + N @ dq2) * free
                hit = ((q + dq < self.lo + 1e-6) & (dq < 0)) | ((q + dq > self.hi - 1e-6) & (dq > 0))
                if not hit.any():
                    break
                free &= ~hit
            n = np.linalg.norm(dq)
            if n > 0.25:
                dq *= 0.25 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q
