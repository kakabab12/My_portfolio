"""처음 자세 (11/20): 사람이 손으로 잡아 둔 팔 자세를 저장해 두고, 버튼 하나로 그 자세로 돌아간다.

저장: python save_home.py   (팔을 교시 모드로 손으로 옮겨 둔 채 실행. CAN 을 듣기만 해서 팔은 안 움직임)
돌아가기: robot_server.py 가 도는 동안 teleop 화면에서 B (또는 robot_server 창에서 B).
  따라가기는 꺼지고, 관절마다 같은 시간에 출발·도착하도록 부드럽게(최소 저크) 간다. 가장 많이 도는 관절도 초당 max_rate 를 넘지 않는다.
  가는 중 SPACE 를 누르면 그 자리에 멈춘다. 그리퍼는 그대로 둔다 (물체를 쥐고 있으면 떨어뜨리지 않게).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOME_FILE = os.path.join(HERE, "home_pose.json")


def load_home(path=HOME_FILE):
    """저장한 처음 자세 (관절 6개, rad). 없으면 None."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    q = np.asarray(d["q_rad"], np.float64)
    return q if q.shape == (6,) and np.all(np.isfinite(q)) else None


def path_ok(kin, q0, q1, n=60, below=0.01):
    """q0 → q1 관절 직선 길에서 팔 마디가 책상보다 below(m) 넘게 아래로 가는지 본다 (가상 팔 모델 기준).
    돌려줌 (괜찮음?, 가장 낮은 마디 높이 − 책상 m)."""
    from piper_kin import TABLE_Z
    low = np.inf
    for s in np.linspace(0.0, 1.0, n):
        q = q0 + (q1 - q0) * s
        pts = np.asarray(kin.link_points(q))[1:]              # 0번은 밑동 바닥
        low = min(low, float(pts[:, 2].min()) - TABLE_Z)
    return low > -below, low


class HomeMove:
    """q0 → q1 최소 저크 (출발·도착에서 속도·가속도 0). 가장 많이 도는 관절의 최고 속도가 max_rate(rad/s) 가 되게 시간을 잡는다."""

    def __init__(self, q0, q1, max_rate=0.6, min_time=1.5):
        self.q0, self.q1 = np.asarray(q0, np.float64).copy(), np.asarray(q1, np.float64).copy()
        dq = float(np.max(np.abs(self.q1 - self.q0)))
        self.T = max(min_time, 1.875 * dq / max_rate)          # 최소 저크의 최고 속도 = 1.875 × 거리 / 시간
        self.t = 0.0

    def step(self, dt):
        """dt 만큼 나아간 관절 목표와 도착했는지."""
        self.t += float(dt)
        s = min(self.t / self.T, 1.0)
        m = s ** 3 * (10 - 15 * s + 6 * s * s)
        return self.q0 + (self.q1 - self.q0) * m, s >= 1.0
