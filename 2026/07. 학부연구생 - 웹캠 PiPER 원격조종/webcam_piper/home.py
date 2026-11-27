"""처음 자세 (11/20): 사람이 손으로 잡아 둔 팔 자세를 저장해 두고, 버튼 하나로 그 자세로 돌아간다.

저장: python save_home.py   (팔을 교시 모드로 손으로 옮겨 둔 채 실행. CAN 을 듣기만 해서 팔은 안 움직임)
돌아가기: robot_server.py 가 도는 동안 teleop 화면에서 B (또는 robot_server 창에서 B).
  따라가기는 꺼지고, 관절마다 같은 시간에 출발·도착하도록 부드럽게(최소 저크) 간다. 가장 많이 도는 관절도 초당 max_rate 를 넘지 않는다.
  가는 중 SPACE 를 누르면 그 자리에 멈춘다. 그리퍼는 11/23 부터 닫는다 (robot_server --home-grip close 기본, 사용자 요청).
  11/20 처럼 그대로 두려면 --home-grip keep (닫는 것이라 쥐고 있던 물체를 떨어뜨리지는 않는다).

11/23 오후 사용자: "B 를 누르면 로봇팔이 책상에 박고 힘이 빠지네".
  저장한 자세가 가상 팔 모델로 집게 끝 책상 −0.8 cm 였고, 실물은 2번 관절 86.6° (저장 93.0°) 에서 책상에 닿아 힘이 빠졌다 (대기 모드).
  힘이 빠져 책상에 얹힌 집게 끝이 모델로 +0.9 cm 라서, 이 자리 책상은 모델보다 1 cm 쯤 높다. 저장할 때 팔이 위치 제어로 처져 있었다면
  읽은 각도를 그대로 명령할 때 그만큼 더 내려간다. 그래서 B 로 갈 자세는 집게 끝을 모델로 책상 위 HOME_CLEAR(3 cm) 까지 같은 방향으로 들어 올리고
  (safe_target), 가는 길은 1 cm 아래로 내려가지 않게 본다 (이미 그보다 낮은 데서 출발하면 출발 높이 − 3 mm 까지).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOME_FILE = os.path.join(HERE, "home_pose.json")
HOME_CLEAR = 0.03          # B 로 갈 자세의 집게 끝은 가상 팔 모델로 책상 위 이만큼 (m, 11/23 오후)
PATH_FLOOR = 0.01          # 가는 길의 팔 마디·집게 끝은 모델로 책상 위 이만큼 (m)


def load_home(path=HOME_FILE):
    """저장한 처음 자세 (관절 6개, rad). 없으면 None."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    q = np.asarray(d["q_rad"], np.float64)
    return q if q.shape == (6,) and np.all(np.isfinite(q)) else None


def lowest(kin, q):
    """팔 마디·집게 끝 중 가장 낮은 높이 − 책상 (m, 가상 팔 모델)."""
    from piper_kin import TABLE_Z
    return float(np.asarray(kin.link_points(q))[1:, 2].min()) - TABLE_Z     # 0번은 밑동 바닥


def path_ok(kin, q0, q1, n=60, floor=PATH_FLOOR):
    """q0 → q1 관절 직선 길이 책상 위 floor(m) 아래로 내려가는지 본다 (가상 팔 모델 기준).
    출발 자세가 이미 floor 보다 낮으면 출발 높이 − 3 mm 까지만 허락한다 (11/23 오후, 11/20 은 책상 −1 cm 까지 허락해서 책상에 박았다).
    돌려줌 (괜찮음?, 가장 낮은 높이 − 책상 m)."""
    q0, q1 = np.asarray(q0, np.float64), np.asarray(q1, np.float64)
    allowed = min(floor, lowest(kin, q0) - 0.003)
    low = min(lowest(kin, q0 + (q1 - q0) * s) for s in np.linspace(0.0, 1.0, n))
    return low >= allowed, low


def safe_target(kin, q, clear=HOME_CLEAR):
    """저장한 처음 자세의 집게 끝이 모델로 책상 위 clear 보다 낮으면, 같은 방향·같은 앞뒤·옆 자리에서 clear 높이까지 들어 올린 관절 자세.
    돌려줌 (관절 자세, 들어 올린 높이 m)."""
    from piper_kin import TABLE_Z
    q = np.asarray(q, np.float64)
    p, m = kin.fk(q)
    lift = TABLE_Z + clear - float(p[2])
    if lift <= 0:
        return q.copy(), 0.0
    tgt = p + np.array([0.0, 0.0, lift])
    q2 = q.copy()
    for _ in range(6):
        q2 = kin.ik(tgt, m, q2)
    p2, _ = kin.fk(q2)
    if np.linalg.norm(p2 - tgt) > 0.003:
        raise ValueError("처음 자세를 들어 올린 자세를 못 찾음 (%.1f mm 차)" % (np.linalg.norm(p2 - tgt) * 1000))
    return q2, lift


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
