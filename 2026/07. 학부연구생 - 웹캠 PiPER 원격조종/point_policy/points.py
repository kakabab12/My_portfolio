"""논문 식 바꾸기: 사람 손 점 → 로봇 자세 → 로봇 점, 그리고 거꾸로 (numpy 만, 2026-11-27).

Point Policy (arXiv 2502.20391) 3장:
  - 로봇 위치  = 엄지 끝과 검지 끝의 가운데
  - 로봇 방향  = 첫 장면의 로봇 방향에, 첫 장면 손 → 지금 손의 회전(강체 변환)을 곱한 것 (식 2)
  - 집게      = 엄지 끝 ↔ 검지 끝이 7 cm 보다 가까우면 닫힘, 아니면 열림
  - 로봇 점    = 로봇 자세에 붙은 점 N 개 (식 3). 정책은 이 점들의 앞날 궤적을 배우고,
                실행할 때는 점들에 강체를 맞춰(Kabsch) 6 자유도 자세를 되찾는다.
손의 회전은 손바닥 다섯 점(손목·네 손가락 뿌리)으로 잰다. 쥐어도 거의 안 움직이는 점이라 (webcam_piper 11/18 과 같음).
"""
import numpy as np

import common  # noqa: F401  (webcam_piper 를 import 길에 넣음)
from geometry import INDEX_TIP, PALM, THUMB_TIP, kabsch
from piper_kin import DOWN

# 로봇 점: 집게 끝 사이트 좌표 (x = 집게가 닫히는 줄, z = 손목 → 집게 끝, m).
# 가운데 하나, 두 손가락 자리, 손목 쪽 하나, 옆 하나 (세 축이 다 드러나야 회전을 하나로 되찾음).
ROBOT_OFFSETS = np.array([[0.0, 0.0, 0.0],
                          [0.035, 0.0, 0.0],
                          [-0.035, 0.0, 0.0],
                          [0.0, 0.0, -0.08],
                          [0.0, 0.04, -0.04]])
N_ROBOT = len(ROBOT_OFFSETS)


def robot_points(pos, R):
    """집게 끝 위치 (3,) 와 방향 (3×3, 열 = 사이트 x·y·z 축) → 로봇 점 (N×3)."""
    return np.asarray(pos)[None, :] + ROBOT_OFFSETS @ np.asarray(R).T


def pose_from_points(P):
    """로봇 점 (N×3, 정책이 낸 것이라 강체가 아닐 수 있음) → 가장 잘 맞는 (위치, 방향) 와 맞춘 뒤 남은 차 (m)."""
    P = np.asarray(P, np.float64)
    c_off = ROBOT_OFFSETS.mean(axis=0)
    c_p = P.mean(axis=0)
    R = kabsch(ROBOT_OFFSETS - c_off, P - c_p)
    pos = c_p - R @ c_off
    res = float(np.sqrt(np.mean(np.sum((robot_points(pos, R) - P) ** 2, axis=1))))
    return pos, R, res


def desk_to_robot(p_desk, task):
    """사람 책상 좌표 (원점 A4 가운데, x 앞, y 사람 왼쪽, z 위) → 시뮬 로봇 좌표. 축은 같고 원점만 옮긴다."""
    return np.asarray(p_desk, np.float64) + np.asarray(task["desk_origin_robot"])


class HandToRobot:
    """한 시범 안에서 손 21점 (로봇 좌표, 시간 순) → 로봇 자세·집게. 첫 장면 손바닥을 기준으로 잡는다."""

    def __init__(self, task, R0=DOWN):
        self.close_m = float(task.get("grip_close_m", 0.07))
        self.R0 = np.asarray(R0, np.float64)
        self.palm0 = None

    def __call__(self, hand21):
        h = np.asarray(hand21, np.float64)
        thumb, index = h[THUMB_TIP], h[INDEX_TIP]
        pos = 0.5 * (thumb + index)
        gap = float(np.linalg.norm(thumb - index))
        palm = h[list(PALM)]
        if self.palm0 is None:
            self.palm0 = palm.copy()
        dR = kabsch(self.palm0 - self.palm0.mean(axis=0), palm - palm.mean(axis=0))
        R = dR @ self.R0
        closed = gap < self.close_m
        return pos, R, closed, gap


def rot_angle_deg(Ra, Rb):
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
