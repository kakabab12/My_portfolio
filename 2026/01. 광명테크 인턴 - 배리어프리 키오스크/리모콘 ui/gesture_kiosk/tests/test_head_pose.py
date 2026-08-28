"""머리 3차원 자세(HeadPose) 단위 테스트 — 2026-08-28 신설.

카메라도 MediaPipe도 없이 순수 수학만 검증한다.

[왜 이 테스트가 필요한가]
처음 구현할 때 Z-Y-X 순서의 오일러각 추출식을 그대로 가져다 썼는데, 실제
합성 순서는 Y-X-Z(yaw-pitch-roll)라 **yaw와 pitch가 서로 뒤바뀌어 나왔다.**
고개를 좌우로 돌렸는데 커서가 위아래로 움직이는 꼴이 됐을 것이다.
합성 -> 분해 왕복으로 검증하니 바로 드러났다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.face_estimator import (   # noqa: E402
    HeadPose, _extract_head_pose, _rotation_matrix_to_euler_deg,
)


def _compose(yaw_deg, pitch_deg, roll_deg):
    """(yaw, pitch, roll) -> 3x3 회전행렬. Y-X-Z 순서로 합성한다.

    _rotation_matrix_to_euler_deg 가 되짚어야 할 바로 그 순서다.
    """
    y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0],
                   [0, math.cos(p), -math.sin(p)],
                   [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0],
                   [math.sin(r), math.cos(r), 0],
                   [0, 0, 1]])
    return ry @ rx @ rz


class EulerRoundTripTest(unittest.TestCase):
    """합성 -> 분해 왕복이 정확한가 — 이 클래스가 yaw/pitch 뒤바뀜을 잡아냈다."""

    CASES = [
        (0, 0, 0),
        (30, 0, 0), (-30, 0, 0),        # 좌우만
        (0, 20, 0), (0, -20, 0),        # 상하만
        (0, 0, 15), (0, 0, -15),        # 갸웃만
        (25, -10, 8), (-40, 15, -12),   # 섞인 것
        (60, -35, 20),                  # 큰 각도
    ]

    def test_round_trip_is_exact(self):
        for yaw, pitch, roll in self.CASES:
            with self.subTest(yaw=yaw, pitch=pitch, roll=roll):
                got = _rotation_matrix_to_euler_deg(_compose(yaw, pitch, roll))
                for name, g, e in zip(("yaw", "pitch", "roll"), got, (yaw, pitch, roll)):
                    self.assertAlmostEqual(g, e, places=6, msg=f"{name} 불일치")

    def test_yaw_and_pitch_are_not_swapped(self):
        """★회귀 검사 — 순수 yaw 회전이 pitch로 새면 안 된다.

        처음 버그가 정확히 이 형태였다: 좌우로 30° 돌렸는데 pitch=30 이 나왔다.
        """
        yaw, pitch, roll = _rotation_matrix_to_euler_deg(_compose(30, 0, 0))
        self.assertAlmostEqual(yaw, 30.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6, msg="yaw가 pitch로 샜다")

        yaw, pitch, roll = _rotation_matrix_to_euler_deg(_compose(0, 20, 0))
        self.assertAlmostEqual(pitch, 20.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6, msg="pitch가 yaw로 샜다")

    def test_sign_convention(self):
        """부호 약속이 문서와 맞는가 — 뒤집히면 커서가 반대로 움직인다."""
        self.assertGreater(_rotation_matrix_to_euler_deg(_compose(30, 0, 0))[0], 0)
        self.assertLess(_rotation_matrix_to_euler_deg(_compose(-30, 0, 0))[0], 0)
        self.assertGreater(_rotation_matrix_to_euler_deg(_compose(0, 20, 0))[1], 0)
        self.assertLess(_rotation_matrix_to_euler_deg(_compose(0, -20, 0))[1], 0)
        self.assertGreater(_rotation_matrix_to_euler_deg(_compose(0, 0, 15))[2], 0)


class GimbalLockTest(unittest.TestCase):
    """pitch가 ±90°에 붙는 구간에서 값이 튀지 않는가.

    키오스크에서 고개를 90° 젖힐 일은 없지만, 각도가 NaN이나 무한대로
    튀면 커서가 화면 밖으로 순간이동한다. 그 사고를 막는 방어선이다.
    """

    def test_no_crash_and_finite_at_lock(self):
        for pitch in (90.0, -90.0, 89.9999, -89.9999):
            with self.subTest(pitch=pitch):
                got = _rotation_matrix_to_euler_deg(_compose(0, pitch, 0))
                for v in got:
                    self.assertFalse(math.isnan(v), "NaN이 나왔다")
                    self.assertFalse(math.isinf(v), "무한대가 나왔다")

    def test_pitch_is_clamped_at_lock(self):
        self.assertAlmostEqual(_rotation_matrix_to_euler_deg(_compose(0, 90, 0))[1], 90.0, places=3)
        self.assertAlmostEqual(_rotation_matrix_to_euler_deg(_compose(0, -90, 0))[1], -90.0, places=3)

    def test_numerical_overshoot_does_not_crash(self):
        """수치 오차로 |sin(pitch)|가 1을 아주 살짝 넘어도 asin이 죽으면 안 된다."""
        bad = np.eye(3)
        bad[1][2] = -1.0000001      # sin(pitch) = 1.0000001
        got = _rotation_matrix_to_euler_deg(bad)
        self.assertFalse(any(math.isnan(v) for v in got))


class _FakeResult:
    def __init__(self, matrices):
        self.facial_transformation_matrixes = matrices


class ExtractHeadPoseTest(unittest.TestCase):
    """MediaPipe 결과에서 뽑아내는 부분 — 없거나 깨져도 죽지 않아야 한다."""

    @staticmethod
    def _matrix(yaw, pitch, roll, tx=1.0, ty=2.0, tz=-30.0):
        m = np.eye(4)
        m[:3, :3] = _compose(yaw, pitch, roll)
        m[0][3], m[1][3], m[2][3] = tx, ty, tz
        return m

    def test_extracts_angles_and_translation(self):
        pose = _extract_head_pose(_FakeResult([self._matrix(25, -10, 8)]), 0)
        self.assertIsInstance(pose, HeadPose)
        self.assertAlmostEqual(pose.yaw_deg, 25.0, places=5)
        self.assertAlmostEqual(pose.pitch_deg, -10.0, places=5)
        self.assertAlmostEqual(pose.roll_deg, 8.0, places=5)
        self.assertAlmostEqual(pose.tz, -30.0, places=5)

    def test_missing_matrices_returns_none(self):
        """★가장 중요 — MediaPipe 옵션이 꺼져 있거나 버전이 달라 행렬이 안 와도
        None만 돌려주고 넘어가야 한다. 부가 정보 하나 때문에 추론이 죽으면 안 된다."""
        class NoAttr:
            pass
        self.assertIsNone(_extract_head_pose(NoAttr(), 0))
        self.assertIsNone(_extract_head_pose(_FakeResult([]), 0))
        self.assertIsNone(_extract_head_pose(_FakeResult(None), 0))

    def test_index_out_of_range_returns_none(self):
        self.assertIsNone(_extract_head_pose(_FakeResult([self._matrix(0, 0, 0)]), 5))

    def test_malformed_matrix_returns_none_not_raises(self):
        self.assertIsNone(_extract_head_pose(_FakeResult([[1, 2, 3]]), 0))


if __name__ == "__main__":
    unittest.main()
