"""각도 기반 커서 매핑(head_pose_mapping) 테스트 — 2026-08-28 신설.

[무엇을 검증하나]
지금까지 커서는 **화면에 투영된 2D 랜드마크 위치**로 움직였다. 코처럼 얼굴에서
튀어나온 점은 고개를 돌리면 원근 때문에 비선형으로 움직여서, 좌우로만 돌려도
세로가 활처럼 휘었다(ARC_COMPENSATION이 2차식으로 사후 보정하던 문제).

머리의 **회전각**을 직접 쓰면 투영 왜곡이 원리적으로 없다. 이 테스트가 확인하는
핵심은 그것이다 — **좌우로만 돌렸을 때 세로가 안 휘는가.**

카메라도 MediaPipe도 필요 없다. 가짜 자세 값을 넣어 매핑 계산만 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.face_estimator import HeadPose      # noqa: E402
from src.postprocess.head_tracker import (             # noqa: E402
    HEAD_POSE_MAX_ANGLE_DEG, _CursorMapper,
)

FRAME_DT_SEC = 1.0 / 30.0


def _mapper(**overrides):
    kwargs = dict(
        calibration_window_sec=0.3, sensitivity_x=1.0, sensitivity_y=1.0,
        smoothing_alpha=1.0,          # 평활 없음 — 매핑 자체만 본다
        distance_smoothing_alpha=0.08, max_offset_ratio=0.5,
        head_pose_mapping=True,
    )
    kwargs.update(overrides)
    return _CursorMapper(**kwargs)


def _pose(yaw=0.0, pitch=0.0, roll=0.0):
    return HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                    tx=0.0, ty=0.0, tz=-35.0)


# 각도 경로는 눈 좌표를 안 쓰지만 update() 시그니처상 필요하다
EYES = ((100.0, 200.0), (160.0, 200.0))


def _settle(mapper, pose, frames, start_sec=0.0):
    """같은 자세를 여러 프레임 넣고 마지막 커서 좌표를 돌려준다."""
    t = start_sec
    out = (None, None)
    for _ in range(frames):
        t += FRAME_DT_SEC
        out = mapper.update((130.0, 200.0), *EYES, t, head_pose=pose)
    return out, t


class NoCurvatureTest(unittest.TestCase):
    """★핵심 — 좌우로만 돌리면 세로가 안 움직여야 한다.

    이 방식을 도입하는 이유 자체다. 2D 랜드마크 방식에서는 여기서 세로가
    포물선으로 휘어서 ARC_COMPENSATION 상수를 카메라 배치마다 다시 재야 했다.
    """

    def test_pure_yaw_does_not_move_cursor_vertically(self):
        mapper = _mapper()
        _settle(mapper, _pose(), 20)          # 정면으로 캘리브레이션

        ys = []
        for yaw in range(-40, 41, 5):         # 좌우로만 크게 훑는다
            (x, y), _ = _settle(mapper, _pose(yaw=yaw), 3)
            self.assertIsNotNone(y)
            ys.append(y)

        # 세로가 전혀 안 움직여야 한다 (평활도 껐으므로 완전히 0이어야 정상)
        self.assertAlmostEqual(max(ys), min(ys), places=9,
                               msg="좌우로만 돌렸는데 세로가 움직였다 — 곡률이 남아 있다")

    def test_horizontal_actually_moves(self):
        """세로가 안 움직인다는 검사가 헛돌지 않으려면, 가로는 실제로 움직여야 한다."""
        mapper = _mapper()
        _settle(mapper, _pose(), 20)
        (x_left, _), _ = _settle(mapper, _pose(yaw=-30), 3)
        (x_right, _), _ = _settle(mapper, _pose(yaw=30), 3)
        self.assertLess(x_left, 0.45)
        self.assertGreater(x_right, 0.55)

    def test_pure_pitch_does_not_move_cursor_horizontally(self):
        mapper = _mapper()
        _settle(mapper, _pose(), 20)
        xs = []
        for pitch in range(-20, 21, 5):
            (x, y), _ = _settle(mapper, _pose(pitch=pitch), 3)
            xs.append(x)
        self.assertAlmostEqual(max(xs), min(xs), places=9,
                               msg="위아래로만 움직였는데 가로가 움직였다")


class DirectionTest(unittest.TestCase):
    """부호가 뒤집히면 커서가 반대로 간다 — 실기에서 바로 드러나지만 미리 고정한다."""

    def test_look_up_moves_cursor_up(self):
        mapper = _mapper()
        _settle(mapper, _pose(), 20)
        (_, y_up), _ = _settle(mapper, _pose(pitch=20), 3)
        # 화면 좌표는 위쪽이 0 — 위를 보면 y가 작아져야 한다
        self.assertLess(y_up, 0.5, "위를 봤는데 커서가 아래로 갔다")

    def test_look_down_moves_cursor_down(self):
        mapper = _mapper()
        _settle(mapper, _pose(), 20)
        (_, y_down), _ = _settle(mapper, _pose(pitch=-20), 3)
        self.assertGreater(y_down, 0.5, "아래를 봤는데 커서가 위로 갔다")

    def test_look_right_moves_cursor_right(self):
        mapper = _mapper()
        _settle(mapper, _pose(), 20)
        (x_right, _), _ = _settle(mapper, _pose(yaw=25), 3)
        self.assertGreater(x_right, 0.5)


class CalibrationTest(unittest.TestCase):
    """기울어진 카메라 대응 — 이 방식을 쓰는 두 번째 이유."""

    def test_tilted_camera_is_absorbed_by_calibration(self):
        """카메라가 비스듬히 달려 있어 평상시 각도가 0이 아니어도,
        그 자세를 중심으로 캘리브레이션하면 커서는 중앙에서 시작해야 한다.

        연구실 카메라(아래에서 위를 봄)와 키오스크 카메라(정면)에서 곡률 보정
        상수를 각각 다시 재야 했던 문제가 여기서 사라진다.
        """
        resting = _pose(yaw=-12.0, pitch=18.0)   # 카메라가 비뚤게 달린 상황
        mapper = _mapper()
        (x, y), _ = _settle(mapper, resting, 20)
        self.assertAlmostEqual(x, 0.5, places=6)
        self.assertAlmostEqual(y, 0.5, places=6)

    def test_returns_none_while_calibrating(self):
        mapper = _mapper()
        got = mapper.update((130.0, 200.0), *EYES, FRAME_DT_SEC, head_pose=_pose())
        self.assertEqual(got, (None, None))


class SafetyTest(unittest.TestCase):
    def test_extreme_angle_is_clamped_not_infinite(self):
        """tan은 90°에서 발산한다 — 잘리지 않으면 커서가 화면 밖으로 순간이동한다."""
        mapper = _mapper()
        _settle(mapper, _pose(), 20)
        (x, y), _ = _settle(mapper, _pose(yaw=89.9, pitch=89.9), 3)
        for v in (x, y):
            self.assertFalse(math.isnan(v))
            self.assertFalse(math.isinf(v))
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_beyond_limit_saturates(self):
        """한도를 넘어 더 돌려도 결과가 더 커지지 않아야 한다."""
        mapper = _mapper()
        _settle(mapper, _pose(), 20)
        (x1, _), _ = _settle(mapper, _pose(yaw=HEAD_POSE_MAX_ANGLE_DEG + 5), 3)
        (x2, _), _ = _settle(mapper, _pose(yaw=HEAD_POSE_MAX_ANGLE_DEG + 25), 3)
        self.assertAlmostEqual(x1, x2, places=9)

    def test_missing_pose_falls_back_to_landmarks(self):
        """★자세 정보가 안 와도 커서가 멈추면 안 된다.

        MediaPipe 옵션이 꺼졌거나 모델이 바뀌어 행렬이 안 오는 상황에서,
        각도 매핑을 켜 뒀다는 이유로 트래커가 죽거나 멈추면 최악이다.
        기존 랜드마크 경로로 조용히 되돌아가야 한다.
        """
        mapper = _mapper()
        t = 0.0
        for _ in range(20):
            t += FRAME_DT_SEC
            mapper.update((130.0, 200.0), *EYES, t, head_pose=None)
        t += FRAME_DT_SEC
        x, y = mapper.update((130.0, 200.0), *EYES, t, head_pose=None)
        self.assertIsNotNone(x, "자세가 없을 때 커서가 확정되지 않았다")
        self.assertIsNotNone(y)


class DefaultsUnchangedTest(unittest.TestCase):
    """★기존 트래커 보호 — 옵션을 안 켜면 각도 정보가 와도 무시해야 한다."""

    def test_disabled_by_default(self):
        mapper = _CursorMapper(
            calibration_window_sec=0.3, sensitivity_x=1.0, sensitivity_y=1.0,
            smoothing_alpha=1.0, distance_smoothing_alpha=0.08, max_offset_ratio=0.5)
        self.assertFalse(mapper._head_pose_mapping)

    def test_pose_is_ignored_when_disabled(self):
        """같은 입력에 자세를 줬을 때와 안 줬을 때 결과가 완전히 같아야 한다."""
        def run(pose):
            m = _CursorMapper(
                calibration_window_sec=0.3, sensitivity_x=1.0, sensitivity_y=1.0,
                smoothing_alpha=0.5, distance_smoothing_alpha=0.08,
                max_offset_ratio=0.5, face_local=True)
            t = 0.0
            out = None
            for i in range(30):
                t += FRAME_DT_SEC
                out = m.update((130.0 + i * 0.3, 200.0), *EYES, t, head_pose=pose)
            return out

        self.assertEqual(run(None), run(_pose(yaw=35, pitch=-20)))


if __name__ == "__main__":
    unittest.main()
