"""배경 광학 흐름으로 카메라 이동·회전·배율 변화를 손 좌표에서 제거한다."""
import math

import cv2
import numpy as np


class CameraMotionCompensator:
    """연속 프레임의 배경 이동을 누적해 화면 좌표를 안정 좌표로 바꾼다."""

    def __init__(self, cfg):
        self._enabled = cfg.get("enabled", True)
        self._scale = float(cfg.get("analysis_scale", 0.25))
        self._max_corners = int(cfg.get("max_corners", 160))
        self._min_points = int(cfg.get("min_points", 12))
        self._max_shift_ratio = float(cfg.get("max_shift_ratio", 0.08))
        self._max_rotation_deg = float(cfg.get("max_rotation_deg", 8.0))
        self._min_scale = float(cfg.get("min_scale", 0.85))
        self._max_scale = float(cfg.get("max_scale", 1.15))
        self.reset()

    def reset(self):
        self._prev_gray = None
        # 현재 프레임 좌표 -> 최초 프레임의 안정 좌표.
        self._current_to_reference = np.eye(3, dtype=np.float64)

    @property
    def offset_px(self):
        return (float(self._current_to_reference[0, 2]),
                float(self._current_to_reference[1, 2]))

    def update(self, frame, exclude_box=None):
        """새 프레임을 넣고 직전 프레임 대비 카메라 이동량을 누적한다."""
        if not self._enabled or frame is None:
            return self.offset_px
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._scale != 1.0:
            gray = cv2.resize(gray, None, fx=self._scale, fy=self._scale,
                              interpolation=cv2.INTER_AREA)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return self.offset_px

        mask = np.full(self._prev_gray.shape, 255, dtype=np.uint8)
        if exclude_box is not None:
            x1, y1, x2, y2 = (int(round(value * self._scale)) for value in exclude_box)
            pad = 12
            cv2.rectangle(mask, (max(0, x1 - pad), max(0, y1 - pad)),
                          (min(mask.shape[1] - 1, x2 + pad),
                           min(mask.shape[0] - 1, y2 + pad)), 0, -1)

        previous = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=self._max_corners, qualityLevel=0.01,
            minDistance=7, mask=mask, blockSize=7)
        if previous is not None:
            current, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, previous, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            if current is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                previous_valid = previous.reshape(-1, 2)[valid]
                current_valid = current.reshape(-1, 2)[valid]
                if len(previous_valid) >= self._min_points:
                    affine, inliers = cv2.estimateAffinePartial2D(
                        previous_valid, current_valid, method=cv2.RANSAC,
                        ransacReprojThreshold=2.0, maxIters=1000,
                        confidence=0.99, refineIters=10)
                    if (affine is not None and inliers is not None
                            and int(inliers.sum()) >= self._min_points):
                        transform = np.eye(3, dtype=np.float64)
                        transform[:2] = affine
                        # 저해상도에서 구한 평행이동만 원본 픽셀 단위로 환산한다.
                        transform[0, 2] /= self._scale
                        transform[1, 2] /= self._scale
                        if self._is_plausible(transform, frame.shape[:2]):
                            self._current_to_reference = (
                                self._current_to_reference @ np.linalg.inv(transform))
        self._prev_gray = gray
        return self.offset_px

    def compensate(self, point_px):
        """현재 화면 좌표를 카메라 이동이 제거된 안정 좌표로 변환한다."""
        if point_px is None or not self._enabled:
            return point_px
        point = self._current_to_reference @ np.array(
            [float(point_px[0]), float(point_px[1]), 1.0])
        return float(point[0] / point[2]), float(point[1] / point[2])

    def _is_plausible(self, transform, frame_shape):
        a, b = float(transform[0, 0]), float(transform[0, 1])
        scale = math.hypot(a, b)
        rotation_deg = abs(math.degrees(math.atan2(float(transform[1, 0]), a)))
        shift = math.hypot(float(transform[0, 2]), float(transform[1, 2]))
        max_shift = self._max_shift_ratio * min(frame_shape)
        return (self._min_scale <= scale <= self._max_scale
                and rotation_deg <= self._max_rotation_deg
                and shift <= max_shift)
