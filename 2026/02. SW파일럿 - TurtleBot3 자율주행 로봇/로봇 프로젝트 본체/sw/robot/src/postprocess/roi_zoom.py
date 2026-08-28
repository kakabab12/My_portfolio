"""postprocess 모듈 — 손 주변 디지털 줌(ROI crop). 거리가 멀어져도 모델이 보는
손 크기를 유지해 원거리 인식률을 지킨다.

gesture_kiosk(realtime_loop.resolve_roi_box)는 별도 포즈 모델의 머리 위치를
앵커로 썼지만, 여기서는 조작자 1인을 가정하므로 **손 자신의 마지막 박스**를
앵커로 써서 모델을 하나 더 두지 않고 단순화한다.
"""
import math

import numpy as np


def resolve_roi_box(prev_box, last_hand_box, frame_width_px, frame_height_px, roi_cfg):
    """다음 프레임 추론에 쓸 crop 박스 -> (x1, y1, x2, y2) | None(전체 프레임 사용).

    prev_box: 직전에 실제로 쓴 crop 박스 — 히스테리시스 유지용(None이면 처음).
    last_hand_box: 손의 마지막 관측 박스 (x1, y1, x2, y2) px | None(손 소실 —
    다음 프레임은 전체 프레임을 봐야 재검출이 가능하다).
    """
    if last_hand_box is None:
        return None
    hx1, hy1, hx2, hy2 = last_hand_box
    hand_side_px = max(hx2 - hx1, hy2 - hy1)
    if hand_side_px <= 0.0:
        return None

    half_px = max(hand_side_px * roi_cfg.get("pad_ratio", 2.5) / 2.0,
                  roi_cfg.get("min_side_px", 320) / 2.0)
    if 2.0 * half_px >= min(frame_width_px, frame_height_px):
        return None   # 근거리 — crop이 프레임을 거의 다 덮음: 전체 프레임으로 충분

    center_x = (hx1 + hx2) / 2.0
    center_y = (hy1 + hy2) / 2.0
    side_px = int(2.0 * half_px)
    x1 = int(max(0.0, min(center_x - half_px, frame_width_px - side_px)))
    y1 = int(max(0.0, min(center_y - half_px, frame_height_px - side_px)))
    target = (x1, y1, x1 + side_px, y1 + side_px)

    if prev_box is not None:
        prev_center = ((prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0)
        prev_side_px = float(max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]))
        target_center = ((target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0)
        if (math.dist(prev_center, target_center)
                <= roi_cfg.get("move_ratio", 0.15) * prev_side_px
                and abs(side_px - prev_side_px)
                <= roi_cfg.get("resize_ratio", 0.2) * prev_side_px):
            return prev_box   # 문턱 미달 — 창 유지(추적 안정)
    return target


def offset_landmarks_xy(landmarks, x_offset_px, y_offset_px):
    """crop 안에서 나온 랜드마크(화면 좌표)를 원본 프레임 좌표로 되돌린다.

    world_landmarks(미터 단위, 손 중심 원점)는 crop과 무관해 그대로 쓰면 된다 —
    이 함수는 screen-space landmarks(x_px, y_px, z_px)에만 적용한다.
    """
    shifted = landmarks.copy()
    shifted[:, 0] += x_offset_px
    shifted[:, 1] += y_offset_px
    return shifted


def crop_frame(frame, box):
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(frame[y1:y2, x1:x2])
