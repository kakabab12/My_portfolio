"""정규화된 랜드마크 시퀀스에 적용하는 학습용 증강.

3명만 촬영해도 실전에서 다양한 사람/각도/속도에 덜 흔들리도록,
정규화로 없앨 수 없는 부분(카메라 각도, 좌우잡이 차이, 검출 노이즈)을 랜덤하게 흔들어줌.
"""
import numpy as np

from landmarks import mirror_frame, rotate_frame, add_noise, drop_hand

MAX_ROTATION_RAD = 0.26  # 약 ±15도
HAND_DROPOUT_PROB = 0.20  # 한쪽 손이 인식 안 되는 상황(장애/보조기구/일시적 가림)에 대한 강건성용


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    out = seq
    if np.random.rand() < 0.5:
        out = np.stack([mirror_frame(f) for f in out])

    angle = np.random.uniform(-MAX_ROTATION_RAD, MAX_ROTATION_RAD)
    out = np.stack([rotate_frame(f, angle) for f in out])

    if np.random.rand() < HAND_DROPOUT_PROB:
        side = "left" if np.random.rand() < 0.5 else "right"
        out = np.stack([drop_hand(f, side) for f in out])

    out = np.stack([add_noise(f) for f in out])
    return out
