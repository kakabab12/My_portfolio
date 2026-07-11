"""MediaPipe HolisticLandmarker(Tasks API) 래퍼 + 랜드마크 시퀀스 전처리 유틸.

한 프레임 -> 고정 길이 raw 벡터(FRAME_DIM)로 뽑는 것부터, 사람/거리/좌우손 차이를 없애는
정규화, 가변 길이 시퀀스를 고정 길이로 맞추는 리샘플링, 증강에 쓰는 좌우반전/회전까지
학습(dataset.py)과 실시간 추론(infer_demo.py)이 동일한 코드를 공유하도록 여기 모아둠.
(학습 때 쓴 정규화와 추론 때 쓴 정규화가 어긋나면 모델이 무조건 틀어지기 때문에 단일 소스로 유지)
"""
import time

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    HandLandmarksConnections,
    PoseLandmarksConnections,
    drawing_utils,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from config import ARM_LANDMARK_IDS, HOLISTIC_MODEL_PATH

ARM_NAMES = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
ARM_DIM = len(ARM_NAMES) * 4        # x, y, z, visibility 를 6개 점에 대해
HAND_DIM = 21 * 3 + 1                # 21개 랜드마크 * xyz + presence 플래그
FRAME_DIM = ARM_DIM + HAND_DIM * 2   # 팔 + 왼손 + 오른손

_ARM_SLICE = slice(0, ARM_DIM)
_LEFT_HAND_SLICE = slice(ARM_DIM, ARM_DIM + HAND_DIM)
_RIGHT_HAND_SLICE = slice(ARM_DIM + HAND_DIM, FRAME_DIM)
_ARM_MIRROR_PAIRS = [(0, 1), (2, 3), (4, 5)]  # (왼쪽,오른쪽) 인덱스 쌍: 어깨/팔꿈치/손목
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
_EPS = 1e-6


class HolisticExtractor:
    """BGR 프레임을 넣으면 (raw_vector[FRAME_DIM], HolisticLandmarkerResult)를 돌려주는 래퍼."""

    def __init__(self, model_path=HOLISTIC_MODEL_PATH, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if not model_path.exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {model_path}\n"
                "holistic_landmarker.task 를 먼저 다운로드해서 models/ 폴더에 넣어주세요."
            )
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_landmarks_confidence=min_tracking_confidence,
            min_hand_landmarks_confidence=min_detection_confidence,
        )
        self._landmarker = HolisticLandmarker.create_from_options(options)
        self._start = time.monotonic()
        self._last_ts = -1

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _next_timestamp_ms(self):
        ts = int((time.monotonic() - self._start) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        vec = np.zeros(FRAME_DIM, dtype=np.float32)

        if results.pose_landmarks:
            for i, name in enumerate(ARM_NAMES):
                p = results.pose_landmarks[ARM_LANDMARK_IDS[name]]
                vec[i * 4:i * 4 + 4] = (p.x, p.y, p.z, p.visibility or 0.0)

        if results.left_hand_landmarks:
            vec[_LEFT_HAND_SLICE] = _hand_to_vec(results.left_hand_landmarks)
        if results.right_hand_landmarks:
            vec[_RIGHT_HAND_SLICE] = _hand_to_vec(results.right_hand_landmarks)

        return vec, results


def _hand_to_vec(hand_landmarks):
    coords = np.array([(p.x, p.y, p.z) for p in hand_landmarks], dtype=np.float32).reshape(-1)
    out = np.zeros(HAND_DIM, dtype=np.float32)
    out[:63] = coords
    out[63] = 1.0  # presence flag: 이 손이 이 프레임에서 감지됐는지
    return out


def draw_landmarks_on_frame(frame, results):
    """collect_data.py / infer_demo.py 미리보기 창에 스켈레톤을 그려서 추적 상태를 눈으로 확인할 수 있게 함."""
    if results.pose_landmarks:
        drawing_utils.draw_landmarks(frame, results.pose_landmarks, PoseLandmarksConnections.POSE_LANDMARKS)
    if results.left_hand_landmarks:
        drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, HandLandmarksConnections.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, HandLandmarksConnections.HAND_CONNECTIONS)


def normalize_frame(raw: np.ndarray) -> np.ndarray:
    """팔은 어깨중심/어깨너비, 손은 손목/손크기 기준으로 정규화 -> 카메라와의 거리, 체형, 사람 차이를 흡수."""
    out = raw.copy()

    arm = out[_ARM_SLICE].reshape(6, 4).copy()
    l_sh, r_sh = arm[0, :3], arm[1, :3]
    center = (l_sh + r_sh) / 2.0
    scale = max(float(np.linalg.norm(l_sh[:2] - r_sh[:2])), _EPS)
    arm[:, :3] = (arm[:, :3] - center) / scale
    out[_ARM_SLICE] = arm.reshape(-1)

    for hand_slice in (_LEFT_HAND_SLICE, _RIGHT_HAND_SLICE):
        block = out[hand_slice]
        if block[63] == 0.0:
            continue  # 손이 감지 안 된 프레임은 0으로 유지 (presence flag로 "없음"을 구분)
        pts = block[:63].reshape(21, 3).copy()
        wrist = pts[WRIST_IDX].copy()
        hand_scale = max(float(np.linalg.norm(pts[MIDDLE_MCP_IDX] - wrist)), _EPS)
        pts = (pts - wrist) / hand_scale
        block[:63] = pts.reshape(-1)

    return out


def resample_sequence(seq: np.ndarray, target_len: int, jitter: bool = False) -> np.ndarray:
    """가변 길이 프레임 시퀀스를 시간축 선형보간으로 target_len 프레임으로 맞춤.
    jitter=True면 샘플링 지점을 살짝 흔들어 제스처 속도 변화(빠르게/느리게)를 흉내내는 증강 효과."""
    seq = np.asarray(seq, dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, num=len(seq))
    t_dst = np.linspace(0.0, 1.0, num=target_len)
    if jitter:
        t_dst = np.clip(t_dst + np.random.uniform(-0.03, 0.03, size=target_len), 0.0, 1.0)
        t_dst = np.sort(t_dst)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, seq[:, d])
    return out


def mirror_frame(vec: np.ndarray) -> np.ndarray:
    """좌우 반전 증강: x부호 반전 + 왼팔<->오른팔, 왼손<->오른손 블록 교체."""
    out = vec.copy()

    arm = out[_ARM_SLICE].reshape(6, 4).copy()
    arm[:, 0] *= -1
    for a, b in _ARM_MIRROR_PAIRS:
        arm[[a, b]] = arm[[b, a]]
    out[_ARM_SLICE] = arm.reshape(-1)

    left_hand = out[_LEFT_HAND_SLICE].copy()
    right_hand = out[_RIGHT_HAND_SLICE].copy()
    left_hand[0:63:3] *= -1
    right_hand[0:63:3] *= -1
    out[_LEFT_HAND_SLICE] = right_hand
    out[_RIGHT_HAND_SLICE] = left_hand

    return out


def rotate_frame(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    """정규화된 좌표는 이미 원점 중심이라, 팔/양손에 같은 각도를 적용하면 카메라-사용자 각도 차이를 흉내낼 수 있음."""
    out = vec.copy()
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    arm = out[_ARM_SLICE].reshape(6, 4).copy()
    arm[:, :2] = arm[:, :2] @ rot.T
    out[_ARM_SLICE] = arm.reshape(-1)

    for hand_slice in (_LEFT_HAND_SLICE, _RIGHT_HAND_SLICE):
        block = out[hand_slice].copy()
        if block[63] != 0.0:
            pts = block[:63].reshape(21, 3)
            pts[:, :2] = pts[:, :2] @ rot.T
            block[:63] = pts.reshape(-1)
        out[hand_slice] = block

    return out


def drop_hand(vec: np.ndarray, side: str) -> np.ndarray:
    """side('left'/'right') 손 데이터를 통째로 지움 (presence=0).
    실제로 그 손이 카메라에 안 잡힌 것과 똑같은 입력이 되어, 학습 때 이런 케이스를 섞어주면
    모델이 팔 움직임/반대쪽 손만으로도 제스처를 구분하는 법을 익히게 됨 (손 인식이 잘 안 되는
    사용자나, 일시적으로 손이 가려지는 상황에 대한 강건성)."""
    out = vec.copy()
    hand_slice = _LEFT_HAND_SLICE if side == "left" else _RIGHT_HAND_SLICE
    out[hand_slice] = 0.0
    return out


def add_noise(vec: np.ndarray, sigma: float = 0.015) -> np.ndarray:
    """좌표에 약한 가우시안 노이즈를 더해 랜드마크 검출 흔들림(jitter)을 흉내냄. visibility/presence 값은 건드리지 않음."""
    out = vec.copy()
    noise = np.random.normal(0.0, sigma, size=out.shape).astype(np.float32)
    noise[3:ARM_DIM:4] = 0.0                                   # arm visibility 채널
    noise[_LEFT_HAND_SLICE.stop - 1] = 0.0                     # left hand presence
    noise[_RIGHT_HAND_SLICE.stop - 1] = 0.0                    # right hand presence
    out += noise
    return out


if __name__ == "__main__":
    # 인덱스 계산이 손으로 짠 코드라 실수하기 쉬워서, 실행할 때마다 기본 불변식을 바로 확인함.
    rng = np.random.default_rng(0)
    sample = rng.uniform(-1, 1, size=FRAME_DIM).astype(np.float32)
    sample[_LEFT_HAND_SLICE.stop - 1] = 1.0
    sample[_RIGHT_HAND_SLICE.stop - 1] = 1.0
    sample[3:ARM_DIM:4] = rng.uniform(0, 1, size=6)  # visibility는 0~1

    assert sample.shape == (FRAME_DIM,)
    assert FRAME_DIM == 152, FRAME_DIM

    mirrored_twice = mirror_frame(mirror_frame(sample))
    assert np.allclose(mirrored_twice, sample, atol=1e-5), "mirror_frame이 대합(involution)이 아님"

    rotated_back = rotate_frame(rotate_frame(sample, 0.7), -0.7)
    assert np.allclose(rotated_back, sample, atol=1e-4), "rotate_frame 각도 합이 원복되지 않음"

    seq = np.stack([sample] * 10)
    resampled = resample_sequence(seq, 30)
    assert resampled.shape == (30, FRAME_DIM)

    normed = normalize_frame(sample)
    assert normed.shape == (FRAME_DIM,)

    dropped = drop_hand(sample, "left")
    assert np.allclose(dropped[_LEFT_HAND_SLICE], 0.0)
    assert np.allclose(dropped[_RIGHT_HAND_SLICE], sample[_RIGHT_HAND_SLICE])
    assert np.allclose(dropped[_ARM_SLICE], sample[_ARM_SLICE])

    print("landmarks.py self-test 통과 (FRAME_DIM =", FRAME_DIM, ")")
