"""손 랜드마크 -> 학습된 분류기로 "손등팔등"인지 판정한다.

detector_mediapipe.py가 지금 쓰는 기하 규칙(is_back_of_hand, 손목→검지MCP/새끼MCP
외적 부호)과 별개로, 우리가 직접 녹화한 데이터로 학습시킨 소형 분류기(ONNX)도
만든다:
    scripts/collect_landmarks.py  — 손모양별 랜드마크 녹화 (data/<label>/*.npy)
    scripts/train_classifier.py   — PyTorch로 학습 후 ONNX export
    (이 파일)                      — 학습된 ONNX를 onnxruntime으로 추론

normalize_landmarks()는 데이터 녹화·학습·추론이 전부 공유하는 단일 소스다 —
정규화가 어긋나면 학습된 모델이 실전에서 무조건 틀어지기 때문에 절대 따로
구현하면 안 된다.

라벨: 손등팔등(주먹 쥔 채 손등·팔등이 카메라를 향함) + none(그 외 모든 자연스러운
손 모양 — 손바닥이 보이거나, 손을 안 들었거나 등). 2클래스 이진 분류.
"""
import json

import numpy as np

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
LABELS = ["손등팔등", "none"]
FEATURE_DIM = 21 * 2  # 21개 랜드마크 * (x, y)


def normalize_landmarks(points_px):
    """21개 (x, y) 픽셀 좌표 -> 손목 원점 + 손크기(손목~중지MCP 거리) 스케일로
    정규화한 42차원 벡터. 카메라 거리·손 크기·사람 차이를 흡수한다."""
    pts = np.asarray(points_px, dtype=np.float32)
    wrist = pts[WRIST_IDX]
    scale = max(float(np.linalg.norm(pts[MIDDLE_MCP_IDX] - wrist)), 1e-6)
    normalized = (pts - wrist) / scale
    return normalized.reshape(-1).astype(np.float32)


class HandPoseClassifier:
    """학습된 ONNX 손모양 분류기. predict(landmarks_px) -> (label, confidence).

    ONNX 모델 자체가 마지막에 softmax를 포함하고 있어(scripts/train_classifier.py의
    export 참고) 출력이 곧 클래스별 확률이다.
    """

    def __init__(self, onnx_path, label_map_path):
        import onnxruntime as ort

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"학습된 손모양 분류기가 없습니다: {onnx_path}\n"
                "scripts/collect_landmarks.py로 데이터를 녹화하고 "
                "scripts/train_classifier.py로 학습부터 진행하세요."
            )
        with open(label_map_path, encoding="utf-8") as f:
            meta = json.load(f)
        self._labels = meta["labels"]
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, points_px):
        """21개 (x, y) 픽셀 좌표 -> (label, confidence). 'none'도 그대로 돌려주므로
        호출부(class_map)에서 제스처가 아닌 것으로 걸러야 한다."""
        feature = normalize_landmarks(points_px)[None, :]  # (1, FEATURE_DIM)
        (probs,) = self._session.run(None, {self._input_name: feature})
        idx = int(np.argmax(probs[0]))
        return self._labels[idx], float(probs[0][idx])
