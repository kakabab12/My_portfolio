"""inference 모듈 — 사람 포즈(RTMPose)를 추론해 얼굴·손목 키포인트를 얻는다.

person_lock이 "누가 카메라 앞에 있는 사용자인지" 판단하는 데 쓴다 (얼굴 크기·
선명도로 잠금 대상 선정, 손목 위치로 제스처를 그 사람 손에 귀속). rtmlib
(Apache-2.0, RTMPose 계열 + ONNX Runtime)를 쓴다 — 출력은 COCO 17 키포인트 규격.

모델 파일은 첫 실행 때 자동으로 내려받아 캐시(~/.cache/rtmlib)에 둔다 (인터넷 필요).

키포인트 번호는 COCO 17 규격이다 (0=코, 1·2=눈, 3·4=귀, 9=왼손목, 10=오른손목).
주의: 이 라벨은 "화면에 보이는 사람" 기준의 해부학적 좌/우다. 거울 반전된
프레임에서는 사용자의 실제 좌/우와 반대가 되며, 그 보정은 person_lock이 담당한다.
"""
from dataclasses import dataclass, field

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

# COCO 17 키포인트 인덱스 (RTMPose body 계열 출력 순서)
KPT_NOSE = 0
KPT_HEAD_INDICES = (0, 1, 2, 3, 4)  # 코·양눈·양귀 — 얼굴 영역 추정에 사용
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

BBOX_PAD_RATIO = 0.10  # 키포인트 묶음 -> 사람 박스로 넓히는 패딩 (추적용)


@dataclass
class PersonPose:
    """사람 1명의 포즈 추정 결과."""

    bbox: tuple                 # (x1, y1, x2, y2) 픽셀 좌표 — 키포인트 묶음 기반
    conf: float
    keypoints: np.ndarray       # shape (17, 3) — (x_px, y_px, conf)
    head_points: list = field(default_factory=list)  # 신뢰도 통과한 머리 키포인트 [(x, y)]

    def wrist(self, index, min_conf):
        """키포인트 신뢰도가 통과하면 (x_px, y_px), 아니면 None."""
        x, y, conf = self.keypoints[index]
        if conf < min_conf:
            return None
        return float(x), float(y)


def _bbox_from_keypoints(keypoints, kpt_conf, frame_shape):
    """신뢰도 통과 키포인트를 감싸는 박스. 통과점이 없으면 None."""
    valid = keypoints[keypoints[:, 2] >= kpt_conf]
    if len(valid) == 0:
        return None
    x1, y1 = valid[:, 0].min(), valid[:, 1].min()
    x2, y2 = valid[:, 0].max(), valid[:, 1].max()
    pad = max(x2 - x1, y2 - y1, 20.0) * BBOX_PAD_RATIO
    h_px, w_px = frame_shape[:2]
    return (
        max(0.0, float(x1 - pad)), max(0.0, float(y1 - pad)),
        min(w_px - 1.0, float(x2 + pad)), min(h_px - 1.0, float(y2 + pad)),
    )


class PoseEstimator:
    """RTMPose 포즈 추정기. infer(frame) -> list[PersonPose]."""

    def __init__(self, config):
        from rtmlib import Body  # 무거운 의존 — person_lock을 끈 환경에선 임포트하지 않는다

        model = config["model"]
        self._kpt_conf_threshold = config["person_lock"]["kpt_conf_threshold"]
        # mode: lightweight(빠름) | balanced | performance(정확) — 첫 실행 시 자동 다운로드
        self._body = Body(mode=model["pose_mode"], backend="onnxruntime", device=model["device"])
        logger.info("포즈 모델 로딩 완료: rtmlib Body(mode=%s, device=%s)",
                    model["pose_mode"], model["device"])

    def infer(self, frame):
        """프레임에서 사람 포즈를 추정한다."""
        keypoints_xy, scores = self._body(frame)  # (N,17,2), (N,17)
        persons = []
        for xy, score in zip(keypoints_xy, scores):
            keypoints = np.concatenate([xy, score[:, None]], axis=1).astype(np.float32)
            bbox = _bbox_from_keypoints(keypoints, self._kpt_conf_threshold, frame.shape)
            if bbox is None:
                continue
            head_points = [
                (float(keypoints[i][0]), float(keypoints[i][1]))
                for i in KPT_HEAD_INDICES
                if keypoints[i][2] >= self._kpt_conf_threshold
            ]
            persons.append(
                PersonPose(
                    bbox=bbox,
                    conf=float(score.mean()),
                    keypoints=keypoints,
                    head_points=head_points,
                )
            )
        return persons
