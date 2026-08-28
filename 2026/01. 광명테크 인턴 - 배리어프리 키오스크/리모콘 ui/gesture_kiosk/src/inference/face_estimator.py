"""inference 모듈 — 얼굴 랜드마크(MediaPipe FaceLandmarker, Apache-2.0)를 추론한다.

2026-07-30 헤드트래커 병합 도입: 손 제스처(hand_tracker)는 그대로 두고, **머리 흔들기로
전환되는 헤드트래커 모드**의 입력으로 얼굴 랜드마크를 추가한다 — 코끝(커서 포인터)·
입 벌림/눈 감김/입 오므림(블렌드셰이프)이 전부 이 모델 하나에서 나온다.

★ face_anchor(FaceDetector, 얼굴 박스만)와는 다른 모델이다 — 그쪽은 손 인식용
"가장 가까운 사람" 앵커(bbox만 필요, 저FPS)이고, 이 모듈은 헤드트래커 모드의
커서·블렌드셰이프 판정에 랜드마크 478점이 필요해 별도로 둔다. 두 모델이 함께
도는 대가로 프레임당 비용이 늘어난다(hand 모드 중에도 머리 흔들기 감지를 위해
이 모듈은 상시 실행 — realtime_loop.py 참고).

구 프로토타입(헤드트래커_프로토타입_win.ver)에서 이식하며 **아이트래커(시선) 경로는
제외**했다(사용자 결정 2026-07-30) — 홍채·눈꺼풀 랜드마크 상수와 시선 비율 계산이
전부 그쪽 전용이라 함께 제거. 커서는 코끝 기준 하나만 남는다.

라이선스: MediaPipe(Apache-2.0) — hand_tracker와 동일 엔진·동일 기준이라 의존성이
새로 늘지 않는다 (2026-07-11 라이선스 B안 유지: 상업 허용·카피레프트 없음).

모델 파일: models/weights/face_landmarker.task — download_weights.py가 받는다.
"""
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

# FaceLandmarker 랜드마크 인덱스 (478점 — 468 얼굴 + 10 홍채. 홍채는 아이트래커 전용이라 미사용)
LMK_NOSE_TIP = 1
LMK_LEFT_EYE_OUTER = 33     # 안구간 거리 = 카메라 거리 정규화 자(尺)
LMK_RIGHT_EYE_OUTER = 263

# 코끝 주변 묶음 — 커서 기준점을 점 하나가 아니라 이 묶음의 평균으로 잡는다.
#
# ★2026-08-20 신설. 사용자 실기 보고 "코 기준은 커서가 많이 떨렸고 미간은 별로
# 안 떨리더라"(특히 어두울 때). 실제 세션 녹화 4개에서 커서 궤적을 뽑아 재보니
# 그대로였다 — 커서가 목적지까지 가는 데 실제로 움직인 거리 / 순수 이동 거리:
#
#     코 기준  어두울 때 1.83배   밝을 때 1.13배
#     미간 기준 어두울 때 1.15배   밝을 때 1.07배
#
# 즉 "어두울 때의 코"만 유독 튄다. 이유는 두 기준점의 성질 차이다:
#   · 코끝(랜드마크 1개)은 매끈하고 무늬가 없는 면 위에 있다. 어두우면 그 부근에
#     명암 차가 거의 사라져 모델이 정확한 위치를 못 잡고 값이 계속 흔들린다.
#   · 미간은 코드상 양쪽 눈 바깥쪽 끝 **두 점의 평균**이다. 눈꼬리는 명암이
#     뚜렷해 잘 잡히고, 두 점을 평균 내면 서로 무관한 흔들림이 절반쯤 상쇄된다.
#
# 그래서 코도 똑같이 여러 점의 평균으로 만든다. 서로 무관한 흔들림은 점 개수의
# 제곱근만큼 줄어든다. 고르는 기준은 두 가지:
#   ① 콧대가 아니라 **코 아래쪽(끝·기둥·콧방울)** 만 쓴다 — 콧대는 두 눈 사이에
#      거의 붙어 있어 고개를 돌려도 눈에 대해 잘 안 움직인다. 그런 점을 섞으면
#      정작 필요한 신호(회전)가 묽어진다(미간이 안 되는 이유와 같다).
#   ② 좌우 대칭으로 짝지어 넣는다 — 한쪽만 넣으면 기준점이 옆으로 치우친다.
# 전부 MediaPipe가 공식적으로 코로 분류한 점들이다(FACEMESH_NOSE).
LMK_NOSE_CLUSTER = (
    1, 2, 4, 19, 94,      # 코끝~코기둥 정중선
    97, 326,              # 콧방울 안쪽 (좌/우 대칭)
    98, 327,              # 콧방울 바깥쪽 (좌/우 대칭)
)

BBOX_PAD_RATIO = 0.10  # 랜드마크 묶음 -> 얼굴 박스로 넓히는 패딩


def _rotation_matrix_to_euler_deg(rot):
    """3x3 회전행렬 -> (yaw, pitch, roll) 도(degree).

    ★2026-08-28 신설 — 머리의 실제 3차원 자세를 각도로 쓰기 위한 변환
    (head_pose 설명은 FaceLandmarks.head_pose 참고).

    회전 순서는 컴퓨터비전에서 널리 쓰는 Tait-Bryan Y-X-Z(yaw-pitch-roll)를
    따른다. 짐벌락(pitch가 ±90°에 가까워 yaw와 roll이 한 축으로 붙어버리는
    구간)은 별도 분기로 처리한다 — 키오스크에서 고개를 그만큼 젖힐 일은
    없지만, 그 구간에서 각도가 튀면 커서가 순간이동하므로 막아 둔다.

    부호 약속 (사용자 기준, 거울 반전된 화면 기준이 아니다):
      yaw   왼쪽을 보면 음수 / 오른쪽을 보면 양수
      pitch 위를 보면 양수 / 아래를 보면 음수
      roll  오른쪽으로 갸웃하면 양수

    R = Ry(yaw) @ Rx(pitch) @ Rz(roll) 로 합성된 행렬을 되짚는 식이다. 전개하면

        R[1][2] = -sin(pitch)
        R[0][2] =  sin(yaw)·cos(pitch)      R[2][2] = cos(yaw)·cos(pitch)
        R[1][0] = cos(pitch)·sin(roll)      R[1][1] = cos(pitch)·cos(roll)

    이므로 아래처럼 각각 뽑아낼 수 있다. (2026-08-28: 처음에 Z-Y-X 순서의
    식을 잘못 가져다 써서 yaw와 pitch가 서로 뒤바뀌어 나왔다 — 단위 테스트로
    잡았다. tests/test_head_pose.py 참고)
    """
    sin_pitch = -rot[1][2]
    sin_pitch = max(-1.0, min(1.0, sin_pitch))   # 수치 오차로 |sin|>1 이 되는 것 방지
    # 짐벌락 — cos(pitch)가 0에 붙으면 yaw와 roll이 한 축으로 겹쳐 구분이 안 된다.
    # 위 네 성분이 전부 0에 수렴하므로 atan2가 무의미해진다. roll을 0으로 두고
    # 남은 자유도를 전부 yaw로 몰아준다(키오스크에서 고개를 90° 젖힐 일은 없지만,
    # 그 구간에서 각도가 튀면 커서가 순간이동하므로 막아 둔다)
    if sin_pitch > 0.99999:
        return (math.degrees(math.atan2(rot[0][1], rot[0][0])), 90.0, 0.0)
    if sin_pitch < -0.99999:
        return (math.degrees(math.atan2(-rot[0][1], rot[0][0])), -90.0, 0.0)
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(rot[0][2], rot[2][2])
    roll = math.atan2(rot[1][0], rot[1][1])
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


@dataclass
class HeadPose:
    """머리의 3차원 자세 — MediaPipe가 돌려주는 4x4 변환행렬에서 뽑아낸 값.

    ★2026-08-28 신설.

    [왜 필요한가]
    지금까지 커서는 **화면에 투영된 2D 랜드마크 위치**로 움직였다. 그런데
    코처럼 얼굴에서 튀어나온 점은 고개를 돌리면 원근(perspective) 때문에
    비선형으로 움직인다 — 좌우로만 돌려도 세로가 활처럼 휘는 현상
    (ARC_COMPENSATION이 2차식으로 사후 보정하던 그 문제)의 근본 원인이다.

    머리의 **회전각 자체**를 쓰면 애초에 투영 왜곡이 없다. 보정 상수도,
    그 상수를 카메라 배치마다 다시 재는 일도 필요 없어진다.

    [무엇을 받는가]
    MediaPipe FaceLandmarker의 facial transformation matrix는 표준 얼굴
    모형을 지금 검출된 얼굴에 맞추는 4x4 행렬이다(회전 + 평행이동).
    회전 부분에서 각도를, 평행이동 부분에서 위치를 얻는다.
    z는 카메라로부터의 거리라, 안구간거리로 거리를 짐작하던 기존 방식보다
    직접적이다.
    """

    yaw_deg: float      # 좌우 회전 (도)
    pitch_deg: float    # 상하 회전 (도)
    roll_deg: float     # 갸웃 (도)
    tx: float           # 머리 위치 x (모델 단위 — 대략 cm)
    ty: float           # 머리 위치 y
    tz: float           # 머리 위치 z — 카메라로부터의 거리(음수 방향)


@dataclass
class FaceLandmarks:
    """얼굴 1개의 추정 결과 (기획서 4.6 공통 데이터 구조 스타일)."""

    bbox: tuple                       # (x1, y1, x2, y2) 픽셀 좌표
    conf: float
    landmarks_px: np.ndarray          # shape (478, 2) — (x_px, y_px)
    blendshapes: dict = field(default_factory=dict)   # category_name -> score(0~1)
    # 3차원 머리 자세 — HeadPose 설명 참고. MediaPipe 옵션이 꺼져 있거나
    # 행렬이 안 오면 None이므로, 쓰는 쪽은 반드시 None을 확인해야 한다
    head_pose: object = None

    def landmark_px(self, index):
        """랜드마크 픽셀 좌표 (x, y). 인덱스는 항상 존재 — 신뢰도 게이트가 없다."""
        x, y = self.landmarks_px[index]
        return float(x), float(y)

    def landmarks_mean_px(self, indices):
        """여러 랜드마크의 평균 픽셀 좌표 (x, y) — 흔들리는 점 하나 대신 묶음의
        평균을 기준점으로 쓸 때(LMK_NOSE_CLUSTER 설명 참고). numpy로 한 번에
        평균 내므로 점 개수가 늘어도 비용은 사실상 그대로다."""
        pts = self.landmarks_px[list(indices)]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def blendshape(self, category_name, default=0.0):
        return self.blendshapes.get(category_name, default)

    @property
    def area_px(self):
        """얼굴 박스 면적 — 가까운 사용자 선별 기준 (select_user_face)."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_user_face(faces):
    """여러 얼굴 중 사용자 1명 -> FaceLandmarks | None.

    가장 큰 얼굴 = 카메라에 가장 가까운 사람. hand_select의 face_anchor가
    "가장 큰 얼굴 = 사용자"로 앵커를 고르는 것과 같은 기준이라 두 모드가 같은
    사람을 가리킨다 — 별도 얼굴 잠금을 새로 만들지 않는다.
    """
    if not faces:
        return None
    return max(faces, key=lambda face: face.area_px)


def _extract_head_pose(result, face_idx):
    """MediaPipe 결과에서 얼굴 1개의 HeadPose를 뽑는다. 없으면 None.

    ★방어적으로 감쌌다 — 이 값은 "있으면 좋은" 부가 정보라, 뽑다가 실패해도
    기존 랜드마크 기반 동작은 그대로 굴러가야 한다. 모델이나 mediapipe 버전이
    바뀌어 행렬이 안 오는 상황에서 트래커 전체가 죽으면 안 된다.
    """
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if not matrices or face_idx >= len(matrices):
        return None
    try:
        m = np.asarray(matrices[face_idx], dtype=np.float64).reshape(4, 4)
        yaw, pitch, roll = _rotation_matrix_to_euler_deg(m[:3, :3])
        return HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                        tx=float(m[0][3]), ty=float(m[1][3]), tz=float(m[2][3]))
    except Exception:   # noqa: 방어적 — 부가 정보 하나 때문에 추론이 죽으면 안 된다
        logger.exception("머리 자세(변환행렬) 해석 실패 - 랜드마크 기반 동작은 계속됩니다")
        return None


def _landmarks_to_bbox_px(landmarks_px, frame_shape):
    """랜드마크 전체를 감싸는 박스 + 패딩."""
    xs = landmarks_px[:, 0]
    ys = landmarks_px[:, 1]
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    pad = max(x2 - x1, y2 - y1, 20.0) * BBOX_PAD_RATIO
    h_px, w_px = frame_shape[:2]
    return (
        max(0.0, float(x1 - pad)), max(0.0, float(y1 - pad)),
        min(w_px - 1.0, float(x2 + pad)), min(h_px - 1.0, float(y2 + pad)),
    )


class FaceEstimator:
    """MediaPipe FaceLandmarker 래퍼. infer(frame) -> list[FaceLandmarks]."""

    def __init__(self, config):
        face_cfg = config["face_tracker"]
        # mediapipe는 무거운 의존이라 사용 시점에 임포트한다 (단위 테스트가 가벼워지게)
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode,
        )

        # 2026-07-31 실기 — 한글 경로 대응 (hand_tracker.py와 동일 사유·동일 처방:
        # mediapipe 0.10.14가 model_asset_path의 한글 경로를 못 연다)
        with open(face_cfg["model_path"], "rb") as model_file:
            model_bytes = model_file.read()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=RunningMode.VIDEO,
            num_faces=face_cfg["max_num_faces"],
            min_face_detection_confidence=face_cfg["min_detection_conf"],
            min_face_presence_confidence=face_cfg["min_presence_conf"],
            min_tracking_confidence=face_cfg["min_tracking_conf"],
            output_face_blendshapes=True,   # 입 벌림·눈 감김·입 오므림 판정의 입력
            # ★2026-08-28 신설 — 머리의 3차원 자세(HeadPose 설명 참고).
            # 추가 비용은 사실상 없다: MediaPipe가 랜드마크를 뽑는 과정에서
            # 이미 계산해 둔 행렬을 결과에 실어 보내는 것뿐이라, 새 추론이
            # 돌지 않는다. 실측으로 확인할 것(scripts/measure_head_pose.py).
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._infer_scale_ratio = face_cfg["infer_scale_ratio"]
        # VIDEO 모드는 단조 증가 타임스탬프(ms)가 필수 — 프레임 간 추적에 쓰인다
        self._start_sec = time.monotonic()
        self._last_timestamp_ms = -1
        logger.info(
            "얼굴 랜드마크 모델 로딩 완료: MediaPipe FaceLandmarker (max_num_faces=%d, infer_scale=%.2f, %s)",
            face_cfg["max_num_faces"], self._infer_scale_ratio, face_cfg["model_path"],
        )

    def infer(self, frame):
        """프레임(BGR·거울 반전 후)에서 얼굴 랜드마크를 추정한다 -> list[FaceLandmarks].

        infer_scale_ratio < 1.0이면 추론 입력만 축소한다 — MediaPipe는 정규화(0~1) 좌표를
        돌려주므로 원본 프레임 크기로 곱하면 좌표는 자동으로 원본 기준이 된다
        (시각화·판정 코드는 축소 여부를 몰라도 된다).
        """
        infer_frame = frame
        if self._infer_scale_ratio < 1.0:
            infer_frame = cv2.resize(
                frame, None, fx=self._infer_scale_ratio, fy=self._infer_scale_ratio,
                interpolation=cv2.INTER_AREA,
            )
        # cvtColor는 SIMD 최적화 + 연속 메모리 출력 — numpy 역순 슬라이스+복사보다 빠르다
        rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._start_sec) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1   # 단조 증가 보장 (고FPS 보호)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h_px, w_px = frame.shape[:2]
        faces = []
        for face_idx, landmarks in enumerate(result.face_landmarks):
            landmarks_px = np.array(
                [(pt.x * w_px, pt.y * h_px) for pt in landmarks], dtype=np.float32
            )
            blendshapes = {}
            if result.face_blendshapes:
                blendshapes = {c.category_name: c.score for c in result.face_blendshapes[face_idx]}
            faces.append(FaceLandmarks(
                bbox=_landmarks_to_bbox_px(landmarks_px, frame.shape), conf=1.0,
                landmarks_px=landmarks_px, blendshapes=blendshapes,
                head_pose=_extract_head_pose(result, face_idx),
            ))
        return faces

    def close(self):
        self._landmarker.close()
