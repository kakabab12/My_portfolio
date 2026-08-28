"""inference 모듈 — 머리 앵커 검출 (MediaPipe Pose Landmarker/BlazePose, Apache-2.0).

2026-07-31 도입(사용자 결정 — 얼굴 검출 앵커의 가림 한계): 마스크·썬글라스·모자·
색상 조합으로 얼굴 랜드마크가 가려지면 BlazeFace가 실패했다. 포즈는 **몸
실루엣으로 사람을 잡고 머리 위치를 골격의 일부로** 출력하므로 얼굴에 뭘 썼든
무관하다 — 학습·액세서리별 대응이 아예 불필요. 얼굴 검출기(face_detector.py)는
전면 교체·삭제(사용자 결정: 회사 시선추적 모듈과의 얼굴 처리 중복·충돌 회피 —
엔진은 얼굴을 다루지 않는다).

머리 관측 정의: 중심 = 양귀 키포인트(7·8) 중점, 폭 = 귀-귀 거리(≈얼굴 폭) —
구 face_anchor의 "폭 × reach 배수 = 팔 도달" 체계를 숫자 그대로 잇는다.
관측 인원(num_poses)은 config로 정한다. ★2026-08-04 실기(사용자 보고 — "다른
사람 난입 시 노이즈"): 종전 1인 고정에는 구조적 구멍이 있었다. 손은 최대 4개까지
보면서 **몸은 한 사람만** 보니 "저 손이 누구 팔 끝인지"를 물을 수가 없었고,
더 나쁘게는 모델이 상대방으로 옮겨가면 내 머리가 관측에서 사라져 앵커가 낡고
→ 유예 후 해제 → **게이트가 통째로 꺼지고** → 상대가 앵커를 차지해 내 손이
거부됐다(사용자 보고의 "인식 손이 꺼짐"·"골격 선 요동침"이 같은 연쇄다).
2명을 보면 내 골격이 매 프레임 유지돼 그 연쇄가 시작되지 않는다.
귀 키포인트는 가려져도 몸 맥락으로 추정돼 좌표가 나온다.

라이선스: HandLandmarker와 동일 Apache-2.0 — 상업 사용 가능·코드 공개(카피레프트)
의무 없음·제품 화면 표시 의무 없음 (라이선스 B안 유지, №9).
모델 파일: models/weights/pose_landmarker_lite.task (~5MB) — download_weights.py.
※7-29에 제거한 포즈(rtmlib+ONNX Runtime — 별도 의존 2개·다인·~190MB)와 달리
mediapipe 내장이라 의존성 추가가 없다.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

NOSE_IDX = 0
LEFT_EAR_IDX = 7
RIGHT_EAR_IDX = 8
LEFT_SHOULDER_IDX = 11   # BlazePose 어깨 — 몸 기준 높이선(추정 대신 실측)
RIGHT_SHOULDER_IDX = 12
LEFT_ELBOW_IDX = 13      # BlazePose 팔꿈치 — 디버그 창의 팔 선
RIGHT_ELBOW_IDX = 14
LEFT_WRIST_IDX = 15      # BlazePose 손목 — 손이 이 사람 팔 끝인지 판단용
RIGHT_WRIST_IDX = 16


def _points(pose_landmarks, indices, w_px, h_px):
    """지정 랜드마크를 (x_px, y_px, visibility) 튜플 목록으로 — 신뢰도를 살려 넘긴다."""
    return tuple(
        (pose_landmarks[idx].x * w_px, pose_landmarks[idx].y * h_px,
         float(pose_landmarks[idx].visibility))
        for idx in indices
    )


@dataclass
class HeadDetection:
    """사람 1명의 관측 — 앵커 선별(hand_select)에 필요한 최소 정보.

    이름은 "머리"지만 2026-08-03부터 **손목 좌표**도 싣는다 — 같은 포즈 추론
    한 번에서 나오므로 비용이 0이고, "이 손이 저 사람 팔 끝에 붙어 있나"를
    물으려면 손목이 필요하다 (wrists 필드 주석 참고).
    """

    center_x_px: float
    center_y_px: float
    width_px: float     # 귀-귀 거리 — 클수록 가깝다 (앵커 선별 기준 + 도달 반경 자)
    conf: float
    shoulders: tuple = ()  # ((x_px, y_px, visibility), ...) — 양 어깨 (BlazePose 11·12).
                        #   ★2026-08-04 사용자 지적("몸통 팔 손목 손 다 잡고 있는 거
                        #   아니냐"): 포즈는 33점을 다 주는데 엔진은 코·귀만 쓰고
                        #   어깨선은 **머리 위치로 추정**하고 있었다. 실측 어깨가 있으면
                        #   "배 밑 손은 후보가 아니다" 같은 몸 기준 판단이 정확해진다
    elbows: tuple = ()  # ((x_px, y_px, visibility), ...) — 양 팔꿈치 (13·14) — 표시용
    wrists: tuple = ()  # ((x_px, y_px, visibility), ...) — 양 손목 (BlazePose 15·16).
                        #   ★visibility를 함께 싣는 이유(2026-08-04 사용자 관찰):
                        #   BlazePose는 가려지거나 **화면 밖이어도 33점을 항상 채운다**
                        #   — 안 보이는 손목의 추측 좌표를 "내 팔 끝"의 근거로 쓰면
                        #   있지도 않은 곳에 판정 영역이 생긴다. 쓰는 쪽이 신뢰도를
                        #   보고 거를 수 있게 값을 그대로 넘긴다 (hand_select)
                        #   ★2026-08-03 실기(사용자 보고: "내 몸에서 뻗어나가는 손이
                        #   아닌데도 남의 몸 손이 잡힌다"): 머리 중심 반경 하나로는
                        #   못 가른다 — 팔을 옆으로 뻗으면 머리~내 손 거리가 옆 사람
                        #   손까지의 거리와 같아진다. 손목을 알면 "내 팔 끝인가"를
                        #   직접 물을 수 있다 (hand_select._filter_hands_by_anchor).
                        #   기본 ()는 하위 호환 — 없으면 손목 게이트가 꺼진다
    segmentation_mask: np.ndarray = field(default=None)  # 이 사람의 픽셀 단위
                        #   실루엣(0~1 float, 프레임과 같은 h×w) | None(비활성·추출
                        #   실패). ★2026-08-07 신설(사용자 요청 — "블러를 사람 누끼
                        #   딴거 제외하고"): 어깨 사각형(anchor_keep_sharp_box)은
                        #   가까이 붙은 두 사람을 못 가른다 — 사각형 안에 옆 사람
                        #   몸이 같이 들어가면 그대로 선명하게 남는다. Pose
                        #   Landmarker의 output_segmentation_masks가 사람별
                        #   실루엣을 준다 — 이걸 쓰면 옆 사람이 같은 사각형 안에
                        #   있어도 실루엣 밖이라 블러된다. 비용은 추가 실측 필요
                        #   (이미 낮은 FPS 3.7~7.0을 더 깎을 수 있다 — 사용 시 확인)



class HeadDetector:
    """MediaPipe Pose Landmarker 래퍼. infer(frame) -> list[HeadDetection] (0~1개)."""

    def __init__(self, config):
        detector_cfg = config["head_anchor"]
        self._model_path = detector_cfg["model_path"]
        # mediapipe는 무거운 의존이라 사용 시점에 임포트한다 (단위 테스트가 가벼워지게)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        # 2026-08-07 신설 — 인식 대상 외 블러가 실루엣 기반일 때만 켠다(비용
        # 발생 — segmentation_mask 필드 주석 참고). 기본 False = 종전과 동일 비용
        self._want_segmentation = detector_cfg.get("output_segmentation_masks", False)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.VIDEO,
            # 2명 이상을 봐야 "이 손이 누구 것인가"를 물을 수 있다(모듈 주석)
            num_poses=detector_cfg.get("num_poses", 2),
            min_pose_detection_confidence=detector_cfg.get("min_detection_conf", 0.5),
            output_segmentation_masks=self._want_segmentation,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        # VIDEO 모드는 단조 증가 타임스탬프(ms)가 필수 (hand_tracker와 동일 규약)
        self._start_sec = time.monotonic()
        self._last_timestamp_ms = -1
        self._segmentation_warn_logged = False   # 마스크 추출 실패 로그 1회 제한
        logger.info("머리(포즈) 모델 로딩 완료: MediaPipe Pose Landmarker (%s)",
                    self._model_path)

    def infer(self, frame):
        """프레임(BGR·거울 반전 후)의 머리 관측 -> list[HeadDetection].

        거울 보정 불필요 — 머리에는 좌/우 라벨이 없고 좌표는 손과 같은
        (반전 후) 프레임 좌표계라 그대로 비교 가능하다.
        """
        h_px, w_px = frame.shape[:2]
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame[:, :, ::-1].copy()
        )
        timestamp_ms = int((time.monotonic() - self._start_sec) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1   # 단조 증가 보장 (고FPS 보호)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        heads = []
        for person_idx, pose_landmarks in enumerate(result.pose_landmarks):
            left_ear = pose_landmarks[LEFT_EAR_IDX]
            right_ear = pose_landmarks[RIGHT_EAR_IDX]
            width_px = (((left_ear.x - right_ear.x) * w_px) ** 2
                        + ((left_ear.y - right_ear.y) * h_px) ** 2) ** 0.5
            if width_px <= 0.0:
                continue
            # 신뢰도 = 머리 키포인트 가시성 평균 — 가려져도 몸 맥락 추정값이 나오며
            # visibility가 낮아질 뿐이다 (마스크·모자는 좌표 정확도에 영향 적음)
            conf = (pose_landmarks[NOSE_IDX].visibility
                    + left_ear.visibility + right_ear.visibility) / 3.0
            shoulders = _points(pose_landmarks,
                               (LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX), w_px, h_px)
            heads.append(HeadDetection(
                center_x_px=(left_ear.x + right_ear.x) / 2.0 * w_px,
                center_y_px=(left_ear.y + right_ear.y) / 2.0 * h_px,
                width_px=width_px,
                conf=float(conf),
                # ★2026-08-04 배선 누락 정정(사용자 보고 — "손이 안 잡히니 손목쪽에서
                # 오염이 나온다"): 아래 네 줄은 종전 (x, y) 2-튜플을 만들었고, 그래서
                # hand_select.reliable_wrists의 가시성 관문(len(wrist) > 2 조건부)이
                # **한 번도 실행되지 않았다** — 주머니 속 손목의 추측 좌표가 관문 없이
                # 통과해 있지도 않은 곳에 판정 영역을 만들었다. _points가 정의만 되고
                # 안 불리고 있었던 것이 원인. 관문이 꺼지면 심사 생략(인식 우선 폴백)이라
                # 손이 잘리는 쪽으로는 안 기운다
                shoulders=shoulders,
                elbows=_points(pose_landmarks,
                               (LEFT_ELBOW_IDX, RIGHT_ELBOW_IDX), w_px, h_px),
                wrists=_points(pose_landmarks,
                               (LEFT_WRIST_IDX, RIGHT_WRIST_IDX), w_px, h_px),
                segmentation_mask=self._segmentation_mask_for(result, person_idx),
            ))
        return heads

    def _segmentation_mask_for(self, result, person_idx):
        """이 사람의 실루엣 마스크(numpy, 0~1) | None — 실패해도 엔진은 계속 돈다.

        ★방어적으로 감싼 이유: output_segmentation_masks는 이 프로젝트에서
        처음 켜보는 옵션이라(2026-08-07), 버전별 반환 형태 차이나 예외로 손
        추적 스레드 전체가 죽는 것보다 이 사람만 마스크 없이(None) 진행하는
        편이 안전하다 — realtime_loop의 "오버레이 실패해도 엔진은 계속" 원칙과
        같은 정신. 실패는 처음 1회만 로그(반복 실패로 로그가 덮이는 것 방지)
        """
        if not self._want_segmentation:
            return None
        try:
            masks = result.segmentation_masks
            if not masks or person_idx >= len(masks):
                return None
            # ★.copy()가 **필수**다(2026-08-07 코드 리뷰 지적). numpy_view()는
            # MediaPipe 내부 버퍼를 복사 없이 가리키는 뷰인데, 이 값은
            # ①이 스레드(포즈)에서 만들어져 hand_select에 **저장**되고
            # ②다른 스레드(추론 루프)에서 ③여러 프레임 뒤에 읽힌다.
            # 뷰인 채로 넘기면 result가 GC되는 순간 버퍼가 사라지거나, 더 나쁘게는
            # MediaPipe가 버퍼를 재사용해 **저장해 둔 마스크가 조용히 다음 프레임
            # 내용으로 바뀐다**(크래시 없이 앵커 상태와 마스크만 어긋난다 —
            # 간헐적이라 실기에서 가장 잡기 어려운 종류의 버그).
            # 비용: 720p float32 ≈ 3.7MB/프레임, 10Hz면 ≈37MB/s 할당. 정확성과
            # 바꿀 값이 아니다 — 부담되면 여기서 임계를 적용해 uint8로 떨굴 것
            return masks[person_idx].numpy_view().copy()
        except Exception:   # noqa: BLE001 — 마스크 추출 실패가 포즈 추론 전체를 못 죽이게
            if not self._segmentation_warn_logged:
                self._segmentation_warn_logged = True
                logger.exception("실루엣 마스크 추출 실패 — 이후 프레임은 마스크 없이 진행")
            return None
