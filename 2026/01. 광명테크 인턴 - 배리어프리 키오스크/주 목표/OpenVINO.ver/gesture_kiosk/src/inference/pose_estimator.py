"""inference 모듈 — 사람 포즈(RTMPose)를 추론해 얼굴·어깨·손 키포인트를 얻는다.

2026-07-15 2차 개편으로 **유일한 추론 모델**이 됐다 — 손 모양(주먹/한 손가락)·
손 중심 궤적·사용자 잠금(얼굴)이 전부 이 포즈 키포인트로 판정된다
(2026-07-23 새 스펙 — 손 모양도 CNN 없이 손 21점 기하 규칙: hand_shape.py).

2026-07-11 교체(라이선스 B안): ultralytics yolo11n-pose(AGPL-3.0)를 제거하고
rtmlib(Apache-2.0, RTMPose 계열 + ONNX Runtime)로 바꿨다.

모델 파일은 첫 실행 때 자동으로 내려받아 캐시(~/.cache/rtmlib)에 둔다 —
내부망 반입 시에는 make_offline_bundle.bat이 이 캐시를 함께 담는다.

키포인트 번호는 COCO 17 규격이다 (0=코, 1·2=눈, 3·4=귀, 5·6=어깨, 9·10=손목).
pose_engine=wholebody면 COCO-WholeBody 133 규격 — 앞 17개 번호는 COCO 17과
동일하고, 91~132가 양손 21점씩이다 (손 모양 판별·손 중심 추적의 입력 —
hand_shape.py가 유일한 사용처. 새 스펙은 wholebody 필수: body 17은 손이 없다).
주의: 이 라벨은 "화면에 보이는 사람" 기준의 해부학적 좌/우다. 거울 반전된
프레임에서는 사용자의 실제 좌/우와 반대가 되며, 그 보정은 person_lock이 담당한다.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

# COCO 17 키포인트 인덱스 (RTMPose body 계열 출력 순서)
KPT_NOSE = 0
KPT_HEAD_INDICES = (0, 1, 2, 3, 4)  # 코·양눈·양귀 — 얼굴 영역 추정에 사용
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

BBOX_PAD_RATIO = 0.10  # 키포인트 묶음 -> 사람 박스로 넓히는 패딩 (추적용)


def ensure_cuda_dlls():
    """윈도우: onnxruntime CUDA는 torch(cu128)가 등록하는 CUDA DLL 경로에 의존한다.

    onnxruntime-gpu가 설치되면 CUDAExecutionProvider는 항상 목록에 보이지만
    DLL 로드는 세션 생성 시점에 일어나므로(실패 시 조용히 CPU 폴백), CUDA를
    쓰려면 세션을 만들기 전에 반드시 torch를 먼저 임포트해 둬야 한다.
    (2026-07-15 2차: 구 detector.py 삭제로 이 모듈로 옮겨 왔다)"""
    if sys.platform.startswith("win"):
        try:
            import torch  # noqa: F401 — DLL 경로 등록 부수효과만 목적
        except ImportError:
            pass


@dataclass
class PersonPose:
    """사람 1명의 포즈 추정 결과 (기획서 4.6 공통 데이터 구조 스타일)."""

    bbox: tuple                 # (x1, y1, x2, y2) 픽셀 좌표 — 키포인트 묶음 기반
    conf: float
    keypoints: np.ndarray       # shape (17|133, 3) — (x_px, y_px, conf). 133=wholebody 엔진
    head_points: list = field(default_factory=list)  # 신뢰도 통과한 머리 키포인트 [(x, y)]

    def keypoint(self, index, min_conf):
        """키포인트 신뢰도가 통과하면 (x_px, y_px), 아니면 None (어깨·손 등 공용)."""
        x, y, conf = self.keypoints[index]
        if conf < min_conf:
            return None
        return float(x), float(y)


def _resolve_device(device):
    """auto -> onnxruntime에 CUDA가 있으면 cuda, 없으면 cpu."""
    if device == "cpu":
        return device
    # rtmlib의 ORT 세션도 torch의 CUDA DLL 경로가 있어야 GPU로 열린다 —
    # 없으면 조용히 CPU로 폴백해 30 FPS가 무너진다 (2026-07-10 실측: 10 FPS)
    ensure_cuda_dlls()
    if device != "auto":
        return device
    import onnxruntime as ort

    return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"


def _resolve_pose_mode(pose_mode, device):
    """auto -> GPU(cuda)면 balanced(정확), CPU면 lightweight(30 FPS 확보)."""
    if pose_mode != "auto":
        return pose_mode
    return "balanced" if device == "cuda" else "lightweight"


def _build_solution(solution_cls, pose_mode, backend, device):
    """rtmlib 모델 생성 — openvino 백엔드 실패 시 onnxruntime로 즉시 복귀.

    2026-07-27 실측(실카메라, 검출+포즈): onnxruntime 558.7ms → openvino
    443.8ms(약 20% 개선, device=cpu 고정 — rtmlib의 openvino 경로는 GPU
    미지원). 모델 가중치·정밀도는 그대로라 정확도 영향 없음(순수 그래프
    컴파일·커널 차이). openvino 패키지가 없거나 초기화가 실패하는 배포
    환경(예: 오프라인 번들에 안 담겼거나 버전 문제)에서도 설치 자체가
    막히지 않도록 onnxruntime로 조용히 되돌아간다 — quantized의 "파일
    없으면 fp32 유지"와 같은 철학
    """
    if backend != "openvino":
        return solution_cls(mode=pose_mode, backend=backend, device=device)
    try:
        return solution_cls(mode=pose_mode, backend="openvino", device=device)
    except Exception as error:   # noqa: BLE001 — 배포 환경 다양성(패키지 미설치 등) 대응
        logger.warning("openvino 백엔드 초기화 실패(%s) — onnxruntime로 복귀", error)
        return solution_cls(mode=pose_mode, backend="onnxruntime", device=device)


def _recompile_openvino_device(tool, device_name):
    """이미 만들어진 rtmlib 도구(YOLOX/RTMPose)를 다른 OpenVINO 디바이스로 재컴파일한다.

    2026-07-27 신설(CPU+Intel iGPU 동시 추론용) — rtmlib의 openvino 백엔드는
    device_name='CPU'로 하드코딩돼 있어(rtmlib/tools/base.py) 다른 디바이스를 쓰려면
    컴파일된 모델을 직접 교체해야 한다. 실패(컴퓨트 드라이버 미설치·디바이스 없음 등)
    하면 False를 돌려주고 tool은 원래(CPU) 상태 그대로 둔다 — dual_device 폴백 판단은
    이 반환값으로 한다. ★키오스크에 dGPU가 없다는 전제 — dGPU가 있는 PC에서
    device_name="GPU"는 그 dGPU로 잡히니 여기서 켜지 말 것(개발 PC 실측: MX250이 잡힘)
    """
    if getattr(tool, "backend", None) != "openvino":
        return False
    try:
        from openvino import Core

        core = Core()
        if device_name not in core.available_devices:
            return False
        model_onnx = core.read_model(model=tool.onnx_model)
        compiled = core.compile_model(
            model=model_onnx, device_name=device_name,
            config={"PERFORMANCE_HINT": "LATENCY"},
        )
        tool.compiled_model = compiled
        tool.input_layer = compiled.input(0)
        tool.output_layer0 = compiled.output(0)
        tool.output_layer1 = compiled.output(1)
        tool.device = device_name
        return True
    except Exception as error:   # noqa: BLE001 — 드라이버·디바이스 다양성(배포 환경) 대응
        logger.warning("OpenVINO %s 재컴파일 실패(%s) — 이 워커는 CPU로 유지", device_name, error)
        return False


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
        # cv2 자체 스레드풀을 끈다(2026-07-27 실측) — OpenCV(warpAffine 등 전처리)와
        # onnxruntime이 각자 멀티스레드로 같은 8코어를 다투면서 서로 느려짐을 확인
        # (실카메라 파이프라인 349ms/프레임 → cv2 단일 스레드로 236ms, 약 33% 개선,
        # 모델 정밀도·정확도는 무관 — 순수 스레드 경합 해소). onnxruntime 세션이
        # 자체 스레드 풀을 쓰므로 cv2 쪽만 양보하면 된다
        cv2.setNumThreads(1)
        model = config["model"]
        device = _resolve_device(model["device"])
        pose_mode = _resolve_pose_mode(model["pose_mode"], device)
        self._kpt_conf_threshold = config["person_lock"]["kpt_conf_threshold"]
        engine = model.get("pose_engine", "body")
        # mode: auto(장치 따라 자동) | lightweight(빠름) | balanced(기본) | performance(정확) — 첫 실행 시 자동 다운로드.
        # rtmlib은 무거운 의존이라 사용 시점에 임포트한다 (단위 테스트가 가벼워지게)
        if engine == "wholebody":
            # 전신 133 — 손 모양 판별(2026-07-23 스펙) 필수. body(17)는 손 키포인트가 없어 제스처 불가
            from rtmlib import Wholebody as solution
        else:
            from rtmlib import Body as solution

        # 검출·포즈 직접 지휘(2026-07-20 2차 — rtmlib PoseTracker 대체 미니 트래커):
        # ① 검출이 0명이면 **포즈를 생략**한다 — rtmlib은 빈 화면에서도 화면 전체를
        #    사람으로 가정하고 포즈를 돌려 허수(쓰레기) 포즈를 만들며 CPU를 태운다
        # ② 검출은 det_interval_frames마다 1회, 사이엔 **신뢰도 통과 사람의 박스만**
        #    재사용 — 쓰레기 포즈가 다음 프레임 박스로 전파되던 것 차단
        # ③ 사람이 사라지면 캐시가 비어 다음 프레임에 검출이 즉시 다시 돈다
        # (solution 인스턴스의 det_model/pose_model을 그대로 쓰므로 NPU 세션 교체 호환)
        self._det_interval_frames = int(model.get("det_interval_frames", 1))
        self._cached_bboxes = []      # 신뢰도 통과 사람 박스 — 검출 건너뛰는 프레임의 포즈 입력
        self._frames_since_det = 0
        backend = model.get("backend", "onnxruntime")
        self._pose = _build_solution(solution, pose_mode, backend, device)
        logger.info(
            "포즈 모델 로딩 완료: rtmlib %s(mode=%s, backend=%s, device=%s, "
            "det_interval=%d프레임, 허수 포즈 생략)",
            "Wholebody" if engine == "wholebody" else "Body",
            pose_mode, backend, device, self._det_interval_frames,
        )

    def infer(self, frame):
        """프레임에서 사람 포즈를 추정한다. 사람이 없으면 포즈 추론 없이 빈 목록."""
        self._frames_since_det += 1
        if not self._cached_bboxes or self._frames_since_det >= self._det_interval_frames:
            bboxes = list(self._pose.det_model(frame))
            self._frames_since_det = 0
        else:
            bboxes = self._cached_bboxes

        if len(bboxes) == 0:
            return []   # 사람 없음 — 포즈 생략(허수 차단). 캐시가 비어 다음 프레임도 검출부터

        keypoints_xy, scores = self._pose.pose_model(frame, bboxes=bboxes)
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
                    # 몸 17점만 평균 — wholebody의 얼굴 68점이 사람 신뢰도를 지배하지 않게
                    conf=float(score[:17].mean()),
                    keypoints=keypoints,
                    head_points=head_points,
                )
            )
        # 신뢰도 통과 사람의 박스만 다음 프레임에 재사용 — 전원 미달이면 캐시가 비어
        # 다음 프레임에 검출이 다시 돈다 (신규 접근자·복귀 사용자 포착 안전장치)
        self._cached_bboxes = [person.bbox for person in persons]
        return persons


class DualDevicePoseEstimator:
    """CPU + 보조 디바이스(Intel iGPU 등)를 동시에 돌리는 포즈 추정기 (2026-07-27 신설).

    실제 프레임을 한 프레임씩만 받는 실시간 루프 특성상(미래 프레임을 미리 못 받음),
    한 프레임 지연을 대가로 2단 소프트웨어 파이프라인을 쓴다: 이번 프레임은 유휴 디바이스에
    배정만 해두고(논블로킹), 직전 호출 때 배정해 둔 다른 디바이스의 결과를 돌려준다 —
    그래야 두 디바이스가 서로 다른 프레임을 동시에 처리해 처리량이 늘어난다. 첫 프레임은
    돌려줄 이전 결과가 없어 빈 목록([])을 준다(콜드 스타트, 세션당 1회 한정).
    구성은 build_pose_estimator()가 담당 — 보조 디바이스 컴파일이 실패하면 여기까지
    오지 않고 PoseEstimator 단독으로 폴백한다(위 함수 참고).
    """

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._next_seq = 0
        self._prev_future = None

    def infer(self, frame):
        device = self._primary if self._next_seq % 2 == 0 else self._secondary
        self._next_seq += 1
        future = self._executor.submit(device.infer, frame)

        if self._prev_future is None:
            # 콜드 스타트 — 되돌려줄 이전 결과가 없다. 지금 프레임은 백그라운드에서
            # 계속 처리되고, 다음 호출 때 그 결과를 돌려준다(아래 분기)
            self._prev_future = future
            return []

        result = self._prev_future.result()
        self._prev_future = future
        return result


def build_pose_estimator(config):
    """설정에 따라 단일(CPU) 또는 이중(CPU+보조 디바이스) 포즈 추정기를 만든다.

    2026-07-27 신설 — model.dual_device.enabled가 꺼져 있거나, backend가 openvino가
    아니거나, 보조 디바이스 컴파일이 실패하면(컴퓨트 드라이버 미설치 등) 안전하게
    CPU 단독(PoseEstimator)으로 돌아간다 — 그 경우 이 함수를 쓰기 전과 동작이 같다.
    ★개발 PC는 dGPU(MX250)가 OpenVINO GPU 슬롯을 차지해 여기서 검증 불가 —
    실제 키오스크(dGPU 없음, Intel iGPU만 있음)에서 실측해 켤 것.
    """
    primary = PoseEstimator(config)
    dual_cfg = config["model"].get("dual_device", {})
    if not dual_cfg.get("enabled", False):
        return primary
    if config["model"].get("backend") != "openvino":
        logger.warning("dual_device는 backend: openvino 에서만 지원 — CPU 단독으로 진행")
        return primary

    secondary_device = dual_cfg.get("secondary_device", "GPU")
    secondary = PoseEstimator(config)
    det_ok = _recompile_openvino_device(secondary._pose.det_model, secondary_device)
    pose_ok = _recompile_openvino_device(secondary._pose.pose_model, secondary_device)
    if not (det_ok and pose_ok):
        logger.warning("dual_device 보조(%s) 초기화 실패 — CPU 단독으로 진행", secondary_device)
        return primary

    logger.info("dual_device 활성화 — CPU + %s 동시 추론", secondary_device)
    return DualDevicePoseEstimator(primary, secondary)
