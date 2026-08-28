"""pipeline 모듈 — 캡처·추론·판정·전송을 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (2026-07-29 포즈 제거 — 손 단독 추론, hand_select.py 참고):
  카메라(스레드) → 거울 반전 → 손 랜드마크(MediaPipe — 유일한 추론 모델)
  → 사용자 손 선별(hand_select: 크기+연속성, 손 실측 자)
  → 동작 판정(gesture_filter: 손 모양 래치 + 첫 선 궤적 4방향)
  → 이벤트 전송(stdio — stdout 한 줄, 델파이가 파이프로 수신)

2026-07-29 포즈(ONNX Runtime + rtmlib) 제거(사용자 결정): 제스처 판정은 손
데이터만으로 끝난다 — 포즈가 하던 잠금·자(尺)·게이트는 hand_select가 손 기반으로
대체. 추론 엔진 = MediaPipe 내장 TFLite(XNNPACK) 하나 (Apache-2.0 — 라이선스
B안 유지: 상업 허용·카피레프트 없음).

2026-07-23: 웹소켓·UDP·데모 웹 서버 제거(회사 결정 — 네트워크 철회, print 연동).
디버그는 main.py --debug 로컬 창(cv2)이 담당한다.

2026-07-31 머리 앵커 추론 스레드 분리(키오스크 실기 — FPS 저하·획 끊김): 포즈
lite는 BlazeFace(수 ms)와 달리 호출당 수십 ms라 손 루프 인라인이면 10Hz마다
33ms 예산을 넘겨 프레임을 놓쳤다 — 빠른 쓸기(모션 블러 + 큰 이동폭)의 추적이
바로 그 회차에 끊긴다. 앵커는 느리게 변하는 값이라 비동기 반영으로 충분하다.

PipelineState가 디버그 창과 공유되는 유일한 상태 저장소다.
"""
import math
import threading
import time

from src.capture.camera_probe import select_camera
from src.capture.camera_stream import CameraStream, init_camera
from src.utils.env_report import log_environment
from src.inference.head_detector import HeadDetector
from src.inference.hand_tracker import HandTracker
from src.inference.preprocessor import (
    Preprocessor, anchor_keep_sharp_box, arm_reach_mask, blur_outside_mask,
    blur_outside_region, combine_reach_masks, hand_box_reach_mask,
)
from src.pipeline.event_sender import create_event_sender
from src.postprocess.hand_arbiter import HandArbiter
from src.postprocess.hand_select import HandSelector
from src.postprocess.rotor import RotorController
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import draw_debug_panel, draw_status, draw_user_hands

logger = get_logger("pipeline")

EVENT_LOG_MAX_COUNT = 200
EVENT_OVERLAY_HOLD_SEC = 1.5


def resolve_loop_interval_sec(model_config, is_active):
    """추론 루프의 최소 간격 — 활성(손 사용 중)일 땐 max_infer_fps, 유휴일 땐
    idle_infer_fps로 낮춰 CPU·전력을 아낀다 (2026-07-20 추론 부담 절감).
    idle_infer_fps 미설정 브랜치는 종전대로 상시 max_infer_fps."""
    max_fps = model_config["max_infer_fps"]
    idle_fps = model_config.get("idle_infer_fps", max_fps)
    return 1.0 / (max_fps if is_active else min(idle_fps, max_fps))


def resolve_roi_box(prev_box, anchor_box, frame_width_px, frame_height_px, roi_cfg,
                    reach_px):
    """머리 앵커 기반 손 추론 크롭 창 -> (x1, y1, x2, y2) | None(전체 프레임).

    원거리 디지털 줌(2026-07-31 키오스크 실기 — 거리별 인식 편차): MediaPipe
    손바닥 검출기는 프레임 전체를 고정 크기(~192px)로 줄여 보므로, 먼 사용자의
    손은 캡처 해상도와 무관하게 몇 픽셀로 뭉개져 검출이 끊기고 모양(주먹/한
    손가락) 판별이 무너진다. 앵커 주변 팔 도달 반경만 잘라 넣으면 손이 모델
    입력에서 그 비율만큼 커진다 — 크롭은 원본 픽셀 그대로(리사이즈 없음)라
    좌표는 크롭 원점만 더하면 프레임 좌표와 동일하다.

    - 앵커 없음(상체 미노출)·근거리(크롭이 프레임 짧은 변을 덮음) → None.
    - 히스테리시스: 크롭 창을 매 프레임 옮기면 VIDEO 모드의 추적 ROI가 어긋나
      재검출이 반복된다 — 중심·크기가 문턱 이상 변할 때만 창을 갱신한다.
    reach_px: 팔 도달 반경(픽셀) — 앵커가 계산해 넘긴다(hand_select.anchor_reach_px).
    ★2026-08-04: 종전엔 머리 폭 × 배수로 여기서 직접 구했는데, 귀-귀 폭은 고개만
    돌려도 71%까지 줄어(실측) **크롭까지 함께 쪼그라들었다**. 어깨 기준 반경을
    받아 쓰면 그 결함이 사라진다. 크롭 중심은 계속 머리 — 손은 머리 주변에 있다.
    순수 함수 — tests/test_infer_optimization.py.
    """
    if anchor_box is None or reach_px is None or reach_px <= 0.0:
        return None
    if float(anchor_box[2] - anchor_box[0]) <= 0.0:
        return None
    half_px = max(reach_px * roi_cfg.get("pad_reach_ratio", 1.3),
                  roi_cfg.get("min_side_px", 320) / 2.0)
    if 2.0 * half_px >= min(frame_width_px, frame_height_px):
        return None   # 근거리 — 크롭이 프레임을 사실상 다 덮는다: 전체 프레임 사용
    anchor_center_x = (anchor_box[0] + anchor_box[2]) / 2.0
    anchor_center_y = (anchor_box[1] + anchor_box[3]) / 2.0
    side_px = int(2.0 * half_px)
    x1 = int(max(0.0, min(anchor_center_x - half_px, frame_width_px - side_px)))
    y1 = int(max(0.0, min(anchor_center_y - half_px, frame_height_px - side_px)))
    target = (x1, y1, x1 + side_px, y1 + side_px)
    if prev_box is not None:
        prev_center = ((prev_box[0] + prev_box[2]) / 2.0,
                       (prev_box[1] + prev_box[3]) / 2.0)
        prev_side_px = float(max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]))
        target_center = ((target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0)
        if (math.dist(prev_center, target_center)
                <= roi_cfg.get("move_ratio", 0.15) * prev_side_px
                and abs(side_px - prev_side_px)
                <= roi_cfg.get("resize_ratio", 0.2) * prev_side_px):
            return prev_box   # 문턱 미달 — 창 유지 (추적 ROI 안정)
    return target


class PipelineState:
    """추론 결과·성능 수치를 스레드 안전하게 공유한다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self.capture_fps = 0.0
        self.infer_fps = 0.0
        self.last_event = None
        self.event_log = []
        self.is_running = False
        self.is_user_locked = False
        self.debug = {}                # 판정 계기판(gesture_filter.debug) — 실기 튜닝용
        self._viewer_count = 0         # 디버그 창 시청자 수 — 0이면 오버레이 렌더링 생략

    def add_viewer(self):
        """디버그 창 열림 — 다음 루프부터 오버레이를 그린다 (2026-07-20 최적화)."""
        with self._lock:
            self._viewer_count += 1

    def remove_viewer(self):
        with self._lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    @property
    def has_viewer(self):
        return self._viewer_count > 0

    def update_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def append_event(self, gesture_event):
        with self._lock:
            self.last_event = gesture_event
            self.event_log.append(gesture_event)
            if len(self.event_log) > EVENT_LOG_MAX_COUNT:
                self.event_log.pop(0)


def run_pipeline(config):
    """파이프라인 전체를 조립해 시작하고 PipelineState를 돌려준다 (기획서 4.6 계약)."""
    startup_sec = time.monotonic()
    state = PipelineState()
    log_environment(config)   # 어느 하드웨어에서 돈 기록인지 로그 첫머리에 남긴다 (2026-07-16)
    # 시작 병렬화(2026-08-03 — 키오스크 시작 20초 단축): 카메라 오픈(MSMF —
    # 키오스크 실측 ~11초)과 모델 로딩(~수 초)이 서로 독립이라 겹친다. 프로브
    # (auto_select)가 켜져 있으면 오픈에 모델이 필요해 종전 순차 경로 유지
    probe_cfg = config["camera"].get("auto_select") or {}
    pre_open = None
    if not probe_cfg.get("enabled"):
        pre_open = {"cap": None, "error": None}

        def _open_main_camera():
            try:
                pre_open["cap"] = init_camera(config)
            except RuntimeError as error:
                pre_open["error"] = error   # 메인 스레드에서 다시 던진다

        pre_open["thread"] = threading.Thread(target=_open_main_camera, daemon=True)
        pre_open["thread"].start()
    # A안: 모델을 먼저 만들고 손 인식 품질로 카메라를 프로브해 메인을 고른다
    # (2026-07-29 포즈 제거 — 프로브 채점도 손 품질 단독)
    preprocessor = Preprocessor(config)
    hand_tracker = HandTracker(config)   # 주 추론 모델 (2026-07-29 포즈 제거)
    # 머리 앵커(2026-07-31 몸통판 — 얼굴 검출 교체, 사용자 결정): 포즈(BlazePose)가
    # 몸 실루엣으로 잡은 머리 위치의 사람 손만 인식 — 마스크·썬글라스·모자·색상
    # 무관, 옆 사람 손 난입 차단. config에 head_anchor 섹션이 없으면 비활성(종전)
    head_cfg = config.get("head_anchor") or {}
    head_detector = HeadDetector(config) if head_cfg else None
    models_elapsed_sec = time.monotonic() - startup_sec
    if pre_open is not None:
        pre_open["thread"].join()   # 모델 로딩과 겹쳐 돌던 카메라 오픈 대기
        if pre_open["error"] is not None:
            raise pre_open["error"]
        main_device_id, main_cap = config["camera"]["device_id"], pre_open["cap"]
    else:
        main_device_id, main_cap = select_camera(config, hand_tracker, preprocessor)
    camera = CameraStream(config, device_id=main_device_id, cap=main_cap).start()

    first_frame = camera.capture_frame()
    logger.info("시작 소요: 모델 %.1f초 · 카메라 포함 총 %.1f초 (%s)",
                models_elapsed_sec, time.monotonic() - startup_sec,
                "병렬 오픈" if pre_open is not None else "프로브 경로")
    frame_height_px, frame_width_px = first_frame.shape[:2]
    hand_selector = HandSelector(config, frame_width_px, frame_height_px)
    # ★2026-08-06 10차(사용자 확정 — "잡히고 고정"): hand_bend 섹션이 있으면
    # 중재기가 슬롯마다 **자세 판정기**를 쥔다 — 먼저 자세를 발화시킨 손이
    # 잠기고 그 발화가 곧 이벤트다(정지한 손도 잡힌다 — 이동 획득 전제 소멸).
    # 구 쓸기 스택(GestureFilter)은 bend 모드에서 완전히 빠진다.
    # hand_bend 섹션이 없으면 종전(쓸기 경쟁 — 로터 등 다른 판이 그 위에서 돈다)
    hand_arbiter = HandArbiter(
        config, judge_kind="bend" if config.get("hand_bend") else "filter")
    event_sender = create_event_sender(config)
    # 로터(리모컨) 모드(2026-08-05 feat/rotor_remote 판, 사용자 결정): 펼친 손
    # 위 쓸기(up)가 설정 토글 — 켜지면 검지 4방향 단축키(상 home·하 back·
    # 우 confirm·좌 select)가 돌고 나머지 쓸기는 무시된다(rotor.py 독스트링).
    # config에 rotor 섹션이 없으면 종전(up이 파이프로 나간다)
    rotor_cfg = config.get("rotor") or {}
    rotor = (RotorController(config, frame_width_px, frame_height_px)
             if rotor_cfg else None)
    if rotor is not None and (rotor_cfg.get("ui") or {}).get("enabled"):
        # 지연 임포트 — UI를 끈 환경(테스트·서버)은 tkinter 비용을 지지 않는다
        from src.ui.rotor_window import RotorWindow
        RotorWindow(config, rotor).start()   # 데몬 스레드 — 창은 로터 on/off를 따라간다
    # ※손 꺾임(포스처) 판은 2026-08-06 10차부터 중재기 안에서 돈다(위 주석) —
    # 별도 HandBendController를 여기서 만들면 중재기가 낸 자세 이벤트를
    # update()가 쓸기로 오인해 삼키는 이중 판정이 된다. 접점은 중재기 하나다

    state.is_running = True

    # 머리 앵커 스레드 ↔ 손 루프 공유 관측함 — 최신 1건, 소비 즉시 비움
    head_result_lock = threading.Lock()
    head_result = [None]

    # 원거리 디지털 줌 설정(2026-07-31) — 키(roi_zoom) 삭제 시 종전(전체 프레임)
    roi_cfg = config["hand_tracker"].get("roi_zoom")
    # 인식 대상 외 블러(2026-08-07 신설·실험적, 사용자 제안 — "인식되는 사람
    # 제외하고 모든 것을 블러 처리"): 앵커 밖을 흐려 손 모델이 다른 사람의
    # 손을 아예 못 보게 한다. 기본 꺼짐 — 이미 낮게 실측된 FPS(3.7~7.0,
    # docs/TODO.md)에 프레임마다 블러 연산이 더해지는 비용 + "누가 앵커인가"
    # 판정 자체가 아직 흔들리는 채(같은 문서 9차)라 잘못된 순간에 켜지면
    # 진짜 사용자를 블러해 완전히 놓칠 위험이 있다 — 켜서 실기로 확인할 것
    blur_cfg = config["hand_tracker"].get("isolate_anchor_blur") or {}
    blur_enabled = blur_cfg.get("enabled", False)
    blur_keep_sharp_ratio = blur_cfg.get("keep_sharp_shoulder_widths", 1.6)
    # 실루엣 정밀 블러(2026-08-07 2차 — 사용자 결정 "블러를 사람 누끼 딴거
    # 제외하고"): 마스크가 있으면 이 임계로 blur_outside_mask를 쓰고, 없으면
    # (head_anchor.output_segmentation_masks 비활성·추출 실패) 위 사각형으로 폴백
    blur_mask_threshold = blur_cfg.get("mask_threshold", 0.5)
    # ★2026-08-07 3차(사용자 실기 보고 — "손가락 손바닥 쪽에서 블러를 잘 못
    # 잡는다"): 몸통보다 신뢰도가 낮게 나오는 손끝이 문턱 밑으로 떨어져 내
    # 손까지 블러되는 역효과 방지 — 선명 영역을 이만큼(px) 팽창(preprocessor.
    # blur_outside_mask 주석 참고). 0이면 팽창 없음(종전)
    # ★2026-08-07 4차(사용자 실기 재확인 — "뒤에 사람이 손 올리면 블러가
    # 풀리면서 뒷사람 손이 인식된다"): 20px는 내 손끝은 살렸지만 겹침 경계의
    # 여유도 같이 키워, 뒷사람 손이 내 실루엣에 닿을 만큼 다가오면(손을 들 때
    # 특히) 그 경계까지 선명 영역으로 끌려들어왔다. 8로 축소 — 손끝 구제와
    # 겹침 차단은 같은 축(마스크 경계 관대함)의 반대 방향이라 완전한 동시
    # 해결은 어렵다(config.yaml mask_dilate_px 주석 참고). 손끝이 다시 잘리면
    # ↑, 뒷사람이 여전히 새면 더 ↓(0까지)
    blur_mask_dilate_px = blur_cfg.get("mask_dilate_px", 8)
    # 1인 제한 상자(2026-08-07 사용자 지적 — "블러 사람 한명만"): 마스크 경로의
    # 선명 영역이 이 사각형을 못 넘는다. 폴백 상자(keep_sharp_shoulder_widths)와
    # 별도 값인 이유는 역할이 반대이기 때문 — 폴백은 마스크가 없으니 **좁아야**
    # 옆 사람을 덜 담고, 이건 마스크가 정밀도를 맡으니 **내 팔 도달을 다 담을
    # 만큼 넉넉해야** 뻗은 손이 안 잘린다
    blur_mask_bound_ratio = blur_cfg.get("mask_bound_shoulder_widths", 3.6)
    # 팔 주변 반경(2026-08-07 — 옆 사람 차단): 앵커 관절에서 어깨너비 × N 안만
    # 선명하게 남긴다. 0이면 이 제약 없음(종전 — 사각형까지만)
    blur_arm_reach_ratio = blur_cfg.get("arm_reach_shoulder_widths", 0.9)
    # 손 위치 반경(2026-08-07 4차 — 관절 반경의 사각지대 보강). 추적 손의 최근
    # 박스 대각선 × N/2을 반지름으로 — N=1.0이면 원이 박스 모서리에 딱 닿는
    # 최소값이라 여유가 없다. 0이면 이 보강 없음(관절 반경만)
    blur_hand_box_ratio = blur_cfg.get("hand_box_reach_ratio", 1.4)

    def _inference_loop():
        infer_fps_meter = FpsMeter()
        was_active = True   # 유휴↔활성 전환을 로그로 남기기 위한 직전 상태
        last_frame_seq = 0  # 새 프레임 동기화(2026-07-20) — 같은 프레임 중복 추론 방지
        roi_box = None      # 손 추론 크롭 창 — 히스테리시스 상태 (resolve_roi_box)
        overlay_failed = False   # 오버레이 실패 로그를 한 번만 남기기 위한 표시
        while state.is_running:
            loop_start_sec = time.monotonic()

            frame, last_frame_seq = camera.capture_new_frame(last_frame_seq)
            input_tensor = preprocessor.preprocess_frame(frame)

            # 인식 대상 외 블러 — 손 추론(아래) 전에 적용해 다른 사람의 손을
            # 모델 입력 단계에서부터 지운다. 판정(hand_selector) 입력은 안 건드림
            # — 원본 프레임 좌표계를 그대로 유지해야 좌표 계산이 안 어긋난다.
            # 머리(포즈) 스레드에는 적용하지 않는다: 포즈가 앵커 재확인·교정의
            # 유일한 창구라, 그 입력까지 앵커 기준으로 흐리면 앵커가 틀렸을 때
            # 스스로 고칠 방법이 없어진다(자기강화 오류 — 순환 참조)
            if blur_enabled:
                # ★2026-08-07 2차: 실루엣 마스크가 있으면 정밀(픽셀 단위) 블러를
                # 우선한다 — 사각형과 달리 바로 옆에 붙어 앉은 사람도 갈라낸다.
                # 마스크가 없으면(비활성·추출 실패) 종전 사각형으로 폴백
                seg_mask = hand_selector.anchor_segmentation_mask()
                anchor_frame_px = hand_selector.anchor_center_width_px()
                other_mask = hand_selector.other_segmentation_mask()
                if seg_mask is not None:
                    # 1인 제한(2026-08-07 사용자 지적): 마스크(정밀)에 상자(범위
                    # 한계)를 겹친다 — 마스크가 옆 사람으로 새도 이 상자를 못
                    # 넘는다. 상자는 **내 팔 도달을 다 담을 만큼** 넉넉해야
                    # 하므로(뻗은 손이 잘리면 안 된다) 폴백용 상자보다 크다
                    input_tensor = blur_outside_mask(
                        input_tensor, seg_mask, blur_mask_threshold, blur_mask_dilate_px,
                        bound_box=anchor_keep_sharp_box(
                            anchor_frame_px, blur_mask_bound_ratio,
                            frame_width_px, frame_height_px),
                        # 앵커가 아닌 사람들의 실루엣은 **반드시** 흐린다 —
                        # 겹치는 애매한 구간도 흐리는 쪽으로(사용자 요청)
                        exclude_mask=other_mask,
                        # 앵커 몸에서 픽셀이 **이어지지 않는** 덩어리는 통째로
                        # 뺀다 — 마스크가 오작동해도 떨어져 선 사람은 못 남는다
                        seed_point=(None if anchor_frame_px is None
                                    else anchor_frame_px[:2]),
                        # 앵커의 **팔 주변만** 선명하게 — 옆 사람은 거리가
                        # 같아 깊이로 못 가르므로, 판정 영역이 내 팔을 따라
                        # 다니게 해서 가른다(preprocessor.arm_reach_mask).
                        # ★혼자일 때는 걸지 않는다: 배제할 사람이 없어 얻는
                        # 것이 없는데, 손목 추정이 틀리면 **내 손만 잘린다**.
                        # 비용도 5.6ms 아낀다(실측 15.3 -> 9.7ms)
                        # ★2026-08-07 4차(사용자 보고 — "제스처를 하면 블러 때문에
                        # 관절값이 사라진다"): 관절 반경만으로는 부족했다 — 손을
                        # 들거나 펼치는 정상 동작만으로도 손끝이 몸통 관절 기준
                        # 반경을 넘었다(실측 화면 캡처로 확인). 추적 중인 손의
                        # 최근 위치(hand_selector.locked_box) 주변 반경을 **더해**
                        # 관절 반경의 사각지대를 메운다 — 둘 중 하나에만 들어도
                        # 선명(preprocessor.combine_reach_masks, 합집합)
                        reach_mask=(None if (anchor_frame_px is None
                                             or other_mask is None)
                                    else combine_reach_masks(
                                        arm_reach_mask(
                                            input_tensor.shape,
                                            hand_selector.anchor_arm_points(),
                                            blur_arm_reach_ratio * anchor_frame_px[2]),
                                        hand_box_reach_mask(
                                            input_tensor.shape, hand_selector.locked_box,
                                            blur_hand_box_ratio))))
                else:
                    keep_sharp_box = anchor_keep_sharp_box(
                        anchor_frame_px, blur_keep_sharp_ratio,
                        frame_width_px, frame_height_px)
                    input_tensor = blur_outside_region(input_tensor, keep_sharp_box)

            # 원거리 디지털 줌 — 앵커(머리) 주변만 잘라 손 추론: 먼 손이 모델
            # 입력에서 커진다 (resolve_roi_box 독스트링). 크롭은 원본 픽셀
            # 그대로라 크롭 원점을 더하면 프레임 좌표와 동일 (z는 px 궤적·판정에
            # 미사용 — hand_shape.hand_center_point 주석)
            if roi_cfg is not None:
                roi_box = resolve_roi_box(
                    roi_box, hand_selector.anchor_head_box,
                    frame_width_px, frame_height_px, roi_cfg,
                    hand_selector.anchor_reach_px(),
                )
            if roi_box is not None:
                rx1, ry1, rx2, ry2 = roi_box
                hands = hand_tracker.infer(input_tensor[ry1:ry2, rx1:rx2])
                for hand in hands:
                    hand.landmarks[:, 0] += rx1
                    hand.landmarks[:, 1] += ry1
            else:
                hands = hand_tracker.infer(input_tensor)
            heads = None   # None = 새 머리 관측 없음 — 앵커 유지(hand_select)
            if head_detector is not None:
                with head_result_lock:
                    heads = head_result[0]   # 앵커 스레드의 최신 관측 — 1회만 반영
                    head_result[0] = None
            is_engaged = hand_selector.update(hands, heads)
            state.is_user_locked = is_engaged

            # 유휴 판정 — 손이 보이거나 최근 사용 중이면 활성 (idle_infer_fps 절감)
            is_active = bool(hands) or is_engaged
            if is_active != was_active:
                logger.info("추론 %s 전환 (hands=%d)", "활성" if is_active else "유휴",
                            len(hands))
                was_active = is_active

            # ★2026-08-04 두 손 병렬 판정 → 2026-08-06 10차 자세 경쟁: 게이트를
            # 통과한 손(최대 2)을 각자의 판정기로 나란히 보고, 먼저 발화한
            # 손으로 확정한다(hand_arbiter — bend 모드는 자세, 종전은 쓸기).
            # 정지한 손도 자세만 보이면 잡힌다 — 이동 획득 전제 소멸
            gesture_event = hand_arbiter.update(hand_selector, frame_width_px)
            state.debug = hand_arbiter.debug

            if rotor is not None:
                # up = 로터 토글로 소비 · 로터 중엔 나머지 쓸기 무시 · 로터 dwell
                # 확정(home/back/confirm/select)이 대신 이벤트가 된다 (2026-08-05)
                gesture_event = rotor.update(gesture_event, hand_selector)

            if gesture_event is not None:
                # stdio: stdout 한 줄 — 델파이 파이프 수신. True면 수신부가 파이프를
                # 닫은 것(2026-08-04 사고: 종전엔 예외가 이 스레드를 조용히 죽여
                # 창만 멈춘 좀비가 됐다) — 보낼 곳이 없으니 엔진을 접는다
                if event_sender.send(gesture_event):
                    logger.warning("이벤트를 받을 곳이 없다 — 엔진을 종료한다")
                    state.is_running = False
                    break
                state.append_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            # 오버레이(추적점·계기판·상태)는 디버그 창 시청자가 있을 때만 그린다 —
            # 실전(회사 UI는 이벤트만 수신)에서는 매 프레임 그리기·복사가 순수 낭비다
            # (2026-07-20 최적화. 판정·이벤트 경로는 위에서 이미 끝났으므로 무영향)
            if state.has_viewer:
                # ★그리기 실패가 엔진을 죽이지 않게 격리(2026-08-03 키오스크 실기):
                # 신호 형식이 바뀐 뒤 시각화가 구 형식으로 언팩해 ValueError가 났고,
                # cam on 하는 순간 **추론 스레드가 통째로 죽어** 화면이 멈췄다.
                # 오버레이는 진단용 부가 기능이라 판정·이벤트를 중단시킬 이유가 없다 —
                # 실패는 한 번만 로그로 남기고 엔진은 계속 돈다 (무인 키오스크 안전).
                try:
                    annotated = draw_user_hands(input_tensor, hand_selector)
                    annotated = draw_debug_panel(annotated, state.debug)
                    overlay_event = state.last_event
                    if overlay_event is not None and (
                        time.monotonic() - overlay_event.ts_sec > EVENT_OVERLAY_HOLD_SEC
                    ):
                        overlay_event = None
                    annotated = draw_status(annotated, state.infer_fps, overlay_event)
                    state.update_frame(annotated)
                except Exception:   # noqa: BLE001 — 진단 기능이 본 기능을 못 죽이게
                    if not overlay_failed:
                        overlay_failed = True
                        logger.exception("오버레이 그리기 실패 — 엔진은 계속 구동한다"
                                         " (카메라 창만 갱신 중단)")

            # FPS 상한 — 개발 PC에서 200+ FPS로 도는 낭비를 막는다.
            # 유휴(손 없음)일 땐 idle_infer_fps까지 더 낮춘다 (2026-07-20)
            min_loop_interval_sec = resolve_loop_interval_sec(config["model"], is_active)
            elapsed_sec = time.monotonic() - loop_start_sec
            if elapsed_sec < min_loop_interval_sec:
                time.sleep(min_loop_interval_sec - elapsed_sec)

    # 머리 앵커 추론 스레드(2026-07-31 키오스크 실기 — 모듈 독스트링): 손 루프와
    # 분리해 포즈 비용(호출당 수십 ms)이 손 추적 프레임을 밀어내지 않게 한다.
    # 관측은 최신 1건만 유지, 손 루프가 소비하는 즉시 비운다 — hand_select의
    # "None = 관측 없음(앵커 유지)" 규약이 그대로 성립한다
    def _head_anchor_loop():
        interval_sec = 1.0 / head_cfg.get("infer_fps", 10)
        while state.is_running:
            infer_start_sec = time.monotonic()
            frame = preprocessor.preprocess_frame(camera.capture_frame())
            heads = head_detector.infer(frame)
            with head_result_lock:
                head_result[0] = heads
            elapsed_sec = time.monotonic() - infer_start_sec
            if elapsed_sec < interval_sec:
                time.sleep(interval_sec - elapsed_sec)

    threading.Thread(target=_inference_loop, daemon=True).start()
    if head_detector is not None:
        threading.Thread(target=_head_anchor_loop, daemon=True).start()
    logger.info("실시간 파이프라인 시작 (frame_width_px=%d)", frame_width_px)
    return state
