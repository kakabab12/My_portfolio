"""pipeline 모듈 — 다중 카메라 캡처 -> 손 추론 -> 손모양 -> D-pad 방향 존 매핑을
백그라운드 스레드에서 반복 실행하고 PipelineState로 공유한다.

카메라마다 별도 HandTracker 인스턴스를 둔다 — MediaPipe VIDEO 모드는 단일
연속 영상을 전제해, 서로 다른 물리 카메라의 프레임을 한 인스턴스에 섞으면
내부 프레임 간 추적 상태(모션 예측)가 오염된다.

안전: 손이 안 보이거나, 잠겨(armed=False) 있거나, engage_shape가 아니거나,
방향 존이 아니면(center·애매) 그 프레임부터 유예 없이 (0, 0)을 낸다 —
DpadController.update()가 이미 그 계약을 보장하므로 이 모듈은 그 결과를
그대로 전달하기만 한다(추가 지연 없음 — "멀어져서 안 보이면 계속 달린다"를
막는 첫 번째 방어선. 두 번째 방어선은 PipelineState.snapshot()의 age_sec을
보는 ros2_bridge 워치독).
"""
import threading
import time

import cv2

from src.capture.dual_camera import CameraArbiter, MultiCameraManager, score_hand_quality
from src.control.dpad import DpadController, ZONE_CENTER
from src.control.finger_commands import (
    WaveDetector, classify_finger_command,
)
from src.inference.hand_tracker import HandTracker
from src.postprocess.camera_motion import CameraMotionCompensator
from src.postprocess.hand_lock import PrimaryHandTracker
from src.postprocess.hand_shape import HandShapeStabilizer, classify_hand_shape, hand_bbox_px
from src.postprocess.roi_zoom import crop_frame, offset_landmarks_xy, resolve_roi_box
from src.utils.logger import get_logger

logger = get_logger("pipeline")

INDEX_FINGERTIP_LMK = 8
LOOP_IDLE_SLEEP_SEC = 0.005   # 두 카메라 모두 새 프레임이 없을 때 짧게 양보(바쁜 대기 방지)


class PipelineState:
    """추론 결과를 스레드 세이프하게 공유한다. Flask 라우트는 이 상태만 읽는다."""

    def __init__(self):
        self._lock = threading.Lock()
        self.is_running = False
        self._cameras = None
        self._linear_x = 0.0
        self._angular_z = 0.0
        self._hand_detected = False
        self._engaged = False
        self._armed = False
        self._shape = None
        self._zone = None
        self._active_camera = None
        self._confidence = 0.0
        self._last_update_sec = time.monotonic()
        self._latest_frame = None
        self._camera_frames = {}
        self._camera_sequences = {}
        self._viewer_count = 0
        self._inference_status = []

    def update(self, linear_x, angular_z, hand_detected, engaged, armed, shape,
               zone, active_camera, confidence):
        with self._lock:
            self._linear_x = linear_x
            self._angular_z = angular_z
            self._hand_detected = hand_detected
            self._engaged = engaged
            self._armed = armed
            self._shape = shape
            self._zone = zone
            self._active_camera = active_camera
            self._confidence = confidence
            self._last_update_sec = time.monotonic()

    def snapshot(self):
        """/cmd 응답용. age_sec은 호출 시점 기준으로 매번 새로 계산한다 —
        파이프라인 스레드가 멈춰도(hang/crash) age_sec은 계속 커져, 이를 보는
        ros2_bridge 워치독이 감지할 수 있다(모듈 독스트링 참고)."""
        with self._lock:
            return {
                "linear_x": self._linear_x,
                "angular_z": self._angular_z,
                "hand_detected": self._hand_detected,
                "engaged": self._engaged,
                "armed": self._armed,
                "shape": self._shape,
                "zone": self._zone,
                "active_camera": self._active_camera,
                "confidence": self._confidence,
                "age_sec": time.monotonic() - self._last_update_sec,
            }

    def update_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def update_camera_frame(self, camera_index, frame, sequence):
        """웹 진단 화면용 카메라별 최신 프레임을 저장한다."""
        with self._lock:
            self._camera_frames[camera_index] = frame
            self._camera_sequences[camera_index] = sequence

    def get_camera_frame(self, camera_index):
        with self._lock:
            frame = self._camera_frames.get(camera_index)
            return None if frame is None else frame.copy()

    def camera_status(self, device_ids):
        with self._lock:
            return [
                {
                    "index": index,
                    "device_id": device_id,
                    "frame_received": index in self._camera_frames,
                    "sequence": self._camera_sequences.get(index, 0),
                }
                for index, device_id in enumerate(device_ids)
            ]

    def set_inference_status(self, trackers):
        """모델 생성 직후 확정된 CPU/GPU delegate 상태를 보관한다."""
        with self._lock:
            self._inference_status = [tracker.inference_status() for tracker in trackers]

    def inference_status(self):
        """/health 응답용 delegate 상태 사본."""
        with self._lock:
            trackers = [dict(status) for status in self._inference_status]
        return {
            "trackers": trackers,
            "gpu_accelerated": bool(trackers) and all(
                status["gpu_accelerated"] for status in trackers),
        }

    def add_viewer(self):
        """/video_feed 시청자 등록 — 다음 루프부터 오버레이 그리기가 켜진다."""
        with self._lock:
            self._viewer_count += 1

    def remove_viewer(self):
        with self._lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    @property
    def has_viewer(self):
        return self._viewer_count > 0

    def stop(self):
        self.is_running = False
        if self._cameras is not None:
            self._cameras.stop()


class _CameraCache:
    """카메라 1대分 상태 — 전용 HandTracker·주 조작자 손 추적기, ROI 히스테리시스,
    마지막 추론 결과."""

    def __init__(self, config):
        self.tracker = HandTracker(config)
        self.primary_hand = PrimaryHandTracker(config.get("primary_hand") or {})
        self.camera_motion = CameraMotionCompensator(config.get("camera_motion") or {})
        self.roi_box = None
        self.last_hand_box = None
        self.last_seq = -1
        self.frame = None
        self.hands = []
        self.chosen = None
        self.score = 0.0


def _infer_camera(cache, frame, roi_cfg):
    """프레임 1장을 추론하고 cache(roi_box·last_hand_box·hands·chosen)를 갱신한다.

    chosen은 매 프레임 최고 신뢰도 손이 아니라 PrimaryHandTracker가 고른다 —
    배경의 다른 사람 손이 잠깐 신뢰도가 더 높아도 이미 추적 중인(조작자로 보이는)
    손을 계속 우선한다(hand_lock.py 모듈독스트링 참고).
    """
    if roi_cfg.get("enabled", True):
        cache.roi_box = resolve_roi_box(
            cache.roi_box, cache.last_hand_box, frame.shape[1], frame.shape[0], roi_cfg)
    if cache.roi_box is not None:
        x1, y1, x2, y2 = cache.roi_box
        hands = cache.tracker.infer(crop_frame(frame, cache.roi_box))
        for hand in hands:
            hand.landmarks = offset_landmarks_xy(hand.landmarks, x1, y1)
    else:
        hands = cache.tracker.infer(frame)

    cache.hands = hands
    cache.chosen = cache.primary_hand.select(hands, frame.shape[1])
    cache.last_hand_box = (
        hand_bbox_px(cache.chosen.landmarks) if cache.chosen is not None else None)


COLOR_ARMED = (80, 220, 120)
COLOR_LOCKED = (90, 90, 220)
COLOR_NEUTRAL_RING = (200, 200, 200)
COLOR_LOCK_PROGRESS = (60, 210, 245)
ZONE_ARROW_OFFSETS = {"up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
CURSOR_RADIUS_PX = 9


def _clamp_cursor_to_frame(center, offset_px, frame_shape, margin_px=CURSOR_RADIUS_PX):
    """증폭된 커서 중심을 점 전체가 영상 안에 보이는 범위로 제한한다."""
    frame_h, frame_w = frame_shape[:2]
    raw_x = int(center[0] + offset_px[0])
    raw_y = int(center[1] + offset_px[1])
    return (
        min(max(raw_x, margin_px), max(margin_px, frame_w - 1 - margin_px)),
        min(max(raw_y, margin_px), max(margin_px, frame_h - 1 - margin_px)),
    )


def _draw_debug_overlay(frame, active_camera_id, shape, zone, engaged, armed, linear_x,
                        angular_z, offset_px, hand_box, graphic_center_px, lock_progress,
                        center_radius_ratio):
    """디버그 창/영상 스트림용 오버레이 — D-pad 나침반(중앙 원 + 상하좌우 화살표)을
    그리고 현재 판정된 존을 밝게 강조한다. 진단 실패가 판정을 막지 않도록 호출부가 감싼다.

    graphic_center_px: 나침반을 그리는 **화면 고정 위치**(main_dpad.py처럼 재중심·
    캘리브레이션과 무관하게 항상 같은 자리) — neutral_px가 아니다.
    offset_px: DpadController가 cursor_sensitivity를 적용해 증폭한 (dx, dy) —
    이 고정 위치에 더해서 커서를 그린다. 실제 존 판정에 쓰는 값과 동일해
    화면에 보이는 점과 판정이 항상 일치한다.
    """
    status_color = COLOR_ARMED if armed else COLOR_LOCKED

    if graphic_center_px is not None:
        center = (int(graphic_center_px[0]), int(graphic_center_px[1]))
        short_side_px = min(frame.shape[0], frame.shape[1])
        radius_px = max(20, int(center_radius_ratio * short_side_px))

        center_color = status_color if zone == ZONE_CENTER else COLOR_NEUTRAL_RING
        cv2.circle(frame, center, radius_px, center_color,
                  4 if zone == ZONE_CENTER else 2, cv2.LINE_AA)
        if lock_progress > 0.0:
            cv2.ellipse(frame, center, (radius_px + 10,) * 2, -90, 0,
                       360.0 * lock_progress, COLOR_LOCK_PROGRESS, 4, cv2.LINE_AA)

        for name, (ux, uy) in ZONE_ARROW_OFFSETS.items():
            tip = (int(center[0] + ux * radius_px * 2.2), int(center[1] + uy * radius_px * 2.2))
            is_active = zone == name
            cv2.arrowedLine(frame, center, tip, status_color if is_active else COLOR_NEUTRAL_RING,
                           5 if is_active else 2, cv2.LINE_AA, tipLength=0.25)

        if offset_px is not None:
            cursor = _clamp_cursor_to_frame(center, offset_px, frame.shape)
            cv2.line(frame, center, cursor, status_color, 1, cv2.LINE_AA)
            cv2.circle(frame, cursor, CURSOR_RADIUS_PX, status_color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 4, COLOR_NEUTRAL_RING, -1, cv2.LINE_AA)

    if hand_box is not None:
        x1, y1, x2, y2 = (int(v) for v in hand_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2, cv2.LINE_AA)

    lines = [
        # 오버레이 텍스트는 영어 고정 — cv2.putText(Hershey 폰트)가 한글을
        # 못 그리는 OpenCV 빌드가 있어(젯슨 등 플랫폼별 차이), 진단용 텍스트는
        # 플랫폼 무관하게 항상 보이는 ASCII로 통일한다
        "LOCKED - hold OPEN hand at center to unlock" if not armed else "ARMED",
        f"cam={active_camera_id} shape={shape or '-'} zone={zone or '-'} engaged={engaged}",
        f"linear_x={linear_x:+.2f} m/s  angular_z={angular_z:+.2f} rad/s",
    ]
    for line_idx, line in enumerate(lines):
        cv2.putText(frame, line, (10, 26 + 24 * line_idx), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, status_color, 2, cv2.LINE_AA)
    return frame


def run_pipeline(config):
    """파이프라인을 조립해 백그라운드 스레드로 시작하고 PipelineState를 돌려준다."""
    state = PipelineState()
    cameras = MultiCameraManager(config).start()
    state._cameras = cameras
    caches = [_CameraCache(config) for _ in cameras.device_ids]
    state.set_inference_status([cache.tracker for cache in caches])
    dual_cfg = config["dual_camera"]
    arbiter = CameraArbiter(
        dual_cfg["switch_margin"], dual_cfg["switch_frames"],
        sticky_ownership=dual_cfg.get("sticky_ownership", False),
        release_frames=dual_cfg.get("release_frames", 10),
        cooldown_frames=dual_cfg.get("cooldown_frames", 30))
    dpad = DpadController(config["dpad"], config["point_filter"])
    control_mode = (config.get("control") or {}).get("mode", "dpad")
    wave_detector = WaveDetector((config.get("control") or {}).get("wave") or {})
    shape_stabilizer = HandShapeStabilizer(
        config["hand_shape"].get("hold_open_on_unknown_sec", 0.3),
        config["hand_shape"].get("confirm_frames", 2),
        config["hand_shape"].get("confirm_frames_by_shape") or {})
    roi_cfg = config.get("roi_zoom") or {}
    good_span_ratio = config["dual_camera"]["good_hand_span_ratio"]
    is_mirror = config["camera"]["mirror"]
    hand_cfg = config["hand_shape"]
    fallback_w, fallback_h = config["camera"]["width_px"], config["camera"]["height_px"]
    # D-pad 그래픽(나침반)을 그리는 화면 고정 위치 — main_dpad.py의 cx_ratio/cy_ratio와
    # 같은 발상: 재중심·캘리브레이션으로 neutral이 바뀌어도 그래픽 자체는 안 흔들린다
    graphic_cx_ratio = config["dpad"].get("graphic_cx_ratio", 0.5)
    graphic_cy_ratio = config["dpad"].get("graphic_cy_ratio", 0.5)

    state.is_running = True

    def _loop():
        logger.info("제스처 파이프라인 시작 (카메라 %s)", cameras.device_ids)
        previous_active_index = arbiter.active_index
        while state.is_running:
            any_new_frame = False
            for idx, (frame, seq) in enumerate(cameras.read_frames()):
                cache = caches[idx]
                if frame is None or seq == cache.last_seq:
                    continue
                cache.last_seq = seq
                any_new_frame = True
                frame = cv2.flip(frame, 1) if is_mirror else frame
                cache.frame = frame
                cache.camera_motion.update(frame, cache.last_hand_box)
                _infer_camera(cache, frame, roi_cfg)
                cache.score = score_hand_quality(cache.hands, frame.shape[1], good_span_ratio)

            active_index = arbiter.update(
                [cache.score for cache in caches],
                [cache.chosen is not None for cache in caches])
            if active_index != previous_active_index:
                # 서로 다른 카메라의 픽셀 좌표를 같은 필터/중립점에 이어 붙이면 커서가
                # 순간이동한다. 전환 프레임에서 새 좌표계를 즉시 center로 재보정한다.
                dpad.reset_tracking()
                shape_stabilizer.reset()
                previous_active_index = active_index
            active = caches[active_index]

            fingertip_px = None
            shape = None
            confidence = 0.0
            if active.chosen is not None:
                tip = active.chosen.landmarks[INDEX_FINGERTIP_LMK]
                fingertip_px = active.camera_motion.compensate(
                    (float(tip[0]), float(tip[1])))
                shape = classify_hand_shape(
                    active.chosen.world_landmarks, hand_cfg["extend_ratio"],
                    hand_cfg["min_valid_fingers"], hand_cfg["curl_confirm_ratio"])
                if control_mode == "finger_commands":
                    shape = classify_finger_command(
                        active.chosen.world_landmarks, active.chosen.landmarks,
                        active.chosen.user_side, hand_cfg["extend_ratio"],
                        hand_cfg["curl_confirm_ratio"])
                confidence = active.chosen.conf

            now_sec = time.monotonic()
            if active.frame is not None:
                frame_h, frame_w = active.frame.shape[:2]
            else:
                frame_w, frame_h = fallback_w, fallback_h

            wave_fired = False
            if (control_mode == "finger_commands" and active.chosen is not None
                    and shape == "open"):
                wave_fired = wave_detector.update(
                    active.chosen.landmarks, frame_w, now_sec)
                if wave_detector.in_progress:
                    shape = None
            # 흔드는 중 빠른 움직임 때문에 한두 프레임 검출이 빠져도 누적 왕복을
            # 유지한다. WaveDetector 자체 timeout을 넘겼을 때만 자동 초기화된다.

            shape = shape_stabilizer.update(
                shape, active.chosen is not None, now_sec)
            if wave_fired:
                shape = "wave"

            if control_mode == "dpad":
                linear_x, angular_z, engaged = dpad.update(
                    fingertip_px, shape, frame_w, frame_h, now_sec)
            else:
                linear_x, angular_z, engaged = 0.0, 0.0, False

            state.update(
                linear_x=linear_x, angular_z=angular_z,
                hand_detected=active.chosen is not None, engaged=engaged,
                armed=dpad.armed, shape=shape, zone=dpad.zone,
                active_camera=cameras.device_ids[active_index],
                confidence=confidence,
            )

            # 개별 /camera_feed/<index>에도 모든 카메라의 D-pad가 보이게 한다.
            # 제어 판정은 가장 품질이 좋은 active 카메라 하나만 사용하지만, 웹 진단
            # 화면에서는 두 카메라의 구도와 손 위치를 동시에 확인할 수 있어야 한다.
            for idx, cache in enumerate(caches):
                if cache.frame is None:
                    continue
                camera_frame = cache.frame
                if state.has_viewer:
                    try:
                        graphic_center_px = (graphic_cx_ratio * camera_frame.shape[1],
                                             graphic_cy_ratio * camera_frame.shape[0])
                        camera_frame = _draw_debug_overlay(
                            camera_frame.copy(), cameras.device_ids[idx], shape,
                            dpad.zone, engaged, dpad.armed, linear_x, angular_z,
                            dpad.offset_px if idx == active_index else None,
                            cache.last_hand_box, graphic_center_px,
                            dpad.debug.get("lock_progress", 0.0),
                            config["dpad"].get("center_radius_ratio", 0.10))
                    except Exception:   # noqa: BLE001 — 진단 화면 실패가 판정을 막으면 안 된다
                        logger.exception("카메라 %s 오버레이 그리기 실패", cameras.device_ids[idx])
                        camera_frame = cache.frame
                state.update_camera_frame(idx, camera_frame, cache.last_seq)

            if active.frame is not None:
                # /video_feed를 보는 사람이 없으면 오버레이 그리기(복사+cv2 호출)를
                # 통째로 건너뛴다 — 판정 경로는 이미 끝났으므로 무영향, 시청자가
                # 없는 동안의 순수 낭비만 없앤다 (add_viewer/remove_viewer는 server/app.py).
                if not state.has_viewer:
                    state.update_frame(active.frame)
                else:
                    try:
                        graphic_center_px = (graphic_cx_ratio * active.frame.shape[1],
                                             graphic_cy_ratio * active.frame.shape[0])
                        overlay = _draw_debug_overlay(
                            active.frame.copy(), cameras.device_ids[active_index], shape,
                            dpad.zone, engaged, dpad.armed, linear_x, angular_z, dpad.offset_px,
                            active.last_hand_box, graphic_center_px,
                            dpad.debug.get("lock_progress", 0.0),
                            config["dpad"].get("center_radius_ratio", 0.10))
                        state.update_frame(overlay)
                    except Exception:   # noqa: BLE001 — 디버그 화면 실패가 판정을 막으면 안 된다
                        logger.exception("디버그 오버레이 그리기 실패 — 판정은 계속 진행")
                        state.update_frame(active.frame)

            if not any_new_frame:
                time.sleep(LOOP_IDLE_SLEEP_SEC)

    threading.Thread(target=_loop, daemon=True).start()
    return state
