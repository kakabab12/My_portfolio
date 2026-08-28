"""control 모듈 — 손끝 위치·손모양을 방향 존(위/아래/좌/우/중앙)으로 판정해
그 존에 대응하는 **고정 속도**의 로봇 명령(linear_x, angular_z)을 내는 D-pad
방식 원격조종. gesture_kiosk의 main_dpad.py가 쓰던 5존 판정(중심 반경 안이면
center, 그 밖은 각도로 상/하/좌/우 — 축 우세 미달이면 애매)을 그대로 쓰되,
main_dpad.py는 애매하면 "직전 방향 유지"였던 것과 달리 여기서는 **정지**로
처리한다 — 로봇 구동은 불확실할 때 도는 것보다 멈추는 쪽이 안전하다.

존을 가리키는 동안 계속 그 방향으로 움직이고(리모컨 버튼처럼), 존을 벗어나거나
손을 놓치면 그 프레임부터 즉시 멈춘다 — dwell(정해진 시간 채우기)로 "한 번"
발화하는 방식이 아니다.

기준점(neutral)은 main_dpad.py와 동일하게 **손이 처음 나타난 자리**로 즉시
캘리브레이션된다(화면 중앙일 필요 없음 — 조작자가 어디 서든 그 자리가 0점).
그 뒤 손을 한동안 가만히 두면(recenter) 그 자리가 새 기준으로 재조정된다.
offset_px(손끝 오프셋에 cursor_sensitivity를 곱해 증폭한 값)를 존 판정에 쓴다
(main_dpad.py의 CURSOR_SENSITIVITY와 동일 발상: 팔을 크게 휘두르지 않아도
존에 닿게 한다). ★D-pad 그래픽 자체는 화면에서 움직이지 않는다 — 오버레이가
고정된 화면 위치에 그리고 offset_px만 그 위에 얹어 커서를 그린다(neutral_px가
재조정돼도 그래픽은 안 흔들린다). offset_px는 판정에 실제로 쓰는 값이라
그림과 판정이 항상 일치한다.

주행 잠금 토글: center 존에서 engage_shape로 hold_sec만큼 머물면 armed가
반전된다(D-pad 중앙 버튼의 PLAY/PAUSE에 대응). armed가 False면 손이 무엇을
하든 항상 정지.
"""
import math

from src.postprocess.point_filter import PointFilter
from src.utils.logger import get_logger

logger = get_logger("control")

ZONE_UP = "up"
ZONE_DOWN = "down"
ZONE_LEFT = "left"
ZONE_RIGHT = "right"
ZONE_CENTER = "center"

# ROS REP103 관례: angular_z 양수 = 반시계(좌회전).
_ZONE_DIRECTION = {
    ZONE_UP: (1.0, 0.0),      # 전진
    ZONE_DOWN: (-1.0, 0.0),   # 후진
    ZONE_LEFT: (0.0, 1.0),    # 좌회전
    ZONE_RIGHT: (0.0, -1.0),  # 우회전
}


def classify_zone(dx_px, dy_px, center_radius_px, axis_dominance):
    """중심 기준 오프셋(px) -> "up"|"down"|"left"|"right"|"center"|None(애매/정지).

    거리가 아니라 각도로 상/하/좌/우를 가른다 — 중심 반경만 벗어나면 손이
    버튼까지 정확히 안 닿아도 그 방향을 향하기만 하면 인식된다. 대각선처럼
    두 축이 비슷하면(축 우세 미달) None — 호출부가 정지로 처리한다.
    """
    if math.hypot(dx_px, dy_px) <= center_radius_px:
        return ZONE_CENTER
    abs_dx, abs_dy = abs(dx_px), abs(dy_px)
    if abs_dx >= abs_dy * axis_dominance:
        return ZONE_RIGHT if dx_px > 0 else ZONE_LEFT
    if abs_dy >= abs_dx * axis_dominance:
        return ZONE_DOWN if dy_px > 0 else ZONE_UP
    return None


def zone_velocity(zone, cfg):
    """존 -> (linear_x, angular_z). center/None/미지정 존은 (0, 0)."""
    linear_sign, angular_sign = _ZONE_DIRECTION.get(zone, (0.0, 0.0))
    return linear_sign * cfg["linear_mps"], angular_sign * cfg["angular_radps"]


class DpadController:
    """프레임마다 update()를 호출해 (linear_x, angular_z, engaged)를 얻는 상태기.

    armed(주행 잠금 해제 여부)·zone(현재 판정 존)·neutral_px·offset_px는 공개
    속성으로 노출 — 상태 표시·디버그 오버레이용.
    """

    def __init__(self, cfg, point_filter_cfg):
        self._cfg = cfg
        self._fingertip_filter = PointFilter(
            point_filter_cfg["min_cutoff_hz"], point_filter_cfg["beta"],
            point_filter_cfg["d_cutoff_hz"])
        self._neutral_px = None
        self._still_anchor_px = None
        self._still_start_sec = None
        self._still_fired = False
        self._had_hand_prev = False
        self._v_recenter_latched = False

        lock_cfg = cfg.get("lock_toggle") or {}
        # 기본값(config에 키 자체가 없을 때)도 안전 쪽으로 — 로봇은 잠금 상태로
        # 시작해 조작자가 한 번은 명시적으로 풀어야 움직인다
        self.armed = (True if not lock_cfg.get("enabled", True)
                      else lock_cfg.get("start_armed", False))
        self._lock_start_sec = None
        self._lock_fired = False

        self.neutral_px = None   # 시각화용 — update() 후 항상 현재 기준점을 담는다
        self.offset_px = None    # 시각화·판정 공용 — 증폭 적용된 (dx, dy) 오프셋 벡터.
                                 #   화면 표시는 호출부가 원하는 고정 지점에 이 벡터를
                                 #   더해 그린다(D-pad 그래픽이 화면에서 안 흔들리게)
        self.zone = None         # 시각화용 — 이번 프레임 판정 존
        self.debug = {"lock_progress": 0.0}

    def reset_tracking(self):
        """카메라/조작자가 바뀐 다음 좌표계를 새 손 위치에서 다시 시작한다."""
        self._fingertip_filter.reset()
        self._neutral_px = None
        self._still_anchor_px = None
        self._still_start_sec = None
        self._still_fired = False
        self._had_hand_prev = False
        self._v_recenter_latched = False
        self.neutral_px = None
        self.offset_px = None
        self.zone = None

    def update(self, fingertip_px, shape, frame_width_px, frame_height_px, now_sec):
        """fingertip_px: (x_px, y_px) | None(손 없음). shape: classify_hand_shape 결과."""
        if fingertip_px is None:
            self._had_hand_prev = False
            self._v_recenter_latched = False
            self._still_anchor_px = None
            self._lock_start_sec = None
            self.zone = None
            self.offset_px = None
            self.debug["lock_progress"] = 0.0
            return 0.0, 0.0, False

        is_new_appearance = not self._had_hand_prev
        if is_new_appearance:
            self._fingertip_filter.reset()
        self._had_hand_prev = True

        filtered_px = self._fingertip_filter.filter(fingertip_px, now_sec)
        if is_new_appearance:
            # main_dpad.py와 동일 — 손이 새로 나타날 때마다(최초 포함) 그 자리를
            # 곧바로 기준(0점)으로 캘리브레이션한다. 화면 중앙에 서 있을 필요가 없다
            self._neutral_px = filtered_px

        if self._update_v_sign_recenter(shape):
            self._neutral_px = filtered_px
            self._still_anchor_px = filtered_px
            self._still_start_sec = now_sec
            self._still_fired = False

        short_side_px = min(frame_width_px, frame_height_px)
        recenter_cfg = self._cfg.get("recenter") or {}
        if recenter_cfg.get("enabled", True):
            self._update_still_recenter(filtered_px, recenter_cfg, now_sec, short_side_px)

        self.neutral_px = self._neutral_px
        sensitivity = self._cfg.get("cursor_sensitivity", 3.0)
        dx_px = (filtered_px[0] - self._neutral_px[0]) * sensitivity
        dy_px = (filtered_px[1] - self._neutral_px[1]) * sensitivity
        if self._cfg.get("invert_x"):
            dx_px = -dx_px
        if self._cfg.get("invert_y"):
            dy_px = -dy_px
        self.offset_px = (dx_px, dy_px)
        center_radius_px = self._cfg.get("center_radius_ratio", 0.10) * short_side_px
        axis_dominance = self._cfg.get("axis_dominance", 1.3)
        zone = classify_zone(dx_px, dy_px, center_radius_px, axis_dominance)
        self.zone = zone

        lock_cfg = self._cfg.get("lock_toggle") or {}
        lock_toggle_enabled = lock_cfg.get("enabled", True)
        if lock_toggle_enabled:
            self._update_lock_toggle(zone, shape, lock_cfg, now_sec)

        linear_x, angular_z = zone_velocity(zone, self._cfg)

        # lock_toggle이 꺼져 있으면 잠금 기능 자체가 없는 것 — armed 값과 무관하게
        # 항상 통과시킨다(다른 enabled 플래그들과 같은 관례: 꺼짐 = 그 기능 무영향)
        if lock_toggle_enabled and not self.armed:
            return 0.0, 0.0, False
        if shape != self._cfg.get("engage_shape", "open"):
            return 0.0, 0.0, False
        if zone in (None, ZONE_CENTER):
            return 0.0, 0.0, False
        return linear_x, angular_z, True

    def _update_v_sign_recenter(self, shape):
        """V 자세가 새로 나타난 첫 프레임에만 재중심을 발화한다."""
        cfg = self._cfg.get("v_sign_recenter") or {}
        if not cfg.get("enabled", False):
            return False
        if shape != "v":
            self._v_recenter_latched = False
            return False
        if self._v_recenter_latched:
            return False
        self._v_recenter_latched = True
        logger.info("V 제스처 재중심")
        return True

    def _update_still_recenter(self, point_px, recenter_cfg, now_sec, short_side_px):
        # 기본값은 main_dpad.py의 STILL_RADIUS_RATIO(0.035)·STILL_RECALIBRATE_SEC(2.5)와 동일
        radius_px = recenter_cfg.get("still_radius_ratio", 0.035) * short_side_px
        still_sec = recenter_cfg.get("still_sec", 2.5)
        if (self._still_anchor_px is None
                or math.hypot(point_px[0] - self._still_anchor_px[0],
                              point_px[1] - self._still_anchor_px[1]) > radius_px):
            self._still_anchor_px = point_px
            self._still_start_sec = now_sec
            self._still_fired = False
        elif not self._still_fired and now_sec - self._still_start_sec >= still_sec:
            self._neutral_px = point_px
            self._still_fired = True

    def _update_lock_toggle(self, zone, shape, lock_cfg, now_sec):
        """center 존에서 engage_shape로 hold_sec만큼 머물면 armed를 반전한다.

        center 존 판정 자체(classify_zone)가 이미 "중심 반경 안"을 걸러주므로,
        옛 joystick.py처럼 별도 반경 검사를 다시 할 필요가 없다 — D-pad로
        바꾸며 단순해진 부분.
        """
        hold_sec = lock_cfg.get("hold_sec", 1.2)
        required_shape = lock_cfg.get("require_shape", "open")
        is_candidate = zone == ZONE_CENTER and shape == required_shape
        if not is_candidate:
            self._lock_start_sec = None
            self._lock_fired = False
            self.debug["lock_progress"] = 0.0
            return

        if self._lock_start_sec is None:
            self._lock_start_sec = now_sec
            self._lock_fired = False
        elapsed_sec = now_sec - self._lock_start_sec
        self.debug["lock_progress"] = 0.0 if self._lock_fired else min(1.0, elapsed_sec / hold_sec)
        if not self._lock_fired and elapsed_sec >= hold_sec:
            self.armed = not self.armed
            self._lock_fired = True
            logger.info("주행 잠금 토글 -> armed=%s", self.armed)
