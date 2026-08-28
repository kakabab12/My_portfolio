"""postprocess 모듈 — 로터(리모컨) 모드: 위 쓸기로 켜고 끄는 손 다이얼 단축키.

2026-08-05 신설 (feat/rotor_remote 판, 사용자 결정):
- 일반 모드: 펼친 손 좌/우 쓸기(left/right)만 파이프로 나간다 — 나머지 칸은
  같은 날 비활성 (gesture_filter.EVENT_BY_SHAPE 주석).
- 펼친 손 **위 쓸기(up)** = 설정 토글: 로터 모드를 켜고, 다시 위 쓸기로 끈다.
  up 자체는 파이프로 나가지 않는다(모드 스위치로 소비).
- 로터 모드 중에는 토글(up)을 뺀 모든 쓸기 이벤트를 무시한다.

★판정 원리 확정 이력(같은 날 사용자 확인 8차): ①가리키고 머무름 → ②90도
스텝 → ③회전 누적 바늘 → ④버튼 원판 회전 → ⑤고정 버튼+노브 → ⑥검지 절대
각도+차징 → ⑦검지 조준+주먹 확정 → ⑧**펼친 손 다이얼(기아 자동차 다이얼
방식)** ("검지 제어 말고 — 다 펼친 손을 오른쪽으로 제자리 회전하면 오른쪽,
왼쪽으로 회전하면 왼쪽, 선택은 주먹"). 현재 방식:

- **조작 자세 = 펼친 손**: 손 방향각 = 손목(랜드마크 0) → 중지 뿌리(랜드마크
  9) 벡터의 시계 각도 — 검지 끝(⑦까지)보다 기선이 길고 관절 판별이 안 끼어
  훨씬 안정적이다. 펼친 손을 제자리에서 비틀면 이 각도가 돈다.
- **래칫 스텝(사용자 확인 — 손목은 무한으로 못 돈다)**: 중립(기준) 대비
  step_threshold_deg 이상 비틀면 포커스가 1칸 이동(오른쪽 비틀기=시계 다음,
  왼쪽=이전), **중립 근처로 되돌아와야**(threshold × rearm_ratio 미만) 다음
  칸이 장전된다 — 자동차 다이얼의 찰칵 감각. 비튼 채 유지해도 1칸이다.
- **중립 지속 재중심(recenter_alpha)**: 장전 상태로 중립 근처에 있는 동안
  중립이 현재 각도로 천천히 따라간다 — 손 자세가 흘러도 기준이 함께 흘러
  좌/우 비대칭이 안 생긴다(사용자가 제안한 "지속적 캘리브레이션"의 구현).
  펼친 손이 새로 나타날 때(등장·주먹 확정 후 다시 폄)마다 그 각도로 재캡처.
- **확정 = 주먹 쥐기**: 펼친 손을 본 뒤 주먹으로 쥐면(연속 fist_confirm_frames
  프레임 판별 — 블러 프레임은 계수 유지) 현재 포커스 버튼이 발화한다.
  재발화는 다시 펼친 손으로 돌아간 뒤에만(쥔 채 유지 = 1회). 펼친 손을 본 적
  없는 주먹(휴식 손)은 발화하지 않는다.
- **머무름(차징) 없음**: ⑦에서 이미 제거 — 확정은 주먹뿐이다.
- **입력 손 1개 제한(사용자 확인)**: 로터 입력은 hand_select의 **추적 손만**
  쓴다(후보 폴백 없음 — 끊기면 대기: 포커스 유지·발화 금지).
- **버튼 확장(사용자 요구 — "4개에서 9개로")**: 버튼은 config rotor.buttons
  목록 — 항목을 추가하면 포커스 순환·UI가 자동으로 N개를 다룬다.
- 시작 유예(start_grace_sec): 켜짐 직후에는 스텝·발화하지 않고, 유예 동안의
  손 방향이 중립으로 잡힌다 (2026-08-05 실기 로그 10:38:28 — 복귀 팔 오발).
- ⑦까지의 검지 전용 장치(One Euro 필터·angle_gain·조준 게이트)는 소멸 —
  손목→중지 뿌리 기선은 필터 없이도 각도가 안정적이다.
수치는 전부 config rotor 섹션 (기획서 4.7 — 숫자 하드코딩 금지).

화면 표시는 src/ui/rotor_window.py(tkinter 오버레이)가 snapshot()으로 읽어
그린다 — 판정(추론 스레드)과 그리기(UI 스레드)가 달라 스냅숏은 락으로 지킨다.
"""
import math
import threading
import time

from src.postprocess.gesture_filter import GestureEvent
from src.postprocess.hand_shape import SHAPE_FIST, SHAPE_OPEN, classify_hand_shape
from src.utils.logger import get_logger

logger = get_logger("postprocess")

WRIST_LMK = 0        # MediaPipe 21점 중 손목 (hand_tracker.py 독스트링)
MIDDLE_MCP_LMK = 9   # 중지 뿌리 — 손목과 함께 손의 장축(방향각)을 이룬다


def clock_angle_deg(from_px, to_px):
    """from -> to 벡터의 시계 각도(0=12시·시계방향+, 화면 y는 아래로 증가).

    위 = (0,-1)이 0도, 오른쪽 = (1,0)이 90도가 되도록 atan2(dx, -dy)를 쓴다.
    순수 함수 — tests/test_rotor.py가 4방위 실좌표로 검증.
    """
    return math.degrees(math.atan2(to_px[0] - from_px[0], -(to_px[1] - from_px[1])))


class RotorController:
    """로터 모드 상태기 — 토글·펼친 손 래칫·주먹 확정 판정을 맡고 확정 이벤트를 돌려준다."""

    def __init__(self, config, frame_width_px, frame_height_px, clock=time.monotonic):
        rotor_cfg = config["rotor"]
        self._toggle_event = rotor_cfg["toggle_event"]
        # 버튼 목록 — 포커스 순환 순서(오른쪽 비틀기 = 다음). 항목 추가 = 버튼 추가
        self._buttons = [dict(button) for button in rotor_cfg["buttons"]]
        if not self._buttons:
            raise ValueError("rotor.buttons가 비어 있다 — 버튼이 최소 1개 필요")
        self._step_threshold_deg = rotor_cfg["step_threshold_deg"]
        self._rearm_ratio = rotor_cfg["rearm_ratio"]
        self._recenter_alpha = rotor_cfg["recenter_alpha"]
        self._fist_confirm_frames = rotor_cfg["fist_confirm_frames"]
        self._flash_sec = rotor_cfg["flash_sec"]
        self._start_grace_sec = rotor_cfg["start_grace_sec"]
        # 손 모양 판별 임계 — 본 엔진 판별(hand_select.hand_shape)과 같은 값 재사용:
        # 조작(펼친 손)·확정(주먹)이 화면·쓸기와 같은 기준으로 읽히게
        shape_cfg = config["hand_select"]["hand_shape"]
        self._extend_ratio = shape_cfg["extend_ratio"]
        self._min_valid_fingers = shape_cfg["min_valid_fingers"]
        self._curl_confirm_ratio = shape_cfg["curl_confirm_ratio"]
        self._clock = clock
        # 프레임 크기는 판정에 안 쓴다(각도만 본다) — 생성자 계약 유지용
        _ = (frame_width_px, frame_height_px)

        self.is_rotor_on = False
        self._grace_until_sec = 0.0
        self._neutral_deg = None      # 중립 방향 — 래칫 비틀림의 기준 (지속 재중심)
        self._twist_deg = None        # 중립 대비 비틀림 각 | None(펼친 손 아님)
        self._focus_idx = 0           # 현재 포커스 버튼 — 래칫 스텝으로만 움직인다
        self._is_step_armed = True    # 스텝 장전 — 중립 복귀로만 재장전(찰칵 1칸)
        self._was_open = False        # 직전 프레임이 펼친 손이었나 — 재등장 재캡처용
        self._is_open_seen = False    # 펼친 손을 본 뒤인가 — 주먹 확정의 전제
        self._fist_streak = 0         # 주먹 판별 연속 프레임 수 — 한 프레임 오판 방어
        self._flash_name = None
        self._flash_start_sec = 0.0

        self._snapshot_lock = threading.Lock()
        self._snapshot = {"is_on": False}

    # ----- 파이프라인 접점 -----

    def update(self, gesture_event, hand_selector):
        """쓸기 이벤트와 손 관측을 반영 -> 파이프로 보낼 이벤트 | None.

        - 토글 이벤트(up)는 여기서 소비된다 — 파이프로 나가지 않는다.
        - 로터가 꺼져 있으면 나머지 이벤트(left/right)는 그대로 통과.
        - 로터가 켜져 있으면 토글 외 쓸기는 전부 무시(사용자 결정)하고,
          펼친 손 래칫 + 주먹 확정 판정만 이벤트가 된다.
        """
        now_sec = self._clock()
        if gesture_event is not None and gesture_event.class_name == self._toggle_event:
            self._toggle()
            gesture_event = None
        elif self.is_rotor_on and gesture_event is not None:
            logger.info("로터 모드 중 쓸기 무시: %s", gesture_event.class_name)
            gesture_event = None

        if not self.is_rotor_on:
            self._write_snapshot(now_sec)
            return gesture_event

        return self._update_rotor(hand_selector, now_sec)

    def _toggle(self):
        self.is_rotor_on = not self.is_rotor_on
        self._reset_rotor_state()
        if self.is_rotor_on:
            # 켜진 직후 시작 유예 — 토글(위 쓸기) 복귀 동작이 스텝·확정으로
            # 오발되지 않게 + 유예 동안의 손 방향이 중립으로 잡힌다
            self._grace_until_sec = self._clock() + self._start_grace_sec
        logger.info("로터 모드 %s (위 쓸기 토글)", "켜짐" if self.is_rotor_on else "꺼짐")

    def _reset_rotor_state(self):
        """모드 전환 — 이전 세션의 중립·포커스·확정 상태를 승계하지 않는다."""
        self._neutral_deg = None
        self._twist_deg = None
        self._focus_idx = 0
        self._is_step_armed = True
        self._was_open = False
        self._is_open_seen = False
        self._fist_streak = 0
        self._flash_name = None

    # ----- 다이얼 판정 -----

    def _update_rotor(self, hand_selector, now_sec):
        """다이얼 1프레임 판정 -> 확정 GestureEvent | None."""
        hand = hand_selector.tracked_hand()
        if hand is None:
            # ★추적 손 1개 제한(사용자 확인 — 모듈 독스트링): 폴백 없이 **대기**.
            # 포커스 표시는 유지하고, 주먹 연속 계수는 끊는다
            self._fist_streak = 0
            self._was_open = False
            self._twist_deg = None
            self._write_snapshot(now_sec)
            return None

        shape = classify_hand_shape(hand.world_landmarks, self._extend_ratio,
                                    self._min_valid_fingers, self._curl_confirm_ratio)
        if shape == SHAPE_OPEN:
            self._fist_streak = 0
            self._is_open_seen = True
            self._update_ratchet(hand, now_sec)
            self._was_open = True
            self._write_snapshot(now_sec)
            return None

        # 펼친 손이 아니다 — 주먹이면 확정 계수, 그 외(검지·불명)는 대기.
        # 판별 불가(None)는 계수를 유지만 한다 — 쥐는 도중 블러 한두 프레임이
        # 연속 조건을 끊어 확정이 씹히는 것 방지 (shape_latch와 같은 취지)
        self._was_open = False
        self._twist_deg = None
        if shape == SHAPE_FIST:
            self._fist_streak += 1
        elif shape is not None:
            self._fist_streak = 0

        event_name = self._resolve_fist_fire(now_sec)
        self._write_snapshot(now_sec)
        if event_name is None:
            return None
        return GestureEvent(class_name=event_name, conf=1.0, ts_sec=now_sec)

    def _update_ratchet(self, hand, now_sec):
        """펼친 손 프레임 — 비틀림 각 갱신 + 래칫 스텝 판정."""
        landmarks = hand.landmarks
        wrist_px = (float(landmarks[WRIST_LMK][0]), float(landmarks[WRIST_LMK][1]))
        mcp_px = (float(landmarks[MIDDLE_MCP_LMK][0]),
                  float(landmarks[MIDDLE_MCP_LMK][1]))
        angle_deg = clock_angle_deg(wrist_px, mcp_px)
        if (self._neutral_deg is None or not self._was_open
                or now_sec < self._grace_until_sec):
            # 중립 재캡처 — 최초·재등장(주먹 확정 뒤 다시 폄·소실 복귀)·유예 중:
            # 지금 이 자세가 기준이다. 이전 자세의 비틀림을 승계하지 않는다
            self._neutral_deg = angle_deg
            self._is_step_armed = True
        self._twist_deg = self._wrap180_deg(angle_deg - self._neutral_deg)
        if now_sec < self._grace_until_sec:
            return   # 유예 — 스텝 판정 없음 (중립만 따라간다)

        rearm_deg = self._step_threshold_deg * self._rearm_ratio
        if self._is_step_armed:
            if self._twist_deg >= self._step_threshold_deg:
                self._step_focus(1, now_sec)     # 오른쪽 비틀기 = 다음(시계)
            elif self._twist_deg <= -self._step_threshold_deg:
                self._step_focus(-1, now_sec)    # 왼쪽 비틀기 = 이전
            else:
                # 장전 + 중립 근처 — 중립을 현재 각도로 천천히 따라가게 한다
                # (지속 재중심 — 손 자세가 흘러도 좌/우 문턱이 비대칭이 안 되게)
                if abs(self._twist_deg) < rearm_deg:
                    self._neutral_deg += self._recenter_alpha * self._twist_deg
                    self._neutral_deg = self._wrap180_deg(self._neutral_deg)
        elif abs(self._twist_deg) < rearm_deg:
            self._is_step_armed = True   # 중립 복귀 — 다음 찰칵 장전

    def _step_focus(self, direction_step, now_sec):
        self._focus_idx = (self._focus_idx + direction_step) % len(self._buttons)
        self._is_step_armed = False   # 되돌아와야 다음 칸 — 비튼 채 유지는 1칸
        logger.info("로터 포커스: %s (%s 비틀기)",
                    self._buttons[self._focus_idx]["name"],
                    "오른쪽" if direction_step > 0 else "왼쪽")
        _ = now_sec   # 스텝 시각은 현재 미사용 — 연속 이동 방식 도입 시 기준점

    def _resolve_fist_fire(self, now_sec):
        """주먹 확정 판정 -> 발화할 이벤트명 | None.

        조건 3중: ①펼친 손을 본 뒤(_is_open_seen — 휴식 주먹 오발 방지)
        ②주먹 판별이 연속 확인됨(한 프레임 오판 방어) ③시작 유예 밖.
        발화 후 표식을 내린다 — 재발화는 다시 펼친 손으로 돌아간 뒤에만.
        """
        if (not self._is_open_seen
                or self._fist_streak < self._fist_confirm_frames
                or now_sec < self._grace_until_sec):
            return None
        self._is_open_seen = False
        self._fist_streak = 0
        button = self._buttons[self._focus_idx]
        self._flash_name = button["name"]
        self._flash_start_sec = now_sec
        logger.info("로터 발화: %s -> %s (주먹 확정)", button["name"], button["event"])
        return button["event"]

    @staticmethod
    def _wrap180_deg(angle_deg):
        """각도를 [-180, 180) 범위로 접는다 — 중립 대비 비틀림의 부호 있는 표현."""
        return (angle_deg + 180.0) % 360.0 - 180.0

    # ----- UI 스냅숏 -----

    def _write_snapshot(self, now_sec):
        flash_elapsed_sec = now_sec - self._flash_start_sec
        is_flashing = (self._flash_name is not None
                       and flash_elapsed_sec < self._flash_sec)
        with self._snapshot_lock:
            self._snapshot = {
                "is_on": self.is_rotor_on,
                "twist_deg": self._twist_deg,   # 중립 대비 비틀림 | None(펼친 손 아님)
                "focus_name": self._buttons[self._focus_idx]["name"],
                "is_step_armed": self._is_step_armed,
                "flash_name": self._flash_name if is_flashing else None,
                "flash_progress": (1.0 - flash_elapsed_sec / self._flash_sec
                                   if is_flashing else 0.0),
            }

    def snapshot(self):
        """UI 스레드용 상태 사본 — 락 보호 (판정 상태는 건드리지 못한다)."""
        with self._snapshot_lock:
            return dict(self._snapshot)
