"""postprocess 모듈 — 손 관측(HandObservation)을 동작 이벤트로 확정한다.

동작 하나만 쓰기로 단순화(2026-07 결정): 손등이 카메라를 향한 채로
stable_frame_count 프레임 연속 유지되면 이벤트 하나(next_item)를 확정한다.
어느 손이든 상관없다. 이전에 있던 select/pause_voice/cancel/go_home/sos_call/
prev_item 등 다른 동작들은 전부 제거했다 — 필요해지면 git 히스토리에서 복구.

이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지).
모든 수치는 config에서 읽는다.
"""
import time
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

BACK_HAND_ARM = "손등팔등"


@dataclass
class GestureEvent:
    """확정된 동작 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float
    hand_side: str = None
    data: dict = None


class _StableTracker:
    """정적 상태 유지 판정 — 같은 제스처 N프레임 연속 + 거의 제자리."""

    def __init__(self, stable_frame_count, max_move_ratio):
        self._stable_frame_count = stable_frame_count
        self._max_move_ratio = max_move_ratio
        self._gesture = None
        self._count = 0
        self._start_cx_ratio = None

    def update(self, gesture, cx_ratio):
        """관측 1건을 반영하고, 유지 확정이면 True."""
        is_same_run = gesture == self._gesture and (
            abs(cx_ratio - self._start_cx_ratio) <= self._max_move_ratio
        )
        if is_same_run:
            self._count += 1
        else:
            self._gesture = gesture
            self._count = 1
            self._start_cx_ratio = cx_ratio
        if self._count >= self._stable_frame_count:
            self.reset()
            return True
        return False

    def reset(self):
        self._gesture = None
        self._count = 0
        self._start_cx_ratio = None


class GestureFilter:
    def __init__(self, config, clock=time.monotonic):
        gestures = config["gestures"]
        self._cooldown_sec = config["detect"]["cooldown_sec"]
        self._clock = clock

        next_item = gestures["next_item"]
        self._tracker = _StableTracker(
            next_item["stable_frame_count"], next_item["max_static_move_ratio"]
        )

        self._last_event_ts_sec = None

    def filter_observations(self, observations, raised=None, raised_high=None):
        """observations(손모양 관측 목록) -> gesture_event | None.

        raised/raised_high는 더 이상 쓰지 않지만(예전 손 들기 판정 제거), 호출부
        시그니처 호환을 위해 인자만 받아두고 무시한다.
        """
        now_sec = self._clock()
        if self._is_in_cooldown(now_sec):
            return None

        for obs in observations:
            fired = self._tracker.update(obs.gesture, obs.cx_ratio)
            if fired and obs.gesture == BACK_HAND_ARM:
                return self._confirm("next_item", obs.conf, now_sec, hand_side=obs.side)
        return None

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _confirm(self, class_name, conf, now_sec, hand_side=None, data=None):
        self._last_event_ts_sec = now_sec
        self._tracker.reset()
        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
