"""postprocess 모듈 — 여러 손 후보 중 "그 조작자"의 손 하나만 계속 추적한다.

붐비는 공간에서 배경의 다른 사람 손이 함께 잡혀도 반응하지 않도록:
1) 너무 작은(먼) 손은 애초에 후보에서 제외 — 조작자는 카메라 가까이서
   손을 들어 보여준다고 가정한다. 배경 사람 손은 대개 더 멀어(작아) 걸러진다.
2) 이미 추적 중인 손이 있으면, 신뢰도가 더 높은 다른 손이 나타나도 **공간
   연속성**(마지막 위치에서 가까운 쪽)을 우선한다 — 매 프레임 최고 신뢰도
   손을 다시 고르면 배경 사람에게 매 프레임 옮겨갈 수 있다.
3) 추적 손이 잠깐(release_sec 이내) 안 보여도 정체성을 유지한다(가림·모션
   블러 흡수). 그보다 오래 안 보이면 정체성을 놓고, 다음은 다시 크기 기준
   1순위 후보로 새로 잡는다.
4) 아직 추적 중인 손이 없을 때는 같은 위치의 후보가 여러 프레임 연속 보여야
   새 손으로 획득한다. 얼굴·책상 무늬가 한 프레임만 손처럼 보이는 오탐은 이
   단계에서 버린다.

gesture_kiosk의 hand_select.py(포즈 기반 머리 앵커 게이트) 중 "정체성 연속
추적" 아이디어만 가져오고, 별도 포즈 모델이 필요한 머리 앵커 게이트는 뺐다
— 지금은 크기+연속성만으로 충분하고, 부족하면 나중에 확장한다.
"""
import math
import time

from src.postprocess.hand_shape import hand_center_point, hand_span_px


class PrimaryHandTracker:
    """select(hands, frame_width_px) -> 추적 대상 HandDetection | None."""

    def __init__(self, cfg, clock=time.monotonic):
        self._min_span_ratio = cfg.get("min_span_ratio", 0.05)
        self._continuity_span_ratio = cfg.get("continuity_span_ratio", 3.5)
        self._release_sec = cfg.get("release_sec", 1.5)
        self._acquire_frames = max(1, int(cfg.get("acquire_frames", 1)))
        self._acquire_center_span_ratio = float(
            cfg.get("acquire_center_span_ratio", 1.25))
        self._clock = clock
        self._locked_center = None
        self._locked_sec = None
        self._pending_center = None
        self._pending_count = 0

    def select(self, hands, frame_width_px):
        now_sec = self._clock()
        min_span_px = self._min_span_ratio * frame_width_px
        candidates = [hand for hand in hands if hand_span_px(hand.landmarks) >= min_span_px]

        if self._locked_center is not None:
            continued = self._match_continuation(candidates)
            if continued is not None:
                self._locked_center = hand_center_point(continued.landmarks)
                self._locked_sec = now_sec
                self._reset_pending()
                return continued
            if now_sec - self._locked_sec <= self._release_sec:
                return None   # 유예 중 — 다른 손으로 넘어가지 않고 잠깐의 소실로 취급
            self._locked_center = None
            self._locked_sec = None

        if not candidates:
            self._reset_pending()
            return None
        chosen = max(candidates, key=lambda hand: hand_span_px(hand.landmarks))
        center = hand_center_point(chosen.landmarks)
        acquire_radius_px = (
            self._acquire_center_span_ratio * hand_span_px(chosen.landmarks))
        if (self._pending_center is None
                or math.dist(center, self._pending_center) > acquire_radius_px):
            self._pending_center = center
            self._pending_count = 1
        else:
            self._pending_center = center
            self._pending_count += 1
        if self._pending_count < self._acquire_frames:
            return None

        self._locked_center = hand_center_point(chosen.landmarks)
        self._locked_sec = now_sec
        self._reset_pending()
        return chosen

    def _reset_pending(self):
        self._pending_center = None
        self._pending_count = 0

    def _match_continuation(self, candidates):
        best, best_dist = None, None
        for hand in candidates:
            center = hand_center_point(hand.landmarks)
            radius_px = self._continuity_span_ratio * hand_span_px(hand.landmarks)
            dist_px = math.dist(center, self._locked_center)
            if dist_px <= radius_px and (best_dist is None or dist_px < best_dist):
                best, best_dist = hand, dist_px
        return best
