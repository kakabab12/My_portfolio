"""capture 모듈 — 여러 카메라 동시 운용 + 프레임별 활성 카메라 선택.

로봇이 움직이면서 조작자와 카메라의 상대 위치가 계속 바뀌므로, 시작 시
1회만 고르는 방식(gesture_kiosk의 camera_probe.py)이 아니라 **매 프레임**
모든 카메라의 손 품질을 비교해 활성 카메라를 정한다. 손 품질 공식은
camera_probe.py의 _hand_quality와 동일(크기×신뢰도) — 검증된 방식을 승계.

여러 카메라의 프레임을 같은 HandTracker(VIDEO 모드) 인스턴스에 번갈아 넣으면
내부 프레임 간 추적 상태가 서로 다른 물리적 장면을 잇는 것으로 오염된다 —
그래서 카메라마다 별도 HandTracker 인스턴스를 두는 것을 전제로 한다
(인스턴스 생성은 pipeline/gesture_loop.py 책임 — 이 모듈은 프레임과 활성
선택만 맡는다).
"""
from src.capture.camera_stream import CameraStream
from src.postprocess.hand_shape import hand_span_px
from src.utils.logger import get_logger

logger = get_logger("capture")


def score_hand_quality(hands, frame_width_px, good_span_ratio):
    """프레임의 손 품질(0.0~1.0) — 가장 좋은 손의 (크기/기준)×신뢰도."""
    best_quality = 0.0
    for hand in hands:
        span_ratio = hand_span_px(hand.landmarks) / frame_width_px if frame_width_px else 0.0
        quality = min(1.0, span_ratio / good_span_ratio) * hand.conf
        best_quality = max(best_quality, quality)
    return best_quality


class CameraArbiter:
    """한 카메라에 손 제어권을 고정하고 손을 놓쳤을 때만 넘기는 상태기.

    비활성 카메라가 switch_margin 이상 앞서는 상태가 switch_frames 연속돼야
    실제로 전환한다 — 점수가 근소하게 오가는 매 프레임 깜빡임(flicker) 방지.
    순수 상태기 — 카메라·모델 없이 단위 테스트 가능.
    """

    def __init__(self, switch_margin, switch_frames, initial_active=0,
                 sticky_ownership=False, release_frames=10, cooldown_frames=30):
        self._switch_margin = switch_margin
        self._switch_frames = switch_frames
        self.active_index = initial_active
        self._candidate_index = None
        self._candidate_streak = 0
        self._sticky_ownership = sticky_ownership
        self._release_frames = release_frames
        self._cooldown_frames = cooldown_frames
        self._active_missing_streak = 0
        self._cooldown_remaining = 0

    def update(self, scores, visible=None):
        """scores와 카메라별 선택 손 존재 여부 -> 이번 프레임의 active_index."""
        if self._sticky_ownership:
            return self._update_sticky(scores, visible)

        best_index = max(range(len(scores)), key=lambda i: scores[i])
        if best_index == self.active_index or scores[best_index] - scores[self.active_index] < self._switch_margin:
            self._candidate_index = None
            self._candidate_streak = 0
            return self.active_index

        if best_index == self._candidate_index:
            self._candidate_streak += 1
        else:
            self._candidate_index = best_index
            self._candidate_streak = 1

        if self._candidate_streak >= self._switch_frames:
            logger.info("활성 카메라 전환: %d -> %d (점수 %.2f -> %.2f)",
                        self.active_index, best_index,
                        scores[self.active_index], scores[best_index])
            self.active_index = best_index
            self._candidate_index = None
            self._candidate_streak = 0
        return self.active_index

    def _update_sticky(self, scores, visible):
        visible = ([score > 0.0 for score in scores] if visible is None else visible)
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if visible[self.active_index]:
            self._active_missing_streak = 0
            self._candidate_index = None
            self._candidate_streak = 0
            return self.active_index

        self._active_missing_streak += 1
        if (self._active_missing_streak < self._release_frames
                or self._cooldown_remaining > 0):
            return self.active_index

        candidates = [i for i, is_visible in enumerate(visible)
                      if i != self.active_index and is_visible]
        if not candidates:
            self._candidate_index = None
            self._candidate_streak = 0
            return self.active_index

        best_index = max(candidates, key=lambda i: scores[i])
        if best_index == self._candidate_index:
            self._candidate_streak += 1
        else:
            self._candidate_index = best_index
            self._candidate_streak = 1

        if self._candidate_streak >= self._switch_frames:
            logger.info("활성 카메라 소유권 전환: %d -> %d", self.active_index, best_index)
            self.active_index = best_index
            self._candidate_index = None
            self._candidate_streak = 0
            self._active_missing_streak = 0
            self._cooldown_remaining = self._cooldown_frames
        return self.active_index


class MultiCameraManager:
    """설정된 모든 카메라를 구동하고 각 카메라의 최신 프레임을 돌려준다.

    한쪽 카메라가 빠지거나, 애초에 장치 자체가 없어 열기부터 실패해도(개발용
    PC에 웹캠 1대뿐인 경우 포함) 나머지 한 대만으로 계속 동작한다 — 열기
    실패는 경고만 남기고 넘어간다(try_get_frame은 그런 카메라에 대해 계속
    (None, seq)만 돌려주므로 이후 루프도 자연히 안전하다).
    """

    def __init__(self, config):
        self.device_ids = list(config["camera"]["devices"])
        self._streams = [CameraStream(config, device_id=d) for d in self.device_ids]

    def start(self):
        started_count = 0
        for device_id, stream in zip(self.device_ids, self._streams):
            try:
                stream.start()
                started_count += 1
            except RuntimeError as error:
                logger.warning("카메라 시작 실패(device_id=%s) — 나머지 카메라로 계속: %s",
                               device_id, error)
        if started_count == 0:
            raise RuntimeError("카메라를 하나도 열지 못했습니다 — 장치 연결/번호를 확인하세요")
        return self

    def read_frames(self):
        """각 카메라의 (프레임, seq) 리스트 -> [(frame0, seq0), ...] (미도착 시 frame=None)."""
        return [stream.try_get_frame() for stream in self._streams]

    def stop(self):
        for stream in self._streams:
            stream.stop()


# 기존 import/API와 설정 이름을 깨지 않기 위한 호환 별칭.
DualCameraManager = MultiCameraManager
