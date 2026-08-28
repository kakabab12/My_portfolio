"""postprocess 모듈 — 오토포커스 사용자 잠금: 초점 맞은 사람에게 잠그고 그 사람만 인식한다.

요구사항(2026-07-10): 오토포커스 카메라 기준, 초점이 맞춰진 사람의 얼굴을 기준으로
잠금(lock)하고 그 사람만 판정에 쓴다 — 다른 사람은 무시한다.

판정 절차(모든 수치는 config person_lock에서 읽는다):
1. 후보 점수 = 얼굴 크기 × 초점 선명도(라플라시안 분산) 가중 평균
   — 오토포커스가 맞은 사람이 가장 선명하고, 가까운 사람이 가장 크다
2. 최고 점수 후보가 lock_frame_count 프레임 연속이면 그 사람에게 잠금
3. 잠금 중에는 follow_radius 안에서 같은 사람을 추적, release_sec 이상
   사라지면 해제하고 다음 사용자를 받는다

2026-07-18 헤드트래커 전환: 이 모듈은 "누구를 볼지"만 결정한다 — 잠긴 사람의
FaceLandmarks(랜드마크·블렌드셰이프)를 그대로 노출하고, 그걸로 어떤 제스처를
판정할지는 head_tracker.py의 몫이다(제스처 체계가 또 바뀌어도 이 모듈은 안 건드리게).
lock_generation은 잠금이 새로 걸릴 때만 증가하는 카운터 — head_tracker가 이걸로
"새 사용자라 커서 캘리브레이션을 다시 해야 하는지"를 판단한다.

구 팔 쓸기 체계의 좌/우 거울 보정(user_side_points)은 삭제했다 — 코·입 등 헤드트래커가
쓰는 신호는 전부 얼굴 중앙의 단일 지점이거나 좌우 대칭 판정이라 거울 반전이
결과에 영향을 주지 않는다.
"""
import math
import time

import cv2

from src.utils.logger import get_logger

logger = get_logger("postprocess")

SHARPNESS_SQUASH = 300.0      # 라플라시안 분산 정규화 상수 (v/(v+K) — 0~1로 압축)


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _laplacian_sharpness(frame, face_box):
    """얼굴 영역의 초점 선명도 — 라플라시안 분산. 클수록 초점이 맞은 것."""
    x1, y1, x2, y2 = face_box
    crop = frame[int(y1):int(y2), int(x1):int(x2)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class PersonLock:
    def __init__(self, config, frame_width_px, frame_height_px,
                 clock=time.monotonic, sharpness_fn=None):
        lock_cfg = config["person_lock"]
        self.enabled = lock_cfg["enabled"]
        self._lock_frame_count = lock_cfg["lock_frame_count"]
        self._follow_radius_px = lock_cfg["follow_radius_ratio"] * frame_width_px
        self._release_sec = lock_cfg["release_sec"]
        self._sharpness_weight = lock_cfg["sharpness_weight"]

        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._clock = clock
        self._sharpness_fn = sharpness_fn or _laplacian_sharpness

        self.locked_face = None        # 잠긴 FaceLandmarks (최신 프레임 것으로 갱신)
        self.lock_generation = 0       # None -> 얼굴 전환 시에만 증가 (head_tracker 캘리브레이션 트리거)
        self._candidate_center = None  # 잠금 전 최고 후보 추적
        self._candidate_count = 0
        self._last_seen_sec = None

    # ----- 사용자 선정·추적 -----

    def _score(self, frame, face):
        """후보 점수 — 얼굴 크기와 초점 선명도의 가중 평균 (둘 다 0~1 정규화)."""
        x1, y1, x2, y2 = face.bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame.shape[0] * frame.shape[1])
        sharpness = self._sharpness_fn(frame, face.bbox)
        sharpness_norm = sharpness / (sharpness + SHARPNESS_SQUASH)
        weight = self._sharpness_weight
        return (1.0 - weight) * min(area_ratio * 10.0, 1.0) + weight * sharpness_norm

    def update(self, frame, faces):
        """프레임의 얼굴 목록으로 잠금 상태를 갱신한다. 잠긴 얼굴(or None)을 돌려준다."""
        if not self.enabled:
            # 잠금 비활성이어도 head_tracker는 기준 인물이 필요하다 — 최고 신뢰도 얼굴을 추적
            self.locked_face = max(faces, key=lambda f: f.conf) if faces else None
            return self.locked_face
        now_sec = self._clock()

        if self.locked_face is not None:
            return self._follow_locked(faces, now_sec)

        # 점수(라플라시안 선명도 포함)는 잠금 전 후보 선정에만 필요하다 — 잠금 중에는
        # 위치 추적만 하므로 매 프레임 선명도 계산이 순수 낭비였다 (2026-07-20 추론 FPS 개선)
        scored = [(self._score(frame, face), face) for face in faces]

        if not scored:
            self._candidate_center = None
            self._candidate_count = 0
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        _, best_face = scored[0]
        best_center = _center(best_face.bbox)

        is_same_candidate = self._candidate_center is not None and (
            math.dist(best_center, self._candidate_center) <= self._follow_radius_px
        )
        self._candidate_count = self._candidate_count + 1 if is_same_candidate else 1
        self._candidate_center = best_center

        if self._candidate_count >= self._lock_frame_count:
            self.locked_face = best_face
            self._last_seen_sec = now_sec
            self.lock_generation += 1
            logger.info(
                "사용자 잠금 — 얼굴 기준 (score 후보 %d프레임 연속, generation=%d)",
                self._candidate_count, self.lock_generation,
            )
        return self.locked_face

    def _follow_locked(self, faces, now_sec):
        """잠긴 사람을 follow_radius 안에서 계속 추적한다. 오래 사라지면 해제."""
        locked_center = _center(self.locked_face.bbox)
        best_match = None
        best_dist = None
        for face in faces:
            dist = math.dist(_center(face.bbox), locked_center)
            if dist <= self._follow_radius_px and (best_dist is None or dist < best_dist):
                best_match = face
                best_dist = dist

        if best_match is not None:
            self.locked_face = best_match
            self._last_seen_sec = now_sec
            return self.locked_face

        if now_sec - self._last_seen_sec > self._release_sec:
            logger.info("사용자 잠금 해제 — %.1f초 미검출", now_sec - self._last_seen_sec)
            self.locked_face = None
            self._candidate_center = None
            self._candidate_count = 0
        return self.locked_face
