"""얼굴 기준 "뒷사람" 방어 — 여러 얼굴 중 진짜 사용자(가장 가까운 사람) 하나를
계속 같은 사람으로 추적한다 (2026-08-18 신설, 사용자 요청 — "C:\\dev 에서
뒷사람 인식 안되게 하는것도 참고해서 리모콘 ui 모두 적용시켜줘").

★2026-08-18 현재 상태 — 사실상 비활성: configs/config.yaml의
face_tracker.max_num_faces가 성능 우선 결정으로 3→1로 바뀌었다("얼굴 최대
3명 잡을 필요 없어 무조건 1명만 잡으면 되니까", 트레이드오프를 알린 뒤 사용자
확인받음). FaceEstimator가 애초에 얼굴을 1개만 돌려주면 이 클래스는 여러
후보를 비교할 수가 없어 — 매 프레임 유일한 후보를 그대로 앵커로 받아들이는
것 말고 할 일이 없다(중앙값 스파이크 필터 정도의 잔여 효과만 남는다). 코드
자체는 그대로 두었다 — max_num_faces를 다시 3으로 올리면 아래 로직이 그대로
다시 살아난다. 이 파일의 나머지 설명은 max_num_faces가 여러 명을 받던
시절(그리고 다시 그렇게 되었을 때) 기준으로 쓰여 있다.

★배경 — C:\\dev\\GMtech_project(광명테크 "뒷사람 인식 주도권" 방어 작업,
2026-08-04~08-07에 실기로 다듬어짐)의 src/postprocess/hand_select.py에는
손 추적용 "머리/어깨 앵커" 시스템이 있다: 여러 사람이 보일 때 어깨가 가장
넓은(=가장 가까운) 사람을 앵커로 고정하고, 그 뒤로는 위치 연속성 + 크기
연속성으로 "같은 사람"을 계속 이어 붙이며, 이상치(폭·위치가 사람이 순간
이동할 수 없을 만큼 튀는 관측)는 거부하되 계속 거부되면 안전판으로 강제
수용한다.

그 원본을 그대로 복사해 올 수는 없다 — GMtech_project는 그 하드닝이 끝난
뒤 얼굴 검출기 자체를 삭제하고(회사 결정: 별도 시선추적 모듈과의 얼굴 처리
충돌 회피) 몸 자세(BlazePose: 어깨·팔꿈치·손목·실루엣)로 전환을 마친
상태라, "얼굴 버전"의 앵커 코드가 그쪽엔 아예 없다(face_estimator.py도
없음). 이 프로젝트(리모콘 ui)는 반대로 몸 자세 데이터 없이 얼굴 랜드마크만
쓰므로, 원본의 **원리**(첫 선정은 크기로, 이후는 위치+크기 연속성으로 이어
붙이기, 이상치는 크기 급변·위치 급튐 둘 다로 거부, 판정이 계속 막히면
안전판으로 강제 수용, 짧은 소실은 유예 기간으로 버팀)를 얼굴 랜드마크에
맞게 새로 짠 것이다 — hand_select.py의 _update_head_anchor·
_drop_farther_candidates·_median_observed_frame·_is_implausible_observation·
_accept_anchor_frame과 이름·구조를 최대한 맞췄다.

★hand_select.py와 다른 점 — 이쪽엔 "앵커에 딸린 손 여러 개 중 내 것만 고르기"에
해당하는 층이 없다: 손 추적은 이미 정해진 앵커 주변의 여러 손(내 손 vs
옆사람 손)을 걸러내는 2단계 문제지만, 얼굴 추적은 "앵커 얼굴 자체가 곧
추적 대상"이라 앵커 선정 로직 하나로 끝난다(hand_select.py의
_filter_hands_by_anchor·_is_far_person_hand·손목/실루엣 소유권 판정에
해당하는 부분은 필요 없다).

★알려진 한계(원본 개발 이력에서 그대로 이어받음): 카메라 거리 근사값으로
안구간거리(interocular distance, px)를 쓰는데 — 이 값은 얼굴 자체의
치수라 고개를 좌우로 크게 돌리면 원근으로 줄어든다. 원본이 처음 쓰던
"귀-귀 폭" 자를 버리고 "어깨너비"로 바꾼 이유가 정확히 이 문제였다(귀 폭은
고개를 돌리면 71.1% 붕괴, 어깨너비는 8.5%만 붕괴 — 몸통이라 고개 회전의
영향을 덜 받는다). 이 프로젝트엔 몸 자세 데이터가 없어 그 회피책을 그대로
쓸 수 없다 — 고개를 거의 옆으로 돌린 채로는 앵커 판정이 흔들릴 수 있다는
한계를 안고 쓴다.
"""
import math
import time
from collections import deque

from src.inference.face_estimator import LMK_LEFT_EYE_OUTER, LMK_RIGHT_EYE_OUTER
from src.utils.logger import get_logger

logger = get_logger("postprocess")

# 구조적 상수 — hand_select.py의 동급 상수와 같은 성격(현장에서 매번 바꿀
# 값이 아니라 로직 자체에 속하는 값이라 config가 아니라 모듈 상수로 둔다)
_REJECT_STREAK_LIMIT = 20   # 이상치 관측이 이만큼 연속되면 안전판으로 강제 수용 — 앵커가 영영 안 풀리는 것 방지
_DRIFT_ALPHA = 0.1          # 이상치로 판정된 관측은 이 낮은 비율로만 앵커에 반영(완전 무시도 완전 수용도 아님)
_ACCEPT_ALPHA = 0.4         # 정상 관측의 EMA 반영 비율 — hand_select.py BOX_SMOOTH_ALPHA와 동일값
_DEPTH_LOG_MIN_INTERVAL_SEC = 2.0   # "먼 얼굴 제외" 로그 최소 간격 — 매 프레임 찍으면 로그가 그것만 남는다


def _face_center_px(face):
    """양쪽 눈 바깥쪽 끝의 중점(미간) — 앵커의 위치 기준. bbox 중심보다 안정적이다
    (bbox 경계는 머리카락·턱선 등 랜드마크 검출 잡음의 영향을 더 받는다)."""
    left_px = face.landmark_px(LMK_LEFT_EYE_OUTER)
    right_px = face.landmark_px(LMK_RIGHT_EYE_OUTER)
    return ((left_px[0] + right_px[0]) / 2.0, (left_px[1] + right_px[1]) / 2.0)


def face_depth_px(face):
    """이 얼굴의 안구간거리(px) — 카메라까지 거리의 근사 자(尺).

    hand_select.py의 hand_shoulder_px와 같은 역할(멀수록 작아지는 값)이지만,
    이 프로젝트는 몸 자세 데이터가 없어 안구간거리로 대신한다 — 모듈
    독스트링의 "알려진 한계" 참고.
    """
    left_px = face.landmark_px(LMK_LEFT_EYE_OUTER)
    right_px = face.landmark_px(LMK_RIGHT_EYE_OUTER)
    return math.dist(left_px, right_px)


class FaceAnchor:
    """여러 얼굴 중 사용자 1명을 계속 같은 사람으로 추적한다 — select_user_face()의
    상태 있는(stateful) 대체재.

    select_user_face()는 매 프레임 독립적으로 "가장 큰 얼굴"만 고른다 — 뒷사람이
    일어서거나 앞으로 기울여 순간적으로 화면상 더 크게(또는 가깝게) 보이면
    그 즉시, 아무 저항 없이 "사용자"가 바뀐다. FaceAnchor는 첫 선정 이후엔
    위치·크기 연속성으로 같은 사람을 계속 붙잡고, 그 사람이 아닌 얼굴이
    갑자기 더 크게 보여도 앵커를 넘겨주지 않는다.
    """

    def __init__(self, config, clock=time.monotonic):
        anchor_cfg = config.get("face_anchor") or {}
        # 이음 반경(코스) — "같은 사람일 수 있다"는 느슨한 1차 거름망. 이후
        # move_reject_ratio(정교)가 더 엄격하게 한 번 더 거른다 — 반드시
        # 이 값보다 작아야 한다(그래야 2차 관문이 실제로 뭔가를 거른다)
        self._continuity_ratio = anchor_cfg.get("continuity_ratio", 1.5)
        # 대기줄 방어 — 이음 후보 중 앵커보다 뚜렷이 먼 사람을 뺀다
        self._continuity_depth_ratio = anchor_cfg.get("continuity_depth_ratio", 0.75)
        # 이상치 관문(정교) — 크기 급변·위치 급튐, 각각 독립적으로 거부(OR)
        self._jump_reject_ratio = anchor_cfg.get("jump_reject_ratio", 0.30)
        self._move_reject_ratio = anchor_cfg.get("move_reject_ratio", 0.6)
        # 작동 최소 거리(안구간거리 px 절대값) — 현장 실측 전이라 기본은 끔(None).
        # hand_select.py의 min_operating_depth_px처럼 카메라·설치 환경별 실측이
        # 필요한 값이라, 검증 없이 임의로 켜지 않는다(이 세션에서 반복된 원칙 —
        # ACCURACY_CONF를 임의로 올렸다가 되돌린 head.py 이력 참고)
        self._min_operating_depth_px = anchor_cfg.get("min_operating_depth_px")
        self._median_count = anchor_cfg.get("median_count", 3)
        self._grace_sec = anchor_cfg.get("anchor_grace_sec", 1.0)
        self._clock = clock

        self._anchor = None   # (center_x_px, center_y_px, depth_px) — EMA 평활
        self._anchor_seen_sec = None
        self._reject_streak = 0
        self._observed = deque()
        self._farther_log_sec = None

    def update(self, faces):
        """이번 프레임 얼굴 목록 -> 사용자로 판정된 FaceLandmarks | None.

        None은 "이번 프레임엔 신뢰할 얼굴이 없다"는 뜻이다(호출부의 짧은-소실
        유예 로직이 이어받는다 — head.py/eyebrow.py의 DROPOUT_GRACE_SEC처럼).
        앵커 자체(누가 사용자인지에 대한 기억)는 anchor_grace_sec 동안 이보다
        더 오래 살아남는다 — 짧은 소실 뒤 같은 사람이 돌아왔을 때, 그 사이
        누가 새로 들어왔든 다시 그 사람에게 이어 붙인다.
        """
        now_sec = self._clock()
        self._drop_expired_anchor(now_sec)
        if not faces:
            return None

        framed = [(face, (_face_center_px(face)[0], _face_center_px(face)[1], face_depth_px(face)))
                  for face in faces]
        framed = [(face, frame) for face, frame in framed if frame[2] > 0.0]
        if self._min_operating_depth_px is not None:
            framed = [(face, frame) for face, frame in framed
                      if frame[2] >= self._min_operating_depth_px]
        if not framed:
            return None

        if self._anchor is None:
            return self._acquire(framed, now_sec)
        return self._continue_anchor(framed, now_sec)

    def _acquire(self, framed, now_sec):
        # 첫 선정 — 안구간거리가 가장 큰(=가장 가까운) 얼굴 = 사용자
        # (hand_select.py "가장 넓은 어깨 = 가장 가까운 사람" 첫 선정 규칙과 동일 원리.
        # select_user_face()의 "가장 큰 얼굴" 규칙과도 같은 기준이라 두 선정이 갈리지 않는다)
        chosen, frame = max(framed, key=lambda pair: pair[1][2])
        self._anchor = frame
        self._anchor_seen_sec = now_sec
        self._reject_streak = 0
        self._observed.clear()
        logger.info("얼굴 앵커 획득: 안구간거리 %.0fpx · 중심 (%.0f,%.0f) · 관측 %d명",
                    frame[2], frame[0], frame[1], len(framed))
        return chosen

    def _continue_anchor(self, framed, now_sec):
        anchor_x, anchor_y, anchor_depth = self._anchor
        # 이음(1차, 코스) — 앵커 위치에서 이 반경(현재/앵커 안구간거리 중 큰 쪽
        # 기준) 밖의 얼굴은 애초에 "같은 사람일 후보"조차 아니다
        continuous = [
            (face, frame) for face, frame in framed
            if math.dist((frame[0], frame[1]), (anchor_x, anchor_y))
            <= self._continuity_ratio * max(frame[2], anchor_depth)
        ]
        if not continuous:
            return None   # 앵커 얼굴이 이번 프레임엔 안 보인다 — _drop_expired_anchor의 유예 기간 동안 앵커 상태는 유지

        continuous = self._drop_farther_candidates(continuous, anchor_depth)
        # 이음 후보 중에선 크기가 아니라 "위치가 가장 가까운" 얼굴을 고른다 —
        # 첫 선정 이후의 정체성 연속은 크기가 아니라 위치로 잇는다(hand_select.py와 동일 원칙)
        chosen, frame = min(continuous, key=lambda pair: math.dist(
            (pair[1][0], pair[1][1]), (anchor_x, anchor_y)))
        frame = self._median_observed_frame(frame)

        if not self._is_implausible_observation(self._anchor, frame):
            self._reject_streak = 0
            self._accept_anchor_frame(frame)
        else:
            self._reject_streak += 1
            if self._reject_streak > _REJECT_STREAK_LIMIT:
                logger.info("얼굴 앵커 관측 거부가 %d회 연속 — 받아들인다(앵커가 굳는 것 방지)",
                            _REJECT_STREAK_LIMIT)
                self._reject_streak = 0
                self._accept_anchor_frame(frame)
            else:
                self._accept_anchor_frame(frame, alpha=_DRIFT_ALPHA)
        self._anchor_seen_sec = now_sec
        return chosen

    def _drop_farther_candidates(self, continuous, anchor_depth_px):
        """이음 후보 중 앵커보다 뚜렷이 먼(안구간거리가 작은) 사람을 뺀다 — 대기줄
        방어. hand_select.py의 _drop_farther_candidates와 동일 원리: 위치
        연속성만으론 대기줄에서 앞뒤로 자리를 바꾸는 사람들을 못 가른다 —
        위치는 비슷해도 거리(안구간거리)는 뚜렷이 다르기 때문에 이걸로 가른다.
        후보를 전부 비우게 되면(가장 가까운 후보까지 앵커보다 훨씬 멀 때)
        원래 후보 목록을 그대로 돌려준다 — 이 관문이 앵커를 완전히 잃게 만들면
        안 된다."""
        if anchor_depth_px <= 0.0:
            return continuous
        kept = [pair for pair in continuous
                if pair[1][2] >= self._continuity_depth_ratio * anchor_depth_px]
        if not kept or len(kept) == len(continuous):
            return continuous if not kept else kept
        now_sec = self._clock()
        if (self._farther_log_sec is None
                or now_sec - self._farther_log_sec >= _DEPTH_LOG_MIN_INTERVAL_SEC):
            self._farther_log_sec = now_sec
            logger.info("이음 후보에서 먼 얼굴 %d명 제외(대기줄) — 앵커 안구간거리 %.0fpx",
                        len(continuous) - len(kept), anchor_depth_px)
        return kept

    def _median_observed_frame(self, frame):
        """중앙값 필터 — 순간적인 랜드마크 튐(스파이크) 한두 프레임을 걸러낸다.
        hand_select.py의 _median_observed_frame과 동일 원리(평균 대신 중앙값 —
        스파이크 하나가 기준 전체를 끌고 가지 않는다)."""
        if self._median_count < 2:
            return frame
        self._observed.append(frame)
        while len(self._observed) > self._median_count:
            self._observed.popleft()
        if len(self._observed) < self._median_count:
            return frame
        middle_idx = len(self._observed) // 2
        return tuple(
            sorted(observed[axis] for observed in self._observed)[middle_idx]
            for axis in range(3)
        )

    def _is_implausible_observation(self, prev_frame, new_frame):
        """이번 관측이 "사람일 수 없는가" — 크기(안구간거리) 급변·위치 급튐 둘 다
        본다(OR — 하나만 넘어도 거부). hand_select.py의 _is_implausible_observation
        (2026-08-07 위치 튐 관문 추가 이력 참고)과 동일 원리 — 크기만 보면
        "비슷한 크기의 옆/뒷사람 사이에서 위치만 바뀌는" 경우를 놓친다."""
        prev_x, prev_y, prev_depth = prev_frame
        new_x, new_y, new_depth = new_frame
        if prev_depth <= 0.0:
            return False
        depth_ratio = abs(new_depth - prev_depth) / prev_depth
        move_ratio = math.dist((new_x, new_y), (prev_x, prev_y)) / prev_depth
        depth_bad = depth_ratio > self._jump_reject_ratio
        move_bad = move_ratio > self._move_reject_ratio
        return depth_bad or move_bad

    def _accept_anchor_frame(self, frame, alpha=_ACCEPT_ALPHA):
        anchor_x, anchor_y, anchor_depth = self._anchor
        self._anchor = (
            anchor_x + alpha * (frame[0] - anchor_x),
            anchor_y + alpha * (frame[1] - anchor_y),
            anchor_depth + alpha * (frame[2] - anchor_depth),
        )

    def _drop_expired_anchor(self, now_sec):
        """유예 기간(anchor_grace_sec)을 넘겨 앵커 얼굴이 안 보이면 앵커를 완전히
        놓는다 — 다음 얼굴 등장 때 처음부터 다시(가장 가까운 사람으로) 선정한다.
        hand_select.py의 _drop_expired_head_anchor와 동일 목적."""
        if self._anchor is None or self._anchor_seen_sec is None:
            return
        if now_sec - self._anchor_seen_sec > self._grace_sec:
            logger.info("얼굴 앵커 유예 %.1fs 초과 — 앵커 해제(다음 얼굴로 새로 획득)",
                        self._grace_sec)
            self._anchor = None
            self._anchor_seen_sec = None
            self._reject_streak = 0
            self._observed.clear()
