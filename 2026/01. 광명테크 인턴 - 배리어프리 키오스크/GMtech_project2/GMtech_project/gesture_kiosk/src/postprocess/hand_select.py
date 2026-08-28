"""postprocess 모듈 — 사용자 손 선별: 프레임의 손들 중 사용자 손 **하나**를 추적한다.

2026-07-31 단일 손 추적 재작성(사용자 결정 — "무조건 한 손"): 종전에는 물리적
손을 left/right **라벨**을 키로 좌/우 슬롯에서 따로 추적했는데, MediaPipe
handedness가 불안정해 보정(위치 재라벨·중앙 띠·획 교차 보호·플랩 브리지)이
보정을 부르며 실기 버그의 주 원인이 됐다(획 씹힘·배구 토스 — 당일 실기 5건).
라벨을 정체성 키에서 제거하고 **공간 연속성**으로 손 하나를 직접 추적한다:
- 획득: 모양이 보이는 손이 실제로 이동(acquire.move_dist_shoulder)하면 그 손
  — 구 지시 손 고정(v2)의 획득 규칙을 라벨 없이 계승. 쉬는 손은 영원히 못 잡힌다
- 이음: 직전 추적점 근처(REENTRY 반경) 재등장 = 같은 손 (신선 0.5초는 게이트
  면제 겸용, 이후 release_sec까지는 재합류 — 화면 가리킴의 수 초 소실 흡수)
- 해제: release_sec 초과 소실 — 다음 손은 획득부터 (다른 사용자 승계 차단)
handedness 라벨은 이벤트의 정보용 필드(hand_side)로만 전달한다.

몸통판(2026-07-31): 앵커 = 포즈(BlazePose) 머리 관측(head_detector.py) —
마스크·썬글라스·모자 무관. 앵커 동작(sticky·게이트·어깨선)은 얼굴판과 동일,
관측 소스만 다르다.

2026-07-29 포즈 스택 제거(사용자 결정)의 대체 구조는 유지:
- 어깨너비 자(거리 무관 임계) → **손 실측 자**: 월드 랜드마크(미터)와 화면
  px의 비로 가상 어깨너비를 만든다 — 기존 임계 체계(어깨너비 배수) 무변경
- 어깨선 → 앵커(머리) 기준 몸 비례 추정 (2026-07-31 — 카메라 거치 무관)
"""
import math
import time
from collections import deque

import numpy as np

from src.postprocess.hand_shape import classify_hand_shape, finger_states, hand_center_point
from src.utils.logger import get_logger
import logging

logger = get_logger("postprocess")

STANDARD_SHOULDER_M = 0.4     # 가상 어깨 자 환산용 표준 어깨너비 — 기존 임계(어깨너비
                              #   배수) 체계를 숫자 그대로 유지하기 위한 고정 상수
EXEMPT_FRESH_SEC = 0.5        # 반경 면제의 신선도 — 추적점이 이 시간 안일 때만 게이트
                              #   면제 (모션 블러 소실 0.35초는 덮고, 내린 손 자리에
                              #   나타난 옆 사람 손의 면제 승계는 차단)
CONTINUITY_SPAN_RATIO = 1.5   # 프레임 간 연속·획득 궤적 연결 반경 — 손 폭의 N배
                              #   (프레임 간 이동보다 넉넉, 다른 손보다 좁게)
REENTRY_SPAN_RATIO = 4.0      # 소실 재등장 이음 반경(2026-07-31 키오스크 실기 — 획 중간
                              #   끊김): 빠른 쓸기 중 추적이 잠깐 끊기면 재등장 손은
                              #   마지막 추적점에서 수 손폭 밖이다. 연속 반경(1.5배)만
                              #   쓰면 "다른 손" 취급돼 획이 죽는다. 승계 오탐이 보이면 3.0으로
BOX_SMOOTH_ALPHA = 0.4        # 표시 박스 EMA — 랜드마크 떨림 노이즈 저감 (1.0=평활 없음)
ACQUIRE_GAP_SEC = 0.25        # 획득 궤적의 프레임 연결 허용 공백 — 짧은 검출 끊김 흡수
WRIST_MIN_VISIBILITY = 0.5    # 손목 심사에 쓸 최소 관측 신뢰도 — 2026-08-04
                              #   사용자 관찰("카메라에 안 보여도 추론이 손을
                              #   만들어 낸다"): BlazePose는 가려지거나 화면
                              #   밖이어도 33점을 항상 채운다. 추측 좌표를
                              #   "내 팔 끝"의 근거로 쓰면 안 된다
WRIST_ARM_SPAN_RATIO = 1.6    # 어깨→손목 허용 거리 = 어깨너비 × N (팔 길이).
                              #   사용자 관찰("남의 손 난입 시 내 팔이 내려가
                              #   있어도 그 손을 내 손목으로 잡아 올린다"):
                              #   끌려간 손목은 몸에서 팔 길이를 넘는다
ANCHOR_CONTINUITY_RATIO = 0.47  # 앵커 이음 반경 = 어깨너비 × N — 이 안의 관측만
                              #   "같은 사람"으로 본다. ★2026-08-04 단위 환산과 함께
                              #   신설: 종전엔 손 연결용 CONTINUITY_SPAN_RATIO(1.5)를
                              #   같이 썼는데, 기준이 머리폭→어깨너비로 바뀌면서 반경이
                              #   3.16배로 부풀어 **멀리 있는 사람도 "연속"으로 잡혀
                              #   앵커가 점프했다**(테스트가 잡았다). 1.5 ÷ 3.16 = 0.47로
                              #   환산해 종전과 같은 실거리를 유지한다
REJECT_STREAK_LIMIT = 5       # 어깨 관측을 이만큼 연속 거부하면 그냥 받아들인다
                              #   (앵커 주기 10 FPS 기준 0.5초). 안전판이다 —
                              #   사람이 정말 멀어졌는데 계속 거부하면 앵커가 낡은
                              #   채 굳어, 실제 위치와 어긋난 반경으로 손을 자른다
REJECT_STREAK_LIMIT_WHILE_TRACKED = 30  # 2026-08-07 3차 — 손이 지금 활성 추적
                              #   중일 때의 안전판은 훨씬 느리게(≈3초). 실기 로그:
                              #   REJECT_STREAK_LIMIT(5=0.5초)이 활성 추적 중에도
                              #   무조건 강제 수용해, 뒷사람이 있을 때의 지속
                              #   드리프트(30~44%대 점프)가 이 뒷문으로 계속
                              #   들어왔다(2연속 합의 경로를 막은 것만으로는 부족—
                              #   같은 함수 _update_head_anchor 참고). 활성 추적이
                              #   없을 때는 종전 5 그대로(빠른 재동기화 유지)
TRACKED_DRIFT_ALPHA = 0.1     # 2026-08-07 4차 — 활성 추적 중 "의심스러운"(폭·위치
                              #   관문에 걸린) 관측을 완전히 무시하는 대신 이만큼만
                              #   반영한다(BOX_SMOOTH_ALPHA 0.4의 1/4 — 정상 관측보다
                              #   훨씬 조심스럽게). 3차("얼려서 보호")의 역설을
                              #   해소하기 위한 값: 완전히 얼리면 앵커가 실제 손
                              #   위치와 어긋나 반경 밖으로 밀려나고, 그 손이
                              #   release_sec(2초) 뒤 추적 해제되면 "추적 없음"
                              #   상태가 되어 보호 자체가 풀리는 악순환이 실측됐다.
                              #   저역통과 필터 원리 — 노이즈는 방향이 들쭉날쭉해
                              #   누적되며 상쇄되고, 진짜(느린) 이동만 같은 방향으로
                              #   쌓여 몇 프레임에 걸쳐 따라잡는다. 내 손이 여전히
                              #   반경 밖으로 밀리면 ↑, 하이재킹이 이 경로로
                              #   새면 ↓(단, REJECT_STREAK_LIMIT_WHILE_TRACKED가
                              #   결국은 다 받아들이므로 0에 가깝게만 낮출 것)
EAR_TO_SHOULDER_RATIO = 3.16  # 어깨너비 = 귀-귀 폭 × N (2026-08-04 정면 실측:
                              #   귀 130px / 어깨 413px).
                              #   ★같은 날 조명 켜고 재측정: 자세별 중앙값이 2.94(앉아서)
                              #   ~3.82(서서 정면)로 흩어지고 고개를 돌리면 5.07까지 튄다.
                              #   값 하나로 대표가 안 된다는 것 자체가 기준을 어깨로 옮긴
                              #   이유다 — 그래서 이 상수를 다시 맞추지 않는다. 실측
                              #   1029프레임에서 어깨 미검출 0이라 이 경로는 거의 안 탄다.
                              #   어깨가 안 잡히는 프레임에서만
                              #   귀로 앵커를 대신 세울 때 쓰는 환산 계수 — 단위를
                              #   어깨너비 하나로 통일하기 위한 것이다
SHOULDER_BELOW_EAR_RATIO = 0.51  # 귀 폴백 시 어깨선 = 귀 중점 y + 어깨너비 × N.
                              #   구 shoulder_below_head_widths 1.6(머리폭)을 어깨
                              #   단위로 환산한 값(1.6 ÷ 3.16)
HAND_SPAN_OF_SHOULDER = 0.2   # 펼친 손의 화면 크기 ÷ 가상 어깨너비 — stable_span_px가
                              #   모양 무관 자를 **기존 눈금에 맞추는** 환산 계수.
                              #   픽스처·실측 모두 펼친 손 80px / 가상 어깨 400px
                              #   = 0.2라, 기존 임계(REENTRY 4.0 · CONTINUITY 1.5)의
                              #   숫자 의미가 그대로 유지된다(재보정 불필요)
SILHOUETTE_THRESHOLD = 0.5    # 실루엣 마스크를 "이 사람"으로 볼 확률 문턱 —
                              #   블러 쪽 mask_threshold와 같은 기본값이지만,
                              #   이쪽은 **판정**이라 config로 열지 않는다
                              #   (튜닝 지점이 늘면 실기에서 원인 추적이 어렵다)
DEPTH_LOG_MIN_INTERVAL_SEC = 2.0  # 깊이 관문 거부 로그의 최소 간격 — 프레임마다
                              #   찍으면 파일 핸들러가 동기 flush라 FPS를 갉는다
                              #   (2026-08-07 7차에 실제로 겪은 문제)
OTHER_TRACK_KEEP_SEC = 1.0    # 추적 손이 아닌 손의 궤적을 안 보이는 동안도 유지하는
                              #   시간 — 한두 프레임 미검출(블러·가림·중복 억제)로
                              #   "내 손과 공존" 표시를 잃지 않게. 소실 유예
                              #   (release_sec 2.0)보다 짧게 — 진짜 떠난 손의
                              #   잔상이 오래 남지 않게


def smooth_box(prev_box, new_box, alpha=BOX_SMOOTH_ALPHA):
    """표시 박스 EMA — 이전 박스에서 새 박스 쪽으로 alpha만큼만 이동 (떨림 흡수)."""
    if prev_box is None or new_box is None:
        return new_box
    return tuple(
        int(round(prev_value + alpha * (new_value - prev_value)))
        for prev_value, new_value in zip(prev_box, new_box)
    )


def hand_span_px(landmarks):
    """손 화면 크기 — 랜드마크 묶음의 최대 폭(px). 연속·재등장 반경의 자."""
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    return max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def hand_span_world_m(world_landmarks):
    """손 실측 크기 — 월드 랜드마크(미터)의 최대 폭. px/m 환산의 분모."""
    xs = world_landmarks[:, 0]
    ys = world_landmarks[:, 1]
    return max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def reliable_wrists(wrists, shoulders):
    """**믿을 수 있는** 손목만 골라 돌려준다 — 못 믿으면 빈 목록(심사 생략).

    ★2026-08-04 사용자 관찰 2건으로 신설. 종전엔 포즈가 준 손목을 그대로 썼는데,
    그러면 손목 심사가 바로 무력화된다:
    ① "다른 사람 손 난입 시, 내 팔이 내려가 있어도 (포즈가) 손으로 잡고 올린다"
       — 남의 손이 **내 손목**으로 추정되면 그 손은 "내 팔 끝"이 되어 통과한다.
    ② "카메라에 안 보여도 추론이 손을 만들어 낸다" — BlazePose는 가려지거나
       화면 밖이어도 33점을 항상 채운다. 그 추측 좌표에 판정 영역이 생긴다.
    그래서 두 관문을 둔다:
      · 관측 신뢰도(visibility)가 WRIST_MIN_VISIBILITY 이상 — 추측은 못 쓴다
      · 같은 쪽 어깨에서 팔 길이(WRIST_ARM_SPAN_RATIO × 어깨너비) 안 —
        남의 손으로 끌려간 손목은 몸에서 너무 멀어 여기서 걸린다
    둘 다 못 넘으면 손목이 없는 셈 친다 (인식 우선 폴백).

    앵커와 **다른 사람 모두**에게 같은 관문을 적용하려고 순수 함수로 뺐다
    (2026-08-04 소유자 판정 — hand_select._is_other_persons_hand): 못 믿을
    추정으로는 아무도 탈락시키지 않아야 내 손이 잘리지 않는다.
    """
    if len(wrists) != 2 or len(shoulders) != 2:
        return []
    shoulder_span_px = math.dist(shoulders[0][:2], shoulders[1][:2])
    if shoulder_span_px <= 0.0:
        return []
    arm_reach_px = WRIST_ARM_SPAN_RATIO * shoulder_span_px
    usable = []
    for side_idx, wrist in enumerate(wrists):
        if len(wrist) > 2 and wrist[2] < WRIST_MIN_VISIBILITY:
            continue   # 안 보이는 손목 — 좌표는 추측이다
        if math.dist(wrist[:2], shoulders[side_idx][:2]) > arm_reach_px:
            continue   # 팔 길이를 넘는다 — 남의 손으로 끌려간 추정
        usable.append(wrist)
    return usable


def anchor_frame(head):
    """앵커의 기준 틀 (중심x, 중심y, **어깨너비**) | None — 중심도 자도 **어깨**다.

    ★2026-08-04 머리 기준점 제거(사용자 결정 — "기준점하고 선이 많이 흔들린다.
    얼굴 쪽도 많이 흔들려서 없애라"). 자(尺)는 같은 날 먼저 어깨로 옮겼고
    (귀-귀 폭이 고개 돌림에 71.1% 무너진다: 130→47px, 같은 조건 어깨는 8.5%),
    이번에 **중심까지** 옮겨 머리 좌표를 판정에서 완전히 뺀다.
    실측이 뒷받침한다 — 귀 중점은 어깨 중점 기준으로 고개 돌릴 때 어깨너비의
    0.104배까지 흔들렸다(제자리 0.035의 3배, 2026-08-04). 어깨 중점은 그 흔들림의
    출처가 아니다.

    ※한 번 되돌렸던 변경이다: 중심만 내리고 반경을 그대로 뒀더니 위쪽 도달이
    머리~어깨 거리만큼 줄어 팔을 끝까지 뻗을 때 게이트 통과율이 100%→88.5%로
    떨어졌다. 이번에는 그 손실을 reach_shoulder_widths로 **보정해서** 내린다
    (config 주석 참고) — 같은 실패를 반복하지 않기 위해 반드시 함께 가야 한다.

    어깨가 없는 프레임에서는 귀에서 환산해 쓴다(폭 = 귀-귀 × 3.16, 중심 =
    귀 중점 + 어깨너비 × SHOULDER_BELOW_EAR_RATIO) — 관측이 빠졌다고 앵커를
    통째로 버리면 게이트가 꺼져 옆 사람 손이 새기 때문이다. 단위는 항상 어깨너비.
    """
    if head.width_px <= 0.0:
        return None
    if len(head.shoulders) == 2:
        span_px = math.dist(head.shoulders[0][:2], head.shoulders[1][:2])
        if span_px > 0.0:
            return ((head.shoulders[0][0] + head.shoulders[1][0]) / 2.0,
                    (head.shoulders[0][1] + head.shoulders[1][1]) / 2.0,
                    span_px)
    span_px = head.width_px * EAR_TO_SHOULDER_RATIO
    return (head.center_x_px,
            head.center_y_px + SHOULDER_BELOW_EAR_RATIO * span_px,
            span_px)


def stable_span_px(hand):
    """손 모양에 **안 흔들리는** 손 크기(px) — 정체성 반경 전용 자.

    ★2026-08-07 신설(사용자 보고 — "특정 손짓을 하면 초록색 관절값이 안 보이거나
    다른 사람한테 이동한다"). 원인은 정체성 반경이 hand_span_px(화면상 손의
    외접 크기)에 비례한 것이었다. 그 값은 **손 모양에 따라 변한다** — 주먹을
    쥐면 펼친 손의 절반 가까이 작아진다(픽스처로도 80->62px, 실제는 더 크다).
    그러면 이음 반경도 같이 줄어(4배 기준 320->248px 이하) 획 중에 주먹을
    쥐는 순간 **추적 손이 제 반경 밖으로 밀려나 정체성을 잃는다.** 그 직후
    다른 손이 새로 획득되면 표시와 판정이 통째로 옆 사람에게 넘어간다.

    해법은 **모양과 무관한 자**를 쓰는 것이다. hand_shoulder_px는 화면 크기를
    월드 크기로 나누므로 손 모양이 약분돼 사라진다(실측: open/fist/finger/ok
    전부 같은 값). 여기에 상수를 곱해 **펼친 손의 화면 크기와 같은 눈금**으로
    맞춘다 — 그래야 기존 임계(REENTRY 4.0·CONTINUITY 1.5)의 숫자 의미가
    그대로 유지되고 재보정이 필요 없다.
    월드 랜드마크를 못 쓰면 종전 값으로 폴백한다.
    """
    shoulder_px = hand_shoulder_px(hand)
    if shoulder_px is None or shoulder_px <= 0.0:
        return hand_span_px(hand.landmarks)
    return shoulder_px * HAND_SPAN_OF_SHOULDER


def hand_shoulder_px(hand):
    """이 손 기준 가상 어깨너비(px) — 획득 이동 임계의 자 (손 실측 자와 동일 원리).

    ★2026-08-07 두 번째 용도(뒷사람 차단 — _is_far_person_hand): 이 값은
    **거리 측정치**이기도 하다. 화면 span_px = f × span_m / d 이므로
    (f = 초점거리 px, d = 카메라~손 거리),
        span_px / span_m = f / d
    가 되어 **손의 실제 크기가 약분돼 사라진다**. 여기에 상수(0.4m)를 곱한
    이 함수의 반환값은 오직 d에만 의존한다 — 손이 큰 사람이든 작은 사람이든
    같은 거리에 있으면 같은 값이 나오고, 멀수록 작아진다.
    그래서 이 하나로 "저 손이 나보다 뒤에 있나"를 직접 물을 수 있다.
    """
    span_px = hand_span_px(hand.landmarks)
    span_m = hand_span_world_m(hand.world_landmarks)
    if span_px <= 0.0 or span_m <= 0.0:
        return None
    return span_px / span_m * STANDARD_SHOULDER_M


class HandSelector:
    """프레임의 손들 중 사용자 손 하나를 추적하고 판정 신호를 공급한다."""

    def __init__(self, config, frame_width_px, frame_height_px, clock=time.monotonic):
        select_cfg = config["hand_select"]
        # 선별 유지 시간 — 손이 안 보여도 이 시간 안엔 "사용 중"으로 취급(유휴 전환·
        # 표시 유지). 추적 정체성의 해제 시각도 이것 — 초과 소실이면 다음 손은
        # 획득(이동+모양)부터 다시 (다른 사용자 승계 차단)
        self._release_sec = select_cfg.get("release_sec", 2.0)
        # 획득 기준(2026-07-31 단일 손 — 구 command_hand_lock v2 계승): 모양이
        # 보이는 손이 창 안에서 이 거리(어깨너비 배수) 이상 움직이면 사용자 손
        acquire_cfg = select_cfg.get("acquire") or {}
        self._acquire_move_shoulder = acquire_cfg.get("move_dist_shoulder", 0.25)
        self._acquire_window_sec = acquire_cfg.get("window_sec", 0.5)
        # 인계 조건(2026-08-03 2차 — _is_tracked_idle): 추적 손이 이만큼 **연속으로**
        # 놀아야 다른 손의 도전을 받는다. 키 없으면 인계 없음(종전 동작)
        self._takeover_idle_sec = select_cfg.get("takeover_idle_sec")
        # 손 모양 판별 임계 — 실측 근거는 config hand_select.hand_shape 주석 참고
        hand_cfg = select_cfg["hand_shape"]
        self._hand_extend_ratio = hand_cfg["extend_ratio"]
        self._hand_min_valid_fingers = hand_cfg["min_valid_fingers"]
        self._hand_curl_confirm_ratio = hand_cfg.get("curl_confirm_ratio",
                                                     hand_cfg["extend_ratio"])
        # 머리 앵커(2026-07-31 몸통판 — 얼굴 검출 교체, 사용자 결정): 포즈가 몸
        # 실루엣으로 잡은 머리 위치의 사람 손만 후보 — 마스크·모자 무관.
        # 섹션(head_anchor) 없으면 게이트 없음(종전)
        anchor_cfg = config.get("head_anchor") or {}
        self._head_reach_widths = anchor_cfg.get("reach_shoulder_widths")
        # 손목 반경(2026-08-03 — _filter_hands_by_anchor 주석): 키 없으면 심사 없음
        self._wrist_reach_widths = anchor_cfg.get("wrist_reach_shoulder_widths")
        # 깊이(거리) 관문 — ★2026-08-07 신설, _is_far_person_hand 참고.
        # 뒷사람 문제는 2D 위치로 원리적으로 못 푼다(겹치면 같은 자리다) —
        # 겉보기 크기가 주는 **거리**로만 갈린다. 키 없으면 심사 없음(종전)
        self._depth_reject_ratio = anchor_cfg.get("depth_reject_ratio")
        self._depth_rel_reject_ratio = anchor_cfg.get("depth_rel_reject_ratio")
        # 작동 거리 창(2026-08-07 — 물류창고 PDA 스캐너 발상, 사용자 제안):
        # 위 두 관문은 임계가 **앵커 어깨너비에 비례**해서, 어깨 관측이 붕괴하면
        # (실측 12~21px) 임계도 같이 붕괴해 관문이 통째로 꺼진다. 이건 앵커와
        # 무관한 **카메라 기준 절대 하한**이라 그 순간에도 살아 있다
        self._min_operating_depth_px = anchor_cfg.get("min_operating_depth_px")
        # 깊이 로그 throttle — FPS 보호. {거부여부: 마지막 기록 시각}로 나눠
        # 두어 흔한 "통과"가 드문 "거부"를 가리지 않게 한다(_log_depth_verdict)
        self._depth_log_sec = {True: None, False: None}
        self._head_anchor_grace_sec = anchor_cfg.get("anchor_grace_sec", 2.0)
        self._head_anchor = None        # (center_x, center_y, width) — EMA 평활
        self._head_anchor_seen_sec = None
        self._anchor_wrists = ()        # 앵커 사람의 손목 좌표 — 손목 심사의 기준
        self._anchor_shoulders = ()     # 앵커 사람의 어깨 — 몸 기준 높이선·표시
        self._anchor_elbows = ()        # 앵커 사람의 팔꿈치 — 표시 전용
        self._anchor_segmentation_mask = None  # 앵커 사람의 실루엣 마스크(numpy 0~1) —
                                        #   인식 대상 외 블러(2026-08-07)의 정밀
                                        #   기준. None = 미관측·비활성(사각형 폴백)
        self._anchor_mask_sec = None    # 그 마스크를 마지막으로 갱신한 시각 —
                                        #   유지 시간(아래)의 기준
        self._other_segmentation_mask = None  # 앵커가 **아닌** 사람들의 실루엣
                                        #   합집합 — 블러에서 빼낸다
                                        #   (_collect_other_masks)
        # 마스크 유지 시간 (2026-08-07 — anchor_segmentation_mask 주석):
        # 앵커가 잠깐 끊겨도 블러가 꺼지지 않게 직전 마스크를 이만큼 붙든다.
        # ★반드시 anchor_grace_sec보다 **길어야** 뜻이 있다 — 둘 다 "마지막
        # 관측"부터 재므로 같은 값이면 앵커가 풀리는 바로 그 순간 마스크도 함께
        # 만료돼 공백을 하나도 못 메운다(단위 테스트가 이 실수를 잡았다)
        self._mask_hold_sec = anchor_cfg.get("segmentation_mask_hold_sec", 1.5)
        self._other_wrists = ()         # **다른 사람들**의 손목 — 소유자 판정용
                                        #   (_is_other_persons_hand)
        # 실사용 중 앵커 이전을 받아 적기 위한 진단 (2026-08-04 — _log_anchor_jump)
        self._anchor_jump_log_ratio = anchor_cfg.get("jump_log_ratio")
        self._is_anchor_lost = False    # 소실 로그 1회만 — 프레임마다 찍으면 묻힌다
        # 붕괴한 어깨 관측 버리기 (2026-08-04 실기 — _is_implausible_observation)
        self._anchor_reject_ratio = anchor_cfg.get("jump_reject_ratio")
        # 위치 튐 거부 (2026-08-07 실기 로그 — _is_implausible_observation 폭 확장):
        # 폭은 거의 안 변했는데 중심만 한 주기에 크게 튀는 사례(비슷한 크기의
        # 옆/뒷사람 사이 전환)는 폭 관문을 통과한다 — 아래 함수 독스트링 참고
        self._anchor_move_reject_ratio = anchor_cfg.get("move_reject_ratio")
        self._reject_streak = 0         # 연속 거부 횟수 — 앵커가 굳는 것 방지
        # 2연속 합의 빠른 수용 (2026-08-07 실기 로그 — 사람이 있으면 자세 노이즈가
        # 상시화돼 REJECT_STREAK_LIMIT(5)까지 매번 얼어붙어 내 손이 통째로 잘렸다):
        # 거부된 관측끼리 서로 가까우면(노이즈라면 매번 다른 방향으로 튀지, 같은
        # 방향으로 두 번 연속 합의하지 않는다) 진짜 변화로 보고 즉시 받아들인다.
        # _reject_candidate = 직전에 거부됐던 (HeadDetection, frame) | None
        self._reject_candidate = None
        # 대기줄 방어 (2026-08-07 — _drop_farther_candidates). 키 없으면 비활성
        self._continuity_depth_ratio = anchor_cfg.get("continuity_depth_ratio")
        self._farther_log_sec = None    # 그 로그의 throttle — FPS 보호
        self._mask_log_sec = None       # 마스크 점검 로그 throttle (_log_mask_coverage)
        # 관측 중앙값 필터 (2026-08-07 — _median_observed_frame). 키 없으면 비활성
        self._median_count = anchor_cfg.get("observation_median_count")
        self._observed_frames = deque()
        self.anchor_head_box = None     # 시각화용 — 앵커 머리 상자
        self._frame_width_px = frame_width_px
        self._frame_height_px = frame_height_px
        self._clock = clock

        self._hands = []                # 이번 프레임 게이트 통과 HandDetection 목록
        self._tracked_center = None     # 추적 손의 마지막 중심 (x_px, y_px)
        self._tracked_sec = None        # 마지막으로 추적 손이 관측된 시각
        self._tracked_hand = None       # 이번 프레임의 추적 손 HandDetection | None
        self._acquire_tracks = []       # 획득 후보 궤적 — {last/start center·sec, shape_sec}
        self._other_tracks = []         # 추적 손이 아닌 손들의 궤적 — 내 손과
                                        #   공존했는지 표시(_update_other_tracks)
        self._tracked_moves = deque()   # 추적 손 최근 위치 — 유휴 판정(인계 조건)
        self._tracked_idle_since = None  # 유휴가 시작된 시각 — 지속 시간의 기준
        self._last_any_hand_sec = None  # 마지막으로 손이 보인 시각 — engaged 판정
        self.locked_box = None          # 시각화용 — 추적 손 주변 박스(EMA)

    # ----- 프레임 갱신 -----

    def update(self, hands, heads=None):
        """이번 프레임의 손(과 머리) 관측을 반영한다. 사용 중 여부(engaged)를 돌려준다.

        heads: HeadDetection 목록 | None(이번 프레임 머리 추론 안 함 — 앵커 유지).
        머리(포즈) 추론은 손보다 낮은 FPS로 돌므로(realtime_loop) None 프레임이 정상이다.
        """
        now_sec = self._clock()
        # 만료 확인을 관측 반영보다 먼저 — 유예가 끝난 그 프레임에서 새 머리가
        # 바로 앵커를 인수할 수 있다 (sticky: 산 앵커는 비연속 머리를 무시하므로)
        self._drop_expired_head_anchor(now_sec)
        if heads is not None:
            self._update_head_anchor(heads, now_sec)
        self._hands = self._filter_hands_by_anchor(
            hands if hands is not None else [])
        self._update_tracked_hand(now_sec)
        if self._hands:
            self._last_any_hand_sec = now_sec
        elif not self.is_engaged():
            # 유예까지 지나 손이 완전히 떠남 — 추적 상태를 비워 다음 사용자에
            # 이전 사용자 정체성·표시를 승계하지 않는다
            self._tracked_center = None
            self._tracked_sec = None
            self._acquire_tracks = []
            self._other_tracks = []
            self._tracked_moves.clear()
            self._tracked_idle_since = None
            self.locked_box = None
        return self.is_engaged()

    # ----- 머리 앵커 (2026-07-30 얼굴 → 2026-07-31 포즈 머리로 교체) -----

    def _update_head_anchor(self, heads, now_sec):
        """머리 관측 반영 — 앵커는 끈끈하다(sticky): 잡히면 그 사람만 따라간다.

        2026-07-31 실기 정정(사용자 보고 — 인식 더 잘되는 관측으로 앵커가 옮겨감):
        앵커가 살아 있는 동안 비연속 머리는 크기와 무관하게 무시하고, 교체는
        앵커가 유예(anchor_grace_sec)를 넘겨 풀린 뒤(사용자가 떠난 뒤) 가장 큰
        머리로만 일어난다. 같은 머리는 EMA 평활(떨림 흡수).
        """
        if self._head_reach_widths is None or not heads:
            return   # 게이트 비활성 또는 관측 실패(앵커는 소실 유예가 관리)
        framed = [(head, anchor_frame(head)) for head in heads]
        framed = [(head, frame) for head, frame in framed if frame is not None]
        if not framed:
            return   # 어깨도 귀도 못 세움 — 앵커는 소실 유예가 관리
        if self._head_anchor is None:
            # 가장 넓은 어깨 = **가장 가까운 사람** = 사용자 (2026-08-04: 종전엔 가장
            # 큰 머리였는데, 귀 폭은 고개 방향에 71%까지 흔들려 기준으로 못 쓴다)
            chosen, self._head_anchor = max(framed, key=lambda pair: pair[1][2])
            self._anchor_wrists = tuple(chosen.wrists)
            self._anchor_shoulders = tuple(chosen.shoulders)
            self._anchor_elbows = tuple(chosen.elbows)
            self._anchor_segmentation_mask = chosen.segmentation_mask
            self._anchor_mask_sec = self._clock()
            self._collect_other_wrists(heads, chosen)
            self._collect_other_masks(heads, chosen)
            if self._anchor_jump_log_ratio is not None:
                # 획득·해제는 **정상 흐름**이라 INFO — WARNING은 이상 신호(점프·소실)만
                # 쓴다. 로그 파일에는 INFO도 남으니 시각 대조에는 지장이 없다
                logger.info("앵커 획득: 어깨폭 %.0fpx · 중심 (%.0f,%.0f) · 관측 %d명",
                            self._head_anchor[2], self._head_anchor[0],
                            self._head_anchor[1], len(heads))
        else:
            anchor_x, anchor_y, anchor_width = self._head_anchor
            continuous = [
                (head, frame) for head, frame in framed
                if math.dist((frame[0], frame[1]), (anchor_x, anchor_y))
                <= ANCHOR_CONTINUITY_RATIO * max(frame[2], anchor_width)
            ]
            if not continuous:
                self._log_anchor_lost(heads)
                return   # 앵커 머리가 이번 관측에 없음 — 다른 머리로 옮기지 않는다
                         # (seen_sec 미갱신 — 사용자가 정말 떠났으면 유예가 앵커를 푼다)
            continuous = self._drop_farther_candidates(continuous, anchor_width)
            # ★2026-08-04 이전 버그 수정(사용자 보고 — "고정이 안 되고 이전되는 현상"):
            # 종전엔 반경 안 후보 중 **가장 큰 머리**를 골랐다. 그래서 옆으로 지나가는
            # 사람이 카메라에 더 가까우면(=더 크면) 그 사람이 이겨, 앵커가 풀리지도
            # 않은 채 그쪽으로 **끌려갔다**(손목·어깨까지 통째로 넘어간다).
            # 정체성 이음은 크기가 아니라 **위치**로 한다 — 크기는 처음 고를 때만
            # 쓴다(가장 가까운 사람 = 사용자).
            chosen, frame = min(continuous, key=lambda pair: math.dist(
                (pair[1][0], pair[1][1]), (anchor_x, anchor_y)))
            frame = self._median_observed_frame(frame)
            if not self._is_implausible_observation(self._head_anchor, frame):
                self._reject_streak = 0
                self._reject_candidate = None
                self._log_anchor_jump(self._head_anchor, frame, heads, len(continuous))
                self._accept_anchor_frame(chosen, frame, heads)
            elif self._tracked_center is not None:
                # ★2026-08-07 4차(실기 로그 — 3차의 "얼려서 보호"가 스스로 무너지는
                # 악순환이었다): 손 확정 로그 뒤 몇 초 만에 "추적 없을 때만 도는"
                # 2연속 합의 로그가 다시 찍혔다 — 얼어붙은 앵커가 실제 손 위치와
                # 어긋나 반경 밖으로 밀려났고, 그 손이 release_sec(2초) 뒤 추적
                # 해제되자 "추적 없음" 상태가 되어 보호 자체가 풀렸다. 완전히
                # 얼리는 대신(0% 반영) **아주 조금만**(TRACKED_DRIFT_ALPHA) 반영해
                # 진짜 사용자의 느린 이동은 몇 프레임에 걸쳐 계속 따라가되, 노이즈나
                # 하이재킹처럼 한 번에 크게 튀는 값은 대부분 무시된다 — 저역통과
                # 필터 원리(노이즈는 방향이 들쭉날쭉해 누적되며 상쇄되고, 진짜 이동만
                # 같은 방향으로 쌓인다). 손목·어깨·팔꿈치는 원래도 무평활 최신값이라
                # 그대로 갱신 — 소유자 판정(_is_other_persons_hand)이 낡은 데이터를
                # 안 쓰게 된다. 안전판(REJECT_STREAK_LIMIT_WHILE_TRACKED)은 유지 —
                # 살살 따라가는 것만으론 진짜 빠른 인수인계를 못 따라잡을 수 있다
                #
                # ※외형(옷 색상 히스토그램)으로 이 판단을 보강하려던 경로가
                # 여기 있었으나 **2026-08-07 제거**했다. DeepSORT 발상으로
                # 도입했는데 실기에서 정반대로 작동했다 — 테스트 인원 둘 다
                # 어두운 상의라 히스토그램 거리가 0.16~0.28로 낮게 나와 다른
                # 사람을 "같은 사람"으로 오판, 뒷사람 쪽으로 **더 빨리** 넘어가는
                # 가속 페달이 됐다(임계를 0.3->0.12로 조여도 재발). 그 뒤 config
                # 키만 빼 비활성 상태로 두었으나, 같은 날 학습 기준선에서
                # "죽은 코드가 나중에 되살아나 버그를 되돌린다"를 겪고 함께
                # 걷어냈다. 재도입은 옷 색이 뚜렷이 다른 조합으로 먼저 검증한
                # 뒤에만 — 이 조건에서는 색상의 변별력 자체가 부족하다
                self._reject_streak += 1
                if self._reject_streak > REJECT_STREAK_LIMIT_WHILE_TRACKED:
                    logger.info(
                        "앵커 관측 거부가 %d회 연속(추적중) — 받아들인다(앵커가 굳는 것 방지)",
                        REJECT_STREAK_LIMIT_WHILE_TRACKED)
                    self._reject_streak = 0
                    self._log_anchor_jump(self._head_anchor, frame, heads, len(continuous))
                    self._accept_anchor_frame(chosen, frame, heads)
                else:
                    self._accept_anchor_frame(chosen, frame, heads, alpha=TRACKED_DRIFT_ALPHA)
                    self._head_anchor_seen_sec = now_sec
                    return
            else:
                # 활성 추적 없음(제스처 사이·대기) — 2연속 합의면 빠르게, 아니면
                # REJECT_STREAK_LIMIT(느린 경로)까지 버틴다. 다음 손이 여기서
                # 정상적으로 새로 잡혀야 하므로 얼려두는 대신 재동기화를 우선한다
                agreed_with_candidate = (
                    self._reject_candidate is not None
                    and not self._is_implausible_observation(
                        self._reject_candidate[1], frame))
                if agreed_with_candidate:
                    logger.info(
                        "앵커 관측 2연속 합의로 수용: 어깨폭 %.0fpx · 중심 (%.0f,%.0f)",
                        frame[2], frame[0], frame[1])
                    self._reject_candidate = None
                    self._reject_streak = 0
                    self._log_anchor_jump(self._head_anchor, frame, heads, len(continuous))
                    self._accept_anchor_frame(chosen, frame, heads)
                else:
                    self._reject_candidate = (chosen, frame)
                    self._reject_streak += 1
                    if self._reject_streak > REJECT_STREAK_LIMIT:
                        logger.info(
                            "앵커 관측 거부가 %d회 연속(합의도 없음) — 받아들인다"
                            "(앵커가 굳는 것 방지)", REJECT_STREAK_LIMIT)
                        self._reject_streak = 0
                        self._reject_candidate = None
                        self._log_anchor_jump(self._head_anchor, frame, heads, len(continuous))
                        self._accept_anchor_frame(chosen, frame, heads)
                    else:
                        logger.info(
                            "앵커 관측 버림: 어깨폭 %.0f->%.0fpx · 중심 이동 — "
                            "사람이 이만큼 못 변한다(합의 대기)",
                            anchor_width, frame[2])
                        self._head_anchor_seen_sec = now_sec
                        return
        self._head_anchor_seen_sec = now_sec
        self._is_anchor_lost = False   # 다시 잡혔다 — 다음 소실을 또 남길 수 있게
        # ★2026-08-04 표시도 어깨로(사용자 보고 — "기준점하고 선이 많이 흔들린다.
        # 얼굴 쪽도 많이 흔들려"): 종전엔 귀 좌표로 머리 원을 그렸는데, 귀는 고개
        # 돌림에 크게 흔들려 **판정은 멀쩡한데 화면만 요동치는** 것처럼 보였다.
        # 이제 앵커(어깨 기준·EMA 평활)를 그대로 그린다 — 화면과 판정이 같은 것을
        # 가리켜야 무엇을 보고 있는지 읽을 수 있다
        anchor_x, anchor_y, anchor_span_px = self._head_anchor
        half_px = max(1.0, anchor_span_px / 2.0)
        self.anchor_head_box = (int(anchor_x - half_px),
                                int(anchor_y - half_px),
                                int(anchor_x + half_px),
                                int(anchor_y + half_px))

    def _drop_farther_candidates(self, continuous, anchor_width_px):
        """이음 후보 중 **앵커보다 뚜렷이 먼**(어깨가 작은) 사람을 뺀다 — 대기줄 방어.

        ★2026-08-07 신설(사용자 보고 — "앵커 잡힌 사람은 누끼 잘 따지는데 뒤에
        있는 사람의 얼굴이랑 손이 보였다가 사라졌다가 반복"). 실측 로그가 원인을
        정확히 짚어줬다. `반경 안 후보 2`가 55회 찍혔고, 그 순간의 점프가
        이런 모양이다:
            어깨폭 173->462px(167%) · 중심 이동 4px
            어깨폭 289->529px( 83%) · 중심 이동 11px
            어깨폭 385->575px( 49%) · 중심 이동 13px
        **중심은 거의 안 움직이는데 폭만 49~167% 요동친다** — 이게 대기줄의
        지문이다. 일렬로 서면 두 사람의 2D 위치가 사실상 같고 거리만 다르다.
        이음(_update_head_anchor)은 후보를 **위치로만** 고르므로(2026-08-04에
        "크기로 고르면 지나가는 사람에게 끌려간다"는 이유로 그렇게 바꿨다) 이
        둘을 구조적으로 구분하지 못하고, chosen이 매 관측 오간다. 그러면
        _accept_anchor_frame이 저장하는 실루엣 마스크도 같이 오가서 뒷사람이
        선명해졌다 흐려졌다 반복한다 — 앵커가 풀리지 않아도 생긴다(그래서
        16차의 마스크 유지 시간으로는 안 잡혔다).
        위치로 못 가르는 것을 **거리**로 가른다: 대기줄 뒷사람은 어깨가 뚜렷이
        작다. 같은 사람의 폭 변화는 중앙값 필터 적용 후 실측 0~6%라 여유가 크다.

        ★비대칭이라 안전하다: **더 먼(작은) 후보만** 뺀다. 더 가까운 후보는
        그대로 두므로, 앵커가 어쩌다 뒷사람에게 가 있어도 앞사람이 후보로 남아
        되돌아올 수 있다 — 이 관문이 앵커를 뒷사람에 가두지 않는다.
        전부 빠지면(앵커 폭이 붕괴한 경우 등) 원래 목록을 그대로 쓴다 —
        못 믿을 관측으로는 아무도 탈락시키지 않는다는 이 파일의 일관된 원칙.
        """
        if self._continuity_depth_ratio is None or anchor_width_px <= 0.0:
            return continuous
        kept = [pair for pair in continuous
                if pair[1][2] >= self._continuity_depth_ratio * anchor_width_px]
        if not kept or len(kept) == len(continuous):
            return continuous if not kept else kept
        now_sec = self._clock()
        if (self._farther_log_sec is None
                or now_sec - self._farther_log_sec >= DEPTH_LOG_MIN_INTERVAL_SEC):
            self._farther_log_sec = now_sec
            logger.info(
                "이음 후보에서 먼 사람 %d명 제외(대기줄) — 앵커 어깨 %.0fpx · "
                "제외된 폭 %s", len(continuous) - len(kept), anchor_width_px,
                ", ".join("%.0f" % pair[1][2] for pair in continuous
                          if pair not in kept))
        return kept

    def _median_observed_frame(self, frame):
        """최근 관측 N개의 성분별 중앙값 — 단발 붕괴(스파이크)를 **입력에서** 제거한다.

        ★2026-08-07 신설. 오늘 쌓은 방어 4겹(폭·위치 관문 · 2연속 합의 · 추적 중
        드리프트 · 스트릭 안전판)은 전부 "이상값이 **들어온 뒤** 어떻게 처리할까"를
        다룬다. 그래서 서로 부딪혔다 — 막으면 앵커가 굳어 내 손이 잘리고, 풀면
        하이재킹이 샌다. 이 둘은 같은 손잡이의 양끝이라 조정으로는 해소가 안 된다
        (3~6차가 그 반복이었다).
        중앙값은 층을 하나 더 쌓는 게 아니라 **문제를 앞으로 옮긴다**: 3개 중
        1개가 튀면 중앙값은 정의상 그 값을 **고르지 않는다** — 이상값이 결과에
        전혀 섞이지 않는다. EMA는 alpha 0.4면 쓰레기 값의 40%를 그대로 받는다는
        점에서 근본적으로 다르다. 실기 로그의 붕괴는 정확히 단발 스파이크
        형태였다(어깨폭 209→18→200px처럼 한 프레임만 무너진다 —
        jump_reject_ratio 주석의 실측), 즉 중앙값이 가장 잘 듣는 형태다.
        지속적인 진짜 이동은 (N-1)/2 프레임 지연 후 그대로 통과한다 —
        앵커 주기 10Hz에서 N=3이면 0.1초로, 사람 이동 속도에 견주면 무시할 만하다.

        키(observation_median_count)가 없으면 원본 그대로(종전 동작).
        """
        if not self._median_count or self._median_count < 2:
            return frame
        self._observed_frames.append(frame)
        while len(self._observed_frames) > self._median_count:
            self._observed_frames.popleft()
        if len(self._observed_frames) < self._median_count:
            return frame   # 아직 창이 안 찼다 — 지연 없이 원본으로 시작한다
        middle_idx = len(self._observed_frames) // 2
        return tuple(
            sorted(observed[axis] for observed in self._observed_frames)[middle_idx]
            for axis in range(3)
        )

    def _is_implausible_observation(self, prev_frame, new_frame):
        """이번 어깨 관측이 **사람일 수 없는가** — 붕괴(폭)·이탈(위치) 둘 다 본다.

        ★2026-08-04 실기 로그(사용자 보고 "기준점·선이 흔들린다 · 검은 옷에 주머니에
        손 넣으면 못 찾는다")가 폭 붕괴 원인을 특정했다. 점프 로그가 거의 전부
        `관측 1명 · 반경 안 후보 1`이었다 — 사람이 바뀐 게 아니라 **같은 사람의
        어깨 관측이 무너진다**:
            어깨폭 209→18px(91%) · 73→192px(163%) · 163→12px(92%) · 188→21px(89%)
        12~21px는 사람 어깨가 아니다. BlazePose가 양 어깨를 거의 같은 점으로
        붕괴시키는 프레임이 있다(검은 옷이라 실루엣이 안 갈리는 것이 유력).
        도달 반경이 어깨폭 × 2.09이므로 어깨폭이 12px면 반경이 25px로 쪼그라들어
        **내 손이 통째로 거부된다** — 화면 흔들림과 인식 끊김이 같은 뿌리다.

        임계 근거(2026-08-04 실측, 오염 없는 단계 2종): 정상 사용의 앵커 1주기당
        어깨폭 변화율은 중앙 0.02 · 95% 0.10이다. 걸어서 다가와도 0.1초에 10%쯤
        (1.5m 거리에서 1.5m/s면 15cm = 10%). 붕괴는 0.2~1.6이라 그 사이가 넉넉히
        빈다. 가시성으로는 못 거른다 — 붕괴한 프레임도 visibility가 0.999다
        (같은 날 실측으로 확인, docs/TODO.md).

        ★2026-08-07 위치 튐 관문 추가(사용자 실기 로그 — "뒷사람이 여전히
        인식된다"): 폭만 보면 놓치는 사례가 실측됐다. 1초 안에 점프 4연속,
        폭은 4~10%만 변했는데 중심은 23~35%(어깨너비 기준) 튀었다 — **비슷한
        크기의 옆/뒷사람 사이에서 위치만 바뀌는 경우**는 폭 관문을 그냥
        통과한다. jump_log_ratio가 이미 재던 move_ratio(같은 계산,
        _log_anchor_jump 참고)를 여기서도 관문으로 승격— 정상 사용에서 사람이
        한 주기(포즈 10Hz ≈0.1초)에 그만큼 이동할 수 없다는 같은 물리적 근거를
        위치에도 적용한다. 폭·위치 **둘 중 하나**만 넘어도 거부(OR) — 각각
        독립적으로 "사람이 이렇게 못 바뀐다"는 근거이기 때문.
        """
        prev_x, prev_y, prev_width = prev_frame
        new_x, new_y, new_width = new_frame
        if prev_width <= 0.0:
            return False
        width_ratio = abs(new_width - prev_width) / prev_width
        move_ratio = math.dist((new_x, new_y), (prev_x, prev_y)) / prev_width
        width_bad = (self._anchor_reject_ratio is not None
                     and width_ratio > self._anchor_reject_ratio)
        move_bad = (self._anchor_move_reject_ratio is not None
                    and move_ratio > self._anchor_move_reject_ratio)
        return width_bad or move_bad

    def _accept_anchor_frame(self, chosen, frame, heads, alpha=BOX_SMOOTH_ALPHA):
        """관측 1건을 앵커로 확정 — EMA 블렌드 + 손목·어깨·팔꿈치 최신화.

        2026-08-07 분리: 정상 수용·2연속 합의 수용·REJECT_STREAK_LIMIT 안전판
        수용 **모두** 같은 확정 절차를 타야 한다(_update_head_anchor가 판단만
        하고 여기로 위임) — 그중 하나라도 이 절차를 건너뛰면 앵커 좌표(판정)와
        화면 표시가 어긋난다(2026-08-04에 이미 한 번 겪은 종류의 버그).

        alpha: 블렌드 비율 — 신뢰할 수 있는 관측은 기본값(BOX_SMOOTH_ALPHA),
        활성 추적 중 의심스러운 관측은 TRACKED_DRIFT_ALPHA로 살살만 반영한다
        (_update_head_anchor 2026-08-07 4차 — "얼려서 보호"의 역설 참고).
        """
        anchor_x, anchor_y, anchor_width = self._head_anchor
        self._head_anchor = (
            anchor_x + alpha * (frame[0] - anchor_x),
            anchor_y + alpha * (frame[1] - anchor_y),
            anchor_width + alpha * (frame[2] - anchor_width),
        )
        self._anchor_wrists = tuple(chosen.wrists)   # 손목은 평활 없이 최신값
        self._anchor_shoulders = tuple(chosen.shoulders)
        self._anchor_elbows = tuple(chosen.elbows)
        self._anchor_segmentation_mask = chosen.segmentation_mask
        self._anchor_mask_sec = self._clock()
        self._collect_other_wrists(heads, chosen)
        self._collect_other_masks(heads, chosen)

    def _log_anchor_jump(self, prev_frame, new_frame, heads, candidate_count):
        """앵커가 한 주기에 크게 움직이면 남긴다 — **실사용 중 이전 순간을 잡는 도구**.

        ★2026-08-04 신설(사용자 보고 "뒷사람하고 나랑 겹치면 생김"). 겹침을 실측으로
        재현하려 했으나 실패했다: 뒷사람이 겹치면 포즈가 **아예 안 잡힌다**(2회 측정
        215관측 중 동시 관측 0회, 검출 문턱 0.3으로 낮춰도 같다). 재현 조건을 모르니
        합성 단계로는 못 잡고, 실사용 중에 나온 순간을 받아 적는 쪽이 확실하다.

        임계는 실측 기준선에서 왔다(2026-08-04, 오염 없는 단계 2종): 정상 사용의
        어깨폭 변화는 앵커 1주기에 중앙 0.02 · 95% 0.10이다. 그 두 배를 넘으면
        사람이 그렇게 못 움직인 것이므로 관측이 다른 사람으로 갔다는 뜻이다.
        정상 사용에서는 거의 안 찍히므로 로그가 넘치지 않는다.
        """
        if self._anchor_jump_log_ratio is None:
            return
        prev_x, prev_y, prev_width = prev_frame
        if prev_width <= 0.0:
            return
        width_ratio = abs(new_frame[2] - prev_width) / prev_width
        move_ratio = math.dist((new_frame[0], new_frame[1]), (prev_x, prev_y)) / prev_width
        if max(width_ratio, move_ratio) < self._anchor_jump_log_ratio:
            return
        logger.info(
            "앵커 점프: 어깨폭 %.0f->%.0fpx(%.0f%%) · 중심 (%.0f,%.0f)->(%.0f,%.0f)"
            "(%.0f%% 어깨너비) · 관측 %d명 · 반경 안 후보 %d",
            prev_width, new_frame[2], width_ratio * 100.0,
            prev_x, prev_y, new_frame[0], new_frame[1], move_ratio * 100.0,
            len(heads), candidate_count)

    def _log_anchor_lost(self, heads):
        """앵커 머리가 반경 안에서 사라진 순간 — 이전의 **방아쇠**라 한 번만 남긴다.

        연속 프레임마다 찍으면 로그가 넘쳐 정작 이전 순간이 묻힌다. 소실이 시작될
        때만 남기고, 다시 잡히면 플래그를 푼다(_update_head_anchor 끝).
        """
        if self._anchor_jump_log_ratio is None or self._is_anchor_lost:
            return
        self._is_anchor_lost = True
        logger.info("앵커 머리 소실 — 반경 안 후보 0 · 관측 %d명 (유예가 관리)",
                    len(heads))

    def _collect_other_masks(self, heads, chosen):
        """앵커가 **아닌** 사람들의 실루엣을 하나로 합친다 — 블러에서 빼낼 영역.

        ★2026-08-07 신설(사용자 요청 — "앵커 잡힌 사람만 보이고 앵커 안잡힌
        사람은 여전히 블러처리 되어있으면 좋겠는데"). 종전에는 앵커 마스크만
        쓰고 **나머지 사람의 마스크는 버렸다**. 그런데 포즈는 사람별 실루엣을
        이미 다 주고 있으므로, "누가 앵커가 아닌가"를 픽셀 단위로 정확히 알 수
        있다 — 추정할 필요가 없는 정보를 안 쓰고 있었던 셈이다.
        이걸 앵커 마스크에서 빼면(blur_outside_mask의 exclude_mask) 두 실루엣이
        겹치는 애매한 구간은 **흐리는 쪽으로** 판정된다. 가까이 붙을수록
        앵커 마스크의 유출이 커지는데, 유출이 정확히 상대 실루엣 위에서
        일어나므로 이 뺄셈이 그 부분을 정조준한다.

        임계는 적용하지 않고 **원본 확률값의 성분별 최댓값**으로 합친다 —
        임계는 소비자(realtime_loop)가 config로 정하는 값이라 여기서 미리
        박으면 그 계약이 깨진다.
        """
        others = [head.segmentation_mask for head in heads
                  if head is not chosen and head.segmentation_mask is not None]
        combined = None
        if others:
            combined = others[0]
            for mask in others[1:]:
                combined = np.maximum(combined, mask)
        self._other_segmentation_mask = combined
        self._log_mask_coverage(heads, chosen, combined)

    def _log_mask_coverage(self, heads, chosen, other_mask):
        """마스크가 실제로 몇 명을 덮고 있는지 남긴다 — **미검증 가설의 판별용**.

        ★2026-08-07 신설(사용자 보고 — 방어 4겹 뒤에도 "손도 얼굴도 여전히
        인식된다"). 여기까지의 수정은 전부 "포즈가 두 사람을 각각 잡는다"를
        전제했는데, 그 전제가 실기에서 성립하는지 **확인한 적이 없다**.
        docs/TODO.md에는 오히려 반대 실측이 있다 — "겹치면 포즈가 둘을 동시에
        못 잡는다(215관측 중 0회)". 겹칠 때 한 사람만 잡히고 **그 하나의
        마스크가 두 사람을 통째로 덮는다면**, 뺄 상대 마스크 자체가 없어
        18차 수정이 작동할 수 없다.
        판별 방법: 혼자일 때 앵커 마스크는 프레임의 2~4%였다(같은 날 실측).
        관측이 1명인데 덮는 비율이 그보다 뚜렷이 크면 = 두 사람을 한 마스크로
        묶은 것이다. 그 경우 픽셀 기반 분리는 원리적으로 불가능하고, 방어는
        깊이 관문(손 판정) 쪽에만 기대야 한다.
        throttle을 걸어 FPS를 갉지 않는다.
        """
        anchor_mask = chosen.segmentation_mask
        if anchor_mask is None:
            return
        now_sec = self._clock()
        if (self._mask_log_sec is not None
                and now_sec - self._mask_log_sec < DEPTH_LOG_MIN_INTERVAL_SEC):
            return
        self._mask_log_sec = now_sec
        anchor_ratio = float((anchor_mask >= 0.5).mean()) * 100.0
        other_ratio = (float((other_mask >= 0.5).mean()) * 100.0
                       if other_mask is not None else None)
        logger.info(
            "마스크 점검: 관측 %d명 · 앵커 마스크가 화면의 %.1f%% · 타인 마스크 %s "
            "(혼자일 때 실측 2~4%%. 1명 관측인데 %%가 크면 한 마스크가 두 사람을 덮는 것)",
            len(heads), anchor_ratio,
            "%.1f%%" % other_ratio if other_ratio is not None else "없음")

    def _collect_other_wrists(self, heads, chosen):
        """앵커가 아닌 사람들의 **믿을 수 있는** 손목만 모은다 — 소유자 판정의 상대편.

        앵커 손목과 같은 관문을 통과시킨다(_usable_wrists와 같은 규칙): 못 믿을
        추정으로는 아무도 탈락시키지 않아야 한다 — 남의 손목이 엉뚱한 곳에
        찍히면 내 손이 잘리기 때문이다.
        """
        wrists = []
        for head in heads:
            if head is chosen:
                continue
            wrists.extend(reliable_wrists(head.wrists, head.shoulders))
        self._other_wrists = tuple(wrists)

    def _drop_expired_head_anchor(self, now_sec):
        """머리가 유예(anchor_grace_sec)를 넘겨 사라짐 — 게이트를 끈다(모든 손 통과).

        머리 미검출(상체 프레임 밖 등)로 키오스크가 먹통이 되는 것보다 방어가
        잠시 꺼지는 쪽이 안전하다 — 앵커 없는 추적으로 폴백.
        ※포즈 머리는 마스크·모자로는 안 끊긴다(몸 맥락 추정).

        ★2026-08-04 앵커 고정(사용자 결정 — "선은 인식되면 고정으로 박아야 할 것
        같다"): **추적 중인 손이 있는 동안엔 앵커를 풀지 않는다.** 사용 중인
        사람이 있는데 앵커가 풀리는 건 명백히 틀렸고, 실기 노이즈의 시작점이
        바로 거기였다 — 사람이 여럿이면 내가 포즈 관측에서 잠깐만 빠져도
        유예가 끝나 앵커가 풀리고, 그 순간 **가장 큰 사람**에게 다시 붙어
        내 손이 통째로 거부됐다(사용자 보고 "인식 손이 꺼짐").
        해제는 세션이 끝날 때만 — 추적 정체성이 release_sec 초과 소실로 풀린
        뒤에야 앵커도 풀린다. 그래야 다음 사용자가 정상적으로 인수한다.

        ★2026-08-07 segmentation_mask는 여기서 비우지 않는다:
        anchor_segmentation_mask()로 realtime_loop가 **앵커 유무와 무관하게 매
        프레임** 읽어 블러 여부를 정한다 — 여기서 안 비우면 앵커가 없는(게이트
        꺼진) 동안에도 떠난 사람의 낡은 실루엣으로 다음 사람을 계속 블러해,
        이 함수의 핵심 취지("방어보다 인식 우선")를 정확히 거스른다.
        """
        if self._tracked_center is not None:
            return   # 사용 중 — 앵커는 그대로 박아 둔다 (다음 사용자 인수는 정체성
                     #   해제 후에. _update_tracked_hand의 release_sec 경로가 먼저 푼다)
        if (self._head_anchor is not None and self._head_anchor_seen_sec is not None
                and now_sec - self._head_anchor_seen_sec > self._head_anchor_grace_sec):
            if self._anchor_jump_log_ratio is not None:
                logger.info("앵커 해제: 소실 유예 %.1f초 초과 — 다음 사람이 인수한다",
                            self._head_anchor_grace_sec)
            self._head_anchor = None
            self._head_anchor_seen_sec = None
            self._anchor_wrists = ()
            self._anchor_shoulders = ()
            self._anchor_elbows = ()
            # ★마스크는 여기서 안 비운다(2026-08-07 — anchor_segmentation_mask
            # 주석): 앵커 해제 순간 블러가 통째로 꺼져 대기줄이 노출되는 깜빡임의
            # 원인이었다. 대신 **유지 시간**으로 늙어 죽게 한다 — 낡은 마스크가
            # 다음 사람에게 승계되지 않는다는 원래 목적은 그대로 지켜진다
            self._other_wrists = ()
            self._observed_frames.clear()   # 떠난 사람의 관측이 다음 사람의
                                           #   중앙값에 섞이면 안 된다
            self.anchor_head_box = None
            self._is_anchor_lost = False

    def _filter_hands_by_anchor(self, hands):
        """앵커(가장 가까운 사람)의 팔에서 나온 손만 남긴다 — 옆 사람 손 차단.

        심사 2중:
        ① 머리 반경 — 앵커 머리 폭(귀-귀) × reach_head_widths (기본 5.0: 성인
           ≈ 0.15m × 5 = 팔 도달 0.75m). 머리 폭 비례라 카메라 거리 무관.
        ② 손목 반경(2026-08-03 신설) — 앵커의 **포즈 손목**에서 머리 폭 ×
           wrist_reach_head_widths 안. 키·손목 관측이 없으면 이 심사는 없다.

        ★②를 넣은 이유(사용자 보고 — "내 몸에서 뻗어나가는 손이 아닌데도 남의 몸
        손이 잡힌다"): ①만으로는 원리적으로 못 가른다. 팔을 옆으로 뻗으면
        내 머리~내 손 거리가 옆 사람 손까지의 거리와 같아진다 — 좁히면 내 손이
        잘리고 넓히면 남이 샌다. 손목을 알면 "내 팔 끝인가"를 직접 물을 수 있고,
        판정 영역이 **내 팔을 따라다녀** 팔을 내리면 옆 사람 자리를 안 덮는다.
        실측(2026-08-03, 포즈 10 FPS·미검출 0프레임): 내 손↔내 손목 거리는
        천천히 최대 2.25 · 빠른 쓸기 최대 1.82 · 팔 끝까지 뻗어 최대 2.30
        (머리 폭 배수) → 임계 3.0은 30% 여유.

        ★경성 게이트(2026-07-31 2차): 앵커가 살아 있는 동안 반경 밖 새 손은
        통과하지 않는다 — fail-open 구멍 봉쇄. 앵커가 아예 없으면 전부
        통과(인식 우선 폴백).
        ★추적 면제(2026-07-31): 반경은 입장 심사만 — 추적 중인 손(신선 +
        재등장 반경)은 밖으로 뻗어도 면제 (획 끝점이 잘리지 않게).
        """
        if self._head_reach_widths is None or self._head_anchor is None:
            return hands
        anchor_x, anchor_y, anchor_width = self._head_anchor
        reach_px = self._head_reach_widths * anchor_width
        # ★2026-08-06 계약 정합(사용자 결정 "어깨만 잡는데 팔목쪽 없애도 되지
        # 않니"): 키(wrist_reach_shoulder_widths)가 없으면 손목 기반 심사 **둘 다**
        # (반경 + 소유자 판정) 꺼진다 — config 주석이 약속한 계약인데 종전 코드는
        # 소유자 판정이 키와 무관하게 돌았다. 포즈 손목 관측이 붕괴 상시(실기
        # 로그: 어깨폭 19~40px 요동)라 심사가 내 손을 자르는 쪽으로만 작동했다
        usable_wrists = (() if self._wrist_reach_widths is None
                         else self._usable_wrists())
        wrist_reach_px = (None if self._wrist_reach_widths is None or not usable_wrists
                          else self._wrist_reach_widths * anchor_width)
        nearest_depth_px = self._nearest_hand_depth_px(hands)
        in_reach = []
        for hand in hands:
            center = hand_center_point(hand.landmarks)
            if center is None:
                continue
            # ★깊이 관문은 **추적 면제보다 먼저** 본다(2026-08-07 — 아래 함수
            # 주석): 면제는 "반경 밖으로 뻗은 획을 자르지 않기" 위한 위치
            # 예외인데, 뒷사람 손이 한 번이라도 추적에 들어오면 그 뒤로는
            # 위치 심사를 통째로 면제받아 **영영 못 쫓아낸다** — 오늘 하루
            # 반복된 "한번 뺏기면 안 돌아온다"의 구조적 뒷문이 여기였다.
            # 거리는 획을 그리는 동안 뒤로 물러나지 않으므로(팔을 뻗으면
            # 오히려 가까워진다) 이 관문이 내 획을 자를 이유가 없다
            if self._is_far_person_hand(hand, anchor_width, nearest_depth_px):
                continue
            # 실루엣 소유자 판정도 **추적 면제보다 먼저**다 — 옆 사람 손이
            # 한 번 추적에 들면 위치 심사를 면제받아 못 쫓아내기 때문(깊이
            # 관문과 같은 이유). 상대 몸 위에 있는 손은 획 중에도 내 손일 수 없다
            if self._is_on_other_silhouette(center):
                continue
            if self._is_tracked_continuation(center, hand):
                in_reach.append(hand)       # 추적 중인 손 — 위치 심사 면제
                continue
            if math.dist(center, (anchor_x, anchor_y)) > reach_px:
                continue
            if wrist_reach_px is not None and all(
                    math.dist(center, wrist[:2]) > wrist_reach_px
                    for wrist in usable_wrists):
                continue                    # 앵커의 어느 손목에서도 멀다 — 남의 손
            if self._is_other_persons_hand(center, usable_wrists):
                continue
            in_reach.append(hand)
        return in_reach

    def _nearest_hand_depth_px(self, hands):
        """이번 프레임 손들 중 **가장 가까운** 손의 깊이값(px) | None — 상대 관문의 기준.

        가장 큰 값 = 가장 가까운 손이다(hand_shoulder_px 주석 — 값은 거리에만
        의존하고 멀수록 작아진다). 손이 하나뿐이면 비교 대상이 없어 상대 관문은
        자동으로 꺼진다(그때는 절대 관문이 담당한다).
        """
        depths = [hand_shoulder_px(hand) for hand in hands]
        depths = [depth for depth in depths if depth is not None and depth > 0.0]
        return max(depths) if depths else None

    def _is_far_person_hand(self, hand, anchor_width_px, nearest_depth_px):
        """이 손이 **앵커보다 뒤에 있는 사람**의 손인가 — 깊이(겉보기 크기) 관문.

        ★2026-08-07 신설 (사용자 보고 13회차 — "뒤에 사람이 손 올리면 인식된다").
        오늘 하루 시도한 방어는 전부 **2D 위치**(반경·손목 거리·소유자 비교)와
        **화면 가림**(실루엣 블러) 두 축에 있었는데, 둘 다 뒷사람에 대해서는
        원리적으로 못 푸는 축이다 — 사람이 내 **뒤**에 서면 화면에서는 내 옆·
        내 위에 겹쳐 보이고, 거리를 모르면 그 손과 내 손이 화면상 구분되지 않는다.
        오늘 조사한 MOT 문헌(8차)이 말한 "geometry만으로는 occlusion에서 진다"가
        정확히 이 얘기다.

        빠져 있던 축이 **깊이**다. 그런데 이 엔진은 이미 깊이를 재고 있었다 —
        hand_shoulder_px(월드 미터 ↔ 화면 px의 비)는 손 크기가 약분돼
        f/d(거리)만 남는 값이다(그 함수 주석의 유도 참고). 쓰던 목적이
        "자(尺)"였을 뿐, 같은 값이 곧 거리계다. 새 모델도 새 의존도 필요 없다.

        관문 둘을 겹친다 — 서로가 서로의 빈틈을 메운다:
        ① **절대**(depth_reject_ratio): 이 손의 깊이 ÷ 앵커 어깨너비(px).
           앵커 어깨너비도 f × 어깨(m) / d_앵커 라, 이 비는
               (0.4 / 내_어깨m) × (d_앵커 / d_손)
           이 된다. 같은 사람이면 뒤 항이 1이라 체형 상수(0.4/어깨)만 남고,
           뒷사람 손이면 d_손이 커져 비가 그만큼 작아진다. 1.5m에 선 나와
           2.5m에 선 뒷사람이면 0.6배다. 체형 편차(어깨 0.34~0.48m → 0.83~1.18)
           보다 뚜렷이 낮아 갈린다. **손이 하나뿐인 프레임에서도 듣는다**는 것이
           이 관문의 값어치 — 뒷사람만 손을 든 순간이 정확히 그 상황이다.
        ② **상대**(depth_rel_reject_ratio): 이 손의 깊이 ÷ 이번 프레임 최근접
           손의 깊이. 체형 상수(0.4/내_어깨)가 분자·분모에서 **약분돼 사라져**
           체형 보정이 아예 필요 없다 — 코드베이스가 이미 신뢰하는 상대 비교
           방식(_is_other_persons_hand의 "임계값 없이 풀린다")과 같은 성질이다.
           내 손과 뒷사람 손이 함께 보이는 순간에 강하다.

        둘 중 하나라도 걸리면 거부(OR) — 각각 독립적인 물리적 근거다.
        키가 없으면 그 관문은 꺼지고, 깊이를 못 재면(월드 랜드마크 이상) 아무도
        안 자른다 — 이 코드베이스의 일관된 원칙(못 믿을 관측으로는 탈락시키지
        않는다, reliable_wrists 주석)을 따른다.

        ※한계: 뒷사람이 **바로 뒤**(수십 cm)에 붙으면 깊이 차가 작아 못 가른다.
        단일 카메라의 물리적 한계다 — 다만 그 거리면 화면에서도 크기가 거의
        같아, 어떤 방법으로도 못 가르는 영역이다. 값은 실측 전 초안
        (config.yaml head_anchor.depth_reject_ratio 주석).
        """
        if (self._depth_reject_ratio is None and self._depth_rel_reject_ratio is None
                and self._min_operating_depth_px is None):
            return False
        depth_px = hand_shoulder_px(hand)
        if depth_px is None or depth_px <= 0.0:
            return False   # 깊이를 못 쟀다 — 인식 우선 폴백(아무도 안 자른다)
        # ★작동 거리 창(2026-08-07 — PDA 스캐너 발상): 아래 두 관문은 앵커
        # 어깨너비에 비례하는데, 어깨 붕괴(실측 12~21px)가 나면 임계가 함께
        # 붕괴해 **가장 필요한 순간에 관문이 꺼진다**(붕괴가 잦은 게 하필
        # 사람이 겹칠 때다). 이 하한은 카메라 기준 절대값이라 그때도 산다.
        # 바코드 스캐너가 정해진 작동 거리 밖은 아예 디코딩을 시도하지 않는
        # 것과 같은 발상 — "이 거리 밖 사람은 이 키오스크 사용자가 아니다"
        if (self._min_operating_depth_px is not None
                and depth_px < self._min_operating_depth_px):
            self._log_depth_verdict(depth_px, anchor_width_px, nearest_depth_px,
                                    False, False, is_out_of_zone=True)
            return True
        # ※사용자별 체형 상수를 **학습**해 이 임계를 대신하려던 시도가 있었으나
        # 2026-08-07 당일 제거했다 — 전제("본인 손이면 비가 체형 상수로 일정")가
        # 틀렸다. 손을 뻗으면 비가 2~3까지 오르므로 이 값은 체형 상수가 아니라
        # '체형 × 팔 뻗음'이고, 그걸로 임계를 올리면 팔을 굽히는 순간 본인 손이
        # 잘린다(config.yaml head_anchor 절의 경고 주석 참고). 다시 만들지 말 것
        reject_ratio = self._depth_reject_ratio
        far_by_anchor = (
            reject_ratio is not None and anchor_width_px > 0.0
            and depth_px < reject_ratio * anchor_width_px)
        # ★기준점 보호(2026-08-07 코드 리뷰 지적): 상대 관문은 "이번 프레임
        # 최근접 손 = 내 손"을 암묵적으로 가정한다. 뒷사람에는 맞지만, 나보다
        # **앞으로** 끼어든 사람이 있으면 그 손이 기준이 되어 **내 손이 잘린다**
        # (1.0m 침입자 vs 1.5m 나 → 0.67 < 0.70). 실측이 해법을 준다 —
        # 내 손의 절대비는 1.07~1.76, 뒷사람 손은 0.51~0.78이었다(config 주석).
        # 절대비가 1.0 이상이면 앵커 몸통만큼 가깝다는 뜻이라 "뒤에 있는 사람"일
        # 수 없다 — 상대 관문에서 면제한다. 보수적 하한이다(같은 거리의 내 손은
        # 체형에 따라 1.07~1.26이라 1.0에 여유가 있다)
        is_at_anchor_depth = anchor_width_px > 0.0 and depth_px >= anchor_width_px
        far_by_peer = (
            self._depth_rel_reject_ratio is not None and nearest_depth_px is not None
            and not is_at_anchor_depth
            and depth_px < self._depth_rel_reject_ratio * nearest_depth_px)
        is_far = far_by_anchor or far_by_peer
        self._log_depth_verdict(depth_px, anchor_width_px, nearest_depth_px,
                                far_by_anchor, far_by_peer)
        return is_far

    def _log_depth_verdict(self, depth_px, anchor_width_px, nearest_depth_px,
                           far_by_anchor, far_by_peer, is_out_of_zone=False):
        """깊이 관문의 판정을 남긴다 — **통과도 함께** 남기되 throttle 건다.

        거부만 남기면 임계 튜닝을 못 한다: 임계(0.70)는 기하 유도값이지 실측이
        아니라(config 주석), 실제 사용에서 **내 손의 비가 몇으로 나오는지**를
        알아야 여유가 얼마인지 판단할 수 있다. 거부 로그만 있으면 "안 잘렸다"는
        사실만 알고 아슬아슬하게 안 잘린 것인지 넉넉했는지를 모른다.
        프레임마다 찍으면 파일 핸들러 동기 flush로 FPS를 갉으므로
        (2026-08-07 7차에 실제로 겪었다) 최소 간격을 둔다.

        ★throttle을 통과/거부 **따로** 둔다(2026-08-07 코드 리뷰 지적): 하나를
        공유하면 통과가 먼저 찍힌 창에서는 **그 뒤의 거부가 조용히 버려진다** —
        정작 튜닝에 필요한 쪽이 흔한 쪽에 가려지는 구조였다. 따로 두면 각각
        초당 0.5줄로 제한되면서도 거부가 통과에 묻히지 않는다.
        """
        is_rejected = far_by_anchor or far_by_peer or is_out_of_zone
        now_sec = self._clock()
        last_sec = self._depth_log_sec[is_rejected]
        if last_sec is not None and now_sec - last_sec < DEPTH_LOG_MIN_INTERVAL_SEC:
            return
        self._depth_log_sec[is_rejected] = now_sec
        anchor_ratio = (depth_px / anchor_width_px if anchor_width_px > 0.0
                        else float("nan"))
        peer_ratio = (depth_px / nearest_depth_px if nearest_depth_px else None)
        if is_out_of_zone:
            reason = "작동거리밖"
        elif far_by_anchor and far_by_peer:
            reason = "절대+상대"
        elif far_by_anchor:
            reason = "절대"
        elif far_by_peer:
            reason = "상대"
        else:
            reason = None
        logger.info(
            "깊이 관문 %s: 손 깊이 %.0fpx · 앵커 어깨 %.0fpx(비 %.2f) · "
            "최근접 손 대비 %s%s",
            "거부" if is_rejected else "통과",
            depth_px, anchor_width_px, anchor_ratio,
            "%.2f" % peer_ratio if peer_ratio is not None else "-",
            " · 사유 %s" % reason if reason else "")

    def _is_on_other_silhouette(self, center):
        """이 손이 **다른 사람의 실루엣 위**에 있는가 — 손목에 기대지 않는 소유자 판정.

        ★2026-08-07 신설(사용자 보고 — "옆사람한테 초록색 관절값이 찍힌다").
        옆 사람은 거리가 같아 깊이 관문이 안 듣고, 남은 방어인 손목 반경과
        소유자 비교(_is_other_persons_hand)는 **내 손목을 못 믿으면 둘 다
        통째로 꺼진다**(_filter_hands_by_anchor의 usable_wrists 계약). 포즈
        손목은 붕괴가 잦아(실기 로그) 그 상황이 드물지 않다.

        그런데 엔진은 이미 더 든든한 신호를 만들어 놓고 블러에만 쓰고 있었다 —
        **사람별 실루엣 마스크**다. 손 중심 화소가 누구 몸 위에 있는지 직접
        물으면 손목 추정이 필요 없다. "저 손이 내 몸에서 나왔나, 옆 사람
        몸에서 나왔나"는 원래 이렇게 묻는 것이 정확하다.

        판정은 **확실할 때만** 자른다 — 상대 실루엣 안이면서 내 실루엣 밖일
        때. 뻗은 팔은 세그멘테이션이 놓치는 일이 있어 양쪽 다 밖일 수 있는데,
        그때 자르면 내 손이 사라진다(오늘 세 번 겪은 실패 유형). 그런 애매한
        경우는 통과시키고 다른 관문에 맡긴다.
        """
        other_mask = self._other_segmentation_mask
        anchor_mask = self._anchor_segmentation_mask
        if other_mask is None or anchor_mask is None:
            return False
        x_px, y_px = int(center[0]), int(center[1])
        height_px, width_px = other_mask.shape[:2]
        if not (0 <= x_px < width_px and 0 <= y_px < height_px):
            return False
        if anchor_mask.shape[:2] != other_mask.shape[:2]:
            return False
        return bool(other_mask[y_px, x_px] >= SILHOUETTE_THRESHOLD
                    and anchor_mask[y_px, x_px] < SILHOUETTE_THRESHOLD)

    def _is_other_persons_hand(self, center, anchor_wrists):
        """다른 사람의 손목이 앵커의 손목보다 이 손에 **더 가까운가** — 소유자 판정.

        ★2026-08-04 실기(사용자 보고 — "다른 사람 난입 시 노이즈"): 반경만으로는
        원리적으로 못 가른다(실측: 팔을 뻗은 내 손까지의 거리 ≈ 옆 사람까지의 거리).
        그런데 **두 사람의 골격을 다 보면** 임계값 없이 풀린다 — 이 손이 누구
        팔 끝에 붙어 있는지는 상대 비교로 정해진다.
        비교 대상은 양쪽 다 신뢰할 수 있는 손목뿐이다(_usable_wrists와 같은
        관문) — 남의 손목 추정이 엉뚱하면 내 손을 자를 수 있으므로, 못 믿을
        관측으로는 아무도 탈락시키지 않는다.
        """
        if not anchor_wrists or not self._other_wrists:
            return False
        nearest_anchor = min(math.dist(center, wrist[:2]) for wrist in anchor_wrists)
        nearest_other = min(math.dist(center, wrist[:2]) for wrist in self._other_wrists)
        return nearest_other < nearest_anchor

    def _usable_wrists(self):
        """앵커 사람의 믿을 수 있는 손목 — 순수 함수 reliable_wrists의 얇은 래퍼."""
        return reliable_wrists(self._anchor_wrists, self._anchor_shoulders)

    def _is_tracked_continuation(self, center, hand):
        """추적 중이던 손의 연장인가 — 추적점 근처(재등장 반경) + 신선할 때만.

        신선도(EXEMPT_FRESH_SEC): 면제는 프레임 연속으로 추적되던 손의 것 —
        내린 손의 오래된 추적점 근처에 나중에 나타난 손(옆 사람)이 면제를
        승계하던 실기 구멍을 막는다.
        """
        now_sec = self._clock()
        return (self._tracked_center is not None and self._tracked_sec is not None
                and now_sec - self._tracked_sec <= EXEMPT_FRESH_SEC
                and math.dist(center, self._tracked_center)
                <= REENTRY_SPAN_RATIO * stable_span_px(hand))

    # ----- 단일 손 추적 (2026-07-31 — 라벨 제거) -----

    def _update_tracked_hand(self, now_sec):
        """추적 손 상태 갱신 — 이음(연속성) 또는 획득(이동+모양).

        이음: 마지막 추적점 근처 후보 중 최근접 = 같은 손. release_sec 안이면
        소실 후 재등장도 잇는다(화면 가리킴의 수 초 소실 — 구 래치 소실 유예·
        rejoin과 같은 역할). 반경 밖 후보는 무시 — 다른 사람 손이 정체성을 뺏지
        못한다. 해제: release_sec 초과 소실 — 다음 손은 획득부터.

        ★2026-08-03 실기(사용자 보고 — "제스처 시전 중인 손보다 가만히 있는 손이
        인식이 더 잘돼서 포커스가 이동된다"): 제스처 중인 손은 빨라서 모션 블러로
        몇 프레임 검출이 끊기는데, 그 프레임에는 **가만히 있는 손이 마지막
        추적점에 더 가깝다** — 재등장 반경(4손폭)이 그 손을 내 손으로 이어
        정체성이 통째로 넘어갔다. 반경을 좁혀도 못 고친다: 끊긴 프레임에는
        그 손밖에 없기 때문이다.
        판단 근거를 추가한다 — **내 손이 추적되던 동안 이미 따로 보이던 손은
        내 손일 수 없다**(_other_tracks). 그런 손은 이음 후보에서 제외하고,
        진짜 재등장(그 순간 없던 손)만 잇는다.

        ★같은 날 2차(사용자 보고 — "아예 배에 있는 손에 고정돼 버린다"): 위
        수정은 "뺏기는" 경로만 막았다. 남은 절반은 **한번 잡힌 손을 영영 안 놓는
        것**이다 — 배 앞의 쉬는 손이 먼저 잡히면 다른 손으로 아무리 크게
        제스처해도 못 들어온다(해제는 2초 소실뿐). 제스처하는 손이 이겨야 한다.
        무작정 뺏게 하면 구 "배구 토스"가 돌아오므로 조건을 단다 —
        **추적 손이 놀고 있을 때만** 도전자를 받는다(_is_tracked_idle +
        기존 획득 규칙 그대로). 획을 그리는 중에는 어떤 손도 못 뺏는다.
        """
        self._tracked_hand = None
        candidates = []
        for hand in self._hands:
            center = hand_center_point(hand.landmarks)
            if center is not None:
                candidates.append((hand, center))
        if self._tracked_center is not None:
            if now_sec - self._tracked_sec > self._release_sec:
                self._tracked_center = None   # 소실 유예 초과 — 정체성 해제
                self._tracked_sec = None
                self._other_tracks = []       # 남 손 이력도 함께 — 다음 사용자에 승계 금지
                self._tracked_moves.clear()
                self._tracked_idle_since = None
            else:
                best = None
                for hand, center in candidates:
                    dist_px = math.dist(center, self._tracked_center)
                    if dist_px > REENTRY_SPAN_RATIO * stable_span_px(hand):
                        continue
                    if self._is_known_other_hand(center, hand):
                        continue   # 내 손이 보이던 때 함께 보이던 손 — 남의 손이다
                    if best is None or dist_px < best[2]:
                        best = (hand, center, dist_px)
                if best is not None:
                    self._tracked_hand = best[0]
                    self._tracked_center = best[1]
                    self._tracked_sec = now_sec
                    self._update_display_box(best[0])
                    self._stamp_tracked_move(best[1], now_sec)
                if self._is_tracked_idle(now_sec):
                    # 추적 손이 논다 — 다른 손이 제대로 지시하면 넘겨준다
                    others = [(hand, center) for hand, center in candidates
                              if hand is not self._tracked_hand]
                    if self._acquire_tracked_hand(others, now_sec):
                        self._other_tracks = []   # 역할이 바뀌었다 — 표시 재수집
                        self._tracked_moves.clear()
                        self._tracked_idle_since = None
                else:
                    self._acquire_tracks = []     # 획 중 — 도전자 궤적은 버린다
                self._update_other_tracks(candidates, now_sec)
                return   # 추적 유지 중 — 아래 획득 경로는 돌지 않는다
        self._update_other_tracks(candidates, now_sec)
        self._acquire_tracked_hand(candidates, now_sec)

    def _stamp_tracked_move(self, center, now_sec):
        """추적 손의 최근 위치 이력 — 유휴 판정(_is_tracked_idle)의 근거."""
        self._tracked_moves.append((now_sec, center))
        # 창 경계에 걸친 표본 **하나는 남긴다** — 전부 잘라내면 가장 오래된
        # 항목이 항상 창보다 어려서 "창을 채웠나" 판정이 영영 참이 안 된다
        while (len(self._tracked_moves) > 1
               and now_sec - self._tracked_moves[1][0] > self._acquire_window_sec):
            self._tracked_moves.popleft()

    def _is_tracked_idle(self, now_sec):
        """추적 손이 **손을 놓아둔 상태**인가 — 인계를 허용할지의 판단.

        조건 둘: ①창(acquire.window_sec) 안 이동이 획득 임계 미만
                 ②그 상태가 takeover_idle_sec 동안 **연속으로** 이어짐

        ★②를 넣은 이유(2026-08-03 실측): ①만 걸었더니 10초에 인계가 4번 일어나
        추적이 두 손 사이를 오갔다(위 손 추적 64%뿐). **쓸기와 쓸기 사이의 짧은
        멈춤**도 유휴로 잡혀 그때마다 도전 틈이 생긴 것이다. 구분해야 할 것은
        "잠깐 멈춤"(0.3~0.5초)과 "손을 놓아둔 상태"(수 초)라, 지속 시간으로 가른다.
        이력이 창을 못 채웠으면(방금 잡힘·재등장 직후) 유휴로 보지 않는다 —
        판단 근거가 모자란 상태에서 넘기지 않는다.
        """
        if self._takeover_idle_sec is None or self._tracked_hand is None:
            return False
        if len(self._tracked_moves) < 2 or (
                now_sec - self._tracked_moves[0][0] < self._acquire_window_sec):
            return False
        shoulder_px = hand_shoulder_px(self._tracked_hand)
        if shoulder_px is None:
            return False
        first_center = self._tracked_moves[0][1]
        travel_px = max(math.dist(center, first_center)
                        for _, center in self._tracked_moves)
        if travel_px >= self._acquire_move_shoulder * shoulder_px:
            self._tracked_idle_since = None   # 움직였다 — 유휴 시계 리셋
            return False
        if self._tracked_idle_since is None:
            self._tracked_idle_since = now_sec
        return now_sec - self._tracked_idle_since >= self._takeover_idle_sec

    def _is_known_other_hand(self, center, hand):
        """이 후보가 "내 손과 함께 보이던 다른 손"인가 — 이음 후보에서 제외한다.

        내 손이 관측된 프레임에 그와 별개로 존재하던 손은 정의상 내 손이 아니다.
        그 사실을 궤적에 표시(is_seen_with_tracked)해 두고, 내 손이 끊긴 사이
        그 손이 정체성을 승계하는 것을 막는다 (2026-08-03 실기 — 함수 호출부 주석).
        """
        span_px = stable_span_px(hand)
        for entry in self._other_tracks:
            if not entry["is_seen_with_tracked"]:
                continue
            if math.dist(center, entry["last_center"]) <= CONTINUITY_SPAN_RATIO * span_px:
                return True
        return False

    def _update_other_tracks(self, candidates, now_sec):
        """추적 손이 아닌 손들의 궤적을 잇는다 — 프레임 간 최근접(연속 반경).

        내 손이 이번 프레임에 관측됐다면, 함께 보인 손들에 "내 손과 공존" 표시를
        남긴다 — 그 표시가 _is_known_other_hand의 판단 근거다.

        ★2026-08-04: 종전엔 이번 프레임에 안 보인 궤적을 **즉시** 버렸다. 손은
        블러·가림·중복 억제·게이트 등 여러 이유로 한두 프레임 사라지는데, 그때마다
        표시가 지워져 그 손이 "처음 보는 손"으로 부활하면 정체성을 가져갈 자격을
        되찾는다 — 표시의 취지(내 손과 공존했다는 사실은 사라지지 않는다)에 어긋난다.
        안 보이는 동안도 OTHER_TRACK_KEEP_SEC 만큼 궤적(과 표시)을 유지한다.
        """
        is_tracked_seen = self._tracked_hand is not None
        matched = []
        for hand, center in candidates:
            if hand is self._tracked_hand:
                continue
            span_px = stable_span_px(hand)
            track, best_dist_px = None, None
            for entry in self._other_tracks:
                if entry in matched or now_sec - entry["last_sec"] > OTHER_TRACK_KEEP_SEC:
                    continue
                dist_px = math.dist(center, entry["last_center"])
                if dist_px <= CONTINUITY_SPAN_RATIO * span_px and (
                        best_dist_px is None or dist_px < best_dist_px):
                    track, best_dist_px = entry, dist_px
            if track is None:
                track = {"last_center": center, "last_sec": now_sec,
                         "is_seen_with_tracked": False}
            track["last_center"] = center
            track["last_sec"] = now_sec
            if is_tracked_seen:
                track["is_seen_with_tracked"] = True
            matched.append(track)
        # 이번 프레임에 못 본 궤적도 유예 안이면 살려 둔다 — 가림·중복 억제로
        # 잠깐 사라진 손이 표시를 잃고 "새 손"이 되는 구멍을 막는다
        for entry in self._other_tracks:
            if entry not in matched and now_sec - entry["last_sec"] <= OTHER_TRACK_KEEP_SEC:
                matched.append(entry)
        self._other_tracks = matched

    def _acquire_tracked_hand(self, candidates, now_sec):
        """획득 — 모양이 보이는 손이 실제로 움직이면 그 손을 추적한다. 성공 시 True.

        구 지시 손 고정 v2(2026-07-30)의 획득 규칙을 라벨 없이 계승: 쉬는 손·
        떠 있는 손은 영원히 안 잡히고, 들어 올리거나 획을 시작하는 손이 잡힌다.
        후보는 라벨이 없으므로 프레임 간 최근접(연속 반경)으로 궤적을 잇는다.
        여럿이 동시에 기준을 넘으면 이동량 최대가 이긴다.
        """
        matched_tracks = []
        acquired = None
        for hand, center in candidates:
            span_px = stable_span_px(hand)
            track = None
            best_dist_px = None
            for entry in self._acquire_tracks:
                if now_sec - entry["last_sec"] > ACQUIRE_GAP_SEC:
                    continue
                dist_px = math.dist(center, entry["last_center"])
                if dist_px <= CONTINUITY_SPAN_RATIO * span_px and (
                        best_dist_px is None or dist_px < best_dist_px):
                    track, best_dist_px = entry, dist_px
            if track is None:
                track = {"start_center": center, "start_sec": now_sec,
                         "last_center": center, "last_sec": now_sec,
                         "shape_sec": None}
            if track in matched_tracks:
                continue   # 한 궤적에 후보 둘 — 먼저 이은 쪽 유지
            track["last_center"] = center
            track["last_sec"] = now_sec
            if now_sec - track["start_sec"] > self._acquire_window_sec:
                # 창 초과 — 원점 재장전: 오래 쉰 손도 방금 움직임만으로 판정
                track["start_center"] = center
                track["start_sec"] = now_sec
            shape = self.classify_hand(hand)
            if shape is not None:
                track["shape_sec"] = now_sec
            matched_tracks.append(track)
            shoulder_px = hand_shoulder_px(hand)
            if shoulder_px is None or track["shape_sec"] is None or (
                    now_sec - track["shape_sec"] > self._acquire_window_sec):
                continue
            travel_px = math.dist(track["last_center"], track["start_center"])
            if travel_px < self._acquire_move_shoulder * shoulder_px:
                continue
            # 여럿이 동시에 기준을 넘으면 **위에 있는 손**이 이긴다 (화면 y가 작은 쪽).
            # ★2026-08-04 사용자 제안("배 위로 있는 손이 잡히도록"): 배 앞에 둔 손이
            # 먼저 잡혀 제스처가 통째로 죽는 일이 반복됐다. 높이로 **자르는** 것은
            # 실측이 반증했다 — 낮은 제스처(어깨 아래 0.55~0.82)와 배 앞 손이
            # 겹쳐서, 자르면 팔을 높이 못 드는 사용자가 못 쓴다(배리어프리 위배).
            # 그래서 자르지 않고 **우선순위**로만 쓴다: 손이 하나뿐이면 높이와
            # 무관하게 잡히고(낮은 제스처 보존), 둘이 경쟁할 때만 위쪽이 이긴다.
            if acquired is None or center[1] < acquired[1][1]:
                acquired = (hand, center, travel_px)
        self._acquire_tracks = matched_tracks   # 미매칭 궤적 폐기 — 소실 손 잔상 제거
        if acquired is None:
            return False
        is_takeover = self._tracked_hand is not None
        self._tracked_hand = acquired[0]
        self._tracked_center = acquired[1]
        self._tracked_sec = now_sec
        self._acquire_tracks = []
        self._update_display_box(acquired[0])
        logger.info("사용자 손 %s: 이동 %.0fpx (라벨 %s — 정보용)",
                    "인계" if is_takeover else "획득",
                    acquired[2], acquired[0].user_side)
        return True

    def is_engaged(self):
        """사용 중인가 — 손이 최근(release_sec 안) 보였는가. 유휴 전환 판단용."""
        return (self._last_any_hand_sec is not None
                and self._clock() - self._last_any_hand_sec <= self._release_sec)

    def _update_display_box(self, hand):
        """추적 손 주변 박스 — 디버그 창 표시용 (판정 미사용)."""
        xs = hand.landmarks[:, 0]
        ys = hand.landmarks[:, 1]
        pad_px = 0.3 * hand_span_px(hand.landmarks)
        new_box = (int(xs.min() - pad_px), int(ys.min() - pad_px),
                   int(xs.max() + pad_px), int(ys.max() + pad_px))
        self.locked_box = smooth_box(self.locked_box, new_box)

    # ----- 판정 신호 (gesture_filter 입력) -----

    def user_hand_signal(self):
        """사용자 손 신호 — (손모양, (x_px, y_px), 라벨, 검지비율) | None(미관측).

        라벨은 handedness(정보용 — 이벤트 hand_side로만 전달)다. 정체성은
        연속성 추적이 보장하므로 판정은 라벨을 쓰지 않는다 (모듈 독스트링).
        검지비율(2026-08-03 추가): 검지 손끝-뿌리 3D 거리 / PIP-뿌리 거리 —
        탭 클릭이 이 값의 **일시적 하강**으로 까딱을 읽는다. 모양 판별이
        "주먹"에 도달하지 않는 작은 까딱까지 잡기 위한 별도 채널이다
        (gesture_filter._update_tap_click). 판별 불가면 None.
        """
        if self._tracked_hand is None:
            return None
        return self._signal_for(self._tracked_hand, self._tracked_center)

    def _signal_for(self, hand, center_px):
        """손 하나의 판정 신호 — user_hand_signal과 candidate_signals의 공통 계산."""
        states = finger_states(hand.world_landmarks, self._hand_extend_ratio,
                              self._hand_curl_confirm_ratio)
        shape = classify_hand_shape(hand.world_landmarks, self._hand_extend_ratio,
                                    self._hand_min_valid_fingers,
                                    self._hand_curl_confirm_ratio)
        index_ratio = float(states[0][0]) if states else None   # HAND_FINGERS[0] = 검지
        if logger.isEnabledFor(logging.DEBUG):
            # 판별 계측(hand_measure) — 실측 튜닝 세션용
            logger.debug("hand_measure shape=%s conf=%.2f f=%s", shape, hand.conf,
                         "|".join(f"{ratio:.2f}:{state}" for ratio, state in states))
        return (shape, center_px, hand.user_side, index_ratio)

    def candidate_hands(self):
        """게이트를 통과한 손들의 (중심, 손, 자, 손폭) 목록 — 두 손 병렬 판정용.

        ★2026-08-04 신설(사용자 결정 — "일단 손을 잡아야지 판정손이 로직이 느린데,
        두 개 다 보고 제스처하는 게 입력되면 나머지 무시"). 종전 획득 규칙은
        **먼저 움직여야** 판정 손이 됐다(move_dist_shoulder 0.25) — 그 이동이
        판정 시작 전에 통째로 버려지는 시간이라 첫 제스처가 늦거나 씹혔다.
        이제 두 손을 나란히 판정하고 **먼저 제스처를 낸 손**이 이긴다(hand_arbiter).

        ※단일 손 추적을 유지할 근거였던 "배구 토스"는 한 손 로직에서도 그대로
        났다(2026-08-04 사용자 보고) — 한 손만 보는 것이 그 증상을 막지 못한다는
        뜻이라, 유지할 이유가 사라졌다.

        자(尺)는 **손마다 따로** 낸다 — 전체 최대(hand_scale_ratio)를 쓰면 가까운
        쪽 손의 자로 먼 손을 재게 돼 임계가 어긋난다.

        ★신호(손모양)는 여기서 **계산하지 않는다**(2026-08-04 실기 — 추론 FPS 저하).
        종전 candidate_signals는 게이트 통과 손 전부에 finger_states +
        classify_hand_shape를 매 프레임 돌렸다. 판정 손 하나만 계산하던 것이
        max_num_hands(4)까지 늘어나 그만큼 느려졌다. 중재기가 **실제로 판정할
        슬롯**(확정 시 1개·경쟁 시 최대 2개)에 대해서만 signal_for를 부른다.
        """
        candidates = []
        for hand in self._hands:
            center_px = hand_center_point(hand.landmarks)
            if center_px is None:
                continue
            # ★네 번째 값은 중재기가 **슬롯 이음 반경**으로 쓴다
            # (hand_arbiter._sync_slots — dist <= match_ratio × span_px).
            # 따라서 손 모양에 흔들리지 않는 자여야 한다: 원본 화면 크기를
            # 넘기면 주먹을 쥐는 순간 반경이 줄어 확정 슬롯이 손을 놓치고,
            # 경쟁이 재시작돼 옆 사람이 이길 수 있다(2026-08-07 — 셀렉터
            # 쪽 정체성 반경과 완전히 같은 버그였다, stable_span_px 주석)
            candidates.append((center_px, hand, self._scale_ratio_for(hand),
                               stable_span_px(hand)))
        return candidates

    def signal_for(self, hand, center_px):
        """중재기가 판정할 손 하나의 신호를 그때 계산한다 — 지연 계산 진입점."""
        return self._signal_for(hand, center_px)

    def adopt_tracked_hand(self, center_px):
        """제스처로 이긴 손을 추적 손으로 삼는다 — 이후 앵커 고정·표시가 그 손을 따른다.

        획득(이동) 경로를 대신하는 진입점이다. 이미 같은 손을 추적 중이면 아무
        일도 하지 않는다(정체성 유지 — 이음 로직이 계속 잇는다).
        """
        best = None
        for hand in self._hands:
            hand_center = hand_center_point(hand.landmarks)
            if hand_center is None:
                continue
            dist_px = math.dist(hand_center, center_px)
            if best is None or dist_px < best[2]:
                best = (hand, hand_center, dist_px)
        if best is None:
            return False
        is_takeover = (self._tracked_center is not None
                       and math.dist(best[1], self._tracked_center)
                       > CONTINUITY_SPAN_RATIO * stable_span_px(best[0]))
        self._tracked_hand = best[0]
        self._tracked_center = best[1]
        self._tracked_sec = self._clock()
        self._acquire_tracks = []
        self._update_display_box(best[0])
        if is_takeover or self._tracked_moves is None:
            self._tracked_moves.clear()
            self._tracked_idle_since = None
            self._other_tracks = []
        return True

    def anchor_shoulder_frame(self):
        """앵커 어깨선 y(px)와 어깨너비(px) | None — 손 들기(select) 판정의 자.

        2026-08-06 신설 (hand_bend 판 — select = 펼친 손을 어깨선 위로 들기):
        앵커(_head_anchor)의 중심 y가 곧 어깨선, 폭이 어깨너비다(anchor_frame).
        어깨 기반이라 카메라 거리·설치 위치 무관. 앵커 부재 시 None — select 대기.
        """
        if self._head_anchor is None:
            return None
        return (self._head_anchor[1], self._head_anchor[2])

    def anchor_reach_px(self):
        """앵커 사람의 팔 도달 반경(픽셀) | None — ROI 줌 크롭 크기의 기준.

        어깨너비 기반이라 고개를 돌려도 안 흔들린다(2026-08-04 — 종전엔 머리 폭에서
        구해 곁눈질만 해도 크롭이 쪼그라들었다).
        """
        if self._head_anchor is None or self._head_reach_widths is None:
            return None
        return self._head_reach_widths * self._head_anchor[2]

    def anchor_center_width_px(self):
        """앵커 중심(x, y)과 어깨너비(px) | None — 2026-08-07 신설.

        인식 대상 외 블러(realtime_loop.blur_outside_region)의 "선명하게 남길
        영역" 기준. anchor_reach_px는 팔 도달(제스처) 반경이라 옆에 붙어 앉은
        사람까지 같이 덮기 쉽다 — 블러는 그보다 좁고 독립적으로 튜닝 가능한
        반경(keep_sharp_shoulder_widths)을 쓴다(realtime_loop 참고).
        """
        return self._head_anchor

    def anchor_segmentation_mask(self):
        """앵커 사람의 실루엣 마스크(numpy, 0~1) | None — 2026-08-07 신설.

        인식 대상 외 블러의 **정밀** 기준(사용자 결정 — "블러를 사람 누끼 딴거
        제외하고"). anchor_center_width_px의 사각형은 옆에 붙어 앉은 사람과
        분리가 안 되지만, 이 마스크는 사람 윤곽을 픽셀 단위로 따라간다.
        None이면(마스크 미관측·output_segmentation_masks 비활성) 호출부가
        사각형(anchor_center_width_px)으로 폴백해야 한다.

        ★2026-08-07 유지 시간 도입(사용자 보고 — "앵커 안 잡혀있는 뒤에 있는
        사람의 얼굴이랑 손이 보였다가 사라졌다가 반복"). 원인은 **블러가
        통째로 꺼지는 구간**이었다: 앵커가 풀리면 마스크가 None이 되고,
        폴백인 anchor_keep_sharp_box도 앵커가 없으니 None을 돌려주며,
        blur_outside_region(frame, None)은 **원본을 그대로 반환**한다 —
        즉 블러가 사라져 뒷사람까지 전부 선명해진다. 실측 로그에서 한 세션에
        앵커 해제 39회·획득 62회였으니 그만큼 깜빡였다.
        앵커가 잠깐 끊기는 것은 사용자가 떠난 게 아니라 거의 항상 검출
        공백이다. 그 사이 **직전 마스크를 붙들어** 블러를 유지한다 —
        0.5초 지난 마스크로 배경을 흐리는 쪽이, 블러를 꺼서 대기줄을 통째로
        노출하는 것보다 명백히 낫다. 유지 시간이 지나면 그때 None이 되어
        낡은 마스크가 다음 사람에게 승계되지 않는다.
        """
        if self._anchor_segmentation_mask is None or self._anchor_mask_sec is None:
            return None
        if self._clock() - self._anchor_mask_sec > self._mask_hold_sec:
            return None
        return self._anchor_segmentation_mask

    def anchor_arm_points(self):
        """앵커의 팔 관절 좌표 목록 [(x, y), ...] — 블러를 팔 주변으로 좁히는 기준.

        2026-08-07 신설(옆 사람 차단 — preprocessor.arm_reach_mask).
        어깨는 항상 쓰고, 팔꿈치·손목은 **믿을 수 있을 때만** 더한다
        (reliable_wrists와 같은 관문 — 가시성이 낮거나 팔 길이를 넘는 추정
        좌표를 쓰면 엉뚱한 곳이 선명해지고 정작 내 손이 흐려진다).
        빈 목록이면 호출부가 이 제약을 걸지 않는다(인식 우선 폴백).
        """
        points = [wrist[:2] for wrist in self._usable_wrists()]
        if not points:
            return []   # 손목을 못 믿으면 팔 위치를 모른다 — 제약을 걸지 않는다
        points.extend(shoulder[:2] for shoulder in self._anchor_shoulders)
        points.extend(
            elbow[:2] for elbow in self._anchor_elbows
            if len(elbow) <= 2 or elbow[2] >= WRIST_MIN_VISIBILITY)
        if self._head_anchor is not None:
            points.append((self._head_anchor[0], self._head_anchor[1]))
        return points

    def other_segmentation_mask(self):
        """앵커가 **아닌** 사람들의 실루엣 합집합(numpy, 0~1) | None.

        2026-08-07 신설 — 블러가 앵커 마스크에서 이 영역을 빼내 "앵커 안 잡힌
        사람은 반드시 흐리게"를 보장한다(_collect_other_masks 주석).
        앵커 마스크와 같은 유지 시간을 따른다 — 둘은 같은 관측에서 나오므로
        수명이 갈리면 앵커만 낡거나 상대만 낡는 어긋남이 생긴다.
        """
        if self._other_segmentation_mask is None or self._anchor_mask_sec is None:
            return None
        if self._clock() - self._anchor_mask_sec > self._mask_hold_sec:
            return None
        return self._other_segmentation_mask

    def body_lines(self):
        """앵커 사람의 골격 선 목록 [(점A, 점B), ...] — 디버그 창 표시 전용.

        2026-08-04 사용자 요청("추적 박스 말고 선으로")로 신설. ★2026-08-06
        팔 선(어깨→팔꿈치→손목) **삭제**(사용자 결정 — "팔목 손목 추적점
        없애라"): 포즈 팔 좌표는 판정 어디에도 안 쓰이므로 화면에도 안 그린다 —
        팔 선의 요동·끌려감이 판정 흔들림처럼 보이던 착시 소멸. 남는 표시 =
        목·어깨선(판정에 실제로 쓰는 관측)뿐이다.
        """
        lines = []
        if len(self._anchor_shoulders) != 2:
            return lines
        left_shoulder = tuple(self._anchor_shoulders[0][:2])
        right_shoulder = tuple(self._anchor_shoulders[1][:2])
        lines.append((left_shoulder, right_shoulder))
        if self._head_anchor is not None:
            neck = ((left_shoulder[0] + right_shoulder[0]) / 2.0,
                    (left_shoulder[1] + right_shoulder[1]) / 2.0)
            lines.append(((self._head_anchor[0], self._head_anchor[1]), neck))
        return lines

    def tracked_hand_landmarks(self):
        """추적 손의 21점 화면 좌표 | None — 디버그 창의 손 골격 표시 전용."""
        if self._tracked_hand is None:
            return None
        return self._tracked_hand.landmarks

    def tracked_hand(self):
        """추적 손 HandDetection | None — 로터(rotor.py)의 입력 손 1개 제한용.

        2026-08-05 신설: 로터가 화면 좌표(검지 방향)와 월드 좌표(검지 폄 게이트 —
        주먹이 다이얼을 돌리지 않게)를 함께 써야 해서 손 객체째 넘긴다.
        """
        return self._tracked_hand

    # ※anchor_arm_near(전완 방향 자 — 포즈 팔꿈치→손목)는 2026-08-06 **최종
    # 삭제**(사용자 결정 — "팔목 손목 추적점 없애라"): 같은 날 팔 고정·팔 길이
    # 관문까지 3중 방어를 시도했으나, 포즈 팔 관측 자체가 붕괴·끌려감(겹침 시
    # 든 팔로 손목이 끌려옴)의 원천이라 판정에서 완전히 뺐다. 꺾임 기준은
    # 화면 수직 고정(hand_bend.py) — 포즈에서 판정에 쓰는 것은 어깨뿐이다.

    def candidate_points(self):
        """게이트 통과 후보 손 중심 목록 — 시각화용(획득 전 상황 확인)."""
        points = []
        for hand in self._hands:
            center = hand_center_point(hand.landmarks)
            if center is not None:
                points.append(center)
        return points

    def classify_hand(self, hand):
        """HandDetection 1건의 모양 판별 — 획득 판정·보조 카메라 표(B안)용."""
        return classify_hand_shape(hand.world_landmarks, self._hand_extend_ratio,
                                   self._hand_min_valid_fingers,
                                   self._hand_curl_confirm_ratio)

    # ----- 거리 자(尺) -----

    def hand_scale_ratio(self):
        """가상 어깨너비 비율 — 기존 임계 체계(어깨너비 배수)를 유지하는 손 실측 자.

        월드 랜드마크(미터)와 화면 랜드마크(px)의 폭 비 = 이 거리의 px/m.
        여기에 표준 어깨너비 0.4m를 곱해 프레임 폭으로 나누면 "이 사용자의 가상
        어깨너비/프레임폭"이 된다. 손이 없으면 None (gesture_filter가 마지막
        값·폴백 사용 — 종전 동작).
        """
        # 가장 가까운 손을 고르는 자리다 — 모양 무관 자를 써야 한다. 원본
        # 화면 크기로 고르면 **가까운 주먹보다 먼 펼친 손이 더 커 보여** 엉뚱한
        # 손의 자를 쓰게 된다(2026-08-07 — stable_span_px 주석)
        best_span_px, best_hand = 0.0, None
        for hand in self._hands:
            span_px = stable_span_px(hand)
            if span_px > best_span_px:
                best_span_px = span_px
                best_hand = hand
        return None if best_hand is None else self._scale_ratio_for(best_hand)

    def _scale_ratio_for(self, hand):
        """이 손 하나의 가상 어깨너비 비율 | None — 두 손을 각자의 자로 재기 위해.

        손마다 따로 내야 한다(2026-08-04 두 손 판정): 전체 최대를 공유하면 가까운
        쪽 손의 자로 먼 손을 재게 돼 쓸기 임계가 어긋난다.
        """
        span_px = hand_span_px(hand.landmarks)
        span_m = hand_span_world_m(hand.world_landmarks)
        if span_px <= 0.0 or span_m <= 0.0:
            return None
        return (span_px / span_m * STANDARD_SHOULDER_M) / self._frame_width_px

    def shoulder_line_y_ratio(self):
        """어깨선 높이(프레임 폭 정규화) | None — **실측 어깨 y**를 그대로 쓴다.

        ★2026-08-04 추정 제거: 종전엔 머리에서 비례로 추정했다(귀 중점 y +
        귀-귀 폭 × shoulder_below_head_widths). 포즈가 어깨를 주므로 추정할 게
        없다 — config 항목 하나와 그 오차가 함께 사라졌다. 어깨 관측이 없는
        프레임에서만 종전과 같은 비례로 되돌린다(SHOULDER_BELOW_EAR_RATIO).
        앵커 부재 시 None — gesture_filter가 화면 하단 띠로 폴백한다(종전).
        """
        if self._head_anchor is None:
            return None
        if len(self._anchor_shoulders) == 2:
            shoulder_y = (self._anchor_shoulders[0][1] + self._anchor_shoulders[1][1]) / 2.0
            return shoulder_y / self._frame_width_px
        _, anchor_y, anchor_width = self._head_anchor
        return (anchor_y + SHOULDER_BELOW_EAR_RATIO * anchor_width) / self._frame_width_px
