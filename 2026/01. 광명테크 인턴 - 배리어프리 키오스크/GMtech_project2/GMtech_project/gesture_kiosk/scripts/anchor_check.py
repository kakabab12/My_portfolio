"""앵커 이음 실측 도구 — 겹침에서 앵커가 뒷사람에게 넘어가는 문제용 (2026-08-04 신설).

★왜 필요한가(2026-08-04 사용자 보고: "뒷사람하고 나랑 겹치면 생김"): 앵커 이음이
**위치만** 보고 같은 사람을 판정한다(hand_select._update_head_anchor). 겹치면 위치가
안 갈리고, 가림으로 내가 관측에서 빠지는 프레임에는 뒷사람만 이음 반경 안에 남아
앵커가 통째로 넘어간 뒤 영구 래치된다(재현: 4프레임 만에 이전, 내가 다시 잡혀도
안 돌아옴). 겹쳐도 갈리는 신호는 **크기(=거리)** 뿐이라 이음에 크기 일관성 관문을
넣으려는데, 임계 하나를 실측으로 정해야 한다:
  · 너무 빡빡 → 카메라 쪽으로 걸어올 때 앵커가 나를 놓친다
  · 너무 헐거움 → 겹침 버그가 남는다
그래서 **정상 사용의 변화율 상한**과 **겹칠 때 갈라야 할 차이**를 같이 잰다.
둘 사이가 벌어져 있어야 임계가 성립한다 — 안 벌어지면 크기로는 못 가른다는 뜻이고,
그때는 다른 신호를 찾아야 한다(보고할 것).

곁들여 재는 것 — 앵커 **중심**을 어깨로 내릴지 판단할 근거(2026-08-04 보류 중):
고개를 돌리면 귀 중점이 얼마나 흔들리는지, 어깨 중점으로 내리면 위쪽 도달을
얼마나 잃는지. 폭은 이미 어깨로 옮겼지만 중심은 머리로 남아 있다.

사용법 (엔진 폴더에서, **쓰는 자리에서 서서** 실행):
    py scripts\\anchor_check.py
    py scripts\\anchor_check.py --device 1

조작: 자세를 잡고 **SPACE** → 3초 뒤 계측 시작 · s 건너뛰기 · q/ESC 중단
마지막에 요약이 로그로 나온다 — 그 숫자를 보고 임계를 정한다.
"""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.capture.camera_stream import CameraStream
from src.inference.head_detector import HeadDetector
from src.inference.preprocessor import Preprocessor
from src.postprocess.hand_select import ANCHOR_CONTINUITY_RATIO
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

WINDOW = "gesture_kiosk anchor check"
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")
SOLO_SEC, PAIR_SEC = 10.0, 12.0
READY_SEC = 3.0               # SPACE를 누르고 자세를 잡을 시간 (calibrate.py와 같은 규약)
SHOULDER_MIN_VISIBILITY = 0.5  # 어깨 관측 최소 신뢰도 — hand_select.WRIST_MIN_VISIBILITY와
                              #   같은 값·같은 이유(가려져도 BlazePose는 추측 좌표를 낸다).
                              #   ★2026-08-04 1차 실측이 이것 때문에 못 쓰게 나왔다: 관문
                              #   없이 재니 "0.1초에 어깨폭 13배 변화"가 찍혔다 — 사람이
                              #   그렇게 못 움직이니 가려진 어깨의 추측 좌표다.
                              #   엔진 anchor_frame()도 이 관문이 없어, **켠 값과 끈 값을
                              #   둘 다** 재서 엔진에도 넣을지 판단한다
FONT_PATHS = (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\gulim.ttc")

logger = get_logger("scripts")


def _font(size):
    """한글 폰트 — cv2.putText는 한글을 못 그려 지시가 깨진다(2026-08-03 실측)."""
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE, FONT_BODY, FONT_SMALL = _font(34), _font(22), _font(18)


def draw(frame, title, detail, status, sub=None):
    view = frame.copy()
    cv2.rectangle(view, (0, 0), (view.shape[1], 122), (0, 0, 0), -1)
    image = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
    pen = ImageDraw.Draw(image)
    pen.text((16, 8), title, font=FONT_TITLE, fill=(255, 220, 0))
    pen.text((16, 52), detail, font=FONT_BODY, fill=(255, 255, 255))
    pen.text((16, 84), status, font=FONT_SMALL, fill=(120, 255, 120))
    if sub:
        pen.text((520, 84), sub, font=FONT_SMALL, fill=(190, 190, 190))
    cv2.imshow(WINDOW, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    return cv2.waitKey(1) & 0xFF


def shoulder_span_px(head, min_visibility=None):
    """어깨너비(px) | None — 앵커 자(尺). 어깨 관측이 없으면 잴 수 없다.

    min_visibility를 주면 그보다 낮은 가시성의 어깨는 **못 쟀다**고 본다(엔진의
    reliable_wrists와 같은 규칙). 엔진 anchor_frame()은 이 관문이 없어서, 관문을
    켠 값과 끈 값을 나란히 재려고 선택형으로 둔다.
    """
    if len(head.shoulders) != 2:
        return None
    left, right = head.shoulders[0], head.shoulders[1]
    if min_visibility is not None and (len(left) < 3 or len(right) < 3
                                       or min(left[2], right[2]) < min_visibility):
        return None
    span_px = ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5
    return span_px if span_px > 0.0 else None


def shoulder_visibility(head):
    """양 어깨 가시성 중 **낮은 쪽** | None — 관문이 실제로 걸리는지 보려고 같이 남긴다."""
    if len(head.shoulders) != 2:
        return None
    left, right = head.shoulders[0], head.shoulders[1]
    if len(left) < 3 or len(right) < 3:
        return None
    return min(left[2], right[2])


def shoulder_mid_xy(head):
    """어깨 중점 (x_px, y_px) | None — 중심을 어깨로 내렸을 때의 후보 기준점."""
    if len(head.shoulders) != 2:
        return None
    left, right = head.shoulders[0], head.shoulders[1]
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)


def span_split(spans_px):
    """관측 어깨폭이 두 무리로 갈라지는지 — (작은무리 중앙, 큰무리 중앙, 비) | None.

    ★2026-08-04 2차: **모든 단계**에 쓴다(사용자 정정 — 1차는 전 단계에 뒷사람이
    같이 있었다). 포즈가 겹친 둘을 동시에 못 잡으면(1차 207관측 중 0회) "2명 관측"
    카운터로는 오염을 못 잡는다 — 매 프레임 1명씩만 보이니까. 대신 관측 어깨폭이
    두 무리로 갈라지면 모델이 두 사람을 **오간** 것이고, 그게 곧 앵커가 넘어가는
    순간이다. 동시 관측 없이 사람 바뀜을 잡아내는 유일한 신호다.

    전체 중앙값을 경계로 아래/위를 갈라 각 중앙을 본다.
    """
    if len(spans_px) < 10:
        return None      # 표본 부족 — 이때만 판정 불가다
    mid_px = statistics.median(spans_px)
    low = [s for s in spans_px if s < mid_px]
    high = [s for s in spans_px if s >= mid_px]
    # 한쪽이 비는 것은 데이터 부족이 아니라 **안 갈렸다**는 뜻이다(값이 다 같음).
    # 종전엔 None을 돌려 "판정 불가"로 새어 나갔다 — 한 사람만 계속 잡힌 정상
    # 상황이 바로 이 경우라, 제일 흔한 결과가 오답이 됐다
    low_px = statistics.median(low) if low else mid_px
    high_px = statistics.median(high) if high else mid_px
    return (low_px, high_px, low_px / high_px)


def _step(bucket, prev_px, span_px):
    """직전 관측 대비 변화율을 bucket에 담고 새 직전값을 돌려준다.

    못 잰 관측(None)은 **잇지 않는다** — 관문에 걸린 프레임을 건너뛰고 이으면
    간격이 늘어난 만큼 변화율이 커져, 관문이 오히려 노이즈를 키운 것처럼 보인다.
    """
    if span_px is None:
        return None
    if prev_px is not None and prev_px > 0.0:
        bucket.append(abs(span_px - prev_px) / prev_px)
    return span_px


def nearest_head(heads):
    """가장 가까운 사람(어깨가 제일 넓은 관측) | None — 1인 단계의 측정 대상."""
    scored = [(head, shoulder_span_px(head)) for head in heads]
    scored = [(head, span_px) for head, span_px in scored if span_px is not None]
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[1])[0]


def summarize(values, unit=""):
    """중앙값·95%·최대 한 줄 — 이상치 하나에 임계가 끌려가지 않게 셋을 같이 본다."""
    if not values:
        return "관측 없음"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return (f"중앙 {statistics.median(ordered):.3f}{unit} · "
            f"95% {p95:.3f}{unit} · 최대 {ordered[-1]:.3f}{unit} · n={len(ordered)}")


def run_phase(camera, preprocessor, detector, title, detail, duration_sec, collect,
              gap_sec):
    """SPACE로 시작 → READY_SEC 준비 → duration_sec 계측. 중단이면 False.

    ★추론은 **운영 주기(gap_sec = 1/infer_fps)로만** 돌린다(2026-08-04 1차 실측 후).
    종전엔 카메라 FPS로 재고 나중에 환산했는데, 그러면 한 프레임짜리 관측 튐이
    환산 배수만큼 부풀어 진짜 움직임과 못 가른다. 운영과 같은 간격으로 관측해야
    나온 숫자를 그대로 임계에 쓸 수 있다. 화면은 카메라 FPS로 계속 갱신하고 마지막
    관측을 겹쳐 그린다.

    관측마다 collect(heads, now_sec). 사람이 안 잡힌 관측도 그대로 넘긴다 —
    **관측이 빠지는 것 자체가 이 버그의 방아쇠**라 빈 관측을 세야 한다.
    시작 게이트는 calibrate.py와 같은 이유로 사람이 누른 뒤에만 연다(자동 시작이면
    앞 단계 자세가 다음 단계에 조용히 섞인다 — 2026-08-03 사고).
    """
    state, start_sec = "wait", None
    heads, last_infer_sec = [], None
    while True:
        frame = preprocessor.preprocess_frame(camera.capture_frame())
        now_sec = time.monotonic()
        is_due = last_infer_sec is None or (now_sec - last_infer_sec) >= gap_sec
        if is_due:
            heads = detector.infer(frame)
            last_infer_sec = now_sec
        _draw_heads(frame, heads)
        seen = f"{len(heads)}명 관측" if heads else "사람 안 잡힘"
        if state == "wait":
            key = draw(frame, title, detail, "자세를 잡고 SPACE를 누르세요",
                       f"{seen} · s 건너뛰기 · q 중단")
            if key == 32:
                state, start_sec = "ready", time.monotonic()
        elif state == "ready":
            remain_sec = READY_SEC - (time.monotonic() - start_sec)
            key = draw(frame, title, detail, f"{max(0.0, remain_sec):.0f} 후 시작", seen)
            if remain_sec <= 0:
                state, start_sec = "measure", time.monotonic()
        else:
            remain_sec = duration_sec - (now_sec - start_sec)
            if is_due:
                collect(heads, now_sec)   # 운영 주기로 들어온 관측만 담는다
            key = draw(frame, title, detail, f"남은 시간 {max(0.0, remain_sec):4.1f}초", seen)
            if remain_sec <= 0:
                return True
        if key in (ord("q"), 27):
            return False
        if key == ord("s") and state == "wait":
            return True


def _draw_heads(frame, heads):
    """관측된 사람마다 귀 중점·어깨선·어깨너비 — 도구가 뭘 보고 있는지 눈으로 확인."""
    for head in heads:
        span_px = shoulder_span_px(head)
        center = (int(head.center_x_px), int(head.center_y_px))
        cv2.circle(frame, center, 8, (80, 220, 255), 2)
        if span_px is None:
            continue
        left, right = head.shoulders[0], head.shoulders[1]
        cv2.line(frame, (int(left[0]), int(left[1])), (int(right[0]), int(right[1])),
                 (120, 255, 120), 2)
        cv2.putText(frame, f"{span_px:.0f}px", (center[0] - 30, center[1] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 2)


class SoloRecorder:
    """1인 단계 수집 — 앵커 주기당 어깨폭 변화율과 귀 중점 흔들림.

    ★2026-08-04 1차 실측 후 재작성: 종전엔 카메라 FPS로 재고 운영 주기(0.1초)로
    **환산**했는데, 그러면 한 프레임짜리 관측 튐까지 같이 부풀어(2배) 진짜 움직임과
    분리가 안 됐다("0.1초에 13배 변화"가 그래서 찍혔다). 이제 run_phase가 관측
    자체를 운영 주기로 던져 주므로 환산 없이 **잰 그대로** 담는다.
    """

    def __init__(self):
        self._prev_px = None          # 직전 관측 어깨폭 (관문 끈 값)
        self._prev_gated_px = None    # 직전 관측 어깨폭 (관문 켠 값)
        self.step_ratios = []         # 앵커 1주기당 어깨폭 변화율 — 관문 끔(현 엔진)
        self.gated_step_ratios = []   # 같은 값 — 관문 켬(엔진에 넣을지 판단용)
        self.visibilities = []        # 어깨 가시성(낮은 쪽) — 관문이 걸리는지 확인
        self.ear_offsets = []         # (귀중점 - 어깨중점) — 어깨 기준 상대 위치
        self.ear_to_shoulder_px = []
        self.observed_spans = []      # 관측 어깨폭 — 사람 바뀜(갈림) 확인용
        self.miss_count = 0           # 사람이 안 잡힌 관측 — 겹침 방아쇠의 빈도
        self.crowd_count = 0          # 2명 이상 잡힌 관측 — 1인 단계 오염 확인.
                                      #   ★이것만으로는 오염을 못 잡는다: 포즈가 겹친
                                      #   둘을 동시에 못 잡으면 2명이 있어도 0이다.
                                      #   그래서 span_split()을 같이 본다

    def __call__(self, heads, now_sec):
        head = nearest_head(heads)
        if head is None:
            self.miss_count += 1
            return
        if len(heads) > 1:
            self.crowd_count += 1
        visibility = shoulder_visibility(head)
        if visibility is not None:
            self.visibilities.append(visibility)
        span_px = shoulder_span_px(head)
        if span_px is not None:
            self.observed_spans.append(span_px)
        self._prev_px = _step(self.step_ratios, self._prev_px, span_px)
        self._prev_gated_px = _step(self.gated_step_ratios, self._prev_gated_px,
                                    shoulder_span_px(head, SHOULDER_MIN_VISIBILITY))
        if span_px is not None:
            self._collect_center(head, span_px)

    def _collect_center(self, head, span_px):
        mid = shoulder_mid_xy(head)
        if mid is None:
            return
        offset_x = (head.center_x_px - mid[0]) / span_px   # 어깨너비 정규화 —
        offset_y = (head.center_y_px - mid[1]) / span_px   #   카메라 거리 무관 비교
        self.ear_offsets.append((offset_x, offset_y))
        self.ear_to_shoulder_px.append(abs(head.center_y_px - mid[1]))

    def span_split(self):
        """이 단계가 사람 바뀜에 오염됐는지 — 1인 단계에서 갈리면 기준선으로 못 쓴다."""
        return span_split(self.observed_spans)

    def drift_ratios(self):
        """귀 중점 흔들림 — 어깨 중점 기준 상대 위치의 중앙값에서 벗어난 거리.

        어깨 중점은 고개 돌림에 안 흔들리므로(2026-08-04 실측: 변동 8.5%) 기준으로
        쓸 수 있다. 이 값이 크면 **중심이 머리라서** 앵커가 고개 따라 움직인다는 뜻.
        """
        if not self.ear_offsets:
            return []
        mid_x = statistics.median(x for x, _ in self.ear_offsets)
        mid_y = statistics.median(y for _, y in self.ear_offsets)
        return [((x - mid_x) ** 2 + (y - mid_y) ** 2) ** 0.5
                for x, y in self.ear_offsets]


class PairRecorder:
    """겹침 단계 수집 — 크기 차이와 머리 간 거리.

    ★2026-08-04 1차 실측 후 보강: 겹치면 포즈가 **둘을 동시에 못 잡는** 일이
    잦았다(207관측 내내 0회). 그러면 "두 사람 비교"가 통째로 빈다 — 그래서 한 명만
    잡힐 때도 **관측된 어깨폭 시계열**을 남긴다. 그 값이 두 무리로 갈라지면
    (예: 316 ↔ 222) 모델이 두 사람 사이를 오간 것이고, 그게 곧 앵커가 넘어가는
    순간이다. 동시 관측이 없어도 크기 차이를 잴 수 있는 유일한 길이다.
    """

    def __init__(self):
        self.span_ratios = []      # 작은쪽 / 큰쪽 — 1.0에 가까울수록 크기로 못 가른다
        self.gap_over_radius = []  # 머리 간 거리 ÷ 이음 반경 — 1 미만이면 "같은 사람"
        self.observed_spans = []   # 한 명만 잡혀도 남기는 어깨폭 — 갈라지는지 본다
        self.single_count = 0      # 겹쳤는데 한 명만 잡힌 관측 = **버그 방아쇠**
        self.pair_count = 0
        self.miss_count = 0

    def __call__(self, heads, now_sec):
        spans = [(head, shoulder_span_px(head)) for head in heads]
        spans = [(head, span_px) for head, span_px in spans if span_px is not None]
        if not spans:
            self.miss_count += 1
            return
        spans.sort(key=lambda pair: pair[1], reverse=True)
        self.observed_spans.append(spans[0][1])
        if len(spans) < 2:
            self.single_count += 1
            return
        (near, near_px), (far, far_px) = spans[0], spans[1]
        self.pair_count += 1
        self.span_ratios.append(far_px / near_px)
        gap_px = ((near.center_x_px - far.center_x_px) ** 2
                  + (near.center_y_px - far.center_y_px) ** 2) ** 0.5
        radius_px = ANCHOR_CONTINUITY_RATIO * max(near_px, far_px)
        if radius_px > 0.0:
            self.gap_over_radius.append(gap_px / radius_px)

    def span_split(self):
        return span_split(self.observed_spans)


def _log_split(prefix, split):
    """어깨폭 갈림 한 줄 — 1인 단계에서 갈리면 그 단계 값은 **오염된 것**이다."""
    if split is None:
        logger.warning("%s표본 부족 — 갈림 판정 불가", prefix)
        return
    low_px, high_px, ratio = split
    mark = "★갈림" if ratio < 0.85 else "안 갈림"
    logger.warning("%s%s (작은무리 %.0fpx · 큰무리 %.0fpx · 비 %.3f)",
                   prefix, mark, low_px, high_px, ratio)


def report(solo_phases, pair):
    """실측 요약 — 이 숫자로 크기 일관성 임계를 정한다."""
    logger.warning("=" * 78)
    logger.warning("앵커 이음 실측 요약 (2026-08-04) — 크기 일관성 임계 근거")
    logger.warning("=" * 78)
    logger.warning("[A] 정상 사용의 앵커 1주기당 어깨폭 변화율 — 임계는 이보다 **커야** 한다")
    logger.warning("    (운영 주기로 관측한 값 그대로 · 환산 없음)")
    for label, rec in solo_phases:
        logger.warning("  %-12s 관문 끔(현 엔진) %s", label, summarize(rec.step_ratios))
        logger.warning("  %-12s 관문 켬(가시성)  %s", "", summarize(rec.gated_step_ratios))
        logger.warning("  %-12s 어깨 가시성      %s · 안 잡힘 %d · 2명 이상 %d",
                       "", summarize(rec.visibilities), rec.miss_count, rec.crowd_count)
        _log_split("  %-12s 사람 바뀜 확인   " % "", rec.span_split())
    logger.warning("")
    logger.warning("[B] 겹침 — 크기로 갈라야 할 차이. 임계는 이보다 **작아야** 한다")
    logger.warning("  두 사람 어깨폭 비(작은쪽/큰쪽)  %s", summarize(pair.span_ratios))
    logger.warning("  → 크기 차이(1 - 비)             %s",
                   summarize([1.0 - r for r in pair.span_ratios]))
    logger.warning("  머리 간 거리 ÷ 이음 반경        %s", summarize(pair.gap_over_radius))
    logger.warning("  둘 다 잡힘 %d · **한 명만 잡힘 %d**(버그 방아쇠) · 아무도 안 잡힘 %d",
                   pair.pair_count, pair.single_count, pair.miss_count)
    _log_split("  관측 어깨폭 시계열  ", pair.span_split())
    logger.warning("  → 동시 관측이 없어도 이 비가 갈리면 모델이 두 사람을 오간 것이다")
    logger.warning("")
    logger.warning("[C] 앵커 중심을 어깨로 내릴지 — 보류 중인 판단의 근거")
    for label, rec in solo_phases:
        logger.warning("  %-14s 귀 중점 흔들림(어깨너비 배수) %s",
                       label, summarize(rec.drift_ratios()))
    for label, rec in solo_phases:
        logger.warning("  %-14s 귀↔어깨 중점 거리(px) %s",
                       label, summarize(rec.ear_to_shoulder_px))
    logger.warning("=" * 78)
    _verdict(solo_phases, pair)


def _verdict(solo_phases, pair):
    """A와 B가 벌어져 있는지 — 안 벌어지면 크기로는 못 가른다(다른 신호를 찾아야 한다).

    B는 동시 관측이 우선이지만, 겹치면 포즈가 둘을 동시에 못 잡는 일이 잦다
    (2026-08-04 실측 207관측 중 0회). 그때는 관측 어깨폭 시계열의 갈림을 대신 쓴다 —
    모델이 두 사람을 오간 것 자체가 갈라야 할 크기 차이다.
    """
    for label, key in (("관문 켬", "gated_step_ratios"), ("관문 끔", "step_ratios")):
        normal = [r for _, rec in solo_phases for r in getattr(rec, key)]
        gaps = [1.0 - r for r in pair.span_ratios]
        source = "동시 관측"
        if not gaps:
            split = pair.span_split()
            if split is not None:
                gaps, source = [1.0 - split[2]], "어깨폭 시계열 갈림"
        if not normal or not gaps:
            logger.warning("[%s] 판정 불가 — 단계를 건너뛰었거나 관측이 없다", label)
            continue
        ordered = sorted(normal)
        upper = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        lower = min(gaps)
        logger.warning("[%s] 정상 변화율 95%% = %.3f  vs  겹침 크기차(%s) = %.3f",
                       label, upper, source, lower)
        if lower > upper:
            logger.warning("  → 벌어져 있다. 임계를 이 사이(예: %.3f)에 두면 갈린다",
                           (upper + lower) / 2.0)
        else:
            logger.warning("  → 겹친다. 이 조합으로는 못 가른다 — 다른 신호가 필요하다")


def main():
    parser = argparse.ArgumentParser(description="앵커 이음 실측 — 겹침 크기 일관성 임계")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--device", type=int, default=None,
                        help="카메라 장치 번호 (기본: config device_id)")
    parser.add_argument("--min-conf", type=float, default=None,
                        help="포즈 검출 문턱 (기본: config min_detection_conf). "
                             "★2026-08-04: 겹친 뒷사람이 아예 안 잡히는지 확인용 — "
                             "0.3으로 낮춰 재보면 문턱 때문인지 모델 한계인지 갈린다")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)
    head_cfg = config.get("head_anchor") or {}
    if not head_cfg.get("model_path"):
        logger.error("config에 head_anchor가 없습니다 — 몸통판 브랜치에서 실행하세요")
        return 1
    if args.min_conf is not None:
        head_cfg = dict(head_cfg)
        head_cfg["min_detection_conf"] = args.min_conf
        config["head_anchor"] = head_cfg
        logger.warning("포즈 검출 문턱을 %.2f로 낮춰 실행합니다 (실측 전용)", args.min_conf)
    if args.device is not None:
        config["camera"]["device_id"] = args.device
        auto_select = config["camera"].get("auto_select")
        if auto_select is not None:
            auto_select["enabled"] = False   # 실측은 지정 장치 그대로

    gap_sec = 1.0 / float(head_cfg.get("infer_fps", 10))
    detector = HeadDetector(config)
    preprocessor = Preprocessor(config)
    camera = CameraStream(config).start()
    ctx = (camera, preprocessor, detector)

    solo_phases, pair = [], PairRecorder()
    try:
        # ★1~4는 **혼자** 서야 한다(2026-08-04 사용자 정정 — 1차는 전 단계에 뒷사람이
        # 같이 있었다). 정상 변화율 기준선을 재는 단계라 사람이 하나라도 더 있으면
        # 모델이 둘을 오간 값이 섞여 기준선이 무너진다 — 실제로 1차에서 "0.1초에
        # 어깨폭 13배"가 그렇게 찍혔다. 화면 지시에 인원을 명시한다
        for label, title, detail in (
            ("제자리 사용", "1) 제자리 사용 [혼자]", "화각에 나 혼자 — 평소처럼 손을 움직이세요"),
            ("걸어오기", "2) 카메라 쪽으로 [혼자]", "화각에 나 혼자 — 뒤에서 카메라 쪽으로 걸어오세요"),
            ("앉기·서기", "3) 앉기·서기 [혼자]", "화각에 나 혼자 — 앉았다 섰다를 반복하세요"),
            ("고개 돌리기", "4) 고개 돌리기 [혼자]", "화각에 나 혼자 — 몸은 두고 고개만 좌우로"),
        ):
            recorder = SoloRecorder()
            if not run_phase(*ctx, title, detail, SOLO_SEC, recorder, gap_sec):
                return 1
            solo_phases.append((label, recorder))
        # ★뒷사람이 **반드시** 있어야 한다(2026-08-04 1차 실측이 이 단계만 비었다):
        # 207관측 내내 2명이 동시에 안 잡혔다. 혼자 서 있었던 것인지 포즈가 겹친
        # 둘을 못 잡은 것인지 로그로는 못 갈라, 화면 지시에 인원을 명시한다
        if not run_phase(*ctx, "5) 뒷사람과 겹치기",
                         "뒷사람이 내 어깨 뒤로 겹쳐 서야 합니다 — 나는 평소처럼 조작",
                         PAIR_SEC, pair, gap_sec):
            return 1
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    report(solo_phases, pair)
    return 0


if __name__ == "__main__":
    sys.exit(main())
