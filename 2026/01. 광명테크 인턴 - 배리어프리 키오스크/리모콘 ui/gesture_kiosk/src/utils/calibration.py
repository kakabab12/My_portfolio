"""utils 모듈 — 보정 측정값에서 임계값을 계산하고 config에 반영한다 (2026-08-03 신설).

★도입 배경(사용자 결정): 임계값을 감으로 정하면 현장에서 어긋난다. 2026-08-03
보정 세션에서 책상 앞에 **앉아서** 잰 값으로 좌우 임계를 0.55→0.20으로 내렸다가
키오스크에서 오히려 나빠져 되돌린 사고가 있었다 — 사람은 서서 할 때 동작이 크다.
그래서 **쓰는 자리에서 재고 그 자리에서 반영**하는 도구가 필요하다
(scripts/calibrate.py가 이 모듈을 쓴다).

여기 있는 함수는 전부 순수 함수 — 카메라·모델 없이 단위 테스트한다
(tests/test_calibration.py). 측정(카메라)과 계산(여기)을 분리한 이유다.
"""
import re

from src.utils.logger import get_logger

logger = get_logger("utils")

# 안전 범위 — 측정이 이상해도 config가 망가지지 않게 가둔다 (튜닝 사고 방지).
# 하한은 "잡음(정지 떨림)보다 확실히 큰" 값, 상한은 "사람이 낼 수 있는" 값 기준.
LIMITS = {
    "min_dist_x_shoulder": (0.15, 0.90),
    "min_dist_y_shoulder": (0.10, 0.60),
    "flick_min_dist_shoulder": (0.08, 0.45),
    "extend_ratio": (0.80, 1.30),
    "dip_drop_ratio": (0.08, 0.35),
    "move_dip_shoulder": (0.03, 0.30),
}
SWIPE_MARGIN = 0.90   # 쓸기 임계 = "작게" 한 동작의 최소값 × 이 비율 — 가장 작게 한
                      #   동작도 확실히 잡히도록 살짝 아래에 둔다
FLICK_RATIO = 0.80    # 플릭 임계 = 본 임계 × 이 비율 (본 임계보다 낮아야 경로가 산다)
TAP_TARGET_MARGIN = 0.85   # 탭 임계 = 실제 까딱 깊이 × 이 비율 (얕게 해도 잡히게)


def _clamp(key, value):
    low, high = LIMITS[key]
    return max(low, min(high, round(value, 3)))


def recommend_thresholds(measured, current):
    """측정값 -> {키: 권장값} + 사유 (순수 함수).

    measured: scripts/calibrate.py가 모은 요약
      swipe_x_small_min / swipe_y_small_min : "작게" 쓸기 최소 이동(어깨너비 배수)
      shape_curl_p90 / shape_extend_p10     : 굽힘·폄 손가락 비율 분포 경계
      tap_index_drop  : 검지 까딱의 실제 하강 비율(기준선 대비 0~1)
      tap_wrist_drop  : 손목 까딱의 실제 하강(어깨너비 배수)
    current: 현재 config 값 — 측정이 없는 항목은 그대로 둔다.
    반환: {키: (권장값, 사유 문자열)} — 값이 그대로면 항목을 넣지 않는다.
    """
    picks = {}

    def put(key, value, reason):
        value = _clamp(key, value)
        if current.get(key) != value:
            picks[key] = (value, reason)

    x_min = measured.get("swipe_x_small_min")
    if x_min:
        put("min_dist_x_shoulder", x_min * SWIPE_MARGIN,
            f"작게 한 좌/우 쓸기 최소 {x_min:.3f} × {SWIPE_MARGIN}")
    y_min = measured.get("swipe_y_small_min")
    if y_min:
        put("min_dist_y_shoulder", y_min * SWIPE_MARGIN,
            f"작게 한 위 쓸기 최소 {y_min:.3f} × {SWIPE_MARGIN}")

    # 플릭은 본 임계보다 낮아야 "짧고 빠른 까딱" 경로가 산다 — 둘 중 작은 쪽 기준.
    # ★쓸기를 실제로 잰 경우에만 손댄다: 측정 없이 현재값에서 파생시키면 재지도
    # 않은 항목이 바뀐다 (tests/test_calibration.py가 잡은 결함, 2026-08-03)
    base_x = picks.get("min_dist_x_shoulder", (current.get("min_dist_x_shoulder"), ""))[0]
    base_y = picks.get("min_dist_y_shoulder", (current.get("min_dist_y_shoulder"), ""))[0]
    bases = [v for v in (base_x, base_y) if v]
    if bases and (x_min or y_min):
        put("flick_min_dist_shoulder", min(bases) * FLICK_RATIO,
            f"본 임계 최소({min(bases):.2f}) × {FLICK_RATIO}")

    # 손 모양: 굽힘 분포 위끝과 폄 분포 아래끝 사이의 **한가운데**가 가장 안전하다
    curl_p90 = measured.get("shape_curl_p90")
    extend_p10 = measured.get("shape_extend_p10")
    if curl_p90 and extend_p10 and extend_p10 > curl_p90:
        put("extend_ratio", (curl_p90 + extend_p10) / 2.0,
            f"굽힘 상단 {curl_p90:.2f} ~ 폄 하단 {extend_p10:.2f}의 중간")

    index_drop = measured.get("tap_index_drop")
    if index_drop:
        put("dip_drop_ratio", index_drop * TAP_TARGET_MARGIN,
            f"검지 까딱 실측 하강 {index_drop:.2f} × {TAP_TARGET_MARGIN}")

    wrist_drop = measured.get("tap_wrist_drop")
    if wrist_drop:
        # ★제약: 손목 까딱 임계는 **위 쓸기 임계보다 작아야** 한다 — 크면 까딱이
        # select로 먼저 확정된다 (2026-08-03 설계 근거, config 주석과 동일)
        cap = (base_y or current.get("min_dist_y_shoulder") or 0.16) * 0.8
        value = min(wrist_drop * TAP_TARGET_MARGIN, cap)
        put("move_dip_shoulder", value,
            f"손목 까딱 실측 {wrist_drop:.2f} × {TAP_TARGET_MARGIN}"
            + (f" (위 쓸기 임계의 80%={cap:.2f}로 제한)" if value >= cap else ""))
    return picks


def apply_to_config_text(text, picks, today):
    """config.yaml 원문에서 해당 키의 숫자만 바꾼다 -> (새 원문, 미적용 키 목록).

    주석을 그대로 보존하는 게 핵심이다 — 이 저장소의 config는 값마다 실측 근거와
    날짜가 붙어 있고(기획서 4.7), 통째로 다시 쓰면 그 이력이 사라진다.
    바뀐 줄 끝에는 자동 보정 표시를 남겨 나중에 손으로 정한 값과 구분한다.
    """
    missed = []
    for key, (value, reason) in picks.items():
        pattern = re.compile(rf"^(\s*){key}:\s*[0-9.]+(.*)$", re.MULTILINE)
        match = pattern.search(text)
        if match is None:
            missed.append(key)
            continue
        indent, tail = match.group(1), match.group(2)
        tail = re.sub(r"\s*#\s*\[자동보정[^\]]*\]", "", tail)   # 이전 표시 제거(중복 방지)
        stamp = f"  # [자동보정 {today}] {reason}"
        text = pattern.sub(lambda _m: f"{indent}{key}: {value}{tail}{stamp}", text, count=1)
    return text, missed


def format_report(picks, missed, measured):
    """사람이 읽을 요약 — 콘솔 출력·로그용."""
    lines = ["=== 보정 결과 ==="]
    if measured:
        lines.append("측정: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(measured.items()) if v is not None))
    if not picks:
        lines.append("변경 없음 — 현재 값이 측정과 이미 맞습니다.")
    for key, (value, reason) in sorted(picks.items()):
        lines.append(f"  {key}: {value}   ({reason})")
    if missed:
        lines.append("⚠ config에서 키를 못 찾아 건너뜀: " + ", ".join(missed))
    return "\n".join(lines)
