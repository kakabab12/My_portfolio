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
SWIPE_MARGIN = 0.90   # 쓸기 임계 = "작게" 한 동작의 하한 × 이 비율 — 가장 작게 한
                      #   동작도 확실히 잡히도록 살짝 아래에 둔다
FLICK_RATIO = 0.80    # 플릭 임계 = 본 임계 × 이 비율 (본 임계보다 낮아야 경로가 산다)
TAP_TARGET_MARGIN = 0.85   # 탭 임계 = 실제 까딱 깊이 × 이 비율 (얕게 해도 잡히게)
OUTLIER_FLOOR_RATIO = 0.5  # 이상치 방어 — "작게" 5회 중 하나가 중단된 동작이면 최소값이
                      #   비정상적으로 작다. 중앙값의 이 비율 아래로는 내려가지 않는다
                      #   (2026-08-03 사용자 지적: 최소만 재면 이상치인지 알 수 없다)
ASYMMETRY_WARN = 1.5  # 좌/우 동작 크기 비가 이 배수를 넘으면 보고 — 임계는 작은 쪽에
                      #   맞출 수밖에 없어(단일 값) 큰 쪽이 과민해질 수 있다
SIZE_SANITY_RATIO = 1.3  # "크게"가 "작게"보다 이 배수는 커야 측정이 유효하다 —
                      #   둘이 비슷하면 두 단계를 같은 크기로 한 것(측정 무효)


def _clamp(key, value):
    low, high = LIMITS[key]
    return max(low, min(high, round(value, 3)))


def small_floor(measured, direction):
    """그 방향의 "작게" 동작 하한 -> 값 | None (이상치 방어 포함).

    최소값만 쓰면 5회 중 하나가 중단된 동작일 때 임계가 과하게 낮아진다 —
    중앙값의 절반 아래로는 내려가지 않게 받친다 (2026-08-03 사용자 지적).
    """
    low = measured.get(f"swipe_{direction}_small_min")
    if low is None:
        return None
    median = measured.get(f"swipe_{direction}_small_median")
    if median:
        low = max(low, median * OUTLIER_FLOOR_RATIO)
    return low


def check_measurements(measured):
    """측정이 쓸 만한지 점검 -> 경고 문자열 목록 (순수 함수).

    ①"크게"가 "작게"와 비슷하면 두 단계를 같은 크기로 한 것 — 임계 근거가 약하다.
    ②좌/우 크기가 많이 다르면 단일 임계로는 한쪽이 과민해진다 — 사용자에게 알린다.
    """
    warnings = []
    for direction, label in (("left", "왼쪽"), ("right", "오른쪽"), ("up", "위")):
        small = measured.get(f"swipe_{direction}_small_median")
        big = measured.get(f"swipe_{direction}_big_median")
        if small and big and big < small * SIZE_SANITY_RATIO:
            warnings.append(
                f"{label} 쓸기: 작게({small:.2f})와 크게({big:.2f})가 비슷합니다 — "
                "'작게'를 더 작게 다시 재면 임계가 정확해집니다")
    left = small_floor(measured, "left")
    right = small_floor(measured, "right")
    if left and right:
        ratio = max(left, right) / min(left, right)
        if ratio >= ASYMMETRY_WARN:
            bigger = "오른쪽" if right > left else "왼쪽"
            warnings.append(
                f"좌/우 동작 크기가 {ratio:.1f}배 차이납니다(작게: 좌 {left:.2f} · "
                f"우 {right:.2f}) — 임계는 작은 쪽에 맞추므로 {bigger} 쓸기가 "
                "상대적으로 민감해집니다")
    return warnings


def recommend_thresholds(measured, current):
    """측정값 -> {키: (권장값, 사유)} (순수 함수).

    measured: scripts/calibrate.py가 모은 요약 — 방향별·크기별로 따로 잰다
      swipe_{left|right|up}_{small|big}_{min|median|max} : 획 이동량(어깨너비 배수)
      shape_curl_p90 / shape_extend_p10 : 굽힘·폄 손가락 비율 분포 경계
      tap_index_drop : 검지 까딱의 실제 하강 비율(기준선 대비 0~1)
      tap_wrist_drop : 손목 까딱의 실제 하강(어깨너비 배수)
    current: 현재 config 값 — 측정이 없는 항목은 그대로 둔다.
    값이 현재와 같으면 항목을 넣지 않는다(불필요한 config 수정 방지).
    """
    picks = {}

    def put(key, value, reason):
        value = _clamp(key, value)
        if current.get(key) != value:
            picks[key] = (value, reason)

    # 좌/우는 config가 임계 하나(min_dist_x)를 공유한다 — **작은 쪽**에 맞춰야
    # 양쪽 다 잡힌다 (큰 쪽은 여유가 생길 뿐 못 잡히지는 않는다)
    left, right = small_floor(measured, "left"), small_floor(measured, "right")
    sides = {"왼쪽": left, "오른쪽": right}
    available = {name: value for name, value in sides.items() if value}
    x_floor = min(available.values()) if available else None
    if x_floor:
        harder = min(available, key=available.get)
        detail = " · ".join(f"{name} {value:.3f}" for name, value in available.items())
        put("min_dist_x_shoulder", x_floor * SWIPE_MARGIN,
            f"작게 한 쓸기({detail}) 중 작은 쪽 {harder} 기준 × {SWIPE_MARGIN}")
    y_floor = small_floor(measured, "up")
    if y_floor:
        put("min_dist_y_shoulder", y_floor * SWIPE_MARGIN,
            f"작게 한 위 쓸기 하한 {y_floor:.3f} × {SWIPE_MARGIN}")
    x_min, y_min = x_floor, y_floor

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


def format_report(picks, missed, measured, warnings=()):
    """사람이 읽을 요약 — 콘솔 출력·로그용.

    쓸기는 **방향별·크기별 표**로 보여준다 — 좌/우 비대칭이나 "작게=크게"
    (측정 무효)를 사용자가 눈으로 바로 알아채게 하기 위해서다 (2026-08-03).
    """
    lines = ["=== 보정 결과 ==="]
    rows = []
    for direction, label in (("left", "왼쪽"), ("right", "오른쪽"), ("up", "위")):
        cells = []
        for size, size_label in (("small", "작게"), ("big", "크게")):
            low = measured.get(f"swipe_{direction}_{size}_min")
            median = measured.get(f"swipe_{direction}_{size}_median")
            high = measured.get(f"swipe_{direction}_{size}_max")
            if median is None:
                continue
            cells.append(f"{size_label} {low:.2f}~{high:.2f}(중앙 {median:.2f})")
        if cells:
            rows.append(f"  {label:4s}| " + "  |  ".join(cells))
    if rows:
        lines.append("[쓸기 측정 — 방향별·크기별 (어깨너비 배수)]")
        lines.extend(rows)
    others = {key: value for key, value in measured.items()
              if not key.startswith("swipe_") and value is not None}
    if others:
        lines.append("[그 밖의 측정] " + ", ".join(
            f"{key}={value:.3f}" for key, value in sorted(others.items())))
    for warning in warnings:
        lines.append("⚠ " + warning)
    lines.append("[반영할 값]")
    if not picks:
        lines.append("  변경 없음 — 현재 값이 측정과 이미 맞습니다.")
    for key, (value, reason) in sorted(picks.items()):
        lines.append(f"  {key}: {value}   ({reason})")
    if missed:
        lines.append("⚠ config에서 키를 못 찾아 건너뜀: " + ", ".join(missed))
    return "\n".join(lines)
