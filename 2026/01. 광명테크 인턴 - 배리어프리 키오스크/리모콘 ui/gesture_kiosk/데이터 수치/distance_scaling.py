# -*- coding: utf-8 -*-
"""거리 조절(orientation_distance_scaling)은 언제 효과가 있나 (2026-09-11).

왜 다시 측정했나
----------------
논문 Ⅴ-3은 이 기능을 "켜고 끈 차이가 0.06%p"라고 적었는데, 그 측정이 어디에도
남아 있지 않았다. 게다가 표 2(04번 문서 ④)는 같은 기능이 "도달 범위를 42→91%로
복구했다"고 한다. 둘은 조건이 달라서 둘 다 맞을 수 있다. 이 스크립트가 가른다.

  · 제자리 — 사용자가 선 자리에서 캘리브레이션하고, 그 자리에서 쓴다
  · 이동   — 설치 기준 거리(700 mm)에서 캘리브레이션하고, 다른 거리로 걸어가서 쓴다

코드에서 거리 비율은 `캘리브레이션 때 얼굴 크기 / 지금 얼굴 크기`다
(head_orientation.py). 그러면 제자리에서는 비율이 늘 1이라 켜도 달라질 게 없고,
이동에서는 켜야 맞아야 한다. 코드를 읽고 한 추론을 측정으로 확인한다.

지표 — 조준 오차 (화면 폭·높이 대비 %, 낮을수록 좋다)
------------------------------------------------------
화면 중심에서 가로 ±25·50·75% 지점을 **그 거리에서 기하학적으로 보이는 각도**
(atan(위치 mm / 거리 mm))만큼 고개를 좌우로 돌려 겨누고, 커서가 그 지점에서 얼마나
벗어났는지 측정한다. 세로도 같은 방식(고개만 위아래로). 두 회전을 섞지 않는 이유는
회전 순서 규약이 오차에 섞여 들어가기 때문이다.

07번 문서의 "얼굴이 향한 곳과 커서 사이 거리"(화면 15점, 대각선 대비)와는
**정의가 다르다. 숫자를 직접 견주면 안 된다.**

커서는 화면 끝에서 멈춘다. 너무 가까이 서서 이득이 과하면 오차가 끝에서 잘려
실제보다 작게 나오므로, 끝에 붙은 표본 수를 함께 적는다.

첫 시도가 틀렸던 두 가지 (2026-09-11 진단)
-------------------------------------------
1) 카메라 거리를 **한 프레임에 순간이동**시켰다. 사람은 그렇게 안 움직인다.
   지금은 2초에 걸쳐 걸어가듯 옮기고, 고개도 코사인으로 서서히 돌린다.
2) **응시 재정렬(recenter_dwell)이 켜진 채**였다. 커서가 2.5초 반경 안에 머물면
   지금 자세를 새 중립으로 잡는 기능이라, 옮긴 뒤 가만히 서 있는 동안 새 거리에서
   중립을 다시 잡아 '이동'이 몰래 '제자리'가 됐다(이동 조건 20개 중 19개 오염).
   아래 _load_config_without_interaction 참고.

조건
----
24인치 16:9 (531×299 mm) · 기준 거리 700 mm · 정면 배치 · 렌즈 두 가지
(왜곡없음 = 기하만 보는 대조군, 광각 90도 = 04번 문서와 같은 조건) ·
렌즈 자가 보정·곡률 자동 보정·평활은 끔 (ablation.py의 build()와 같다) ·
응시 재정렬·응시 선택은 끔.

    py -3 "데이터 수치/distance_scaling.py"
"""
import copy
import csv
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ablation을 먼저 불러온다 — 거기서 표준출력을 UTF-8로 감싼다. 여기서 한 번 더
# 감싸면 앞의 래퍼가 정리되면서 출력 버퍼가 닫힌다.
import ablation  # noqa: E402
from ablation import SCREEN_H_MM, SCREEN_W_MM, _Face  # noqa: E402
from tests.virtual_camera import VirtualCamera, rotation  # noqa: E402


def _load_config_without_interaction(path, _load=ablation.load_config):
    """ablation.build()가 읽는 설정에서 **측정 도중 스스로 상태를 바꾸는 기능만** 끈다.

    recenter_dwell(2.5초 응시 → 지금 자세를 새 중립으로)이 켜져 있으면, 거리를 옮긴
    뒤 가만히 서 있는 동안 새 거리에서 중립을 다시 잡아 버린다. dwell_click(1.5초
    응시 → 선택)은 커서를 바꾸지 않지만 같은 부류라 함께 끈다.

    실제 제품에서는 둘 다 켜져 있다 — 사람이 물러난 뒤 가만히 서 있으면 똑같이 새
    거리에서 중립을 다시 잡는다. 그건 따로 적어 둘 발견이고, 여기서는 거리 조절 기능
    **자체**가 무엇을 하는지만 본다.
    """
    cfg = copy.deepcopy(_load(path))
    for key in ("recenter_dwell", "dwell_click"):
        cfg["head_tracker"][key]["enabled"] = False
    return cfg


ablation.load_config = _load_config_without_interaction
build = ablation.build

REF_MM = 700.0
DISTANCES = (400.0, 500.0, 600.0, 700.0, 900.0, 1200.0)
FRACTIONS = (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75)
LENSES = ("왜곡없음", "광각 90도")

# (모드, 기하 경로, 거리 조절, 표 이름)
CONDITIONS = (
    ("제자리", False, False, "기하 끔 (고정 15°/10°)"),
    ("제자리", True, False, "기하 켬 · 거리조절 끔"),
    ("제자리", True, True, "기하 켬 · 거리조절 켬"),
    ("이동", True, False, "기하 켬 · 거리조절 끔"),
    ("이동", True, True, "기하 켬 · 거리조절 켬"),
)


def run(lens, calib_mm, use_mm, geometry, scaling):
    """한 조건에서 조준 오차를 측정한다 -> dict."""
    overrides = {"orientation_distance_scaling": bool(scaling),
                 "orientation_auto_arc": False}
    if geometry:
        overrides.update(screen_width_mm=SCREEN_W_MM,
                         screen_height_mm=SCREEN_H_MM,
                         reference_distance_mm=REF_MM)
    tracker, clock = build(overrides)
    mapper = tracker._cursor_mapper
    max_offset = getattr(mapper, "_max_offset_ratio", 0.5)
    cam = VirtualCamera(lens=lens, distance_mm=calib_mm, seed=3)
    cam.reset_noise()

    def step(R=None, frames=1):
        result = None
        for _ in range(frames):
            clock.now += 1.0 / 30.0
            result = tracker.update(_Face(cam.observe(R).landmarks_3d))
        return result

    def turn(axis, deg, ramp=6, hold=8):
        """사람처럼 서서히 돌리고(코사인) 머문다 -> 머문 마지막 프레임의 결과."""
        for k in range(ramp):
            ratio = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / ramp)
            step(rotation(axis, deg * ratio))
        return step(rotation(axis, deg), frames=hold)

    def turn_back(axis, deg, ramp=6, hold=6):
        for k in range(ramp):
            ratio = 0.5 + 0.5 * math.cos(math.pi * (k + 1) / ramp)
            step(rotation(axis, deg * ratio))
        step(frames=hold)

    step(frames=70)                           # 중립 캘리브레이션
    if use_mm != calib_mm:
        # 걸어서 옮기듯 2초에 걸쳐 거리를 바꾼다 — 한 프레임 순간이동은 비현실적 입력(Ⅵ-1)
        for k in range(60):
            ratio = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / 60)
            cam.distance_mm = calib_mm + (use_mm - calib_mm) * ratio
            step()
        step(frames=120)                      # 거리 비율 EMA(α 0.06)가 따라올 시간
    distance_ratio = getattr(mapper._orientation, "distance_ratio", float("nan"))
    cam.reset_noise()                         # 조준 구간의 잡음을 조건끼리 같게

    r = turn((0, 1, 0), 6.0)                  # 부호 확인 — +로 돌리면 커서가 어디로 가나
    if r is None or r.cursor_x_ratio is None:
        raise RuntimeError("커서가 없다: %s %.0f→%.0f 기하%d 조절%d"
                           % (lens, calib_mm, use_mm, geometry, scaling))
    sign_x = 1.0 if r.cursor_x_ratio > 0.5 else -1.0
    turn_back((0, 1, 0), 6.0)
    r = turn((1, 0, 0), 5.0)
    sign_y = 1.0 if r.cursor_y_ratio > 0.5 else -1.0
    turn_back((1, 0, 0), 5.0)

    per_axis, at_edge, missing = [], 0, 0
    for axis, size_mm, sign, attr in (
            ((0, 1, 0), SCREEN_W_MM, sign_x, "cursor_x_ratio"),
            ((1, 0, 0), SCREEN_H_MM, sign_y, "cursor_y_ratio")):
        errors = []
        for frac in FRACTIONS:
            deg = sign * math.degrees(math.atan(frac * size_mm * 0.5 / use_mm))
            r = turn(axis, deg)
            got = None if r is None else getattr(r, attr)
            if got is None:
                missing += 1
            else:
                if abs(got - 0.5) >= max_offset - 1e-6:
                    at_edge += 1
                errors.append(abs(got - (0.5 + frac * 0.5)) * 100.0)
            turn_back(axis, deg)
        per_axis.append(errors)
    ex, ey = per_axis
    both = ex + ey
    return {"가로": sum(ex) / max(1, len(ex)), "세로": sum(ey) / max(1, len(ey)),
            "평균": sum(both) / max(1, len(both)), "최악": max(both) if both else float("nan"),
            "끝": at_edge, "없음": missing, "거리비율": distance_ratio}


def main():
    started = time.time()
    rows = []
    for lens in LENSES:
        for mode, geometry, scaling, label in CONDITIONS:
            for dist in DISTANCES:
                calib = dist if mode == "제자리" else REF_MM
                m = run(lens, calib, dist, geometry, scaling)
                rows.append([lens, mode, label, int(calib), int(dist),
                             int(geometry), int(scaling),
                             round(m["가로"], 3), round(m["세로"], 3),
                             round(m["평균"], 3), round(m["최악"], 3), m["끝"],
                             m["없음"], round(m["거리비율"], 4)])

    path = os.path.join(HERE, "거리조절_제자리와_이동.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["렌즈", "모드", "조건", "캘리브레이션 거리(mm)",
                         "사용 거리(mm)", "기하", "거리조절",
                         "가로 조준오차(%)", "세로 조준오차(%)", "평균(%)",
                         "최악(%)", "끝에 붙은 표본", "커서 없던 표본", "거리 비율"])
        writer.writerows(rows)

    for lens in LENSES:
        print("\n■ %s — 평균 조준오차(%%), 기준 거리 %d mm" % (lens, int(REF_MM)))
        print("  %-6s %-24s" % ("모드", "조건")
              + "".join("%8d" % int(d) for d in DISTANCES))
        for mode, geometry, scaling, label in CONDITIONS:
            picked = [r for r in rows
                      if r[0] == lens and r[1] == mode and r[2] == label]
            edge = sum(r[11] for r in picked)
            missing = sum(r[12] for r in picked)
            note = []
            if edge:
                note.append("끝에 붙음 %d" % edge)
            if missing:
                note.append("커서 없음 %d" % missing)
            print("  %-6s %-24s" % (mode, label)
                  + "".join("%8.2f" % r[9] for r in picked)
                  + ("   (%s)" % ", ".join(note) if note else ""))
        moved = [r for r in rows if r[0] == lens and r[1] == "이동" and r[6] == 1]
        print("  %-31s" % "  이동 후 거리 비율 (켬)"
              + "".join("%8.3f" % r[13] for r in moved))
        print("  %-31s" % "  기대값 (사용거리/700)"
              + "".join("%8.3f" % (d / REF_MM) for d in DISTANCES))
        for mode in ("제자리", "이동"):
            off = [r[9] for r in rows
                   if r[0] == lens and r[1] == mode and r[5] == 1 and r[6] == 0]
            on = [r[9] for r in rows
                  if r[0] == lens and r[1] == mode and r[5] == 1 and r[6] == 1]
            diffs = [abs(a - b) for a, b in zip(on, off)]
            print("  → %s: 거리조절 켬/끔 차이 평균 %.3f%%p, 최대 %.3f%%p"
                  % (mode, sum(diffs) / len(diffs), max(diffs)))
    print("\n저장: %s (%.0f초)" % (path, time.time() - started))


if __name__ == "__main__":
    main()
