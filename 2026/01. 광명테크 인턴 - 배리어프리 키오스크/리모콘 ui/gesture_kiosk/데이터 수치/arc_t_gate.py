# -*- coding: utf-8 -*-
"""곡률 자동 보정의 판정을 결정계수에서 t값으로 바꾸면 (2026-09-11 재측정).

왜 다시 측정했나
----------------
논문 Ⅵ-3의 수치(t = 3.70, 6배 악화, 계수 4배 추정)는 예전 초안에만 있고 측정
기록이 없었다. 그 실험 코드는 결과가 나빠 되돌리면서 함께 지워졌다. 그래서
**같은 질문을 다시 세워 처음부터 측정한다.** 예전 숫자를 맞추려는 게 아니라,
나오는 대로 적는다.

질문
----
auto_arc는 2차 회귀의 결정계수 R²가 0.15 미만이면 그 창을 버린다. 사람이 고개를
좌우·상하로 섞어 움직이면 R²가 낮아 한 번도 발동하지 않는다(05번 문서 §2).
그러면 판정을 "계수가 통계적으로 0이 아닌가"(|t| ≥ 2)로 바꾸고 창을 2분으로
늘리면, 실사용에서도 곡률을 배울 수 있지 않을까?

방법
----
1) 가상 사용자가 키오스크 화면의 무작위 지점을 차례로 겨눈다 — 좌우와 상하가
   섞인 실사용 움직임(이동 0.4~0.9초 코사인 보간, 머묾 0.6~2.0초, 떨림 0.12도).
   2분(3600프레임)을 실제 HeadTracker에 통과시키며 auto_arc가 받는 입력(탄젠트
   단위 가로·세로)을 그대로 모은다. 모으는 동안 auto_arc는 재적합하지 않게 막는다.
2) auto_arc와 같은 거름(화면 안 |x| ≤ 0.30, 표본 수·가로 폭 하한)을 거친 표본으로
   y = a + b·x + c·x² 적합 → c, R², c의 t값(보통최소제곱 표준오차).
3) 같은 카메라로 고개를 **좌우로만** 20초 훑어 얻은 c를 참값(c_참)으로 둔다 —
   보정기가 지워야 할 계통 곡률이다.
4) 추정한 c를 보정기에 실제로 걸고(상한 ±2.0, 재적합 막음) 좌우로만 훑어 휨을
   측정한다(ablation.py와 같은 휨 정의). 보정 없음·참값 보정과 견준다.
5) 사용자가 **의도한** 세로 움직임(겨눈 지점의 세로 각도 탄젠트)과 가로²의
   상관을 직접 측정한다 — "우연한 상관" 설명이 맞는지 보는 값이다.
6) 시드를 바꿔 반복한다(논문 절차 6단계).

주의 — t값의 표준오차는 표본이 서로 독립이라고 가정한다. 30fps 영상은 이웃
프레임이 거의 같아서 이 가정이 크게 틀리고, 그만큼 t가 부풀려진다. 이 스크립트는
그걸 바로잡지 않고 **소박한 t 판정이 실제로 무엇을 하는지**를 측정한다.

    py -3 "데이터 수치/arc_t_gate.py"            # 광각 90도 20시드 + 초광각 120도 10시드
    py -3 "데이터 수치/arc_t_gate.py" --quick    # 광각 90도 3시드 (시간 확인용)
"""
import copy
import csv
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ablation을 먼저 불러온다 — 표준출력 UTF-8 래핑이 거기 있다(두 번 감싸면 버퍼가 닫힌다)
import ablation  # noqa: E402
from ablation import _Face  # noqa: E402
import numpy as np  # noqa: E402
from src.postprocess.auto_arc import (  # noqa: E402
    FIT_X_LIMIT, MAX_COEF, MIN_R2, MIN_SAMPLES, MIN_X_SPAN)
from tests.virtual_camera import VirtualCamera, rotation  # noqa: E402


def _load_config_without_interaction(path, _load=ablation.load_config):
    """측정 도중 스스로 상태를 바꾸는 응시 기능 둘을 끈다 (2026-09-11 추가).

    recenter_dwell(커서가 2.5초 반경 안에 머물면 지금 자세를 새 중립으로)은 가상
    사용자가 가까운 지점을 연달아 겨누며 오래 머물 때 **2분 수집 도중 중립을 옮겨
    버릴 수 있다.** 그러면 표본에 계단이 섞여 곡률 추정이 오염된다 — 거리 조절 측정
    (distance_scaling.py)에서 실제로 이 기능이 이동 조건 20개 중 19개를 오염시켰다.
    dwell_click(1.5초 응시 → 선택)은 커서를 바꾸지 않지만 같은 부류라 함께 끈다.
    처음 돌린 결과는 이 두 기능이 켜진 채였다.
    """
    cfg = copy.deepcopy(_load(path))
    for key in ("recenter_dwell", "dwell_click"):
        cfg["head_tracker"][key]["enabled"] = False
    return cfg


ablation.load_config = _load_config_without_interaction
build = ablation.build

FPS = 30.0
WINDOW_SEC = 120.0
T_GATE = 2.0
HALF_X_DEG, HALF_Y_DEG = 15.0, 10.0   # 기준 거리가 없을 때의 반폭 (build() 기본)
DISTANCE_MM = 600.0
SWEEP_DEG = 14.0


def kiosk_user(seconds, seed):
    """키오스크 화면의 무작위 지점을 차례로 겨눈다 -> [(yaw도, pitch도, 의도 세로 tan)]."""
    rng = np.random.default_rng(seed)
    total = int(seconds * FPS)
    yaw = pitch = 0.0
    poses = []
    while len(poses) < total:
        fx, fy = rng.uniform(-0.9, 0.9, 2)
        target_yaw = math.degrees(math.atan(fx * math.tan(math.radians(HALF_X_DEG))))
        target_pitch = math.degrees(math.atan(fy * math.tan(math.radians(HALF_Y_DEG))))
        move = max(1, int(round(rng.uniform(0.4, 0.9) * FPS)))
        start_yaw, start_pitch = yaw, pitch
        for k in range(move):
            ratio = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / move)
            yaw = start_yaw + (target_yaw - start_yaw) * ratio
            pitch = start_pitch + (target_pitch - start_pitch) * ratio
            poses.append((yaw, pitch))
        dwell = int(round(rng.uniform(0.6, 2.0) * FPS))
        poses.extend([(yaw, pitch)] * dwell)
    poses = poses[:total]
    shake = rng.normal(0.0, 0.12, (total, 2))
    return [(y + s[0], p + s[1], math.tan(math.radians(p)))
            for (y, p), s in zip(poses, shake)]


def fit(xs, ys):
    """y = a + b·x + c·x² 보통최소제곱 -> c, R², c의 t값."""
    n = len(xs)
    if n < 6:
        return None
    X = np.column_stack([np.ones(n), xs, xs * xs])
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
    resid = ys - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return None
    cov = (ss_res / (n - 3)) * np.linalg.inv(X.T @ X)
    c = float(beta[2])
    return {"c": c, "r2": 1.0 - ss_res / ss_tot,
            "t": c / math.sqrt(max(float(cov[2, 2]), 1e-30)),
            "n": n, "span": float(xs.max() - xs.min())}


def make(lens, seed):
    """캘리브레이션까지 마친 추적기. auto_arc는 스스로 재적합하지 않게 막는다."""
    tracker, clock = build({"orientation_auto_arc": True})
    comp = tracker._cursor_mapper._auto_arc
    if comp is None:
        raise SystemExit("auto_arc가 만들어지지 않았다 — 설정을 확인할 것")
    comp._maybe_refit = lambda: None
    cam = VirtualCamera(lens=lens, distance_mm=DISTANCE_MM, seed=seed)
    cam.reset_noise()

    def step(R=None):
        clock.now += 1.0 / FPS
        return tracker.update(_Face(cam.observe(R).landmarks_3d))

    for _ in range(70):
        step()
    return comp, step


def record_inputs(comp, with_intent):
    """auto_arc가 받는 (가로, 세로[, 의도])를 가로챈다."""
    records, intent = [], [0.0]
    original = comp.update

    def recording(x, y):
        records.append((x, y, intent[0]) if with_intent else (x, y))
        return original(x, y)

    comp.update = recording
    return records, intent


def screen_only(xs, *others):
    keep = np.abs(xs) <= FIT_X_LIMIT
    return (xs[keep],) + tuple(o[keep] for o in others)


def true_curvature(lens, seed):
    comp, step = make(lens, seed)
    records, _ = record_inputs(comp, with_intent=False)
    for i in range(int(20.0 * FPS)):
        yaw = SWEEP_DEG * math.sin(2.0 * math.pi * i / (8.0 * FPS))
        step(rotation((0, 1, 0), yaw))
    xs = np.array([p[0] for p in records])
    ys = np.array([p[1] for p in records])
    xs, ys = screen_only(xs, ys)
    return fit(xs, ys)


def bow_with(lens, seed, coef):
    """보정 계수를 걸고 좌우로만 훑었을 때의 휨(%) — ablation.py와 같은 정의."""
    comp, step = make(lens, seed)
    comp.coef = float(coef)
    ys = []
    for deg in np.arange(-SWEEP_DEG, SWEEP_DEG + 0.01, 1.0):
        r = None
        for _ in range(3):
            r = step(rotation((0, 1, 0), float(deg)))
        if r is not None and r.cursor_y_ratio is not None:
            ys.append(r.cursor_y_ratio)
    ys = np.array(ys)
    return 100.0 * float(np.abs(ys - ys.mean()).max())


def one_seed(lens, seed):
    comp, step = make(lens, seed)
    records, intent = record_inputs(comp, with_intent=True)
    for yaw, pitch, tan_pitch in kiosk_user(WINDOW_SEC, seed):
        intent[0] = tan_pitch
        step(rotation((0, 1, 0), yaw) @ rotation((1, 0, 0), pitch))
    xs = np.array([p[0] for p in records])
    ys = np.array([p[1] for p in records])
    intents = np.array([p[2] for p in records])
    xs, ys, intents = screen_only(xs, ys, intents)

    est = fit(xs, ys)
    truth = true_curvature(lens, seed)
    enough = (est is not None and est["n"] >= MIN_SAMPLES // 2
              and est["span"] >= MIN_X_SPAN)
    applied = max(-MAX_COEF, min(MAX_COEF, est["c"]))
    bow_none = bow_with(lens, seed, 0.0)
    bow_truth = bow_with(lens, seed, truth["c"])
    bow_est = bow_with(lens, seed, applied)
    return {
        "렌즈": lens, "시드": seed, "표본": est["n"], "R2": est["r2"],
        "t": est["t"], "c_추정": est["c"], "c_참": truth["c"],
        "참값_R2": truth["r2"],
        "추정/참": est["c"] / truth["c"] if truth["c"] else float("nan"),
        "의도세로_가로제곱_상관": float(np.corrcoef(intents, xs * xs)[0, 1]),
        "R2판정": int(enough and est["r2"] >= MIN_R2),
        "t판정": int(enough and abs(est["t"]) >= T_GATE),
        "휨_보정없음": bow_none, "휨_참값보정": bow_truth, "휨_추정보정": bow_est,
        "추정보정/보정없음": bow_est / bow_none if bow_none else float("nan"),
    }


COLUMNS = ["렌즈", "시드", "표본", "R2", "t", "c_추정", "c_참", "참값_R2", "추정/참",
           "의도세로_가로제곱_상관", "R2판정", "t판정", "휨_보정없음", "휨_참값보정",
           "휨_추정보정", "추정보정/보정없음"]


def summarize(rows, lens):
    picked = [r for r in rows if r["렌즈"] == lens]
    if not picked:
        return

    def med(values):
        return float(np.median(values)) if values else float("nan")

    passed = [r for r in picked if r["t판정"]]
    print("\n■ %s — 시드 %d개, 2분 창" % (lens, len(picked)))
    print("  참값 c (좌우만) 중앙값 %+.3f, 그때 R² %.3f"
          % (med([r["c_참"] for r in picked]), med([r["참값_R2"] for r in picked])))
    print("  섞인 움직임 R² 중앙값 %.3f → R² 판정 통과 %d/%d"
          % (med([r["R2"] for r in picked]), sum(r["R2판정"] for r in picked), len(picked)))
    print("  t 중앙값 %.2f (범위 %.2f ~ %.2f) → t 판정 통과 %d/%d"
          % (med([abs(r["t"]) for r in picked]),
             min(abs(r["t"]) for r in picked), max(abs(r["t"]) for r in picked),
             len(passed), len(picked)))
    print("  |의도 세로·가로² 상관| 중앙값 %.3f (범위 %.3f ~ %.3f)"
          % (med([abs(r["의도세로_가로제곱_상관"]) for r in picked]),
             min(abs(r["의도세로_가로제곱_상관"]) for r in picked),
             max(abs(r["의도세로_가로제곱_상관"]) for r in picked)))
    print("  휨: 보정 없음 %.3f%% · 참값 보정 %.3f%% (중앙값)"
          % (med([r["휨_보정없음"] for r in picked]), med([r["휨_참값보정"] for r in picked])))
    if passed:
        ratios = [r["추정보정/보정없음"] for r in passed]
        print("  t 판정 통과한 경우 추정값을 걸면: 휨 %.2f배 (중앙값, 범위 %.2f ~ %.2f),"
              " 나빠진 경우 %d/%d"
              % (med(ratios), min(ratios), max(ratios),
                 sum(1 for x in ratios if x > 1.0), len(ratios)))
        print("  그때 추정 c / 참값 c: 중앙값 %.2f배 (범위 %.2f ~ %.2f)"
              % (med([r["추정/참"] for r in passed]),
                 min(r["추정/참"] for r in passed), max(r["추정/참"] for r in passed)))


def main():
    quick = "--quick" in sys.argv
    plan = ([("광각 90도", range(1, 4))] if quick
            else [("광각 90도", range(1, 21)), ("초광각 120도", range(1, 11))])
    rows = []
    started = time.time()
    for lens, seeds in plan:
        for seed in seeds:
            t0 = time.time()
            row = one_seed(lens, seed)
            rows.append(row)
            print("  %s 시드 %2d  R² %.3f  t %+7.2f  c %+.3f (참 %+.3f)  상관 %+.3f"
                  "  휨 %.2f→%.2f%%  (%.1f초)"
                  % (lens, seed, row["R2"], row["t"], row["c_추정"], row["c_참"],
                     row["의도세로_가로제곱_상관"], row["휨_보정없음"],
                     row["휨_추정보정"], time.time() - t0))
            sys.stdout.flush()

    if not quick:
        path = os.path.join(HERE, "곡률_t값판정_시드별.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(COLUMNS)
            for row in rows:
                writer.writerow([round(row[k], 5) if isinstance(row[k], float) else row[k]
                                 for k in COLUMNS])
        print("\n저장:", path)
    for lens in dict.fromkeys(r["렌즈"] for r in rows):
        summarize(rows, lens)
    print("\n(%.0f초)" % (time.time() - started))


if __name__ == "__main__":
    main()
