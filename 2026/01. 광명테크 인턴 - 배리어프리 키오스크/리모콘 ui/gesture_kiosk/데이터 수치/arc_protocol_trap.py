# -*- coding: utf-8 -*-
"""렌즈 비교 측정 도중 auto_arc가 언제, 무엇을 보고 발동했나 (2026-09-11).

왜 측정하나
-----------
05번 문서 §2와 논문 Ⅵ-2는 이렇게 설명했다 — "사람이 실제로 쓸 때는 결정계수가
0.051로 낮아 auto_arc가 발동하지 않는데, 휨을 측정하려고 만든 순수 1축 회전은
0.568로 높아 안전장치를 통과했다. 그래서 측정 도중 auto_arc가 학습해 휨이 최대
8배 나빠졌다."

그런데 0.051·0.568은 저장소 어디에도 측정 기록이 없다(출처로 적힌 개발일지가
저장소에 없다). 게다가 렌즈 비교 측정의 순수 좌우 훑기는 87프레임뿐이라, auto_arc가
적합을 시작하는 최소 표본(240)에 못 미친다. 발동했다면 그 앞의 예열 도중이었을 수도
있다. 그래서 **그 측정을 그대로 다시 돌리면서, auto_arc가 적합을 시도할 때마다
단계·결정계수·계수·통과 여부를 기록한다.** 예전 설명을 맞추려는 게 아니라, 나오는
대로 적는다.

무엇을 돌리나 — 렌즈 3종 × 입력 3종, 배치 '오른쪽에서 25도', 600 mm
---------------------------------------------------------------------
  ① 렌즈 비교 측정 그대로 (lens_ablation.py ⑩의 '보정 끔' 줄): 중립 70프레임 →
     예열 70초(ablation.warm_up_lens — 좌우·상하·몸 이동이 섞인 사인파) → 좌우 ±14°
     훑기(도마다 3프레임). 훑기에서 측정한 휨이 렌즈보정_전후비교.csv와 같아야
     이 기록이 그 측정의 기록이다 — 먼저 그걸 확인한다.
  ② 실사용 — 키오스크 화면의 무작위 지점을 차례로 겨누는 2분(arc_t_gate.py와 같은
     생성기, 시드 1).
  ③ 순수 좌우 왕복 — 60초 동안 ±14° 사인파(8초 주기).

설정은 렌즈 비교 측정 때와 같게 둔다 — 응시 기능을 포함한 배포 설정 그대로다.
(arc_t_gate.py는 import만 해도 응시 기능을 끄므로 불러오지 않고 생성기를 옮겨 왔다.)

    py -3 "데이터 수치/arc_protocol_trap.py"
"""
import csv
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ablation을 먼저 불러온다 — 표준출력 UTF-8 래핑이 거기 있다(두 번 감싸면 버퍼가 닫힌다)
from ablation import _Face, build, warm_up_lens  # noqa: E402
import numpy as np  # noqa: E402
try:
    from src.postprocess import auto_arc as arc_mod  # noqa: E402
except ImportError:
    raise SystemExit("auto_arc는 2026-09-24 삭제됐습니다. 이 측정은 삭제 전 판에서 "
                     "돌려야 합니다 — 데이터 수치/00_README.md '삭제한 기법' 참고")
from tests.virtual_camera import MOUNTS, VirtualCamera, rotation  # noqa: E402

LENSES = ("일반 65도", "광각 90도", "초광각 120도")
MOUNT = "오른쪽에서 25도"
DISTANCE_MM = 600.0
FPS = 30.0
HALF_X_DEG, HALF_Y_DEG = 15.0, 10.0

LOG = []
STATE = {"lens": None, "input": None, "phase": None, "clock": None}
_original_refit = arc_mod.OnlineArcCompensator._maybe_refit


def _recording_refit(self):
    """원래 판정을 그대로 따라가며 기록하고, 계수 갱신은 원래 함수에 맡긴다."""
    pairs = [(x, y) for x, y in zip(self._xs, self._ys)
             if abs(x) <= arc_mod.FIT_X_LIMIT]
    reason, r2, c, span = "", float("nan"), float("nan"), float("nan")
    if len(pairs) < arc_mod.MIN_SAMPLES // 2:
        reason = "표본 부족"
    else:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        span = max(xs) - min(xs)
        if span < arc_mod.MIN_X_SPAN:
            reason = "가로 폭 부족"
        else:
            fit = arc_mod._fit_quadratic(xs, ys)
            if fit is None:
                reason = "적합 실패"
            else:
                c, r2 = fit
                if r2 < arc_mod.MIN_R2:
                    reason = "결정계수 미달"
    _original_refit(self)
    LOG.append({
        "렌즈": STATE["lens"], "입력": STATE["input"], "단계": STATE["phase"],
        "시각(초)": round(STATE["clock"].now, 2), "표본": len(pairs),
        "가로 폭": span, "결정계수": r2, "계수": c,
        "통과": int(reason == ""), "거절 이유": reason or "-",
        "적용 뒤 계수": self.coef,
    })


arc_mod.OnlineArcCompensator._maybe_refit = _recording_refit


def make(lens, input_name):
    tracker, clock = build({"orientation_lens_calibration": False,
                            "orientation_auto_arc": True})
    cam = VirtualCamera(lens=lens, distance_mm=DISTANCE_MM, seed=3,
                        mount=MOUNTS[MOUNT])
    cam.reset_noise()
    STATE.update(lens=lens, input=input_name, clock=clock)

    def step(R=None, off=(0.0, 0.0, 0.0)):
        clock.now += 1.0 / FPS
        return tracker.update(_Face(cam.observe(R, offset_mm=off).landmarks_3d))

    STATE["phase"] = "중립"
    for _ in range(70):
        step()
    return tracker, clock, cam, step


def protocol_replica(lens):
    """① lens_ablation.py ⑩ '보정 끔'과 같은 순서 -> (휨 %, 훑기 직전 계수, 끝난 뒤 계수)."""
    tracker, clock, cam, step = make(lens, "① 렌즈 비교 측정 그대로")
    STATE["phase"] = "예열 70초"
    warm_up_lens(tracker, clock, cam)
    coef_before_sweep = tracker._cursor_mapper._auto_arc.coef
    cam.reset_noise()
    STATE["phase"] = "좌우 ±14° 훑기"
    ys = []
    for deg in np.arange(-14.0, 14.01, 1.0):
        r = None
        for _ in range(3):
            r = step(rotation((0, 1, 0), float(deg)))
        if r is not None and r.cursor_x_ratio is not None:
            ys.append(r.cursor_y_ratio)
    ys = np.array(ys)
    bow = 100.0 * float(np.abs(ys - ys.mean()).max())
    return bow, coef_before_sweep, tracker._cursor_mapper._auto_arc.coef


def kiosk_user(seconds, seed):
    """arc_t_gate.py와 같은 생성기 — 화면의 무작위 지점을 차례로 겨눈다."""
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
    return [(y + s[0], p + s[1]) for (y, p), s in zip(poses, shake)]


def realistic_use(lens):
    tracker, _clock, _cam, step = make(lens, "② 실사용 (무작위 지점 겨누기 2분)")
    STATE["phase"] = "실사용 2분"
    for yaw, pitch in kiosk_user(120.0, seed=1):
        step(rotation((0, 1, 0), yaw) @ rotation((1, 0, 0), pitch))
    return tracker._cursor_mapper._auto_arc.coef


def pure_sweep(lens):
    tracker, _clock, _cam, step = make(lens, "③ 순수 좌우 왕복 60초")
    STATE["phase"] = "좌우 왕복 60초"
    for i in range(int(60.0 * FPS)):
        step(rotation((0, 1, 0), 14.0 * math.sin(2.0 * math.pi * i / (8.0 * FPS))))
    return tracker._cursor_mapper._auto_arc.coef


def recorded_bow(lens):
    """렌즈보정_전후비교.csv의 ⑩ '보정 끔' 휨 — 재현이 맞는지 견줄 기준."""
    with open(os.path.join(HERE, "렌즈보정_전후비교.csv"), encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if row["조건"].startswith("⑩") and row["설정"] == "%s / 보정 끔" % lens:
                return float(row["세로 휨(%)"])
    return None


def main():
    started = time.time()
    replica = {}
    finals = {}
    for lens in LENSES:
        replica[lens] = protocol_replica(lens)
        finals[(lens, "②")] = realistic_use(lens)
        finals[(lens, "③")] = pure_sweep(lens)
        print("  %s 끝 (%.0f초)" % (lens, time.time() - started))
        sys.stdout.flush()

    # 재현 확인도 파일로 남긴다 — 문서의 숫자를 여기서 가져간다
    check_path = os.path.join(HERE, "곡률보정_재현확인.csv")
    with open(check_path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["렌즈", "재현한 휨(%)", "기록된 휨(%)", "같음",
                         "훑기 직전 계수", "끝난 뒤 계수"])
        for lens in LENSES:
            bow, coef_before, coef_after = replica[lens]
            ref = recorded_bow(lens)
            writer.writerow([lens, round(bow, 4), ref,
                             int(ref is not None and abs(bow - ref) < 0.05),
                             round(coef_before, 4), round(coef_after, 4)])

    path = os.path.join(HERE, "곡률보정_적합기록.csv")
    columns = ["렌즈", "입력", "단계", "시각(초)", "표본", "가로 폭", "결정계수", "계수",
               "통과", "거절 이유", "적용 뒤 계수"]
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        for row in LOG:
            writer.writerow([round(row[k], 4) if isinstance(row[k], float) else row[k]
                             for k in columns])

    print("\n■ ① 재현 확인 — 휨이 CSV와 같아야 이 기록이 그 측정의 기록이다")
    for lens in LENSES:
        bow, coef_before, coef_after = replica[lens]
        ref = recorded_bow(lens)
        same = ref is not None and abs(bow - ref) < 0.05
        print("  %-10s 휨 %.3f%% (CSV %s) %s · 훑기 직전 계수 %+.3f → 끝난 뒤 %+.3f"
              % (lens, bow, "%.3f%%" % ref if ref is not None else "없음",
                 "✓ 같음" if same else "✗ 다름", coef_before, coef_after))

    print("\n■ 입력별 적합 시도")
    for lens in LENSES:
        for name in dict.fromkeys(r["입력"] for r in LOG):
            rows = [r for r in LOG if r["렌즈"] == lens and r["입력"] == name]
            if not rows:
                continue
            fitted = [r for r in rows if not math.isnan(r["결정계수"])]
            passed = [r for r in rows if r["통과"]]
            r2s = sorted(r["결정계수"] for r in fitted)
            phases = {}
            for r in passed:
                phases[r["단계"]] = phases.get(r["단계"], 0) + 1
            print("  %-10s %-26s 시도 %3d · 적합 %3d · 통과 %3d · 결정계수 %s%s"
                  % (lens, name, len(rows), len(fitted), len(passed),
                     ("%.3f~%.3f (중앙값 %.3f)" % (r2s[0], r2s[-1], r2s[len(r2s) // 2]))
                     if r2s else "-",
                     ("  · 통과한 단계 " + ", ".join("%s %d" % kv for kv in phases.items()))
                     if phases else ""))
            if passed:
                best = max(passed, key=lambda r: r["결정계수"])
                print("  %-10s %-26s   통과 중 결정계수 최대 %.3f (계수 %+.3f, %s %.1f초)"
                      % ("", "", best["결정계수"], best["계수"], best["단계"], best["시각(초)"]))
    print("\n저장: %s (%d행, %.0f초)" % (path, len(LOG), time.time() - started))


if __name__ == "__main__":
    main()
