# -*- coding: utf-8 -*-
"""기법별 적용 전/후 비교 (ablation) — 2026-09-09.

논문에 "이 기법을 넣어서 좋아졌다"고 쓰려면 **끄고 재고 켜고 재야** 한다.
각 기법을 하나씩만 바꿔서 같은 시나리오·같은 지표로 비교한다. 모든 값은
**실제 HeadTracker를 통과한 커서 위치**에서 나온다.

지표 (전부 화면 비율 %, 낮을수록 좋다)
  · 몸 평행이동 끌림 — 몸만 움직였을 때 커서가 따라간 양. 0이어야 한다
  · 가로 선형성 이탈 — 고개를 좌우로 돌릴 때 커서가 직선에서 벗어난 양
  · 세로 휨(곡률)   — 좌우로만 돌렸는데 커서가 위아래로 움직인 양
  · 정지 떨림       — 가만히 있을 때 커서 표준편차

★측정이 실제로 무언가를 걸고 있는지 스스로 확인한다. 커서가 회전에 반응하지
않으면(가로 이동폭이 너무 작으면) 수치를 내지 않고 "측정 불가"로 표시한다 —
0.000%가 "완벽하다"가 아니라 "안 움직였다"인 경우를 걸러내기 위해서다.

돌리는 법 (프로젝트 루트에서):
    py "데이터 수치/ablation.py"          # 렌즈 자가 보정 빼고 (빠름)
    py "데이터 수치/ablation.py" --lens   # 렌즈 자가 보정까지 (10분+)
"""
import copy
import csv
import io
import math
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from src.postprocess.head_tracker import HeadTracker
from src.utils.config_loader import load_config
from tests.virtual_camera import (FRAME_H_PX, FRAME_W_PX, MOUNTS,
                                  VirtualCamera, rotation)

CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")
SCREEN_W_MM, SCREEN_H_MM = 531.0, 299.0     # 24인치 16:9 실측 표시영역
MIN_TRAVEL_PCT = 8.0                        # 이만큼은 움직여야 측정으로 인정한다


class _Face:
    frame_size = (FRAME_W_PX, FRAME_H_PX)

    def __init__(self, pts):
        self.landmarks_3d = np.asarray(pts, dtype=np.float64)
        self.landmarks_px = self.landmarks_3d[:, :2].astype(np.float32)
        self.blendshapes = {}

    def landmark_px(self, i):
        x, y = self.landmarks_px[i]
        return float(x), float(y)

    def landmarks_mean_px(self, idx):
        p = self.landmarks_px[list(idx)]
        return float(p[:, 0].mean()), float(p[:, 1].mean())

    def blendshape(self, name, default=0.0):
        return self.blendshapes.get(name, default)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


VC_NOSE = 4   # 가상 카메라가 채우는 코끝. 기본값 1번은 합성되지 않아 (0,0)이다


def nose_point(face):
    """2차원 경로용 커서 기준점. 가상 카메라가 그리는 점이라야 측정이 성립한다."""
    return face.landmark_px(VC_NOSE)


def build(overrides, two_d=False):
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    p = cfg["head_tracker"]["pointer"]
    p["orientation_mapping"] = True
    p["one_euro_enabled"] = False
    p["smoothing_alpha"] = 1.0
    p["orientation_lens_calibration"] = False
    p["orientation_distance_scaling"] = False
    p["orientation_invert_x"] = False
    p["orientation_reach_gain"] = 1.0
    p["screen_width_mm"] = p["screen_height_mm"] = None
    p["reference_distance_mm"] = None
    p["arc_compensation"] = 0.0
    p.update(overrides)
    clock = _Clock()
    fn = nose_point if two_d else None
    return HeadTracker(cfg, clock=clock, cursor_point_fn=fn), clock


def warm_up_lens(tracker, clock, cam, seconds=70.0, fps=30.0):
    rng = np.random.default_rng(5)
    for i in range(int(seconds * fps)):
        t = i / fps
        off = (200.0 * math.sin(t * 0.32) + rng.normal(0, 8),
               110.0 * math.sin(t * 0.21 + 1.0) + rng.normal(0, 6),
               90.0 * math.sin(t * 0.14))
        R = (rotation((0, 1, 0), 13.0 * math.sin(t * 1.1))
             @ rotation((1, 0, 0), 8.0 * math.sin(t * 0.8 + 0.5)))
        clock.now += 1.0 / fps
        tracker.update(_Face(cam.observe(R, offset_mm=off).landmarks_3d))
        cal = tracker._cursor_mapper._lens_calibrator
        if cal is not None:
            end = time.time() + 20.0
            while (cal._thread is not None and cal._thread.is_alive()
                   and time.time() < end):
                time.sleep(0.004)


def measure(overrides, lens="광각 90도", mount="정면", calib_distance=600.0,
            measure_distance=None, noise=None, do_warmup=False, swing=14.0):
    """한 설정에서 네 지표를 잰다 -> dict. 측정이 성립 안 하면 None.

    calib_distance 에서 중립을 잡고, measure_distance 로 옮겨서 잰다
    (다르게 주면 '사용자가 앞뒤로 움직인' 상황이 된다).
    """
    measure_distance = measure_distance or calib_distance
    two_d = overrides.get("orientation_mapping", True) is False
    tracker, clock = build(overrides, two_d=two_d)

    def make_cam(dist):
        kw = dict(lens=lens, distance_mm=dist, seed=3)
        if mount != "정면":
            kw["mount"] = MOUNTS[mount]
        if noise is not None:
            kw["noise_px"] = noise
        c = VirtualCamera(**kw)
        c.reset_noise()
        return c

    cam = make_cam(calib_distance)

    def step(R=None, off=(0.0, 0.0, 0.0)):
        clock.now += 1.0 / 30.0
        return tracker.update(_Face(cam.observe(R, offset_mm=off).landmarks_3d))

    for _ in range(70):                      # 중립 캘리브레이션
        step()
    if do_warmup:
        warm_up_lens(tracker, clock, cam)

    if measure_distance != calib_distance:   # 사용자가 옮겨 간다
        cam = make_cam(measure_distance)
        for _ in range(120):                 # 거리 비율 EMA가 수렴할 시간
            step()
    cam.reset_noise()

    xs, ys, tans = [], [], []
    for d in np.arange(-swing, swing + 0.01, 1.0):
        R = rotation((0, 1, 0), float(d))
        r = None
        for _ in range(3):
            r = step(R)
        if r is None or r.cursor_x_ratio is None:
            continue
        xs.append(r.cursor_x_ratio); ys.append(r.cursor_y_ratio)
        tans.append(math.tan(math.radians(float(d))))
    if len(xs) < 8:
        return None
    xs, ys, tans = np.array(xs), np.array(ys), np.array(tans)
    travel = 100.0 * float(xs.max() - xs.min())
    if travel < MIN_TRAVEL_PCT:              # 커서가 회전에 반응하지 않았다
        return {"측정불가": True, "이동폭": travel}
    a, b = np.polyfit(tans, xs, 1)
    linearity = 100.0 * float(np.abs(xs - (a * tans + b)).max())
    bow = 100.0 * float(np.abs(ys - ys.mean()).max())

    cam.reset_noise()
    still = []
    for _ in range(90):
        r = step()
        if r and r.cursor_x_ratio is not None:
            still.append(r.cursor_x_ratio)
    jitter = 100.0 * float(np.std(still[30:])) if len(still) > 40 else float("nan")

    cam.reset_noise()
    base = None
    for _ in range(30):
        base = step()
    drift = 0.0
    for sh in ((60, 0, 0), (-60, 0, 0), (0, 50, 0), (0, 0, 120)):
        r = None
        for _ in range(12):
            r = step(off=sh)
        if r and base and r.cursor_x_ratio is not None:
            drift = max(drift, 100.0 * math.hypot(
                r.cursor_x_ratio - base.cursor_x_ratio,
                r.cursor_y_ratio - base.cursor_y_ratio))
        for _ in range(12):                  # 원위치로 돌아온 뒤 다음 방향
            step()
    return {"끌림": drift, "선형성": linearity, "휨": bow, "떨림": jitter,
            "이동폭": travel}


ROWS = []


def compare(name, sides, note=""):
    """sides = [(라벨, overrides, kwargs), ...] 를 나란히 재서 비교한다."""
    print(f"\n■ {name}")
    if note:
        print(f"   {note}")
    print(f"   {'설정':<26s} {'끌림':>8s} {'선형성':>8s} {'휨':>8s} {'떨림':>8s} {'이동폭':>8s}")
    got = []
    for label, ov, kw in sides:
        m = measure(ov, **kw)
        if m is None:
            print(f"   {label:<26s}  측정 실패 (표본 부족)")
            got.append(None); continue
        if m.get("측정불가"):
            print(f"   {label:<26s}  ★측정 불가 — 커서가 회전에 반응하지 않음"
                  f" (이동폭 {m['이동폭']:.2f}%)")
            ROWS.append([name, label, "측정불가", "측정불가", "측정불가",
                         "측정불가", round(m["이동폭"], 3)])
            got.append(None); continue
        print(f"   {label:<26s} {m['끌림']:7.3f}% {m['선형성']:7.3f}%"
              f" {m['휨']:7.3f}% {m['떨림']:7.3f}% {m['이동폭']:7.2f}%")
        ROWS.append([name, label, round(m["끌림"], 4), round(m["선형성"], 4),
                     round(m["휨"], 4), round(m["떨림"], 4), round(m["이동폭"], 3)])
        got.append(m)
    if len(got) == 2 and got[0] and got[1]:
        for k in ("끌림", "선형성", "휨", "떨림"):
            if got[0][k] > 1e-6:
                chg = (got[1][k] - got[0][k]) / got[0][k] * 100.0
                if abs(chg) >= 5.0:
                    print(f"      {k}: {got[0][k]:.3f}% -> {got[1][k]:.3f}%  ({chg:+.0f}%)")


def main():
    do_lens = "--lens" in sys.argv
    print("=" * 78)
    print(" 기법별 적용 전/후 비교 — 실제 HeadTracker 통과, 가상 카메라 관측")
    print(" 지표는 전부 화면 비율 %. 낮을수록 좋다.")
    print("=" * 78)

    compare("① 상대 회전 매핑 (Kabsch 정합)", [
        ("2차원 투영점 (예전)", {"orientation_mapping": False}, {}),
        ("상대 회전 (현재)", {"orientation_mapping": True}, {}),
    ], note="몸이 움직여도 커서가 안 밀리는가 — 이 프로젝트의 핵심 변경")

    compare("② 얼굴 기준 좌표계 (2차원 경로 안에서)", [
        ("화면 절대좌표", {"orientation_mapping": False, "face_local": False}, {}),
        ("얼굴 기준 좌표계", {"orientation_mapping": False, "face_local": True}, {}),
    ], note="상대 회전을 쓰기 전, 2차원 경로에서 몸 밀림을 줄이던 방법")

    compare("③ 곡률 보정 (2차원 경로 전용 — 상대 회전에서는 무시됨)", [
        ("보정 없음 (0.0)",
         {"orientation_mapping": False, "arc_compensation": 0.0}, {}),
        ("보정 적용 (-0.8936)",
         {"orientation_mapping": False, "arc_compensation": -0.8936}, {}),
    ], note="세로 휨을 가로오프셋의 2차식으로 눌러 편다")

    compare("④ 거리 비례 반폭 (600mm에서 잡고 1300mm로 물러남)", [
        ("끔", {"orientation_distance_scaling": False},
         {"calib_distance": 600.0, "measure_distance": 1300.0, "swing": 6.5}),
        ("켬", {"orientation_distance_scaling": True},
         {"calib_distance": 600.0, "measure_distance": 1300.0, "swing": 6.5}),
    ], note="1300mm에서 화면 끝에 닿는 각도는 ±6.5도다. 그만큼만 돌려 커서 이동폭을 본다")

    compare("⑤ 화면 기하 기반 반폭", [
        ("고정 15도", {}, {}),
        ("화면 치수 (531x299mm / 700mm)",
         {"screen_width_mm": SCREEN_W_MM, "screen_height_mm": SCREEN_H_MM,
          "reference_distance_mm": 700.0}, {}),
    ], note="반폭을 atan((화면 절반)/거리)로 계산 — 감도가 바뀌므로 다른 지표도 함께 변한다")

    compare("⑥ 도달 배율 (목이 7도만 돌아가는 사용자용)", [
        ("1.0 (손대지 않음)", {"orientation_reach_gain": 1.0}, {"swing": 7.0}),
        ("2.18 (7도 사용자)", {"orientation_reach_gain": 2.18}, {"swing": 7.0}),
    ], note="좌우 7도까지만 돌린다. 커서 이동폭이 화면의 몇 %까지 가는지가 핵심")

    compare("⑦ 카메라 배치 (광각 90도)", [
        ("정면", {}, {"mount": "정면"}),
        ("오른쪽에서 25도", {}, {"mount": "오른쪽에서 25도"}),
        ("심하게 비스듬히", {}, {"mount": "심하게 비스듬히"}),
    ], note="설치 각도가 나빠지면 얼마나 나빠지나")

    compare("⑧ 렌즈 화각 (정면 배치)", [
        ("왜곡없음", {}, {"lens": "왜곡없음"}),
        ("일반 65도", {}, {"lens": "일반 65도"}),
        ("광각 90도", {}, {"lens": "광각 90도"}),
        ("초광각 120도", {}, {"lens": "초광각 120도"}),
    ], note="렌즈 왜곡이 커질수록 얼마나 나빠지나 (렌즈 자가 보정은 끈 상태)")

    compare("⑨ 랜드마크 잡음 (어두운 곳)", [
        ("0.35px (기본)", {}, {"noise": 0.35}),
        ("0.70px", {}, {"noise": 0.70}),
        ("1.40px (어두움)", {}, {"noise": 1.40}),
    ], note="조명이 나빠지면 커서가 얼마나 흔들리나")

    if do_lens:
        print("\n■ ⑩ 렌즈 자가 보정 (조건마다 70초 예열 — 느림)")
        print(f"   {'설정':<30s} {'끌림':>8s} {'선형성':>8s} {'휨':>8s}")
        for lens in ("일반 65도", "광각 90도", "초광각 120도"):
            for on in (False, True):
                m = measure({"orientation_lens_calibration": on},
                            lens=lens, mount="오른쪽에서 25도", do_warmup=on)
                label = f"{lens} / 보정 {'켬' if on else '끔'}"
                if not m or m.get("측정불가"):
                    print(f"   {label:<30s}  측정 불가")
                    continue
                print(f"   {label:<30s} {m['끌림']:7.3f}% {m['선형성']:7.3f}%"
                      f" {m['휨']:7.3f}%")
                ROWS.append(["⑩ 렌즈 자가 보정", label, round(m["끌림"], 4),
                             round(m["선형성"], 4), round(m["휨"], 4),
                             round(m["떨림"], 4), round(m["이동폭"], 3)])
    else:
        print("\n■ ⑩ 렌즈 자가 보정 — 건너뜀 (--lens 를 붙이면 측정한다)")

    path = os.path.join(HERE, "기법별_전후비교.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["기법", "설정", "몸평행이동 끌림(%)", "가로 선형성 이탈(%)",
                    "세로 휨(%)", "정지 떨림(%)", "커서 이동폭(%)"])
        w.writerows(ROWS)
    print(f"\n저장: 기법별_전후비교.csv ({len(ROWS)}행)")


if __name__ == "__main__":
    main()
