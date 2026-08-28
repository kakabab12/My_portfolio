"""보정 도구 — 실제 제스처를 재서 임계값을 자동으로 맞춘다 (2026-08-03 신설).

사용법 (엔진 폴더에서, **쓰는 자리에서 서서** 실행):
    py scripts\\calibrate.py              # 측정 → config.yaml 자동 반영(백업 남김)
    py scripts\\calibrate.py --dry-run    # 측정·권장값만 보고 반영은 안 함
    py scripts\\calibrate.py --only swipe # 일부만 (shape | swipe | tap)

★왜 필요한가(2026-08-03 사고): 임계값을 개발 PC에서 **앉아서** 재어 맞췄더니
키오스크에서 오히려 나빠졌다 — 사람은 서서 할 때 동작이 크다. 값은 반드시
**쓰는 자리에서** 재야 한다. 그래서 재는 것과 반영을 한 번에 한다.

측정 항목 → 반영되는 값:
  A. 손 모양 3종(정지)              → extend_ratio (굽힘·폄 분포의 빈 구간 한가운데)
  B. 쓸기 — **좌·우·위 × 작게·크게** → min_dist_x/y_shoulder, flick_min_dist_shoulder
     (방향마다 최소·중앙·최대를 따로 남긴다: 좌/우 비대칭과 이상치를 가려내기 위해)
  C. 검지 까딱                      → dip_drop_ratio
  D. 손목 까딱                      → move_dip_shoulder
계산·config 반영은 src/utils/calibration.py (순수 함수 — 단위 테스트 있음).

조작: 자세를 잡고 **SPACE** → 3초 뒤 계측 시작 · s 건너뛰기 · q/ESC 중단
"""
import argparse
import datetime
import os
import shutil
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.capture.camera_stream import CameraStream, init_camera
from src.inference.hand_tracker import HandTracker
from src.inference.preprocessor import Preprocessor
from src.postprocess.hand_select import (
    STANDARD_SHOULDER_M, hand_span_px, hand_span_world_m,
)
from src.postprocess.hand_shape import finger_states, hand_center_point
from src.utils.calibration import (
    apply_to_config_text, check_measurements, format_report, recommend_thresholds,
)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

WINDOW = "gesture_kiosk calibration"
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")
SHAPE_SEC, SWIPE_SEC, TAP_SEC = 5.0, 7.0, 8.0
READY_SEC = 3.0               # SPACE를 누르고 자세를 잡을 시간 (run_phase 주석 참고)
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
        pen.text((420, 84), sub, font=FONT_SMALL, fill=(190, 190, 190))
    cv2.imshow(WINDOW, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    return cv2.waitKey(1) & 0xFF


def shoulder_px(hand):
    """이 손 기준 가상 어깨너비(px) — 임계 단위(어깨너비 배수)의 자(尺)."""
    span_px, span_m = hand_span_px(hand.landmarks), hand_span_world_m(hand.world_landmarks)
    if span_px <= 0 or span_m <= 0:
        return None
    return span_px / span_m * STANDARD_SHOULDER_M


def run_phase(camera, preprocessor, tracker, title, detail, duration_sec, collect):
    """SPACE로 시작 → READY_SEC 준비 → duration_sec 계측. 중단 시 False.

    매 프레임 collect(hand, shoulder_px, now).

    ★2026-08-03 사고로 재작성: 종전 게이트가 "손이 보이면 시작"이라, 앞 단계가
    끝난 뒤 손이 화면에 남아 있으면 **다음 단계가 곧바로 이어서 시작**됐다.
    사용자가 다음 자세를 취하기도 전에 6~8초가 지나가, 앞 자세나 쉬고 있는 손이
    그 단계의 값으로 담겼다(진단 세션에서 3단계가 통째로 오염). 값을 config에
    자동 반영하는 도구라 조용한 오염이 제일 위험하다 — 사람이 누른 뒤에만 시작한다.
    """
    state, start = "wait", None
    while True:
        frame = preprocessor.preprocess_frame(camera.capture_frame())
        hands = tracker.infer(frame)
        hand = max(hands, key=lambda h: hand_span_px(h.landmarks)) if hands else None
        if hand is not None:
            center = hand_center_point(hand.landmarks)
            if center:
                cv2.circle(frame, (int(center[0]), int(center[1])), 12, (60, 220, 60), 2)
        seen = "손 보임" if hand is not None else "손 안 보입니다"
        if state == "wait":
            key = draw(frame, title, detail, "자세를 잡고 SPACE를 누르세요",
                       f"{seen} · s 건너뛰기 · q 중단")
            if key == 32:
                state, start = "ready", time.monotonic()
        elif state == "ready":
            remain = READY_SEC - (time.monotonic() - start)
            key = draw(frame, title, detail, f"{max(0.0, remain):.0f} 후 시작", seen)
            if remain <= 0:
                state, start = "measure", time.monotonic()
        else:
            remain = duration_sec - (time.monotonic() - start)
            if hand is not None:
                sp = shoulder_px(hand)
                collect(hand, sp, time.monotonic())
                note = f"측정 중 (어깨 자 {sp:.0f}px)" if sp else "측정 중"
            else:
                note = seen
            key = draw(frame, title, detail, f"남은 시간 {max(0.0, remain):4.1f}초", note)
            if remain <= 0:
                return True
        if key in (ord("q"), 27):
            return False
        if key == ord("s") and state == "wait":
            return True


def measure_shape(ctx, cfg, measured):
    """모양 3종 정지 — 굽힌 손가락 상단(p90)과 편 손가락 하단(p10)을 얻는다."""
    shape_cfg = cfg["hand_select"]["hand_shape"]
    ext, curl = shape_cfg["extend_ratio"], shape_cfg.get("curl_confirm_ratio", 0.85)
    curled, extended = [], []
    for key, label, detail, bucket in (
        ("fist", "1) 주먹", "주먹을 쥐고 가슴 높이에서 정지", curled),
        ("finger", "2) 한 손가락", "검지만 펴고 정지", None),
        ("open", "3) 손바닥", "손가락을 전부 펴고 정지", extended),
    ):
        rows = []

        def collect(hand, sp, now, _r=rows):
            _r.extend(float(r) for r, _ in finger_states(hand.world_landmarks, ext, curl))

        if not run_phase(*ctx, label, detail, SHAPE_SEC, collect):
            return False
        if not rows:
            continue
        if bucket is curled:
            curled.extend(rows)
        elif bucket is extended:
            extended.extend(rows)
        else:   # 한 손가락 — 편 1개·굽힘 3개가 섞여 있어 위/아래로 나눠 담는다
            rows.sort()
            curled.extend(rows[:len(rows) * 3 // 4])
            extended.extend(rows[len(rows) * 3 // 4:])
    if len(curled) > 10:
        measured["shape_curl_p90"] = statistics.quantiles(curled, n=10)[8]
    if len(extended) > 10:
        measured["shape_extend_p10"] = statistics.quantiles(extended, n=10)[0]
    return True


def measure_swipe(ctx, cfg, measured):
    """쓸기를 **방향별(좌·우·위) × 크기별(작게·크게)** 로 따로 잰다 — 6단계.

    ★2026-08-03 사용자 지적으로 재작성: 종전엔 좌/우를 절댓값으로 합치고
    "작게"만 쟀다. 그러면 ①좌/우 비대칭(몸을 가로지르는 쪽이 대체로 작다)을
    못 보고 ②최소값이 이상치(중단된 동작)인지 판단할 근거가 없다.
    방향마다 최소·중앙·최대를 남겨 계산부가 이상치를 걸러내고, 좌/우 차이와
    "작게=크게"(측정 무효)를 경고할 수 있게 한다 (src/utils/calibration.py).
    """
    window_sec = cfg["gestures"]["swipe"]["window_sec"]
    phases = (
        ("left", "small", "4) 왼쪽 — 작게", "왼쪽으로 5회 — 인식되길 바라는 최소 크기로"),
        ("left", "big", "5) 왼쪽 — 크게", "왼쪽으로 5회 — 평소 시원하게 쓰는 크기로"),
        ("right", "small", "6) 오른쪽 — 작게", "오른쪽으로 5회 — 최소 크기로"),
        ("right", "big", "7) 오른쪽 — 크게", "오른쪽으로 5회 — 시원하게"),
        ("up", "small", "8) 위 — 작게", "위로 5회 — 최소 크기로"),
        ("up", "big", "9) 위 — 크게", "위로 5회 — 시원하게"),
    )
    # 방향 -> (좌표 축, 그 방향을 양수로 만드는 부호): 화면 y는 아래로 증가
    AXIS = {"left": (0, -1.0), "right": (0, 1.0), "up": (1, -1.0)}
    for direction, size, label, detail in phases:
        pts, sps = [], []

        def collect(hand, sp, now, _p=pts, _s=sps):
            center = hand_center_point(hand.landmarks)
            if center and sp:
                _p.append((now, center[0], center[1]))
                _s.append(sp)

        if not run_phase(*ctx, label, detail, SWIPE_SEC, collect):
            return False
        if not pts:
            continue
        axis, sign = AXIS[direction]
        sp_med = statistics.median(sps)
        peaks = []   # 엔진과 같은 창에서 그 방향으로 나아간 최대 이동
        for i, (t0, x0, y0) in enumerate(pts):
            best = 0.0
            for t1, x1, y1 in pts[i:]:
                if t1 - t0 > window_sec:
                    break
                delta = ((x1 - x0) if axis == 0 else (y1 - y0)) * sign
                best = max(best, delta)
            if best > 0:
                peaks.append((t0, best / sp_med))
        strokes = []   # 시간상 떨어진 정점만 = 서로 다른 획 (되돌리는 손 제외)
        for t0, travel in sorted(peaks, key=lambda peak: -peak[1]):
            if travel >= 0.05 and all(abs(t0 - s[0]) > 0.6 for s in strokes):
                strokes.append((t0, travel))
            if len(strokes) >= 5:
                break
        if not strokes:
            continue
        travels = sorted(travel for _, travel in strokes)
        prefix = f"swipe_{direction}_{size}"
        measured[f"{prefix}_min"] = travels[0]
        measured[f"{prefix}_median"] = statistics.median(travels)
        measured[f"{prefix}_max"] = travels[-1]
        logger.info("쓸기 %s/%s 획 %d개: %s", direction, size, len(travels),
                    ", ".join(f"{travel:.3f}" for travel in travels))
    return True


def measure_tap(ctx, cfg, measured):
    """검지 까딱·손목 까딱 — 각 채널의 실제 하강 폭을 얻는다."""
    shape_cfg = cfg["hand_select"]["hand_shape"]
    ext, curl = shape_cfg["extend_ratio"], shape_cfg.get("curl_confirm_ratio", 0.85)
    for key, label, detail in (
        ("index", "10) 검지 까딱", "검지만 빠르게 까딱 2회 × 4세트 (손은 고정)"),
        ("wrist", "11) 손목 까딱", "손목으로 손을 아래로 톡 떨궜다 올리기 2회 × 4세트"),
    ):
        rows = []

        def collect(hand, sp, now, _r=rows):
            states = finger_states(hand.world_landmarks, ext, curl)
            center = hand_center_point(hand.landmarks)
            if states and center and sp:
                _r.append((float(states[0][0]), center[1], sp))

        if not run_phase(*ctx, label, detail, TAP_SEC, collect):
            return False
        if len(rows) < 20:
            continue
        if key == "index":
            ratios = [r for r, _, _ in rows]
            base = statistics.quantiles(ratios, n=10)[8]
            bottom = statistics.quantiles(ratios, n=20)[0]   # 하위 5% = 까딱 바닥
            if base > 0:
                measured["tap_index_drop"] = max(0.0, (base - bottom) / base)
        else:
            sp_med = statistics.median(s for _, _, s in rows)
            ys = [y for _, y, _ in rows]
            top = statistics.quantiles(ys, n=10)[0]          # 최고점(화면 y 최소)
            bottom = statistics.quantiles(ys, n=20)[18]      # 하위 = 떨군 지점
            measured["tap_wrist_drop"] = max(0.0, (bottom - top) / sp_med)
    return True


def main():
    parser = argparse.ArgumentParser(description="gesture_kiosk 임계값 보정")
    parser.add_argument("--dry-run", action="store_true",
                        help="측정·권장값만 출력하고 config는 건드리지 않는다")
    parser.add_argument("--only", choices=("shape", "swipe", "tap"),
                        help="일부 단계만 측정")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    init_logging(cfg)
    preprocessor = Preprocessor(cfg)
    tracker = HandTracker(cfg)
    camera = CameraStream(cfg, device_id=cfg["camera"]["device_id"],
                          cap=init_camera(cfg)).start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    ctx = (camera, preprocessor, tracker)

    measured, aborted = {}, False
    try:
        for name, func in (("shape", measure_shape), ("swipe", measure_swipe),
                           ("tap", measure_tap)):
            if args.only in (None, name) and not func(ctx, cfg, measured):
                aborted = True
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    swipe_cfg = cfg["gestures"]["swipe"]
    current = {
        "min_dist_x_shoulder": swipe_cfg.get("min_dist_x_shoulder"),
        "min_dist_y_shoulder": swipe_cfg.get("min_dist_y_shoulder"),
        "flick_min_dist_shoulder": swipe_cfg.get("flick_min_dist_shoulder"),
        "extend_ratio": cfg["hand_select"]["hand_shape"].get("extend_ratio"),
        "dip_drop_ratio": (cfg["gestures"].get("tap_click") or {}).get("dip_drop_ratio"),
        "move_dip_shoulder": (cfg["gestures"].get("tap_click") or {}).get("move_dip_shoulder"),
    }
    picks = recommend_thresholds(measured, current)
    warnings = check_measurements(measured)   # 측정 품질 — 사람에게 먼저 알린다

    if aborted:
        print("[중단] 측정을 끝내지 않아 config를 바꾸지 않습니다.")
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        text = handle.read()
    today = datetime.date.today().isoformat()
    new_text, missed = apply_to_config_text(text, picks, today)
    print(format_report(picks, missed, measured, warnings))

    if aborted or args.dry_run or not picks:
        if args.dry_run:
            print("[안내] --dry-run — config는 그대로입니다.")
        return
    backup = CONFIG_PATH + f".bak-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy2(CONFIG_PATH, backup)   # 되돌릴 수 있게 항상 백업부터
    with open(CONFIG_PATH, "w", encoding="utf-8", newline="") as handle:
        handle.write(new_text)
    print(f"[적용] configs\\config.yaml 갱신 — 백업: {os.path.basename(backup)}")
    print("[안내] 엔진을 다시 시작하면 새 값으로 동작합니다 (py main.py)")
    logger.info("자동 보정 적용: %s", {k: v for k, (v, _) in picks.items()})


if __name__ == "__main__":
    main()
