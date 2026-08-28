"""보정 도구 — 실제 제스처를 재서 임계값을 자동으로 맞춘다 (2026-08-03 신설).

사용법 (엔진 폴더에서, **쓰는 자리에서 서서** 실행):
    py scripts\\calibrate.py              # 측정 → config.yaml 자동 반영(백업 남김)
    py scripts\\calibrate.py --dry-run    # 측정·권장값만 보고 반영은 안 함
    py scripts\\calibrate.py --only swipe # 일부만 (shape | swipe | tap)

★왜 필요한가(2026-08-03 사고): 임계값을 개발 PC에서 **앉아서** 재어 맞췄더니
키오스크에서 오히려 나빠졌다 — 사람은 서서 할 때 동작이 크다. 값은 반드시
**쓰는 자리에서** 재야 한다. 그래서 재는 것과 반영을 한 번에 한다.

측정 항목 → 반영되는 값:
  A. 손 모양 3종(정지)   → extend_ratio (굽힘·폄 분포의 빈 구간 한가운데)
  B. 쓸기 좌/우/위(작게)  → min_dist_x/y_shoulder, flick_min_dist_shoulder
  C. 검지 까딱           → dip_drop_ratio
  D. 손목 까딱           → move_dip_shoulder
계산·config 반영은 src/utils/calibration.py (순수 함수 — 단위 테스트 있음).

조작: 손을 화면에 보이면 각 단계가 시작 · SPACE 건너뛰기 · q/ESC 중단
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
from src.utils.calibration import apply_to_config_text, format_report, recommend_thresholds
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging
from src.utils.console import enable_utf8_output

WINDOW = "gesture_kiosk calibration"
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")
SHAPE_SEC, SWIPE_SEC, TAP_SEC = 5.0, 8.0, 8.0
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
    """손이 보이면 계측 시작 — 매 프레임 collect(hand, shoulder_px, now). 중단 시 False.

    대기 게이트(2026-08-03): 창이 뜨자마자 타이머가 돌면 지시를 읽는 사이에
    단계가 끝나 데이터가 빈다 — 손이 들어와야 시작한다.
    """
    start = None
    while True:
        frame = preprocessor.preprocess_frame(camera.capture_frame())
        hands = tracker.infer(frame)
        hand = max(hands, key=lambda h: hand_span_px(h.landmarks)) if hands else None
        if hand is not None:
            center = hand_center_point(hand.landmarks)
            if center:
                cv2.circle(frame, (int(center[0]), int(center[1])), 12, (60, 220, 60), 2)
        if start is None:
            if hand is not None:
                start = time.monotonic()
            key = draw(frame, title, detail,
                       "손을 화면에 보이면 시작합니다..." if hand is None else "측정 시작!",
                       "SPACE 건너뛰기 · q 중단")
        else:
            remain = duration_sec - (time.monotonic() - start)
            if hand is not None:
                sp = shoulder_px(hand)
                collect(hand, sp, time.monotonic())
                note = f"측정 중 (어깨 자 {sp:.0f}px)" if sp else "측정 중"
            else:
                note = "손이 안 보입니다"
            key = draw(frame, title, detail, f"남은 시간 {max(0.0, remain):4.1f}초", note)
            if remain <= 0:
                return True
        if key in (ord("q"), 27):
            return False
        if key == 32:
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
    """좌/우/위 "작게" 쓸기 — 획별 이동량의 최소값이 임계의 근거다."""
    window_sec = cfg["gestures"]["swipe"]["window_sec"]
    for axis_key, label, detail, axis, sign in (
        ("x", "4) 좌/우 쓸기 — 작게", "좌우로 5회 — 인식되길 바라는 **최소 크기**로", 0, None),
        ("y", "5) 위 쓸기 — 작게", "위로 5회 — 인식되길 바라는 **최소 크기**로", 1, -1.0),
    ):
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
        sp_med = statistics.median(sps)
        peaks = []
        for i, (t0, x0, y0) in enumerate(pts):
            best = 0.0
            for t1, x1, y1 in pts[i:]:
                if t1 - t0 > window_sec:
                    break
                d = (x1 - x0) if axis == 0 else (y1 - y0)
                d = abs(d) if sign is None else d * sign
                best = max(best, d)
            if best > 0:
                peaks.append((t0, best / sp_med))
        strokes = []
        for t0, travel in sorted(peaks, key=lambda p: -p[1]):
            if travel >= 0.05 and all(abs(t0 - s[0]) > 0.6 for s in strokes):
                strokes.append((t0, travel))
            if len(strokes) >= 5:
                break
        if strokes:
            travels = sorted(travel for _, travel in strokes)
            measured[f"swipe_{axis_key}_small_min"] = travels[0]
            logger.info("쓸기(%s) 획 %d개: %s", axis_key, len(travels),
                        ", ".join(f"{travel:.3f}" for travel in travels))
    return True


def measure_tap(ctx, cfg, measured):
    """검지 까딱·손목 까딱 — 각 채널의 실제 하강 폭을 얻는다."""
    shape_cfg = cfg["hand_select"]["hand_shape"]
    ext, curl = shape_cfg["extend_ratio"], shape_cfg.get("curl_confirm_ratio", 0.85)
    for key, label, detail in (
        ("index", "6) 검지 까딱", "검지만 빠르게 까딱 2회 × 4세트 (손은 고정)"),
        ("wrist", "7) 손목 까딱", "손목으로 손을 아래로 톡 떨궜다 올리기 2회 × 4세트"),
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
    enable_utf8_output()   # cp949 콘솔에서 줄표(—) 등으로 죽는 것 방지
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

    if aborted:
        print("[중단] 측정을 끝내지 않아 config를 바꾸지 않습니다.")
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        text = handle.read()
    today = datetime.date.today().isoformat()
    new_text, missed = apply_to_config_text(text, picks, today)
    print(format_report(picks, missed, measured))

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
