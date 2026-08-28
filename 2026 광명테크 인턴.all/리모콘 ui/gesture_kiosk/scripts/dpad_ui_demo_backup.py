"""가상 D-pad 데모 — 카메라 화면에 방향 버튼 UI를 띄우고 손끝으로 눌러본다.

기존 엔진(main.py)은 **쓸기(swipe) 궤적**으로 동작을 판정하지만, 이 스크립트는
전혀 다른 상호작용을 시험하는 독립 프로토타입이다: 화면에 위/아래/좌/우 +
중앙 버튼으로 이뤄진 D-pad를 그려 놓고, 검지 손끝(landmark 8)이 한 버튼 위에
일정 시간(dwell) 머무르면 그 버튼에 연결된 함수를 실행한다. 손 추적은 기존
HandTracker(MediaPipe)를 그대로 재사용하고, 손끝 좌표는 point_filter의 One
Euro 필터로 떨림을 줄인다 — 판정 로직만 이 파일 안에서 새로 구성한다.

버튼마다 실행되는 함수는 build_zone_functions()에 있다 — 지금은 표시용 출력만
하는 자리표시자(placeholder)이니, 실제 동작이 필요하면 그 함수 본문을 바꾸면
된다 (예: event_sender로 보내기, 델파이 쪽 실제 명령 호출 등).

D-pad는 화면 고정 위치가 아니라 **손을 따라다닌다** — 기준점은 손목 하나가
아니라 손목+4개 손가락 MCP(5·9·13·17) 평균, 즉 손바닥 쪽으로 옮긴 중심점이다
(순수 손목 기준은 아래 방향으로 손끝이 못 내려가 잘 안 잡히는 문제가 있었다 —
2026-08-04 실기 피드백). 손바닥 폭(검지-새끼 MCP 거리)을 자(尺) 삼아 크기도
같이 커지고 작아진다 (카메라와의 거리가 달라져도 손 대비 존 크기 체감은 비슷
하게 — hand_select.py의 손 실측 자와 같은 취지). 손이 안 보이면 기준점이
없으니 D-pad 자체를 그리지 않는다.

판정도 "버튼 원 안에 정확히 들어와야 함"이 아니라 **중심점 대비 손끝의 각도**로
본다 — 손가락 길이가 사람마다 달라도 그 방향을 향하기만 하면 인식된다(거리는
안 봄). 중심점과 아주 가까울 때만 예외로 CENTER로 본다.

사용법 (프로젝트 루트에서):
    py scripts\\dpad_ui_demo.py [--device N] [--dwell-sec 0.55]
종료: q/ESC
"""
import argparse
import math
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)   # main.py와 동일한 이유 — 어느 작업 디렉터리에서 띄우든 config의
                     # 모델·로그 상대 경로(models/weights/... 등)가 항상 성립하게

import cv2
import numpy as np

from src.capture.camera_stream import CameraStream
from src.inference.hand_tracker import HandTracker
from src.inference.preprocessor import Preprocessor
from src.postprocess.point_filter import PointFilter
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
WINDOW_NAME = "gesture_kiosk dpad ui demo"

INDEX_FINGERTIP_LMK = 8   # MediaPipe 21점 중 검지 끝 (hand_tracker.py 독스트링)
# 기준점 = 손목(0) 단독이 아니라 손목+4개 손가락 MCP(5·9·13·17) 평균 — 손목보다
# 손바닥 쪽으로 옮겨서, 손목 기준일 때 손끝이 내려가기 어려워 잘 안 잡히던
# 아래 방향도 자연스럽게 닿게 한다 (2026-08-04 사용자 실기 피드백)
PALM_CENTER_LMK_IDXS = [0, 5, 9, 13, 17]
PALM_SCALE_LMK_A = 5      # 검지 MCP
PALM_SCALE_LMK_B = 17     # 새끼 MCP — 이 둘 사이 거리를 손바닥 폭(거리 자)으로 쓴다

# 손끝·중심점 One Euro 필터 — config gestures.swipe.point_filter의 실기 튜닝값을 그대로 재사용
FILTER_MIN_CUTOFF_HZ = 1.5
FILTER_BETA = 1.0
FILTER_D_CUTOFF_HZ = 1.0
SCALE_ALPHA = 0.15   # 손바닥 폭 EMA 평활 — hand_select.body_scale.alpha(0.1)와 같은 취지
AXIS_DOMINANCE = 1.2  # 방향 판정 — 한 축이 이만큼 우세해야 확정, 대각선 애매구간은
                      # 직전 방향 유지 (gesture_filter.py의 swipe axis_dominance와 같은 개념)

DWELL_SEC_DEFAULT = 0.55   # 버튼 확정까지 머물러야 하는 시간
FLASH_SEC = 0.30           # 확정 직후 버튼이 밝게 반짝이는 시간

# 레이아웃 크기 = 손바닥 폭(palm_scale_px)의 배수 — 손이 카메라에 가깝든 멀든
# "내 손 크기 대비"는 같아서 존을 누르는 손끝 체감이 거리와 무관해진다.
# (offset > petal_r + center_r 라야 버튼끼리·버튼-중앙 사이에 간격이 보인다)
OFFSET_PALM_MULT = 1.55
PETAL_RADIUS_PALM_MULT = 0.72
CENTER_RADIUS_PALM_MULT = 0.62

COLOR_BG = (54, 46, 43)             # 버튼 배경(평상시) — 어두운 남색 계열
COLOR_BG_HOVER = (90, 76, 50)       # 손끝이 올라와 있는 동안
COLOR_FLASH = (255, 255, 255)       # 확정 순간 반짝임
COLOR_BORDER = (24, 21, 19)
COLOR_ACCENT = (61, 163, 232)       # 화살표·아이콘 (BGR) — 참고 이미지의 amber 톤
COLOR_PROGRESS = (255, 221, 156)    # dwell 진행 링
COLOR_FINGERTIP = (80, 220, 120)
COLOR_TEXT = (255, 255, 255)
PLATE_ALPHA = 0.55                  # 버튼들 뒤 받침판 불투명도

logger = get_logger("scripts")


def draw_rounded_rect(img, pt1, pt2, radius, color):
    """모서리가 둥근 채워진 사각형 — cv2엔 없어 사각형 2개 + 모서리 원 4개로 근사."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    return img


def draw_chevron(img, cx, cy, direction, size, color, thickness):
    """방향 화살표(꺾쇠) — up/down/left/right."""
    if direction == "up":
        pts = [(cx - size, cy + size * 0.5), (cx, cy - size * 0.5), (cx + size, cy + size * 0.5)]
    elif direction == "down":
        pts = [(cx - size, cy - size * 0.5), (cx, cy + size * 0.5), (cx + size, cy - size * 0.5)]
    elif direction == "left":
        pts = [(cx + size * 0.5, cy - size), (cx - size * 0.5, cy), (cx + size * 0.5, cy + size)]
    else:   # right
        pts = [(cx - size * 0.5, cy - size), (cx + size * 0.5, cy), (cx - size * 0.5, cy + size)]
    points = np.array([(int(x), int(y)) for x, y in pts])
    cv2.polylines(img, [points], False, color, thickness, cv2.LINE_AA)


def draw_pause_icon(img, cx, cy, size, color, thickness):
    """중앙 버튼 아이콘 — 참고 이미지의 일시정지(││) 표시."""
    gap = size * 0.5
    for dx in (-gap, gap):
        x = int(cx + dx)
        cv2.line(img, (x, int(cy - size)), (x, int(cy + size)), color, thickness, cv2.LINE_AA)


def compute_zones(center_px, scale_px):
    """D-pad 5개 존(up/down/left/right/center) -> {name: (cx_px, cy_px, radius_px)}.

    center_px: D-pad가 따라다닐 기준점(손목+손바닥 MCP 평균, 화면 픽셀 좌표).
    scale_px: 손바닥 폭(픽셀) — 카메라 거리 무관하게 손 대비 같은 크기로 보이게 하는 자.
    """
    cx_px, cy_px = int(center_px[0]), int(center_px[1])
    offset_px = int(scale_px * OFFSET_PALM_MULT)
    petal_r_px = int(scale_px * PETAL_RADIUS_PALM_MULT)
    center_r_px = int(scale_px * CENTER_RADIUS_PALM_MULT)
    zones = {
        "up": (cx_px, cy_px - offset_px, petal_r_px),
        "down": (cx_px, cy_px + offset_px, petal_r_px),
        "left": (cx_px - offset_px, cy_px, petal_r_px),
        "right": (cx_px + offset_px, cy_px, petal_r_px),
        "center": (cx_px, cy_px, center_r_px),
    }
    return zones, offset_px, petal_r_px


def classify_direction(dx_px, dy_px):
    """중심점(팔목·손바닥) -> 손끝 벡터 -> "up"/"down"/"left"/"right" | None.

    거리가 아니라 **각도만** 본다 — 손가락이 길든 짧든(사람마다 다른 손 크기)
    버튼 원 안에 정확히 들어오지 않아도, 그 방향을 향하기만 하면 인식되게
    하기 위함(2026-08-04 사용자 요청). 대각선처럼 두 축이 비슷하면(축 우세
    미달) None을 돌려줘 호출부가 직전 방향을 유지하게 한다 — 그래야 대각선
    근처에서 두 방향 사이를 매 프레임 오가며 dwell이 계속 리셋되지 않는다.
    """
    abs_dx_px, abs_dy_px = abs(dx_px), abs(dy_px)
    if abs_dx_px >= abs_dy_px * AXIS_DOMINANCE:
        return "right" if dx_px > 0 else "left"
    if abs_dy_px >= abs_dx_px * AXIS_DOMINANCE:
        return "down" if dy_px > 0 else "up"
    return None


def draw_dpad(frame, zones, offset_px, petal_r_px, hover_zone, hover_progress,
              flash_zone, flash_progress):
    """D-pad 전체(받침판 + 5개 버튼 + 아이콘 + 진행 링 + 반짝임)를 frame에 그린다."""
    cx_px, cy_px, _ = zones["center"]
    pad_px = int(petal_r_px * 0.4)
    half_px = offset_px + petal_r_px + pad_px
    plate_radius_px = int(petal_r_px * 0.55)

    overlay = frame.copy()
    draw_rounded_rect(overlay, (cx_px - half_px, cy_px - half_px),
                       (cx_px + half_px, cy_px + half_px), plate_radius_px, COLOR_BG)
    cv2.addWeighted(overlay, PLATE_ALPHA, frame, 1 - PLATE_ALPHA, 0, frame)

    for name in ("up", "down", "left", "right", "center"):
        zx, zy, zr = zones[name]
        color = COLOR_BG_HOVER if name == hover_zone else COLOR_BG
        if name == flash_zone and flash_progress > 0:
            color = tuple(int(base * (1 - flash_progress) + flash * flash_progress)
                          for base, flash in zip(color, COLOR_FLASH))
        cv2.circle(frame, (zx, zy), zr, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (zx, zy), zr, COLOR_BORDER, 2, cv2.LINE_AA)
        icon_thickness = max(3, zr // 8)
        if name == "center":
            draw_pause_icon(frame, zx, zy, int(zr * 0.42), COLOR_ACCENT, icon_thickness)
        else:
            draw_chevron(frame, zx, zy, name, int(zr * 0.5), COLOR_ACCENT, icon_thickness)
        if name == hover_zone and hover_progress > 0:
            end_angle_deg = 360.0 * hover_progress
            cv2.ellipse(frame, (zx, zy), (zr + 6, zr + 6), -90, 0, end_angle_deg,
                        COLOR_PROGRESS, 4, cv2.LINE_AA)
    return frame


def build_zone_functions():
    """존 이름 -> 실행 함수. 지금은 자리표시자(콘솔 출력)만 한다 —
    실제 기능이 정해지면 각 함수 본문을 원하는 동작으로 바꿔 끼우면 된다
    (예: event_sender.send(...)로 기존 이벤트 파이프에 합류시키기).
    콘솔 출력은 영문 ASCII로 고정한다 — cmd 코드페이지에 따라 한글이 깨질 수
    있다 (event_sender.py가 stdout을 ascii로 encode하는 것과 같은 이유)."""
    state = {"is_playing": False}

    def on_up():
        print("[D-PAD] UP fired -> plug your action into on_up()")
        return "UP"

    def on_down():
        print("[D-PAD] DOWN fired -> plug your action into on_down()")
        return "DOWN"

    def on_left():
        print("[D-PAD] LEFT fired -> plug your action into on_left()")
        return "LEFT"

    def on_right():
        print("[D-PAD] RIGHT fired -> plug your action into on_right()")
        return "RIGHT"

    def on_center():
        state["is_playing"] = not state["is_playing"]
        label = "PLAY" if state["is_playing"] else "PAUSE"
        print(f"[D-PAD] CENTER fired -> {label}")
        return label

    return {"up": on_up, "down": on_down, "left": on_left, "right": on_right, "center": on_center}


def main():
    parser = argparse.ArgumentParser(
        description="카메라 화면 위 가상 D-pad — 손끝으로 방향 존을 눌러 함수를 실행하는 데모")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=int, default=None, help="카메라 장치 번호 (기본: config device_id)")
    parser.add_argument("--dwell-sec", type=float, default=DWELL_SEC_DEFAULT,
                        help="존에 머물러야 확정되는 시간(초)")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)
    if args.device is not None:
        config["camera"]["device_id"] = args.device
        auto_select = config["camera"].get("auto_select")
        if auto_select is not None:
            config["camera"]["auto_select"] = {**auto_select, "enabled": False}

    preprocessor = Preprocessor(config)
    hand_tracker = HandTracker(config)
    camera = CameraStream(config).start()

    fingertip_filter = PointFilter(FILTER_MIN_CUTOFF_HZ, FILTER_BETA, FILTER_D_CUTOFF_HZ)
    center_filter = PointFilter(FILTER_MIN_CUTOFF_HZ, FILTER_BETA, FILTER_D_CUTOFF_HZ)
    palm_scale_px = None   # 손바닥 폭 EMA — None이면 아직 손을 못 봄(재등장 시 새로 시작)
    last_direction = None  # 직전 확정 방향(대각선 애매구간 유지용) — 손 재등장 시 초기화
    had_hand_prev = False
    hover_zone, hover_start_sec, armed = None, 0.0, True
    flash_zone, flash_start_sec = None, 0.0
    last_action_label = ""
    zone_funcs = build_zone_functions()

    logger.warning("D-pad UI demo started - it now follows your palm. Point toward a "
                   "direction for %.2fs to fire it (no need to touch the icon). Quit: q or ESC",
                   args.dwell_sec)

    try:
        while True:
            frame = preprocessor.preprocess_frame(camera.capture_frame(), apply_crop=False)
            now_sec = time.monotonic()

            hands = hand_tracker.infer(frame)
            center_px, fingertip_px = None, None
            if hands:
                # 데모 단순화: 신뢰도가 가장 높은 손 하나만 본다 (본 엔진의 단일 손
                # 추적·앵커 게이트는 main.py 쪽 hand_select.py가 담당 — 여기선 재사용 안 함)
                hand = max(hands, key=lambda h: h.conf)
                if not had_hand_prev:
                    fingertip_filter.reset()   # 재등장 — 이전 궤적과 섞이지 않게
                    center_filter.reset()
                    palm_scale_px = None
                    last_direction = None
                had_hand_prev = True

                raw_tip = (float(hand.landmarks[INDEX_FINGERTIP_LMK][0]),
                          float(hand.landmarks[INDEX_FINGERTIP_LMK][1]))
                fingertip_px = fingertip_filter.filter(raw_tip, now_sec)

                palm_pts = hand.landmarks[PALM_CENTER_LMK_IDXS, :2]   # (5, 2) — 손목+4 MCP
                raw_center = (float(palm_pts[:, 0].mean()), float(palm_pts[:, 1].mean()))
                center_px = center_filter.filter(raw_center, now_sec)

                raw_scale_px = math.hypot(
                    hand.landmarks[PALM_SCALE_LMK_A][0] - hand.landmarks[PALM_SCALE_LMK_B][0],
                    hand.landmarks[PALM_SCALE_LMK_A][1] - hand.landmarks[PALM_SCALE_LMK_B][1],
                )
                palm_scale_px = (raw_scale_px if palm_scale_px is None else
                                 palm_scale_px * (1 - SCALE_ALPHA) + raw_scale_px * SCALE_ALPHA)
            else:
                had_hand_prev = False

            zones = offset_px = petal_r_px = None
            zone = None
            if center_px is not None:
                zones, offset_px, petal_r_px = compute_zones(center_px, palm_scale_px)
                dx_px = fingertip_px[0] - center_px[0]
                dy_px = fingertip_px[1] - center_px[1]
                center_r_px = zones["center"][2]
                if math.hypot(dx_px, dy_px) < center_r_px:
                    zone = "center"
                else:
                    direction = classify_direction(dx_px, dy_px)
                    if direction is not None:
                        last_direction = direction
                    zone = last_direction   # 대각선 애매구간이면 직전 방향 유지

            if zone != hover_zone:
                hover_zone, hover_start_sec, armed = zone, now_sec, True
            hover_progress = 0.0
            if hover_zone is not None:
                hover_progress = min(1.0, (now_sec - hover_start_sec) / args.dwell_sec)
                if armed and hover_progress >= 1.0:
                    armed = False   # 재확정하려면 존을 벗어났다 다시 들어와야 함
                    flash_zone, flash_start_sec = hover_zone, now_sec
                    last_action_label = zone_funcs[hover_zone]()

            flash_progress = 0.0
            if flash_zone is not None:
                flash_elapsed_sec = now_sec - flash_start_sec
                if flash_elapsed_sec >= FLASH_SEC:
                    flash_zone = None
                else:
                    flash_progress = 1.0 - flash_elapsed_sec / FLASH_SEC

            if zones is not None:
                frame = draw_dpad(frame, zones, offset_px, petal_r_px, hover_zone, hover_progress,
                                  flash_zone, flash_progress)
            if fingertip_px is not None:
                fx_px, fy_px = int(fingertip_px[0]), int(fingertip_px[1])
                cv2.circle(frame, (fx_px, fy_px), 10, COLOR_FINGERTIP, 2, cv2.LINE_AA)
                cv2.circle(frame, (fx_px, fy_px), 2, COLOR_FINGERTIP, -1, cv2.LINE_AA)

            cv2.putText(frame, "D-PAD UI DEMO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, COLOR_TEXT, 2, cv2.LINE_AA)
            if last_action_label:
                cv2.putText(frame, f"FIRED: {last_action_label}", (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ACCENT, 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
