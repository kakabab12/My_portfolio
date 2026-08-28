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

D-pad는 화면 비율에 맞춘 **고정 위치·작은 크기**로 화면 상단 중앙에 그린다
(2026-08-04 사용자 결정 — 화면을 거의 채우는 중앙 대형판은
dpad_ui_demo_fixed_center.py에, 손을 따라다니던 판은
dpad_ui_demo_palm_follow.py에, 우상단 구석·CALIB 이전 판은
dpad_ui_demo_corner_cursor.py·dpad_ui_demo_corner_calib.py에 보존 —
"왼쪽 조금 작게" -> "오른쪽 위 구석" -> "상단 중앙" 순으로 위치가 바뀌었다).

화면 **좌우 끝에는 세로로 긴 PREV·NEXT 버튼**을 D-pad와 별개로 고정한다
(2026-08-04 사용자 요청 — 이전/다음 화면 전환용). D-pad의 각도 판정과 달리
이 둘은 커서가 사각형 안에 들어왔는지로 판정하고, D-pad보다 항상 먼저
검사한다(compute_edge_buttons·point_in_rect, main()의 존 판정 순서 참고).

D-pad 자체는 고정이지만 **커서는 손끝을 따라 D-pad 주변에서 움직인다** —
손끝의 화면 절대 좌표가 아니라 **화면 중앙(평상시 손이 쉬는 자리로 가정)
대비 손끝의 오프셋**을 D-pad 중심에 그대로 옮겨 찍는다. 그래서 D-pad가
화면 어느 구석에 있든, 사용자는 몸 앞 편한 위치에서 손을 움직이기만 하면
되고 커서는 항상 D-pad 근처에 보인다(카메라 앞 어디서든 조작 가능 — 실제로
D-pad를 향해 손을 뻗을 필요 없음, 2026-08-04 사용자 요청).

판정은 "버튼 원 안에 정확히 들어와야 함"이 아니라 **D-pad 중심 대비 커서의
각도**로 본다 — 그 방향을 향하기만 하면 인식된다(거리는 안 봄). 중심과 아주
가까울 때만 예외로 CENTER로 본다.

**재정렬(CALIB)**: 손이 쉬는 위치가 사람마다 달라 화면 중앙이라는 가정이
안 맞을 수 있다 — D-pad 아래 작은 원(CALIB, 조준선 아이콘)에 커서를 dwell
시키거나 키보드 c를 누르면, **그 순간의 손끝 위치**가 새 기준점이 되어
커서가 즉시 D-pad 정중앙으로 옮겨진다(2026-08-04 사용자 요청). 재정렬은
손이 안 보여도 유지된다 — 손을 잠깐 내렸다 다시 들었다고 풀리지 않는다.
프로그램을 막 시작해 손이 **처음** 잡히는 순간에도 같은 원리로 자동 1회
재정렬돼(화면 중앙 가정 없이) 바로 D-pad 중앙에서 시작한다 — 이후로는
수동 재정렬만 갱신한다.

**민감도(CURSOR_SENSITIVITY)**: 손끝이 기준점에서 벗어난 픽셀을 이 배수만큼
키워 커서 오프셋으로 쓴다 — 손가락을 조금만 움직여도 커서가 크게 움직이게
하려면 이 값을 올린다(2026-08-04 사용자 요청으로 대폭 상향).

CALIB의 dwell(CALIB_DWELL_SEC)은 나머지 존(--dwell-sec)과 별도로 3초로
길게 잡는다 — 스치기만 해도 기준점이 바뀌면 안 되니 확실히 오래 머물러야
확정된다(2026-08-04 사용자 요청). 그동안 CALIB 버튼 아래에 남은 시간을
보여주는 안내 문구("HOLD TO RECALIBRATE N.Ns")가 뜬다.

사용법 (프로젝트 루트에서):
    py scripts\\dpad_ui_demo.py [--device N] [--dwell-sec 0.55] [--fullscreen]
                              [--cx-ratio 0.50] [--cy-ratio 0.30]
종료: q/ESC (전체화면 중에도 동일 — 창에 테두리·닫기 버튼이 없어져도 키보드는 그대로 먹는다)
재정렬: CALIB 존에 3초 dwell 또는 키보드 c
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

# 손끝 One Euro 필터 — config gestures.swipe.point_filter의 실기 튜닝값을 그대로 재사용
FILTER_MIN_CUTOFF_HZ = 1.5
FILTER_BETA = 1.0
FILTER_D_CUTOFF_HZ = 1.0
AXIS_DOMINANCE = 1.2  # 방향 판정 — 한 축이 이만큼 우세해야 확정, 대각선 애매구간은
                      # 직전 방향 유지 (gesture_filter.py의 swipe axis_dominance와 같은 개념)
CURSOR_SENSITIVITY = 3.0   # 손끝이 기준점(재정렬 전엔 화면 중앙)에서 벗어난
                           # 픽셀 오프셋 -> 커서 오프셋 배율. 1.0=그대로 옮김.
                           # 2026-08-04 사용자 요청으로 대폭 상향(1.0->3.0) —
                           # 손가락을 조금만 움직여도 존까지 닿게. 너무
                           # 예민해 떨림에 오발되면 ↓ (2.0 정도까지)

DWELL_SEC_DEFAULT = 0.55   # 버튼 확정까지 머물러야 하는 시간
FLASH_SEC = 0.30           # 확정 직후 버튼이 밝게 반짝이는 시간
CALIB_DWELL_SEC = 3.0      # 재정렬은 훨씬 길게 대기 — 실수로 잠깐 스쳐서
                           # 기준점이 엉뚱하게 바뀌면 안 되니(2026-08-04 사용자
                           # 요청). 델파이 프로덕션의 recenter_dwell(2.5초)이
                           # dwell_click(1.5초)보다 긴 것과 같은 이유

# 가장자리 세로 버튼(PREV/NEXT) — D-pad와 별개로 화면 좌/우 끝에 고정 (2026-08-04 신설)
EDGE_BUTTON_WIDTH_RATIO = 0.07     # 프레임 폭 비율
EDGE_BUTTON_HEIGHT_RATIO = 0.42    # 프레임 높이 비율 — 세로로 긴 모양
EDGE_BUTTON_MARGIN_RATIO = 0.015   # 화면 가장자리와의 여백

# 레이아웃 크기 = 프레임 짧은 변(scale_px = min(w_px, h_px))의 비율 — 구석에
# 고정할 거라 작게(2026-08-04 사용자 요청: 화면 중앙 대형판 -> 구석에 작게).
# offset+petal_r+받침판 여유가 구석 위치(cx_ratio·cy_ratio)에서 화면 밖으로
# 안 잘리게 맞춰야 한다 (offset > petal_r + center_r 라야 버튼끼리·버튼-중앙
# 사이에 간격이 보인다)
OFFSET_SCREEN_RATIO = 0.14
PETAL_RADIUS_SCREEN_RATIO = 0.065
CENTER_RADIUS_SCREEN_RATIO = 0.055

# 재정렬(CALIB) 버튼 — D-pad 아래에 따로 떠 있는 작은 원 (2026-08-04 신설)
CALIB_RADIUS_SCREEN_RATIO = 0.045
CALIB_GAP_SCREEN_RATIO = 0.035   # D-pad 아래쪽 끝과 CALIB 버튼 사이 간격

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


def draw_calib_icon(img, cx, cy, size, color, thickness):
    """재정렬(CALIB) 버튼 아이콘 — 조준선(원 + 십자)."""
    cv2.circle(img, (cx, cy), size, color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)


def compute_zones(center_px, scale_px):
    """D-pad 6개 존(up/down/left/right/center/calib) -> {name: (cx_px, cy_px, radius_px)}.

    center_px: D-pad 중심(화면 픽셀 좌표, 기본은 프레임 정중앙).
    scale_px: 크기 기준 자(프레임의 짧은 변 픽셀 수) — 화면 비율에 맞춰 큼직하게 그린다.
    calib은 방향 4개·center와 달리 D-pad 아래에 따로 떠 있는 재정렬 버튼 —
    각도가 아니라 거리로 판정한다(호출부에서 별도 처리, classify_direction 미사용).
    """
    cx_px, cy_px = int(center_px[0]), int(center_px[1])
    offset_px = int(scale_px * OFFSET_SCREEN_RATIO)
    petal_r_px = int(scale_px * PETAL_RADIUS_SCREEN_RATIO)
    center_r_px = int(scale_px * CENTER_RADIUS_SCREEN_RATIO)
    calib_r_px = int(scale_px * CALIB_RADIUS_SCREEN_RATIO)
    calib_gap_px = int(scale_px * CALIB_GAP_SCREEN_RATIO)
    calib_cy_px = cy_px + offset_px + petal_r_px + calib_gap_px + calib_r_px
    zones = {
        "up": (cx_px, cy_px - offset_px, petal_r_px),
        "down": (cx_px, cy_px + offset_px, petal_r_px),
        "left": (cx_px - offset_px, cy_px, petal_r_px),
        "right": (cx_px + offset_px, cy_px, petal_r_px),
        "center": (cx_px, cy_px, center_r_px),
        "calib": (cx_px, calib_cy_px, calib_r_px),
    }
    return zones, offset_px, petal_r_px


def compute_edge_buttons(w_px, h_px):
    """화면 좌/우 끝의 세로로 긴 PREV·NEXT 버튼 -> {"prev": rect, "next": rect}.

    rect = (x1_px, y1_px, x2_px, y2_px). D-pad와 무관하게 프레임 크기에서
    바로 계산 — D-pad가 중앙이든 구석이든 화면 양끝에 그대로 고정된다.
    """
    btn_w_px = int(w_px * EDGE_BUTTON_WIDTH_RATIO)
    btn_h_px = int(h_px * EDGE_BUTTON_HEIGHT_RATIO)
    margin_px = int(w_px * EDGE_BUTTON_MARGIN_RATIO)
    y1_px = h_px // 2 - btn_h_px // 2
    y2_px = y1_px + btn_h_px
    return {
        "prev": (margin_px, y1_px, margin_px + btn_w_px, y2_px),
        "next": (w_px - margin_px - btn_w_px, y1_px, w_px - margin_px, y2_px),
    }


def point_in_rect(point_px, rect):
    """점이 rect=(x1,y1,x2,y2) 안에 있는지."""
    x1_px, y1_px, x2_px, y2_px = rect
    return x1_px <= point_px[0] <= x2_px and y1_px <= point_px[1] <= y2_px


def draw_edge_button(frame, rect, direction, is_hover, hover_progress, is_flash, flash_progress):
    """세로로 긴 가장자리 버튼(PREV/NEXT) — 채움 + 아래->위 진행 바 + 화살표."""
    x1_px, y1_px, x2_px, y2_px = rect
    color = COLOR_BG_HOVER if is_hover else COLOR_BG
    if is_flash and flash_progress > 0:
        color = tuple(int(base * (1 - flash_progress) + flash * flash_progress)
                      for base, flash in zip(color, COLOR_FLASH))
    radius_px = int((x2_px - x1_px) * 0.35)
    draw_rounded_rect(frame, (x1_px, y1_px), (x2_px, y2_px), radius_px, color)
    if is_hover and hover_progress > 0:
        fill_top_y_px = int(y2_px - (y2_px - y1_px) * hover_progress)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1_px, fill_top_y_px), (x2_px, y2_px), COLOR_PROGRESS, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    icon_size_px = int((x2_px - x1_px) * 0.35)
    icon_thickness = max(3, (x2_px - x1_px) // 6)
    cx_px, cy_px = (x1_px + x2_px) // 2, (y1_px + y2_px) // 2
    draw_chevron(frame, cx_px, cy_px, direction, icon_size_px, COLOR_ACCENT, icon_thickness)
    return frame


def classify_direction(dx_px, dy_px):
    """D-pad 중심 -> 손끝 벡터 -> "up"/"down"/"left"/"right" | None.

    거리가 아니라 **각도만** 본다 — 팔 길이가 짧아 버튼 원까지 못 닿아도
    그 방향을 향하기만 하면 인식되게 하기 위함(2026-08-04 사용자 요청).
    대각선처럼 두 축이 비슷하면(축 우세
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

    for name in ("up", "down", "left", "right", "center", "calib"):
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
        elif name == "calib":
            draw_calib_icon(frame, zx, zy, int(zr * 0.5), COLOR_ACCENT, icon_thickness)
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

    def on_prev():
        print("[D-PAD] PREV fired -> plug your action into on_prev()")
        return "PREV"

    def on_next():
        print("[D-PAD] NEXT fired -> plug your action into on_next()")
        return "NEXT"

    return {"up": on_up, "down": on_down, "left": on_left, "right": on_right, "center": on_center,
            "prev": on_prev, "next": on_next}


def main():
    parser = argparse.ArgumentParser(
        description="카메라 화면 위 가상 D-pad — 손끝으로 방향 존을 눌러 함수를 실행하는 데모")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=int, default=None, help="카메라 장치 번호 (기본: config device_id)")
    parser.add_argument("--dwell-sec", type=float, default=DWELL_SEC_DEFAULT,
                        help="존에 머물러야 확정되는 시간(초)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="창을 전체화면으로 띄운다 (키오스크 시연용 — 종료는 q/ESC 그대로)")
    parser.add_argument("--cx-ratio", type=float, default=0.50, help="D-pad 중심 X (화면 폭 비율) — 기본 정중앙")
    parser.add_argument("--cy-ratio", type=float, default=0.30, help="D-pad 중심 Y (화면 높이 비율) — 기본 상단")
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
    last_direction = None  # 직전 확정 방향(대각선 애매구간 유지용) — 손 재등장 시 초기화
    neutral_px = None      # 재정렬 기준점 — None이면 화면 중앙 사용(자동 보정 전 폴백).
                           # 손이 없어져도 유지(사용자가 다시 재정렬하기 전까지 그대로)
    has_auto_calibrated = False   # 프로그램 시작 후 손이 처음 잡히는 그 프레임에
                           # 딱 1회만 자동 재정렬(2026-08-04 사용자 요청 — 화면
                           # 중앙 가정 없이 바로 D-pad 중앙에서 시작). 그 뒤로는
                           # 수동 재정렬(CALIB·키보드 c)만 갱신 — 손을 뗐다 다시
                           # 들 때마다 재보정되면 위 "재정렬 유지" 요청과 충돌한다
    had_hand_prev = False
    hover_zone, hover_start_sec, armed = None, 0.0, True
    flash_zone, flash_start_sec = None, 0.0
    last_action_label = ""
    zone_funcs = build_zone_functions()

    if args.fullscreen:
        # 카메라 프레임 픽셀 좌표는 그대로 두고(존·손끝 판정은 프레임 기준이라
        # 무관) 창 표시만 전체화면으로 — 실제 모니터 해상도로 OS가 확대해서 보여준다
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logger.warning("D-pad UI demo started - top-center D-pad + PREV/NEXT edge buttons. "
                   "Point toward a direction for %.2fs to fire it (CALIB needs %.1fs). "
                   "Quit: q or ESC", args.dwell_sec, CALIB_DWELL_SEC)

    try:
        while True:
            frame = preprocessor.preprocess_frame(camera.capture_frame(), apply_crop=False)
            now_sec = time.monotonic()
            h_px, w_px = frame.shape[:2]

            hands = hand_tracker.infer(frame)
            fingertip_px = None
            if hands:
                # 데모 단순화: 신뢰도가 가장 높은 손 하나만 본다 (본 엔진의 단일 손
                # 추적·앵커 게이트는 main.py 쪽 hand_select.py가 담당 — 여기선 재사용 안 함)
                hand = max(hands, key=lambda h: h.conf)
                if not had_hand_prev:
                    fingertip_filter.reset()   # 재등장 — 이전 궤적과 섞이지 않게
                    last_direction = None
                had_hand_prev = True

                raw_tip = (float(hand.landmarks[INDEX_FINGERTIP_LMK][0]),
                          float(hand.landmarks[INDEX_FINGERTIP_LMK][1]))
                fingertip_px = fingertip_filter.filter(raw_tip, now_sec)
                if not has_auto_calibrated:
                    neutral_px = fingertip_px   # 첫 등장 손끝 = 시작 기준점 -> 커서가 D-pad 중앙에서 출발
                    has_auto_calibrated = True
            else:
                had_hand_prev = False

            # D-pad는 화면 비율 고정 위치·고정 크기 — 손 유무와 무관하게 항상 그 자리
            center_px = (args.cx_ratio * w_px, args.cy_ratio * h_px)
            zones, offset_px, petal_r_px = compute_zones(center_px, min(w_px, h_px))
            edge_buttons = compute_edge_buttons(w_px, h_px)   # PREV/NEXT — D-pad와 무관, 화면 양끝 고정

            # 커서 = 손끝의 절대 좌표가 아니라 "기준점(재정렬 전엔 화면 중앙 —
            # 평상시 손 위치로 가정) 대비 오프셋"을 D-pad 중심에 그대로 옮겨
            # 찍는다 — D-pad가 구석에 작게 있어도 몸 앞 편한 위치에서 손을
            # 움직이면 커서가 D-pad 근처에서 그 방향으로 움직인다
            # (2026-08-04 사용자 요청)
            cursor_px = None
            zone = None
            if fingertip_px is not None:
                reference_px = neutral_px if neutral_px is not None else (w_px / 2.0, h_px / 2.0)
                dx_px = (fingertip_px[0] - reference_px[0]) * CURSOR_SENSITIVITY
                dy_px = (fingertip_px[1] - reference_px[1]) * CURSOR_SENSITIVITY
                cursor_px = (center_px[0] + dx_px, center_px[1] + dy_px)

                if point_in_rect(cursor_px, edge_buttons["prev"]):
                    zone = "prev"
                elif point_in_rect(cursor_px, edge_buttons["next"]):
                    zone = "next"
                else:
                    # CALIB은 D-pad 각도 판정과 별개 — 자기 자리(zones["calib"])까지
                    # 커서의 실제 거리로 본다 (호출부 전용, classify_direction 미사용)
                    calib_cx, calib_cy, calib_r_px = zones["calib"]
                    if math.hypot(cursor_px[0] - calib_cx, cursor_px[1] - calib_cy) < calib_r_px:
                        zone = "calib"
                    else:
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
                dwell_sec = CALIB_DWELL_SEC if hover_zone == "calib" else args.dwell_sec
                hover_progress = min(1.0, (now_sec - hover_start_sec) / dwell_sec)
                if armed and hover_progress >= 1.0:
                    armed = False   # 재확정하려면 존을 벗어났다 다시 들어와야 함
                    flash_zone, flash_start_sec = hover_zone, now_sec
                    if hover_zone == "calib":
                        # 재정렬 — 지금 이 순간의 손끝 위치를 새 기준점으로:
                        # 다음 프레임부터 dx_px·dy_px가 0이 되어 커서가 D-pad
                        # 정중앙에 나타난다 (zone_funcs를 안 거치는 이유:
                        # fingertip_px라는 루프 지역 상태가 필요해서)
                        neutral_px = fingertip_px
                        print("[D-PAD] CALIB fired -> cursor recentered")
                        last_action_label = "CALIB"
                    else:
                        last_action_label = zone_funcs[hover_zone]()

            flash_progress = 0.0
            if flash_zone is not None:
                flash_elapsed_sec = now_sec - flash_start_sec
                if flash_elapsed_sec >= FLASH_SEC:
                    flash_zone = None
                else:
                    flash_progress = 1.0 - flash_elapsed_sec / FLASH_SEC

            frame = draw_dpad(frame, zones, offset_px, petal_r_px, hover_zone, hover_progress,
                              flash_zone, flash_progress)
            for edge_name, edge_direction in (("prev", "left"), ("next", "right")):
                frame = draw_edge_button(
                    frame, edge_buttons[edge_name], edge_direction,
                    hover_zone == edge_name, hover_progress if hover_zone == edge_name else 0.0,
                    flash_zone == edge_name, flash_progress if flash_zone == edge_name else 0.0,
                )
            if hover_zone == "calib" and hover_progress > 0:
                # 3초 대기 안내 문구 — CALIB 버튼 바로 아래에 남은 시간 표시
                # (2026-08-04 사용자 요청: 긴 대기 중 "지금 뭐 하는 중인지" 안내)
                calib_cx, calib_cy, calib_r_px = zones["calib"]
                remaining_sec = max(0.0, CALIB_DWELL_SEC - (now_sec - hover_start_sec))
                guide_text = f"HOLD TO RECALIBRATE {remaining_sec:.1f}s"
                text_w_px = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
                text_x_px = max(4, calib_cx - text_w_px // 2)
                text_y_px = calib_cy + calib_r_px + 22
                cv2.putText(frame, guide_text, (text_x_px, text_y_px), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, COLOR_PROGRESS, 2, cv2.LINE_AA)
            if cursor_px is not None:
                cx_px, cy_px = int(cursor_px[0]), int(cursor_px[1])
                cv2.circle(frame, (cx_px, cy_px), 10, COLOR_FINGERTIP, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx_px, cy_px), 2, COLOR_FINGERTIP, -1, cv2.LINE_AA)

            cv2.putText(frame, "D-PAD UI DEMO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, COLOR_TEXT, 2, cv2.LINE_AA)
            if last_action_label:
                cv2.putText(frame, f"FIRED: {last_action_label}", (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ACCENT, 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and fingertip_px is not None:
                neutral_px = fingertip_px   # 키보드 재정렬 — CALIB 존 dwell과 동일 효과
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
