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

**손 모양 계층**: 같은 방향이라도 손 모양(한 손가락/주먹/손바닥)에 따라
다른 함수가 나온다(2026-08-04 사용자 요청) — 본 엔진의 "손 모양이 계층을,
방향이 기능을 정한다"는 원칙과 같은 개념. 판별은 새로 만들지 않고 본 엔진의
src/postprocess/hand_shape.py(classify_hand_shape)를 그대로 재사용하고,
임계값도 config.yaml의 hand_select.hand_shape를 그대로 읽어 실기로 이미
보정된 값을 그대로 물려받는다. 커서 옆에 현재 판별된 모양(FINGER/FIST/
OPEN)이 태그로 뜬다.

D-pad는 화면 비율에 맞춘 **고정 위치·작은 크기**로 화면 상단 중앙에 그린다
(2026-08-04 사용자 결정 — 화면을 거의 채우는 중앙 대형판은
dpad_ui_demo_fixed_center.py에, 손을 따라다니던 판은
dpad_ui_demo_palm_follow.py에, 우상단 구석·CALIB 이전 판은
dpad_ui_demo_corner_cursor.py·dpad_ui_demo_corner_calib.py에 보존 —
"왼쪽 조금 작게" -> "오른쪽 위 구석" -> "상단 중앙" 순으로 위치가 바뀌었다).

(가장자리 PREV·NEXT 세로 버튼은 한 번 만들어봤다가 2026-08-04 폐기 —
dpad_ui_demo_with_prevnext.py에 보존. 필요해지면 그 파일에서 되살릴 것)

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
안 맞을 수 있다. 흐름은 2단계다(2026-08-04 사용자 정정 — 처음엔 CALIB에서
3초를 그대로 버텨야 하는 판이었는데, 그건 손을 편한 자리로 옮길 틈이 없어
잘못됐다는 지적):
  1) D-pad 아래 작은 원(CALIB, 조준선 아이콘)을 다른 버튼과 같은 dwell로
     "누른다"(키보드 c도 동일) — 여기까지는 즉시 반응.
  2) 누른 순간부터 화면 중앙에 큰 안내 문구가 뜨고 **3초 뒤**, 그 시점의
     손끝 위치가 새 기준점이 된다. 그 3초 동안 손을 원하는 편한 자리로
     옮겨 놓으면 된다 — CALIB 버튼 위에 계속 머물러 있을 필요 없음.
재정렬은 손이 안 보여도 유지된다 — 손을 잠깐 내렸다 다시 들었다고 풀리지
않는다. 프로그램을 막 시작해 손이 **처음** 잡히는 순간에도(즉시, 3초 대기
없이) 자동 1회 재정렬돼 바로 D-pad 중앙에서 시작한다 — 이후로는 위 수동
2단계 재정렬만 갱신한다.

**민감도(CURSOR_SENSITIVITY)**: 손끝이 기준점에서 벗어난 픽셀을 이 배수만큼
키워 커서 오프셋으로 쓴다 — 손가락을 조금만 움직여도 커서가 크게 움직이게
하려면 이 값을 올린다(2026-08-04 사용자 요청으로 대폭 상향).

사용법 (프로젝트 루트에서):
    py scripts\\dpad_ui_demo.py [--device N] [--dwell-sec 0.55] [--fullscreen]
                              [--cx-ratio 0.50] [--cy-ratio 0.30]
종료: q/ESC (전체화면 중에도 동일 — 창에 테두리·닫기 버튼이 없어져도 키보드는 그대로 먹는다)
재정렬: CALIB 존 dwell 또는 키보드 c -> 안내 문구 -> 3초 뒤 확정
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
from src.postprocess.hand_shape import classify_hand_shape
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
CALIB_DELAY_SEC = 3.0      # CALIB "누르기"(dwell, 다른 버튼과 동일)는 즉시
                           # 반응하고, 실제 재정렬은 그 뒤로 이 시간만큼 더
                           # 늦춘다 — 손을 편한 자리로 옮길 시간을 준 다음
                           # 그 위치를 기준점으로 쓰기 위함 (2026-08-04 사용자
                           # 정정: 처음엔 이 시간만큼 CALIB 위에서 안 움직이고
                           # 버텨야 하는 판이었는데, 그럼 손을 편한 자리로
                           # 옮길 틈이 없다는 지적)

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

SHAPE_TAG = {"fist": "FIST", "finger": "FINGER", "open": "OPEN"}   # 커서 옆 표시용
                                    # — classify_hand_shape 반환값(hand_shape.py)과 1:1

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
    """존 이름 -> 실행 함수(shape_label: str 인자 하나). 지금은 자리표시자
    (콘솔 출력)만 한다 — 실제 기능이 정해지면 각 함수 본문을 원하는 동작으로
    바꿔 끼우면 된다(예: event_sender.send(...)로 기존 이벤트 파이프에
    합류시키기). shape_label은 "FINGER"/"FIST"/"OPEN"/"UNKNOWN" 중 하나 —
    같은 방향이라도 손 모양에 따라 다르게 반응하고 싶으면 이 값으로 분기하면
    된다(2026-08-04 사용자 요청).
    콘솔 출력은 영문 ASCII로 고정한다 — cmd 코드페이지에 따라 한글이 깨질 수
    있다 (event_sender.py가 stdout을 ascii로 encode하는 것과 같은 이유)."""
    state = {"is_playing": False}

    def on_up(shape_label):
        print(f"[D-PAD] UP+{shape_label} fired -> plug your action into on_up()")
        return f"UP+{shape_label}"

    def on_down(shape_label):
        print(f"[D-PAD] DOWN+{shape_label} fired -> plug your action into on_down()")
        return f"DOWN+{shape_label}"

    def on_left(shape_label):
        print(f"[D-PAD] LEFT+{shape_label} fired -> plug your action into on_left()")
        return f"LEFT+{shape_label}"

    def on_right(shape_label):
        print(f"[D-PAD] RIGHT+{shape_label} fired -> plug your action into on_right()")
        return f"RIGHT+{shape_label}"

    def on_center(shape_label):
        state["is_playing"] = not state["is_playing"]
        label = "PLAY" if state["is_playing"] else "PAUSE"
        print(f"[D-PAD] CENTER+{shape_label} fired -> {label}")
        return f"{label}+{shape_label}"

    return {"up": on_up, "down": on_down, "left": on_left, "right": on_right, "center": on_center}


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
    shape_cfg = config["hand_select"]["hand_shape"]   # 본 엔진과 같은 실기 보정 임계값 재사용
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
    calib_pending_until_sec = None   # None이면 대기 없음. CALIB을 누르면 이
                           # 시각(now_sec + CALIB_DELAY_SEC)까지 대기했다가
                           # 그때의 fingertip_px로 실제 재정렬한다
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

    logger.warning("D-pad UI demo started - top-center D-pad. Point toward a direction "
                   "for %.2fs to fire it. CALIB then waits %.1fs before recentering. "
                   "Quit: q or ESC", args.dwell_sec, CALIB_DELAY_SEC)

    try:
        while True:
            frame = preprocessor.preprocess_frame(camera.capture_frame(), apply_crop=False)
            now_sec = time.monotonic()
            h_px, w_px = frame.shape[:2]

            hands = hand_tracker.infer(frame)
            fingertip_px = None
            current_shape = None   # "finger"/"fist"/"open" | None(불명 — hand_shape.py 규칙)
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

                # world_landmarks 권장(화면 좌표 z는 노이즈가 커서 오판 위험 —
                # hand_shape.py classify_hand_shape 독스트링)
                current_shape = classify_hand_shape(
                    hand.world_landmarks, shape_cfg["extend_ratio"],
                    shape_cfg["min_valid_fingers"], shape_cfg["curl_confirm_ratio"],
                )
            else:
                had_hand_prev = False

            # D-pad는 화면 비율 고정 위치·고정 크기 — 손 유무와 무관하게 항상 그 자리
            center_px = (args.cx_ratio * w_px, args.cy_ratio * h_px)
            zones, offset_px, petal_r_px = compute_zones(center_px, min(w_px, h_px))

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
                hover_progress = min(1.0, (now_sec - hover_start_sec) / args.dwell_sec)
                if armed and hover_progress >= 1.0:
                    armed = False   # 재확정하려면 존을 벗어났다 다시 들어와야 함
                    flash_zone, flash_start_sec = hover_zone, now_sec
                    if hover_zone == "calib":
                        # "누르기"는 다른 버튼과 동일하게 즉시 반응 — 실제
                        # 재정렬은 CALIB_DELAY_SEC 뒤로 미룬다(아래 대기 처리
                        # 블록). 그동안 손을 편한 자리로 옮길 수 있다
                        calib_pending_until_sec = now_sec + CALIB_DELAY_SEC
                        print(f"[D-PAD] CALIB pressed -> recentering in {CALIB_DELAY_SEC:.1f}s")
                        last_action_label = "CALIB"
                    else:
                        shape_label = SHAPE_TAG.get(current_shape, "UNKNOWN")
                        last_action_label = zone_funcs[hover_zone](shape_label)

            # 재정렬 대기 처리 — CALIB을 누른 뒤 hover_zone이 뭐든(손을 이미
            # 옮겼을 수 있으니) 상관없이, 정해진 시각이 되면 그 순간의
            # fingertip_px를 새 기준점으로 확정한다
            if calib_pending_until_sec is not None and now_sec >= calib_pending_until_sec:
                if fingertip_px is not None:
                    neutral_px = fingertip_px
                    print("[D-PAD] CALIB fired -> cursor recentered")
                calib_pending_until_sec = None

            flash_progress = 0.0
            if flash_zone is not None:
                flash_elapsed_sec = now_sec - flash_start_sec
                if flash_elapsed_sec >= FLASH_SEC:
                    flash_zone = None
                else:
                    flash_progress = 1.0 - flash_elapsed_sec / FLASH_SEC

            frame = draw_dpad(frame, zones, offset_px, petal_r_px, hover_zone, hover_progress,
                              flash_zone, flash_progress)
            if calib_pending_until_sec is not None:
                # 재정렬 대기 안내 — 화면 중앙에 크게. CALIB 버튼 근처가 아니라
                # 화면 중앙인 이유: 이 몇 초 동안 사용자는 CALIB 버튼이 아니라
                # 자기가 옮겨갈 편한 자리를 보고 있다 (2026-08-04 사용자 요청)
                remaining_sec = max(0.0, calib_pending_until_sec - now_sec)
                guide_text = f"HOLD STILL - RECALIBRATING IN {remaining_sec:.1f}s"
                text_w_px = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
                text_x_px = max(4, w_px // 2 - text_w_px // 2)
                text_y_px = h_px // 2
                cv2.putText(frame, guide_text, (text_x_px, text_y_px), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, COLOR_PROGRESS, 2, cv2.LINE_AA)
            if cursor_px is not None:
                cx_px, cy_px = int(cursor_px[0]), int(cursor_px[1])
                cv2.circle(frame, (cx_px, cy_px), 10, COLOR_FINGERTIP, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx_px, cy_px), 2, COLOR_FINGERTIP, -1, cv2.LINE_AA)
                cv2.putText(frame, SHAPE_TAG.get(current_shape, "?"), (cx_px + 14, cy_px + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FINGERTIP, 2, cv2.LINE_AA)

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
                calib_pending_until_sec = now_sec + CALIB_DELAY_SEC   # CALIB 존 dwell과 동일 효과
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
