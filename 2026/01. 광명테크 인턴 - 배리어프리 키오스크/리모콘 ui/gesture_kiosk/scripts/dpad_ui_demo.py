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

★손가락↔주먹 혼동(2026-08-04 사용자 보고) — 두 겹으로 완화한다. 근본 원인은
hand_shape.py에 이미 기록된 기존 한계: 검지가 카메라 정면을 향하면 world
깊이 추정이 흔들려 손끝-손목 **거리**비(extend_ratio)가 굽힘 판정과 겹친다.
  1) crosscheck_fist_joints() — "주먹"인데 관절 **각도**(국소적이라 원근에
     덜 흔들림)로는 쫙 편 손가락이 있으면 그 판정을 취소(불명)한다
     (2026-08-04 사용자 제안).
  2) update_shape_latch() — 그래도 남는 순간 오판은 몇 프레임 연속 확인
     되기 전엔 이전 모양을 유지해 체감을 줄인다.
둘 다 완화일 뿐 근본 해결은 아니다 — 손가락을 카메라에 정면으로 몇 초씩
겨누면(관절도 함께 왜곡될 만큼 극단적이면) 결국 주먹으로 넘어갈 수 있다 —
그럴 땐 손가락을 살짝 옆으로 기울여 가리킬 것.

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

**재정렬(자동, 버튼 없음)**: 손이 쉬는 위치가 사람마다 달라 화면 중앙이라는
가정이 안 맞을 수 있다. 2026-08-04 사용자 결정으로 버튼(CALIB) 자체를 없애고
**허공에서 손을 가만히 STILL_RECALIBRATE_SEC(2.5)초 이상 멈추면** 자동으로
재정렬되게 바꿨다 — 손끝이 작은 반경(STILL_RADIUS_RATIO) 안에 머무는 시간을
매 프레임 재는 것으로, 별도 조준이 필요 없다(어디를 보고 있든 손만 멈추면
됨). 그 반경을 벗어나면(=움직이면) 시계가 그 자리에서 다시 시작한다. 정지
중엔 커서 옆에 진행 링과 안내 문구가 뜨고, 문턱을 넘는 순간 그 위치가 새
기준점이 된다 — 그 뒤로도 손이 계속 멈춰 있다고 매 프레임 재발화하지는
않는다(한 번 쏘면 다시 움직여야 재무장). 키보드 c는 그 자리에서 즉시
재정렬하는 수동 단축키로 남겨뒀다(테스트 편의용).
손이 없다가 **다시 나타날 때마다**(최초 등장 포함, 매번 즉시·대기 없이)
그 자리를 새 기준점으로 무조건 재정렬한다(2026-08-04 사용자 결정 — 이전엔
프로그램 시작 후 첫 등장에만 자동 보정하고 그다음부터는 손을 뗐다 들어도
직전 보정을 그대로 유지했으나, 그러면 이전 사용자·이전 위치의 보정을 다음
사람이 물려받는다는 문제가 있어 "손이 안 보이면 무조건 재정렬"로 바꿨다).
손이 계속 보이는 동안에는 위 정지 감지·키보드 c로만 갱신한다.

**민감도(CURSOR_SENSITIVITY)**: 손끝이 기준점에서 벗어난 픽셀을 이 배수만큼
키워 커서 오프셋으로 쓴다 — 손가락을 조금만 움직여도 커서가 크게 움직이게
하려면 이 값을 올린다(2026-08-04 사용자 요청으로 대폭 상향).

★커서 떨림(2026-08-04 사용자 보고): 위 3배 증폭이 손끝 필터의 잔여 노이즈도
그대로 3배 키운 것이 원인 — 필터 자체를 "정지 시 더 강하게 누르도록"
재조정했다(FILTER_MIN_CUTOFF_HZ·FILTER_BETA 정의부 주석 참고, 1.5/1.0 ->
0.7/0.4). 그래도 떨리면 그 두 값을 더 낮추고, 반대로 손 재배치가 굼뜨게
느껴지면 다시 올릴 것.

사용법 (프로젝트 루트에서):
    py scripts\\dpad_ui_demo.py [--device N] [--dwell-sec 0.55] [--fullscreen]
                              [--cx-ratio 0.50] [--cy-ratio 0.30]
종료: q/ESC (전체화면 중에도 동일 — 창에 테두리·닫기 버튼이 없어져도 키보드는 그대로 먹는다)
재정렬: 허공에 손을 2.5초 가만히 멈추거나 키보드 c
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
from src.postprocess.hand_shape import HAND_FINGERS, SHAPE_FIST, classify_hand_shape, finger_states
from src.postprocess.point_filter import PointFilter
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
WINDOW_NAME = "gesture_kiosk dpad ui demo"

INDEX_FINGERTIP_LMK = 8   # MediaPipe 21점 중 검지 끝 (hand_tracker.py 독스트링)

# 손끝 One Euro 필터 — 처음엔 config gestures.swipe.point_filter 값(min_cutoff 1.5·
# beta 1.0)을 그대로 재사용했으나, 그건 "빠른 쓸기 추종"용 튜닝이라 정지 시
# 떨림 억제가 약하다. 이 데모는 반대로 "가만히 멈춰있기"가 훨씬 잦은 조작(존
# dwell·정지 재정렬)이라 정지 쪽에 더 강하게 맞춘다 — beta를 낮춰 순간 노이즈를
# "빠른 움직임"으로 오인해 필터를 풀어버리는 일을 줄인다(2026-08-04 사용자
# 보고: CURSOR_SENSITIVITY 3배 증폭 위에 이 노이즈가 얹혀 커서가 갑자기 떤다).
# 빠른 재배치가 굼뜨게 느껴지면 beta를 다시 올릴 것
FILTER_MIN_CUTOFF_HZ = 0.7   # (1.5에서 하향) 정지 시 컷오프 — 낮을수록 떨림 억제 강함
FILTER_BETA = 0.4            # (1.0에서 하향) 속도 반응 — 낮을수록 노이즈를 "빠른 움직임"으로
                              # 덜 오인해 순간 튐이 덜 새어나간다
FILTER_D_CUTOFF_HZ = 1.0
AXIS_DOMINANCE = 1.2  # 방향 판정 — 한 축이 이만큼 우세해야 확정, 대각선 애매구간은
                      # 직전 방향 유지 (gesture_filter.py의 swipe axis_dominance와 같은 개념)
CURSOR_SENSITIVITY = 3.0   # 손끝이 기준점(재정렬 전엔 화면 중앙)에서 벗어난
                           # 픽셀 오프셋 -> 커서 오프셋 배율. 1.0=그대로 옮김.
                           # 2026-08-04 사용자 요청으로 대폭 상향(1.0->3.0) —
                           # 손가락을 조금만 움직여도 존까지 닿게. ★이 배율이
                           # 필터를 통과한 잔여 떨림도 그대로 3배 키우므로,
                           # 커서가 떨리면 필터(위 FILTER_MIN_CUTOFF_HZ)부터
                           # 볼 것 — 그래도 예민하면 이 값부터 낮춘다(2.0까지)

DEMO_EXTEND_RATIO = 0.80   # 본 엔진 config.yaml 기본값(1.0)보다 낮춤 — 2026-08-04
                           # 실측(SHAPE-DEBUG 로그, zone까지 찍어서 확인) 3차 조정:
                           # 1.0/0.75 — zone=up(위쪽 D-pad 존)으로 손을 뻗을 때 idx
                           # 비율이 0.55~0.75대로 낮게 나와 주먹으로 오판(카메라가
                           # 위에 있어 위로 뻗을수록 손가락이 렌즈를 더 향하는
                           # 원근 왜곡 — hand_shape.py 기존 한계 재현).
                           # 0.65 — idx는 잡히는데 이번엔 **중지(mid)까지** 자주
                           # 0.65를 넘어 "폄"으로 같이 잡혀 "손가락 1개만 폄"
                           # 조건이 깨짐(검지+중지 둘 다 폄 = 정의된 모양 없음 =
                           # UNKNOWN). 즉 idx가 낮을 때(0.55~) 잡으려 하면 mid의
                           # 노이즈 상한(~0.7~0.86)과 겹쳐버려 한 값으로 완벽히
                           # 못 가른다 — 0.8로 절충(중지 대부분은 걸러지고 검지
                           # 일부는 여전히 샐 수 있음). --extend-ratio로 실행 중
                           # 값 바꿔가며 재조정 가능. config.yaml 원본은 안 건드림
                           # (본 엔진 카메라 거리·자세 기준으로 이미 실기 보정된
                           # 값이라 — 여기 값만 이 데모 전용)
SHAPE_LATCH_FRAMES = 4     # 무래치 -> 고정: 같은 판별이 이 프레임 연속이면 그 모양으로 고정
SHAPE_SWITCH_FRAMES = 5    # (8에서 하향, 2026-08-04) 고정 -> 다른 모양 전환: 다른
                           # 판별이 이 프레임 연속이어야 전환 — 잡음이 큰 신호에서는
                           # 8이 너무 오래 걸려 진짜 모양 변화도 늦게 반영됐다
                           # (본 엔진 gesture_filter.py shape_latch와 같은 개념을 단순
                           # 이식 — 2026-08-04 사용자 보고: 손가락↔주먹 혼동. 근본 원인은
                           # hand_shape.py 모듈독스트링에 이미 기록된 기존 한계: 검지가
                           # 카메라 정면을 향하면 world_landmarks 깊이 추정이 흔들려
                           # 굽힘 판정 구간과 겹친다 — 임계값만으로는 못 고친다(2D 보조
                           # 신호가 필요하다고 적혀 있음, 미착수). 래치는 그 순간 오판이
                           # 몇 프레임 안에 그치면 이전 모양을 유지해 체감을 줄여줄 뿐,
                           # 손가락을 카메라에 정면으로 몇 초씩 겨누면 결국 주먹으로
                           # 넘어간다 — 그럴 땐 손가락을 살짝 옆으로 기울여 가리킬 것

STRAIGHT_JOINT_ANGLE_DEG = 80.0   # (30에서 상향, 2026-08-04 실측) 이 각도(도)
                           # 미만이면 관절이 사실상 폄으로 본다 — 실측 SHAPE-DEBUG
                           # 로그를 보니 실제로 편 검지도 각도가 12~94도까지
                           # 널뛰어(원래 "0도 근처"라는 가정이 틀렸다) 30도는
                           # 사실상 한 번도 안 걸렸다. 이제 진짜 문제(extend_ratio
                           # 잡음, DEMO_EXTEND_RATIO 참고)를 직접 고쳤으니 이건
                           # 보조 안전망 정도 — 진짜 주먹의 손가락이 이 각도
                           # 밑으로 내려가 잘못 취소되면 ↓
MIN_STRAIGHT_FINGERS_TO_VETO = 1   # 주먹 판정인데 관절상 폄으로 보이는 손가락이
                           # 이 개수 이상이면 그 판정을 취소(None, 불명) —
                           # 카메라 정면을 향한 손가락 1개만 오판되는 것이
                           # 이 버그의 실제 양상이라 1로 충분할 가능성이 큼
                           # (2026-08-04 사용자 제안 — 관절 각도 교차검증 추가)

DWELL_SEC_DEFAULT = 0.55   # 버튼 확정까지 머물러야 하는 시간
FLASH_SEC = 0.30           # 확정 직후 버튼이 밝게 반짝이는 시간
STILL_RECALIBRATE_SEC = 2.5   # 손끝이 이 시간 이상 한자리(STILL_RADIUS_RATIO
                           # 반경)에 머물면 자동 재정렬 — 버튼 없이 "허공에
                           # 가만히"만으로 트리거 (2026-08-04 사용자 요청:
                           # CALIB 버튼 폐기, 정지 감지로 대체)
STILL_RADIUS_RATIO = 0.035    # "가만히"로 인정하는 손끝 이동 허용 반경 —
                           # 프레임 짧은 변 비율. 벗어나면 시계가 그 자리에서
                           # 재시작한다. 너무 자주 오발화되면 ↑, 손이 많이
                           # 움직여도 재정렬되면 ↓

# 레이아웃 크기 = 프레임 짧은 변(scale_px = min(w_px, h_px))의 비율 — 구석에
# 고정할 거라 작게(2026-08-04 사용자 요청: 화면 중앙 대형판 -> 구석에 작게).
# offset+petal_r+받침판 여유가 구석 위치(cx_ratio·cy_ratio)에서 화면 밖으로
# 안 잘리게 맞춰야 한다 (offset > petal_r + center_r 라야 버튼끼리·버튼-중앙
# 사이에 간격이 보인다)
OFFSET_SCREEN_RATIO = 0.14
PETAL_RADIUS_SCREEN_RATIO = 0.065
CENTER_RADIUS_SCREEN_RATIO = 0.055

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


def compute_zones(center_px, scale_px):
    """D-pad 5개 존(up/down/left/right/center) -> {name: (cx_px, cy_px, radius_px)}.

    center_px: D-pad 중심(화면 픽셀 좌표, 기본은 프레임 정중앙).
    scale_px: 크기 기준 자(프레임의 짧은 변 픽셀 수) — 화면 비율에 맞춰 큼직하게 그린다.
    """
    cx_px, cy_px = int(center_px[0]), int(center_px[1])
    offset_px = int(scale_px * OFFSET_SCREEN_RATIO)
    petal_r_px = int(scale_px * PETAL_RADIUS_SCREEN_RATIO)
    center_r_px = int(scale_px * CENTER_RADIUS_SCREEN_RATIO)
    zones = {
        "up": (cx_px, cy_px - offset_px, petal_r_px),
        "down": (cx_px, cy_px + offset_px, petal_r_px),
        "left": (cx_px - offset_px, cy_px, petal_r_px),
        "right": (cx_px + offset_px, cy_px, petal_r_px),
        "center": (cx_px, cy_px, center_r_px),
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


def update_shape_latch(latched_shape, streak_value, streak_count, raw_shape):
    """손 모양 판별 출렁임을 걸러 프레임 하나짜리 오판이 그대로 새지 않게 한다.

    raw_shape: 이번 프레임 classify_hand_shape() 결과("finger"/"fist"/"open"/None).
    무래치 -> 고정은 같은 판별 SHAPE_LATCH_FRAMES연속, 고정 -> 다른 모양 전환은
    다른 판별 SHAPE_SWITCH_FRAMES연속이 필요 — 본 엔진 gesture_filter.py의
    shape_latch와 같은 발상(래치 상수 정의부 주석 참고). 순수 함수 —
    (latched_shape, streak_value, streak_count) -> 같은 모양의 다음 상태 3종.
    """
    if raw_shape == streak_value:
        streak_count += 1
    else:
        streak_value, streak_count = raw_shape, 1

    if latched_shape is None:
        if raw_shape is not None and streak_count >= SHAPE_LATCH_FRAMES:
            latched_shape = raw_shape
    elif raw_shape is not None and raw_shape != latched_shape and streak_count >= SHAPE_SWITCH_FRAMES:
        latched_shape = raw_shape
    return latched_shape, streak_value, streak_count


def _finger_joint_angle_deg(world_landmarks, mcp, pip, tip):
    """PIP 관절에서 두 뼈(MCP->PIP, PIP->TIP) 사이 꺾임 각도(도).

    0도=일직선(폄), 클수록 굽음. hand_shape.py의 extend_ratio(손끝-손목
    **거리**비)와 달리 이건 두 인접 마디 사이의 **국소** 각도라, 손가락이
    카메라 쪽 어느 방향을 향하든(원근으로 3D 거리 추정이 흔들려도) 상대적으로
    덜 흔들린다는 것이 이 교차검증의 전제다.
    """
    vec_a = world_landmarks[pip] - world_landmarks[mcp]
    vec_b = world_landmarks[tip] - world_landmarks[pip]
    norm_a, norm_b = float(np.linalg.norm(vec_a)), float(np.linalg.norm(vec_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    cos_angle = max(-1.0, min(1.0, float(np.dot(vec_a, vec_b)) / (norm_a * norm_b)))
    return math.degrees(math.acos(cos_angle))


def crosscheck_fist_joints(shape, world_landmarks):
    """주먹 판정을 손가락 관절 각도로 교차검증(2026-08-04 사용자 제안).

    "주먹"인데 관절상으로는 쫙 편 손가락이 MIN_STRAIGHT_FINGERS_TO_VETO개
    이상이면 그 판정을 취소(None, 불명)한다 — 카메라 정면을 향한 검지 하나가
    extend_ratio 오판으로 주먹에 섞여 들어오는 경우를 걸러낸다(2026-08-04
    사용자 보고 버그, hand_shape.py 모듈독스트링에 이미 기록된 기존 한계).
    finger/open 판정은 그대로 둔다 — 이미 폄 신호가 있어 이 교차검증이
    필요 없다. update_shape_latch와 함께 쓴다: 여기서 None이 되면 래치가
    직전 모양을 그대로 유지하므로, 오판이 래치를 뒤집기 전에 걸러진다.
    """
    if shape != SHAPE_FIST:
        return shape
    straight_count = 0
    for mcp, pip, _dip, tip in HAND_FINGERS:
        angle_deg = _finger_joint_angle_deg(world_landmarks, mcp, pip, tip)
        if angle_deg is not None and angle_deg < STRAIGHT_JOINT_ANGLE_DEG:
            straight_count += 1
    if straight_count >= MIN_STRAIGHT_FINGERS_TO_VETO:
        return None
    return shape


def draw_shape_debug(frame, w_px, h_px, raw_before_veto, latched_shape, finger_angles_deg):
    """손 모양 판정 계기판(좌하단, 본 엔진 visualize.draw_debug_panel과 같은 자리) —
    RAW(교차검증 전 원시 판별)·ANGLES(손가락별 PIP 관절 각도, STRAIGHT_JOINT_
    ANGLE_DEG 미만이면 폄으로 봄)·LATCH(화면·발화에 실제로 쓰이는 최종값)를
    한 줄씩 보여준다. 오판이 어느 단계(판별 자체 vs 관절 교차검증 vs 래치)에서
    나는지 화면 하나로 구분하기 위한 진단용(2026-08-04 사용자 요청 — 버그를
    직접 보면서 고치기). 밝은 벽 등 배경에서 흰 글자가 묻힐 수 있어 어두운
    받침띠를 깔고 그 위에 쓴다(2026-08-04 실기 — 실제로 안 보여서 추가)."""
    finger_names = ("idx", "mid", "ring", "pinky")
    angle_text = " ".join(
        f"{name}={angle:.0f}" if angle is not None else f"{name}=-"
        for name, angle in zip(finger_names, finger_angles_deg)
    )
    lines = [
        f"RAW {raw_before_veto or '-'}",
        f"ANGLES {angle_text} (<{STRAIGHT_JOINT_ANGLE_DEG:.0f}=straight)",
        f"LATCH {latched_shape or '-'}",
    ]
    panel_h_px = 14 + 22 * len(lines) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h_px - panel_h_px), (w_px, h_px), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for line_idx, line in enumerate(lines):
        y_px = h_px - 14 - 22 * (len(lines) - 1 - line_idx)
        cv2.putText(frame, line, (10, y_px), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


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
    parser.add_argument("--extend-ratio", type=float, default=DEMO_EXTEND_RATIO,
                        help="손가락 '폄' 판정 기준 — 낮출수록 잘 펴진다(기본은 본 엔진 "
                             "config보다 낮은 이 데모 전용 실측값, DEMO_EXTEND_RATIO 참고)")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)
    shape_cfg = config["hand_select"]["hand_shape"]   # min_valid_fingers·curl_confirm_ratio는
                           # 본 엔진 값 그대로 재사용 — extend_ratio만 이 데모 전용으로 덮어씀
    if args.device is not None:
        config["camera"]["device_id"] = args.device
        auto_select = config["camera"].get("auto_select")
        if auto_select is not None:
            config["camera"]["auto_select"] = {**auto_select, "enabled": False}

    preprocessor = Preprocessor(config)
    hand_tracker = HandTracker(config)
    camera = CameraStream(config).start()

    fingertip_filter = PointFilter(FILTER_MIN_CUTOFF_HZ, FILTER_BETA, FILTER_D_CUTOFF_HZ)
    last_shape_debug_print_sec = 0.0   # 화면 진단 계기판이 창 가림으로 안 보일 때도
                           # 콘솔 로그로 같은 정보를 볼 수 있게(2026-08-04 사용자 요청 —
                           # 실기로 직접 보면서 고치기, 초당 다다다 찍히지 않게 간격 제한)
    last_direction = None  # 직전 확정 방향(대각선 애매구간 유지용) — 손 재등장 시 초기화
    latched_shape = None    # 래치된 손 모양 — update_shape_latch가 매 프레임 갱신,
                           # 표시·발화 모두 이 값을 쓴다(원시 판별 아님)
    shape_streak_value, shape_streak_count = None, 0   # 래치 내부 상태(연속 판별 카운터)
    neutral_px = None      # 재정렬 기준점 — None이면 화면 중앙 사용(자동 보정 전 폴백).
                           # 손이 없어져도 유지(사용자가 다시 재정렬하기 전까지 그대로)
    still_anchor_px = None   # 정지 추적 시작점 — None이면 추적 안 함(손 없음).
                           # 손끝이 이 점에서 STILL_RADIUS_RATIO를 벗어나면 그
                           # 자리로 재설정(시계 재시작)
    still_start_sec = 0.0
    still_fired = False    # 이번 정지 구간에서 이미 재정렬했는지 — 다시
                           # 움직였다 멈춰야 재무장(연속 재발화 방지)
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
                   "for %.2fs to fire it. Hold your hand still for %.1fs to recalibrate "
                   "(no button needed). Quit: q or ESC", args.dwell_sec, STILL_RECALIBRATE_SEC)

    try:
        while True:
            frame = preprocessor.preprocess_frame(camera.capture_frame(), apply_crop=False)
            now_sec = time.monotonic()
            h_px, w_px = frame.shape[:2]

            hands = hand_tracker.infer(frame)
            fingertip_px = None
            raw_shape_before_veto = None    # 진단용 — 교차검증 전 원시 판별
            debug_finger_angles_deg = None  # 진단용 — 손가락별 PIP 관절 각도
            if hands:
                # 데모 단순화: 신뢰도가 가장 높은 손 하나만 본다 (본 엔진의 단일 손
                # 추적·앵커 게이트는 main.py 쪽 hand_select.py가 담당 — 여기선 재사용 안 함)
                hand = max(hands, key=lambda h: h.conf)
                is_new_appearance = not had_hand_prev
                if is_new_appearance:
                    fingertip_filter.reset()   # 재등장 — 이전 궤적과 섞이지 않게
                    last_direction = None
                    latched_shape, shape_streak_value, shape_streak_count = None, None, 0
                had_hand_prev = True

                raw_tip = (float(hand.landmarks[INDEX_FINGERTIP_LMK][0]),
                          float(hand.landmarks[INDEX_FINGERTIP_LMK][1]))
                fingertip_px = fingertip_filter.filter(raw_tip, now_sec)
                if is_new_appearance:
                    # 손이 없다가 다시 나타날 때마다(최초 등장 포함) 무조건
                    # 그 자리를 새 기준점으로 — 이전 사용자·이전 위치의 보정을
                    # 물려받지 않는다(2026-08-04 사용자 요청: "손이 안 보이면
                    # 무조건 캘리브레이션"). 손을 잠깐 화면 밖으로 내렸다
                    # 올려도 매번 그 자리에서 커서가 D-pad 중앙으로 시작한다
                    neutral_px = fingertip_px

                # world_landmarks 권장(화면 좌표 z는 노이즈가 커서 오판 위험 —
                # hand_shape.py classify_hand_shape 독스트링). 원시 판별에
                # 관절 각도 교차검증 -> 래치 순으로 두 겹 필터를 거쳐야 쓴다 —
                # 프레임 하나짜리 오판(예: 손가락이 카메라 정면을 향해 순간
                # 주먹으로 잘못 읽히는 경우)이 그대로 새지 않게
                raw_shape_before_veto = classify_hand_shape(
                    hand.world_landmarks, args.extend_ratio,
                    shape_cfg["min_valid_fingers"], shape_cfg["curl_confirm_ratio"],
                )
                raw_shape = crosscheck_fist_joints(raw_shape_before_veto, hand.world_landmarks)
                latched_shape, shape_streak_value, shape_streak_count = update_shape_latch(
                    latched_shape, shape_streak_value, shape_streak_count, raw_shape,
                )
                # 진단용 — 손가락별 관절 각도를 계기판에 그대로 노출해 오판이
                # 어느 단계(판별 자체 vs 교차검증 vs 래치)에서 나는지 한눈에
                # 보이게 한다 (2026-08-04 사용자 요청 — 버그를 직접 보고 고치기)
                debug_finger_angles_deg = [
                    _finger_joint_angle_deg(hand.world_landmarks, mcp, pip, tip)
                    for mcp, pip, _dip, tip in HAND_FINGERS
                ]
                # 화면 계기판이 창 가림으로 안 보일 때를 대비한 콘솔판 — 0.4초
                # 간격으로만 찍는다(2026-08-04 사용자 요청: 버그를 직접 보면서
                # 고치기. 콘솔 로그는 창 z-순서와 무관하게 항상 읽을 수 있다)
                if now_sec - last_shape_debug_print_sec >= 0.4:
                    last_shape_debug_print_sec = now_sec
                    angle_text = " ".join(
                        f"{n}={a:.0f}" if a is not None else f"{n}=-"
                        for n, a in zip(("idx", "mid", "ring", "pinky"), debug_finger_angles_deg)
                    )
                    # extend_ratio 원시값도 같이 본다 — classify_hand_shape가 내부에서
                    # 쓰는 바로 그 값(hand_shape.finger_states)이라, "판별기가 왜 이
                    # 손가락을 굽음으로 봤는지"를 관절 각도보다 직접적으로 보여준다
                    # (2026-08-04 사용자 확인: 검지 하나만 계속 편 채로 있었는데도
                    # raw=fist가 대부분이었음 — extend_ratio 자체가 이 카메라
                    # 거리/자세에서 안 맞을 가능성 점검)
                    states = finger_states(hand.world_landmarks, args.extend_ratio,
                                           shape_cfg["curl_confirm_ratio"])
                    ratio_text = " ".join(
                        f"{n}={ratio:.2f}/{state}"
                        for n, (ratio, state) in zip(("idx", "mid", "ring", "pinky"), states)
                    )
                    # zone(직전 프레임 값 — 이 시점엔 이번 프레임 zone이 아직
                    # 계산 전이라 최대 한 프레임 지연, 진단 목적엔 무관) — 사용자가
                    # "특정 구간(방향)에서만 fist가 나온다"고 보고해 위치 상관관계를
                    # 직접 보려는 것(2026-08-04)
                    print(f"[SHAPE-DEBUG] zone={hover_zone or '-'} "
                          f"raw={raw_shape_before_veto or '-'} "
                          f"veto={'YES' if raw_shape_before_veto != raw_shape else 'no'} "
                          f"latch={latched_shape or '-'} angles[{angle_text}] "
                          f"ratios(>={args.extend_ratio:.2f}=extend)[{ratio_text}]")
            else:
                had_hand_prev = False
                latched_shape, shape_streak_value, shape_streak_count = None, None, 0

            # 정지 감지 — 손끝이 STILL_RADIUS_RATIO 반경 안에 STILL_RECALIBRATE_SEC
            # 이상 머물면 자동 재정렬 (2026-08-04 사용자 요청: CALIB 버튼 폐기,
            # 조준 없이 "허공에 가만히"만으로 트리거). 반경을 벗어나면(=움직이면)
            # 그 자리를 새 정지 시작점 삼아 시계를 다시 켠다
            if fingertip_px is None:
                still_anchor_px = None
            else:
                still_radius_px = STILL_RADIUS_RATIO * min(w_px, h_px)
                if (still_anchor_px is None
                        or math.hypot(fingertip_px[0] - still_anchor_px[0],
                                      fingertip_px[1] - still_anchor_px[1]) > still_radius_px):
                    still_anchor_px, still_start_sec, still_fired = fingertip_px, now_sec, False
                elif not still_fired and now_sec - still_start_sec >= STILL_RECALIBRATE_SEC:
                    neutral_px = fingertip_px   # 재확정하려면 다시 움직였다 멈춰야 함(연속 재발화 방지)
                    still_fired = True
                    print("[D-PAD] STILL fired -> cursor recentered")
                    last_action_label = "RECALIBRATED"

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
                    shape_label = SHAPE_TAG.get(latched_shape, "UNKNOWN")
                    last_action_label = zone_funcs[hover_zone](shape_label)

            flash_progress = 0.0
            if flash_zone is not None:
                flash_elapsed_sec = now_sec - flash_start_sec
                if flash_elapsed_sec >= FLASH_SEC:
                    flash_zone = None
                else:
                    flash_progress = 1.0 - flash_elapsed_sec / FLASH_SEC

            frame = draw_dpad(frame, zones, offset_px, petal_r_px, hover_zone, hover_progress,
                              flash_zone, flash_progress)
            if debug_finger_angles_deg is not None:
                frame = draw_shape_debug(frame, w_px, h_px, raw_shape_before_veto, latched_shape,
                                         debug_finger_angles_deg)
            if cursor_px is not None:
                cx_px, cy_px = int(cursor_px[0]), int(cursor_px[1])
                cv2.circle(frame, (cx_px, cy_px), 10, COLOR_FINGERTIP, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx_px, cy_px), 2, COLOR_FINGERTIP, -1, cv2.LINE_AA)
                cv2.putText(frame, SHAPE_TAG.get(latched_shape, "?"), (cx_px + 14, cy_px + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FINGERTIP, 2, cv2.LINE_AA)
                # 정지 재정렬 진행 — 커서 둘레 링 + 안내 문구. CALIB 버튼이
                # 없어진 자리라 커서 옆이 유일하게 사용자가 항상 보는 위치
                # (2026-08-04 사용자 요청)
                if still_anchor_px is not None and not still_fired:
                    still_progress = min(1.0, (now_sec - still_start_sec) / STILL_RECALIBRATE_SEC)
                    if still_progress > 0:
                        end_angle_deg = 360.0 * still_progress
                        cv2.ellipse(frame, (cx_px, cy_px), (18, 18), -90, 0, end_angle_deg,
                                    COLOR_PROGRESS, 3, cv2.LINE_AA)
                        cv2.putText(frame, "HOLD STILL TO RECALIBRATE", (cx_px + 14, cy_px + 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PROGRESS, 2, cv2.LINE_AA)

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
                neutral_px = fingertip_px   # 수동 단축키 — 정지 대기 없이 즉시 재정렬(테스트 편의용)
                still_fired = True          # 방금 잡은 자리를 정지 앵커로도 인정 — 곧바로 재발화 방지
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
