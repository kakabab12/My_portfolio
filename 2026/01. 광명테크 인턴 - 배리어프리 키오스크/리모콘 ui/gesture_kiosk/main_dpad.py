"""gesture_kiosk D-pad UI 공식 진입점 (2026-08-04 신설, 사용자 결정).

scripts/dpad_ui_demo.py에서 실기로 다듬어진 손동작 D-pad 프로토타입을
main.py와 같은 격의 **공식 진입점**으로 승격한다 — 손 모양(한 손가락/주먹/
손바닥)·각도 기반 방향 판정·오프셋 커서·정지 자동 재정렬 등 판정 로직은
그 파일에서 그대로 가져왔다(각 상수 옆 주석이 실기 튜닝 이력을 그대로
보존한다 — dpad_ui_demo.py 자체는 체크포인트로 손대지 않고 그대로 둔다).

★이번에 새로 추가된 것 — 헤드트래커 통합(2026-08-04 사용자 요청): 본 엔진
(src/pipeline/realtime_loop.py)의 "머리 좌우 흔들기 -> 입력 모드 전환" 구조를
그대로 재사용한다. config mode_switch.enabled(기본 true)면 얼굴 랜드마크
(FaceEstimator)가 매 프레임 함께 돌며, 머리를 흔들면(HeadShakeDetector) 손
↔머리 입력 모드가 토글된다 — 새 모드로 들어가면 그 모드의 트래커부터
새로 시작한다(HeadTracker.reset() / 손은 다음 등장을 "새 등장"으로 취급해
기존 재정렬 로직을 그대로 태운다). 얼굴 추론은 모드와 무관하게 항상 도는데,
손 모드 중에도 흔들기로 전환해야 하기 때문이다.

★머리 모드는 D-pad를 쓰지 않는다(2026-08-04 사용자 결정 — 처음엔 D-pad
존을 머리로도 조준하게 만들었으나, "헤드트래커는 dpad 쓰면 안되는거지"라는
피드백으로 전면 교체): 머리 모드로 들어가면 D-pad 자체를 그리지 않고, 손
추론도 건너뛴다(본 엔진과 동일한 최적화 — realtime_loop.py 참고). 대신 본
엔진이 헤드트래커 모드에서 원래 쓰던 화면을 그대로 재사용한다 —
src/utils/visualize.py의 draw_cursor(코끝 커서 + 응시 재정렬 진행 링)·
draw_head_debug_panel(JAW/EYE/DWELL/RECENTER/PUCKER 계기판)·draw_mode_banner
(상단 중앙 HAND MODE/HEAD MODE 배너)를 재구현 없이 그대로 가져다 쓴다. 프레임
크롭 기본값도 본 엔진과 동일하게 머리 모드에서만 9:16 세로로 크롭한다(config
camera.portrait_crop, preprocessor는 이미 그 옵션을 갖고 있었고 손 모드는
원본 16:9 그대로 받는다) — 단 --overlay일 때는 예외로, 모드와 무관하게
항상 9:16 고정이다(아래 참고). 확정(select)·처음으로(home)·재정렬
(calibration) 이벤트도 D-pad 존에 억지로
끼워 맞추지 않고 본 엔진 그대로 소비한다(발화만 하고 콘솔엔 안 찍는다 — 아래
참고). 즉 머리 모드는 "D-pad를 머리로 조작"이 아니라 "이 데모 안에서 본
엔진의 헤드트래커 모드를 그대로 미리보기"하는 것에 가깝다 — 손 모드(D-pad)와
완전히 분리된 화면·판정이다.

손 모양(FINGER/FIST/OPEN) 태그 등은 D-pad 전용(손 모드 전용) 개념이라 머리
모드에는 등장하지 않는다.

★콘솔(cmd 창) 출력 방침은 2026-08-04 두 차례 사용자 피드백으로 정리됐다.
처음엔 SHAPE-DEBUG·존 발화·모드 전환·머리 이벤트를 전부 echo했으나
"cmd 창에 xy 좌표가 안 떠, [HEAD] [shape debug] 이것도 없애주고 둘다 xy
좌표만 뜨게 해줘"라는 요청으로 SHAPE-DEBUG·모드 전환·머리 이벤트 echo·존
발화 echo를 전부 없애고 xy 연속값 하나만 남겼다가, "방향 버튼 누르면 함수는
나와야지 각각 모두 다"라는 후속 요청으로 **존 발화 알림만 다시 추가**했다
(build_zone_functions 참고) — 그래서 최종 콘솔은 (1) 매 프레임 연속으로
찍는 xy 좌표 + (2) 5개 존이 실제 발화하는 순간 각각 구분되는 한 줄, 이
둘로 정리됐다. xy는 손 모드가 D-pad 커서(cursor_px)를 화면 폭·높이로 나눠
0~1 비율로, 머리 모드가 head_tracker의 cursor_x/y_ratio(이미 0~1 비율)를
그대로 찍는다 — 단위를 맞춰 두 모드를 오가도 숫자 의미가 달라지지 않는다.
화면 표시(cursor_px 원, "CURSOR x=.. y=.." 텍스트, "FIRED:" 라벨 등)는
이번 정리와 무관하게 그대로다. 커서가 없는 프레임(손 미검출·머리
캘리브레이션 중)은 xy를 찍을 값이 없어 건너뛴다.

★--blank-bg(2026-08-05 사용자 요청): 카메라 영상 대신 빈 배경 위에 D-pad·
활성 모드 커서만 그려서 보여준다 — "실제 키오스크 화면엔 뭐가 보일지"를
카메라 프리뷰 없이 미리 보는 용도. camera_frame(추론용 실제 카메라 픽셀)과
frame(화면에 그릴 캔버스)을 분리해 구현했다 — 판정 경로(hand_tracker·
face_estimator·head_tracker 전부 camera_frame을 본다)는 이 옵션과 무관하게
완전히 동일하고, 바뀌는 건 draw_*·imshow가 그리는 대상뿐이다. 손 모드면
D-pad+손끝 커서, 머리 모드면 본 엔진 헤드트래커 커서+계기판이 그 캔버스
위에 그려진다 — 모드 전환 자체는 기존과 동일하게 한 번에 하나만 보인다
(두 커서를 동시에 띄우는 것은 이번 요청 범위 밖으로 확인함).

★--overlay(2026-08-05, 네 차례 요청을 거쳐 지금 형태): 1차 "빈 화면 말고
뒤에 바탕화면 보이게 dpad만 띄워줄 수는 없나"에 --blank-bg의 빈 배경 대신
진짜 바탕화면(또는 뒤 창)이 비쳐 보이는 반투명 창으로 답했고, 2차 "dpad
배경 투명하게 해주고 모든것을 다 투명하게 해줘"에 창 전체 알파를
추가했다가, 3차 "dpad만 뜨게 하고 나머지는 안보이게 해줘 그리고 dpad 색 다
없애줘"로 (1) D-pad를 받침판·버튼 면 채우기 없는 **와이어프레임**으로
바꾸고(draw_dpad의 wireframe 인자 — 면이 있던 자리는 색상 키 그대로라
완전히 투명), (2) 진단용 계기판·라벨(손 모양 계기판·"D-PAD UI"·"FIRED:"·
헤드트래커 계기판·좌표 텍스트·모드 배너·커서 옆 FINGER/FIST 라벨)을 전부
빼 D-pad 본체와 커서만 남겼다. 4차 "dpad 반투명 흰색으로 보여줘 커서는
초록색으로"로 색을 확정했다 — 와이어프레임 테두리·아이콘은
OVERLAY_DPAD_COLOR(흰색), 커서는 OVERLAY_CURSOR_COLOR(초록, 머리 모드
커서와 같은 색)이고, --overlay-alpha 기본값을 170(반투명)으로 낮췄다.
★색상 키 방식의 한계로 알파는 창 전체에 균일 적용된다 — "D-pad만 더
반투명하고 커서는 더 또렷하게" 같은 요소별 차등은 지원하지 않는다(정말
필요해지면 UpdateLayeredWindow 기반 픽셀별 알파로 다시 설계해야 함 —
enable_transparent_overlay 독스트링 참고). --blank-bg와 동시 지정 불가
(상호 배타 — 인자 정의부 참고). 5차 "dpad overay만 헤드모드 핸드모드 둘다
비율 9:16으로 바꿔주라"로 프레임 크롭도 손댔다 — 기본은 머리 모드에서만
9:16(realtime_loop.py와 동일 조건)이지만, --overlay는 모드와 무관하게 항상
9:16으로 고정한다(camera_frame 캡처 줄의 apply_crop 조건식 참고): 오버레이는
실제 세로 키오스크 화면을 흉내 내는 용도라, 화면 모양 자체가 손↔머리 전환
때마다 바뀌면 미리보기로서 의미가 없기 때문이다. --blank-bg·카메라
프리뷰(기본, --overlay 아님)는 이 예외에 안 걸린다 — 손 모드에서 원본 16:9
그대로.

사용법 (프로젝트 루트에서):
    py main_dpad.py [--device N] [--dwell-sec 0.55] [--fullscreen]
                    [--cx-ratio 0.50] [--cy-ratio 0.30] [--no-head-mode]
                    [--min-cutoff-hz 0.25] [--filter-beta 0.12] [--cursor-sensitivity 2.5]
                    [--blank-bg | --overlay [--overlay-alpha 170]]
종료: q 또는 ESC
재정렬(손 모드): 허공에 손을 2.5초 가만히 멈추거나 키보드 c
확정(머리 모드): 입 벌리기 또는 응시(config head_tracker.dwell_click.dwell_sec, 기본 1.5초)
모드 전환: 머리를 좌우로 흔들기 (config mode_switch.enabled=true일 때만 — 기본 켜짐)
"""
import argparse
import math
import os
import sys
import time

PROCESS_START_SEC = time.monotonic()   # 시작 시간 계측 기점 — main.py와 동일 관례
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)   # 델파이/exe가 어느 작업 디렉터리에서 띄우든 config의 모델·로그
                     # 상대 경로가 항상 성립하게 (main.py와 동일 이유)

import cv2
import numpy as np

from src.capture.camera_stream import CameraStream
from src.inference.face_estimator import (
    FaceEstimator, LMK_LEFT_EYE_OUTER, LMK_NOSE_TIP, LMK_RIGHT_EYE_OUTER,
)
from src.inference.hand_tracker import HandTracker
from src.inference.preprocessor import Preprocessor
from src.postprocess.face_anchor import FaceAnchor
from src.postprocess.hand_shape import HAND_FINGERS, SHAPE_FIST, classify_hand_shape
from src.postprocess.head_shake import HeadShakeDetector
from src.postprocess.head_tracker import HeadTracker
from src.postprocess.point_filter import PointFilter
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging
from src.utils.visualize import draw_cursor, draw_head_debug_panel, draw_mode_banner

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
WINDOW_NAME = "gesture_kiosk dpad"

INDEX_FINGERTIP_LMK = 8   # MediaPipe 21점 중 검지 끝 (hand_tracker.py 독스트링)

MODE_HAND = "hand"
MODE_HEAD = "head"

# 손끝 One Euro 필터 — dpad_ui_demo.py 2026-08-04 실기 조정값(1.5/1.0 -> 0.7/0.4)을
# 물려받았으나, 2026-08-05 --overlay 도입 후 두 차례 연달아 떨림 재보고
# ("핸드모드일때 커서가 좀 떨리더라" -> 필터 재조정 -> "오버레이 커서 왜이렇게
# 떨리지"): 카메라 영상 위에서는 배경 자체가 계속 흔들려 잔여 떨림이 묻혔지만,
# --overlay의 정지된 배경(바탕화면) 위 밝은 색 커서 하나만 보면 같은 크기의
# 떨림도 훨씬 눈에 띈다. 짐작으로 값을 두 번 내렸는데도 부족하다는 신호라,
# 이번엔 세 값(아래 CURSOR_SENSITIVITY 포함) 전부 --min-cutoff-hz·
# --filter-beta·--cursor-sensitivity로 실행 중 조정 가능하게 열었다 —
# extend_ratio 때처럼 카메라 앞에서 직접 보며 맞는 값을 찾는 편이 여기서
# 더 낮은 숫자를 또 추측하는 것보다 낫다고 판단(2026-08-05). 아래 상수들은
# 그 조정 전 시작값(기본값)일 뿐이다.
FILTER_MIN_CUTOFF_HZ = 0.25   # 정지 시 컷오프 — 더 낮출수록 떨림 억제가 강해지는
                           # 대신 움직이기 시작할 때 반응이 굼떠진다
FILTER_BETA = 0.12            # 속도 반응 계수 — 낮을수록 순간적인 잡음을 "빠른
                           # 움직임"으로 오인해 억제를 풀어버리는 일이 줄어든다
                           # (그만큼 실제 빠른 손짓엔 살짝 더 지연)
FILTER_D_CUTOFF_HZ = 1.0
AXIS_DOMINANCE = 1.2   # 방향 판정 — 한 축이 이만큼 우세해야 확정, 대각선 애매구간은
                       # 직전 방향 유지 (gesture_filter.py의 swipe axis_dominance와 같은 개념)
CURSOR_SENSITIVITY = 2.5   # 손끝 오프셋 -> 커서 오프셋 배율 — 필터를 통과한 잔여
                           # 떨림도 그대로 이 배수만큼 커진다. 3.0(2026-08-04
                           # 실기값)에서 소폭 하향 — 반응성은 웬만큼 유지하면서
                           # 떨림 증폭도 같이 줄이는 절충. --cursor-sensitivity로
                           # 조정 가능
DEMO_EXTEND_RATIO = 0.80   # 본 엔진 config.yaml 기본값(1.0)보다 낮춤 — 2026-08-04
                           # 실측(SHAPE-DEBUG 로그) 3차 조정 끝 확정값. 조정 이력 전문은
                           # scripts/dpad_ui_demo.py의 동일 상수 주석 참고. --extend-ratio로
                           # 실행 중 값 변경 가능. config.yaml 원본은 안 건드림(본 엔진
                           # 카메라 거리·자세 기준으로 이미 실기 보정된 값이라 — 여기 값만
                           # 이 데모 전용)
SHAPE_LATCH_FRAMES = 4     # 무래치 -> 고정: 같은 판별이 이 프레임 연속이면 그 모양으로 고정
SHAPE_SWITCH_FRAMES = 5    # 고정 -> 다른 모양 전환에 필요한 연속 프레임 수 (dpad_ui_demo.py
                           # 2026-08-04 8->5 하향 이력 — 잡음이 큰 신호에서 8은 너무 굼떴다)

STRAIGHT_JOINT_ANGLE_DEG = 80.0   # 이 각도(도) 미만이면 관절이 사실상 폄으로 본다 —
                           # dpad_ui_demo.py 2026-08-04 실측 상향(30->80) 이력 참고
MIN_STRAIGHT_FINGERS_TO_VETO = 1   # 주먹 판정인데 관절상 폄으로 보이는 손가락이 이
                           # 개수 이상이면 그 판정을 취소(None, 불명) — 관절 각도
                           # 교차검증(crosscheck_fist_joints) 문턱

DWELL_SEC_DEFAULT = 0.55   # 버튼 확정까지 머물러야 하는 시간
FLASH_SEC = 0.30           # 확정 직후 버튼이 밝게 반짝이는 시간
STILL_RECALIBRATE_SEC = 2.5   # 손끝이 이 시간 이상 한자리(STILL_RADIUS_RATIO 반경)에
                           # 머물면 자동 재정렬 — 손 모드 전용(버튼 없이 "허공에 가만히")
STILL_RADIUS_RATIO = 0.035    # "가만히"로 인정하는 손끝 이동 허용 반경 — 프레임 짧은 변 비율

# 레이아웃 크기 = 프레임 짧은 변(scale_px = min(w_px, h_px))의 비율
OFFSET_SCREEN_RATIO = 0.14
PETAL_RADIUS_SCREEN_RATIO = 0.065
CENTER_RADIUS_SCREEN_RATIO = 0.055

COLOR_BG = (54, 46, 43)
COLOR_BG_HOVER = (90, 76, 50)
COLOR_FLASH = (255, 255, 255)
COLOR_BORDER = (24, 21, 19)
COLOR_ACCENT = (61, 163, 232)
COLOR_PROGRESS = (255, 221, 156)
COLOR_FINGERTIP = (80, 220, 120)
COLOR_TEXT = (255, 255, 255)
PLATE_ALPHA = 0.55
BLANK_BG_COLOR = (18, 18, 18)   # --blank-bg 배경색 — D-pad 존(COLOR_BG=(54,46,43))·
                           # 테두리(COLOR_BORDER=(24,21,19))보다 확실히 어둡게 잡아
                           # 배경에 묻히지 않게 한다
TRANSPARENT_KEY_COLOR = (255, 0, 255)   # --overlay 크로마키 색(BGR 마젠타) — 이
                           # 정확한 색의 픽셀만 SetLayeredWindowAttributes(LWA_
                           # COLORKEY)로 완전 투명 처리된다. D-pad 팔레트(어두운
                           # 남색·파란 강조·흰 텍스트) 어디에도 없는 색이라 UI
                           # 요소를 잘못 뚫을 위험이 없다 — 고전적인 크로마키 관례색
OVERLAY_DPAD_COLOR = (255, 255, 255)   # --overlay 와이어프레임 색(2026-08-05
                           # 사용자 요청: "dpad 반투명 흰색으로") — 테두리·아이콘
                           # 전부 이 색. COLOR_TEXT와 값은 같지만 의도가 달라 별도
                           # 상수로 둔다(D-pad 전용 팔레트임을 이름으로 드러냄)
OVERLAY_CURSOR_COLOR = (0, 220, 0)   # --overlay 커서 색(2026-08-05 사용자 요청:
                           # "커서는 초록색으로") — visualize.py의 CURSOR_COLOR와
                           # 값을 맞춰 손 모드·머리 모드 커서 색이 서로 통일된다

SHAPE_TAG = {"fist": "FIST", "finger": "FINGER", "open": "OPEN"}   # 커서 옆 표시용
                                    # — classify_hand_shape 반환값(hand_shape.py)과 1:1

logger = get_logger("scripts")


def disable_console_quick_edit():
    """콘솔 빠른 편집(QuickEdit) 해제 — main.py와 동일 로직·동일 이유(그 파일 독스트링
    참고: 터치스크린 키오스크에서 콘솔을 스치기만 해도 출력이 멈추는 것 방지).
    콘솔이 없으면(파이프 실행) 조용히 통과 — 실패해도 시작을 막지 않는다."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        std_input_handle = kernel32.GetStdHandle(-10)   # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(std_input_handle, ctypes.byref(mode)):
            return
        quick_edit_flag = 0x0040
        extended_flags = 0x0080
        kernel32.SetConsoleMode(std_input_handle,
                                (mode.value & ~quick_edit_flag) | extended_flags)
    except Exception:   # noqa: 방어적 — 콘솔 설정 실패가 시작을 막으면 안 된다
        pass


def enable_transparent_overlay(window_name, key_color_bgr, whole_window_alpha):
    """--overlay 전용 — cv2 창을 반투명 창으로 바꾼다.

    2026-08-05 사용자 요청 두 차례로 지금 형태가 됐다: 처음엔 "빈 화면 말고
    뒤에 바탕화면 보이게 dpad만 띄워줄 수는 없나"(→ 배경만 색상 키로 뚫는
    1차 구현), 이어서 "dpad 배경 투명하게 해주고 모든것을 다 투명하게
    해줘"(→ 배경뿐 아니라 D-pad·커서·텍스트까지 전부 함께 옅어지도록 확장).

    cv2/HighGUI 자체는 창 투명도를 지원하지 않아, 창이 만들어진 뒤 Win32
    API(ctypes — disable_console_quick_edit와 같은 이유로 pywin32 등 새 의존성
    없이 표준 라이브러리만 사용)로 직접 확장 스타일을 바꾼다: WS_EX_LAYERED를
    켜고 SetLayeredWindowAttributes를 **두 플래그를 함께** 써서 적용한다 —
    LWA_COLORKEY로 key_color_bgr와 정확히 일치하는 배경 픽셀은 완전 투명(그
    자리는 바탕화면·뒤 창이 그대로 비침), LWA_ALPHA로 나머지 모든 픽셀(D-pad
    받침판·버튼·아이콘·커서·텍스트 전부 포함)에 whole_window_alpha(0~255,
    255=완전 불투명)를 균일하게 곱한다 — "모든 것"이 요청대로 함께 옅어진다.

    ★진짜 픽셀별 알파 블렌딩(UpdateLayeredWindow)이 아니라 균일 배율이라는
    한계는 남는다 — 받침판만 더 짙게/버튼만 더 옅게 같은 요소별 차등은 안
    된다. 그 대신 구현이 훨씬 단순·안전하고(GDI 비트맵·premultiplied alpha
    등 다루지 않음), 매 프레임 렌더링 비용도 그대로다. --overlay-alpha로
    실행 중 값 조정 가능 — 너무 옅어 안 보이면 올리고, 배경이 덜 비치면 내릴 것.

    창은 항상 마우스 클릭을 받는 상태로 둔다(WS_EX_TRANSPARENT/클릭통과는
    적용 안 함) — 그래야 q/ESC 종료가 그대로 먹는다.

    창이 아직 화면에 실제로 만들어지지 않은 시점에 찾으면 실패하므로, 호출
    전에 이미 한 번 이상 cv2.imshow + cv2.waitKey를 거쳐야 한다."""
    if os.name != "nt":
        logger.warning("Transparent overlay is Windows-only - skipping (not on Windows)")
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            logger.warning("Transparent overlay failed - could not find window %r", window_name)
            return
        gwl_exstyle = -20
        ws_ex_layered = 0x00080000
        lwa_colorkey = 0x00000001
        lwa_alpha = 0x00000002
        exstyle = user32.GetWindowLongW(hwnd, gwl_exstyle)
        user32.SetWindowLongW(hwnd, gwl_exstyle, exstyle | ws_ex_layered)
        b, g, r = key_color_bgr
        colorref = r | (g << 8) | (b << 16)   # Windows COLORREF = 0x00BBGGRR
        alpha = max(0, min(255, int(whole_window_alpha)))
        user32.SetLayeredWindowAttributes(hwnd, colorref, alpha, lwa_colorkey | lwa_alpha)
    except Exception:   # noqa: 방어적 — 오버레이 설정 실패가 시작을 막으면 안 된다
        logger.exception("Transparent overlay setup failed - continuing with a normal opaque window")


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
    """D-pad 5개 존(up/down/left/right/center) -> {name: (cx_px, cy_px, radius_px)}."""
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
    """D-pad 중심 -> 커서 벡터 -> "up"/"down"/"left"/"right" | None.

    거리가 아니라 **각도만** 본다 — 손/고개가 버튼 원까지 정확히 안 닿아도 그
    방향을 향하기만 하면 인식되게 하기 위함. 대각선처럼 두 축이 비슷하면(축
    우세 미달) None을 돌려줘 호출부가 직전 방향을 유지하게 한다."""
    abs_dx_px, abs_dy_px = abs(dx_px), abs(dy_px)
    if abs_dx_px >= abs_dy_px * AXIS_DOMINANCE:
        return "right" if dx_px > 0 else "left"
    if abs_dy_px >= abs_dx_px * AXIS_DOMINANCE:
        return "down" if dy_px > 0 else "up"
    return None


def update_shape_latch(latched_shape, streak_value, streak_count, raw_shape):
    """손 모양 판별 출렁임을 걸러 프레임 하나짜리 오판이 그대로 새지 않게 한다.

    무래치 -> 고정은 같은 판별 SHAPE_LATCH_FRAMES연속, 고정 -> 다른 모양 전환은
    다른 판별 SHAPE_SWITCH_FRAMES연속이 필요 (본 엔진 gesture_filter.py의
    shape_latch와 같은 발상). 순수 함수."""
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

    0도=일직선(폄), 클수록 굽음. extend_ratio(손끝-손목 **거리**비)와 달리 두
    인접 마디 사이의 **국소** 각도라, 손가락이 카메라 쪽 어느 방향을 향하든
    상대적으로 덜 흔들린다는 것이 이 교차검증의 전제다."""
    vec_a = world_landmarks[pip] - world_landmarks[mcp]
    vec_b = world_landmarks[tip] - world_landmarks[pip]
    norm_a, norm_b = float(np.linalg.norm(vec_a)), float(np.linalg.norm(vec_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    cos_angle = max(-1.0, min(1.0, float(np.dot(vec_a, vec_b)) / (norm_a * norm_b)))
    return math.degrees(math.acos(cos_angle))


def crosscheck_fist_joints(shape, world_landmarks):
    """주먹 판정을 손가락 관절 각도로 교차검증.

    "주먹"인데 관절상으로는 쫙 편 손가락이 MIN_STRAIGHT_FINGERS_TO_VETO개
    이상이면 그 판정을 취소(None, 불명)한다 — 카메라 정면을 향한 검지 하나가
    extend_ratio 오판으로 주먹에 섞여 들어오는 경우를 걸러낸다. finger/open
    판정은 그대로 둔다 — 이미 폄 신호가 있어 이 교차검증이 필요 없다."""
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
    """손 모양 판정 계기판(좌하단) — RAW(교차검증 전 원시 판별)·ANGLES(손가락별 PIP
    관절 각도)·LATCH(화면·발화에 실제로 쓰이는 최종값)를 한 줄씩 보여준다. 손
    모드에서만 채워진다(머리 모드엔 대응 개념이 없다)."""
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
              flash_zone, flash_progress, wireframe=False):
    """D-pad 전체(받침판 + 5개 버튼 + 아이콘 + 진행 링 + 반짝임)를 frame에 그린다.

    wireframe=True(--overlay 전용, 2026-08-05 사용자 요청 — "dpad 색 다
    없애줘 투명하게 뒤에 바탕화면 보이게", 이어서 "dpad 반투명 흰색으로
    보여줘"): 받침판·버튼 채우기(면 색)를 전부 생략한다 — 그 자리는
    --overlay의 색상 키 배경 그대로 남아 완전히 투명해진다(투명 창 자체는
    enable_transparent_overlay가 담당, 여기선 "칠하지 않는다"만 책임진다).
    테두리·아이콘은 OVERLAY_DPAD_COLOR(흰색)로 그려 면 없이도 존 경계가
    또렷이 보이게 하고, 확정 순간의 반짝임은 흰 바탕과 구분되도록
    OVERLAY_CURSOR_COLOR(초록, 커서와 같은 색)로 테두리를 바꾼다. 진행 링은
    평소와 동일(COLOR_PROGRESS) — 실질 정보라 그대로 둔다. "반투명"은 창
    전체 알파(--overlay-alpha, enable_transparent_overlay 참고)가 담당 —
    이 함수는 색만 정할 뿐 흐림 정도는 모른다(원소별 차등 알파는 색상 키
    방식의 한계로 아직 지원 안 함)."""
    cx_px, cy_px, _ = zones["center"]
    pad_px = int(petal_r_px * 0.4)
    half_px = offset_px + petal_r_px + pad_px
    plate_radius_px = int(petal_r_px * 0.55)

    if not wireframe:
        overlay = frame.copy()
        draw_rounded_rect(overlay, (cx_px - half_px, cy_px - half_px),
                           (cx_px + half_px, cy_px + half_px), plate_radius_px, COLOR_BG)
        cv2.addWeighted(overlay, PLATE_ALPHA, frame, 1 - PLATE_ALPHA, 0, frame)

    icon_color = OVERLAY_DPAD_COLOR if wireframe else COLOR_ACCENT
    for name in ("up", "down", "left", "right", "center"):
        zx, zy, zr = zones[name]
        is_flashing = name == flash_zone and flash_progress > 0
        if wireframe:
            border_color = OVERLAY_CURSOR_COLOR if is_flashing else OVERLAY_DPAD_COLOR
            cv2.circle(frame, (zx, zy), zr, border_color, 3, cv2.LINE_AA)
        else:
            color = COLOR_BG_HOVER if name == hover_zone else COLOR_BG
            if is_flashing:
                color = tuple(int(base * (1 - flash_progress) + flash * flash_progress)
                              for base, flash in zip(color, COLOR_FLASH))
            cv2.circle(frame, (zx, zy), zr, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (zx, zy), zr, COLOR_BORDER, 2, cv2.LINE_AA)
        icon_thickness = max(3, zr // 8)
        if name == "center":
            draw_pause_icon(frame, zx, zy, int(zr * 0.42), icon_color, icon_thickness)
        else:
            draw_chevron(frame, zx, zy, name, int(zr * 0.5), icon_color, icon_thickness)
        if name == hover_zone and hover_progress > 0:
            end_angle_deg = 360.0 * hover_progress
            cv2.ellipse(frame, (zx, zy), (zr + 6, zr + 6), -90, 0, end_angle_deg,
                        COLOR_PROGRESS, 4, cv2.LINE_AA)
    return frame


def build_zone_functions():
    """존 이름 -> 실행 함수(shape_label: str 인자 하나). 지금은 자리표시자(발화될
    때마다 콘솔에 한 줄 + 화면 "FIRED:" 갱신)만 한다 — 실제 기능이 정해지면 각
    함수 본문을 원하는 동작으로 바꿔 끼우면 된다(예: event_sender.send(...)로
    기존 이벤트 파이프에 합류시키기). shape_label은 "FINGER"/"FIST"/"OPEN"/
    "HEAD"/"UNKNOWN" 중 하나 — 같은 방향이라도 손 모양·입력 모드에 따라 다르게
    반응하고 싶으면 이 값으로 분기하면 된다.
    콘솔 출력 방침(2026-08-04 사용자 요청): 매 프레임 연속으로 찍는 xy 좌표와
    별개로, 버튼이 실제 발화하는 순간엔 5개 존 각각 구분되는 한 줄을 낸다 —
    "둘 다"(xy 연속값 + 발화 알림)를 함께 요구한 것이라 여기서만 print한다
    (SHAPE-DEBUG·모드 전환처럼 매 프레임 반복되는 진단 로그는 계속 제거된
    상태). 화면 "FIRED:" 라벨과 같은 문자열을 그대로 찍어 콘솔·화면이 항상
    일치한다."""
    state = {"is_playing": False}

    def on_up(shape_label):
        label = f"UP+{shape_label}"
        print(f"FIRED: {label}")
        return label

    def on_down(shape_label):
        label = f"DOWN+{shape_label}"
        print(f"FIRED: {label}")
        return label

    def on_left(shape_label):
        label = f"LEFT+{shape_label}"
        print(f"FIRED: {label}")
        return label

    def on_right(shape_label):
        label = f"RIGHT+{shape_label}"
        print(f"FIRED: {label}")
        return label

    def on_center(shape_label):
        state["is_playing"] = not state["is_playing"]
        play_label = "PLAY" if state["is_playing"] else "PAUSE"
        label = f"{play_label}+{shape_label}"
        print(f"FIRED: {label}")
        return label

    return {"up": on_up, "down": on_down, "left": on_left, "right": on_right, "center": on_center}


def _face_nose_signal(face):
    """FaceLandmarks -> (nose_px, interocular_dist_px) | (None, 0.0) — head_shake 입력.

    realtime_loop.py의 동명 함수와 동일 로직 — 헤드트래커 통합은 본 엔진과
    최대한 같은 코드를 쓴다(중복 재구현 방지)."""
    if face is None:
        return None, 0.0
    nose_px = face.landmark_px(LMK_NOSE_TIP)
    left_px = face.landmark_px(LMK_LEFT_EYE_OUTER)
    right_px = face.landmark_px(LMK_RIGHT_EYE_OUTER)
    interocular_dist_px = ((left_px[0] - right_px[0]) ** 2 + (left_px[1] - right_px[1]) ** 2) ** 0.5
    return nose_px, interocular_dist_px


def main():
    parser = argparse.ArgumentParser(
        description="gesture_kiosk 공식 D-pad UI — 손끝/고개로 방향 존을 향해 함수를 실행한다")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=int, default=None, help="카메라 장치 번호 (기본: config device_id)")
    parser.add_argument("--dwell-sec", type=float, default=DWELL_SEC_DEFAULT,
                        help="존에 머물러야 확정되는 시간(초)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="창을 전체화면으로 띄운다 (키오스크 시연용 — 종료는 q/ESC 그대로)")
    parser.add_argument("--cx-ratio", type=float, default=0.50, help="D-pad 중심 X (화면 폭 비율) — 기본 정중앙")
    parser.add_argument("--cy-ratio", type=float, default=0.30, help="D-pad 중심 Y (화면 높이 비율) — 기본 상단")
    parser.add_argument("--extend-ratio", type=float, default=DEMO_EXTEND_RATIO,
                        help="손가락 '폄' 판정 기준 — 낮출수록 잘 펴진다(DEMO_EXTEND_RATIO 참고)")
    parser.add_argument("--min-cutoff-hz", type=float, default=FILTER_MIN_CUTOFF_HZ,
                        help="손끝 필터 정지 시 컷오프(Hz) — 낮을수록 떨림 억제가 강해지는 대신 "
                             "반응이 굼떠진다(FILTER_MIN_CUTOFF_HZ 참고)")
    parser.add_argument("--filter-beta", type=float, default=FILTER_BETA,
                        help="손끝 필터 속도 반응 계수 — 낮을수록 순간 잡음을 빠른 움직임으로 덜 "
                             "오인해 떨림이 줄지만, 실제 빠른 손짓엔 살짝 더 지연된다(FILTER_BETA 참고)")
    parser.add_argument("--cursor-sensitivity", type=float, default=CURSOR_SENSITIVITY,
                        help="손끝 오프셋 -> 커서 오프셋 배율 — 높을수록 조금만 움직여도 커서가 "
                             "크게 움직이지만, 필터를 통과한 잔여 떨림도 그만큼 커진다"
                             "(CURSOR_SENSITIVITY 참고)")
    parser.add_argument("--no-head-mode", action="store_true",
                        help="config mode_switch.enabled와 무관하게 헤드트래커 통합을 끈다(손 전용으로 실행)")
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--blank-bg", action="store_true",
                        help="카메라 영상 대신 빈 배경 위에 D-pad·커서만 그린다 — 실제 키오스크 "
                             "화면에 뭐가 보일지 미리보기하는 용도(추론은 항상 실제 카메라 프레임을 씀)")
    display_group.add_argument("--overlay", action="store_true",
                        help="빈 배경 대신 진짜 바탕화면(또는 뒤 창)이 비쳐 보이는 반투명 창으로 "
                             "D-pad·커서만 띄운다 — Windows 전용, enable_transparent_overlay 참고")
    parser.add_argument("--overlay-alpha", type=int, default=170,
                        help="--overlay일 때 D-pad·커서(배경 제외)의 불투명도, 0(완전 투명)~255"
                             "(완전 불투명) — 기본 170(반투명). 창 전체에 균일하게 적용되어 "
                             "D-pad와 커서가 같이 옅어진다(색상 키 방식의 한계 — 원소별 차등 불가)")
    args = parser.parse_args()

    disable_console_quick_edit()
    config = load_config(args.config)
    init_logging(config)
    shape_cfg = config["hand_select"]["hand_shape"]   # min_valid_fingers·curl_confirm_ratio는
                           # 본 엔진 값 그대로 재사용 — extend_ratio만 이 데모 전용으로 덮어씀
    if args.device is not None:
        config["camera"]["device_id"] = args.device
        auto_select = config["camera"].get("auto_select")
        if auto_select is not None:
            config["camera"]["auto_select"] = {**auto_select, "enabled": False}

    # mode_switch 섹션이 없거나 enabled: false거나 --no-head-mode면 얼굴 관련 객체를
    # 아예 만들지 않는다 — realtime_loop.py와 동일 관례(얼굴 코드 경로 완전 미실행)
    mode_switch_enabled = (config.get("mode_switch") or {}).get("enabled", False) and not args.no_head_mode

    preprocessor = Preprocessor(config)
    hand_tracker = HandTracker(config)
    face_estimator = FaceEstimator(config) if mode_switch_enabled else None
    # "뒷사람" 방어 — head.py/eyebrow.py와 동일 이유(src/postprocess/face_anchor.py
    # 독스트링 참고). 이 파일은 머리 모드 커서(head_tracker.update, 아래)뿐 아니라
    # 손↔머리 모드 전환 감지(head_shake)에도 같은 얼굴을 쓴다 — 둘 다 여기서
    # 이어받는다
    face_anchor = FaceAnchor(config) if mode_switch_enabled else None
    head_shake = HeadShakeDetector(config) if mode_switch_enabled else None
    head_tracker = HeadTracker(config) if mode_switch_enabled else None
    # 모델 로딩 완료를 연동 GUI(델파이)에 알린다 — 팀장님 요청(2026-08-31).
    # 델파이는 엔진을 자식 프로세스로 띄우고 stdout을 익명 파이프로 줄 단위
    # 수신하며 준비 완료를 기다린다 (docs/델파이7_연동가이드.md 참고).
    #
    # ★flush가 반드시 필요하다. 파이프로 나가는 stdout은 줄 단위가 아니라
    # 블록 단위(약 8KB)로 버퍼링된다 — 그냥 print만 하면 이 한 줄이 버퍼에
    # 갇혀 델파이가 못 받는다. 한참 뒤 좌표가 쌓여 버퍼가 찰 때 함께 밀려
    # 나가는데, 그때는 "준비됐다"는 신호로서 이미 쓸모가 없다.
    # (이 프로젝트는 -u 나 PYTHONUNBUFFERED 를 쓰지 않는다 — 확인함)
    #
    # 카메라를 열기 **전**에 찍는다. 요청은 "모델 로딩 완료"이고, 카메라
    # 오픈은 정상 장치도 십수 초 걸리는 경우가 있어(2026-08-26 실측) 그것까지
    # 기다리면 신호가 그만큼 늦어진다.
    print("Models Loaded", flush=True)

    camera = CameraStream(config, config_path=args.config).start()

    fingertip_filter = PointFilter(args.min_cutoff_hz, args.filter_beta, FILTER_D_CUTOFF_HZ)
    last_direction = None
    latched_shape = None
    shape_streak_value, shape_streak_count = None, 0
    neutral_px = None
    still_anchor_px = None
    still_start_sec = 0.0
    still_fired = False
    had_hand_prev = False
    hover_zone, hover_start_sec, armed = None, 0.0, True
    flash_zone, flash_start_sec = None, 0.0
    last_action_label = ""
    input_mode = MODE_HAND   # 항상 손 모드로 시작 (PipelineState.input_mode 기본값과 동일)
    zone_funcs = build_zone_functions()

    if args.fullscreen:
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if args.overlay:
        if not args.fullscreen:
            cv2.namedWindow(WINDOW_NAME)
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)   # 바탕화면·다른 창 위로 항상 떠 있게
        except Exception:   # noqa: 방어적 — 이 OpenCV 빌드가 TOPMOST를 지원 안 해도 계속 진행
            logger.warning("Could not set window topmost - continuing anyway")
        # 창이 실제로 화면에 만들어질 때까지 한 번 그려야 FindWindowW로 찾을 수
        # 있다(enable_transparent_overlay 독스트링 참고) — 실제 D-pad 내용은
        # 아래 루프 첫 프레임에서 다시 그려 덮어씀, 이건 순전히 창 생성용
        cv2.imshow(WINDOW_NAME, np.full((10, 10, 3), TRANSPARENT_KEY_COLOR, dtype=np.uint8))
        cv2.waitKey(1)
        enable_transparent_overlay(WINDOW_NAME, TRANSPARENT_KEY_COLOR, args.overlay_alpha)

    logger.warning("D-pad UI started - top-center D-pad. Point toward a direction for %.2fs "
                   "to fire it. Hold your hand still for %.1fs to recalibrate. Head mode: %s "
                   "(shake head left-right to toggle input). Quit: q or ESC",
                   args.dwell_sec, STILL_RECALIBRATE_SEC, "ON" if mode_switch_enabled else "OFF")
    logger.info("Startup %.1fs (interpreter + imports + models)", time.monotonic() - PROCESS_START_SEC)

    try:
        while True:
            # 프레임 크롭도 모드별로 다르다 — 기본은 머리 모드일 때만 9:16 세로
            # 크롭(config camera.portrait_crop, realtime_loop.py와 동일 조건식).
            # 다만 --overlay면 모드와 무관하게 항상 9:16으로 고정한다(2026-08-05
            # 사용자 요청: "dpad overlay만 헤드모드 핸드모드 둘다 비율 9:16으로
            # 바꿔주라") — 오버레이는 실제(세로) 키오스크 화면을 흉내 내는
            # 용도라, 화면 모양 자체가 입력 모드 전환으로 바뀌면 안 되기
            # 때문이다. --blank-bg·카메라 프리뷰(기본)는 종전 그대로 손 모드에서
            # 원본 16:9. 이번 프레임 시작 시점의 input_mode를 쓴다 — 전환 자체가
            # 이 프레임 아래에서 판정되므로 전환 프레임 당일은 한 프레임 지연
            # (본 엔진도 같은 구조)
            camera_frame = preprocessor.preprocess_frame(
                camera.capture_frame(), apply_crop=(args.overlay or input_mode == MODE_HEAD))
            # 화면에 그려 보여줄 캔버스 — --blank-bg면 빈 배경, --overlay면
            # 크로마키 색(뒤에 실제 바탕화면이 비쳐 보임, enable_transparent_
            # overlay 참고)으로 채운다(2026-08-05 사용자 요청: "헤드 모드
            # 커서랑 핸드모드 커서 그리고 dpad를 띄워서 윈도우에서 되게" →
            # 후속 "빈 화면 말고 뒤에 바탕화면 보이게" — 카메라 프리뷰 없이
            # 실제 키오스크 화면에 뭐가 보일지 미리 보고 싶다는 취지). 아래
            # draw_*·cv2.putText·imshow는 전부 이 frame에 그린다 —
            # camera_frame(실제 카메라 픽셀)은 추론 전용으로 따로 유지, 판정
            # 경로는 화면 표시 방식과 무관하게 완전히 동일하다
            if args.overlay:
                frame = np.full_like(camera_frame, TRANSPARENT_KEY_COLOR)
            elif args.blank_bg:
                frame = np.full_like(camera_frame, BLANK_BG_COLOR)
            else:
                frame = camera_frame
            now_sec = time.monotonic()
            h_px, w_px = frame.shape[:2]

            # 얼굴 추론(모드 전환 감지)은 모드 무관 상시 실행 — 손 모드 중에도
            # 흔들면 머리 모드로 전환해야 한다 (realtime_loop.py와 동일 이유)
            user_face = None
            if mode_switch_enabled:
                faces = face_estimator.infer(camera_frame)
                user_face = face_anchor.update(faces)
                nose_px, interocular_dist_px = _face_nose_signal(user_face)
                if head_shake.update(nose_px, interocular_dist_px):
                    input_mode = MODE_HEAD if input_mode == MODE_HAND else MODE_HAND
                    if input_mode == MODE_HEAD:
                        head_tracker.reset()   # 새 모드 진입 — 캘리브레이션부터 다시
                    else:
                        had_hand_prev = False   # 다음 손 등장을 "새 등장"으로 취급 -> 기존
                                                # 재정렬 로직(neutral_px 재캡처)이 그대로 태워진다

            if input_mode == MODE_HAND:
                # --- 손 모드: D-pad 전체 (기존 로직 그대로) ---
                cursor_offset_px = None   # (dx_px, dy_px) — D-pad 중심 기준
                shape_label = "UNKNOWN"
                fingertip_px = None
                raw_shape_before_veto = None
                debug_finger_angles_deg = None

                hands = hand_tracker.infer(camera_frame)
                if hands:
                    # 데모 단순화: 신뢰도가 가장 높은 손 하나만 본다
                    hand = max(hands, key=lambda h: h.conf)
                    is_new_appearance = not had_hand_prev
                    if is_new_appearance:
                        fingertip_filter.reset()
                        last_direction = None
                        latched_shape, shape_streak_value, shape_streak_count = None, None, 0
                    had_hand_prev = True

                    raw_tip = (float(hand.landmarks[INDEX_FINGERTIP_LMK][0]),
                              float(hand.landmarks[INDEX_FINGERTIP_LMK][1]))
                    fingertip_px = fingertip_filter.filter(raw_tip, now_sec)
                    if is_new_appearance:
                        # 손이 없다가 다시 나타날 때마다(최초 등장 포함) 무조건 그
                        # 자리를 새 기준점으로 — 이전 사용자·이전 위치의 보정을 물려받지 않는다
                        neutral_px = fingertip_px

                    raw_shape_before_veto = classify_hand_shape(
                        hand.world_landmarks, args.extend_ratio,
                        shape_cfg["min_valid_fingers"], shape_cfg["curl_confirm_ratio"],
                    )
                    raw_shape = crosscheck_fist_joints(raw_shape_before_veto, hand.world_landmarks)
                    latched_shape, shape_streak_value, shape_streak_count = update_shape_latch(
                        latched_shape, shape_streak_value, shape_streak_count, raw_shape,
                    )
                    debug_finger_angles_deg = [
                        _finger_joint_angle_deg(hand.world_landmarks, mcp, pip, tip)
                        for mcp, pip, _dip, tip in HAND_FINGERS
                    ]

                    shape_label = SHAPE_TAG.get(latched_shape, "UNKNOWN")
                    reference_px = neutral_px if neutral_px is not None else (w_px / 2.0, h_px / 2.0)
                    dx_px = (fingertip_px[0] - reference_px[0]) * args.cursor_sensitivity
                    dy_px = (fingertip_px[1] - reference_px[1]) * args.cursor_sensitivity
                    cursor_offset_px = (dx_px, dy_px)
                else:
                    had_hand_prev = False
                    latched_shape, shape_streak_value, shape_streak_count = None, None, 0

                # 정지 감지 — 손끝이 STILL_RADIUS_RATIO 반경 안에 STILL_RECALIBRATE_SEC
                # 이상 머물면 자동 재정렬 (버튼 없이 "허공에 가만히"만으로 트리거)
                if fingertip_px is None:
                    still_anchor_px = None
                else:
                    still_radius_px = STILL_RADIUS_RATIO * min(w_px, h_px)
                    if (still_anchor_px is None
                            or math.hypot(fingertip_px[0] - still_anchor_px[0],
                                          fingertip_px[1] - still_anchor_px[1]) > still_radius_px):
                        still_anchor_px, still_start_sec, still_fired = fingertip_px, now_sec, False
                    elif not still_fired and now_sec - still_start_sec >= STILL_RECALIBRATE_SEC:
                        neutral_px = fingertip_px
                        still_fired = True
                        last_action_label = "RECALIBRATED"

                # D-pad는 화면 비율 고정 위치·고정 크기 — 손 유무와 무관하게 항상 그 자리
                center_px = (args.cx_ratio * w_px, args.cy_ratio * h_px)
                zones, offset_px, petal_r_px = compute_zones(center_px, min(w_px, h_px))

                cursor_px = None
                zone = None
                if cursor_offset_px is not None:
                    dx_px, dy_px = cursor_offset_px
                    cursor_px = (center_px[0] + dx_px, center_px[1] + dy_px)
                    # 콘솔에는 xy 좌표만 매 프레임(연속값) 찍는다(2026-08-04 사용자
                    # 요청 — SHAPE-DEBUG·fired 로그는 전부 제거하고 이것만 남긴다).
                    # 머리 모드와 같은 단위(화면 비율 0~1)로 맞춰 둘을 직접 비교할 수
                    # 있게 한다 — 화면 표시(cursor_px, 픽셀)는 그대로, 콘솔만 정규화
                    print(f"x={cursor_px[0] / w_px:.3f} y={cursor_px[1] / h_px:.3f}")
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
                        last_action_label = zone_funcs[hover_zone](shape_label)

                flash_progress = 0.0
                if flash_zone is not None:
                    flash_elapsed_sec = now_sec - flash_start_sec
                    if flash_elapsed_sec >= FLASH_SEC:
                        flash_zone = None
                    else:
                        flash_progress = 1.0 - flash_elapsed_sec / FLASH_SEC

                frame = draw_dpad(frame, zones, offset_px, petal_r_px, hover_zone, hover_progress,
                                  flash_zone, flash_progress, wireframe=args.overlay)
                # --overlay면 진단용 계기판·라벨은 전부 뺀다(2026-08-05 사용자
                # 요청 — "dpad만 뜨게 하고 나머지는 안보이게 해줘"). D-pad
                # 본체·커서(방향·확정에 실제로 필요한 정보)만 남긴다
                if not args.overlay and debug_finger_angles_deg is not None:
                    frame = draw_shape_debug(frame, w_px, h_px, raw_shape_before_veto, latched_shape,
                                             debug_finger_angles_deg)
                if cursor_px is not None:
                    cx_px, cy_px = int(cursor_px[0]), int(cursor_px[1])
                    # --overlay면 커서를 초록으로(2026-08-05 사용자 요청 — 머리 모드
                    # 커서(visualize.draw_cursor)와 색을 맞춘 OVERLAY_CURSOR_COLOR)
                    cursor_color = OVERLAY_CURSOR_COLOR if args.overlay else COLOR_FINGERTIP
                    cv2.circle(frame, (cx_px, cy_px), 10, cursor_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx_px, cy_px), 2, cursor_color, -1, cv2.LINE_AA)
                    # 손 모양 영문 라벨(FINGER/FIST/OPEN)은 --overlay면 뺀다(2026-08-05
                    # 사용자 요청: "커서 옆에 finger fist 이런 영어는 안나오게 해줘") —
                    # 진행 링(도형)은 그대로 두고 글자만 없앤다
                    if not args.overlay:
                        cv2.putText(frame, shape_label, (cx_px + 14, cy_px + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FINGERTIP, 2, cv2.LINE_AA)
                    if still_anchor_px is not None and not still_fired:
                        still_progress = min(1.0, (now_sec - still_start_sec) / STILL_RECALIBRATE_SEC)
                        if still_progress > 0:
                            end_angle_deg = 360.0 * still_progress
                            cv2.ellipse(frame, (cx_px, cy_px), (18, 18), -90, 0, end_angle_deg,
                                        COLOR_PROGRESS, 3, cv2.LINE_AA)
                            if not args.overlay:
                                cv2.putText(frame, "HOLD STILL TO RECALIBRATE", (cx_px + 14, cy_px + 24),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PROGRESS, 2, cv2.LINE_AA)

                if not args.overlay:
                    cv2.putText(frame, "D-PAD UI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, COLOR_TEXT, 2, cv2.LINE_AA)
                    if last_action_label:
                        cv2.putText(frame, f"FIRED: {last_action_label}", (10, 62),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ACCENT, 2, cv2.LINE_AA)
            else:
                # --- 머리 모드: D-pad 없음 — 본 엔진 헤드트래커 화면을 그대로 재사용 ---
                fingertip_px = None   # 아래 "c" 단축키 가드용 — 머리 모드에선 항상 비활성
                result = head_tracker.update(user_face)
                # draw_cursor(코끝 커서)는 --overlay에서도 유지 — 방향·확정에 실제로
                # 쓰이는 정보다. draw_head_debug_panel·좌표 텍스트는 진단용이라
                # --overlay면 뺀다(2026-08-05 사용자 요청 — 위 손 모드와 동일 방침)
                frame = draw_cursor(frame, result.cursor_x_ratio, result.cursor_y_ratio,
                                    head_tracker.debug.get("recenter_progress", 0.0))
                if not args.overlay:
                    frame = draw_head_debug_panel(frame, head_tracker.debug)
                    if result.cursor_x_ratio is None:
                        cursor_text = "CURSOR -- (calibrating)"
                    else:
                        cursor_text = f"CURSOR x={result.cursor_x_ratio:.3f} y={result.cursor_y_ratio:.3f}"
                    cv2.putText(frame, cursor_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, COLOR_TEXT, 2, cv2.LINE_AA)
                # 콘솔: xy는 매 프레임(연속값), 발화는 select/home/calibration이
                # 실제로 뜨는 순간에만 — 손 모드의 xy 연속값 + FIRED 발화 알림과
                # 같은 방침(2026-08-04 사용자 요청: "핸드 모드는 함수값 다 뜨는데
                # 헤드 모드는 select calibration home 이런거 다 어디갔어" — 이전에
                # SHAPE-DEBUG와 함께 지운 [HEAD] echo를 손 모드와 같은 "FIRED:"
                # 형식으로 복원). 손 모드 콘솔과 같은 단위(화면 비율 0~1,
                # cursor_x/y_ratio 자체가 이미 그 단위)
                if result.cursor_x_ratio is not None:
                    print(f"x={result.cursor_x_ratio:.3f} y={result.cursor_y_ratio:.3f}")
                for event in result.events:
                    print(f"FIRED: {event.class_name.upper()}")

            if mode_switch_enabled and not args.overlay:
                frame = draw_mode_banner(frame, input_mode)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and fingertip_px is not None:
                neutral_px = fingertip_px   # 수동 단축키 — 정지 대기 없이 즉시 재정렬(테스트 편의용)
                still_fired = True          # 방금 잡은 자리를 정지 앵커로도 인정 — 곧바로 재발화 방지
    finally:
        camera.stop()
        if face_estimator is not None:
            face_estimator.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
