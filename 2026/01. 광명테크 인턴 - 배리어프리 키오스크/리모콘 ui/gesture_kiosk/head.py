"""헤드트래커 단독 실행기 — 실제 OS 마우스 포인터를 제어한다 (2026-08-13 신설,
사용자 요청: "헤드트래커 기능만 따로 빼서 head.py랑 head.exe 만들어주라").

main_dpad.py의 헤드트래커 모드(입력 로직은 src/postprocess/head_tracker.py)를
그대로 재사용하되, 이 파일은 D-pad·손 모드·모드 전환(head_shake)을 전부 빼고
머리 트래킹 하나만 남긴 단독 실행기다. main_dpad.py처럼 화면 위에 커서를
"그리기만" 하지 않고, Win32 API(SetCursorPos/mouse_event, ctypes — 프로젝트
관례상 pyautogui 등 새 의존 추가하지 않음. enable_transparent_overlay와 동일
방식)로 **실제 OS 마우스 포인터를 이 위치로 옮기고, select 이벤트(입 벌리기·
1.5초 응시)에서 실제 좌클릭까지 낸다** — 그래서 바탕화면 파일 위에 커서를
가져가 select를 발화하면 실제로 그 파일이 클릭된다.

★세로 절반 제한(2026-08-13 사용자 요청 — "9:16 전체 화면 비율에서 세로 절반
위쪽만 커서가 움직이게"): head_tracker의 cursor_y_ratio(0~1, 캘리브레이션
중심 기준 상하 고개 움직임 전체 범위)를 화면 픽셀로 매핑하기 직전에 0.5를
곱한다(CURSOR_Y_SPAN) — 절반 지점에서 그냥 잘라내는 클램프가 아니라
**전체 고개 상하 가동범위를 화면 위쪽 절반 전체에 다시 늘려 매핑**하는
방식이다. 클램프만 하면 고개를 절반 이상 숙였을 때 커서가 그냥 경계에
멈춰버려 나머지 가동범위가 낭비되지만, 이 방식은 평소 쓰던 고개 움직임
그대로 위쪽 절반을 전부 쓸 수 있다. 가로(X)는 원래 그대로 화면 전체 폭을 쓴다
(제한 요청이 X에는 없었음).

카메라 프레임은 머리 모드 관례대로 항상 9:16 세로로 크롭한다(config
camera.portrait_crop — main_dpad.py 머리 모드와 동일 조건, 이 파일은 애초에
머리 모드 전용이라 조건 분기 없이 항상 적용).

안전장치: 시작 직후 곧바로 실제 마우스를 뺏으면 당황스러울 수 있어, 콘솔에
"p" + Enter로 실제 OS 제어를 켜고 끌 수 있다(is_control_active 토글) — 꺼도
오버레이 커서 표시는 계속 갱신되어 캘리브레이션 상태는 그대로 볼 수 있다.
기본값은 켜짐(요청대로 "마우스 포인터랑 같이 움직여서 클릭되게").

★오버레이 표시(2026-08-13 사용자 요청 — "카메라 안보여도되는데 오버레이로
바탕화면에 뜨게 해줘", main_dpad.py --overlay와 동일 기법 재사용): 카메라
영상을 보여줄 필요가 없다는 요청이라, 창에는 카메라 프레임을 전혀 그리지
않는다. 대신 main_dpad.py의 enable_transparent_overlay와 완전히 같은
방식으로 — cv2/HighGUI가 지원하지 않는 창 투명도를 창 생성 후 Win32 API
(SetLayeredWindowAttributes, LWA_COLORKEY + LWA_ALPHA)로 직접 걸어 화면
전체 크기의 완전 투명(바탕화면이 그대로 비침) 풀스크린 창을 띄우고, 그 위에
커서 점 하나만 그린다. 창 크기를 카메라 프레임이 아니라 **실제 화면
해상도**(_Win32Mouse.screen_w/h_px)로 잡는다 — 그래야 오버레이에 그려지는
커서 점 위치와 SetCursorPos로 옮기는 진짜 OS 커서 위치가 화면상 같은 자리에
겹쳐 보인다. dpad_overlay처럼 계기판·좌표 텍스트 등 진단용 요소는 넣지
않고 커서(+응시 진행 링)만 남긴다.

사용법 (프로젝트 루트에서):
    py head.py [--device N] [--config path] [--no-window]
종료: 오버레이 창에서 q/ESC, 또는 콘솔에 quit + Enter
실제 마우스 제어 토글: 콘솔에 p + Enter
"""
import argparse
import atexit
import ctypes
import math
import os
import signal
import sys
import threading
import time

PROCESS_START_SEC = time.monotonic()   # main.py와 동일 관례 — 시작 시간 계측 기점
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)   # main.py와 동일 이유 — 어디서 띄우든 config·모델 상대 경로가 성립

def _make_console_crash_proof():
    """★출력이 프로그램을 죽이지 못하게 한다 (2026-08-25 신설).

    한글 윈도우의 기본 출력 인코딩은 cp949인데, 여기엔 줄표(—)·화살표 같은
    문자가 아예 없다. 그런 글자를 찍으려 하면 UnicodeEncodeError가 나고
    **프로그램이 시작하자마자 죽는다**. 실측으로 확인했다:

        py head.py --check
        UnicodeEncodeError: 'cp949' codec can't encode character '—'

    개발 중에는 `py -X utf8`로 돌려서 이 문제가 안 보였다 — 정작 현장에서
    그냥 실행하면 죽는, 가장 나쁜 종류의 차이였다.

    두 가지를 한다:
      · 콘솔이 아닌 곳(파이프·파일로 받아가는 경우)엔 UTF-8로 못박는다 —
        받는 쪽이 한글을 제대로 읽을 수 있다
      · 어느 경우든 못 찍는 글자는 '?'로 바꾸게 한다(errors="replace") —
        글자 하나 때문에 트래커가 죽는 일은 이제 없다

    콘솔일 때 인코딩을 바꾸지 않는 건 의도적이다. 창의 코드페이지가 cp949인데
    UTF-8로 써버리면 한글이 통째로 깨져 보인다.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:   # 콘솔 없는 실행(pythonw) 등 — 찍을 데가 없다
            continue
        try:
            is_console = bool(getattr(stream, "isatty", lambda: False)())
            if is_console:
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:   # noqa: 방어적 — 안전장치가 실패해도 실행은 계속
            pass


_make_console_crash_proof()

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.capture.camera_stream import CameraStream
from src.inference.face_estimator import (
    FaceEstimator, LMK_LEFT_EYE_OUTER, LMK_NOSE_CLUSTER, LMK_NOSE_TIP, LMK_RIGHT_EYE_OUTER,
)
from src.inference.preprocessor import Preprocessor
from src.postprocess.face_anchor import FaceAnchor
from src.postprocess.head_tracker import EVENT_SELECT, HeadTracker
from src.utils.config_loader import load_config
from src.utils import console, preflight, version
from src.utils.logger import get_logger, init_logging
# 카메라 미리보기 전용(2026-08-13 사용자 요청 — "cam on 하면 카메라 보이게") —
# main.py의 --debug/"cam on" 카메라 창과 같은 목적이라, 그 창에 쓰이는 원본
# draw_cursor·draw_head_debug_panel(카메라 해상도 기준 크기)을 그대로 가져다
# 쓴다. 오버레이용으로 이 파일 안에 새로 만든 더 큰 draw_cursor와는 이름이
# 겹쳐 카메라 쪽만 접두어를 붙여 구분한다
from src.utils.visualize import draw_cursor as draw_cursor_camera
from src.utils.visualize import draw_head_debug_panel

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

# 같은 오류가 계속될 때 다시 기록하기까지의 간격(초) — _inference_loop 설명 참고
ERROR_REPEAT_LOG_SEC = 30.0
# ★스레드 감시 (2026-08-25 신설, 사용자 요청 — "추론이 멈춰도 아무도 모른다").
#
# 무인 키오스크에서 가장 나쁜 고장은 "죽는 것"이 아니라 **조용히 멈추는 것**
# 이다. 프로그램은 살아 있고 창도 떠 있는데 커서만 반응하지 않으면, 앞에 선
# 사람은 자기가 잘못한 줄 알고 계속 시도하다 그냥 간다. 관리자도 로그를
# 열어보기 전에는 모른다.
#
# 두 가지를 따로 본다 — 원인이 다르면 할 일도 다르기 때문이다:
#   · 추론 루프가 한 바퀴도 못 돌고 있다  -> 프로그램 내부 문제
#   · 루프는 도는데 새 카메라 프레임이 없다 -> 카메라/USB 문제
# 화면에도 알리고 로그에도 남긴다. 로그는 간격을 둬서 폭주하지 않게 한다.
WATCHDOG_STALL_SEC = 5.0        # 이만큼 소식이 없으면 고장으로 본다
WATCHDOG_LOG_SEC = 30.0         # 같은 고장을 다시 기록하기까지의 간격
WATCHDOG_LABEL_INTERNAL = "프로그램에 문제가 생겼습니다 - 관리자에게 알려주세요"
WATCHDOG_LABEL_CAMERA = "카메라 신호가 없습니다 - 연결을 확인해주세요"
WATCHDOG_LABEL_COLOR = (0, 80, 255)   # 빨강(BGR) — 커서(초록)·안내(초록)와 뚜렷이 구분

WINDOW_NAME = "head tracker overlay"
CAMERA_WINDOW_NAME = "head tracker camera"

# ★ 2026-08-20 원인 규명 완료 — 키 확인은 이제 공짜다.
#
# 그동안 렌더 루프가 예산을 넘긴 가장 큰 원인이 cv2.waitKey(1) 하나였다.
# 이전 조사에서는 "다른 스레드가 CPU를 쓸 때만 15~16ms가 걸린다"고 적어둔데,
# 다시 재보니 **다른 스레드가 놀고 있어도 항상 15~16ms**였다. 원인도 GIL이
# 아니었다 — waitKey(1)은 창 메시지를 기다리며 실제로 "1ms 자려고" 하는데,
# 윈도우의 기본 타이머 눈금이 약 15.6ms라 1ms를 요청해도 한 눈금을 통째로
# 자버린다. 그래서 예외 없이 16ms가 나왔던 것이다.
#
# cv2.pollKey()는 기다리지 않고 밀린 창 메시지만 처리하고 즉시 돌아온다 —
# 하는 일(메시지 펌프 + 키 확인)은 waitKey와 같은데 실측 0.00ms다.
#
#   실측(1536x864 화면, OpenCV 4.10)
#     cv2.waitKey(1)  중앙 10.5ms / 최악 17.1ms
#     cv2.pollKey()   중앙  0.00ms / 최악  0.06ms
#
# 공짜가 됐으므로 "몇 틱에 한 번만 부르기"라는 예전 우회는 없쥜다 —
# 매 틱 부른다. 덕분에 q/ESC 반응도 다시 즉각적이다.
# pollKey는 OpenCV 4.5+에만 있어, 없으면 예전 방식으로 돌아간다
_HAS_POLL_KEY = hasattr(cv2, "pollKey")


def _pump_window_events():
    """창 메시지를 처리하고 눌린 키를 돌려준다(없으면 255). 위 설명 참고 —
    pollKey는 기다리지 않아 렌더 예산을 전혀 먹지 않는다."""
    return (cv2.pollKey() if _HAS_POLL_KEY else cv2.waitKey(1)) & 0xFF

# 세로 절반 위쪽 제한 — 모듈 독스트링 참고. 0.5 = 화면 상단 절반까지만 커서가 감
# 세로 커서 도달 범위 — 화면 세로 중 커서가 실제로 쓰는 구간.
#
# ★2026-08-27 사용자 요청 — "head.py·eyebrow.py 둘 다 밑에 화면 쓰게 바꿔줘".
# 실제 키오스크는 사용자가 화면 아래쪽을 주로 조작한다(휠체어 사용자·키가
# 작은 사용자에게도 아래쪽이 닿기 쉽다). **폭(0.5)은 그대로 두고 붙는 쪽만**
# 위 -> 아래로 뒤집었다 — 지금까지 실기로 맞춰 온 조작 감각은 그대로다.
#
#   CURSOR_Y_ANCHOR_BOTTOM = True   -> 화면 아래 절반 (세로 50%~100%)
#   CURSOR_Y_ANCHOR_BOTTOM = False  -> 화면 위  절반 (세로  0%~ 50%, 예전 동작)
#
# 폭과 방향을 나눠 뒀으니 전체 화면을 쓰려면 SPAN을 1.0으로 올리면 된다.
CURSOR_Y_SPAN = 0.5
CURSOR_Y_ANCHOR_BOTTOM = True
CURSOR_Y_OFFSET = (1.0 - CURSOR_Y_SPAN) if CURSOR_Y_ANCHOR_BOTTOM else 0.0


def _cursor_y_to_screen(cursor_y_ratio):
    """트래커 내부 세로 좌표(0~1) -> 실제 화면 세로 비율(0~1).

    커서 계산은 언제나 0~1 전체를 쓰고, 화면의 어느 구간에 앉힐지는 여기서만
    정한다 — 예전엔 `* TOP_HALF_Y_SCALE`이 코드 여러 곳에 흩어져 있어 한 곳만
    빠뜨려도 커서와 안내 문구가 서로 다른 자리에 놓였다.
    """
    return CURSOR_Y_OFFSET + cursor_y_ratio * CURSOR_Y_SPAN

# 렌더 스레드가 매 틱마다 "표시 위치 -> 목표 위치" 남은 거리의 이 비율만큼만
# 따라간다(선형 보간·"추격" 방식) — 2026-08-13 도입("화면 주사율 맞게 부드럽게
# 움직이게"), 반응성 문제로 값을 한 번 올림(아래 참고). 1.0이면 매번 목표로
# 순간 이동(보간 없음), 낮을수록 부드럽지만 반응이 처진다. head_tracker
# 자체의 EMA(smoothing_alpha)는 "원시 좌표의 떨림"을 줄이고, 이 값은 "그 결과를
# 화면에 얼마나 매끈하게 이어 그리는가"를 담당 — 서로 다른 층의 스무딩이라 함께 둔다.
# ★반응성↔떨림 절충 이력(2026-08-13): 처음 0.35(부드러움 우선) → 반응이
# 느리다는 요청으로 0.6까지 올림 → 그랬더니 "가만히 있을 때 포인터가 튀는"
# 잔떨림이 두드러진다는 실기 보고로 0.45로 다시 낮춤.
# ★2026-08-14 추가 하향(0.45→0.38, "커서움직임 부드럽게해줘" + "딜레이 없이"):
# 예전엔 이 값을 낮추면 정지 시 잔떨림이 심해지는 게 값을 못 낮추는 이유였는데,
# 이제 그 잔떨림은 RESUME_GAP_RATIO 정지 히스테리시스가 이 값과 무관하게 따로
# 잡아준다 — 그래서 옛 트레이드오프가 더 이상 그대로 적용되지 않는다. 움직이는
# 동안의 반응 지연은 순수하게 "목표가 갱신되는 주기(추론 스레드, 20~30Hz =
# 33~50ms) 대비 몇 프레임 만에 따라잡는가"로 결정되는데, alpha 0.38이면 렌더
# 주사율(보통 60Hz+)에서 두세 프레임(약 30~50ms) 안에 대부분 따라잡아 사람이
# 체감하는 지연은 거의 없이 궤적만 더 매끄러워진다
RENDER_LERP_ALPHA = 0.38

# 렌더 상한(2026-08-20 사용자 요청 — "수직 동기화 기능 못넣나 30으로
# 고정하고싶은데"): 화면 창을 쓰는 구조라 진짜 수직 동기화(디스플레이의 수직
# 귀선에 맞춰 내보내기)는 걸 수 없다 — 대신 시간 기준으로 이 값에 맞춰
# 페이싱한다. 30으로 잡는 근거:
#  ① 커서 위치의 원본인 얼굴 인식이 카메라 속도(30fps)로 들어오므로, 그보다
#     자주 그려봐야 새 정보가 없다.
#  ② 60Hz 화면에서 30은 정확히 절반이라 매 그림이 화면 갱신 두 번에 고르게
#     걸린다(어중간한 45 같은 값은 어떤 그림은 한 번, 어떤 건 두 번 걸려
#     떨려 보인다).
#  ③ 한 틱에 쓸 수 있는 시간이 16.7ms → 33.3ms로 두 배가 되어, 늦는 일이
#     크게 줄어든다(늦는 게 곧 커서가 끊겨 보이는 원인이었다).
# 화면 주사율이 30보다 낮으면 그쪽을 따른다(min으로 고른다)
RENDER_FPS_CAP = 30

# 위 alpha 값들은 "한 틱에 남은 거리의 몇 %를 따라갈지"라, 틱이 늦어지면
# 그만큼 커서가 덜 따라와 느려 보인다. 실제 걸린 시간에 맞춰 alpha를 보정해
# 틱 간격이 흔들려도 움직임이 일정하게 보이도록 한다 — 기준은 위 상한(30fps)
# 이라, 정상 속도로 돌 때는 보정이 없어 기존 손맛이 그대로 유지된다
def _dt_adjusted_alpha(alpha, dt_sec, ref_dt_sec):
    if dt_sec <= 0.0 or abs(dt_sec - ref_dt_sec) < 1e-6:
        return alpha
    return max(0.0, min(1.0, 1.0 - (1.0 - alpha) ** (dt_sec / ref_dt_sec)))

# 커서 민감도·입벌림 문턱(2026-08-13 최초 상향 후, "가만히 있을 때 튀고
# 민감도 조금만 낮추자" 요청으로 살짝 되낮춤): sensitivity는 코/미간 오프셋에
# 곱해지는 배율이라, 값이 높을수록 의도한 움직임뿐 아니라 랜드마크 검출의
# 미세한 잡음까지 그대로 증폭돼 정지 상태에서도 커서가 튀어 보인다.
# config.yaml 기본값(0.9/1.0)은 다른 진입점과 공유해 여기선 안 건드리고 이
# 실행기 config 사본에만 적용. mouth_click은 open_margin(확정 문턱)을
# 낮추면서 close_margin(재장전 문턱)도 그보다 낮게 함께 낮춰야 히스테리시스
# 순서(open > close)가 안 깨진다
SENSITIVITY_X_OVERRIDE = 1.12    # config 기본 0.9 (2026-08-13: 1.6→1.3, 2026-08-18: "커서 민감도 너무 빠르다" 1.3→1.05,
                                 #   2026-08-24: "아주 미세하게 살짝만" 요청으로 1.05→1.12 (약 7%))
SENSITIVITY_Y_OVERRIDE = 1.46   # config 기본 1.0 (2026-08-13: 1.7→1.4, 2026-08-14: 1.4→1.6, 2026-08-18: 1.6→1.3, 2026-08-26: 1.39→1.46 살짝 상향,
                                 #   2026-08-24: 1.3→1.39 (가로와 같은 비율로 약 7%))
# 2026-08-18: "입벌리는걸 인식을 잘 못하는거 같아 다방면으로 입벌리는걸
# 인식할수 있게" 대응으로 더 낮춤(0.18→0.12) — 고개를 위/아래로 크게
# 기울인 채로(이 커서는 화면 위쪽 절반만 써서 고개를 든 자세가 잦다) 입을
# 벌리면 각도 때문에 jawOpen 블렌드셰이프 점수 자체가 낮게 잡히는 경우가
# 있다(정면일 때보다 둔감) — 문턱을 낮춰 그런 각도에서도 웬만하면 확정되게
# 여유를 더 준다
MOUTH_OPEN_MARGIN_OVERRIDE = 0.12    # config 기본 0.35 — 이만큼만 벌어져도 확정
MOUTH_CLOSE_MARGIN_OVERRIDE = 0.05   # config 기본 0.15 — open보다 낮게 유지(2026-08-18: 0.08→0.05, 위와 같은 비율로 하향)

# 입 제스처 확장(2026-08-18 신설, 사용자 요청 — "입 빠르게 두번 누르면
# 더블클릭 입 벌린채로 2초 이상 있으면 꾹 누르기로 해줘... 마우스 좌 클릭
# 꾹누르기 말하는거야[드래그]"): head_tracker.py의 기존 EVENT_SELECT
# (trigger=mouth)는 입을 벌리는 순간 무조건 클릭 1회뿐이라 이 세 갈래를
# 표현할 수 없다 — 그래서 head_tracker의 공유 판정은 그대로 두고(dwell 트리거는
# 지금처럼 즉시 단일 클릭), 입 트리거만 이 파일에서 직접 원시 값
# (head_tracker.debug의 jaw_open/jaw_base, MOUTH_OPEN/CLOSE_MARGIN_OVERRIDE와
# 같은 문턱)을 보고 열림/닫힘을 자체적으로 추적한다 — _update_mouth_gesture
# 함수 참고.
# ★2026-08-18 재설계("바로바로 클릭 되게 만들어줘" 대응): 처음엔 짧게
# 열었다 닫힌 뒤 일정 시간 더 기다려 두 번째 탭이 오는지 보고서야 단일/더블을
# 확정해 보냈는데("더블클릭인지 확인하려면 기다려야 한다"는 생각으로), 이러면
# 모든 단일 클릭에 그 대기 시간만큼 지연이 생겨 "바로바로" 요청과 정면으로
# 부딪혔다. 다시 보니 그 대기 자체가 불필요했다 — 더블클릭 판정은 Windows가
# 이미 하고 있는 일이다(GetDoubleClickTime 기준으로, 짧은 시간 안에 같은
# 자리에서 클릭이 두 번 들어오면 그 앱이 알아서 더블클릭으로 받아들인다 —
# 물리 마우스로 두 번 눌러도 원리는 똑같다). 그래서 이제 입을 짧게 벌렸다
# 닫을 때마다 그 즉시 클릭 1회를 보낸다 — 지연도 없고, 사용자가 입을 빠르게
# 두 번 벌리면 그 두 클릭이 실제로 짧은 간격으로 들어가 Windows가 자연스럽게
# 더블클릭으로 인식한다. 아래는 이제 "꾹 누르기" 판정에만 쓰인다
# ★2026-08-20 하향(2.0→1.2, 사용자 실기 보고 "드래그 잘 안되더라"): 입을
# 2초 동안 흔들림 없이 벌리고 있는 건 생각보다 어렵다 — 아래 닫힘 확인
# (MOUTH_CLOSE_CONFIRM_SEC)과 함께 드래그가 실제로 걸리게 만드는 조치다.
# 일반 클릭은 입을 0.3~0.5초 정도만 벌렸다 닫으므로 1.2초와는 충분히 구분된다
MOUTH_HOLD_SEC = 0.7   # 이 이상 계속 벌리고 있으면 "꾹 누르기"(드래그 시작)로 전환.
# ★2026-08-24 1.2 -> 0.7 (사용자 요청 "드래그하는거 좀 더 짧게, 2초는 너무 긴 것 같은데").
#
# 얼마까지 줄여도 되는지는 "일반 클릭이 실수로 드래그가 되는가"가 정한다. 상태
# 기계를 그대로 재현해 입을 벌리고 있는 시간별로 판정을 돌려봤다:
#
#     유지시간   입을 0.2s  0.3s  0.4s  0.5s  0.6s  0.8s  1.0s 벌렸을 때
#       0.5s      클릭  클릭  클릭  클릭  드래그 드래그 드래그   <- 0.6초짜리 느린 클릭이 샌다
#       0.7s      클릭  클릭  클릭  클릭  클릭  드래그 드래그   <- 채택
#       1.2s      클릭  클릭  클릭  클릭  클릭  클릭  클릭     <- 예전값. 드래그가 너무 멀다
#
# 사람이 클릭하려고 입을 여닫는 시간은 보통 0.2~0.5초라, 0.7초면 그보다 확실히
# 길어서 안 새면서도 예전보다 40% 빠르다. 드래그 시작 순간 커서 색이 바뀌므로
# (CursorFeedback) 짧아져도 "지금 눌렸다"를 바로 알 수 있다.

# 닫힘 확인 시간(2026-08-20 신설 — "드래그 잘 안되더라"의 핵심 원인): 입을
# 벌린 채 유지해도 턱 떨림 때문에 입벌림 값이 순간적으로 닫힘 문턱 아래로
# 내려갔다 올라오는 일이 잦다. 예전 판정은 그 한 순간을 곧바로 "입을 다물었다"로
# 보고 드래그를 끝내(또는 클릭으로 처리해) 버려서, 2초를 채우기 전에 매번
# 끊겼다. 이제 닫힘 문턱 아래로 내려간 상태가 이 시간만큼 이어져야 진짜
# 닫힘으로 본다 — 순간적인 흔들림은 무시되고, 그동안 벌린 시간은 계속 쌓인다.
# 사람이 못 느낄 만큼 짧게 잡아 "바로바로 클릭" 요구와도 부딪히지 않는다
DOUBLE_CLICK_WINDOW_SEC = 0.9
# 두 클릭이 이 안에 들어오면 더블클릭으로 본다 (_Win32Mouse.click 독스트링 참고).
# 윈도우 기본 한계(500ms)보다 넉넉한 이유는 실측 때문이다 — 입을 0.25초씩
# 여닫는 평범한 속도가 이미 485ms라, 500ms에 맞추면 조금만 느려도 실패한다.
# 0.9초는 "빠르게 두 번"으로 읽히는 상한이면서, 서로 다른 두 번의 클릭이
# 잘못 묶이지 않을 만큼 짧다.

MOUTH_CLOSE_CONFIRM_SEC = 0.08

# 드래그 중에는 "닫혔다"를 더 엄격하게 본다 (2026-08-24 신설).
#
# 왜: 드래그는 입을 벌린 채로 고개를 움직이는 동작이다. 고개가 움직이면 얼굴
# 랜드마크가 흔들려 턱 벌림 추정값도 같이 출렁이는데, 그때 잠깐 닫힘 문턱 아래로
# 스치면 드래그가 중간에 끊긴다. 사용자 입장에서는 "드래그하다 놓쳤다"가 된다.
#
# 그래서 드래그를 이미 시작한 뒤에는 ① 더 오래 닫혀 있어야 하고 ② 더 깊이
# 닫아야 놓아준다. 시작할 때보다 끝낼 때를 어렵게 만드는 비대칭 히스테리시스로,
# 스위치 채터링을 막는 것과 같은 발상이다. 일부러 입을 다물면 이 정도는 쉽게
# 넘으므로 의도한 종료는 그대로 잘 걸린다.
MOUTH_HOLD_RELEASE_CONFIRM_SEC = 0.20   # 드래그 종료에 필요한 닫힘 지속(평소 0.08)
MOUTH_HOLD_RELEASE_MARGIN = 0.03        # 드래그 종료에 필요한 닫힘 깊이(평소 0.05 — 더 낮다=더 깊이)

# 커서 EMA 반응 속도(2026-08-13: 정지 시 떨림 완화를 위해 되낮춤 — 위
# RENDER_LERP_ALPHA·SENSITIVITY와 같은 이력): config 기본값(smoothing_alpha=0.2,
# distance_smoothing_alpha=0.15)은 다른 진입점과 공유해 여기서 안 건드리고
# 이 실행기 config 사본에만 적용
# ★2026-08-14 세 번째 조정(0.16→0.22, "커서가 늦게 따라오는 현상... 살짝만
# 돌려서 선택하려고 할 때 끊기는 느낌" 대응): 직전에 0.28→0.16까지 크게
# 낮췄더니 이번엔 반대로 랙(지연)이 느껴진다는 보고가 왔다 — EMA를 세게
# 걸수록 원시 잡음은 줄지만 실제 움직임도 그만큼 늦게 반영된다는 근본적인
# 트레이드오프라, 계속 한쪽으로만 밀어붙이면 반드시 반대쪽 불만이 나온다.
# 그래서 이 값은 다시 절반쯤 되돌리고, 대신 "정지 히스테리시스" 쪽에 지속시간
# 확인(RESUME_CONFIRM_SEC)을 새로 추가해 잡음 억제 역할을 더 맡겼다 — 진폭
# (RESUME_GAP_RATIO)만으로 잡음/의도한 움직임을 가르면 작은 의도한 움직임까지
# 막혀 "끊기는" 느낌이 나니, 진폭 문턱은 다시 낮추고 그 대신 그 문턱을 넘은
# 상태가 짧게라도 유지되는지를 봐서 순간 잡음만 걸러낸다(아래 RESUME_CONFIRM_SEC
# 참고) — 이러면 EMA는 덜 세게 걸어도 되어 지연이 준다.
# ★2026-08-18 추가 상향(0.22→0.32, 사용자 실기 보고 "고개를 돌리면 커서가
# 뒤늦게 따라오는 현상"): 이 값은 head_tracker.py _CursorMapper.update()에서
# **추론 프레임마다**(20~30Hz) 걸리는 EMA라, 렌더 루프의 RENDER_LERP_ALPHA
# (매 렌더 틱·60Hz+마다 걸림)보다 체감 지연에 훨씬 크게 기여한다 — 정상 상태
# 지연은 대략 (1/alpha) 프레임이라, 0.22면 약 4.5추론프레임(25Hz 기준 ~180ms),
# 0.32면 약 3.1프레임(~125ms)로 줄어든다. 이 값을 낮게 잡아야 했던 이유(정지
# 시 잡음 억제)는 이제 렌더 층의 정지 히스테리시스(RESUME_GAP_RATIO+
# RESUME_CONFIRM_SEC+SETTLE_CONFIRM_SEC, 전부 이 값과 독립적으로 작동)가
# 대신 맡고 있어서, 이 값을 잡음 방어의 1차 수단으로 쓸 필요가 줄었다 — 더
# 올려도 될 여지가 생긴 것
POINTER_SMOOTHING_ALPHA_OVERRIDE = 0.32           # config 기본 0.2 (2026-08-13: 0.45→0.28, 2026-08-14: 0.28→0.16→0.22, 2026-08-18: 0.22→0.32)
# ★2026-08-14 distance_smoothing_alpha 추가 하향(0.2→0.08, 사용자 실기 보고
# "고개를 너무 들어올리면 커서가 와들와들 부들부들 되는 현상"): _CursorMapper.update()
# (head_tracker.py)는 dx/dy를 안구간거리(interocular_dist_px)로 나눠 정규화한다 —
# 고개를 크게 들면 카메라에서 보이는 눈 사이 간격이 원근 때문에 실제로 좁아지는
# 데다, 극단적인 각도에서는 랜드마크 검출 자체도 덜 안정적이라 이 거리값이
# 프레임마다 흔들린다. 나누는 값(분모)이 작고 떨리면 그 잡음이 나눗셈으로
# 그대로 증폭돼 커서가 격하게 떨리는 것으로 보인다. distance_smoothing_alpha를
# 크게 낮춰(더 강하게 평활) 이 분모 자체를 훨씬 천천히만 바뀌게 만들면, 순간적인
# 거리 추정 잡음이 증폭되기 전에 걸러진다 — 실제 카메라와의 거리(사람이 앞뒤로
# 움직이는 것)는 이 정도로 느리게 반응해도 체감상 문제없을 만큼 드물게 바뀐다
POINTER_DISTANCE_SMOOTHING_ALPHA_OVERRIDE = 0.08   # config 기본 0.15 (2026-08-13: 0.3→0.2, 2026-08-14: 0.2→0.08)

# 데드존(2026-08-13 사용자 요청 — "엄청 미세하게 떨리긴하는데 수정해주면
# 좋긴해"): 스무딩을 더 세게 걸면 이번엔 반응이 처지므로, 대신 목표 위치와
# 현재 표시 위치의 거리가 이 값(화면 비율)보다 작으면 렌더 루프가 아예
# 움직이지 않는다 — 진짜 의도한 움직임은 거리가 금방 이 문턱을 넘어 그대로
# 반응하고, 랜드마크 검출의 미세한 잡음(가만히 있을 때의 1px 안팎 흔들림)만
# 걸러진다. 반응 속도를 전혀 희생하지 않고 정지 시 떨림만 없애는 방식
# ★2026-08-14 재상향(0.0035→0.006, 사용자 실기 보고 "가만히 있을때 커서
# 떨림이 있어"): is_moving 상태에선 이 문턱 안까지 목표를 따라잡아야 비로소
# 멈추고 anchor를 다시 고정한다(아래 RESUME_GAP_RATIO 히스테리시스 참고) —
# 이 문턱이 랜드마크 잡음의 흔들림 폭보다 작으면, 목표 자체가 잡음으로 계속
# 문턱 밖에서 맴돌아 is_moving이 영영 꺼지지 못하고 미세하게 계속 쫓아가는
# 것처럼 보인다(떨림의 실제 원인). 문턱을 잡음 폭보다 확실히 크게 잡아야
# "따라잡았다"는 판정이 안정적으로 성립한다. ★같은 날 재차 상향(0.006→0.008)
# 했다가, "끊기는 느낌" 보고로 다시 낮춤(0.008→0.006) — 아래 RESUME_GAP_RATIO
# 주석의 재설계 배경과 같은 이유.
# ★2026-08-20 재하향(0.006→0.0035, 사용자 요청 "작은 아이콘들도 정확하게
# 커서에 대고 클릭할 수 있게"): 이 문턱은 "여기까지 왔으면 도착으로 친다"는
# 값이라, 그대로가 곧 **커서가 낼 수 있는 최소 이동 단위**가 된다. 0.006이면
# 가로 1536px 화면에서 약 9px — 32px짜리 바탕화면 아이콘을 겨냥하기엔 너무
# 굵어서, 아이콘 위에 정확히 얹지 못하고 언저리에서 멈춰 버렸다. 절반으로
# 낮추면 약 5px 단위로 미세 조정이 되어 작은 대상도 겨냥할 수 있다.
# 떨림이 다시 살아나지 않는 이유: 이제 잡음은 이 진폭 문턱이 아니라
# 지속시간 확인(RESUME_CONFIRM_SEC·SETTLE_CONFIRM_SEC)이 걸러내기 때문이다
RENDER_DEAD_ZONE_RATIO = 0.0035

# 정지 히스테리시스(2026-08-14 사용자 실기 보고 — 키오스크에서 "고개를 가만히
# 있어도 커서가 조금씩 움직인다"): 위 데드존만으로는 못 막는 종류의 문제다 —
# 데드존은 "이번 틱의 목표-표시 거리"만 보는데, 표시 위치가 매 틱 목표를
# 살짝씩 따라가다 보면 목표 자체가 아주 느리게 한쪽으로 흘러갈 때(개발 PC보다
# 키오스크 카메라의 조명·노출이 불안정해 랜드마크 추정치가 미세하게 계속
# 흔들리는 경우 등) 매 틱 문턱을 살짝씩만 넘는 게 누적돼 서서히 떠내려가는
# "drift"까지는 못 잡는다. 그래서 "정지 중"에는 매 틱 갱신되는 표시 위치가
# 아니라 **정지가 시작된 그 순간의 고정된 지점(anchor)** 하나만 기준으로
# 삼는다 — 목표가 이 anchor에서 이 문턱만큼 벌어져야 비로소 "진짜 움직였다"고
# 보고 다시 따라가기 시작한다. anchor는 절대 서서히 갱신되지 않으므로(움직이기
# 시작할 때·다시 멈출 때만 그 시점 위치로 갱신) 아무리 작은 흔들림이 오래
# 누적돼도 anchor 자체가 밀리지 않아 drift가 원천 차단된다.
# ★2026-08-14 재설계(진폭 문턱만 계속 올렸다가(0.01→0.015→0.02) "커서가
# 늦게 따라오고... 살짝만 돌려서 선택하려고 할 때 끊기는 느낌" 역풍을 맞음):
# 문턱을 크게 잡으면 잡음은 잘 걸러지지만 작고 의도한 움직임(정밀 조준)까지
# 같이 막혀버린다 — 진폭만으로는 "잡음"과 "작은 의도한 움직임"을 구분할 수가
# 없다. 그래서 문턱은 원래 크기로 되돌리고, 대신 아래 RESUME_CONFIRM_SEC로
# "그 문턱을 넘은 상태가 잠깐이라도 유지되는지"를 추가로 본다 — 잡음은
# 프레임마다 방향이 들쭉날쭉해 문턱을 넘어도 금방 다시 밑으로 떨어지지만,
# 진짜 의도한 움직임은 작더라도 그 방향으로 계속 유지된다는 차이를 이용한다.
# ★2026-08-20 재하향(0.01→0.006, 위 RENDER_DEAD_ZONE_RATIO와 같은 이유 —
# "작은 아이콘 정확히 클릭"): 이 값은 멈춰 있다가 **다시 움직이기 시작하는
# 데 필요한 최소 거리**다. 0.01이면 약 15px이라, 아이콘을 살짝 비껴 맞췄을 때
# 그보다 작게 보정하려 해도 커서가 아예 반응하지 않아 조준을 못 고쳤다
RESUME_GAP_RATIO = 0.006
# 문턱(RESUME_GAP_RATIO)을 넘은 상태가 이 시간(초) 동안 끊기지 않고 이어져야
# "진짜 움직임"으로 확정한다 — 카메라 프레임 1~2장 분량(추론 스레드 20~30Hz
# 기준)이라 순간적인 잡음 스파이크 한두 장은 걸러진다. 넘었다가 문턱 아래로
# 떨어지면 그 순간 다시 0부터 잰다(아래 렌더 루프 참고).
# ★2026-08-18 재하향(0.1→0.07, 사용자 실기 보고 "커서 1cm 정도 이동하려고
# 하면 뚝뚝 끊긴다"): 확정 전까지 화면이 완전히 얼어 있다가 확정되는 순간
# 한 번에 따라잡는 구조라, 작은(1cm급) 움직임은 "정지 → 순간 스냅"으로
# 보였다 — 이동 거리 자체가 작아서 확정 후 따라잡기가 1~2프레임 만에 끝나
# 버려 눈엔 순간이동처럼 보인다. 대기(미확정) 중에도 완만히(CATCHUP_LERP_ALPHA)
# 미리 따라가기 시작하도록 바꿔서(아래 렌더 루프 참고) "정지→스냅"을
# "서서히 가속"으로 바꿨고, 대기 시간 자체도 줄여 전체 지연을 낮췄다
RESUME_CONFIRM_SEC = 0.07

# ★2026-08-18 실측 진단으로 발견(head.py를 직접 실행해 [MOVE ...] 진단 로그를
# 실시간으로 확인) — 연속으로 고개를 움직이는 중에도 "뚝뚝 끊기는" 게 재현됐다.
# 원인: 렌더 루프(60Hz+)가 추론 스레드(20~30Hz)보다 훨씬 빠르게 도는데,
# RENDER_LERP_ALPHA(0.38)로 목표를 쫓다 보면 다음 실측 갱신이 오기 전에
# "그 순간의(약간 낡은) 목표"를 완전히 따라잡아버린다(gap<=RENDER_DEAD_ZONE_RATIO) —
# 그러면 즉시 is_moving=False로 떨어져 anchor를 다시 얼리는데, 바로 다음
# 추론 프레임에서 목표가 또 움직이면 RESUME_GAP_RATIO+RESUME_CONFIRM_SEC
# 전체를 처음부터 다시 통과해야 한다. 그 결과 "빠르게 따라잡기(~80ms) →
# 완전 정지(다음 추론 갱신까지) → 대기 재확정(~70~100ms) → 다시 따라잡기"가
# 사람이 실제로는 계속 움직이는 중에도 약 300ms 주기로 반복됐다 — 이게 실측
# 로그로 확인된 "뚝뚝 끊김"의 실제 정체다. 그래서 "따라잡음(is_moving=True)"
# 상태에서 벗어나는 쪽에도 같은 지속시간 확인을 추가한다(아래
# SETTLE_CONFIRM_SEC) — 목표가 실제로 그만큼 오래 안 움직여야 비로소 정지로
# 확정하고, 잠깐 따라잡았다고 바로 얼어붙지 않는다(아래 렌더 루프 참고)
SETTLE_CONFIRM_SEC = 0.12

# 캘리브레이션 직후 정지 유예(2026-08-13 사용자 요청 — "캘리브레이션 초 다
# 지나면 화면중앙에 돌아오는데 바로 움직이게 하지말고 5초있다가 움직이게"):
# 얼굴이 새로 잡히는(=캘리브레이션이 막 끝나는) 시점부터 이 시간(초) 동안은
# 커서(실제 마우스 포함)가 아주 느리게만 반응해 거의 그 자리에 붙잡혀 있는
# 것처럼 보인다 — 캘리브레이션 직후 사용자가 자세를 잡는 시간을 벌어준다.
# 이 동안은 커서 아래에 안내 문구를 띄운다(아래 SETTLING_LABEL). ★이 유예
# 구간이 프로그램 시작 때 한 번만이 아니라 실사용 중에도(recenter_dwell
# 자동 재정렬로) 수시로 걸린다는 게 2026-08-14에 로그로 드러나면서, 유예
# 처리 방식이 여러 차례 바뀌었다 — 최종 구현·이유는 아래 렌더 루프의
# "if is_settling:" 분기 주석 참고. 2026-08-18: "캘리브레이션 초 조금만 더
# 늘리자" 요청으로 5.0 → 7.0, 이어서 "8초로 늘리고" 요청으로 7.0 → 8.0
SETTLE_DELAY_SEC = 8.0
SETTLING_LABEL_LINE1 = "커서 재정렬 중입니다"
# 2026-08-14 사용자 요청 — 정지 유예 중 무엇을 해야 하는지 안내가 없어 헷갈릴
# 수 있어 안내 문구 둘째 줄 추가. ★같은 날 문구 수정("화면 정면" → "커서
# 중앙") — 유예가 끝나는 순간 head_tracker 내부 기준으로는 지금 보고 있는
# 곳이 곧 캘리브레이션 중심(비율 0.5)이 되므로, 화면 어디가 아니라 지금
# 보이는 커서(유예 시작 직전 위치에 얼어 있는 점)를 계속 보고 있어 달라는
# 뜻이 더 정확하다
# ★2026-08-20 "입은 다물어 주세요" 추가 — 실측으로 확인된 실패 모드 대응.
# 이 유예 구간은 커서 중심만 잡는 게 아니라 **입 벌림의 평상시 기준선**도 함께
# 잡는다(head_tracker의 jaw_baseline). 그래서 이 8초 동안 입이 벌어져 있으면
# 기준선이 그만큼 높게 잡히고, 이후 "평소보다 이만큼 더 벌리면 클릭"의 기준이
# 통째로 어긋나 클릭이 안 먹거나 제멋대로 나간다.
# 실측: 유예 중에 입을 여닫는 상태로 캘리브레이션하면 더블클릭이 주기당 2번이
# 아니라 1번만 나갔고, 입을 다물고 캘리브레이션하니 정상으로 돌아왔다.
# 말하거나 하품하면서 서 있으면 실사용에서도 그대로 생기는 문제라 안내에 넣는다.
SETTLING_LABEL_LINE2 = "커서 중앙을 봐주시고 입은 다문 채 편한자세로 있어주세요"

# 큰 격차 완만한 합류(2026-08-14 최초 도입 — "5초후 갑자기 커서가 순간이동을
# 하는데"). ★같은 날 재설계(로그로 확인한 실제 원인 — 위 SETTLE_DELAY_SEC·
# 렌더 루프 "if is_settling:" 분기 주석 참고: recenter_dwell 자동 재정렬이
# 실사용 중 수시로 걸리는데, 유예 중 화면 표시와 실제 목표가 따로 놀다가
# 유예 종료 시점에 격차를 드러내는 구조 자체가 문제였다) 이후로 CATCHUP_LERP_ALPHA는
# 정지 유예 "전체 구간" 동안 계속 쓰인다(유예 종료 순간에만 잠깐 쓰는 게
# 아니다) — 목표를 항상 느리게 뒤쫓아 화면 표시와 실제 목표가 벌어지는 순간
# 자체를 안 만든다. 평소(유예가 아닌 정상 추적) 정상 추적 중 갑자기 큰
# 격차가 생기는 경우(짧은 미검출 후 재검출 등)에도 이 낮은 alpha로 완만히
# 합류한다 — 평소의 작은 격차(프레임 간 정상 이동)는 이 문턱(LARGE_GAP_RATIO)을
# 넘지 않아 RENDER_LERP_ALPHA의 기존 반응성 그대로 유지된다.
# ★2026-08-18 상향(0.05→0.18, 사용자 실기 보고 "고개를 돌리면 커서가 뒤늦게
# 따라오는 현상"): 0.05(화면 폭의 5%)는 생각보다 낮은 문턱이라, 미검출 복귀
# 같은 진짜 불연속 상황이 아니어도 **평범하게 빠르게 고개를 돌리는 정상
# 동작만으로도** 추론 프레임(20~30Hz) 사이 격차가 이 문턱을 쉽게 넘어 느린
# CATCHUP_LERP_ALPHA(0.08)로 떨어졌다 — 정작 빠른 반응이 필요한 순간에 가장
# 느린 alpha를 쓰고 있었던 셈이다. 문턱을 화면 폭의 18%로 크게 올려서, 이제
# 이 느린 경로는 진짜 불연속적인 상황(짧은 미검출 후 재검출, 정지 유예 등)에만
# 걸리고 평범한 고개 돌리기(아무리 빨라도)는 RENDER_LERP_ALPHA의 빠른 반응을 그대로 쓴다
DRAG_LERP_ALPHA_SCALE = 0.5
DRAG_DEAD_ZONE_SCALE = 2.0
# ★드래그(꾹 누르기) 중에만 커서를 더 무디게 한다 (2026-08-26 사용자 실기 —
# "드래그할때 커서가 좀 많이 떨린다").
#
# 왜 드래그 때만 더 떨려 보이나 — 실측으로 계산한 결과:
# 커서 위치는 (미간이 움직인 픽셀 / 두 눈 사이 거리)로 정해진다. 그래서 얼굴이
# 작게 잡힐수록(멀수록) **같은 픽셀 잡음이 커서를 더 크게 흔든다**.
#
#     안구간거리 90px (노트북, 가까움)  잡음 0.6px -> 커서 흔들림 2.0px
#     안구간거리 60px (키오스크 앞)      잡음 0.6px -> 커서 흔들림 3.0px
#     안구간거리 40px (좀 물러섬)        잡음 0.6px -> 커서 흔들림 4.5px
#
# 화면에서 흔들림을 눌러주는 데드존이 약 3.5px이라, 노트북에서는 잡음이 그 안에
# 들어가 안 보이지만 키오스크 거리에서는 딱 그 경계를 넘나들며 떨려 보인다.
# 게다가 드래그는 무언가를 집어 옮기는 동작이라 사용자가 커서를 계속 주시하고
# 있어서, 같은 흔들림도 훨씬 크게 느껴진다.
#
# 조준할 때는 정밀함이 중요하지만 드래그 중에는 안정성이 더 중요하다 —
# 이미 목표를 집은 상태이기 때문이다. 그래서 드래그 중에만 따라가는 속도를
# 절반으로, 무시하는 범위를 두 배로 키운다. 평소 조준 감각은 건드리지 않는다.

LARGE_GAP_RATIO = 0.18
CATCHUP_LERP_ALPHA = 0.08

# ★2026-08-13: "정확도 값 높여줘" 요청으로 얼굴 검출·존재·추적 신뢰도 문턱을
# 0.5→0.7로 올렸다가, 곧바로 "입 벌리거나 응시해도 클릭이 안 된다"는 실기
# 보고로 되돌림. 원인으로 추정되는 것: select/home 판정은 캘리브레이션 직후
# 잡히는 "평상시 기준선"이 있어야 동작하는데(jaw_baseline 등, head_tracker.py
# 참고), 문턱을 올리면 얼굴 각도·조명이 조금만 흔들려도 "미검출"로 판정돼
# head_tracker가 자주 리셋된다 — 리셋될 때마다 기준선을 처음부터 다시 잡아야
# 해서 그 사이엔 입을 벌려도 select가 발화하지 않는다(클릭이 "안 되는" 것처럼
# 보이는 원인). 검출 신뢰도는 이렇게 임의로 올리기보다, 클릭이 정상 동작하는
# 걸 먼저 확인한 뒤 실측하며 조정해야 하는 값이라 config.yaml 기본값(0.5)을
# 그대로 둔다 — 이 실행기에서 더 이상 오버라이드하지 않음
ACCURACY_CONF = None

# 추론 입력 배율 — ★2026-08-20 실측으로 0.75 -> 1.0 되돌림(원래 config 기본값).
#
# 2026-08-18에 "프레임 개선" 목적으로 0.75를 넣었는데, 실제로 재보니 **이득이
# 0이었다**. 프레임을 줄이는 resize 자체가 줄여서 아끼는 추론 시간과 거의 같은
# 시간을 먹는다:
#
#   실측(실제 크기 — 카메라 1280x720을 9:16으로 크롭하면 405x720)
#     resize 0.75배 (INTER_AREA)              1.24 ms   <- 줄이려고 내는 비용
#     추론 scale=1.0 (405x720)                5.19 ms
#     추론 scale=0.75 (304x540)               4.09 ms   <- 줄여서 번 시간 1.10ms
#   => 1.24ms 내고 1.10ms 벌었다. 오히려 손해.
#
# 처음에 0.75가 맞다고 판단한 건 프레임 크기를 1080x1920으로 잘못 알고 재봤기
# 때문이다. 실제로는 9:16 크롭 후 405x720이라 추론이 이미 충분히 가볍고
# (전체 예산 33ms 중 5ms), 줄여서 얻을 게 없다.
#
# 되돌리면 덤으로 두 가지가 좋아진다 — 랜드마크 정밀도가 원래대로 돌아오고,
# 어두운 곳에서 유리하다(픽셀을 줄이면 그만큼 얼굴 정보가 사라져 저조도에서
# 먼저 무너진다). 프레임의 진짜 원인은 추론이 아니라 렌더 쪽이었다
# (_pump_window_events / _blank_rect 설명 참고)
INFER_SCALE_RATIO_OVERRIDE = 1.0

# 커서를 "얼굴이 향한 방향"으로만 움직이는 방식 — 켬(2026-08-20 사용자 요청
# "내가 바라보는 지점이 그대로 커서가 가있게 정확하게 수정해줘").
# 원리와 근거는 head_tracker.py _CursorMapper 독스트링에 자세히 적었다. 요약하면
# 코 위치를 화면이 아니라 **얼굴 자신을 기준으로** 재서, 몸이 움직인 것과 고개를
# 돌린 것을 구분한다 — 코는 얼굴에서 튀어나와 있어 고개를 돌리면 두 눈 중점에서
# 확실히 벗어나므로 이 방식이 잘 맞는다.
FACE_LOCAL_MAPPING = True

# 위 방식의 속도 보정 배수 — ★커서가 흔들리면 여기부터 만진다.
#
# 새 매핑은 "코가 얼굴에서 튀어나온 길이"만큼의 작은 움직임으로 회전을 읽는다.
# 예전 방식은 회전 중심(목 부근)에서 코끝까지의 훨씬 긴 거리를 썼다. 그 차이를
# 배수로 되돌리는 값인데, **신호와 잡음이 똑같이 커진다** — 올린다고 떨림 대비
# 이득이 생기지 않는다. 즉 이 값은 "정밀도와 편함" 사이의 손잡이다:
#
#     gain   떨림(예전 대비)   같은 거리 가려면 고개를
#     ----   -------------   --------------------
#     1.5        1.00배            2.0배 더 돌림   <- 가장 안 떨림, 목이 피곤
#     2.0        1.33배            1.5배 더 돌림   <- 기본값
#     3.0        2.00배            예전과 동일     <- 예전 손맛, 어두우면 많이 떨림
#
# 기본을 2.0으로 잡은 근거: 사용자 실기 보고 "코 기준은 커서가 많이 떨렸고
# 미간은 별로 안 떨리더라"(특히 어두울 때). 3.0은 예전 손맛을 그대로 주지만
# 떨림을 두 배로 만들어 그 문제를 정면으로 악화시킨다. 코 묶음 평균
# (NOSE_CLUSTER_AVERAGING)이 흔들림을 1.5배쯤 줄여주므로, 2.0이면 실제 떨림은
# 예전의 1.33배 수준에서 그친다.
#
#   너무 느리다 / 목이 아프다  -> 2.5, 3.0으로 올린다
#   커서가 떨린다 / 조준이 어렵다 -> 1.5로 내린다
FACE_LOCAL_GAIN = 2.0

# 커서 기준점을 코끝 한 점이 아니라 코 아래쪽 여러 점의 평균으로 잡을지
# (face_estimator.LMK_NOSE_CLUSTER 설명에 실측 근거). 어두운 곳에서 커서가
# 떨리는 문제의 직접 대응이라 기본은 켬 — 끄면 예전처럼 코끝 한 점만 쓴다
NOSE_CLUSTER_AVERAGING = True

# 자동 재정렬(가만히 있으면 스스로 중심을 다시 잡는 기능) — ★2026-08-20 끔.
#
# 사용자 결정: "몇 초 가만히 있으면 캘리브레이션 되는 거 없애자."
# 시작할 때 화면 중앙을 보며 자세를 잡는 캘리브레이션(SETTLE_DELAY_SEC)은
# 그대로 두고, 사용 중에 스스로 다시 잡는 것만 없앤다.
#
# 원래 이 기능은 커서가 조금씩 밀려가는 것을 주기적으로 되돌리려고 있었다 —
# 즉 증상을 덮는 반창고였다. 밀려가는 원인 자체(고개를 돌린 것과 몸이 움직인
# 것을 구분하지 못하던 것)를 _CursorMapper에서 고쳤으므로 반창고가 필요 없다
# (head_tracker.py _CursorMapper 독스트링 참고).
#
# 게다가 이 기능은 실사용에서 방해가 됐다: 작은 아이콘을 겨냥하려고 커서를
# 멈춰 두는 동작이 그대로 재정렬 조건이라, 조준하는 도중에 커서가 중앙으로
# 튀어 버렸다.
RECENTER_DWELL_ENABLED_OVERRIDE = False

# --overlay 크로마키 색(BGR 마젠타, main_dpad.py TRANSPARENT_KEY_COLOR와 동일값) —
# 이 정확한 색의 픽셀만 완전 투명 처리된다. 커서 색은 초록으로 맞춰 dpad_overlay·
# 본 엔진 헤드트래커 커서와 통일한다(visualize.CURSOR_COLOR와 동일 계열)
TRANSPARENT_KEY_COLOR = (255, 0, 255)
OVERLAY_ALPHA = 220   # 0(완전 투명)~255(완전 불투명) — 커서가 잘 보이게 진하게
CURSOR_COLOR = (0, 220, 0)   # visualize.CURSOR_COLOR와 동일값(색은 그대로 통일)

# 클릭·드래그 피드백 색 (2026-08-24 사용자 요청 — "클릭할 때 커서가 깜빡이는
# 그런 피드백 넣어줘. 입 한 번 벌리면 한 번 색이 바뀌고, 두 번 클릭은 두 번
# 깜빡이고, 드래그는 바뀐 색상 유지").
#
# 왜 필요한가: 지금은 입을 벌려 클릭해도 화면에 아무 변화가 없어서, 클릭이
# 먹었는지 아닌지를 결과(창이 열렸는지 등)로만 알 수 있다. 인식이 실패했을 때
# "안 눌렸구나"를 즉시 알 수 없으니 같은 동작을 반복하게 되고, 그게 다시
# 오작동으로 이어진다. 커서 자체가 반응하면 그 순환이 끊긴다.
CURSOR_CLICK_COLOR = (0, 235, 255)    # 클릭 순간 잠깐 (노랑)
CURSOR_HOLD_COLOR = (255, 140, 0)     # 꾹 누르는 동안 계속 (파랑) — 드래그 중임을 계속 알린다
CURSOR_CLICK_FLASH_SEC = 0.12
# 0.12초인 이유: 30fps 화면에서 3~4프레임이라 확실히 보이면서, 더블클릭의 두
# 클릭 간격(실측 340~380ms)보다 훨씬 짧아 **두 번이 각각 따로 깜빡여 보인다**.
# 0.2초를 넘기면 두 번이 하나로 이어져 보여서 "두 번 눌렸다"가 전달되지 않는다


class CursorFeedback:
    """커서 색으로 클릭·드래그 상태를 알린다. 추론 스레드가 쓰고 렌더 스레드가 읽는다.

    값이 float/bool 하나씩뿐이라 파이썬에서는 대입이 쪼개지지 않는다 — 락 없이도
    렌더 쪽이 찢어진 값을 보는 일이 없다(락을 잡으면 렌더 틱이 추론에 물린다).
    """

    def __init__(self):
        self.flash_until_sec = 0.0
        self.is_holding = False

    def flash(self, now_sec):
        """클릭 1회 — 잠깐 색이 바뀐다. 연달아 부르면 그때마다 다시 깜빡인다."""
        self.flash_until_sec = now_sec + CURSOR_CLICK_FLASH_SEC

    def set_holding(self, holding):
        """드래그(꾹 누르기) 시작/끝 — 누르는 동안 색이 계속 바뀐 채로 있는다."""
        self.is_holding = holding

    def color(self, now_sec):
        if self.is_holding:
            return CURSOR_HOLD_COLOR
        if now_sec < self.flash_until_sec:
            return CURSOR_CLICK_COLOR
        return CURSOR_COLOR

    def state_key(self, now_sec):
        """지금 어떤 상태인지 — 다시 그릴지 판단하는 signature에 넣는다.
        이걸 빼먹으면 커서가 제자리일 때 다시 그리기를 건너뛰어 **깜빡임이
        화면에 안 나타난다**(색만 바뀐 건 위치 변화가 아니라서)."""
        return 2 if self.is_holding else (1 if now_sec < self.flash_until_sec else 0)
RECENTER_PROGRESS_COLOR = (0, 165, 255)   # visualize.RECENTER_PROGRESS_COLOR와 동일값

# 커서 크기(2026-08-13 사용자 요청 — "포인터를 둘다 좀더 크게"): visualize.py의
# draw_cursor를 그대로 쓰지 않고 이 파일에서 직접 그린다 — 그쪽 크기(반경
# 14px)는 main_dpad.py가 **카메라 해상도** 프레임(보통 세로 720px 안팎) 위에
# 그리는 걸 기준으로 잡은 값인데, 이 오버레이는 **실제 화면 해상도**(보통
# 1080~1440px대, 훨씬 큼) 캔버스 위에 그려서 상대적으로 훨씬 작아 보인다.
# 공용 함수 크기를 화면 해상도에 맞게 고치면 main_dpad.py 쪽까지 함께
# 바뀌어버려서, 이 파일 전용으로 더 큰 값을 따로 둔다
CURSOR_RADIUS_PX = 28
CURSOR_MARKER_SIZE_PX = 22
CURSOR_THICKNESS_PX = 3

# 추적 기준점 시각화(2026-08-13 — 처음엔 오버레이에 "인식됨" 문구로 시도했다가,
# 사용자가 원한 건 그게 아니라 "cam on으로 카메라를 켰을 때 코가 실제로 어디를
# 잡고 있는지 시각적으로" 보는 것이라는 정정을 받고 방식을 바꿈): 카메라
# 미리보기(cam on)에 원본 커서(계산된 화면 비율 위치)와는 별개로, 지금 이
# 프레임에서 실제로 잡은 랜드마크 원시 픽셀 위치에 점을 찍는다 — "커서가 어디
# 있는지"가 아니라 "지금 얼굴의 어느 점을 보고 있는지"를 곧이곧대로 보여준다.
# TRACKING_POINT_LABEL만 다르면 eyebrow.py도 그대로 재사용할 수 있게 이름을 상수로 뺐다
TRACKING_POINT_LABEL = "코"
TRACKING_POINT_COLOR = (0, 255, 255)   # 노랑(BGR) — 커서(초록)와 뚜렷이 구분

# 얼굴 변형 전환 순간 좌표 보류(2026-08-18 사용자 실기 보고 — "더블클릭 하는게
# 입 두번 움직이는 거잖아 그럼 커서가 위아래로 흔들리는 현상이 있어"):
# MediaPipe는 얼굴 전체를 하나로 묶어 추정해서, 입을 벌리거나(턱이 움직임)
# 눈을 감으면 코끝 좌표도 같이 살짝 흔들린다 — eyebrow.py가 미간(눈 사이)
# 좌표에서 이미 겪은 것과 같은 문제(그 파일 _glabella_point 독스트링 참고)다.
# 다만 코끝은 eyebrow.py의 미간과 처방이 다르다 — 이제 입을 0.7초 이상 벌리고
# 있으면 "꾹 누르기"(드래그)가 되는데(MOUTH_HOLD_SEC), 벌린 "동안" 내내
# 좌표를 얼려버리면 드래그 자체가 안 움직인다. 그래서 "벌려져 있는 동안
# 전부"가 아니라 "열림↔닫힘이 막 바뀐 순간"에만 짧게(NOSE_DEFORM_SUPPRESS_SEC)
# 좌표를 얼린다 — 더블클릭처럼 짧은 시간에 전환이 몰리면(입을 두 번 여닫음)
# 그 전환들이 전부 억제돼 흔들림이 사라지고, 꾹 누르기로 오래 벌리고 있는
# 동안은 첫 전환 억제가 끝난 뒤로는 평소처럼 정상 추적되어 드래그가 그대로 된다
# ★2026-08-18 문턱 재조정(사용자 실기 보고 "입벌릴때 커서가 움직여" — 재발):
# 그 직후 "입벌리는걸 인식을 잘 못하는거 같아" 요청으로 클릭 판정 문턱
# (MOUTH_OPEN_MARGIN_OVERRIDE)을 0.18→0.12로 낮췄는데, 이 억제 문턱은 그때
# 같이 안 낮춰서 0.15로 남아 있었다 — 그 결과 입을 벌리는 도중 jawOpen이
# 0.12~0.15 사이를 지나는 구간에서는 "클릭 판정 기준(jaw_base+0.12)으론 이미
# 열렸다"고 보는데 "억제 기준(절대값 0.15)으론 아직 안 열렸다"고 보는 어긋난
# 구간이 생겼다 — 그 구간 동안은 보호가 아예 안 걸려 코끝이 그대로 흔들렸다.
# 억제 문턱을 클릭 문턱보다 확실히 낮게 잡아 항상 억제가 먼저 걸리게 한다
# 입이 열린 뒤 랜드마크가 가라앉기를 기다리는 시간 — 이 뒤에 변형량을 잰다
NOSE_DEFORM_SETTLE_SEC = 0.12
# 변형 보정량의 상한(픽셀). 이보다 큰 차이는 얼굴이 일그러진 게 아니라 고개를
# 돌린 것이므로 보정하면 안 된다 — 안 막으면 입 벌린 채 고개를 돌릴 때 그 움직임
# 까지 빼버려서 드래그가 아예 안 먹는다
NOSE_DEFORM_MAX_PX = 14.0
NOSE_DEFORM_SUPPRESS_SEC = 0.2   # (옛 방식 잔재 — 지금은 안 쓴다)
NOSE_DEFORM_EYE_THRESHOLD = 0.35    # eyebrow.py EYE_CLOSE_SUPPRESS_THRESHOLD와 동일값
NOSE_DEFORM_MOUTH_THRESHOLD = 0.08  # config 기본 jaw_base가 보통 0에 가까워, MOUTH_OPEN_MARGIN_OVERRIDE(0.12)보다 확실히 낮게
_nose_deform_state = {
    "neutral": None,        # 입 다물었을 때의 얼굴 기준 코 위치 (천천히 갱신)
    "neutral_jaw": 0.0,     # 그때의 jawOpen 값
    "k": None,              # 턱 벌림 1당 코가 밀리는 양 (x, y) — 사람마다 다르다
    "was_open": False,
    "opened_at_sec": 0.0,
    "pre_open": None,       # 입이 열리기 직전의 위치 — k를 재는 기준
}


def _stable_nose_point(face):
    """커서 기준점(코) 픽셀 좌표 — HeadTracker의 cursor_point_fn으로 쓰인다.

    ★2026-08-24 재설계 — 입을 벌리고 있는 **내내** 커서가 안 움직이게.

    무엇이 문제였나: 예전엔 입/눈 상태가 바뀌는 **순간에만** 0.2초 코 좌표를
    얼렸다. 클릭(입을 잠깐 여닫음)에는 그걸로 충분했지만 드래그는 다르다 —
    드래그는 입을 0.7초 이상 계속 벌리고 있어야 시작되는데, 0.2초가 지난 나머지
    1초 동안은 **변형된 코 위치를 그대로 썼다**. 그래서 드래그가 시작될 때쯤엔
    커서가 이미 엉뚱한 데로 밀려 가 있었다. 사용자 보고 "입 벌려서 클릭하고
    드래그하려면 커서가 흔들려서 잘 안 된다"가 정확히 이것이다.

    어떻게 고쳤나: 얼리는 대신 **변형량을 빼준다**.
      1. 입을 다물고 있는 동안 얼굴 기준 코 위치를 평상시 값으로 계속 기억한다
      2. 입이 열리면, 열리기 직전 값과 열린 뒤 값의 차이 = 순수 변형량
      3. 입을 벌리고 있는 동안 그 차이를 계속 빼준다
    빼는 건 상수 벡터라 그 위에 얹히는 **진짜 고개 움직임은 그대로 통과한다** —
    입을 벌린 채로 고개를 돌려 드래그하는 게 정상 동작한다. 얼리는 방식으로는
    이게 불가능했다(얼리면 드래그 자체가 안 되니 0.2초로 짧게 둘 수밖에 없었다).

    좌표는 화면 픽셀이 아니라 **두 눈 중점 기준**으로 다룬다. 화면 좌표로 다루면
    그 사이 몸이 조금만 움직여도 보정량이 통째로 어긋난다(2026-08-20에 겪은 버그).
    """
    now_sec = time.monotonic()
    jaw = face.blendshape("jawOpen")
    is_open = jaw >= NOSE_DEFORM_MOUTH_THRESHOLD
    point = (face.landmarks_mean_px(LMK_NOSE_CLUSTER) if NOSE_CLUSTER_AVERAGING
             else face.landmark_px(LMK_NOSE_TIP))

    eye_left_px = face.landmark_px(LMK_LEFT_EYE_OUTER)
    eye_right_px = face.landmark_px(LMK_RIGHT_EYE_OUTER)
    eye_mid_x = (eye_left_px[0] + eye_right_px[0]) / 2.0
    eye_mid_y = (eye_left_px[1] + eye_right_px[1]) / 2.0
    local = (point[0] - eye_mid_x, point[1] - eye_mid_y)
    st = _nose_deform_state

    if not is_open:
        # 입을 다물고 있다 — 평상시 값을 천천히 따라가며 기억해 둔다.
        # 천천히(0.1) 가는 이유: 빨리 따라가면 고개를 돌리는 중의 값까지 평상시로
        # 기억해 버려서, 정작 입이 열렸을 때 뺄 기준이 흔들린다
        st["neutral"] = local if st["neutral"] is None else (
            st["neutral"][0] + 0.1 * (local[0] - st["neutral"][0]),
            st["neutral"][1] + 0.1 * (local[1] - st["neutral"][1]))
        st["neutral_jaw"] = st["neutral_jaw"] + 0.1 * (jaw - st["neutral_jaw"])
        st["pre_open"] = st["neutral"]
        st["was_open"] = False
        return point

    # 입이 벌어져 있다
    if not st["was_open"]:
        st["was_open"] = True
        st["opened_at_sec"] = now_sec
        st["k"] = None                     # 이번 벌림에 대해 계수를 새로 잰다

    # ★보정량을 "턱이 벌어진 정도에 비례"하게 만든다.
    #
    # 처음엔 벌리고 있는 내내 (지금 위치 - 열기 직전 위치)를 계속 재서 뺐다.
    # 그랬더니 **입을 벌린 채 고개를 돌리면 그 움직임까지 보정에 먹혔다** —
    # 드래그가 아예 안 되는 상태였다(실측: 6px 움직여야 하는데 0.23px만 움직임).
    #
    # 코가 밀리는 건 턱이 벌어졌기 때문이지 고개를 돌렸기 때문이 아니다. 그래서
    # "턱 벌림 1당 코가 몇 px 밀리는가"라는 계수 k를 벌림 시작 직후에 한 번 재고,
    # 그 뒤로는 k x (지금 턱 벌림) 을 뺀다. 고개를 돌려도 턱 벌림은 그대로이니
    # 보정량이 안 변해서 **움직임이 그대로 통과한다**. 입을 더 벌리거나 덜 벌리면
    # 그만큼 보정도 따라 커지고 작아진다.
    jaw_delta = jaw - st["neutral_jaw"]
    if st["k"] is None and st["pre_open"] is not None and jaw_delta > 1e-3:
        # 열린 직후 잠깐은 랜드마크가 출렁이므로 가라앉은 뒤에 잰다
        if now_sec - st["opened_at_sec"] >= NOSE_DEFORM_SETTLE_SEC:
            shift = (local[0] - st["pre_open"][0], local[1] - st["pre_open"][1])
            limit = NOSE_DEFORM_MAX_PX
            shift = (max(-limit, min(limit, shift[0])), max(-limit, min(limit, shift[1])))
            st["k"] = (shift[0] / jaw_delta, shift[1] / jaw_delta)

    if st["k"] is None:
        # ★계수를 아직 못 잰 구간(입이 열린 직후 NOSE_DEFORM_SETTLE_SEC 동안)은
        # 변형된 좌표를 그대로 내보내면 안 된다 — 실측에서 이 짧은 구간에만
        # 커서가 화면 세로의 16%나 튀었다(0.250 -> 0.414). 그동안은 평상시
        # 위치를 지금의 두 눈 중점에 얹어 내보내 커서를 붙들어 둔다.
        # 화면 좌표가 아니라 얼굴 기준이라, 그 사이 몸이 움직여도 안전하다.
        if st["neutral"] is not None:
            return (eye_mid_x + st["neutral"][0], eye_mid_y + st["neutral"][1])
        return point
    dx = st["k"][0] * jaw_delta
    dy = st["k"][1] * jaw_delta
    return (point[0] - dx, point[1] - dy)


def draw_cursor(frame, cursor_x_ratio, cursor_y_ratio, recenter_progress_ratio=0.0,
                color=None, filled=False):
    """visualize.draw_cursor와 같은 모양(원 + 십자 + 재정렬 진행 링)이지만 화면
    해상도 캔버스에 맞게 더 크게 그린다 — 위 CURSOR_RADIUS_PX 등 상수 설명 참고.

    color/filled는 클릭·드래그 피드백용(CursorFeedback 참고). 드래그 중엔 속을
    채워서, 색 구분이 어려운 사람도 "지금 누르고 있다"를 형태로 알 수 있게 한다.
    """
    if cursor_x_ratio is None:
        return frame
    color = color or CURSOR_COLOR
    h_px, w_px = frame.shape[:2]
    x_px, y_px = int(cursor_x_ratio * w_px), int(cursor_y_ratio * h_px)
    if filled:
        cv2.circle(frame, (x_px, y_px), CURSOR_RADIUS_PX - 6, color, -1)
    cv2.circle(frame, (x_px, y_px), CURSOR_RADIUS_PX, color, CURSOR_THICKNESS_PX)
    cv2.drawMarker(frame, (x_px, y_px), color, cv2.MARKER_CROSS,
                   CURSOR_MARKER_SIZE_PX, CURSOR_THICKNESS_PX)
    if recenter_progress_ratio > 0.0:
        end_angle_deg = 360.0 * recenter_progress_ratio
        cv2.ellipse(frame, (x_px, y_px), (CURSOR_RADIUS_PX + 8, CURSOR_RADIUS_PX + 8), -90, 0,
                    end_angle_deg, RECENTER_PROGRESS_COLOR, CURSOR_THICKNESS_PX)
    return frame


# ★2026-08-20 신설 — 오버레이 지우기를 "화면 전체"에서 "커서 주변만"으로.
#
# 이 오버레이는 화면 전체 해상도 캔버스(예: 1536x864 = 4MB)라, 매 틱
# `canvas[:] = 배경색`으로 통째로 지우는 비용이 컸다. 실측:
#
#     canvas[:] = 배경색 (전체 4MB)   중앙  6.9ms / 최악 11.1ms
#       └ 추론 스레드가 같이 돌 때는 13.1ms까지 올라감
#     커서 주변 작은 사각형만 지우기      중앙  0.02ms
#
# 30fps 예산이 33ms인데 지우기 하나가 7~13ms를 먹고 있었다는 뜻이다. 게다가
# 하필 **커서가 움직이는 동안**에만 이 비용을 낸다 — 정지 중엔 다시 그리기
# 자체를 건너뛰므로. 커서가 움직일 때가 곧 부드러움이 중요한 순간인데 그때만
# 느려지니, 체감되는 "뚝뚝 끊김"의 큰 원인이었다.
#
# 해법은 화면 갱신의 기본기인 더티 사각형(dirty rectangle) — 지난 틱에 실제로
# 뭔가 그린 영역만 배경색으로 되돌리고, 나머지는 이미 배경색이니 건드리지
# 않는다. 지우는 넓이가 4MB에서 커서 한 개 크기로 줄어 350배 빨라진다.
def _blank_rect(canvas, rect):
    """지난 틱에 그린 영역만 배경색으로 되돌린다. rect=None이면 할 일 없음."""
    if rect is None:
        return
    x0, y0, x1, y1 = rect
    canvas[y0:y1, x0:x1] = TRANSPARENT_KEY_COLOR


def _union_rect(a, b):
    """두 사각형을 모두 감싸는 사각형 — 커서와 안내 문구를 한 번에 지우려고."""
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _clip_rect(rect, w_px, h_px):
    """캔버스 밖으로 나간 좌표를 잘라낸다 — numpy 슬라이스는 음수를 반대편
    끝으로 해석하므로 그대로 두면 화면 반대쪽이 지워지는 사고가 난다."""
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w_px, int(x1)), min(h_px, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _cursor_rect(cursor_x_ratio, cursor_y_ratio, w_px, h_px):
    """draw_cursor가 실제로 칠하는 범위 — 원/십자/진행 링 중 가장 큰 것 기준에
    선 두께와 안티에일리어싱 여유를 더한다(모자라면 잔상이 남는다)."""
    if cursor_x_ratio is None:
        return None
    x_px, y_px = int(cursor_x_ratio * w_px), int(cursor_y_ratio * h_px)
    reach = max(CURSOR_RADIUS_PX + 8, CURSOR_MARKER_SIZE_PX // 2) + CURSOR_THICKNESS_PX + 2
    return _clip_rect((x_px - reach, y_px - reach, x_px + reach, y_px + reach), w_px, h_px)


# 한글 텍스트 렌더링(2026-08-13 사용자 실기 보고 — "글자가 ??? ???? ??????
# 이렇게 뜨는데"): cv2.putText는 OpenCV 내장 Hershey 폰트를 쓰는데, 이 폰트가
# 라틴 문자만 담고 있어 한글은 전부 "?"로 깨진다 — OpenCV 자체의 알려진 한계로,
# 옵션을 바꿔서 고칠 수 있는 문제가 아니다. 한글을 실제로 그리려면 한글
# 글리프가 있는 트루타입 폰트(맑은 고딕 — 이 PC 어디에나 있는 Windows 기본
# 글꼴, 이 프로젝트의 다른 문서 작업에서도 계속 써온 폰트)를 PIL로 그린 뒤
# numpy 배열로 합성하는 우회가 필요하다. cv2 그리기 함수들과 같은 관례로
# canvas를 in-place로 수정한다(반환값은 안 써도 되지만 체이닝 편의상 반환도 함)
_KOREAN_FONT_CACHE = {}


def _get_korean_font(size_px):
    font = _KOREAN_FONT_CACHE.get(size_px)
    if font is None:
        font = ImageFont.truetype("malgun.ttf", size_px)
        _KOREAN_FONT_CACHE[size_px] = font
    return font


def put_korean_text(canvas_bgr, text, org, font_size_px, color_bgr):
    """★2026-08-14 최적화("최적화해줘"): 원래는 canvas 전체를 PIL로 왕복
    변환했는데, 이 캔버스가 보통 화면 해상도(1920x1080 등) 풀사이즈라 글자
    한 줄 그리려고 매 렌더 틱마다 그 큰 배열 전체를 BGR<->RGB로 두 번씩(변환
    +되돌리기) 변환하고 있었다 — 특히 정지 유예 문구는 화면 주사율만큼(초당
    60회 이상) x 5초 x 두 줄이라 이 낭비가 눈에 띄게 누적된다. 실제 글자가
    차지하는 작은 영역(ROI)만 잘라내 그 부분만 변환·그리기·되붙이기 하도록
    바꿔, 변환 비용을 캔버스 전체 크기가 아니라 텍스트 영역 크기에 비례하게
    줄인다 — 그려지는 결과(캔버스에 보이는 픽셀)는 이전과 완전히 동일하다.
    """
    x_px, y_px = int(org[0]), int(org[1])
    canvas_h_px, canvas_w_px = canvas_bgr.shape[:2]
    font = _get_korean_font(font_size_px)
    text_w_px = int(font.getlength(text))
    pad_px = font_size_px // 2   # 글자가 폰트 박스 밖으로 살짝 삐져나오는 경우(획 삐침 등) 대비 여유
    x0, y0 = max(0, x_px - pad_px), max(0, y_px - pad_px)
    x1 = min(canvas_w_px, x_px + text_w_px + pad_px)
    y1 = min(canvas_h_px, y_px + font_size_px + pad_px * 2)
    if x1 <= x0 or y1 <= y0:
        return canvas_bgr   # 캔버스 밖 — 그릴 영역이 없다
    b, g, r = color_bgr
    roi_bgr = canvas_bgr[y0:y1, x0:x1]
    pil_image = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil_image).text((x_px - x0, y_px - y0), text, font=font, fill=(r, g, b))
    canvas_bgr[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return canvas_bgr


def _korean_text_width_px(text, font_size_px):
    """두 줄짜리 정지 유예 문구를 커서 중심에 맞춰 가운데 정렬하는 데 쓴다 —
    줄마다 글자 수·글꼴 크기가 달라 폭이 다르므로, 고정 오프셋을 하드코딩하는
    대신 실제 렌더 폭을 재서 매번 정확히 가운데 맞춘다."""
    return _get_korean_font(font_size_px).getlength(text)


logger = get_logger("scripts")


def enable_transparent_overlay(window_name, key_color_bgr, whole_window_alpha):
    """--overlay 전용 — cv2 창을 반투명 창으로 바꾼다.

    main_dpad.py의 동명 함수를 기반으로 하되, head.py는 **클릭 통과(click-through)**
    스타일을 하나 더 얹는다는 점이 다르다(2026-08-13 사용자 실기 보고로 추가
    — "select는 찍히는데 클릭이 잘 안되는데... cmd 창 누르니까 앞으로 안 와":
    main_dpad.py의 오버레이는 화면에 "보여주기만" 하고 실제 OS 클릭을 보낼
    일이 없어 WS_EX_LAYERED만으로 충분했지만, head.py는 mouse.click()으로
    **진짜 클릭을 배경 창·바탕화면까지 보내야** 한다. 그런데 이 오버레이 창은
    화면 전체를 덮는 최상위(topmost) 창이라, 눈에는 안 보여도(색상 키 투명)
    윈도우 자체는 그 자리에 그대로 있어서 클릭 입력을 이 창이 먼저 가로채
    버렸다 — 배경까지 클릭이 못 뚫고 내려간 원인. WS_EX_TRANSPARENT를
    WS_EX_LAYERED와 함께 켜면 이 창이 "마우스 입력에 대해 투명"해져서, 모든
    클릭이 이 창을 그냥 통과해 실제로 그 자리에 있는 배경 창(탐색기·cmd 등)에
    닿는다 — 시각적 투명(색상 키)과 입력 투명(클릭 통과)은 서로 다른 속성이라
    둘 다 켜야 한다.
    """
    if os.name != "nt":
        logger.warning("Transparent overlay is Windows-only - skipping (not on Windows)")
        return
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            logger.warning("Transparent overlay failed - could not find window %r", window_name)
            return
        gwl_exstyle = -20
        ws_ex_layered = 0x00080000
        ws_ex_transparent = 0x00000020   # 클릭 통과 — 모듈 함수 독스트링 참고
        lwa_colorkey = 0x00000001
        lwa_alpha = 0x00000002
        exstyle = user32.GetWindowLongW(hwnd, gwl_exstyle)
        user32.SetWindowLongW(hwnd, gwl_exstyle, exstyle | ws_ex_layered | ws_ex_transparent)
        b, g, r = key_color_bgr
        colorref = r | (g << 8) | (b << 16)   # Windows COLORREF = 0x00BBGGRR
        alpha = max(0, min(255, int(whole_window_alpha)))
        user32.SetLayeredWindowAttributes(hwnd, colorref, alpha, lwa_colorkey | lwa_alpha)
    except Exception:   # noqa: 방어적 — 오버레이 설정 실패가 시작을 막으면 안 된다
        logger.exception("Transparent overlay setup failed - continuing with a normal opaque window")


def disable_console_quick_edit():
    """콘솔 빠른 편집 해제 — main.py/main_dpad.py와 동일 로직·동일 이유
    (터치스크린 등에서 콘솔을 스치기만 해도 출력이 멈추는 것 방지)."""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        std_input_handle = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(std_input_handle, ctypes.byref(mode)):
            return
        quick_edit_flag = 0x0040
        extended_flags = 0x0080
        kernel32.SetConsoleMode(std_input_handle,
                                (mode.value & ~quick_edit_flag) | extended_flags)
    except Exception:   # noqa: 방어적
        pass



class _Win32Mouse:
    """실제 OS 마우스 포인터 제어 — ctypes만 사용(새 의존 추가 없음,
    enable_transparent_overlay와 동일 원칙). Windows 전용.

    화면 해상도는 GetSystemMetrics로 실측한다 — 이 PC/키오스크 모니터가
    실제로 몇 x 몇인지 가정하지 않는다(포터블).
    """

    SM_CXSCREEN = 0
    SM_CYSCREEN = 1
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("head.py의 실제 마우스 제어는 Windows 전용입니다")
        self._user32 = ctypes.windll.user32
        self.screen_w_px = self._user32.GetSystemMetrics(self.SM_CXSCREEN)
        self.screen_h_px = self._user32.GetSystemMetrics(self.SM_CYSCREEN)
        # press()로 눌러둔 상태인지 — release_if_pressed 독스트링 참고
        self.is_pressed = False
        # 마지막 클릭 시각 — 더블클릭 판정용(click 독스트링 참고)
        self._last_click_sec = 0.0

    def move(self, x_ratio, y_ratio):
        x_px = int(max(0.0, min(1.0, x_ratio)) * (self.screen_w_px - 1))
        y_px = int(max(0.0, min(1.0, y_ratio)) * (self.screen_h_px - 1))
        self._user32.SetCursorPos(x_px, y_px)

    def click(self):
        """현재 포인터 위치에서 실제 좌클릭 1회(다운+업) — 파일 아이콘 등을 클릭한다.

        ★두 번째 클릭이 DOUBLE_CLICK_WINDOW_SEC 안에 들어오면 윈도우가 확실히
        더블클릭으로 알아듣도록 **두 번을 바로 붙여서** 보낸다 (2026-08-26).

        왜 이렇게 하나 (실측):
        예전엔 "빠르게 두 번 벌리면 클릭이 두 번 나가고 윈도우가 알아서 묶는다"에
        맡겼다. 그런데 재보니 사람의 입 벌림 주기가 조금만 느려도 간격이 윈도우
        한계(GetDoubleClickTime, 보통 500ms)에 걸린다:

            입 열기 0.12 + 닫기 0.10초 -> 클릭 간격 313ms  (묶임)
            입 열기 0.25 + 닫기 0.25초 -> 클릭 간격 485ms  (아슬아슬)

        0.25초씩은 결코 느린 동작이 아닌데 485ms다. 조금만 더 느려지면 그냥
        단일 클릭 두 번이 되어, 사용자는 "더블클릭이 안 된다"고 느낀다.

        그래서 판정을 윈도우에 맡기지 않고 우리가 한다. 우리 창(0.9초)은 사람의
        실제 동작 속도에 맞춰 넉넉히 잡고, 두 번째 클릭일 때는 다운-업을 두 번
        연속으로 보낸다 — 그 둘 사이 간격은 사실상 0이라 윈도우가 무조건 더블
        클릭으로 처리한다.
        """
        now_sec = time.monotonic()
        is_second = (now_sec - self._last_click_sec) <= DOUBLE_CLICK_WINDOW_SEC
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        if is_second:
            self._user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self._user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self._last_click_sec = 0.0   # 세 번째가 또 묶이지 않게 창을 닫는다
        else:
            self._last_click_sec = now_sec

    def press(self):
        """좌클릭 누르기만(업 없음) — MOUTH_HOLD_SEC 이상 입을 벌리고 있을 때
        시작하는 "꾹 누르기"(드래그) 전용 (2026-08-18 신설, 사용자 요청 —
        "입 벌린채로 2초 이상 있으면 꾹 누르기로 해줘... 드래그 같은거").
        release()로 반드시 짝을 맞춰야 한다 — 안 그러면 왼쪽 버튼이 계속
        눌린 상태로 남는다."""
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.is_pressed = True

    def release(self):
        """press()로 시작한 좌클릭을 뗀다."""
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.is_pressed = False

    def release_if_pressed(self):
        """★눌러둔 게 있으면 무조건 뗀다 (2026-08-25 신설).

        드래그(꾹 누르기)로 왼쪽 버튼을 눌러둔 채 프로그램이 끝나면, 그 버튼은
        OS 수준에서 계속 눌린 상태로 남는다 — 바탕화면 전체가 드래그 상태가
        되어 아이콘이 딸려다니고 클릭이 전부 영역 선택이 된다. 화살표가
        사라지는 것보다 훨씬 심각하다(재부팅 말고는 사용자가 손쓸 방법이 없다).

        실측으로 두 경로에서 실제로 남는 걸 확인했다:
          · 드래그 중에 q/ESC·quit으로 종료      -> finally에서 뗄 사람이 없었다
          · 드래그 중에 p로 실제 제어를 끔        -> 뗄 때 "제어 중일 때만"이라는
                                                조건이 걸려 있어 건너뛰었다

        그래서 "제어가 켜져 있는지"가 아니라 **우리가 실제로 눌렀는지**만 본다 —
        눌렀으면 무슨 일이 있어도 뗀다. 여러 번 불려도 안전하고, 종료 경로에서
        불리므로 로거가 이미 닫혔을 수 있어 예외는 조용히 삼킨다."""
        if not self.is_pressed:
            return
        try:
            self.release()
        except Exception:   # noqa: 방어적 — 종료 중이라 더 할 수 있는 게 없다
            self.is_pressed = False


class _SystemCursorHider:
    """네이티브 OS 마우스 포인터(화살표)를 화면에서 숨긴다 (2026-08-13 사용자
    요청 — "마우스 포인터 말고 차라리 커서를 보이게 해줘": 진짜 포인터는
    SetCursorPos로 계속 옮기고 클릭도 그 위치에서 실제로 내되, 화면에는 그
    화살표 대신 오버레이가 그리는 초록 점(draw_cursor)만 보이게 한다).

    ★ShowCursor(FALSE)로는 안 된다 — 그 API는 "호출한 스레드 자신의 창" 위에서만
    적용되는 스레드 지역 카운터라, 바탕화면(탐색기, 다른 프로세스)을 가리키는
    동안에는 화살표가 그대로 나온다. 대신 SetSystemCursor로 **시스템 전역
    커서 리소스 자체**를 완전 투명 이미지로 바꿔치기한다(AND 마스크 전부
    1·XOR 마스크 전부 0 = 화면색 그대로 통과 = 안 보임, 고전 모노크롬 커서
    규칙) — 어느 창 위에 있든 동일하게 적용된다. 기본 화살표(OCR_NORMAL)와
    손가락 커서(OCR_HAND, 링크·일부 아이콘 호버)만 바꾼다 — 텍스트 입력창의
    I-빔 등 나머지 커서 모양은 이 도구의 사용 시나리오(바탕화면 아이콘
    클릭)에서 마주칠 일이 거의 없어 범위 밖으로 남겨둔다.

    복구는 SystemParametersInfoW(SPI_SETCURSORS)로 레지스트리 기본 커서를
    다시 불러오는 표준 방식 — SetSystemCursor로 바꾼 것 전부가 한 번에
    원복된다(Windows 공식 동작). 프로그램이 비정상 종료돼도(작업 관리자로
    강제 종료 등) 다음 SPI_SETCURSORS 호출이나 재로그인 시 자동 복구된다.
    """

    OCR_NORMAL = 32512
    OCR_HAND = 32649
    SPI_SETCURSORS = 0x0057

    def __init__(self):
        self._user32 = ctypes.windll.user32
        self._is_hidden = False

    def _blank_cursor_handle(self):
        w_px, h_px = 32, 32
        plane_size = (w_px // 8) * h_px
        and_mask = (ctypes.c_ubyte * plane_size)(*([0xFF] * plane_size))
        xor_mask = (ctypes.c_ubyte * plane_size)(*([0x00] * plane_size))
        return self._user32.CreateCursor(0, 0, 0, w_px, h_px, and_mask, xor_mask)

    def hide(self):
        try:
            for cursor_id in (self.OCR_NORMAL, self.OCR_HAND):
                handle = self._blank_cursor_handle()
                if handle:
                    # SetSystemCursor가 handle 소유권을 가져가 내부적으로 파괴한다 —
                    # 여기서 따로 DestroyCursor를 호출하면 안 된다(이중 해제)
                    self._user32.SetSystemCursor(handle, cursor_id)
            self._is_hidden = True
        except Exception:   # noqa: 방어적 — 실패해도 실제 클릭 기능엔 영향 없음
            logger.exception("시스템 커서 숨기기 실패 - 화살표가 계속 보일 수 있습니다")

    def restore(self, force=False):
        """네이티브 화살표를 되돌린다. force=True면 숨긴 적이 없더라도 실행한다
        (지난 실행이 강제 종료돼 화살표가 사라진 채 남아 있는 경우 대비 —
        install_exit_guards 설명 참고). 여러 번 불려도 안전하다."""
        if not self._is_hidden and not force:
            return
        try:
            self._user32.SystemParametersInfoW(self.SPI_SETCURSORS, 0, None, 0)
        except Exception:   # noqa: 방어적
            logger.exception("시스템 커서 복구 실패")
        self._is_hidden = False

    def install_exit_guards(self, mouse=None):
        """어떤 식으로 꺼지든 화살표가 반드시 돌아오게 안전장치를 건다
        (2026-08-20 사용자 요청 — "프로그램이 무조건 꺼지면 마우스 커서는
        되돌려 놓게 해줘").

        정상 종료 때는 main의 finally가 restore()를 부르지만, 그 경로를 안 타는
        종료가 여러 가지 있다. 각각을 따로 막는다:

          · 예외로 죽음 / sys.exit          -> atexit          (실측 확인)
          · 콘솔에서 Ctrl+C                 -> SIGINT
          · 콘솔 창의 X 버튼, 로그오프, 종료 -> SetConsoleCtrlHandler
                                              (파이썬 signal로는 안 오는 경로)
          · SIGTERM                        -> 등록은 해두지만 윈도우에서는 거의
                                              의미 없다(윈도우엔 진짜 SIGTERM이
                                              없다). 다른 OS 대비 + 형식상 등록

        못 막는 경로가 하나 있고, 이건 정직하게 인정하는 게 맞다: `taskkill /F`
        (내부적으로 TerminateProcess)처럼 OS가 프로세스를 즉시 없애버리는 경우다.
        실측으로도 이 경로에서만 복구가 안 됐다 — 어떤 프로그램도 이때는 정리
        코드를 돌릴 수 없다. 그래서 **시작할 때 무조건 한 번 복구**하는 것으로
        보완한다(main의 restore(force=True) 참고) — 그렇게 죽어 화살표가 사라진
        채였더라도 다시 실행하는 순간 곧바로 돌아온다. 트래커를 다시 켜기도
        싫을 땐 scripts/restore_cursor.py를 실행하면 된다.
        """
        # ★2026-08-25 — 화살표뿐 아니라 **눌러둔 마우스 버튼**도 함께 되돌린다.
        # 드래그 도중에 꺼지면 왼쪽 버튼이 눌린 채 남아 PC를 못 쓰게 된다
        # (_Win32Mouse.release_if_pressed 독스트링에 실측 경로가 있다). 버튼을
        # 먼저 떼고 화살표를 되돌린다 — 순서가 바뀌면 화살표가 돌아온 뒤에도
        # 잠깐 드래그 상태로 보인다
        def _cleanup():
            if mouse is not None:
                mouse.release_if_pressed()
            self.restore()

        atexit.register(_cleanup)

        def _on_signal(signum, _frame):
            _cleanup()
            # 기본 동작(종료)을 이어간다 — 여기서 붙잡고 있으면 안 꺼진다
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, _on_signal)
            except (ValueError, OSError):   # 메인 스레드가 아니거나 미지원 플랫폼
                pass

        # 콘솔 창 닫기·로그오프·시스템 종료 — signal로는 안 오는 경로라 Win32로 직접 받는다.
        # 핸들러 함수 객체는 파이썬이 수거하면 OS가 죽은 주소를 부르게 되므로 참조를 붙들어 둔다
        try:
            handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)

            def _on_console_event(_event):
                _cleanup()
                return False   # False = 기본 처리(종료)도 계속 진행

            self._console_handler = handler_type(_on_console_event)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(self._console_handler, True)
        except Exception:   # noqa: 방어적 — 안전장치 하나 실패해도 본 기능은 정상
            logger.exception("콘솔 종료 감지 등록 실패 - 창을 X로 닫으면 화살표가 안 돌아올 수 있습니다")


def _stdin_command_loop(state):
    """main.py의 stdin 명령 관례(cam on/off, quit)와 완전히 같은 이름·형식을
    쓴다 — p(마우스 제어 토글)만 이 파일 전용으로 추가.

    콘솔 없이 띄우면(pythonw, 서비스 등) sys.stdin이 None이라 그대로 두면 이
    스레드가 예외로 조용히 죽는다 — stderr도 없어 흔적조차 안 남는다. 명령
    입력만 포기하고 나머지는 정상 동작하게 한다."""
    if sys.stdin is None:
        logger.info("표준 입력이 없어 콘솔 명령(quit/p/cam)을 받지 않습니다")
        return
    try:
        _read_commands(state)
    except Exception:   # noqa: 방어적 — 명령 입력이 끊겨도 트래킹은 계속돼야 한다
        logger.exception("콘솔 명령 읽기 중단 - 트래킹은 계속됩니다")


def _read_commands(state):
    for line in sys.stdin:
        command = line.strip().lower()
        if command == "quit":
            state["should_quit"] = True
            break
        if command == "p":
            state["is_control_active"] = not state["is_control_active"]
            logger.info("실제 마우스 제어: %s", "ON" if state["is_control_active"] else "OFF")
        elif command == "cam on":
            state["show_camera"] = True
            logger.info("카메라 창: ON")
        elif command == "cam off":
            state["show_camera"] = False
            logger.info("카메라 창: OFF")


def _get_refresh_rate_hz(default_hz=60):
    """주 모니터의 실제 화면 주사율(Hz) — 2026-08-13 사용자 요청("화면 주사율에
    맞게 부드럽게") 대응. 렌더 루프를 이 값에 맞춰 페이싱한다.

    GetDeviceCaps(VREFRESH)를 쓴다 — EnumDisplaySettingsW도 후보였지만 그건
    DEVMODEW 전체 구조체(유니온 포함, 필드가 많고 버전별로 미묘하게 다르다)를
    정확한 크기로 직접 선언해야 해서, 크기를 하나라도 잘못 잡으면 Win32가 그
    버퍼 경계를 넘어 쓸 위험이 있다(메모리 손상). GetDeviceCaps는 단순
    정수 하나만 돌려주는 훨씬 안전한 API라 이쪽을 쓴다. 옛 문서엔 "일부 드라이버가
    VREFRESH로 0/1(=하드웨어 기본값)을 돌려줄 수 있다"는 주의가 있어, 그 경우
    (그리고 이 함수가 실패하는 모든 경우) default_hz로 조용히 대체한다 — 이
    값은 렌더 주기만 정할 뿐이라 틀려도 기능이 막히지 않고 부드러움만 덜해진다.
    """
    if os.name != "nt":
        return default_hz
    try:
        VREFRESH = 116
        user32 = ctypes.windll.user32
        hdc = user32.GetDC(0)
        if not hdc:
            return default_hz
        try:
            hz = ctypes.windll.gdi32.GetDeviceCaps(hdc, VREFRESH)
        finally:
            user32.ReleaseDC(0, hdc)
        if hz and hz > 1:
            return int(hz)
    except Exception:   # noqa: 방어적
        logger.exception("화면 주사율 조회 실패 - 기본값(%dHz)으로 대체", default_hz)
    return default_hz


def main():
    parser = argparse.ArgumentParser(
        description="헤드트래커 단독 실행 — 코끝 위치로 실제 OS 마우스 포인터를 옮기고 클릭한다")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--check", action="store_true",
                        help="설정·모델·카메라만 점검하고 끝낸다. 설치한 뒤 "
                             "제대로 돌아갈 상태인지 확인할 때 쓴다")
    parser.add_argument("--device", type=int, default=None, help="카메라 장치 번호 (기본: config device_id)")
    parser.add_argument("--no-window", action="store_true",
                        help="디버그 창을 띄우지 않는다 (콘솔의 quit로만 종료 가능해짐)")
    args = parser.parse_args()

    disable_console_quick_edit()
    config = load_config(args.config)
    # 자가 점검보다 먼저 초기화한다 — 점검 실패야말로 나중에
    # 로그로 확인해야 할 일이라, 이 순서가 뒤바뀌면 정작
    # 그 기록만 파일에 안 남는다
    init_logging(config)
    # ★어느 빌드가 돌고 있는지 먼저 남긴다 (src/utils/version.py 설명 참고) —
    # 현장 연락을 받았을 때 이 두 줄이 없으면 조사 자체를 시작할 수 없다
    logger.warning("%s", version.describe("head.py"))
    logger.info("실행 환경 — %s", version.environment())

    # ★시작 전 자가 점검 (src/utils/preflight.py 설명 참고).
    # 설정이 빠졌거나 모델이 없으면 예전에는 KeyError·FileNotFoundError만 뜨고
    # 죽었다 — 설치하는 사람이 개발자가 아니면 손을 쓸 수가 없다. 무엇이
    # 잘못됐고 어떻게 고치는지를 한 번에 모아서 알려준다.
    # 카메라 점검은 평소엔 건너뛴다 — 여는 데 2~3초가 걸리는데 곧바로 다시
    # 열기 때문이다. --check 로 돌릴 때만 카메라까지 본다.
    problems = preflight.run(config, check_cam=args.check)
    if preflight.report(problems):
        return 2
    if args.check:
        print(version.describe("head.py"))
        print(version.environment())
        print("점검 완료 — 바로 실행할 수 있는 상태입니다.")
        return 0
    if args.device is not None:
        config["camera"]["device_id"] = args.device
    if ACCURACY_CONF is not None:
        config["face_tracker"]["min_detection_conf"] = ACCURACY_CONF
        config["face_tracker"]["min_presence_conf"] = ACCURACY_CONF
        config["face_tracker"]["min_tracking_conf"] = ACCURACY_CONF
    config["face_tracker"]["infer_scale_ratio"] = INFER_SCALE_RATIO_OVERRIDE
    config["head_tracker"]["pointer"]["sensitivity_x"] = SENSITIVITY_X_OVERRIDE
    config["head_tracker"]["pointer"]["sensitivity_y"] = SENSITIVITY_Y_OVERRIDE
    config["head_tracker"]["pointer"]["smoothing_alpha"] = POINTER_SMOOTHING_ALPHA_OVERRIDE
    config["head_tracker"]["pointer"]["distance_smoothing_alpha"] = POINTER_DISTANCE_SMOOTHING_ALPHA_OVERRIDE
    config["head_tracker"]["mouth_click"]["open_margin"] = MOUTH_OPEN_MARGIN_OVERRIDE
    config["head_tracker"]["mouth_click"]["close_margin"] = MOUTH_CLOSE_MARGIN_OVERRIDE
    config["head_tracker"]["recenter_dwell"]["enabled"] = RECENTER_DWELL_ENABLED_OVERRIDE
    config["head_tracker"]["pointer"]["face_local"] = FACE_LOCAL_MAPPING
    config["head_tracker"]["pointer"]["face_local_gain"] = FACE_LOCAL_GAIN

    mouse = _Win32Mouse()
    logger.info("화면 해상도 %dx%d 감지 — 커서는 세로 %d~%dpx 구간(%s %.0f%%)만 사용",
                mouse.screen_w_px, mouse.screen_h_px,
                int(_cursor_y_to_screen(0.0) * mouse.screen_h_px),
                int(_cursor_y_to_screen(1.0) * mouse.screen_h_px),
                "하단" if CURSOR_Y_ANCHOR_BOTTOM else "상단", CURSOR_Y_SPAN * 100)
    # dpad_overlay의 D-pad 중심(cx_ratio=0.50, cy_ratio=0.30 고정 앵커)과 같은
    # 발상 — 얼굴이 아직 검출되기 전(캘리브레이션 완료 전)에도 커서가 항상
    # 정해진 자리(허용 영역 정중앙)에서 시작하게, 프로그램 시작과 동시에
    # 실제 OS 커서를 옮겨둔다. head_tracker 쪽도 캘리브레이션이 끝나는 순간
    # cursor_x/y_ratio를 0.5/0.5로 정의하므로(_CursorMapper.update 참고),
    # 이후 매 프레임 갱신되는 실제 위치와 자연스럽게 이어진다.
    CENTER_X_RATIO, CENTER_Y_RATIO = 0.5, 0.5
    mouse.move(CENTER_X_RATIO, _cursor_y_to_screen(CENTER_Y_RATIO))

    # ★2026-08-14 사용자 실기 보고("둘다 실행시키고나면 커서가 안보여") 대응 —
    # 예전엔 여기서 프로그램 시작과 동시에 네이티브 화살표를 숨겼는데, 오버레이
    # 초록 점은 첫 캘리브레이션이 끝나야(모델 로딩 수 초 + 캘리브레이션
    # 0.5초, 카메라·조명 사정에 따라 더 걸릴 수도 있다) 그려지기 시작한다 —
    # 그 사이엔 화살표도 없고 점도 없어 커서가 통째로 안 보이는 구간이 생겼다.
    # 이제 hide()는 여기서 바로 부르지 않고, 아래 렌더 루프에서 실제로 추적이
    # 시작되는(오버레이 점을 그리기 시작하는) 첫 틱에만 한 번 부른다 — 그
    # 전까지는 원래 화살표가 그대로 보여 "커서가 없다"는 상태 자체가 없어진다
    cursor_hider = _SystemCursorHider()
    # ★2026-08-20 사용자 요청 — "프로그램이 무조건 꺼지면 마우스 커서는 되돌려
    # 놓게 해줘". 두 방향으로 보장한다:
    #  ① 시작할 때 무조건 한 번 복구 — 지난 실행이 강제 종료(taskkill /F 등)돼
    #     화살표가 사라진 채 남아 있어도 여기서 되돌아온다. 강제 종료는 어떤
    #     프로그램도 정리 코드를 못 돌리므로, "다음 실행 때 고친다"가 유일한 답이다
    #  ② 끝날 때 어떤 경로로든 복구 — 예외·Ctrl+C·종료 요청·콘솔 창 닫기까지
    #     (install_exit_guards 독스트링에 경로별로 정리)
    cursor_hider.restore(force=True)
    cursor_hider.install_exit_guards(mouse)

    preprocessor = Preprocessor(config)
    face_estimator = FaceEstimator(config)
    # "뒷사람" 방어(2026-08-18 사용자 요청 — "뒷사람 인식 안되게 하는것도
    # 참고해서 리모콘 ui 모두 적용시켜줘") — 매 프레임 독립적으로 "가장 큰
    # 얼굴"만 고르던 select_user_face() 대신, 한 번 정한 사용자를 위치·크기
    # 연속성으로 계속 같은 사람으로 붙잡는다(src/postprocess/face_anchor.py
    # 독스트링 참고 — C:\dev\GMtech_project의 손 추적 앵커 시스템과 같은 원리)
    face_anchor = FaceAnchor(config)
    # 코끝 좌표 안정화(_stable_nose_point 독스트링 참고) — "커서가 위아래로
    # 흔들리는" 문제 대응, 기본(코끝 그대로)을 이 안정화 버전으로 교체
    head_tracker = HeadTracker(config, cursor_point_fn=_stable_nose_point)
    camera = CameraStream(config, config_path=args.config).start()

    state = {"is_control_active": True, "should_quit": False, "show_camera": False}
    stdin_thread = threading.Thread(target=_stdin_command_loop, args=(state,), daemon=True)
    stdin_thread.start()

    # 카메라 미리보기 공유 상태(2026-08-13 사용자 요청 — "cam on 하면 카메라
    # 보이게") — 추론 스레드가 매 프레임 채워두고, 렌더 스레드(cv2 창을 도맡는
    # 쪽)가 state["show_camera"]일 때만 그걸 꺼내 보여준다. camera_frame은
    # preprocessor가 이미 만든 실제 카메라 픽셀 프레임 자체를 재사용한다
    camera_preview_lock = threading.Lock()
    camera_preview_state = {"frame": None}

    # 추론 스레드(카메라·MediaPipe, ~20~30Hz)와 렌더 스레드(마우스 이동·오버레이
    # 그리기, 화면 주사율)를 분리한다 — 2026-08-13 사용자 요청("화면 주사율 맞게
    # 부드럽게 움직이게"). 공유 상태는 target_lock으로 보호. 클릭은 판정이
    # 확정되는 그 순간 정확히 한 번만 나가야 하는 이산 이벤트라 보간 없이
    # 추론 스레드에서 바로 처리한다(렌더 스레드에서 하면 프레임 사이 값이
    # 보간되는 도중 이벤트 자체가 뭉개지거나 중복될 위험이 있다)
    target_lock = threading.Lock()
    target_state = {
        "x_ratio": CENTER_X_RATIO, "y_ratio": _cursor_y_to_screen(CENTER_Y_RATIO),
        "is_tracking": False, "recenter_progress": 0.0, "became_tracking_sec": 0.0,
    }
    # is_tracking의 False->True 전이(=캘리브레이션이 막 끝난 순간)를 감지하는
    # 용도 — 매 프레임 덮어써지는 target_state와 달리 "직전 프레임엔 어땠는지"
    # 기억해야 해서 별도로 둔다(아래 SETTLE_DELAY_SEC 대응)
    tracking_edge_state = {"was_tracking": False}
    # ★2026-08-13 사용자 실기 보고 "커서가 좀 튀어서" 대응 — 원인 추정: head_tracker
    # 는 얼굴이 단 한 프레임만 미검출돼도 즉시 전체 리셋(reset())한다
    # (head_tracker.py update() 참고) — 캘리브레이션·기준선이 전부 날아가서
    # 다음 정상 프레임에서 커서가 화면(위쪽 절반) 중앙으로 순간 이동했다가
    # 새로 캘리브레이션되며 다시 쫓아오는데, 이게 "튀는" 것처럼 보였을 것으로
    # 추정된다. mode_switch.head_shake.dropout_grace_sec와 같은 발상으로,
    # 아주 짧은 순간의 미검출은 마지막으로 검출됐던 얼굴을 그대로 재사용해
    # 봐준다 — head_tracker 쪽 리셋 로직(다른 곳에서도 쓰는 검증된 코드)은
    # 건드리지 않고, 그 앞단에서 "진짜 놓쳤을 때만" None을 넘기도록 걸러낸다
    DROPOUT_GRACE_SEC = 0.3
    last_face_state = {"face": None, "sec": 0.0}
    # 마지막으로 추론한 카메라 프레임 번호 — 아래 capture_new_frame 주석 참고
    frame_seq_state = {"seq": -1}

    # ★감시용 건강 신호 (WATCHDOG_STALL_SEC 상수 설명 참고). 추론 스레드만
    # 쓰고 렌더 스레드만 읽는다 — 값이 실수 하나뿐이라 파이썬에서 쓰기가
    # 쪼개지지 않으므로 락이 필요 없다(설령 한 틱 늦게 읽혀도 판정 기준이
    # 5초라 아무 차이가 없다). 0.0은 "아직 시작 안 함"이라 감시도 쉰다
    health_state = {"loop_sec": 0.0, "frame_sec": 0.0}

    # 입 제스처 상태(MOUTH_HOLD_SEC 상수 설명 참고)
    mouth_gesture_state = {
        "is_open": False,        # 지금 입이 "벌어진" 상태로 판정 중인지(히스테리시스 래치)
        "open_since_sec": None,  # 이번에 벌어지기 시작한 시각
        "is_holding": False,     # MOUTH_HOLD_SEC를 넘어 이미 마우스를 누르고 있는 중인지
        "close_since_sec": None,  # 닫힘 문턱 아래로 내려간 시각 — MOUTH_CLOSE_CONFIRM_SEC 확인용
    }

    feedback = CursorFeedback()   # 클릭·드래그를 커서 색으로 알린다

    def _reset_recenter_timer():
        """클릭·드래그가 일어나면 자동 재정렬(캘리브레이션) 대기 시간을 처음부터
        다시 세게 한다 (2026-08-20 사용자 요청 — "클릭하거나 드래그 등 할 때
        캘리브레이션 시간 초기화").

        조작 중이라는 건 사용자가 화면을 잘 쓰고 있다는 뜻이라 재정렬이 필요
        없는 상태다. 그런데 재정렬은 '커서가 한자리에 머무는 시간'만 보기 때문에,
        아이콘을 겨냥하려고 커서를 멈춰 두는 동작이 그대로 재정렬 조건이 되어
        조작 도중에 커서가 중앙으로 튀는 일이 있었다. 클릭·드래그 때마다 시계를
        0으로 되돌려 조작 중에는 재정렬이 끼어들지 못하게 한다.

        ★2026-08-20 수정 — 처음엔 reset_event_gates()(모든 판정 초기화)를 불렀는데
        그게 재정렬을 아예 불가능하게 만들었다. 가만히 있으면 1.5초마다 응시
        클릭이 나가고 → 그 클릭이 다시 재정렬 시계를 0으로 되돌리는 무한 루프라
        10초를 영영 못 채운다(실측 로그: 27초 동안 응시 클릭 13번, 재정렬 0번).
        덤으로 응시 클릭의 '한 번 누르면 커서가 반경을 벗어나야 다시 누름' 규칙도
        같이 지워져서 클릭이 계속 반복됐다. 이제 재정렬 시계만 콕 집어 되돌린다 —
        head_tracker.reset_recenter_dwell 독스트링 참고."""
        head_tracker.reset_recenter_dwell()

    def _release_mouth_hold_if_stuck():
        """추적을 잃으면(진짜 미검출·재캘리브레이션 등) 입 제스처 상태를
        통째로 되돌린다 — 특히 꾹 누르기(드래그) 도중 얼굴을 놓치면 마우스
        왼쪽 버튼이 영영 눌린 채로 남는 사고를 막는다(2026-08-18 신설)."""
        if mouth_gesture_state["is_holding"]:
            try:
                # ★제어 토글(p) 상태와 무관하게 뗀다 — 누른 건 우리이므로
                # 떼는 것도 우리 책임이다(release_if_pressed 독스트링 참고)
                mouse.release_if_pressed()
                logger.info("추적 끊김 - 꾹 누르기 강제 해제 (drag release)")
            except Exception:   # noqa: 방어적
                logger.exception("꾹 누르기 강제 해제 실패")
        mouth_gesture_state["is_open"] = False
        mouth_gesture_state["open_since_sec"] = None
        mouth_gesture_state["is_holding"] = False
        mouth_gesture_state["close_since_sec"] = None
        feedback.set_holding(False)   # 색도 같이 되돌린다 — 안 그러면 드래그 색이 남는다

    def _update_mouth_gesture(now_sec):
        """입을 짧게 벌렸다 닫으면 그 즉시 클릭 1회(지연 없음 — 위
        MOUTH_HOLD_SEC 상수 설명 참고, "바로바로 클릭 되게" 요청 대응).
        MOUTH_HOLD_SEC 이상 계속 벌리고 있으면 꾹 누르기로 전환해 누른 채
        유지하고, 입을 다물 때 뗀다 — 그동안 고개를 움직이면 실제 드래그가
        된다. head_tracker.debug의 jaw_open/jaw_base를 직접 읽어
        MOUTH_OPEN/CLOSE_MARGIN_OVERRIDE와 같은 문턱으로 히스테리시스를
        건다(head_tracker.py의 mouth_gate와 같은 문턱값, 별개의 판정 상태) —
        dwell(응시) select는 이 함수와 무관하게 기존 그대로 즉시 단일 클릭이다.
        더블클릭은 여기서 판정하지 않는다 — 입을 빠르게 두 번 벌리면 클릭이
        실제로 짧은 간격으로 두 번 나가고, Windows가 그 간격만 보고 알아서
        더블클릭으로 인식한다(물리 마우스와 원리가 같다)."""
        jaw_open = head_tracker.debug.get("jaw_open")
        jaw_base = head_tracker.debug.get("jaw_base")
        if jaw_base is None:   # 기준선 미확정(캘리브레이션 중) — 판정 보류
            return

        if not mouth_gesture_state["is_open"]:
            if jaw_open >= jaw_base + MOUTH_OPEN_MARGIN_OVERRIDE:
                mouth_gesture_state["is_open"] = True
                mouth_gesture_state["open_since_sec"] = now_sec
                mouth_gesture_state["close_since_sec"] = None
        else:
            held_sec = now_sec - mouth_gesture_state["open_since_sec"]
            # 드래그 중이면 더 깊이·더 오래 닫아야 놓아준다
            # (MOUTH_HOLD_RELEASE_* 상수 설명 참고)
            holding = mouth_gesture_state["is_holding"]
            close_margin = (MOUTH_HOLD_RELEASE_MARGIN if holding
                            else MOUTH_CLOSE_MARGIN_OVERRIDE)
            close_confirm = (MOUTH_HOLD_RELEASE_CONFIRM_SEC if holding
                             else MOUTH_CLOSE_CONFIRM_SEC)
            is_below = jaw_open <= jaw_base + close_margin
            if is_below:
                # 닫힘 문턱 아래 — 다만 이 상태가 MOUTH_CLOSE_CONFIRM_SEC 동안
                # 이어져야 진짜 닫힘으로 본다(상수 설명 참고). 턱 떨림으로 잠깐
                # 내려간 것까지 닫힘으로 세면 드래그가 매번 중간에 끊긴다
                if mouth_gesture_state["close_since_sec"] is None:
                    mouth_gesture_state["close_since_sec"] = now_sec
                elif now_sec - mouth_gesture_state["close_since_sec"] >= close_confirm:
                    mouth_gesture_state["is_open"] = False
                    mouth_gesture_state["close_since_sec"] = None
                    if mouth_gesture_state["is_holding"]:
                        mouth_gesture_state["is_holding"] = False
                        feedback.set_holding(False)      # 드래그 색 해제
                        # 제어를 끈 상태여도 반드시 뗀다 — 안 그러면 버튼이
                        # 눌린 채 남는다(release_if_pressed 독스트링 참고)
                        mouse.release_if_pressed()
                        console.emit("hold_end")
                        logger.info("꾹 누르기 종료 (trigger=mouth, drag release)")
                    else:
                        if state["is_control_active"]:
                            mouse.click()
                        feedback.flash(now_sec)          # 클릭 1회 = 한 번 깜빡
                        console.emit("select")
                        logger.info("클릭 (trigger=mouth)")
                    _reset_recenter_timer()
            else:
                # 잠깐 내려갔다 다시 올라옴 — 닫힘 후보를 취소하고 벌린 시간은 계속 쌓는다
                mouth_gesture_state["close_since_sec"] = None
                # ★홀드 시작 판정을 여기(닫힘 문턱 위)에서만 하는 건 의도적이다.
                # 닫히는 중일 수도 있는 프레임에서 드래그를 시작하면, 시작하자마자
                # 끝나는 헛동작이 된다
                if not mouth_gesture_state["is_holding"] and held_sec >= MOUTH_HOLD_SEC:
                    # 계속 벌리고 있다 — 꾹 누르기 시작
                    mouth_gesture_state["is_holding"] = True
                    feedback.set_holding(True)           # 누르는 동안 색 유지
                    if state["is_control_active"]:
                        mouse.press()
                    console.emit("hold_start")
                    logger.info("꾹 누르기 시작 (trigger=mouth, drag press)")
                    _reset_recenter_timer()
                elif mouth_gesture_state["is_holding"]:
                    # 드래그 중에는 계속 시계를 되돌린다 — 드래그하려고 커서를
                    # 천천히 움직이는 동안 재정렬이 끼어들면 안 된다
                    _reset_recenter_timer()

    def _process_one_frame():
        # 머리 모드는 항상 9:16 세로 크롭 — main_dpad.py 머리 모드와 동일 조건
        # (이 스크립트는 애초에 머리 모드 전용이라 조건 분기 없이 항상 적용한다).
        # 기본은 화면에 안 그리지만(카메라를 안 보여줘도 된다는 요청, 모듈
        # 독스트링 참고), "cam on"으로 켜면 아래에서 이 프레임에 계기판을
        # 그려 미리보기로 내보낸다
        # ★2026-08-20 capture_frame → capture_new_frame 교체(사용자 요청 "프레임
        # 잘나오게"): capture_frame은 지금 있는 프레임을 곧바로 돌려주기만 해서,
        # 추론이 카메라(30fps)보다 빨라지는 순간 **같은 프레임을 반복해서 추론**한다.
        # 같은 입력은 같은 결과라 얻는 정보는 0인데 CPU는 그대로 쓰고, 무엇보다
        # 그동안 이 스레드가 계속 돌면서 커서를 그리는 렌더 쪽을 방해한다.
        # capture_new_frame은 새 프레임이 올 때까지 재우므로(그동안 렌더가
        # 원활해진다) 추론 속도가 카메라 속도에 자연스럽게 맞춰진다 —
        # src/pipeline/realtime_loop.py는 원래 이 방식을 쓰고 있었는데
        # head.py만 빠져 있었다
        previous_seq = frame_seq_state["seq"]
        raw_frame, frame_seq_state["seq"] = camera.capture_new_frame(previous_seq)
        if frame_seq_state["seq"] != previous_seq:
            # 진짜 새 프레임일 때만 찍는다 — 카메라가 멈추면 같은 그림이 계속
            # 돌아오는데, 그것까지 "살아 있다"로 세면 고장을 영영 못 잡는다
            health_state["frame_sec"] = time.monotonic()
        frame = preprocessor.preprocess_frame(raw_frame, apply_crop=True)

        faces = face_estimator.infer(frame)
        user_face = face_anchor.update(faces)
        now_sec = time.monotonic()
        if user_face is not None:
            last_face_state["face"] = user_face
            last_face_state["sec"] = now_sec
        elif now_sec - last_face_state["sec"] < DROPOUT_GRACE_SEC:
            user_face = last_face_state["face"]   # 짧은 순간의 미검출 — 마지막 얼굴로 봐준다
        result = head_tracker.update(user_face)

        if state["show_camera"]:
            # main_dpad.py 머리 모드의 기본(비-overlay) 미리보기와 같은 구성 —
            # 카메라 해상도 기준으로 만들어진 원본 draw_cursor_camera·
            # draw_head_debug_panel을 그대로 쓴다(이 카메라 프레임 자체가 그
            # 해상도라 크기를 새로 맞출 필요가 없다). 캘리브레이션 중이라
            # cursor_x_ratio가 None이어도 계기판은 보여준다 — 그래야 캘리브레이션
            # 진행 상황을 볼 수 있다(draw_cursor_camera는 None이면 커서만 생략)
            preview_frame = draw_cursor_camera(
                frame, result.cursor_x_ratio, result.cursor_y_ratio,
                head_tracker.debug.get("recenter_progress", 0.0),
            )
            preview_frame = draw_head_debug_panel(preview_frame, head_tracker.debug)
            # 추적 기준점 실측 위치(위 TRACKING_POINT_COLOR 주석 참고) — 얼굴이
            # 잡혀 있을 때만 그린다(user_face가 None이면 찍을 점 자체가 없다)
            if user_face is not None:
                point_x_px, point_y_px = user_face.landmark_px(LMK_NOSE_TIP)
                point_px = (int(point_x_px), int(point_y_px))
                cv2.circle(preview_frame, point_px, 10, TRACKING_POINT_COLOR, -1, cv2.LINE_AA)
                cv2.circle(preview_frame, point_px, 14, TRACKING_POINT_COLOR, 2, cv2.LINE_AA)
                # 한글이라 cv2.putText로는 안 그려진다 — put_korean_text 독스트링 참고
                put_korean_text(preview_frame, TRACKING_POINT_LABEL,
                                (point_px[0] + 18, point_px[1] - 10), 22, TRACKING_POINT_COLOR)
            with camera_preview_lock:
                camera_preview_state["frame"] = preview_frame

        if result.cursor_x_ratio is None:
            with target_lock:
                target_state["is_tracking"] = False
            tracking_edge_state["was_tracking"] = False
            _release_mouth_hold_if_stuck()
            return

        # False->True 전이 순간(캘리브레이션이 막 끝난 프레임)을 기록 — 렌더
        # 스레드가 이 시각 기준으로 SETTLE_DELAY_SEC 동안 정지 유예를 준다.
        # became_tracking_sec을 이 스레드 자체에도 따로 들고 있는다(target_state의
        # 값과 같은 값이지만, 매 프레임 락을 잡고 읽어오는 대신 이미 아는 값을
        # 그대로 재사용 — 아래 이벤트 억제 판단에 쓴다)
        if not tracking_edge_state["was_tracking"]:
            tracking_edge_state["became_tracking_sec"] = now_sec
            with target_lock:
                target_state["became_tracking_sec"] = now_sec
            tracking_edge_state["was_tracking"] = True
            # 이번 정지 유예 사이클의 "유예 끝날 때 재중심" 여부 — 아래
            # recenter_cursor() 호출부 참고. 새 유예가 시작될 때마다 다시 False로
            tracking_edge_state["recentered_after_settle"] = False

        # 세로는 위쪽 절반으로 재매핑(모듈 독스트링 참고) — 가로는 그대로
        screen_y_ratio = _cursor_y_to_screen(result.cursor_y_ratio)
        with target_lock:
            target_state["x_ratio"] = result.cursor_x_ratio
            target_state["y_ratio"] = screen_y_ratio
            target_state["is_tracking"] = True
            target_state["recenter_progress"] = head_tracker.debug.get("recenter_progress", 0.0)

        # 콘솔 출력 방침(2026-08-13 사용자 요청 — main_dpad.py와 동일 관례):
        # 매 프레임 연속으로 xy 좌표, select/home/calibration 발화 시 각각
        # 한 줄(접두어 없이 이름만). 여기 xy는 실제 커서가 놓이는 좌표(화면
        # 위쪽 절반 재매핑 반영 후) — head_tracker 원본 비율이 아니라 이
        # 화면에 실제로 쓰이는 값을 보여준다
        console.emit(f"x={result.cursor_x_ratio:.3f} y={screen_y_ratio:.3f}")

        # 정지 유예 중엔 모든 키(select/home/calibration)를 무시한다(2026-08-13
        # 사용자 요청 — "모든 키가 안되게 해줘야해 5초 정지 안에"): 커서는 render
        # 스레드가 화면에서 안 움직이는 척만 하고 있을 뿐, head_tracker 자체의
        # 판정(입 벌리기·응시 등)은 이 스레드에서 평소와 다름없이 계속 돌고 있어
        # 그대로 두면 화면은 멈춰 보이는데 뒤에서 클릭이 실제로 나가는 모순이
        # 생긴다 — 그래서 이벤트 자체를 여기서 걸러낸다(발화 로그도 함께 생략,
        # 안 그러면 "아무 반응 없어 보이는데 select는 찍히는" 혼란을 또 만든다).
        # ★2026-08-14 추가 — "5초지나도 가만히있으면 또 캘리브레이션 되는
        # 현상": 위에서 이벤트 "발화"만 걸렀지, head_tracker 내부의 응시(dwell)
        # 게이트는 update() 호출 때마다 매 프레임 계속 진행이 쌓이고 있었다 —
        # 유예 중에 가만히 있으면 정지 유예가 끝나기도 전에 이미 다 채워져
        # 있다가, 유예가 끝나자마자(또는 직후) 곧바로 재정렬이 발화한 것.
        # reset_event_gates()로 게이트 진행 자체를 매 프레임 지워서, 유예가
        # 끝난 시점부터 응시 판정이 0부터 다시 시작하게 한다(head_tracker.py
        # reset_event_gates 독스트링 참고 — 커서 캘리브레이션은 안 건드림)
        if now_sec - tracking_edge_state["became_tracking_sec"] < SETTLE_DELAY_SEC:
            head_tracker.reset_event_gates()
            return

        # 정지 유예가 막 끝난 첫 프레임에 딱 한 번 — 지금 이 자세를 새 중심으로
        # 확정한다(2026-08-18 사용자 요청 — "캘리브레이션 되면 백그라운드 커서
        # 값도 무조건 사용자가 어떤 자세를 취하던 커서가 중앙에 가있게 고정해줘").
        # head_tracker.recenter_cursor 독스트링 참고 — reset()과 달리 재수집
        # 구간이 없어 is_tracking이 끊기지 않으므로, 이 재중심 자체가 새로운
        # "얼굴이 새로 잡힘" 판정으로 오인돼 정지 유예가 재귀적으로 다시
        # 시작되는 일이 없다
        if not tracking_edge_state["recentered_after_settle"]:
            # 성공했을 때만 "했다"고 표시한다 — 실패했는데 표시해 버리면 이번
            # 유예 사이클에선 다시 시도하지 않아, "유예가 끝나면 커서는 무조건
            # 화면 중앙"이라는 보장이 조용히 깨진다
            if head_tracker.recenter_cursor(user_face):
                tracking_edge_state["recentered_after_settle"] = True

        # 입 트리거 select는 이제 _update_mouth_gesture가 전담한다(위 함수
        # 독스트링·MOUTH_HOLD_SEC 상수 설명 참고 — 단일/더블/꾹 누르기 세 갈래로
        # 나누려면 head_tracker의 "벌리면 즉시 클릭 1회" 판정을 그대로 쓸 수
        # 없다) — 그 트리거만 건너뛴다. dwell(응시)은 예전처럼 즉시 단일 클릭
        for event in result.events:
            trigger = event.data.get("trigger") if event.data else None
            if event.class_name == EVENT_SELECT and trigger == "mouth":
                continue
            console.emit(event.class_name)
            if event.class_name == EVENT_SELECT and state["is_control_active"]:
                try:
                    mouse.click()
                    feedback.flash(time.monotonic())
                    logger.info("클릭 (trigger=%s)", trigger)
                    _reset_recenter_timer()   # 응시 클릭도 재정렬 시계를 되돌린다
                except Exception:   # noqa: 방어적 — 클릭 실패로 프레임 처리가 중단되면
                                    # 이후 select가 발화해도 아무 반응이 없어 보여 디버깅이
                                    # 훨씬 어려워진다. 로그만 남기고 다음 프레임을 계속 처리한다
                    logger.exception("클릭 시도 실패")

        _update_mouth_gesture(now_sec)

    def _inference_loop():
        # ★같은 오류가 반복될 때 기록을 줄인다 (2026-08-25 신설).
        #
        # 예외를 삼키고 계속 도는 건 맞다 — 스레드가 조용히 죽으면 이후 조작이
        # 전부 먹통이 되는데 원인을 알 수 없다. 문제는 **오류가 계속되는 상황**
        # 이었다. 카메라를 뽑아두면 매 프레임 실패하고, 초당 수십 건의 기록이
        # 쌓인다. 무인 키오스크에서 밤새 그러면 로그가 디스크를 채우고 결국
        # 기계가 멈춘다(로그 회전도 함께 넣었지만, 애초에 덜 쓰는 게 낫다).
        #
        # 그래서 같은 오류가 이어지면 처음 한 번만 자세히 남기고, 그 뒤로는
        # 간격을 두고 "몇 번째로 반복 중"만 적는다. 오류가 바뀌거나 한 번이라도
        # 정상 처리되면 다시 처음부터 자세히 남긴다 — 새 문제를 놓치지 않게.
        last_err_key = None
        err_streak = 0
        last_err_log_sec = 0.0
        while not state["should_quit"]:
            # 예외가 나도 찍는다 — 이건 "루프가 돌고 있는가"만 보는 신호다.
            # 오류 자체는 아래에서 따로 기록한다
            health_state["loop_sec"] = time.monotonic()
            try:
                _process_one_frame()
                if err_streak:
                    logger.info("프레임 처리 정상 복구 (오류 %d회 뒤)", err_streak)
                last_err_key, err_streak = None, 0
            except Exception as exc:   # noqa: 방어적 — 위 설명 참고
                key = (type(exc).__name__, str(exc)[:120])
                now_sec = time.monotonic()
                if key != last_err_key:
                    logger.exception("프레임 처리 중 오류 - 다음 프레임에서 계속 시도")
                    last_err_key, err_streak, last_err_log_sec = key, 1, now_sec
                else:
                    err_streak += 1
                    if now_sec - last_err_log_sec >= ERROR_REPEAT_LOG_SEC:
                        logger.warning("같은 오류가 %d회째 반복 중: %s",
                                       err_streak, key[1])
                        last_err_log_sec = now_sec

    # ★콘솔 출력을 전담 스레드로 넘긴다 (src/utils/console.py 설명 참고) —
    # 출력을 파이프로 받아가는 쪽이 읽지 않으면 print가 영영 안 돌아와서
    # 추론이 통째로 멈춘다(실측 8.2초). 큐가 차면 버릴 뿐 막히지 않는다
    console.start()
    inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    inference_thread.start()

    # 오버레이 캔버스 — 카메라가 아니라 실제 화면 해상도 크기로 잡는다(모듈
    # 독스트링 참고: 화면에 그려지는 커서 점과 진짜 OS 커서가 겹쳐 보이려면
    # 캔버스 좌표계가 화면 좌표계와 같아야 한다)
    overlay_h_px, overlay_w_px = mouse.screen_h_px, mouse.screen_w_px
    # 최적화(2026-08-13 사용자 요청 — "최대한 최적화해줘"): np.full로 매 렌더
    # 틱(주사율만큼, 보통 초당 60회 이상)마다 새 배열을 할당하는 대신, 버퍼
    # 하나를 미리 만들어두고 매 틱 내용만 in-place로 지운다(canvas[:] = 색)
    # — 새로 메모리를 할당·해제하는 비용이 없다. draw_cursor도 cv2 그리기
    # 함수라 같은 배열을 그대로 수정하고 반환해 추가 할당이 없다
    overlay_canvas = np.empty((overlay_h_px, overlay_w_px, 3), dtype=np.uint8)

    if not args.no_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception:   # noqa: 방어적 — 이 OpenCV 빌드가 TOPMOST를 지원 안 해도 계속 진행
            logger.warning("Could not set window topmost - continuing anyway")
        # 창이 실제로 화면에 만들어질 때까지 한 번 그려야 FindWindowW로 찾을 수 있다
        # (enable_transparent_overlay 독스트링 참고) — 실제 내용은 루프 첫 프레임에서 덮어씀
        overlay_canvas[:] = TRANSPARENT_KEY_COLOR
        cv2.imshow(WINDOW_NAME, overlay_canvas)
        cv2.waitKey(1)
        enable_transparent_overlay(WINDOW_NAME, TRANSPARENT_KEY_COLOR, OVERLAY_ALPHA)

    refresh_hz = _get_refresh_rate_hz()
    # 화면 주사율과 상한(RENDER_FPS_CAP) 중 낮은 쪽에 맞춘다 — 상수 설명 참고
    render_hz = min(refresh_hz, RENDER_FPS_CAP)
    frame_interval_sec = 1.0 / render_hz
    logger.warning(
        "헤드트래커 시작 — 코끝으로 커서 이동, 입 벌리기/1.5초 응시로 클릭. "
        "커서는 화면 %s 절반만 사용. 화면 주사율 %dHz · 렌더 %dHz 고정. "
        "콘솔에 p+Enter로 실제 마우스 제어 켜기/끄기, quit+Enter로 종료. "
        "창이 있으면 q/ESC로도 종료.",
        "하단" if CURSOR_Y_ANCHOR_BOTTOM else "상단", refresh_hz, render_hz,
    )
    logger.info("Startup %.1fs (interpreter + imports + models)", time.monotonic() - PROCESS_START_SEC)

    # 렌더 스레드(=메인 스레드)가 표시하는 현재 위치 — 매 틱 target_state를
    # 향해 RENDER_LERP_ALPHA만큼씩 다가간다(위 상수 설명 참고)
    displayed_x_ratio, displayed_y_ratio = CENTER_X_RATIO, _cursor_y_to_screen(CENTER_Y_RATIO)
    # 정지 히스테리시스용 anchor·상태(RESUME_GAP_RATIO 상수 설명 참고) — anchor는
    # "마지막으로 멈춰 선 지점", is_moving은 지금 그 anchor를 벗어나 목표를
    # 쫓아가는 중인지
    anchor_x_ratio, anchor_y_ratio = displayed_x_ratio, displayed_y_ratio
    is_moving = False
    resume_pending_since_sec = None   # RESUME_CONFIRM_SEC 문턱 초과가 언제부터 이어지고 있는지 — 안 넘는 중이면 None
    arrived_since_sec = None   # SETTLE_CONFIRM_SEC 확인용 — 목표를 따라잡은 상태(gap<=데드존)가 언제부터 이어지고 있는지
    native_cursor_hidden = False   # 추적이 처음 시작되는 틱에 딱 한 번만 cursor_hider.hide() 호출 — 위 cursor_hider 생성부 주석 참고
    camera_window_open = False   # "cam on"/"cam off" 전환 시 창을 새로 만들지/닫을지 추적
    was_settling = False   # 정지 유예 구간에 막 들어선 첫 틱만 감지 — 렌더 스레드 전용 지역 변수
                           # (tracking_edge_state는 추론 스레드 전용이라 여긴 따로 둔다)
    watchdog_fault = None        # 지금 감지된 고장 종류(None이면 정상)
    watchdog_logged_sec = 0.0    # 같은 고장을 마지막으로 기록한 시각
    last_drawn_signature = None   # 오버레이 다시 그리기 생략용(아래 cv2.imshow 직전 주석 참고)
    # 지난 틱에 실제로 뭔가 그린 영역 — 다음 틱에 여기만 지운다
    # (_blank_rect / _cursor_rect 설명 참고)
    last_drawn_rect = None
    render_tick_count = 0   # 진단 로그 주기 계산용

    prev_tick_sec = time.monotonic()   # 직전 틱 시각 — 아래 tick_dt_sec 계산용
    try:
        while not state["should_quit"]:
            tick_start_sec = time.monotonic()
            # 이번 틱이 실제로 얼마 만에 왔는지 — alpha 보정에 쓴다
            # (_dt_adjusted_alpha 설명 참고). 창 전환 등으로 크게 튄 값은
            # 커서가 한 번에 확 튀지 않도록 상한을 둔다
            tick_dt_sec = min(tick_start_sec - prev_tick_sec, frame_interval_sec * 3.0)
            prev_tick_sec = tick_start_sec

            with target_lock:
                target_x_ratio = target_state["x_ratio"]
                target_y_ratio = target_state["y_ratio"]
                is_tracking = target_state["is_tracking"]
                recenter_progress = target_state["recenter_progress"]
                became_tracking_sec = target_state["became_tracking_sec"]

            # ★감시 판정 (WATCHDOG_STALL_SEC 상수 설명 참고). 추론 스레드가
            # 살아 있는지, 카메라가 새 그림을 주고 있는지를 따로 본다.
            # health_state의 값이 0.0이면 아직 시작 전이라 판정하지 않는다 —
            # 모델 로딩·카메라 예열에 몇 초가 걸리는데 그걸 고장으로 세면 안 된다
            previous_fault = watchdog_fault
            watchdog_fault = None
            if not inference_thread.is_alive():
                watchdog_fault = WATCHDOG_LABEL_INTERNAL
            elif (health_state["loop_sec"]
                  and tick_start_sec - health_state["loop_sec"] > WATCHDOG_STALL_SEC):
                watchdog_fault = WATCHDOG_LABEL_INTERNAL
            elif (health_state["frame_sec"]
                  and tick_start_sec - health_state["frame_sec"] > WATCHDOG_STALL_SEC):
                watchdog_fault = WATCHDOG_LABEL_CAMERA
            if watchdog_fault:
                # 고장이 바뀌었으면 즉시, 같은 고장이면 간격을 두고 기록한다 —
                # 매 틱 남기면 초당 30줄이 쌓인다
                if (watchdog_fault != previous_fault
                        or tick_start_sec - watchdog_logged_sec >= WATCHDOG_LOG_SEC):
                    logger.error("감시: %s", watchdog_fault)
                    watchdog_logged_sec = tick_start_sec
            elif previous_fault:
                logger.warning("감시: 정상으로 돌아왔습니다 (직전 상태: %s)", previous_fault)

            # 정지 유예(SETTLE_DELAY_SEC 상수 설명 참고) — 캘리브레이션 직후
            # 이 시간 동안은 목표를 아주 느리게만 따라간다(아래 is_settling 분기 참고)
            is_settling = is_tracking and (tick_start_sec - became_tracking_sec) < SETTLE_DELAY_SEC
            # 드래그 중엔 잡음을 더 넓게 무시한다 (DRAG_DEAD_ZONE_SCALE 설명 참고)
            dead_zone_ratio = RENDER_DEAD_ZONE_RATIO * (
                DRAG_DEAD_ZONE_SCALE if feedback.is_holding else 1.0)

            if is_settling:
                # ★2026-08-18 재설계(사용자 요청 — "캘리브레이션 할떄 5초 후에
                # 커서가 화면 중앙에 가게해줘"): 유예 중엔 실시간 target이 아니라
                # 고정된 화면 중앙을 향해 완만히(CATCHUP_LERP_ALPHA) 다가간다 —
                # "화면 정면을 보고 편한 자세로 있어달라"는 안내 문구대로 사용자가
                # 가만히 있으면 target 자체도 중앙 근처라 결과는 이전 방식(target을
                # 쫓아가기)과 사실상 같지만, 사용자가 유예 중 고개를 움직여도 5초
                # 시점엔 반드시 중앙에 가 있다는 걸 보장한다. 유예가 끝난 뒤엔
                # 그 시점의 실제 목표(고개 위치)로 정상 추적이 자연스럽게 이어받는데
                # (아래 "elif is_tracking:" 분기), 이때 격차가 있어도 이미 부드러운
                # 히스테리시스·완만한 합류 로직(RESUME_GAP_RATIO 등)을 그대로
                # 타므로 순간이동으로 보이지 않는다(예전에 "커서가 튀어" 문제를
                # 일으켰던 건 유예 "종료" 시점의 강제 스냅이었지, 유예 "중" 목표를
                # 뭘로 잡느냐가 아니었다 — 이제 종료 시점엔 아무것도 강제하지 않는다)
                center_x_ratio, center_y_ratio = CENTER_X_RATIO, _cursor_y_to_screen(CENTER_Y_RATIO)
                dx = center_x_ratio - displayed_x_ratio
                dy = center_y_ratio - displayed_y_ratio
                gap = math.hypot(dx, dy)
                if gap > dead_zone_ratio:
                    settle_alpha = _dt_adjusted_alpha(
                        CATCHUP_LERP_ALPHA, tick_dt_sec, frame_interval_sec)
                    displayed_x_ratio += settle_alpha * dx
                    displayed_y_ratio += settle_alpha * dy
                    if state["is_control_active"]:
                        mouse.move(displayed_x_ratio, displayed_y_ratio)
                anchor_x_ratio, anchor_y_ratio = displayed_x_ratio, displayed_y_ratio
                is_moving = False
                resume_pending_since_sec = None
                arrived_since_sec = None
            elif is_tracking:
                if was_settling:
                    # 유예 종료 — 위 분기에서 이미 매 틱 target을 느리게 계속
                    # 따라오고 있었으므로 갑자기 좁혀야 할 격차 자체가 없다
                    # (2026-08-14 재설계 — 위 is_settling 분기 주석 참고). 그냥
                    # anchor만 지금 위치로 갱신해 정지 히스테리시스를 재개한다
                    anchor_x_ratio, anchor_y_ratio = displayed_x_ratio, displayed_y_ratio
                    is_moving = False
                    resume_pending_since_sec = None
                    arrived_since_sec = None
                elif is_moving:
                    # 확정된 뒤(RESUME_GAP_RATIO/RESUME_CONFIRM_SEC 상수 설명 참고) —
                    # 목표를 계속 쫓는다. ★2026-08-18 재설계(SETTLE_CONFIRM_SEC 상수
                    # 설명 참고 — 실측 로그로 "연속으로 움직이는 중에도 뚝뚝 끊긴다"의
                    # 실제 원인을 확인) — 데드존 안까지 따라잡아도(gap<=데드존) 그
                    # 즉시 정지로 확정하지 않는다. 그 상태가 SETTLE_CONFIRM_SEC 동안
                    # 끊기지 않고 이어져야("목표가 그만큼 오래 안 움직였다") 비로소
                    # is_moving을 끄고 anchor를 새로 고정한다 — 잠깐 따라잡았다가
                    # 다음 추론 갱신에서 목표가 또 움직이면(연속 동작 중엔 흔하다)
                    # 얼어붙지 않고 그대로 계속 쫓아간다
                    dx = target_x_ratio - displayed_x_ratio
                    dy = target_y_ratio - displayed_y_ratio
                    gap = math.hypot(dx, dy)
                    if gap <= dead_zone_ratio:
                        if arrived_since_sec is None:
                            arrived_since_sec = tick_start_sec
                        elif tick_start_sec - arrived_since_sec >= SETTLE_CONFIRM_SEC:
                            is_moving = False
                            anchor_x_ratio, anchor_y_ratio = displayed_x_ratio, displayed_y_ratio
                            arrived_since_sec = None
                    else:
                        arrived_since_sec = None   # 격차가 다시 벌어졌다 — "도착" 타이머 리셋
                        # 격차가 크면(짧은 미검출 후 재검출 등) 완만한 합류 — LARGE_GAP_RATIO 참고
                        alpha = CATCHUP_LERP_ALPHA if gap > LARGE_GAP_RATIO else RENDER_LERP_ALPHA
                        if feedback.is_holding:
                            alpha *= DRAG_LERP_ALPHA_SCALE   # 드래그 중엔 더 부드럽게
                        alpha = _dt_adjusted_alpha(alpha, tick_dt_sec, frame_interval_sec)
                        displayed_x_ratio += alpha * dx
                        displayed_y_ratio += alpha * dy
                        if state["is_control_active"]:
                            mouse.move(displayed_x_ratio, displayed_y_ratio)
                else:
                    # 정지 히스테리시스 + 지속시간 확인 — 멈춰 있는 동안(is_moving=False)은
                    # 목표가 anchor(멈춰 선 지점)에서 문턱만큼 벌어진 상태가
                    # RESUME_CONFIRM_SEC 동안 끊기지 않고 이어져야 "진짜 움직였다"고
                    # 확정한다 — 문턱만 넘고 금방 다시 떨어지면(잡음) 리셋된다.
                    # ★2026-08-18 추가(사용자 실기 보고 "1cm 정도 이동하려고 하면
                    # 뚝뚝 끊긴다" — RESUME_CONFIRM_SEC 상수 설명 참고): 확정 전
                    # "대기" 중에도 화면을 완전히 얼려두지 않고 CATCHUP_LERP_ALPHA로
                    # 미리 완만하게 따라가기 시작한다 — anchor 자체(재확정 판단 기준)는
                    # 안 건드리므로, 정지 중 잡음만으로 이 분기가 열리는 일은 여전히
                    # 없다(그 경우는 애초에 RESUME_GAP_RATIO를 못 넘어 아래 if를 안 탄다) —
                    # 열린 뒤에만, 확정되기 전까지 완만히 움직인다
                    resume_dx = target_x_ratio - anchor_x_ratio
                    resume_dy = target_y_ratio - anchor_y_ratio
                    if math.hypot(resume_dx, resume_dy) > RESUME_GAP_RATIO:
                        if resume_pending_since_sec is None:
                            resume_pending_since_sec = tick_start_sec
                        dx = target_x_ratio - displayed_x_ratio
                        dy = target_y_ratio - displayed_y_ratio
                        gap = math.hypot(dx, dy)
                        if gap > dead_zone_ratio:
                            pend_alpha = _dt_adjusted_alpha(
                                CATCHUP_LERP_ALPHA, tick_dt_sec, frame_interval_sec)
                            displayed_x_ratio += pend_alpha * dx
                            displayed_y_ratio += pend_alpha * dy
                            if state["is_control_active"]:
                                mouse.move(displayed_x_ratio, displayed_y_ratio)
                        if tick_start_sec - resume_pending_since_sec >= RESUME_CONFIRM_SEC:
                            is_moving = True
                            resume_pending_since_sec = None
                    else:
                        resume_pending_since_sec = None
            was_settling = is_settling

            if not args.no_window:
                # ★2026-08-18 다시 그리기 생략 최적화(사용자 실기 보고 "프레임
                # 잘나오게 뚝뚝 끊기지 않게 최적화해줘" — 실측으로 렌더 틱이
                # 화면 주사율 예산을 거의 매번 넘기고 있는 걸 확인함): 이 오버레이는
                # 화면 전체 해상도 캔버스라, 커서가 정지해 있어(이 세션 내내
                # 공들인 정지 히스테리시스 덕분에 대부분의 틱이 여기 해당한다)
                # 화면에 그려질 내용이 지난 틱과 완전히 같을 때도 매번
                # cv2.imshow로 그 큰 이미지를 다시 올리고 있었다 — 그게 매 틱의
                # 실제 비용 대부분이었다(내용을 안 바꾸는 fill·그리기 자체는
                # 저렴하다). 화면에 보이는 값(커서 위치·정지 유예 여부·추적
                # 여부·재정렬 진행률)이 지난 틱과 똑같으면 다시 그리지도, 다시
                # imshow하지도 않는다 — 창엔 이미 그 내용이 그대로 떠 있으니
                # 다시 그릴 이유가 없다. cv2.waitKey는 이 분기와 무관하게 아래에서
                # 매 틱 그대로 호출된다(키 입력·메시지 펌프는 이 창에 새로 그릴
                # 게 없어도 계속 필요하다)
                # ★피드백 상태를 signature에 넣는다 — 빼먹으면 커서가 제자리일 때
                # 다시 그리기를 건너뛰어 깜빡임이 화면에 안 나타난다
                feedback_key = feedback.state_key(tick_start_sec)
                draw_signature = (
                    round(displayed_x_ratio, 4), round(displayed_y_ratio, 4),
                    is_settling, is_tracking, round(recenter_progress, 2), feedback_key,
                    watchdog_fault,   # 빼먹으면 커서가 제자리일 때 경고가 화면에 안 뜬다
                )
                if draw_signature != last_drawn_signature:
                    # 2026-08-13 사용자 최종 정리 — "마우스 포인터 말고 차라리 커서를
                    # 보이게 해줘": 네이티브 화살표는 cursor_hider.hide()로 화면에서
                    # 숨기고(클릭·이동 자체는 실제 포인터 위치에서 그대로 동작),
                    # 화면에 보이는 건 이 오버레이가 그리는 초록 점 하나뿐이다.
                    # 버퍼 재사용(위 overlay_canvas 주석 참고) — 매 틱 새로 할당하지 않는다
                    # 화면 전체가 아니라 지난 틱에 그린 영역만 되돌린다 —
                    # 실측 6.9~13.1ms -> 0.02ms (_blank_rect 설명 참고)
                    _blank_rect(overlay_canvas, last_drawn_rect)
                    drawn_rect = None
                    if is_tracking:
                        if not native_cursor_hidden:
                            # 오버레이 점을 처음 그리는 이 틱에만 네이티브 화살표를
                            # 숨긴다(위 cursor_hider 생성부 주석 참고) — 그 전까지는
                            # 원래 화살표가 그대로 보여 커서가 아예 안 보이는 구간이 없다
                            cursor_hider.hide()
                            native_cursor_hidden = True
                        # 정지 유예 중엔 재정렬(응시) 진행 링을 끈다(2026-08-13 사용자
                        # 요청 — "그때는 캘리브레이션 초 안나오게 해줘"): head_tracker
                        # 자체의 응시 재정렬 감지는 유예 중에도 내부적으로 계속 돌고
                        # 있어(아래 이벤트 억제와 별개 — 그쪽은 select/home/calibration
                        # "발화"만 막고, 이 진행 링은 발화 전 진행 상태를 그때그때
                        # 보여주는 표시라 따로 꺼야 한다) 화면엔 방금 만든 정지 유예
                        # 문구 하나만 보이게 한다
                        draw_cursor(overlay_canvas, displayed_x_ratio, displayed_y_ratio,
                                   0.0 if is_settling else recenter_progress,
                                   color=feedback.color(tick_start_sec),
                                   filled=feedback.is_holding)
                        drawn_rect = _union_rect(drawn_rect, _cursor_rect(
                            displayed_x_ratio, displayed_y_ratio, overlay_w_px, overlay_h_px))
                        if is_settling:
                            # 정지 유예 안내 문구 — 화면 중앙 고정 두 줄(2026-08-18 사용자
                            # 요청 — "이 문구는 무조건 화면 가운데에 뜨게 해주고"): 예전엔
                            # displayed_x/y_ratio(커서 위치)를 따라 그려서, 커서가 유예 중
                            # 중앙으로 다가가는 동안 문구도 같이 움직였다 — 커서는 이제
                            # 중앙을 향해 실제로 움직이는 중이니(위 is_settling 분기 참고),
                            # 안내 문구까지 같이 움직이면 "어디를 보라는 건지" 기준점 자체가
                            # 흔들린다. 문구는 커서가 최종적으로 도착할 자리(화면 중앙)에
                            # 고정해 흔들리지 않는 기준점으로 삼는다. cv2.putText는 한글을
                            # 못 그려("?"로 깨짐, put_korean_text 독스트링 참고)
                            # put_korean_text로 그린다. 줄마다 폭이 달라
                            # _korean_text_width_px로 실측해 가운데 정렬한다
                            line1_font_px, line2_font_px = 30, 22
                            line1_w_px = _korean_text_width_px(SETTLING_LABEL_LINE1, line1_font_px)
                            line2_w_px = _korean_text_width_px(SETTLING_LABEL_LINE2, line2_font_px)
                            center_x_px = int(CENTER_X_RATIO * overlay_w_px)
                            line1_y_px = (int(_cursor_y_to_screen(CENTER_Y_RATIO) * overlay_h_px)
                                         + CURSOR_RADIUS_PX + 20)
                            line2_y_px = line1_y_px + line1_font_px + 10
                            put_korean_text(overlay_canvas, SETTLING_LABEL_LINE1,
                                            (int(center_x_px - line1_w_px / 2), line1_y_px),
                                            line1_font_px, CURSOR_COLOR)
                            put_korean_text(overlay_canvas, SETTLING_LABEL_LINE2,
                                            (int(center_x_px - line2_w_px / 2), line2_y_px),
                                            line2_font_px, CURSOR_COLOR)
                            # 두 줄 문구가 칠한 범위도 지울 목록에 넣는다 — 안 넣으면 유예가
                            # 끝난 뒤에도 글자가 화면에 남는다
                            text_w_px = max(line1_w_px, line2_w_px)
                            drawn_rect = _union_rect(drawn_rect, _clip_rect((
                                center_x_px - text_w_px // 2 - 4, line1_y_px - line1_font_px - 4,
                                center_x_px + text_w_px // 2 + 4, line2_y_px + 8),
                                overlay_w_px, overlay_h_px))
                    # ★고장 알림 (WATCHDOG_STALL_SEC 상수 설명 참고) — 추적
                    # 여부와 무관하게 그린다. 오히려 추적이 멈춘 상태가
                    # 바로 이걸 봐야 하는 상황이다. 화면 한가운데보다 조금
                    # 아래에 둬서 커서·안내 문구와 겹치지 않게 한다
                    if watchdog_fault:
                        fault_font_px = 26
                        fault_w_px = _korean_text_width_px(watchdog_fault, fault_font_px)
                        fault_x_px = int(CENTER_X_RATIO * overlay_w_px - fault_w_px / 2)
                        fault_y_px = int(_cursor_y_to_screen(1.0) * overlay_h_px) - 40
                        put_korean_text(overlay_canvas, watchdog_fault,
                                        (fault_x_px, fault_y_px), fault_font_px,
                                        WATCHDOG_LABEL_COLOR)
                        drawn_rect = _union_rect(drawn_rect, _clip_rect((
                            fault_x_px - 6, fault_y_px - fault_font_px - 6,
                            fault_x_px + fault_w_px + 6, fault_y_px + 8),
                            overlay_w_px, overlay_h_px))
                    cv2.imshow(WINDOW_NAME, overlay_canvas)
                    last_drawn_signature = draw_signature
                    last_drawn_rect = drawn_rect

            # 카메라 미리보기(2026-08-13 사용자 요청 — "cam on 하면 카메라 보이게") —
            # main.py의 런타임 cam on/off 관례와 같은 이름. 오버레이(--no-window)
            # 여부와 무관하게 독립적으로 켜고 끌 수 있다
            if state["show_camera"]:
                with camera_preview_lock:
                    preview_frame = camera_preview_state["frame"]
                if preview_frame is not None:
                    cv2.imshow(CAMERA_WINDOW_NAME, preview_frame)
                    camera_window_open = True
            elif camera_window_open:
                cv2.destroyWindow(CAMERA_WINDOW_NAME)
                camera_window_open = False

            # 창이 하나라도 있으면 메시지 펌프 겸 키 입력 확인 — 이 호출 하나가
            # 이 프로세스의 모든 cv2 창(오버레이·카메라 미리보기)를 함께 갱신한다.
            # 매 틱 부른다 — pollKey는 기다리지 않아 공짜다(_pump_window_events 참고)
            render_tick_count += 1
            if not args.no_window or camera_window_open:
                if _pump_window_events() in (ord("q"), 27):
                    break

            # 화면 주사율에 맞춰 페이싱 — 이번 틱에 실제로 걸린 시간을 빼고 남는
            # 만큼만 잔다(처리 시간이 길어져도 과도하게 밀리지 않게)
            elapsed_sec = time.monotonic() - tick_start_sec
            sleep_sec = frame_interval_sec - elapsed_sec
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    finally:
        state["should_quit"] = True
        # 무엇보다 먼저 — 드래그 중이었다면 왼쪽 버튼을 뗀다. 아래 정리가
        # 오래 걸리거나 실패해도 이것만은 이미 끝나 있어야 한다
        mouse.release_if_pressed()
        console.stop()   # 남은 출력 내보내기 (막혀 있으면 기다리지 않는다)
        inference_thread.join(timeout=2.0)   # face_estimator.close() 전에 추론 스레드가 먼저 끝나야 안전
        camera.stop()
        # ★join이 실패했으면(추론 스레드가 아직 살아 있으면) 모델을 해제하지
        # 않는다 — infer() 안에서 쓰고 있는 네이티브 객체를 그 밑에서 없애는
        # 꼴이라 종료 순간에 프로세스가 죽는다. 어차피 프로세스가 끝나는
        # 참이므로, 해제를 건너뛰어 생기는 손해는 없다
        if inference_thread.is_alive():
            logger.warning("추론 스레드가 제때 끝나지 않아 모델 해제를 건너뜁니다")
        else:
            face_estimator.close()
        cv2.destroyAllWindows()
        cursor_hider.restore()   # 종료 시 반드시 네이티브 화살표를 되돌려놓는다
    return 0


if __name__ == "__main__":
    sys.exit(main())
