"""헤드트래커 단독 실행기(코-눈 혼합점 기준) — 실제 OS 마우스 포인터를 제어한다
(2026-08-27 신설, 사용자 요청: "몸이 움직이면 커서도 움직이는 문제 해결하자" —
eyebrow.py를 그대로 둔 채 같은 문제를 근본적으로 고친 새 버전).

★eyebrow.py와 이 파일의 차이 — 딱 하나, 커서 기준점을 재는 방식.

eyebrow.py의 "미간"은 사실 LMK_LEFT_EYE_OUTER·LMK_RIGHT_EYE_OUTER(양쪽 눈
바깥쪽 끝) **두 점의 평균**이다. 그런데 head.py가 쓰는 "얼굴 기준 좌표계"
(face_local — head_tracker.py _CursorMapper 참고)는 원점을 정확히 그 두 점의
평균으로 잡는다. 즉 eyebrow.py는 "기준점 - 원점"이 항상 정확히 0인 계산을
하고 있었던 셈이라(같은 두 점으로 같은 식을 두 번 계산), face_local을 켤 수
없었다 — 켜면 커서가 아예 안 움직인다. 그래서 eyebrow.py는 예전 방식(화면
절대 픽셀 기준)을 그대로 썼고, 그 대가로 **몸이 움직이면 커서도 그만큼
밀리는 문제**가 남아 있었다(FACE_LOCAL_MAPPING 주석 참고, eyebrow.py에
그대로 남겨 뒀다).

이 파일은 **새 랜드마크 인덱스를 추측하지 않는다** — 대신 이미 검증된 두
종류의 점(코끝 LMK_NOSE_TIP, 눈 바깥쪽 끝 LMK_LEFT/RIGHT_EYE_OUTER)만으로
원점과 겹치지 않는 기준점을 만든다:

    기준점 = 눈_중점 + FOREHEAD_NOSE_BLEND_RATIO x (코끝 - 눈_중점)

비율(FOREHEAD_NOSE_BLEND_RATIO)만큼 코끝 쪽으로 당긴 점이다. 비율이 0보다
크기만 하면 기준점이 원점(눈_중점)과 달라져 face_local이 정상 작동한다 —
몸 전체가 움직여도 커서는 안 움직이고, 고개를 돌리거나 끄덕이는 움직임만
반영된다(head.py가 몸 밀림을 없앤 것과 같은 원리 — 2026-08-20). 비율을
작게 잡을수록(기본 0.25) 위치는 눈 사이 근처에 남아 있으면서 코의 회전
신호를 그 비율만큼만 빌려 오고, 입 벌림에 의한 변형 영향도 같은 비율만큼만
받는다 — eyebrow.py가 애초에 코 대신 미간을 쓰려던 이유(입 벌림 영향을
줄이기)를 상당 부분 유지하면서 face_local을 켤 수 있다.

이 조합이 실기로 회전 감도가 부족하면 FOREHEAD_NOSE_BLEND_RATIO를 올릴 것
(1.0이면 head.py의 코 기준과 동일해진다). 감도값(SENSITIVITY_X/Y_OVERRIDE)은
eyebrow.py 값을 그대로 시작점으로 가져왔을 뿐이라 실기로 다시 맞춰야 할
가능성이 높다.

그 밖의 구조(오버레이 표시, 실제 마우스 이동·클릭, 상단 제한 범위, 화면
주사율 보간, 네이티브 커서 숨김, 입 벌림 변형 보정 등)는 head.py·eyebrow.py와
동일하다 — 그 파일들 독스트링에 각 설계 이유가 자세히 있다(상단 제한 비율은
이 파일은 실제 발급기 화면에 맞춰 CURSOR_Y_SPAN/ANCHOR로 따로 조정했다).

사용법 (프로젝트 루트에서):
    py forehead.py [--device N] [--config path] [--no-window]
종료: 오버레이 창에서 q/ESC, 또는 콘솔에 quit + Enter
실제 마우스 제어 토글: 콘솔에 p + Enter
"""
import argparse
import atexit
import ctypes
import json
import math
import os
import signal
import subprocess
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
# head.py와 동일 이유로 원본(카메라 해상도 기준) draw_cursor·draw_head_debug_panel을
# 그대로 쓴다. 오버레이용 큰 draw_cursor와 이름이 겹쳐 카메라 쪽만 접두어로 구분
from src.utils.visualize import draw_cursor as draw_cursor_camera
from src.utils.visualize import draw_head_debug_panel

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

# ★2026-08-28 신설 — 실시간 조절 UI(scripts/tuning_ui.py, "tune"+Enter로
# 켠다) 대응. eyebrow.py의 동명 상수·함수와 완전히 동일한 이유·구조다(그
# 파일 TUNING_FILE_PATH 설명 참고) — 파일 이름만 트래커별로 다르다.
TUNING_FILE_PATH = os.path.join(ROOT_DIR, "forehead_tuning.json")
TUNING_POLL_INTERVAL_SEC = 0.5


def _load_tuning_overrides(path):
    """튜닝 파일이 있으면 dict, 없거나 손상됐으면 None — eyebrow.py 동명
    함수와 동일(그 파일 설명 참고)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        logger.warning("튜닝 파일이 손상돼 무시합니다: %s", path)
        return None

# 같은 오류가 계속될 때 다시 기록하기까지의 간격(초) — head.py와 동일
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

WINDOW_NAME = "forehead tracker overlay"
CAMERA_WINDOW_NAME = "forehead tracker camera"

# 키 확인 — head.py와 동일 이유·동일 처방(그 파일 _pump_window_events 설명 참고).
# ★ 2026-08-20: cv2.waitKey(1)이 항상 15~16ms를 먹던 것을 cv2.pollKey()(실측
# 0.00ms)로 교체했다. 공짜가 됐으므로 호출을 건너뛰던 우회도 없쥜다
_HAS_POLL_KEY = hasattr(cv2, "pollKey")


def _pump_window_events():
    """창 메시지 처리 + 키 확인 — head.py 동명 함수와 동일."""
    return (cv2.pollKey() if _HAS_POLL_KEY else cv2.waitKey(1)) & 0xFF

# 세로 커서 도달 범위 — 화면 세로 중 커서가 실제로 쓰는 구간.
#
# 이력: head.py 원래 기본값은 0.5(상단 절반)였다. 2026-08-27 실기에서 발급기
# 화면(9:16) 하단의 "화면이 높으면 이곳을 터치해주세요" 배너·자막까지 커서가
# 안 닿아 0.7로 넓혔다.
#
# ★2026-08-27 재요청 — "위쪽만 되는 걸 반전해서 아래쪽만 되게".
# 실제 키오스크는 사용자가 화면 아래쪽을 주로 조작한다(휠체어 사용자·키가
# 작은 사용자에게도 아래쪽이 닿기 쉽다). 그래서 같은 폭(0.7)을 유지하되
# **화면 아래쪽에 붙여** 매핑한다.
#
#   CURSOR_Y_ANCHOR_BOTTOM = True   -> 화면 아래 70% 구간 (세로 30%~100%)
#   CURSOR_Y_ANCHOR_BOTTOM = False  -> 화면 위  70% 구간 (세로  0%~ 70%)
#
# 폭(SPAN)과 어느 쪽에 붙일지(ANCHOR)를 나눠 뒀으니, 전체 화면을 쓰려면
# SPAN을 1.0으로 올리면 된다(그러면 ANCHOR는 의미가 없어진다).
#
# ★2026-08-28 재요청 — 9:16 키오스크 모니터에서 커서 영역이 화면 가운데까지
# 올라와 있는 느낌이라, 0.7 -> 0.5(하단 절반만, head.py·eyebrow.py와 동일
# 폭)로 다시 줄였다. 발급기(FINOK) 화면의 하단 자막까지 커서가 닿아야 하는
# 요구(위 이력 참고)와는 상충할 수 있다 — 그 화면에서 다시 문제가 되면 그때
# 다시 넓힐 것.
CURSOR_Y_SPAN = 0.5
CURSOR_Y_ANCHOR_BOTTOM = True
# 커서 세로 구간의 시작점(화면 비율) — 아래쪽에 붙이면 그만큼 밀어서 시작한다
CURSOR_Y_OFFSET = (1.0 - CURSOR_Y_SPAN) if CURSOR_Y_ANCHOR_BOTTOM else 0.0


def _cursor_y_to_screen(cursor_y_ratio):
    """트래커 내부 세로 좌표(0~1) -> 실제 화면 세로 비율(0~1).

    커서 계산은 언제나 0~1 전체 범위를 쓰고, 화면의 어느 구간에 앉힐지는
    여기서만 정한다 — 위/아래 전환이 이 함수 하나로 끝나게 하려는 것이다
    (예전엔 `* TOP_HALF_Y_SCALE`이 코드 여러 곳에 흩어져 있어 한 곳만
    빠뜨려도 커서와 안내 문구가 서로 다른 자리에 놓였다).
    """
    return CURSOR_Y_OFFSET + cursor_y_ratio * CURSOR_Y_SPAN

# 렌더 스레드 보간 계수 — head.py와 동일 이력·동일 값(그 파일 상수 설명 참고:
# 0.35→0.6→0.45→0.38, 반응성↔정지 시 떨림 절충 + 정지 히스테리시스 도입 후
# 재하향)
RENDER_LERP_ALPHA = 0.38

# 렌더 상한 — ★2026-08-27 30 -> 60 상향 (사용자 요청 "프레임도 60fps처럼").
#
# 원래 30으로 잡았던 근거 ①이 **이 구조에서는 성립하지 않는다**는 걸 확인해서
# 올렸다. 당시 근거와 지금 판단:
#
#  ① (옛) "얼굴 인식이 카메라 속도(30fps)로 들어오니 더 자주 그려도 새 정보가
#     없다" -> **틀렸다.** 렌더 루프는 같은 좌표를 다시 그리는 게 아니라 목표를
#     향해 보간(lerp)한다. 추론과 추론 사이에도 매 틱 새로운 중간 위치가
#     만들어지므로, 두 배로 그리면 그 궤적이 두 배로 촘촘하게 표시된다 —
#     같은 움직임이 눈에 띄게 매끄러워진다.
#  ② (옛) "60Hz 화면에서 30은 정확히 절반이라 고르게 걸린다" -> 여전히 맞지만,
#     60은 1:1이라 더 고르다. 어중간한 값(45 등)만 피하면 된다.
#  ③ (옛) "틱당 예산이 두 배" -> 맞지만 실측해 보니 여유가 압도적이다:
#         커서 영역 지우기 0.001ms + 커서 그리기 0.172ms = 약 0.17ms
#         60fps 예산 16.7ms 대비 100배 가까운 여유
#
# ★함께 바꿔야 하는 것: 아래 LERP_REFERENCE_FPS. 보간 계수를 렌더 주기에
# 그대로 묶어 두면 60fps에서 커서가 두 배 빨리 목표에 붙어 손맛이 통째로
# 바뀐다 — 그래서 기준을 렌더 속도와 분리했다.
# 화면 주사율이 이 값보다 낮으면 그쪽을 따른다(min으로 고른다)
RENDER_FPS_CAP = 60

# 보간 계수(RENDER_LERP_ALPHA 등)가 "몇 초당" 적용되는지의 기준 — 렌더 속도와
# **분리해서** 고정한다 (2026-08-27 신설).
#
# 왜 필요한가: alpha는 "한 틱에 남은 거리의 몇 %를 따라갈지"다. 기준을 렌더
# 주기로 두면 렌더가 30 -> 60이 될 때 같은 alpha가 두 배 자주 적용돼 커서가
# 두 배 빨리 목표에 붙는다 — 그동안 실기로 맞춰 온 감각(0.35→0.6→0.45→0.38
# 이력)이 통째로 어긋난다.
#
# 30으로 고정해 두면 렌더를 몇으로 올리든 **커서가 시간축에서 그리는 궤적은
# 완전히 동일**하고, 그 궤적을 더 촘촘히 표시할 뿐이다. 즉 "손맛은 그대로,
# 화면만 매끄럽게"가 성립한다.
LERP_REFERENCE_FPS = 30

# 위 alpha 값들은 "한 틱에 남은 거리의 몇 %를 따라갈지"라, 틱이 늦어지면
# 그만큼 커서가 덜 따라와 느려 보인다. 실제 걸린 시간에 맞춰 alpha를 보정해
# 틱 간격이 흔들려도 움직임이 일정하게 보이도록 한다 — 기준은 위 상한(30fps)
# 이라, 정상 속도로 돌 때는 보정이 없어 기존 손맛이 그대로 유지된다
def _dt_adjusted_alpha(alpha, dt_sec, ref_dt_sec):
    if dt_sec <= 0.0 or abs(dt_sec - ref_dt_sec) < 1e-6:
        return alpha
    return max(0.0, min(1.0, 1.0 - (1.0 - alpha) ** (dt_sec / ref_dt_sec)))

# 커서 민감도·입벌림 문턱.
# ★2026-08-27 1차 — 가로(X)는 FOREHEAD_NOSE_BLEND_RATIO를 0.5→0.3으로 낮춘
# 것(그 상수 설명 참고 — U자 포물선·입벌림 커서 밀림 대응)을 보충하려고
# 1.124 → 2.0으로 올렸다(세로는 eyebrow.py 값 그대로 유지).
# ★2026-08-27 2차 실기 확인 후 — 저조도 인식은 해결됐고, 발급기 실제 화면에서
# 테스트하며 "수직 수평 민감도 살짝 높여달라"는 추가 요청으로 둘 다
# 한 단계 더 올렸다(가로 2.0→2.2, 세로 1.80→2.0). 실기로 다시 맞출 것 —
# 여전히 둔하면 더 올리고, 떨리거나 과하면 낮출 것
# ★2026-08-27 3차 — "화면 밑 끝까지 커서가 안 닿는다" + "수평·수직 다 올려줘"
#
# 세로는 단순히 "좀 둔하다"가 아니라 **기하학적으로 못 닿는** 상태였다.
# 커서가 화면 끝에 닿으려면 (dy x 감도)가 클램프값(config max_offset_ratio
# = 0.5)에 도달해야 하는데, 혼합점은 코를 30%만 섞으므로 세로로 튀어나온
# 양이 아주 작다 — 고개를 45도 숙여도 신호가 0.5에 한참 못 미친다.
# 어림 계산으로 지금 감도(2.0)에서는 화면의 약 84%가 한계였고, 실제 보고
# ("밑 끝까지 안 닿아")와 맞는다. 닿게 하려면 세로 감도를 약 1.9배 올려야 한다.
#
#   2.0 -> 3.8  (세로: 화면 바닥까지 닿게)
#   2.2 -> 2.8  (가로: 요청대로 상향)
#
# 감도를 올리면 잡음도 같이 증폭되지만, 1€ 필터를 제대로 튜닝해 둔 덕에
# (0.25/1.5 — 정지 떨림 -24%) 예전 감도 때보다 여유가 있다.
# 여전히 안 닿으면 세로를 더 올리고, 너무 예민하면 낮출 것
SENSITIVITY_X_OVERRIDE = 2.8
SENSITIVITY_Y_OVERRIDE = 3.8
# 2026-08-18: "입벌리는걸 인식을 잘 못하는거 같아 다방면으로" 대응으로 더
# 낮춤 — head.py와 동일 이유(그 파일 상수 설명 참고)
MOUTH_OPEN_MARGIN_OVERRIDE = 0.12    # config 기본 0.35
MOUTH_CLOSE_MARGIN_OVERRIDE = 0.05   # config 기본 0.15

# 입 제스처 확장(단일 클릭 즉시/꾹 누르기) — head.py와 동일 이유·동일 값(그
# 파일 상수 설명 참고 — 더블클릭은 이제 별도 판정 없이 Windows의 기본
# 더블클릭 인식에 맡긴다)
# 2026-08-20: 드래그가 잘 안 걸린다는 실기 보고로 2.0→1.2 하향 + 닫힘 확인
# 시간 신설 — head.py와 동일 이유·동일 값(그 파일 상수 설명 참고)
MOUTH_HOLD_SEC = 0.7   # 2026-08-24 1.2 -> 0.7 — head.py와 동일 이유·동일 값
                       # (그 파일 상수 설명에 "얼마까지 줄여도 클릭이 안 새는가" 실측표)
DOUBLE_CLICK_WINDOW_SEC = 0.9
# 두 클릭이 이 안에 들어오면 더블클릭으로 본다 (_Win32Mouse.click 독스트링 참고).
# 윈도우 기본 한계(500ms)보다 넉넉한 이유는 실측 때문이다 — 입을 0.25초씩
# 여닫는 평범한 속도가 이미 485ms라, 500ms에 맞추면 조금만 느려도 실패한다.
# 0.9초는 "빠르게 두 번"으로 읽히는 상한이면서, 서로 다른 두 번의 클릭이
# 잘못 묶이지 않을 만큼 짧다.

MOUTH_CLOSE_CONFIRM_SEC = 0.08

# 드래그 중에는 "닫혔다"를 더 엄격하게 본다 — head.py와 동일 이유·동일 값
# (그 파일 상수 설명 참고: 고개를 움직이면 턱 벌림 추정이 출렁여 드래그가 끊긴다)
MOUTH_HOLD_RELEASE_CONFIRM_SEC = 0.20
MOUTH_HOLD_RELEASE_MARGIN = 0.03

# 커서 EMA 반응 속도 — head.py와 동일 이력·동일 이유(그 파일 상수 설명 참고,
# "고개를 돌리면 커서가 뒤늦게 따라오는 현상" 대응으로 0.18→0.26 추가 상향).
# 여전히 head.py(0.32)보다 낮다 — 이 파일만 수직 감도(2.0)가 더 높아 원시
# 잡음도 그만큼 더 크게 증폭되기 때문
POINTER_SMOOTHING_ALPHA_OVERRIDE = 0.26           # config 기본 0.2
POINTER_DISTANCE_SMOOTHING_ALPHA_OVERRIDE = 0.08   # config 기본 0.15

# 데드존·정지 히스테리시스·정지 유예·재확정 지속시간 — head.py와 동일 이유
# (그 파일 상수 설명 참고, "끊기는 느낌" 대응으로 진폭 문턱을 다시 낮추고
# RESUME_CONFIRM_SEC로 지속시간을 확인하는 방식으로 재설계). 진폭 문턱은
# head.py보다 조금 더 크게 잡았다 — 이 파일만 수직 감도(2.0)가 더 높아
# 같은 랜드마크 잡음도 더 크게 증폭되기 때문
# ★2026-08-20 재하향 — head.py와 동일 이유(그 파일 상수 설명 참고, "작은
# 아이콘도 정확하게 클릭"): 이 두 값이 곧 커서의 최소 이동 단위라 굵으면
# 작은 아이콘을 겨냥할 수 없다. 떨림은 이제 지속시간 확인이 걸러낸다
RENDER_DEAD_ZONE_RATIO = 0.004
RESUME_GAP_RATIO = 0.007
RESUME_CONFIRM_SEC = 0.07   # head.py와 동일 이유(그 파일 상수 설명 참고, "1cm 정도 이동하려고 하면 뚝뚝 끊긴다" 대응)
SETTLE_CONFIRM_SEC = 0.12   # head.py와 동일 이유(그 파일 상수 설명 참고 — 연속 동작 중 반복되는 "뚝뚝 끊김"의 실측 원인 대응)
SETTLE_DELAY_SEC = 8.0   # head.py와 동일 이유(그 파일 주석 참고, 2026-08-18: 5.0→7.0→8.0)
SETTLING_LABEL_LINE1 = "커서 재정렬 중입니다"
# 정지 유예 안내 둘째 줄 — head.py와 동일 이유·동일 값(그 파일 주석 참고)
# ★2026-08-20 "입은 다물어 주세요" 추가 — 실측으로 확인된 실패 모드 대응.
# 이 유예 구간은 커서 중심만 잡는 게 아니라 **입 벌림의 평상시 기준선**도 함께
# 잡는다(head_tracker의 jaw_baseline). 그래서 이 8초 동안 입이 벌어져 있으면
# 기준선이 그만큼 높게 잡히고, 이후 "평소보다 이만큼 더 벌리면 클릭"의 기준이
# 통째로 어긋나 클릭이 안 먹거나 제멋대로 나간다.
# 실측: 유예 중에 입을 여닫는 상태로 캘리브레이션하면 더블클릭이 주기당 2번이
# 아니라 1번만 나갔고, 입을 다물고 캘리브레이션하니 정상으로 돌아왔다.
# 말하거나 하품하면서 서 있으면 실사용에서도 그대로 생기는 문제라 안내에 넣는다.
SETTLING_LABEL_LINE2 = "커서 중앙을 봐주시고 입은 다문 채 편한자세로 있어주세요"

# 큰 격차 완만한 합류 — head.py와 동일 이유·동일 값(그 파일 상수 설명 참고,
# "고개를 돌리면 커서가 뒤늦게 따라오는 현상" 대응으로 0.05→0.18 상향 —
# 평범한 고개 돌리기만으로도 이 낮은 문턱을 넘어 정작 빠른 반응이 필요할 때
# 가장 느린 alpha를 쓰고 있었다)
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

# 얼굴 검출 신뢰도 오버라이드 — head.py의 "정확도" 되돌림 이력과 동일 이유로
# 기본은 끔(config.yaml 값 그대로 사용)
ACCURACY_CONF = None

# 추론 입력 배율 — head.py와 동일 이유·동일 값(그 파일 상수 설명 참고).
# ★2026-08-20 실측으로 0.75 -> 1.0 되돌림: 줄이려고 내는 resize 비용(1.24ms)이
# 줄여서 버는 추론 시간(1.10ms)보다 커서 이득이 0이었다. 되돌리면 랜드마크
# 정밀도와 저조도 성능이 함께 좋아진다
INFER_SCALE_RATIO_OVERRIDE = 1.0

# 커서를 "얼굴이 향한 방향"으로만 움직이는 방식 — ★이 파일은 켠다(핵심 변경).
#
# eyebrow.py는 이 방식을 못 썼다 — 그 파일의 기준점(미간)이 두 눈 중점 그
# 자체로 계산돼(LMK_LEFT/RIGHT_EYE_OUTER 평균) 원점과 완전히 같은 값이라,
# "벗어난 양"이 항상 정확히 0이 되어 커서가 아예 안 움직이기 때문이다
# (eyebrow.py의 이 상수 설명에 자세히 남겨 뒀다).
#
# 이 파일은 원점과 다른 실제 랜드마크(LMK_GLABELLA)를 쓰므로 이 문제가 없다.
# 켜면 head.py와 똑같은 효과를 얻는다 — 몸 전체가 움직여도 커서는 그대로고,
# 고개를 돌리거나 끄덕이는 움직임만 반영된다(2026-08-20 head.py 도입 이유와
# 동일. head_tracker.py _CursorMapper 독스트링 참고).
FACE_LOCAL_MAPPING = True
FACE_LOCAL_GAIN = 2.0   # head.py와 동일 시작값 — 미간은 코보다 덜 튀어나와
                        # 실제 신호 크기가 다를 수 있다. 실기로 테스트 후 조정할 것

# 1€ 필터(OneEuroFilter, head_tracker.py 독스트링 참고) — ★이 파일만 켠다.
# head.py·eyebrow.py는 예전 그대로 단순 EMA를 쓴다(설정 키 기본값이 False라
# 동작 변화 없음). forehead.py는 새로 만든 파일이라 이 검증된 논문 기법을
# 바로 적용해 본다 — 정지 시 떨림은 억제하면서 빠른 고개 움직임엔 지연 없이
# 반응하는 절충을 EMA 하나보다 더 잘 해낸다.
# ★2026-08-27 재튜닝 — 논문의 표준 절차대로 실측해서 값을 정했다.
#
# 처음엔 논문 권장 시작값(min_cutoff=1.0, beta=0.0)을 그대로 뒀는데, **beta=0은
# 속도 적응을 꺼버리는 값**이라 1€ 필터를 쓰면서 정작 핵심 기능을 안 쓰고 있었다
# (고정 저역통과 필터와 다를 게 없는 상태). 시뮬레이션으로 조합을 훑어 보니
# 지금 값이 세 지표 **전부** 열세였다:
#
#   min_cutoff  beta |  정지 떨림  느린 조준 지연  빠른 이동 지연
#   -----------------+--------------------------------------------
#      1.00     0.0  |   0.89px      30.49px        274.40px   <- 기존
#      0.25     1.5  |   0.68px      21.60px         61.41px   <- 채택
#                    |    -24%         -29%            -78%
#
# 절충이 아니라 순수 개선이다 — 떨림·정밀 조준·빠른 이동이 동시에 좋아진다.
# beta가 지연을 압도적으로 줄이면서 떨림 비용은 거의 없기 때문이다(움직일 때만
# 평활이 풀리므로). 그래서 min_cutoff는 낮춰 정지 떨림을 더 잡고, beta를 올려
# 움직임 추종을 되살리는 조합이 성립한다.
#
# 재튜닝이 필요하면 논문 절차를 그대로 따를 것:
#   1) beta=0으로 두고 정지 떨림이 잡힐 때까지 min_cutoff를 낮춘다
#   2) 그 다음 움직일 때 지연이 사라질 때까지 beta를 올린다
ONE_EURO_ENABLED = True
ONE_EURO_MIN_CUTOFF = 0.25
ONE_EURO_BETA = 1.5

# 거리 적응 평활 — 멀수록 자동으로 더 세게 평활한다(head_tracker.py
# _CursorMapper.__init__ 독스트링에 실측 근거와 원리가 정리돼 있다).
# 8/26에 이 프로젝트가 직접 잰 값이 "커서 흔들림 x 안구간거리 = 180"으로
# 정확히 1/거리 비례라, 그 증폭을 같은 비율로 되돌린다.
# 기준 60px = 키오스크 실사용 거리 — 그 거리에선 배율 1.0이라 지금 손맛 그대로다
ONE_EURO_DISTANCE_ADAPTIVE = True
ONE_EURO_REFERENCE_DIST_PX = 60.0

# 가로로 움직일 때 세로가 활처럼 휘는 것(뒤집힌 U) 보정 — 원리와 부호는
# head_tracker.py `_arc_compensation` 설명 참고.
#
# 원인: 기준점에 섞인 코가 얼굴 밖으로 튀어나온 3차원 점이라, 고개를 좌우로
# 돌리면 원근 때문에 세로까지 밀린다. 그 밀림이 가로 이동량의 **제곱에 비례**해
# 궤적이 2차 곡선이 된다. 그래서 같은 형태를 빼면 원리적으로 상쇄된다.
#
# 8/27에 코 혼합 비율을 0.5 -> 0.3으로 낮춰 곡률을 40% 줄였지만, 그건 **완화**지
# 제거가 아니다 — 남은 30%만큼은 그대로 휜다. 이 보정이 나머지를 없앤다.
#
# ★2026-08-27 4차 — 1.0을 넣었다가 다시 0.0으로 되돌렸다.
#
# 3차에서 "여전히 ∩로 휜다"는 보고로 어림값 1.0을 넣었는데, 실기 결과는
# "좌우로 머리를 이동하면 무조건 커서가 아래로 수직 하강"이었다 — **과보정**이다.
# 원인은 두 가지가 겹친 것: ① 이 계수 자체가 실측이 아니라 어림값이었고,
# ② 같은 커밋에서 SENSITIVITY_X를 2.2→2.8로 올려 offset_x가 더 쉽게 클램프
# (0.5)에 닿게 됐다 — 보정식이 offset_x²이라 클램프 근처에서 밀림량이 급격히
# 커진다(계산: offset_x=0.5, 계수=1.0일 때 화면 높이의 약 17.5%를 강제로 내림).
# 이 밀림은 좌우 어느 쪽으로 돌리든 항상 **양수**(제곱이라 부호가 사라짐)라
# "좌우 이동하면 다 아래로 간다"는 보고와 정확히 일치한다.
#
# ★2026-08-28 5차 — measure_arc.py로 실측해 넣었다(더 이상 어림짐작 아님).
#
# 표본 800개, 가로 훑은 폭 0.99(충분): 맞춘 곡선 세로 = -0.0425 + 0.0397·가로
# + 1.1029·가로² (R² = 0.825). 화면 좌우 끝에서 세로로 +27.6% 밀림(뒤집힌
# U자와 일치). 곡률 +1.1029를 상쇄하려면 부호를 뒤집은 -1.1029를 더한다.
#
# R²=0.825는 완벽하진 않다 — 실제 사람 손으로 좌우'만' 완벽하게 왕복하긴
# 어려워 위아래 미세한 흔들림이 섞여 있을 수 있다는 뜻. 적용 후
# measure_arc.py를 다시 돌려 남은 휨이 줄었는지 재확인할 것.
#
# ★2026-08-28 6차 — -1.1029가 과보정이었다. 재측정 두 번(R²=0.19, 0.22로
# 낮아 그때는 보류) 모두 잔여 곡률이 -0.27 ~ -0.36으로 **반대 방향**으로
# 나왔는데, 이걸 "잡음이라 무시"했더니 실사용에서 "양쪽 끝에서 커서가 위로
# 뜬다"는 정확히 그 증상으로 나타났다 — 신뢰도 낮은 측정이라고 실제 신호가
# 아닌 건 아니었다.
#
# 세 번 측정 전부(보정 0일 때 +1.1029, 보정 -1.1029일 때 -0.27과 -0.358)를
# "결과곡률 = 원래곡률 + 보정값" 관계로 역산해 평균 낸 값 — 새로 추측한
# 값이 아니라 이미 잰 데이터 3개를 조합한 값이다: (1.1029 + 0.8329 + 0.7449)
# / 3 = 0.8936. 상쇄하려면 부호를 뒤집는다.
ARC_COMPENSATION = -0.8936

# 자동 재정렬(가만히 있으면 스스로 중심을 다시 잡는 기능) — ★2026-08-20 끔.
# head.py와 동일 이유(그 파일 상수 설명 참고) — 사용자 결정 "몇 초 가만히
# 있으면 캘리브레이션 되는 거 없애자". 시작할 때의 캘리브레이션은 그대로 둔다
RECENTER_DWELL_ENABLED_OVERRIDE = False

# --overlay 크로마키 색(BGR 마젠타) — head.py와 동일값
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

# 커서 크기 — head.py와 동일(그 파일 상수 설명 참고: 화면 해상도 캔버스 기준으로 확대)
CURSOR_RADIUS_PX = 28
CURSOR_MARKER_SIZE_PX = 22
CURSOR_THICKNESS_PX = 3

# 추적 기준점 시각화 — head.py와 동일 이유(그 파일 상수 설명 참고: "cam on"으로
# 카메라를 켰을 때 지금 어느 점을 잡고 있는지 실측 위치로 보여준다)
# ★2026-08-27 라벨 정정 — "이마"였는데 실제로는 이마(눈보다 위)가 아니라
# 눈_중점에서 코끝 쪽(아래)으로 당긴 점이라 눈과 코 사이(콧대 부근)에 찍힌다.
# 실측 위치와 라벨이 어긋나 있던 걸 사용자가 지적해 정확한 이름으로 바꿨다
TRACKING_POINT_LABEL = "혼합점"
TRACKING_POINT_COLOR = (0, 255, 255)   # 노랑(BGR) — 커서(초록)와 뚜렷이 구분

# ★기준 재료 전부를 화면에 보여준다(2026-08-27 사용자 요청 — "cam on 하면
# 어디가 기준인지 특징점 잡는거 다 보이게 해줘"). 커서 기준점(위 노란 점)이
# 눈 두 점 + 코끝, 세 원재료로 어떻게 만들어지는지 그 재료들까지 화면에
# 그대로 노출한다 — 오작동을 의심할 때 "어느 랜드마크가 흔들렸는지"를
# 눈으로 바로 짚을 수 있어야 하기 때문이다.
EYE_CORNER_COLOR = (255, 180, 0)     # 하늘색(BGR) — 눈 바깥쪽 끝 두 점
ORIGIN_POINT_COLOR = (255, 0, 255)   # 마젠타(BGR) — 원점(두 눈 중점, face_local 기준)
NOSE_RAW_COLOR = (0, 0, 255)         # 빨강(BGR) — 코끝 원본 랜드마크(혼합 전)
BLEND_LINE_COLOR = (180, 180, 180)   # 회색(BGR) — 원점→코끝 축(노란 점이 이 선 위 어디에 있는지)


# 기준점을 코끝 쪽으로 당기는 비율 — 위 모듈 독스트링 참고. 0이면 눈_중점과
# 완전히 같아져(eyebrow.py와 동일한 문제로) face_local 신호가 0이 되고, 1.0이면
# head.py와 완전히 같은 코끝 기준이 된다. 그 사이 값으로 "미간 근처에 남으면서
# 신호는 살아있는" 절충점을 잡는다.
# ★2026-08-27 1차 실기 확인 — 0.25: 몸이 움직여도 커서는 안 움직임(성공) /
# 회전이 둔함. 0.5로 올림: 회전은 좋아졌으나 새 부작용 2건 보고:
#   · 좌우로 움직이면 커서가 U자 뒤집힌 포물선을 그림 — 코끝은 얼굴 밖으로
#     튀어나온 3D 점이라, 고개를 돌리면 원근(패럴랙스) 때문에 순수 수평이
#     아니라 살짝 휘어진 궤적으로 움직인다. 코 비율이 높을수록 이 휘어짐도 커진다
#   · 입을 벌려 클릭하려 하면 커서가 내려감 — 입 벌림에 의한 코끝 랜드마크
#     밀림(변형 보정 구조는 있지만, 코 비율이 높으면 그만큼 영향도 커진다)
# 두 부작용 모두 "코 성분이 너무 많이 섞였다"가 공통 원인이라 비율을 다시
# 낮췄다 — 회전 반응성은 비율 대신 아래 SENSITIVITY_X_OVERRIDE를 직접 올려
# 별도로 보충한다(사용자 요청 — "좌우 수평 민감도 살짝 올려줘")
FOREHEAD_NOSE_BLEND_RATIO = 0.3

# 코 기준점을 코끝 한 점이 아니라 코 아래쪽 여러 점의 평균으로 잡을지 —
# head.py와 동일 상수·동일 근거(그 파일 NOSE_CLUSTER_AVERAGING 설명 참고).
# ★2026-08-27 사용자 실기 보고 — "미간(eyebrow.py)은 불 꺼도 인식됐는데 이번
# 것(코 성분이 섞인 혼합점)은 어두우면 바로 인식이 안 된다". 원인은 이미
# 8/20에 실측된 것과 같다 — 코끝(랜드마크 1개)은 매끈하고 무늬 없는 면이라
# 어두우면 모델이 위치를 잘 못 잡고 흔들리는데, 미간(두 눈 바깥쪽 끝의 평균)은
# 눈꼬리 명암이 뚜렷해 어두워도 잘 잡힌다(그때 실측: 어두울 때 코 1.83배
# 흔들림 vs 미간 1.15배). 이 파일은 코끝을 30%(FOREHEAD_NOSE_BLEND_RATIO)
# 섞어 쓰므로 그 불안정성을 그만큼 물려받는다 — 코 아래쪽 여러 점의 평균을
# 쓰면(서로 무관한 흔들림이 점 개수의 제곱근만큼 상쇄) 코 성분 자체의
# 저조도 안정성이 head.py 수준으로 올라간다. 기본은 켬 — 끄면 예전처럼
# 코끝 한 점만 쓴다
NOSE_CLUSTER_AVERAGING = True

# 얼굴 변형 전환 순간 좌표 보류(2026-08-14 사용자 실기 보고 — 처음엔
# "눈감을때마다 커서가 움직이는데", 이어서 "입벌리면 커서가 움직여";
# 2026-08-18 head.py에서 재설계된 것과 같은 이유(그 파일 _stable_nose_point
# 독스트링 참고) — 이 기준점도 코끝을 일부 섞으므로 입을 크게 벌리면(턱이
# 크게 움직이며 얼굴 형태가 변해 코끝 랜드마크 추정치가 밀림) 그 영향이
# FOREHEAD_NOSE_BLEND_RATIO만큼 섞여 들어온다. 코끝 100%인 head.py보다는
# 작겠지만 0은 아니므로, 안전하게 같은 보정 구조를 그대로 적용한다 — 실기로
# 보정량(k)이 거의 0으로 나오면 이 구조는 사실상 비활성인 채로 남아도 무해하다.
# 입이 열린 뒤 랜드마크가 가라앉기를 기다리는 시간 — head.py와 동일 값
NOSE_DEFORM_SETTLE_SEC = 0.12
# 보정량 상한(px). 이보다 크면 얼굴이 일그러진 게 아니라 고개를 돌린 것이다
NOSE_DEFORM_MAX_PX = 14.0
EYE_CLOSE_SUPPRESS_THRESHOLD = 0.35
MOUTH_OPEN_SUPPRESS_THRESHOLD = 0.08
_forehead_deform_state = {
    "neutral": None,        # 입 다물었을 때의 얼굴 기준 혼합점 위치
    "neutral_jaw": 0.0,
    "k": None,              # 턱 벌림 1당 혼합점이 밀리는 양 (x, y)
    "was_open": False,
    "opened_at_sec": 0.0,
    "pre_open": None,
}


def _stable_forehead_point(face):
    """커서 기준점(미간 쪽으로 당긴 코-눈 혼합점) 픽셀 좌표 — HeadTracker의
    cursor_point_fn으로 쓰인다.

    head.py의 _stable_nose_point와 거의 같은 구조다(그 파일 독스트링에 설계
    이유가 자세히 있다) — 다른 점은 순수 코끝이 아니라 "눈_중점에서 코끝
    쪽으로 FOREHEAD_NOSE_BLEND_RATIO만큼 당긴 점"을 쓴다는 것뿐이다(위 모듈
    독스트링 참고 — 새 랜드마크 인덱스를 추측하지 않고 이미 검증된 두 점만
    쓰기 위한 선택). eyebrow.py의 _glabella_point와 달리 **좌표를 화면 절대
    픽셀이 아니라 두 눈 중점 기준(local)으로 다룬다** — 이 점은 원점(두 눈
    중점)과 다른 위치라 상대 좌표가 의미를 가지고(eyebrow.py는 기준점=원점이라
    이 방식을 못 썼다), 화면 좌표로 다루면 그 사이 몸이 움직였을 때 보정량이
    통째로 어긋나는 문제(2026-08-20 head.py에서 겪음)도 함께 피한다.
    """
    now_sec = time.monotonic()
    eye_left_px = face.landmark_px(LMK_LEFT_EYE_OUTER)
    eye_right_px = face.landmark_px(LMK_RIGHT_EYE_OUTER)
    eye_mid_x = (eye_left_px[0] + eye_right_px[0]) / 2.0
    eye_mid_y = (eye_left_px[1] + eye_right_px[1]) / 2.0
    nose_px = (face.landmarks_mean_px(LMK_NOSE_CLUSTER) if NOSE_CLUSTER_AVERAGING
               else face.landmark_px(LMK_NOSE_TIP))
    ratio = FOREHEAD_NOSE_BLEND_RATIO
    point = (eye_mid_x + ratio * (nose_px[0] - eye_mid_x),
             eye_mid_y + ratio * (nose_px[1] - eye_mid_y))
    local = (point[0] - eye_mid_x, point[1] - eye_mid_y)
    jaw = face.blendshape("jawOpen")
    is_open = jaw >= MOUTH_OPEN_SUPPRESS_THRESHOLD
    st = _forehead_deform_state

    if not is_open:
        # 입을 다물고 있다 — 평상시 값을 천천히 기억한다(빨리 따라가면 고개를
        # 돌리는 중의 값까지 평상시로 삼아 기준이 흔들린다)
        st["neutral"] = local if st["neutral"] is None else (
            st["neutral"][0] + 0.1 * (local[0] - st["neutral"][0]),
            st["neutral"][1] + 0.1 * (local[1] - st["neutral"][1]))
        st["neutral_jaw"] = st["neutral_jaw"] + 0.1 * (jaw - st["neutral_jaw"])
        st["pre_open"] = st["neutral"]
        st["was_open"] = False
        return point

    if not st["was_open"]:
        st["was_open"] = True
        st["opened_at_sec"] = now_sec
        st["k"] = None

    jaw_delta = jaw - st["neutral_jaw"]
    if st["k"] is None and st["pre_open"] is not None and jaw_delta > 1e-3:
        if now_sec - st["opened_at_sec"] >= NOSE_DEFORM_SETTLE_SEC:
            shift = (local[0] - st["pre_open"][0], local[1] - st["pre_open"][1])
            limit = NOSE_DEFORM_MAX_PX
            shift = (max(-limit, min(limit, shift[0])), max(-limit, min(limit, shift[1])))
            st["k"] = (shift[0] / jaw_delta, shift[1] / jaw_delta)

    if st["k"] is None:
        # 계수를 아직 못 잰 구간은 변형된 좌표를 내보내면 안 된다 — 그 짧은
        # 사이에만 커서가 크게 튄다(head.py 실측: 화면 세로의 16%)
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


# 한글 텍스트 렌더링 — head.py의 동명 함수와 동일(그 파일 독스트링에 cv2.putText가
# 한글을 못 그리는 이유가 자세히 있다)
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


_KOREAN_FONT_CACHE = {}


def _get_korean_font(size_px):
    font = _KOREAN_FONT_CACHE.get(size_px)
    if font is None:
        font = ImageFont.truetype("malgun.ttf", size_px)
        _KOREAN_FONT_CACHE[size_px] = font
    return font


def put_korean_text(canvas_bgr, text, org, font_size_px, color_bgr):
    """ROI만 PIL 왕복 변환하는 최적화 — head.py와 동일 이유(그 파일 독스트링 참고)."""
    x_px, y_px = int(org[0]), int(org[1])
    canvas_h_px, canvas_w_px = canvas_bgr.shape[:2]
    font = _get_korean_font(font_size_px)
    text_w_px = int(font.getlength(text))
    pad_px = font_size_px // 2
    x0, y0 = max(0, x_px - pad_px), max(0, y_px - pad_px)
    x1 = min(canvas_w_px, x_px + text_w_px + pad_px)
    y1 = min(canvas_h_px, y_px + font_size_px + pad_px * 2)
    if x1 <= x0 or y1 <= y0:
        return canvas_bgr
    b, g, r = color_bgr
    roi_bgr = canvas_bgr[y0:y1, x0:x1]
    pil_image = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil_image).text((x_px - x0, y_px - y0), text, font=font, fill=(r, g, b))
    canvas_bgr[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return canvas_bgr


def _korean_text_width_px(text, font_size_px):
    """두 줄짜리 정지 유예 문구 가운데 정렬용 — head.py와 동일 이유(그 파일
    주석 참고)."""
    return _get_korean_font(font_size_px).getlength(text)


logger = get_logger("scripts")


def enable_transparent_overlay(window_name, key_color_bgr, whole_window_alpha):
    """--overlay 전용 — cv2 창을 반투명 + 클릭 통과 창으로 바꾼다.

    head.py의 동명 함수와 완전히 동일한 구현(그 파일 독스트링에 클릭 통과가
    왜 필요한지 자세히 있다) — 두 실행기가 독립 배포되는 별개 exe라 함수만
    그대로 복사해 둔다.
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
        ws_ex_transparent = 0x00000020   # 클릭 통과 — head.py의 동명 함수 독스트링 참고
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
    """콘솔 빠른 편집 해제 — main.py/head.py와 동일 로직·동일 이유
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
    """실제 OS 마우스 포인터 제어 — ctypes만 사용(새 의존 추가 없음). Windows 전용.

    화면 해상도는 GetSystemMetrics로 실측한다 — 이 PC/키오스크 모니터가
    실제로 몇 x 몇인지 가정하지 않는다(포터블).
    """

    SM_CXSCREEN = 0
    SM_CYSCREEN = 1
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("forehead.py의 실제 마우스 제어는 Windows 전용입니다")
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
        """좌클릭 누르기만(업 없음) — "꾹 누르기"(드래그) 전용. head.py와
        동일 이유(그 파일 동명 메서드 독스트링 참고)."""
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.is_pressed = True

    def release(self):
        """press()로 시작한 좌클릭을 뗀다."""
        self._user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.is_pressed = False

    def release_if_pressed(self):
        """눌러둔 게 있으면 무조건 뗀다 — head.py 동명 메서드 독스트링에
        왜 "제어가 켜져 있는지"를 보면 안 되는지가 실측과 함께 적혀 있다."""
        if not self.is_pressed:
            return
        try:
            self.release()
        except Exception:   # noqa: 방어적 — 종료 중이라 더 할 수 있는 게 없다
            self.is_pressed = False


class _SystemCursorHider:
    """네이티브 OS 마우스 포인터(화살표)를 화면에서 숨긴다 — head.py의 동명
    클래스와 완전히 동일(그 클래스 독스트링에 ShowCursor 대신 SetSystemCursor를
    쓰는 이유가 자세히 있다)."""

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
        logger.info("표준 입력이 없어 콘솔 명령(quit/p/cam/tune)을 받지 않습니다")
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
        elif command == "tune":
            _launch_tuning_ui()


_TUNING_UI_SCRIPT = os.path.join(ROOT_DIR, "scripts", "tuning_ui.py")


def _launch_tuning_ui():
    """"tune"+Enter로 실시간 조절 창을 띄운다 — eyebrow.py 동명 함수와 완전히
    동일(그 파일 설명 참고, 별도 프로세스로 띄우는 이유 포함)."""
    if not os.path.exists(_TUNING_UI_SCRIPT):
        logger.warning("실시간 조절 UI를 찾을 수 없습니다: %s", _TUNING_UI_SCRIPT)
        return
    try:
        subprocess.Popen([sys.executable, _TUNING_UI_SCRIPT,
                          "--tracker", "forehead", "--file", TUNING_FILE_PATH])
        logger.info("실시간 조절 창을 열었습니다")
    except OSError:
        logger.exception("실시간 조절 창을 여는 데 실패했습니다")


class _TuningReloader:
    """TUNING_FILE_PATH 변경을 주기적으로 확인해 head_tracker에 반영한다 —
    eyebrow.py 동명 클래스와 완전히 동일(그 파일 설명 참고)."""

    def __init__(self, path, poll_interval_sec, clock=time.monotonic):
        self._path = path
        self._poll_interval_sec = poll_interval_sec
        self._clock = clock
        # -inf로 시작 — eyebrow.py와 동일 이유(그 파일 설명 참고, 0.0으로
        # 두면 첫 확인을 건너뛸 수 있다)
        self._last_checked_sec = float("-inf")
        self._last_mtime = None

    def maybe_reload(self, head_tracker):
        now_sec = self._clock()
        if now_sec - self._last_checked_sec < self._poll_interval_sec:
            return
        self._last_checked_sec = now_sec
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return
        if mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        tuning = _load_tuning_overrides(self._path)
        if tuning is None:
            return
        head_tracker.set_pointer_tuning(
            sensitivity_x=tuning.get("sensitivity_x"),
            sensitivity_y=tuning.get("sensitivity_y"),
            arc_compensation=tuning.get("arc_compensation"))


def _get_refresh_rate_hz(default_hz=60):
    """주 모니터의 실제 화면 주사율(Hz) — head.py의 동명 함수와 완전히 동일
    (그 함수 독스트링에 GetDeviceCaps를 쓰는 이유가 자세히 있다)."""
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


# ★저조도 대응 — CLAHE (2026-08-27 사용자 요청 — "조도 문제를 완화할만한
# 기술이나 논문 없나 찾아서 적용해줘").
#
# Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization."
# Graphics Gems IV, Academic Press, 1994.
#
# 일반 히스토그램 평활화는 영상 전체의 명암 분포를 한 번에 넓게 펴는데,
# 이러면 원래 어둡던 영역의 잡음까지 같이 크게 증폭된다 — 코끝처럼 무늬
# 없는 매끈한 면은 정확히 이 잡음 증폭에 가장 취약한 부위다(위
# NOSE_CLUSTER_AVERAGING 설명 참고). CLAHE는 화면을 작은 타일로 나눠
# **타일마다 국소적으로**, 그리고 **대비 상한(clip limit)을 두고** 명암을
# 편다 — 어두운 부위의 국소 대비만 끌어올리고 잡음 폭주는 상한으로 막는다.
# 얼굴 인식·저조도 영상 전처리에 널리 쓰이는 표준 기법이고, OpenCV에
# 이미 내장돼 있어 새 의존성 없이 CPU에서 실시간으로 돌릴 수 있다.
#
# **밝을 때는 아예 건드리지 않는다** — 평균 밝기가 문턱 이상이면 원본
# 프레임을 그대로 돌려준다. 이미 잘 보이는 영상에 대비를 더 올리면 얻는
# 것 없이 부작용(과다 대비, 색 얼룩)만 생길 위험이 있어, "필요할 때만
# 켜지는" 안전한 방식으로 설계했다. head.py·eyebrow.py는 이 함수를 아예
# 호출하지 않으므로 영향이 없다.
#
# Y(밝기) 채널에만 적용하고 색(Cr·Cb)은 그대로 둔다 — 색상 왜곡 없이
# 명암 대비만 올리는 표준 관례(컬러 CLAHE를 BGR 채널마다 따로 걸면 색이
# 어긋난다).
LOW_LIGHT_ENHANCEMENT_ENABLED = True
LOW_LIGHT_MEAN_LUMA_THRESHOLD = 80.0   # 0~255. 이 평균 밝기 미만일 때만 보정 적용
LOW_LIGHT_CLAHE_CLIP_LIMIT = 3.0       # 클수록 대비를 더 올리되 잡음도 더 증폭
LOW_LIGHT_CLAHE_TILE_GRID = (8, 8)     # 타일 개수(가로, 세로) — 국소 처리 단위
# 밝기 사전 판정용 축소 썸네일 한 변 길이(px) — 아래 성능 설명 참고
LOW_LIGHT_PROBE_SIZE_PX = 32

_low_light_clahe = cv2.createCLAHE(
    clipLimit=LOW_LIGHT_CLAHE_CLIP_LIMIT, tileGridSize=LOW_LIGHT_CLAHE_TILE_GRID)


def _enhance_low_light(frame_bgr):
    """어두운 프레임만 CLAHE로 국소 대비를 올려 돌려준다 — 위 상수 설명 참고.
    밝은 프레임은 그대로 통과시킨다(원본 객체를 그대로 반환, 복사 없음).

    ★성능 — 밝기 판정을 실제 해상도(1280x720)에서 바로 하면 색공간 변환
    자체가 실측 약 5ms 들어(카메라 fps 대비 비싸다), 키오스크는 대부분
    시간을 이미 밝은 상태로 돌아가는데 그 다수 프레임에서마다 이 비용을
    낸다. 32x32로 확 줄인 썸네일의 평균 밝기로 먼저 거르면(부정확해도
    "밝다/어둡다" 이분 판정에는 충분하다) 이 사전판정이 실측 0.02ms — 밝은
    프레임에서는 사실상 공짜가 되고, 실제로 어두운 프레임에서만(드묾) 전체
    처리 비용(~8ms)을 낸다. 8/20 waitKey 최적화와 같은 원리 — "대부분의
    경우엔 안 낸다"로 평균 비용을 낮춘다.
    """
    if not LOW_LIGHT_ENHANCEMENT_ENABLED:
        return frame_bgr
    probe = cv2.resize(frame_bgr, (LOW_LIGHT_PROBE_SIZE_PX, LOW_LIGHT_PROBE_SIZE_PX),
                        interpolation=cv2.INTER_NEAREST)
    if float(probe.mean()) >= LOW_LIGHT_MEAN_LUMA_THRESHOLD:
        return frame_bgr   # 이미 충분히 밝다 — 원본 그대로(부작용 방지 + 비용 없음)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    y_equalized = _low_light_clahe.apply(y_channel)
    enhanced = cv2.merge((y_equalized, cr_channel, cb_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)


def main():
    parser = argparse.ArgumentParser(
        description="헤드트래커(코-눈 혼합점 기준) 단독 실행 — 실제 OS 마우스 포인터를 옮기고 클릭한다")
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
    logger.warning("%s", version.describe("forehead.py"))
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
        print(version.describe("forehead.py"))
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

    # 실시간 조절 UI가 저장해 둔 값이 있으면 그걸로 시작한다 — eyebrow.py와
    # 동일 이유(그 파일 TUNING_FILE_PATH 설명 참고)
    tuning = _load_tuning_overrides(TUNING_FILE_PATH) or {}
    sensitivity_x = tuning.get("sensitivity_x", SENSITIVITY_X_OVERRIDE)
    sensitivity_y = tuning.get("sensitivity_y", SENSITIVITY_Y_OVERRIDE)
    arc_compensation = tuning.get("arc_compensation", ARC_COMPENSATION)

    config["head_tracker"]["pointer"]["sensitivity_x"] = sensitivity_x
    config["head_tracker"]["pointer"]["sensitivity_y"] = sensitivity_y
    config["head_tracker"]["pointer"]["smoothing_alpha"] = POINTER_SMOOTHING_ALPHA_OVERRIDE
    config["head_tracker"]["pointer"]["distance_smoothing_alpha"] = POINTER_DISTANCE_SMOOTHING_ALPHA_OVERRIDE
    config["head_tracker"]["mouth_click"]["open_margin"] = MOUTH_OPEN_MARGIN_OVERRIDE
    config["head_tracker"]["mouth_click"]["close_margin"] = MOUTH_CLOSE_MARGIN_OVERRIDE
    config["head_tracker"]["recenter_dwell"]["enabled"] = RECENTER_DWELL_ENABLED_OVERRIDE
    config["head_tracker"]["pointer"]["face_local"] = FACE_LOCAL_MAPPING
    config["head_tracker"]["pointer"]["face_local_gain"] = FACE_LOCAL_GAIN
    config["head_tracker"]["pointer"]["one_euro_enabled"] = ONE_EURO_ENABLED
    config["head_tracker"]["pointer"]["one_euro_min_cutoff"] = ONE_EURO_MIN_CUTOFF
    config["head_tracker"]["pointer"]["one_euro_beta"] = ONE_EURO_BETA
    config["head_tracker"]["pointer"]["one_euro_distance_adaptive"] = ONE_EURO_DISTANCE_ADAPTIVE
    config["head_tracker"]["pointer"]["one_euro_reference_dist_px"] = ONE_EURO_REFERENCE_DIST_PX
    config["head_tracker"]["pointer"]["arc_compensation"] = arc_compensation

    mouse = _Win32Mouse()
    logger.info("화면 해상도 %dx%d 감지 — 커서는 세로 %d~%dpx 구간(%s %.0f%%)만 사용",
                mouse.screen_w_px, mouse.screen_h_px,
                int(_cursor_y_to_screen(0.0) * mouse.screen_h_px),
                int(_cursor_y_to_screen(1.0) * mouse.screen_h_px),
                "하단" if CURSOR_Y_ANCHOR_BOTTOM else "상단", CURSOR_Y_SPAN * 100)
    CENTER_X_RATIO, CENTER_Y_RATIO = 0.5, 0.5
    mouse.move(CENTER_X_RATIO, _cursor_y_to_screen(CENTER_Y_RATIO))

    # hide()는 여기서 바로 부르지 않고 추적이 처음 시작되는 렌더 틱에만 부른다 —
    # head.py와 동일 이유(그 파일 cursor_hider 생성부 주석 참고, "실행시키고나면
    # 커서가 안보여" 대응)
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
    # "뒷사람" 방어 — head.py와 동일 이유(그 파일 face_anchor 생성부 주석 참고)
    face_anchor = FaceAnchor(config)
    head_tracker = HeadTracker(config, cursor_point_fn=_stable_forehead_point)   # 코끝 대신 코-눈 혼합점
    tuning_reloader = _TuningReloader(TUNING_FILE_PATH, TUNING_POLL_INTERVAL_SEC)
    camera = CameraStream(config, config_path=args.config).start()

    state = {"is_control_active": True, "should_quit": False, "show_camera": False}
    stdin_thread = threading.Thread(target=_stdin_command_loop, args=(state,), daemon=True)
    stdin_thread.start()

    # 카메라 미리보기 공유 상태 — head.py와 동일 이유(그 파일 주석 참고)
    camera_preview_lock = threading.Lock()
    camera_preview_state = {"frame": None}

    target_lock = threading.Lock()
    target_state = {
        "x_ratio": CENTER_X_RATIO, "y_ratio": _cursor_y_to_screen(CENTER_Y_RATIO),
        "is_tracking": False, "recenter_progress": 0.0, "became_tracking_sec": 0.0,
    }
    # is_tracking의 False->True 전이 감지용 — head.py와 동일 이유(그 파일 주석 참고)
    tracking_edge_state = {"was_tracking": False}
    # 짧은 순간의 미검출로 인한 전체 리셋(캘리브레이션 소실)을 봐준다 — head.py와
    # 동일한 이유·동일한 값(그 파일 주석 참고)
    DROPOUT_GRACE_SEC = 0.3
    last_face_state = {"face": None, "sec": 0.0}
    frame_seq_state = {"seq": -1}   # 마지막으로 추론한 카메라 프레임 번호 — head.py와 동일 이유

    # 감시용 건강 신호 — head.py와 동일(그 파일 주석에 락이 왜 필요 없는지 설명)
    health_state = {"loop_sec": 0.0, "frame_sec": 0.0}

    # 입 제스처 상태 — head.py와 동일 이유(그 파일 주석 참고)
    mouth_gesture_state = {
        "is_open": False,
        "open_since_sec": None,
        "is_holding": False,
        "close_since_sec": None,   # MOUTH_CLOSE_CONFIRM_SEC 확인용 — head.py와 동일 이유
    }

    feedback = CursorFeedback()   # 클릭·드래그를 커서 색으로 알린다

    def _reset_recenter_timer():
        """클릭·드래그가 일어나면 자동 재정렬 대기 시간을 처음부터 다시 센다 —
        head.py와 동일 이유(그 파일 동명 함수 독스트링 참고). ★2026-08-20:
        reset_event_gates()(모든 판정 초기화)에서 reset_recenter_dwell()(재정렬
        시계만)로 대상을 좁혔다 — 전자는 응시 클릭이 재정렬 시계를 계속
        되돌리는 무한 루프를 만들어 재정렬이 영영 안 됐다(head.py 동명 함수
        독스트링에 실측 근거)."""
        head_tracker.reset_recenter_dwell()

    def _release_mouth_hold_if_stuck():
        """추적을 잃으면 입 제스처 상태를 되돌린다 — head.py와 동일 이유(그
        파일 동명 함수 독스트링 참고)."""
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
        feedback.set_holding(False)

    def _update_mouth_gesture(now_sec):
        """단일 클릭 즉시/꾹 누르기 판정 — head.py와 동일 이유·동일 로직(그
        파일 동명 함수 독스트링 참고 — 더블클릭은 별도 판정 없이 Windows의
        기본 더블클릭 인식에 맡긴다)."""
        jaw_open = head_tracker.debug.get("jaw_open")
        jaw_base = head_tracker.debug.get("jaw_base")
        if jaw_base is None:
            return

        if not mouth_gesture_state["is_open"]:
            if jaw_open >= jaw_base + MOUTH_OPEN_MARGIN_OVERRIDE:
                mouth_gesture_state["is_open"] = True
                mouth_gesture_state["open_since_sec"] = now_sec
                mouth_gesture_state["close_since_sec"] = None
        else:
            held_sec = now_sec - mouth_gesture_state["open_since_sec"]
            # 드래그 중이면 더 깊이·더 오래 닫아야 놓아준다 — head.py와 동일
            holding = mouth_gesture_state["is_holding"]
            close_margin = (MOUTH_HOLD_RELEASE_MARGIN if holding
                            else MOUTH_CLOSE_MARGIN_OVERRIDE)
            close_confirm = (MOUTH_HOLD_RELEASE_CONFIRM_SEC if holding
                             else MOUTH_CLOSE_CONFIRM_SEC)
            is_below = jaw_open <= jaw_base + close_margin
            if is_below:
                # 닫힘 확인 — head.py와 동일 이유(턱 떨림으로 잠깐 내려간 것을
                # 닫힘으로 세면 드래그가 매번 중간에 끊긴다)
                if mouth_gesture_state["close_since_sec"] is None:
                    mouth_gesture_state["close_since_sec"] = now_sec
                elif now_sec - mouth_gesture_state["close_since_sec"] >= close_confirm:
                    mouth_gesture_state["is_open"] = False
                    mouth_gesture_state["close_since_sec"] = None
                    if mouth_gesture_state["is_holding"]:
                        mouth_gesture_state["is_holding"] = False
                        feedback.set_holding(False)
                        # 제어를 끈 상태여도 반드시 뗀다 — 안 그러면 버튼이
                        # 눌린 채 남는다(release_if_pressed 독스트링 참고)
                        mouse.release_if_pressed()
                        console.emit("hold_end")
                        logger.info("꾹 누르기 종료 (trigger=mouth, drag release)")
                    else:
                        if state["is_control_active"]:
                            mouse.click()
                        feedback.flash(now_sec)
                        console.emit("select")
                        logger.info("클릭 (trigger=mouth)")
                    _reset_recenter_timer()
            else:
                mouth_gesture_state["close_since_sec"] = None
                if not mouth_gesture_state["is_holding"] and held_sec >= MOUTH_HOLD_SEC:
                    mouth_gesture_state["is_holding"] = True
                    feedback.set_holding(True)
                    if state["is_control_active"]:
                        mouse.press()
                    console.emit("hold_start")
                    logger.info("꾹 누르기 시작 (trigger=mouth, drag press)")
                    _reset_recenter_timer()
                elif mouth_gesture_state["is_holding"]:
                    # 드래그 중에는 계속 시계를 되돌린다 — head.py와 동일 이유
                    _reset_recenter_timer()

    def _process_one_frame():
        tuning_reloader.maybe_reload(head_tracker)

        # 머리 모드는 항상 9:16 세로 크롭 — main_dpad.py 머리 모드와 동일 조건
        # capture_frame → capture_new_frame (같은 프레임 중복 추론 방지) —
        # head.py와 동일 이유(그 파일 주석 참고, "프레임 잘나오게" 대응)
        previous_seq = frame_seq_state["seq"]
        raw_frame, frame_seq_state["seq"] = camera.capture_new_frame(previous_seq)
        if frame_seq_state["seq"] != previous_seq:
            # 진짜 새 프레임일 때만 — head.py와 동일 이유(그 파일 주석 참고)
            health_state["frame_sec"] = time.monotonic()
        frame = preprocessor.preprocess_frame(raw_frame, apply_crop=True)
        # ★저조도 보정(CLAHE) — _enhance_low_light 설명 참고. 인식용 프레임에만
        # 적용하고, 화면에 보여주는 미리보기(cam on)는 아래에서 원본 frame을
        # 그대로 쓴다 — 사용자가 보는 화면 색감은 안 바뀌고 인식만 좋아진다
        infer_frame = _enhance_low_light(frame)

        faces = face_estimator.infer(infer_frame)
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
            # head.py와 동일 이유(그 파일 주석 참고)
            preview_frame = draw_cursor_camera(
                frame, result.cursor_x_ratio, result.cursor_y_ratio,
                head_tracker.debug.get("recenter_progress", 0.0),
            )
            preview_frame = draw_head_debug_panel(preview_frame, head_tracker.debug)
            # 추적 기준점 실측 위치(head.py와 동일 이유·동일 방식 — 그 파일 주석 참고)
            if user_face is not None:
                # ★원재료 세 점(눈 좌/우, 코끝)부터 그린다 — 최종 기준점(노란 점)이
                # 이 점들로 어떻게 만들어지는지 먼저 보이도록, 최종 점보다 먼저 그려
                # 뒤에 깔리게 한다(TRACKING_POINT_COLOR 위 설명 참고)
                eye_left_raw = user_face.landmark_px(LMK_LEFT_EYE_OUTER)
                eye_right_raw = user_face.landmark_px(LMK_RIGHT_EYE_OUTER)
                nose_raw = (user_face.landmarks_mean_px(LMK_NOSE_CLUSTER) if NOSE_CLUSTER_AVERAGING
                            else user_face.landmark_px(LMK_NOSE_TIP))
                origin_px = (int((eye_left_raw[0] + eye_right_raw[0]) / 2.0),
                             int((eye_left_raw[1] + eye_right_raw[1]) / 2.0))
                nose_px_i = (int(nose_raw[0]), int(nose_raw[1]))

                # 원점(두 눈 중점) -> 코끝 축 — 노란 점(최종 기준점)이 이 선 위
                # FOREHEAD_NOSE_BLEND_RATIO 지점에 있다는 걸 한눈에 보여준다
                cv2.line(preview_frame, origin_px, nose_px_i, BLEND_LINE_COLOR, 1, cv2.LINE_AA)

                for eye_raw in (eye_left_raw, eye_right_raw):
                    eye_px = (int(eye_raw[0]), int(eye_raw[1]))
                    cv2.circle(preview_frame, eye_px, 5, EYE_CORNER_COLOR, -1, cv2.LINE_AA)
                put_korean_text(preview_frame, "눈",
                                (int(eye_left_raw[0]) - 26, int(eye_left_raw[1]) - 6),
                                18, EYE_CORNER_COLOR)

                cv2.drawMarker(preview_frame, origin_px, ORIGIN_POINT_COLOR,
                               cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
                put_korean_text(preview_frame, "원점(눈 중점)",
                                (origin_px[0] + 10, origin_px[1] + 18), 18, ORIGIN_POINT_COLOR)

                cv2.circle(preview_frame, nose_px_i, 6, NOSE_RAW_COLOR, -1, cv2.LINE_AA)
                put_korean_text(preview_frame, "코끝",
                                (nose_px_i[0] + 10, nose_px_i[1] - 6), 18, NOSE_RAW_COLOR)

                # 최종 커서 기준점(위 세 점을 섞어 만든 값) — 맨 위에 그린다
                point_x_px, point_y_px = _stable_forehead_point(user_face)
                point_px = (int(point_x_px), int(point_y_px))
                cv2.circle(preview_frame, point_px, 10, TRACKING_POINT_COLOR, -1, cv2.LINE_AA)
                cv2.circle(preview_frame, point_px, 14, TRACKING_POINT_COLOR, 2, cv2.LINE_AA)
                # 한글이라 cv2.putText로는 안 그려진다 — head.py의 put_korean_text와 동일 이유
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

        if not tracking_edge_state["was_tracking"]:
            tracking_edge_state["became_tracking_sec"] = now_sec
            with target_lock:
                target_state["became_tracking_sec"] = now_sec
            tracking_edge_state["was_tracking"] = True
            # 이번 유예 사이클의 재중심 여부 — head.py와 동일 이유(그 파일 주석 참고)
            tracking_edge_state["recentered_after_settle"] = False

        screen_y_ratio = _cursor_y_to_screen(result.cursor_y_ratio)
        with target_lock:
            target_state["x_ratio"] = result.cursor_x_ratio
            target_state["y_ratio"] = screen_y_ratio
            target_state["is_tracking"] = True
            target_state["recenter_progress"] = head_tracker.debug.get("recenter_progress", 0.0)

        console.emit(f"x={result.cursor_x_ratio:.3f} y={screen_y_ratio:.3f}")

        # 정지 유예 중엔 모든 키를 무시한다 — head.py와 동일 이유(그 파일 주석 참고).
        # reset_event_gates()도 매 프레임 함께 호출 — head_tracker 내부 응시 게이트가
        # 유예 중에 몰래 진행되다가 유예 끝나자마자 재정렬이 발화하는 문제 대응
        # (head.py의 동일 블록 주석 참고)
        if now_sec - tracking_edge_state["became_tracking_sec"] < SETTLE_DELAY_SEC:
            head_tracker.reset_event_gates()
            return

        # 유예 종료 직후 딱 한 번 재중심 — head.py와 동일 이유(그 파일 주석 참고)
        if not tracking_edge_state["recentered_after_settle"]:
            # 성공했을 때만 "했다"고 표시한다 — 실패했는데 표시해 버리면 이번
            # 유예 사이클에선 다시 시도하지 않아, "유예가 끝나면 커서는 무조건
            # 화면 중앙"이라는 보장이 조용히 깨진다
            if head_tracker.recenter_cursor(user_face):
                tracking_edge_state["recentered_after_settle"] = True

        # 입 트리거는 _update_mouth_gesture가 전담 — head.py와 동일 이유(그 파일 주석 참고)
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
                except Exception:   # noqa: 방어적 — head.py와 동일 이유
                    logger.exception("클릭 시도 실패")

        _update_mouth_gesture(now_sec)

    def _inference_loop():
        # 같은 오류가 반복될 때 기록을 줄인다 — head.py와 동일 이유·동일 방식
        # (그 파일 _inference_loop 설명 참고: 카메라가 빠지면 초당 수십 건이
        # 쌓여 로그가 디스크를 채운다).
        last_err_key = None
        err_streak = 0
        last_err_log_sec = 0.0
        while not state["should_quit"]:
            # 예외가 나도 찍는다 — "루프가 돌고 있는가"만 보는 신호다
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

    overlay_h_px, overlay_w_px = mouse.screen_h_px, mouse.screen_w_px
    overlay_canvas = np.empty((overlay_h_px, overlay_w_px, 3), dtype=np.uint8)

    if not args.no_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception:   # noqa: 방어적 — 이 OpenCV 빌드가 TOPMOST를 지원 안 해도 계속 진행
            logger.warning("Could not set window topmost - continuing anyway")
        overlay_canvas[:] = TRANSPARENT_KEY_COLOR
        cv2.imshow(WINDOW_NAME, overlay_canvas)
        cv2.waitKey(1)
        enable_transparent_overlay(WINDOW_NAME, TRANSPARENT_KEY_COLOR, OVERLAY_ALPHA)

    refresh_hz = _get_refresh_rate_hz()
    # 화면 주사율과 상한(RENDER_FPS_CAP) 중 낮은 쪽에 맞춘다 — 상수 설명 참고
    render_hz = min(refresh_hz, RENDER_FPS_CAP)
    frame_interval_sec = 1.0 / render_hz
    # 보간 계수의 시간 기준 — 렌더 속도와 분리한다(LERP_REFERENCE_FPS 설명 참고).
    # 이 값을 frame_interval_sec으로 두면 렌더를 올릴 때 커서 손맛이 같이 바뀐다
    lerp_ref_dt_sec = 1.0 / LERP_REFERENCE_FPS
    logger.warning(
        "헤드트래커(코-눈 혼합점 기준) 시작 — 코-눈 혼합점으로 커서 이동, 입 벌리기/1.5초 응시로 클릭. "
        "커서는 화면 %s %.0f%%만 사용. 화면 주사율 %dHz · 렌더 %dHz 고정. "
        "콘솔에 p+Enter로 실제 마우스 제어 켜기/끄기, quit+Enter로 종료. "
        "창이 있으면 q/ESC로도 종료.",
        "하단" if CURSOR_Y_ANCHOR_BOTTOM else "상단", CURSOR_Y_SPAN * 100,
        refresh_hz, render_hz,
    )
    logger.info("Startup %.1fs (interpreter + imports + models)", time.monotonic() - PROCESS_START_SEC)

    displayed_x_ratio, displayed_y_ratio = CENTER_X_RATIO, _cursor_y_to_screen(CENTER_Y_RATIO)
    # 정지 히스테리시스용 anchor·상태 — head.py와 동일 이유(그 파일 주석 참고)
    anchor_x_ratio, anchor_y_ratio = displayed_x_ratio, displayed_y_ratio
    is_moving = False
    resume_pending_since_sec = None   # RESUME_CONFIRM_SEC 지속시간 확인용 — head.py와 동일 이유
    arrived_since_sec = None   # SETTLE_CONFIRM_SEC 확인용 — head.py와 동일 이유
    native_cursor_hidden = False   # 추적이 처음 시작되는 틱에 딱 한 번만 hide() — head.py와 동일 이유
    camera_window_open = False   # "cam on"/"cam off" 전환 시 창을 새로 만들지/닫을지 추적
    was_settling = False   # 정지 유예 구간에 막 들어선 첫 틱만 감지 — head.py와 동일 이유
    watchdog_fault = None        # 지금 감지된 고장 종류(None이면 정상)
    watchdog_logged_sec = 0.0    # 같은 고장을 마지막으로 기록한 시각
    last_drawn_signature = None   # 오버레이 다시 그리기 생략용 — head.py와 동일 이유
    # 지난 틱에 실제로 뭔가 그린 영역 — 다음 틱에 여기만 지운다
    # (_blank_rect / _cursor_rect 설명 참고)
    last_drawn_rect = None
    render_tick_count = 0   # 진단 로그 주기 계산용

    prev_tick_sec = time.monotonic()   # 직전 틱 시각 — 위 tick_dt_sec 계산용
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

            # 감시 판정 — head.py와 동일(그 파일 주석에 판정 근거가 설명돼 있다)
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
                if (watchdog_fault != previous_fault
                        or tick_start_sec - watchdog_logged_sec >= WATCHDOG_LOG_SEC):
                    logger.error("감시: %s", watchdog_fault)
                    watchdog_logged_sec = tick_start_sec
            elif previous_fault:
                logger.warning("감시: 정상으로 돌아왔습니다 (직전 상태: %s)", previous_fault)

            # 정지 유예 — head.py와 동일 이유(그 파일 주석 참고)
            is_settling = is_tracking and (tick_start_sec - became_tracking_sec) < SETTLE_DELAY_SEC
            # 드래그 중엔 잡음을 더 넓게 무시한다 (DRAG_DEAD_ZONE_SCALE 설명 참고)
            dead_zone_ratio = RENDER_DEAD_ZONE_RATIO * (
                DRAG_DEAD_ZONE_SCALE if feedback.is_holding else 1.0)

            if is_settling:
                # 유예 중엔 고정된 화면 중앙을 향해 완만히(CATCHUP_LERP_ALPHA)
                # 다가간다(2026-08-18 재설계, 사용자 요청 — "캘리브레이션 할떄 5초
                # 후에 커서가 화면 중앙에 가게해줘") — head.py와 동일 이유(그 파일
                # "if is_settling:" 분기 주석 참고)
                center_x_ratio, center_y_ratio = CENTER_X_RATIO, _cursor_y_to_screen(CENTER_Y_RATIO)
                dx = center_x_ratio - displayed_x_ratio
                dy = center_y_ratio - displayed_y_ratio
                gap = math.hypot(dx, dy)
                if gap > dead_zone_ratio:
                    settle_alpha = _dt_adjusted_alpha(
                        CATCHUP_LERP_ALPHA, tick_dt_sec, lerp_ref_dt_sec)
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
                    # 유예 종료 — head.py와 동일 이유(그 파일 주석 참고, 이제
                    # 유예 종료 시점엔 아무것도 강제하지 않고 정상 추적이 자연스럽게 이어받는다)
                    anchor_x_ratio, anchor_y_ratio = displayed_x_ratio, displayed_y_ratio
                    is_moving = False
                    resume_pending_since_sec = None
                    arrived_since_sec = None
                elif is_moving:
                    # 확정된 뒤 — head.py와 동일 이유(그 파일 "elif is_moving:" 분기
                    # 주석 참고, SETTLE_CONFIRM_SEC로 "연속 동작 중 뚝뚝 끊김" 대응)
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
                        arrived_since_sec = None
                        alpha = CATCHUP_LERP_ALPHA if gap > LARGE_GAP_RATIO else RENDER_LERP_ALPHA
                        if feedback.is_holding:
                            alpha *= DRAG_LERP_ALPHA_SCALE   # 드래그 중엔 더 부드럽게
                        alpha = _dt_adjusted_alpha(alpha, tick_dt_sec, lerp_ref_dt_sec)
                        displayed_x_ratio += alpha * dx
                        displayed_y_ratio += alpha * dy
                        if state["is_control_active"]:
                            mouse.move(displayed_x_ratio, displayed_y_ratio)
                else:
                    # 정지 히스테리시스 + 지속시간 확인 — head.py와 동일 이유(그 파일
                    # "else:" 분기 주석 참고, 대기 중에도 완만히 미리 따라간다)
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
                                CATCHUP_LERP_ALPHA, tick_dt_sec, lerp_ref_dt_sec)
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
                # 다시 그리기 생략 최적화 — head.py와 동일 이유(그 파일 "if not
                # args.no_window:" 분기 주석 참고, "프레임 잘나오게 뚝뚝 끊기지
                # 않게 최적화해줘" 대응)
                feedback_key = feedback.state_key(tick_start_sec)
                draw_signature = (
                    round(displayed_x_ratio, 4), round(displayed_y_ratio, 4),
                    is_settling, is_tracking, round(recenter_progress, 2), feedback_key,
                    watchdog_fault,   # 빼먹으면 커서가 제자리일 때 경고가 화면에 안 뜬다
                )
                if draw_signature != last_drawn_signature:
                    # 화면 전체가 아니라 지난 틱에 그린 영역만 되돌린다 —
                    # 실측 6.9~13.1ms -> 0.02ms (_blank_rect 설명 참고)
                    _blank_rect(overlay_canvas, last_drawn_rect)
                    drawn_rect = None
                    if is_tracking:
                        if not native_cursor_hidden:
                            # 오버레이 점을 처음 그리는 틱에만 hide() — head.py와 동일 이유(그 파일 주석 참고)
                            cursor_hider.hide()
                            native_cursor_hidden = True
                        # 정지 유예 중엔 재정렬 진행 링을 끈다 — head.py와 동일 이유(그 파일 주석 참고)
                        draw_cursor(overlay_canvas, displayed_x_ratio, displayed_y_ratio,
                                   0.0 if is_settling else recenter_progress,
                                   color=feedback.color(tick_start_sec),
                                   filled=feedback.is_holding)
                        drawn_rect = _union_rect(drawn_rect, _cursor_rect(
                            displayed_x_ratio, displayed_y_ratio, overlay_w_px, overlay_h_px))
                        if is_settling:
                            # 화면 중앙 고정 — head.py와 동일 이유(그 파일 주석 참고,
                            # "이 문구는 무조건 화면 가운데에 뜨게 해주고" 요청). 한글이라
                            # cv2.putText로는 안 그려진다 — head.py와 동일 이유. 두 줄
                            # 가운데 정렬도 head.py와 동일 방식(그 파일 주석 참고)
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
            # head.py와 동일 이유(그 파일 주석 참고)
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
