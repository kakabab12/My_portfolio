"""헤드트래커 단독 실행기(미간 기준) — 실제 OS 마우스 포인터를 제어한다
(2026-08-13 신설, 사용자 요청: "eyebrow.py랑 eyebrow.exe 만들고... 미간
그러니까 양쪽 눈 사이로 해서 움직이게 해줘 head.py 처럼").

head.py와 완전히 동일한 구조·기능(오버레이 표시, 실제 마우스 이동·클릭, 위쪽
절반 제한, 화면 주사율 보간, 네이티브 커서 숨김 등 — 그 파일 독스트링에 각
설계 이유가 자세히 있다)이고, **커서 위치의 기준점 하나만 다르다**: head.py는
코끝(LMK_NOSE_TIP)을 쓰고, 이 파일은 미간(양쪽 눈 사이 중점)을 쓴다.

★기준점 교체 방식 — src/postprocess/head_tracker.py의 HeadTracker가 이번에
cursor_point_fn 매개변수를 새로 받도록 확장됐다(2026-08-13, 이 파일 대응으로
추가 — 그 클래스 독스트링 참고). EMA 스무딩·안구간거리 정규화·캘리브레이션·
select/home/calibration 판정 등 나머지 로직은 기준점이 코끝이든 미간이든
완전히 동일해서, 이 로직 전체를 복제하지 않고 "어디를 볼지"만 바꿔 끼운다.
미간 좌표는 이미 이 프로젝트 전역에서 안구간거리 계산에 쓰는 검증된 두 점
(LMK_LEFT_EYE_OUTER=33, LMK_RIGHT_EYE_OUTER=263, 양쪽 눈 바깥쪽 끝)의 중점으로
구한다 — 새로운 랜드마크 인덱스를 추측해서 쓰지 않고 이미 신뢰할 수 있다고
검증된 값만 재사용한다.

이 파일 자체(오버레이·마우스 제어·스레드 구조·클릭 통과·주사율 페이싱·커서
크기 등)는 head.py를 그대로 복사한 것이다 — 두 실행기가 독립 배포되는 별개
exe라 코드를 공유 모듈로 더 쪼개기보다 이쪽이 낫다고 판단했다(dpad_overlay
launcher 계열과 같은 선례).

사용법 (프로젝트 루트에서):
    py eyebrow.py [--device N] [--config path] [--no-window]
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
from collections import deque

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
    FaceEstimator, LMK_LEFT_EYE_OUTER, LMK_RIGHT_EYE_OUTER,
)
from src.inference.preprocessor import Preprocessor
from src.postprocess.face_anchor import FaceAnchor
from src.postprocess.head_tracker import EVENT_SELECT, HeadTracker
from src.utils.config_loader import load_config
from src.utils.cursor_render import (   # 커서 크기·그리기 공용 (2026-08-31)
    CURSOR_MARKER_SIZE_PX, CURSOR_RADIUS_PX, CURSOR_THICKNESS_PX,
    cursor_reach_px, draw_cursor as _draw_cursor_shared,
)
from src.utils import console, preflight, version
from src.utils.logger import get_logger, init_logging
# 카메라 미리보기 전용(2026-08-13 사용자 요청 — "cam on 하면 카메라 보이게") —
# head.py와 동일 이유로 원본(카메라 해상도 기준) draw_cursor·draw_head_debug_panel을
# 그대로 쓴다. 오버레이용 큰 draw_cursor와 이름이 겹쳐 카메라 쪽만 접두어로 구분
from src.utils.visualize import draw_cursor as draw_cursor_camera
from src.utils.visualize import draw_head_debug_panel

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

# ★2026-08-28 신설 — 실시간 조절 UI(scripts/tuning_ui.py, "tune"+Enter로 켠다)
# 대응. 이 파일이 있으면 그 값으로 아래 SENSITIVITY_*_OVERRIDE·ARC_COMPENSATION
# 을 덮어써서 시작하고, 실행 중에도 몇 틱마다 수정 시각을 확인해 바뀌었으면
# 다시 읽어 head_tracker에 바로 반영한다(_maybe_reload_tuning 참고). UI 창을
# 닫아도 이 파일에 값이 남아 있으므로 마지막 조절값이 계속 유지되고, 트래커를
# 재시작해도 그 값 그대로 시작한다 — "UI 꺼도 적용 유지" 요청이 이 파일 하나로
# 자연히 해결된다.
TUNING_FILE_PATH = os.path.join(ROOT_DIR, "eyebrow_tuning.json")
TUNING_POLL_INTERVAL_SEC = 0.5   # 이 정도면 "실시간"으로 느껴지면서도 매 틱
                                 # 파일 I/O를 하지 않는다(대부분 안 바뀐 상태로
                                 # 확인만 하므로 os.path.getmtime 정도는 공짜에
                                 # 가깝지만, 그래도 굳이 매 프레임 할 이유는 없다)


def _load_tuning_overrides(path):
    """튜닝 파일이 있으면 dict, 없거나 손상됐으면 None.

    scripts/tuning_ui.py가 쓰는 형식과 같다(그 파일 save_tuning 참고) —
    다만 이쪽에서 새로 import하지 않고 똑같은 최소 로직만 복제해 둔다.
    UI 도구가 없어도(또는 옛 버전이어도) 트래커 자체는 독립적으로 돌아가야
    하기 때문이다.
    """
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

WINDOW_NAME = "eyebrow tracker overlay"
CAMERA_WINDOW_NAME = "eyebrow tracker camera"

# 키 확인 — head.py와 동일 이유·동일 처방(그 파일 _pump_window_events 설명 참고).
# ★ 2026-08-20: cv2.waitKey(1)이 항상 15~16ms를 먹던 것을 cv2.pollKey()(실측
# 0.00ms)로 교체했다. 공짜가 됐으므로 호출을 건너뛰던 우회도 없쥜다
_HAS_POLL_KEY = hasattr(cv2, "pollKey")


def _pump_window_events():
    """창 메시지 처리 + 키 확인 — head.py 동명 함수와 동일."""
    return (cv2.pollKey() if _HAS_POLL_KEY else cv2.waitKey(1)) & 0xFF

# 세로 절반 위쪽 제한 — head.py와 동일. 0.5 = 화면 상단 절반까지만 커서가 감
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

# 렌더 스레드 보간 계수 — head.py와 동일 이력·동일 값(그 파일 상수 설명 참고:
# 0.35→0.6→0.45→0.38, 반응성↔정지 시 떨림 절충 + 정지 히스테리시스 도입 후
# 재하향)
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

# 커서 민감도·입벌림 문턱 — head.py와 동일 이력·동일 값(그 파일 상수 설명 참고:
# 1.6/1.7→1.3/1.4로 되낮춤, 정지 시 떨림 완화. 2026-08-14: 수직 감도만 head.py와
# 별도로 두 차례 재상향(1.4→1.6→2.0) — "eyebrow 커서 수직 민감도만 더 키워줘"
# 요청으로 head.py(1.6)보다 eyebrow만 더 높다. 감도가 높을수록 잡음도 더
# 증폭되니 아래 RENDER_DEAD_ZONE_RATIO/RESUME_GAP_RATIO도 head.py보다 더 크게 잡았다
SENSITIVITY_X_OVERRIDE = 2.05   # 2026-08-28 연구실 키오스크 실기로 확정(팀장님 확인) — 1.124 -> 2배(2.248) 시도 후 추가 조정
SENSITIVITY_Y_OVERRIDE = 6.0    # 2026-08-28 연구실 키오스크 실기로 확정(팀장님 확인) — 1.80 -> 2배(3.60) 시도 후 추가 조정. 감도가 커서 아래 눈 깜빡임 흔들림(GLABELLA_JITTER_MEDIAN_WINDOW 설명 참고)·좌우 이동 시 곡률(ARC_COMPENSATION 설명 참고)이 더 크게 드러난다

# 가로 이동 시 세로가 활처럼 휘는 것(U자) 보정 — 원리·계산식은
# forehead.py의 동명 상수 설명과 head_tracker.py `_arc_compensation` 설명
# 참고. 기본 0.0(꺼짐)이라 이 상수를 추가하는 것만으로는 아무 동작도
# 안 바뀐다 — head.py도 이 필드를 안 쓰므로 영향 없음.
#
# ★2026-08-28 신설 — 연구실 키오스크(카메라가 아래에서 위를 보는 각도)에서
# "살짝 U자로 휜다"는 보고로 추가했다. forehead.py 때와 같은 이유로 **값을
# 추측해 넣지 않는다** — 이번엔 그럴 만한 근거도 없다.
#
# ★카메라 각도·위치에 무관한 만능 계수는 없다 — 이 휨의 진짜 원인(미간이
# 회전축보다 카메라 쪽으로 살짝 튀어나와 있어 고개를 돌리면 원근으로 세로까지
# 밀리는 것)이 카메라가 어디서 얼굴을 보고 있느냐에 따라 달라지기 때문이다.
# 카메라가 정면에 있을 때와 아래에서 올려다볼 때는 같은 고개 회전도 이미지에
# 다르게 투영된다 — 사용자가 직접 짚은 것과 같은 이유다. 그래서 "장소마다
# 다시 잰다"가 유일하게 정직한 답이다:
#
#     py eyebrow.py                   (트래커 실행, 캘리브레이션 끝날 때까지 대기)
#     (측정 도구 measure_arc.py는 2026-08-31 정리로 삭제 — 이제 잴 필요가
#      없다: 상대 회전 매핑 + 자동 곡률 소거(auto_arc.py)가 실행 중에 스스로
#      맞춘다. 아래 기록은 그 이전 방식의 이력이다)
#
# measure_arc.py는 화면에 실제로 나타나는 커서 궤적만 재므로 카메라 각도가
# 무엇이든 그 배치에서 실제로 필요한 값을 그대로 알려준다 — 공식을 새로
# 유도하는 대신, 이미 있는 이 측정 도구를 "그 자리에서" 돌리는 것 자체가
# 카메라 각도 독립적인 해법이다. 연구실 카메라에서 측정한 값을 여기 넣고,
# 카메라 각도·위치가 바뀌면(예: 이 키오스크를 다른 카메라 배치로 옮기면)
# 다시 측정할 것.
#
# ★2026-08-28 연구실 실기로 확정 — -0.7 → -0.75로 미세조정. measure_arc.py의
# 정확한 출력값이 아니라 실사용 중 직접 시험해 나온 값이다 — forehead.py
# 때처럼 정밀한 R² 측정을 거치진 않았지만, 실제 사용 중 확인이 가장 확실한
# 검증이라 이 값을 그대로 쓴다. (measure_arc.py는 삭제 — 위 참고) 다듬으려면
# 정밀 재측정할 것.
ARC_COMPENSATION = -0.75
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

# ★2026-08-28 신설 — "버벅거림(프레임 문제 같다)" 대응. 고정 alpha EMA는
# 잡음 억제량이 항상 똑같아서, 감도를 6.0까지 올린 지금은 카메라 프레임마다
# 조금씩 있는 위치 흔들림도 그만큼 증폭돼 그대로 나간다 — 가만히 있을 때도
# 빠르게 움직일 때도 같은 세기로 누르다 보니, 가만히 있을 때는 덜 눌러서
# 떨리고 빠르게 움직일 때는 더 눌러서 늦게 따라오는 두 문제를 동시에 겪는다.
#
# forehead.py에 이미 적용해 검증한 1€ 필터(ONE_EURO_ENABLED, OneEuroFilter
# 설명 참고)를 그대로 가져온다 — 속도를 같이 봐서 가만히 있을 땐 세게
# 누르고 빠르게 움직일 땐 풀어준다. 값은 forehead.py에서 실기로 맞춘 값을
# 그대로 시작점으로 쓴다 — 완전히 다른 감도(forehead X=2.8·Y=3.8 vs
# eyebrow X=2.05·Y=6.0)라 최적값은 다를 수 있으니 실기로 다시 다듬을 것.
ONE_EURO_ENABLED = True
ONE_EURO_MIN_CUTOFF = 0.25
ONE_EURO_BETA = 1.5
ONE_EURO_DISTANCE_ADAPTIVE = True
ONE_EURO_REFERENCE_DIST_PX = 60.0

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

# 커서를 "얼굴이 향한 방향"으로만 움직이는 방식 — ★이 파일은 끈다.
#
# head.py는 켜서 커서 밀림을 없앴는데(그 파일 상수 설명 + head_tracker.py
# _CursorMapper 독스트링 참고), 그 방식은 이 파일에 **원리적으로 적용할 수 없다**.
#
# 그 방식은 "기준점이 두 눈 중점에서 얼마나 벗어났는가"로 고개 회전을 읽는다.
# 그런데 이 파일의 기준점인 미간은 _glabella_point()가 두 눈 중점 그 자체로
# 계산한다 — 즉 벗어난 양이 항상 정확히 0이라 커서가 아예 안 움직이게 된다.
#
# 더 근본적으로, 미간은 얼굴에서 튀어나와 있지 않아 고개를 돌려도 두 눈에 대해
# 거의 제자리다. 코와 달리 "회전만" 담긴 신호를 뽑아낼 수가 없다 — 미간 모드는
# 본질적으로 머리의 화면상 위치(이동+회전이 섞인 값)를 따라가는 방식이다.
#
# 그래서 이 파일은 예전 매핑을 그대로 쓴다. 대가로 커서 밀림이 남는다(자동
# 재정렬도 껐으므로 되돌려주는 것도 없다) — 정확도가 중요하면 head.py를 쓸 것.
FACE_LOCAL_MAPPING = False

# ★상대 회전 매핑 (2026-08-31 신설) — src/postprocess/head_orientation.py 참고.
#
# 켜면 커서를 "중립 자세 대비 머리가 얼마나 돌아갔는가"로 정한다. 시작할 때
# 얼굴 랜드마크를 한 벌 저장해 두고, 매 프레임 지금 랜드마크를 그 중립에
# 겹치는 최적 회전을 SVD로 구한다 (Kabsch 1976 / Umeyama 1991).
#
# [무엇을 해결하나] 카메라를 옮길 때마다 다시 재던 일이 사라진다.
#   카메라 위치·기울기는 중립에도 똑같이 들어 있어 상대 회전에서 소거된다.
#   밑에서 올려보는 연구실 배치든 정면인 키오스크 배치든 그대로 동작한다.
#   ARC_COMPENSATION도 이 경로에서는 무시된다 - 투영을 안 거치니 휠 것이 없다.
#
# [실측 2026-08-31] 같은 프레임에서 기존 2D 방식과 동시에 재 비교한 신호 대 잡음:
#     가로 19.1 -> 24.2 (+27%)      세로 10.8 -> 14.9 (+38%)
#   강체 랜드마크 22개를 한꺼번에 정합해 개별 점 떨림이 평균되기 때문이다.
#   비용은 프레임당 0.7ms로 사실상 무시할 수 있다.
#
# [켜면 바뀌는 것] SENSITIVITY_X/Y_OVERRIDE와 ARC_COMPENSATION이 무시되고,
#   아래 두 각도가 그 자리를 대신한다. 각도는 카메라와 무관한 사람 기준
#   설계값이라 자리를 옮겨도 다시 맞출 필요가 없다.
ORIENTATION_MAPPING = True
# "고개를 몇 도 돌리면 커서가 화면 끝에 닿는가".
# 작으면 조금만 돌려도 끝까지(민감), 크면 더 돌려야 한다(정밀).
ORIENTATION_HALF_SPAN_X_DEG = 15.0
ORIENTATION_HALF_SPAN_Y_DEG = 10.0
FACE_LOCAL_GAIN = 2.0   # 위가 False라 실제로는 안 쓰인다(값 통일용)

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

# 추적 기준점 시각화 — head.py와 동일 이유(그 파일 상수 설명 참고: "cam on"으로
# 카메라를 켰을 때 지금 어느 점을 잡고 있는지 실측 위치로 보여준다)
TRACKING_POINT_LABEL = "미간"
TRACKING_POINT_COLOR = (0, 255, 255)   # 노랑(BGR) — 커서(초록)와 뚜렷이 구분


# 얼굴 변형 전환 순간 좌표 보류(2026-08-14 사용자 실기 보고 — 처음엔
# "눈감을때마다 커서가 움직이는데", 이어서 "입벌리면 커서가 움직여";
# 2026-08-18 재설계 — "더블클릭 하는게 입 두번 움직이는 거잖아 그럼 커서가
# 위아래로 흔들리는 현상이 있어"): 미간은 양쪽 눈 바깥쪽 끝(LMK_LEFT/RIGHT_EYE_OUTER)의
# 중점인데, 이 두 점은 MediaPipe가 얼굴 전체를 하나로 묶어 추정하는 랜드마크라
# 눈을 감거나(눈꺼풀이 그 자리를 덮음) 입을 크게 벌려도(턱이 크게 움직이며
# 아래쪽 얼굴 형태가 변해 위쪽 랜드마크 추정치도 살짝 같이 밀림) 위치 자체가
# 흔들린다 — head.py의 코끝도 같은 문제를 겪는다(그 파일 _stable_nose_point
# 독스트링 참고, 이 함수와 같은 방식으로 재설계됨).
# ★2026-08-18 이전엔 "눈이 감겨있거나 입이 벌어져 있는 동안 내내" 좌표를
# 얼렸는데, 그 뒤 입을 MOUTH_HOLD_SEC(2초) 이상 벌리고 있으면 "꾹 누르기"
# (드래그)가 되는 기능이 생기면서 이 방식이 드래그 자체를 막아버리는
# 문제가 됐다(벌리고 있는 내내 좌표가 얼어있으니 드래그가 전혀 안 움직임).
# 그래서 "벌려져/감겨 있는 동안 전부"가 아니라 "열림↔닫힘·뜸↔감음이 막
# 바뀐 순간"에만 짧게(NOSE_DEFORM_SUPPRESS_SEC) 얼리는 방식으로 바꿨다 —
# 더블클릭처럼 짧은 시간에 전환이 몰리면 그 전환들이 전부 억제돼 흔들림이
# 사라지고, 꾹 누르기로 오래 벌리고 있는 동안은 첫 전환 억제가 끝난 뒤로는
# 평소처럼 정상 추적되어 드래그가 그대로 된다
# ★2026-08-18 문턱 재조정 — head.py와 동일 이유(그 파일 상수 설명 참고,
# "입벌릴때 커서가 움직여" 재발 — MOUTH_OPEN_MARGIN_OVERRIDE를 0.12로 낮추면서
# 이 억제 문턱과의 순서가 어긋났던 것)
# 입이 열린 뒤 랜드마크가 가라앉기를 기다리는 시간 — head.py와 동일 값
NOSE_DEFORM_SETTLE_SEC = 0.12
# 보정량 상한(px). 이보다 크면 얼굴이 일그러진 게 아니라 고개를 돌린 것이다
NOSE_DEFORM_MAX_PX = 14.0
NOSE_DEFORM_SUPPRESS_SEC = 0.2   # (옛 방식 잔재 — 지금은 안 쓴다)
EYE_CLOSE_SUPPRESS_THRESHOLD = 0.35
MOUTH_OPEN_SUPPRESS_THRESHOLD = 0.08
_glabella_deform_state = {
    "neutral": None,        # 입 다물었을 때의 미간 위치
    "neutral_jaw": 0.0,
    "k": None,              # 턱 벌림 1당 미간이 밀리는 양 (x, y)
    "was_open": False,
    "opened_at_sec": 0.0,
    "pre_open": None,
}

# ★2026-08-28 신설 — 눈 깜빡일 때 커서가 아래로 튀는 문제 대응(연구실 키오스크
# 실기 보고, 수직 감도를 6.0까지 올린 뒤 두드러짐 — 위 SENSITIVITY_Y_OVERRIDE
# 이력 참고).
#
# 미간 기준점이 정확히 "두 눈 바깥쪽 끝의 중점"이라(_glabella_point 독스트링
# 참고), 눈을 깜빡이는 순간 눈가 근육 수축으로 이 랜드마크가 1~2프레임 살짝
# 흔들리면 그 잡음이 감도 6배로 그대로 증폭된다.
#
# _median() 설명(head_tracker.py)이 이미 짚은 원칙 그대로다 — "눈 깜빡임 같은
# 순간적 이상치는 평균이 아니라 중앙값으로 거른다." 거기서는 캘리브레이션
# 한 번에만 썼는데, 여기서는 매 프레임 흐르는 신호에 상시 적용한다. 평균이면
# 이상치 한 프레임에도 끌려가지만, 중앙값은 창의 절반 이상이 정상이면 이상치를
# 완전히 무시한다.
#
# 창을 5프레임(30fps 기준 약 0.17초)으로 잡은 근거 — 일반적인 눈 깜빡임은
# 보통 0.1~0.4초라 이 창 안에 거의 다 들어오면서도, 창이 너무 길면 실제 고개
# 이동에 그만큼 반응이 늦어진다(체감 지연). 정확한 값이 아니라 근사치다 —
# 실기에서 여전히 흔들리면 창을 늘리고, 반응이 둔하면 줄일 것.
GLABELLA_JITTER_MEDIAN_WINDOW = 5


class _RollingMedianPoint:
    """최근 N개 (x, y) 표본의 중앙값 — 순간적 이상치(눈 깜빡임 등) 흡수용."""

    def __init__(self, window):
        self._xs = deque(maxlen=window)
        self._ys = deque(maxlen=window)

    def update(self, point):
        self._xs.append(point[0])
        self._ys.append(point[1])
        return (self._median(self._xs), self._median(self._ys))

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2]


_glabella_jitter_filter = _RollingMedianPoint(GLABELLA_JITTER_MEDIAN_WINDOW)

# ★2026-08-28 추가, 같은 날 재설계 — 중앙값 필터만으로는 부족했다(사용자
# 실기 재보고: "여전히 눈감을때 커서가 내려가더라"). 이유 — 중앙값은 "창의
# 절반 이상이 정상값"일 때만 이상치를 걸러내는데, 눈을 **감고 있는 동안
# 내내**(수백 ms, 창의 절반을 넘길 수 있다) 미간 랜드마크가 계속 낮게
# 잡히면 중앙값 입장에선 그게 "이상치"가 아니라 "다수"가 돼버려서 못 거른다.
#
# ★1차 시도(점 단위 고정)도 부족했다 — _glabella_point에서 미간 좌표만
# 얼렸는데도 여전히 흔들렸다. 원인을 다시 보니 head_tracker.py의
# HeadTracker.update()가 eye_left_px/eye_right_px를 cursor_point_fn과는
# **별도로 face에서 직접** 다시 읽어(그 파일 670~671행), 안구간거리
# (interocular_dist_px -> _smoothed_dist_px)를 계산한다. eyebrow.py는
# face_local이 아니라(FACE_LOCAL_MAPPING=False) dx = (좌표차이) /
# _smoothed_dist_px 로 나누는데, 이 분모가 눈 깜빡임 중 살짝만 흔들려도
# 분자(미간 좌표)를 아무리 얼려도 결과(dx, dy)가 같이 흔들린다 — 점 하나만
# 얼리는 걸로는 이 경로를 못 막는다.
#
# 그래서 이번엔 **더 위 단계**에서 막는다 — 눈을 감은 동안은 아예
# head_tracker.update() 자체를 안 부르고 직전 결과를 그대로 재사용한다
# (_process_one_frame의 _BLINK_GATE 사용부 참고). 호출 자체를 건너뛰므로
# 안구간거리 평활 상태·EMA·1€ 필터 등 내부 상태가 전부 통째로 멈췄다가
# 눈을 뜨면 멈췄던 지점에서 그대로 이어간다 — 분자든 분모든 어떤 내부
# 계산이 얽혀 있든 상관없이 통하는 방식이다.
#
# 고정 임계값을 안 쓰는 이유는 head_tracker.py의 "home"(양눈 감고 버티기)
# 판정과 같다 — 사람마다 평상시 eyeBlink 점수가 0.1~0.6까지 벌어져서, 고정
# 값 하나면 어떤 사람은 항상 "감은 걸로" 잘못 잡혀 커서가 아예 안 움직이는
# 심각한 오탐이 날 수 있다. 그 대신 각자의 평상시 점수를 천천히 따라가며
# 그보다 마진 이상 튀었을 때만 "지금 감았다"고 본다(아래 _BlinkGate).
#
# EYE_BLINK_FREEZE_MARGIN=0.15는 MOUTH_OPEN_MARGIN_OVERRIDE(0.12)와 같은
# 자릿수로 잡은 시작값이다 — 실측 도구가 없어 정확한 값은 아니다. 실기에서
# 눈 깜빡임이 여전히 새면(고정이 안 걸림) 낮추고, 고개를 자연스럽게 움직이는
# 중에도 커서가 자꾸 멈칫하면(오탐) 높일 것.
EYE_BLINK_FREEZE_MARGIN = 0.15

# ★2026-08-28 추가 — 진입 문턱 하나로는 "감고 뜨는" 전환 램프 구간이 새서
# 흔들림이 남았다(_BlinkGate 독스트링 참고). 해제는 진입보다 낮은 문턱까지
# 내려와야 하고(EYE_BLINK_RELEASE_MARGIN), 한 번 얼면 최소 이 시간은
# 유지한다(EYE_BLINK_MIN_HOLD_SEC) — 둘 다 실측 도구가 없는 시작값이라
# 실기로 조정할 것. 최소 유지 시간은 일반적인 깜빡임 길이(0.1~0.4초)보다
# 짧게 잡아 진짜 응시·클릭 동작을 방해하지 않게 했다.
EYE_BLINK_RELEASE_MARGIN = 0.05
EYE_BLINK_MIN_HOLD_SEC = 0.08


class _BlinkGate:
    """지금 이 프레임이 "눈 깜빡이는 중"인지 판단 — 위 설명 참고.

    _process_one_frame에서 head_tracker.update() 호출 여부를 결정하는 데
    쓴다. face 하나만 넣으면 되고, 내부적으로 평상시 기준선을 스스로 갱신한다.

    ★2026-08-28 재수정 — 진입 문턱 하나만으론 "덜하지만 여전히" 흔들렸다
    (사용자 재보고: "눈감고 뜨면 커서 움직이는건 여전해, 전보다 덜할 뿐").
    원인 — 실제 깜빡임은 순간 on/off가 아니라 **눈꺼풀이 감기고 뜨이는 동안
    점수가 서서히 오르내리는 램프**다. 진입 문턱(margin)을 넘는 "확실히 감은"
    구간만 얼렸더니, 그 앞뒤로 눈이 반쯤 감기고/반쯤 뜨이는 몇 프레임은 문턱
    아래라 그대로 통과해 흔들림이 남았다 — 정확히 "감고 뜨는" 전환 순간이라는
    보고와 맞아떨어진다.

    04_참고_논문_기술.md에 이미 정리해 둔 "히스테리시스 + 쿨다운" 원칙(입
    벌림 등 다른 판정에도 쓰는 방식)을 여기도 그대로 적용한다 — 켜는 문턱과
    끄는 문턱을 다르게 잡는다:
      · 진입(EYE_BLINK_FREEZE_MARGIN) — 이 이상 튀면 "감기 시작"
      · 해제(EYE_BLINK_RELEASE_MARGIN, 진입보다 낮음) — 완전히 평상시 근처로
        가라앉아야 "다 떴다"고 보고 얼림을 푼다 — 눈이 덜 뜬 채로(램프 하강
        구간) 다시 풀리는 걸 막는다
      · 최소 유지 시간(EYE_BLINK_MIN_HOLD_SEC) — 한 번 얼면 최소 이 시간은
        무조건 유지 — 램프 상승 구간에서 몇 프레임 만에 다시 풀리는 걸 막는다
    """

    def __init__(self, enter_margin, release_margin, min_hold_sec, neutral_alpha=0.03,
                clock=time.monotonic):
        self._enter_margin = enter_margin
        self._release_margin = release_margin
        self._min_hold_sec = min_hold_sec
        self._neutral_alpha = neutral_alpha
        self._clock = clock
        self._neutral = None
        self._blinking = False
        self._blink_started_sec = None

    def is_blinking(self, face):
        score = max(face.blendshape("eyeBlinkLeft"), face.blendshape("eyeBlinkRight"))
        now_sec = self._clock()

        if self._blinking:
            held_sec = now_sec - self._blink_started_sec
            if held_sec < self._min_hold_sec or (
                    self._neutral is not None
                    and score > self._neutral + self._release_margin):
                return True   # 아직 최소 유지 시간이거나, 아직 평상시 근처로 안 내려왔다
            self._blinking = False   # 완전히 풀렸다 — 아래에서 기준선 갱신 재개

        if self._neutral is not None and score > self._neutral + self._enter_margin:
            self._blinking = True
            self._blink_started_sec = now_sec
            return True

        self._neutral = score if self._neutral is None else (
            self._neutral + self._neutral_alpha * (score - self._neutral))
        return False


_BLINK_GATE = _BlinkGate(EYE_BLINK_FREEZE_MARGIN, EYE_BLINK_RELEASE_MARGIN,
                         EYE_BLINK_MIN_HOLD_SEC)


def _glabella_point(face):
    """미간(양쪽 눈 사이 중점) 픽셀 좌표 — HeadTracker의 cursor_point_fn으로 쓰인다.

    정확한 "미간" 전용 랜드마크 인덱스를 새로 추측하지 않고, 이미 이 코드베이스
    전역에서 안구간거리 계산에 쓰이며 검증된 두 점(양쪽 눈 바깥쪽 끝)의 중점을
    쓴다 — 기하학적으로 두 눈 사이의 중앙이라 "양쪽 눈 사이"라는 요청과
    정확히 일치하고, 신뢰도가 이미 확인된 값만 재사용해 안전하다.

    ★2026-08-24 — 입 벌림 보정을 head.py와 같은 방식으로 바꿨다.

    예전엔 입/눈 상태가 바뀌는 **순간에만** 0.2초 좌표를 얼렸다. 클릭에는 충분
    했지만 드래그(입을 0.7초 이상 벌리고 유지)에는 턱없이 짧아, 나머지 시간
    동안은 변형된 좌표가 그대로 나갔다 — 사용자 보고 "eyebrow 입 벌리면 커서가
    움직여"가 그것이다.

    이제 얼리는 대신 **턱 벌린 정도에 비례해 변형량을 뺀다**. "턱 벌림 1당 미간이
    몇 px 밀리는가" 계수를 벌림 직후 한 번 재고, 그 뒤로는 그 계수 x 지금 턱
    벌림을 뺀다. 고개를 돌려도 턱 벌림은 그대로라 보정량이 안 변하므로, 입을
    벌린 채 고개를 돌리는 **드래그 동작이 그대로 통과한다**.

    ★head.py와 다른 점: 저쪽은 코가 기준점이라 "두 눈 중점 대비 상대 위치"로
    다룰 수 있지만, 여기서는 기준점이 두 눈 중점 그 자체다. 그래서 상대 좌표를
    쓸 수 없고 화면 좌표로 잰다. 계수를 재는 0.12초 사이에 고개가 크게 움직이면
    그만큼 계수가 오염되는데, 아래 상한(NOSE_DEFORM_MAX_PX)으로 막는다.
    """
    now_sec = time.monotonic()
    left_px = face.landmark_px(LMK_LEFT_EYE_OUTER)
    right_px = face.landmark_px(LMK_RIGHT_EYE_OUTER)
    point = ((left_px[0] + right_px[0]) / 2.0, (left_px[1] + right_px[1]) / 2.0)
    # 미세한 순간적 흔들림 완화 — _RollingMedianPoint 설명 참고. 눈 감김
    # 자체로 인한 지속형 흔들림은 여기가 아니라 _process_one_frame의
    # _BlinkGate가 더 위 단계(head_tracker.update 호출 자체를 건너뜀)에서
    # 막는다 — 그 이유는 _BlinkGate 설명 참고.
    point = _glabella_jitter_filter.update(point)
    jaw = face.blendshape("jawOpen")
    is_open = jaw >= MOUTH_OPEN_SUPPRESS_THRESHOLD
    st = _glabella_deform_state

    if not is_open:
        # 입을 다물고 있다 — 평상시 값을 천천히 기억한다(빨리 따라가면 고개를
        # 돌리는 중의 값까지 평상시로 삼아 기준이 흔들린다)
        st["neutral"] = point if st["neutral"] is None else (
            st["neutral"][0] + 0.1 * (point[0] - st["neutral"][0]),
            st["neutral"][1] + 0.1 * (point[1] - st["neutral"][1]))
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
            shift = (point[0] - st["pre_open"][0], point[1] - st["pre_open"][1])
            limit = NOSE_DEFORM_MAX_PX
            shift = (max(-limit, min(limit, shift[0])), max(-limit, min(limit, shift[1])))
            st["k"] = (shift[0] / jaw_delta, shift[1] / jaw_delta)

    if st["k"] is None:
        # 계수를 아직 못 잰 구간은 변형된 좌표를 내보내면 안 된다 — 그 짧은
        # 사이에만 커서가 크게 튄다(head.py 실측: 화면 세로의 16%)
        return st["neutral"] if st["neutral"] is not None else point
    return (point[0] - st["k"][0] * jaw_delta, point[1] - st["k"][1] * jaw_delta)


def draw_cursor(frame, cursor_x_ratio, cursor_y_ratio, recenter_progress_ratio=0.0,
                color=None, filled=False):
    """visualize.draw_cursor와 같은 모양(원 + 십자 + 재정렬 진행 링)이지만 화면
    해상도 캔버스에 맞게 더 크게 그린다 — 위 CURSOR_RADIUS_PX 등 상수 설명 참고.

    color/filled는 클릭·드래그 피드백용(CursorFeedback 참고). 드래그 중엔 속을
    채워서, 색 구분이 어려운 사람도 "지금 누르고 있다"를 형태로 알 수 있게 한다.
    """
    # ★2026-08-31 — 실제 그리기는 src/utils/cursor_render.py 로 옮겼다.
    # 세 트래커에 같은 함수가 통째로 복사돼 있어(해시까지 동일) 커서를 고치려면
    # 세 곳을 똑같이 고쳐야 했다. 한 곳으로 모으면서 안티에일리어싱·부분 픽셀
    # 위치·대비 테두리·조준점을 함께 넣었다 (그 파일 독스트링 참고).
    return _draw_cursor_shared(
        frame, cursor_x_ratio, cursor_y_ratio,
        recenter_progress_ratio=recenter_progress_ratio,
        color=color or CURSOR_COLOR, filled=filled,
        progress_color=RECENTER_PROGRESS_COLOR)


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
    # 그리는 쪽(cursor_render)이 자기가 칠하는 최대 반경을 알려준다 — 여기서
    # 따로 계산하면 커서 모양을 바꿀 때마다 어긋나 잔상이 남는다
    reach = cursor_reach_px()
    return _clip_rect((x_px - reach, y_px - reach, x_px + reach, y_px + reach), w_px, h_px)


_KOREAN_FONT_CACHE = {}


def _get_korean_font(size_px):
    font = _KOREAN_FONT_CACHE.get(size_px)
    if font is None:
        font = ImageFont.truetype("malgun.ttf", size_px)
        _KOREAN_FONT_CACHE[size_px] = font
    return font


def put_korean_text(canvas_bgr, text, org, font_size_px, color_bgr):
    """한글을 그리고 **실제로 칠한 범위**를 돌려준다 (x0, y0, x1, y1). 못 그리면 None.

    ROI만 PIL 왕복 변환하는 최적화 — head.py와 동일 이유(그 파일 독스트링 참고).
    돌려주는 범위는 더티 사각형(지울 목록)에 그대로 넣으면 된다 — 아래 설명 참고."""
    x_px, y_px = int(org[0]), int(org[1])
    canvas_h_px, canvas_w_px = canvas_bgr.shape[:2]
    font = _get_korean_font(font_size_px)
    text_w_px = int(font.getlength(text))
    pad_px = font_size_px // 2
    x0, y0 = max(0, x_px - pad_px), max(0, y_px - pad_px)
    x1 = min(canvas_w_px, x_px + text_w_px + pad_px)
    y1 = min(canvas_h_px, y_px + font_size_px + pad_px * 2)
    if x1 <= x0 or y1 <= y0:
        return None
    b, g, r = color_bgr
    roi_bgr = canvas_bgr[y0:y1, x0:x1]
    pil_image = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil_image).text((x_px - x0, y_px - y0), text, font=font, fill=(r, g, b))
    canvas_bgr[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # ★2026-08-31 — 실제로 칠한 범위를 돌려준다 (사용자 실기 보고:
    # "처음에 커서 뜨면 한글이 안 지워진다").
    #
    # 예전에는 부르는 쪽이 글자 범위를 **따로 계산해서** 지울 목록에 넣었는데,
    # 여기서 실제로 칠하는 범위(글자 크기의 절반을 여백으로 두르고, 아래로는
    # 그 두 배)와 어긋나 있었다. 두 줄 문구 기준으로 아래쪽 36px, 좌우 11px씩이
    # 지울 목록 밖이라 영영 안 지워졌다.
    #
    # 두 곳에서 따로 계산하는 한 언젠가 또 어긋난다. 칠한 쪽이 알려준다.
    return (x0, y0, x1, y1)


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
            raise RuntimeError("eyebrow.py의 실제 마우스 제어는 Windows 전용입니다")
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


_TUNING_UI_SCRIPT = os.path.join(ROOT_DIR, "scripts", "tuning_ui.py")


def _launch_tuning_ui():
    """"tune"+Enter로 실시간 조절 창을 띄운다 (TUNING_FILE_PATH 설명 참고).

    별도 프로세스로 띄운다 — Tkinter의 자체 이벤트 루프를 이 트래커의
    OpenCV·mediapipe 스레드와 한 프로세스에 같이 두면 서로의 이벤트 루프가
    부딪힌다. 파일 하나로만 이어지는 구조라 프로세스가 갈라져 있어도 아무
    문제가 없다.
    """
    if not os.path.exists(_TUNING_UI_SCRIPT):
        logger.warning("실시간 조절 UI를 찾을 수 없습니다: %s", _TUNING_UI_SCRIPT)
        return
    try:
        subprocess.Popen([sys.executable, _TUNING_UI_SCRIPT,
                          "--tracker", "eyebrow", "--file", TUNING_FILE_PATH])
        logger.info("실시간 조절 창을 열었습니다")
    except OSError:
        logger.exception("실시간 조절 창을 여는 데 실패했습니다")


class _TuningReloader:
    """TUNING_FILE_PATH의 수정 시각을 주기적으로 확인해 바뀌었으면 다시
    읽어 head_tracker에 반영한다 — TUNING_POLL_INTERVAL_SEC 설명 참고.

    매 프레임 파일을 열어보지 않는다(그럴 필요도 없고, 카메라 프레임 루프
    안에서 불필요한 디스크 접근을 늘리고 싶지 않다) — 마지막으로 확인한
    시각과 비교해 간격이 지났을 때만, 그것도 mtime만 먼저 본다.
    """

    def __init__(self, path, poll_interval_sec, clock=time.monotonic):
        self._path = path
        self._poll_interval_sec = poll_interval_sec
        self._clock = clock
        # -inf로 시작 — 0.0으로 두면 첫 호출 시각이 우연히 0.0에 가까울 때
        # (예: 테스트의 가짜 시계, 혹은 time.monotonic()이 0에 가까운 극단적
        # 환경) "이미 확인한 지 얼마 안 됐다"고 오판해 첫 확인을 건너뛴다
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
            return   # 파일이 아직 없다 — UI를 한 번도 안 켰다는 뜻, 정상
        if mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        tuning = _load_tuning_overrides(self._path)
        if tuning is None:
            return
        head_tracker.set_pointer_tuning(
            sensitivity_x=tuning.get("sensitivity_x"),
            sensitivity_y=tuning.get("sensitivity_y"),
            arc_compensation=tuning.get("arc_compensation"),
            # ★2026-08-31 — 상대 회전 매핑에서는 위 세 값이 안 쓰인다.
            # 이 두 각도를 함께 넘겨야 조절 UI가 실제로 먹는다
            half_span_x_deg=tuning.get("orientation_half_span_x_deg"),
            half_span_y_deg=tuning.get("orientation_half_span_y_deg"))


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


def main():
    parser = argparse.ArgumentParser(
        description="헤드트래커(미간 기준) 단독 실행 — 미간 위치로 실제 OS 마우스 포인터를 옮기고 클릭한다")
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
    logger.warning("%s", version.describe("eyebrow.py"))
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
        print(version.describe("eyebrow.py"))
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

    # 실시간 조절 UI가 저장해 둔 값이 있으면 그걸로 시작한다 — 위
    # TUNING_FILE_PATH 설명 참고. UI를 한 번도 안 켰으면 파일 자체가 없어
    # 아래 하드코딩된 값 그대로 시작한다(기존과 동일)
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
    config["head_tracker"]["pointer"]["orientation_mapping"] = ORIENTATION_MAPPING
    config["head_tracker"]["pointer"]["orientation_half_span_x_deg"] = ORIENTATION_HALF_SPAN_X_DEG
    config["head_tracker"]["pointer"]["orientation_half_span_y_deg"] = ORIENTATION_HALF_SPAN_Y_DEG
    config["head_tracker"]["pointer"]["face_local"] = FACE_LOCAL_MAPPING
    config["head_tracker"]["pointer"]["face_local_gain"] = FACE_LOCAL_GAIN
    config["head_tracker"]["pointer"]["arc_compensation"] = arc_compensation
    config["head_tracker"]["pointer"]["one_euro_enabled"] = ONE_EURO_ENABLED
    config["head_tracker"]["pointer"]["one_euro_min_cutoff"] = ONE_EURO_MIN_CUTOFF
    config["head_tracker"]["pointer"]["one_euro_beta"] = ONE_EURO_BETA
    config["head_tracker"]["pointer"]["one_euro_distance_adaptive"] = ONE_EURO_DISTANCE_ADAPTIVE
    config["head_tracker"]["pointer"]["one_euro_reference_dist_px"] = ONE_EURO_REFERENCE_DIST_PX

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
    head_tracker = HeadTracker(config, cursor_point_fn=_glabella_point)   # 코끝 대신 미간
    tuning_reloader = _TuningReloader(TUNING_FILE_PATH, TUNING_POLL_INTERVAL_SEC)
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
    # 눈 깜빡임 중 head_tracker.update()를 건너뛸 때 재사용할 직전 결과 —
    # _BlinkGate 사용부(_process_one_frame) 참고
    last_result_state = {"result": None}

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

        faces = face_estimator.infer(frame)
        user_face = face_anchor.update(faces)
        now_sec = time.monotonic()
        if user_face is not None:
            last_face_state["face"] = user_face
            last_face_state["sec"] = now_sec
        elif now_sec - last_face_state["sec"] < DROPOUT_GRACE_SEC:
            user_face = last_face_state["face"]   # 짧은 순간의 미검출 — 마지막 얼굴로 봐준다

        # ★2026-08-28 눈 깜빡임 커서 흔들림 대응 — _BlinkGate 설명 참고.
        # 눈을 감은 동안은 head_tracker.update() 자체를 안 부르고 직전
        # 결과(last_result_state)를 그대로 재사용한다 — 내부 상태(안구간거리
        # 평활 등)까지 통째로 멈춰야 분자만 얼려선 못 막던 흔들림이 없어진다.
        if (user_face is not None and _BLINK_GATE.is_blinking(user_face)
                and last_result_state["result"] is not None):
            result = last_result_state["result"]
        else:
            result = head_tracker.update(user_face)
            last_result_state["result"] = result

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
                point_x_px, point_y_px = _glabella_point(user_face)
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
    logger.warning(
        "헤드트래커(미간 기준) 시작 — 미간으로 커서 이동, 입 벌리기/1.5초 응시로 클릭. "
        "커서는 화면 %s 절반만 사용. 화면 주사율 %dHz · 렌더 %dHz 고정. "
        "콘솔에 p+Enter로 실제 마우스 제어 켜기/끄기, quit+Enter로 종료. "
        "창이 있으면 q/ESC로도 종료.",
        "하단" if CURSOR_Y_ANCHOR_BOTTOM else "상단", refresh_hz, render_hz,
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
                        alpha = _dt_adjusted_alpha(alpha, tick_dt_sec, frame_interval_sec)
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
                            # ★2026-08-31 — 글자가 실제로 칠한 범위를 그리는 쪽에서 받아 지울 목록에
                            # 넣는다. 예전엔 여기서 따로 계산했는데 put_korean_text의 실제 범위와
                            # 어긋나(아래 36px·좌우 11px) 문구가 안 지워졌다 — 사용자 실기 보고
                            for _text_rect in (
                                    put_korean_text(overlay_canvas, SETTLING_LABEL_LINE1,
                                                    (int(center_x_px - line1_w_px / 2), line1_y_px),
                                                    line1_font_px, CURSOR_COLOR),
                                    put_korean_text(overlay_canvas, SETTLING_LABEL_LINE2,
                                                    (int(center_x_px - line2_w_px / 2), line2_y_px),
                                                    line2_font_px, CURSOR_COLOR)):
                                drawn_rect = _union_rect(drawn_rect, _clip_rect(
                                    _text_rect, overlay_w_px, overlay_h_px))
                    # ★고장 알림 (WATCHDOG_STALL_SEC 상수 설명 참고) — 추적
                    # 여부와 무관하게 그린다. 오히려 추적이 멈춘 상태가
                    # 바로 이걸 봐야 하는 상황이다. 화면 한가운데보다 조금
                    # 아래에 둬서 커서·안내 문구와 겹치지 않게 한다
                    if watchdog_fault:
                        fault_font_px = 26
                        fault_w_px = _korean_text_width_px(watchdog_fault, fault_font_px)
                        fault_x_px = int(CENTER_X_RATIO * overlay_w_px - fault_w_px / 2)
                        fault_y_px = int(_cursor_y_to_screen(1.0) * overlay_h_px) - 40
                        _fault_rect = put_korean_text(overlay_canvas, watchdog_fault,
                                                     (fault_x_px, fault_y_px), fault_font_px,
                                                     WATCHDOG_LABEL_COLOR)
                        drawn_rect = _union_rect(drawn_rect, _clip_rect(
                            _fault_rect, overlay_w_px, overlay_h_px))
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
