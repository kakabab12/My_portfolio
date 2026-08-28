"""카메라 자동 협상 — 설정된 조합이 안 먹히는 장치에서도 자동으로 맞는 조합을
찾는다 (2026-08-27 신설, 사용자 요청 — "MSMF + MJPG + 1280x720 이거 아니여도
모든 카메라 장치값을 받을수있게 해줘").

배경 — 사용자가 연구실에서 로지텍 카메라(0번)로 eyebrow.exe를 실행했더니
카메라가 안 켜졌다. 팀장님이 만든 다른 프로그램은 같은 0번을 잘 여는 걸로
봐서 카메라 자체 문제가 아니라 **이 프로젝트가 항상 MSMF+MJPG+1280x720
하나로만 여는 게 문제**였다. 지금까지는 안 되면 사람이 직접
`camera_check.py --diagnose`를 돌려 맞는 조합을 찾아 config.yaml을 고쳐야
했는데, 이제 트래커가 시작할 때 스스로 이 과정을 한다.

★안전 설계 — 반드시 지킬 것: 새 조합을 시험하는 동안은 **절대 본 프로세스
안에서 cv2.VideoCapture를 열지 않는다.**

이 프로젝트는 8/26에 실측으로 이미 확인한 사실이 있다 — MSMF는 문제 있는
장치를 열 때 파이썬 예외가 아니라 **프로세스 자체를 죽이는 크래시**를 낸다.
`scripts/camera_check.py`가 진단용 조합을 자식 프로세스에서만 여는 것도
같은 이유다. 만약 이 자동 협상을 트래커 본 프로세스 안에서 직접 여러 조합을
돌려가며 시도했다면, 그중 하나가 크래시를 내는 순간 **헤드트래커 전체가
아무 예고 없이 죽는다** — "카메라가 안 켜진다"보다 훨씬 나쁜 결과다.

그래서 탐색은 전부 `camera_check.py --try-combo`를 자식 프로세스로 불러
수행한다(그 스크립트가 이미 만들고 검증해 둔 격리 방식 그대로 재사용) —
새로 만들지 않고 검증된 것을 가져다 쓴다.

비용 — 이미 설정대로 잘 열리는 절대다수의 경우엔 이 모듈이 아예 호출되지
않는다(camera_stream.init_camera가 기존 방식으로 여는 데 성공하면 그걸로
끝). 탐색은 **그게 실패했을 때만** 시작되고, 그 실패는 어차피 지금까지도
"카메라가 안 켜진다"는 완전한 실패였으므로 몇 초 더 걸려서라도 여는 쪽이
확실히 낫다.
"""
import json
import os
import subprocess
import sys

from src.utils.logger import get_logger

logger = get_logger("capture")

# camera_check.py --diagnose와 완전히 같은 목록 — 그 도구가 이미 실기로
# 검증해 둔 순서다(가능성 높은 조합부터). 하나를 늘리거나 줄이면 여기와
# camera_check.py 양쪽 다 손볼 것 — 지금은 의도적으로 중복시켜 뒀다(그
# 스크립트를 건드리지 않기 위해). 둘이 벌어지면 진단 결과와 자동 협상
# 결과가 달라질 수 있으니 바꿀 때 같이 바꿔야 한다
COMBOS = (
    ("msmf", "mjpg", 1280, 720),
    ("msmf", "auto", 1280, 720),
    ("msmf", "auto", 640, 480),
    ("dshow", "mjpg", 1280, 720),
    ("dshow", "auto", 1280, 720),
    ("dshow", "auto", 640, 480),
    ("auto", "auto", 640, 480),
)

# 조합 하나 시험에 주는 시간 한도 — ★반드시 camera_check.py의 PROBE_TIMEOUT_SEC
# (자식 프로세스 안 OPEN_TIMEOUT_SEC=15초 + 모델 로딩 + 표본 촬영 여유를 포함한
# 값, 40.0)과 같아야 한다. 처음엔 20.0으로 뒀다가 발견해서 고쳤다 — 정상
# 장치도 MSMF 오픈에 11초 넘게 걸리는 경우가 이 프로젝트에서 실측됐는데,
# 20초로는 그 여유(15초 오픈 + 표본 촬영)를 못 채워 **정상인데 느린 카메라를
# 실패로 오판**할 뻔했다. 더 짧게 잡으면 안 되는 값이다
PROBE_TIMEOUT_SEC = 40.0

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CAMERA_CHECK_SCRIPT = os.path.join(_ROOT_DIR, "scripts", "camera_check.py")


def _try_combo_isolated(config_path, device_id, backend, fourcc, width, height):
    """조합 하나를 자식 프로세스에서 시험한다. 죽어도 부모(이 트래커)는 산다.

    camera_check.py의 try_combo()와 완전히 같은 프로토콜 — 그 스크립트를
    `--try-combo`로 불러 결과를 `__RESULT__` 접두사가 붙은 JSON 한 줄로
    받는다. 실패·타임아웃·크래시 전부 None으로 통일해 호출자를 단순하게 한다.
    """
    if not os.path.exists(_CAMERA_CHECK_SCRIPT):
        logger.warning("카메라 자동 협상 불가 — camera_check.py를 찾을 수 없습니다: %s",
                       _CAMERA_CHECK_SCRIPT)
        return None
    cmd = [sys.executable, "-X", "utf8", _CAMERA_CHECK_SCRIPT,
           "--try-combo", str(device_id), "--config", config_path,
           "--backend", backend, "--fourcc", fourcc,
           "--width", str(width), "--height", str(height)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=PROBE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return None
    except OSError:   # noqa: 방어적 — 자식 프로세스 자체를 못 띄우는 극단적 상황
        logger.exception("카메라 협상 자식 프로세스 실행 실패")
        return None
    for line in (done.stdout or "").splitlines():
        if line.startswith("__RESULT__"):
            try:
                return json.loads(line[len("__RESULT__"):])
            except (ValueError, KeyError):
                return None
    return None


def find_working_combo(config_path, device_id):
    """이 장치가 실제로 화면을 주는 조합을 찾아 돌려준다. 없으면 None.

    목록 순서대로 하나씩 시도하고 **처음 성공하는 것**을 쓴다(diagnose_device
    처럼 전부 시도해 최고 FPS를 고르지 않는다) — 트래커 시작 지연을 줄이기
    위해서다. 이미 순서 자체가 "가능성 높은 조합부터"로 짜여 있다.
    """
    for backend, fourcc, width, height in COMBOS:
        logger.info("카메라(device_id=%s) 협상 시도 — %s/%s/%dx%d",
                    device_id, backend, fourcc, width, height)
        res = _try_combo_isolated(config_path, device_id, backend, fourcc, width, height)
        if res and res.get("opened") and res.get("frames", 0) > 0:
            return {"windows_backend": backend, "fourcc": fourcc,
                   "width_px": width, "height_px": height}
    return None
