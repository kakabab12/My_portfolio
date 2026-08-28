"""로깅 설정 — 기획서 4.7의 단일 포맷만 사용한다.

포맷 예: [2026-07-20 14:03:22] [INFO] [pipeline] gesture_event: right (conf=1.00)

2026-07-31(사용자 요청 — cmd 창은 이벤트 한 줄만): 콘솔은 console_level
(기본 WARNING)부터만 찍고, 파일은 level(INFO) 그대로 남긴다 — 실기 추적
기록은 유지하면서 콘솔 소음(시작 리포트·전환 로그·이벤트 중복 줄)을 없앤다.
"""
import logging
from logging.handlers import RotatingFileHandler
import os

LOG_MAX_BYTES = 5 * 1024 * 1024     # 한 파일 5MB
LOG_BACKUP_COUNT = 3               # 최대 4개 = 20MB로 상한
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def init_logging(config):
    """logging 루트를 config(logging.level / console_level / save_dir)로 초기화한다."""
    logging_cfg = config["logging"]
    level_name = logging_cfg["level"]
    # 키 없으면 종전(콘솔도 level) — 하위 호환
    console_level_name = logging_cfg.get("console_level", level_name)
    save_dir = logging_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # MediaPipe/TFLite C++ 로그 억제 시도 — mediapipe 임포트 전에 걸려야 해서
    # 모든 진입점이 거치는 여기서 설정한다. ※실측(2026-07-31): 모델 로딩 시
    # W0000 경고 몇 줄은 absl 직접 출력이라 환경변수로 안 잡히고 남는다 —
    # 시작 1회성이고 제스처 판정 출력에는 안 섞인다
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level_name))
    # ★로그 회전 (2026-08-25 신설). 무인 키오스크는 몇 달을 켜두는 물건인데
    # 그동안 로그 파일이 한없이 커지고 있었다. 특히 카메라가 빠지는 등으로
    # 오류가 반복되면 초당 수십 건이 쌓여 디스크를 채우고, 그러면 키오스크가
    # 통째로 멈춘다. 파일 하나가 이 크기를 넘으면 다음 파일로 넘기고, 오래된
    # 것부터 지운다 — 전체가 이 크기 x (backupCount+1)을 절대 못 넘는다.
    file_handler = RotatingFileHandler(
        os.path.join(save_dir, "gesture_kiosk.log"),
        maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    logging.basicConfig(
        level=getattr(logging, level_name),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[console_handler, file_handler],
        force=True,
    )


def get_logger(module_name):
    """모듈 이름표가 붙은 로거를 돌려준다 (예: get_logger("pipeline"))."""
    return logging.getLogger(module_name)
