"""로깅 설정 — 단일 포맷, 콘솔+파일 동시 기록."""
import logging
import os

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def init_logging(config):
    """logging 루트를 config(logging.level / save_dir)로 초기화한다."""
    logging_cfg = config["logging"]
    level_name = logging_cfg["level"]
    save_dir = logging_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # mediapipe/TFLite C++ 로그 억제 — mediapipe 임포트 전에 걸려야 해서 여기서 설정
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(save_dir, "gesture_engine.log"), encoding="utf-8")
    logging.basicConfig(
        level=getattr(logging, level_name),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[console_handler, file_handler],
        force=True,
    )


def get_logger(module_name):
    return logging.getLogger(module_name)
