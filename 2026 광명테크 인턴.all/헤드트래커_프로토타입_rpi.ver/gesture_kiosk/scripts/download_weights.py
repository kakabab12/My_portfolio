"""모델 준비 — 얼굴 랜드마크(FaceLandmarker) 모델 파일을 내려받는다.

2026-07-18 헤드트래커 전환: rtmlib(RTMPose) 캐시 대신 mediapipe FaceLandmarker의
.task 모델 파일 하나를 models/weights/에 받는다. 빌드 타임 1회 다운로드 —
런타임 네트워크 접근이 없어 내부망(폐쇄망) 반입과 호환된다
(캐시 방식이 아니라 고정 파일이라 make_offline_bundle.bat이 models/weights/를
그대로 반출하면 된다 — bundle_models/ 특수 처리 불필요).

사용법:
    python scripts/download_weights.py
"""
import os
import sys
import urllib.request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
# Apache-2.0 (MediaPipe) — blendshapes 포함 float16 빌드
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

logger = get_logger("scripts")


def main():
    config = load_config(DEFAULT_CONFIG_PATH)
    model_path = config["model"]["face_landmarker_path"]   # load_config가 절대경로로 정규화

    if os.path.exists(model_path):
        logger.info("얼굴 랜드마크 모델 이미 있음 — 건너뜀 (%s)", model_path)
        return

    logger.info("얼굴 랜드마크 모델 다운로드 중... (%s, 약 4MB)", MODEL_URL)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    logger.info("[DONE] 모델 저장 완료: %s", model_path)


if __name__ == "__main__":
    main()
