"""제스처 인식 파이프라인 공통 설정.

새 제스처를 추가하려면 GESTURES에 이름만 추가하면 됨 -> collect_data.py / train.py / infer_demo.py 모두 자동 반영.
"""
from pathlib import Path

GESTURES = [
    "idle",         # 대기 / 제스처 아님 (오탐 방지를 위해 반드시 데이터를 모아야 하는 필수 클래스)
    "select",       # 선택
    "confirm",      # 확인
    "home",         # 처음화면
    "prev_menu",    # 이전메뉴
    "next_menu",    # 다음메뉴
    "call_staff",   # 직원호출
]

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
HOLISTIC_MODEL_PATH = MODELS_DIR / "holistic_landmarker.task"

# 한 제스처 구간의 길이(초)와, 그걸 리샘플링해서 맞출 고정 프레임 수.
# 녹화/추론 fps가 기기마다 달라도 항상 SEQ_LEN 프레임으로 맞춰서 모델 입력 shape을 통일함.
WINDOW_SECONDS = 1.5
SEQ_LEN = 30

# MediaPipe PoseLandmark 인덱스 중 팔 동작 인식에 쓸 지점 (양쪽 어깨/팔꿈치/손목)
ARM_LANDMARK_IDS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

RANDOM_SEED = 42
