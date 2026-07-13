"""실시간 웹캠 데모 — MediaPipe Hand Landmarker + 기하 규칙으로 제스처를 인식한다.

python scripts/run_demo.py 로 실행. q로 종료.

on_gesture_detected()가 팀원 키오스크 프레임워크와 연동할 지점 -> 지금은 콘솔
출력만 하니, 실제 키오스크 동작(화면 전환 등)을 호출하도록 이 함수 내용만
바꿔 끼우면 된다. label은 move_left/move_right/select/go_home 등
configs/config.yaml의 gestures 설정에 따라 확정된 이벤트 이름이다.
"""
import os
import sys

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.pipeline.realtime_loop import run_pipeline
from src.utils.config_loader import load_config
from src.utils.logger import init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
WINDOW_NAME = "gesture_model"


def on_gesture_detected(label: str, confidence: float):
    print(f">>> GESTURE: {label} ({confidence:.2f})")


def main():
    config = load_config(DEFAULT_CONFIG_PATH)
    init_logging(config)

    runner = run_pipeline(
        config, on_event=lambda event: on_gesture_detected(event.class_name, event.conf)
    )
    try:
        while True:
            annotated, _event = runner.step()
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        runner.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
