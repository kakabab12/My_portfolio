"""웹캠으로 손모양별 랜드마크를 녹화해서 data/<label>/*.npy 로 저장하는 도구.

우리가 직접 학습시킨 손모양 분류기(scripts/train_classifier.py)의 학습 데이터를
만드는 첫 단계. 손 21개 랜드마크의 "픽셀 좌표 원본"을 그대로 저장한다 (정규화는
학습·추론 쪽에서 hand_pose_classifier.normalize_landmarks()로 공통 처리).

사용법:
    python scripts/collect_landmarks.py
    1. 촬영 대상 이름을 입력 (예: kim, lee - 여러 명이면 각자 한 번씩 실행)
    2. 화면 안내: SPACE=1초 녹화, n/p=다음/이전 클래스, q=종료
    3. 클래스당 최소 15~20회 이상, 매번 손 위치/각도/거리를 조금씩 바꿔가며 녹화 권장
    4. 'none' 클래스에는 5개 제스처 어디에도 안 속하는 자연스러운 손 모양(반쯤 쥔
       손, 브이 사인, 손 내리는 중 등)을 다양하게 담을 것 — 모델이 애매한 손모양을
       억지로 5개 중 하나로 분류하는 오탐을 줄이는 데 제일 중요한 클래스
"""
import os
import sys
import time

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.inference.hand_landmark_extractor import HandLandmarkExtractor  # noqa: E402
from src.inference.hand_pose_classifier import LABELS  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
DATA_DIR = os.path.join(ROOT_DIR, "data")
COUNTDOWN_SECONDS = 2
RECORD_SECONDS = 1.0
POINT_COLOR = (0, 220, 120)


def main():
    config = load_config(DEFAULT_CONFIG_PATH)
    person = input("촬영 대상 이름을 입력하세요 (예: kim): ").strip() or "unknown"

    cap = cv2.VideoCapture(config["camera"]["device_id"])
    if not cap.isOpened():
        raise SystemExit("웹캠을 열 수 없습니다.")

    mp_cfg = config["model"]["mediapipe"]
    extractor = HandLandmarkExtractor(
        mp_cfg["hand_landmarker_path"],
        num_hands=1,  # 데이터 녹화는 한 손씩, 한 프레임에 하나만
        min_detection_confidence=config["detect"]["conf_threshold"],
        min_tracking_confidence=mp_cfg["min_tracking_confidence"],
    )

    label_idx = 0
    rep_counts = {label: _existing_rep_count(label, person) for label in LABELS}

    try:
        while True:
            label = LABELS[label_idx]
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hands_px, _ = extractor.detect(frame)
            if hands_px:
                _draw_points(frame, hands_px[0])
            _overlay(frame, label, rep_counts[label], person, has_hand=bool(hands_px))
            cv2.imshow("collect_landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                label_idx = (label_idx + 1) % len(LABELS)
            elif key == ord("p"):
                label_idx = (label_idx - 1) % len(LABELS)
            elif key == ord(" "):
                _record_one(cap, extractor, label, person, rep_counts)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


def _existing_rep_count(label, person):
    ldir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(ldir):
        return 0
    return len([f for f in os.listdir(ldir) if f.startswith(f"{person}_{label}_")])


def _record_one(cap, extractor, label, person, rep_counts):
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        ok, frame = cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, str(remaining), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
        cv2.imshow("collect_landmarks", frame)
        cv2.waitKey(1)
        time.sleep(1)

    frames = []
    t0 = time.time()
    while time.time() - t0 < RECORD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hands_px, _ = extractor.detect(frame)
        disp = frame.copy()
        if hands_px:
            frames.append(np.asarray(hands_px[0], dtype=np.float32))
            _draw_points(disp, hands_px[0])
        cv2.putText(disp, "REC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("collect_landmarks", disp)
        cv2.waitKey(1)

    if len(frames) < 3:
        print("손이 잘 안 잡혀서 저장하지 않았습니다. 손을 카메라에 더 가까이 대고 다시 시도하세요.")
        return

    out_dir = os.path.join(DATA_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    rep_counts[label] += 1
    fname = os.path.join(out_dir, f"{person}_{label}_{rep_counts[label]:03d}_{int(time.time())}.npy")
    np.save(fname, np.stack(frames))  # (n_frames, 21, 2) — 프레임별 21랜드마크 원본 픽셀좌표
    print(f"저장: {fname} ({len(frames)} frames)")


def _draw_points(frame, points_px):
    for x, y in points_px:
        cv2.circle(frame, (int(x), int(y)), 4, POINT_COLOR, -1)


def _overlay(frame, label, rep_count, person, has_hand):
    cv2.putText(frame, f"person: {person}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"label: {label}  (count: {rep_count})", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if not has_hand:
        cv2.putText(frame, "손이 안 잡힘", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "SPACE: record   n/p: switch label   q: quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


if __name__ == "__main__":
    main()
