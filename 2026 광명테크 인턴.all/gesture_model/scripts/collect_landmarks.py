"""웹캠으로 "손등팔등" 여부별 랜드마크를 녹화해서 data/<label>/*.npy 로 저장하는 도구.

우리가 직접 학습시킨 손등팔등 분류기(scripts/train_classifier.py)의 학습 데이터를
만드는 첫 단계. 손 21개 랜드마크의 "픽셀 좌표 원본"을 그대로 저장한다 (정규화는
학습·추론 쪽에서 hand_pose_classifier.normalize_landmarks()로 공통 처리).

사용법:
    python scripts/collect_landmarks.py
    1. 촬영 대상 이름을 입력 (예: kim, lee - 여러 명이면 각자 한 번씩 실행)
    2. 화면 안내: SPACE=1초 녹화, n/p=다음/이전 클래스, q=종료
    3. 라벨은 2개뿐: 손등팔등(주먹 쥔 채 손등이 카메라를 향함) / none(그 외 전부)
    4. 라벨당 최소 15~20회 이상, 매번 손 위치·각도·거리·회전 정도를 조금씩
       바꿔가며 녹화 권장
    5. 'none'에는 손바닥이 보이는 경우뿐 아니라, 주먹은 아니지만 방향만 애매한
       경우, 손을 안 든 자연스러운 자세 등도 다양하게 담을 것 — 모델이 애매한
       상황을 손등팔등으로 오판하는 걸 줄이는 데 제일 중요한 클래스

실전(run_demo.py)이 항상 양손을 동시에 추적하는 것과 똑같이, 이 스크립트도
한 번에 최대 두 손을 인식한다 — 양손을 같이 들고 같은 모양을 만들면 한 번
녹화로 샘플이 2배로 쌓인다 (분류기는 손 하나하나를 독립적으로 판정하므로
왼손/오른손 구분 없이 그냥 각각 별도 샘플로 저장됨).
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
        num_hands=2,  # run_demo.py와 동일 — 양손을 동시에 들고 녹화하면 샘플이 2배로 쌓임
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
            for hand in hands_px:
                _draw_points(frame, hand)
            _overlay(frame, label, rep_counts[label], person, n_hands=len(hands_px))
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

    samples = []  # 각 항목 = 손 하나의 21랜드마크 (한 프레임에 양손이면 2개씩 쌓임)
    t0 = time.time()
    while time.time() - t0 < RECORD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hands_px, _ = extractor.detect(frame)
        disp = frame.copy()
        for hand in hands_px:
            samples.append(np.asarray(hand, dtype=np.float32))
            _draw_points(disp, hand)
        cv2.putText(disp, "REC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("collect_landmarks", disp)
        cv2.waitKey(1)

    if len(samples) < 3:
        print("손이 잘 안 잡혀서 저장하지 않았습니다. 손을 카메라에 더 가까이 대고 다시 시도하세요.")
        return

    out_dir = os.path.join(DATA_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    rep_counts[label] += 1
    fname = os.path.join(out_dir, f"{person}_{label}_{rep_counts[label]:03d}_{int(time.time())}.npy")
    np.save(fname, np.stack(samples))  # (n_samples, 21, 2) — 손 하나당 21랜드마크 원본 픽셀좌표
    print(f"저장: {fname} ({len(samples)} samples, 양손이면 프레임당 최대 2개)")


def _draw_points(frame, points_px):
    for x, y in points_px:
        cv2.circle(frame, (int(x), int(y)), 4, POINT_COLOR, -1)


def _overlay(frame, label, rep_count, person, n_hands):
    cv2.putText(frame, f"person: {person}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"label: {label}  (count: {rep_count})", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if n_hands == 0:
        cv2.putText(frame, "손이 안 잡힘", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"손 {n_hands}개 인식 중", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)
    cv2.putText(frame, "SPACE: record   n/p: switch label   q: quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


if __name__ == "__main__":
    main()
