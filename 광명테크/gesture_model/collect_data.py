"""웹캠으로 제스처별 랜드마크 시퀀스를 녹화해서 data/<gesture>/*.npy 로 저장하는 도구.

사용법:
    python collect_data.py
    1. 촬영 대상 이름을 입력 (예: kim, lee, park - 3명이면 각자 한 번씩 실행)
    2. 화면 안내: SPACE=녹화, n/p=제스처 전환, q=종료
    3. 제스처당(특히 idle) 최소 15~20회 이상, 매번 위치/각도/속도를 조금씩 바꿔가며 녹화 권장
    4. idle에는 가만히 서있기, 자연스럽게 움직이기, 얼굴 만지기 등 "제스처 아닌 동작"을 다양하게 담을 것
       (이게 있어야 실제 키오스크에서 아무 동작에나 반응하는 오탐을 줄일 수 있음)
"""
import time

import cv2
import numpy as np

from config import GESTURES, DATA_DIR, WINDOW_SECONDS
from landmarks import HolisticExtractor, draw_landmarks_on_frame

COUNTDOWN_SECONDS = 3
RECORD_SECONDS = WINDOW_SECONDS


def main():
    person = input("촬영 대상 이름을 입력하세요 (예: kim): ").strip() or "unknown"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("웹캠을 열 수 없습니다.")

    gesture_idx = 0
    rep_counts = {g: _existing_rep_count(g, person) for g in GESTURES}

    with HolisticExtractor() as extractor:
        while True:
            gesture = GESTURES[gesture_idx]
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            _, results = extractor.process(frame)
            draw_landmarks_on_frame(frame, results)
            _overlay(frame, gesture, rep_counts[gesture], person)
            cv2.imshow("collect_data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                gesture_idx = (gesture_idx + 1) % len(GESTURES)
            elif key == ord("p"):
                gesture_idx = (gesture_idx - 1) % len(GESTURES)
            elif key == ord(" "):
                _record_one(cap, extractor, gesture, person, rep_counts)

    cap.release()
    cv2.destroyAllWindows()


def _existing_rep_count(gesture, person):
    gdir = DATA_DIR / gesture
    if not gdir.exists():
        return 0
    return len(list(gdir.glob(f"{person}_{gesture}_*.npy")))


def _record_one(cap, extractor, gesture, person, rep_counts):
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        ok, frame = cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, str(remaining), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
        cv2.imshow("collect_data", frame)
        cv2.waitKey(1)
        time.sleep(1)

    frames = []
    t0 = time.time()
    while time.time() - t0 < RECORD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        vec, results = extractor.process(frame)
        frames.append(vec)
        disp = frame.copy()
        draw_landmarks_on_frame(disp, results)
        cv2.putText(disp, "REC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("collect_data", disp)
        cv2.waitKey(1)

    if len(frames) < 3:
        print("녹화된 프레임이 너무 적어 저장하지 않았습니다. 다시 시도하세요.")
        return

    out_dir = DATA_DIR / gesture
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_counts[gesture] += 1
    fname = out_dir / f"{person}_{gesture}_{rep_counts[gesture]:03d}_{int(time.time())}.npy"
    np.save(fname, np.stack(frames))
    print(f"저장: {fname} ({len(frames)} frames)")


def _overlay(frame, gesture, rep_count, person):
    cv2.putText(frame, f"person: {person}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"gesture: {gesture}  (count: {rep_count})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE: record   n/p: switch gesture   q: quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


if __name__ == "__main__":
    main()
