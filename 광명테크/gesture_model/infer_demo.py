"""학습된 모델로 웹캠에서 실시간 제스처 인식을 테스트하는 데모.

python infer_demo.py 로 실행. q로 종료.

on_gesture_detected() 가 팀원 키오스크 프레임워크와 연동할 지점 -> 지금은 콘솔 출력만 하니,
실제 키오스크 동작(화면 전환 등)을 호출하도록 이 함수 내용만 바꿔 끼우면 됨.
"""
import json
from collections import Counter, deque

import cv2
import numpy as np
import torch

from config import MODELS_DIR, SEQ_LEN
from landmarks import HolisticExtractor, draw_landmarks_on_frame, normalize_frame, resample_sequence
from model import GestureGRU

CONFIDENCE_THRESHOLD = 0.80
CONSECUTIVE_HITS_REQUIRED = 4   # 최근 예측 중 이만큼이 같은 제스처여야 확정 (떨림 방지)
COOLDOWN_FRAMES = 20            # 한 번 인식되면 이 프레임 수만큼 재발화 방지
BUFFER_SECONDS = 1.8            # SEQ_LEN(WINDOW_SECONDS)보다 살짝 긴 버퍼를 유지하다 리샘플
ASSUMED_FPS = 20


def on_gesture_detected(label: str, confidence: float):
    print(f">>> GESTURE: {label} ({confidence:.2f})")


def load_model():
    with open(MODELS_DIR / "label_map.json", encoding="utf-8") as f:
        meta = json.load(f)
    model = GestureGRU(
        input_dim=meta["frame_dim"],
        num_classes=len(meta["gestures"]),
        hidden_dim=meta.get("hidden_dim", 64),
        num_layers=meta.get("num_layers", 2),
    )
    model.load_state_dict(torch.load(MODELS_DIR / "gesture_model.pt", map_location="cpu"))
    model.eval()
    return model, meta


def main():
    model, meta = load_model()
    gestures = meta["gestures"]
    idle_idx = gestures.index("idle") if "idle" in gestures else -1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("웹캠을 열 수 없습니다.")

    buf_len = int(BUFFER_SECONDS * ASSUMED_FPS)
    buffer = deque(maxlen=buf_len)
    recent_preds = deque(maxlen=CONSECUTIVE_HITS_REQUIRED)
    cooldown = 0

    with HolisticExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            raw_vec, results = extractor.process(frame)
            draw_landmarks_on_frame(frame, results)
            buffer.append(normalize_frame(raw_vec))

            label_text = ""
            if len(buffer) >= max(10, buf_len // 2):
                seq = resample_sequence(np.stack(buffer), SEQ_LEN, jitter=False)
                with torch.no_grad():
                    logits = model(torch.from_numpy(seq).unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(probs.argmax())
                pred_conf = float(probs[pred_idx])
                label_text = f"{gestures[pred_idx]} ({pred_conf:.2f})"

                recent_preds.append(pred_idx if pred_conf >= CONFIDENCE_THRESHOLD else idle_idx)

                if cooldown > 0:
                    cooldown -= 1
                elif len(recent_preds) == CONSECUTIVE_HITS_REQUIRED:
                    top_idx, hits = Counter(recent_preds).most_common(1)[0]
                    if top_idx != idle_idx and hits >= CONSECUTIVE_HITS_REQUIRED - 1:
                        on_gesture_detected(gestures[top_idx], pred_conf)
                        cooldown = COOLDOWN_FRAMES
                        recent_preds.clear()

            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("infer_demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
