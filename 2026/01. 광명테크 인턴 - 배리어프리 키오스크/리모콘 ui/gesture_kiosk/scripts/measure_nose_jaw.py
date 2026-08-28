"""입을 벌릴 때 코 랜드마크가 얼마나 따라 움직이는지 실측한다.

왜 필요한가: 커서 기준점을 코끝 한 점에서 코 아래쪽 9점 평균으로 바꿔 떨림을
줄였는데(어두울 때 유리), 그 묶음에 코기둥·콧방울처럼 **입 바로 위에 있는 점들**이
들어 있다. 턱이 벌어지면 얼굴 메시가 통째로 다시 맞춰지면서 그 점들이 딸려
내려갈 수 있고, 그러면 "입 벌려 클릭할 때 커서가 흔들린다"가 된다.

어느 점이 실제로 딸려 움직이는지는 얼굴마다 다를 수 있으니 추측하지 말고 잰다.
점마다 "입 벌림 정도와 위치 변화의 상관"을 구해서, 입에 둔감한 점만 골라 쓴다.

같은 실행에서 **드래그가 왜 잘 안 걸리는지**도 같이 잰다. 드래그는 입을 1.2초
이상 계속 벌리고 있어야 시작되는데, 그동안 턱이 잠깐이라도 닫힘 문턱 아래로
내려가면 드래그 대신 클릭으로 처리된다. 그게 실제로 얼마나 자주 일어나는지
숫자로 봐야 문턱을 얼마로 고칠지 정할 수 있다.

사용법:
    py -3.11 scripts/measure_nose_jaw.py
    -> ① 입을 천천히 크게 벌렸다 닫았다 5회쯤
       ② 이어서 **드래그하듯 입을 벌린 채로 2초 유지**를 5회쯤 (고개도 조금 움직이며)
       q로 종료.
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.camera_stream import CameraStream
from src.inference.face_estimator import (
    FaceEstimator, LMK_LEFT_EYE_OUTER, LMK_NOSE_CLUSTER, LMK_RIGHT_EYE_OUTER)
from src.inference.preprocessor import Preprocessor
from src.utils.config_loader import load_config

NAMES = {1: "코끝", 2: "코기둥위", 4: "코끝위", 19: "코기둥", 94: "코기둥아래",
         97: "콧방울안L", 326: "콧방울안R", 98: "콧방울밖L", 327: "콧방울밖R",
         5: "콧대아래", 6: "콧대중간", 195: "콧대위", 197: "콧대맨위", 168: "미간아래"}
# 후보를 넓게 본다 — 지금 쓰는 9점 + 콧대(입에서 먼 쪽) 5점
CANDIDATES = tuple(LMK_NOSE_CLUSTER) + (5, 6, 195, 197, 168)


def main():
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "config.yaml"))
    config["face_tracker"]["max_num_faces"] = 1
    camera = CameraStream(config).start()
    pre = Preprocessor(config)
    est = FaceEstimator(config)

    jaws, offs, stamps = [], [], []
    print("입을 천천히 크게 벌렸다 닫았다 10회쯤 반복하세요. q로 종료.")
    seq = -1
    try:
        while True:
            frame, seq = camera.capture_new_frame(seq)
            frame = pre.preprocess_frame(frame, apply_crop=True)
            faces = est.infer(frame)
            if not faces:
                continue
            f = faces[0]
            jaw = f.blendshape("jawOpen")
            eL = np.array(f.landmark_px(LMK_LEFT_EYE_OUTER), np.float32)
            eR = np.array(f.landmark_px(LMK_RIGHT_EYE_OUTER), np.float32)
            mid = (eL + eR) / 2.0
            scale = float(np.linalg.norm(eR - eL))
            if scale < 1e-3:
                continue
            # 얼굴 좌표계 기준 위치 — 머리가 움직여도 값이 안 흔들리게
            row = [( np.array(f.landmark_px(i), np.float32) - mid) / scale for i in CANDIDATES]
            jaws.append(jaw)
            offs.append(np.array(row, np.float32))
            stamps.append(time.monotonic())

            vis = frame.copy()
            cv2.putText(vis, f"jawOpen {jaw:.2f}   samples {len(jaws)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for i in CANDIDATES:
                x, y = f.landmark_px(i)
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)
            cv2.imshow("nose vs jaw", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.stop()
        est.close()
        cv2.destroyAllWindows()

    jaws = np.array(jaws, np.float32)
    offs = np.array(offs, np.float32)          # (N, 점수, 2)
    stamps = np.array(stamps, np.float64)
    if len(jaws) < 60 or jaws.max() - jaws.min() < 0.15:
        print(f"\n표본이 부족하거나 입을 충분히 안 벌렸습니다 "
              f"(표본 {len(jaws)}개, jawOpen 범위 {jaws.min():.2f}~{jaws.max():.2f}). 다시 해주세요.")
        return 1

    closed = jaws < np.percentile(jaws, 25)
    opened = jaws > np.percentile(jaws, 75)
    print(f"\n표본 {len(jaws)}개 (닫힘 {closed.sum()} / 벌림 {opened.sum()}), "
          f"jawOpen {jaws.min():.2f}~{jaws.max():.2f}\n")
    print(f"{'점':>5s} {'이름':>10s} {'입벌림에 딸려간 거리':>20s} {'제자리 떨림':>12s} {'판정':>8s}")
    print("-" * 66)
    rows = []
    for k, idx in enumerate(CANDIDATES):
        shift = float(np.linalg.norm(offs[opened, k].mean(0) - offs[closed, k].mean(0)))
        noise = float(np.linalg.norm(offs[closed, k].std(0)))
        rows.append((idx, shift, noise))
    med = float(np.median([r[1] for r in rows]))
    for idx, shift, noise in rows:
        verdict = "안정" if shift <= med else "입에 딸림"
        print(f"{idx:5d} {NAMES.get(idx,'?'):>10s} {shift:19.4f} {noise:11.4f} {verdict:>8s}")
    stable = [r[0] for r in sorted(rows, key=lambda r: r[1])[:6]]
    print(f"\n입에 가장 둔감한 6점: {tuple(stable)}")
    print("이 값을 face_estimator.py의 LMK_NOSE_CLUSTER로 쓰면 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
