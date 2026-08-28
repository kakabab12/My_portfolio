"""MediaPipe Tasks 모델 파일(.task)을 models/ 에 내려받는다."""

import os
import urllib.request

BASE = "https://storage.googleapis.com/mediapipe-models"
FILES = {
    "pose_landmarker_lite.task":
        BASE + "/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task":
        BASE + "/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(out_dir, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst):
            print("skip  " + name)
            continue
        print("get   " + name + " ...", end="", flush=True)
        urllib.request.urlretrieve(url, dst)
        print(" %.1f MB" % (os.path.getsize(dst) / 1e6))


if __name__ == "__main__":
    main()
