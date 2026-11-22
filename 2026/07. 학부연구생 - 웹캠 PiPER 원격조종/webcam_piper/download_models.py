"""MediaPipe 손 모델(hand_landmarker.task, 약 7.5 MB)을 models/ 에 내려받는다. 처음 한 번만 실행."""

import os
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
       "hand_landmarker.task")


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "hand_landmarker.task")
    if os.path.exists(dst):
        print("이미 있음:", dst)
        return
    print("내려받는 중 ...", end="", flush=True)
    urllib.request.urlretrieve(URL, dst)
    print(f" {os.path.getsize(dst) / 1e6:.1f} MB → {dst}")


if __name__ == "__main__":
    main()
