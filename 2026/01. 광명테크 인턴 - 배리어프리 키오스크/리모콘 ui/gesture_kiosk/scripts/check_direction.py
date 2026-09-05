"""커서가 좌우로 맞게 가는지 **테스트에서 측정한다** (2026-09-05 신설).

왜 필요한가
-----------
"좌우가 반대로 움직인다"는 보고를 가상 카메라로는 재현하지 못했다.
시뮬레이션에서는 배포 기본값(camera.mirror=true)에서 방향이 맞다고 나온다.
그러면 남는 가능성은 두 가지고, 둘 다 **실제 카메라 없이는 가릴 수 없다.**

  1) 이 카메라가 이미 좌우 반전된 영상을 내보낸다. 그러면 config의
     mirror=true가 그걸 되돌려 놓아 반전이 사라진다. (노트북 내장 웹캠과
     일부 UVC 카메라가 그렇게 동작한다.)
  2) 가상 카메라의 거울 모델이 실제 장치와 부호가 반대다. 그러면 지금까지의
     시뮬레이션 검증이 반대쪽을 통과시켜 온 것이다.

이 도구는 **추측하지 않고 측정한다.** 고개를 한쪽으로 돌리는 동안 두 가지를
같이 기록해서, 어느 쪽이 어긋났는지 이름으로 알려준다.

  · 얼굴이 화면(카메라 영상)에서 어느 쪽으로 갔나  <- 영상이 거울인지 알려준다
  · 커서 오프셋이 어느 쪽으로 갔나                 <- 사용자가 겪는 결과

어떻게 쓰나
-----------
평소 쓰는 자리에서 실행하고, 안내가 나오면 **고개를 오른쪽으로** 천천히
크게 돌렸다가 정면으로 돌아온다. 손은 쓰지 않는다.

    py scripts/check_direction.py
    py scripts/check_direction.py --tracker forehead

"오른쪽"은 **사용자 본인 기준**이다(오른손 쪽). 화면 기준이 아니다.
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.capture.camera_stream import CameraStream                        # noqa: E402
from src.inference.face_estimator import (LMK_LEFT_EYE_OUTER,             # noqa: E402
                                          LMK_RIGHT_EYE_OUTER,
                                          FaceEstimator, select_user_face)
from src.inference.preprocessor import Preprocessor                       # noqa: E402
from src.postprocess.head_tracker import HeadTracker                      # noqa: E402
from src.utils.config_loader import load_config                           # noqa: E402
from src.utils.console import enable_utf8_output                          # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
FIRST_FRAME_TIMEOUT_SEC = 10.0
# 이만큼은 움직여야 "돌렸다"로 본다 — 잡음으로 방향을 판정하지 않는다
MIN_FACE_SHIFT_PX = 8.0
MIN_CURSOR_SHIFT = 0.03


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="커서 좌우 방향을 테스트에서 측정")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--tracker", choices=("eyebrow", "forehead", "head"),
                        default="head")
    args = parser.parse_args()

    config = load_config(args.config)
    module = __import__(args.tracker)
    pointer = config["head_tracker"]["pointer"]
    pointer["orientation_mapping"] = module.ORIENTATION_MAPPING
    pointer["orientation_half_span_x_deg"] = module.ORIENTATION_HALF_SPAN_X_DEG
    pointer["orientation_half_span_y_deg"] = module.ORIENTATION_HALF_SPAN_Y_DEG
    mirror_cfg = config["camera"]["mirror"]

    preprocessor = Preprocessor(config)
    estimator = FaceEstimator(config)
    tracker = HeadTracker(config)
    camera = CameraStream(config, config_path=args.config).start()

    print("=" * 68)
    print(" 커서 좌우 방향 측정 (%s 설정)" % args.tracker)
    print("=" * 68)
    print("   설정된 거울 모드(camera.mirror) : %s" % ("켜짐" if mirror_cfg else "꺼짐"))
    print("   카메라 준비 중...", end="", flush=True)
    seq = -1
    deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
    while True:
        try:
            _raw, seq = camera.capture_new_frame(seq)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                camera.stop()
                print("\n 카메라에서 프레임이 안 옵니다.")
                return 2
            print(".", end="", flush=True)
    print(" 준비됨")
    print()
    print(" 먼저 3초간 **정면**을 봐 주세요 (기준을 잡습니다).")

    base_face_x = base_cursor = None
    face_x_at_peak = cursor_at_peak = None
    peak = 0.0
    started = time.monotonic()
    told = False
    try:
        while time.monotonic() - started < args.seconds + 3.0:
            try:
                raw, seq = camera.capture_new_frame(seq)
            except RuntimeError:
                continue
            frame = preprocessor.preprocess_frame(raw, apply_crop=True)
            face = select_user_face(estimator.infer(frame))
            result = tracker.update(face)
            elapsed = time.monotonic() - started
            if face is None or result.cursor_x_ratio is None:
                continue
            # 얼굴이 영상에서 어디 있나 — 두 눈 바깥쪽의 가운데
            left = face.landmark_px(LMK_LEFT_EYE_OUTER)
            right = face.landmark_px(LMK_RIGHT_EYE_OUTER)
            face_x = 0.5 * (left[0] + right[0])

            if elapsed < 3.0:
                base_face_x, base_cursor = face_x, result.cursor_x_ratio
                continue
            if not told:
                told = True
                print()
                print(" 이제 **고개를 본인 기준 오른쪽(오른손 쪽)으로** 천천히 크게")
                print(" 돌렸다가 정면으로 돌아와 주세요. %.0f초 동안 기록합니다." % args.seconds)
                print()
            if base_face_x is None:
                continue
            shift = result.cursor_x_ratio - base_cursor
            if abs(shift) > abs(peak):
                peak = shift
                face_x_at_peak = face_x
                cursor_at_peak = result.cursor_x_ratio
            print("\r   기록 중... 커서 이동 %+.3f" % peak, end="", flush=True)
    finally:
        camera.stop()

    print("\r" + " " * 40)
    if base_face_x is None or face_x_at_peak is None:
        print(" 얼굴이 충분히 안 잡혔습니다. 조명을 밝게 하고 다시 해 주세요.")
        return 1

    d_face = face_x_at_peak - base_face_x
    d_cursor = cursor_at_peak - base_cursor
    if abs(d_face) < MIN_FACE_SHIFT_PX or abs(d_cursor) < MIN_CURSOR_SHIFT:
        print(" 움직임이 너무 작아 방향을 판정할 수 없습니다"
              " (얼굴 %+.1fpx, 커서 %+.3f)." % (d_face, d_cursor))
        print(" 고개를 더 크게 돌려서 다시 해 주세요.")
        return 1

    print("=" * 68)
    print(" 측정 결과")
    print("=" * 68)
    print("   고개를 오른쪽으로 돌렸을 때")
    print("     영상 속 얼굴은  %s (%+.1f px)"
          % ("오른쪽으로" if d_face > 0 else "왼쪽으로", d_face))
    print("     커서는          %s (%+.3f)"
          % ("오른쪽으로" if d_cursor > 0 else "왼쪽으로", d_cursor))
    print()

    frame_is_mirrored = d_face > 0        # 거울 영상이면 본인의 오른쪽이 영상 오른쪽
    cursor_ok = d_cursor > 0              # 커서는 본인 오른쪽 = 화면 오른쪽으로 가야 한다

    print("   영상이 거울인가 : %s" % ("예" if frame_is_mirrored else "아니오"))
    print("   설정값과 일치하나 : %s"
          % ("예" if frame_is_mirrored == bool(mirror_cfg) else
             "★아니오 — 카메라가 이미 좌우 반전된 영상을 주고 있습니다★"))
    print()
    if cursor_ok:
        print("   >> 커서 방향은 맞습니다.")
        return 0

    print("   >> 커서가 반대로 갑니다.")
    print()
    if frame_is_mirrored != bool(mirror_cfg):
        print("   원인: 이 카메라가 이미 좌우 반전된 영상을 내보냅니다.")
        print("   조치: configs/config.yaml 의")
        print("           camera.mirror: %s" % ("false" if mirror_cfg else "true"))
        print("         로 바꾸고 다시 실행하세요. 손 쓸기 방향도 같이 맞습니다.")
    else:
        print("   설정과 영상은 맞는데 커서만 반대입니다 — 부호 로직 쪽입니다.")
        print("   (손 쓸기가 정상이라면 이쪽입니다. camera.mirror는 건드리지 마세요 —")
        print("    바꾸면 손 제스처가 깨집니다.)")
        print()
        print("   임시 조치: configs/config.yaml 의")
        print("           orientation_invert_x: true")
        print("         로 바꾸면 머리 커서 가로만 뒤집힙니다.")
        print()
        print("   그리고 아래 값을 알려 주세요 — 부호를 제대로 고치는 데 필요합니다:")
        print("     mirror=%s  얼굴 %+.1fpx  커서 %+.3f" % (mirror_cfg, d_face, d_cursor))
    return 1


if __name__ == "__main__":
    sys.exit(main())
