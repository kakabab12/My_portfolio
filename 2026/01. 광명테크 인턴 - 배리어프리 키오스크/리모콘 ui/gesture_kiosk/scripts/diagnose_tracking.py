"""커서가 안 움직일 때 **어디서 끊기는지** 집어낸다 (2026-08-31 신설).

왜 만들었나
-----------
"옆으로 살짝 기운 카메라에서는 커서가 아예 안 움직인다"는 실기 보고가 있었다.
그런데 같은 상황을 이미지 회전으로 흉내 내 봤더니(0~30도) 검출률 99~100%,
커서도 정상 이동이라 **재현이 안 됐다**. 즉 실제로 카메라를 기울였을 때만
생기는 무언가가 있고, 그건 추측으로 고칠 수 없다.

커서가 화면에서 움직이기까지는 여러 단계를 거친다. 어느 한 곳만 막혀도
증상은 똑같이 "커서가 안 움직인다"로 보인다:

    카메라 프레임 -> 얼굴 검출 -> 3차원 랜드마크 -> 중립 잡기
      -> 상대 회전 추정 -> 감도 나누기 -> 클램프 -> 평활 -> 화면

이 도구는 그 단계를 하나씩 세어서 **막힌 곳을 이름으로** 알려준다.

어떻게 쓰나
-----------
카메라를 문제가 생기는 그 각도로 둔 채, 평소처럼 서서 실행한다.

    py scripts/diagnose_tracking.py
    py scripts/diagnose_tracking.py --seconds 20

측정 중에는 **고개를 좌우로 크게, 위아래로도 크게** 움직인다. 커서가 실제로
안 움직이는 상태 그대로 재는 것이 목적이므로, 평소 쓰던 자세를 바꾸지 않는다.
"""
import argparse
import math
import os
import statistics
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.capture.camera_stream import CameraStream                        # noqa: E402
from src.inference.face_estimator import FaceEstimator, select_user_face  # noqa: E402
from src.inference.preprocessor import Preprocessor                       # noqa: E402
from src.postprocess.head_tracker import HeadTracker                      # noqa: E402
from src.utils.config_loader import load_config                           # noqa: E402
from src.utils.console import enable_utf8_output                          # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
FIRST_FRAME_TIMEOUT_SEC = 45.0

# 이만큼도 안 움직였으면 "커서가 안 움직인다"고 본다 (화면 비율)
CURSOR_STUCK_SPAN = 0.05

# 고개를 이만큼은 돌려 줘야 판정할 수 있다
MIN_HEAD_SWING_DEG = 10.0


def _span(values):
    return (max(values) - min(values)) if values else float("nan")


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="커서가 안 움직이는 원인 진단")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--tracker", choices=("eyebrow", "forehead", "head"),
                        default="eyebrow", help="어느 트래커 설정으로 진단할지")
    args = parser.parse_args()

    config = load_config(args.config)
    # 트래커 파일의 상수를 그대로 반영한다 — 실제로 도는 것과 같은 조건이어야 한다
    module = __import__(args.tracker)
    pointer = config["head_tracker"]["pointer"]
    pointer["orientation_mapping"] = module.ORIENTATION_MAPPING
    pointer["orientation_half_span_x_deg"] = module.ORIENTATION_HALF_SPAN_X_DEG
    pointer["orientation_half_span_y_deg"] = module.ORIENTATION_HALF_SPAN_Y_DEG

    preprocessor = Preprocessor(config)
    estimator = FaceEstimator(config)
    tracker = HeadTracker(config)
    mapper = tracker._cursor_mapper      # 진단 목적 — 내부 상태를 들여다본다
    camera = CameraStream(config, config_path=args.config).start()

    print("=" * 68)
    print(" 커서가 안 움직이는 원인 진단 (%s 설정)" % args.tracker)
    print("=" * 68)
    print("   카메라 준비 중...", end="", flush=True)
    import time
    seq = -1
    deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
    while True:
        try:
            _raw, seq = camera.capture_new_frame(seq)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                camera.stop()
                print("\n [막힌 곳] 카메라 — 프레임이 아예 안 옵니다.")
                return 2
            print(".", end="", flush=True)
    print(" 준비됨")
    print()
    print(" %.0f초간 **고개를 좌우로 크게, 위아래로도 크게** 움직여 주세요." % args.seconds)
    print()

    n_frame = n_face = n_depth = n_pose = 0
    n_offset_ok = n_offset_fail = n_calibrating = 0
    yaws, pitches, rolls = [], [], []
    raw_x, raw_y, cur_x, cur_y = [], [], [], []
    mapping_on_frames = 0

    started = time.monotonic()
    last_note = 0.0
    try:
        while time.monotonic() - started < args.seconds:
            try:
                raw, seq = camera.capture_new_frame(seq)
            except RuntimeError:
                continue
            n_frame += 1
            frame = preprocessor.preprocess_frame(raw, apply_crop=True)
            face = select_user_face(estimator.infer(frame))
            if face is not None:
                n_face += 1
                if getattr(face, "landmarks_3d", None) is not None:
                    n_depth += 1
                if getattr(face, "head_pose", None) is not None:
                    n_pose += 1
                    yaws.append(face.head_pose.yaw_deg)
                    pitches.append(face.head_pose.pitch_deg)
                    rolls.append(face.head_pose.roll_deg)
                # 상대 회전이 이 프레임에서 나오는지 (트래커와 같은 상태를 본다)
                if mapper._orientation is not None and mapper._orientation.is_ready:
                    offset = mapper._orientation.pointing_offset(face)
                    if offset is None:
                        n_offset_fail += 1
                    else:
                        n_offset_ok += 1
                        raw_x.append(offset[0])
                        raw_y.append(offset[1])

            if mapper._orientation_mapping:
                mapping_on_frames += 1
            if mapper._orientation_calibrating:
                n_calibrating += 1

            result = tracker.update(face)
            if result.cursor_x_ratio is not None:
                cur_x.append(result.cursor_x_ratio)
                cur_y.append(result.cursor_y_ratio)

            elapsed = time.monotonic() - started
            if elapsed - last_note >= 1.0:
                last_note = elapsed
                print("\r   %3.0f초 / %.0f초" % (elapsed, args.seconds), end="", flush=True)
    finally:
        camera.stop()

    print("\r" + " " * 30)
    print("=" * 68)
    print(" 단계별 통과 개수")
    print("=" * 68)
    print(f"   카메라 프레임          {n_frame}")
    print(f"   얼굴 검출              {n_face}  ({n_face / max(1, n_frame) * 100:.0f}%)")
    print(f"   3차원 랜드마크         {n_depth}")
    print(f"   머리 자세(부가정보)    {n_pose}")
    print(f"   상대 회전 추정 성공    {n_offset_ok}   실패 {n_offset_fail}")
    print(f"   상대 회전 매핑 켜짐    {mapping_on_frames} / {n_frame}")
    print(f"   중립 잡는 중이던 프레임 {n_calibrating}")
    print()
    print(" 움직인 폭")
    print(f"   머리 yaw(좌우)   {_span(yaws):7.2f}도    pitch(상하) {_span(pitches):7.2f}도")
    print(f"   머리 roll(갸웃)  {_span(rolls):7.2f}도   "
          f"(중앙 {statistics.median(rolls) if rolls else float('nan'):+.1f}도 "
          f"= 카메라가 기울어진 정도)")
    print(f"   원시 오프셋      가로 {_span(raw_x):7.4f}  세로 {_span(raw_y):7.4f}")
    print(f"   최종 커서        가로 {_span(cur_x):7.4f}  세로 {_span(cur_y):7.4f}")
    print()

    # ---- 판정 ----
    print("=" * 68)
    print(" 판정")
    print("=" * 68)
    problems = []
    if n_frame == 0:
        problems.append("카메라에서 프레임이 안 옵니다.")
    elif n_face / max(1, n_frame) < 0.7:
        problems.append("얼굴 검출률이 %.0f%%로 낮습니다 — 이 각도에서 얼굴이 잘 안 잡힙니다."
                        % (n_face / n_frame * 100))
    if n_face and n_depth < n_face * 0.9:
        problems.append("3차원 랜드마크가 안 옵니다 — 상대 회전 매핑이 못 돕니다.")
    if mapping_on_frames < n_frame * 0.9:
        problems.append("상대 회전 매핑이 꺼졌습니다 — 예전 2D 방식으로 되돌아갔습니다.")
    if n_offset_fail > n_offset_ok:
        problems.append("상대 회전 추정이 자주 실패합니다(%d/%d) — 중립 대비 너무 많이 "
                        "돌았거나 랜드마크가 불안정합니다."
                        % (n_offset_fail, n_offset_fail + n_offset_ok))
    if n_calibrating > n_frame * 0.5:
        problems.append("중립을 잡다가 계속 되돌아갑니다 — 얼굴이 자꾸 끊겨 "
                        "캘리브레이션이 안 끝납니다.")

    swing = max(_span(yaws) if yaws else 0.0, _span(pitches) if pitches else 0.0)
    if swing < MIN_HEAD_SWING_DEG:
        print("   고개를 %.0f도밖에 안 움직였습니다 — 더 크게 움직여 다시 재 주세요."
              % swing)
    elif _span(raw_x) < 0.02 and _span(raw_y) < 0.02:
        problems.append("고개는 %.0f도 움직였는데 오프셋이 거의 0입니다 — "
                        "회전 추정 자체가 머리 움직임을 못 따라갑니다." % swing)
    elif _span(cur_x) < CURSOR_STUCK_SPAN and _span(cur_y) < CURSOR_STUCK_SPAN:
        problems.append("오프셋은 나오는데 커서가 안 움직입니다 — "
                        "감도(half_span)나 클램프·평활 쪽 문제입니다.")

    if problems:
        for i, text in enumerate(problems, 1):
            print("   [%d] %s" % (i, text))
    else:
        print("   막힌 곳을 못 찾았습니다 — 이 조건에서는 커서가 정상 동작합니다.")
        print("   커서 가로 %.3f / 세로 %.3f 만큼 움직였습니다."
              % (_span(cur_x), _span(cur_y)))
    print()
    print(" 이 출력을 그대로 알려 주시면 어느 단계를 고쳐야 하는지 특정됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
