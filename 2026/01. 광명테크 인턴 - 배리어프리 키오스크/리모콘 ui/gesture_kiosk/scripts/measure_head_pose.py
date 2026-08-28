"""머리 3차원 자세(yaw/pitch/roll)가 실제로 잘 나오는지 눈으로 확인한다 (2026-08-28 신설).

무엇을 보여주나
---------------
MediaPipe가 돌려주는 얼굴 변환행렬에서 뽑아낸 **머리 회전각**을 실시간으로
찍는다. 지금까지 커서는 화면에 투영된 2D 랜드마크 위치로 움직였는데, 그
방식은 코처럼 튀어나온 점의 원근 왜곡 때문에 좌우로만 돌려도 세로가 활처럼
휘었다(ARC_COMPENSATION이 2차식으로 사후 보정하던 문제).

회전각을 직접 쓰면 그 왜곡이 원리적으로 없다. 다만 **정말 그런지 먼저 재
본다** — 이 프로젝트가 8월 내내 지켜 온 순서다(어림값을 먼저 넣었다가 두 번
되돌린 기록이 forehead.py ARC_COMPENSATION 주석에 남아 있다).

어떻게 쓰나
-----------
1) 이 프로그램을 켠다 (트래커와 따로 돌아간다 — 카메라를 따로 연다)
2) 화면 안내대로 고개를 움직이며 숫자가 어떻게 변하는지 본다
3) 특히 확인할 것:

   · 좌우로만 돌릴 때  -> yaw 만 변하고 pitch 는 거의 고정인가?
     (2D 방식이라면 여기서 세로가 딸려 움직였다)
   · 앞뒤로 다가갔다 물러날 때 -> 각도가 그대로인가? (거리 무관성)
   · 카메라를 기울여 놓아도 -> 고개를 안 움직이면 각도가 안 변하는가?

4) q 를 누르면 끝난다. 마지막에 요약이 나온다.

실행:
    py scripts/measure_head_pose.py
    py scripts/measure_head_pose.py --seconds 30
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2   # noqa: E402

from src.capture.camera_stream import CameraStream       # noqa: E402
from src.inference.face_estimator import FaceEstimator, select_user_face   # noqa: E402
from src.inference.preprocessor import Preprocessor      # noqa: E402
from src.utils.config_loader import load_config          # noqa: E402
from src.utils.console import enable_utf8_output         # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

# 첫 프레임을 기다리는 한도 — camera_stream의 1초 한도와 별개다.
# 이 프로젝트는 정상 장치도 MSMF 오픈에 11초 넘게 걸리는 경우를 실측했고
# (2026-08-26), camera_negotiate의 조합 탐색은 조합당 40초까지 잡는다.
# 그 둘을 감당하도록 넉넉히 잡는다.
FIRST_FRAME_TIMEOUT_SEC = 45.0


def _fmt(v):
    return f"{v:+7.2f}" if v is not None else "   --  "


def main():
    # 한글·줄표가 섞인 안내문을 찍어도 죽지 않게 (console.enable_utf8_output 설명 참고)
    enable_utf8_output()

    parser = argparse.ArgumentParser(description="머리 3차원 자세 실측")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="최대 측정 시간 (기본 60초, q로 언제든 종료)")
    parser.add_argument("--no-window", action="store_true", help="카메라 창 없이 숫자만")
    args = parser.parse_args()

    config = load_config(args.config)
    preprocessor = Preprocessor(config)
    face_estimator = FaceEstimator(config)
    camera = CameraStream(config, config_path=args.config).start()

    print("=" * 66)
    print(" 머리 3차원 자세 실측")
    print("=" * 66)
    print(" 이렇게 움직여 보세요 — 숫자가 어떻게 변하는지 보는 게 목적입니다.")
    print()
    print("   1. 고개를 좌우로만 천천히      -> yaw 만 변해야 합니다")
    print("   2. 고개를 위아래로만 천천히    -> pitch 만 변해야 합니다")
    print("   3. 고개를 갸웃                 -> roll 만 변해야 합니다")
    print("   4. 앞뒤로 다가갔다 물러나기    -> 각도는 그대로, 거리(tz)만 변해야 합니다")
    print()
    print(" q 를 누르면 종료. 최대 %.0f초." % args.seconds)
    print("=" * 66)
    print()
    print("      yaw(좌우)  pitch(상하)  roll(갸웃)   거리(tz)")

    # ★첫 프레임 대기 — camera_stream의 대기 한도는 1초인데, 이 프로젝트는
    # 정상 장치도 MSMF 오픈에 11초 넘게 걸리는 경우를 실측했다(2026-08-26 기록).
    # 트래커 본체는 카메라를 켠 뒤 모델 로딩·스레드 준비를 하면서 자연히 시간을
    # 벌지만, 이 도구는 켜자마자 바로 프레임을 요구해서 첫 호출이 그대로
    # 터졌다(2026-08-28). 여기서 넉넉히 기다린다.
    print("   카메라 준비 중...", end="", flush=True)
    frame_seq = -1
    warmup_deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
    while True:
        try:
            _raw, frame_seq = camera.capture_new_frame(frame_seq)
            break
        except RuntimeError:
            if time.monotonic() >= warmup_deadline:
                camera.stop()
                print()
                print(f" 카메라가 {FIRST_FRAME_TIMEOUT_SEC:.0f}초 안에 프레임을 주지 않았습니다.")
                print(" 다른 프로그램이 카메라를 쓰고 있지 않은지 확인하고,")
                print(" 장치 번호는 scripts/camera_check.py 로 진단하세요.")
                return 2
            print(".", end="", flush=True)
    print(" 준비됨")
    print()

    started = time.monotonic()
    samples = []
    last_print = 0.0
    try:
        while time.monotonic() - started < args.seconds:
            raw, frame_seq = camera.capture_new_frame(frame_seq)
            frame = preprocessor.preprocess_frame(raw, apply_crop=True)
            face = select_user_face(face_estimator.infer(frame))

            pose = face.head_pose if face is not None else None
            now = time.monotonic()
            if pose is not None:
                samples.append((pose.yaw_deg, pose.pitch_deg, pose.roll_deg, pose.tz))

            if now - last_print >= 0.1:
                last_print = now
                if pose is None:
                    line = "   얼굴이 안 잡힙니다" if face is None else "   자세 정보 없음(옵션 확인 필요)"
                else:
                    line = (f"   {_fmt(pose.yaw_deg)}   {_fmt(pose.pitch_deg)}"
                            f"   {_fmt(pose.roll_deg)}   {_fmt(pose.tz)}")
                print(line + " " * 10, end="\r", flush=True)

            if not args.no_window:
                if pose is not None:
                    for i, (label, val) in enumerate((
                            ("yaw  (left-right)", pose.yaw_deg),
                            ("pitch(up-down)", pose.pitch_deg),
                            ("roll (tilt)", pose.roll_deg),
                            ("tz   (distance)", pose.tz))):
                        cv2.putText(frame, f"{label}: {val:+7.2f}", (12, 30 + i * 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "no face / no pose", (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("head pose", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        if not args.no_window:
            cv2.destroyAllWindows()

    print()
    print()
    print("=" * 66)
    print(" 결과")
    print("=" * 66)
    if not samples:
        print(" 표본이 없습니다 — 얼굴이 한 번도 안 잡혔거나 자세 정보가 안 왔습니다.")
        print(" face_estimator.py 의 output_facial_transformation_matrixes 설정을 확인하세요.")
        return 2

    print(f" 표본 {len(samples)}개")
    print()
    print("            최소       최대     움직인 폭")
    for idx, name in enumerate(("yaw  (좌우)", "pitch(상하)", "roll (갸웃)", "tz   (거리)")):
        vals = [s[idx] for s in samples]
        lo, hi = min(vals), max(vals)
        print(f" {name}  {lo:+8.2f}  {hi:+8.2f}   {hi - lo:8.2f}")
    print()
    print(" 판단 기준")
    print("   · 좌우로만 움직였는데 pitch 폭이 yaw 폭만큼 크다면 -> 축이 섞인 것")
    print("   · 앞뒤로만 움직였는데 yaw·pitch 폭이 크다면        -> 거리 의존이 남은 것")
    print("   · 위 두 가지가 없으면, 각도 기반 커서로 넘어갈 근거가 됩니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:   # noqa: 방어적 — 측정 도구가 죽어도 원인은 보여야 한다
        import traceback
        traceback.print_exc()
        input("\n계속하려면 Enter를 누르세요...")
        sys.exit(1)
