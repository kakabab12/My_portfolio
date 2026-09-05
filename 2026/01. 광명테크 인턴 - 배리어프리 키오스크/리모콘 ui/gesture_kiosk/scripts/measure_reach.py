"""이 사람이 화면 어디까지 닿는지 재고, 도달 배율을 알려준다 (2026-09-05 신설).

왜 필요한가
-----------
커서는 "화면 절반 폭에 닿는 각도"를 기준으로 움직인다. 목을 그만큼 못 돌리는
사람은 화면 가장자리에 아예 못 닿는다:

    좌우로 돌릴 수 있는 최대   화면 폭에서 닿는 범위
          5.0도                     32.7%
          7.0도                     45.8%
         10.0도                     65.8%
         15.0도                    100.0%

마비가 있는 사용자가 정확히 이 경우다. 가장자리 버튼을 못 누르므로 정확도가
아니라 "쓸 수 있냐 없냐"의 문제다.

고치는 방법은 있다 — config의 orientation_reach_gain을 올리면 그만큼 적게
돌려도 끝까지 간다. 문제는 **그 값을 얼마로 할지 모른다**는 것이다.
사람마다 다르고, 추측해서 넣으면 목이 멀쩡한 사람은 커서가 예민해지고
떨림이 그 배율만큼 그대로 커진다.

그래서 **잰다.** 이 도구가 그 사람의 실제 가동범위와 떨림을 재서 권장값을
알려 준다. (자동으로 안 정하는 이유는 src/postprocess/reach_measure.py의
독스트링 참고 — 관측만으로는 "못 돌리는 사람"과 "안 돌린 사람"이 안 갈린다.)

어떻게 쓰나
-----------
평소 쓰는 자리에, 평소 자세로 앉거나 서서 실행한다.

    py scripts/measure_reach.py
    py scripts/measure_reach.py --seconds 60 --tracker head

측정 중에는 **평소처럼 화면 네 귀퉁이를 차례로 보려고 한다.** 억지로 더
돌리지 말 것 — 평소에 낼 수 있는 만큼만 내야 그 사람에게 맞는 값이 나온다.
중간중간 정면을 보고 잠깐 가만히 있는 구간이 있어야 떨림도 함께 잰다.
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.capture.camera_stream import CameraStream                        # noqa: E402
from src.inference.face_estimator import FaceEstimator, select_user_face  # noqa: E402
from src.inference.preprocessor import Preprocessor                       # noqa: E402
from src.postprocess.head_tracker import HeadTracker                      # noqa: E402
from src.postprocess.reach_measure import ReachMeasure                    # noqa: E402
from src.utils.config_loader import load_config                           # noqa: E402
from src.utils.console import enable_utf8_output                          # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
FIRST_FRAME_TIMEOUT_SEC = 10.0


def _fmt(value, unit="", digits=1):
    return "못 잼" if value is None else f"{value:.{digits}f}{unit}"


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="가동범위를 재서 도달 배율을 권한다")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--tracker", choices=("eyebrow", "forehead", "head"),
                        default="head", help="어느 트래커 설정으로 잴지")
    args = parser.parse_args()

    config = load_config(args.config)
    # 실제로 도는 것과 같은 조건이어야 한다 — 트래커 파일의 상수를 반영한다
    module = __import__(args.tracker)
    pointer = config["head_tracker"]["pointer"]
    pointer["orientation_mapping"] = module.ORIENTATION_MAPPING
    pointer["orientation_half_span_x_deg"] = module.ORIENTATION_HALF_SPAN_X_DEG
    pointer["orientation_half_span_y_deg"] = module.ORIENTATION_HALF_SPAN_Y_DEG
    # 배율은 1.0으로 두고 잰다 — 이미 걸린 배율 위에서 재면 값이 곱해져 버린다
    pointer["orientation_reach_gain"] = 1.0

    preprocessor = Preprocessor(config)
    estimator = FaceEstimator(config)
    tracker = HeadTracker(config)
    mapper = tracker._cursor_mapper       # 측정 목적 — 내부 상태를 들여다본다
    camera = CameraStream(config, config_path=args.config).start()

    print("=" * 68)
    print(" 가동범위 측정 (%s 설정)" % args.tracker)
    print("=" * 68)
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
    print(" %.0f초 동안 **평소처럼** 화면 네 귀퉁이를 차례로 보려고 해 주세요." % args.seconds)
    print(" 억지로 더 돌리지 마세요 — 평소에 낼 수 있는 만큼만 내야 맞는 값이 나옵니다.")
    print(" 중간중간 정면을 보고 잠깐 가만히 계시면 떨림도 함께 잽니다.")
    print()

    measure = None
    n_frame = n_sample = 0
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
            tracker.update(face)
            if face is None:
                continue
            orientation = mapper._orientation
            if orientation is None or not orientation.is_ready:
                continue
            if measure is None:
                measure = ReachMeasure(mapper._orientation_tan_x,
                                       mapper._orientation_tan_y)
            offset = orientation.pointing_offset(face)
            if offset is None:
                continue
            # 거리 보정을 먹인 뒤의 값으로 잰다 — 앞뒤로 움직인 것이
            # "가동범위가 바뀌었다"로 새지 않게 (head_tracker와 같은 순서)
            scale = getattr(orientation, "distance_ratio", 1.0) or 1.0
            measure.add(offset[0] * scale, offset[1] * scale)
            n_sample += 1

            elapsed = time.monotonic() - started
            if elapsed - last_note >= 1.0:
                last_note = elapsed
                print("\r   %3.0f초 / %.0f초" % (elapsed, args.seconds),
                      end="", flush=True)
    finally:
        camera.stop()

    print("\r" + " " * 30)
    if measure is None or n_sample < 30:
        print(" 표본이 모자랍니다 (얼굴 %d프레임 / 전체 %d프레임)." % (n_sample, n_frame))
        print(" 카메라에 얼굴이 잘 잡히는지 확인하고 다시 재 주세요.")
        return 1

    report = measure.report()
    print("=" * 68)
    print(" 잰 결과")
    print("=" * 68)
    for axis, label in (("x", "좌우"), ("y", "상하")):
        r = report[axis]
        print(f"   [{label}]")
        print(f"     화면 끝에 필요한 각도   {r['span_deg']:.1f}도")
        print(f"     실제로 돌린 각도        {_fmt(r['reach_deg'], '도')}"
              f"   (왕복 {r['peaks']}번)")
        reach = r["reach_ratio"]
        print("     닿는 범위               "
              + ("못 잼" if reach is None else f"화면의 {reach * 100:.0f}%"))
        jitter = r["jitter_ratio"]
        print("     정지 시 커서 떨림       "
              + ("못 잼" if jitter is None else f"화면의 {jitter * 100:.2f}%"))
        print()

    print("=" * 68)
    print(" 권장 설정 — configs/config.yaml 의 head_tracker.pointer")
    print("=" * 68)
    gain_x = report["x"]["recommended_gain"]
    gain_y = report["y"]["recommended_gain"]
    gains = [g for g in (gain_x, gain_y) if g is not None]
    if not gains:
        print("   아직 못 정합니다:")
        print("     좌우 — " + report["x"]["reason"])
        print("     상하 — " + report["y"]["reason"])
        print("   더 길게(--seconds 90) 다시 재 주세요.")
        return 1
    # 한 값으로 두 축을 함께 쓰므로 더 막힌 쪽에 맞춘다 — 덜 막힌 축이
    # 조금 예민해지는 것보다 못 닿는 축이 남는 쪽이 나쁘다
    gain = max(gains)
    print(f"   orientation_reach_gain: {gain:.2f}")
    print()
    print("   좌우 — " + report["x"]["reason"])
    print("   상하 — " + report["y"]["reason"])
    if gain <= 1.0:
        print()
        print("   (1.0은 '손대지 않음'입니다. 지금 설정으로 화면 전체에 닿습니다.)")
    else:
        print()
        print(f"   이 값을 넣으면 정지 시 떨림도 {gain:.2f}배가 됩니다.")
        print("   넣은 뒤 실제로 써 보고, 떨려서 못 쓰겠으면 조금 낮추세요.")
    tight_x = report["x"]["ceiling_tightness"]
    if tight_x is not None:
        print()
        print(f"   [참고] 최대치가 한 점에 몰린 정도 {tight_x:.2f}"
              " (1에 가까울수록 '늘 같은 데서 멈춤')")
        print("          이 값만으로는 '못 돌린 것'과 '안 돌린 것'이 안 갈립니다 —")
        print("          그래서 자동으로 정하지 않고 사람이 보고 정합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
