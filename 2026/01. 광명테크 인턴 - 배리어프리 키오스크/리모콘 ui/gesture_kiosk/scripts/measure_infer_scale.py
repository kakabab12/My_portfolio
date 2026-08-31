"""추론 입력 축소 배율(infer_scale_ratio)의 속도 vs 정확도를 실측한다 (2026-08-31 신설).

왜 만들었나
-----------
헤드트래커의 프레임 예산을 8/31에 처음 단계별로 재 봤더니 얼굴 추론이 압도적
병목이었다 (405x720 입력, 실측):

    카메라 대기   중앙 3.0ms   p95  13ms   최대  32ms
    전처리        중앙 3.4ms   p95  10ms   최대  28ms
    얼굴 추론     중앙 29.8ms  p95 121ms   최대 383ms   <- 여기
    ------------------------------------------------
    실효 27.7 FPS

특히 **꼬리가 길다**. 중앙값은 30ms인데 p95가 121ms, 최대가 383ms다. 이 꼬리가
사용자가 보고한 "커서가 버벅거린다"의 정체다 — 평균이 아니라 가끔 튀는 프레임이
끊김으로 느껴진다.

config.yaml의 infer_scale_ratio는 추론 입력만 줄이는 손잡이이고 기본이 1.0이다.
주석에는 "더 약한 CPU에서 FPS 부족하면 0.5~0.75"라고만 적혀 있고 **얼마나
빨라지고 얼마나 부정확해지는지는 재 본 적이 없다**. 그래서 잰다.

어떻게 재나
-----------
배율마다 따로 카메라를 돌리면 사람이 그 사이에 움직여서 비교가 오염된다.
그래서 **프레임을 한 번만 모아 두고, 같은 프레임에 모든 배율을 돌린다**.
입력이 완전히 같으므로 차이는 순수하게 배율 때문이다.

정확도는 배율 1.0(원본)을 기준으로 잰다:
  · 랜드마크가 몇 px 어긋나는가 — 안구간거리로 나눠 얼굴 크기와 무관하게
  · 머리 회전각(yaw/pitch)이 몇 도 어긋나는가
  · 커서 좌표로 환산하면 화면의 몇 %인가  <- 실사용에서 체감되는 값

마지막 항목이 판단 기준이다. 화면 1%면 1920px 기준 19px이라 눈에 띄지만,
0.1%면 2px이라 손떨림보다 작다.

어떻게 쓰나
-----------
    py scripts/measure_infer_scale.py
    py scripts/measure_infer_scale.py --frames 200 --scales 1.0,0.75,0.5

측정 중에는 **평소 쓰는 자세로 고개를 천천히 좌우·상하로 움직이세요**.
가만히 있으면 쉬운 프레임만 모여서 실사용보다 좋게 나옵니다.
"""
import argparse
import copy
import math
import os
import statistics
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.capture.camera_stream import CameraStream                        # noqa: E402
from src.inference.face_estimator import (                                # noqa: E402
    FaceEstimator, LMK_LEFT_EYE_OUTER, LMK_RIGHT_EYE_OUTER, select_user_face,
)
from src.inference.preprocessor import Preprocessor                       # noqa: E402
from src.utils.config_loader import load_config                           # noqa: E402
from src.utils.console import enable_utf8_output                          # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
FIRST_FRAME_TIMEOUT_SEC = 45.0

# 커서 환산에 쓰는 감도 — eyebrow.py의 실기 확정값(팀장님 확인).
# "랜드마크가 이만큼 어긋나면 커서는 화면의 몇 %가 어긋나는가"를 내기 위한 것
SENS_X, SENS_Y = 2.05, 6.0


def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def _capture_frames(config, config_path, count, seconds):
    """프레임을 미리 모아 둔다 — 모든 배율이 **같은 입력**을 보게 하려고."""
    preprocessor = Preprocessor(config)
    camera = CameraStream(config, config_path=config_path).start()
    frames = []
    seq = -1
    print("   카메라 준비 중...", end="", flush=True)
    deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
    while True:
        try:
            _raw, seq = camera.capture_new_frame(seq)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                camera.stop()
                print()
                print(" 카메라가 %.0f초 안에 프레임을 주지 않았습니다." % FIRST_FRAME_TIMEOUT_SEC)
                return None
            print(".", end="", flush=True)
    print(" 준비됨")
    print()
    print(" 지금부터 %d장을 모읍니다 - 평소 자세로 고개를 천천히 좌우·상하로 움직이세요."
          % count)
    started = time.monotonic()
    last = 0.0
    try:
        while len(frames) < count and time.monotonic() - started < seconds:
            try:
                raw, seq = camera.capture_new_frame(seq)
            except RuntimeError:
                continue
            frames.append(preprocessor.preprocess_frame(raw, apply_crop=True))
            elapsed = time.monotonic() - started
            if elapsed - last >= 1.0:
                last = elapsed
                print("\r   %d / %d 장" % (len(frames), count), end="", flush=True)
    finally:
        camera.stop()
    print("\r   %d장 모았습니다.          " % len(frames))
    return frames


def _run_scale(config, scale, frames):
    """한 배율로 모든 프레임을 돌린다 -> (추론시간 리스트, 프레임별 결과).

    결과는 (미간 x, 미간 y, 안구간거리, yaw, pitch) — 얼굴을 못 찾으면 None.
    """
    cfg = copy.deepcopy(config)
    cfg["face_tracker"]["infer_scale_ratio"] = scale
    estimator = FaceEstimator(cfg)
    times, out = [], []
    for frame in frames:
        t = time.perf_counter()
        faces = estimator.infer(frame)
        times.append((time.perf_counter() - t) * 1000.0)
        face = select_user_face(faces)
        if face is None:
            out.append(None)
            continue
        lx, ly = face.landmark_px(LMK_LEFT_EYE_OUTER)
        rx, ry = face.landmark_px(LMK_RIGHT_EYE_OUTER)
        dist = math.hypot(rx - lx, ry - ly)
        pose = face.head_pose
        out.append(((lx + rx) * 0.5, (ly + ry) * 0.5, dist,
                    pose.yaw_deg if pose else None,
                    pose.pitch_deg if pose else None))
    return times, out


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="추론 축소 배율 속도/정확도 실측")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--frames", type=int, default=150, help="모을 프레임 수 (기본 150)")
    parser.add_argument("--capture-seconds", type=float, default=20.0)
    parser.add_argument("--scales", default="1.0,0.75,0.6,0.5",
                        help="비교할 배율 (쉼표 구분, 첫 값이 정확도 기준)")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    if not scales:
        print(" 배율을 하나 이상 지정하세요.")
        return 2

    config = load_config(args.config)
    print("=" * 70)
    print(" 추론 입력 축소 배율 — 속도 vs 정확도 실측")
    print("=" * 70)

    frames = _capture_frames(config, args.config, args.frames, args.capture_seconds)
    if not frames:
        return 2
    if len(frames) < 30:
        print(" 프레임이 %d장뿐이라 분석하지 않습니다 (최소 30장)." % len(frames))
        return 2

    height, width = frames[0].shape[:2]
    print()
    print(" 입력 %dx%d, 프레임 %d장" % (width, height, len(frames)))
    print()

    results = []
    for scale in scales:
        print("   배율 %.2f 로 %d장 추론 중..." % (scale, len(frames)), end="", flush=True)
        times, out = _run_scale(config, scale, frames)
        found = sum(1 for o in out if o is not None)
        print("\r   배율 %.2f  완료 (얼굴 %d/%d)          " % (scale, found, len(out)))
        results.append((scale, times, out))

    base_scale, base_times, base_out = results[0]

    print()
    print("=" * 70)
    print(" [1] 속도")
    print("=" * 70)
    print("   배율    중앙값     p95      최대    이론FPS   기준대비")
    base_med = statistics.median(base_times)
    for scale, times, _out in results:
        med = statistics.median(times)
        print("   %.2f  %7.1fms %7.1f %8.1f %8.1f %8.2f배"
              % (scale, med, _pct(times, 0.95), max(times), 1000.0 / med, base_med / med))

    print()
    print("=" * 70)
    print(" [2] 정확도 — 배율 %.2f 기준, 같은 프레임에서의 차이" % base_scale)
    print("=" * 70)
    print("   배율   검출률   미간오차   yaw오차   pitch오차   커서오차(화면%)")
    for scale, _times, out in results:
        dl, dyaw, dpitch, dcur_x, dcur_y = [], [], [], [], []
        found = 0
        for base, cur in zip(base_out, out):
            if cur is None:
                continue
            found += 1
            if base is None:
                continue
            dist = base[2] if base[2] > 1e-6 else 1.0
            ex, ey = cur[0] - base[0], cur[1] - base[1]
            dl.append(math.hypot(ex, ey))
            # 커서는 (좌표차이 / 안구간거리) x 감도 로 만들어진다 -> 화면 비율
            dcur_x.append(abs(ex / dist) * SENS_X * 100.0)
            dcur_y.append(abs(ey / dist) * SENS_Y * 100.0)
            if base[3] is not None and cur[3] is not None:
                dyaw.append(abs(cur[3] - base[3]))
                dpitch.append(abs(cur[4] - base[4]))

        def med(xs):
            return statistics.median(xs) if xs else float("nan")

        print("   %.2f  %5.0f%%  %7.2fpx %8.2f도 %9.2f도   %.2f%% / %.2f%%"
              % (scale, found / len(out) * 100.0, med(dl), med(dyaw), med(dpitch),
                 med(dcur_x), med(dcur_y)))

    print()
    print(" 읽는 법")
    print("   · 커서오차는 화면 폭·높이 대비 %입니다. 1920px 화면에서 0.1%면 약 2px,")
    print("     1.0%면 약 19px입니다. 손떨림·랜드마크 자체 잡음이 이미 2~4px 있으므로")
    print("     그보다 작은 오차는 실사용에서 구분되지 않습니다.")
    print("   · 검출률이 떨어지는 배율은 속도가 아무리 빨라도 쓰면 안 됩니다.")
    print("   · p95와 최대가 중앙값 대비 얼마나 큰지 보세요 - 긴 꼬리가 곧 버벅거림입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
