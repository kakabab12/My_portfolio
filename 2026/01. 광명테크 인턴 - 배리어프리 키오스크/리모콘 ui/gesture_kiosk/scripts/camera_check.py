"""카메라 진단 — 이 PC에 어떤 카메라가 몇 번으로 잡히는지 눈으로 확인한다
(2026-08-26 신설, 키오스크 실기 — "카메라를 못잡는거같아. 노트북은 잘되는데").

왜 필요한가
-----------
같은 프로그램이 노트북에서는 되고 키오스크에서는 안 될 때, 열에 아홉은 **장치
번호**가 다르기 때문이다. 노트북은 웹캠이 하나라 0번이 곧 그 웹캠이지만,
키오스크는 카메라가 두 대인 데다 노트북 화면용 IR 카메라나 가상 카메라가
끼어들면 번호가 밀린다. config.yaml의 camera.device_id가 엉뚱한 장치를
가리키면 프로그램은 "카메라를 못 잡는" 것처럼 보인다.

그런데 지금까지는 그걸 확인할 방법이 없었다. 프로그램을 띄워 보고 안 되면
device_id를 바꿔가며 다시 띄우는 수밖에 없었는데, 모델 로딩까지 매번 기다려야
해서 한 번 시도에 10초가 넘게 걸렸다. 이 도구는 카메라만 본다.

무엇을 보여주나
---------------
장치 번호를 0부터 차례로 열어보고, 각각에 대해:

  · 열리는가
  · 실제로 그림이 들어오는가 (열리기만 하고 안 주는 장치가 흔하다)
  · 해상도와 초당 장수
  · **얼굴이 잡히는가** — 헤드트래커가 쓸 수 있는 장치인지 최종 판단
  · 그 순간의 사진 (logs/camera_check_N.jpg)

얼굴까지 확인하는 이유는, 그림이 들어와도 IR 카메라처럼 얼굴 인식이 안 되는
장치가 있기 때문이다. 사진까지 남기는 건 "위쪽 카메라인지 아래쪽 카메라인지"를
사람이 눈으로 구분해야 하기 때문이다 — 번호만으로는 알 수 없다.

장치마다 프로세스를 따로 띄우는 이유
-----------------------------------
★2026-08-26 키오스크 실기 — "1번 카메라에서 꺼지는데".

윈도우 MSMF는 문제 있는 장치를 열 때 파이썬 예외가 아니라 **프로세스 자체를
죽이는 크래시**를 낸다(드라이버 안에서 나는 접근 위반). 그러면 try/except도
시간 제한도 소용이 없다 — 진단 도구가 1번에서 죽어버려 2번 이후는 아예 못
본다. 그래서 장치 하나를 확인할 때마다 **자식 프로세스를 따로 띄운다**. 자식이
죽어도 부모는 "이 번호는 프로그램을 죽인다"고 적고 다음 번호로 넘어간다.

쓰는 법
-------
    py -3.11 scripts/camera_check.py
    py -3.11 scripts/camera_check.py --devices 6   (0~5번까지 훑기)
    py -3.11 scripts/camera_check.py --no-face     (얼굴 확인 없이 빠르게)

끝나면 "config.yaml에 이 번호를 쓰세요"까지 알려준다.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from src.utils.config_loader import load_config   # noqa: E402

DEFAULT_DEVICE_COUNT = 4
WARMUP_FRAMES = 5        # 자동 노출이 잡히기 전 몇 장은 버린다
SAMPLE_FRAMES = 20       # 초당 장수를 재는 표본
OPEN_TIMEOUT_SEC = 15.0  # config probe_open_timeout_sec와 같은 값 — 정상 장치도
                         # MSMF 오픈에 10초 넘게 걸리는 PC가 있다(실측 2026-07-31)


def _open_with_timeout(config, device_id, timeout_sec):
    """장치 열기를 시간 한도로 감싼다 — 없는 번호에서 영영 안 돌아오는 걸 막는다.

    camera_probe._open_with_timeout과 같은 이유·같은 방식(그 함수 설명 참고).
    여기서 다시 쓰는 건 이 스크립트가 그 모듈을 안 거치고도 혼자 돌 수 있어야
    하기 때문이다 — 진단 도구가 진단 대상에 의존하면 곤란하다.
    """
    import threading
    from src.capture.camera_stream import init_camera

    holder = {"cap": None, "abandoned": False}
    lock = threading.Lock()
    opened = threading.Event()

    def _open():
        try:
            cap = init_camera(config, device_id=device_id)
        except Exception:
            cap = None
        with lock:
            if holder["abandoned"]:
                if cap is not None:
                    cap.release()
                return
            holder["cap"] = cap
            opened.set()

    threading.Thread(target=_open, daemon=True).start()
    if opened.wait(timeout_sec):
        return holder["cap"]
    with lock:
        if opened.is_set():
            return holder["cap"]
        holder["abandoned"] = True
    return None


def _save_jpg(path, frame):
    """한글 경로에서도 저장되게 — cv2.imwrite는 한글 경로를 못 연다(실측)."""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def check_device(config, device_id, face_estimator, preprocessor):
    """장치 하나를 확인해 결과 dict로 돌려준다."""
    result = {"device_id": device_id, "opened": False, "frames": 0,
              "width": 0, "height": 0, "fps": 0.0, "faces": None, "photo": None}

    cap = _open_with_timeout(config, device_id, OPEN_TIMEOUT_SEC)
    if cap is None:
        return result
    result["opened"] = True
    try:
        for _ in range(WARMUP_FRAMES):
            cap.read()

        last_frame = None
        got = 0
        t0 = time.monotonic()
        for _ in range(SAMPLE_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None:
                got += 1
                last_frame = frame
        elapsed = time.monotonic() - t0
        result["frames"] = got
        if got:
            result["fps"] = got / max(elapsed, 1e-6)
            result["height"], result["width"] = last_frame.shape[:2]

            if face_estimator is not None:
                # 헤드트래커와 똑같은 전처리를 거쳐야 판단이 의미가 있다
                prepared = preprocessor.preprocess_frame(last_frame, apply_crop=True)
                faces = face_estimator.infer(prepared)
                result["faces"] = len(faces)
                shot = prepared.copy()
                for face in faces:
                    x1, y1, x2, y2 = face.bbox
                    cv2.rectangle(shot, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
            else:
                shot = last_frame

            photo = os.path.join("logs", "camera_check_%d.jpg" % device_id)
            os.makedirs("logs", exist_ok=True)
            if _save_jpg(photo, shot):
                result["photo"] = photo
    finally:
        cap.release()
    return result


PROBE_TIMEOUT_SEC = 40.0   # 자식 하나가 이보다 오래 걸리면 그 장치는 포기한다
                           # (오픈 한도 15초 + 모델 로딩 + 표본 촬영 여유)


def probe_one_in_child(config, device_id, want_face):
    """자식 프로세스 본체 — 장치 하나를 확인하고 결과를 한 줄 JSON으로 찍는다."""
    face_estimator = preprocessor = None
    if want_face:
        from src.inference.face_estimator import FaceEstimator
        from src.inference.preprocessor import Preprocessor
        preprocessor = Preprocessor(config)
        face_estimator = FaceEstimator(config)
    try:
        res = check_device(config, device_id, face_estimator, preprocessor)
    finally:
        if face_estimator is not None:
            face_estimator.close()
    print("__RESULT__" + json.dumps(res, ensure_ascii=False))
    return 0


def probe_one(config_path, device_id, want_face):
    """장치 하나를 **자식 프로세스**에서 확인한다.

    자식이 죽어도(MSMF 크래시) 부모는 멀쩡하다 — 위 모듈 설명 참고.
    돌아오는 dict의 crashed=True면 그 번호가 프로그램을 죽였다는 뜻이다.
    """
    cmd = [sys.executable, "-X", "utf8", os.path.abspath(__file__),
           "--probe-one", str(device_id), "--config", config_path]
    if not want_face:
        cmd.append("--no-face")
    dead = {"device_id": device_id, "opened": False, "frames": 0, "width": 0,
            "height": 0, "fps": 0.0, "faces": None, "photo": None, "crashed": True}
    try:
        done = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=PROBE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        dead["timed_out"] = True
        return dead
    for line in (done.stdout or "").splitlines():
        if line.startswith("__RESULT__"):
            res = json.loads(line[len("__RESULT__"):])
            res["crashed"] = False
            return res
    return dead


# 한 장치가 안 잡힐 때 시험해 볼 조합 — 백엔드 x 형식 x 해상도.
# 설정이 강제하는 조합(msmf + mjpg + 1280x720)을 그 장치가 거부하면 "카메라를
# 못 잡는" 것처럼 보인다. 기종이 다르면 받아주는 조합도 다르다(config 주석의
# Brio 100 실측 참고 — dshow는 YUY2 720p로만 협상돼 ~5장/초였다).
COMBOS = [
    ("msmf", "mjpg", 1280, 720),
    ("msmf", "auto", 1280, 720),
    ("msmf", "auto", 640, 480),
    ("dshow", "mjpg", 1280, 720),
    ("dshow", "auto", 1280, 720),
    ("dshow", "auto", 640, 480),
    ("auto", "auto", 640, 480),
]


def try_combo_in_child(config, device_id, backend, fourcc, width, height):
    """설정을 이 조합으로 바꿔 끼운 뒤 장치를 열어 본다 (자식 프로세스 본체)."""
    cam = config["camera"]
    cam["windows_backend"] = backend
    cam["fourcc"] = fourcc
    cam["width_px"] = width
    cam["height_px"] = height
    res = check_device(config, device_id, None, None)
    print("__RESULT__" + json.dumps(res, ensure_ascii=False))
    return 0


def try_combo(config_path, device_id, combo):
    """조합 하나를 자식 프로세스에서 시험한다 — 죽어도 부모는 산다."""
    backend, fourcc, width, height = combo
    cmd = [sys.executable, "-X", "utf8", os.path.abspath(__file__),
           "--try-combo", str(device_id), "--config", config_path,
           "--backend", backend, "--fourcc", fourcc,
           "--width", str(width), "--height", str(height)]
    dead = {"opened": False, "frames": 0, "fps": 0.0, "width": 0, "height": 0,
            "crashed": True}
    try:
        done = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=PROBE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        dead["timed_out"] = True
        return dead
    for line in (done.stdout or "").splitlines():
        if line.startswith("__RESULT__"):
            res = json.loads(line[len("__RESULT__"):])
            res["crashed"] = False
            return res
    return dead


def diagnose_device(config_path, device_id):
    """안 잡히는 장치 하나를 조합을 바꿔가며 파고든다."""
    print("=" * 66)
    print(" %d번 장치 정밀 진단 — 어떤 조합이면 잡히는지 찾습니다" % device_id)
    print("=" * 66)
    print(" %-8s %-8s %-12s %s" % ("백엔드", "형식", "요청 해상도", "결과"))
    print(" " + "-" * 60)
    winners = []
    for combo in COMBOS:
        backend, fourcc, width, height = combo
        print(" %-8s %-8s %-12s " % (backend, fourcc, "%dx%d" % (width, height)),
              end="", flush=True)
        res = try_combo(config_path, device_id, combo)
        if res.get("crashed"):
            print("★프로그램이 죽음")
        elif res.get("timed_out"):
            print("응답 없음")
        elif not res["opened"]:
            print("열리지 않음")
        elif res["frames"] == 0:
            print("열렸지만 화면 없음")
        else:
            print("OK — 실제 %dx%d, 초당 %.0f장"
                  % (res["width"], res["height"], res["fps"]))
            winners.append((combo, res))
    print()
    if not winners:
        print(" ★어떤 조합으로도 이 장치에서 화면을 받지 못했습니다.")
        print("   USB 연결·전원, 그리고 장치 관리자에 카메라로 보이는지 확인하세요.")
        return 1
    best_combo, best_res = max(winners, key=lambda pair: pair[1]["fps"])
    backend, fourcc, width, height = best_combo
    print(" 이 장치는 아래 설정이면 잡힙니다 — configs/config.yaml 의 camera 항목:")
    print()
    print("   device_id: %d" % device_id)
    print("   width_px: %d" % width)
    print("   height_px: %d" % height)
    print("   windows_backend: %s" % backend)
    print("   fourcc: %s" % fourcc)
    print()
    print(" (실제 %dx%d, 초당 %.0f장으로 들어옵니다)"
          % (best_res["width"], best_res["height"], best_res["fps"]))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="이 PC에 어떤 카메라가 몇 번으로 잡히는지 확인한다")
    parser.add_argument("--devices", type=int, default=DEFAULT_DEVICE_COUNT,
                        help="0번부터 몇 개까지 훑을지 (기본 %d)" % DEFAULT_DEVICE_COUNT)
    parser.add_argument("--no-face", action="store_true",
                        help="얼굴 확인을 건너뛴다 (모델 로딩 없이 빠르게)")
    parser.add_argument("--config", default=os.path.join("configs", "config.yaml"))
    parser.add_argument("--probe-one", type=int, default=None,
                        help="(내부용) 이 번호 하나만 확인하고 결과를 찍는다")
    parser.add_argument("--diagnose", type=int, default=None,
                        help="안 잡히는 장치 번호 — 설정 조합을 바꿔가며 원인을 찾는다")
    parser.add_argument("--try-combo", type=int, default=None, help="(내부용)")
    parser.add_argument("--backend", default=None, help="(내부용)")
    parser.add_argument("--fourcc", default=None, help="(내부용)")
    parser.add_argument("--width", type=int, default=None, help="(내부용)")
    parser.add_argument("--height", type=int, default=None, help="(내부용)")
    args = parser.parse_args()

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                if stream.isatty():
                    reconfigure(errors="replace")
                else:
                    reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    config = load_config(args.config)
    current_id = config["camera"]["device_id"]

    # 자식 모드 — 장치 하나만 보고 결과를 찍고 끝낸다
    if args.probe_one is not None:
        return probe_one_in_child(config, args.probe_one, not args.no_face)
    if args.try_combo is not None:
        return try_combo_in_child(config, args.try_combo, args.backend,
                                  args.fourcc, args.width, args.height)
    if args.diagnose is not None:
        return diagnose_device(args.config, args.diagnose)

    print("=" * 66)
    print(" 카메라 진단")
    print("=" * 66)
    print(" 지금 설정된 번호 : camera.device_id = %s" % current_id)
    print(" 훑어볼 범위      : 0 ~ %d번" % (args.devices - 1))
    print(" 얼굴 확인        : %s" % ("건너뜀" if args.no_face else "함"))
    print()
    print(" ※ 확인하는 동안 카메라 앞에 얼굴이 보이게 서 계세요.")
    print()

    print(" 장치마다 프로세스를 따로 띄웁니다 — 한 장치가 죽어도 나머지는 계속 봅니다.")
    print()

    results = []
    for device_id in range(args.devices):
        print(" %d번 확인 중..." % device_id, end="", flush=True)
        res = probe_one(args.config, device_id, not args.no_face)
        results.append(res)
        if res.get("timed_out"):
            print(" 응답 없음 (%.0f초 초과) — 이 장치는 건너뜁니다" % PROBE_TIMEOUT_SEC)
        elif res.get("crashed"):
            print(" ★프로그램이 죽음 — 이 번호는 쓰면 안 됩니다")
        elif not res["opened"]:
            print(" 열리지 않음")
        elif res["frames"] == 0:
            print(" 열렸지만 화면이 안 들어옴")
        else:
            face_text = ""
            if res["faces"] is not None:
                face_text = "  얼굴 %d명" % res["faces"]
            print(" %dx%d  초당 %.0f장%s" %
                  (res["width"], res["height"], res["fps"], face_text))

    print()
    print("=" * 66)
    print(" 결과")
    print("=" * 66)
    print(" %-6s %-10s %-12s %-10s %s" % ("번호", "열림", "해상도", "초당장수", "얼굴"))
    print(" " + "-" * 60)
    for res in results:
        if res.get("crashed") or res.get("timed_out"):
            mark = "★죽음" if res.get("crashed") else "응답없음"
            print(" %-6d %-10s %-12s %-10s %s" % (res["device_id"], mark, "-", "-", "-"))
            continue
        if not res["opened"]:
            print(" %-6d %-10s %-12s %-10s %s" % (res["device_id"], "아니오", "-", "-", "-"))
            continue
        face_text = "-" if res["faces"] is None else ("%d명" % res["faces"])
        print(" %-6d %-10s %-12s %-10.0f %s" %
              (res["device_id"], "예",
               "%dx%d" % (res["width"], res["height"]) if res["frames"] else "-",
               res["fps"], face_text))

    usable = [r for r in results if r["opened"] and r["frames"] > 0]
    with_face = [r for r in usable if r["faces"]]

    print()
    if not usable:
        print(" ★쓸 수 있는 카메라가 하나도 없습니다.")
        print("   · USB 연결을 확인하세요")
        print("   · 다른 프로그램(화상회의 등)이 카메라를 쓰고 있는지 확인하세요")
        print("   · --devices 8 로 더 넓게 훑어보세요")
        return 1

    print(" 사진을 남겼습니다 — 어느 쪽 카메라인지 눈으로 확인하세요:")
    for res in usable:
        if res["photo"]:
            print("   %d번 -> %s" % (res["device_id"], res["photo"]))

    print()
    if with_face:
        best = max(with_face, key=lambda r: r["fps"])
        print(" 권장: camera.device_id = %d" % best["device_id"])
        print("   (얼굴이 잡히고 초당 %.0f장이 나옵니다)" % best["fps"])
        if best["device_id"] != current_id:
            print()
            print(" ★지금 설정(%s)과 다릅니다. configs/config.yaml 의" % current_id)
            print("   camera.device_id 를 %d 로 바꾸세요." % best["device_id"])
        else:
            print("   지금 설정과 같습니다 — 장치 번호 문제는 아닙니다.")
    else:
        print(" 화면은 들어오는데 어느 장치에서도 얼굴이 안 잡혔습니다.")
        print("   · 카메라 앞에 얼굴이 보이게 서서 다시 해보세요")
        print("   · 사진(logs/camera_check_*.jpg)을 열어 무엇이 찍혔는지 확인하세요")
        print("   · 너무 어둡거나, IR 카메라만 잡혔을 수 있습니다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
