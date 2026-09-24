# -*- coding: utf-8 -*-
"""세 트래커의 main()을 가상 카메라·가상 사용자로 처음부터 끝까지 돌린다 (2026-09-24 신설).

    py -3 tests/tracker_smoke.py head          # head.py, 9:16 키오스크
    py -3 tests/tracker_smoke.py eyebrow --aspect desktop
    py -3 tests/tracker_smoke.py all           # 셋 다, 두 화면비 모두

왜 필요한가
-----------
테스트 800여 개는 판정·매핑 모듈을 따로 본다. 세 트래커의 main()은 거의 아무도
거치지 않는다. 2026-09-10에 리팩터가 main() 안쪽의 정의 두 개를 지웠을 때
컴파일·자가 점검·테스트가 전부 통과했고, 사용자가 실행한 뒤에야 NameError가
났다. 그 구간을 실제로 돌려 보는 것이 이 파일이다.

무엇을 가짜로 바꾸나 — 사용자 화면·마우스·파일을 건드리는 것만
-----------------------------------------------------------------
  · 실제 마우스(_Win32Mouse)      → 동작을 기록만 하는 FakeMouse
  · 시스템 커서 숨기기            → 아무것도 안 함
  · 카메라(CameraStream)          → 30 fps로 검은 프레임을 주는 FakeCamera
  · 얼굴 인식(FaceEstimator)      → 가상 카메라가 만든 얼굴을 FaceLandmarks로 감싸 준다
  · OpenCV 창·투명 오버레이        → 아무것도 안 함
  · 로그 파일                     → 메모리에 모아 오류를 센다
  · 콘솔 입력                     → 시나리오가 끝나면 "quit"을 넣는다
그 밖의 전부(전처리, 얼굴 앵커, HeadTracker, 입 판정, 클릭 붙잡기, 렌더 루프, 스레드,
종료 절차)는 진짜 코드가 돈다.

무엇을 확인하나
---------------
  · main()이 0을 돌려주고 제시간에 끝나는가
  · 로그에 ERROR가 하나도 없는가
  · 입으로 한 클릭이 3번 이상, 드래그가 1번 이상 OS 마우스까지 갔는가
  · 버튼을 누른 수와 뗀 수가 같고, 끝났을 때 눌린 채가 아닌가
  · 오른쪽·왼쪽을 가리키면 커서가 서로 반대쪽에 있는가
  · 얼굴이 사라졌다 돌아온 뒤에도 커서가 다시 따라오는가
  · 옆에 다른 사람이 들어와도 커서가 튀지 않는가
  · 끝난 뒤 남은 스레드가 없는가
장시간 모드(--soak 분)에서는 같은 흐름을 되풀이하며 여기에 더해
  · 메모리 추세(표본 전체에 맞춘 직선의 기울기)와 스레드 수
  · 위의 위치 확인 셋을 **첫 회차와 마지막 회차 모두**에서 — 오래 켜 둔 뒤에도 맞는가
"""
import argparse
import bisect
import importlib.util
import io
import logging
import math
import os
import subprocess
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.inference.face_estimator import (  # noqa: E402
    FaceLandmarks, HeadPose, _landmarks_to_bbox_px, _rotation_matrix_to_euler_deg)
import tests.virtual_camera as vc  # noqa: E402
from tests.virtual_camera import VirtualCamera, rotation  # noqa: E402
from tests.virtual_user import FPS, VirtualUser, aim, mouth_open, rest  # noqa: E402

TRACKERS = ("head", "eyebrow", "forehead")
SCREEN_PX = {"kiosk": (1080, 1920), "desktop": (1920, 1080)}   # (가로, 세로)
FRAME_SHAPE = (720, 1280, 3)        # 웹캠이 주는 가로 프레임 — 키오스크는 405x720으로 잘린다
WATCHDOG_SEC = 120.0
SHUT, WIDE = 0.05, 0.45
# 얼굴을 새로 잡으면 트래커가 8초 동안 커서를 가운데 붙잡고 "커서 중앙을 봐주시고 입은
# 다문 채 편한 자세로 있어주세요"라고 안내한다(SETTLE_DELAY_SEC, 사용자 요청으로 5->7->8초).
# 그동안은 클릭도 받지 않는다. 사람은 이 안내를 따르므로 가상 사용자도 따른다.
SETTLE_WAIT_SEC = 9.2
# 장시간 모드에서 마우스 이동 기록을 이만큼 넘으면 MOVES_TRIM개를 지운다(첫 회차는 남김)
MOVES_KEEP_MAX = 20000
MOVES_TRIM = 10000


def _click(yaw, pitch=0.0, label="클릭"):
    return [mouth_open(0.12, jaw_open=WIDE, yaw_deg=yaw, pitch_deg=pitch, label=label),
            mouth_open(0.30, jaw_open=WIDE, yaw_deg=yaw, pitch_deg=pitch, label=label),
            mouth_open(0.15, jaw_open=SHUT, yaw_deg=yaw, pitch_deg=pitch, label=label),
            rest(1.0, yaw_deg=yaw, pitch_deg=pitch, label=label + " 뒤")]


# 시나리오 — 라벨이 곧 확인할 구간 이름이다
ACTIONS = (
    [rest(SETTLE_WAIT_SEC, label="캘리브레이션"),
     aim(1.0, yaw_deg=8.0, label="오른쪽 겨누기"), rest(0.8, yaw_deg=8.0, label="오른쪽 머묾")]
    + _click(8.0, label="클릭1")
    + [aim(1.0, yaw_deg=-8.0, pitch_deg=4.0, label="왼쪽 겨누기"),
       rest(0.8, yaw_deg=-8.0, pitch_deg=4.0, label="왼쪽 머묾")]
    + _click(-8.0, 4.0, label="클릭2")
    + [mouth_open(0.12, jaw_open=WIDE, yaw_deg=-8.0, pitch_deg=4.0, label="드래그"),
       mouth_open(1.5, jaw_open=WIDE, yaw_deg=8.0, pitch_deg=0.0, label="드래그"),
       mouth_open(0.15, jaw_open=SHUT, yaw_deg=8.0, label="드래그 놓기"),
       rest(1.2, yaw_deg=8.0, label="드래그 놓기"),
       rest(2.0, yaw_deg=0.0, label="얼굴 사라짐"),
       rest(SETTLE_WAIT_SEC, yaw_deg=0.0, label="다시 나타남"),
       aim(1.0, yaw_deg=6.0, label="돌아와 오른쪽"), rest(0.6, yaw_deg=6.0, label="돌아와 머묾")]
    + _click(6.0, label="클릭3")
    + [rest(2.0, yaw_deg=6.0, label="옆에 다른 사람"),
       rest(1.0, yaw_deg=6.0, label="끝")]
)


class Scenario:
    """시각 -> (얼굴들, 라벨). 시작 시각은 첫 추론 때 정한다."""

    def __init__(self, frame_w_px):
        self.frames = list(VirtualUser(ACTIONS).frames())
        self.duration = self.frames[-1][0]
        self.t0 = None
        self.t0_epoch = None
        self.loop = False
        self.run_sec = None     # 장시간 모드에서 quit을 넣을 시각
        self._lock = threading.Lock()
        vc.FRAME_W_PX = frame_w_px
        self.camera = VirtualCamera(distance_mm=650.0, lens="일반 65도", noise_px=0.35, seed=7)
        self.other = VirtualCamera(distance_mm=650.0, lens="일반 65도", noise_px=0.35, seed=8)

    def start(self):
        with self._lock:
            if self.t0 is None:
                self.t0 = time.monotonic()
                self.t0_epoch = time.time()   # 로그 기록 시각(에포크)과 맞추려고
        return self.t0

    def now(self):
        return None if self.t0 is None else time.monotonic() - self.t0

    def label_at(self, t):
        idx = min(len(self.frames) - 1, max(0, int(t * FPS)))
        return self.frames[idx][4]

    def faces(self, frame_shape):
        t = time.monotonic() - self.start()
        if self.loop:
            t = t % self.duration     # 장시간 모드 — 같은 사용 흐름을 계속 되풀이한다
        idx = min(len(self.frames) - 1, int(t * FPS))
        _now, yaw, pitch, jaw, label = self.frames[idx]
        if label == "얼굴 사라짐":
            return []
        rot = rotation((0, 1, 0), yaw) @ rotation((1, 0, 0), pitch)
        faces = [self._wrap(self.camera.observe(rot, expression=self._expr(jaw)), jaw, frame_shape)]
        if label == "옆에 다른 사람":
            # 더 멀리, 옆에 선 사람 — 앵커가 원래 사용자를 지켜야 한다
            far = self.other.observe(rotation((0, 1, 0), -20.0), offset_mm=(170.0, -30.0, 450.0))
            faces.append(self._wrap(far, SHUT, frame_shape))
        return faces

    @staticmethod
    def _expr(jaw):
        return max(0.0, min(1.0, (jaw - SHUT) / 0.40))

    @staticmethod
    def _wrap(virtual, jaw, frame_shape):
        """실제 FaceEstimator.infer가 만드는 것과 같은 모양으로 감싼다."""
        h_px, w_px = frame_shape[:2]
        pts3 = np.asarray(virtual.landmarks_3d, dtype=np.float32)
        pts2 = pts3[:, :2].copy()
        yaw, pitch, roll = _rotation_matrix_to_euler_deg(np.asarray(virtual.head_rotation))
        return FaceLandmarks(
            bbox=_landmarks_to_bbox_px(pts2, frame_shape), conf=1.0,
            landmarks_px=pts2, blendshapes={"jawOpen": float(jaw)},
            landmarks_3d=pts3, frame_size=(w_px, h_px),
            head_pose=HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                               tx=0.0, ty=0.0, tz=-65.0),
            head_rotation=np.asarray(virtual.head_rotation, dtype=np.float64))


class FakeMouse:
    """_Win32Mouse와 같은 모양 — OS에 보내는 대신 기록만 한다."""

    screen_px = (1920, 1080)
    scenario = None

    def __init__(self):
        self.screen_w_px, self.screen_h_px = FakeMouse.screen_px
        self.is_pressed = False
        self.moves = []          # (t, x_ratio, y_ratio)
        self.move_count = 0      # 지운 것까지 센 이동 수
        self.log = []            # (t, 동작)
        self._first_pass_len = None
        FakeMouse.instance = self

    def _t(self):
        t = self.scenario.now()
        return -1.0 if t is None else t

    def move(self, x_ratio, y_ratio):
        self.moves.append((self._t(), float(x_ratio), float(y_ratio)))
        self.move_count += 1
        if self.scenario.loop and len(self.moves) > MOVES_KEEP_MAX:
            # 장시간 모드 — 시험 도구가 메모리를 먹으면 안 된다. 단 **첫 회차는 남기고**
            # 그 뒤의 오래된 것부터 지운다. 위치 확인이 첫 회차와 마지막 회차를 본다.
            # (2026-09-25: 맨 앞부터 지웠더니 커서를 자주 옮기는 forehead만 30분에
            # 2만 번을 넘어 첫 회차 기록이 사라졌고, 위치 확인 셋이 값 없이 실패했다)
            if self._first_pass_len is None:
                self._first_pass_len = bisect.bisect_right(
                    self.moves, (self.scenario.duration, math.inf, math.inf))
            keep = self._first_pass_len
            del self.moves[keep:keep + MOVES_TRIM]

    def click(self):
        self.log.append((self._t(), "click"))

    def press(self):
        self.log.append((self._t(), "press"))
        self.is_pressed = True

    def release(self):
        self.log.append((self._t(), "release"))
        self.is_pressed = False

    def release_click(self):
        if not self.is_pressed:
            return
        self.log.append((self._t(), "release_click"))
        self.is_pressed = False

    def release_if_pressed(self):
        if self.is_pressed:
            self.log.append((self._t(), "release_if_pressed"))
            self.is_pressed = False


class FakeCursorHider:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def hide(self):
        self.calls.append("hide")

    def restore(self, force=False):
        self.calls.append("restore")

    def install_exit_guards(self, mouse=None):
        self.calls.append("guards")


class FakeCamera:
    """30 fps로 새 프레임 번호를 준다. 내용은 검은 화면 — 얼굴은 FakeFaceEstimator가 준다."""

    def __init__(self, config, *args, **kwargs):
        self.portrait_crop = (config.get("camera", {}).get("portrait_crop") or {})
        self._frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        self._t0 = None
        self.stopped = False

    def start(self):
        self._t0 = time.monotonic()
        return self

    def capture_new_frame(self, last_seq):
        target = self._t0 + (last_seq + 1) / FPS
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return self._frame.copy(), last_seq + 1

    def stop(self):
        self.stopped = True


class FakeFaceEstimator:
    scenario = None

    def __init__(self, *args, **kwargs):
        pass

    def infer(self, frame):
        return self.scenario.faces(frame.shape)

    def close(self):
        pass


class ScriptedStdin:
    """시나리오가 끝나면 quit을 넣는다 — 진짜 종료 경로(콘솔 명령)를 그대로 탄다."""

    def __init__(self, scenario):
        self._scenario = scenario
        self._sent = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._sent:
            raise StopIteration
        while self._scenario.t0 is None:
            time.sleep(0.05)
        end = self._scenario.t0 + (self._scenario.run_sec or self._scenario.duration + 0.5)
        while time.monotonic() < end:
            time.sleep(0.05)
        self._sent = True
        return "quit\n"

    def readline(self):
        try:
            return next(self)
        except StopIteration:
            return ""


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _load_tracker(name):
    path = os.path.join(ROOT, name + ".py")
    spec = importlib.util.spec_from_file_location("tracker_under_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pos_at(moves, t):
    """그 시각에 OS 커서가 있던 자리 — 마지막으로 옮긴 곳. 커서는 멈춰 있으면
    옮기지 않으므로(렌더 루프의 데드존) 이동 기록이 없는 구간도 이렇게 읽는다."""
    last = None
    for mt, x, y in moves:
        if mt > t:
            break
        last = (x, y)
    return last


def _window_times(scenario, labels, samples=12, offset=0.0):
    """라벨 구간의 시각들. offset은 회차의 시작 시각(장시간 모드에서 n회차 = n x 한 바퀴)."""
    ts = [offset + i / FPS for i, f in enumerate(scenario.frames) if f[4] in labels]
    if not ts:
        return []
    step = max(1, len(ts) // samples)
    return ts[::step]


def _window_mean(moves, scenario, labels, offset=0.0):
    pts = [p for p in (_pos_at(moves, t) for t in _window_times(scenario, labels, offset=offset)) if p]
    if not pts:
        return None
    arr = np.array(pts)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _check_passes(scenario):
    """위치를 확인할 회차들 — (이름, 시작 시각). 장시간 모드면 첫 회차와 **끝나기 전
    마지막으로 온전히 돈 회차**를 함께 본다(오래 켜 둔 뒤에도 커서가 맞는가)."""
    if not scenario.loop or not scenario.run_sec:
        return [("", 0.0)]
    last = int(scenario.run_sec // scenario.duration) - 1
    if last < 1:
        return [("", 0.0)]
    start = last * scenario.duration
    return [("첫 회차", 0.0), ("%.0f분 뒤 %d회차" % (start / 60.0, last + 1), start)]


def _position_checks(moves, scenario, offset):
    """한 회차의 위치 확인 셋 — (좌우, 복귀, 옆 사람) 각각 (통과, 보고 문구)."""
    right = _window_mean(moves, scenario, ("오른쪽 머묾",), offset)
    left = _window_mean(moves, scenario, ("왼쪽 머묾",), offset)
    back = _window_mean(moves, scenario, ("돌아와 머묾",), offset)
    side = (bool(right and left and abs(right[0] - left[0]) > 0.15),
            "좌우를 가리키면 커서가 반대쪽 (오른쪽 %s · 왼쪽 %s)" % (
                "%.2f" % right[0] if right else "-", "%.2f" % left[0] if left else "-"))
    ret = (bool(right and back and back[0] and (back[0] - 0.5) * (right[0] - 0.5) > 0),
           "얼굴이 돌아온 뒤에도 커서가 따라옴 (%s)" % ("%.2f" % back[0] if back else "-"))
    times = _window_times(scenario, ("옆에 다른 사람",), samples=40, offset=offset)
    before = _pos_at(moves, times[0] - 1.0 / FPS) if times else None
    during = [p for p in (_pos_at(moves, t) for t in times) if p]
    jump = (max(math.hypot(x - before[0], y - before[1]) for x, y in during)
            if (before and during) else None)
    other = (jump is not None and jump < 0.10,
             "옆에 다른 사람이 와도 커서가 안 튐 (최대 %s)" % ("%.3f" % jump if jump is not None else "-"))
    return side, ret, other


def _memory_trend(samples, soak_min):
    """(시작 MB, 끝 MB, 최소, 최대, 시간당 추세 MB). 첫 1분과 quit 뒤 표본은 뺀다.

    추세는 **표본 전체에 맞춘 직선의 기울기**다. 처음엔 두 끝점의 차로 셌는데,
    RSS는 30초 간격 표본 사이에서도 ±10MB씩 오르내려 끝점 하나에 크게 흔들렸다
    (2026-09-25 forehead 30분: 시작 109 -> 끝 110MB인데 "시간당 +23.4MB")."""
    warm = [s for s in samples if 1.0 <= s[0] < soak_min - 0.05] or samples
    mb = [s[1] for s in warm]
    if len(warm) >= 3:
        per_hour = float(np.polyfit([s[0] for s in warm], mb, 1)[0]) * 60.0
    else:
        per_hour = (mb[-1] - mb[0]) / max(1e-6, warm[-1][0] - warm[0][0]) * 60.0
    return samples[0][1], samples[-1][1], min(mb), max(mb), per_hour, warm


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:   # noqa: psutil이 없으면 메모리는 못 잰다
        return float("nan")


def run_one(name, aspect, soak_min=0.0):
    """한 트래커를 한 번 돌리고 (통과 여부, 보고 줄들)을 돌려준다."""
    import cv2
    os.chdir(ROOT)
    crop = (aspect == "kiosk")
    scenario = Scenario(frame_w_px=405 if crop else FRAME_SHAPE[1])
    if soak_min:
        scenario.loop = True
        scenario.run_sec = soak_min * 60.0
        global WATCHDOG_SEC
        WATCHDOG_SEC = soak_min * 60.0 + 120.0
    samples = []     # (분, 메모리 MB, 스레드 수)
    def _sampler():
        while scenario.t0 is None:
            time.sleep(0.2)
        while True:
            samples.append(((time.monotonic() - scenario.t0) / 60.0, _rss_mb(), threading.active_count()))
            time.sleep(30.0 if soak_min else 5.0)
    threading.Thread(target=_sampler, daemon=True).start()
    FakeMouse.screen_px = SCREEN_PX[aspect]
    FakeMouse.scenario = scenario
    FakeFaceEstimator.scenario = scenario

    module = _load_tracker(name)
    capture = _Capture()

    def fake_init_logging(_config):
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(capture)

    module._Win32Mouse = FakeMouse
    module._SystemCursorHider = FakeCursorHider
    module.CameraStream = FakeCamera
    module.FaceEstimator = FakeFaceEstimator
    module.init_logging = fake_init_logging
    module.enable_transparent_overlay = lambda *a, **k: None
    module.disable_console_quick_edit = lambda *a, **k: None
    popen_calls = []
    module.subprocess = types.SimpleNamespace(
        Popen=lambda *a, **k: popen_calls.append(a), DEVNULL=subprocess.DEVNULL,
        PIPE=subprocess.PIPE)
    module.console.emit = lambda *a, **k: None
    for fn in ("namedWindow", "imshow", "setWindowProperty", "destroyAllWindows",
               "destroyWindow", "moveWindow", "resizeWindow", "setMouseCallback"):
        setattr(cv2, fn, lambda *a, **k: None)
    cv2.waitKey = lambda *a, **k: -1
    cv2.pollKey = lambda *a, **k: -1
    cv2.getWindowProperty = lambda *a, **k: 1.0

    sys.stdin = ScriptedStdin(scenario)
    sys.argv = [name + ".py", "--no-window", "--aspect", aspect]
    threads_before = set(threading.enumerate())

    watchdog = threading.Timer(WATCHDOG_SEC, lambda: (
        print("  ✗ %s(%s) %d초 안에 끝나지 않음 — 멈춤" % (name, aspect, WATCHDOG_SEC), flush=True),
        os._exit(3)))
    watchdog.daemon = True
    watchdog.start()
    started = time.monotonic()
    crash = None
    try:
        rc = module.main()
    except SystemExit as exc:
        rc = exc.code
    except Exception as exc:   # noqa: 무엇이 터졌는지 그대로 보고한다
        import traceback
        rc, crash = None, traceback.format_exc()
    watchdog.cancel()
    elapsed = time.monotonic() - started
    time.sleep(0.3)   # 데몬 스레드가 마무리할 시간

    mouse = getattr(FakeMouse, "instance", None)
    moves = mouse.moves if mouse else []
    log = mouse.log if mouse else []
    kinds = [k for _t, k in log]
    errors = [r for r in capture.records if r.levelno >= logging.ERROR]
    checks = []

    def check(ok, text):
        checks.append((bool(ok), text))

    check(crash is None and rc in (0, None), "main()이 정상 종료 (rc=%r, %.1f초)" % (rc, elapsed))
    check(not errors, "로그 ERROR 없음" + ("" if not errors else " — %d건: %s" % (
        len(errors), errors[0].getMessage()[:80])))
    presses = kinds.count("press")
    ups = sum(kinds.count(k) for k in ("release", "release_click", "release_if_pressed"))
    check(kinds.count("release_click") >= 3, "입 클릭이 마우스까지 감 (%d번)" % kinds.count("release_click"))
    drags = 0
    down_t = None
    for t, k in log:
        if k == "press":
            down_t = t
        elif k in ("release", "release_if_pressed") and down_t is not None:
            if t - down_t >= 0.7:
                drags += 1
            down_t = None
        elif k == "release_click":
            down_t = None
    check(drags >= 1, "드래그가 마우스까지 감 (%d번)" % drags)
    check(presses == ups and not (mouse and mouse.is_pressed),
          "누름 %d번 = 뗌 %d번, 끝에 눌린 채 아님" % (presses, ups))
    bad = [m for m in moves if not all(math.isfinite(v) for v in m[1:])]
    move_count = mouse.move_count if mouse else 0
    check(move_count > 200 and not bad, "커서 이동 %d번, 숫자 아닌 값 %d개" % (move_count, len(bad))
          + ("" if len(moves) == move_count else " (기록은 첫 회차와 최근 %d개만 보관)" % len(moves)))
    for pass_name, offset in _check_passes(scenario):
        for ok_, text in _position_checks(moves, scenario, offset):
            check(ok_, text + (" — " + pass_name if pass_name else ""))
    leftover = [t for t in threading.enumerate()
                if t not in threads_before and t.is_alive() and not t.daemon]
    check(not leftover, "남은 스레드 없음" + ("" if not leftover else ": %s" % [t.name for t in leftover]))
    check(not popen_calls, "다른 프로세스를 띄우지 않음")

    if soak_min and len(samples) >= 3:
        # 첫 1분은 모델·캐시가 자리 잡는 시간이라 빼고, 그 뒤 증가량을 본다.
        # quit 뒤의 표본도 뺀다 — 종료하며 스레드가 정리된 뒤라 수가 줄어 보인다
        # (처음 돌렸을 때 이것 때문에 "2~6개"로 잘못 걸렸다)
        first_mb, last_mb, low_mb, high_mb, per_hour, warm = _memory_trend(samples, soak_min)
        check(per_hour < 50.0,
              "메모리 %.0f -> %.0f MB, 1분 뒤부터 %.0f~%.0f MB (추세 시간당 %+.1f MB)"
              % (first_mb, last_mb, low_mb, high_mb, per_hour))
        check(max(s[2] for s in warm) - min(s[2] for s in warm) <= 1,
              "스레드 수 %d~%d개로 고정" % (min(s[2] for s in warm), max(s[2] for s in warm)))
    ok = all(c[0] for c in checks)
    lines = ["■ %s.py · %s — %s" % (name, "9:16 키오스크" if crop else "16:9 데스크탑",
                                     "통과" if ok else "실패")]
    lines += ["   %s %s" % ("○" if c[0] else "✗", c[1]) for c in checks]
    if crash:
        lines.append("   예외:" + crash.replace("\n", "\n     "))
    if os.environ.get("SMOKE_VERBOSE"):
        lines.append("   --- 로그 (INFO 이상) ---")
        for r in capture.records:
            lines.append("   %6.2f %-7s %s" % (r.created - (scenario.t0_epoch or r.created), r.levelname, r.getMessage()[:150]))
        lines.append("   --- 마우스 ---")
        lines += ["   %6.2f %s" % (tt, k) for tt, k in log]
        step = max(1, len(moves) // 40)
        lines += ["   %6.2f move %.3f %.3f  [%s]" % (tt, x, y, scenario.label_at(tt)) for tt, x, y in moves[::step]]
    for r in errors[:5]:
        lines.append("   ERROR %s" % r.getMessage()[:120])
        if r.exc_text:
            lines.append("     " + r.exc_text.strip().splitlines()[-1][:120])
    return ok, lines


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker", choices=TRACKERS + ("all",))
    parser.add_argument("--aspect", choices=("kiosk", "desktop", "both"), default="kiosk")
    parser.add_argument("--soak", type=float, default=0.0,
                        help="장시간 모드(분) — 사용 흐름을 되풀이하며 메모리·스레드를 잰다")
    args = parser.parse_args()
    if args.tracker == "all" or args.aspect == "both":
        # 트래커마다 새 프로세스 — 모듈 전역 상태와 가짜 부품이 서로 섞이지 않게
        names = TRACKERS if args.tracker == "all" else (args.tracker,)
        aspects = ("kiosk", "desktop") if args.aspect == "both" or args.tracker == "all" else (args.aspect,)
        procs = [(n, a, subprocess.Popen([sys.executable, os.path.abspath(__file__), n, "--aspect", a],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
                 for n in names for a in aspects]
        ok = True
        for n, a, p in procs:
            out, _ = p.communicate()
            sys.stdout.write(out.decode("utf-8", errors="replace"))
            ok = ok and p.returncode == 0
        print("모두 통과" if ok else "실패가 있습니다")
        return 0 if ok else 1
    real_stdin = sys.stdin
    try:
        ok, lines = run_one(args.tracker, args.aspect, soak_min=args.soak)
    finally:
        sys.stdin = real_stdin
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    os._exit(0 if ok else 1)   # 남은 데몬 스레드를 기다리지 않는다


if __name__ == "__main__":
    main()
