"""카메라 -> AI 깊이 추정 -> 깊이에 반응하는 실시간 조명 (CPU 전용).

원본(Konrad Reczko, TypeGPU/WebGPU)과의 차이 — 왜 구조가 달라야 하는가
-----------------------------------------------------------------------
원본의 자랑은 "AI 추론 -> 조명 -> 렌더링을 command encoder 하나로 이어서
처리하고, 중간 결과를 GPU 밖으로 꺼내지 않는다"이다. GPU에서 깊이 추정이 8ms면
한 프레임(16ms) 안에 전부 끝나므로 순차 처리가 성립한다.

CPU에서는 그 전제가 무너진다:
  · "GPU 밖으로 안 꺼낸다"는 최적화 자체가 의미가 없다 — 처음부터 다 같은 메모리다.
  · 대신 깊이 추정이 훨씬 느리다(이 PC 실측 70~160ms). 순차로 이으면 화면 전체가
    그 속도로 떨어져, 카메라 영상이 뚝뚝 끊겨 보인다.

그래서 여기서는 **깊이 추정만 따로 떼어 자기 속도로 돌린다**:
  · 화면과 조명은 카메라 속도(30fps)로 매끄럽게 계속 돌고,
  · 깊이 맵은 준비되는 대로(8~12fps) 갈아 끼운다.
깊이는 장면이 크게 안 바뀌면 몇 프레임 묵어도 티가 안 나는 정보라 이 거래가
성립한다 — 반면 영상이 끊기는 건 바로 보인다. 이 프로젝트의 gesture_kiosk도
같은 이유로 추론과 렌더를 분리했다.

--sync 옵션을 주면 원본처럼 한 줄로 순차 처리해서, 두 방식의 체감 차이를
직접 비교할 수 있다.

조작: q/ESC 종료 · d 깊이맵 보기 · l 조명 끄기 · space 광원 회전 멈춤
"""
import argparse
import threading
import time

import cv2
import numpy as np

from depth_lighting import DepthLighting
from depth_model import MidasDepth
from depth_stabilizer import DepthStabilizer

LIGHT_ORBIT_SEC = 6.0      # 광원이 한 바퀴 도는 데 걸리는 시간
LIGHT_RADIUS = 0.75        # 광원이 도는 반경(화면 -1~1 좌표)
LIGHT_Z = 0.12             # 광원의 기본 거리 — 사람(0.35)보다 앞이라 얼굴이 밝아진다
SHADE_SIZE = 128           # 조명 계산 해상도 — 실측으로 고른 값(아래 설명)
# 조명을 깊이 맵과 같은 256에서 계산하면 17.8ms, 128이면 11.7ms, 96이면 오히려
# 17.1ms로 다시 나빠진다(작을수록 계산은 주는데 화면 크기로 늘리는 비용이 커져서
# 어느 지점부터 손해로 뒤집힌다). 128이 그 골짜기 바닥이다.


class DepthWorker:
    """깊이 추정을 자기 속도로 돌리는 백그라운드 일꾼.

    최신 프레임 하나만 붙들고 있다가 추정이 끝나면 결과를 갈아 끼운다 — 밀린
    프레임을 큐에 쌓아 뒤늦게 처리하면 깊이가 점점 과거를 가리키게 되므로,
    큐를 두지 않고 **항상 최신 프레임만** 쓴다.
    """

    def __init__(self, model):
        self.model = model
        self._frame = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self.depth = None
        self.infer_ms = 0.0
        self.count = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def submit(self, frame):
        with self._new_frame:
            self._frame = frame
            self._new_frame.notify()

    def _loop(self):
        while not self._stop:
            with self._new_frame:
                while self._frame is None and not self._stop:
                    self._new_frame.wait(timeout=0.5)
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            t = time.perf_counter()
            try:
                depth = self.model.infer(frame)
            except Exception:
                continue
            elapsed = (time.perf_counter() - t) * 1000.0
            with self._lock:
                self.depth = depth
                # 지수이동평균 — 한 번 튄 값에 표시가 요동치지 않게
                self.infer_ms = elapsed if self.count == 0 else self.infer_ms * 0.8 + elapsed * 0.2
                self.count += 1

    def latest(self):
        with self._lock:
            return self.depth, self.infer_ms, self.count

    def stop(self):
        self._stop = True
        with self._new_frame:
            self._new_frame.notify_all()


def draw_hud(canvas, lines):
    """왼쪽 위에 반투명 판을 깔고 흰 글씨로 상태를 적는다."""
    pad, lh = 8, 20
    h = pad * 2 + lh * len(lines)
    w = 330
    roi = canvas[0:h, 0:w]
    cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, roi)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (pad, pad + lh * (i + 1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def flatten_lighting(bgr, sigma=40):
    """입력에 이미 구워져 있는 조명을 걷어내 '재질 색'에 가깝게 만든다.

    왜 필요한가: 밝은 흰 벽을 등지고 찍으면 카메라가 벽에 노출을 맞춰서 얼굴이
    아주 어둡게 담긴다(실측: 얼굴 15~47 / 벽 124~192, 최대 8.6배 차이). 그 상태로
    우리 빛을 아무리 세게 비춰도 얼굴은 안 밝아진다 — 애초에 담긴 정보가 없어서
    키워봐야 잡음만 커진다. 실측으로 확인한 한계였다.

    크게 흐린 자기 자신으로 나누면 넓은 범위의 밝기 차(=원래 조명)는 사라지고
    무늬·색만 남는다. 그러면 우리 빛이 밝기를 온전히 결정할 수 있다. 사진에서
    역광 보정할 때 쓰는 것과 같은 원리이고, 실측으로 8.6배 차이가 2.4배까지 줄었다.

    ★근본 해결은 아니다. 가장 확실한 건 흰 벽을 등지지 않거나 얼굴 쪽에 불을
    켜는 것이다 — 잘 노출된 입력에서는 이 보정 없이도 원본 데모와 같은 그림이 나온다.
    """
    # 속도: 큰 흐림을 원본 크기에서 하면 프레임이 반토막 난다(실측 20fps -> 4.5fps).
    # 흐림 결과는 원래 뭉개진 그림이라 작게 계산해서 키워도 똑같다 — 조명 계산을
    # 작은 해상도에서 하는 것과 같은 요령이다
    f = bgr.astype(np.float32)
    h, w = f.shape[:2]
    tiny = cv2.resize(f, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    base = cv2.GaussianBlur(tiny, (0, 0), max(sigma / 8.0, 1.0))
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(f / (base + 1e-3) * float(base.mean()), 0, 255).astype(np.uint8)


class LightControl:
    """광원 위치를 사람이 직접 정하게 한다 — 원본 데모의 핵심 조작.

    원본에서 가장 눈에 띄는 건 빛나는 점을 **마우스로 끌고 다니는** 것이다.
    자동으로 도는 것과 체감이 완전히 다르다 — 내가 옮긴 곳에 맞춰 얼굴의 명암과
    그림자가 즉시 따라오니 "AI가 공간을 이해하고 있다"가 바로 와닿는다.

    x, y는 화면 좌표(-1~1), z는 카메라로부터의 거리(0=코앞, 1=배경).
    z는 휠로 조절한다 — 빛을 사람 앞뒤로 옮기면 조명이 확 달라지는 게 재미있다.
    """

    def __init__(self, x=0.45, y=-0.10, z=0.12):
        self.x, self.y, self.z = x, y, z
        self.dragging = False
        self.auto = False          # 마우스를 안 쓰면 예전처럼 자동으로 돌 수도 있게
        # ★깊이 자동 잡기 (2026-08-21 사용자 요청 — "뎁스는 오토로 안 될려나,
        # 가까이 가면 사라지고 이런 느낌으로").
        #
        # 켜면 광원을 **가리킨 표면 바로 앞**에 놓는다. 벽을 가리키면 빛이 벽에
        # 가서 붙고, 그 상태로 얼굴이나 손을 앞으로 내밀면 그것들이 빛을 가려
        # 사라진다 — 빛이 화면에 붙은 게 아니라 방 안 한 자리에 놓여 있다는
        # 느낌이 여기서 나온다.
        #
        # 이게 가능해진 건 깊이 안정화(scale-shift 정렬)를 넣은 뒤부터다. 그전엔
        # 모델이 매 프레임 다시 정규화해서 "장면 속 고정된 깊이"라는 개념 자체가
        # 성립하지 않았다 — 사람이 조금만 움직여도 기준이 통째로 바뀌었다.
        self.auto_z = True
        self.z_offset = 0.06       # 가리킨 표면보다 이만큼 앞에 둔다(표면에 파묻히지 않게)
        self._anchored = False     # 한 번이라도 깊이를 정했는지

    def on_mouse(self, event, mx, my, flags, param):
        W, H = param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging, self.auto = True, False
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            # 휠을 굴리면 빛이 앞뒤로 — 위로 굴리면 카메라 쪽(가까이).
            # 손으로 조절하는 순간 자동 깊이는 꺼진다(수동이 우선)
            self.auto_z = False
            delta = 0.04 if flags > 0 else -0.04
            self.z = float(np.clip(self.z - delta, 0.0, 1.2))
        if self.dragging and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.x = float(np.clip((mx / max(W - 1, 1)) * 2 - 1, -1.3, 1.3))
            self.y = float(np.clip((my / max(H - 1, 1)) * 2 - 1, -1.3, 1.3))

    def follow_surface(self, depth_small, to_distance):
        """광원을 **끌고 있는 동안에만** 가리키는 표면 앞에 붙인다. 놓으면 그 자리에 남는다.

        ★2026-08-21 버그 수정 — 처음엔 매 프레임 커서 자리의 깊이를 다시 읽었다.
        그러면 손을 광원 앞으로 가져가는 순간 광원이 **손 앞으로 도망가서**,
        "가까이 가면 빛이 가려진다"가 영원히 일어나지 않는다. 빛이 물체를
        피해 다니니 장면 안에 놓인 게 아니라 커서에 붙어 다니는 꼴이었다.

        빛은 **놓아둔 자리에 머물러야** 가려질 수 있다. 그래서 깊이를 정하는 건
        끌고 있는 동안(과 처음 한 번)뿐이고, 손을 떼면 그 깊이에 고정된다.

        depth_small : 조명 계산에 쓰는 작은 깊이 맵 (1=가까움)
        to_distance : 깊이를 거리로 바꾸는 함수 (DepthLighting.to_distance)
        """
        if not self.auto_z:
            return
        # 끌고 있지 않으면 그 자리에 그대로 — 이게 가림이 성립하는 조건이다
        if not self.dragging and self._anchored:
            return
        h, w = depth_small.shape
        ix = int(np.clip((self.x * 0.5 + 0.5) * (w - 1), 0, w - 1))
        iy = int(np.clip((self.y * 0.5 + 0.5) * (h - 1), 0, h - 1))
        # 한 점만 읽으면 깊이 잡음에 광원이 덜덜 떨린다 — 작은 창의 중앙값을 쓴다
        x0, x1 = max(0, ix - 2), min(w, ix + 3)
        y0, y1 = max(0, iy - 2), min(h, iy + 3)
        surface = to_distance(float(np.median(depth_small[y0:y1, x0:x1])))
        target = max(0.0, surface - self.z_offset)
        # 끌기 시작한 첫 프레임엔 곧바로, 그 뒤엔 부드럽게 따라간다
        self.z = target if not self._anchored else self.z + 0.5 * (target - self.z)
        self._anchored = True

    def xyz(self):
        return (self.x, self.y, self.z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--sync", action="store_true",
                    help="원본처럼 한 줄로 순차 처리(추론->조명->표시). 비교용")
    ap.add_argument("--seconds", type=float, default=0.0, help="이 시간 뒤 자동 종료(측정용)")
    ap.add_argument("--raw-depth", action="store_true",
                    help="깊이 안정화·경계정렬을 끄고 모델 출력을 그대로 쓴다(비교용)")
    ap.add_argument("--relight", action="store_true",
                    help="역광 보정 — 입력에 이미 들어있는 조명을 걷어내고 우리 빛만 남긴다. "
                         "흰 벽을 등지고 찍어 얼굴이 어둡게 나올 때 쓴다(아래 설명 참고)")
    ap.add_argument("--threads", type=int, default=None,
                    help="깊이 추정에 쓸 스레드 수. 화면 루프도 CPU를 쓰므로 물리 코어"
                         "수보다 하나 적게 주는 편이 전체적으로 나을 수 있다")
    args = ap.parse_args()

    model = MidasDepth(threads=args.threads)
    lighting = DepthLighting()
    # 깊이 흔들림 제거 + 경계 정렬 — depth_stabilizer.py 설명 참고
    stabilizer = DepthStabilizer(align=False,
                                 ema=0.0 if args.raw_depth else 0.35,
                                 guided=not args.raw_depth)
    print(f"MiDaS v2.1 small | 입력 {model.size}x{model.size} | 스레드 {model.threads}개")
    print(f"방식: {'순차(원본과 같은 구조)' if args.sync else '분리(깊이만 따로, 화면은 카메라 속도)'}")

    cap = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("카메라를 열지 못했습니다.")
        return 1

    worker = None if args.sync else DepthWorker(model).start()
    show_depth, use_light = False, True
    light = LightControl()
    WIN = "depth reactive lighting (CPU)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, args.width, args.height)
    cv2.setMouseCallback(WIN, light.on_mouse, (args.width, args.height))

    start = time.monotonic()
    frames, fps_ema, last = 0, 0.0, time.perf_counter()
    infer_ms_sync, angle = 0.0, 0.0
    surface_dist, occluded = 0.0, False
    display_ms = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)          # 거울 모드 — 내 움직임과 화면을 일치시킨다
            if args.relight:
                frame = flatten_lighting(frame)
            tick = time.perf_counter()

            if args.sync:
                t = time.perf_counter()
                depth = model.infer(frame)
                infer_ms_sync = infer_ms_sync * 0.8 + (time.perf_counter() - t) * 1000 * 0.2 \
                    if infer_ms_sync else (time.perf_counter() - t) * 1000
                infer_ms, count = infer_ms_sync, frames
            else:
                worker.submit(frame)
                depth, infer_ms, count = worker.latest()

            if depth is None:
                cv2.putText(frame, "waiting for first depth...", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                canvas = frame
            else:
                if light.auto:      # 마우스를 안 건드리면 예전처럼 자동으로 돈다
                    a = (time.monotonic() - start) / LIGHT_ORBIT_SEC * 2 * np.pi
                    light.x, light.y = LIGHT_RADIUS * np.cos(a), LIGHT_RADIUS * np.sin(a)
                # 흔들림 제거 + 컬러 영상 기준 경계 정렬을 한 번에.
                # 단순 resize를 대신한다 — 안정화는 사실상 공짜고 경계 정렬은 1.5ms
                small = stabilizer(depth, frame, (SHADE_SIZE, SHADE_SIZE))
                if show_depth:
                    d8 = (small * 255).astype(np.uint8)
                    canvas = cv2.applyColorMap(
                        cv2.resize(d8, (frame.shape[1], frame.shape[0])), cv2.COLORMAP_INFERNO)
                light.follow_surface(small, lighting.to_distance)
                # 지금 광원이 가려져 있는지 — 화면에 보여주면 "왜 구체가 안 보이지"를
                # 바로 알 수 있다(빛이 물체 뒤로 들어간 것이지 고장난 게 아니다)
                _ix = int(np.clip((light.x * .5 + .5) * (SHADE_SIZE - 1), 0, SHADE_SIZE - 1))
                _iy = int(np.clip((light.y * .5 + .5) * (SHADE_SIZE - 1), 0, SHADE_SIZE - 1))
                surface_dist = lighting.to_distance(float(small[_iy, _ix]))
                occluded = surface_dist < light.z
                if show_depth:
                    pass
                elif use_light:
                    # 빛나는 구체는 shade_frame이 직접 그린다(_add_glow) —
                    # 나중에 원을 덧그리면 빛무리와 따로 놀아 붙여놓은 티가 난다
                    canvas = lighting.shade_frame(frame, small, light.xyz())
                else:
                    canvas = frame

            display_ms = display_ms * 0.9 + (time.perf_counter() - tick) * 1000 * 0.1
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            fps_ema = inst if frames == 0 else fps_ema * 0.9 + inst * 0.1
            frames += 1

            draw_hud(canvas, [
                f"screen {fps_ema:5.1f} fps    depth {1000.0 / max(infer_ms, 1e-6):4.1f} fps",
                f"depth infer {infer_ms:6.1f} ms   frame work {display_ms:5.1f} ms",
                f"light z {light.z:.2f}   surface here {surface_dist:.2f}   "
                f"{'HIDDEN behind object' if occluded else 'visible'}",
                "drag=move  wheel=depth  z=auto-depth  d=depthmap  a=auto  q=quit",
            ])
            cv2.imshow(WIN, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_depth = not show_depth
            if key == ord("l"):
                use_light = not use_light
            if key == ord("a"):
                light.auto = not light.auto
            if key == ord("z"):
                light.auto_z = not light.auto_z
                if light.auto_z:
                    light._anchored = False   # 켜는 순간 지금 가리킨 표면에 다시 붙는다
            if args.seconds and time.monotonic() - start > args.seconds:
                break
    finally:
        if worker:
            worker.stop()
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - start
    _, infer_ms, count = (None, infer_ms_sync, frames) if args.sync else worker.latest()
    print(f"\n{elapsed:.1f}초 동안")
    print(f"  화면      {frames}프레임  =  {frames / elapsed:.1f} fps")
    print(f"  깊이 추정 {count}회       =  {count / elapsed:.1f} fps  (1회 {infer_ms:.0f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
