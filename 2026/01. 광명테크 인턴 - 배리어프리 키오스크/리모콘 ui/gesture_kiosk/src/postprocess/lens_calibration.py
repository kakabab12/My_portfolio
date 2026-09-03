"""렌즈 자가 보정 — 사용자의 얼굴을 보정판 삼아 초점거리와 왜곡을 알아낸다.

왜 필요한가 (2026-09-03 신설)
-----------------------------
"광각 카메라에서도 되나"를 그동안 초점거리(focal_px)만 줄여서 시험했는데,
그것은 **틀린 시험**이었다. 초점거리는 배율일 뿐이고 상대 회전 매핑은 배율에
원리적으로 면역이라, 무엇을 넣어도 통과한다.

광각 렌즈의 진짜 문제는 **배럴 왜곡**이다. 가상 카메라에 왜곡을 넣어 재 보니
(tests/virtual_camera.py, 얼굴이 화면 옆+위에 있을 때 커서 오차):

    렌즈           가로     세로
    왜곡없음       4.17%    3.67%
    일반 65도      4.14%    3.16%
    광각 90도      5.05%    6.48%
    초광각 120도   6.31%   12.75%

왜곡을 되돌리면 이 오차가 줄어든다. 그런데 **왜곡 계수는 카메라마다 다르고,
현장에서 체커보드를 들고 측정할 수는 없다** — 이 프로젝트의 규칙은 "재지
않는다"이다.

그래서 **사람 얼굴을 보정판으로 쓴다.** 얼굴은 강체이고, 사용자는 어차피
키오스크 앞에서 이리저리 움직인다. 같은 강체를 화면 여러 곳에서 여러 자세로
본 관측이 쌓이면, 그것이 곧 카메라 보정에 필요한 자료다.

  · **Zhang, Z. (2000).** "A Flexible New Technique for Camera Calibration."
    *IEEE TPAMI* 22(11), 1330-1334. — 여러 시점의 관측에서 내부 파라미터를
    푸는 표준 기법. 여기서 쓰는 cv2.calibrateCamera의 바탕이다.
  · **Brown, D.C. (1966).** "Decentering Distortion of Lenses."
    *Photogrammetric Engineering*, 32(3), 444-462. — k1 방사왜곡 모형.
  · **Hartley, R., Zisserman, A. (2004).** *Multiple View Geometry*, 2nd ed.,
    Cambridge University Press, 6.1절 — 핀홀 모형과 원근 되돌리기의 근거.

무엇을 얻고 무엇에 쓰나
-----------------------
두 값을 얻는다. **둘 다 쓸모가 있고, 쓰임이 다르다.**

  1) k1 (왜곡 계수) — 휘어진 랜드마크를 펴는 데 쓴다.
  2) f  (초점거리)  — **원근을 되돌리는** 데 쓴다. 이쪽이 오히려 더 크게 듣는다.

2)를 설명하면: MediaPipe 랜드마크는 화면 좌표(x, y)와 상대 깊이(z)가 섞인
값이다. 이것을 그대로 3차원 점으로 보고 정합하면, 사람이 옆으로 걸어갔을 때
원근 단축이 **회전으로 오해**된다 — 고개를 안 돌렸는데 커서가 따라간다.

핀홀 모형에서 u = f*X/Z + cx 이고 MediaPipe의 z 는 f*dZ/Z0 에 해당하므로

    X 는 (u - cx) * (1 + z/f) 에 비례한다

즉 중심을 뺀 좌표에 (1 + z/f)를 곱하면 원근이 풀린다. 가상 카메라로 잰
몸 평행이동 끌림(작을수록 좋다, 한도 0.020):

    배치              되돌리기 전   참 f로    f가 40% 틀려도
    정면                0.0230      0.0062     0.0088
    밑에서 35도          0.0384      0.0059     0.0107
    심하게 비스듬히       0.0449      0.0078     0.0161

**f가 40% 틀려도 안 고친 것보다 훨씬 낫다.** 정확한 f가 필요한 게 아니라
대략만 맞으면 되므로, 자가 보정으로 충분하다.

안전장치 — 틀린 값을 쓰느니 안 쓴다
-----------------------------------
자가 보정은 **사용자가 충분히 움직여야** 성립한다. 얼굴이 화면 한 곳에만
머물면 왜곡의 증거가 없고, 그때 나오는 값은 부호까지 틀린다. 실제로 쟀다
(가상 카메라, 뷰 80개를 반으로 갈라 두 번 보정):

    움직임        전반 k1   후반 k1   차이     참값
    +-60mm       -0.413    -0.205    0.208    -0.30   <- 못 믿는다
    +-120mm      -0.295    -0.266    0.029    -0.30   <- 믿을 만하다
    +-200mm      -0.279    -0.279    0.001    -0.30   <- 믿을 만하다

그래서 다음을 모두 통과해야 채택한다.

  1) 뷰가 MIN_VIEWS개 이상 쌓였다
  2) 얼굴 중심이 화면에서 MIN_RADIUS_SPAN_PX 이상 퍼졌다 (움직임의 증거)
  3) **교차검증** — 뷰를 반으로 갈라 따로 보정했을 때 두 k1이 CROSS_TOL 안에서
     일치한다. 위 표가 보여주듯 이 하나가 못 믿을 경우를 전부 걸러낸다
  4) 재투영 오차(RMS)가 MAX_RMS_PX 이하다
  5) 값이 물리적으로 말이 되는 범위다 (f, k1 모두)
  6) 채택 뒤에는 더 시도하지 않는다. 실패해도 MAX_ATTEMPTS까지만 — 못 믿을
     상황에서 계속 CPU를 태우지 않는다

보정 계산은 140~270ms가 걸려 **프레임 루프에 넣을 수 없다.** 별도 스레드에서
돌리고, 끝나면 결과만 받는다.

그런데 딴 스레드에 두는 것만으로는 부족했다. 보정이 도는 동안 추론 대역의
부하를 재 보니 최대 멈춤이 6.78ms에서 109ms로 뛰었다 — CPU를 나눠 쓰기
때문이다. 그래서 보정 스레드의 **우선순위를 낮춘다**(_lower_priority).
보정은 몇 초 늦어도 아무 일이 없지만, 커서가 100ms 멈추면 사용자가 느낀다.

실제로 몇 번이나 도는가: 가상 카메라 흐름에서 **27.8초 만에 1회 실행하고
채택한 뒤 다시 돌지 않았다.** 상시 부담이 아니라 시작 한 번의 비용이다.

보정 한 번은 60뷰 기준 60~81ms이고, 교차검증까지 세 번 부르므로 한 덩어리로
두면 200ms 가까이 CPU를 붙잡는다. 사이사이 짧게 쉬어(_BREATH_SEC) 그 덩어리를
쪼갠다 — 총 시간은 그대로지만 커서가 한 번에 오래 멈추지 않는다.

(cv2.setNumThreads(1)로 OpenCV를 단일 스레드로 돌리는 것도 재 봤는데 오히려
나빴고 — 최대 지연 52ms 대 99ms — 전역 설정이라 다른 스레드까지 건드린다.
채택하지 않았다.)
"""
import math
import os
import sys
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:                          # 시험 환경에서 없을 수 있다
    cv2 = None

# 보정판으로 쓸 얼굴 점 — head_orientation.RIGID_LANDMARKS와 같은 22점.
# 말하기, 깜빡임, 눈썹처럼 움직이는 부위는 강체가 아니라 보정판이 못 된다
from src.postprocess.head_orientation import RIGID_LANDMARKS

# 정규 얼굴 모형(mm 어림). 보정판의 "실제 치수"에 해당한다.
# tests/virtual_camera.py의 FACE_MODEL과 같은 얼굴을 쓴다 — 두 곳이 같은
# 얼굴이어야 시험 결과를 그대로 읽을 수 있다
CANONICAL_FACE = {
    6: (0.0, -1.0, -10.0), 197: (0.0, 3.0, -13.0), 195: (0.0, 8.0, -17.0),
    5: (0.0, 13.0, -21.0), 4: (0.0, 18.0, -25.0), 168: (0.0, -6.0, -6.0),
    8: (0.0, -11.0, -3.0),
    33: (-32.0, 0.0, 4.0), 133: (-12.0, 1.0, 0.0),
    263: (32.0, 0.0, 4.0), 362: (12.0, 1.0, 0.0),
    234: (-46.0, 2.0, 22.0), 454: (46.0, 2.0, 22.0),
    10: (0.0, -58.0, 2.0), 151: (0.0, -46.0, -2.0), 9: (0.0, -34.0, -5.0),
    107: (-14.0, -30.0, -4.0), 336: (14.0, -30.0, -4.0),
    117: (-34.0, 20.0, 8.0), 346: (34.0, 20.0, 8.0),
    50: (-38.0, 12.0, 10.0), 280: (38.0, 12.0, 10.0),
}

# ── 안전장치 상수 (위 독스트링의 측정 참고)
MIN_VIEWS = 60               # 이보다 적으면 추정이 흔들린다 (40개 -0.242, 60개 -0.284)
MAX_VIEWS = 120              # 계산 시간 상한 — 뷰가 늘면 보정도 느려진다
MIN_RADIUS_SPAN_PX = 100.0   # 얼굴 중심이 이만큼은 퍼져야 한다 (+-60mm=63px는 실패)
MIN_VIEW_GAP_PX = 6.0        # 같은 자리의 뷰를 쌓아 봐야 새 정보가 없다
CROSS_TOL = 0.06             # 교차검증 허용 차이 (거부 0.082~0.208, 채택 0.001~0.029)
MAX_RMS_PX = 3.0             # 재투영 오차 상한
MIN_FOCAL_PX = 150.0         # 이보다 짧으면 어안 — 이 모형이 다루는 범위 밖
MAX_FOCAL_PX = 4000.0
MAX_ABS_K1 = 0.60            # 이보다 세면 Brown 1항 모형으로 감당이 안 된다
MAX_ATTEMPTS = 6             # 못 믿을 상황에서 계속 시도하지 않는다
_BREATH_SEC = 0.02           # 보정 사이에 쉬는 시간 — 커서에 자리를 내준다


class LensModel:
    """알아낸 렌즈 — 랜드마크를 펴고 원근을 되돌린다.

    focal_px  초점거리(픽셀)
    k1        Brown-Conrady 1차 방사왜곡 계수 (음수면 배럴)
    """

    __slots__ = ("focal_px", "k1", "width", "height", "mirrored")

    def __init__(self, focal_px, k1, width, height, mirrored=True):
        self.focal_px = float(focal_px)
        self.k1 = float(k1)
        self.width = float(width)
        self.height = float(height)
        self.mirrored = bool(mirrored)

    def rectify(self, points):
        """(N,3) 랜드마크 -> 왜곡과 원근을 되돌린 (N,3).

        순서가 중요하다 — 렌즈가 먼저 휘게 하고 센서가 배율을 먹였으므로,
        되돌릴 때는 **왜곡을 먼저** 풀고 그 다음에 원근을 편다.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return pts
        pts = pts.copy()
        cx, cy = self.width * 0.5, self.height * 0.5
        # 거울이 켜져 있으면 원래 방향으로 되돌려 놓고 계산한다 — 왜곡은
        # 렌즈의 성질이라 반전되기 **전** 좌표에서 걸린 것이다
        x = (self.width - pts[:, 0] if self.mirrored else pts[:, 0]) - cx
        y = pts[:, 1] - cy

        if self.k1:
            f = self.focal_px
            xu, yu = x / f, y / f
            xd, yd = xu.copy(), yu.copy()
            # 부동점 반복 — Brown 모형에는 닫힌 역함수가 없다. 6회면 수렴한다
            for _ in range(6):
                rr = xd * xd + yd * yd
                gain = 1.0 + self.k1 * rr
                # gain이 0에 가까우면 발산한다 — 물리적으로 못 오는 값이지만 막는다
                gain = np.where(np.abs(gain) < 0.2, 0.2, gain)
                xd, yd = xu / gain, yu / gain
            x, y = xd * f, yd * f

        # 원근 되돌리기 (독스트링 2) 참고)
        gain = 1.0 + pts[:, 2] / self.focal_px
        gain = np.where(gain < 0.2, 0.2, gain)      # 얼굴 뒤쪽까지 뒤집히지 않게
        x, y = x * gain, y * gain

        pts[:, 0] = (self.width - (x + cx)) if self.mirrored else (x + cx)
        pts[:, 1] = y + cy
        return pts

    def __repr__(self):
        return f"LensModel(focal_px={self.focal_px:.1f}, k1={self.k1:+.4f})"


class LensSelfCalibrator:
    """쓰는 동안 조용히 렌즈를 알아낸다. 못 믿겠으면 아무것도 하지 않는다.

        cal = LensSelfCalibrator(width, height, mirrored=True)
        cal.add(face.landmarks_3d)      # 매 프레임 (내부에서 알아서 걸러 쌓는다)
        if cal.model is not None:       # 채택되면 그때부터 값이 있다
            points = cal.model.rectify(points)
    """

    def __init__(self, width, height, mirrored=None, enabled=True):
        """mirrored=None이면 랜드마크를 보고 **스스로 판정한다**(아래 참고)."""
        self.width = float(width)
        self.height = float(height)
        self.mirrored = None if mirrored is None else bool(mirrored)
        self.enabled = bool(enabled) and cv2 is not None
        self._lock = threading.Lock()
        self._views = []              # [(N,2) 픽셀좌표]
        self._centers = []            # 각 뷰의 얼굴 중심
        self._last_center = None
        self._model = None
        self._thread = None
        self._attempts = 0
        self._done = False
        self.reject_reason = None     # 진단용 — 왜 안 됐는지

    # ------------------------------------------------------------------ 상태
    @property
    def model(self):
        with self._lock:
            return self._model

    @property
    def view_count(self):
        with self._lock:
            return len(self._views)

    @property
    def finished(self):
        with self._lock:
            return self._done

    def reset(self):
        """사람이 바뀌어도 렌즈는 그대로다 — 뷰만 비우고 결과는 남긴다."""
        with self._lock:
            self._views = []
            self._centers = []
            self._last_center = None

    # ------------------------------------------------------------------ 수집
    def add(self, landmarks_3d):
        """한 프레임의 랜드마크를 넣는다. 쌓을 값어치가 있을 때만 쌓는다."""
        if not self.enabled or self._done:
            return
        pts = self._pick(landmarks_3d)
        if pts is None:
            return
        center = pts.mean(axis=0)
        # 같은 자리에서 찍은 뷰는 새 정보가 없다 — 움직였을 때만 쌓는다
        if self._last_center is not None:
            if math.hypot(*(center - self._last_center)) < MIN_VIEW_GAP_PX:
                return
        with self._lock:
            if len(self._views) >= MAX_VIEWS:
                self._views.pop(0)
                self._centers.pop(0)
            self._views.append(pts)
            self._centers.append(center)
            ready = len(self._views) >= MIN_VIEWS
        self._last_center = center
        if ready:
            self._maybe_start()

    def _pick(self, landmarks_3d):
        """강체 22점만 (N,2)로. 거울이면 되돌려 놓는다 (왜곡은 반전 전의 성질)."""
        if landmarks_3d is None:
            return None
        try:
            arr = np.asarray(landmarks_3d, dtype=np.float64)
            picked = arr[list(RIGID_LANDMARKS)][:, :2]
        except (IndexError, ValueError, TypeError):
            return None
        if picked.shape != (len(RIGID_LANDMARKS), 2) or not np.all(np.isfinite(picked)):
            return None
        if self.mirrored is None:
            # ★거울 여부를 랜드마크에서 곧바로 읽는다 — 설정에 기대지 않는다.
            # 반전이 없으면 사람의 왼눈 바깥(33)이 오른눈 바깥(263)보다
            # 화면 왼쪽에 찍힌다. 반전하면 뒤집힌다.
            try:
                self.mirrored = bool(arr[33][0] > arr[263][0])
            except (IndexError, ValueError, TypeError):
                return None
        picked = picked.copy()
        if self.mirrored:
            picked[:, 0] = self.width - picked[:, 0]
        return picked

    # ------------------------------------------------------------------ 보정
    def _maybe_start(self):
        """조건이 되면 **딴 스레드에서** 보정한다 (140~270ms — 루프에 못 넣는다)."""
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._done:
                return
            if self._attempts >= MAX_ATTEMPTS:
                self._done = True
                self.reject_reason = "시도 횟수 소진"
                return
            centers = np.array(self._centers)
            views = list(self._views)
        # ★안전장치 2) — 얼굴이 화면에서 충분히 퍼졌나 (움직임의 증거).
        # 비싼 계산 **전에** 본다
        radius = np.hypot(centers[:, 0] - self.width * 0.5,
                          centers[:, 1] - self.height * 0.5)
        if float(radius.max() - radius.min()) < MIN_RADIUS_SPAN_PX:
            self.reject_reason = "움직임 부족"
            return
        with self._lock:
            self._attempts += 1
        self._thread = threading.Thread(target=self._solve, args=(views,), daemon=True)
        self._thread.start()

    def _solve(self, views):
        """딴 스레드에서 도는 몸통 — 교차검증까지 통과해야 채택한다."""
        _lower_priority()
        try:
            half = len(views) // 2
            whole = _calibrate(views, self.width, self.height)
            if whole is None:
                self.reject_reason = "보정 실패"
                return
            focal, k1, rms = whole
            # ★안전장치 4) 5)
            if rms > MAX_RMS_PX:
                self.reject_reason = f"재투영오차 {rms:.2f}px"
                return
            if not (MIN_FOCAL_PX <= focal <= MAX_FOCAL_PX) or abs(k1) > MAX_ABS_K1:
                self.reject_reason = "값이 범위 밖"
                return
            # ★안전장치 3) — 교차검증. 이 하나가 못 믿을 경우를 전부 걸러낸다
            time.sleep(_BREATH_SEC)             # 커서에 자리를 내준다
            first = _calibrate(views[:half], self.width, self.height)
            time.sleep(_BREATH_SEC)
            second = _calibrate(views[half:], self.width, self.height)
            if first is None or second is None:
                self.reject_reason = "교차검증 실패"
                return
            gap = abs(first[1] - second[1])
            if gap > CROSS_TOL:
                self.reject_reason = f"교차검증 불일치 {gap:.3f}"
                return
            with self._lock:
                self._model = LensModel(focal, k1, self.width, self.height,
                                        bool(self.mirrored))
                self._done = True
                self.reject_reason = None
        except Exception as exc:            # 이 스레드가 죽어도 커서는 살아야 한다
            self.reject_reason = f"예외 {type(exc).__name__}: {exc}"


def _lower_priority():
    """지금 스레드를 뒤로 물린다. 안 되면 그냥 넘어간다(있으면 좋은 것일 뿐).

    보정이 CPU를 다투면 커서가 멈춘다 — 추론 대역 부하의 최대 지연이
    6.78ms에서 109ms로 뛰는 것을 쟀다. 보정은 몇 초 늦어도 상관없다.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            THREAD_PRIORITY_LOWEST = -2
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(),
                                       THREAD_PRIORITY_LOWEST)
        elif hasattr(os, "nice"):
            os.nice(10)
    except Exception:
        pass          # 우선순위를 못 낮춰도 보정 자체는 해야 한다


def _calibrate(views, width, height):
    """뷰 묶음 -> (focal_px, k1, rms). 못 풀면 None.

    **2모수(f, k1)만 푼다.** k2까지 풀면 둘이 서로를 상쇄해 오히려 나빠진다 —
    가상 카메라로 쟀다 (참 k1=-0.30일 때 2모수 -0.284, 3모수 -0.236).
    """
    if cv2 is None or len(views) < 4:
        return None
    model = np.array([CANONICAL_FACE[i] for i in RIGID_LANDMARKS], dtype=np.float32)
    obj = [model.copy() for _ in views]
    img = [np.asarray(v, dtype=np.float32).copy() for v in views]
    # 초기 추정 — 여기서 출발해 수렴한다. 틀려도 된다는 것을 확인했다
    # (모든 렌즈에 f=700으로 출발시켜도 참값 부근으로 갔다)
    guess = np.array([[700.0, 0.0, width * 0.5],
                      [0.0, 700.0, height * 0.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    try:
        rms, mtx, dist, _r, _t = cv2.calibrateCamera(
            obj, img, (int(width), int(height)), guess, np.zeros(5), flags=flags)
    except cv2.error:
        return None
    focal = float(mtx[0, 0])
    k1 = float(np.asarray(dist).ravel()[0])
    if not (math.isfinite(focal) and math.isfinite(k1) and math.isfinite(rms)):
        return None
    return focal, k1, float(rms)
