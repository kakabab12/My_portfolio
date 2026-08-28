"""MiDaS v2.1 small (ONNX) 단안 깊이 추정 래퍼.

원본 데모(TypeGPU)는 WebGPU 위에서 모델을 돌리지만, 여기서는 ONNX Runtime의
CPU 백엔드를 쓴다. 모델 자체는 같은 계열(경량 단안 깊이 추정)이다.

이 PC(Intel i5-10210U, 물리 4코어) 실측:
    intra_op 스레드  1개 149ms / 2개 100ms / 4개 69ms / 8개 114ms
스레드를 논리 코어 수(8)에 맞추면 오히려 느려진다 — 하이퍼스레딩으로 늘어난
논리 코어는 연산 유닛을 공유하므로, 이런 순수 계산 작업에서는 서로 방해만 한다.
그래서 **물리 코어 수**에 맞추는 게 맞다.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

_HERE = os.path.dirname(os.path.abspath(__file__))

# 모델을 둘 폴더 — weights/를 먼저 본다.
# ★2026-08-21: 원래 models/에 뒀는데 이 PC에서 그 폴더의 파일이 계속 사라졌다
# (4바이트 텍스트 파일까지 지워졌고, 폴더 목록은 0개인데 Test-Path는 True를
# 돌려주는 이상 상태였다 — 폴더 색인이 깨진 것으로 보인다). 같은 위치의 다른
# 이름 폴더(weights/)에서는 멀쩡히 유지돼서 그쪽으로 옮겼다.
_CANDIDATES = [os.path.join(_HERE, "weights", "midas_v21_small.onnx"),
               os.path.join(_HERE, "models", "midas_v21_small.onnx")]
DEFAULT_MODEL = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])

# MiDaS는 ImageNet 통계로 정규화된 입력을 기대한다 — 학습 때와 같은 전처리를
# 안 쓰면 깊이가 엉뚱하게 나온다(모델은 멀쩡한데 결과만 이상해져 찾기 어렵다)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
_MEAN255 = tuple(float(v) * 255.0 for v in (0.485, 0.456, 0.406))


def physical_core_count():
    """물리 코어 수 추정 — 위 설명대로 스레드 수를 여기에 맞춘다."""
    try:
        import subprocess
        out = subprocess.run(["wmic", "cpu", "get", "NumberOfCores"],
                             capture_output=True, text=True, timeout=5).stdout
        nums = [int(t) for t in out.split() if t.isdigit()]
        if nums:
            return max(nums)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 4) // 2)


class MidasDepth:
    """BGR 프레임 -> 깊이 맵(0~1, 클수록 가까움).

    MiDaS는 '역깊이(inverse depth)'를 내놓는다 — 값이 클수록 카메라에 가깝다.
    미터 단위 절대 거리가 아니라 상대적인 원근이며, 조명 계산에는 그걸로 충분하다.
    """

    def __init__(self, model_path=DEFAULT_MODEL, threads=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {model_path}\n"
                "  download_model.py를 실행하거나 아래 주소에서 내려받아 weights/에 두세요.\n"
                "  https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = threads or physical_core_count()
        self.threads = opts.intra_op_num_threads
        self.session = ort.InferenceSession(model_path, opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # 이 모델은 256x256 고정 입력이다(동적 크기를 지원하지 않아 다른 크기로는
        # 아예 실행이 안 된다) — 실측으로 확인했다
        self.size = int(shape[2]) if isinstance(shape[2], int) else 256

    def infer(self, frame_bgr):
        """프레임 1장 -> (깊이맵 0~1, 클수록 가까움). 크기는 self.size x self.size."""
        # cv2.dnn.blobFromImage가 축소·BGR->RGB·평균빼기·CHW 전치를 최적화된
        # C++ 한 번에 처리한다. 직접 numpy로 하면 전치 결과가 비연속 메모리라
        # 뒤따르는 정규화가 느려진다(실측 3.05ms -> 1.19ms)
        small = cv2.resize(frame_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            small, 1.0 / 255.0, (self.size, self.size),
            _MEAN255,   # OpenCV는 numpy 스칼라를 안 받는다 — 순수 float 튜플이어야 한다
            swapRB=True)
        np.divide(blob, _STD, out=blob)      # 채널별 표준편차는 blobFromImage가 못 한다
        out = self.session.run(None, {self.input_name: blob})[0]
        depth = np.squeeze(out).astype(np.float32)
        lo, hi = float(depth.min()), float(depth.max())
        return (depth - lo) / (hi - lo) if hi > lo else np.zeros_like(depth)


if __name__ == "__main__":
    import time

    m = MidasDepth()
    print(f"MiDaS v2.1 small | 입력 {m.size}x{m.size} | 스레드 {m.threads}개")
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    m.infer(frame)
    ts = []
    for _ in range(10):
        t = time.perf_counter()
        d = m.infer(frame)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print(f"1회 추정: 중앙 {ts[len(ts)//2]:.1f}ms  최악 {ts[-1]:.1f}ms")
    print(f"깊이 범위 {d.min():.2f}~{d.max():.2f}  (1에 가까울수록 카메라에 가까움)")
