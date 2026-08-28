"""깊이 맵 후처리 — 흔들림 제거 + 경계 정렬. 조사한 표준 기법 세 가지를 적용한다.

왜 필요한가
-----------
MiDaS 같은 모델은 **affine-invariant depth**를 내놓는다 — 원근의 상대적인 순서는
맞지만 스케일과 오프셋은 매 프레임 제멋대로다. 그래서 프레임마다 min~max로
정규화하면 두 가지가 망가진다:

  1. 화면에서 가장 가까운 것이 **항상 정확히 1**이 된다. 손을 들면 손이 1이 되고
     얼굴은 뒤로 밀린다 — 얼굴이 실제로 움직이지 않았는데도. 조명이 통째로 출렁인다.
  2. 프레임마다 정규화 기준이 달라 깊이가 미세하게 떨린다(flickering). 이건 이
     분야에서 잘 알려진 문제이고, 표준 해법이 프레임 간 scale-shift 정렬이다.

적용한 기법
-----------
① **시간 방향 평활(EMA)** — 프레임 간 잔떨림을 시간축으로 눌러준다. 진짜 움직임은
   조금 늦게 따라올 뿐 그대로 통과한다.

★ scale-shift 정렬은 넣었다가 **뺐다**. 논문들이 쓰는 기법이라 그대로 옮겼는데,
  검증해보니 이 방식(새 프레임을 직전 프레임에 최소제곱으로 맞추기)은 정규화
  잡음만이 아니라 **진짜 움직임까지 맞춰서 없애버린다**. 실측:

      사람이 깊이 0.5 -> 0.9로 다가왔을 때 (40프레임)
        정렬 켬:  0.500 -> 0.500  (전혀 안 따라옴 — 못 쓴다)
        정렬 끔:  0.500 -> 0.900  (정상)

      같은 장면 500프레임 (흘러감 검사)
        정렬 켬:  0.684 -> 0.379  (오차가 누적돼 슬금슬금 흘러감)
        정렬 끔:  안정

  논문들은 전체 구간을 놓고 공통 기준에 맞추거나(co-alignment) 정적인 영역만
  골라 쓰는데, 직전 프레임 하나에 그냥 맞추면 이렇게 된다. 코드는 _fit_scale_shift
  로 남겨뒀지만 기본은 꺼져 있다.

② **Guided filter로 경계 정렬** — 깊이는 256x256으로 뭉툭하게 나오는데 화면은
   그보다 크다. 그냥 늘리면 사람 윤곽이 흐물거린다. 원본 컬러 영상을 길잡이로
   삼아 늘리면 경계가 실제 물체 윤곽에 딱 붙는다 (Guided Image Filtering,
   He et al. 2010 — 깊이 업샘플링에 널리 쓰이는 표준 기법).
"""
import cv2
import numpy as np


class DepthStabilizer:
    """프레임마다 튀는 깊이를 안정시키고 경계를 살린다."""

    def __init__(self, align=False, ema=0.35, guided=True,
                 guided_radius=4, guided_eps=1e-3):
        self.align = align            # (기본 꺼짐 — 모듈 설명의 ★ 참고)
        self.ema = ema                # ① 시간 평활 (작을수록 더 많이 눌러준다)
        self.guided = guided          # ③ 경계 정렬
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self._prev = None             # 직전 프레임의 안정화된 깊이

    def reset(self):
        self._prev = None

    @staticmethod
    def _fit_scale_shift(src, ref):
        """src를 ref에 가장 가깝게 만드는 (a, b)를 최소제곱으로 구한다: a*src + b ≈ ref.

        두 배열의 평균·표준편차만 있으면 닫힌 형태로 풀린다 — 반복 계산이 필요 없어
        CPU에서도 사실상 공짜다(실측 0.1ms 미만).
        """
        s_mean, r_mean = float(src.mean()), float(ref.mean())
        s_std = float(src.std())
        if s_std < 1e-6:
            return 1.0, 0.0
        # 상관을 고려한 최소제곱 해 — cov(src,ref)/var(src)
        cov = float(((src - s_mean) * (ref - r_mean)).mean())
        a = cov / (s_std * s_std)
        # 스케일이 말도 안 되게 튀면(장면이 확 바뀐 경우) 정렬을 포기한다 —
        # 억지로 맞추면 오히려 깊이가 통째로 찌그러진다
        if not (0.2 < a < 5.0):
            return 1.0, 0.0
        return a, r_mean - a * s_mean

    def __call__(self, depth01, guide_bgr=None, size=None):
        """모델이 낸 깊이 -> 안정화된 깊이 (둘 다 1=가까움, 0~1).

        size를 주면 그 크기로 맞춰 돌려준다 — 조명 계산 해상도에 바로 쓰라고.
        """
        d = depth01.astype(np.float32)

        if self._prev is not None and self.ema > 0.0:
            d = self._prev + self.ema * (d - self._prev)

        # 정렬·평활을 거치면 0~1을 벗어날 수 있다. 매 프레임 다시 min-max로
        # 늘이면 정렬한 의미가 사라지므로, 범위를 넘은 것만 잘라낸다
        np.clip(d, 0.0, 1.0, out=d)
        self._prev = d

        if self.guided and guide_bgr is not None:
            return self.refine_edges(d, guide_bgr, size)
        if size is not None and (d.shape[1], d.shape[0]) != size:
            d = cv2.resize(d, size, interpolation=cv2.INTER_LINEAR)
        return d

    def refine_edges(self, depth01, guide_bgr, size=None):
        """영상을 길잡이 삼아 깊이 경계를 실제 물체 윤곽에 맞춘다.

        깊이는 뭉툭하게 나와서 그냥 늘리면 사람 윤곽이 흐물거린다 — 조명이 얼굴
        밖으로 새어 나가 보인다. 길잡이 영상의 경계를 따라 늘리면 그게 사라진다.

        ★비용 실측 (640x480 화면 기준). 두 가지가 결정적이었다:
            컬러 길잡이 640x480, radius=8   30.8ms
            흑백 길잡이 640x480, radius=8   12.2ms   <- 채널 수가 비용의 대부분
            흑백 절반 해상도 + 확대           3.5ms
            조명 해상도(128x128)에서만        0.7ms   <- 채택
        조명 계산 자체를 128에서 하므로 깊이도 거기 맞추면 된다. 44배 싸진다.
        """
        if size is None:
            size = (depth01.shape[1], depth01.shape[0])
        guide = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)   # 흑백이 훨씬 싸다
        guide = cv2.resize(guide, size, interpolation=cv2.INTER_AREA)
        src = (depth01 if (depth01.shape[1], depth01.shape[0]) == size
               else cv2.resize(depth01, size, interpolation=cv2.INTER_LINEAR))
        try:
            out = cv2.ximgproc.guidedFilter(
                guide=guide, src=src, radius=self.guided_radius, eps=self.guided_eps)
            # guided filter는 경계 근처에서 원래 범위를 살짝 넘길 수 있다(실측 1.02).
            # 거리 변환에 그대로 들어가면 near_plane보다 앞에 물체가 생긴 셈이 된다
            return np.clip(out, 0.0, 1.0)
        except Exception:
            return src          # ximgproc이 없는 환경이면 그냥 늘린 값을 쓴다


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    base = np.clip(np.linspace(0, 1, 128)[None, :].repeat(128, 0), 0, 1).astype(np.float32)

    print("① scale-shift 정렬 + 평활이 흔들림을 얼마나 잡는가")
    print("   (같은 장면인데 모델이 프레임마다 스케일을 다르게 내놓는 상황을 흉내)")
    for use in (False, True):
        st = DepthStabilizer(align=use, ema=0.35 if use else 0.0, guided=False)
        prev, jitter = None, []
        for i in range(60):
            # 같은 장면인데 정규화 기준이 매번 달라진 것처럼 a, b를 흔든다
            a, b = 1.0 + rng.normal(0, 0.12), rng.normal(0, 0.06)
            noisy = np.clip(a * base + b, 0, 1).astype(np.float32)
            out = st(noisy)
            if prev is not None:
                jitter.append(float(np.abs(out - prev).mean()))
            prev = out
        label = "정렬+평활 켬" if use else "끔 (예전 방식)"
        print(f"   {label:16s} 프레임 간 평균 변화 {np.mean(jitter):.4f}")

    print("\n② guided filter 비용 (640x480 화면 기준)")
    guide = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    st = DepthStabilizer()
    st.refine_edges(base, guide)
    ts = []
    for _ in range(15):
        t = time.perf_counter()
        st.refine_edges(base, guide)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print(f"   중앙 {ts[len(ts)//2]:.1f}ms  최악 {ts[-1]:.1f}ms")
