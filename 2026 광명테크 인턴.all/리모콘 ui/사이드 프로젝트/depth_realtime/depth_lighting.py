"""깊이 맵 -> 깊이에 반응하는 실시간 조명 (CPU, numpy 배열 연산).

원본 데모(TypeGPU)의 조명 단계를 CPU에서 재현. 원본은 GPU 셰이더가 픽셀마다
돌지만, CPU에서 픽셀 반복문은 재앙이라 "모든 픽셀을 한 번에" 계산한다.

깊이 값의 약속 (★중요)
-----------------------
이 모듈에 넘기는 depth01은 **1이 가깝고 0이 멀다** — MiDaS가 내놓는 방향
그대로다. 처음엔 이 모듈만 반대(0이 가깝다)로 가정하고 있었고, 그래서 실제
앱에서는 원근이 뒤집힌 채로 조명이 계산됐다. 그럴듯한 그림이 나와서 한동안
못 알아챘다 — 규약이 어긋나도 "틀린 티가 안 나는" 종류의 버그다.
내부 계산은 카메라로부터의 거리(dist = 1 - depth01, 0이 가깝다)로 바꿔서 한다.

좌표계: 화면 가로/세로는 -1~1. z는 카메라로부터의 거리로 0(코앞)~1(배경).
광원의 z도 같은 자로 잰다 — 0.35면 사람과 배경 사이쯤에 떠 있는 셈이다.
"""
import cv2
import numpy as np


class DepthLighting:
    """깊이 맵 + 원본 프레임 -> 조명이 적용된 프레임. 버퍼를 재사용한다."""

    def __init__(self, base=0.85, ambient=0.0, specular=0.30, falloff=4.0,
                 color=(0.72, 0.86, 1.0), intensity=5.0,
                 shadow_steps=8, shadow_bias=0.02, shadow_min=0.25,
                 near_plane=0.35, far_plane=1.0, max_slope=0.035):
        # ★2026-08-21 재설계 — 방의 원래 조명을 살려두고 **빛을 더한다**.
        #
        # 처음엔 화면 전체에 밝기를 곱하는 방식이었다(out = frame x shade). 그러면
        # 빛이 안 닿는 곳은 전부 캄캄해져서, 참고 영상처럼 "평상시 방에 빛이 하나
        # 더 생긴" 느낌이 아니라 "암흑 속 손전등"이 된다. 사용자 지적 그대로다.
        # 게다가 원본이 어둡게 찍힌 얼굴은 곱셈으로 더 어두워져 영영 안 밝아졌다.
        #
        # base = 원래 화면을 얼마나 살려둘지(1.0이면 그대로). 여기에 광원의
        # 기여분을 **더해서** 올린다 — 실제로 방에 조명을 하나 더 켠 것과 같다.
        self.base = base
        self.ambient = ambient          # 빛과 무관하게 더하는 최소 밝기(보통 0)
        self.specular = specular        # 반짝임 세기
        self.falloff = falloff          # 거리 감쇠 — 클수록 빛이 좁게 떨어진다
        # base/intensity/falloff 기본값은 실기 실측으로 골랐다. 더하는 방식이라
        # 세게 올리면 밝은 벽부터 하얗게 타버린다 — 광원 근처 밝아짐 / 타버린 픽셀:
        #   세기 2.6  1.41배 / 0.0%
        #   세기 5.0  2.10배 / 0.0%   <- 채택 (효과는 두 배, 타는 곳 없음)
        #   세기 8.0  2.75배 / 2.9%
        #   세기 14   2.83배 / 55.5%  (효과는 더 안 늘고 화면만 탄다)
        # 빛 색 (B, G, R 배수). 파랑을 줄이고 빨강을 남기면 백열등 같은 따뜻한 빛이 된다
        self.color = np.array(color, np.float32)
        self.intensity = intensity      # 빛 세기
        self.shadow_steps = shadow_steps    # 그림자 판정 행진 횟수 (0이면 그림자 끔)
        self.shadow_bias = shadow_bias      # 자기 그림자 방지 여유값
        self.shadow_min = shadow_min        # 그림자 속 최소 밝기(0이면 새까맣다)
        # ★장면을 카메라에서 얼마나 떨어진 구간에 놓을지 (near_plane ~ far_plane).
        #
        # 이게 없으면 얼굴이 절대 안 밝아진다. MiDaS는 **매 프레임 0~1로 정규화**해서
        # 내놓기 때문에, 화면에서 가장 가까운 것(보통 사람 얼굴)이 항상 정확히
        # 거리 0이 된다. 그러면 광원을 사람 **앞**에 둘 자리가 아예 없어서, 빛을
        # 어디에 놓든 항상 사람 뒤쪽이 된다 — 뒤에서 비추니 얼굴은 계속 어둡고
        # 배경만 밝아진다. 실제로 처음에 그 그림이 나왔다.
        #
        # 그래서 장면 전체를 near_plane만큼 뒤로 밀어, 그 앞의 0~near_plane 구간을
        # "빛을 놓을 수 있는 빈 공간"으로 비워둔다.
        # 표면 기울기 상한 — 실루엣에서 밝은 테두리가 생기는 걸 막는다
        # (surface_normals 설명 참고). 작을수록 테두리가 확실히 사라지지만
        # 너무 작으면 코·볼 같은 굴곡도 밋밋해진다
        self.max_slope = max_slope
        self.near_plane = near_plane
        self.far_plane = far_plane
        self._shape = None

    def to_distance(self, depth01):
        """모델 출력(1=가까움)을 카메라로부터의 거리로 바꾼다 (near~far 구간에 배치)."""
        return self.near_plane + (1.0 - depth01) * (self.far_plane - self.near_plane)

    def _ensure(self, h, w):
        if self._shape == (h, w):
            return
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self.px = (xs / max(w - 1, 1)) * 2.0 - 1.0
        self.py = (ys / max(h - 1, 1)) * 2.0 - 1.0
        self.shade = np.empty((h, w), np.float32)
        self._shape = (h, w)

    def surface_normals(self, dist):
        """거리 맵에서 표면 법선을 구한다 (dist: 0=가까움, 1=멂).

        이웃과의 거리 차이가 곧 표면의 기울기다. np.gradient가 배열 전체를 한 번에
        미분하므로 픽셀 반복문이 필요 없다. z 성분이 -1인 이유: 이 좌표계에서
        멀어지는 방향이 +z라, 표면이 바라보는 쪽(카메라 쪽)은 -z다.
        """
        ddy, ddx = np.gradient(dist)

        # ★실루엣 테두리 제거 (2026-08-21). 사람 윤곽처럼 깊이가 뚝 끊기는 자리는
        # 기울기가 폭발해서 "거의 수직으로 선 벽"으로 계산된다. 그런 면은 옆에서
        # 오는 빛을 정면으로 받아 **얇고 밝은 테두리**가 생긴다 — 실제로는 벽이
        # 아니라 그냥 물체의 가장자리(뒤가 안 보이는 경계)일 뿐인데도.
        # 실측 그림에서 머리와 손 둘레에 흰 실선처럼 나타났다.
        #
        # 표면의 기울기에는 현실적인 한계가 있다고 보고 잘라낸다. 자른 뒤에도
        # 완만한 굴곡(코, 볼)은 그대로 살아 있고 테두리만 사라진다.
        lim = self.max_slope
        np.clip(ddx, -lim, lim, out=ddx)
        np.clip(ddy, -lim, lim, out=ddy)

        nx, ny, nz = -ddx * 8.0, -ddy * 8.0, -np.ones_like(dist)
        inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
        return nx * inv, ny * inv, nz * inv

    def shadow_mask(self, dist, light_xyz):
        """각 점이 광원을 볼 수 있는지 (1=빛 받음, 0=그림자).

        점에서 광원 쪽으로 몇 걸음 걸어가며, 가는 길에 **광선보다 카메라에 가까운**
        표면이 있으면 그게 빛을 막고 있는 것으로 본다. 깊이 맵만 보는 근사지만
        손을 들면 얼굴에 그림자가 지는 정도는 제대로 나온다.

        CPU에서 되게 만든 요령: 픽셀마다 도는 대신 **걸음 수만큼만** 반복하고
        각 걸음에서 화면 전체를 한 번에 계산한다 — 12걸음이면 반복은 12번뿐이다.
        """
        if self.shadow_steps <= 0:
            return None
        h, w = dist.shape
        dx, dy = light_xyz[0] - self.px, light_xyz[1] - self.py
        dz = light_xyz[2] - dist
        # ★이진(0 또는 1) 대신 **가려진 걸음의 비율**을 센다. 한 걸음이라도
        # 막히면 곧바로 새까맣게 만들면 경계가 칼처럼 서고 행진 간격이 그대로
        # 계단 무늬로 드러난다(처음에 그렇게 만들었다가 실제로 그런 그림이 나왔다).
        # 비율로 세면 가장자리가 자연스럽게 반그림자처럼 흐려진다.
        blocked = np.zeros((h, w), np.float32)
        for i in range(1, self.shadow_steps + 1):
            t = i / (self.shadow_steps + 1.0)
            sx = np.clip(((self.px + dx * t) * 0.5 + 0.5) * (w - 1), 0, w - 1).astype(np.int32)
            sy = np.clip(((self.py + dy * t) * 0.5 + 0.5) * (h - 1), 0, h - 1).astype(np.int32)
            ray = dist + dz * t              # 광선이 이 지점에서 가져야 할 거리
            here = dist[sy, sx]              # 그 자리에 실제로 있는 것의 거리
            # 실제 표면이 광선보다 **가까우면**(값이 작으면) 앞을 가로막은 것이다
            blocked += (here < ray - self.shadow_bias)
        # 전부 막혀도 완전히 검게 두지 않는다 — 실제 그림자에도 사방에서 튄
        # 빛이 조금은 들어온다(shadow_min). 새까만 구멍은 눈에 가짜로 보인다
        visible = 1.0 - (blocked / self.shadow_steps) * (1.0 - self.shadow_min)
        return cv2.GaussianBlur(visible, (0, 0), 2.0)

    def shade_frame(self, frame_bgr, depth01, light_xyz, glow=True):
        """조명을 적용한 프레임(BGR uint8). depth01은 1=가까움(모듈 설명 참고)."""
        h, w = depth01.shape
        self._ensure(h, w)
        dist = self.to_distance(depth01)     # 내부 계산은 거리(작을수록 가까움)로

        nx, ny, nz = self.surface_normals(dist)
        lx, ly = light_xyz[0] - self.px, light_xyz[1] - self.py
        lz = light_xyz[2] - dist
        dist2 = lx * lx + ly * ly + lz * lz
        inv = 1.0 / np.sqrt(dist2 + 1e-6)
        lx = lx * inv
        ly = ly * inv
        lz = lz * inv

        diffuse = np.maximum(nx * lx + ny * ly + nz * lz, 0.0)
        diffuse *= 1.0 / (1.0 + self.falloff * dist2)
        if self.specular > 0.0:
            diffuse += np.maximum(-lz, 0.0) ** 16 * self.specular
        diffuse *= self.intensity

        vis = self.shadow_mask(dist, light_xyz)
        if vis is not None:
            diffuse *= vis                   # 가려진 곳은 직접광이 사라지고 환경광만 남는다

        np.add(diffuse, self.ambient, out=self.shade)
        np.clip(self.shade, 0.0, 3.0, out=self.shade)

        H, W = frame_bgr.shape[:2]
        shade = (self.shade if (h, w) == (H, W)
                 else cv2.resize(self.shade, (W, H), interpolation=cv2.INTER_LINEAR))
        # 원래 화면(방의 조명)을 살려두고 그 위에 광원 기여분을 더한다.
        # 색은 더해지는 쪽에만 입힌다 — 원래 화면까지 물들이면 방 전체가
        # 주황빛으로 변해버려 "빛 하나가 더 생겼다"로 안 보인다
        src = frame_bgr.astype(np.float32)
        out = src * self.base + src * shade[:, :, None] * self.color
        if glow:
            self._add_glow(out, light_xyz, W, H, dist)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _add_glow(self, out_f32, light_xyz, W, H, dist):
        """광원 자리에 실제로 빛나는 구체를 그린다 — **앞에 뭐가 있으면 가려진다.**

        원본 데모에서 가장 눈에 띄는 요소이고, 끌고 다닐 손잡이 역할도 한다.
        화면 전체에 빛무리를 계산하면 비싸므로 광원 주변 작은 사각형에만
        더한다(더티 사각형과 같은 발상).

        ★2026-08-21 가림 처리 추가: 처음엔 깊이를 무시하고 맨 위에 덧그렸다.
        그래서 광원을 머리 **뒤로** 옮겨도 구체가 얼굴 위에 그대로 떠 있었다 —
        조명은 "뒤에서 비추는" 계산을 하는데 구체만 앞에 보이니 앞뒤가 어긋나고,
        장면 안에 있는 빛이 아니라 화면에 붙인 스티커처럼 보였다.
        이제 그 자리의 물체가 광원보다 카메라에 가까우면 구체를 가린다.
        """
        cx = int((light_xyz[0] * 0.5 + 0.5) * W)
        cy = int((light_xyz[1] * 0.5 + 0.5) * H)
        r = max(12, int(min(W, H) * 0.055))
        x0, x1 = max(0, cx - r * 3), min(W, cx + r * 3)
        y0, y1 = max(0, cy - r * 3), min(H, cy + r * 3)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        core = np.clip(1.0 - d / r, 0.0, 1.0) ** 0.5      # 가운데 밝은 알
        halo = np.exp(-(d / (r * 1.7)) ** 2) * 0.5        # 부드럽게 퍼지는 빛무리
        glow = np.clip(core + halo, 0.0, 1.0)

        # 이 자리에 있는 물체가 광원보다 앞이면(거리가 더 작으면) 구체를 가린다.
        # 깊이 맵은 작으니 이 사각형만 잘라 키워 쓴다
        scene = cv2.resize(dist, (W, H), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        # 딱 잘라내면 경계가 칼처럼 서니 광원 앞뒤 좁은 구간에서 부드럽게 넘긴다
        visible = np.clip((scene - light_xyz[2]) / 0.06, 0.0, 1.0)
        glow *= visible

        add = glow[:, :, None] * 255.0 * self.color
        out_f32[y0:y1, x0:x1] += add


if __name__ == "__main__":
    import time

    H, W, dh, dw = 480, 640, 128, 128
    yy, xx = np.mgrid[0:dh, 0:dw].astype(np.float32)
    r = np.sqrt(((xx / dw - 0.5) * 2) ** 2 + ((yy / dh - 0.5) * 2) ** 2)
    depth = np.clip(1.0 - r, 0, 1)        # 가운데가 가까움(1) = 모델과 같은 방향
    frame = np.full((H, W, 3), 200, np.uint8)

    lit = DepthLighting()
    lit.shade_frame(frame, depth, (0.0, 0.0, 0.35))
    ts = []
    for i in range(20):
        a = i / 20.0 * 2 * np.pi
        t = time.perf_counter()
        out = lit.shade_frame(frame, depth, (0.6 * np.cos(a), 0.6 * np.sin(a), 0.35))
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    g = out.mean(axis=2)
    print("조명 단계 (%dx%d 깊이 -> %dx%d 출력, 그림자 %d걸음)" % (dw, dh, W, H, lit.shadow_steps))
    print("  중앙 %.1fms  최악 %.1fms" % (ts[len(ts) // 2], ts[-1]))
    print("  밝기 최소 %.0f 중앙 %.0f 최대 %.0f -> %s"
          % (g.min(), np.median(g), g.max(),
             "명암 생김 OK" if g.max() - g.min() > 40 else "평평함 FAIL"))
    ok, buf = cv2.imencode(".png", out)
    if ok:
        open("lighting_selftest.png", "wb").write(buf.tobytes())
        print("  결과: lighting_selftest.png")
