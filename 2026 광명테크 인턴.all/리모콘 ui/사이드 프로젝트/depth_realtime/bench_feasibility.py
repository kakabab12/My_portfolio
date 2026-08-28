"""이 PC의 CPU가 단안 깊이 추정을 실시간으로 돌릴 수 있는지 재본다.

가중치는 필요 없다 — 신경망의 **속도**는 가중치 값이 아니라 구조와 입력 크기로
정해지므로, 무작위 가중치로 재도 실제 모델과 같은 시간이 나온다. 그래서 모델
파일을 내려받기 전에 "될지 안 될지"를 먼저 확정할 수 있다.

비교 기준: 원본 데모(Konrad Reczko, TypeGPU)는 448x448에서 M4 Pro GPU로 8ms.
"""
import time

import torch
import torch.nn as nn

torch.set_grad_enabled(False)
print(f"torch {torch.__version__} | 스레드 {torch.get_num_threads()}개\n")


class DepthNetLike(nn.Module):
    """MiDaS-small 계열(경량 인코더-디코더)과 비슷한 계산량의 깊이 추정망.

    구조를 그대로 베낀 게 아니라 **계산량 규모를 맞춘 대역**이다: 입력을 1/32까지
    5단계로 줄이며 채널을 늘리는 인코더 + 다시 키우며 합치는 디코더, 마지막에
    1채널(깊이) 출력. MiDaS-small, DepthAnything-small 모두 이 형태다.
    """

    def __init__(self, width=32):
        super().__init__()
        w = width

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.e1 = block(3, w, 2)          # 1/2
        self.e2 = block(w, w * 2, 2)      # 1/4
        self.e3 = block(w * 2, w * 4, 2)  # 1/8
        self.e4 = block(w * 4, w * 8, 2)  # 1/16
        self.e5 = block(w * 8, w * 8, 2)  # 1/32
        self.d4 = block(w * 16, w * 4)
        self.d3 = block(w * 8, w * 2)
        self.d2 = block(w * 4, w)
        self.d1 = block(w * 2, w)
        self.out = nn.Conv2d(w, 1, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2)
        e4 = self.e4(e3); e5 = self.e5(e4)
        d = self.d4(torch.cat([self.up(e5), e4], 1))
        d = self.d3(torch.cat([self.up(d), e3], 1))
        d = self.d2(torch.cat([self.up(d), e2], 1))
        d = self.d1(torch.cat([self.up(d), e1], 1))
        return self.out(self.up(d))


def bench(model, size, runs=8):
    x = torch.randn(1, 3, size, size)
    model(x)                      # 워밍업(최초 호출은 메모리 할당 등이 섞인다)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


print("입력 크기별 1회 깊이 추정 시간 (이 PC의 CPU)")
print(f"{'크기':>10s} {'경량(w=16)':>12s} {'표준(w=32)':>12s} {'fps(표준)':>11s}")
print("-" * 50)
models = {16: DepthNetLike(16).eval(), 32: DepthNetLike(32).eval()}
rows = []
for size in (448, 384, 256, 192, 128):
    a = bench(models[16], size)
    b = bench(models[32], size)
    rows.append((size, a, b))
    print(f"{size:>7d}px {a:10.1f}ms {b:10.1f}ms {1000.0 / b:10.1f}")

print("\n원본 데모 기준: 448x448을 M4 Pro GPU에서 8ms (=125fps)")
base = [r for r in rows if r[0] == 448][0]
print(f"이 PC 448x448 표준 모델: {base[2]:.0f}ms  ->  M4 Pro GPU 대비 약 {base[2] / 8:.0f}배 느림")
print("\n실시간(30fps=33ms) 기준으로 가능한 조합:")
ok = [(s, a, b) for s, a, b in rows if a < 33 or b < 33]
if ok:
    for s, a, b in ok:
        which = []
        if b < 33:
            which.append(f"표준 {b:.0f}ms")
        if a < 33:
            which.append(f"경량 {a:.0f}ms")
        print(f"  {s}px — " + ", ".join(which))
else:
    print("  없음 — 이 CPU로는 어떤 조합도 30fps를 못 맞춘다")
