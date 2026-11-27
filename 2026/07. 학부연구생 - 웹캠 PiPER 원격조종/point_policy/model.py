"""Point Policy 정책 (BAKU 식 트랜스포머, 논문 3장 C, 2026-11-27).

  - 점 하나(로봇 점 5개, 물체 점 K개)마다 지난 H 걸음(논문 10) 위치를 한 줄로 펴서 MLP 로 토큰 하나를 만든다.
  - 집게 지난 H 걸음으로 토큰 하나를 더 만든다.
  - 점마다 다른 이름표(학습되는 벡터)를 더해 누가 누군지 알게 한다.
  - 인과 가림 없는(non-causal) 트랜스포머를 지나, 로봇 점 토큰마다 앞으로 C 걸음(논문 20 = 3초)의 위치, 집게 토큰에서 앞으로 C 걸음의 닫힘을 낸다.
  - 학습 손실은 로봇 점 궤적 (평균 제곱) + 집게 (이진 교차 엔트로피).
사진은 넣지 않는다 (논문도 점만).
"""
import torch
from torch import nn


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class PointPolicy(nn.Module):
    def __init__(self, n_robot, n_obj, hist=10, chunk=20, d=256, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.n_robot, self.n_obj, self.hist, self.chunk = n_robot, n_obj, hist, chunk
        self.enc_pt = mlp(hist * 3, d, d)
        self.enc_grip = mlp(hist, d, d)
        self.ident = nn.Parameter(torch.randn(n_robot + n_obj + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head_pt = mlp(d, d, chunk * 3)
        self.head_grip = mlp(d, d, chunk)

    def forward(self, pts, grip):
        """pts (B, n_robot + n_obj, H, 3) 고른 값, grip (B, H) 0/1 → (앞날 로봇 점 (B, n_robot, C, 3), 집게 로짓 (B, C))."""
        B = pts.shape[0]
        tok = torch.cat([self.enc_pt(pts.reshape(B, pts.shape[1], -1)), self.enc_grip(grip)[:, None]], dim=1)
        z = self.norm(self.tf(tok + self.ident[None]))
        fut = self.head_pt(z[:, :self.n_robot]).reshape(B, self.n_robot, self.chunk, 3)
        g = self.head_grip(z[:, -1])
        return fut, g
