"""랜드마크 시퀀스 (T, F) -> 제스처 클래스를 분류하는 경량 GRU."""
import torch
import torch.nn as nn


class GestureGRU(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.gru(x)       # (B, T, 2*hidden_dim)
        pooled = out.mean(dim=1)   # 시간축 평균 풀링 (마지막 프레임만 보는 것보다 노이즈에 강함)
        return self.head(pooled)
