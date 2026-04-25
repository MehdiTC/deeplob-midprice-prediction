"""Simple 2-layer LSTM baseline for LOB mid-price direction prediction."""

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """Two-layer LSTM operating on sequences of LOB snapshots.

    Input:  (batch, seq_len=100, input_size=40)
    Output: (batch, num_classes=3) logits
    """

    def __init__(
        self,
        input_size: int = 40,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 100, 40)  →  (batch, 3)."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
