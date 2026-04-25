"""PyTorch reimplementation of the DeepLOB architecture (Zhang et al., 2019).

Reference Keras implementation: data/raw/DeepLOB.ipynb
Parameter count should equal 60,691 (verifiable with sum(p.numel() for p in DeepLOB().parameters())).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Nine convolutional layers that extract spatial structure from the LOB.

    Input:  (batch, 1, 100, 40)   — 1 channel, 100 time steps, 40 features
    Output: (batch, 16, 100, 1)   — 16 channels, time preserved, features collapsed

    Layer groups:
      - Group 1: (1×2, stride 1×2) pairs price+volume at each level → (100, 20)
      - Group 2: (1×2, stride 1×2) integrates bid/ask across levels  → (100, 10)
      - Group 3: (1×10) collapses all 10 levels                      → (100, 1)
    Each group is followed by two (4×1, same-padded) temporal smoothing layers.
    """

    def __init__(self):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(negative_slope=0.01)

        bn = lambda: nn.BatchNorm2d(16)
        self.layers = nn.Sequential(
            # Group 1
            nn.Conv2d(1,  16, kernel_size=(1, 2), stride=(1, 2)), lrelu(), bn(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding="same"), lrelu(), bn(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding="same"), lrelu(), bn(),
            # Group 2
            nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2)), lrelu(), bn(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding="same"), lrelu(), bn(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding="same"), lrelu(), bn(),
            # Group 3
            nn.Conv2d(16, 16, kernel_size=(1, 10)), lrelu(), bn(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding="same"), lrelu(), bn(),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding="same"), lrelu(), bn(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InceptionModule(nn.Module):
    """Inception module with three towers capturing features at different time scales.

    Input:  (batch, 16, 100, 1)
    Output: (batch, 96, 100, 1)  — 3 towers of 32 channels each, concatenated

    Tower 1: 1×1 → 3×1 (short-range temporal)
    Tower 2: 1×1 → 5×1 (mid-range temporal)
    Tower 3: MaxPool 3×1 → 1×1 (pooling branch)
    """

    def __init__(self):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(negative_slope=0.01)

        self.tower_1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), padding="same"), lrelu(),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding="same"), lrelu(),
        )
        self.tower_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1), padding="same"), lrelu(),
            nn.Conv2d(32, 32, kernel_size=(5, 1), padding="same"), lrelu(),
        )
        self.tower_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(16, 32, kernel_size=(1, 1), padding="same"), lrelu(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.tower_1(x), self.tower_2(x), self.tower_3(x)], dim=1)


class DeepLOB(nn.Module):
    """Full DeepLOB model: ConvBlock → InceptionModule → LSTM → Linear.

    Input:  (batch, 1, 100, 40)
    Output: (batch, 3) logits — use nn.CrossEntropyLoss, not softmax here.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv_block = ConvBlock()
        self.inception = InceptionModule()
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
        self._init_conv_weights()

    def _init_conv_weights(self):
        """Re-initialise Conv2d layers for LeakyReLU(0.01).

        PyTorch's default kaiming_uniform uses a=sqrt(5), calibrated for a slope of
        ~2.24 — not 0.01.  With the wrong slope, each of the 9 conv layers attenuates
        activations by ~0.41×, leaving std≈0.05 at the conv block output and causing
        vanishing gradients (~28,000× weaker at layer 0 vs the FC).  Using a=0.01
        matches the actual activation and preserves variance across all 9 layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 100, 40) or (batch, 1, 100, 40) — channel dim added if missing."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.inception(self.conv_block(x))   # (batch, 96, 100, 1)
        x = x.squeeze(-1).permute(0, 2, 1)       # (batch, 100, 96)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])                    # (batch, 3)
