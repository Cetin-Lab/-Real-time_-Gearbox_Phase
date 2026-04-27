# -*- coding: utf-8 -*-
"""
TPNWHT — Two-Pronged Network with Walsh-Hadamard Transform Layer.

Architecture (Fig. 2a in paper):
  Each branch:  1024 → 512 → 128 → 64 → WHT1D(64) → 128 → 32 → WHT1D(32) → 16
  Upper branch → Re(Y),  Lower branch → Im(Y)
  Phase = arctan2(Im, Re)

Two separate instances are trained independently:
  real_model  targets y[:, :16]
  imag_model  targets y[:, 16:]
Total parameters per model: 611,890  →  1,223,780 combined.
"""

import torch
import torch.nn as nn
from .wht_layers import WHT1D


class TPNWHT(nn.Module):
    """
    Single branch of the Two-Pronged Network with WHT layers.
    Instantiate twice (real / imaginary) and train separately.

    Input  : (B, 1024)
    Output : (B, 16)   — either the real or imaginary part of the analytic signal
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
        )
        self.wht1 = WHT1D(length=64, in_channels=1, out_channels=1)

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 32),
        )
        self.wht2 = WHT1D(length=32, in_channels=1, out_channels=1)

        self.proj = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)                          # (B, 64)
        x = self.wht1(x.unsqueeze(1)).squeeze(1)     # (B, 64)
        x = self.decoder(x)                          # (B, 32)
        x = self.wht2(x.unsqueeze(1)).squeeze(1)     # (B, 32)
        x = self.proj(x)                             # (B, 16)
        return x
