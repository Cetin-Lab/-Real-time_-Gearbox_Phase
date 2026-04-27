# -*- coding: utf-8 -*-
"""
TPNWHT2 — Two-Pronged Network with WHT Layer and Learnable Decimation Layer.

Architecture (Fig. 2b / Algorithm 1 in paper):
  Each branch:
    1024  →  DownsampleLY (×4)  →  256
  Concatenate both branches  →  512

  Encoder:  512 → 512 → 128 → 64
  WHT1D(64)
  Decoder:  64  → 128 → 32
  WHT1D(32)
  Projection: 32 → 16

Two separate instances are trained independently:
  real_model  targets y[:, :16]
  imag_model  targets y[:, 16:]
Total parameters per model: 611,890  →  1,223,780 combined.
"""

import torch
import torch.nn as nn
from .wht_layers import WHT1D, DownsampleLY


class TPNWHT2(nn.Module):
    """
    Single branch of TPNWHT2.
    Instantiate twice (real / imaginary) and train separately.

    Input  : (B, 1024)
    Output : (B, 16)   — either the real or imaginary part of the analytic signal
    """

    def __init__(self):
        super().__init__()
        # Two independent learnable decimation branches (each 1024 → 256)
        self.downsample1 = DownsampleLY(learnable=True)
        self.downsample2 = DownsampleLY(learnable=True)

        # Encoder: concat output is 256+256=512
        self.encoder = nn.Sequential(
            nn.Linear(512, 512),
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
        R = self.downsample1(x)                      # (B, 256)
        Z = self.downsample2(x)                      # (B, 256)
        x = torch.cat((R, Z), dim=1)                 # (B, 512)

        x = self.encoder(x)                          # (B, 64)
        x = self.wht1(x.unsqueeze(1)).squeeze(1)     # (B, 64)
        x = self.decoder(x)                          # (B, 32)
        x = self.wht2(x.unsqueeze(1)).squeeze(1)     # (B, 32)
        x = self.proj(x)                             # (B, 16)
        return x
