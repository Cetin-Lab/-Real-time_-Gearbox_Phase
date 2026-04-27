# -*- coding: utf-8 -*-
"""
JWHT — Joint WHT Model with Learnable Downsampling Layer.

Architecture (Fig. 2c / Algorithm 2 in paper):
  1024  →  DownsampleLY (×4)  →  256   (branch 1: R)
  1024  →  DownsampleLY (×4)  →  256   (branch 2: Z)
  Concatenate  →  512

  Encoder:  512 → 512 → 128 → 64
  WHT1D(64)
  Decoder:  64  → 128 → 32
  WHT1D(32)
  Final projection: 32 → 32

Output convention:
  y[:, 0:16]  = real part of the analytic signal
  y[:, 16:32] = imaginary part of the analytic signal

Single model trained jointly on both parts.
Total parameters: 350,286.
"""

import torch
import torch.nn as nn
from .wht_layers import WHT1D, DownsampleLY


class JWHT(nn.Module):
    """
    Joint Walsh-Hadamard Transform model.

    Input  : (B, L_in)   — raw gearbox signal window (default L_in = 1024)
    Output : (B, 32)
               [:, 0:16]  → real part
               [:, 16:32] → imaginary part
    """

    def __init__(self):
        super().__init__()
        # Dual learnable decimation: each 1024 → 256, concat → 512
        self.downsample1 = DownsampleLY(learnable=True)
        self.downsample2 = DownsampleLY(learnable=True)

        # Encoder: 512 → 512 → 128 → 64
        self.encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
        )
        self.wht1 = WHT1D(length=64, in_channels=1, out_channels=1)

        # Decoder: 64 → 128 → 32
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 32),
        )
        self.wht2 = WHT1D(length=32, in_channels=1, out_channels=1)

        # Final projection: 32 → 32  (preserves dimension)
        self.proj = nn.Linear(32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1-4: dual learnable decimation + concat
        R = self.downsample1(x)                      # (B, 256)
        Z = self.downsample2(x)                      # (B, 256)
        x = torch.cat((R, Z), dim=1)                 # (B, 512)

        # Step 5-8: encoder
        x = self.encoder(x)                          # (B, 64)

        # Step 9-12: first WHT layer
        x = self.wht1(x.unsqueeze(1)).squeeze(1)     # (B, 64)

        # Step 13-15: decoder
        x = self.decoder(x)                          # (B, 32)

        # Step 16-19: second WHT layer
        x = self.wht2(x.unsqueeze(1)).squeeze(1)     # (B, 32)

        # Step 20-22: final projection
        y = self.proj(x)                             # (B, 32)
        return y
        # Training split:  y[:, 0:16] = real,  y[:, 16:32] = imaginary
