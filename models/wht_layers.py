# -*- coding: utf-8 -*-
"""
Shared Walsh-Hadamard Transform primitives.

Contains:
  - find_min_power   : next power-of-2 helper
  - fwht / ifwht     : forward / inverse Fast Walsh-Hadamard Transform
  - SoftThresholding : learnable soft-threshold operator  (S_T)
  - HardThresholding : learnable hard-threshold operator
  - WHT1D            : full WHT layer  (FWHT → spectral gate → conv → ST → iFFHT)
  - DownsampleLY     : learnable anti-aliasing decimation layer (×4 down)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import hadamard


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_min_power(x: int) -> int:
    """Return the smallest power of 2 that is >= x."""
    if x <= 0:
        raise ValueError("Input must be a positive integer")
    return 1 << (x - 1).bit_length()


# ──────────────────────────────────────────────────────────────────────────────
# Fast Walsh-Hadamard Transform
# ──────────────────────────────────────────────────────────────────────────────

def fwht(u: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Forward Walsh-Hadamard Transform along *axis*. Length must be a power of 2."""
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    n = u.shape[-1]
    assert n == find_min_power(n), f"Signal length {n} must be a power of 2"
    H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
    y = u @ H
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y


def ifwht(u: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Inverse Walsh-Hadamard Transform along *axis*. Length must be a power of 2."""
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    n = u.shape[-1]
    assert n == find_min_power(n), f"Signal length {n} must be a power of 2"
    H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
    y = u @ H / n
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y


# ──────────────────────────────────────────────────────────────────────────────
# Thresholding operators
# ──────────────────────────────────────────────────────────────────────────────

class SoftThresholding(nn.Module):
    """Learnable soft-threshold:  S_T(x) = sign(x) · max(|x| - |T|, 0)."""

    def __init__(self, num_features):
        super().__init__()
        self.T = nn.Parameter(torch.rand(num_features) / 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(torch.sign(x),
                         nn.functional.relu(torch.abs(x) - torch.abs(self.T)))


class HardThresholding(nn.Module):
    """Learnable hard-threshold operator."""

    def __init__(self, num_features):
        super().__init__()
        self.T = nn.Parameter(torch.rand(num_features) / 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_soft = torch.mul(torch.sign(x),
                           nn.functional.relu(torch.abs(x) - torch.abs(self.T)))
        return x_soft + torch.sign(x_soft) * torch.abs(self.T)


# ──────────────────────────────────────────────────────────────────────────────
# WHT Layer  (Fig. 1 in paper)
# ──────────────────────────────────────────────────────────────────────────────

class WHT1D(nn.Module):
    """
    Walsh-Hadamard Transform layer.

    Pipeline per pod i:
        x  →  zero-pad to 2^⌈log₂L⌉  →  FWHT  →  v_i * (·)
           →  Conv_i (1×1)  →  ST_i  →  sum over pods  →  iFFHT  →  crop
        (+ optional residual skip)
    """

    def __init__(self, length: int, in_channels: int, out_channels: int,
                 pods: int = 1, residual: bool = True):
        super().__init__()
        self.length     = length
        self.length_pad = find_min_power(length)
        self.pods       = pods
        self.residual   = residual

        self.conv = nn.ModuleList(
            [nn.Conv1d(in_channels, out_channels, 1, bias=False)
             for _ in range(pods)]
        )
        self.ST = nn.ModuleList(
            [SoftThresholding((self.length_pad,)) for _ in range(pods)]
        )
        self.v = nn.ParameterList(
            [nn.Parameter(torch.rand(self.length_pad)) for _ in range(pods)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.length:
            raise ValueError(f"Expected length {self.length}, got {x.shape[-1]}")

        f0 = x
        if self.length_pad > self.length:
            f0 = nn.functional.pad(f0, (0, self.length_pad - self.length))

        f1 = fwht(f0, axis=-1)

        f5 = [self.ST[i](self.conv[i](self.v[i] * f1)) for i in range(self.pods)]
        f6 = torch.stack(f5, dim=-1).sum(dim=-1)

        y = ifwht(f6, axis=-1)[..., :self.length]

        if self.residual:
            y = y + x
        return y


# ──────────────────────────────────────────────────────────────────────────────
# Learnable anti-aliasing decimation layer  (×4 down-sampling)
# ──────────────────────────────────────────────────────────────────────────────

class DownsampleLY(nn.Module):
    """
    Two-stage learnable low-pass filter + stride-2 decimation.
    Input  : (B, L)  →  Output: (B, L//4)

    Each stage applies a learnable 1-D convolution (kernel=3, padding=1)
    followed by stride-2 sub-sampling, reducing the length by ×2 twice.
    """

    def __init__(self, learnable: bool = True):
        super().__init__()
        self.filter1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.filter2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.filter1.weight = nn.Parameter(
            torch.randn(1, 1, 3), requires_grad=learnable
        )
        self.filter2.weight = nn.Parameter(
            torch.randn(1, 1, 3), requires_grad=learnable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (B, 1, L)
        x = self.filter1(x)[:, :, ::2]   # LPF + ↓2
        x = self.filter2(x)[:, :, ::2]   # LPF + ↓2
        return x.squeeze(1)         # (B, L//4)
