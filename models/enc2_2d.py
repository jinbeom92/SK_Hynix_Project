"""
This module implements the Enc2 component of HDN, which applies a stack
of 2D convolutions to the sinogram plane (X×A).  Given an input sinogram
with shape [B,1,X,A,Z], where X is the detector pixel dimension, A is the
number of projection angles, and Z is the depth (number of slices), it
flattens the batch and depth dimensions to process each [X,A] slice via
Conv2D→GroupNorm→ReLU blocks.  The output is reshaped back to
[B,C_out,X,A,Z] so that downstream modules (e.g., Sino2XYAlign) can fuse it
with other features.

As with the 1D encoder, this module ensures that inputs of shape
[B,1,X,A] or [B,X,A,Z] are canonicalised to [B,1,X,A,Z] before processing.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def _gn(out_ch: int, prefer: int = 8) -> int:
    """Select a GroupNorm group count via gcd(out_ch, prefer).

    GroupNorm prefers channel counts divisible by small integers.  This
    helper returns the greatest common divisor of out_ch and a preferred
    factor (default 8), ensuring it is at least 1.
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class Conv2DReLU(nn.Module):
    """A convenience block: Conv2D → GroupNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_gn(out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        return self.act(x)


class Enc2_2D_Sino(nn.Module):
    """2D encoder over sinogram planes (X×A).

    Parameters:
        base: Base number of channels for each convolutional block.
        depth: Number of Conv2DReLU blocks.

    Inputs:
        S_xaz: Sinogram tensor [B,1,X,A,Z], [B,1,X,A], or [B,X,A,Z].
            If channel or depth dimensions are missing, they are added.

    Returns:
        A tensor [B,C_out,X,A,Z], where C_out = base.  Internally,
        the (B,Z) dimensions are merged to process each [X,A] slice in
        parallel, then restored afterwards.  This captures joint patterns
        across detector pixels and projection angles, complementing the 1D
        angle-wise encoder.
    """

    def __init__(self, base: int = 32, depth: int = 3) -> None:
        super().__init__()
        ch = [1] + [base] * depth
        self.blocks = nn.ModuleList([Conv2DReLU(ch[i], ch[i + 1]) for i in range(depth)])
        self.out_ch = ch[-1]

    def forward(self, S_xaz: torch.Tensor) -> torch.Tensor:
        # Convert input to [B,1,X,A,Z] form
        if S_xaz.dim() == 5:
            if S_xaz.shape[1] != 1:
                raise ValueError(f"Expected channel=1 at dim=1, got shape {tuple(S_xaz.shape)}")
            B, _, X, A, Z = S_xaz.shape
            x5 = S_xaz
        elif S_xaz.dim() == 4:
            # [B,1,X,A] → add depth; [B,X,A,Z] → add channel
            if S_xaz.shape[1] == 1:
                B, _, X, A = S_xaz.shape
                Z = 1
                x5 = S_xaz.unsqueeze(-1)
            else:
                B, X, A, Z = S_xaz.shape
                x5 = S_xaz.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported sinogram shape: {tuple(S_xaz.shape)}")

        # Flatten (B,Z) and apply 2D conv stack on each [X,A] slice
        x = x5.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, 1, X, A)
        for blk in self.blocks:
            x = blk(x)
        C_out = self.out_ch
        # Restore [B,C_out,X,A,Z]
        x = x.view(B, Z, C_out, X, A).permute(0, 2, 3, 4, 1).contiguous()
        return x
