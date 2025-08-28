"""
This module implements the Enc1 component of HDN, which applies a stack of
1D convolutions along the projection angle dimension.  The sinogram input
has shape [B,1,X,A,Z], where X is the detector pixel count, A is the number of
angles, and Z is the depth.  The encoder flattens the (B,X,Z) dimensions,
applies residual Conv1D→GroupNorm→ReLU blocks (each block adds its output to
its input when the channel and length match) to each [1,A] signal, and then
restores the original ordering, producing a feature tensor of shape
[B,C_out,X,A,Z].  This residual design encourages the network to learn
correction terms rather than full mappings, improving gradient flow and
preserving low‑frequency angular patterns:contentReference[oaicite:0]{index=0}.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def _gn(out_ch: int, prefer: int = 8) -> int:
    """Choose a GroupNorm group count based on gcd(out_ch, prefer).

    GroupNorm performs best when the number of channels is divisible by a small
    integer.  This helper returns max(1, gcd(out_ch, prefer)) to stabilise
    normalisation across various channel sizes.
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class Conv1DReLU(nn.Module):
    """A small block consisting of Conv1D → GroupNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 7, s: int = 1, p: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_gn(out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a 1D convolution followed by group normalisation and ReLU activation.

        A residual skip connection is applied when the input and output shapes
        match exactly.  This encourages the network to learn residuals of the
        angular signal rather than the full mapping, improving gradient flow in
        deep stacks:contentReference[oaicite:1]{index=1}.

        Args:
            x: Input tensor of shape [N,C_in,L].

        Returns:
            Tensor of shape [N,C_out,L] with optional residual addition.
        """
        y = self.conv(x)
        y = self.gn(y)
        y = self.act(y)
        # If the shapes match exactly, add a skip connection
        if y.shape == x.shape:
            return y + x
        return y


class Enc1_1D_Angle(nn.Module):
    """Angle-wise 1D encoder for sinogram inputs.

    Parameters:
        base: Number of output channels for each intermediate Conv1D.
        depth: Number of Conv1DReLU blocks in the encoder.

    Inputs:
        S_xaz: A sinogram tensor with shape [B,1,X,A,Z], [B,1,X,A], or [B,X,A,Z].
            If no explicit channel dimension is present, one is added automatically.

    Returns:
        A tensor of shape [B,C_out,X,A,Z], where C_out = base and the angle
        dimension has been processed by depth Conv1D blocks.  The batch, X,
        and Z dimensions are preserved across the reshape operations.
    """

    def __init__(self, base: int = 32, depth: int = 3) -> None:
        super().__init__()
        ch = [1] + [base] * depth
        self.blocks = nn.ModuleList([Conv1DReLU(ch[i], ch[i + 1]) for i in range(depth)])
        self.out_ch = ch[-1]

    def forward(self, S_xaz: torch.Tensor) -> torch.Tensor:
        # Normalise input to [B,1,X,A,Z] shape
        if S_xaz.dim() == 5:
            if S_xaz.shape[1] != 1:
                raise ValueError(f"Expected channel=1 at dim=1, got shape {tuple(S_xaz.shape)}")
            B, _, X, A, Z = S_xaz.shape
            x5 = S_xaz
        elif S_xaz.dim() == 4:
            # Accept [B,1,X,A] or [B,X,A,Z] and insert missing dimension
            if S_xaz.shape[1] == 1:
                B, _, X, A = S_xaz.shape
                Z = 1
                x5 = S_xaz.unsqueeze(-1)
            else:
                B, X, A, Z = S_xaz.shape
                x5 = S_xaz.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported sinogram shape: {tuple(S_xaz.shape)}")

        # Flatten batch, X, Z to apply 1D convolution along angle dimension
        x = x5.permute(0, 2, 4, 1, 3).contiguous().view(B * X * Z, 1, A)
        for blk in self.blocks:
            x = blk(x)
        C_out = self.out_ch
        # Reshape back to [B,C_out,X,A,Z]
        x = x.view(B, X, Z, C_out, A).permute(0, 3, 1, 4, 2).contiguous()
        return x
