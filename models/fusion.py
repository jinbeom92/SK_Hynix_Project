"""
HDN optionally incorporates ground‑truth voxel information during training via
a 2D cheat path. `VoxelCheat2D` encodes each XY slice of the voxel volume
using 2D convolutions, while `Fusion2D` concatenates these cheat features
with sino‑derived XY features and mixes them through a small Conv2D block.
Both modules canonicalise inputs to [B,C,X,Y,Z] tensors and return outputs
in the same format.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def _gn(out_ch: int, prefer: int = 8) -> int:
    """Select a GroupNorm group count based on gcd(out_ch, prefer)."""
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class VoxelCheat2D(nn.Module):
    """Encode ground‑truth voxel slices into 2D feature maps.

    This encoder processes each depth slice of a voxel volume using a stack
    of 2D convolutions with GroupNorm and ReLU.  It accepts inputs of shape
    [B,1,X,Y,Z] or [B,1,X,Y], where X and Y are spatial dimensions and Z is
    depth.  The batch and depth axes are flattened, the 2D network is
    applied, and the result is reshaped back to [B,C_out,X,Y,Z].  The
    encoded features are later fused with Sino→XY features in `Fusion2D`.

    Parameters:
        base: Base number of channels for intermediate layers.
        depth: Number of Conv2D layers in the encoder.
    """

    def __init__(self, base: int = 16, depth: int = 2) -> None:
        super().__init__()
        ch = [1] + [base] * depth
        layers = []
        for i in range(depth):
            layers.extend(
                [
                    nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(_gn(ch[i + 1]), ch[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.out_ch = ch[-1]

    def forward(self, v_slice: torch.Tensor) -> torch.Tensor:
        # Canonicalise to [B,1,X,Y,Z] for processing
        if v_slice.dim() == 5:
            B, C, X, Y, Z = v_slice.shape
            if C != 1:
                raise ValueError(f"VoxelCheat2D expects channel=1, got {tuple(v_slice.shape)}")
            x = v_slice.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, 1, X, Y)
        elif v_slice.dim() == 4:
            B, C, X, Y = v_slice.shape
            if C != 1:
                raise ValueError(f"VoxelCheat2D expects channel=1, got {tuple(v_slice.shape)}")
            Z = 1
            x = v_slice  # [B,1,X,Y] → treat Z=1
        else:
            raise ValueError(f"Unsupported voxel shape: {tuple(v_slice.shape)}")
        x = self.net(x)
        # Reshape back to [B,C_out,X,Y,Z]
        x = x.view(B, Z, self.out_ch, X, Y).permute(0, 2, 3, 4, 1).contiguous()
        return x


class Fusion2D(nn.Module):
    """Fuse sino‑derived XY features with voxel cheat features.

    Given sino features [B,C_s,X,Y,Z] and optional voxel cheat features
    [B,C_c,X,Y,Z], this module concatenates them along the channel dimension
    and applies a single Conv2D→GroupNorm→ReLU block per slice.  It
    canonically handles missing channel or depth dimensions, broadcasting Z
    when one of the inputs has Z=1.  The output has shape [B,C_out,X,Y,Z].

    Parameters:
        in_ch_sino: Number of channels from Sino→XY features.
        in_ch_cheat: Number of channels from voxel cheat features (0 if disabled).
        out_ch: Number of output channels after fusion.
    """

    def __init__(self, in_ch_sino: int, in_ch_cheat: int, out_ch: int = 64) -> None:
        super().__init__()
        self.in_ch_sino = int(in_ch_sino)
        self.in_ch_cheat = int(in_ch_cheat)
        in_ch = self.in_ch_sino + self.in_ch_cheat
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_gn(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def _canon_5d(self, t: torch.Tensor, expect_c: int, name: str):
        """Ensure tensor has shape [B,C,X,Y,Z]; return tensor and shape tuple."""
        if t.dim() == 5:
            B, C, X, Y, Z = t.shape
            if expect_c is not None and C != expect_c:
                raise ValueError(f"{name} channels={C} != expected {expect_c}")
            return t, (B, C, X, Y, Z)
        elif t.dim() == 4:
            B, C, X, Y = t.shape
            if expect_c is not None and C != expect_c:
                raise ValueError(f"{name} channels={C} != expected {expect_c}")
            return t.unsqueeze(-1), (B, C, X, Y, 1)
        else:
            raise ValueError(f"{name} must be [B,C,X,Y,(Z)], got {tuple(t.shape)}")

    def forward(self, F_xy_sino: torch.Tensor, cheat_xy: torch.Tensor | None = None) -> torch.Tensor:
        # Canonicalise sinogram features
        F5, (B, Cs, X, Y, Z) = self._canon_5d(F_xy_sino, self.in_ch_sino, "F_xy_sino")
        # Handle cheat features
        if self.in_ch_cheat > 0:
            if cheat_xy is None:
                # Create zero cheat feature if none provided
                Cc = self.in_ch_cheat
                cheat5 = F5.new_zeros(B, Cc, X, Y, Z)
            else:
                cheat5, (B2, Cc, X2, Y2, Z2) = self._canon_5d(cheat_xy, self.in_ch_cheat, "cheat_xy")
                if (B2, X2, Y2) != (B, X, Y):
                    raise ValueError(
                        f"cheat_xy spatial/batch mismatch: {(B2, X2, Y2)} vs {(B, X, Y)}"
                    )
                # Expand along Z if one of them has Z=1
                if Z2 != Z:
                    if Z2 == 1:
                        cheat5 = cheat5.expand(B, Cc, X, Y, Z)
                    elif Z == 1:
                        F5 = F5.expand(B, Cs, X, Y, Z2)
                        Z = Z2
                    else:
                        raise ValueError(f"Z mismatch: F_xy_sino Z={Z}, cheat Z={Z2}")
            x_cat = torch.cat([F5, cheat5], dim=1)
        else:
            x_cat = F5
        # Flatten batch and depth for 2D fusion
        x2d = x_cat.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, x_cat.shape[1], X, Y)
        fused = self.mix(x2d)
        # Reshape back to [B,C_out,X,Y,Z]
        out = fused.view(B, Z, self.out_ch, X, Y).permute(0, 2, 3, 4, 1).contiguous()
        return out
