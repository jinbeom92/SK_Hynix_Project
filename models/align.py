"""
This module implements the Sino→XY alignment block described in the HDN paper.
It consumes feature maps in the sinogram domain – where dimensions correspond
to detector pixel (X), projection angle (A), and depth (Z) – and produces
feature maps on an (X,Y) Cartesian grid. Internally, it uses a lightweight 2D
convolutional stack to mix channel information and then resizes the (X,A) plane
to (X,Y) using bilinear interpolation, similar to the way scikit-image
resamples images during iradon processing.  The output
preserves the batch and depth dimensions, returning a tensor of shape
[B,C_out,X,Y,Z].
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(out_ch: int, prefer: int = 8) -> int:
    """Return a GroupNorm group count based on the greatest common divisor.

    GroupNorm works best when the number of channels is divisible by a
    reasonably small integer.  This helper picks gcd(out_ch, prefer) but
    never less than 1.  See the original GN paper for details.
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class Sino2XYAlign(nn.Module):
    """Align sinogram feature maps to an XY plane.

    The input F_sino is expected to have shape [B,C,X,A,Z] or [B,C,X,A],
    where X is the detector pixel dimension, A is the number of projection
    angles, and Z is the depth (number of slices).  A small stack of
    3×3 convolutions with GroupNorm and ReLU mixes the channel dimension, and
    then the feature map is interpolated from (X×A) to (X×Y) using the
    specified mode (default: bilinear).  The output is returned as
    [B,C_out,X,Y,Z], maintaining the original batch and depth ordering.
    """

    def __init__(self, in_ch: int, out_ch: int = 64, depth: int = 2, mode: str = "bilinear"):
        super().__init__()
        self.mode = str(mode)
        # Build a sequence of Conv2D → GroupNorm → ReLU layers
        ch = [in_ch] + [out_ch] * depth
        layers = []
        for i in range(depth):
            layers.extend(
                [
                    nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(_gn(ch[i + 1]), ch[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
        self.stem = nn.Sequential(*layers)
        self.out_proj = nn.Identity()

    def forward(self, F_sino: torch.Tensor, out_hw) -> torch.Tensor:
        """Resample sinogram features to the XY grid.

        Args:
            F_sino: Input feature tensor of shape [B,C,X,A,Z] or [B,C,X,A].
            out_hw: Tuple (X_tgt, Y_tgt) specifying the target XY dimensions.

        Returns:
            Tensor [B,C_out,X_tgt,Y_tgt,Z] containing resampled features.

        Raises:
            ValueError: If F_sino has an unexpected number of dimensions.
        """
        # Collapse batch and depth dimensions to apply 2D convolutions on (X,A)
        if F_sino.dim() == 5:
            B, C, X, A, Z = F_sino.shape
            x2d = F_sino.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, A)
        elif F_sino.dim() == 4:
            B, C, X, A = F_sino.shape
            Z = 1
            x2d = F_sino.view(B, C, X, A)
        else:
            raise ValueError(
                f"Sino2XYAlign expects [B,C,X,A,Z] or [B,C,X,A], got {tuple(F_sino.shape)}"
            )
        # Convolutional mixing in (X,A) plane
        x2d = self.stem(x2d)
        X_tgt, Y_tgt = int(out_hw[0]), int(out_hw[1])
        # Bilinear or nearest-neighbour resizing to (X_tgt,Y_tgt)
        if self.mode in ("bilinear", "bicubic"):
            x2d = F.interpolate(x2d, size=(X_tgt, Y_tgt), mode=self.mode, align_corners=False)
        else:
            x2d = F.interpolate(x2d, size=(X_tgt, Y_tgt), mode=self.mode)
        # Reshape back to [B,C_out,X_tgt,Y_tgt,Z]
        x = x2d.view(-1, Z, x2d.shape[1], X_tgt, Y_tgt).permute(0, 2, 3, 4, 1).contiguous()
        return self.out_proj(x)
