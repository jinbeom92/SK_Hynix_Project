"""
This module converts XY feature maps back into X×A sinograms. In the HDN
architecture, after the sinogram features are aligned onto the XY plane and
optionally fused with cheat features, the resulting tensor has shape
[B,C,X,Y,Z] where X and Y are spatial dimensions and Z is depth.  SinoDecoder2D
applies a stack of 2D convolutional layers to refine these features and then
resizes them from (X,Y) to (X,A) along the second spatial axis.  The predicted
sinogram has shape [B,1,X,A,Z] and is optionally passed through a bounding
function (none/sigmoid/tanh).

The interpolation of dimension Y→A mirrors the bilinear resampling used in the
HDN paper and aligns with scikit‑image’s philosophy of continuous Radon
inversion. Passing the entire volume
[B,C,X,Y,Z] rather than per‑slice avoids collapsing the Z dimension during this
interpolation step, ensuring that the decoder stacks along the angle dimension A
rather than depth Z.
"""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(out_ch: int, prefer: int = 8) -> int:
    """Compute a stable group count for GroupNorm using the gcd heuristic."""
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class SinoDecoder2D(nn.Module):
    """Decode XY features into XA sinograms.

    Inputs:
        F_xy: Tensor of shape [B,C,X,Y,Z] or [B,C,X,Y] containing XY
            feature maps.  The channel dimension C corresponds to fused
            features from earlier stages.
        target_A: Number of output angles A.  If None, the Y dimension
            (F_xy.shape[-1]) is used as A.

    Returns:
        A tensor [B,1,X,A,Z] representing the predicted sinogram volume.

    The decoder first rearranges [B,C,X,Y,Z] into [B·Z,C,X,Y] to process
    each depth slice independently.  A series of 2D convolutions with
    GroupNorm and ReLU are applied, followed by a 1×1 convolution to
    collapse the channel dimension to 1.  Finally, the (X,Y) grid is
    interpolated to (X,A) using the specified mode (default bilinear), and
    the tensor is reshaped back to [B,1,X,A,Z].  Optionally, a bound
    function (sigmoid or tanh) can be applied to constrain the output.
    """

    def __init__(
        self,
        in_ch: int,
        mid_ch: int = 64,
        depth: int = 3,
        *,
        interp_mode: str = "bilinear",   # recommended: "bilinear" or "nearest"
        bound: str = "none",             # {"none","sigmoid","tanh"}
    ) -> None:
        super().__init__()
        assert interp_mode in ("bilinear", "nearest")
        assert bound in ("none", "sigmoid", "tanh")
        self.interp_mode = interp_mode
        self.bound = bound
        # Construct convolutional layers: in_ch → mid_ch → … → mid_ch → 1
        ch = [in_ch] + [mid_ch] * depth + [1]
        layers = []
        for i in range(depth):
            layers.extend(
                [
                    nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(_gn(ch[i + 1]), ch[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(ch[depth], ch[depth + 1], kernel_size=1, stride=1, padding=0))
        self.net = nn.Sequential(*layers)

    def _apply_bound(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optional output bounding (sigmoid or tanh)."""
        if self.bound == "sigmoid":
            return torch.sigmoid(x)
        if self.bound == "tanh":
            return torch.tanh(x)
        return x

    def forward(self, F_xy: torch.Tensor, target_A: Optional[int] = None) -> torch.Tensor:
        """Perform the decoding and angle resizing.

        Args:
            F_xy: XY feature tensor [B,C,X,Y,Z] or [B,C,X,Y].
            target_A: Desired number of projection angles; if None, Y is used.

        Returns:
            [B,1,X,A,Z] sinogram tensor.

        Raises:
            ValueError: If the input dimensionality is unsupported or target_A is not positive.
        """
        # Flatten depth dimension for 2D convolution processing
        if F_xy.dim() == 5:
            B, C, X, Y, Z = F_xy.shape
            x = F_xy.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, Y)
        elif F_xy.dim() == 4:
            B, C, X, Y = F_xy.shape
            Z = 1
            x = F_xy
        else:
            raise ValueError(
                f"SinoDecoder2D expects [B,C,X,Y,Z] or [B,C,X,Y], got {tuple(F_xy.shape)}"
            )
        # 2D convolutional stack
        x = self.net(x)  # [B*Z, 1, X, Y]
        # Determine target angle dimension
        A = int(target_A) if target_A is not None else int(x.shape[-1])
        if A <= 0:
            raise ValueError(f"target_A must be positive, got {A}")
        # Resize along Y→A if necessary
        if A != x.shape[-1]:
            x = F.interpolate(x, size=(X, A), mode=self.interp_mode, align_corners=False)
        # Apply bounding function if requested
        x = self._apply_bound(x)
        # Reshape back to [B,1,X,A,Z]
        out = x.view(B, Z, 1, X, A).permute(0, 2, 3, 4, 1).contiguous()
        return out
