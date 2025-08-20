import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(out_ch: int, prefer: int = 8) -> int:
    """
    Compute a suitable GroupNorm divisor.

    Args:
        out_ch (int): number of channels
        prefer (int): preferred group count (default 8)

    Returns:
        int: number of groups for GroupNorm such that
             • divides out_ch evenly (via gcd)
             • clamped between 1 and out_ch
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class Sino2XYAlign(nn.Module):
    """
    Alignment block mapping sinogram feature maps [B,C,U,A] → voxel-plane features [B,C,X,Y].

    Purpose
    -------
    • Consumes concatenated sinogram features (from Enc1_1D_Angle + Enc2_2D_Sino).
    • Applies multiple Conv2D + GroupNorm + ReLU layers for local feature mixing.
    • Resizes features deterministically to match voxel spatial resolution (X,Y).
    • Output channel count is out_ch, depth controls number of conv blocks.

    Args
    ----
    in_ch (int)   : number of input feature channels
    out_ch (int)  : number of channels for each conv block (default 64)
    depth (int)   : number of conv blocks (default 2)
    mode (str)    : interpolation mode for resize, e.g. {"bilinear","bicubic","nearest"}

    Shapes
    ------
    Input:
        F_sino : [B, in_ch, U, A]
          • U = detector dimension
          • A = angle dimension
        out_hw : (X, Y) tuple, desired voxel-plane size

    Output:
        Tensor [B, out_ch, X, Y]
    """

    def __init__(self, in_ch: int, out_ch: int = 64, depth: int = 2, mode: str = "bilinear"):
        super().__init__()
        self.mode = str(mode)
        ch = [in_ch] + [out_ch] * depth
        blocks = []
        for i in range(depth):
            blocks += [
                nn.Conv2d(ch[i], ch[i+1], 3, 1, 1, bias=False),
                nn.GroupNorm(_gn(ch[i+1]), ch[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.stem = nn.Sequential(*blocks)
        self.out_proj = nn.Identity()

    def forward(self, F_sino: torch.Tensor, out_hw):
        """
        Forward pass.

        Args:
            F_sino (Tensor): [B, in_ch, U, A] sinogram feature map
            out_hw (tuple): (X, Y) target spatial size

        Returns:
            Tensor: [B, out_ch, X, Y] aligned voxel-plane feature map
        """
        X, Y = int(out_hw[0]), int(out_hw[1])
        x = self.stem(F_sino)  # [B, out_ch, U, A]

        # Deterministic-friendly resize to voxel plane size
        if self.mode in ("bilinear", "bicubic"):
            x = F.interpolate(x, size=(X, Y), mode=self.mode, align_corners=False)
        else:
            x = F.interpolate(x, size=(X, Y), mode=self.mode)

        return self.out_proj(x)
