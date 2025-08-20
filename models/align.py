# =================================================================================================
# Sino2XYAlign — Learned mapping from (U,A) plane to (X,Y) plane
# -------------------------------------------------------------------------------------------------
# Converts sinogram-plane features [B, C, U, A] to XY-aligned features [B, C_out, X, Y].
# Uses conv blocks + resize (interpolate) to the target (X,Y).
# =================================================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _gn(out_ch: int, prefer: int = 8) -> int:
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))

class Sino2XYAlign(nn.Module):
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
        X, Y = int(out_hw[0]), int(out_hw[1])
        x = self.stem(F_sino)  # [B,C,U,A]
        # deterministic-friendly resize
        if self.mode in ("bilinear", "bicubic"):
            x = F.interpolate(x, size=(X, Y), mode=self.mode, align_corners=False)
        else:
            x = F.interpolate(x, size=(X, Y), mode=self.mode)
        return self.out_proj(x)
