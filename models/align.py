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
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_gn(ch[i + 1]), ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.stem = nn.Sequential(*blocks)
        self.out_proj = nn.Identity()  # placeholder if you want a final 1x1 later

    def forward(self, F_sino: torch.Tensor, out_hw):
        if F_sino.dim() == 5:
            B, C, X, A, Z = F_sino.shape
            # [B,C,X,A,Z] → [B,Z,C,X,A] → [B*Z,C,X,A]
            x2d = F_sino.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, A)
            BZ = B * Z
        elif F_sino.dim() == 4:
            B, C, X, A = F_sino.shape
            Z = 1
            x2d = F_sino.view(B, C, X, A)   # [B,C,X,A]
            BZ = B
        else:
            raise ValueError(f"Sino2XYAlign expects [B,C,X,A,Z] or [B,C,X,A], got {tuple(F_sino.shape)}")

        # Local mixing on (X,A)
        x2d = self.stem(x2d)  # [BZ, out_ch, X, A]

        # Deterministic-friendly resize to (X,Y)
        X_tgt, Y_tgt = int(out_hw[0]), int(out_hw[1])
        if self.mode in ("bilinear", "bicubic"):
            x2d = F.interpolate(x2d, size=(X_tgt, Y_tgt), mode=self.mode, align_corners=False)
        else:
            x2d = F.interpolate(x2d, size=(X_tgt, Y_tgt), mode=self.mode)

        # Restore [B,out_ch,X,Y,Z]
        x = x2d.view(B, Z, x2d.shape[1], X_tgt, Y_tgt).permute(0, 2, 3, 4, 1).contiguous()
        return self.out_proj(x)
