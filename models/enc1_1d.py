# =================================================================================================
# Enc1_1D_Angle — 1D Conv along Angle Axis (ReLU)
# -------------------------------------------------------------------------------------------------
# Input
#   S_ua : [B, 1, U, A]
# Output
#   F1   : [B, C1, U, A]
# =================================================================================================
import math
import torch
import torch.nn as nn

def _gn(out_ch: int, prefer: int = 8) -> int:
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))

class Conv1DReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, p=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_gn(out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class Enc1_1D_Angle(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3):
        super().__init__()
        ch = [1] + [base] * depth
        self.blocks = nn.ModuleList([Conv1DReLU(ch[i], ch[i+1]) for i in range(depth)])
        self.out_ch = ch[-1]

    def forward(self, S_ua: torch.Tensor) -> torch.Tensor:
        # S_ua: [B,1,U,A] → [B*U,1,A] → 1D conv → [B,C,U,A]
        B, C, U, A = S_ua.shape
        x = S_ua.permute(0, 2, 1, 3).contiguous().view(B * U, 1, A)
        for blk in self.blocks:
            x = blk(x)
        C1 = self.out_ch
        x = x.view(B, U, C1, A).permute(0, 2, 1, 3).contiguous()  # [B,C1,U,A]
        return x
