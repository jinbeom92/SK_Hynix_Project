# =================================================================================================
# Enc2_2D_Sino — 2D Conv over (U,A) Plane (ReLU)
# -------------------------------------------------------------------------------------------------
# Input
#   S_ua : [B, 1, U, A]
# Output
#   F2   : [B, C2, U, A]
# =================================================================================================
import math
import torch.nn as nn

def _gn(out_ch: int, prefer: int = 8) -> int:
    import math
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))

class Conv2DReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_gn(out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class Enc2_2D_Sino(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3):
        super().__init__()
        ch = [1] + [base] * depth
        self.blocks = nn.ModuleList([Conv2DReLU(ch[i], ch[i+1]) for i in range(depth)])
        self.out_ch = ch[-1]
    def forward(self, S_ua):
        x = S_ua
        for blk in self.blocks:
            x = blk(x)
        return x  # [B,C2,U,A]
