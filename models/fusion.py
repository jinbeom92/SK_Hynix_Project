# =================================================================================================
# VoxelCheat2D & Fusion2D
# -------------------------------------------------------------------------------------------------
# Inputs
#   F_xy_sino : [B, C_s, X, Y]
#   cheat_xy  : [B, C_c, X, Y] or None
# Output
#   fused     : [B, C_f, X, Y]
# =================================================================================================
import math
import torch
import torch.nn as nn

def _gn(out_ch: int, prefer: int = 8) -> int:
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))

class VoxelCheat2D(nn.Module):
    def __init__(self, base: int = 16, depth: int = 2):
        super().__init__()
        ch = [1] + [base] * depth
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i+1], 3, 1, 1, bias=False),
                nn.GroupNorm(_gn(ch[i+1]), ch[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)
        self.out_ch = ch[-1]
    def forward(self, v_slice):
        return self.net(v_slice)

class Fusion2D(nn.Module):
    def __init__(self, in_ch_sino: int, in_ch_cheat: int, out_ch: int = 64):
        super().__init__()
        self.in_ch_sino  = int(in_ch_sino)
        self.in_ch_cheat = int(in_ch_cheat)
        in_ch = self.in_ch_sino + self.in_ch_cheat
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(_gn(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def forward(self, F_xy_sino: torch.Tensor, cheat_xy: torch.Tensor = None) -> torch.Tensor:
        if self.in_ch_cheat > 0:
            if cheat_xy is None:
                # pad missing cheat with zeros to match expected channels
                B, _, X, Y = F_xy_sino.shape
                zeros = F_xy_sino.new_zeros(B, self.in_ch_cheat, X, Y)
                x = torch.cat([F_xy_sino, zeros], dim=1)
            else:
                x = torch.cat([F_xy_sino, cheat_xy], dim=1)
        else:
            x = F_xy_sino
        return self.mix(x)
