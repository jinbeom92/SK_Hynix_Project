# =================================================================================================
# DecoderSlice2D — Direct Slice Reconstruction from XY Latent
# -------------------------------------------------------------------------------------------------
# Input : [B, C_in, X, Y]
# Output: [B, 1, X, Y]
# =================================================================================================
import math
import torch.nn as nn

def _gn(out_ch: int, prefer: int = 8) -> int:
    import math
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))

class DecoderSlice2D(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 64, depth: int = 3):
        super().__init__()
        ch = [in_ch] + [mid_ch] * depth + [1]
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i+1], 3, 1, 1, bias=False),
                nn.GroupNorm(_gn(ch[i+1]), ch[i+1]),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(ch[depth], ch[depth+1], 1, 1, 0)]
        self.net = nn.Sequential(*layers)
    def forward(self, F_xy):
        return self.net(F_xy)
