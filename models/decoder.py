import math
import torch.nn as nn

def _gn(out_ch: int, prefer: int = 8) -> int:
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class DecoderSlice2D(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 64, depth: int = 3):
        super().__init__()
        # Channel schedule: [in_ch → mid_ch … → mid_ch → 1]
        ch = [in_ch] + [mid_ch] * depth + [1]
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_gn(ch[i + 1]), ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        # Final 1×1 projection to 1 output channel
        layers += [nn.Conv2d(ch[depth], ch[depth + 1], kernel_size=1, stride=1, padding=0)]
        self.net = nn.Sequential(*layers)

    def forward(self, F_xy):
        if F_xy.dim() == 5:
            B, C, X, Y, Z = F_xy.shape
            # [B, C, X, Y, Z] → [B, Z, C, X, Y] → [B*Z, C, X, Y]
            x = F_xy.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, Y)
        elif F_xy.dim() == 4:
            # Z=1 case
            B, C, X, Y = F_xy.shape
            Z = 1
            x = F_xy.view(B * Z, C, X, Y)
        else:
            raise ValueError(f"DecoderSlice2D expects [B,C,X,Y,Z] or [B,C,X,Y], got {tuple(F_xy.shape)}")

        # 2D Conv stack per slice
        x = self.net(x)  # [B*Z, 1, X, Y]

        # Restore to [B, 1, X, Y, Z]
        out = x.view(B, Z, 1, X, Y).permute(0, 2, 3, 4, 1).contiguous()  # [B,1,X,Y,Z]
        return out
