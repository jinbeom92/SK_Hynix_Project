import math
import torch
import torch.nn as nn


def _gn(out_ch: int, prefer: int = 8) -> int:
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class Conv2DReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_gn(out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class Enc2_2D_Sino(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3):
        super().__init__()
        ch = [1] + [base] * depth
        self.blocks = nn.ModuleList([Conv2DReLU(ch[i], ch[i + 1]) for i in range(depth)])
        self.out_ch = ch[-1]

    def forward(self, S_xaz: torch.Tensor) -> torch.Tensor:
        # Canonicalize to [B,1,X,A,Z] without permuting axes
        if S_xaz.dim() == 5:
            if S_xaz.shape[1] != 1:
                raise ValueError(f"Expected channel=1 at dim=1, got shape {tuple(S_xaz.shape)}")
            B, _, X, A, Z = S_xaz.shape
            x5 = S_xaz
        elif S_xaz.dim() == 4:
            # Either [B,X,A,Z] or [B,1,X,A]
            if S_xaz.shape[1] == 1:        # [B,1,X,A] → add Z=1
                B, _, X, A = S_xaz.shape
                Z = 1
                x5 = S_xaz.unsqueeze(-1)
            else:                           # [B,X,A,Z] → add channel dim
                B, X, A, Z = S_xaz.shape
                x5 = S_xaz.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported sinogram shape: {tuple(S_xaz.shape)}")

        # Flatten (B,Z) into batch for 2D conv over (X,A)
        # [B,1,X,A,Z] → [B,Z,1,X,A] → [B*Z, 1, X, A]
        x = x5.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, 1, X, A)

        # Apply Conv2D stack
        for blk in self.blocks:
            x = blk(x)  # [B*Z, C2, X, A]

        C2 = self.out_ch
        # Reshape back to [B,C2,X,A,Z]
        x = x.view(B, Z, C2, X, A).permute(0, 2, 3, 4, 1).contiguous()
        return x
