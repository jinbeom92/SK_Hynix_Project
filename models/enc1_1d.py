# =================================================================================================
# Enc1_1D — Angle-Axis 1D Feature Encoder for Sinograms
# -------------------------------------------------------------------------------------------------
# Purpose
#   Extracts 1D features along the projection-angle axis for each detector pixel (V,U). This
#   encoder treats the sinogram as a collection of angle-wise signals per detector location,
#   enabling the model to learn angular context (e.g., periodicity, harmonics) independent of
#   spatial mixing across detector coordinates.
#
# Inputs
#   sino : torch.Tensor [B, A, V, U]
#     Batch of sinograms with A angles and detector grid (V,U).
#
# Pipeline
#   1) Reshape to per-pixel 1D sequences: (B, A, V, U) → (B·V·U, 1, A).
#   2) Angle-axis Conv1D stack: Conv1d → GroupNorm → SiLU repeated `depth` times to build
#      angular features; followed by a 1×1 Conv1d projection.
#   3) Reshape back to [B, C, A, V, U].
#   4) Augment with per-angle statistics:
#        • angle mean map (broadcast to A)
#        • angle std  map (broadcast to A)
#      Final output channels = base + 2 (mean/std).
#
# Design Notes
#   • GroupNorm uses gcd-based grouping to ensure divisibility for a wide range of channel sizes.
#   • SiLU (a.k.a. Swish) provides smooth, non-saturating activation along the angle axis.
#   • Angle mean/std augmentation stabilizes downstream alignment by exposing coarse intensity
#     context that is invariant across angles for a given detector pixel.
#
# Constructor Args
#   in_ch : int (default=1)   — input channels per angle (usually 1 for scalar sinograms)
#   base  : int (default=32)  — base channel width of the 1D stack
#   depth : int (default=3)   — number of Conv1D blocks along the angle axis
#
# Output
#   torch.Tensor [B, C_out, A, V, U], where C_out = base + 2 (mean, std)
# =================================================================================================
import math
import torch
import torch.nn as nn

class Conv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, p=2, s=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = nn.GroupNorm(num_groups=max(1, math.gcd(out_ch, 8)), num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class Enc1_1D(nn.Module):
    """Input: sino [B,A,V,U] -> [B,C_out,A,V,U] with C_out = base + 2 (mean/std)."""
    def __init__(self, in_ch=1, base=32, depth=3):
        super().__init__()
        ch = [in_ch] + [base] * (depth - 1) + [base]
        self.blocks = nn.Sequential(*[Conv1DBlock(ch[i], ch[i+1]) for i in range(depth)])
        self.proj = nn.Conv1d(base, base, kernel_size=1, bias=True)
        self.out_ch = int(base) + 2

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        B, A, V, U = sino.shape
        x = sino.permute(0, 2, 3, 1).contiguous().view(B * V * U, 1, A)  # [B*V*U,1,A]
        x = self.blocks(x)                                               # [B*V*U,base,A]
        x = self.proj(x)                                                 # [B*V*U,base,A]
        feat = x.view(B, V, U, -1, A).permute(0, 3, 4, 1, 2).contiguous()  # [B,base,A,V,U]

        # angle mean/std replicated along A to match [B,1,A,V,U]
        mean = sino.mean(dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, A, 1, 1)  # [B,1,A,V,U]
        std  = sino.std(dim=1, unbiased=False, keepdim=True).unsqueeze(2).repeat(1, 1, A, 1, 1)
        return torch.cat([feat, mean, std], dim=1)  # [B, base+2, A, V, U]
