# =================================================================================================
# Enc3_3D — Optional Volumetric Encoder (Training-Only Prior Path)
# -------------------------------------------------------------------------------------------------
# Purpose
#   Extracts hierarchical 3D features directly from the ground-truth volume V_gt to provide an
#   optional volumetric prior during training. This path can supply high-level spatial context
#   to the alignment stage (e.g., via concatenation), while being disabled at inference time.
#
# Inputs
#   vol : torch.Tensor [B, 1, D, H, W]
#         Ground-truth volume (or an auxiliary volumetric signal) provided during training.
#
# Architecture
#   • Depth-wise feature expansion with isotropic 3D convolutions:
#       Conv3D → GroupNorm → SiLU  × depth
#     where channel width grows geometrically: base · 2^i for block i.
#   • GroupNorm group count is chosen via gcd(out_ch, gn_groups) for stable normalization.
#
# Constructor Args
#   in_ch : int (default=1)   — input channels (typically 1 for scalar volume)
#   base  : int (default=16)  — base channel width for the first block
#   depth : int (default=3)   — number of Conv3D blocks; channels scale as base·2^i
#
# Output
#   torch.Tensor [B, C, D, H, W]
#     Volumetric latent features; `self.out_ch` exposes the final channel width.
#
# Notes
#   • Intended for training-only use (e.g., cheat/teacher path). At inference, this branch
#     is omitted and the alignment block operates without the volumetric prior.
# =================================================================================================
import torch, math
import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, gn_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, p)
        g = math.gcd(out_ch, gn_groups)
        if g <= 0: g = 1
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class Enc3_3D(nn.Module):
    """Optional volumetric encoder for V_gt (training-only).
    Input: V_gt [B,1,D,H,W]
    Output: latent3d [B,C,D,H,W]
    """
    def __init__(self, in_ch=1, base=16, depth=3):
        super().__init__()
        ch = [in_ch] + [base*(2**i) for i in range(depth)]
        blocks = []
        for i in range(depth):
            blocks.append(Conv3DBlock(ch[i], ch[i+1]))
        self.blocks = nn.Sequential(*blocks)
        self.out_ch = ch[-1]
    def forward(self, vol):
        return self.blocks(vol)
