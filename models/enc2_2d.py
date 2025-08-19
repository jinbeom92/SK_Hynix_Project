# =================================================================================================
# Enc2_2D — Per-Angle 2D Detector-Plane Encoder with Harmonic Angle Embedding
# -------------------------------------------------------------------------------------------------
# Purpose
#   Builds rich 2D feature maps on the detector plane for each projection angle and augments
#   them with explicit angle embeddings. This module complements Enc1_1D by capturing spatial
#   structure (V,U) per angle while injecting harmonic encodings of the acquisition angle.
#
# Inputs
#   S_in    : torch.Tensor [B, A, V, U]
#             Sinograms arranged as angle-indexed 2D detector slices.
#   angles  : torch.Tensor [A]
#             Acquisition angles in radians.
#   cheat2d : torch.Tensor [B, Cc, V, U] or None
#             Optional “cheat” channels (e.g., train-time priors); will be zeroed or padded/cropped
#             to the expected width `Cc = cheat_in_ch`, then broadcast to all angles.
#   gate    : float
#             Scalar gate applied to `cheat2d` channels (0 disables their contribution).
#
# Pipeline
#   1) Construct input channel stack per angle:
#        [ sinogram(1), angle-mean(1), harmonic maps(2K), cheat(Cc) ]  →  C_in = 2 + 2K + Cc
#   2) Process with a depth `depth` stack of Conv2D → GroupNorm → SiLU blocks at constant width `base`.
#   3) Reshape result back to angle-indexed 5D layout: [B, C_out, A, V, U].
#
# Harmonic Angle Embedding
#   • For K harmonics, builds sin/cos maps: {sin(k·θ), cos(k·θ)} for k=1..K and broadcasts them
#     to the detector grid. These maps are concatenated as 2K channels and help the network model
#     periodic angular structure.
#
# Shape & Stability Notes
#   • GroupNorm group count is chosen via gcd(out_ch, 8) to ensure divisibility and stable training.
#   • Angle-mean map (per sample) provides coarse intensity context invariant across angles.
#   • If `cheat_in_ch>0`, the network always receives exactly Cc channels for cheat input by
#     zero-padding or channel-slicing; this keeps layer shapes fixed irrespective of gating.
#
# Constructor Args
#   in_ch       : int — number of per-angle sinogram channels (usually 1)
#   base        : int — constant channel width of each Conv2D block
#   depth       : int — number of Conv2D blocks
#   harm_K      : int — number of angular harmonics (adds 2K channels)
#   cheat_in_ch : int — fixed width for optional cheat channels Cc (0 to disable)
#
# Output
#   torch.Tensor [B, C_out, A, V, U]  — detector-plane features per angle
# =================================================================================================
from typing import Optional
import math
import torch
import torch.nn as nn


def _safe_gn_groups(out_ch: int, prefer: int = 8) -> int:
    # choose a divisor of out_ch, close to 'prefer'
    g = math.gcd(out_ch, prefer)
    if g == 0:
        g = 1
    return max(1, min(out_ch, g))


class Conv2DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_safe_gn_groups(out_ch, 8), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class Enc2_2D(nn.Module):
    def __init__(self, in_ch: int, base: int, depth: int, harm_K: int, cheat_in_ch: int = 0):
        """
        in_ch: per-angle sino channels (usually 1)
        base : number of channels for each conv stage (constant width)
        depth: number of Conv2DBlocks
        harm_K: number of harmonics (sin/cos 1..K -> 2K channels)
        cheat_in_ch: expected cheat channels (Cc). If >0, layers are built with
                     Cc included in C_in and the forward() must always supply Cc
                     channels (zeros if gate==0 or cheat2d=None).
        """
        super().__init__()
        self.harm_K = int(harm_K)
        self.cheat_in_ch = int(cheat_in_ch)
        # Per-angle input channel plan: [sino(1), mean(1), harmonic(2K), cheat(Cc?)]
        c_in = in_ch + 1 + 2 * self.harm_K + self.cheat_in_ch

        ch = [c_in] + [base] * depth
        self.blocks = nn.ModuleList([Conv2DBlock(ch[i], ch[i + 1]) for i in range(depth)])
        self.out_ch = ch[-1]

    def _harmonic_maps(self, angles: torch.Tensor, V: int, U: int, B: int, device, dtype):
        """
        angles: [A]
        returns broadcasted harmonic maps for each BA sample: [B*A, 2K, V, U]
        """
        A = angles.numel()
        if self.harm_K <= 0:
            return torch.zeros(B * A, 0, V, U, device=device, dtype=dtype)

        k = torch.arange(1, self.harm_K + 1, device=device, dtype=angles.dtype).view(1, -1)  # [1,K]
        ang = angles.view(-1, 1)  # [A,1]
        s = torch.sin(k * ang)    # [A,K]
        c = torch.cos(k * ang)    # [A,K]
        hc = torch.stack([s, c], dim=2).reshape(A, 2 * self.harm_K)  # [A,2K]
        hc = hc.view(1, A, 2 * self.harm_K, 1, 1).expand(B, A, -1, V, U)  # [B,A,2K,V,U]
        hc = hc.reshape(B * A, 2 * self.harm_K, V, U).to(dtype)
        return hc

    def forward(
        self,
        S_in: torch.Tensor,               # [B,A,V,U]
        angles: torch.Tensor,             # [A]
        cheat2d: Optional[torch.Tensor],  # [B,Cc,V,U] or None
        gate: float = 0.0,
    ) -> torch.Tensor:
        B, A, V, U = S_in.shape
        device, dtype = S_in.device, S_in.dtype

        # per-angle sino slices -> [B*A,1,V,U]
        sino_pa = S_in.reshape(B * A, 1, V, U)

        # angle-mean map -> [B,1,V,U] -> tile for each angle -> [B*A,1,V,U]
        mean_map = S_in.mean(dim=1, keepdim=True).expand(-1, A, -1, -1)          # [B,A,V,U]
        mean_map = mean_map.contiguous().reshape(B * A, 1, V, U)

        # harmonic maps -> [B*A, 2K, V, U]
        hmaps = self._harmonic_maps(angles, V, U, B, device, dtype)

        # cheat maps (always provide Cc channels to match layer in_channels)
        Cc = self.cheat_in_ch
        if Cc > 0:
            if (cheat2d is None) or (gate <= 0.0):
                cheat = torch.zeros(B, Cc, V, U, device=device, dtype=dtype)
            else:
                # if channels mismatch (e.g., different dft_K), slice/pad to Cc
                c_in = cheat2d.shape[1]
                if c_in > Cc:
                    cheat = cheat2d[:, :Cc]
                elif c_in < Cc:
                    pad = torch.zeros(B, Cc - c_in, V, U, device=device, dtype=dtype)
                    cheat = torch.cat([cheat2d, pad], dim=1)
                else:
                    cheat = cheat2d
                cheat = cheat * float(gate)
            cheat = cheat.unsqueeze(1).expand(-1, A, -1, -1, -1).reshape(B * A, Cc, V, U)
        else:
            cheat = torch.zeros(B * A, 0, V, U, device=device, dtype=dtype)

        # concat per-angle input channels: [1] + [1] + [2K] + [Cc]
        x = torch.cat([sino_pa, mean_map, hmaps, cheat], dim=1)  # [B*A, C_in, V, U]

        # stacks of 2D convs
        for blk in self.blocks:
            x = blk(x)  # [B*A, C_mid, V, U]

        # reshape back to [B, C_out, A, V, U]
        x = x.view(B, A, -1, V, U).permute(0, 2, 1, 3, 4).contiguous()
        return x
