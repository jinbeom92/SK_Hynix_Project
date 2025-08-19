# =================================================================================================
# Align2Dto3D — Physics‑Consistent 2D→3D Alignment Block
# -------------------------------------------------------------------------------------------------
# Purpose
#   Bridges angle‑indexed detector‑plane features (sinogram domain) and volumetric latent features
#   (voxel domain) by compressing per‑angle 2D features and backprojecting them into 3D using a
#   differentiable, physics‑consistent projector.
#
# Inputs
#   f1 : torch.Tensor [B, C1, A, V, U]  — features from the 1D angle encoder (Enc1_1D)
#   f2 : torch.Tensor [B, C2, A, V, U]  — features from the 2D detector‑plane encoder (Enc2_2D)
#   f3 : torch.Tensor [B, C3, D, H, W] or None — optional volumetric prior (e.g., Enc3_3D)
#
# Pipeline
#   1) Per‑angle compression: concat(f1, f2) → 2D conv stack → [B, bp_ch, A, V, U]
#   2) (Optional) PSF‑transpose: apply detector PSFᵀ per channel for FP/BP symmetry
#   3) Backprojection: for channel chunks of size `bp_chunk`, run `projector.backproject` with
#      non‑reentrant activation checkpointing to limit activation memory → [B, bp_ch, D, H, W]
#   4) Volumetric mixing: (concat f3 if provided) → 1×1×1 Conv + GroupNorm + SiLU
#      → aligned latent volume [B, out_ch, D, H, W]
#
# Memory/Performance Notes
#   • Channel‑chunking (`bp_chunk`) bounds peak VRAM during backprojection.
#   • Uses `torch.utils.checkpoint(..., use_reentrant=False)` to reduce activation memory while
#     preserving autograd correctness and improving debuggability.
#   • Explicit `del` of temporaries and optional `torch.cuda.synchronize()` mitigate fragmentation.
#
# Constructor Args
#   projector      : BaseProjector3D (e.g., Joseph/Siddon)
#   c1, c2, c3     : int — input channel sizes for Enc1/Enc2/Enc3 paths
#   out_ch         : int — output channel size after volumetric mixing
#   n_bp_ch        : int or None — compressed channels for BP (default: heuristic from out_ch)
#   psf            : nn.Module or None — separable Gaussian PSF (sinogram domain)
#   psf_consistent : bool — if True, applies PSFᵀ prior to BP for FP/BP consistency
#
# Output
#   torch.Tensor [B, out_ch, D, H, W] — aligned 3D latent representation
# =================================================================================================
from typing import Optional
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from physics.projector import BaseProjector3D  # FP/BP interface

class Align2Dto3D(nn.Module):
    def __init__(self,
                 projector: BaseProjector3D,
                 c1: int, c2: int, c3: int = 0,
                 out_ch: int = 64,
                 n_bp_ch: Optional[int] = None,
                 psf: Optional[nn.Module] = None,
                 psf_consistent: bool = False):
        super().__init__()
        self.proj = projector
        self.psf = psf
        self.psf_consistent = bool(psf_consistent)

        c_in = int(c1) + int(c2)
        bp_ch = int(n_bp_ch) if n_bp_ch is not None else max(4, min(32, out_ch // 2))

        g2 = max(1, math.gcd(bp_ch, 8))
        self.compress2d = nn.Sequential(
            nn.Conv2d(c_in, bp_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g2, bp_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(bp_ch, bp_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g2, bp_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(bp_ch, bp_ch, 1, bias=True),
        )

        g3 = max(1, math.gcd(out_ch, 8))
        self.mix3d = nn.Sequential(
            nn.Conv3d(bp_ch + max(0, int(c3)), out_ch, 1, bias=True),
            nn.GroupNorm(g3, out_ch),
            nn.SiLU(inplace=True),
        )

        self.bp_chunk = max(1, int(getattr(projector, "c_chunk", 4)))

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C1, A, V, U = f1.shape
        assert f2.shape[0] == B and f2.shape[2:] == (A, V, U)

        x = torch.cat([f1, f2], dim=1)                              # [B,C1+C2,A,V,U]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * A, C1 + f2.shape[1], V, U)
        x = self.compress2d(x)                                      # [B*A,bp_ch,V,U]
        bp_ch = x.shape[1]
        x = x.view(B, A, bp_ch, V, U).permute(0, 2, 1, 3, 4).contiguous()  # [B,bp_ch,A,V,U]

        if self.psf_consistent and (self.psf is not None) and getattr(self.psf, "enabled", False):
            xs = []
            for c in range(bp_ch):
                xs.append(self.psf.transpose(x[:, c]))             # [B,A,V,U]
            x = torch.stack(xs, dim=1)                              # [B,bp_ch,A,V,U]
            del xs

        D, H, W = self.proj.geom.D, self.proj.geom.H, self.proj.geom.W
        latent = torch.zeros(B, bp_ch, D, H, W, device=x.device, dtype=x.dtype)
        for c0 in range(0, bp_ch, self.bp_chunk):
            c1 = min(c0 + self.bp_chunk, bp_ch)
            sino_chunk = x[:, c0:c1].contiguous()                   # [B,c,A,V,U]
            vol_chunk = checkpoint(lambda t: self.proj.backproject(t),
                                   sino_chunk, use_reentrant=False) # [B,c,D,H,W]
            latent[:, c0:c1].add_((vol_chunk))
            del sino_chunk, vol_chunk
            if torch.cuda.is_available(): torch.cuda.synchronize()

        if f3 is not None:
            latent = torch.cat([latent, f3], dim=1)
        out = self.mix3d(latent)
        return out
