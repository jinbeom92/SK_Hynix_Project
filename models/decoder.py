# =================================================================================================
# DecoderSinogram — 3D→2D Physics-Guided Projection with Chunked, Checkpointed FP
# -------------------------------------------------------------------------------------------------
# Purpose
#   Converts a volumetric latent representation into an angle-indexed sinogram by applying
#   a shallow 3D convolutional head followed by a physics-consistent forward projector.
#   Projection is executed per-channel in chunks to bound peak memory, and each chunk is
#   wrapped in non‑reentrant activation checkpointing to reduce activation storage.
#
# Inputs
#   latent3d : torch.Tensor [B, C_in, D, H, W]
#     Volumetric latent features produced by the alignment stage.
#
# Pipeline
#   1) 3D Head: Conv3D → GroupNorm → SiLU → 1×1×1 Conv to re‑mix channels.
#   2) Chunked FP: For channel blocks of size `proj_chunk`, invoke the projector’s
#      forward operator (FP) under checkpointing:
#          sino_chunk = projector(vol_chunk)   # → [B, c, A, V, U]
#   3) Accumulation: Write each chunk into a preallocated accumulator and, after all
#      chunks, sum over channels to obtain the final sinogram:
#          sino = sino_accum.sum(dim=1)        # → [B, A, V, U]
#
# Memory/Performance Notes
#   • `proj_chunk` is derived from the projector’s `c_chunk` to keep FP consistent with
#     projector-side streaming knobs.
#   • Uses `torch.utils.checkpoint(..., use_reentrant=False)` to lower activation memory
#     without sacrificing gradient flow or debuggability.
#   • Explicit deletion of temporaries and optional `torch.cuda.synchronize()` help
#     mitigate fragmentation and smooth memory spikes.
#
# Constructor Args
#   projector : BaseProjector3D  — differentiable physics FP/BP backend (e.g., Joseph/Siddon)
#   in_ch     : int              — number of input channels in the volumetric latent
#   mid_ch    : int (default=64) — width of the 3D head
#   n_proj_ch : int (default=4)  — fallback used if projector does not expose `c_chunk`
#
# Output
#   torch.Tensor [B, A, V, U] — per‑angle sinogram predicted from the 3D latent
# =================================================================================================
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from physics.projector import BaseProjector3D  # FP/BP interface

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, gn_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, p)
        g = math.gcd(out_ch, gn_groups); g = 1 if g <= 0 else g
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.gn(self.conv(x)))

class DecoderSinogram(nn.Module):
    def __init__(self, projector: BaseProjector3D, in_ch: int, mid_ch: int = 64, n_proj_ch: int = 4):
        super().__init__()
        self.proj = projector
        self.proj_chunk = max(1, int(getattr(projector, "c_chunk", n_proj_ch)))
        self.head = nn.Sequential(
            Conv3DBlock(in_ch, mid_ch),
            nn.Conv3d(mid_ch, in_ch, kernel_size=1, bias=True),
        )

    def forward(self, latent3d: torch.Tensor) -> torch.Tensor:
        x = self.head(latent3d)  # [B,C,D,H,W]
        B, C, D, H, W = x.shape
        A = self.proj.geom.A; V = self.proj.geom.V; U = self.proj.geom.U

        sino_accum = torch.zeros(B, C, A, V, U, device=x.device, dtype=x.dtype)
        for c0 in range(0, C, self.proj_chunk):
            c1 = min(c0 + self.proj_chunk, C)
            vol_chunk = x[:, c0:c1].contiguous()
            sino_chunk = checkpoint(lambda t: self.proj(t),
                                    vol_chunk, use_reentrant=False)  # [B,c,A,V,U]
            sino_accum[:, c0:c1].add_(sino_chunk)
            del vol_chunk, sino_chunk
            if torch.cuda.is_available(): torch.cuda.synchronize()

        # c.sum → [B,A,V,U]
        sino = sino_accum.sum(dim=1)
        del sino_accum
        return sino
