# =================================================================================================
# HDNSystem — High‑Dimensional Neural Tomographic Reconstruction Core
# -------------------------------------------------------------------------------------------------
# Purpose
#   End‑to‑end sinogram→volume model that couples learned encoders/decoders with a
#   differentiable, physics‑consistent CT operator. The network predicts a sinogram
#   proxy from a 3D latent and enforces data fidelity through forward/back‑projection
#   consistency, while volumetric losses supervise the reconstruction quality.
#
# Architecture
#   • Encoders
#       - Enc1_1D: angle‑axis 1D encoder over sinogram angles → [B, C1, A, V, U]
#       - Enc2_2D: per‑angle 2D encoder with harmonic angle embeddings (+ optional cheat)
#                  → [B, C2, A, V, U]
#       - Enc3_3D (optional): volumetric encoder over V_gt (training‑only prior)
#                  → [B, C3, D, H, W]
#   • Align (2D→3D)
#       - Concatenates Enc1/Enc2 features (and optionally PSFᵀ), then backprojects per‑channel
#         in chunks with activation checkpointing → latent volume [B, C*, D, H, W]
#       - Concatenates Enc3 (if enabled) and mixes via 1×1×1 Conv + GN + SiLU
#   • Decoder (3D→2D)
#       - Shallow 3D head, then per‑channel forward projection in chunks with checkpointing
#         → predicted sinogram `sino_hat` ≥ 0
#   • Physics Head
#       - Optional PSFᵀ, Backproject(sino_hat) → R_hat, then ForwardProject(R_hat) → S_pred
#         (+ optional PSF) to form a data‑consistency path
#
# Geometry/Physics Integration
#   • `Parallel3DGeometry` provides (D,H,W), (V,U), angles, and sampling/spacing.
#   • `make_projector(method)` selects the CT operator (e.g., Joseph or Siddon).
#   • `rebind_geometry(...)` hot‑swaps geometry (angles, shapes, spacing) at runtime so a
#     single model can be trained across multiple resolutions/angle sets.
#   • PSF module in sinogram domain is kept in sync with active geometry (angle count).
#
# Memory/Throughput Considerations
#   • Channel/step/angle chunking parameters are propagated to Align/Decoder to bound peak
#     memory during FP/BP.
#   • Non‑reentrant activation checkpointing around FP/BP reduces activation storage while
#     preserving gradients.
#
# Cheat‑Gate (Training‑Only Prior)
#   • If enabled, derives 2D statistics (mean/std across depth) from V_gt, gates them, and
#     injects into Enc2_2D to provide a weak volumetric prior at train time (disabled at eval).
#
# I/O Shapes
#   • Input:  S_in  [B, A, V, U], angles [A], (optional) V_gt [B, 1, D, H, W]
#   • Output: sino_hat [B, A, V, U], R_hat [B, 1, D, H, W], S_pred [B, A, V, U]
#
# Usage
#   model = HDNSystem(geom, cfg)
#   sino_hat, R_hat, S_pred = model(S_in, angles, V_gt=..., train_mode=True/False)
# =================================================================================================
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from physics.geometry import Parallel3DGeometry
from physics.projector import BaseProjector3D, make_projector
from physics.psf import SeparableGaussianPSF2D
from .align import Align2Dto3D
from .decoder import DecoderSinogram
from .enc1_1d import Enc1_1D
from .enc2_2d import Enc2_2D
from .enc3_3d import Enc3_3D

class HDNSystem(nn.Module):
    def __init__(self, geom: Parallel3DGeometry, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.method = cfg["projector"].get("method", "joseph3d")

        # Projector + streaming knobs
        self.projector: BaseProjector3D = make_projector(self.method, geom)
        jp = self.projector
        if hasattr(jp, "c_chunk"):    jp.c_chunk   = int(cfg["projector"].get("c_chunk", jp.c_chunk))
        if hasattr(jp, "step_chunk"): jp.step_chunk= int(cfg["projector"].get("step_chunk", getattr(jp, "step_chunk", 8)))

        # PSF (sinogram domain, self-adjoint for Gaussian)
        psf_cfg = cfg["projector"].get("psf", {"enabled": False})
        self.psf = SeparableGaussianPSF2D(enabled=psf_cfg.get("enabled", False),
                                          angle_variant=psf_cfg.get("angle_variant", False),
                                          sigma_u=float(psf_cfg.get("sigma_u", 0.7)),
                                          sigma_v=float(psf_cfg.get("sigma_v", 0.7)))
        self.psf_consistent = bool(cfg.get("cheat", {}).get("psf_consistent", False))
        self.psf.configure(A=geom.A, device=geom.angles.device, dtype=geom.angles.dtype)

        # Encoders
        mcfg = cfg.get("model", {})
        e1 = mcfg.get("enc1", {"base": 16, "depth": 2})
        e2 = mcfg.get("enc2", {"base": 16, "depth": 2, "harm_K": 2})
        self.enc1 = Enc1_1D(in_ch=1, base=int(e1.get("base", 16)), depth=int(e1.get("depth", 2)))
        self.enc2 = Enc2_2D(in_ch=1, base=int(e2.get("base", 16)), depth=int(e2.get("depth", 2)),
                            harm_K=int(e2.get("harm_K", 2)), cheat_in_ch=2)

        # Optional volumetric encoder (honor cfg)
        e3 = mcfg.get("enc3", {"enabled": False, "base": 8, "depth": 2})
        self.enc3_enabled = bool(e3.get("enabled", False))
        self.enc3 = Enc3_3D(base=int(e3.get("base", 8)), depth=int(e3.get("depth", 2))) if self.enc3_enabled else None

        # Align 2D→3D (BP per-channel, PSF-consistent option)
        align_out = int(mcfg.get("align", {}).get("out_ch", 64))
        bp_ch_opt = int(mcfg.get("align", {}).get("bp_ch", max(4, align_out // 2)))
        self.align = Align2Dto3D(projector=self.projector,
                                 c1=self.enc1.out_ch, c2=self.enc2.out_ch,
                                 c3=(self.enc3.out_ch if self.enc3_enabled else 0),
                                 out_ch=align_out, n_bp_ch=bp_ch_opt,
                                 psf=self.psf, psf_consistent=self.psf_consistent)

        # Decoder 3D→2D (FP per-channel)
        dcfg = mcfg.get("dec", {"mid_ch": 64, "n_proj_ch": 4})
        self.dec = DecoderSinogram(self.projector, in_ch=align_out,
                                   mid_ch=int(dcfg.get("mid_ch", 64)),
                                   n_proj_ch=int(dcfg.get("n_proj_ch", 4)))
        if hasattr(self.projector, "c_chunk"):
            self.dec.proj_chunk = max(1, int(self.projector.c_chunk))
            self.align.bp_chunk = max(1, int(self.projector.c_chunk))

        # Cheat control
        self.cheat_cfg      = cfg.get("cheat", {"enabled": False, "aggregate": "angle_mean",
                                                "dft_K": 4, "gate": 0.0, "train_only": True,
                                                "psf_consistent": False})
        self.cheat_enabled   = bool(self.cheat_cfg.get("enabled", False))
        self.cheat_gate      = float(self.cheat_cfg.get("gate", 0.0))
        self.cheat_train_only= bool(self.cheat_cfg.get("train_only", True))

    @torch.no_grad()
    def rebind_geometry(self, geom: Parallel3DGeometry, angles: torch.Tensor):
        """Swap geometry/projector on-the-fly for multi-resolution training."""
        method = self.cfg["projector"].get("method", self.method)
        if method != self.method:
            self.method = method
            self.projector = make_projector(self.method, geom)
            self.align.proj = self.projector
            self.dec.proj   = self.projector
        else:
            self.projector.reset_geometry(geom)

        self.psf.configure(A=geom.A, device=angles.device, dtype=angles.dtype)
        self.align.psf = self.psf
        if hasattr(self.projector, "c_chunk"):
            self.dec.proj_chunk = max(1, int(self.projector.c_chunk))
            self.align.bp_chunk = max(1, int(self.projector.c_chunk))

    def _voxel_to_2d_cheat(self, V_gt: torch.Tensor, V: int, U: int) -> torch.Tensor:
        mean_map = V_gt.mean(dim=2)
        std_map  = V_gt.std(dim=2, unbiased=False) + 1e-6
        cheat = torch.cat([mean_map, std_map], dim=1)
        return F.interpolate(cheat, size=(V, U), mode='bilinear', align_corners=False)

    def forward(self, S_in: torch.Tensor, angles: torch.Tensor,
                V_gt: Optional[torch.Tensor] = None, train_mode: bool = True):
        use_cheat = self.cheat_enabled and (train_mode or (not self.cheat_train_only))
        B, A, V_dim, U_dim = S_in.shape

        f1 = self.enc1(S_in)  # [B,C1,A,V,U]
        if use_cheat and (V_gt is not None):
            with torch.no_grad():
                cheat2d = self._voxel_to_2d_cheat(V_gt, V_dim, U_dim)
        else:
            cheat2d = torch.zeros((B, 2, V_dim, U_dim), device=S_in.device, dtype=S_in.dtype)
        f2 = self.enc2(S_in, angles, cheat2d=cheat2d, gate=self.cheat_gate)
        f3 = self.enc3(V_gt) if (self.enc3_enabled and V_gt is not None) else None

        latent3d = self.align(f1, f2, f3)
        sino_hat = self.dec(latent3d)                           # [B,A,V,U]
        sino_for_bp = self.psf.transpose(sino_hat) if self.psf.enabled else sino_hat
        R_hat = self.projector.backproject(sino_for_bp.unsqueeze(1))  # [B,1,D,H,W]
        S_pred = self.projector(R_hat).squeeze(1)                     # [B,A,V,U]
        if self.psf.enabled:
            S_pred = self.psf(S_pred)
        return sino_hat, R_hat, S_pred
