"""
High‑Dimensional Network (HDN) implementation with encoders, alignment,
cheat fusion, 2D decoding, optional PSF and physics‑based backprojection.

The HDN architecture follows the pipeline Enc1→Enc2→Sino2XYAlign→VoxelCheat2D
(optional)→Fusion2D→SinoDecoder2D→Backprojection.  Sinogram inputs
[B,1,X,A,Z] are processed slice‑wise along the depth dimension.  Enc1
performs 1D convolutions along the angle axis; Enc2 performs 2D convolutions
over the (X×A) plane; Sino2XYAlign resamples these features onto an (X×Y)
Cartesian grid using bilinear/bicubic interpolation; VoxelCheat2D encodes
ground truth voxel slices to serve as hints during training; Fusion2D mixes
sinogram and cheat features; SinoDecoder2D upsamples from (X,Y) back to
(X,A) and stacks along the angle dimension to form a predicted sinogram
[B,1,X,A,Z].  Finally, a Joseph ray‑driven backprojector applies a
scikit‑image‑like filtered inverse Radon transform on each slice.  If a
360° sinogram is provided, the backprojector automatically averages opposing
angles to form a 180° sinogram before inversion and scales the result by
π/(2·A_eff), matching scikit‑image’s normalisation:contentReference[oaicite:0]{index=0}.

This module defines helper functions `_gn` and `_has_antialias_arg`, the
`SinoDecoder2D` class for XY→XA decoding with optional output bounding and
anti‑aliasing, and the top‑level `HDNSystem` class integrating all
components.  The `HDNSystem.forward` method can optionally accept voxel
volumes during training to enable the cheat path and produces both the
predicted sinogram and reconstructed volume.  A residual skip connection
is applied between the input sinogram and the decoder output so that the
network learns only a correction term; this helps propagate low‑frequency
information and improves gradient flow:contentReference[oaicite:1]{index=1}.
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .enc1_1d import Enc1_1D_Angle
from .enc2_2d import Enc2_2D_Sino
from .align import Sino2XYAlign
from .fusion import VoxelCheat2D, Fusion2D
from .decoder import SinoDecoder2D
from physics.psf import SeparableGaussianPSF2D


def _gn(c: int, prefer: int = 8) -> int:
    """Return a GroupNorm group size based on gcd(c, prefer)."""
    g = math.gcd(c, prefer)
    return max(1, g if g > 0 else 1)


def _has_antialias_arg() -> bool:
    """Check whether torch.nn.functional.interpolate supports an 'antialias' kwarg."""
    try:
        return "antialias" in inspect.signature(F.interpolate).parameters
    except Exception:
        return False


class HDNSystem(nn.Module):
    """Top‑level HDN system with encoders, alignment, cheat, fusion, decoding and BP."""

    def __init__(self, cfg: dict, projector: nn.Module) -> None:
        super().__init__()
        if projector is None:
            raise ValueError("projector must implement backproject()")
        self.projector = projector
        m = cfg.get("model", cfg)
        dbg = cfg.get("debug", {})

        # Propagate only supported projector attributes (ir_circle, fbp_filter)
        proj_cfg = cfg.get("projector", {})
        attr_map = {
            "ir_circle": "ir_circle",
            "fbp_filter": "fbp_filter",
            "ir_filter": "fbp_filter",  # legacy key maps to fbp_filter
        }
        for cfg_key, attr_name in attr_map.items():
            if hasattr(self.projector, attr_name):
                if cfg_key in m or cfg_key in proj_cfg:
                    val = m.get(cfg_key) if cfg_key in m else proj_cfg.get(cfg_key)
                    if val is not None:
                        setattr(self.projector, attr_name, val)

        # Encoders
        self.enc1 = Enc1_1D_Angle(
            base=int(m.get("enc1", {}).get("base", 32)),
            depth=int(m.get("enc1", {}).get("depth", 3)),
        )
        self.enc2 = Enc2_2D_Sino(
            base=int(m.get("enc2", {}).get("base", 32)),
            depth=int(m.get("enc2", {}).get("depth", 3)),
        )

        # Sino→XY alignment
        align_cfg = m.get("align", {})
        self.align = Sino2XYAlign(
            in_ch=self.enc1.out_ch + self.enc2.out_ch,
            out_ch=int(align_cfg.get("out_ch", 64)),
            depth=int(align_cfg.get("depth", 2)),
            mode=str(align_cfg.get("interp_mode", "bilinear")),
        )

        # Cheat path
        c_cfg = m.get("cheat2d", {})
        self.cheat_enabled = bool(c_cfg.get("enabled", True))
        self.cheat = VoxelCheat2D(
            base=int(c_cfg.get("base", 16)),
            depth=int(c_cfg.get("depth", 2)),
        )

        # Fusion
        self.fusion = Fusion2D(
            in_ch_sino=int(align_cfg.get("out_ch", 64)),
            in_ch_cheat=(self.cheat.out_ch if self.cheat_enabled else 0),
            out_ch=int(m.get("fusion", {}).get("out_ch", 64)),
        )

        # Decoder
        sd_cfg = m.get("sino_dec", {})
        self.sino_dec = SinoDecoder2D(
            in_ch=self.fusion.out_ch,
            mid_ch=int(sd_cfg.get("mid_ch", 64)),
            depth=int(sd_cfg.get("depth", 2)),
            interp_mode=str(sd_cfg.get("interp_mode", align_cfg.get("interp_mode", "bilinear"))),
            bound=str(sd_cfg.get("bound", "none")).lower(),
        )

        # PSF
        psf_cfg = cfg.get("psf", {})
        self.psf = SeparableGaussianPSF2D(
            enabled=bool(psf_cfg.get("enabled", False)),
            angle_variant=bool(psf_cfg.get("angle_variant", False)),
            sigma_u=float(psf_cfg.get("sigma_u", 0.7)),
            sigma_v=float(psf_cfg.get("sigma_v", 0.7)),
        )
        self.clamp_recon = bool(m.get("clamp_recon", False))
        self.strict_geometry = bool(m.get("strict_geometry", dbg.get("strict_geometry", False)))

    @staticmethod
    def _to_b1xaz(sino: torch.Tensor) -> torch.Tensor:
        """Ensure a sinogram has shape [B,1,X,A,Z]."""
        if sino.dim() == 5:
            return sino
        if sino.dim() == 4:
            return sino.unsqueeze(1)
        raise ValueError(f"sino must be [B,C,X,A,Z] or [B,X,A,Z], got {sino.shape}")

    @staticmethod
    def _to_b1xyz(vol: torch.Tensor) -> torch.Tensor:
        """Ensure a voxel volume has shape [B,1,X,Y,Z]."""
        if vol is None:
            return None
        if vol.dim() == 5:
            return vol
        if vol.dim() == 4:
            if vol.shape[1] != 1:
                raise ValueError(f"volume channel must be 1, got {vol.shape}")
            return vol
        if vol.dim() == 3:
            return vol.unsqueeze(1)
        raise ValueError(f"V_vol must be [B,1,X,Y,Z] or [B,X,Y,Z], got {vol.shape}")

    def forward(
        self,
        sino_xaz: torch.Tensor,
        v_vol: Optional[torch.Tensor] = None,
        train_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the HDN forward pass.

        Args
        ----
        sino_xaz : Tensor [B,1,X,A,Z]
            Input sinogram.
        v_vol : Tensor [B,1,X,Y,Z] or None
            Ground-truth voxel volume used as a cheat during training.
        train_mode : bool
            If True and cheat is enabled, include the cheat path.

        Returns
        -------
        sino_hat_xaz : Tensor [B,1,X,A,Z]
            Predicted sinogram after residual skip.
        recon_xyz : Tensor [B,1,X,Y,Z]
            Reconstructed volume via physics-based backprojection.
        """
        S = self._to_b1xaz(sino_xaz)
        V = self._to_b1xyz(v_vol) if self.cheat_enabled else None
        B, _, X, A, Z = S.shape

        # Geometry checks and target Y dimension
        geom = getattr(self.projector, "geom", None)
        if geom is not None:
            if hasattr(geom, "A") and (A != geom.A):
                raise ValueError(f"A mismatch: input A={A} vs geom.A={geom.A}")
            Y_target = int(getattr(geom, "H", (V.shape[3] if V is not None else X)))
            if self.strict_geometry:
                candidates = []
                if hasattr(geom, "U"):
                    candidates.append(int(geom.U))
                if hasattr(geom, "W"):
                    candidates.append(int(geom.W))
                if candidates and all(X != c for c in candidates):
                    raise ValueError(f"X mismatch under strict_geometry: X={X} vs geom.U/W={candidates}")
        else:
            Y_target = (V.shape[3] if V is not None else X)

        # Encoders
        f1 = self.enc1(S)
        f2 = self.enc2(S)
        f_sino = torch.cat([f1, f2], 1)

        # Sino→XY alignment
        f_xy = self.align(f_sino, (X, Y_target))

        # Optional cheat fusion
        if self.cheat_enabled and train_mode and (V is not None):
            cheat_xy = self.cheat(V)
            fused_xy = self.fusion(f_xy, cheat_xy)
        else:
            fused_xy = self.fusion(f_xy, None)

        # Decode to [X,A,Z]
        sino_hat_xaz = self.sino_dec(fused_xy, target_A=A)

        # Residual skip connection (add input sinogram)
        sino_hat_xaz = sino_hat_xaz + S

        # Optional PSF convolution
        sino_for_bp = sino_hat_xaz
        if self.psf.enabled:
            if getattr(self.psf, "angle_variant", False):
                need_cfg = (
                    not hasattr(self.psf, "_A")
                    or (getattr(self.psf, "_A", None) != A)
                )
                if need_cfg and hasattr(self.psf, "configure"):
                    self.psf.configure(A, device=sino_for_bp.device, dtype=sino_for_bp.dtype)
            sino_for_bp = self.psf(sino_for_bp)

        # Backproject using the configured projector (scikit-image compatible)
        recon_xyz = self.projector.backproject(sino_for_bp)
        if self.clamp_recon:
            recon_xyz = recon_xyz.clamp(0.0, 1.0)

        return sino_hat_xaz, recon_xyz
