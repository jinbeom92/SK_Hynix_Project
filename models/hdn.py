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
[B,1,X,A,Z].  Finally, a Joseph ray‑driven backprojector
applies an unfiltered or filtered inverse Radon transform on each slice,
scaling by θ_span/(2·A_eff) to match scikit‑image’s `iradon(filter=None)`
normalisation.

This module defines helper functions `_gn` and `_has_antialias_arg`, the
`SinoDecoder2D` class for XY→XA decoding with optional output bounding and
anti‑aliasing, and the top‑level `HDNSystem` class integrating all
components.  The `HDNSystem.forward` method can optionally accept voxel
volumes during training to enable the cheat path and produces both the
predicted sinogram and reconstructed volume.
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
from physics.psf import SeparableGaussianPSF2D


def _gn(c: int, prefer: int = 8) -> int:
    """Return a GroupNorm group size based on gcd(c, prefer).

    GroupNorm stabilises training when the number of channels is divisible
    by a small integer.  This helper chooses gcd(c, prefer) but guarantees
    at least 1.
    """
    g = math.gcd(c, prefer)
    return max(1, g if g > 0 else 1)


def _has_antialias_arg() -> bool:
    """Check whether torch.nn.functional.interpolate supports an 'antialias' kwarg."""
    try:
        return "antialias" in inspect.signature(F.interpolate).parameters
    except Exception:
        return False


class SinoDecoder2D(nn.Module):
    """Decode XY feature maps back into XA sinograms.

    Parameters:
        in_ch: Number of input channels from fusion module.
        mid_ch: Intermediate channels in the decoder.
        depth: Number of Conv2D→GroupNorm→ReLU blocks.
        mode: Interpolation mode ("bilinear", "bicubic", "nearest").
        bound: Output bounding ("none", "clamp", "sigmoid").
        antialias: Override for anti‑aliasing; if None, anti‑aliasing is
            applied automatically when downsampling.
        prefer_gn: Preferred divisor for GroupNorm group count.
        az_sanity_check: If True, check whether angle and depth axes have
            been accidentally swapped by looking at variance.

    Inputs:
        F_xy: Feature tensor [B,C,X,Y,Z] or [B,C,X,Y].
        A: Target number of angles to upsample to.

    Returns:
        Predicted sinogram [B,1,X,A,Z].
    """

    def __init__(
        self,
        in_ch: int,
        mid_ch: int = 64,
        depth: int = 2,
        mode: str = "bilinear",
        bound: str = "none",
        antialias: Optional[bool] = None,
        prefer_gn: int = 8,
        az_sanity_check: bool = False,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.bound = str(bound).lower()
        self.antialias = antialias
        self._aa_supported = _has_antialias_arg()
        self.az_sanity_check = bool(az_sanity_check)
        # Build convolutional mixing layers
        ch = [in_ch] + [mid_ch] * depth
        layers = []
        for i in range(depth):
            layers.extend(
                [
                    nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(_gn(ch[i + 1], prefer=prefer_gn), ch[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
        self.mix = nn.Sequential(*layers)
        self.proj = nn.Conv2d(ch[-1], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def _interpolate_xy_to_xa(self, x: torch.Tensor, tgt_x: int, tgt_a: int) -> torch.Tensor:
        """Resize from (X,Y) to (X,A) with the specified mode."""
        size = (int(tgt_x), int(tgt_a))
        kwargs = {}
        if self.mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        if self._aa_supported:
            if self.antialias is None:
                is_down = (size[0] < x.shape[2]) or (size[1] < x.shape[3])
                kwargs["antialias"] = bool(is_down)
            else:
                kwargs["antialias"] = bool(self.antialias)
        return F.interpolate(x, size=size, mode=self.mode, **kwargs)

    def _bound_out(self, t: torch.Tensor) -> torch.Tensor:
        """Apply optional output bounding."""
        if self.bound == "none":
            return t
        if self.bound == "clamp":
            return t.clamp_(0.0, 1.0)
        if self.bound == "sigmoid":
            return torch.sigmoid(t)
        raise ValueError(f"Unknown bound option: {self.bound}")

    @torch.no_grad()
    def _sanity_check_az(self, s: torch.Tensor) -> None:
        """Warn if angle and depth axes appear swapped based on variance."""
        if (not self.az_sanity_check) or (s.ndim != 5):
            return
        _, _, _, A, Z = s.shape
        if A != Z:
            return
        d = s - s.mean(dim=3, keepdim=True)
        var_A = d.pow(2).mean(dim=3).mean().item()
        var_Z = d.pow(2).mean(dim=4).mean().item()
        if var_A < 1e-10 and var_Z > var_A * 10:
            print(
                "[warn][SinoDecoder2D] A-collapse vs Z-variance detected. "
                "Verify: stacking must be along A, not Z."
            )

    def forward(self, F_xy: torch.Tensor, A: int) -> torch.Tensor:
        if F_xy.dim() == 5:
            B, C, X, Y, Z = F_xy.shape
            x = F_xy.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, Y)
        elif F_xy.dim() == 4:
            B, C, X, Y = F_xy.shape
            Z = 1
            x = F_xy.view(B * Z, C, X, Y)
        else:
            raise ValueError(f"SinoDecoder2D expects [B,C,X,Y,(Z)], got {F_xy.shape}")
        x = self.mix(x)
        x = self._interpolate_xy_to_xa(x, X, A)
        x = self.proj(x)  # [B*Z,1,X,A]
        sino = x.view(B, Z, 1, X, A).permute(0, 2, 3, 4, 1).contiguous()
        self._sanity_check_az(sino)
        return self._bound_out(sino)


class HDNSystem(nn.Module):
    """Top‑level HDN system with encoders, alignment, cheat, fusion, decoding and BP.

    Parameters:
        cfg: Configuration dictionary containing model/projector settings.
        projector: A physics projector/backprojector implementing backproject().

    The forward method takes a sinogram [B,1,X,A,Z] and an optional voxel
    volume [B,1,X,Y,Z] (cheat) and returns the predicted sinogram and
    reconstructed volume.  During training, train_mode=True enables the
    cheat path; otherwise the cheat path is bypassed.
    """

    def __init__(self, cfg: dict, projector: nn.Module) -> None:
        super().__init__()
        if projector is None:
            raise ValueError("projector must implement backproject()")
        self.projector = projector
        m = cfg.get("model", cfg)
        dbg = cfg.get("debug", {})
        # Propagate projector attributes from cfg
        proj_cfg = cfg.get("projector", {})
        for k in ("ir_impl", "ir_interpolation", "ir_circle", "bp_span", "dc_mode", "ir_filter"):
            if hasattr(self.projector, k) and (k in m or k in proj_cfg):
                setattr(self.projector, k, (m.get(k) if k in m else proj_cfg.get(k)))
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
            mode=str(sd_cfg.get("interp_mode", align_cfg.get("interp_mode", "bilinear"))),
            bound=str(sd_cfg.get("bound", "none")).lower(),
            antialias=sd_cfg.get("antialias", None),
            prefer_gn=int(sd_cfg.get("prefer_gn", 8)),
            az_sanity_check=bool(dbg.get("az_sanity_check", False)),
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
        """Run the HDN forward pass.

        Args:
            sino_xaz: Sinogram tensor [B,1,X,A,Z].
            v_vol: Ground‑truth voxel volume [B,1,X,Y,Z] or None.
            train_mode: When True and cheat is enabled, use cheat path.

        Returns:
            sino_hat_xaz: Predicted sinogram [B,1,X,A,Z].
            recon_xyz: Reconstructed volume [B,1,X,Y,Z].
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
        sino_hat_xaz = self.sino_dec(fused_xy, A=A)
        sino_for_bp = sino_hat_xaz
        # Optional PSF
        if self.psf.enabled:
            if getattr(self.psf, "angle_variant", False):
                need_cfg = (
                    not hasattr(self.psf, "_A")
                    or (getattr(self.psf, "_A", None) != A)
                )
                if need_cfg and hasattr(self.psf, "configure"):
                    self.psf.configure(A, device=sino_for_bp.device, dtype=sino_for_bp.dtype)
            sino_for_bp = self.psf(sino_for_bp)
        # Backproject
        recon_xyz = self.projector.backproject(sino_for_bp)
        if self.clamp_recon:
            recon_xyz = recon_xyz.clamp(0.0, 1.0)
        return sino_hat_xaz, recon_xyz
