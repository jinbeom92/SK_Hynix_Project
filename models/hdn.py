"""
HDN system (encoders → align → fusion → sino decoder → physics BP).

Overview
--------
This module defines the high-level HDNSystem used for sinogram-to-volume
reconstruction. The forward path is:

  Enc1_1D_Angle + Enc2_2D_Sino  →  Sino2XYAlign  →  (optional VoxelCheat2D) + Fusion2D
  →  SinoDecoder2D (XY→XA, **per z-slice**)  →  physics backprojection (unfiltered BP).

Axis conventions
----------------
• Sinogram S[x,a,z] is used as [B, 1, X, A, Z], i.e., detector=X, angle=A, depth=Z.
• Volumes are [B, 1, X, Y, Z].
• The decoder **must stack along angle A** (never Z). Any Z-tiling would
  over-accumulate during BP and whiten the reconstruction.

Backprojection reference
------------------------
The projector's unfiltered backprojection mirrors scikit-image's `iradon` with
`filter_name=None` (circular support mask + π/(2·A) scaling). See our physics
projector implementation for details. :contentReference[oaicite:0]{index=0}
"""

from typing import Optional, Tuple
import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .enc1_1d import Enc1_1D_Angle                 # [B,1,X,A,Z] → [B,C1,X,A,Z]  :contentReference[oaicite:1]{index=1}
from .enc2_2d import Enc2_2D_Sino                  # [B,1,X,A,Z] → [B,C2,X,A,Z]  :contentReference[oaicite:2]{index=2}
from .align   import Sino2XYAlign                  # [B,C,X,A,Z] → [B,Ca,X,Y,Z]  :contentReference[oaicite:3]{index=3}
from .fusion  import VoxelCheat2D, Fusion2D        # [B,*,X,Y,Z] fuse            :contentReference[oaicite:4]{index=4}
from physics.psf import SeparableGaussianPSF2D     # optional PSF over sino(x,a,z) :contentReference[oaicite:5]{index=5}


# ------------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------------
def _gn(c: int, prefer: int = 8) -> int:
    """Group count for GroupNorm = gcd(c, prefer), clamped to [1, c]."""
    g = math.gcd(c, prefer)
    return max(1, g if g > 0 else 1)


def _has_antialias_arg() -> bool:
    """Return True if torch.nn.functional.interpolate supports the `antialias` kwarg."""
    try:
        return "antialias" in inspect.signature(F.interpolate).parameters
    except Exception:
        return False


# ------------------------------------------------------------------------------------------
# XY → XA (per z-slice) decoder
# ------------------------------------------------------------------------------------------
class SinoDecoder2D(nn.Module):
    """
    Lightweight decoder mapping fused XY features to a single-channel sinogram (XA),
    applied **per z-slice**.

    Shape contract
    --------------
    Input : [B, C, X, Y, Z] (or [B, C, X, Y] treated as Z=1)
    Output: [B, 1, X, A, Z] — stacking axis is **A** (angle)

    Notes
    -----
    • Internally flattens (B,Z) into batch for 2D conv on (X,Y).
    • Resize (X,Y) → (X,A) with deterministic settings (align_corners=False for bilinear/bicubic).
    • Optional output bounding: {"none","clamp","sigmoid"}.
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
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.bound = str(bound).lower()
        self.antialias = antialias
        self._aa_supported = _has_antialias_arg()
        self.az_sanity_check = bool(az_sanity_check)

        ch = [in_ch] + [mid_ch] * depth
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_gn(ch[i + 1], prefer=prefer_gn), ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.mix = nn.Sequential(*layers)
        self.proj = nn.Conv2d(ch[-1], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def _interpolate_xy_to_xa(self, x: torch.Tensor, tgt_x: int, tgt_a: int) -> torch.Tensor:
        """Resize a 4D tensor [N,C,X,Y] to [N,C,X,A] with deterministic-friendly settings."""
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
        if self.bound == "none":
            return t
        elif self.bound == "clamp":
            return t.clamp_(0.0, 1.0)
        elif self.bound == "sigmoid":
            return torch.sigmoid(t)
        else:
            raise ValueError(f"Unknown bound option: {self.bound}")

    @torch.no_grad()
    def _sanity_check_az(self, sino_b1xaz: torch.Tensor) -> None:
        """
        Light-weight A↔Z sanity: if content collapses along A while varying along Z (a sign
        of 'stacked-along-Z'), print a warning to surface the bug early.
        """
        if not self.az_sanity_check:
            return
        if sino_b1xaz.ndim != 5:
            return
        _, _, _, A, Z = sino_b1xaz.shape
        if A != Z:
            return
        s = sino_b1xaz - sino_b1xaz.mean(dim=3, keepdim=True)
        var_A = s.pow(2).mean(dim=3).mean().item()
        var_Z = s.pow(2).mean(dim=4).mean().item()
        if var_A < 1e-10 and var_Z > var_A * 10:
            print("[warn][SinoDecoder2D] Detected A-collapse vs Z-variance; "
                  "verify decoder stacks along A, not Z.")

    def forward(self, F_xy: torch.Tensor, A: int) -> torch.Tensor:
        """
        Parameters
        ----------
        F_xy : Tensor
            Fused features in XY layout, [B,C,X,Y,(Z)].
        A : int
            Number of projection angles to produce along the **A** axis.

        Returns
        -------
        Tensor
            Predicted sinogram [B,1,X,A,Z].
        """
        if F_xy.dim() == 5:
            B, C, X, Y, Z = F_xy.shape
            x = F_xy.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, Y)  # [B*Z,C,X,Y]
        elif F_xy.dim() == 4:
            B, C, X, Y = F_xy.shape
            Z = 1
            x = F_xy.view(B * Z, C, X, Y)
        else:
            raise ValueError(f"SinoDecoder2D expects [B,C,X,Y,(Z)], got {tuple(F_xy.shape)}")

        # Local conv mixing at the native XY resolution
        x = self.mix(x)  # [B*Z, mid, X, Y]

        # Deterministic resize: fix X, map Y→A
        x = self._interpolate_xy_to_xa(x, tgt_x=X, tgt_a=A)  # [B*Z, mid, X, A]

        # Project to a single-channel sinogram
        x = self.proj(x)  # [B*Z, 1, X, A]

        # Restore [B,1,X,A,Z] (stack along **A** only)
        sino = x.view(B, Z, 1, X, A).permute(0, 2, 3, 4, 1).contiguous()  # [B,1,X,A,Z]

        self._sanity_check_az(sino)
        return self._bound_out(sino)


# ------------------------------------------------------------------------------------------
# HDN System
# ------------------------------------------------------------------------------------------
class HDNSystem(nn.Module):
    """
    High-level HDN wrapper that couples encoders/decoders with a physics projector.

    Components
    ----------
    • enc1 (1D over angles)  + enc2 (2D over X×A)    →  sino features.  
    • align (XA→XY)                                  →  XY features.     :contentReference[oaicite:7]{index=7}
    • (optional) cheat features from GT voxel        →  fusion in XY.    :contentReference[oaicite:8]{index=8}
    • sino_dec (XY→XA per z-slice; stacks along A).                     (this file)
    • projector.backproject for unfiltered BP (π/(2·A), circle mask).   :contentReference[oaicite:9]{index=9}

    Notes
    -----
    • Forward supports a 'cheat' path only during training (train_mode=True and v_vol provided).
    • Geometry checks (A vs geom.A, etc.) can be enforced via `strict_geometry`.
    """

    def __init__(self, cfg: dict, projector: nn.Module):
        super().__init__()
        if projector is None:
            raise ValueError("projector must be provided and implement .backproject().")
        self.projector = projector

        m = cfg.get("model", cfg)
        dbg = cfg.get("debug", {})

        # Sinogram-domain encoders
        self.enc1 = Enc1_1D_Angle(
            base=int(m.get("enc1", {}).get("base", 32)),
            depth=int(m.get("enc1", {}).get("depth", 3)),
        )
        self.enc2 = Enc2_2D_Sino(
            base=int(m.get("enc2", {}).get("base", 32)),
            depth=int(m.get("enc2", {}).get("depth", 3)),
        )

        # (X,A) → (X,Y) alignment
        align_cfg = m.get("align", {})
        self.align = Sino2XYAlign(
            in_ch=self.enc1.out_ch + self.enc2.out_ch,
            out_ch=int(align_cfg.get("out_ch", 64)),
            depth=int(align_cfg.get("depth", 2)),
            mode=str(align_cfg.get("interp_mode", "bilinear")),
        )

        # Optional 'cheat' encoder (inject GT voxel features during training)
        c_cfg = m.get("cheat2d", {})
        self.cheat_enabled = bool(c_cfg.get("enabled", True))
        self.cheat = VoxelCheat2D(
            base=int(c_cfg.get("base", 16)),
            depth=int(c_cfg.get("depth", 2)),
        )

        # XY fusion
        self.fusion = Fusion2D(
            in_ch_sino=int(align_cfg.get("out_ch", 64)),
            in_ch_cheat=(self.cheat.out_ch if self.cheat_enabled else 0),
            out_ch=int(m.get("fusion", {}).get("out_ch", 64)),
        )

        # Sino decoder (XY → XA)
        sd_cfg = m.get("sino_dec", {})
        self.sino_dec = SinoDecoder2D(
            in_ch=self.fusion.out_ch,
            mid_ch=int(sd_cfg.get("mid_ch", 64)),
            depth=int(sd_cfg.get("depth", 2)),
            mode=str(sd_cfg.get("interp_mode", align_cfg.get("interp_mode", "bilinear"))),
            bound=str(sd_cfg.get("bound", "none")).lower(),      # {"none","clamp","sigmoid"}
            antialias=sd_cfg.get("antialias", None),             # None → auto when downsampling
            prefer_gn=int(sd_cfg.get("prefer_gn", 8)),
            az_sanity_check=bool(dbg.get("az_sanity_check", False)),
        )

        # Optional PSF on sino(x,a,z)
        psf_cfg = cfg.get("psf", {})
        self.psf = SeparableGaussianPSF2D(
            enabled=bool(psf_cfg.get("enabled", False)),
            angle_variant=bool(psf_cfg.get("angle_variant", False)),
            sigma_u=float(psf_cfg.get("sigma_u", 0.7)),
            sigma_v=float(psf_cfg.get("sigma_v", 0.7)),
        )

        # Output / checks
        self.clamp_recon: bool = bool(m.get("clamp_recon", False))
        self.strict_geometry: bool = bool(m.get("strict_geometry", dbg.get("strict_geometry", False)))

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _to_b1xaz(sino: torch.Tensor) -> torch.Tensor:
        """
        Canonicalize sinogram to [B,1,X,A,Z] without permuting physics axes.

        Accepts:
          • [B,1,X,A,Z] → returned as-is.
          • [B,X,A,Z]   → add channel dim.
        """
        if sino.dim() == 5:
            if sino.shape[1] != 1:
                raise ValueError(f"sino channel must be 1: got {tuple(sino.shape)}")
            return sino
        if sino.dim() == 4:
            return sino.unsqueeze(1)
        raise ValueError(f"sino_xaz must be [B,1,X,A,Z] or [B,X,A,Z], got {tuple(sino.shape)}")

    @staticmethod
    def _to_b1xyz(vol: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Canonicalize volume to [B,1,X,Y,Z] (or None). No axis permutation is performed.
        """
        if vol is None:
            return None
        if vol.dim() == 5:
            if vol.shape[1] != 1:
                raise ValueError(f"volume channel must be 1: got {tuple(vol.shape)}")
            return vol
        if vol.dim() == 4:
            return vol.unsqueeze(1)
        raise ValueError(f"v_vol must be [B,1,X,Y,Z] or [B,X,Y,Z], got {tuple(vol.shape)}")

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(
        self,
        sino_xaz: torch.Tensor,
        v_vol: Optional[torch.Tensor] = None,
        train_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HDN forward pass.

        Parameters
        ----------
        sino_xaz : Tensor
            Input sinogram [B,1,X,A,Z] (or [B,X,A,Z]).
        v_vol : Optional[Tensor]
            GT volume [B,1,X,Y,Z] (or [B,X,Y,Z]) used only when `train_mode=True`
            and `cheat_enabled=True` to inject XY features.
        train_mode : bool
            Enable 'cheat' feature injection when True.

        Returns
        -------
        (sino_hat_xaz, recon_xyz) : Tuple[Tensor, Tensor]
            Predicted sinogram [B,1,X,A,Z] and reconstructed volume [B,1,X,Y,Z].
        """
        # Canonicalize inputs
        S = self._to_b1xaz(sino_xaz)                               # [B,1,X,A,Z]
        V = self._to_b1xyz(v_vol) if self.cheat_enabled else None  # [B,1,X,Y,Z] or None
        B, _, X, A, Z = S.shape

        # Geometry: derive Y target and optionally enforce strict checks
        geom = getattr(self.projector, "geom", None)
        if geom is not None:
            # A must match geometry to keep encoders/decoder and BP consistent.
            if hasattr(geom, "A") and (A != geom.A):
                raise ValueError(f"A mismatch: input A={A} vs geom.A={getattr(geom,'A','?')}")
            Y_target = int(getattr(geom, "H", (V.shape[3] if V is not None else X)))

            if self.strict_geometry:
                candidates = []
                if hasattr(geom, "U"):
                    candidates.append(int(geom.U))
                if hasattr(geom, "W"):
                    candidates.append(int(geom.W))
                if len(candidates) > 0 and all(X != c for c in candidates):
                    raise ValueError(
                        f"X mismatch under strict_geometry: input X={X} vs geom.U/W candidates={candidates}"
                    )
        else:
            Y_target = (V.shape[3] if V is not None else X)

        # 1) Sinogram encoders (x,a,z)
        f1 = self.enc1(S)                         # [B,C1,X,A,Z]
        f2 = self.enc2(S)                         # [B,C2,X,A,Z]
        f_sino = torch.cat([f1, f2], dim=1)       # [B,C1+C2,X,A,Z]

        # 2) (X,A) → (X,Y) alignment
        f_xy = self.align(f_sino, (X, Y_target))  # [B,Ca,X,Y,Z]

        # 3) Optional cheat + fusion in XY domain
        if self.cheat_enabled and train_mode and (V is not None):
            cheat_xy = self.cheat(V)              # [B,Cc,X,Y,Z]
            fused_xy = self.fusion(f_xy, cheat_xy)  # [B,Cf,X,Y,Z]
        else:
            fused_xy = self.fusion(f_xy, None)      # [B,Cf,X,Y,Z]

        # 4) Sino decoder (XY → XA): predicted sinogram
        sino_hat_xaz = self.sino_dec(fused_xy, A=A)  # [B,1,X,A,Z]

        # 5) Optional PSF before backprojection
        sino_for_bp = sino_hat_xaz
        if self.psf.enabled:
            if getattr(self.psf, "angle_variant", False):
                need_cfg = (not hasattr(self.psf, "_A")) or (getattr(self.psf, "_A", None) != A)
                if need_cfg and hasattr(self.psf, "configure"):
                    self.psf.configure(A, device=sino_for_bp.device, dtype=sino_for_bp.dtype)
            sino_for_bp = self.psf(sino_for_bp)  # [B,1,X,A,Z]

        # 6) Backprojection to (x,y,z) — unfiltered BP with circular mask and π/(2·A) scaling.
        recon_xyz = self.projector.backproject(sino_for_bp)  # [B,1,X,Y,Z]  :contentReference[oaicite:10]{index=10}

        if self.clamp_recon:
            recon_xyz = recon_xyz.clamp(0.0, 1.0)
        return sino_hat_xaz, recon_xyz
