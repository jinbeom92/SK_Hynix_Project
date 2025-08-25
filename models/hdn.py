from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .enc1_1d import Enc1_1D_Angle                 # [B,1,X,A,Z] → [B,C1,X,A,Z]
from .enc2_2d import Enc2_2D_Sino                  # [B,1,X,A,Z] → [B,C2,X,A,Z]
from .align   import Sino2XYAlign                  # [B,C,X,A,Z] → [B,Ca,X,Y,Z]
from .fusion  import VoxelCheat2D, Fusion2D        # [B,*,X,Y,Z] fuse
from physics.psf import SeparableGaussianPSF2D     # optional PSF over sino(x,a,z)


# ------------------------------------------------------------------------------------------
# Small helper: XY → XA (per z-slice) with Conv stack + resize
# ------------------------------------------------------------------------------------------
class SinoDecoder2D(nn.Module):
    """
    Decode fused **(x, y, z)** features into a **(x, a, z)** sinogram.

    Parameters
    ----------
    in_ch : int
        Input channel width (from fusion output).
    mid_ch : int, optional
        Hidden channel width for conv mixing (default: 64).
    depth : int, optional
        Number of Conv2D→GN→ReLU blocks before projection (default: 2).
    mode : str, optional
        Interp mode for Y→A resizing: {"bilinear","bicubic","nearest"} (default: "bilinear").

    Shapes
    ------
    Input : F_xy   → [B, in_ch, X, Y, Z]
    Arg   : A      → int (number of angles, target size for 'A' axis)
    Output: sino   → [B, 1,     X, A, Z]
    """

    def __init__(self, in_ch: int, mid_ch: int = 64, depth: int = 2, mode: str = "bilinear"):
        super().__init__()
        self.mode = str(mode)
        ch = [in_ch] + [mid_ch] * depth
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=max(1, min(ch[i + 1], 8)), num_channels=ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        # final 1×1 to 1-channel sinogram
        self.mix = nn.Sequential(*layers)
        self.proj = nn.Conv2d(ch[-1], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, F_xy: torch.Tensor, A: int) -> torch.Tensor:
        """
        Decode fused XY-features into a sinogram with A angles.

        Parameters
        ----------
        F_xy : Tensor
            [B, in_ch, X, Y, Z] fused features.
        A : int
            Number of projection angles (target A-size).

        Returns
        -------
        Tensor
            [B, 1, X, A, Z] sinogram in (x, a, z) layout, clamped to [0, 1].
        """
        if F_xy.dim() == 5:
            B, C, X, Y, Z = F_xy.shape
            x = F_xy.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, Y)
        elif F_xy.dim() == 4:  # Z=1
            B, C, X, Y = F_xy.shape
            Z = 1
            x = F_xy.view(B * Z, C, X, Y)
        else:
            raise ValueError(f"SinoDecoder2D expects [B,C,X,Y,(Z)], got {tuple(F_xy.shape)}")

        # Conv mixing (resolution preserved)
        x = self.mix(x)  # [B*Z, mid, X, Y]

        # Y → A resize
        if self.mode in ("bilinear", "bicubic"):
            x = F.interpolate(x, size=(X, A), mode=self.mode, align_corners=False)
        else:
            x = F.interpolate(x, size=(X, A), mode=self.mode)

        # 1×1 projection to sinogram
        x = self.proj(x)  # [B*Z, 1, X, A]

        # [B*Z,1,X,A] → [B,1,X,A,Z]
        sino = x.view(B, Z, 1, X, A).permute(0, 2, 3, 4, 1).contiguous()
        return sino.clamp_(0.0, 1.0)


class HDNSystem(nn.Module):
    """
    High-Dimensional Neural (HDN) System producing a **sino(x,a,z)** and its BP **recon(x,y,z)**.

    Pipeline
    --------
    1) Enc1_1D_Angle: A-axis 1D encoding → [B,C1,X,A,Z]
    2) Enc2_2D_Sino : (X,A) 2D encoding → [B,C2,X,A,Z]
    3) Sino2XYAlign : (X,A) features → (X,Y) alignment → [B,Ca,X,Y,Z]
    4) (optional) VoxelCheat2D: GT voxel → [B,Cc,X,Y,Z]
    5) Fusion2D      : fuse aligned+cheat → [B,Cf,X,Y,Z]
    6) SinoDecoder2D : (X,Y) → **sino [B,1,X,A,Z]**
    7) Projector.BP  : **backproject** → **recon [B,1,X,Y,Z]**

    Notes
    -----
    • Cheat path is active only when `train_mode=True` and GT volume is provided.
    • Target Y is taken from `projector.geom.H` to keep alignment consistent.
    """

    def __init__(self, cfg: dict, projector: nn.Module):
        super().__init__()
        if projector is None:
            raise ValueError("projector must be provided and implement .backproject().")
        self.projector = projector

        m = cfg.get("model", cfg)

        # 1D/2D encoders (sinogram domain)
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

        # Cheat encoder (optional)
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

        # Sinogram decoder (XY → XA)
        sd_cfg = m.get("sino_dec", {})
        self.sino_dec = SinoDecoder2D(
            in_ch=self.fusion.out_ch,
            mid_ch=int(sd_cfg.get("mid_ch", 64)),
            depth=int(sd_cfg.get("depth", 2)),
            mode=str(sd_cfg.get("interp_mode", align_cfg.get("interp_mode", "bilinear"))),
        )

        # Optional PSF over sino(x,a,z)
        psf_cfg = cfg.get("psf", {})
        self.psf = SeparableGaussianPSF2D(
            enabled=bool(psf_cfg.get("enabled", False)),
            angle_variant=bool(psf_cfg.get("angle_variant", False)),
            sigma_u=float(psf_cfg.get("sigma_u", 0.7)),
            sigma_v=float(psf_cfg.get("sigma_v", 0.7)),
        )

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _to_b1xaz(sino: torch.Tensor) -> torch.Tensor:
        """Ensure sinogram is [B,1,X,A,Z]."""
        if sino.dim() == 5:
            if sino.shape[1] != 1:
                raise ValueError(f"sino channel must be 1: got {tuple(sino.shape)}")
            return sino
        if sino.dim() == 4:
            return sino.unsqueeze(1)
        raise ValueError(f"sino_xaz must be [B,1,X,A,Z] or [B,X,A,Z], got {tuple(sino.shape)}")

    @staticmethod
    def _to_b1xyz(vol: torch.Tensor) -> torch.Tensor:
        """Ensure volume is [B,1,X,Y,Z] (optional input for cheat)."""
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
        Forward pass.

        Parameters
        ----------
        sino_xaz : Tensor
            Input sinogram **[B,1,X,A,Z]** or **[B,X,A,Z]**.
        v_vol : Tensor, optional
            Ground-truth voxel volume **[B,1,X,Y,Z]** or **[B,X,Y,Z]** (cheat path).
        train_mode : bool
            If True, enables cheat injection (default: True).

        Returns
        -------
        (sino_hat_xaz, recon_xyz) : Tuple[Tensor, Tensor]
            - sino_hat_xaz : **[B,1,X,A,Z]** decoder-predicted sinogram.
            - recon_xyz    : **[B,1,X,Y,Z]** backprojected reconstruction.
        """
        # Canonicalize inputs
        S = self._to_b1xaz(sino_xaz)             # [B,1,X,A,Z]
        V = self._to_b1xyz(v_vol) if self.cheat_enabled else None  # [B,1,X,Y,Z] or None

        B, _, X, A, Z = S.shape

        # Y target from geometry (if available)
        geom = getattr(self.projector, "geom", None)
        if geom is not None:
            if A != geom.A:
                raise ValueError(f"A mismatch: input A={A} vs geom.A={geom.A}")
            Y_target = geom.H
        else:
            Y_target = (V.shape[3] if V is not None else X)

        # 1) Sinogram encoders (x,a,z)
        f1 = self.enc1(S)                         # [B,C1,X,A,Z]
        f2 = self.enc2(S)                         # [B,C2,X,A,Z]
        f_sino = torch.cat([f1, f2], dim=1)       # [B,C1+C2,X,A,Z]

        # 2) (X,A) → (X,Y) alignment
        f_xy = self.align(f_sino, (X, Y_target))  # [B,Ca,X,Y,Z]

        # 3) Cheat features (optional) + fusion (XY domain)
        if self.cheat_enabled and train_mode and (V is not None):
            cheat_xy = self.cheat(V)              # [B,Cc,X,Y,Z]
            fused_xy = self.fusion(f_xy, cheat_xy)  # [B,Cf,X,Y,Z]
        else:
            fused_xy = self.fusion(f_xy, None)      # [B,Cf,X,Y,Z]

        # 4) Sinogram decoder (XY → XA): predicted sinogram (x,a,z)
        sino_hat_xaz = self.sino_dec(fused_xy, A=A)  # [B,1,X,A,Z]

        # 5) Optional PSF before backprojection
        sino_for_bp = sino_hat_xaz
        if self.psf.enabled:
            if self.psf.angle_variant and self.psf._A != A:
                self.psf.configure(A, device=sino_for_bp.device, dtype=sino_for_bp.dtype)
            sino_for_bp = self.psf(sino_for_bp)  # [B,1,X,A,Z]

        # 6) Backprojection to reconstruction (x,y,z)
        recon_xyz = self.projector.backproject(sino_for_bp)  # [B,1,X,Y,Z]
        recon_xyz = recon_xyz.clamp_(0.0, 1.0)

        return sino_hat_xaz, recon_xyz
