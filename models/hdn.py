import torch
import torch.nn as nn
from .enc1_1d import Enc1_1D_Angle
from .enc2_2d import Enc2_2D_Sino
from .align import Sino2XYAlign
from .fusion import VoxelCheat2D, Fusion2D
from .decoder import DecoderSlice2D


class HDNSystem(nn.Module):
    """
    High-Dimensional Neural (HDN) System for slice-wise tomographic reconstruction.

    Purpose
    -------
    • Consumes a sinogram [B,1,U,A] (detectors × angles).
    • Extracts features via 1D (angle-axis) and 2D (sino-plane) encoders.
    • Aligns features into voxel-plane coordinates (X,Y).
    • Optionally encodes ground-truth voxel slices (cheat path) during training.
    • Fuses sinogram and cheat features, then decodes into a reconstructed slice.

    Args
    ----
    cfg (dict): configuration dictionary (expects "model" sub-dict with keys):
        enc1   : {"base": int, "depth": int} for 1D encoder
        enc2   : {"base": int, "depth": int} for 2D encoder
        align  : {"out_ch": int, "depth": int, "interp_mode": str}
        cheat2d: {"enabled": bool, "base": int, "depth": int}
        fusion : {"out_ch": int}
        dec    : {"mid_ch": int, "depth": int}

    Shapes
    ------
    Input:
        sino_ua : [B,1,U,A] or [B,U,A]
        v_slice : [B,1,X,Y] (optional GT voxel slice for cheat injection)

    Output:
        R_hat   : [B,1,X,Y] reconstructed voxel slice

    Notes
    -----
    • If v_slice is not provided, alignment falls back to a square size (X=Y=U).
    • Cheat path is active only when:
        - cheat_enabled=True,
        - train_mode=True,
        - v_slice is provided.
    • During inference, only the sinogram path is used.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        # ----------------------------------------------------------------------
        # Encoders: 1D (angle-wise) and 2D (sinogram plane)
        # ----------------------------------------------------------------------
        self.enc1 = Enc1_1D_Angle(
            base=int(m["enc1"]["base"]),
            depth=int(m["enc1"]["depth"])
        )
        self.enc2 = Enc2_2D_Sino(
            base=int(m["enc2"]["base"]),
            depth=int(m["enc2"]["depth"])
        )

        # ----------------------------------------------------------------------
        # Alignment: map (U,A) features → (X,Y) voxel-plane features
        # ----------------------------------------------------------------------
        align_cfg = m.get("align", {})
        self.align = Sino2XYAlign(
            in_ch=self.enc1.out_ch + self.enc2.out_ch,
            out_ch=int(align_cfg.get("out_ch", 64)),
            depth=int(align_cfg.get("depth", 2)),
            mode=str(align_cfg.get("interp_mode", "bilinear")),
        )

        # ----------------------------------------------------------------------
        # Cheat encoder (optional, training only)
        # ----------------------------------------------------------------------
        self.cheat_enabled = bool(m.get("cheat2d", {}).get("enabled", True))
        self.cheat = VoxelCheat2D(
            base=int(m.get("cheat2d", {}).get("base", 16)),
            depth=int(m.get("cheat2d", {}).get("depth", 2)),
        )

        # ----------------------------------------------------------------------
        # Fusion: combine aligned sino features + (optional) cheat features
        # ----------------------------------------------------------------------
        self.fusion = Fusion2D(
            in_ch_sino=int(align_cfg.get("out_ch", 64)),
            in_ch_cheat=(self.cheat.out_ch if self.cheat_enabled else 0),
            out_ch=int(m["fusion"]["out_ch"]),
        )

        # ----------------------------------------------------------------------
        # Decoder: reconstruct slice from fused features
        # ----------------------------------------------------------------------
        self.dec = DecoderSlice2D(
            in_ch=self.fusion.out_ch,
            mid_ch=int(m["dec"]["mid_ch"]),
            depth=int(m["dec"].get("depth", 3)),
        )

    def forward(self, sino_ua, v_slice=None, train_mode=True):
        """
        Forward pass.

        Args:
            sino_ua (Tensor): [B,1,U,A] or [B,U,A] sinogram
            v_slice (Tensor, optional): [B,1,X,Y] voxel slice (GT), used only if cheat is enabled
            train_mode (bool): if True, allow cheat injection (default True)

        Returns:
            Tensor: [B,1,X,Y] reconstructed voxel slice
        """
        if sino_ua.ndim == 3:   # [B,U,A] → add channel
            sino_ua = sino_ua.unsqueeze(1)  # [B,1,U,A]
        B, C, U, A = sino_ua.shape

        # 1. Encode sinogram along angle axis (1D) and as 2D plane
        f1 = self.enc1(sino_ua)         # [B,C1,U,A]
        f2 = self.enc2(sino_ua)         # [B,C2,U,A]
        f  = torch.cat([f1, f2], dim=1) # [B,C1+C2,U,A]

        # 2. Align features to voxel plane
        if v_slice is not None:
            X, Y = int(v_slice.shape[-2]), int(v_slice.shape[-1])
        else:
            X = Y = U  # fallback to square if GT slice is missing
        f_xy = self.align(f, (X, Y))    # [B,Ca,X,Y]

        # 3. Fuse with cheat features (zeros padded internally if cheat=None)
        if self.cheat_enabled and train_mode and (v_slice is not None):
            cheat_xy = self.cheat(v_slice)      # [B,Cc,X,Y]
            fused = self.fusion(f_xy, cheat_xy) # [B,Cf,X,Y]
        else:
            fused = self.fusion(f_xy, None)     # [B,Cf,X,Y]

        # 4. Decode into reconstructed voxel slice
        R_hat = self.dec(fused)                 # [B,1,X,Y]
        return R_hat
