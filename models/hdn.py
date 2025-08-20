# =================================================================================================
# HDNSystem — Physics-Free per-Depth Slice Reconstructor
# =================================================================================================
import torch
import torch.nn as nn
from .enc1_1d import Enc1_1D_Angle
from .enc2_2d import Enc2_2D_Sino
from .align import Sino2XYAlign
from .fusion import VoxelCheat2D, Fusion2D
from .decoder import DecoderSlice2D

class HDNSystem(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        # Encoders
        self.enc1 = Enc1_1D_Angle(base=int(m["enc1"]["base"]), depth=int(m["enc1"]["depth"]))
        self.enc2 = Enc2_2D_Sino(base=int(m["enc2"]["base"]), depth=int(m["enc2"]["depth"]))

        # Align (U,A) → (X,Y)
        align_cfg = m.get("align", {})
        self.align = Sino2XYAlign(
            in_ch=self.enc1.out_ch + self.enc2.out_ch,
            out_ch=int(align_cfg.get("out_ch", 64)),
            depth=int(align_cfg.get("depth", 2)),
            mode=str(align_cfg.get("interp_mode", "bilinear")),
        )

        # Cheat + Fusion
        self.cheat_enabled = bool(m.get("cheat2d", {}).get("enabled", True))
        self.cheat = VoxelCheat2D(
            base=int(m.get("cheat2d", {}).get("base", 16)),
            depth=int(m.get("cheat2d", {}).get("depth", 2)),
        )
        self.fusion = Fusion2D(
            in_ch_sino=int(align_cfg.get("out_ch", 64)),
            in_ch_cheat=(self.cheat.out_ch if self.cheat_enabled else 0),
            out_ch=int(m["fusion"]["out_ch"]),
        )

        # Decoder
        self.dec = DecoderSlice2D(
            in_ch=self.fusion.out_ch,
            mid_ch=int(m["dec"]["mid_ch"]),
            depth=int(m["dec"].get("depth", 3)),
        )

    def forward(self, sino_ua, v_slice=None, train_mode=True):
        if sino_ua.ndim == 3:   # [B,U,A]
            sino_ua = sino_ua.unsqueeze(1)  # [B,1,U,A]
        B, C, U, A = sino_ua.shape

        # Encoders on (U,A)
        f1 = self.enc1(sino_ua)         # [B,C1,U,A]
        f2 = self.enc2(sino_ua)         # [B,C2,U,A]
        f  = torch.cat([f1, f2], dim=1) # [B,C1+C2,U,A]

        # Align to (X,Y)
        if v_slice is not None:
            X, Y = int(v_slice.shape[-2]), int(v_slice.shape[-1])
        else:
            X = Y = U  # fallback
        f_xy = self.align(f, (X, Y))    # [B,Ca,X,Y]

        # Cheat fusion (zeros padded internally when cheat is None)
        if self.cheat_enabled and train_mode and (v_slice is not None):
            cheat_xy = self.cheat(v_slice)      # [B,Cc,X,Y]
            fused = self.fusion(f_xy, cheat_xy) # [B,Cf,X,Y]
        else:
            fused = self.fusion(f_xy, None)     # [B,Cf,X,Y]

        # Decode slice
        R_hat = self.dec(fused)                 # [B,1,X,Y]
        return R_hat
