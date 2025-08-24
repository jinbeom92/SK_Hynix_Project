import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(out_ch: int, prefer: int = 8) -> int:
    """
    Compute a suitable GroupNorm divisor.

    Args:
        out_ch (int): number of channels
        prefer (int): preferred group count (default 8)

    Returns:
        int: number of groups for GroupNorm such that
             • divides out_ch evenly (via gcd)
             • clamped between 1 and out_ch
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class Sino2XYAlign(nn.Module):
    """
    Align sinogram features **(x,a,z)** to voxel-plane features **(x,y,z)**.

    Purpose
    -------
    • Consumes concatenated sinogram features from Enc1/Enc2 laid out as (x,a,z).
    • Applies Conv2D+GroupNorm+ReLU on the (X,A) plane for each z-slice.
    • Deterministically resizes (X,A) → (X,Y) to match voxel plane size.
    • Outputs per-slice aligned features with channel width `out_ch`.

    Args
    ----
    in_ch : int
        Number of input feature channels (after fusion).
    out_ch : int, optional
        Channel width for each conv block (default: 64).
    depth : int, optional
        Number of conv blocks before projection (default: 2).
    mode : str, optional
        Interpolation mode for resize: {"bilinear","bicubic","nearest"}.

    Shapes
    ------
    Input:
        F_sino : [B, in_ch, X, A, Z]   or  [B, in_ch, X, A] (treated as Z=1)
        out_hw : (X, Y)   target voxel-plane size
    Output:
        Tensor  : [B, out_ch, X, Y, Z]

    Notes
    -----
    • This is a deterministic alignment (resize) on (X,A) — **not** a BP operator.
    • Implementation flattens (B,Z) → batch to process all slices efficiently.
    """

    def __init__(self, in_ch: int, out_ch: int = 64, depth: int = 2, mode: str = "bilinear"):
        super().__init__()
        self.mode = str(mode)
        ch = [in_ch] + [out_ch] * depth
        blocks = []
        for i in range(depth):
            blocks += [
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_gn(ch[i + 1]), ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.stem = nn.Sequential(*blocks)
        self.out_proj = nn.Identity()  # placeholder if you want a final 1x1 later

    def forward(self, F_sino: torch.Tensor, out_hw):
        """
        Forward pass.

        Args
        ----
        F_sino : Tensor
            Sinogram feature map in (x,a,z) layout:
            - [B, in_ch, X, A, Z]  (preferred), or
            - [B, in_ch, X, A]     (treated as Z=1).
        out_hw : tuple
            (X, Y) target spatial size on the voxel plane.

        Returns
        -------
        Tensor
            Aligned voxel-plane feature map with shape [B, out_ch, X, Y, Z].
        """
        if F_sino.dim() == 5:
            B, C, X, A, Z = F_sino.shape
            # [B,C,X,A,Z] → [B,Z,C,X,A] → [B*Z,C,X,A]
            x2d = F_sino.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, A)
            BZ = B * Z
        elif F_sino.dim() == 4:
            B, C, X, A = F_sino.shape
            Z = 1
            x2d = F_sino.view(B, C, X, A)   # [B,C,X,A]
            BZ = B
        else:
            raise ValueError(f"Sino2XYAlign expects [B,C,X,A,Z] or [B,C,X,A], got {tuple(F_sino.shape)}")

        # Local mixing on (X,A)
        x2d = self.stem(x2d)  # [BZ, out_ch, X, A]

        # Deterministic-friendly resize to (X,Y)
        X_tgt, Y_tgt = int(out_hw[0]), int(out_hw[1])
        if self.mode in ("bilinear", "bicubic"):
            x2d = F.interpolate(x2d, size=(X_tgt, Y_tgt), mode=self.mode, align_corners=False)
        else:
            x2d = F.interpolate(x2d, size=(X_tgt, Y_tgt), mode=self.mode)

        # Restore [B,out_ch,X,Y,Z]
        x = x2d.view(B, Z, x2d.shape[1], X_tgt, Y_tgt).permute(0, 2, 3, 4, 1).contiguous()
        return self.out_proj(x)
