import math
import torch.nn as nn


def _gn(out_ch: int, prefer: int = 8) -> int:
    """
    Compute a suitable number of groups for GroupNorm.

    Args:
        out_ch (int): output channel count
        prefer (int): preferred divisor (default 8)

    Returns:
        int: number of groups (divides out_ch, clamped ≥1)
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class Conv2DReLU(nn.Module):
    """
    Basic 2D Conv → GroupNorm → ReLU block.

    Args:
        in_ch (int): input channel count
        out_ch (int): output channel count
        k (int): kernel size (default 3)
        s (int): stride (default 1)
        p (int): padding (default 1)

    Notes:
        • Bias is disabled in Conv2D since GroupNorm follows.
        • ReLU is applied in-place.
    """

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_gn(out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, C_in, H, W]

        Returns:
            Tensor: [B, C_out, H, W]
        """
        return self.act(self.gn(self.conv(x)))


class Enc2_2D_Sino(nn.Module):
    """
    2D Sinogram Encoder.

    Purpose
    -------
    • Extracts **joint spatial–angular features** directly from sinogram slices.
    • Complements Enc1_1D_Angle by capturing local 2D correlations in the (U,A) plane.
    • Used as the second encoder branch in SVTR/HDN pipelines.

    Args
    ----
    base (int): channel width for each conv block (default 32)
    depth (int): number of Conv2DReLU blocks (default 3)

    Shapes
    ------
    Input:
        S_ua : [B, 1, U, A]
          • U = detector bins
          • A = projection angles
    Output:
        F2 : [B, C2, U, A]
          • C2 = base (final channel count)

    Notes
    -----
    • This encoder treats the sinogram as a 2D image.
    • Each Conv2D block increases representational power while maintaining resolution.
    """

    def __init__(self, base: int = 32, depth: int = 3):
        super().__init__()
        ch = [1] + [base] * depth
        self.blocks = nn.ModuleList([Conv2DReLU(ch[i], ch[i+1]) for i in range(depth)])
        self.out_ch = ch[-1]

    def forward(self, S_ua):
        """
        Forward pass.

        Args:
            S_ua (Tensor): [B,1,U,A] sinogram

        Returns:
            Tensor: [B,C2,U,A] encoded feature map
        """
        x = S_ua
        for blk in self.blocks:
            x = blk(x)
        return x  # [B,C2,U,A]
