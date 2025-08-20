import math
import torch
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


class Conv1DReLU(nn.Module):
    """
    Basic 1D Conv → GroupNorm → ReLU block.

    Args:
        in_ch (int): input channel count
        out_ch (int): output channel count
        k (int): kernel size (default 7)
        s (int): stride (default 1)
        p (int): padding (default 3)

    Notes:
        • Bias is disabled in Conv1D since GroupNorm follows.
        • ReLU is applied in-place.
    """

    def __init__(self, in_ch, out_ch, k=7, s=1, p=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(_gn(out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, C_in, L] input sequence

        Returns:
            Tensor: [B, C_out, L] output after conv + norm + ReLU
        """
        return self.act(self.gn(self.conv(x)))


class Enc1_1D_Angle(nn.Module):
    """
    1D Angle Encoder.

    Purpose
    -------
    • Extracts features along the **angle axis** of sinograms.
    • Processes each detector row independently with a shared 1D Conv stack.
    • Used as the first encoder branch in SVTR/HDN pipelines.

    Args
    ----
    base (int): channel width for each conv block (default 32)
    depth (int): number of Conv1DReLU blocks (default 3)

    Shapes
    ------
    Input:
        S_ua : [B, 1, U, A]
          • B = batch
          • U = detector bins
          • A = projection angles
    Output:
        F1 : [B, C1, U, A]
          • C1 = base (final channel count)

    Notes
    -----
    • Input is reshaped to [B*U, 1, A] so that each detector row
      is treated as an independent 1D sequence of angles.
    • After Conv1D blocks, reshape back to [B,C1,U,A].
    """

    def __init__(self, base: int = 32, depth: int = 3):
        super().__init__()
        ch = [1] + [base] * depth
        self.blocks = nn.ModuleList([Conv1DReLU(ch[i], ch[i+1]) for i in range(depth)])
        self.out_ch = ch[-1]

    def forward(self, S_ua: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            S_ua (Tensor): [B,1,U,A] sinogram (1 channel)

        Returns:
            Tensor: [B,C1,U,A] angle-encoded features
        """
        B, C, U, A = S_ua.shape
        # Flatten detector dimension into batch
        x = S_ua.permute(0, 2, 1, 3).contiguous().view(B * U, 1, A)
        for blk in self.blocks:
            x = blk(x)
        C1 = self.out_ch
        # Reshape back to [B,C1,U,A]
        x = x.view(B, U, C1, A).permute(0, 2, 1, 3).contiguous()
        return x
