import math
import torch
import torch.nn as nn


def _gn(out_ch: int, prefer: int = 8) -> int:
    """
    Compute a suitable number of groups for GroupNorm.

    Args:
        out_ch (int): number of output channels
        prefer (int): preferred divisor (default 8)

    Returns:
        int: group count, chosen as gcd(out_ch, prefer),
             clamped to [1, out_ch].
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class VoxelCheat2D(nn.Module):
    """
    Voxel Cheat Encoder (2D).

    Purpose
    -------
    • Encodes ground-truth voxel slices into a feature representation.
    • Used only during training as a **cheat path** to inject GT information
      into the fusion stage, improving convergence.

    Args
    ----
    base (int): base channel width (default 16)
    depth (int): number of Conv2D→GN→ReLU layers (default 2)

    Architecture
    ------------
    Input : v_slice [B, 1, X, Y] (single-channel voxel slice)
    Output: feature [B, base, X, Y] (by default)

    Notes
    -----
    • All convs are 3×3, stride=1, padding=1, bias=False.
    • GroupNorm is applied with groups chosen by _gn().
    • ReLU is applied in-place.
    """

    def __init__(self, base: int = 16, depth: int = 2):
        super().__init__()
        ch = [1] + [base] * depth
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i+1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_gn(ch[i+1]), ch[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)
        self.out_ch = ch[-1]

    def forward(self, v_slice: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_slice (Tensor): [B,1,X,Y] voxel slice

        Returns:
            Tensor: [B, out_ch, X, Y] cheat features
        """
        return self.net(v_slice)


class Fusion2D(nn.Module):
    """
    Fusion block combining sinogram-aligned features and cheat features.

    Purpose
    -------
    • Concatenates features from the sinogram branch and optional
      voxel cheat branch.
    • Applies a Conv2D + GN + ReLU mix to produce fused representation.

    Args
    ----
    in_ch_sino (int): number of channels from sino-aligned features
    in_ch_cheat (int): number of channels from cheat features (0 disables cheat path)
    out_ch (int): output channel width after fusion (default 64)

    Architecture
    ------------
    Input : F_xy_sino [B, in_ch_sino, X, Y]
            cheat_xy   [B, in_ch_cheat, X, Y] (optional)
    Output: fused     [B, out_ch, X, Y]

    Behavior
    --------
    • If cheat channels are expected but `cheat_xy` is None, zeros are padded.
    • If in_ch_cheat=0, the module just processes F_xy_sino.

    Notes
    -----
    • This design allows toggling cheat injection via config.
    • Inference typically sets in_ch_cheat=0 (no GT available).
    """

    def __init__(self, in_ch_sino: int, in_ch_cheat: int, out_ch: int = 64):
        super().__init__()
        self.in_ch_sino  = int(in_ch_sino)
        self.in_ch_cheat = int(in_ch_cheat)
        in_ch = self.in_ch_sino + self.in_ch_cheat

        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_gn(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def forward(self, F_xy_sino: torch.Tensor, cheat_xy: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            F_xy_sino (Tensor): [B, in_ch_sino, X, Y]
            cheat_xy (Tensor, optional): [B, in_ch_cheat, X, Y] cheat features

        Returns:
            Tensor: [B, out_ch, X, Y] fused representation
        """
        if self.in_ch_cheat > 0:
            if cheat_xy is None:
                # pad missing cheat with zeros to match expected channels
                B, _, X, Y = F_xy_sino.shape
                zeros = F_xy_sino.new_zeros(B, self.in_ch_cheat, X, Y)
                x = torch.cat([F_xy_sino, zeros], dim=1)
            else:
                x = torch.cat([F_xy_sino, cheat_xy], dim=1)
        else:
            x = F_xy_sino

        return self.mix(x)
