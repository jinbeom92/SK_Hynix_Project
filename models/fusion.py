import math
import torch
import torch.nn as nn


def _gn(out_ch: int, prefer: int = 8) -> int:
    """
    Compute a suitable number of groups for GroupNorm.

    Parameters
    ----------
    out_ch : int
        Number of output channels.
    prefer : int
        Preferred divisor (default: 8).

    Returns
    -------
    int
        Group count chosen as gcd(out_ch, prefer), clamped to [1, out_ch].
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class VoxelCheat2D(nn.Module):
    """
    Voxel Cheat Encoder (2D, slice-wise along z).

    Purpose
    -------
    Encode ground-truth voxel **slices** into a feature representation used
    only during training (cheat path) to aid fusion and convergence.

    Axis convention
    ---------------
    Volumes are laid out as **(x, y, z)**:
      - Input  : ``[B, 1, X, Y, Z]`` (preferred) or ``[B, 1, X, Y]`` (Z=1)
      - Output : ``[B, base, X, Y, Z]``

    Notes
    -----
    • Processes each z-slice with a shared 2D Conv → GN → ReLU stack.
    • No axis permutation relative to (x, y, z); only (B, Z) flattening.
    """

    def __init__(self, base: int = 16, depth: int = 2):
        super().__init__()
        ch = [1] + [base] * depth
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_gn(ch[i + 1]), ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)
        self.out_ch = ch[-1]

    def forward(self, v_slice: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        v_slice : Tensor
            Ground-truth voxel slices, ``[B, 1, X, Y, Z]`` or ``[B, 1, X, Y]``.

        Returns
        -------
        Tensor
            Cheat features ``[B, out_ch, X, Y, Z]``.
        """
        if v_slice.dim() == 5:
            B, C, X, Y, Z = v_slice.shape
            if C != 1:
                raise ValueError(f"VoxelCheat2D expects channel=1, got {tuple(v_slice.shape)}")
            # [B,1,X,Y,Z] → [B,Z,1,X,Y] → [B*Z,1,X,Y]
            x = v_slice.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, 1, X, Y)
        elif v_slice.dim() == 4:
            B, C, X, Y = v_slice.shape
            if C != 1:
                raise ValueError(f"VoxelCheat2D expects channel=1, got {tuple(v_slice.shape)}")
            Z = 1
            x = v_slice  # [B,1,X,Y] == [B*1,1,X,Y]
        else:
            raise ValueError(f"Unsupported voxel shape: {tuple(v_slice.shape)}")

        x = self.net(x)  # [B*Z, base, X, Y]

        # Restore to [B, base, X, Y, Z]
        x = x.view(B, Z, self.out_ch, X, Y).permute(0, 2, 3, 4, 1).contiguous()
        return x


class Fusion2D(nn.Module):
    """
    Fuse sinogram-aligned features with optional voxel cheat features (slice-wise).

    Purpose
    -------
    Concatenate features from the sinogram branch and (optionally) the voxel
    cheat branch, then mix with Conv2D → GN → ReLU on each (x, y) slice.

    Axis convention
    ---------------
    Inputs/outputs are **(x, y, z)**:
      - ``F_xy_sino`` : ``[B, C_s, X, Y, Z]`` or ``[B, C_s, X, Y]`` (Z=1)
      - ``cheat_xy``  : ``[B, C_c, X, Y, Z]`` or ``[B, C_c, X, Y]`` (Z=1)
      - Output        : ``[B, out_ch, X, Y, Z]``

    Behavior
    --------
    • If `in_ch_cheat > 0` and `cheat_xy is None`, zeros with the correct
      shape are concatenated as the cheat branch.
    • If one input has Z=1 and the other has Z>1, the Z=1 tensor is
      broadcast along Z for convenience.

    Notes
    -----
    • This module does not permute axes; it only flattens (B, Z) for efficient
      batched 2D processing, then restores the (x, y, z) layout.
    """

    def __init__(self, in_ch_sino: int, in_ch_cheat: int, out_ch: int = 64):
        super().__init__()
        self.in_ch_sino = int(in_ch_sino)
        self.in_ch_cheat = int(in_ch_cheat)
        in_ch = self.in_ch_sino + self.in_ch_cheat

        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_gn(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def _canon_5d(self, t: torch.Tensor, expect_c: int, name: str):
        """Canonicalize to [B,C,X,Y,Z] and also return (B,X,Y,Z)."""
        if t.dim() == 5:
            B, C, X, Y, Z = t.shape
            if expect_c is not None and C != expect_c:
                raise ValueError(f"{name} channels={C} != expected {expect_c}")
            return t, (B, C, X, Y, Z)
        elif t.dim() == 4:
            B, C, X, Y = t.shape
            if expect_c is not None and C != expect_c:
                raise ValueError(f"{name} channels={C} != expected {expect_c}")
            return t.unsqueeze(-1), (B, C, X, Y, 1)
        else:
            raise ValueError(f"{name} must be [B,C,X,Y,(Z)], got {tuple(t.shape)}")

    def forward(self, F_xy_sino: torch.Tensor, cheat_xy: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        F_xy_sino : Tensor
            Sinogram-aligned features, ``[B, C_s, X, Y, Z]`` or ``[B, C_s, X, Y]``.
        cheat_xy : Tensor, optional
            Cheat features, ``[B, C_c, X, Y, Z]`` or ``[B, C_c, X, Y]``.
            If None and `in_ch_cheat > 0`, zeros are injected.

        Returns
        -------
        Tensor
            Fused representation, ``[B, out_ch, X, Y, Z]``.
        """
        # Canonicalize inputs to [B,C,X,Y,Z]
        F5, (B, Cs, X, Y, Z) = self._canon_5d(F_xy_sino, self.in_ch_sino, "F_xy_sino")

        if self.in_ch_cheat > 0:
            if cheat_xy is None:
                Cc = self.in_ch_cheat
                cheat5 = F5.new_zeros(B, Cc, X, Y, Z)
            else:
                cheat5, (B2, Cc, X2, Y2, Z2) = self._canon_5d(cheat_xy, self.in_ch_cheat, "cheat_xy")
                if (B2, X2, Y2) != (B, X, Y):
                    raise ValueError(f"cheat_xy spatial/batch mismatch: {(B2,X2,Y2)} vs {(B,X,Y)}")
                # Broadcast Z if needed
                if Z2 != Z:
                    if Z2 == 1:
                        cheat5 = cheat5.expand(B, Cc, X, Y, Z)
                    elif Z == 1:
                        F5 = F5.expand(B, Cs, X, Y, Z2)
                        Z = Z2
                    else:
                        raise ValueError(f"Z mismatch: F_xy_sino Z={Z}, cheat Z={Z2}")
            x_cat = torch.cat([F5, cheat5], dim=1)  # [B, Cs+Cc, X, Y, Z]
        else:
            x_cat = F5  # [B, Cs, X, Y, Z]

        # Flatten (B, Z) → batch for 2D fusion
        # [B,C,X,Y,Z] → [B,Z,C,X,Y] → [B*Z,C,X,Y]
        x2d = x_cat.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, x_cat.shape[1], X, Y)

        fused = self.mix(x2d)  # [B*Z, out_ch, X, Y]

        # Restore [B,out_ch,X,Y,Z]
        out = fused.view(B, Z, self.out_ch, X, Y).permute(0, 2, 3, 4, 1).contiguous()
        return out
