import math
import torch.nn as nn

def _gn(out_ch: int, prefer: int = 8) -> int:
    """
    Compute a suitable group count for GroupNorm.

    Parameters
    ----------
    out_ch : int
        Number of output channels.
    prefer : int
        Preferred divisor (default: 8).

    Returns
    -------
    int
        Number of groups for GroupNorm (gcd(out_ch, prefer), clamped to [1, out_ch]).
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class DecoderSlice2D(nn.Module):
    """
    2D decoder over (x, y) applied **slice‑wise along z**.

    Purpose
    -------
    Map fused voxel‑plane features to a single‑channel reconstruction,
    processing each z‑slice with a shared Conv2D → GroupNorm → ReLU stack.

    Axis convention
    ---------------
    Volumes are laid out as **(x, y, z)**:
      - Input  : ``[B, in_ch, X, Y, Z]``  (or ``[B, in_ch, X, Y]`` treated as ``Z=1``)
      - Output : ``[B, 1,     X, Y, Z]``

    Notes
    -----
    • The module flattens (B, Z) → batch to run 2D convolutions efficiently:
        ``[B, C, X, Y, Z] → [B*Z, C, X, Y] → conv stack → [B*Z, 1, X, Y]``,
      then reshapes back to ``[B, 1, X, Y, Z]``.
    • Final 1×1 convolution projects to one output channel; result is clamped
      to ``[0, 1]`` to enforce a valid intensity range.
    """

    def __init__(self, in_ch: int, mid_ch: int = 64, depth: int = 3):
        super().__init__()
        # Channel schedule: [in_ch → mid_ch … → mid_ch → 1]
        ch = [in_ch] + [mid_ch] * depth + [1]
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_gn(ch[i + 1]), ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        # Final 1×1 projection to 1 output channel
        layers += [nn.Conv2d(ch[depth], ch[depth + 1], kernel_size=1, stride=1, padding=0)]
        self.net = nn.Sequential(*layers)

    def forward(self, F_xy):
        """
        Decode fused features into a reconstructed **volume** slice‑wise.

        Parameters
        ----------
        F_xy : Tensor
            Fused, aligned features with shape:
              - ``[B, in_ch, X, Y, Z]``  (preferred), or
              - ``[B, in_ch, X, Y]``     (treated as ``Z=1``).

        Returns
        -------
        Tensor
            Reconstructed volume ``[B, 1, X, Y, Z]`` with values clamped to ``[0, 1]``.
        """
        if F_xy.dim() == 5:
            B, C, X, Y, Z = F_xy.shape
            # [B, C, X, Y, Z] → [B, Z, C, X, Y] → [B*Z, C, X, Y]
            x = F_xy.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, C, X, Y)
        elif F_xy.dim() == 4:
            # Z=1 case
            B, C, X, Y = F_xy.shape
            Z = 1
            x = F_xy.view(B * Z, C, X, Y)
        else:
            raise ValueError(f"DecoderSlice2D expects [B,C,X,Y,Z] or [B,C,X,Y], got {tuple(F_xy.shape)}")

        # 2D Conv stack per slice
        x = self.net(x)  # [B*Z, 1, X, Y]

        # Restore to [B, 1, X, Y, Z] and clamp to [0, 1]
        out = x.view(B, Z, 1, X, Y).permute(0, 2, 3, 4, 1).contiguous()  # [B,1,X,Y,Z]
        return out.clamp_(min=0.0, max=1.0)
