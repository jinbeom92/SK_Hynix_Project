import math
import torch.nn as nn

def _gn(out_ch: int, prefer: int = 8) -> int:
    """
    Compute a suitable group count for GroupNorm.

    Args:
        out_ch (int): number of output channels
        prefer (int): preferred divisor (default 8)

    Returns:
        int: number of groups for GroupNorm
             • gcd(out_ch, prefer) if > 0
             • clamped between 1 and out_ch
    """
    g = math.gcd(out_ch, prefer)
    return max(1, min(out_ch, g if g > 0 else 1))


class DecoderSlice2D(nn.Module):
    """
    2D Decoder block that maps aligned + fused voxel-plane features [B,C,X,Y]
    into a reconstructed slice [B,1,X,Y].

    Purpose
    -------
    • Refines fused feature maps through multiple Conv2D + GroupNorm + ReLU blocks.
    • Projects down to a single-channel voxel slice with a final 1×1 convolution.
    • Typically used after Fusion2D inside SVTRSystem.

    Args
    ----
    in_ch (int)   : number of input channels
    mid_ch (int)  : intermediate channel width for hidden conv layers (default 64)
    depth (int)   : number of conv→norm→ReLU blocks before final projection (default 3)

    Shapes
    ------
    Input:
        F_xy : [B, in_ch, X, Y]   — fused aligned features
    Output:
        out  : [B, 1, X, Y]       — reconstructed voxel slice
    """

    def __init__(self, in_ch: int, mid_ch: int = 64, depth: int = 3):
        super().__init__()
        # Channel schedule: [in_ch → mid_ch … → mid_ch → 1]
        ch = [in_ch] + [mid_ch] * depth + [1]
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(ch[i], ch[i+1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_gn(ch[i+1]), ch[i+1]),
                nn.ReLU(inplace=True),
            ]
        # Final 1×1 projection to 1 output channel
        layers += [nn.Conv2d(ch[depth], ch[depth+1], kernel_size=1, stride=1, padding=0)]
        self.net = nn.Sequential(*layers)

    def forward(self, F_xy):
        """
        Decode fused features into a reconstructed slice and clamp to [0, 1].

        This decoder processes the aligned and fused feature map ``F_xy``
        through a series of convolutional blocks and a final 1×1
        projection.  The raw output may contain negative values or
        exceed unity because convolutional filters can introduce
        oscillations and overshoot, analogous to the negative intensities
        observed in ramp‑filtered backprojection【508774924062640†L39-L44】.  To
        enforce a physically meaningful intensity range, we clamp the
        reconstruction to the interval ``[0, 1]`` before returning it.

        Args:
            F_xy (Tensor): `[B, in_ch, X, Y]` fused feature map from the encoder,
                alignment and fusion stages.

        Returns:
            Tensor: `[B, 1, X, Y]` reconstructed voxel slice with values in
            ``[0, 1]``.
        """
        out = self.net(F_xy)
        # Clamp in-place: negative values → 0, values >1 → 1.
        return out.clamp_(min=0.0, max=1.0)
