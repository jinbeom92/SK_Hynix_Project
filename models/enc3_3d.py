import torch, math
import torch.nn as nn

# ==========================================================================================
# 3D volumetric encoder (planned integration)
#   • Purpose: extract 3D context from ground-truth volumes (V_gt) to inject as priors.
#   • Status: implemented as a standalone module, but not integrated into the main
#             SVTR pipeline yet (training-only branch).
#   • TODO:
#       - Decide fusion interface (concat / attention / gating).
#       - Add options for downsampling strategies to reduce memory.
#       - Exclude from checkpoints/ONNX exports during inference (train-only).
# ==========================================================================================

class Conv3DBlock(nn.Module):
    """
    Conv3D → GroupNorm → SiLU block.

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        k (int): kernel size (default 3)
        s (int): stride (default 1)
        p (int): padding (default 1)
        gn_groups (int): preferred number of groups for GroupNorm

    Notes:
        • Group count is chosen as gcd(out_ch, gn_groups) to divide evenly.
        • SiLU activation is used for smooth non-linearity.
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, gn_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, p)
        g = math.gcd(out_ch, gn_groups)
        if g <= 0: 
            g = 1
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, C_in, D, H, W]

        Returns:
            Tensor: [B, C_out, D, H, W]
        """
        return self.act(self.gn(self.conv(x)))


class Enc3_3D(nn.Module):
    """
    3D Volumetric Encoder (planned; training-only)

    Purpose
    -------
    • Extract hierarchical volumetric features from the ground-truth 3D volume (V_gt).
    • Intended to serve as a **training-only prior** that augments the 2D path
      (Align/Fusion/Decoder) with additional context.
    • Disabled during inference/evaluation.

    Status
    ------
    • Implemented as a module but **not integrated** into the default SVTR pipeline.
    • Future integration planned via Fusion2D or a dedicated 3D→2D adapter
      (e.g., average pooling, projection, or attention-based reduction).

    Args:
        in_ch (int): number of input channels (default 1 for grayscale volumes)
        base  (int): base channel width for the first block
        depth (int): number of Conv3DBlocks, each doubling channels

    Input/Output
    ------------
    Input : V_gt [B, 1, D, H, W]
    Output: latent3d [B, C, D, H, W], where C = base * 2^(depth-1)

    Attributes:
        out_ch (int): number of output channels from the last block
    """
    def __init__(self, in_ch=1, base=16, depth=3):
        super().__init__()
        ch = [in_ch] + [base * (2 ** i) for i in range(depth)]
        blocks = []
        for i in range(depth):
            blocks.append(Conv3DBlock(ch[i], ch[i+1]))
        self.blocks = nn.Sequential(*blocks)
        self.out_ch = ch[-1]

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            vol (Tensor): [B, 1, D, H, W] (normalized ~[0,1])

        Returns:
            Tensor: [B, out_ch, D, H, W] volumetric latent features
        """
        return self.blocks(vol)
