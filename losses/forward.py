from typing import Any, Dict
import torch
import torch.nn.functional as F

__all__ = [
    "radon_forward_slice",
    "forward_consistency_loss",
]

def _to_b1xyz(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 5:
        B, C, X, Y, Z = img.shape
        if C != 1:
            raise ValueError(f"img with shape {tuple(img.shape)} has C={C} ≠ 1.")
        return img
    elif img.dim() == 4:
        if img.shape[1] == 1:
            return img.unsqueeze(-1)
        else:
            return img.unsqueeze(1)
    else:
        raise ValueError(
            "img must be [B,X,Y,Z], [B,1,X,Y,Z], or [B,1,X,Y]. "
            f"Got {tuple(img.shape)}"
        )


def radon_forward_slice(img: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    # Canonicalize to [B,1,X,Y,Z]
    img = _to_b1xyz(img)
    device, dtype = img.device, img.dtype
    B, _, X, Y, Z = img.shape

    angles = angles.to(device=device, dtype=dtype)
    A = int(angles.numel())

    sino = img.new_zeros((B, X, A, Z))

    # Flatten (B, Z) into a single batch
    img_bz = img.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, 1, X, Y)

    for i in range(A):
        theta = angles[i]
        c = torch.cos(theta)
        s = torch.sin(theta)

        # [B*Z, 2, 3] rotation matrices
        rot = img_bz.new_zeros((B * Z, 2, 3))
        rot[:, 0, 0] = c
        rot[:, 0, 1] = -s
        rot[:, 1, 0] = s
        rot[:, 1, 1] = c

        grid = F.affine_grid(rot, size=(B * Z, 1, X, Y), align_corners=True)
        rotated = F.grid_sample(
            img_bz, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [B*Z,1,X,Y]

        proj = rotated.sum(dim=3)  # [B*Z,1,X]
        proj_bxz = proj.view(B, Z, 1, X).permute(0, 3, 2, 1).squeeze(2)  # [B,X,Z]
        sino[:, :, i, :] = proj_bxz

    return sino


def forward_consistency_loss(
    pred_slice: torch.Tensor,
    gt_slice: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    sino_pred = radon_forward_slice(pred_slice, angles)  # [B, X, A, Z]
    sino_gt   = radon_forward_slice(gt_slice, angles)    # [B, X, A, Z]
    return F.mse_loss(sino_pred, sino_gt)
