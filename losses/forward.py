import torch
import torch.nn.functional as F

__all__ = [
    "radon_forward_slice",
    "forward_consistency_loss",
]

def _to_b1xyz(img: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize input to [B, 1, X, Y, Z] without permuting axes.

    Accepted shapes
    ---------------
    - [B, X, Y, Z]      → unsqueeze channel → [B, 1, X, Y, Z]
    - [B, 1, X, Y, Z]   → returned as-is
    - [B, 1, X, Y]      → add Z=1          → [B, 1, X, Y, 1]

    Raises
    ------
    ValueError if shape doesn't match any of the above.
    """
    if img.dim() == 5:
        B, C, X, Y, Z = img.shape
        if C != 1:
            raise ValueError(f"img with shape {tuple(img.shape)} has C={C} ≠ 1.")
        return img
    elif img.dim() == 4:
        # Ambiguity: could be [B,1,X,Y] or [B,X,Y,Z].
        if img.shape[1] == 1:
            # [B,1,X,Y] → [B,1,X,Y,1]
            return img.unsqueeze(-1)
        else:
            # [B,X,Y,Z] → [B,1,X,Y,Z]
            return img.unsqueeze(1)
    else:
        raise ValueError(
            "img must be [B,X,Y,Z], [B,1,X,Y,Z], or [B,1,X,Y]. "
            f"Got {tuple(img.shape)}"
        )


def radon_forward_slice(img: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Parallel-beam Radon transform of an entire **(x,y,z)** volume into a
    **(x,a,z)** sinogram, computed slice-by-slice along z.

    Parameters
    ----------
    img : Tensor
        Reconstructed volume with shape **[B, X, Y, Z]**, **[B, 1, X, Y, Z]**,
        or a single 2D slice **[B, 1, X, Y]** (treated as Z=1).
        The second dimension is interpreted as channel and must be 1 when present.
    angles : Tensor
        1D tensor of length **A** with projection angles (radians).

    Returns
    -------
    Tensor
        Sinogram with shape **[B, X, A, Z]** where:
        - X: detector bins (integration is along **Y**),
        - A: number of angles,
        - Z: depth (slice index).

    Notes
    -----
    * Rotation uses `torch.nn.functional.grid_sample` with bilinear sampling
      and zero padding, centered on the image.
    * Integration is the discrete sum along the **Y** axis to produce **X**
      detector bins, so the result matches the (x, a, z) convention.
    * Differentiable w.r.t. `img`.
    """
    # Canonicalize to [B,1,X,Y,Z]
    img = _to_b1xyz(img)
    device, dtype = img.device, img.dtype
    B, _, X, Y, Z = img.shape

    # Angles
    angles = angles.to(device=device, dtype=dtype)
    A = int(angles.numel())

    # Prepare output [B, X, A, Z]
    sino = img.new_zeros((B, X, A, Z))

    # Flatten (B, Z) into a single batch so we rotate all slices at once per angle.
    # img_bz: [B*Z, 1, X, Y]
    img_bz = img.permute(0, 4, 1, 2, 3).contiguous().view(B * Z, 1, X, Y)

    # Process each angle; per-angle grid (shared across all B*Z slices)
    for i in range(A):
        theta = angles[i]
        c = torch.cos(theta)
        s = torch.sin(theta)

        # Build [B*Z, 2, 3] rotation matrices on the correct device/dtype
        rot = img_bz.new_zeros((B * Z, 2, 3))
        rot[:, 0, 0] = c
        rot[:, 0, 1] = -s
        rot[:, 1, 0] = s
        rot[:, 1, 1] = c

        # Grid and rotate all (B*Z) slices together
        grid = F.affine_grid(rot, size=(B * Z, 1, X, Y), align_corners=False)
        rotated = F.grid_sample(
            img_bz, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # [B*Z,1,X,Y]

        # Integrate along Y → detector bins X
        proj = rotated.sum(dim=3)  # [B*Z,1,X]

        # Reshape back to [B,X,Z] and place into angle slot i
        proj_bxz = proj.view(B, Z, 1, X).permute(0, 3, 2, 1).squeeze(2)  # [B,X,Z]
        sino[:, :, i, :] = proj_bxz

    return sino


def forward_consistency_loss(
    pred_slice: torch.Tensor,
    gt_slice: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """
    Forward-projection consistency loss on **(x,y,z)** volumes.

    Given predicted and ground-truth recon volumes (any of
    ``[B,X,Y,Z]``, ``[B,1,X,Y,Z]`` or 2D slices ``[B,1,X,Y]``),
    compute their sinograms with ``radon_forward_slice`` and return the
    mean squared error over **[B, X, A, Z]**.

    Parameters
    ----------
    pred_slice : Tensor
        Predicted volume/slice, accepts ``[B,X,Y,Z]``, ``[B,1,X,Y,Z]`` or ``[B,1,X,Y]``.
    gt_slice : Tensor
        Ground-truth volume/slice, accepts the same shapes as `pred_slice`.
    angles : Tensor
        1D tensor of projection angles in radians (length **A**).

    Returns
    -------
    Tensor
        Scalar MSE loss between the two sinograms.
    """
    sino_pred = radon_forward_slice(pred_slice, angles)  # [B, X, A, Z]
    sino_gt   = radon_forward_slice(gt_slice, angles)    # [B, X, A, Z]
    return F.mse_loss(sino_pred, sino_gt)
