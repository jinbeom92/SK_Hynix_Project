import math
import torch
import torch.nn.functional as F

__all__ = [
    "radon_forward_slice",
    "forward_consistency_loss",
]

def radon_forward_slice(img: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Compute a parallel-beam Radon transform (forward projection) of a batch
    of 2D images using rotation and integration.  This function maps a
    reconstructed slice into a sinogram by rotating it through the given
    angles and summing pixel values along one axis, matching the physics
    of parallel-beam CT.【470627143552496†L17-L23】

    Parameters
    ----------
    img : Tensor
        Reconstructed slices with shape ``[B, C, X, Y]``, where ``C`` is the
        channel dimension (typically 1), ``X`` and ``Y`` are the spatial
        dimensions (height and width).
    angles : Tensor
        1D tensor of length ``A`` containing projection angles in radians.

    Returns
    -------
    Tensor
        A sinogram tensor of shape ``[B, C, X, A]``.  The first spatial
        dimension ``X`` corresponds to detector bins (U) and the last
        dimension ``A`` corresponds to projection angles.

    Notes
    -----
    * The transform is differentiable with respect to ``img``.  It uses
      ``torch.nn.functional.grid_sample`` to implement rotation and then
      integrates along the width axis to simulate detector integration.
    * The rotation is performed about the center of the image.  Pixels
      outside the field of view are implicitly zero.
    """
    B, C, X, Y = img.shape
    device = img.device
    dtype = img.dtype
    A = angles.numel()
    # Prepare output tensor
    sino = torch.zeros((B, C, X, A), device=device, dtype=dtype)
    # Construct base grid for one image (normalized coordinates)
    base_grid = None
    # For each angle, rotate the image and integrate along width
    for i, theta in enumerate(angles):
        c = torch.cos(theta).to(device=device, dtype=dtype)
        s = torch.sin(theta).to(device=device, dtype=dtype)
        # Affine matrix for rotation (no translation)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=device, dtype=dtype)
        rot = rot.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, 3]
        # Compute grid only once for the current angle and batch
        grid = F.affine_grid(rot, size=img.shape, align_corners=False)
        # Sample the rotated image
        rotated = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # Integrate along width (axis=3) to form detector bin values
        # rotated shape: [B, C, X, Y]; integrate along Y → [B, C, X]
        proj = rotated.sum(dim=3)
        sino[:, :, :, i] = proj
    return sino


def forward_consistency_loss(pred_slice: torch.Tensor,
                             gt_slice: torch.Tensor,
                             angles: torch.Tensor) -> torch.Tensor:
    """
    Compute a forward-projection consistency loss between predicted and
    ground‑truth slices.  This loss encourages the network's predicted
    reconstruction to produce the same sinogram as the ground‑truth when
    projected with a parallel-beam Radon transform.

    Given a predicted slice ``pred_slice`` and a ground‑truth slice
    ``gt_slice`` (both shaped ``[B, 1, X, Y]``), this function
    computes their sinograms using ``radon_forward_slice`` and then
    returns the mean squared error (MSE) between the two sinograms.

    Parameters
    ----------
    pred_slice : Tensor
        Predicted reconstructed slice with shape ``[B, 1, X, Y]``.
    gt_slice : Tensor
        Ground‑truth reconstructed slice with shape ``[B, 1, X, Y]``.
    angles : Tensor
        1D tensor of projection angles in radians.  Its length defines
        the number of projections ``A``.

    Returns
    -------
    Tensor
        Scalar tensor representing the MSE between the sinograms of the
        predicted and ground‑truth slices.  The result is differentiable
        with respect to the inputs and can be used directly as a loss.
    """
    # Compute forward projections (sinograms) of predicted and ground‑truth slices
    sino_pred = radon_forward_slice(pred_slice, angles)  # [B,1,X,A]
    sino_gt = radon_forward_slice(gt_slice, angles)      # [B,1,X,A]
    # Mean squared error over all elements
    return F.mse_loss(sino_pred, sino_gt)
