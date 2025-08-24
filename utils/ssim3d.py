import torch
import torch.nn.functional as F


def _gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype):
    """
    Build a normalized 1D Gaussian kernel centered at zero.

    Parameters
    ----------
    kernel_size : int
        Odd kernel length (e.g., 7). Must be >= 1 and odd.
    sigma : float
        Standard deviation of the Gaussian. Should be > 0 for non-trivial blur.
    device, dtype :
        Torch device/dtype for the returned tensor.

    Returns
    -------
    Tensor
        1D kernel with shape ``[K]`` and sum = 1.

    Notes
    -----
    • Support is symmetric around zero: indices in
      {-(K-1)/2, ..., 0, ..., +(K-1)/2}.
    • Kernel is normalized to unit sum for numerical stability.
    """
    if kernel_size < 1 or (kernel_size % 2) != 1:
        raise ValueError(f"`kernel_size` must be odd and >=1, got {kernel_size}.")
    if sigma <= 0:
        # Degenerate "identity" kernel
        w = torch.zeros(kernel_size, device=device, dtype=dtype)
        w[(kernel_size - 1) // 2] = 1.0
        return w

    k = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    w = torch.exp(-(k ** 2) / (2 * sigma * sigma))
    w = w / w.sum()
    return w


def _gaussian_kernel3d(kernel_size: int, sigma: float, device, dtype):
    """
    Build a separable, normalized 3D Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        Odd kernel length per axis (same for all three axes).
    sigma : float
        Standard deviation (same on all three axes).
    device, dtype :
        Torch device/dtype.

    Returns
    -------
    Tensor
        3D kernel ``[1, 1, K, K, K]`` with sum = 1 (ready for conv3d).

    Implementation
    --------------
    Constructed via outer products of the 1D kernel along each axis.
    """
    k1 = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    k3 = torch.einsum("i,j,k->ijk", k1, k1, k1)
    k3 = k3 / k3.sum()
    return k3.view(1, 1, kernel_size, kernel_size, kernel_size)


def ssim3d(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 1.5,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
):
    """
    3D Structural Similarity (SSIM) loss for **(x,y,z)** volumes — returns ``1 − SSIM``.

    Axis convention (model-facing)
    ------------------------------
    This function expects volumes in **[B, 1, Z, X, Y]**.  
    If your tensors are **[B, 1, X, Y, Z]**, permute to ``[B,1,Z,X,Y]`` before calling
    (as done in the project’s `_ssim_safe`).  The last three dims map to
    conv3d’s **(D,H,W) = (Z,X,Y)**.

    Parameters
    ----------
    x : Tensor
        Predicted volume, shape ``[B, 1, Z, X, Y]``, typically normalized to ~[0,1].
    y : Tensor
        Target volume, shape ``[B, 1, Z, X, Y]``.
    kernel_size : int
        Gaussian window size (odd, e.g., 7).
    sigma : float
        Gaussian std for the window (e.g., 1.5).
    C1 : float
        Small constant for luminance term (assumes L=1).
    C2 : float
        Small constant for contrast term  (assumes L=1).

    Returns
    -------
    Tensor
        Loss tensor ``[B, 1, 1, 1, 1]`` equal to ``1 − SSIM``, i.e., averaged
        over Z/X/Y while preserving broadcastable batch dims.

    Implementation details
    ----------------------
    • Local means/variances/covariance are computed by Gaussian smoothing
      (conv3d) and then plugged into the standard SSIM formula:
        SSIM = ((2 μ_x μ_y + C1)(2 σ_xy + C2)) / ((μ_x^2 + μ_y^2 + C1)(σ_x^2 + σ_y^2 + C2))
    • Single-channel volumes are assumed; for multi-channel, apply per-channel
      and average upstream if needed.
    • Padding uses ``padding=kernel_size//2`` (zero padding).
    """
    if x.ndim != 5 or y.ndim != 5:
        raise ValueError(f"ssim3d expects 5D tensors [B,1,Z,X,Y], got {x.shape} and {y.shape}.")
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError(f"ssim3d expects single-channel inputs, got C={x.shape[1]} and C={y.shape[1]}.")

    device, dtype = x.device, x.dtype
    if y.device != device or y.dtype != dtype:
        y = y.to(device=device, dtype=dtype)

    if kernel_size < 1 or (kernel_size % 2) != 1:
        raise ValueError(f"`kernel_size` must be odd and >=1, got {kernel_size}.")

    # Separable 3D Gaussian window
    k = _gaussian_kernel3d(kernel_size, sigma, device, dtype)

    # Local means
    mu_x = F.conv3d(x, k, padding=kernel_size // 2, groups=1)
    mu_y = F.conv3d(y, k, padding=kernel_size // 2, groups=1)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # Local variances and covariance
    sigma_x2 = F.conv3d(x * x, k, padding=kernel_size // 2, groups=1) - mu_x2
    sigma_y2 = F.conv3d(y * y, k, padding=kernel_size // 2, groups=1) - mu_y2
    sigma_xy = F.conv3d(x * y, k, padding=kernel_size // 2, groups=1) - mu_xy

    # SSIM map and loss (1 − SSIM)
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    loss = 1.0 - ssim_map

    # Reduce over spatial/volumetric dims, keep broadcastable batch dims
    return loss.mean(dim=list(range(1, loss.ndim)), keepdim=True)
