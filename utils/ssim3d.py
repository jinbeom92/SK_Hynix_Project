import torch
import torch.nn.functional as F


def _gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype):
    """
    Build a normalized 1D Gaussian kernel centered at zero.

    Args:
        kernel_size (int): Odd kernel length (e.g., 7).
        sigma (float): Standard deviation of the Gaussian.
        device, dtype: Torch device/dtype for the returned tensor.

    Returns:
        Tensor: [K] 1D kernel with sum = 1.

    Notes:
        • The support is symmetric around zero: indices in
          {-(K-1)/2, ..., 0, ..., +(K-1)/2}.
        • For stability and consistency, the kernel is normalized to unit sum.
    """
    k = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    w = torch.exp(-(k ** 2) / (2 * sigma * sigma))
    w = w / w.sum()
    return w


def _gaussian_kernel3d(kernel_size: int, sigma: float, device, dtype):
    """
    Build a separable, normalized 3D Gaussian kernel.

    Args:
        kernel_size (int): Odd kernel length per axis.
        sigma (float): Standard deviation (same on all three axes).
        device, dtype: Torch device/dtype.

    Returns:
        Tensor: [1, 1, K, K, K] 3D kernel with sum = 1 (ready for conv3d).
    """
    k1 = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    # Outer products to form separable 3D kernel
    k3 = torch.einsum("i,j,k->ijk", k1, k1, k1)
    k3 = k3 / k3.sum()
    k3 = k3.view(1, 1, kernel_size, kernel_size, kernel_size)
    return k3


def ssim3d(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 1.5,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
):
    """
    3D Structural Similarity (SSIM) loss (returns 1 − SSIM).

    Args:
        x (Tensor): [B, 1, D, H, W] predicted volume (typically normalized to ~[0,1]).
        y (Tensor): [B, 1, D, H, W] target volume   (typically normalized to ~[0,1]).
        kernel_size (int): Gaussian window size (odd, e.g., 7).
        sigma (float): Gaussian std for the window (e.g., 1.5).
        C1 (float): Small constant for luminance term (default assumes L=1).
        C2 (float): Small constant for contrast term  (default assumes L=1).

    Returns:
        Tensor: [B, 1, 1, 1, 1] loss = 1 − SSIM, averaged over spatial/volumetric dims.

    Implementation details
    ----------------------
    • Local means/variances/covariance are computed by Gaussian smoothing
      (via conv3d) and the standard SSIM formula:
        SSIM = ((2μ_xμ_y + C1)(2σ_xy + C2)) / ((μ_x^2 + μ_y^2 + C1)(σ_x^2 + σ_y^2 + C2))
    • This implementation assumes single-channel 3D volumes; for multi-channel,
      apply per-channel and average.
    • Padding uses `padding=kernel_size//2` to preserve size; boundary handling
      follows conv3d’s zero-padding convention.

    Notes
    -----
    • This function returns (1 − SSIM) so it can be used directly as a loss term.
    • Ensure inputs are on the same device/dtype; the Gaussian kernel is created
      to match x’s device/dtype.
    """
    # x,y: [B,1,D,H,W]
    device, dtype = x.device, x.dtype

    # Gaussian window (separable 3D)
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
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    ssim_val = 1.0 - ssim_map.mean(dim=list(range(1, ssim_map.ndim)), keepdim=True)
    return ssim_val
