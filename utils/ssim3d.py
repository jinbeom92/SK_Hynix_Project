# =================================================================================================
# ssim3d — 3D Structural Similarity Index (SSIM) for Volumetric Data
# -------------------------------------------------------------------------------------------------
# Purpose
#   Computes the Structural Similarity Index (SSIM) between two 3D volumes using a Gaussian
#   smoothing kernel. SSIM is a perceptual metric that captures structural fidelity by comparing
#   local means, variances, and covariances of image patches. Returned as (1 − SSIM) so it can
#   be directly used as a loss.
#
# Functions
#   • _gaussian_kernel1d(kernel_size, sigma, device, dtype)
#       Builds a 1D Gaussian kernel of length `kernel_size` with standard deviation `sigma`.
#
#   • _gaussian_kernel3d(kernel_size, sigma, device, dtype)
#       Constructs a 3D Gaussian kernel via outer product of 1D kernels.
#       Normalized to sum=1, shaped [1,1,K,K,K] for conv3d.
#
#   • ssim3d(x, y, kernel_size=7, sigma=1.5, C1=0.01², C2=0.03²)
#       - Inputs:
#           x, y : torch.Tensor [B,1,D,H,W] — volumetric batches.
#       - Steps:
#           1. Compute Gaussian-filtered local means μx, μy.
#           2. Compute variances σx², σy² and covariance σxy.
#           3. Apply SSIM formula:
#                SSIM = ((2μxμy + C1)(2σxy + C2)) /
#                        ((μx²+μy² + C1)(σx²+σy² + C2))
#           4. Return (1 − SSIM) averaged over spatial dims.
#       - Output:
#           [B,1,1,1,1] tensor of per-sample loss values.
#
# Usage
#   loss = ssim3d(pred, target)  # both normalized [B,1,D,H,W]
#
# Notes
#   • Constants C1, C2 stabilize division against weak denominators.
#   • Kernel size and σ control the SSIM window; defaults match image-based SSIM heuristics.
#   • Returns differentiable loss; lower is better (closer structural match).
# =================================================================================================
import torch
import torch.nn.functional as F

def _gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype):
    k = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    w = torch.exp(-(k ** 2) / (2 * sigma * sigma))
    w = w / w.sum()
    return w

def _gaussian_kernel3d(kernel_size: int, sigma: float, device, dtype):
    k1 = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    k3 = torch.einsum("i,j,k->ijk", k1, k1, k1)
    k3 = k3 / k3.sum()
    k3 = k3.view(1, 1, kernel_size, kernel_size, kernel_size)
    return k3

def ssim3d(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5, C1: float = 0.01**2, C2: float = 0.03**2):
    # x,y: [B,1,D,H,W]
    device, dtype = x.device, x.dtype
    k = _gaussian_kernel3d(kernel_size, sigma, device, dtype)
    mu_x = F.conv3d(x, k, padding=kernel_size//2, groups=1)
    mu_y = F.conv3d(y, k, padding=kernel_size//2, groups=1)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv3d(x * x, k, padding=kernel_size//2, groups=1) - mu_x2
    sigma_y2 = F.conv3d(y * y, k, padding=kernel_size//2, groups=1) - mu_y2
    sigma_xy = F.conv3d(x * y, k, padding=kernel_size//2, groups=1) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    ssim_val = 1.0 - ssim_map.mean(dim=list(range(1, ssim_map.ndim)), keepdim=True)
    return ssim_val
