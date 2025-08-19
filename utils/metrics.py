# =================================================================================================
# Metrics & Penalties for Tomographic Reconstruction
# -------------------------------------------------------------------------------------------------
# Purpose
#   Provides quantitative measures and penalty functions to evaluate and regularize
#   reconstructed volumes. Designed to complement structural losses (SSIM, TV) by
#   capturing signal fidelity, occupancy accuracy, and physical plausibility.
#
# Functions
#   • psnr(x, y, eps=1e-8) -> Tensor
#       Peak Signal-to-Noise Ratio (dB). Assumes inputs normalized to [0,1].
#       Computes mean-squared error per batch, converts to dB scale.
#
#   • voxel_error_rate(x, y, thr) -> Tensor
#       Binary classification error of voxel occupancy above threshold `thr`.
#       Returns mean error rate per batch.
#
#   • in_positive_mask_dynamic_range(x, thr, eps=1e-8) -> Tensor
#       Within voxels ≥ thr, computes normalized dynamic range:
#           (max - min) / (mean + eps).
#       Returns zero if no voxel passes threshold.
#
#   • band_penalty(x, low, high) -> Tensor
#       Penalizes values outside the valid band [low, high].
#       Applies ReLU to deviations below/above the band and averages.
#
#   • energy_penalty(pred, gt, eps=1e-8) -> Tensor
#       Penalizes mismatch in total voxel energy (sum of intensities).
#       Computes relative squared error of sums per batch.
#
# Usage
#   p = psnr(recon, gt)
#   ver = voxel_error_rate(recon, gt, thr=0.1)
#   ipdr = in_positive_mask_dynamic_range(recon, thr=0.1)
#   bp = band_penalty(recon, low=0.0, high=1.0)
#   ep = energy_penalty(recon, gt)
#
# Notes
#   • All functions return tensors with shape [B,1,...] for per-batch values.
#   • Use in combination with loss weighting to construct composite objectives.
# =================================================================================================
import torch

def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Assumes inputs normalized to ~[0,1]; we operate in normalized domain anyway
    mse = torch.mean((x - y) ** 2, dim=list(range(1, x.ndim)), keepdim=True)
    return 10.0 * torch.log10(1.0 / (mse + eps))

def voxel_error_rate(x: torch.Tensor, y: torch.Tensor, thr: float) -> torch.Tensor:
    x_b = (x >= thr).to(x.dtype)
    y_b = (y >= thr).to(y.dtype)
    wrong = (x_b != y_b).to(x.dtype)
    return wrong.mean(dim=list(range(1, wrong.ndim)), keepdim=True)

def in_positive_mask_dynamic_range(x: torch.Tensor, thr: float, eps: float = 1e-8) -> torch.Tensor:
    mask = (x >= thr)
    if mask.sum() == 0:
        return torch.zeros_like(x.mean(dim=list(range(1, x.ndim)), keepdim=True))
    vals = x[mask]
    dyn = (vals.max() - vals.min()) / (vals.mean() + eps)
    return dyn.expand(x.shape[0], *([1] * (x.ndim - 1)))

def band_penalty(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    below = torch.relu(low - x)
    above = torch.relu(x - high)
    return (below + above).mean(dim=list(range(1, x.ndim)), keepdim=True)

def energy_penalty(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s_pred = pred.sum(dim=list(range(1, pred.ndim)), keepdim=True)
    s_gt = gt.sum(dim=list(range(1, gt.ndim)), keepdim=True)
    return ((s_pred - s_gt) ** 2) / (s_gt ** 2 + eps)
