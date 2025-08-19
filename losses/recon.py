# =================================================================================================
# Reconstruction Loss Functions
# -------------------------------------------------------------------------------------------------
# Purpose:
#   Provides a composite set of voxel-domain loss functions to evaluate the quality of reconstructed
#   volumes relative to ground truth. These losses enforce structural fidelity, intensity accuracy,
#   and physical plausibility while optionally adding smoothness regularization.
#
# Components:
#   • SSIM (Structural Similarity, 3D):
#       Encourages structural similarity between reconstructed and ground-truth volumes.
#       Implemented via `utils.ssim3d.ssim3d`.
#
#   • PSNR (Peak Signal-to-Noise Ratio):
#       Penalizes intensity mismatches; expressed as a loss term −PSNR/100.
#       Implemented via `utils.metrics.psnr`.
#
#   • Band Penalty:
#       Enforces voxel values to lie within a specified valid band [low, high].
#       Implemented via `utils.metrics.band_penalty`.
#
#   • Energy Penalty:
#       Matches total energy (sum of voxel values) between reconstruction and ground truth.
#       Implemented via `utils.metrics.energy_penalty`.
#
#   • Voxel Error Rate (VER):
#       Binary classification error of occupied voxels vs. ground truth at a given threshold.
#       Implemented via `utils.metrics.voxel_error_rate`.
#
#   • In-Positive Dynamic Range (IPDR):
#       Measures dynamic range consistency in regions above threshold.
#       Implemented via `utils.metrics.in_positive_mask_dynamic_range`.
#
#   • TV (Total Variation, optional):
#       Encourages smoothness/isotropy in reconstructed volumes. Weighted by `tv_weight`.
#
# Usage:
#   loss_dict = reconstruction_losses(R_hat_n, V_gt_n, weights, params)
#   where:
#     - R_hat_n: normalized reconstructed volume [B,1,D,H,W] or [B,D,H,W].
#     - V_gt_n : normalized ground truth volume [B,1,D,H,W] or [B,D,H,W].
#     - weights: dict of loss weights (normalized externally to sum≈1).
#     - params : dict with keys {"band_low","band_high","ver_thr","tv_weight"}.
#
# Returns:
#   loss_dict: dictionary of individual losses plus:
#       • "total_recon" — weighted sum of selected terms.
#
# Notes:
#   • Shapes are internally normalized to [B,1,D,H,W].
#   • TV loss is optional; set params["tv_weight"]>0 to enable.
#   • Supports differentiability for gradient-based training.
# =================================================================================================
import torch
from utils.ssim3d import ssim3d
from utils.metrics import psnr, band_penalty, energy_penalty, voxel_error_rate, in_positive_mask_dynamic_range

def tv_isotropic_3d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    tv = torch.sqrt(dz.pow(2) + dy.pow(2) + dx.pow(2) + eps).mean(dim=list(range(1, x.ndim)), keepdim=True)
    return tv

def reconstruction_losses(R_hat_n: torch.Tensor, V_gt_n: torch.Tensor, weights: dict, params: dict):
    loss_dict = {}
    # Ensure shapes [B,1,D,H,W]
    if R_hat_n.ndim == 4: R_hat_n = R_hat_n.unsqueeze(1)
    if V_gt_n.ndim == 4: V_gt_n = V_gt_n.unsqueeze(1)
    # 1 - SSIM
    loss_dict["ssim"] = ssim3d(R_hat_n, V_gt_n)
    # -PSNR/100
    loss_dict["psnr"] = -psnr(torch.clamp(R_hat_n,0,1), torch.clamp(V_gt_n,0,1)) / 100.0
    # band penalty
    loss_dict["band"] = band_penalty(R_hat_n, params["band_low"], params["band_high"])
    # energy
    loss_dict["energy"] = energy_penalty(R_hat_n, V_gt_n)
    # VER
    loss_dict["ver"] = voxel_error_rate(R_hat_n, V_gt_n, params["ver_thr"])
    # IPDR
    loss_dict["ipdr"] = in_positive_mask_dynamic_range(R_hat_n, params["ver_thr"])
    # TV optional
    if params.get("tv_weight", 0.0) > 0:
        loss_dict["tv"] = tv_isotropic_3d(R_hat_n) * params["tv_weight"]
    else:
        loss_dict["tv"] = torch.zeros_like(loss_dict["ssim"])

    # Weighted sum (weights are normalized to sum 1 outside)
    total = torch.zeros_like(loss_dict["ssim"])
    for k, w in weights.items():
        if k in loss_dict:
            total = total + float(w) * loss_dict[k]
    loss_dict["total_recon"] = total
    return loss_dict
