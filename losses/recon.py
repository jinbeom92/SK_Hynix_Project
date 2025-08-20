# =================================================================================================
# Reconstruction Loss Functions (2D/3D-safe)
# -------------------------------------------------------------------------------------------------
# Purpose:
#   Composite voxel-domain losses for reconstructed vs. ground-truth volumes/slices.
#
# Supports:
#   • 3D volumes [B,1,D,H,W]
#   • 2D slices  [B,1,1,H,W]  (SSIM falls back to 2D window)
# =================================================================================================
import torch
import torch.nn.functional as F
from utils.metrics import psnr, band_penalty, energy_penalty, voxel_error_rate, in_positive_mask_dynamic_range
from utils.ssim3d import ssim3d

def _ssim_safe(R: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # If depth == 1, approximate with 2D SSIM on the slice
    if R.shape[2] == 1 and V.shape[2] == 1:
        # reshape to [B,1,H,W]
        r2 = R[:, :, 0, :, :]
        v2 = V[:, :, 0, :, :]
        # simple 2D SSIM via avg pooling / variance (no external deps)
        # window 7x7, padding 'replicate'
        k = 7
        pad = k // 2
        def _moments(x):
            mu = F.avg_pool2d(x, k, 1, pad, count_include_pad=False)
            mu2 = mu * mu
            sigma2 = F.avg_pool2d(x * x, k, 1, pad, count_include_pad=False) - mu2
            return mu, sigma2
        mu_x, sig2_x = _moments(r2)
        mu_y, sig2_y = _moments(v2)
        cov_xy = F.avg_pool2d(r2 * v2, k, 1, pad, count_include_pad=False) - mu_x * mu_y
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sig2_x + sig2_y + C2) + 1e-12)
        # return 1 - SSIM as loss term to match previous convention
        return 1.0 - ssim_map.mean(dim=list(range(1, ssim_map.ndim)), keepdim=True)
    else:
        return ssim3d(R, V)

def tv_isotropic_3d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    tv = torch.sqrt(dz.pow(2) + dy.pow(2) + dx.pow(2) + eps).mean(dim=list(range(1, x.ndim)), keepdim=True)
    return tv

def reconstruction_losses(R_hat_n: torch.Tensor, V_gt_n: torch.Tensor, weights: dict, params: dict):
    loss_dict = {}
    # shapes [B,1,D,H,W]
    if R_hat_n.ndim == 4: R_hat_n = R_hat_n.unsqueeze(1)
    if V_gt_n.ndim == 4: V_gt_n = V_gt_n.unsqueeze(1)

    # 1 - SSIM
    loss_dict["ssim"] = _ssim_safe(R_hat_n, V_gt_n)

    # PSNR in dB for logging
    psnr_db = psnr(torch.clamp(R_hat_n,0,1), torch.clamp(V_gt_n,0,1))
    loss_dict["psnr"] = psnr_db

    # Normalized PSNR loss (higher PSNR → lower loss)
    # Use a reference PSNR (e.g. 40 dB) to map to [0,1]
    p_ref = float(params.get("psnr_ref", 40.0))
    psnr_loss = torch.clamp((p_ref - psnr_db) / p_ref, min=0.0, max=1.0)
    loss_dict["psnr_loss"] = psnr_loss

    # Other penalties
    loss_dict["band"] = band_penalty(R_hat_n, params["band_low"], params["band_high"])
    loss_dict["energy"] = energy_penalty(R_hat_n, V_gt_n)
    loss_dict["ver"] = voxel_error_rate(R_hat_n, V_gt_n, params["ver_thr"])
    loss_dict["ipdr"] = in_positive_mask_dynamic_range(R_hat_n, params["ver_thr"])
    if params.get("tv_weight", 0.0) > 0:
        loss_dict["tv"] = tv_isotropic_3d(R_hat_n) * params["tv_weight"]
    else:
        loss_dict["tv"] = torch.zeros_like(loss_dict["ssim"])

    # Total loss uses psnr_loss instead of raw psnr_db
    total = torch.zeros_like(loss_dict["ssim"])
    for k, w in weights.items():
        if k == "psnr":
            total = total + float(w) * loss_dict.get("psnr_loss", 0.0)
        elif k in loss_dict:
            total = total + float(w) * loss_dict[k]
    loss_dict["total_recon"] = total
    return loss_dict
