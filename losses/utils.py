# =================================================================================================
# Normalization Utilities for Sinogram and Volume Tensors
# -------------------------------------------------------------------------------------------------
# Purpose:
#   Provides utility functions to normalize sinogram and voxel tensors on a per-sample basis,
#   improving numerical stability and ensuring consistent loss scaling across heterogeneous
#   data samples.
#
# Functions:
#   • per_sample_sino_scale(S_in, q=0.99, eps=1e-6) -> torch.Tensor
#       - Computes a robust per-sample normalization factor for sinograms.
#       - For each batch element, calculates the q-th quantile of |S_in| values (default 99%).
#       - Clamps minimum scale to `eps` to avoid division by zero.
#       - Returns scale tensor of shape [B,1,1,1] (non-trainable, no gradient).
#
#   • normalize_tensors(S_in, sino_hat, R_hat, V_gt, s_sino) -> Tuple[Tensor,...]
#       - Applies per-sample normalization to inputs and outputs using scale `s_sino`.
#       - S_in_n     = S_in     / s_sino       → normalized input sinogram [B,A,V,U]
#       - sino_hat_n = sino_hat / s_sino       → normalized predicted sinogram [B,A,V,U]
#       - R_hat_n    = R_hat    / s_vol        → normalized reconstruction [B,1,D,H,W]
#       - V_gt_n     = V_gt     / s_vol        → normalized ground-truth volume [B,1,D,H,W]
#       - where s_vol is reshaped from s_sino to [B,1,1,1,1].
#
# Usage:
#   s_sino = per_sample_sino_scale(S_in, q=0.99)
#   S_in_n, sino_hat_n, R_hat_n, V_gt_n = normalize_tensors(S_in, sino_hat, R_hat, V_gt, s_sino)
#
# Notes:
#   • Normalization is critical for balancing loss contributions across batches with
#     variable intensity scales.
#   • Robust quantile scaling (instead of max) prevents sensitivity to outliers.
#   • Ensures forward-consistency and reconstruction losses are computed in a normalized domain.
# =================================================================================================
import torch

def per_sample_sino_scale(S_in: torch.Tensor, q: float = 0.99, eps: float = 1e-6) -> torch.Tensor:
    with torch.no_grad():
        B = S_in.shape[0]
        s = torch.quantile(S_in.abs().reshape(B, -1), q, dim=1)
        s = s.clamp_min(eps).view(B, 1, 1, 1)  # [B,1,1,1]
    return s  # requires_grad=False

def normalize_tensors(S_in, sino_hat, R_hat, V_gt, s_sino: torch.Tensor):
    S_in_n    = S_in / s_sino                     # [B,A,V,U]
    sino_hat_n= sino_hat / s_sino                 # [B,A,V,U]
    s_vol     = s_sino.view(-1,1,1,1,1)           # [B,1,1,1,1]
    R_hat_n   = R_hat / s_vol                     # [B,1,D,H,W]
    V_gt_n    = V_gt  / s_vol                     # [B,1,D,H,W]
    return S_in_n, sino_hat_n, R_hat_n, V_gt_n