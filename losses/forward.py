# =================================================================================================
# Forward Consistency Loss
# -------------------------------------------------------------------------------------------------
# Purpose:
#   Encourages agreement between the predicted sinogram and the input sinogram by enforcing
#   frequency-domain consistency. Designed to penalize discrepancies more strongly at lower
#   frequencies, which are critical for stable tomographic reconstruction.
#
# Method:
#   • Compute residual = S_pred_n − S_in_n (normalized domain).
#   • Apply 1D FFT along the detector-bin axis (last dimension).
#   • Take magnitude of the Fourier coefficients (differentiable).
#   • Construct frequency weights w(k) = (1 − k) ^ α, emphasizing low-frequency components.
#   • Compute the weighted mean magnitude as the loss.
#
# Parameters:
#   S_pred_n : torch.Tensor
#       Normalized predicted sinogram. Shape [..., A, V, U].
#   S_in_n   : torch.Tensor
#       Normalized input sinogram (ground truth). Shape [..., A, V, U].
#   alpha    : float (default=0.7)
#       Exponent controlling the strength of low-frequency emphasis.
#
# Output:
#   torch.Tensor (scalar)
#       Forward consistency loss value.
#
# Notes:
#   • Differentiable w.r.t. residual in the spatial domain.
#   • Low-frequency emphasis stabilizes training against noise and aliasing.
#   • Integrates seamlessly with reconstruction + structural losses.
#
# Usage:
#   loss = forward_consistency_loss(S_pred_n, S_in_n, alpha=0.7)
# =================================================================================================
import torch

def forward_consistency_loss(S_pred_n: torch.Tensor, S_in_n: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    resid = (S_pred_n - S_in_n).float()           # keep grad
    fft_u = torch.fft.rfft(resid, dim=-1)         # [..., K]
    mag   = torch.abs(fft_u)                      # magnitude, differentiable
    K     = mag.shape[-1]
    if K > 1:
        k = torch.arange(K, device=mag.device, dtype=mag.dtype) / float(K - 1)
    else:
        k = torch.zeros(K, device=mag.device, dtype=mag.dtype)
    w = (1.0 - k) ** alpha                         # low-freq weighting
    loss = (mag * w).mean()
    return loss