import torch

def per_sample_sino_scale(
    S_in: torch.Tensor,
    q: float = 0.99,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute a per-sample robust scale for sinograms via high-quantile norm.

    Purpose
    -------
    Normalizes each sample by a robust magnitude estimate so that downstream losses
    operate in a comparable range across batches. Uses the q-quantile of |S_in|.

    Shapes
    ------
    S_in : [B, A, V, U]   (any dtype; typically float32/float16)
    return: [B, 1, 1, 1]  (broadcastable per-sample scale; requires_grad=False)

    Args
    ----
    q   : float
          Quantile in (0,1). Recommended 0.95~0.995. Higher → less sensitive to outliers.
    eps : float
          Lower clamp to avoid division by zero.

    Notes
    -----
    • `torch.no_grad()` ensures the scale is not part of the gradient graph.
    • Use this with broadcast division: x / s where s has shape [B,1,1,1].
    """
    with torch.no_grad():
        B = S_in.shape[0]
        # Flatten per sample, take quantile of absolute values
        s = torch.quantile(S_in.abs().reshape(B, -1), q, dim=1)
        s = s.clamp_min(eps).view(B, 1, 1, 1)  # [B,1,1,1]
    return s  # requires_grad=False


def normalize_tensors(
    S_in: torch.Tensor,
    sino_hat: torch.Tensor,
    R_hat: torch.Tensor,
    V_gt: torch.Tensor,
    s_sino: torch.Tensor
):
    """
    Normalize sinograms and volumes using per-sample sinogram scales.

    Purpose
    -------
    Ensures both detector-domain tensors (S_in, sino_hat) and volume-domain tensors
    (R_hat, V_gt) are put on a consistent normalized scale per sample.

    Shapes
    ------
    S_in, sino_hat : [B, A, V, U]
    R_hat, V_gt    : [B, 1, D, H, W]
    s_sino         : [B, 1, 1, 1]  (output from per_sample_sino_scale)
    returns        : (S_in_n, sino_hat_n, R_hat_n, V_gt_n)
                     with same shapes as inputs

    Notes
    -----
    • Volume scaling uses the same per-sample scale as sinograms by broadcasting
      to [B,1,1,1,1], which ties both domains to the same intensity reference.
    • Assumes input arrays are non-negative or loosely in [0, +], but will work
      with signed data as well (scale uses |S_in| quantile).
    """
    # Detector-domain normalization
    S_in_n     = S_in     / s_sino                 # [B,A,V,U]
    sino_hat_n = sino_hat / s_sino                 # [B,A,V,U]

    # Volume-domain normalization (broadcast the same per-sample scale)
    s_vol   = s_sino.view(-1, 1, 1, 1, 1)          # [B,1,1,1,1]
    R_hat_n = R_hat / s_vol                        # [B,1,D,H,W]
    V_gt_n  = V_gt  / s_vol                        # [B,1,D,H,W]

    return S_in_n, sino_hat_n, R_hat_n, V_gt_n
