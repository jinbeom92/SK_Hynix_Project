import torch

def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (in dB).

    Assumptions
    -----------
    • Inputs are roughly normalized to [0, 1].

    Args
    ----
    x, y : Tensor
        Predictions and targets with shape [B, ...].
    eps : float
        Numerical stabilizer to avoid log(0).

    Returns
    -------
    Tensor
        PSNR in dB, reduced over all non-batch dims and kept as a broadcastable
        per-batch scalar: shape [B, 1, 1, ..., 1].
    """
    # Mean-squared error over all non-batch dims
    mse = torch.mean((x - y) ** 2, dim=list(range(1, x.ndim)), keepdim=True)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def voxel_error_rate(x: torch.Tensor, y: torch.Tensor, thr: float) -> torch.Tensor:
    """
    Voxel Error Rate (binary occupancy mismatch above a threshold).

    Args
    ----
    x, y : Tensor
        Predictions and targets, shape [B, ...].
    thr : float
        Occupancy threshold; values >= thr are considered 'positive'.

    Returns
    -------
    Tensor
        Mean fraction of mismatched voxels per batch, broadcastable scalar
        with shape [B, 1, 1, ..., 1].
    """
    x_b = (x >= thr).to(x.dtype)
    y_b = (y >= thr).to(y.dtype)
    wrong = (x_b != y_b).to(x.dtype)
    return wrong.mean(dim=list(range(1, wrong.ndim)), keepdim=True)


def in_positive_mask_dynamic_range(x: torch.Tensor, thr: float, eps: float = 1e-8) -> torch.Tensor:
    """
    In-Positive Dynamic Range (IPDR).

    Definition
    ----------
    Within the set {x >= thr}, compute:
        (max - min) / (mean + eps)

    Args
    ----
    x : Tensor
        Input volume, shape [B, ...].
    thr : float
        Threshold to define the positive mask.
    eps : float
        Numerical stabilizer.

    Returns
    -------
    Tensor
        Broadcastable per-batch scalar with shape [B, 1, 1, ..., 1].
        Returns zeros if no voxel passes the threshold.
    """
    mask = (x >= thr)
    if mask.sum() == 0:
        return torch.zeros_like(x.mean(dim=list(range(1, x.ndim)), keepdim=True))
    vals = x[mask]
    dyn = (vals.max() - vals.min()) / (vals.mean() + eps)
    # Expand to [B, 1, ..., 1] so it’s broadcastable like other metrics
    return dyn.expand(x.shape[0], *([1] * (x.ndim - 1)))


def band_penalty(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """
    Intensity band penalty for values outside [low, high].

    Args
    ----
    x : Tensor
        Input volume, shape [B, ...].
    low, high : float
        Valid intensity band.

    Returns
    -------
    Tensor
        Mean ReLU-penalized deviation outside the band, per batch:
        shape [B, 1, 1, ..., 1].
    """
    below = torch.relu(low - x)
    above = torch.relu(x - high)
    return (below + above).mean(dim=list(range(1, x.ndim)), keepdim=True)


def energy_penalty(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Relative squared error of total energy (sum of intensities).

    Args
    ----
    pred, gt : Tensor
        Predicted and ground-truth volumes, shape [B, ...].
    eps : float
        Numerical stabilizer in the denominator.

    Returns
    -------
    Tensor
        Per-batch relative energy error, broadcastable scalar:
        shape [B, 1, 1, ..., 1].
    """
    s_pred = pred.sum(dim=list(range(1, pred.ndim)), keepdim=True)
    s_gt = gt.sum(dim=list(range(1, gt.ndim)), keepdim=True)
    return ((s_pred - s_gt) ** 2) / (s_gt ** 2 + eps)
