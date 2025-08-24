import torch

def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8, data_range: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (in dB).

    Assumptions
    -----------
    • Inputs are roughly normalized to [0, 1] (default `data_range=1.0`).

    Parameters
    ----------
    x, y : Tensor
        Predictions and targets with shape [B, ...].
    eps : float
        Numerical stabilizer to avoid log(0).
    data_range : float
        Peak-to-peak value of the signal. For [0,1] data use 1.0.

    Returns
    -------
    Tensor
        PSNR in dB, reduced over all non-batch dims and kept as a broadcastable
        per-batch scalar: shape [B, 1, 1, ..., 1].
    """
    reduce_dims = list(range(1, x.ndim))
    mse = torch.mean((x - y) ** 2, dim=reduce_dims, keepdim=True)
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def voxel_error_rate(x: torch.Tensor, y: torch.Tensor, thr: float) -> torch.Tensor:
    """
    Voxel Error Rate (binary occupancy mismatch above a threshold).

    Parameters
    ----------
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
    reduce_dims = list(range(1, x.ndim))
    x_b = (x >= thr)
    y_b = (y >= thr)
    wrong = (x_b != y_b).to(x.dtype)
    return wrong.mean(dim=reduce_dims, keepdim=True)


def in_positive_mask_dynamic_range(x: torch.Tensor, thr: float, eps: float = 1e-8) -> torch.Tensor:
    """
    In-Positive Dynamic Range (IPDR), computed **per batch**.

    Definition
    ----------
    Within the set {x >= thr}, compute:
        (max - min) / (mean + eps)

    Parameters
    ----------
    x : Tensor
        Input volume, shape [B, ...].
    thr : float
        Threshold to define the positive mask.
    eps : float
        Numerical stabilizer for the mean.

    Returns
    -------
    Tensor
        Broadcastable per-batch scalar with shape [B, 1, 1, ..., 1].
        Returns zeros for samples where no voxel passes the threshold.
    """
    B = x.shape[0]
    reduce_dims = tuple(range(1, x.ndim))

    mask = (x >= thr)
    # Count positives per batch
    cnt = mask.reshape(B, -1).sum(dim=1)  # [B]

    # Per-batch mean over positives
    pos_sum = (x * mask.to(x.dtype)).reshape(B, -1).sum(dim=1)                # [B]
    pos_mean = pos_sum / (cnt.to(x.dtype) + eps)                               # [B]

    # Per-batch max/min over positives using +/-inf sentinels
    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)

    x_for_min = torch.where(mask, x, inf)
    x_for_max = torch.where(mask, x, neg_inf)

    pos_min = x_for_min.reshape(B, -1).min(dim=1).values                       # [B]
    pos_max = x_for_max.reshape(B, -1).max(dim=1).values                       # [B]

    dyn = (pos_max - pos_min) / (pos_mean + eps)                               # [B]

    # For batches with cnt == 0, set dyn = 0
    dyn = torch.where(cnt > 0, dyn, torch.zeros_like(dyn))

    # Expand to [B, 1, ..., 1]
    return dyn.view(B, *([1] * (x.ndim - 1)))


def band_penalty(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """
    Intensity band penalty for values outside [low, high].

    Parameters
    ----------
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
    reduce_dims = list(range(1, x.ndim))
    below = torch.relu(low - x)
    above = torch.relu(x - high)
    return (below + above).mean(dim=reduce_dims, keepdim=True)


def energy_penalty(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Relative squared error of total energy (sum of intensities), per batch.

    Parameters
    ----------
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
    reduce_dims = list(range(1, pred.ndim))
    s_pred = pred.sum(dim=reduce_dims, keepdim=True)
    s_gt = gt.sum(dim=reduce_dims, keepdim=True)
    return ((s_pred - s_gt) ** 2) / (s_gt ** 2 + eps)
