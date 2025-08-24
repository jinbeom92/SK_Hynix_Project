import torch
import torch.nn.functional as F
from utils.metrics import psnr, band_penalty, energy_penalty, voxel_error_rate, in_positive_mask_dynamic_range
from utils.ssim3d import ssim3d

def _to_b1xyz(x: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize a volume to channel-first **[B, 1, X, Y, Z]** without
    changing axis order (x,y,z). Accepted shapes:
      - [B, 1, X, Y, Z] → returned as-is
      - [B, X, Y, Z]    → unsqueeze channel → [B, 1, X, Y, Z]
    """
    if x.ndim == 5:
        if x.shape[1] != 1:
            raise ValueError(f"Expected channel dimension C=1, got shape {tuple(x.shape)}.")
        return x
    if x.ndim == 4:
        return x.unsqueeze(1)
    raise ValueError(f"Expected [B,1,X,Y,Z] or [B,X,Y,Z], got {tuple(x.shape)}.")


def _ssim_safe(R_xyz: torch.Tensor, V_xyz: torch.Tensor) -> torch.Tensor:
    """
    SSIM-based loss aligned to **(x,y,z)** volumes.

    - If Z==1 → approximate with 2D SSIM on the single slice [B,1,X,Y].
    - Else    → call `ssim3d` on [B,1,Z,X,Y] (permute only for SSIM).

    Returns
    -------
    Tensor
        Loss tensor shaped like [B,1,1,1,1] (kept dims for easy broadcasting).
        Convention: **1 - SSIM** for 2D path, while `ssim3d` is assumed to
        already return the SSIM-style loss as in the previous setup.
    """
    R = _to_b1xyz(R_xyz)  # [B,1,X,Y,Z]
    V = _to_b1xyz(V_xyz)
    Z = R.shape[-1]

    if Z == 1:
        # 2D fallback on [B,1,X,Y]
        r2 = R[..., 0]  # [B,1,X,Y]
        v2 = V[..., 0]
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
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sig2_x + sig2_y + C2) + 1e-12
        )
        loss = 1.0 - ssim_map
        return loss.mean(dim=list(range(1, loss.ndim)), keepdim=True)  # [B,1,1,1,1]

    # 3D SSIM on [B,1,Z,X,Y]
    R_dhw = R.permute(0, 1, 4, 2, 3).contiguous()
    V_dhw = V.permute(0, 1, 4, 2, 3).contiguous()
    return ssim3d(R_dhw, V_dhw)  # expected to preserve reduced/broadcastable dims


def tv_isotropic_3d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Isotropic total variation on **(x,y,z)** volumes.

    Input
    -----
    x : Tensor
        Volume in shape **[B, 1, X, Y, Z]** (use `_to_b1xyz` before calling).

    Implementation detail
    ---------------------
    Use forward differences and compute TV on the common cropped region
    so that the three directional gradients share the same shape:
        common = x[:, :, 1:, 1:, 1:]  → [B,1,X-1,Y-1,Z-1]
        dx = common - x[:, :, :-1, 1:, 1:]
        dy = common - x[:, :, 1:, :-1, 1:]
        dz = common - x[:, :, 1:, 1:, :-1]
    """
    if x.ndim != 5 or x.shape[1] != 1:
        raise ValueError("tv_isotropic_3d expects [B,1,X,Y,Z].")

    common = x[:, :, 1:, 1:, 1:]
    dx = common - x[:, :, :-1, 1:, 1:]
    dy = common - x[:, :, 1:, :-1, 1:]
    dz = common - x[:, :, 1:, 1:, :-1]

    tv = torch.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + eps)
    return tv.mean(dim=list(range(1, tv.ndim)), keepdim=True)  # [B,1,1,1,1]


def reconstruction_losses(R_hat_n: torch.Tensor,
                          V_gt_n: torch.Tensor,
                          weights: dict,
                          params: dict):
    """
    Aggregate reconstruction-domain losses for **(x,y,z)** volumes.

    Inputs
    ------
    R_hat_n : Tensor
        Predicted volume, shape **[B, X, Y, Z]** or **[B, 1, X, Y, Z]**.
    V_gt_n : Tensor
        Ground-truth volume, same shape conventions as `R_hat_n`.
    weights : dict
        Loss weights, e.g. {"ssim": a, "psnr": b, "band": c, ...}.
        If key == "psnr", we use **psnr_loss** (normalized) instead of raw PSNR.
    params : dict
        Hyper-parameters, must include:
          - "band_low", "band_high"
          - "ver_thr"
          - optional "psnr_ref" (default 40.0)
          - optional "tv_weight" (default 0.0)

    Returns
    -------
    dict
        {
          "ssim":        ...,
          "psnr":        ...,
          "psnr_loss":   ...,
          "band":        ...,
          "energy":      ...,
          "ver":         ...,
          "ipdr":        ...,
          "tv":          ...,
          "total_recon": ...
        }
        Scalar-like tensors are kept broadcastable with shape [B,1,1,1,1].
    """
    loss_dict = {}

    # Canonicalize to [B,1,X,Y,Z] (no axis permutation)
    R = _to_b1xyz(R_hat_n)
    V = _to_b1xyz(V_gt_n)

    # 1 - SSIM (3D via ssim3d, 2D fallback if Z==1)
    loss_dict["ssim"] = _ssim_safe(R, V)

    # PSNR (log-only) using clamped volumes in [0,1]
    psnr_db = psnr(torch.clamp(R, 0, 1), torch.clamp(V, 0, 1))
    loss_dict["psnr"] = psnr_db

    # Normalized PSNR loss (higher dB → lower loss)
    p_ref = float(params.get("psnr_ref", 40.0))
    psnr_loss = torch.clamp((p_ref - psnr_db) / p_ref, min=0.0, max=1.0)
    loss_dict["psnr_loss"] = psnr_loss

    # Other penalties (value-based; axis-order agnostic)
    loss_dict["band"]   = band_penalty(R, params["band_low"], params["band_high"])
    loss_dict["energy"] = energy_penalty(R, V)
    loss_dict["ver"]    = voxel_error_rate(R, V, params["ver_thr"])
    loss_dict["ipdr"]   = in_positive_mask_dynamic_range(R, params["ver_thr"])

    # TV (optional) on [B,1,X,Y,Z]
    tv_w = float(params.get("tv_weight", 0.0))
    if tv_w > 0:
        loss_dict["tv"] = tv_isotropic_3d(R) * tv_w
    else:
        # keep broadcastable shape
        loss_dict["tv"] = torch.zeros_like(loss_dict["ssim"])

    # Total loss: use psnr_loss instead of raw psnr
    total = torch.zeros_like(loss_dict["ssim"])
    for k, w in weights.items():
        if k == "psnr":
            total = total + float(w) * loss_dict.get("psnr_loss", 0.0)
        elif k in loss_dict:
            total = total + float(w) * loss_dict[k]
    loss_dict["total_recon"] = total

    return loss_dict
