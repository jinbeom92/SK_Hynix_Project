"""
This module defines a Near‑Expand Masked Composite Loss tailored to the HDN
training regime. It combines three terms – masked mean squared error (MSE),
masked structural similarity (SSIM), and perceptual peak signal‑to‑noise
ratio (PSNR) – weighted by user‑specified coefficients. The loss operates
on individual 2D slices of the voxel volume and uses a near‑expand mask
derived from the ground truth to focus the penalty near object boundaries.
It also exposes a small factory function to construct a loss instance from a
YAML configuration dictionary.

The near‑expand mask is computed by taking a Euclidean distance transform
(scipy.ndimage.distance_transform_edt) of the background and thresholding
voxels whose distance to the nearest foreground pixel is below `thr`. These
voxels are then assigned a value of `near_value` in the modified target.
This encourages the network to expand object boundaries slightly and reduces
oversmoothing. MSE and PSNR are computed on the masked regions, and SSIM is
computed using a differentiable Gaussian‑window convolution.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

__all__ = ["NearExpandMaskedCompositeLossV2", "build_loss_from_cfg"]

################################################################################
# Additional loss utilities
################################################################################

def _kernel8_no_center(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return a 3×3 kernel of ones with the center element set to zero.

    This helper constructs a convolution kernel that sums the 8
    neighbouring pixels of a 2D map, ignoring the center pixel.  It is
    used by ``edge_contrast_slice_max_torch`` to find boundary pixels.

    Args:
        device: The device on which to allocate the tensor.  If None, the
            current default device is used.
        dtype: Data type of the returned tensor.

    Returns:
        A tensor of shape (1, 1, 3, 3) with dtype ``dtype`` and ones
        everywhere except the center element, which is zero.
    """
    ker = torch.ones((1, 1, 3, 3), dtype=dtype, device=device)
    ker[0, 0, 1, 1] = 0.0
    return ker


@torch.enable_grad()
def edge_contrast_slice_max_torch(
    x: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "per_sample",
) -> torch.Tensor:
    """Compute a simple edge‑contrast metric for 2D slices.

    This function measures how sharply a predicted image ``x`` falls off at
    object boundaries defined by ``mask``.  The metric is computed slice‑wise
    (per sample) and can be returned either per sample, averaged over the
    batch, or with no reduction.

    Args:
        x: A tensor of shape (B, 1, H, W) containing predicted values.  It
            must have ``requires_grad=True`` to allow gradient flow.
        mask: A tensor broadcastable to (B, 1, H, W) containing binary
            foreground/background labels.  Foreground pixels have value 1.
        reduction: Reduction mode – ``"per_sample"`` returns a (B, 1)
            tensor of edge contrast values per batch element; ``"batch_mean"``
            returns a scalar; ``"none"`` returns the full contrast map and
            boundary indicator.

    Returns:
        Depending on ``reduction``: a contrast map and boundary mask (if
        ``none``), a tensor of shape (B, 1) with per‑sample values (if
        ``per_sample``), or a scalar (if ``batch_mean``).

    Raises:
        ValueError: If ``reduction`` is not one of the allowed strings.
    """
    device, dtype = x.device, x.dtype
    # Convert mask to boolean foreground, and compute its complement
    m = mask.to(device=device).bool()
    outN = (~m).float()
    # Convolve the complement to count the number of neighbouring background pixels
    ker = _kernel8_no_center(device=device, dtype=dtype)
    cnt_out = F.conv2d(outN, ker, padding=1)
    # Boundary indicator: a foreground pixel that has at least one background neighbour
    ib = (m & (cnt_out > 0))
    # Maximum of x within a 3×3 neighbourhood on background pixels
    max_out = F.max_pool2d(x * outN, kernel_size=3, stride=1, padding=1)
    contrast = (x - max_out) * ib.float()
    # No reduction: return full contrast map and boundary indicator
    if reduction == "none":
        return contrast, ib
    # Per-sample reduction: average contrast over boundary pixels
    num = ib.float().sum(dim=(2, 3), keepdim=False).clamp_min(1.0)
    ec_per = contrast.sum(dim=(2, 3), keepdim=False) / num
    if reduction == "per_sample":
        return ec_per
    elif reduction == "batch_mean":
        return ec_per.mean()
    else:
        raise ValueError("reduction must be one of: none, per_sample, batch_mean")


class NearExpandMaskedCompositeLossV2(nn.Module):
    """Composite loss using near‑expand masking, MSE, SSIM and PSNR.

    Parameters controlling each term are provided at construction time.
    The `forward` method expects prediction and ground truth tensors
    of shape [B,1,H,W] and returns a scalar loss (or per‑sample loss
    if reduction="none") along with a dictionary of detached metrics
    (mse, psnr, ssim).
    """

    def __init__(
        self,
        thr: float = 0.8,
        near_value: float = 0.8,
        spacing: Optional[Tuple[float, float]] = None,
        *,
        max_val: float = 1.0,
        # SSIM parameters
        ssim_win_size: int = 7,
        ssim_sigma: float = 1.5,
        ssim_grad: bool = True,
        # PSNR parameters
        psnr_grad: bool = True,
        psnr_ref: float = 40.0,
        # Term weights
        w_mse: float = 1.0,
        w_ssim: float = 0.5,
        w_psnr: float = 0.0,
        # Reduction and weighting
        reduction: str = "mean",
        weighted_mean_by_mask: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        assert reduction in ("mean", "none")
        self.thr = float(thr)
        self.near_value = float(near_value)
        self.spacing = None if spacing is None else tuple(map(float, spacing))
        # SSIM constants
        self.max_val = float(max_val)
        self.ssim_win_size = int(ssim_win_size)
        self.ssim_sigma = float(ssim_sigma)
        self.ssim_grad = bool(ssim_grad)
        # PSNR
        self.psnr_grad = bool(psnr_grad)
        self.psnr_ref = float(psnr_ref)
        # Weights
        self.w_mse = float(w_mse)
        self.w_ssim = float(w_ssim)
        self.w_psnr = float(w_psnr)
        # Reduction settings
        self.reduction = reduction
        self.weighted_mean_by_mask = bool(weighted_mean_by_mask)
        self.eps = float(eps)
        # SSIM normalisation constants (L2‑norm based)
        L = self.max_val
        self.C1 = (0.01 * L) ** 2
        self.C2 = (0.03 * L) ** 2

    @torch.no_grad()
    def _make_mask_and_gtmod(
        self, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct a near‑expand mask and modified target.

        For each image in the batch, compute a binary mask where the
        foreground (gt==1) and all background pixels within a Euclidean
        distance `thr` of the foreground are marked as 1.  Replace those
        background pixels in a copy of gt with `near_value`.  Also
        return per‑sample mask pixel counts for weighted reduction.

        Args:
            gt: Ground truth tensor [B,1,H,W] with values in {0,1}.

        Returns:
            mask: [B,1,H,W] float tensor where 1 indicates pixels to include.
            gt_mod: [B,1,H,W] float tensor with near pixels set to near_value.
            mask_sz: [B] tensor containing the count of included pixels.
        """
        dev = gt.device
        gt_np = gt.detach().to("cpu").numpy().astype(np.uint8)
        B, _, H, W = gt_np.shape
        masks, gmods, mask_pix = [], [], []
        for b in range(B):
            g = gt_np[b, 0]
            dist_out = edt((g == 0).astype(np.uint8), sampling=self.spacing)
            near = (g == 0) & (dist_out <= self.thr)
            mask = (g == 1) | near
            gmod = g.astype(np.float32, copy=True)
            gmod[near] = self.near_value
            masks.append(mask[None, None].astype(np.float32))
            gmods.append(gmod[None, None])
            mask_pix.append(float(mask.sum()))
        mask = torch.from_numpy(np.concatenate(masks, axis=0)).to(dev)
        gt_mod = torch.from_numpy(np.concatenate(gmods, axis=0)).to(dev)
        mask_sz = torch.tensor(mask_pix, device=dev, dtype=torch.float32)
        return mask, gt_mod, mask_sz

    def _gaussian_window(
        self, k: int, sigma: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create a 2D Gaussian window of size k×k for SSIM computation."""
        coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / (g1d.sum() + 1e-12)
        g2d = torch.outer(g1d, g1d)
        return g2d.unsqueeze(0).unsqueeze(0)

    def _ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute an SSIM map for two images via convolution."""
        k = int(self.ssim_win_size)
        pad = k // 2
        win = self._gaussian_window(k, float(self.ssim_sigma), x.device, x.dtype)
        mu_x = F.conv2d(x, win, padding=pad)
        mu_y = F.conv2d(y, win, padding=pad)
        x2 = F.conv2d(x * x, win, padding=pad)
        y2 = F.conv2d(y * y, win, padding=pad)
        xy = F.conv2d(x * y, win, padding=pad)
        sigma_x2 = x2 - mu_x * mu_x
        sigma_y2 = y2 - mu_y * mu_y
        sigma_xy = xy - mu_x * mu_y
        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x * mu_x + mu_y * mu_y + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        return num / (den + 1e-12)

    def _masked_ssim_per_sample(
        self, pred: torch.Tensor, gt_mod: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked SSIM per batch element."""
        B = pred.size(0)
        out = []
        for b in range(B):
            mb = mask[b, 0].bool()
            if not torch.any(mb):
                out.append(pred[b, 0].sum() * 0.0)
                continue
            ys, xs = torch.where(mb)
            y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
            x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
            xb = pred[b : b + 1, 0:1, y0:y1, x0:x1]
            yb = gt_mod[b : b + 1, 0:1, y0:y1, x0:x1]
            mm = mb[y0:y1, x0:x1]
            ssim_map = self._ssim_map(xb, yb)
            out.append(ssim_map[0, 0][mm].mean())
        return torch.stack(out, dim=0)

    def _psnr_from_mse(self, mse_b: torch.Tensor) -> torch.Tensor:
        """Compute PSNR from per‑sample MSE (dB)."""
        return 10.0 * torch.log10((self.max_val ** 2) / torch.clamp(mse_b, min=self.eps))

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the composite loss and return metrics.

        Args:
            pred: Predicted slices [B,1,H,W] clamped to [0,1] outside.
            gt: Ground‑truth slices [B,1,H,W] with binary values.

        Returns:
            loss: A scalar loss (or per‑sample if reduction="none").
            info: A dictionary with detached mse, psnr, ssim metrics.

        Raises:
            ValueError: If the shapes of pred and gt are incompatible.
        """
        if not (pred.shape == gt.shape and pred.dim() == 4 and pred.size(1) == 1):
            raise ValueError(
                f"expected pred,gt ∈ [B,1,H,W], got pred={tuple(pred.shape)} gt={tuple(gt.shape)}"
            )
        with torch.no_grad():
            mask, gt_mod, mask_sz = self._make_mask_and_gtmod(gt)
        mask = mask.to(dtype=pred.dtype, device=pred.device)
        gt_mod = gt_mod.to(dtype=pred.dtype, device=pred.device)
        B = pred.size(0)
        mse_b = []
        for b in range(B):
            m = mask[b, 0]
            denom = torch.clamp(m.sum(), min=self.eps)
            d2 = (pred[b, 0] - gt_mod[b, 0]) ** 2
            mse_b.append((d2 * m).sum() / denom)
        mse_b = torch.stack(mse_b, dim=0)
        psnr_b = None
        if self.w_psnr != 0.0:
            if self.psnr_grad:
                psnr_b = self._psnr_from_mse(mse_b)
            else:
                with torch.no_grad():
                    psnr_b = self._psnr_from_mse(mse_b.detach())
        ssim_b = None
        if self.w_ssim != 0.0:
            if self.ssim_grad:
                ssim_b = self._masked_ssim_per_sample(pred, gt_mod, mask)
            else:
                with torch.no_grad():
                    ssim_b = self._masked_ssim_per_sample(pred.detach(), gt_mod, mask)
        loss_b = self.w_mse * mse_b
        if ssim_b is not None:
            loss_b = loss_b + self.w_ssim * (1.0 - torch.clamp(ssim_b, min=-1.0, max=1.0))
        if (psnr_b is not None) and (self.psnr_ref > 0):
            ps = 1.0 - (psnr_b / self.psnr_ref)
            loss_b = loss_b + self.w_psnr * torch.clamp(ps, min=0.0)
        if self.reduction == "none":
            loss = loss_b
        else:
            if self.weighted_mean_by_mask and torch.isfinite(mask_sz).all() and (mask_sz.sum() > 0):
                w = (mask_sz / torch.clamp(mask_sz.sum(), min=self.eps)).to(loss_b.device)
                loss = (loss_b * w).sum()
            else:
                loss = loss_b.mean()
        def _reduce_detached(v: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if v is None:
                return None
            vd = v.detach()
            if self.reduction == "none":
                return vd
            if self.weighted_mean_by_mask and (mask_sz.sum() > 0):
                w = (mask_sz / torch.clamp(mask_sz.sum(), min=self.eps)).to(vd.device)
                return (vd * w).sum()
            return vd.mean()
        info = {"mse": _reduce_detached(mse_b)}
        if psnr_b is not None:
            info["psnr"] = _reduce_detached(psnr_b)
        if ssim_b is not None:
            info["ssim"] = _reduce_detached(ssim_b)
        return loss, info


def build_loss_from_cfg(cfg: Dict) -> NearExpandMaskedCompositeLossV2:
    """Instantiate a composite loss from a nested configuration dictionary.

    The factory reads the `losses` section of the provided config and
    forwards recognised parameters to `NearExpandMaskedCompositeLossV2`.

    Args:
        cfg: Top‑level configuration dict, typically loaded from YAML.

    Returns:
        An initialised `NearExpandMaskedCompositeLossV2` instance.
    """
    l = cfg.get("losses", {}) if cfg is not None else {}
    return NearExpandMaskedCompositeLossV2(
        thr=float(l.get("expand_thr", l.get("thr", 0.8))),
        near_value=float(l.get("expand_near_value", l.get("near_value", 0.8))),
        spacing=l.get("expand_spacing", l.get("spacing", None)),
        max_val=float(l.get("max_val", 1.0)),
        ssim_win_size=int(l.get("ssim_win_size", 7)),
        ssim_sigma=float(l.get("ssim_sigma", 1.5)),
        ssim_grad=bool(l.get("ssim_grad", True)),
        psnr_grad=bool(l.get("psnr_grad", True)),
        psnr_ref=float(l.get("psnr_ref", 40.0)),
        w_mse=float(l.get("w_mse", 1.0)),
        w_ssim=float(l.get("w_ssim", 0.5)),
        w_psnr=float(l.get("w_psnr", 0.0)),
        reduction=str(l.get("reduction", "mean")),
        weighted_mean_by_mask=bool(l.get("weighted_mean_by_mask", True)),
        eps=float(l.get("eps", 1e-8)),
    )
