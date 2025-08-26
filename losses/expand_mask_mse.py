import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt as edt


class ExpandMaskedMSE(nn.Module):
    def __init__(
        self,
        thr: float = 0.8,
        spacing=None,
        include_in_part: bool = True,
        in_value: float = 1.0,
        boundary_low: float = 0.8,
        boundary_high: float = 0.9,
        clamp_pred_gt: bool = True,
    ):
        super().__init__()
        self.thr = float(thr)
        self.spacing = None if spacing is None else tuple(spacing)
        self.include_in_part = bool(include_in_part)
        self.in_value = float(in_value)
        self.boundary_low = float(boundary_low)
        self.boundary_high = float(boundary_high)
        self.clamp_pred_gt = bool(clamp_pred_gt)

        if not (0.0 <= self.boundary_low <= self.boundary_high <= 1.0):
            raise ValueError("Require 0 ≤ boundary_low ≤ boundary_high ≤ 1.")

    @torch.no_grad()
    def _mask_and_soft_target_2d(self, gt: torch.Tensor):
        dev = gt.device
        B, C, H, W = gt.shape
        if C != 1:
            raise ValueError(f"Expected channel=1, got {list(gt.shape)}")
        g_np = gt.detach().cpu().numpy().astype(np.uint8)  # [B,1,H,W]

        masks = []
        targets = []
        thr = float(self.thr)
        lo, hi = float(self.boundary_low), float(self.boundary_high)
        for b in range(B):
            g = g_np[b, 0]  # [H,W], values in {0,1}

            # Distance in the complement (out-of-part pixels only)
            dist_out = edt((g == 0).astype(np.uint8), sampling=self.spacing)

            pos = (g == 1)
            bnd = ((g == 0) & (dist_out <= thr))

            # Mask: include in‑part (optional) + boundary
            m = (bnd | (pos if self.include_in_part else False)).astype(np.float32)

            # Soft targets
            t = np.zeros_like(g, dtype=np.float32)
            if self.include_in_part:
                t[pos] = self.in_value

            # Linear decay from boundary_high (d→0) to boundary_low (d==thr)
            if thr > 0:
                w = np.clip(dist_out / thr, 0.0, 1.0)
                t_bnd = hi - (hi - lo) * w
            else:
                t_bnd = np.full_like(dist_out, fill_value=(lo + hi) * 0.5, dtype=np.float32)
            t[bnd] = t_bnd[bnd].astype(np.float32)

            masks.append(m[None, None])    # [1,1,H,W]
            targets.append(t[None, None])  # [1,1,H,W]

        mask = torch.from_numpy(np.concatenate(masks, axis=0)).to(dev)
        target = torch.from_numpy(np.concatenate(targets, axis=0)).to(dev)
        return mask, target

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if pred.shape != gt.shape or pred.dim() != 4 or pred.size(1) != 1:
            raise ValueError(f"Expected [B,1,H,W] for both pred and gt, got {list(pred.shape)} vs {list(gt.shape)}")

        if self.clamp_pred_gt:
            pred = pred.clamp(0.0, 1.0)
            gt = gt.clamp(0.0, 1.0)

        mask, soft_tgt = self._mask_and_soft_target_2d(gt)
        mask = mask.to(dtype=pred.dtype, device=pred.device)
        soft_tgt = soft_tgt.to(dtype=pred.dtype, device=pred.device)

        denom = mask.sum()
        if denom.item() < eps:
            return pred.new_tensor(0.0, requires_grad=True)

        diff2 = (pred - soft_tgt) ** 2
        return (diff2 * mask).sum() / denom