"""
Adafactor optimizer implementation with factorized second moment estimation.

This optimizer is based on the Adafactor algorithm introduced by Shazeer and
Stern (2018).  Unlike Adam, it stores and updates factorized estimates of
the second moment for matrix/tensor parameters, greatly reducing memory
footprint.  For vector parameters (e.g., bias terms), it falls back to a
full-moment update.  The optimizer also includes an RMS-based update
clipping mechanism to improve stability.  All updates are performed
in-place and without gradient tracking.

Parameters:
    lr: Learning rate applied to the scaled update.
    eps1: Small constant added to second moment estimates to prevent division
        by zero.  Typical values are 1e-30 for factorized case and 1e-16 for
        full-moment case.
    eps2: Small constant used in the RMS update clipping denominator.
    beta2: Decay rate for the running average of squared gradients.
    weight_decay: Optional L2 regularization coefficient.
    clip_threshold: Maximum RMS update magnitude; updates are scaled so
        that the RMS does not exceed this threshold.

Usage:
    optimizer = Adafactor(model.parameters(), lr=1e-3)
    for input, target in data:
        loss = model(input, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class Adafactor(Optimizer):
    """Implementation of the Adafactor optimiser with factorised second moments."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        clip_threshold: float = 1.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            eps1=eps1,
            eps2=eps2,
            beta2=beta2,
            weight_decay=weight_decay,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        This method iterates over all parameter groups and their parameters,
        updating each with its own factored or full second moment estimate.
        RMS-based clipping is applied to the update to ensure stability.

        Args:
            closure: Optional closure that reevaluates the model and returns
                the loss.

        Returns:
            The loss value (if closure is provided) or None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta2 = group["beta2"]
            eps1 = group["eps1"]
            eps2 = group["eps2"]
            wd = group["weight_decay"]
            clip_th = group["clip_threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # Apply weight decay directly to the gradient
                if wd != 0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                # Factorised second moment for matrix/tensor params
                if g.ndim >= 2:
                    if "vr" not in state:
                        state["vr"] = torch.zeros(g.shape[:-1], device=g.device, dtype=g.dtype)
                        state["vc"] = torch.zeros(g.shape[-1], device=g.device, dtype=g.dtype)
                    vr, vc = state["vr"], state["vc"]
                    # Update row and column second moments
                    vr.mul_(beta2).add_(g.pow(2).mean(dim=-1), alpha=(1.0 - beta2))
                    vc.mul_(beta2).add_(g.pow(2).mean(dim=tuple(range(g.ndim - 1))), alpha=(1.0 - beta2))
                    # Construct preconditioner factors
                    r_factor = (vr + eps1).rsqrt().unsqueeze(-1)
                    c_factor = (vc + eps1).rsqrt()
                    update = g * (r_factor * c_factor)
                # Full second moment for vector params
                else:
                    if "v" not in state:
                        state["v"] = torch.zeros_like(g)
                    v = state["v"]
                    v.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2))
                    update = g * (v + eps1).rsqrt()
                # RMS clipping
                rms_update = update.pow(2).mean().sqrt()
                denom = (rms_update / (eps2 + 1e-12) + 1e-12)
                scale = (clip_th / denom).clamp(max=1.0)
                p.add_(update * scale, alpha=-lr)
        return loss
