import torch
from torch.optim import Optimizer


class Adafactor(Optimizer):
    """
    Adafactor Optimizer (memory-efficient adaptive optimization).

    Purpose
    -------
    • Implements a variant of Adam/Adagrad that factorizes the second-moment
      statistics across rows/columns for matrix-shaped parameters, reducing memory use.
    • Suitable for large models where Adam's full-matrix second moment is expensive.

    Args
    ----
    params : iterable
        Model parameters to optimize.
    lr : float (default=1e-3)
        Learning rate.
    eps1 : float (default=1e-30)
        Small constant added inside rsqrt for numerical stability (second-moment factorization).
    eps2 : float (default=1e-3)
        Constant used in the update clipping denominator.
    beta2 : float (default=0.999)
        Exponential decay for second-moment estimates.
    weight_decay : float (default=0.0)
        Weight decay coefficient (L2 regularization).
    clip_threshold : float (default=1.0)
        Threshold for adaptive update clipping (relative to RMS).

    State
    -----
    For each parameter `p`:
      • if p.ndim >= 2:
          vr : row-wise second-moment estimate (shape = p.shape[:-1])
          vc : column-wise second-moment estimate (shape = p.shape[-1])
      • else:
          v  : full second-moment estimate (same shape as p)

    Notes
    -----
    • Factorized variance storage reduces memory from O(n*m) to O(n+m) for matrices.
    • Update is scaled to have controlled RMS magnitude, improving training stability.
    • This is a simplified implementation; it does not support learning-rate warmup
      or relative step size scheduling found in the original paper.
    """

    def __init__(self, params, lr=1e-3, eps1=1e-30, eps2=1e-3,
                 beta2=0.999, weight_decay=0.0, clip_threshold=1.0):
        defaults = dict(lr=lr, eps1=eps1, eps2=eps2,
                        beta2=beta2, weight_decay=weight_decay,
                        clip_threshold=clip_threshold)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model
                                          and returns the loss.

        Returns:
            loss (float or Tensor): Loss from closure if provided, else None.
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

                # Apply weight decay (L2 regularization)
                if wd != 0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]

                # --------------------------
                # Factorized variance (matrix/tensor params)
                # --------------------------
                if g.ndim >= 2:
                    if "vr" not in state:
                        state["vr"] = torch.zeros(g.shape[:-1], device=g.device, dtype=g.dtype)
                        state["vc"] = torch.zeros(g.shape[-1], device=g.device, dtype=g.dtype)
                    vr, vc = state["vr"], state["vc"]

                    # Row/column variance estimates
                    vr.mul_(beta2).add_(g.pow(2).mean(dim=-1), alpha=(1 - beta2))
                    vc.mul_(beta2).add_(g.pow(2).mean(dim=tuple(range(g.ndim - 1))), alpha=(1 - beta2))

                    # Factorized inverse root-mean-square scaling
                    r_factor = (vr + eps1).rsqrt().unsqueeze(-1)  # row factor
                    c_factor = (vc + eps1).rsqrt()               # col factor
                    update = g * (r_factor * c_factor)

                # --------------------------
                # Fallback: full variance (vector params)
                # --------------------------
                else:
                    if "v" not in state:
                        state["v"] = torch.zeros_like(g)
                    v = state["v"]

                    # Exponential moving average of squared gradients
                    v.mul_(beta2).addcmul_(g, g, value=(1 - beta2))
                    update = g * (v + eps1).rsqrt()

                # --------------------------
                # RMS-based clipping
                # --------------------------
                rms_update = update.pow(2).mean().sqrt()
                denom = (rms_update / (eps2 + 1e-12) + 1e-12)
                scale = (clip_th / denom).clamp(max=1.0)

                # Parameter update
                p.add_(update * scale, alpha=-lr)

        return loss
