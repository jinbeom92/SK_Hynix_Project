# =================================================================================================
# Adafactor Optimizer — Memory-Efficient Adaptive Optimization (ICML 2018)
# -------------------------------------------------------------------------------------------------
# Purpose
#   Implements an Adafactor-style optimizer that achieves sublinear memory usage by factoring
#   the second-moment statistics along rows and columns for rank≥2 tensors, while falling back
#   to an unfactored variant for vectors. This substantially reduces optimizer state compared to
#   Adam/AdamW and is well-suited for large models where optimizer memory dominates.
#
# Method (high level)
#   • Matrix/ND params (ndim≥2): maintain EMA of squared gradients as two tensors:
#       - vr: row-wise averages (…, out) and vc: column-wise averages (in)
#     The per-parameter RMS is approximated via (vr ⊗ vc)^(−1/2), yielding O(n+m) memory.
#   • Vector params (ndim=1): keep a standard (unfactored) second-moment accumulator v.
#   • Trust ratio style clipping via a global update RMS and `clip_threshold` for stability.
#   • Optional weight decay applied as additive term to gradients.
#
# Usage
#   opt = Adafactor(model.parameters(), lr=1e-3, weight_decay=0.0, clip_threshold=1.0)
#   loss.backward(); opt.step(); opt.zero_grad()
#
# Notes
#   • This minimal implementation focuses on factored second moments and update clipping.
#   • For very large tensors, the factored approximation dramatically lowers state memory.
#   • Keep learning rate scheduling external as needed for your training regime.
# =================================================================================================
import torch
from torch.optim import Optimizer

class Adafactor(Optimizer):
    def __init__(self, params, lr=1e-3, eps1=1e-30, eps2=1e-3,
                 beta2=0.999, weight_decay=0.0, clip_threshold=1.0):
        defaults = dict(lr=lr, eps1=eps1, eps2=eps2,
                        beta2=beta2, weight_decay=weight_decay,
                        clip_threshold=clip_threshold)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]; beta2 = group["beta2"]
            eps1 = group["eps1"]; eps2 = group["eps2"]
            wd = group["weight_decay"]; clip_th = group["clip_threshold"]

            for p in group["params"]:
                if p.grad is None: 
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]

                if g.ndim >= 2:
                    if "vr" not in state:
                        state["vr"] = torch.zeros(g.shape[:-1], device=g.device, dtype=g.dtype)
                        state["vc"] = torch.zeros(g.shape[-1], device=g.device, dtype=g.dtype)
                    vr, vc = state["vr"], state["vc"]
                    vr.mul_(beta2).add_(g.pow(2).mean(dim=-1), alpha=(1 - beta2))
                    vc.mul_(beta2).add_(g.pow(2).mean(dim=tuple(range(g.ndim - 1))), alpha=(1 - beta2))
                    r_factor = (vr + eps1).rsqrt().unsqueeze(-1)
                    c_factor = (vc + eps1).rsqrt()
                    update = g * (r_factor * c_factor)
                else:
                    if "v" not in state:
                        state["v"] = torch.zeros_like(g)
                    v = state["v"]
                    v.mul_(beta2).addcmul_(g, g, value=(1 - beta2))
                    update = g * (v + eps1).rsqrt()

                rms_update = update.pow(2).mean().sqrt()
                denom = (rms_update / (eps2 + 1e-12) + 1e-12)
                scale = (clip_th / denom).clamp(max=1.0)
                p.add_(update * scale, alpha=-lr)

        return loss
