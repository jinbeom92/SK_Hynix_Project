# =================================================================================================
# project_grads_ — Lightweight GaLore-Style Low-Rank Gradient Projection
# -------------------------------------------------------------------------------------------------
# Purpose
#   Applies a truncated SVD-based low-rank approximation to parameter gradients in-place,
#   inspired by GaLore. By projecting gradients onto a rank‑r subspace, this routine can
#   reduce the effective gradient dimensionality and, when combined with memory‑efficient
#   optimizers (e.g., Adafactor), help curb optimizer/state memory growth.
#
# Behavior
#   • Operates on parameters with ndim ≥ 2 only (matrix/conv kernels).
#   • Reshapes grad to [out, in*kh*kw], runs economy SVD, and keeps top‑r components.
#   • For very large tensors, limits the SVD dimension to `max_svd_dim` for practicality.
#   • Executes at user-specified intervals (`every`, `step`) to control overhead.
#
# Usage
#   Inside the training loop (after backward, before optimizer.step()):
#       project_grads_(model, rank=8, every=10, step=global_step)
#
# Notes
#   • This is a lightweight stub, not a complete GaLore implementation.
#   • SVD may incur nontrivial compute; tune `every`, `rank`, and `max_svd_dim`.
#   • Numerical issues during SVD are caught and safely skipped.
# =================================================================================================
import torch

@torch.no_grad()
def project_grads_(model, rank: int = 8, every: int = 1, step: int = 1, max_svd_dim: int = 4096):
    if (every <= 0) or (step % every != 0):
        return
    for p in model.parameters():
        if (p.grad is None) or (p.grad.ndim < 2):
            continue
        g = p.grad.data
        m = g.reshape(g.shape[0], -1)  # [out, in*kh*kw]
        if min(m.shape) < 2 or rank <= 0:
            continue
        if max(m.shape) > max_svd_dim:
            m = m[:, :max_svd_dim] if m.shape[1] >= m.shape[0] else m[:max_svd_dim, :]
        try:
            U, S, Vh = torch.linalg.svd(m, full_matrices=False)
            r = min(rank, S.numel())
            m_approx = (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :])
            g.copy_(m_approx.view_as(g))
        except Exception:
            pass
