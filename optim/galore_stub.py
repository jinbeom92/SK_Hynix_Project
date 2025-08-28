"""
Gradient projection utility for low‑rank approximation.

During optimisation, projecting gradients onto a lower‑rank subspace can
reduce noise and memory usage.  This function operates in-place on the
gradients of parameters with at least two dimensions, reshaping each
gradient into a matrix [out_features, in_features], computing an SVD, and
reconstructing the gradient using only the top `rank` singular values and
vectors.  Layers with fewer than two dimensions (e.g., biases) are skipped.

Arguments:
    model: A PyTorch module whose parameters may carry gradients.
    rank: The maximum rank to retain in the SVD approximation.  A value
        of zero or negative disables projection.
    every: Only project every `every` steps (use with a step counter).
    step: Current training step (1-based).  Projection occurs if
        `step % every == 0`.
    max_svd_dim: If a gradient matrix is larger than this threshold in
        either dimension, it is truncated to `max_svd_dim` along the
        larger dimension before computing the SVD.  This controls
        computational cost.

Notes:
    The operation is performed under `torch.no_grad()` to avoid tracking
    the SVD in the autograd graph.  Any SVD failure is silently ignored.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def project_grads_(
    model: torch.nn.Module,
    rank: int = 8,
    every: int = 1,
    step: int = 1,
    max_svd_dim: int = 4096,
) -> None:
    """Project gradients of eligible parameters onto a low‑rank subspace.

    This function iterates over all parameters in the model, identifies
    those with at least two dimensions (weight matrices or convolutional
    filters), reshapes their gradients into two‑dimensional matrices, and
    performs an SVD.  The gradient is then approximated by keeping only
    the top `rank` singular values/vectors.  Projection is skipped
    if `every <= 0` or `step` is not a multiple of `every`, or if
    `rank <= 0`.  Gradients on 1‑D tensors (e.g., biases) remain untouched.

    Args:
        model: PyTorch module whose parameter gradients will be modified in-place.
        rank: Maximum number of singular values to keep (low‑rank rank).
        every: Perform projection every `every` steps; set to 1 for each call.
        step: Current global step count.
        max_svd_dim: Maximum dimension along which to compute SVD; larger
            matrices are truncated to this size along the longer dimension.

    Returns:
        None.  Gradients are modified in-place.
    """
    if (every <= 0) or (step % every != 0):
        return
    for p in model.parameters():
        if (p.grad is None) or (p.grad.ndim < 2):
            continue
        g = p.grad.data
        # Flatten all but first dimension: [out_channels, -1]
        m = g.reshape(g.shape[0], -1)
        if min(m.shape) < 2 or rank <= 0:
            continue
        # Truncate to reduce SVD cost
        if max(m.shape) > max_svd_dim:
            if m.shape[1] >= m.shape[0]:
                m = m[:, :max_svd_dim]
            else:
                m = m[:max_svd_dim, :]
        # Compute truncated SVD and reconstruct the gradient
        try:
            U, S, Vh = torch.linalg.svd(m, full_matrices=False)
            r = min(rank, S.numel())
            m_approx = (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :])
            g.copy_(m_approx.view_as(g))
        except Exception:
            # If SVD fails, leave gradient unchanged
            pass
