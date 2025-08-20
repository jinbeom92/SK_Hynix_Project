import torch

@torch.no_grad()
def project_grads_(
    model,
    rank: int = 8,
    every: int = 1,
    step: int = 1,
    max_svd_dim: int = 4096
):
    """
    Gradient low-rank projection via truncated SVD (in-place).

    Purpose
    -------
    • Regularizes large gradient tensors by projecting them onto a low-rank
      subspace (top-k singular values).
    • Helps stabilize training, reduce noise, and improve memory efficiency
      for very high-dimensional parameter gradients (e.g., large conv/linear layers).

    Args
    ----
    model : torch.nn.Module
        Model containing parameters with .grad fields.
    rank : int (default=8)
        Target rank for truncated SVD. (use 0 or negative to disable)
    every : int (default=1)
        Apply projection every N steps. (0 disables entirely)
    step : int (default=1)
        Current global training step (used with `every`).
    max_svd_dim : int (default=4096)
        Maximum dimension along which to compute SVD.
        If gradients exceed this size, they are truncated to avoid expensive SVD.

    Behavior
    --------
    • Only applies to parameter gradients with ndim ≥ 2 (matrices, conv kernels).
    • Reshapes gradient to 2D: [out, in*kernel].
    • If min(m.shape) < 2, skips (too small for SVD).
    • Uses torch.linalg.svd(m) and reconstructs using top-r singular components.
    • Writes the low-rank approximation back into p.grad in-place.

    Notes
    -----
    • Wrapped in @torch.no_grad() to avoid autograd tracking.
    • Failures (e.g., SVD convergence) are silently skipped.
    • Recommended for large models where gradients are very high-dimensional.
    """

    # Skip if not scheduled for this step
    if (every <= 0) or (step % every != 0):
        return

    for p in model.parameters():
        if (p.grad is None) or (p.grad.ndim < 2):
            continue  # skip scalars/vectors

        g = p.grad.data
        m = g.reshape(g.shape[0], -1)  # flatten to 2D [out, in*kh*kw]

        # Skip small tensors or disabled rank
        if min(m.shape) < 2 or rank <= 0:
            continue

        # Truncate large matrices before SVD for efficiency
        if max(m.shape) > max_svd_dim:
            if m.shape[1] >= m.shape[0]:
                m = m[:, :max_svd_dim]
            else:
                m = m[:max_svd_dim, :]

        try:
            # Full SVD and truncated reconstruction
            U, S, Vh = torch.linalg.svd(m, full_matrices=False)
            r = min(rank, S.numel())
            m_approx = (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :])
            g.copy_(m_approx.view_as(g))  # in-place overwrite
        except Exception:
            # Silently ignore numerical issues
            pass
