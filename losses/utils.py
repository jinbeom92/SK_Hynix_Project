import torch

def per_sample_sino_scale(
    S_in: torch.Tensor,
    q: float = 0.99,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute a per-sample robust scale for sinograms via high-quantile norm.

    Axis convention
    ---------------
    Sinogram is in **(x, a, z)** order → tensor shape **[B, X, A, Z]**.

    Parameters
    ----------
    S_in : Tensor
        Input sinograms with shape **[B, X, A, Z]** (any floating dtype).
    q : float
        Quantile in (0,1). Typical 0.95–0.995. Higher → less sensitive to outliers.
    eps : float
        Lower clamp to avoid division by zero.

    Returns
    -------
    Tensor
        Per-sample scale with shape **[B, 1, 1, 1]** (broadcastable).
        `requires_grad=False`.

    Notes
    -----
    • Uses the q-quantile of |S_in| per sample.
    • Wraps computation in `torch.no_grad()` to keep it out of the graph.
    """
    with torch.no_grad():
        B = S_in.shape[0]
        s = torch.quantile(S_in.abs().reshape(B, -1), q, dim=1)  # [B]
        s = s.clamp_min(eps).view(B, 1, 1, 1)  # [B,1,1,1]
    return s  # requires_grad=False


def normalize_tensors(
    S_in: torch.Tensor,
    sino_hat: torch.Tensor,
    R_hat: torch.Tensor,
    V_gt: torch.Tensor,
    s_sino: torch.Tensor
):
    """
    Normalize sinograms and volumes using per-sample sinogram scales.

    Axis convention
    ---------------
    - Sinograms: **(x, a, z)** → **[B, X, A, Z]**
    - Volumes:   **(x, y, z)** → **[B, X, Y, Z]** or **[B, 1, X, Y, Z]**

    Parameters
    ----------
    S_in, sino_hat : Tensor
        Shapes **[B, X, A, Z]**.
    R_hat, V_gt : Tensor
        Shapes **[B, X, Y, Z]** or **[B, 1, X, Y, Z]**.
    s_sino : Tensor
        Per-sample scale **[B, 1, 1, 1]** returned by `per_sample_sino_scale`.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        `(S_in_n, sino_hat_n, R_hat_n, V_gt_n)` with the **same shapes**
        as their respective inputs.

    Notes
    -----
    • The same per-sample scale is broadcast to both detector domain
      (sinograms) and volume domain, tying intensities to a common reference.
    • No axis permutation, no dtype changes — only broadcast division.
    """
    # Detector-domain normalization (x,a,z): [B,X,A,Z] / [B,1,1,1]
    S_in_n     = S_in     / s_sino
    sino_hat_n = sino_hat / s_sino

    # Volume-domain normalization (x,y,z)
    if R_hat.ndim == 5:   # [B,1,X,Y,Z]
        s_vol_R = s_sino.view(-1, 1, 1, 1, 1)
    elif R_hat.ndim == 4: # [B,X,Y,Z]
        s_vol_R = s_sino
    else:
        raise ValueError(f"R_hat must be [B,X,Y,Z] or [B,1,X,Y,Z], got {tuple(R_hat.shape)}")

    if V_gt.ndim == 5:    # [B,1,X,Y,Z]
        s_vol_V = s_sino.view(-1, 1, 1, 1, 1)
    elif V_gt.ndim == 4:  # [B,X,Y,Z]
        s_vol_V = s_sino
    else:
        raise ValueError(f"V_gt must be [B,X,Y,Z] or [B,1,X,Y,Z], got {tuple(V_gt.shape)}")

    R_hat_n = R_hat / s_vol_R
    V_gt_n  = V_gt  / s_vol_V

    return S_in_n, sino_hat_n, R_hat_n, V_gt_n
