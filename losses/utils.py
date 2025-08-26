import torch

def per_sample_sino_scale(
    S_in: torch.Tensor,
    q: float = 0.99,
    eps: float = 1e-6
) -> torch.Tensor:
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
