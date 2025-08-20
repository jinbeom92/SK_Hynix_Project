import torch
from physics.geometry import Parallel3DGeometry
from physics.projector import JosephProjector3D, SiddonProjector3D


def adjoint_inner_product(vol: torch.Tensor, sino: torch.Tensor, proj) -> tuple[float, float]:
    """
    Compute inner products to verify the adjoint relationship:

        ⟨ FP(vol), sino ⟩  vs  ⟨ vol, BP(sino) ⟩

    Args:
        vol  : Tensor [B, C, D, H, W] — test volume.
        sino : Tensor [B, C, A, V, U] — test sinogram.
        proj : BaseProjector3D — projector implementing forward() and backproject().

    Returns:
        (lhs, rhs) as floats:
          lhs = ⟨ FP(vol),  sino ⟩
          rhs = ⟨ vol, BP(sino) ⟩
    """
    fp = proj(vol)                     # [B, C, A, V, U]
    lhs = (fp * sino).sum()
    bp = proj.backproject(sino)        # [B, C, D, H, W]
    rhs = (vol * bp).sum()
    return lhs.item(), rhs.item()


def run_once(method: str, tol_rel: float = 5e-3) -> float:
    """
    Run a single adjoint-consistency check for the chosen projector.

    Args:
        method  : str — "joseph" or "siddon".
        tol_rel : float — relative tolerance for |lhs − rhs| / (|lhs| + |rhs| + ε).

    Returns:
        rel : float — measured relative discrepancy (should be < tol_rel).

    Procedure:
        • Build a small synthetic geometry and random tensors (vol, sino).
        • Compute lhs = ⟨FP(vol), sino⟩ and rhs = ⟨vol, BP(sino)⟩.
        • Assert relative error < tol_rel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small synthetic setup (kept modest for speed/determinism)
    A, V, U = 7, 3, 8
    D, H, W = 4, 6, 7
    angles = torch.linspace(0, 3.14159, A, device=device)  # [0, π)

    geom = Parallel3DGeometry(
        vol_shape=(D, H, W),
        det_shape=(V, U),
        angles=angles,
        angle_chunk=4,
        n_steps_cap=64,
    )

    if method == "joseph":
        proj = JosephProjector3D(geom, n_steps=64).to(device)
    else:
        proj = SiddonProjector3D(geom).to(device)

    B, C = 2, 1
    vol  = torch.rand(B, C, D, H, W, device=device)
    sino = torch.rand(B, C, A, V, U, device=device)

    lhs, rhs = adjoint_inner_product(vol, sino, proj)
    rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12)
    assert rel < tol_rel, (
        f"Adjoint mismatch ({method}): rel={rel:.6e}  lhs={lhs:.6e}  rhs={rhs:.6e}"
    )
    return rel


if __name__ == "__main__":
    # Smoke test both projectors
    r1 = run_once("joseph")
    r2 = run_once("siddon")
    print(f"[adjoint] joseph rel={r1:.3e} | siddon rel={r2:.3e}")
