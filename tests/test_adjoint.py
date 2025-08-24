import torch
from physics.geometry import Parallel3DGeometry
from physics.projector import JosephProjector3D, SiddonProjector3D

# ======================================================================================
# Adjoint (FP,BP) consistency test in our pipeline's shape convention:
#   - Volume  : [B, C, D, H, W]  where (D,H,W) == (Z,Y,X)
#   - Sinogram: [B, C, X, A, Z]  (X = detector-u, Z = detector-v)
#
# Most projectors implement:
#   forward(vol)      -> [B, C, A, V, U]  (A = angles, V ≡ Z, U ≡ X)
#   backproject(sino) -> [B, C, D, H, W]
#
# We convert between [A,V,U] and [X,A,Z] as:
#   [B,C,A,V,U]  --permute(0,1,4,2,3)-->  [B,C,U(=X),A,V(=Z)]
#   [B,C,X,A,Z]  --permute(0,1,3,4,2)-->  [B,C,A,Z,X]
# ======================================================================================

def _fp_to_xaz(fp_avu: torch.Tensor) -> torch.Tensor:
    """
    Convert projector forward output [B,C,A,V,U] to our xaz layout [B,C,X,A,Z].
    """
    # [B,C,A,V,U] -> [B,C,U(=X),A,V(=Z)]
    return fp_avu.permute(0, 1, 4, 2, 3).contiguous()

def _xaz_to_avu(sino_xaz: torch.Tensor) -> torch.Tensor:
    """
    Convert our sinogram layout [B,C,X,A,Z] to projector input [B,C,A,V,U].
    """
    # [B,C,X,A,Z] -> [B,C,A,Z(=V),X(=U)]
    return sino_xaz.permute(0, 1, 3, 4, 2).contiguous()


def adjoint_inner_product_xaz(vol_bcdhw: torch.Tensor,
                              sino_bcxaz: torch.Tensor,
                              proj) -> tuple[float, float]:
    """
    Compute < FP(vol), sino > and < vol, BP(sino) > **in our (X,A,Z) convention**.

    Parameters
    ----------
    vol_bcdhw : Tensor
        Volume [B, C, D, H, W] with (D,H,W) = (Z,Y,X).
    sino_bcxaz : Tensor
        Sinogram [B, C, X, A, Z] (our pipeline layout).
    proj : BaseProjector3D
        Projector implementing forward() -> [B,C,A,V,U] and
        backproject() -> [B,C,D,H,W].

    Returns
    -------
    (lhs, rhs) : tuple[float, float]
        lhs = < FP(vol), sino > computed after converting FP(vol) to [B,C,X,A,Z]
        rhs = < vol, BP(sino) >
    """
    # Forward: vol -> [B,C,A,V,U] then convert to [B,C,X,A,Z]
    fp_avu = proj(vol_bcdhw)                  # [B,C,A,V,U]
    fp_xaz = _fp_to_xaz(fp_avu)               # [B,C,X,A,Z]
    lhs = (fp_xaz * sino_bcxaz).sum()

    # Backproject: [B,C,X,A,Z] -> [B,C,A,V,U] then BP -> [B,C,D,H,W]
    sino_avu = _xaz_to_avu(sino_bcxaz)        # [B,C,A,V,U]
    bp = proj.backproject(sino_avu)           # [B,C,D,H, W]
    rhs = (vol_bcdhw * bp).sum()

    return lhs.item(), rhs.item()


def run_once(method: str = "joseph",
             tol_rel: float = 2e-2,
             device: torch.device | None = None,
             dtype: torch.dtype = torch.float32) -> float:
    """
    Single adjoint-consistency check under our (X,A,Z) I/O.

    Args
    ----
    method  : {"joseph", "siddon"}
        Projector to test.
    tol_rel : float
        Relative tolerance on   |lhs - rhs| / (|lhs| + |rhs| + eps).
        Note: Joseph (sampled integration) is not exactly adjoint in practice;
              use a looser tol (e.g., 1e-2~3e-2). Siddon can be tighter.
    device  : torch.device
        Target device. Defaults to "cuda" if available else "cpu".
    dtype   : torch.dtype
        Tensor dtype for the test tensors.

    Returns
    -------
    rel : float
        Measured relative discrepancy.

    Geometry
    --------
    We pick small shapes for speed/determinism:
      X=8, Y=6, Z=5, A=9  -> vol: [B,C,D=Z,H=Y,W=X], sino: [B,C,X,A,Z]
      Detector: (V=Z, U=X) to match our mapping.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Synthetic geometry (small)
    X, Y, Z = 8, 6, 5
    A = 9
    D, H, W = Z, Y, X
    V, U = Z, X
    angles = torch.linspace(0.0, torch.pi, steps=A, device=device, dtype=dtype)  # [0, π)

    geom = Parallel3DGeometry(
        vol_shape=(D, H, W),   # (Z, Y, X)
        det_shape=(V, U),      # (Z, X)
        angles=angles,
        angle_chunk=4,
        n_steps_cap=64,
    )

    if method.lower().startswith("joseph"):
        proj = JosephProjector3D(geom, n_steps=64).to(device)
    elif method.lower().startswith("siddon"):
        proj = SiddonProjector3D(geom).to(device)
    else:
        raise ValueError(f"unknown method: {method}")

    # --- Random test tensors
    B, C = 2, 1
    g = torch.Generator(device=device).manual_seed(1337)
    vol  = torch.rand((B, C, D, H, W), generator=g, device=device, dtype=dtype)
    sino = torch.rand((B, C, X, A, Z), generator=g, device=device, dtype=dtype)

    # --- Adjoint check in our x-a-z layout
    lhs, rhs = adjoint_inner_product_xaz(vol, sino, proj)
    rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12)

    # Relaxed tol for Joseph; Siddon is typically tighter
    assert rel < tol_rel, (
        f"Adjoint mismatch ({method}): rel={rel:.6e}  lhs={lhs:.6e}  rhs={rhs:.6e}"
    )
    return rel


if __name__ == "__main__":
    # Smoke test both projectors (with slightly different tolerances)
    r1 = run_once("joseph", tol_rel=2e-2)
    r2 = run_once("siddon", tol_rel=5e-3)
    print(f"[adjoint-xaz] joseph rel={r1:.3e} | siddon rel={r2:.3e}")
