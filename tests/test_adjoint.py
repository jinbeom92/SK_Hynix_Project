# =================================================================================================
# Adjoint Consistency Test — ⟨FP(x), y⟩ ≈ ⟨x, BP(y)⟩ for 3D Parallel-Beam CT
# -------------------------------------------------------------------------------------------------
# Purpose
#   Verifies the adjoint relationship between the forward projector (FP) and the backprojector (BP)
#   by checking inner-product equality within a specified tolerance:
#       ⟨ FP(vol), sino ⟩  ≈  ⟨ vol, BP(sino) ⟩
#   A small relative discrepancy indicates numerical consistency of the operator pair.
#
# What It Does
#   • Builds a small random geometry (D,H,W) volume grid and (A,V,U) detector grid.
#   • Instantiates either Joseph (voxel-driven, grid_sample-based) or Siddon (ray-driven) projector.
#   • Draws random inputs vol and sino on CPU/GPU.
#   • Compares inner products ⟨FP(vol), sino⟩ and ⟨vol, BP(sino)⟩ and asserts that the
#     relative error falls below `tol_rel`.
#
# Usage
#   python tests/test_adjoint.py
#   # or import run_once("joseph"|"siddon", tol_rel=5e-3) in your test harness.
#
# Notes
#   • This is a lightweight, deterministic sanity check—use small sizes to keep runtime modest.
#   • For more thorough validation, sweep multiple random seeds and geometries.
# =================================================================================================
import torch
from physics.geometry import Parallel3DGeometry
from physics.projector import JosephProjector3D, SiddonProjector3D

def adjoint_inner_product(vol, sino, proj):
    fp = proj(vol)                     # [B,C,A,V,U]
    lhs = (fp * sino).sum()
    bp = proj.backproject(sino)        # [B,C,D,H,W]
    rhs = (vol * bp).sum()
    return lhs.item(), rhs.item()

def run_once(method: str, tol_rel: float = 5e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A,V,U = 7, 3, 8
    D,H,W = 4, 6, 7
    angles = torch.linspace(0, 3.14159, A, device=device)
    geom = Parallel3DGeometry(vol_shape=(D,H,W), det_shape=(V,U), angles=angles, angle_chunk=4, n_steps_cap=64)
    if method=="joseph":
        proj = JosephProjector3D(geom, n_steps=64).to(device)
    else:
        proj = SiddonProjector3D(geom).to(device)

    B,C = 2, 1
    vol  = torch.rand(B,C,D,H,W, device=device)
    sino = torch.rand(B,C,A,V,U, device=device)

    lhs, rhs = adjoint_inner_product(vol, sino, proj)
    rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12)
    assert rel < tol_rel, f"Adjoint mismatch ({method}): rel={rel:.6e}  lhs={lhs:.6e}  rhs={rhs:.6e}"
    return rel

if __name__ == "__main__":
    r1 = run_once("joseph")
    r2 = run_once("siddon")
    print(f"[adjoint] joseph rel={r1:.3e} | siddon rel={r2:.3e}")
