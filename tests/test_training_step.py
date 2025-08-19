# =================================================================================================
# Sanity Test — One-Step Forward Pass & Cheat-Gate Enforcement
# -------------------------------------------------------------------------------------------------
# Purpose
#   Validates that the HDNSystem’s forward path produces tensors with the expected shapes and that
#   the cheat-gate behaves correctly (enabled during training, disabled during evaluation when
#   `train_only=True`). This serves as a quick smoke test for model wiring and configuration.
#
# What It Checks
#   1) Shape sanity:
#        sino_hat : [B, A, V, U]
#        R_hat    : [B, 1, D, H, W]
#        S_pred   : [B, A, V, U]
#   2) Cheat-gate enforcement:
#        With `cheat = {enabled: True, gate: 1.0, train_only: True}`,
#        outputs between train_mode=True vs False must differ (gate off in eval).
#
# Geometry/Config
#   • Small synthetic geometry (D,H,W) and detector grid (A,V,U) to keep test fast.
#   • Joseph projector is used by default; PSF disabled for simplicity.
#
# Usage
#   python tests/test_training_step.py
#
# Notes
#   • Runs on CUDA if available; otherwise defaults to CPU.
#   • This is a minimal sanity check; more comprehensive tests should include numerical
#     comparisons, deterministic seeds, and multi-geometry sweeps.
# =================================================================================================
import torch
from physics.geometry import Parallel3DGeometry
from models.hdn import HDNSystem

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B=1; A=8; V=4; U=8; D=4; H=8; W=8
    angles = torch.linspace(0, 3.14159, A, device=device)
    geom = Parallel3DGeometry(vol_shape=(D,H,W), det_shape=(V,U), angles=angles, angle_chunk=4, n_steps_cap=64)
    cfg = {
        "projector": {
            "method": "joseph3d",
            "joseph":{"n_steps_cap":64},
            "angle_chunk":4,
            "psf":{"enabled": False}
        },
        "model": {
            "enc1":{"base":16, "depth":2},
            "enc2":{"base":16, "depth":2, "harm_K":2},
            "enc3":{"enabled": True, "base":8, "depth":2},
            "align":{"out_ch":32},
            "dec":{"mid_ch":32, "n_proj_ch":4}
        },
        "cheat":{"enabled": True, "aggregate": "angle_mean", "dft_K": 2, "gate": 1.0, "train_only": True, "psf_consistent": False},
    }
    model = HDNSystem(geom, cfg).to(device)

    S_in = torch.rand(B,A,V,U, device=device)
    V_gt = torch.rand(B,1,D,H,W, device=device)

    sino_hat_train, R_hat_train, S_pred_train = model(S_in, angles, V_gt=V_gt, train_mode=True)
    sino_hat_eval,  R_hat_eval,  S_pred_eval  = model(S_in, angles, V_gt=V_gt, train_mode=False)

    assert sino_hat_train.shape == (B,A,V,U)
    assert R_hat_train.shape  == (B,1,D,H,W)
    assert S_pred_train.shape == (B,A,V,U)

    # Cheat-gate OFF at eval → outputs must differ from train (gate=1.0 makes injection strong).
    diff = (sino_hat_train - sino_hat_eval).abs().mean().item()
    assert diff > 1e-6, f"Cheat-gate train_only not enforced; diff={diff}"

    print("OK: shapes & cheat-gate enforcement")

if __name__ == "__main__":
    run()
