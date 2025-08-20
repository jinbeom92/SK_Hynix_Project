import torch
from physics.geometry import Parallel3DGeometry
from models.hdn import HDNSystem


def run():
    """
    Sanity test for the HDNSystem forward path and cheat-gate behavior.

    What this checks
    ----------------
    1) **Tensor shapes**:
         - sino_hat : [B, A, V, U]
         - R_hat    : [B, 1, D, H, W]
         - S_pred   : [B, A, V, U]    (if your implementation returns it)
    2) **Cheat-gate enforcement**:
         - With `cheat = {enabled: True, gate: 1.0, train_only: True}`,
           the model's outputs in train_mode=True vs train_mode=False
           must differ (cheat path off at eval).

    Notes
    -----
    • This is a lightweight smoke test meant to catch wiring mistakes.
    • Geometry is small to keep runtime minimal.
    • If your `HDNSystem` constructor signature differs (e.g., cfg-only),
      adapt instantiation accordingly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small synthetic geometry (kept modest for speed/determinism)
    B = 1
    A, V, U = 8, 4, 8
    D, H, W = 4, 8, 8

    angles = torch.linspace(0, 3.14159, A, device=device)  # [0, π)
    geom = Parallel3DGeometry(
        vol_shape=(D, H, W),
        det_shape=(V, U),
        angles=angles,
        angle_chunk=4,
        n_steps_cap=64,
    )

    # Minimal model config for sanity pass
    cfg = {
        "projector": {
            "method": "joseph3d",
            "joseph": {"n_steps_cap": 64},
            "angle_chunk": 4,
            "psf": {"enabled": False},
        },
        "model": {
            "enc1": {"base": 16, "depth": 2},
            "enc2": {"base": 16, "depth": 2, "harm_K": 2},
            "enc3": {"enabled": True, "base": 8, "depth": 2},  # optional (planned)
            "align": {"out_ch": 32},
            "dec": {"mid_ch": 32, "n_proj_ch": 4},
        },
        "cheat": {
            "enabled": True,
            "aggregate": "angle_mean",
            "dft_K": 2,
            "gate": 1.0,
            "train_only": True,
            "psf_consistent": False,
        },
    }

    # If your implementation is cfg-only, use: model = HDNSystem(cfg).to(device)
    model = HDNSystem(geom, cfg).to(device)

    # Random inputs
    S_in = torch.rand(B, A, V, U, device=device)        # detector-domain input
    V_gt = torch.rand(B, 1, D, H, W, device=device)     # ground-truth volume

    # Forward (train mode: cheat ON)
    sino_hat_train, R_hat_train, S_pred_train = model(
        S_in, angles, V_gt=V_gt, train_mode=True
    )
    # Forward (eval mode: cheat OFF)
    sino_hat_eval, R_hat_eval, S_pred_eval = model(
        S_in, angles, V_gt=V_gt, train_mode=False
    )

    # --- Shape checks ---------------------------------------------------------
    assert sino_hat_train.shape == (B, A, V, U), "sino_hat_train shape mismatch"
    assert R_hat_train.shape == (B, 1, D, H, W), "R_hat_train shape mismatch"
    assert S_pred_train.shape == (B, A, V, U), "S_pred_train shape mismatch"

    # --- Cheat-gate OFF at eval: outputs should differ from train -------------
    # With gate=1.0, the difference should be easily detectable.
    diff = (sino_hat_train - sino_hat_eval).abs().mean().item()
    assert diff > 1e-6, f"Cheat-gate train_only not enforced; diff={diff}"

    print("OK: shapes & cheat-gate enforcement")


if __name__ == "__main__":
    run()
