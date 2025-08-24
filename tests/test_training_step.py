import math
import torch
from physics.geometry import Parallel3DGeometry
from physics.projector import JosephProjector3D  # or SiddonProjector3D
from models.hdn import HDNSystem


def run():
    """
    Sanity test for the HDNSystem forward path and cheat gating under our axis convention.

    What this checks
    ----------------
    1) Tensor shapes (our project convention):
         - sino_opt_xaz : [B, 1, X, A, Z]   (optimized sinogram in (x,a,z))
         - recon_xyz    : [B, 1, X, Y, Z]   (backprojected reconstruction in (x,y,z))

    2) Cheat gating:
         - In train_mode=True with GT volume provided, the cheat path is active.
         - In train_mode=False (or with v_vol=None), the cheat path is disabled.
         - We assert that train vs eval produce different outputs.

    Notes
    -----
    • Geometry is intentionally small for speed/determinism.
    • If your HDNSystem constructor differs, adapt the instantiation line:
        - model = HDNSystem(cfg)                         # if model builds its own projector
        - model = HDNSystem(cfg, projector=projector)    # if projector is injected
        - model = HDNSystem(geom, cfg)                   # if your signature is (geom, cfg)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Small synthetic geometry
    # -----------------------------
    B = 2
    X, Y, Z = 8, 8, 1       # Z=1로 슬라이스 모드 검증 (원하면 Z>1로 올려도 무방)
    A = 9

    angles = torch.linspace(0.0, math.pi, steps=A, device=device, dtype=torch.float32)

    # Our geometry uses (D,H,W)=(Z,Y,X), detector (V,U)=(Z,X)
    geom = Parallel3DGeometry(
        vol_shape=(Z, Y, X),
        det_shape=(Z, X),
        angles=angles,
        angle_chunk=4,
        n_steps_cap=64,
    )

    # -----------------------------
    # Minimal model config
    # -----------------------------
    cfg = {
        "projector": {"method": "joseph3d"},
        "model": {
            "enc1":  {"base": 16, "depth": 2},
            "enc2":  {"base": 16, "depth": 2},
            "align": {"out_ch": 32, "depth": 2, "interp_mode": "bilinear"},
            "cheat2d": {"enabled": True, "base": 8, "depth": 1},
            "fusion": {"out_ch": 32},
            "dec":    {"mid_ch": 32, "depth": 2},
        },
    }

    # -----------------------------
    # Projector & model
    # -----------------------------
    projector = JosephProjector3D(geom, n_steps=geom.n_steps_cap).to(device)

    # (1) If your model expects (cfg, projector):
    model = HDNSystem(cfg, projector=projector).to(device)

    # (2) If your model expects (geom, cfg): use this instead
    # model = HDNSystem(geom, cfg).to(device)

    # (3) If your model builds projector internally from cfg:
    # model = HDNSystem(cfg).to(device)

    model.eval()  # we’ll toggle train/eval per forward call via train_mode flag

    # -----------------------------
    # Random inputs (our layouts)
    # -----------------------------
    S_xaz = torch.rand(B, 1, X, A, Z, device=device)    # sinogram input [B,1,X,A,Z]
    V_gt  = torch.rand(B, 1, X, Y, Z, device=device)    # GT volume    [B,1,X,Y,Z]

    # -----------------------------
    # Forward passes
    # -----------------------------
    # Train mode: cheat ON (provide v_vol)
    sino_opt_tr, recon_tr = model(S_xaz, v_vol=V_gt, train_mode=True)
    # Eval mode: cheat OFF (no v_vol)
    sino_opt_ev, recon_ev = model(S_xaz, v_vol=None, train_mode=False)

    # -----------------------------
    # Shape checks
    # -----------------------------
    assert sino_opt_tr.shape == (B, 1, X, A, Z), f"sino_opt_tr shape mismatch: {tuple(sino_opt_tr.shape)}"
    assert recon_tr.shape    == (B, 1, X, Y, Z), f"recon_tr shape mismatch: {tuple(recon_tr.shape)}"
    assert sino_opt_ev.shape == (B, 1, X, A, Z), f"sino_opt_ev shape mismatch: {tuple(sino_opt_ev.shape)}"
    assert recon_ev.shape    == (B, 1, X, Y, Z), f"recon_ev shape mismatch: {tuple(recon_ev.shape)}"

    # -----------------------------
    # Cheat-gate difference check
    # -----------------------------
    # With cheat enabled in train pass, outputs should differ from eval pass.
    # We check both sino and recon; tolerance kept tiny to catch wiring mistakes.
    diff_sino  = (sino_opt_tr - sino_opt_ev).abs().mean().item()
    diff_recon = (recon_tr    - recon_ev).abs().mean().item()
    assert diff_sino > 1e-6 or diff_recon > 1e-6, (
        f"Cheat-gate not enforced (train vs eval look identical): "
        f"diff_sino={diff_sino:.3e}, diff_recon={diff_recon:.3e}"
    )

    print("OK: shapes & cheat-gate enforcement.")


if __name__ == "__main__":
    run()
