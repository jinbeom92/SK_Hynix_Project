from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from utils.yaml_config import load_config, deep_update
from utils.seed import set_seed
from physics.geometry import Parallel3DGeometry
from physics.projector import make_projector
from models.hdn import HDNSystem


def _prefix(name: str) -> str:
    """Return the filename stem without the ``_sino`` or ``_voxel`` suffix."""
    stem = Path(name).stem
    for suf in ("_sino", "_voxel"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _resolve_first_pair(sino_dir: str, voxel_dir: str) -> Tuple[str, str]:
    """Return the first matching sinogram/voxel file pair.

    The function scans the ``sino_dir`` for ``*_sino.npy`` files and the
    ``voxel_dir`` for ``*_voxel.npy`` files, pairing them by prefix.  It
    returns the first match found.  If no matching pair exists, an error
    is raised.
    """
    p_s = Path(sino_dir)
    p_v = Path(voxel_dir)
    sinos = sorted(p_s.glob("*_sino.npy"))
    voxels = sorted(p_v.glob("*_voxel.npy"))
    if not sinos:
        raise FileNotFoundError(f"No sinograms found in {sino_dir}")
    if not voxels:
        raise FileNotFoundError(f"No voxels found in {voxel_dir}")
    v_map = {_prefix(v.name): v for v in voxels}
    for s in sinos:
        key = _prefix(s.name)
        if key in v_map:
            return str(s), str(v_map[key])
    raise RuntimeError("No matching sinogram/voxel prefix found")


def _load_sino_xaz(path: str) -> np.ndarray:
    """Load a sinogram file and ensure shape ``[X, A, Z]``."""
    arr = np.load(path)
    if arr.ndim == 2:
        X, A = arr.shape
        arr = arr.reshape(X, A, 1)
    if arr.ndim != 3:
        raise ValueError(f"sinogram must be [X,A] or [X,A,Z], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _load_voxel_xyz(path: str) -> np.ndarray:
    """Load a voxel volume and ensure shape ``[X, Y, Z]``."""
    arr = np.load(path)
    if arr.ndim == 2:
        X, Y = arr.shape
        arr = arr.reshape(X, Y, 1)
    if arr.ndim != 3:
        raise ValueError(f"voxel must be [X,Y] or [X,Y,Z], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _assert_compatible(S_xaz: np.ndarray, V_xyz: np.ndarray) -> Tuple[int, int, int, int]:
    """Check that sinogram and voxel volumes are shape compatible."""
    Xs, A, Zs = S_xaz.shape
    Xv, Y, Zv = V_xyz.shape
    if Xs != Xv or Zs != Zv:
        raise AssertionError(
            f"Shape mismatch: sino[X,A,Z]={S_xaz.shape}, voxel[X,Y,Z]={V_xyz.shape}"
        )
    return Xs, Y, A, Zs


def _angles_from_cfg(A: int, cfg: dict) -> torch.Tensor:
    """Compute evenly spaced angles based on the ``bp_span`` configuration."""
    span_key = str(cfg.get("projector", {}).get("bp_span", "auto")).lower()
    if span_key == "full":
        theta_span = 2.0 * math.pi
    elif span_key == "half":
        theta_span = math.pi
    else:
        theta_span = (2.0 * math.pi) if A >= 300 else math.pi
    angles = torch.linspace(0.0, theta_span, steps=A + 1, dtype=torch.float32)[:-1]
    return angles


def _build_geometry_z1(cfg: dict, X: int, Y: int, A: int) -> Parallel3DGeometry:
    """Construct a Z=1 parallel‑beam geometry for slice‑mode inference."""
    angles = _angles_from_cfg(A, cfg)
    gcfg = cfg.get("geom", {})
    return Parallel3DGeometry.from_xyz(
        vol_shape_xyz=(X, Y, 1),
        det_shape_xz=(X, 1),
        angles=angles,
        voxel_size_xyz=tuple(map(float, gcfg.get("voxel_size_xyz", (1.0, 1.0, 1.0)))),
        det_spacing_xz=tuple(map(float, gcfg.get("det_spacing_xz", (1.0, 1.0)))),
        angle_chunk=int(gcfg.get("angle_chunk", 16)),
        n_steps_cap=int(gcfg.get("n_steps_cap", 256)),
    )


def _load_model_slice_mode(cfg: dict, geom_z1: Parallel3DGeometry, ckpt_path: str, device: torch.device) -> HDNSystem:
    """Load a trained HDNSystemTS and its checkpoint for slice‑mode inference."""
    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = state.get("config", {}) or {}
    # Merge stored configuration into current cfg for keys we care about
    for k in ("model", "psf", "debug"):
        if k in ckpt_cfg:
            cfg = deep_update(cfg, {k: ckpt_cfg[k]})
    projector = make_projector(
        method=str(cfg.get("projector", {}).get("method", "joseph3d")).lower(),
        geom=geom_z1,
    ).to(device)
    proj_cfg = cfg.get("projector", {})
    model_cfg = cfg.get("model", {})
    # Set projector options from configuration
    projector.fbp_filter = str(proj_cfg.get("fbp_filter", getattr(projector, "fbp_filter", "ramp"))).lower()
    projector.bp_span = str(proj_cfg.get("bp_span", getattr(projector, "bp_span", "auto"))).lower()
    projector.ir_circle = bool(model_cfg.get("ir_circle", getattr(projector, "ir_circle", True)))
    # Build HDN system (slice mode)
    model = HDNSystem(cfg, projector=projector).to(device)
    # Disable PSF during inference
    if hasattr(model, "psf"):
        model.psf.enabled = False
    # Load state dictionary; prune keys related to geometry or PSF
    state_dict = state.get("model_state", state)
    drop_keys = (
        "projector.angles", "projector.cos_angles", "projector.sin_angles",
        "projector.u_phys", "projector.v_phys",
        "psf._sigma_u", "psf._sigma_v", "psf._A",
    )
    pruned = {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in drop_keys)}
    model.load_state_dict(pruned, strict=False)
    # Assign geometry on the projector in case it was overwritten by state_dict
    model.projector.geom = geom_z1
    model.eval()
    return model


def _metrics(R_hat: np.ndarray, V_gt: np.ndarray) -> dict:
    """Compute 2D MSE and PSNR between reconstruction and ground truth."""
    Vc = np.clip(V_gt, 0.0, 1.0)
    Rc = np.clip(R_hat, 0.0, 1.0)
    mse = float(np.mean((Rc - Vc) ** 2))
    psnr = float(20.0 * np.log10(1.0 / (np.sqrt(mse) + 1e-12))) if mse > 0 else float("inf")
    return {"mse": mse, "psnr": psnr}


def _visualize(V_xyz: np.ndarray, R_xyz: np.ndarray, out_png: Path) -> None:
    """Create and save a visual comparison of GT, reconstruction and error."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import torch
    import torch.nn.functional as F

    # Convert volumes [X,Y,Z] → [Y,X,Z] for intuitive indexing
    V_yxz = V_xyz.transpose(1, 0, 2)
    R_yxz = R_xyz.transpose(1, 0, 2)
    Ydim, Xdim, Zdim = V_yxz.shape
    # Mid-slice indices
    z0, y0, x0 = Zdim // 2, Ydim // 2, Xdim // 2

    # Extract planes
    vam_ax = V_yxz[:, :, z0]
    vam_co = V_yxz[y0, :, :].T
    vam_sa = V_yxz[:, x0, :].T
    rec_ax = R_yxz[:, :, z0]
    rec_co = R_yxz[y0, :, :].T
    rec_sa = R_yxz[:, x0, :].T
    dif_ax = np.abs(vam_ax - rec_ax)
    dif_co = np.abs(vam_co - rec_co)
    dif_sa = np.abs(vam_sa - rec_sa)

    # Quantile-based scaling for intensity and difference
    ints_all = np.concatenate([
        vam_ax.ravel(), rec_ax.ravel(),
        vam_co.ravel(), rec_co.ravel(),
        vam_sa.ravel(), rec_sa.ravel(),
    ])
    vmin_int, vmax_int = np.quantile(ints_all, [0.01, 0.99])
    diff_all = np.concatenate([dif_ax.ravel(), dif_co.ravel(), dif_sa.ravel()])
    vmin_diff, vmax_diff = 0.0, np.quantile(diff_all, 0.99)
    norm_int = Normalize(vmin=vmin_int, vmax=vmax_int)
    norm_diff = Normalize(vmin=vmin_diff, vmax=vmax_diff)

    # Compute 3D SSIM and PSNR on normalised volumes
    def _to_dhw(yxz: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.transpose(yxz, (2, 0, 1))[None, None].astype(np.float32))
    def _psnr(x: torch.Tensor, y: torch.Tensor) -> float:
        mse = torch.mean((x - y) ** 2)
        return float(10 * torch.log10(1.0 / (mse + 1e-8)))
    def _ssim3d(x: torch.Tensor, y: torch.Tensor) -> float:
        win = 7
        coords = torch.arange(win, dtype=x.dtype) - (win - 1) / 2
        g = torch.exp(-0.5 * (coords / 1.5) ** 2); g = (g / g.sum()).view(1, 1, -1)
        def blur3d(t):
            out = F.conv3d(t, g.view(1, 1, win, 1, 1), padding=(win//2, 0, 0))
            out = F.conv3d(out, g.view(1, 1, 1, win, 1), padding=(0, win//2, 0))
            out = F.conv3d(out, g.view(1, 1, 1, 1, win), padding=(0, 0, win//2))
            return out
        C1, C2 = 0.01**2, 0.03**2
        mx, my = blur3d(x), blur3d(y)
        sx, sy = blur3d(x*x) - mx*mx, blur3d(y*y) - my*my
        sxy    = blur3d(x*y) - mx*my
        ssim_map = ((2*mx*my+C1)*(2*sxy+C2))/((mx*mx+my*my+C1)*(sx+sy+C2))
        return float(ssim_map.mean())

    vam01 = (V_yxz - V_yxz.min()) / max(V_yxz.max() - V_yxz.min(), 1e-8)
    rec01 = (R_yxz - R_yxz.min()) / max(R_yxz.max() - R_yxz.min(), 1e-8)
    vam_t = _to_dhw(vam01)
    rec_t = _to_dhw(rec01)
    psnr3d = _psnr(vam_t, rec_t)
    ssim3d = _ssim3d(vam_t, rec_t)

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    plt.suptitle(
        f"Normalized view — GT vs Recon | SSIM3D={ssim3d:.4f}, PSNR={psnr3d:.2f} dB",
        fontsize=12,
    )
    plane_data = [
        (vam_ax, rec_ax, dif_ax, f"Axial (z={z0})"),
        (vam_co, rec_co, dif_co, f"Coronal (y={y0})"),
        (vam_sa, rec_sa, dif_sa, f"Sagittal (x={x0})"),
    ]
    for row, (gt_img, rec_img, diff_img, plane_title) in enumerate(plane_data):
        for col, (data, title, cmap, norm) in enumerate([
            (gt_img, "GT", "CMRmap", norm_int),
            (rec_img, "Recon", "CMRmap", norm_int),
            (diff_img, "|diff|", "inferno", norm_diff),
        ]):
            ax = axes[row, col]
            ax.imshow(data.T, cmap=cmap, norm=norm, origin="lower")
            if col == 0:
                ax.set_ylabel(plane_title, fontsize=10)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    # Colourbars
    cax_int = fig.add_axes([0.92, 0.56, 0.015, 0.32])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_int, cmap="CMRmap"),
        cax=cax_int, label="Intensity (0–1)",
    )
    cax_diff = fig.add_axes([0.92, 0.12, 0.015, 0.32])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_diff, cmap="inferno"),
        cax=cax_diff, label="|diff|",
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    """Perform inference on the first sinogram/voxel pair and save outputs."""
    # Fixed paths (for clarity and consistency)
    sino_dir = "dataset/sino"
    voxel_dir = "dataset/voxel"
    ckpt_path = "results/ckpt/best_shared.pt"
    cfg_path = "config.yaml"
    out_dir = Path("results/out")

    # Seed and device
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve first matching sinogram/voxel pair
    sino_path, voxel_path = _resolve_first_pair(sino_dir, voxel_dir)
    S_xaz = _load_sino_xaz(sino_path)
    V_xyz = _load_voxel_xyz(voxel_path)
    X, Y, A, Z = _assert_compatible(S_xaz, V_xyz)

    # Load configuration and build geometry
    cfg = load_config(cfg_path)
    geom_z1 = _build_geometry_z1(cfg, X=X, Y=Y, A=A)
    model = _load_model_slice_mode(cfg, geom_z1, ckpt_path, device)

    # Run inference (cheat disabled)
    S = torch.from_numpy(S_xaz).to(device).unsqueeze(0).unsqueeze(0)  # [B=1,1,X,A,Z]
    with torch.no_grad():
        sino_hat, recon_hat = model(S, v_vol=None, train_mode=False)
    sino_hat_np = sino_hat[0, 0].cpu().numpy()
    recon_hat_np = recon_hat[0, 0].cpu().numpy()

    # Save raw outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sinogram_opt.npy", sino_hat_np)
    np.save(out_dir / "reconstruction_opt.npy", recon_hat_np)

    # Generate visualisation
    _visualize(V_xyz, recon_hat_np, out_dir / "visualization.png")

    # Compute metrics (2D) and extended 3D metrics
    metrics = _metrics(recon_hat_np, V_xyz)
    try:
        import torch.nn.functional as F  # noqa: F401
        # Normalise volumes to [0,1]
        def _norm_vol(vol: np.ndarray) -> np.ndarray:
            mn, mx = float(vol.min()), float(vol.max())
            return (vol - mn) / max(mx - mn, 1e-8)
        gt_norm = _norm_vol(V_xyz)
        rec_norm = _norm_vol(recon_hat_np)
        def _to_dhw(vol: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.transpose(vol, (2, 0, 1))[None, None].astype(np.float32))
        gt_t = _to_dhw(gt_norm)
        rec_t = _to_dhw(rec_norm)
        mse_3d = torch.mean((gt_t - rec_t) ** 2)
        psnr3d = float(10.0 * torch.log10(1.0 / (mse_3d + 1e-8)))
        def _ssim3d(x: torch.Tensor, y: torch.Tensor) -> float:
            win = 7
            coords = torch.arange(win, dtype=x.dtype) - (win - 1) / 2
            g = torch.exp(-0.5 * (coords / 1.5) ** 2); g = (g / g.sum()).view(1, 1, -1)
            def blur3d(t):
                out = F.conv3d(t, g.view(1, 1, win, 1, 1), padding=(win//2, 0, 0))
                out = F.conv3d(out, g.view(1, 1, 1, win, 1), padding=(0, win//2, 0))
                out = F.conv3d(out, g.view(1, 1, 1, 1, win), padding=(0, 0, win//2))
                return out
            C1, C2 = 0.01**2, 0.03**2
            mx, my = blur3d(x), blur3d(y)
            sx, sy = blur3d(x*x) - mx*mx, blur3d(y*y) - my*my
            sxy    = blur3d(x*y) - mx*my
            ssim_map = ((2*mx*my+C1)*(2*sxy+C2))/((mx*mx+my*my+C1)*(sx+sy+C2))
            return float(ssim_map.mean())
        ssim3d = _ssim3d(gt_t, rec_t)
        metrics["psnr3d"] = psnr3d
        metrics["ssim3d"] = ssim3d
    except Exception:
        # If anything goes wrong (e.g., missing torch.nn.functional), skip 3D metrics
        pass

    # Save metrics
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Finished inference. Results saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
