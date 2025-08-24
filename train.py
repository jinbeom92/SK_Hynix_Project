from pathlib import Path
import numpy as np
import torch, gc
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler
from torch import autocast
import math

from utils.yaml_config import load_config, save_effective_config
from utils.seed import set_seed
from utils.logging import CSVLogger
from data.dataset import ConcatDepthSliceDataset
from models.hdn import HDNSystem
from physics.geometry import Parallel3DGeometry
from physics.projector import make_projector
from losses.recon import reconstruction_losses
from losses.forward import forward_consistency_loss


def _renorm(w: dict) -> dict:
    """
    Normalize a dict of loss weights so they sum to 1.0 (no-op if already ≈1).

    Parameters
    ----------
    w : dict
        key → weight (float-like)

    Returns
    -------
    dict
        Normalized weights (or original if sum≈1 or sum<=0).
    """
    s = float(sum(float(v) for v in w.values()))
    if s <= 0:
        return w
    if abs(s - 1.0) < 1e-6:
        return w
    return {k: float(v) / s for k, v in w.items()}


@torch.no_grad()
def evaluate_group_by_volume(model, ds, cfg, device):
    """
    Evaluate the model on each sinogram/voxel **file pair** by reconstructing full volumes.

    Protocol
    --------
    For each pair:
      • Read full memmaps: sinogram ``[U, A, D_f]`` and voxel ``[X, Y, D_f]``.
      • Loop depth ``d = 0..D_f-1`` and run **slice inference** with Z=1:
          * input sinogram slice ``[U, A]`` → pack to ``[1,1,X=U,A,1]``.
          * call model with **cheat OFF** → returns
              - ``sino_hat_xaz: [1,1,X,A,1]``
              - ``recon_xyz   : [1,1,X,Y,1]``
          * write ``recon_xyz[..., 0]`` into a 3D prediction buffer.
      • Stack predictions → ``pred_vol [X, Y, D_f]`` and compare with GT volume.

    Loss evaluation
    ---------------
    We reshape both predicted and GT volumes to **[B=1, C=1, Z=D_f, X, Y]** and call
    the same composite losses used during training (SSIM/PSNR/band/energy/VER/IPDR/TV).
    Metrics are averaged across files and returned.
    """
    model.eval()
    recon_w = _renorm(cfg["losses"]["weights"])
    tv_w = float(cfg["losses"].get("tv", 0.0))

    metrics_sum = {
        "ssim": 0.0, "psnr": 0.0, "band": 0.0, "energy": 0.0,
        "ver": 0.0, "ipdr": 0.0, "tv": 0.0, "count": 0
    }

    for f_idx, (s_mem, v_mem) in enumerate(zip(ds.sinos, ds.voxels)):
        # GT volume [X, Y, D_f] (memmap → owning ndarray for safe tensor conversion)
        gt_vol = np.array(v_mem, dtype=np.float32, copy=True, order='C')
        D_f = gt_vol.shape[-1]
        pred_vol = np.zeros_like(gt_vol)

        for d in range(D_f):
            s_slice = np.array(s_mem[:, :, d], dtype=np.float32, copy=True, order='C')  # [U, A]
            S = torch.from_numpy(s_slice).to(device).unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # [1,1,X,A,1]
            # Cheat OFF
            sino_hat, recon = model(S, v_vol=None, train_mode=False)  # [1,1,X,A,1], [1,1,X,Y,1]
            pred_vol[:, :, d] = recon[0, 0, :, :, 0].detach().cpu().numpy()

        # [X, Y, D_f] → [1, 1, Z=D_f, X, Y]
        pred_tensor = torch.from_numpy(pred_vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
        gt_tensor   = torch.from_numpy(gt_vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)

        params = {
            "band_low": float(cfg["losses"]["band_low"]),
            "band_high": float(cfg["losses"]["band_high"]),
            "ver_thr": float(cfg["losses"]["ver_thr"]),
            "tv_weight": tv_w,
        }
        recon_losses = reconstruction_losses(pred_tensor, gt_tensor, recon_w, params)

        # Accumulate scalar metrics per volume
        metrics_sum["ssim"]   += float(recon_losses["ssim"].mean().item())
        metrics_sum["psnr"]   += float(recon_losses["psnr"].mean().item())
        metrics_sum["band"]   += float(recon_losses["band"].mean().item())
        metrics_sum["energy"] += float(recon_losses["energy"].mean().item())
        metrics_sum["ver"]    += float(recon_losses["ver"].mean().item())
        metrics_sum["ipdr"]   += float(recon_losses["ipdr"].mean().item())
        metrics_sum["tv"]     += float(recon_losses["tv"].mean().item())
        metrics_sum["count"]  += 1

    # Average over files
    for k in ["ssim", "psnr", "band", "energy", "ver", "ipdr", "tv"]:
        if metrics_sum["count"] > 0:
            metrics_sum[k] /= metrics_sum["count"]
    return metrics_sum


def main(cfg_path: str):
    """
    Train HDN on grouped sinogram/voxel datasets with **(x,a,z)/(x,y,z)** conventions.

    Data & model axes
    -----------------
    • Sinogram (detector × angle × depth): **[U, A, D]** → model-facing **[X=U, A, Z=D]**  
    • Voxel (x, y, z): **[X, Y, D]** → model-facing **[X, Y, Z]**

    This loop:
      1) Loads YAML config and seeds RNGs.
      2) Groups file pairs into chunks for memory-friendly training.
      3) For each group, builds a dataset over **depth slices**, a DataLoader,
         and a **projector geometry** matching the current shapes.
      4) Runs training with optional **cheat injection** (only on training subset),
         computing reconstruction losses in the image domain.
      5) At each epoch end, runs full-volume evaluation (cheat OFF) and tracks
         the best checkpoint by volumetric SSIM.

    Notes
    -----
    - We pass **Z=1** to the projector during slice training/inference.
      For volumetric metrics we reconstruct each slice independently and
      stack along Z.
    - Forward-projection consistency is optional (`weights.fp`), using a
      differentiable 2D parallel-beam operator.
    """
    # --- Config IO ------------------------------------------------------------
    cfg = load_config(cfg_path)
    save_effective_config(cfg, Path(cfg_path).with_name("effective_config.json"))

    # --- Repro & device -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["train"]["seed"]))

    # --- File enumeration & grouping -----------------------------------------
    data_root = cfg.get("data", {}).get("root", "data")
    sino_glob = cfg.get("data", {}).get("sino_glob", "sino/*_sino.npy")
    voxel_glob = cfg.get("data", {}).get("voxel_glob", "voxel/*_voxel.npy")

    all_sino = sorted(Path(data_root).glob(sino_glob))
    all_vox  = sorted(Path(data_root).glob(voxel_glob))
    if len(all_sino) != len(all_vox):
        raise AssertionError(f"#sino({len(all_sino)}) != #voxel({len(all_vox)})")

    files_per_group = int(cfg["train"].get("files_per_group", 100))
    groups = [
        (all_sino[i:i + files_per_group], all_vox[i:i + files_per_group])
        for i in range(0, len(all_sino), files_per_group)
    ]

    # --- AMP policy (static) --------------------------------------------------
    want = (cfg["train"].get("amp_dtype", "auto") or "auto").lower()
    if want == "bf16":
        amp_dtype = torch.bfloat16
    elif want == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = bool(cfg["train"].get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    # --- Logging & checkpoints -------------------------------------------------
    Path("results").mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(
        str(Path("results") / "train_log.csv"),
        fieldnames=[
            "group", "epoch", "step", "loss_total", "ssim", "psnr", "running_avg",
            "vol_ssim", "vol_psnr", "vol_band", "vol_energy", "vol_ver", "vol_ipdr", "vol_tv",
        ],
    )

    recon_w = _renorm(cfg["losses"]["weights"])
    tv_w = float(cfg["losses"].get("tv", 0.0))
    fp_weight = float(cfg["losses"]["weights"].get("fp", 0.0))

    beta = 0.98  # EMA factor
    best_val = float("inf")
    best_state = None
    steps = 0
    ema = None

    flush_every = max(1, int(cfg["train"]["flush_every"]))
    empty_cache_every = max(1, int(cfg["train"]["empty_cache_every"]))
    grad_clip = float(cfg["train"]["grad_clip"])
    accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    epochs_per_group = int(cfg["train"].get("epochs_per_group", 1)) or int(cfg["train"]["epochs"])

    # Optimizer settings
    name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])

    # --- Group-wise training loop --------------------------------------------
    model = None
    opt = None

    for g_idx, (sino_paths, voxel_paths) in enumerate(groups):
        print(f"--- Training on group {g_idx + 1}/{len(groups)} (files {len(sino_paths)}) ---")

        # Dataset: concatenate depths of all files in the group (slice-level samples)
        ds = ConcatDepthSliceDataset(
            sino_paths=[str(p) for p in sino_paths],
            voxel_paths=[str(p) for p in voxel_paths],
            report=True,
        )
        dl = DataLoader(
            ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
        )

        # Build / update geometry + projector based on current group's shapes
        # Map dataset dims: U→X (detector-u == x), V=1 (single z-slice in training), A as-is
        U, A = ds.U, ds.A
        X, Y = ds.X, ds.Y
        Z = 1  # slice training
        angles = torch.linspace(0.0, math.pi, steps=A, device=device, dtype=torch.float32)

        geom = Parallel3DGeometry.from_xyz(
            vol_shape_xyz=(X, Y, Z),          # (X,Y,Z)
            det_shape_xz=(U, Z),              # (X,Z) == (U,1)
            angles=angles,
            voxel_size_xyz=tuple(map(float, cfg.get("geom", {}).get("voxel_size_xyz", (1.0, 1.0, 1.0)))),
            det_spacing_xz=tuple(map(float, cfg.get("geom", {}).get("det_spacing_xz", (1.0, 1.0)))),
            angle_chunk=int(cfg.get("geom", {}).get("angle_chunk", 16)),
            n_steps_cap=int(cfg.get("geom", {}).get("n_steps_cap", 256)),
        )

        if model is None:
            proj_method = str(cfg.get("projector", {}).get("method", "joseph3d")).lower()
            projector = make_projector(proj_method, geom).to(device)

            # Build HDNSystem with projector
            model = HDNSystem(cfg, projector=projector).to(device)
            if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
                model = torch.compile(model)

            # Optimizer init after model
            if name == "adafactor":
                from optim.adafactor import Adafactor
                opt = Adafactor(model.parameters(), lr=lr, weight_decay=wd)
            else:
                opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            # Update projector geometry in-place (angles and shapes)
            model.projector.reset_geometry(geom)

        # Train/Val split **over depths** (not per file) for this group
        D_total = len(ds)
        all_z = np.arange(D_total)
        rng = np.random.RandomState(int(cfg["train"]["seed"]) + g_idx)
        rng.shuffle(all_z)
        n_train = max(1, int(round(D_total * float(cfg["train"].get("train_ratio", 0.9)))))
        train_set = set(all_z[:n_train])

        for epoch in range(1, epochs_per_group + 1):
            model.train()
            pbar = tqdm(dl, desc=f"group {g_idx + 1}, epoch {epoch}", dynamic_ncols=True)
            opt.zero_grad(set_to_none=True)

            for batch in pbar:
                steps += 1

                # Batch tensors from dataset: S_ua [B,U,A], V_gt [B,1,X,Y]
                S_ua = batch["sino_ua"].to(device, non_blocking=True)    # [B, U, A]
                V_gt = batch["voxel_xy"].to(device, non_blocking=True)   # [B, 1, X, Y]

                # Canonicalize to model-facing (x,a,z)/(x,y,z) with Z=1
                S_xaz = S_ua.unsqueeze(1).unsqueeze(-1)                  # [B, 1, X, A, 1]
                V_xyz = V_gt.unsqueeze(-1)                                # [B, 1, X, Y, 1]

                # Depth index for train/val gating (take first in batch)
                z = int(batch["global_z"][0]) if hasattr(batch["global_z"], "__len__") else int(batch["global_z"])
                is_train = (z in train_set)

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    # Forward: cheat path ON only during training subset
                    sino_hat, recon = model(S_xaz, v_vol=(V_xyz if is_train else None), train_mode=is_train)
                    # recon: [B,1,X,Y,1] → 2D slice for image-space losses
                    R_hat2d = recon[..., 0]                               # [B,1,X,Y]

                    # Clamp and add D=1 for 3D-compatible losses
                    R_hat_n = torch.clamp(R_hat2d, 0.0, 1.0).unsqueeze(2)  # [B,1,1,X,Y]
                    V_gt_n  = torch.clamp(V_gt,   0.0, 1.0).unsqueeze(2)   # [B,1,1,X,Y]

                    params = {
                        "band_low": float(cfg["losses"]["band_low"]),
                        "band_high": float(cfg["losses"]["band_high"]),
                        "ver_thr": float(cfg["losses"]["ver_thr"]),
                        "tv_weight": float(cfg["losses"].get("tv", 0.0)),
                    }
                    recon_terms = reconstruction_losses(R_hat_n, V_gt_n, recon_w, params)

                    # Composite image-space losses (note: use psnr_loss instead of raw psnr)
                    total = (
                        recon_w.get("ssim", 0.0)  * recon_terms["ssim"]
                      + recon_w.get("psnr", 0.0)  * recon_terms["psnr_loss"]
                      + recon_w.get("band", 0.0)  * recon_terms["band"]
                      + recon_w.get("energy", 0.0)* recon_terms["energy"]
                      + recon_w.get("ver", 0.0)   * recon_terms["ver"]
                      + recon_w.get("ipdr", 0.0)  * recon_terms["ipdr"]
                      + recon_w.get("tv", 0.0)    * recon_terms["tv"]
                    )

                    # Optional L1/L2 pixel losses
                    l1_w = recon_w.get("l1", 0.0)
                    l2_w = recon_w.get("l2", 0.0)
                    if l1_w > 0.0:
                        l1_loss = (R_hat_n - V_gt_n).abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                        total = total + l1_w * l1_loss
                    if l2_w > 0.0:
                        l2_loss = ((R_hat_n - V_gt_n) ** 2).mean(dim=[1, 2, 3, 4], keepdim=True)
                        total = total + l2_w * l2_loss

                    # Optional forward-projection consistency (2D)
                    if fp_weight > 0.0 and is_train:
                        angles = model.projector.geom.angles.to(R_hat2d.device, R_hat2d.dtype)
                        fp_loss = forward_consistency_loss(R_hat2d, V_gt, angles)
                        total = total + fp_weight * fp_loss

                    loss = total.mean() / accum_steps

                # Backprop with optional AMP
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (steps % accum_steps) == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    if scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                # --- Running metrics / CSV logging (per step) ------------------
                with torch.no_grad():
                    val_total = float((total.mean() * accum_steps).item())
                    ema = val_total if ema is None else (beta * ema + (1.0 - beta) * val_total)
                    running_avg = ema / (1.0 - beta ** steps)
                    ssim_v = float(recon_terms["ssim"].mean().item())
                    psnr_v = float(recon_terms["psnr"].mean().item())

                csv_logger.log(
                    {
                        "group": g_idx + 1, "epoch": epoch, "step": steps,
                        "loss_total": val_total, "ssim": ssim_v, "psnr": psnr_v,
                        "running_avg": running_avg,
                    },
                    flush=(steps % flush_every == 0),
                )

                pbar.set_postfix({"avg": f"{running_avg:.4f}", "loss": f"{val_total:.4f}"}, refresh=False)

                # Free ASAP & periodic CUDA cache empty to stabilize memory
                del S_ua, V_gt, S_xaz, V_xyz, sino_hat, recon, R_hat2d, R_hat_n, V_gt_n, recon_terms, total, loss
                if (steps % empty_cache_every) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # --- Volumetric evaluation at the end of the epoch ----------------
            vol_metrics = evaluate_group_by_volume(model, ds, cfg, device)

            # Log volumetric metrics (always flush)
            csv_logger.log(
                {
                    "group": g_idx + 1, "epoch": epoch, "step": steps,
                    "loss_total": None, "ssim": None, "psnr": None, "running_avg": None,
                    "vol_ssim": vol_metrics["ssim"], "vol_psnr": vol_metrics["psnr"],
                    "vol_band": vol_metrics["band"], "vol_energy": vol_metrics["energy"],
                    "vol_ver": vol_metrics["ver"], "vol_ipdr": vol_metrics["ipdr"],
                    "vol_tv": vol_metrics["tv"],
                },
                flush=True,
            )

            # Track best checkpoint by (1 − SSIM) on volumetric eval
            current_val = 1.0 - vol_metrics["ssim"]
            if current_val < best_val:
                best_val = current_val
                best_state = {
                    "group": g_idx + 1,
                    "epoch": epoch,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "opt_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "config": cfg,
                }
                torch.save(best_state, str(ckpt_dir / "best_shared.pt"))

    csv_logger.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train HDN slice recon with grouped datasets (x,a,z)/(x,y,z).")
    ap.add_argument("--cfg", type=str, default="config.yaml", help="Path to training YAML config.")
    args = ap.parse_args()
    main(args.cfg)
