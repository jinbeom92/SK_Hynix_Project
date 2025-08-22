from pathlib import Path
import numpy as np
import torch, gc
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler
from torch import autocast

from utils.yaml_config import load_config, save_effective_config
from utils.seed import set_seed
from utils.logging import CSVLogger
from data.dataset import ConcatDepthSliceDataset
from models.hdn import HDNSystem
from losses.recon import reconstruction_losses


def _renorm(w: dict) -> dict:
    """
    Normalize a dict of loss weights so they sum to 1.0 (no-op if already ≈1).

    Args:
        w (dict): key → weight (float-like)

    Returns:
        dict: normalized weights (or original if sum≈1 or sum<=0).
    """
    s = float(sum(float(v) for v in w.values()))
    if s <= 0:
        return w
    if abs(s - 1.0) < 1e-6:
        return w
    return {k: float(v) / s for k, v in w.items()}


def evaluate_group_by_volume(model, ds, cfg, device):
    """
    Evaluate a trained model on a group of sinogram/voxel file pairs.

    This function iterates over each file pair in the given dataset and
    performs **slice‑by‑slice inference** with cheat injection disabled
    (i.e. ``train_mode=False``).  For a single file consisting of a
    sinogram memmap ``[U, A, D_f]`` and voxel ground‑truth memmap
    ``[X, Y, D_f]``, we loop over all depth slices ``d=0…D_f−1`` to
    reconstruct each slice and assemble a predicted volume ``[X, Y, D_f]``.
    We then reshape both the predicted and ground‑truth volumes to
    ``[1, 1, D_f, X, Y]`` and compute the same composite losses used during
    training (SSIM, PSNR, band, energy, ver, ipdr, TV).  Metrics are
    averaged across all files in the group and returned as a summary
    dictionary.

    Notes
    -----
    * Cheat injection is disabled in evaluation mode to avoid leaking
      ground‑truth information into the reconstruction【470627143552496†L75-L99】.  The
      predicted slice is obtained solely from the sinogram input.
    * The volumetric losses use the same weights as training, ensuring
      consistency between slice‑level and volume‑level evaluations.
    """
    model.eval()
    recon_w = _renorm(cfg["losses"]["weights"])
    tv_w = float(cfg["losses"].get("tv", 0.0))

    metrics_sum = {
        "ssim": 0.0, "psnr": 0.0, "band": 0.0, "energy": 0.0,
        "ver": 0.0, "ipdr": 0.0, "tv": 0.0, "count": 0
    }

    with torch.no_grad():
        for f_idx, (s_mem, v_mem) in enumerate(zip(ds.sinos, ds.voxels)):
            # GT volume [X, Y, D_f] (memmap → owning ndarray for safe tensor conversion)
            gt_vol = np.array(v_mem, dtype=np.float32, copy=True, order='C')
            D_f = gt_vol.shape[-1]
            pred_vol = np.zeros_like(gt_vol)

            # Slice-wise inference (cheat OFF)
            for d in range(D_f):
                s_slice = np.array(s_mem[:, :, d], dtype=np.float32, copy=True, order='C')  # [U, A]
                s_tensor = torch.from_numpy(s_slice).unsqueeze(0).to(device)               # [1, U, A]
                r_hat2d = model(s_tensor, v_slice=None, train_mode=False)                  # [1, 1, X, Y]
                pred_vol[:, :, d] = r_hat2d[0, 0].cpu().numpy()

            # [X, Y, D_f] → [1, 1, D_f, X, Y] (match losses.recon expected shape)
            pred_tensor = torch.from_numpy(pred_vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
            gt_tensor   = torch.from_numpy(gt_vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)

            params = {
                "band_low": float(cfg["losses"]["band_low"]),
                "band_high": float(cfg["losses"]["band_high"]),
                "ver_thr": float(cfg["losses"]["ver_thr"]),
                "tv_weight": tv_w,
            }
            recon = reconstruction_losses(pred_tensor, gt_tensor, recon_w, params)

            # Accumulate scalar metrics per volume
            metrics_sum["ssim"]   += float(recon["ssim"].mean().item())
            metrics_sum["psnr"]   += float(recon["psnr"].mean().item())
            metrics_sum["band"]   += float(recon["band"].mean().item())
            metrics_sum["energy"] += float(recon["energy"].mean().item())
            metrics_sum["ver"]    += float(recon["ver"].mean().item())
            metrics_sum["ipdr"]   += float(recon["ipdr"].mean().item())
            metrics_sum["tv"]     += float(recon["tv"].mean().item())
            metrics_sum["count"]  += 1

    # Average over files
    for k in ["ssim", "psnr", "band", "energy", "ver", "ipdr", "tv"]:
        if metrics_sum["count"] > 0:
            metrics_sum[k] /= metrics_sum["count"]
    return metrics_sum


def main(cfg_path: str):
    """
    Run the training loop over grouped depth‑slices with periodic volumetric evaluation.

    This script implements the **HDN/SVTR** training procedure for tomographic
    reconstruction.  It operates on pairs of sinograms and voxel volumes
    loaded by :class:`ConcatDepthSliceDataset`.  Each sinogram has shape
    ``[U, A, D]`` where ``U`` is the number of detector bins, ``A`` the number of
    projection angles and ``D`` the number of depth slices, and each voxel
    volume has shape ``[X, Y, D]``【470627143552496†L17-L23】.  The dataset returns one
    depth slice at a time: a sinogram slice ``[U, A]``, a voxel slice
    ``[1, X, Y]`` and bookkeeping indices for the file pair and depth
    【470627143552496†L75-L99】.

    The goal of training is to predict high‑quality reconstructions of the
    voxel slice from its corresponding sinogram slice.  We clamp the
    reconstructions to the interval ``[0, 1]`` to mimic the non‑negativity
    constraints of filtered backprojection; conventional ramp‑filtered FBP
    introduces negative intensities that must be clipped【508774924062640†L39-L44】.  We
    use a composite loss consisting of SSIM, PSNR, band, energy, ver, ipdr and
    total variation, following the HDN paper.  Unlike previous variants of
    this project, **we do not use VAMToolbox optimized reconstructions as
    supervision**; instead, the ground‑truth voxel slices are the sole
    targets.  This simplifies the training loop to its original form and
    focuses on learning a good mapping from sinogram domain to image domain.

    For reference, the optional forward‑projection consistency term would
    compare the forward projection of the reconstructed slice with the
    input sinogram using a differentiable projector.  ASTRA Toolbox
    implements a CPU forward projection algorithm that “takes a projector and
    a volume as input, and returns the projection data”【941924045700879†L38-L40】.
    We omit this term in the current training loop, but the commented
    structure makes it easy to reintroduce a differentiable projector when
    needed.

    Pipeline overview:

    1. **Load YAML configuration** and save an “effective” copy for
    reproducibility.
    2. **Seed random number generators** and select CUDA or CPU device.
    3. **Enumerate sinogram/voxel files** under ``data_root`` and split into
    groups of at most ``files_per_group`` pairs.
    4. For each group:
    a. **Construct a dataset** by concatenating all depth slices of
        sinogram/voxel pairs and wrap it in a DataLoader.
    b. **Train** for ``epochs_per_group`` epochs over the DataLoader.
    c. **Log** per‑step losses (EMA running average) to a CSV file.
    d. At the end of each epoch, perform **volumetric evaluation** by
        reconstructing entire volumes slice‑by‑slice (cheat OFF) and
        computing the same losses.  The best checkpoint is selected
        based on the volumetric SSIM.

    This file, like the rest of the project, uses English docstrings with
    citations to external references.  Please consult the VAMToolbox
    repository for details on its optimized reconstructions, the HDN paper
    for architectural design choices, and ASTRA Toolbox documentation for
    forward projection implementations.
    """
    # --- Config IO ------------------------------------------------------------
    cfg = load_config(cfg_path)
    # NOTE: we write YAML content to a file named *effective_config.json* (kept for compatibility).
    save_effective_config(cfg, Path(cfg_path).with_name("effective_config.json"))

    # --- Repro & device -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["train"]["seed"]))

    # --- File enumeration & grouping ------------------------------------------
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

    # --- Model ----------------------------------------------------------------
    # If your HDNSystem constructor expects only cfg, adapt accordingly.
    model = HDNSystem(cfg).to(device)
    if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
        # Optional torch.compile for potential speedups (PyTorch 2.x+)
        model = torch.compile(model)

    # --- Optimizer & AMP ------------------------------------------------------
    name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])

    if name == "adafactor":
        from optim.adafactor import Adafactor
        opt = Adafactor(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

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

    # Exponential moving average for smoother display
    beta = 0.98
    best_val = float("inf")
    best_state = None
    steps = 0
    ema = None

    flush_every = int(cfg["train"]["flush_every"])
    empty_cache_every = int(cfg["train"]["empty_cache_every"])
    grad_clip = float(cfg["train"]["grad_clip"])
    accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    epochs_per_group = int(cfg["train"].get("epochs_per_group", 1)) or int(cfg["train"]["epochs"])

    # --- Group-wise training loop ---------------------------------------------
    for g_idx, (sino_paths, voxel_paths) in enumerate(groups):
        print(f"--- Training on group {g_idx + 1}/{len(groups)} (files {len(sino_paths)}) ---")

        # Dataset: concatenate depths of all files in the group
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

                # Batch tensors
                S_ua = batch["sino_ua"].to(device, non_blocking=True)   # [B, U, A]
                V_gt = batch["voxel_xy"].to(device, non_blocking=True)  # [B, 1, X, Y]

                # Depth index for train/val gating
                z = int(batch["global_z"][0]) if hasattr(batch["global_z"], "__len__") else int(batch["global_z"])
                is_train = (z in train_set)

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    # Forward: cheat path ON only during training subset
                    R_hat = model(S_ua, v_slice=(V_gt if is_train else None), train_mode=is_train)  # [B,1,X,Y]

                    # Clamp to [0,1] and add a dummy depth=1 axis for 3D losses
                    R_hat_n = torch.clamp(R_hat, 0.0, 1.0).unsqueeze(2)  # [B,1,1,X,Y]
                    V_gt_n  = torch.clamp(V_gt,  0.0, 1.0).unsqueeze(2)  # [B,1,1,X,Y]

                    params = {
                        "band_low": float(cfg["losses"]["band_low"]),
                        "band_high": float(cfg["losses"]["band_high"]),
                        "ver_thr": float(cfg["losses"]["ver_thr"]),
                        "tv_weight": float(cfg["losses"].get("tv", 0.0)),
                    }
                    recon = reconstruction_losses(R_hat_n, V_gt_n, recon_w, params)

                    # IMPORTANT:
                    # If you want "higher PSNR → lower loss", use recon["psnr_loss"] instead of recon["psnr"].
                    total = (
                        recon_w.get("ssim", 0)  * recon["ssim"]
                      + recon_w.get("psnr", 0)  * recon["psnr_loss"]
                      + recon_w.get("band", 0)  * recon["band"]
                      + recon_w.get("energy", 0)* recon["energy"]
                      + recon_w.get("ver", 0)   * recon["ver"]
                      + recon_w.get("ipdr", 0)  * recon["ipdr"]
                      + recon_w.get("tv", 0)    * recon["tv"]
                    )
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
                    ssim_v = float(recon["ssim"].mean().item())
                    psnr_v = float(recon["psnr"].mean().item())

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
                del S_ua, V_gt, R_hat, R_hat_n, V_gt_n, recon, total, loss
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
    ap = argparse.ArgumentParser(description="Train SVTR/HDN slice recon with grouped datasets.")
    ap.add_argument("--cfg", type=str, default="config.yaml", help="Path to training YAML config.")
    args = ap.parse_args()
    main(args.cfg)
