# =================================================================================================
# Training Script — Grouped per-Depth Slice Reconstruction (No Physics)
# -------------------------------------------------------------------------------------------------
# Features:
#   • Splits the dataset into groups of `files_per_group` sinogram/voxel pairs and trains sequentially.
#   • Keeps a single model/optimizer across groups (weights accumulate).
#   • Uses slice-level loss during training for efficiency.
#   • After training a group, reconstructs each file's full volume and computes 3D metrics (SSIM, PSNR, etc.)
#     against the GT volume; averages them to report group-level evaluation.
# =================================================================================================

import json
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
    """Normalize loss weights to sum to 1 if they don't already."""
    s = float(sum(float(v) for v in w.values()))
    if s <= 0: return w
    if abs(s - 1.0) < 1e-6: return w
    return {k: float(v)/s for k,v in w.items()}

def evaluate_group_by_volume(model, ds, cfg, device):
    """
    Evaluate the model on an entire group at once, computing volumetric losses per file.
    Returns dict with averaged metrics across files in the group.
    """
    model.eval()
    recon_w = _renorm(cfg["losses"]["weights"])
    tv_w = float(cfg["losses"].get("tv", 0.0))

    metrics_sum = {"ssim": 0.0, "psnr": 0.0, "band": 0.0, "energy": 0.0,
                   "ver": 0.0, "ipdr": 0.0, "tv": 0.0, "count": 0}

    with torch.no_grad():
        for f_idx, (s_mem, v_mem) in enumerate(zip(ds.sinos, ds.voxels)):
            # GT volume [X,Y,D_f]
            gt_vol = np.array(v_mem, dtype=np.float32, copy=True, order='C')
            D_f = gt_vol.shape[-1]
            pred_vol = np.zeros_like(gt_vol)
            for d in range(D_f):
                s_slice = np.array(s_mem[:, :, d], dtype=np.float32, copy=True, order='C')
                s_tensor = torch.from_numpy(s_slice).unsqueeze(0).to(device)  # [1,U,A]
                r_hat2d = model(s_tensor, v_slice=None, train_mode=False)     # [1,1,X,Y]
                pred_vol[:, :, d] = r_hat2d[0,0].cpu().numpy()

            # [X,Y,D_f] → [1,1,D_f,X,Y] for reconstruction_losses
            pred_tensor = torch.from_numpy(pred_vol).permute(2,0,1).unsqueeze(0).unsqueeze(0).to(device)
            gt_tensor   = torch.from_numpy(gt_vol).permute(2,0,1).unsqueeze(0).unsqueeze(0).to(device)

            params = {
                "band_low" : float(cfg["losses"]["band_low"]),
                "band_high": float(cfg["losses"]["band_high"]),
                "ver_thr"  : float(cfg["losses"]["ver_thr"]),
                "tv_weight": tv_w,
            }
            recon = reconstruction_losses(pred_tensor, gt_tensor, recon_w, params)
            metrics_sum["ssim"]   += float(recon["ssim"].mean().item())
            metrics_sum["psnr"]   += float(recon["psnr"].mean().item())
            metrics_sum["band"]   += float(recon["band"].mean().item())
            metrics_sum["energy"] += float(recon["energy"].mean().item())
            metrics_sum["ver"]    += float(recon["ver"].mean().item())
            metrics_sum["ipdr"]   += float(recon["ipdr"].mean().item())
            metrics_sum["tv"]     += float(recon["tv"].mean().item())
            metrics_sum["count"]  += 1

    for k in ["ssim","psnr","band","energy","ver","ipdr","tv"]:
        if metrics_sum["count"] > 0:
            metrics_sum[k] /= metrics_sum["count"]
    return metrics_sum

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    save_effective_config(cfg, Path(cfg_path).with_name("effective_config.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["train"]["seed"]))

    # List all sino/voxel paths and chunk into groups
    data_root = cfg.get("data", {}).get("root", "data")
    sino_glob = cfg.get("data", {}).get("sino_glob", "sino/*_sino.npy")
    voxel_glob = cfg.get("data", {}).get("voxel_glob", "voxel/*_voxel.npy")
    all_sino = sorted(Path(data_root).glob(sino_glob))
    all_vox  = sorted(Path(data_root).glob(voxel_glob))
    if len(all_sino) != len(all_vox):
        raise AssertionError(f"#sino({len(all_sino)}) != #voxel({len(all_vox)})")
    files_per_group = int(cfg["train"].get("files_per_group", 100))
    groups = [(all_sino[i:i+files_per_group], all_vox[i:i+files_per_group])
              for i in range(0, len(all_sino), files_per_group)]

    # Build model once
    model = HDNSystem(cfg).to(device)
    if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Optimizer & AMP
    name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"]["lr"]); wd = float(cfg["train"]["weight_decay"])
    if name == "adafactor":
        from optim.adafactor import Adafactor
        opt = Adafactor(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    want = (cfg["train"].get("amp_dtype", "auto") or "auto").lower()
    if want == "bf16": amp_dtype = torch.bfloat16
    elif want == "fp16": amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = bool(cfg["train"].get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    # Logging
    Path("results").mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["train"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(str(Path("results")/"train_log.csv"),
                           fieldnames=["group","epoch","step","loss_total","ssim","psnr","running_avg",
                                       "vol_ssim","vol_psnr","vol_band","vol_energy","vol_ver","vol_ipdr","vol_tv"])

    recon_w = _renorm(cfg["losses"]["weights"])
    tv_w = float(cfg["losses"].get("tv", 0.0))
    beta = 0.98
    best_val = float("inf"); best_state = None
    steps = 0; ema = None
    flush_every = int(cfg["train"]["flush_every"])
    empty_cache_every = int(cfg["train"]["empty_cache_every"])
    grad_clip = float(cfg["train"]["grad_clip"])
    accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    epochs_per_group = int(cfg["train"].get("epochs_per_group", 1)) or int(cfg["train"]["epochs"])

    for g_idx, (sino_paths, voxel_paths) in enumerate(groups):
        print(f"--- Training on group {g_idx+1}/{len(groups)} (files {len(sino_paths)}) ---")

        # Dataset & DataLoader for this group
        ds = ConcatDepthSliceDataset(
            sino_paths=[str(p) for p in sino_paths],
            voxel_paths=[str(p) for p in voxel_paths],
            report=True,
        )
        dl = DataLoader(ds,
                        batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False,
                        num_workers=int(cfg["train"]["num_workers"]),
                        pin_memory=True)

        # train/val split over depths
        D_total = len(ds)
        all_z = np.arange(D_total)
        rng = np.random.RandomState(int(cfg["train"]["seed"]) + g_idx)
        rng.shuffle(all_z)
        n_train = max(1, int(round(D_total * float(cfg["train"].get("train_ratio", 0.9)))))
        train_set = set(all_z[:n_train])

        for epoch in range(1, epochs_per_group + 1):
            model.train()
            pbar = tqdm(dl, desc=f"group {g_idx+1}, epoch {epoch}", dynamic_ncols=True)
            opt.zero_grad(set_to_none=True)

            for batch in pbar:
                steps += 1
                S_ua = batch["sino_ua"].to(device, non_blocking=True)     # [B,U,A]
                V_gt = batch["voxel_xy"].to(device, non_blocking=True)    # [B,1,X,Y]
                z = int(batch["global_z"][0]) if hasattr(batch["global_z"], "__len__") else int(batch["global_z"])
                is_train = (z in train_set)

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    R_hat = model(S_ua, v_slice=(V_gt if is_train else None), train_mode=is_train)  # [B,1,X,Y]
                    R_hat_n = torch.clamp(R_hat, 0.0, 1.0).unsqueeze(2)  # [B,1,1,X,Y]
                    V_gt_n  = torch.clamp(V_gt,  0.0, 1.0).unsqueeze(2)  # [B,1,1,X,Y]
                    params = {
                        "band_low" : float(cfg["losses"]["band_low"]),
                        "band_high": float(cfg["losses"]["band_high"]),
                        "ver_thr"  : float(cfg["losses"]["ver_thr"]),
                        "tv_weight": tv_w,
                    }
                    recon = reconstruction_losses(R_hat_n, V_gt_n, recon_w, params)
                    total = ( recon_w.get("ssim",0)*recon["ssim"]
                            + recon_w.get("psnr",0)*recon["psnr"]
                            + recon_w.get("band",0)*recon["band"]
                            + recon_w.get("energy",0)*recon["energy"]
                            + recon_w.get("ver",0)*recon["ver"]
                            + recon_w.get("ipdr",0)*recon["ipdr"]
                            + recon_w.get("tv",0)*recon["tv"] )
                    loss = total.mean() / accum_steps

                if scaler.is_enabled(): scaler.scale(loss).backward()
                else: loss.backward()

                if (steps % accum_steps) == 0:
                    if scaler.is_enabled(): scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    if scaler.is_enabled(): scaler.step(opt); scaler.update()
                    else: opt.step()
                    opt.zero_grad(set_to_none=True)

                with torch.no_grad():
                    val_total = float((total.mean() * accum_steps).item())
                    ema = val_total if ema is None else (beta*ema + (1.0-beta)*val_total)
                    running_avg = ema / (1.0 - beta**steps)
                    ssim_v = float(recon["ssim"].mean().item())
                    psnr_v = float(recon["psnr"].mean().item())

                # flush every 'flush_every' steps
                csv_logger.log({"group": g_idx+1, "epoch": epoch, "step": steps,
                                "loss_total": val_total, "ssim": ssim_v,
                                "psnr": psnr_v, "running_avg": running_avg},
                               flush=(steps % flush_every == 0))

                pbar.set_postfix({"avg": f"{running_avg:.4f}", "loss": f"{val_total:.4f}"}, refresh=False)

                del S_ua, V_gt, R_hat, R_hat_n, V_gt_n, recon, total, loss
                if (steps % empty_cache_every) == 0:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

            # volumetric evaluation for this group
            vol_metrics = evaluate_group_by_volume(model, ds, cfg, device)
            # update CSV with volumetric metrics (always flush)
            csv_logger.log({"group": g_idx+1, "epoch": epoch, "step": steps,
                            "loss_total": None, "ssim": None, "psnr": None, "running_avg": None,
                            "vol_ssim": vol_metrics["ssim"], "vol_psnr": vol_metrics["psnr"],
                            "vol_band": vol_metrics["band"], "vol_energy": vol_metrics["energy"],
                            "vol_ver": vol_metrics["ver"], "vol_ipdr": vol_metrics["ipdr"],
                            "vol_tv": vol_metrics["tv"]},
                           flush=True)

            # track best model by volumetric metric (smaller 1-SSIM is better)
            current_val = 1.0 - vol_metrics["ssim"]
            if current_val < best_val:
                best_val = current_val
                best_state = {
                    "group": g_idx+1,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    args = ap.parse_args()
    main(args.cfg)
