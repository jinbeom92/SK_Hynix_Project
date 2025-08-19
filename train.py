# =================================================================================================
# Training Script — Shared-Model, Multi-Resolution HDN Training
# -------------------------------------------------------------------------------------------------
# Purpose
#   Orchestrates end-to-end training of the HDNSystem with physics-consistent FP/BP.
#   Supports shared-model training across multiple resolutions/angle sets via per-batch
#   geometry rebinding. Implements mixed precision, gradient accumulation, checkpointing,
#   and CSV logging.
#
# Key Features
#   • Single shared model: one HDN instance trained across varying geometries.
#   • Geometry rebind: dynamically updates projector/backprojector each batch.
#   • Memory optimizations:
#       - Automatic Mixed Precision (AMP, bf16/fp16 auto-detect).
#       - Gradient accumulation (`grad_accum_steps`).
#       - Optional torch.compile on encoders/decoder for speed.
#       - Periodic gc.collect() and torch.cuda.empty_cache().
#   • Optimizer:
#       - Default AdamW, or Adafactor for reduced memory.
#       - Pluggable via cfg.train.optimizer.
#   • Losses:
#       - Reconstruction losses: SSIM, PSNR, band, energy, voxel error rate, IPDR, optional TV.
#       - Forward consistency loss in sinogram domain.
#       - Per-sample normalization based on 99% quantile of |sino|.
#   • Logging:
#       - CSV logger with per-step stats (loss, running avg, RMS residual).
#       - Saves effective config and train/val splits.
#   • Checkpoints:
#       - Best checkpoint saved as best_shared.pt.
#       - Optional periodic checkpoints every `ckpt_interval` epochs.
#
# Usage
#   python train.py --cfg config.yaml
#
# Config Keys (YAML)
#   geometry: { voxel_size, det_spacing }
#   projector: { method, angle_chunk, joseph.n_steps_cap, psf.enabled }
#   model: { enc1, enc2, enc3, align, dec }
#   train: {
#       seed, epochs, batch_size, lr, weight_decay,
#       optimizer: ["adamw"|"adafactor"],
#       grad_accum_steps, amp, amp_dtype, tf32,
#       compile, num_workers, prefetch_factor, persistent_workers,
#       ckpt_dir, ckpt_interval, flush_every, empty_cache_every, grad_clip,
#       shared_model: true
#   }
#   losses: { weights, band_low, band_high, ver_thr, tv, fwd_freq_alpha, renorm }
#
# Notes
#   • Requires dataset organized as data/sino/*_sino.npy and data/voxel/*_voxel.npy.
#   • Angles optionally in data/sino/*_angles.npy.
#   • Designed for PyTorch 2.0+ with AMP and torch.compile support.
# =================================================================================================
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
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
from data.dataset import NpySinoVoxelDataset
from physics.geometry import Parallel3DGeometry
from models.hdn import HDNSystem
from losses.forward import forward_consistency_loss
from losses.recon import reconstruction_losses
from losses.utils import per_sample_sino_scale, normalize_tensors


def _ensure_sino_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shape) == 2:  A, U = shape; V = 1
    elif len(shape) == 3: A, V, U = shape
    else: raise ValueError(f"Unsupported sino shape {shape}")
    return int(A), int(V), int(U)

def _ensure_voxel_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shape) == 2:  H, W = shape; D = 1
    elif len(shape) == 3: D, H, W = shape
    else: raise ValueError(f"Unsupported voxel shape {shape}")
    return int(D), int(H), int(W)

def _angles_for_id(data_root: Path, id_: str, A_guess: int) -> np.ndarray:
    p = data_root / "sino" / f"{id_}_angles.npy"
    if p.exists(): ang = np.load(str(p)).astype(np.float32)
    else:          ang = np.linspace(0, np.pi, A_guess, endpoint=False, dtype=np.float32)
    return ang

def make_geom_dynamic(cfg: Dict[str, Any],
                      vol_shape: Tuple[int,int,int],
                      det_shape: Tuple[int,int],
                      angles: torch.Tensor,
                      device: torch.device) -> Parallel3DGeometry:
    voxel_size = tuple(cfg["geometry"]["voxel_size"])
    det_spacing = tuple(cfg["geometry"]["det_spacing"])
    geom = Parallel3DGeometry(
        vol_shape=tuple(vol_shape),
        det_shape=tuple(det_shape),
        angles=angles.to(device),
        voxel_size=voxel_size,
        det_spacing=det_spacing,
        angle_chunk=cfg["projector"]["angle_chunk"],
        n_steps_cap=cfg["projector"].get("joseph",{}).get("n_steps_cap", 64),
    )
    return geom

def split_ids(all_ids: List[str], train_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(all_ids)); rng.shuffle(idx)
    n = len(all_ids)
    if n == 0: return [], []
    if n == 1: i = int(idx[0]); return [all_ids[i]], [all_ids[i]]
    n_train = max(1, min(n - 1, int(round(n * train_ratio))))
    train_ids = [all_ids[i] for i in idx[:n_train]]
    val_ids   = [all_ids[i] for i in idx[n_train:]]
    return train_ids, val_ids

def _maybe_renorm_loss_weights(w: Dict[str,float], do=True) -> Dict[str,float]:
    if not do: return w
    s = float(sum(float(v) for v in w.values()))
    if abs(s - 1.0) < 1e-6: return w
    return {k: float(v)/s for k,v in w.items()}

def _resolve_amp_dtype(device: torch.device, cfg_amp_dtype: str):
    want = (cfg_amp_dtype or "auto").lower()
    if want == "auto":
        if (device.type == "cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True
        return torch.float16, (device.type == "cuda")
    if want == "bf16": return torch.bfloat16, (device.type == "cuda")
    if want == "fp16": return torch.float16, (device.type == "cuda")
    return None, False  # fp32

def _build_optimizer(model, cfg: Dict[str, Any]):
    name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    if name == "adafactor":
        from optim.adafactor import Adafactor
        return Adafactor(model.parameters(), lr=lr, weight_decay=wd)
    # default
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def train_shared_model(cfg: Dict[str, Any]):
    seed = cfg["train"]["seed"]
    set_seed(seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- numeric knobs -----
    tf32 = bool(cfg["train"].get("tf32", True)) and (device.type == "cuda")
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    use_amp = bool(cfg["train"].get("amp", True)) and (device.type == "cuda")
    amp_dtype_cfg = cfg["train"].get("amp_dtype", "auto")
    amp_torch_dtype, amp_enabled = _resolve_amp_dtype(device, amp_dtype_cfg)
    scaler = GradScaler(enabled=(amp_enabled and amp_torch_dtype == torch.float16))

    data_root = Path(cfg["io"]["data_root"])

    # Discover ids
    sino_glob = sorted((data_root / "sino").glob("*_sino.npy"))
    all_ids = [p.stem.replace("_sino", "") for p in sino_glob]
    if not all_ids:
        raise RuntimeError(f"No training data found under {data_root}/sino/*.npy")

    # split with default 0.9 if omitted
    train_ratio = float(cfg["train"].get("train_ratio", 0.9))
    train_ids, val_ids = split_ids(all_ids, train_ratio, seed)
    ds_train = NpySinoVoxelDataset(train_ids, data_root=str(data_root))
    ds_val   = NpySinoVoxelDataset(val_ids,   data_root=str(data_root))

    # Probe first sample to build model (weights are geometry-agnostic)
    probe = ds_train[0]
    A0, V0, U0 = _ensure_sino_shape(tuple(probe["sino"].shape))
    D0, H0, W0 = _ensure_voxel_shape(tuple(probe["voxel"].shape))
    angles0 = probe["angles"] if probe["angles"] is not None else torch.linspace(0, np.pi, A0, dtype=torch.float32)
    geom0 = make_geom_dynamic(cfg, (D0,H0,W0), (V0,U0), angles0, device)

    out_dir = Path(cfg["train"]["ckpt_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps({"train": train_ids, "val": val_ids}, indent=2), encoding="utf-8")
    save_effective_config(cfg, str(out_dir / "effective_config.yaml"))

    model  = HDNSystem(geom0, cfg).to(device)
    proj = model.projector
    print(f"[projector] method={type(proj).__name__}  A={proj.geom.A}  "
          f"angle_chunk={proj.geom.angle_chunk}  n_steps={getattr(proj,'n_steps',None)}  "
          f"step_chunk={getattr(proj,'step_chunk',None)}  c_chunk={getattr(proj,'c_chunk',None)}")
    if cfg["projector"].get("method", "joseph3d") == "siddon3d":
        print("[warn] Siddon projector is Pythonic and slow—recommended for validation only, not large-scale training.")

    opt    = _build_optimizer(model, cfg)

    # Optional torch.compile for pure-conv blocks (dynamic shapes allowed)
    if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
        try:
            model.enc1 = torch.compile(model.enc1, mode="reduce-overhead", dynamic=True)
            model.enc2 = torch.compile(model.enc2, mode="reduce-overhead", dynamic=True)
            model.dec  = torch.compile(model.dec,  mode="reduce-overhead", dynamic=True)
            print("[compile] enc1/enc2/dec compiled.")
        except Exception as e:
            print(f"[compile] skipped due to: {e}")

    # DataLoaders
    num_workers = int(cfg["train"].get("num_workers", 0))
    prefetch_factor = int(cfg["train"].get("prefetch_factor", 2))
    persistent = bool(cfg["train"].get("persistent_workers", num_workers > 0))
    dl_train = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"),
                          prefetch_factor=prefetch_factor if num_workers > 0 else None,
                          persistent_workers=persistent if num_workers > 0 else False)
    dl_val   = DataLoader(ds_val,   batch_size=cfg["train"]["batch_size"], shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"),
                          prefetch_factor=prefetch_factor if num_workers > 0 else None,
                          persistent_workers=persistent if num_workers > 0 else False)

    # CSV logger
    extra_keys = cfg["train"].get("extra_log_keys", [])
    fieldnames = ["epoch","step","loss_total","loss_recon","loss_fwd","rms_resid","running_avg"]
    fieldnames.extend([k for k in extra_keys if k not in fieldnames])
    csv_logger = CSVLogger(str(out_dir / "train_log.csv"), fieldnames=fieldnames)
    flush_every = int(cfg["train"].get("flush_every", 50))
    empty_cache_every = int(cfg["train"].get("empty_cache_every", 200))

    # Loss weights: optional renorm
    cfg["losses"]["weights"] = _maybe_renorm_loss_weights(cfg["losses"]["weights"],
                                                          do=bool(cfg["losses"].get("renorm", True)))

    steps, ema, beta = 0, None, 0.98
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs = int(cfg["train"]["epochs"])
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 0))
    accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"[shared] epoch {epoch}/{epochs}", leave=False)
        opt.zero_grad(set_to_none=True)

        for batch in pbar:
            steps += 1
            S_in = batch["sino"].to(device, non_blocking=True)                # [B,A,V,U]
            angles_b = batch["angles"].to(device, non_blocking=True)
            V_gt = batch["voxel"].to(device, non_blocking=True).unsqueeze(1)  # [B,1,D,H,W]

            # Per-batch geometry rebind (one model for all resolutions)
            B, A, V, U = S_in.shape
            D, H, W = V_gt.shape[-3:]
            angles_vec = angles_b[0] if angles_b.ndim == 2 else angles_b
            geom_cur = make_geom_dynamic(cfg, (D,H,W), (V,U), angles_vec, device)
            model.rebind_geometry(geom_cur, angles_vec)

            with autocast(device_type=device.type, dtype=amp_torch_dtype, enabled=amp_enabled):
                sino_hat, R_hat, S_pred = model(S_in, angles_vec, V_gt=V_gt, train_mode=True)

                # Per-sample normalization (99% quantile of |S_in|)
                s_sino = per_sample_sino_scale(S_in, q=0.99)
                S_in_n, sino_hat_n, R_hat_n, V_gt_n = normalize_tensors(S_in, sino_hat, R_hat, V_gt, s_sino)
                S_pred_n = S_pred / s_sino

                # Losses
                recon_w = cfg["losses"]["weights"]
                recon_params = {
                    "band_low" : cfg["losses"]["band_low"],
                    "band_high": cfg["losses"]["band_high"],
                    "ver_thr"  : cfg["losses"]["ver_thr"],
                    "tv_weight": cfg["losses"].get("tv", 0.0),
                }
                recon = reconstruction_losses(R_hat_n, V_gt_n, recon_w, recon_params)
                fwd   = forward_consistency_loss(S_pred_n, S_in_n, alpha=cfg["losses"]["fwd_freq_alpha"])

                total = ( recon_w["ssim"]   * recon["ssim"]
                        + recon_w["psnr"]   * recon["psnr"]
                        + recon_w["band"]   * recon["band"]
                        + recon_w["energy"] * recon["energy"]
                        + recon_w["ver"]    * recon["ver"]
                        + recon_w["ipdr"]   * recon["ipdr"]
                        + recon_w.get("tv",0.0) * recon["tv"]
                        + recon_w["forward"]* fwd )
                loss = total.mean() / accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_step = ((steps % accum_steps) == 0)
            if do_step:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["train"]["grad_clip"])
                if scaler.is_enabled():
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            # Running stats
            with torch.no_grad():
                val_total = float((total.mean() * accum_steps).item())
                ema = val_total if ema is None else (beta*ema + (1.0-beta)*val_total)
                running_avg = ema / (1.0 - beta**steps)
                rms = ((S_pred_n - S_in_n)**2).mean().sqrt().item()

            # CSV row (buffered)
            row = {"epoch": epoch, "step": steps,
                   "loss_total": val_total,
                   "loss_recon": float(recon["total_recon"].mean().item()),
                   "loss_fwd":   float(fwd.mean().item()),
                   "rms_resid":  rms,
                   "running_avg": running_avg}
            if "ssim" in extra_keys: row["ssim"] = float(recon["ssim"].mean().item())
            if "psnr" in extra_keys: row["psnr"] = float(recon["psnr"].mean().item())
            csv_logger.log(row, flush=(steps % flush_every == 0))

            # Progress
            pbar.set_postfix({"avg": f"{running_avg:.4f}",
                              "loss": f"{val_total:.4f}",
                              "rms": f"{rms:.4f}"}, refresh=False)

            # Cleanup to mitigate OOM
            del S_in, angles_b, V_gt, sino_hat, R_hat, S_pred
            del s_sino, S_in_n, sino_hat_n, R_hat_n, V_gt_n, S_pred_n
            del recon, fwd, total, loss
            if (steps % empty_cache_every) == 0:
                gc.collect(); 
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ---------- light validation for model selection ----------
        model.eval()
        val_totals = []
        with torch.no_grad():
            for batch in dl_val:
                S_in = batch["sino"].to(device, non_blocking=True)
                angles_b = batch["angles"].to(device, non_blocking=True)
                V_gt = batch["voxel"].to(device, non_blocking=True).unsqueeze(1)
                B, A, V, U = S_in.shape
                D, H, W = V_gt.shape[-3:]
                angles_vec = angles_b[0] if angles_b.ndim == 2 else angles_b
                geom_cur = make_geom_dynamic(cfg, (D,H,W), (V,U), angles_vec, device)
                model.rebind_geometry(geom_cur, angles_vec)

                with autocast(device_type=device.type, dtype=amp_torch_dtype, enabled=amp_enabled):
                    sino_hat, R_hat, S_pred = model(S_in, angles_vec, V_gt=V_gt, train_mode=False)
                    s_sino = per_sample_sino_scale(S_in, q=0.99)
                    S_in_n, sino_hat_n, R_hat_n, V_gt_n = normalize_tensors(S_in, sino_hat, R_hat, V_gt, s_sino)
                    S_pred_n = S_pred / s_sino

                    recon_w = cfg["losses"]["weights"]
                    recon_params = {
                        "band_low" : cfg["losses"]["band_low"],
                        "band_high": cfg["losses"]["band_high"],
                        "ver_thr"  : cfg["losses"]["ver_thr"],
                        "tv_weight": cfg["losses"].get("tv", 0.0),
                    }
                    recon = reconstruction_losses(R_hat_n, V_gt_n, recon_w, recon_params)
                    fwd   = forward_consistency_loss(S_pred_n, S_in_n, alpha=cfg["losses"]["fwd_freq_alpha"])
                    total = ( recon_w["ssim"]   * recon["ssim"]
                            + recon_w["psnr"]   * recon["psnr"]
                            + recon_w["band"]   * recon["band"]
                            + recon_w["energy"] * recon["energy"]
                            + recon_w["ver"]    * recon["ver"]
                            + recon_w["ipdr"]   * recon["ipdr"]
                            + recon_w.get("tv",0.0) * recon["tv"]
                            + recon_w["forward"]* fwd ).mean()
                    val_totals.append(float(total.item()))

                del S_in, angles_b, V_gt, sino_hat, R_hat, S_pred
                del s_sino, S_in_n, sino_hat_n, R_hat_n, V_gt_n, S_pred_n
                del recon, fwd, total

        mean_val = float(np.mean(val_totals)) if val_totals else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {
                "epoch": epoch,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "opt_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": cfg,
                "seed": seed,
            }
            best_epoch = epoch

        # periodic checkpoint
        if ckpt_interval > 0 and (epoch % ckpt_interval == 0):
            mid_path = Path(cfg["train"]["ckpt_dir"]) / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "opt_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": cfg,
                "seed": seed,
            }, str(mid_path))

        gc.collect(); 
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    csv_logger.close()
    if best_state is not None:
        best_path = Path(cfg["train"]["ckpt_dir"]) / "best_shared.pt"
        torch.save(best_state, str(best_path))
        print(f"[shared] saved best checkpoint @ epoch {best_epoch}: {best_path} (val_total={best_val:.6f})")

    del model, opt, scaler, dl_train, dl_val, ds_train, ds_val
    gc.collect(); 
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def train(cfg_path: str):
    cfg = load_config(cfg_path)
    if not cfg["train"].get("shared_model", False):
        raise RuntimeError("Set train.shared_model: true for single-model multi-resolution training.")
    train_shared_model(cfg)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="hdn/config/default.yaml")
    args = ap.parse_args()
    train(args.cfg)
