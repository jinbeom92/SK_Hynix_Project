"""
HDN training script (slice-wise, unfiltered BP) with explicit 80/20 split
and NearExpandMaskedCompositeLossV2 (boundary‑aware + SSIM/PSNR composite).

Core behavior
-------------
• Dataset contract: ConcatDepthSliceDataset yields per-slice
  {sino_ua[U,A], voxel_xy[1,X,Y], global_z}. (U≡X)
  Batching maps to [B,1,X,A,1] / [B,1,X,Y,1] for model I/O.
• Geometry (Z=1) uses evenly spaced angles with span selected by cfg.projector.bp_span
  ∈ {"half","full","auto"}; angles are generated with endpoint=False.
• Projector BP is **unfiltered**, circular XY mask, and span‑aware scaling θ_span/(2·A_eff)
  (mirrors scikit‑image’s iradon(filter=None) convention).
• Loss: NearExpandMaskedCompositeLossV2
  (masked MSE + optional SSIM/PSNR terms; autograd flows only through `pred`).

Notes on model blocks
---------------------
The system expects Sino→XY alignment and 2D slice decoding blocks consistent with your
Sino2XYAlign (mixing on (X,A) then resize to (X,Y)) and DecoderSlice2D; angle‑wise 1D
encoders follow Enc1_1D_Angle. These contracts remain unchanged.
"""

from pathlib import Path
import math
import gc
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.amp import GradScaler
from torch import autocast

from utils.yaml_config import load_config, save_effective_config
from utils.seed import set_seed
from utils.logging import CSVLogger
from data.dataset import ConcatDepthSliceDataset
from models.hdn import HDNSystem
from physics.geometry import Parallel3DGeometry
from physics.projector import make_projector
from losses.losses import (
    NearExpandMaskedCompositeLossV2,
    edge_contrast_slice_max_torch,
)

# ---------------------------------------
# Small utilities
# ---------------------------------------
def _as_float(x):
    return float(x) if not torch.is_tensor(x) else float(x.detach().item())


def _init_amp(device: torch.device, cfg: dict):
    want = (cfg["train"].get("amp_dtype", "auto") or "auto").lower()
    if want == "bf16":
        amp_dtype = torch.bfloat16
    elif want == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = bool(cfg["train"].get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    return amp_enabled, amp_dtype, scaler


def _angles_from_cfg(A: int, device: torch.device, cfg: dict) -> torch.Tensor:
    import math
    span_key = str(cfg.get("projector", {}).get("bp_span", "auto")).lower()
    if span_key == "full":
        theta_span = 2.0 * math.pi
    elif span_key == "half":
        theta_span = math.pi
    else:
        theta_span = (2.0 * math.pi) if A >= 300 else math.pi
    angles = torch.linspace(0.0, theta_span, steps=A + 1, device=device, dtype=torch.float32)[:-1]
    return angles


# ---------------------------------------
# Validation
# ---------------------------------------
@torch.no_grad()
def evaluate_slicewise_composite(
    model: torch.nn.Module,
    dl_val: DataLoader,
    device: torch.device,
    criterion: NearExpandMaskedCompositeLossV2,
    clamp_pred_gt: bool = True,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict:
    model.eval()
    tot_loss, tot_mse, tot_ssim, tot_psnr, tot_ec, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    have_ssim = False
    have_psnr = False
    have_ec = False

    for batch in dl_val:
        S_ua = batch["sino_ua"].to(device, non_blocking=True)   # [B,U,A] ≡ [B,X,A]
        V_gt = batch["voxel_xy"].to(device, non_blocking=True)  # [B,1,X,Y]
        B = int(S_ua.shape[0])

        S_xaz = S_ua.unsqueeze(1).unsqueeze(-1)  # [B,1,X,A,1]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            _, recon = model(S_xaz, v_vol=None, train_mode=False)  # [B,1,X,Y,1]
            R2 = recon[..., 0]                                     # [B,1,X,Y]
            if clamp_pred_gt:
                R2c = R2.clamp(0.0, 1.0)
                Vc = V_gt.clamp(0.0, 1.0)
            else:
                R2c = R2
                Vc = V_gt
            loss_b, info = criterion(R2c, Vc)
            # Edge contrast metric for eval
            mask = (V_gt > 0.0).float()
            ec_tensor = edge_contrast_slice_max_torch(R2, mask, reduction="batch_mean")
            ec_val = _as_float(ec_tensor)

            loss_val = _as_float(loss_b)
            mse_val = _as_float(info["mse"]) if (info.get("mse") is not None) else float("nan")
            ssim_val = _as_float(info.get("ssim")) if (info.get("ssim") is not None) else None
            psnr_val = _as_float(info.get("psnr")) if (info.get("psnr") is not None) else None

        tot_loss += loss_val * B
        if not math.isnan(mse_val):
            tot_mse += mse_val * B
        if ssim_val is not None:
            tot_ssim += ssim_val * B
            have_ssim = True
        if psnr_val is not None:
            tot_psnr += psnr_val * B
            have_psnr = True
        tot_ec += ec_val * B
        have_ec = True
        count += B

    return {
        "loss": (tot_loss / count) if count else float("nan"),
        "mse": (tot_mse / count) if count else float("nan"),
        "ssim": (tot_ssim / count) if (count and have_ssim) else None,
        "psnr": (tot_psnr / count) if (count and have_psnr) else None,
        "ec": (tot_ec / count) if (count and have_ec) else None,
    }


# ---------------------------------------
# Main
# ---------------------------------------
def main(cfg_path: str):
    cfg = load_config(cfg_path)
    # Save the effective configuration for record-keeping
    save_effective_config(cfg, Path(cfg_path).with_name("effective_config.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["train"]["seed"]))
    amp_enabled, amp_dtype, scaler = _init_amp(device, cfg)

    data_root = cfg.get("data", {}).get("root", "data")
    sino_glob = cfg.get("data", {}).get("sino_glob", "sino/*_sino.npy")
    voxel_glob = cfg.get("data", {}).get("voxel_glob", "voxel/*_voxel.npy")

    all_sino = sorted(Path(data_root).glob(sino_glob))
    all_vox = sorted(Path(data_root).glob(voxel_glob))
    if len(all_sino) != len(all_vox):
        raise AssertionError(f"#sino({len(all_sino)}) != #voxel({len(all_vox)})")
    if len(all_sino) == 0:
        raise FileNotFoundError(
            f"No training files found.\n"
            f"  root={Path(data_root).resolve()}\n"
            f"  sino_glob={sino_glob}\n"
            f"  voxel_glob={voxel_glob}"
        )

    files_per_group = int(cfg["train"].get("files_per_group", 100))
    groups = [
        (all_sino[i: i + files_per_group], all_vox[i: i + files_per_group])
        for i in range(0, len(all_sino), files_per_group)
    ]

    Path("results").mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(
        str(Path("results") / "train_log.csv"),
        fieldnames=[
            "group", "epoch", "step",
            "running_avg", "loss_total", "mse", "psnr", "ssim",
            "ec", "val_loss", "val_mse", "val_psnr", "val_ssim", "val_ec",
        ],
    )

    # Instantiate composite loss
    lcfg = cfg.get("losses", {})
    criterion = NearExpandMaskedCompositeLossV2(
        thr=float(lcfg.get("expand_thr", 0.8)),
        near_value=float(lcfg.get("expand_near_value", 0.8)),
        spacing=lcfg.get("expand_spacing", None),
        max_val=float(lcfg.get("max_val", 1.0)),
        ssim_win_size=int(lcfg.get("ssim_win_size", 7)),
        ssim_sigma=float(lcfg.get("ssim_sigma", 1.5)),
        ssim_grad=bool(lcfg.get("ssim_grad", True)),
        psnr_grad=bool(lcfg.get("psnr_grad", True)),
        psnr_ref=float(lcfg.get("psnr_ref", 40.0)),
        w_mse=float(lcfg.get("w_mse", 1.0)),
        w_ssim=float(lcfg.get("w_ssim", 0.0)),
        w_psnr=float(lcfg.get("w_psnr", 0.0)),
        reduction=str(lcfg.get("reduction", "mean")),
        weighted_mean_by_mask=bool(lcfg.get("weighted_mean_by_mask", True)),
        eps=float(lcfg.get("eps", 1e-8)),
    ).to(device)
    clamp_pred_gt = bool(lcfg.get("expand_clamp", True))
    # Weight for optional edge contrast loss
    w_edge_contrast = float(lcfg.get("w_edge_contrast", 0.0))

    name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])

    grad_clip = float(cfg["train"]["grad_clip"])
    accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    epochs_per_group = int(cfg["train"].get("epochs_per_group", 1)) or int(cfg["train"]["epochs"])
    flush_every = max(1, int(cfg["train"]["flush_every"]))
    empty_cache_every = max(1, int(cfg["train"]["empty_cache_every"]))

    best_val = float("inf")
    steps = 0
    beta = 0.98
    ema = None

    model = None
    opt = None

    for g_idx, (sino_paths, voxel_paths) in enumerate(groups):
        print(f"--- Training on group {g_idx + 1}/{len(groups)} (files {len(sino_paths)}) ---")

        ds = ConcatDepthSliceDataset(
            sino_paths=[str(p) for p in sino_paths],
            voxel_paths=[str(p) for p in voxel_paths],
            report=True,
        )

        U, A = ds.U, ds.A
        X, Y = ds.X, ds.Y
        Z = 1
        angles = _angles_from_cfg(A, device, cfg)

        geom = Parallel3DGeometry.from_xyz(
            vol_shape_xyz=(X, Y, Z),
            det_shape_xz=(U, Z),
            angles=angles,
            voxel_size_xyz=tuple(map(float, cfg.get("geom", {}).get("voxel_size_xyz", (1.0, 1.0, 1.0)))),
            det_spacing_xz=tuple(map(float, cfg.get("geom", {}).get("det_spacing_xz", (1.0, 1.0)))),
            angle_chunk=int(cfg.get("geom", {}).get("angle_chunk", 16)),
            n_steps_cap=int(cfg.get("geom", {}).get("n_steps_cap", 256)),
        )

        if model is None:
            projector = make_projector(str(cfg.get("projector", {}).get("method", "joseph3d")).lower(), geom).to(device)

            proj_cfg = cfg.get("projector", {})
            model_cfg = cfg.get("model", {})
            # Propagate selected projector options
            setattr(projector, "fbp_filter",  str(proj_cfg.get("fbp_filter", getattr(projector, "fbp_filter", "none"))).lower())
            setattr(projector, "fbp_cutoff",  float(proj_cfg.get("fbp_cutoff", getattr(projector, "fbp_cutoff", 1.0))))
            setattr(projector, "fbp_pad_mode",str(proj_cfg.get("fbp_pad_mode",getattr(projector, "fbp_pad_mode","next_pow2"))).lower())
            setattr(projector, "ir_impl", str(model_cfg.get("ir_impl", getattr(projector, "ir_impl", "grid"))).lower())
            setattr(projector, "ir_interpolation",
                    str(model_cfg.get("ir_interpolation", getattr(projector, "ir_interpolation", "linear"))).lower())
            setattr(projector, "bp_span", str(proj_cfg.get("bp_span", getattr(projector, "bp_span", "auto"))).lower())
            setattr(projector, "dc_mode", str(proj_cfg.get("dc_mode", getattr(projector, "dc_mode", "detector"))).lower())

            model = HDNSystem(cfg, projector=projector).to(device)

            if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
                model = torch.compile(model)

            if name == "adafactor":
                from optim.adafactor import Adafactor
                opt = Adafactor(model.parameters(), lr=lr, weight_decay=wd)
            else:
                opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            model.projector.reset_geometry(geom)

        D_total = len(ds)
        all_idx = np.arange(D_total)
        rng = np.random.RandomState(int(cfg["train"]["seed"]) + g_idx)
        rng.shuffle(all_idx)

        train_ratio = float(cfg["train"].get("train_ratio", 0.8))
        n_train = int(round(D_total * train_ratio))
        if D_total >= 2:
            n_train = max(1, min(D_total - 1, n_train))
        else:
            n_train = 1

        idx_train = all_idx[:n_train]
        idx_val = all_idx[n_train:]

        dl_train = DataLoader(
            Subset(ds, idx_train),
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
        )
        dl_val = DataLoader(
            Subset(ds, idx_val),
            batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
            shuffle=False,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
        )

        print(f"[split] train={len(idx_train)} ({len(idx_train)/max(1,D_total):.1%})  "
              f"val={len(idx_val)} ({len(idx_val)/max(1,D_total):.1%})")

        for epoch in range(1, epochs_per_group + 1):
            model.train()
            pbar = tqdm(dl_train, desc=f"group {g_idx + 1}, epoch {epoch}", dynamic_ncols=True)
            opt.zero_grad(set_to_none=True)

            for batch in pbar:
                steps += 1

                S_ua = batch["sino_ua"].to(device, non_blocking=True)   # [B,U,A] ≡ [B,X,A]
                V_gt = batch["voxel_xy"].to(device, non_blocking=True)  # [B,1,X,Y]

                S_xaz = S_ua.unsqueeze(1).unsqueeze(-1)  # [B,1,X,A,1]
                V_xyz = V_gt.unsqueeze(-1)               # [B,1,X,Y,1]

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    _, recon = model(S_xaz, v_vol=V_xyz, train_mode=True)
                    R2 = recon[..., 0]  # [B,1,X,Y]
                    if clamp_pred_gt:
                        R2c = R2.clamp(0.0, 1.0)
                        Vc  = V_gt.clamp(0.0, 1.0)
                    else:
                        R2c, Vc = R2, V_gt
                    # Composite loss
                    loss_tensor, info = criterion(R2c, Vc)
                    # Auxiliary edge contrast (batch mean)
                    ec_tensor_batch = edge_contrast_slice_max_torch(
                        R2, (V_gt > 0.0).float(), reduction="batch_mean"
                    )
                    # Incorporate into total loss (maximize contrast)
                    if w_edge_contrast != 0.0:
                        loss_tensor = loss_tensor + (-w_edge_contrast) * ec_tensor_batch
                    loss = loss_tensor / accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

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

                with torch.no_grad():
                    step_loss = float(loss_tensor.item())
                    ema = step_loss if ema is None else (beta * ema + (1.0 - beta) * step_loss)
                    running_avg = ema / (1.0 - beta ** steps)
                    ec_val = float(ec_tensor_batch.detach().item())

                csv_logger.log(
                    {
                        "group": g_idx + 1,
                        "epoch": epoch,
                        "step": steps,
                        "running_avg": running_avg,
                        "loss_total": step_loss,
                        "mse": _as_float(info.get("mse")) if (info.get("mse") is not None) else "",
                        "psnr": _as_float(info.get("psnr")) if (info.get("psnr") is not None) else "",
                        "ssim": _as_float(info.get("ssim")) if (info.get("ssim") is not None) else "",
                        "ec": ec_val,
                        "val_loss": "",
                        "val_mse": "",
                        "val_psnr": "",
                        "val_ssim": "",
                        "val_ec": "",
                    },
                    flush=(steps % flush_every == 0),
                )
                pbar.set_postfix(
                    {
                        "avg": f"{running_avg:.4f}",
                        "loss": f"{step_loss:.4f}",
                        "mse": f"{_as_float(info.get('mse')):.4f}" if info.get("mse") is not None else "na",
                        "ssim": f"{_as_float(info.get('ssim')):.4f}" if info.get("ssim") is not None else "na",
                        "psnr": f"{_as_float(info.get('psnr')):.2f}" if info.get("psnr") is not None else "na",
                        "ec": f"{ec_val:.4f}",
                    },
                    refresh=False,
                )

                del S_ua, V_gt, S_xaz, V_xyz, recon, R2, loss_tensor, loss, info
                if (steps % empty_cache_every) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Validation
            val = evaluate_slicewise_composite(
                model, dl_val, device, criterion,
                clamp_pred_gt=clamp_pred_gt,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype
            )

            csv_logger.log(
                {
                    "group": g_idx + 1,
                    "epoch": epoch,
                    "step": steps,
                    "running_avg": "",
                    "loss_total": "",
                    "mse": "",
                    "psnr": "",
                    "ssim": "",
                    "ec": "",
                    "val_loss": val["loss"],
                    "val_mse": val["mse"],
                    "val_psnr": ("" if val.get("psnr") is None else val["psnr"]),
                    "val_ssim": ("" if val.get("ssim") is None else val["ssim"]),
                    "val_ec": ("" if val.get("ec") is None else val["ec"]),
                },
                flush=True,
            )

            val_loss = float(val["loss"])
            if not (val_loss != val_loss):
                if val_loss < best_val:
                    best_val = val_loss
                    # ------------------------------------------------------------------
                    # Save checkpoint with geometry info included.
                    geom_obj = model.projector.geom
                    geometry_state = {
                        "vol_shape": geom_obj.vol_shape,
                        "det_shape": geom_obj.det_shape,
                        "angles": geom_obj.angles.detach().cpu(),
                        "voxel_size": geom_obj.voxel_size,
                        "det_spacing": geom_obj.det_spacing,
                        "angle_chunk": int(geom_obj.angle_chunk),
                        "n_steps_cap": int(geom_obj.n_steps_cap),
                    }
                    state = {
                        "group": g_idx + 1,
                        "epoch": epoch,
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "opt_state": opt.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "config": cfg,
                        "geometry": geometry_state,
                    }
                    torch.save(state, str(ckpt_dir / "best_shared.pt"))
                    
    csv_logger.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train HDN (slice‑wise) with explicit 80/20 split + composite boundary‑aware loss.")
    ap.add_argument("--cfg", type=str, default="config.yaml", help="Path to YAML config.")
    args = ap.parse_args()
    main(args.cfg)
