"""
Train HDN with **masked MSE only** and **unfiltered backprojection**.

Pipeline (slice training with Z=1)
----------------------------------
Sino(x,a,z) → Enc1(A) + Enc2(XA) → Sino2XYAlign(XA→XY) → (optional) VoxelCheat2D →
Fusion2D(XY) → SinoDecoder2D(XY→XA) → JosephProjector3D.backproject (unfiltered) →
Recon(x,y,z) → ExpandMaskedMSE(pred=Recon2D, soft-boundary mask/targets from GT2D)

Axis convention (model-facing, fixed)
-------------------------------------
• Sinogram : (x, a, z) → [B, 1, X, A, Z]
• Volume   : (x, y, z) → [B, 1, X, Y, Z]

Notes
-----
• Backprojection is **unfiltered by implementation** in `physics/projector.py`.
• Training loss is **ExpandMaskedMSE** only (in-part + near-boundary out-of-part
  with soft labels in [boundary_low, boundary_high]), averaged over the masked pixels.
"""

from pathlib import Path
import math
import gc
import numpy as np
import torch
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
from physics.geometry import Parallel3DGeometry
from physics.projector import make_projector
from losses.expand_mask_mse import ExpandMaskedMSE


def _as_float(x):
    """
    Convert a scalar (Python number or 0-d torch.Tensor) to float safely.

    Parameters
    ----------
    x : Union[float, int, torch.Tensor]

    Returns
    -------
    float
    """
    return float(x) if not torch.is_tensor(x) else float(x.detach().item())


def _init_amp(device: torch.device, cfg: dict):
    """
    Decide AMP dtype and scaler based on config and hardware.

    Parameters
    ----------
    device : torch.device
    cfg : dict

    Returns
    -------
    enabled : bool
        Whether autocast/GradScaler should be enabled.
    amp_dtype : torch.dtype
        Preferred compute dtype for autocast.
    scaler : torch.amp.GradScaler
        Gradient scaler (no-op if disabled).
    """
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


@torch.no_grad()
def evaluate_group_by_volume_mse(
    model: torch.nn.Module,
    ds: ConcatDepthSliceDataset,
    device: torch.device,
    criterion: torch.nn.Module,
) -> float:
    """
    Volumetric evaluation by **masked MSE only** (slice-wise → per-volume mean → mean over files).

    For each file pair in the dataset:
      • Iterate depth d = 0..D-1 and run slice inference with Z=1 (cheat OFF).
      • Compute `criterion(recon2d, gt2d)` per slice and average over depth.
    Returns the mean masked MSE across files.
    """
    model.eval()
    loss_sum = 0.0
    file_cnt = 0

    for s_mem, v_mem in zip(ds.sinos, ds.voxels):
        D = int(v_mem.shape[-1])
        z_sum = 0.0

        for d in range(D):
            s_ua = np.array(s_mem[:, :, d], dtype=np.float32, copy=True, order="C")  # [U,A]
            v_xy = np.array(v_mem[:, :, d], dtype=np.float32, copy=True, order="C")  # [X,Y]

            S = torch.from_numpy(s_ua).to(device).unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # [1,1,X,A,1]
            V = torch.from_numpy(v_xy).to(device).unsqueeze(0).unsqueeze(0)                # [1,1,X,Y]

            _, recon = model(S, v_vol=None, train_mode=False)  # [1,1,X,A,1], [1,1,X,Y,1]
            R2 = recon[..., 0]                                 # [1,1,X,Y]

            z_sum += float(criterion(R2, V).item())

        loss_sum += (z_sum / max(1, D))
        file_cnt += 1

    return loss_sum / max(1, file_cnt)


def main(cfg_path: str):
    """
    Train loop: masked-MSE-only slice training with unfiltered BP.

    Shapes (fixed)
    --------------
    • Input sino  : [B,1,X,A,1]
    • GT voxel    : [B,1,X,Y,1]
    • Recon out   : [B,1,X,Y,1]
    """
    # --- Config / device / seed ------------------------------------------------------------
    cfg = load_config(cfg_path)
    save_effective_config(cfg, Path(cfg_path).with_name("effective_config.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["train"]["seed"]))
    amp_enabled, amp_dtype, scaler = _init_amp(device, cfg)

    # --- Discover files & group them -------------------------------------------------------
    data_root = cfg.get("data", {}).get("root", "data")
    sino_glob = cfg.get("data", {}).get("sino_glob", "sino/*_sino.npy")
    voxel_glob = cfg.get("data", {}).get("voxel_glob", "voxel/*_voxel.npy")

    all_sino = sorted(Path(data_root).glob(sino_glob))
    all_vox = sorted(Path(data_root).glob(voxel_glob))
    if len(all_sino) != len(all_vox):
        raise AssertionError(f"#sino({len(all_sino)}) != #voxel({len(all_vox)})")

    files_per_group = int(cfg["train"].get("files_per_group", 100))
    groups = [
        (all_sino[i : i + files_per_group], all_vox[i : i + files_per_group])
        for i in range(0, len(all_sino), files_per_group)
    ]

    # --- Logging ---------------------------------------------------------------------------
    Path("results").mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(
        str(Path("results") / "train_log.csv"),
        fieldnames=["group", "epoch", "step", "running_avg", "loss_total", "mse", "vol_loss"],
    )

    # --- Loss: ExpandMaskedMSE (soft boundary) ---------------------------------------------
    lcfg = cfg.get("losses", {})
    criterion = ExpandMaskedMSE(
        thr=float(lcfg.get("expand_thr", 0.8)),
        spacing=lcfg.get("expand_spacing", None),
        include_in_part=bool(lcfg.get("expand_include_in", True)),
        in_value=float(lcfg.get("expand_in_value", 1.0)),
        boundary_low=float(lcfg.get("expand_boundary_low", 0.8)),
        boundary_high=float(lcfg.get("expand_boundary_high", 0.9)),
        clamp_pred_gt=bool(lcfg.get("expand_clamp", True)),
    ).to(device)

    # --- Optimizer & misc hyperparams ------------------------------------------------------
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

    # --- Training over groups ---------------------------------------------------------------
    model = None
    opt = None

    for g_idx, (sino_paths, voxel_paths) in enumerate(groups):
        print(f"--- Training on group {g_idx + 1}/{len(groups)} (files {len(sino_paths)}) ---")

        # Dataset / DataLoader (slice-level)
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

        # Geometry & projector (Z=1 for slice training)
        U, A = ds.U, ds.A
        X, Y = ds.X, ds.Y
        Z = 1
        angles = torch.linspace(0.0, math.pi, steps=A, device=device, dtype=torch.float32)

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

        # Train/val split over depths in this group (cheat ON only for train slices)
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
                S_ua = batch["sino_ua"].to(device, non_blocking=True)   # [B,U,A] ≡ [B,X,A]
                V_gt = batch["voxel_xy"].to(device, non_blocking=True)  # [B,1,X,Y]

                # Canonicalize to model I/O
                S_xaz = S_ua.unsqueeze(1).unsqueeze(-1)  # [B,1,X,A,1]
                V_xyz = V_gt.unsqueeze(-1)               # [B,1,X,Y,1]

                # Depth index to gate cheat path
                z = int(batch["global_z"][0]) if hasattr(batch["global_z"], "__len__") else int(batch["global_z"])
                is_train_slice = (z in train_set)

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    sino_hat, recon = model(S_xaz, v_vol=(V_xyz if is_train_slice else None), train_mode=is_train_slice)
                    R2 = recon[..., 0]                  # [B,1,X,Y]
                    loss_tensor = criterion(R2, V_gt)   # masked MSE (2D slice)
                    loss = loss_tensor.mean() / accum_steps

                # Backprop + update (AMP-aware)
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

                # Running EMA + step logging
                with torch.no_grad():
                    step_loss = float((loss_tensor.mean()).item())
                    ema = step_loss if ema is None else (beta * ema + (1.0 - beta) * step_loss)
                    running_avg = ema / (1.0 - beta ** steps)

                mse_val = _as_float(step_loss)  # numeric only for logging
                val_total = mse_val             # only MSE is used

                csv_logger.log(
                    {
                        "group": g_idx + 1,
                        "epoch": epoch,
                        "step": steps,
                        "running_avg": running_avg,
                        "loss_total": val_total,
                        "mse": mse_val,
                    },
                    flush=(steps % flush_every == 0),
                )
                pbar.set_postfix({"avg": f"{running_avg:.4f}", "loss": f"{step_loss:.4f}"}, refresh=False)

                # Housekeeping
                del S_ua, V_gt, S_xaz, V_xyz, sino_hat, recon, R2, loss_tensor, loss
                if (steps % empty_cache_every) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Volumetric eval (masked MSE only; cheat OFF)
            vol_loss = float(evaluate_group_by_volume_mse(model, ds, device, criterion))

            csv_logger.log(
                {
                    "group": g_idx + 1,
                    "epoch": epoch,
                    "step": steps,
                    "running_avg": "",
                    "loss_total": "",
                    "mse": "",
                    "vol_loss": vol_loss,
                },
                flush=True,
            )

            # Best checkpoint by lowest vol_loss
            if not (vol_loss != vol_loss):  # NaN guard
                if vol_loss < best_val:
                    best_val = vol_loss
                    state = {
                        "group": g_idx + 1,
                        "epoch": epoch,
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "opt_state": opt.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "config": cfg,
                    }
                    torch.save(state, str(ckpt_dir / "best_shared.pt"))

    csv_logger.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train HDN with masked MSE only (unfiltered BP).")
    ap.add_argument("--cfg", type=str, default="config.yaml", help="Path to YAML config.")
    args = ap.parse_args()
    main(args.cfg)
