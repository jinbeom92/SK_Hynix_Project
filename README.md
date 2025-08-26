# SVTR: Sinogram-to-Volume Tomographic Reconstruction

## Overview

**SVTR** is a PyTorch-based framework for tomographic reconstruction that maps **sinograms** directly to **3D voxel volumes** through a hybrid neural–physics pipeline.
The system is designed for *end-to-end differentiable training* with physically consistent projectors, supports slice-wise supervision, and uses **unfiltered backprojection (BP)** for reconstruction.

Key features:

* Model-facing fixed conventions:

  * Sinogram (x, a, z) → **\[B, 1, X, A, Z]**
  * Volume (x, y, z)   → **\[B, 1, X, Y, Z]**
* Encoders along angle (1D) and sinogram plane (2D).
* Deterministic **Sino→XY alignment**.
* Optional **voxel cheat path** for stabilized training.
* Differentiable **Joseph forward** and **unfiltered BP** projector.
* Training loss: **ExpandMaskedMSE**, focusing on in-part and near-boundary pixels.

---

## Project Structure & Key Modules

### Data

* **`data/dataset.py`**

  * `ConcatDepthSliceDataset`: loads paired sinogram/voxel slices (`[U,A]`, `[X,Y]` per z).
  * `SinogramVoxelVolumeDataset`: loads full volumes (`[U,A,D]`, `[X,Y,D]`).

* **`data/io.py`**

  * `ensure_sino_shape`, `ensure_voxel_shape`: strict validation of `[X,A,Z]` and `[X,Y,Z]`.
  * Simple `load_npy`, `save_npy` helpers.

### Geometry & Physics

* **`physics/geometry.py`**

  * `Parallel3DGeometry`: stores acquisition parameters (angles, voxel/detector spacing).
  * Provides both legacy fields (ASTRA-style) and model-facing `(X,Y,Z)/(X,Z)` convenience.

* **`physics/projector.py`**

  * `JosephProjector3D`: voxel-driven forward projector, and **unfiltered slice-wise BP**.
  * `SiddonProjector3D`: ray-driven reference projector (voxel intersection lengths).
  * `make_projector`: factory function.

* **`physics/psf.py`**

  * `SeparableGaussianPSF2D`: optional Gaussian blur over sinograms (detector domain).

### Model

* **`models/enc1_1d.py`**

  * `Enc1_1D_Angle`: extracts features along the angle axis (A).

* **`models/enc2_2d.py`**

  * `Enc2_2D_Sino`: 2D conv encoder over (X,A) slices.

* **`models/align.py`**

  * `Sino2XYAlign`: deterministic resize (X,A→X,Y) alignment.

* **`models/fusion.py`**

  * `VoxelCheat2D`: encodes GT voxel slices (train-only).
  * `Fusion2D`: fuses sinogram-aligned and cheat features.

* **`models/decoder.py`**

  * `DecoderSlice2D`: 2D slice-wise voxel-plane decoder (alternative).
  * `SinoDecoder2D`: maps fused XY features back to XA sinogram.

* **`models/hdn.py`**

  * `HDNSystem`: full pipeline wrapper:
    `Enc1+Enc2 → Align → Cheat+Fusion → SinoDecoder → Projector.BP`.

### Training & Utilities

* **`train.py`**

  * Group-wise slice training loop with AMP and gradient accumulation.
  * Supports Adafactor or AdamW optimizer.
  * Logging with `CSVLogger`.

* **`losses/expand_mask_mse.py`**

  * `ExpandMaskedMSE`: masked MSE loss with soft boundary targets (0.8–0.9).

* **`utils/seed.py`**

  * `set_seed`: reproducible RNG seeding and deterministic flags.

* **`utils/yaml_config.py`**

  * Load/save configs with safe YAML.

* **`utils/logging.py`**

  * `CSVLogger`: robust CSV logging with header management.

---

## Training & Execution

### Config File (`config.yaml`)

Example minimal config:

```yaml
data:
  root: dataset
  sino_glob: "sino/*_sino.npy"
  voxel_glob: "voxel/*_voxel.npy"

projector:
  method: "joseph3d"

geom:
  voxel_size_xyz: [1.0, 1.0, 1.0]
  det_spacing_xz: [1.0, 1.0]
  angle_chunk: 16
  n_steps_cap: 256

model:
  enc1: { base: 32, depth: 3 }
  enc2: { base: 32, depth: 3 }
  align: { out_ch: 64, depth: 2, interp_mode: "bilinear" }
  cheat2d: { enabled: true, base: 16, depth: 2 }
  fusion: { out_ch: 64 }
  sino_dec: { mid_ch: 64, depth: 2, interp_mode: "bilinear" }

losses:
  expand_thr: 0.8
  expand_spacing: null
  expand_include_in: true
  expand_in_value: 1.0
  expand_boundary_low: 0.8
  expand_boundary_high: 0.9
  expand_clamp: true

train:
  seed: 1337
  amp: true
  optimizer: "adamw"
  lr: 1.0e-4
  weight_decay: 1.0e-3
  batch_size: 8
  num_workers: 2
  grad_clip: 1.0
  train_ratio: 0.8
  files_per_group: 100
  epochs: 1
  ckpt_dir: "checkpoints/svtr"
  flush_every: 10
  empty_cache_every: 100
```

### Run Training

```bash
python -m train --cfg config.yaml
```

---

## Architecture Diagram

```text
            Input Sinogram [B,1,X,A,Z]
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  Enc1_1D_Angle                   Enc2_2D_Sino
 [B,C1,X,A,Z]                     [B,C2,X,A,Z]
        └───────────────┬───────────────┘
                        ▼
                  Concatenate → [B,C1+C2,X,A,Z]
                        │
                        ▼
               Sino2XYAlign (XA→XY)
               [B,Ca,X,Y,Z]
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   VoxelCheat2D (train)              None (eval)
   [B,Cc,X,Y,Z]                           
        └───────────────┬───────────────┘
                        ▼
                     Fusion2D
                  [B,Cf,X,Y,Z]
                        │
                        ▼
                 SinoDecoder2D
              [B,1,X,A,Z] (predicted)
                        │
                        ▼
         JosephProjector3D.backproject
          (Unfiltered BP, slice-wise)
                        │
                        ▼
              Recon Volume [B,1,X,Y,Z]
```

Loss:

* `ExpandMaskedMSE(pred=recon2D, gt=voxel2D)` applied slice-wise,
  masking out far background, weighting near-boundary with soft targets.