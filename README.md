# SK Hynix Project — High-Dimensional Neural Tomographic Reconstruction

This repository provides a **PyTorch-based framework** for high-dimensional neural (HDN) tomographic reconstruction.  
The system eliminates traditional FBP from the training path and directly optimizes sinograms to produce physically consistent 3D reconstructions.

---

## 🚀 Architecture Overview

- **1D/2D Encoders**
  - `Enc1_1D_Angle`: applies 1D convolutions along the **angle axis**, transforming input sinograms of shape `[B,1,U,A]` into feature maps `[B,C1,U,A]`.
  - `Enc2_2D_Sino`: applies 2D convolutions over the **sinogram plane**, producing `[B,C2,U,A]` features.

- **Alignment Module**
  - `Sino2XYAlign`: interpolates and aligns concatenated sinogram features into the 2D image domain, producing `[B,C_out,X,Y]`.

- **Cheat Injection & Fusion**
  - `VoxelCheat2D`: encodes the ground-truth voxel slice during training.
  - `Fusion2D`: concatenates aligned sinogram features and cheat features along the channel axis. If cheat is disabled, zero-padding is used automatically.

- **Decoder**
  - `DecoderSlice2D`: processes fused features with multiple convolutional blocks and a 1×1 projection, outputting a single voxel slice.

- **HDNSystem**
  - The `HDNSystem` class (`models/hdn.py`) combines the above modules into the forward pipeline:
    1. Extracts features via 1D/2D encoders.
    2. Aligns them to the voxel domain.
    3. Optionally applies cheat injection and fusion.
    4. Decodes into a reconstructed slice.

- **Optional 3D Encoder**
  - `Enc3_3D`: training-only prior that extracts hierarchical volumetric features from ground-truth 3D volumes.  
    Not integrated into the default `HDNSystem`.

- **Physics Modules**
  - `physics/projector.py` and `physics/geometry.py` implement forward/back projectors and acquisition geometry for 3D parallel-beam CT.  
    As the README notes, these are not used in the default training loop but are available for testing or extended setups.

---

## ⚙️ Requirements

- **Python**: 3.10.11  
- **CUDA**: 12.9  
- **Dependencies**: see `requirements.txt`
  - `torch>=2.2`, `torchvision>=0.17`
  - `numpy`, `scipy`, `einops`, `tqdm`, `pyyaml`
  - Optional: `astra-toolbox>=2.1` (for validation only)

---

## 📂 Repository Structure

```

SK\_Hynix\_Project/
├── train.py                  # Main training loop
├── config.yaml               # Training configuration
├── models/                   # Encoders, Decoder, Fusion, HDNSystem
├── physics/                  # Projector, Geometry, PSF
├── losses/                   # Loss functions
├── data/                     # Dataset loaders
├── optim/                    # Optimizers (Adafactor)
├── scripts/                  # Utilities (dataset check, bucket maker)
├── tests/                    # Unit tests
├── utils/                    # Logging, metrics, config utilities

````

---

## 📝 Key Notes

- The **core architecture** (1D/2D encoders → alignment → cheat injection & fusion → decoder) is **fully implemented** and matches the README.  
- The **volumetric encoder (`Enc3_3D`)** and **physics-based projector modules** are implemented but **not integrated into the default `HDNSystem` or training script**.  
- Thus, by default, training proceeds with 1D/2D encoders, alignment, optional cheat injection, and decoding.

---

## ▶️ Usage

### 1. Install requirements
```bash
pip install -r requirements.txt
````

### 2. Prepare dataset

```
data/
├── sino/
│   ├── 0001_sino.npy
│   ├── 0002_sino.npy
│   ...
├── voxel/
│   ├── 0001_voxel.npy
│   ├── 0002_voxel.npy
│   ...
```

### 3. Run training

```bash
python -m train --cfg config.yaml
```

### 4. Run dataset check

```bash
python -m scripts.check_dataset_integrity
```

### 5. Run unit tests

```bash
pytest tests/
```

---

## 🧪 Validation

* **Adjoint consistency**: `tests/test_adjoint.py` checks forward/backprojector adjointness.
* **One-step training test**: `tests/test_training_step.py` validates forward path with/without cheat injection.

---

## 🔒 Constraints

* **FBP/FDK/iradon** are forbidden in training (allowed only for visualization/evaluation).
* Reconstructions must satisfy:

  * `sinogram >= 0`
  * energy & dose-band constraints
  * minimized IPDR (inverse projection data residual)

---

## ✨ Citation

Inspired by:

* HDN: *High-Dimensional Neural Tomographic Reconstruction* (paper reference).
* ASTRA Toolbox forward/backprojection operators.

```
## 🧩 Architecture Diagram

```bash
      Input Sinogram Slice (U×A)
                   │
         ┌─────────┴─────────┐
         │                   │
     Enc1_1D             Enc2_2D
         │                   │
         └───── concat ──────┘
                   │
              Sino2XYAlign
                   │
             [B,Ca,X,Y]
                   │
         ┌─────────┴─────────┐
         │                   │
         │           VoxelCheat2D (train only)
         │                   │
         └───── Fusion2D ────┘
                   │
            DecoderSlice2D
                   │
         Predicted Voxel Slice
```