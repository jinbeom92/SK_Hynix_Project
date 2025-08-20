# SK Hynix Project — High-Dimensional Neural Tomographic Reconstruction

This repository provides a **PyTorch-based framework** for high-dimensional neural (HDN) tomographic reconstruction.  
It integrates **multi-scale encoders (1D, 2D, 3D)**, feature alignment, fusion, and a physics-consistent forward/backprojection module (ASTRA-style, implemented in pure PyTorch).  
The system eliminates traditional FBP from the training path and optimizes sinograms directly to produce physically consistent 3D reconstructions.

---

## 🚀 Features

- **Encoders**
  - `enc1_1d.py`: 1D convolutional encoder for frequency/angle features.
  - `enc2_2d.py`: 2D encoder for sinogram slices.
  - `enc3_3d.py`: 3D encoder for volumetric features.

- **Latent Alignment & Fusion**
  - `align.py`: Aligns 2D → 3D latent spaces.
  - `fusion.py`: Combines multi-modal encoders into a shared latent tensor.

- **Decoder**
  - `decoder.py`: Generates reconstructed volumes from fused latent features.

- **Physics Modules**
  - `physics/projector.py`: Differentiable forward/back projectors (Joseph3D, Siddon3D).
  - `physics/geometry.py`: Defines 3D parallel beam geometry.
  - `physics/psf.py`: Implements separable Gaussian PSF (angle-dependent or invariant).

- **Losses**
  - `losses/recon.py`: Reconstruction losses (SSIM, PSNR, Band, Energy, Forward consistency).
  - `utils/metrics.py`: Structural similarity, error ratios, etc.

- **Optimizers**
  - `optim/adafactor.py`: Adafactor optimizer (memory-efficient).

- **Dataset Loader**
  - `data/dataset.py`: Loads paired `.npy` sinogram/voxel datasets.
  - Format:
    - `data/sino/0001_sino.npy` → shape `(U, A, Z)`
    - `data/voxel/0001_voxel.npy` → shape `(Nx, Ny, Nz)`

---

## ⚙️ Requirements

- **Python**: 3.10.11  
- **CUDA**: 12.9  
- **Dependencies** (see `requirements.txt`):
  - `torch>=2.2`
  - `torchvision>=0.17`
  - `numpy`, `scipy`, `einops`, `tqdm`, `pyyaml`
  - Optional: `astra-toolbox>=2.1` (for validation only)

---

## 📂 Repository Structure

```

SK\_Hynix\_Project/
├── train.py                  # Main training loop
├── config.yaml               # Training configuration
├── models/                   # Encoders, Decoder, Fusion
├── physics/                  # Projector, Geometry, PSF
├── losses/                   # Loss functions
├── data/                     # Dataset loaders
├── optim/                    # Optimizers (Adafactor)
├── scripts/                  # Utilities (dataset check, bucket maker)
├── tests/                    # Unit tests (adjoint test, training step)

````

---

## 📝 Configuration (config.yaml)

Key parameters:

- **projector**
  - `method`: `"joseph3d"` or `"siddon3d"`
  - `c_chunk`, `step_chunk`: Chunking for memory efficiency
  - `psf`: `{ enabled: bool, sigma: float }`

- **training**
  - `epochs`: number of epochs
  - `lr`: learning rate
  - `wd`: weight decay
  - `amp`: automatic mixed precision
  - `grad_clip`: gradient clipping threshold

- **losses**
  - `L_ssim`, `L_psnr`, `L_band`, `L_forward`, etc. with weights

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

### 5. Unit tests

```bash
pytest tests/
```

---

## 🧪 Physics Validation

* The **adjoint test** (`tests/test_adjoint.py`) ensures forward and back projectors are adjoint consistent.
* The system is stable with AMP + FP32 FFT enforced.

---

## 📊 Logging & Checkpoints

* CSV logging per training step (`utils/logging.py`).
* Checkpoints saved in `ckpt/` directory with config snapshot.
* Results stored in `results/`.

---

## 🔒 Constraints

* **FBP/FDK/iradon** are forbidden in training (allowed only for visualization/evaluation).
* All reconstructions must satisfy:

  * `sinogram >= 0`
  * energy & dose-band constraints
  * minimized IPDR (inverse projection data residual)

---

## ✨ Citation

Inspired by:

* HDN: *High-Dimensional Neural Tomographic Reconstruction* (paper reference).
* ASTRA Toolbox forward/backprojection operators.
---

## 🔬 Model Architecture

### 1. **Encoders**
- **Enc1\_1D\_Angle**  
  - Applies Conv1D+GroupNorm+ReLU along the **angle axis (A)** for each detector row (U).  
  - Input: `[B,1,U,A]` → Output: `[B,C1,U,A]`.  

- **Enc2\_2D\_Sino**  
  - Applies Conv2D+GroupNorm+ReLU over the **(U,A)** plane to capture joint spatial/angular features.  
  - Input: `[B,1,U,A]` → Output: `[B,C2,U,A]`.

### 2. **Align Layer (Sino2XYAlign)**  
- Projects sinogram features from the `(U,A)` domain into the voxel `(X,Y)` domain.  
- Uses stacked Conv2D blocks and interpolation (`bilinear`/`nearest`) to match voxel slice size.  
- Output: `[B,Ca,X,Y]`.

### 3. **Cheat Injection (VoxelCheat2D + Fusion2D)**  
- **VoxelCheat2D** encodes the ground-truth voxel slice during training: `[B,1,X,Y]` → `[B,Cc,X,Y]`.  
- **Fusion2D** concatenates sinogram-aligned features with cheat features.  
  - In training: `fused = concat(sino, cheat)`  
  - In evaluation: `fused = concat(sino, zeros)` (cheat disabled).  

### 4. **Decoder (DecoderSlice2D)**  
- Applies multiple Conv2D+GroupNorm+ReLU layers to map fused features back into the image domain.  
- Output: `[B,1,X,Y]` voxel slice prediction.

---

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