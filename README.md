# HDN 3D Reconstruction

This repository implements the **Hybrid Decomposition Network (HDN)**, a model for reconstructing high‑fidelity volumes from parallel‑beam 3D computed tomography (CT) data. It draws on the VAMToolbox and ASTRA Toolbox implementations of forward projection and follows the encoder/decoder architecture introduced in the HDN paper.  The sections below outline the major files and functions, the physical models, network architecture, optional features and how to run training.

## Version and Dependencies

* **Python 3.10+**
* **Key dependencies:** PyTorch 2.8.0 + cu129, numpy, tqdm, yaml (see `requirements.txt` for details).  The Joseph projector leverages `torch.nn.functional.grid_sample` to perform GPU‑accelerated projection.

## Data Organization

### Input Files

Training data are stored under `data/` and require the following files for each sample id:

* `sino/{id}_sino.npy` – sinogram tensor of shape `[A, V, U]` where A is the number of projection angles, V the number of detector rows, and U the number of detector columns.
* `voxel/{id}_voxel.npy` – ground‑truth volume `[D, H, W]` where D is the number of z‑slices, H height and W width.
* *(optional)* `sino/{id}_angles.npy` – projection angles `[A]` in radians.  If absent, a uniform 0–π grid is synthesized.

### `NpySinoVoxelDataset`

The `NpySinoVoxelDataset` in `data/dataset.py` reads the sinogram and volume files, normalizing their shapes to `[A, V, U]` and `[D, H, W]` before returning the sample.  `data/io.py` provides helpers to expand 2D arrays into 3D and enforce shape consistency.

## Physical Model

### Geometry

`physics/geometry.py` defines a `Parallel3DGeometry` dataclass that stores volume shape `(D,H,W)`, detector shape `(V,U)`, projection angles `A` and the physical sampling intervals for voxels and detector pixels.  Convenience properties `A`, `D`, `H`, `W`, `V`, `U` expose these dimensions for use by projectors.

### Forward/Backprojection (Joseph Projector)

The `physics/projector.py` module implements two projectors:

* **JosephProjector3D** – a voxel‑driven forward projector/backprojector.  In the forward pass it integrates rays using `grid_sample`, chunking angles to control memory usage and computing detector coordinates to perform trilinear interpolation.  Backprojection maps detector coordinates back into voxel space and accumulates contributions with `grid_sample`.
* **SiddonProjector3D** – an analytic ray‑driven projector similar to ASTRA Toolbox; it steps through voxels along each ray and computes intersection lengths.  Accurate but slower, it is mainly for validation.
* `make_projector(method, geom)` instantiates the appropriate projector (`'joseph3d'` or `'siddon3d'`).

### Detector PSF

`physics/psf.py` implements `SeparableGaussianPSF2D`, which applies a separable Gaussian blur to the detector plane.  In angle‑invariant mode the same σ is used for all angles; in angle‑variant mode different σ vectors can be provided for each projection.  The PSF can be enabled or disabled independently.

## Network Architecture

### Encoders

* **enc1\_1d.AngleAxisEncoder1D** – encodes the sinogram along the angle axis using 1D convolutions and appends the per‑angle mean and standard deviation to the feature channels.
* **enc2\_2d.SinogramEncoder2D** – applies 2D convolutions to each angle’s detector image, with harmonic angle embeddings and an optional cheat‑channel gating mechanism.
* **enc3\_3d.VolumetricEncoder3D** – (optional) encodes the ground‑truth volume via 3D convolutions to produce a volumetric prior.

### Align & Decoder

* **Align2Dto3D** – backprojects the encoders’ output features to form a 3D latent volume, applies the PSF transpose if configured, concatenates the volumetric prior and mixes them with a 3D convolution.
* **DecoderSinogram** – forward projects the 3D latent volume through the projector to predict the sinogram.  It processes channels in chunks to manage memory and sums them at the end.

### HDNSystem

`models/hdn.py`’s `HDNSystem` composes these modules into an end‑to‑end network.  Its forward method (1) encodes the input sinogram, (2) aligns features to a 3D latent volume and (3) reconstructs the sinogram via the decoder.  It supports PSF, cheat gating and optional volumetric priors, and can compute a forward consistency loss by performing a second forward projection.

## Loss Functions

* **Forward Consistency Loss** – `losses/forward.py`’s `forward_consistency_loss` computes the residual between the predicted and input sinograms in Fourier space, weighting low frequencies more heavily.
* **Reconstruction Loss** – `losses/recon.py` returns a dictionary containing SSIM, negative PSNR, band penalty, energy penalty, voxel error rate (VER) and in‑positive dynamic range (IPDR); a weighted sum yields the total reconstruction loss.
* **Normalization** – `losses/utils.py` provides `per_sample_sino_scale` and `normalize_tensors`, which scale sinograms and volumes per‑sample using a robust quantile to stabilize loss magnitudes.

## Optional Features

* **Cheat Gating** – injects ground‑truth volume information into the encoders during training and can be gated off for evaluation.
* **Volumetric Prior** – encodes the ground‑truth volume with `enc3_3d` and concatenates it to the latent volume.
* **PSF Consistency** – applies the same blur to both forward and backprojected paths for consistency.
* **Optimizers** – supports AdamW and the memory‑efficient Adafactor.
* **Mixed Precision** – Automatic Mixed Precision (AMP) can be enabled via the configuration.

## Training Pipeline

1. **Load and split data:** `train.py` loads `NpySinoVoxelDataset` and splits IDs into train/validation sets.
2. **Set geometry and projector:** instantiate `Parallel3DGeometry` and create a projector via `make_projector`.
3. **Build the model:** initialize `HDNSystem` and configure encoders, decoder, PSF and cheat options.
4. **Compute losses:** evaluate the forward consistency and reconstruction losses and sum them to form the total loss.
5. **Optimization:** update parameters using AdamW or Adafactor with gradient accumulation and clipping.
6. **Validation and checkpoints:** periodically evaluate on the validation set, save the best checkpoint and log metrics with `CSVLogger`.

## Scripts

* **Training:**

  ```bash
  python train.py --cfg config.yaml
  ```

  Adjust geometry, model, loss and training hyperparameters in `config.yaml` as needed.
* **Dataset Integrity:** `scripts/check_dataset_integrity.py` verifies that each sinogram/voxel pair contains no NaN/Inf and has the correct dimensions.
* **Bucket Creation:** `scripts/make_buckets.py` groups samples with the same geometry into buckets and generates specialized configs for each.
* **Bucket Training:** `scripts/run_buckets.py` reads `manifest.json` and launches `train.py` for each bucket sequentially, continuing even if some buckets fail.

## Architecture Diagram
```
               Input S_in (A,V,U)
                      |
                      v  normalize (per_sample_sino_scale)
               +------+------+
               |             |
         (1D conv)        (2D conv)
               ^             ^
               |             |
       cheat gating (train only)  <----  ground‑truth V_gt (B,1,D,H,W)
               \             /
                \           /
                 \         /
                  \       /
                   \     /
                    \   /
                     v v
                 Align2Dto3D
             (backproject & fuse)
                      |
                      v
          Latent volume (B,C,D,H,W)
                      |
                      v
               DecoderSinogram
              (forward project)
                      |
                +-----+-----+
                |           |
           S_hat (B,A,V,U)   R_hat (B,1,D,H,W)
                |           |
                +-----------+
                      |
                      v
 Losses: forward consistency + reconstruction
```


This diagram summarizes the data flow.  The encoders transform the input sinogram into feature maps, Align2Dto3D converts them into a latent volume, and DecoderSinogram uses the physical projector to predict the sinogram while simultaneously computing a backprojected reconstruction.

## Notes

This project references the HDN paper as well as the forward projection algorithms in ASTRA Toolbox and the VAMToolbox to ensure physical fidelity.  The projector is implemented as a linear operation, and non‑negativity is enforced indirectly through the loss functions and normalization.  The model, losses and data handling can be easily tuned via configuration files, making it applicable to various geometries and datasets.