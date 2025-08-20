
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