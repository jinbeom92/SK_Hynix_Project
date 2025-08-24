import os
from glob import glob
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

# ==========================================================================================
# Utility loaders (no behavior change)
# ==========================================================================================

def _load_sino_uaD(path: str) -> np.memmap:
    """
    Memory-map a sinogram .npy file.

    Expected shape
    --------------
    [U, A, D] where
      U: detector bins (≡ X),
      A: projection angles,
      D: depth slices (≡ Z).

    If input is [U, A], it will be reshaped to [U, A, 1].

    Notes
    -----
    * We keep ASTRA's native 3D convention (u, angles, v) at load-time
      as [U, A, D]; the model-facing volume class permutes to z-first.
      When you call ASTRA backprojection, pass [U, A, D] again by
      transposing back from the model layout.
    """
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:  # upgrade to [U,A,1]
        U, A = arr.shape
        arr = arr.reshape(U, A, 1)
    if arr.ndim != 3:
        raise ValueError(f"sino ndim must be 3 (U,A,D), got {arr.ndim} for {path}")
    return arr


def _load_voxel_xyD(path: str) -> np.memmap:
    """
    Memory-map a voxel volume .npy file.

    Expected shape
    --------------
    [X, Y, D] where D is the z-depth. If input is [X, Y], it will be
    reshaped to [X, Y, 1].
    """
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:  # upgrade to [X,Y,1]
        X, Y = arr.shape
        arr = arr.reshape(X, Y, 1)
    if arr.ndim != 3:
        raise ValueError(f"voxel ndim must be 3 (X,Y,D), got {arr.ndim} for {path}")
    return arr


# ==========================================================================================
# Slice-level dataset (unchanged API; only arrays are returned as-is per slice)
# ==========================================================================================

class ConcatDepthSliceDataset(Dataset):
    """
    Paired sinogram/voxel **slice** dataset.

    Each index corresponds to a single depth slice d=z. Returns:
      - "sino_ua":  torch.FloatTensor [U, A]    (≡ [X, A] at that z)
      - "voxel_xy": torch.FloatTensor [1, X, Y] (channel-first for 2D conv)
      - indices/meta for bookkeeping.

    This class does not permute axes beyond extracting the slice.
    It is useful when the model iterates over z one plane at a time
    (e.g., 1D conv along X per angle and 2D conv on (X,A) per z).
    """
    def __init__(self,
                 data_root: str = "data",
                 sino_glob: str = "sino/*_sino.npy",
                 voxel_glob: str = "voxel/*_voxel.npy",
                 sino_paths: Optional[List[str]] = None,
                 voxel_paths: Optional[List[str]] = None,
                 report: bool = True,
                 resample_u: bool = False):
        super().__init__()

        # -------------------------
        # File list
        # -------------------------
        if sino_paths is not None and voxel_paths is not None:
            self.sino_paths = list(sino_paths)
            self.voxel_paths = list(voxel_paths)
        else:
            self.sino_paths = sorted(glob(os.path.join(data_root, sino_glob)))
            self.voxel_paths = sorted(glob(os.path.join(data_root, voxel_glob)))

        if not self.sino_paths:
            raise FileNotFoundError(f"No sinograms found: {sino_paths or os.path.join(data_root, sino_glob)}")
        if not self.voxel_paths:
            raise FileNotFoundError(f"No voxels found: {voxel_paths or os.path.join(data_root, voxel_glob)}")
        if len(self.sino_paths) != len(self.voxel_paths):
            raise AssertionError(f"#sino({len(self.sino_paths)}) != #voxel({len(self.voxel_paths)})")

        # -------------------------
        # Load memmaps
        # -------------------------
        self.sinos  = [_load_sino_uaD(p) for p in self.sino_paths]    # [U,A,D_i]
        self.voxels = [_load_voxel_xyD(p) for p in self.voxel_paths]  # [X,Y,D_i]

        # -------------------------
        # Optional: resample U to match the largest U (linear interp)
        # -------------------------
        if resample_u:
            U_max = max(s.shape[0] for s in self.sinos)
            A_ref = self.sinos[0].shape[1]
            new_sinos = []
            for i, s in enumerate(self.sinos):
                U, A, D = s.shape
                if A != A_ref:
                    raise AssertionError(f"[sino] angle mismatch at {self.sino_paths[i]}: {A} vs {A_ref}")
                if U == U_max:
                    new_sinos.append(np.array(s, copy=False))
                    continue
                # Linear 1D interpolation along detector axis for every (A,D)
                x_old = np.linspace(0.0, 1.0, U, dtype=np.float32)
                x_new = np.linspace(0.0, 1.0, U_max, dtype=np.float32)
                s_res = np.empty((U_max, A, D), dtype=s.dtype)
                for a in range(A):
                    for d in range(D):
                        s_res[:, a, d] = np.interp(x_new, x_old, s[:, a, d])
                new_sinos.append(s_res)
            self.sinos = new_sinos

        # -------------------------
        # Validate shapes & index map
        # -------------------------
        U0, A0 = self.sinos[0].shape[0], self.sinos[0].shape[1]
        X0, Y0 = self.voxels[0].shape[0], self.voxels[0].shape[1]
        self.index: List[Tuple[int, int]] = []  # (file_idx, local_d)
        self.summary_lines: List[str] = []
        total_depth = 0

        for i, (s, v) in enumerate(zip(self.sinos, self.voxels)):
            U, A, Ds = s.shape
            X, Y, Dv = v.shape
            if (U, A) != (U0, A0):
                raise AssertionError(f"[sino] (U,A) mismatch at {self.sino_paths[i]}: {(U,A)} vs {(U0,A0)}")
            if (X, Y) != (X0, Y0):
                raise AssertionError(f"[voxel] (X,Y) mismatch at {self.voxel_paths[i]}: {(X,Y)} vs {(X0,Y0)}")
            if Ds != Dv:
                raise AssertionError(f"Depth mismatch in pair #{i}: sino D={Ds} vs voxel D={Dv}")
            self.summary_lines.append(f"{i+1} → sino{(U,A,Ds)} , voxel{(X,Y,Dv)}")
            for d in range(Ds):
                self.index.append((i, d))
            total_depth += Ds

        self.U, self.A, self.X, self.Y, self.D_total = U0, A0, X0, Y0, total_depth

        if report:
            print("[ConcatDepthSliceDataset]")
            for ln in self.summary_lines:
                print("  " + ln)
            print(f"  Global: U={self.U}  A={self.A}  X={self.X}  Y={self.Y}  D_total={self.D_total}")

    def __len__(self) -> int:
        return self.D_total

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        dict
          - "sino_ua":  torch.FloatTensor [U, A]
          - "voxel_xy": torch.FloatTensor [1, X, Y]
          - "pair_index": int
          - "local_z":   int
          - "global_z":  int
        """
        f, d = self.index[idx]
        s = self.sinos[f][:, :, d]   # [U,A]
        v = self.voxels[f][:, :, d]  # [X,Y]
        s_np = np.array(s, dtype=np.float32, copy=True, order="C")
        v_np = np.array(v, dtype=np.float32, copy=True, order="C")
        return {
            "sino_ua": torch.from_numpy(s_np),                # [U,A]  (≡ [X,A])
            "voxel_xy": torch.from_numpy(v_np).unsqueeze(0),  # [1,X,Y]
            "pair_index": f,
            "local_z": int(d),
            "global_z": int(idx),
        }


# ==========================================================================================
# Volume-level dataset (only permutation to z-first for the model)
# ==========================================================================================

class SinogramVoxelVolumeDataset(Dataset):
    """
    Paired sinogram/voxel **volume** dataset.

    Each index corresponds to one file-pair. Returns:
      - "sino_d_u_a":   torch.FloatTensor [D(=Z), U(=X), A]
      - "voxel_d_x_y":  torch.FloatTensor [D(=Z), X, Y]
      - "pair_index":   int

    Why z-first?
    ------------
    Your encoder slices along z to get (X,A), then along A to feed 1D conv
    across X; having z as the leading dim makes `sino[d] -> (X,A)` trivial.

    ASTRA interop
    -------------
    ASTRA 3D expects sinograms as [U, A, D]. If you pass tensors from this
    dataset to ASTRA, transpose back with `.permute(1, 2, 0)` to [U,A,D].
    """
    def __init__(self,
                 data_root: str = "data",
                 sino_glob: str = "sino/*_sino.npy",
                 voxel_glob: str = "voxel/*_voxel.npy",
                 sino_paths: Optional[List[str]] = None,
                 voxel_paths: Optional[List[str]] = None,
                 report: bool = True,
                 resample_u: bool = False):
        super().__init__()

        # File list
        if sino_paths is not None and voxel_paths is not None:
            self.sino_paths = list(sino_paths)
            self.voxel_paths = list(voxel_paths)
        else:
            self.sino_paths = sorted(glob(os.path.join(data_root, sino_glob)))
            self.voxel_paths = sorted(glob(os.path.join(data_root, voxel_glob)))

        if not self.sino_paths:
            raise FileNotFoundError(f"No sinograms found: {sino_paths or os.path.join(data_root, sino_glob)}")
        if not self.voxel_paths:
            raise FileNotFoundError(f"No voxels found: {voxel_paths or os.path.join(data_root, voxel_glob)}")
        if len(self.sino_paths) != len(self.voxel_paths):
            raise AssertionError(f"#sino({len(self.sino_paths)}) != #voxel({len(self.voxel_paths)})")

        # Load memmaps
        self.sinos  = [_load_sino_uaD(p) for p in self.sino_paths]    # [U,A,D]
        self.voxels = [_load_voxel_xyD(p) for p in self.voxel_paths]  # [X,Y,D]

        # Optional: match max U (linear interp). No angle changes.
        if resample_u:
            U_max = max(s.shape[0] for s in self.sinos)
            A_ref = self.sinos[0].shape[1]
            new_sinos = []
            for i, s in enumerate(self.sinos):
                U, A, D = s.shape
                if A != A_ref:
                    raise AssertionError(f"[sino] angle mismatch at {self.sino_paths[i]}: {A} vs {A_ref}")
                if U == U_max:
                    new_sinos.append(np.array(s, copy=False))
                    continue
                x_old = np.linspace(0.0, 1.0, U, dtype=np.float32)
                x_new = np.linspace(0.0, 1.0, U_max, dtype=np.float32)
                s_res = np.empty((U_max, A, D), dtype=s.dtype)
                for a in range(A):
                    for d in range(D):
                        s_res[:, a, d] = np.interp(x_new, x_old, s[:, a, d])
                new_sinos.append(s_res)
            self.sinos = new_sinos

        # Validate shapes
        U0, A0, D0 = self.sinos[0].shape
        X0, Y0, Dv0 = self.voxels[0].shape
        for i, (s, v) in enumerate(zip(self.sinos, self.voxels)):
            U, A, D = s.shape
            X, Y, Dv = v.shape
            if (U, A) != (U0, A0):
                raise AssertionError(f"[sino] (U,A) mismatch at {self.sino_paths[i]}: {(U,A)} vs {(U0,A0)}")
            if (X, Y) != (X0, Y0):
                raise AssertionError(f"[voxel] (X,Y) mismatch at {self.voxel_paths[i]}: {(X,Y)} vs {(X0,Y0)}")
            if D != Dv:
                raise AssertionError(f"Depth mismatch in pair #{i}: sino D={D} vs voxel D={Dv}")

        self.U, self.A, self.X, self.Y, self.D = U0, A0, X0, Y0, D0
        self.report = report
        if report:
            print(f"[SinogramVoxelVolumeDataset] pairs={len(self.sinos)} "
                  f"shapes: sino[U,A,D]=({self.U},{self.A},{self.D}) "
                  f"voxel[X,Y,D]=({self.X},{self.Y},{self.D})")

    def __len__(self) -> int:
        return len(self.sinos)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        dict
          - "sino_d_u_a":  torch.FloatTensor [D, U, A]  (z-first layout for the model)
          - "voxel_d_x_y": torch.FloatTensor [D, X, Y]  (z-first layout for the model)
          - "pair_index":  int
        """
        s_mem = self.sinos[idx]   # [U, A, D]
        v_mem = self.voxels[idx]  # [X, Y, D]

        # Minimal change: just cast to float32 and permute to z-first.
        s_vol = torch.from_numpy(np.array(s_mem, dtype=np.float32, copy=True)).permute(2, 0, 1)  # [D,U,A]
        v_vol = torch.from_numpy(np.array(v_mem, dtype=np.float32, copy=True)).permute(2, 0, 1)  # [D,X,Y]

        return {
            "sino_d_u_a": s_vol.contiguous(),
            "voxel_d_x_y": v_vol.contiguous(),
            "pair_index": int(idx),
        }
