# =================================================================================================
# ConcatDepthSliceDataset — Per-Depth Loader over Globally Concatenated Sino/Voxel
# -------------------------------------------------------------------------------------------------
# Purpose
#   Pair each sinogram depth-slice with the corresponding voxel depth-slice after concatenating
#   *within modality* along the depth axis:
#       • Sinograms:  cat over D  →  S_all [U, A, D_total]
#       • Voxels:     cat over D  →  V_all [X, Y, D_total]
#   The dataset then serves one global depth index z per item:
#       • sino_ua  : [U, A]  slice at depth z
#       • voxel_xy : [1, X, Y]  slice at depth z
#
# Assumptions
#   • All sinos share identical (U, A).
#   • All voxels share identical (X, Y).
#   • For each file pair i, D_sino[i] == D_voxel[i].
#
# New Features
#   • Can accept explicit `sino_paths` and `voxel_paths` lists (for group-by-group training).
#   • If not provided, falls back to `data_root`, `sino_glob`, `voxel_glob`.
# =================================================================================================
import os
from glob import glob
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

def _load_sino_uaD(path: str) -> np.memmap:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:  # upgrade to [U,A,1]
        U, A = arr.shape
        arr = arr.reshape(U, A, 1)
    if arr.ndim != 3:
        raise ValueError(f"sino ndim must be 3 (U,A,D), got {arr.ndim} for {path}")
    return arr

def _load_voxel_xyD(path: str) -> np.memmap:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:  # upgrade to [X,Y,1]
        X, Y = arr.shape
        arr = arr.reshape(X, Y, 1)
    if arr.ndim != 3:
        raise ValueError(f"voxel ndim must be 3 (X,Y,D), got {arr.ndim} for {path}")
    return arr

class ConcatDepthSliceDataset(Dataset):
    def __init__(self,
                 data_root: str = "data",
                 sino_glob: str = "sino/*_sino.npy",
                 voxel_glob: str = "voxel/*_voxel.npy",
                 sino_paths: Optional[List[str]] = None,
                 voxel_paths: Optional[List[str]] = None,
                 report: bool = True):
        """
        If `sino_paths` and `voxel_paths` are provided, use them directly.
        Otherwise search under data_root using sino_glob and voxel_glob.
        """
        super().__init__()
        # Choose file lists
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
            raise AssertionError(f"#sino({len(self.sino_paths)}) != #voxel({len(self.voxel_paths)}).")

        # Load memmaps
        self.sinos  = [_load_sino_uaD(p) for p in self.sino_paths]    # each [U,A,D_i]
        self.voxels = [_load_voxel_xyD(p) for p in self.voxel_paths]  # each [X,Y,D_i]

        # Validate shapes & build index
        U0, A0 = self.sinos[0].shape[0], self.sinos[0].shape[1]
        X0, Y0 = self.voxels[0].shape[0], self.voxels[0].shape[1]
        total_depth = 0
        self.index: List[Tuple[int,int]] = []  # (file_idx, local_d)
        self.summary_lines: List[str] = []

        for i, (s, v) in enumerate(zip(self.sinos, self.voxels)):
            U, A, Ds = s.shape
            X, Y, Dv = v.shape
            if (U, A) != (U0, A0):
                raise AssertionError(f"[sino] (U,A) mismatch at {self.sino_paths[i]}: {(U,A)} vs {(U0,A0)}")
            if (X, Y) != (X0, Y0):
                raise AssertionError(f"[voxel] (X,Y) mismatch at {self.voxel_paths[i]}: {(X,Y)} vs {(X0,Y0)}")
            if Ds != Dv:
                raise AssertionError(f"Depth mismatch in pair #{i}: sino D={Ds} vs voxel D={Dv}")
            self.summary_lines.append(f"{i+1} \u2192 {(U,A,Ds)} , {(X,Y,Dv)}")
            for d in range(Ds):
                self.index.append((i, d))
            total_depth += Ds

        self.U, self.A = U0, A0
        self.X, self.Y = X0, Y0
        self.D_total = total_depth

        if report:
            print("[ConcatDepthSliceDataset]")
            for ln in self.summary_lines:
                print("  " + ln)
            print(f"  Global: U={self.U}  A={self.A}  X={self.X}  Y={self.Y}  D_total={self.D_total}")

    def __len__(self) -> int:
        return self.D_total

    def __getitem__(self, idx: int):
        f, d = self.index[idx]
        s = self.sinos[f][:, :, d]  # [U,A] read-only view
        v = self.voxels[f][:, :, d] # [X,Y] read-only view
        s_np = np.array(s, dtype=np.float32, copy=True, order='C')
        v_np = np.array(v, dtype=np.float32, copy=True, order='C')
        return {
            "sino_ua": torch.from_numpy(s_np),               # [U,A]
            "voxel_xy": torch.from_numpy(v_np).unsqueeze(0), # [1,X,Y]
            "pair_index": f,
            "local_z": d,
            "global_z": int(idx),
        }
