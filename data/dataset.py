import os
from glob import glob
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

# ==========================================================================================
# Utility loaders
# ==========================================================================================

def _load_sino_uaD(path: str) -> np.memmap:
    """
    Load a sinogram from .npy file as memory-mapped array.

    Expected shape:
      - [U, A, D] where
          U = detector bins,
          A = projection angles,
          D = depth slices (stack).
      - If input is [U, A], it will be reshaped into [U, A, 1].

    Args:
        path (str): Path to sinogram `.npy` file.

    Returns:
        np.memmap: Memory-mapped sinogram array of shape [U, A, D].
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
    Load a voxel volume from .npy file as memory-mapped array.

    Expected shape:
      - [X, Y, D] where
          X = voxel size in x,
          Y = voxel size in y,
          D = depth slices (stack).
      - If input is [X, Y], it will be reshaped into [X, Y, 1].

    Args:
        path (str): Path to voxel `.npy` file.

    Returns:
        np.memmap: Memory-mapped voxel array of shape [X, Y, D].
    """
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:  # upgrade to [X,Y,1]
        X, Y = arr.shape
        arr = arr.reshape(X, Y, 1)
    if arr.ndim != 3:
        raise ValueError(f"voxel ndim must be 3 (X,Y,D), got {arr.ndim} for {path}")
    return arr


# ==========================================================================================
# Dataset
# ==========================================================================================

class ConcatDepthSliceDataset(Dataset):
    """
    Dataset for paired sinogram/voxel slices with depth concatenation.

    Each dataset item corresponds to a single depth slice (z-plane),
    sampled from a matched pair of sinogram [U, A, D] and voxel [X, Y, D].

    Args:
        data_root (str): Root directory for dataset (default "data").
        sino_glob (str): Glob pattern for sinogram files under data_root.
        voxel_glob (str): Glob pattern for voxel files under data_root.
        sino_paths (Optional[List[str]]): Explicit list of sinogram paths (bypass glob).
        voxel_paths (Optional[List[str]]): Explicit list of voxel paths (bypass glob).
        report (bool): If True, prints dataset summary (default True).

    Workflow:
        - Loads all sinogram/voxel `.npy` files as memory maps.
        - Validates that each pair has consistent (U,A) and (X,Y).
        - Checks that sinogram depth D matches voxel depth D.
        - Builds a global index mapping (file_idx, depth_idx).
        - Each __getitem__ returns a dict with:
            * "sino_ua":  [U,A] slice (torch.FloatTensor)
            * "voxel_xy": [1,X,Y] slice (torch.FloatTensor, channel-first)
            * "pair_index": index of the original file pair
            * "local_z":   depth index within that file
            * "global_z":  global depth index across dataset
    """

    def __init__(self,
                 data_root: str = "data",
                 sino_glob: str = "sino/*_sino.npy",
                 voxel_glob: str = "voxel/*_voxel.npy",
                 sino_paths: Optional[List[str]] = None,
                 voxel_paths: Optional[List[str]] = None,
                 report: bool = True):
        super().__init__()
        # ----------------------------------------------------------------------------------
        # File list selection
        # ----------------------------------------------------------------------------------
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

        # ----------------------------------------------------------------------------------
        # Load memmaps
        # ----------------------------------------------------------------------------------
        self.sinos  = [_load_sino_uaD(p) for p in self.sino_paths]    # each [U,A,D_i]
        self.voxels = [_load_voxel_xyD(p) for p in self.voxel_paths]  # each [X,Y,D_i]

        # ----------------------------------------------------------------------------------
        # Validate shapes & build global index
        # ----------------------------------------------------------------------------------
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
            self.summary_lines.append(f"{i+1} → {(U,A,Ds)} , {(X,Y,Dv)}")
            for d in range(Ds):
                self.index.append((i, d))
            total_depth += Ds

        self.U, self.A = U0, A0
        self.X, self.Y = X0, Y0
        self.D_total = total_depth

        # ----------------------------------------------------------------------------------
        # Print summary (optional)
        # ----------------------------------------------------------------------------------
        if report:
            print("[ConcatDepthSliceDataset]")
            for ln in self.summary_lines:
                print("  " + ln)
            print(f"  Global: U={self.U}  A={self.A}  X={self.X}  Y={self.Y}  D_total={self.D_total}")

    def __len__(self) -> int:
        """Total number of depth slices across all file pairs."""
        return self.D_total

    def __getitem__(self, idx: int):
        """
        Fetch a single depth slice.

        Args:
            idx (int): Global depth index.

        Returns:
            dict:
              - "sino_ua": torch.FloatTensor [U,A]
              - "voxel_xy": torch.FloatTensor [1,X,Y]
              - "pair_index": int, file index
              - "local_z": int, depth within file
              - "global_z": int, absolute depth index
        """
        f, d = self.index[idx]
        s = self.sinos[f][:, :, d]  # [U,A]
        v = self.voxels[f][:, :, d] # [X,Y]
        s_np = np.array(s, dtype=np.float32, copy=True, order='C')
        v_np = np.array(v, dtype=np.float32, copy=True, order='C')
        return {
            "sino_ua": torch.from_numpy(s_np),               # [U,A]
            "voxel_xy": torch.from_numpy(v_np).unsqueeze(0), # [1,X,Y]
            "pair_index": f,
            "local_z": d,
            "global_z": int(idx),
        }
