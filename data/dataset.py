# =================================================================================================
# NpySinoVoxelDataset
# -------------------------------------------------------------------------------------------------
# Purpose:
#   A PyTorch Dataset wrapper for paired CT training data stored as NumPy `.npy` arrays.
#   Each sample consists of a sinogram, its corresponding reconstructed voxel volume, and
#   the acquisition angles. Provides standardized tensor outputs for downstream training.
#
# Design:
#   • Input data structure (under `data_root`):
#       - sino/{id}_sino.npy   : sinogram [A, V, U]
#       - voxel/{id}_voxel.npy : voxel volume [D, H, W]
#       - sino/{id}_angles.npy : (optional) angle list [A]
#   • If angle file is missing, generates a default evenly spaced set of angles in [0, π).
#   • Shape safety:
#       - `ensure_sino_shape`: guarantees sinogram shape is [A, V, U] (adds singleton V if needed).
#       - `ensure_voxel_shape`: guarantees voxel shape is [D, H, W] (adds singleton D if needed).
#   • Returns all arrays as `torch.FloatTensor` for direct model input.
#
# Output sample (dict):
#   {
#     "id"     : str           — unique identifier for the sample
#     "sino"   : FloatTensor   — sinogram [A, V, U]
#     "voxel"  : FloatTensor   — voxel volume [D, H, W]
#     "angles" : FloatTensor   — acquisition angles [A]
#   }
#
# Parameters:
#   id_list (List[str])       : list of sample IDs (filenames without suffix).
#   data_root (str, default="data"): root directory containing `sino/` and `voxel/` subdirs.
#   default_angles (int, optional): fallback number of projection angles if angle file missing.
#
# Usage:
#   ds = NpySinoVoxelDataset(id_list=["0001","0002"], data_root="data")
#   loader = DataLoader(ds, batch_size=4, shuffle=True)
#   batch = next(iter(loader))
# =================================================================================================
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
from .io import load_npy, ensure_sino_shape, ensure_voxel_shape

class NpySinoVoxelDataset(Dataset):
    def __init__(self, id_list: List[str], data_root: str = "data", default_angles: Optional[int] = None):
        self.ids = id_list
        self.root = Path(data_root)
        self.default_angles = default_angles

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]
        sino_p = self.root / "sino" / f"{id_}_sino.npy"
        vox_p  = self.root / "voxel" / f"{id_}_voxel.npy"
        ang_p  = self.root / "sino" / f"{id_}_angles.npy"
        sino = ensure_sino_shape(load_npy(str(sino_p)))  # [A,V,U]
        vox  = ensure_voxel_shape(load_npy(str(vox_p)))  # [D,H,W]
        if ang_p.exists():
            angles = load_npy(str(ang_p)).astype(np.float32)  # [A]
        else:
            A = sino.shape[0]
            angles = np.linspace(0, np.pi, A, endpoint=False, dtype=np.float32)
        return {
            "id": id_,
            "sino": torch.from_numpy(sino.astype(np.float32)),   # [A,V,U]
            "voxel": torch.from_numpy(vox.astype(np.float32)),   # [D,H,W]
            "angles": torch.from_numpy(angles),
        }
