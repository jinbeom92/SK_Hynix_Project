# =================================================================================================
# I/O Utilities for Sinogram and Voxel Data
# -------------------------------------------------------------------------------------------------
# Purpose:
#   Provides helper functions to load, save, and standardize `.npy` arrays representing
#   sinograms and voxel volumes. Ensures consistent shapes for downstream training and
#   reconstruction pipelines.
#
# Functions:
#   • load_npy(path: str) -> np.ndarray
#       - Loads a NumPy `.npy` file from the given path.
#
#   • save_npy(path: str, arr: np.ndarray)
#       - Saves a NumPy array to disk at the given path.
#       - Ensures parent directories exist before saving.
#
#   • ensure_sino_shape(arr: np.ndarray) -> np.ndarray
#       - Standardizes sinogram arrays to shape [A, V, U].
#       - Accepts:
#           [A, U]   → reshaped to [A, 1, U] (singleton V-axis).
#           [A, V, U] as-is.
#       - Raises ValueError for other shapes.
#
#   • ensure_voxel_shape(arr: np.ndarray) -> np.ndarray
#       - Standardizes voxel arrays to shape [D, H, W].
#       - Accepts:
#           [H, W]   → reshaped to [1, H, W] (singleton depth).
#           [D, H, W] as-is.
#       - Raises ValueError for other shapes.
#
# Usage:
#   sino = ensure_sino_shape(load_npy("data/sino/0001_sino.npy"))
#   voxel = ensure_voxel_shape(load_npy("data/voxel/0001_voxel.npy"))
#   save_npy("results/sino/0001.npy", sino)
# =================================================================================================
from pathlib import Path
import numpy as np

def load_npy(path: str):
    return np.load(path)

def save_npy(path: str, arr):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def ensure_sino_shape(arr: np.ndarray) -> np.ndarray:
    # Allow [A,U] or [A,V,U]; if [A,U], add V=1 axis
    if arr.ndim == 2:
        A, U = arr.shape
        arr = arr.reshape(A, 1, U)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError("sino must be [A,U] or [A,V,U]")
    return arr

def ensure_voxel_shape(arr: np.ndarray) -> np.ndarray:
    # Allow [H,W] or [D,H,W]; if [H,W], add D=1 axis
    if arr.ndim == 2:
        H, W = arr.shape
        arr = arr.reshape(1, H, W)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError("voxel must be [H,W] or [D,H,W]")
    return arr
