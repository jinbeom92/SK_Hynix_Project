from pathlib import Path
import numpy as np

# ==========================================================================================
# I/O utilities for numpy arrays (sinograms and voxel volumes)
# ==========================================================================================

def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def save_npy(path: str, arr) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # recursively create dirs if needed
    np.save(path, arr)


# ==========================================================================================
# Shape validation / normalization helpers (strictly (X,A,Z) and (X,Y,Z))
# ==========================================================================================

def ensure_sino_shape(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:  # upgrade to [X,A,1]
        X, A = arr.shape
        arr = arr.reshape(X, A, 1)
    elif arr.ndim == 3:
        # Expect [X, A, Z]; do not permute
        pass
    else:
        raise ValueError("sinogram must be shaped [X,A] or [X,A,Z] (x,a,(z)).")
    return arr


def ensure_voxel_shape(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:  # upgrade to [X,Y,1]
        X, Y = arr.shape
        arr = arr.reshape(X, Y, 1)
    elif arr.ndim == 3:
        # Expect [X, Y, Z]; do not permute
        pass
    else:
        raise ValueError("voxel must be shaped [X,Y] or [X,Y,Z] (x,y,(z)).")
    return arr
