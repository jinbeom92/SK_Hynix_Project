from pathlib import Path
import numpy as np

# ==========================================================================================
# I/O utilities for numpy arrays (sinograms and voxel volumes)
# ==========================================================================================

def load_npy(path: str):
    """
    Load a NumPy `.npy` file.

    Args:
        path (str): Path to `.npy` file.

    Returns:
        np.ndarray: Loaded array in memory.
    """
    return np.load(path)


def save_npy(path: str, arr):
    """
    Save an array to `.npy` format, ensuring that parent directories exist.

    Args:
        path (str): Output file path.
        arr  (array-like): Array to save.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # recursively create dirs if needed
    np.save(path, arr)


# ==========================================================================================
# Shape validation / normalization helpers
# ==========================================================================================

def ensure_sino_shape(arr: np.ndarray) -> np.ndarray:
    """
    Ensure sinogram has a canonical shape.

    Supported input shapes:
      - [A, U]        : Angles × Detector bins (single slice).
      - [A, V, U]     : Angles × Views × Detector bins.

    Normalization:
      - If input is [A, U], a singleton 'view' axis (V=1) is inserted,
        yielding [A, 1, U].
      - If input is [A, V, U], it is returned as-is.

    Args:
        arr (np.ndarray): Input sinogram array.

    Returns:
        np.ndarray: Sinogram with shape [A, V, U].
    """
    if arr.ndim == 2:  # upgrade to [A,1,U]
        A, U = arr.shape
        arr = arr.reshape(A, 1, U)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError("sino must be [A,U] or [A,V,U]")
    return arr


def ensure_voxel_shape(arr: np.ndarray) -> np.ndarray:
    """
    Ensure voxel volume has a canonical shape.

    Supported input shapes:
      - [H, W]        : Single 2D slice (Height × Width).
      - [D, H, W]     : Depth × Height × Width (3D volume).

    Normalization:
      - If input is [H, W], a singleton depth axis (D=1) is inserted,
        yielding [1, H, W].
      - If input is [D, H, W], it is returned as-is.

    Args:
        arr (np.ndarray): Input voxel array.

    Returns:
        np.ndarray: Voxel volume with shape [D, H, W].
    """
    if arr.ndim == 2:  # upgrade to [1,H,W]
        H, W = arr.shape
        arr = arr.reshape(1, H, W)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError("voxel must be [H,W] or [D,H,W]")
    return arr
