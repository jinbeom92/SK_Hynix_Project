from pathlib import Path
import numpy as np

# ==========================================================================================
# I/O utilities for numpy arrays (sinograms and voxel volumes)
# ==========================================================================================

def load_npy(path: str) -> np.ndarray:
    """
    Load a NumPy ``.npy`` file into memory as-is.

    Parameters
    ----------
    path : str
        Path to the ``.npy`` file.

    Returns
    -------
    np.ndarray
        Loaded array in memory (no permutation, no dtype casting).
    """
    return np.load(path)


def save_npy(path: str, arr) -> None:
    """
    Save an array to ``.npy`` format, ensuring that parent directories exist.

    Parameters
    ----------
    path : str
        Output file path.
    arr : array-like
        Array to save.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # recursively create dirs if needed
    np.save(path, arr)


# ==========================================================================================
# Shape validation / normalization helpers (strictly (X,A,Z) and (X,Y,Z))
# ==========================================================================================

def ensure_sino_shape(arr: np.ndarray) -> np.ndarray:
    """
    Ensure a sinogram has the canonical model shape **[X, A, Z]** = (x, a, z).

    Supported input shapes
    ----------------------
    - ``[X, A]``         : single-z sinogram → upgraded to ``[X, A, 1]``.
    - ``[X, A, Z]``      : full volume → returned as-is.

    Notes
    -----
    - **No axis permutation** is performed. This function only validates
      and, if needed, inserts a singleton Z-axis at the end.
    - If your data is stored with any other axis order (e.g., ``[A, X]`` or
      ``[Z, X, A]``), convert it **before** calling this function.

    Parameters
    ----------
    arr : np.ndarray
        Input sinogram array.

    Returns
    -------
    np.ndarray
        Sinogram with shape ``[X, A, Z]``.

    Raises
    ------
    ValueError
        If the input does not match ``[X,A]`` or ``[X,A,Z]``.
    """
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
    """
    Ensure a voxel volume has the canonical model shape **[X, Y, Z]** = (x, y, z).

    Supported input shapes
    ----------------------
    - ``[X, Y]``         : single-z slice → upgraded to ``[X, Y, 1]``.
    - ``[X, Y, Z]``      : full volume → returned as-is.

    Notes
    -----
    - **No axis permutation** is performed. This function only validates
      and, if needed, inserts a singleton Z-axis at the end.
    - If your data is stored with any other axis order (e.g., ``[Y, X]`` or
      ``[Z, X, Y]``), convert it **before** calling this function.

    Parameters
    ----------
    arr : np.ndarray
        Input voxel array.

    Returns
    -------
    np.ndarray
        Voxel volume with shape ``[X, Y, Z]``.

    Raises
    ------
    ValueError
        If the input does not match ``[X,Y]`` or ``[X,Y,Z]``.
    """
    if arr.ndim == 2:  # upgrade to [X,Y,1]
        X, Y = arr.shape
        arr = arr.reshape(X, Y, 1)
    elif arr.ndim == 3:
        # Expect [X, Y, Z]; do not permute
        pass
    else:
        raise ValueError("voxel must be shaped [X,Y] or [X,Y,Z] (x,y,(z)).")
    return arr
