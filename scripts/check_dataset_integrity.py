import os
from glob import glob
import numpy as np


def check_pair(sino_path, voxel_path, expect_sino_dims: int = 3, expect_voxel_dims: int = 3):
    """
    Validate a single sinogram/voxel pair.

    Args:
        sino_path (str): Path to sinogram .npy file.
        voxel_path (str): Path to voxel .npy file.
        expect_sino_dims (int, default=3): Expected ndim for sinogram arrays (U,A,D).
        expect_voxel_dims (int, default=3): Expected ndim for voxel arrays (X,Y,D).

    Returns:
        (ok, msg) : tuple
            ok  (bool): True if pair passes all checks.
            msg (str): Diagnostic message(s) if failed; empty string if ok.

    Checks performed:
        • File can be loaded successfully (via np.load with mmap).
        • Arrays contain no NaN/Inf values.
        • Array dimensionality matches expectation.
    """
    s = np.load(sino_path, mmap_mode='r')
    v = np.load(voxel_path, mmap_mode='r')
    ok = True
    msgs = []

    # Check for invalid values
    if np.isnan(s).any() or np.isinf(s).any():
        ok = False
        msgs.append("sino NaN/Inf")
    if np.isnan(v).any() or np.isinf(v).any():
        ok = False
        msgs.append("voxel NaN/Inf")

    # Check dimensionality
    if s.ndim != expect_sino_dims:
        ok = False
        msgs.append(f"sino ndim={s.ndim}")
    if v.ndim != expect_voxel_dims:
        ok = False
        msgs.append(f"voxel ndim={v.ndim}")

    return ok, "; ".join(msgs)


if __name__ == "__main__":
    """
    Command-line usage:
        python check_dataset_integrity.py

    Expected directory structure:
        data/sino/*.npy   — sinograms
        data/voxel/*.npy — voxel volumes

    Behavior:
        • Pairs files by sorted order of *_sino.npy and *_voxel.npy.
        • Asserts equal counts of sino/voxel files.
        • For each pair, runs check_pair() and reports any failures.
        • Prints summary of number of bad pairs.
    """
    sino_dir = "data/sino"
    voxel_dir = "data/voxel"

    sino_files = sorted(glob(os.path.join(sino_dir, "*_sino.npy")))
    voxel_files = sorted(glob(os.path.join(voxel_dir, "*_voxel.npy")))

    # Ensure matching counts
    assert len(sino_files) == len(voxel_files), \
        f"count mismatch: {len(sino_files)} vs {len(voxel_files)}"

    bad = 0
    for s, v in zip(sino_files, voxel_files):
        ok, msg = check_pair(s, v)
        if not ok:
            print("[BAD]", s, v, msg)
            bad += 1

    print("Done. bad pairs =", bad)
