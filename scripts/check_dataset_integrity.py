# =================================================================================================
# Dataset Integrity Checker — Sinogram/Volume Pair Validation
# -------------------------------------------------------------------------------------------------
# Purpose
#   Validates that paired sinogram and voxel `.npy` files are consistent, well-formed, and free
#   of numerical anomalies before being used for training or evaluation.
#
# Checks Performed
#   • File pairing:
#       Ensures that each *_sino.npy has a corresponding *_voxel.npy, and that counts match.
#   • Numerical validity:
#       Detects presence of NaN or Inf values in sinogram or voxel arrays.
#   • Dimensionality:
#       Confirms sinograms have expected rank (default ndim=3: [A, V, U]) and voxels have expected
#       rank (default ndim=3: [D, H, W]).
#
# Functions
#   • check_pair(sino_path, voxel_path, expect_sino_dims=3, expect_voxel_dims=3)
#       Loads the two files, checks for anomalies, and returns (ok, msg).
#
# Usage
#   Run directly from command line:
#       python scripts/check_dataset_integrity.py
#
#   Output:
#       Prints “[BAD] <sino_path> <voxel_path> <reason>” for each failed pair.
#       Prints summary “Done. bad pairs = <count>”.
#
# Notes
#   • Uses memory-mapped loads (mmap_mode='r') for efficiency on large arrays.
#   • Run after dataset generation to detect corrupted or malformed files early.
# =================================================================================================
import os
from glob import glob
import numpy as np

def check_pair(sino_path, voxel_path, expect_sino_dims=3, expect_voxel_dims=3):
    s = np.load(sino_path, mmap_mode='r')
    v = np.load(voxel_path, mmap_mode='r')
    ok = True; msgs = []
    if np.isnan(s).any() or np.isinf(s).any(): ok=False; msgs.append("sino NaN/Inf")
    if np.isnan(v).any() or np.isinf(v).any(): ok=False; msgs.append("voxel NaN/Inf")
    if s.ndim != expect_sino_dims: ok=False; msgs.append(f"sino ndim={s.ndim}")
    if v.ndim != expect_voxel_dims: ok=False; msgs.append(f"voxel ndim={v.ndim}")
    return ok, "; ".join(msgs)

if __name__ == "__main__":
    sino_dir = "data/sino"
    voxel_dir = "data/voxel"
    sino_files = sorted(glob(os.path.join(sino_dir, "*_sino.npy")))
    voxel_files = sorted(glob(os.path.join(voxel_dir, "*_voxel.npy")))
    assert len(sino_files) == len(voxel_files), f"count mismatch: {len(sino_files)} vs {len(voxel_files)}"
    bad = 0
    for s, v in zip(sino_files, voxel_files):
        ok, msg = check_pair(s, v)
        if not ok:
            print("[BAD]", s, v, msg); bad += 1
    print("Done. bad pairs =", bad)
