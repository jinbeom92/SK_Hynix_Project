# =================================================================================================
# set_seed — Reproducibility Helper
# -------------------------------------------------------------------------------------------------
# Purpose
#   Ensures deterministic behavior across Python, NumPy, and PyTorch for reproducible experiments.
#   Optionally enforces deterministic CUDA/cuDNN algorithms (with potential performance trade-offs).
#
# Behavior
#   • Seeds Python's `random`, NumPy, and PyTorch RNGs with the same seed.
#   • Calls `torch.cuda.manual_seed_all` for all available GPUs.
#   • If `deterministic=True`:
#       - Enables deterministic algorithms in PyTorch (warn_only=True to avoid hard errors).
#       - Sets `CUBLAS_WORKSPACE_CONFIG` to enforce reproducibility in cuBLAS GEMM kernels.
#       - Disables cuDNN benchmarking and forces deterministic convolution algorithms.
#
# Parameters
#   seed : int (default=42)
#       Random seed to apply globally.
#   deterministic : bool (default=True)
#       Whether to enforce deterministic algorithm execution in PyTorch/cuDNN.
#
# Usage
#   from utils.seed import set_seed
#   set_seed(1234, deterministic=True)
#
# Notes
#   • Deterministic settings can reduce training speed due to limited algorithm choices.
#   • `warn_only=True` allows PyTorch to fall back to non-deterministic ops if no deterministic
#     alternative is available, while still logging a warning.
# =================================================================================================
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
