import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set global RNG seeds and (optionally) enable deterministic kernels.

    What it does
    ------------
    • Seeds Python's `random`, NumPy, and PyTorch RNGs (CPU + all CUDA devices).
    • If `deterministic=True`, configures PyTorch/cuDNN/cuBLAS to prefer
      deterministic algorithm variants where available.

    Args
    ----
    seed : int, default 42
        Base seed used for all RNGs in this process.
    deterministic : bool, default True
        If True, attempt to make operations deterministic:
          - `torch.use_deterministic_algorithms(True, warn_only=True)`
          - set `CUBLAS_WORKSPACE_CONFIG=":4096:8"` (required by cuBLAS for determinism)
          - `torch.backends.cudnn.benchmark = False`
          - `torch.backends.cudnn.deterministic = True`

    Notes
    -----
    • Call this **at process start** (before creating models/tensors, DataLoaders, etc.).
    • Determinism can reduce performance and may change kernel choices.
    • Some ops have no deterministic implementation; with `warn_only=True`, PyTorch
      will warn instead of throwing. Remove `warn_only` or set it False if you want
      hard failures on nondeterministic ops.
    • cuBLAS determinism requires `CUBLAS_WORKSPACE_CONFIG` to be set *before*
      the first CUDA matmul/convolution executes.
    • For DataLoader workers, also seed per-worker (e.g., via `worker_init_fn` or
      `generator`) to avoid identical worker streams.
    • Full bitwise reproducibility across **different** hardware/drivers/PyTorch
      versions is not guaranteed.

    Examples
    --------
    >>> set_seed(1337)                     # deterministic on
    >>> set_seed(1337, deterministic=False)  # faster, not guaranteed reproducible
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU + CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Prefer deterministic algorithms where available.
        # warn_only=True: emit warnings instead of raising if an op is non-deterministic.
        torch.use_deterministic_algorithms(True, warn_only=True)

        # cuBLAS determinism requirement (must be set before first matmul/conv on CUDA).
        # Alternative allowed value is ":16:8" (smaller workspace), but ":4096:8" is safer.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # cuDNN flags: disable autotuner and force deterministic kernels when possible.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
