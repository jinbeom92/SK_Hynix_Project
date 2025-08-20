def forward_consistency_loss(*args, **kwargs):
    """
    Forward Consistency Loss (planned feature).

    Concept:
        Enforces that the forward-projected reconstruction matches
        the input sinogram:  FP(R̂) ≈ S_in.

    Status:
        - Currently disabled in the physics-free pipeline.
        - Will be implemented once differentiable projectors (e.g., Joseph3D, Siddon3D)
          are fully integrated into the training loop.
        - Intended to add an additional physical constraint by penalizing
          the discrepancy between simulated forward projections of the
          predicted volume and the ground-truth sinogram.

    Args:
        *args, **kwargs: Placeholder for future arguments
            (e.g., reconstructed volume, input sinogram, projector instance).

    Raises:
        RuntimeError: Always, since this feature is not yet implemented.
    """
    raise RuntimeError("forward_consistency_loss is removed in the physics-free pipeline (planned to be reintroduced).")
