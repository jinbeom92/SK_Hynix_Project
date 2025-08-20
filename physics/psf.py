from typing import Optional
import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int = None, device=None, dtype=None):
    """
    Build a normalized 1D Gaussian kernel.

    Args:
        sigma (float): Standard deviation of the Gaussian.
                       If sigma <= 0, returns identity kernel [1.0].
        radius (int, optional): Kernel radius. If None, uses ceil(3*sigma).
        device, dtype: Torch device/dtype for the returned tensor.

    Returns:
        Tensor: [K] kernel, where K = 2*radius + 1, normalized to sum=1.

    Notes:
        • Using radius ≈ 3σ captures ~99.7% of the Gaussian mass.
        • When sigma <= 0, the kernel is the identity (no blur).
    """
    if sigma <= 0:
        k = torch.tensor([1.0], device=device, dtype=dtype)
        return k / k.sum()
    if radius is None:
        radius = int(3.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    w = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    w = w / w.sum()
    return w


class SeparableGaussianPSF2D(torch.nn.Module):
    """
    Angle-optional separable Gaussian PSF on sinograms.

    Purpose
    -------
    • Applies a separable 2D Gaussian blur along (v,u) axes of the sinogram.
    • Supports angle-invariant (single σ_u, σ_v) or angle-variant (per-angle) PSF.
    • Intended as an optional physics component; keep `enabled=False` when PSF
      should be inactive.

    Modes
    -----
    • angle_variant=False (default):
        - Use base_sigma_u/base_sigma_v for all angles.
        - No need to call `configure` before `forward`.
    • angle_variant=True:
        - PSF sigmas may differ per angle (A).
        - You MUST call `configure(A, device, dtype, sigma_u_vec, sigma_v_vec)` first.

    Shapes
    ------
    Input : sino [B, A, V, U]
    Output: same shape [B, A, V, U]

    Args (constructor)
    ------------------
    enabled (bool): Master enable switch. If False, forward is identity.
    angle_variant (bool): Use per-angle sigma vectors if True.
    sigma_u (float): Base σ along detector-u (horizontal).
    sigma_v (float): Base σ along detector-v (vertical).

    Attributes
    ----------
    _sigma_u, _sigma_v:
        • angle_variant=False: scalar tensors (dtype/device-configured).
        • angle_variant=True : per-angle tensors of shape [A].

    Notes
    -----
    • Convolution is implemented via two 1D convs (kv then ku), i.e., separable blur.
    • Gaussian kernels are symmetric; thus transpose() == forward().
    • For mixed precision, kernels are created in the input dtype for safe conv.
    • This module is differentiable w.r.t. inputs, but not w.r.t. σ (kernels are
      created outside autograd). If you need learnable PSF, register σ as params
      and rebuild kernels inside forward with care.
    """

    def __init__(self, enabled: bool = False, angle_variant: bool = False,
                 sigma_u: float = 0.7, sigma_v: float = 0.7):
        super().__init__()
        self.enabled = bool(enabled)
        self.angle_variant = bool(angle_variant)
        self.base_sigma_u = float(sigma_u)
        self.base_sigma_v = float(sigma_v)

        # Internal buffers holding configured sigmas (scalar or per-angle vectors)
        self.register_buffer("_sigma_u", torch.tensor(0.0))
        self.register_buffer("_sigma_v", torch.tensor(0.0))
        self._A = 0  # number of angles (tracked after configure)

    @torch.no_grad()
    def configure(self, A: int, device: torch.device, dtype: torch.dtype = torch.float32,
                  sigma_u_vec: Optional[torch.Tensor] = None,
                  sigma_v_vec: Optional[torch.Tensor] = None):
        """
        Prepare per-angle sigma tensors for the active geometry.

        Required when:
            • angle_variant=True (otherwise optional).

        Args:
            A (int): number of projection angles.
            device (torch.device): target device for internal buffers.
            dtype (torch.dtype): dtype for internal buffers (default float32).
            sigma_u_vec (Tensor, optional): [A] σ_u per angle; if None, fill with base_sigma_u.
            sigma_v_vec (Tensor, optional): [A] σ_v per angle; if None, fill with base_sigma_v.
        """
        self._A = int(A)
        if self.angle_variant:
            if sigma_u_vec is None or sigma_v_vec is None:
                sigma_u_vec = torch.full((A,), self.base_sigma_u, device=device, dtype=dtype)
                sigma_v_vec = torch.full((A,), self.base_sigma_v, device=device, dtype=dtype)
            self._sigma_u = sigma_u_vec.to(device=device, dtype=dtype)
            self._sigma_v = sigma_v_vec.to(device=device, dtype=dtype)
        else:
            # Angle-invariant single σ values
            self._sigma_u = torch.tensor(self.base_sigma_u, device=device, dtype=dtype)
            self._sigma_v = torch.tensor(self.base_sigma_v, device=device, dtype=dtype)

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Apply PSF (forward operator).

        Args:
            sino (Tensor): [B, A, V, U] sinogram.

        Returns:
            Tensor: [B, A, V, U] blurred sinogram (or identity if disabled).

        Implementation
        --------------
        • angle_variant=True:
            - Loop over angles and apply per-angle 1D conv along v then u.
        • angle_variant=False:
            - Vectorized: reshape [B,A,V,U] → [B*A,1,V,U] and apply same kernels once.
        """
        if not self.enabled:
            return sino

        B, A, V, U = sino.shape

        if self.angle_variant:
            # Per-angle kernels; loop over angles
            outs = []
            for a in range(A):
                ku = _gaussian_kernel1d(float(self._sigma_u[a]), device=sino.device, dtype=sino.dtype).view(1, 1, 1, -1)
                kv = _gaussian_kernel1d(float(self._sigma_v[a]), device=sino.device, dtype=sino.dtype).view(1, 1, -1, 1)

                x = sino[:, a:a+1]                               # [B,1,V,U]
                # Separable blur: first along v, then along u
                x = F.conv2d(x, kv, padding=(kv.shape[-2] // 2, 0))
                x = F.conv2d(x, ku, padding=(0, ku.shape[-1] // 2))
                outs.append(x)

            out = torch.cat(outs, dim=1)                         # [B,A,V,U]
            return out

        else:
            # Angle-invariant; single kernels shared across all angles (vectorized)
            ku = _gaussian_kernel1d(float(self._sigma_u), device=sino.device, dtype=sino.dtype).view(1, 1, 1, -1)
            kv = _gaussian_kernel1d(float(self._sigma_v), device=sino.device, dtype=sino.dtype).view(1, 1, -1, 1)

            x = sino.view(B * A, 1, V, U)                        # pack angles into batch
            x = F.conv2d(x, kv, padding=(kv.shape[-2] // 2, 0))  # blur along v
            x = F.conv2d(x, ku, padding=(0, ku.shape[-1] // 2))  # then along u
            return x.view(B, A, V, U)

    def transpose(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose operator PSF^T.

        Returns:
            Tensor: same as forward(sino).

        Notes:
            • Gaussian kernels are symmetric, so PSF is self-adjoint
              (PSF^T == PSF) under these boundary conditions.
        """
        return self.forward(sino)
