"""
Separable 2D Gaussian PSF for sinogram blurring.

In tomographic reconstruction, a point spread function (PSF) can model
detector blur or system response.  This module implements a separable
Gaussian PSF operating on sinograms.  Given a sinogram tensor
[B,C,X,A,Z] (or [B,X,A,Z] which will be unsqueezed to [B,1,X,A,Z]),
it applies a 1D Gaussian blur along the detector pixel axis (X) and
optionally along the depth axis (Z), treating each angle independently
(angle_variant=True) or all angles identically (angle_variant=False).

Parameters:
    enabled: Whether the PSF is active.  If False, inputs are returned unchanged.
    angle_variant: If True, use per-angle σ values; otherwise use single σ values.
    sigma_u: Base standard deviation of the Gaussian kernel along detector X.
    sigma_v: Base standard deviation along depth Z.

Methods:
    configure(A, device, dtype, sigma_u_vec, sigma_v_vec):
        Precompute σ values for angle_variant=True.  Must be called before
        forward if angle_variant is enabled.
    forward(sino):
        Apply the separable Gaussian blur.  Returns a tensor of the same shape
        as sino (with an added channel dimension removed if the input lacked it).
    transpose(sino):
        Alias for forward() since a separable Gaussian is symmetric.

The Gaussian kernel radius defaults to ceil(3·σ) to capture 99.7% of the
mass.  Normalisation ensures the kernel sums to 1.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int | None = None, device=None, dtype=None):
    """Create a 1D Gaussian kernel with standard deviation σ."""
    if sigma <= 0:
        k = torch.tensor([1.0], device=device, dtype=dtype)
        return k / k.sum()
    if radius is None:
        radius = int(3.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    w = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    return w / w.sum()


class SeparableGaussianPSF2D(torch.nn.Module):
    """Apply a separable Gaussian PSF along X and Z axes of a sinogram.

    Args:
        enabled: If False, the forward pass returns the input unchanged.
        angle_variant: If True, σ values may vary per angle; otherwise a single
            σ_u and σ_v are used for all angles.
        sigma_u: Base standard deviation for the detector (X) blur.
        sigma_v: Base standard deviation for the depth (Z) blur.

    The internal buffers `_sigma_u`, `_sigma_v` store per-angle σ values when
    angle_variant=True, and scalars otherwise.  `configure` must be called
    before the first forward pass when angle_variant=True to set these buffers.
    """

    def __init__(
        self,
        enabled: bool = False,
        angle_variant: bool = False,
        sigma_u: float = 0.7,
        sigma_v: float = 0.7,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.angle_variant = bool(angle_variant)
        self.base_sigma_u = float(sigma_u)
        self.base_sigma_v = float(sigma_v)
        self.register_buffer("_sigma_u", torch.tensor(0.0))
        self.register_buffer("_sigma_v", torch.tensor(0.0))
        self._A = 0

    @torch.no_grad()
    def configure(
        self,
        A: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        sigma_u_vec: Optional[torch.Tensor] = None,
        sigma_v_vec: Optional[torch.Tensor] = None,
    ) -> None:
        """Set per-angle σ values when angle_variant=True.

        Args:
            A: Number of projection angles.
            device: Device for σ buffers.
            dtype: Data type for σ buffers.
            sigma_u_vec: Optional tensor of shape [A] specifying σ_u per angle.
            sigma_v_vec: Optional tensor of shape [A] specifying σ_v per angle.

        If angle_variant=False, this method simply copies base σ values to the
        internal buffers as scalars.
        """
        self._A = int(A)
        if self.angle_variant:
            if sigma_u_vec is None or sigma_v_vec is None:
                sigma_u_vec = torch.full((A,), self.base_sigma_u, device=device, dtype=dtype)
                sigma_v_vec = torch.full((A,), self.base_sigma_v, device=device, dtype=dtype)
            self._sigma_u = sigma_u_vec.to(device=device, dtype=dtype)
            self._sigma_v = sigma_v_vec.to(device=device, dtype=dtype)
        else:
            self._sigma_u = torch.tensor(self.base_sigma_u, device=device, dtype=dtype)
            self._sigma_v = torch.tensor(self.base_sigma_v, device=device, dtype=dtype)

    def _canon(self, sino: torch.Tensor):
        """Canonicalise input to [B,1,X,A,Z] and report whether channel was squeezed."""
        if sino.dim() == 4:
            return sino.unsqueeze(1), True
        if sino.dim() == 5:
            return sino, False
        raise ValueError(f"sino must be [B,X,A,Z] or [B,C,X,A,Z], got {tuple(sino.shape)}")

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        """Apply separable Gaussian blur to the sinogram.

        Args:
            sino: Tensor [B,X,A,Z] or [B,C,X,A,Z].

        Returns:
            Tensor with the same shape as sino (channel dim is removed if added).
        """
        if not self.enabled:
            return sino
        x5, squeezed = self._canon(sino)  # [B,C,X,A,Z]
        B, C, X, A, Z = x5.shape
        dev, dt = x5.device, x5.dtype
        if self.angle_variant:
            if self._A != A:
                raise RuntimeError(
                    f"configure(A=...) must be called before forward when angle_variant=True "
                    f"(configured A={self._A}, got A={A})."
                )
            outs = []
            for a in range(A):
                ku = _gaussian_kernel1d(float(self._sigma_u[a]), device=dev, dtype=dt).view(1, 1, 1, -1)
                kv = _gaussian_kernel1d(float(self._sigma_v[a]), device=dev, dtype=dt).view(1, 1, -1, 1)
                xa = x5[:, :, :, a, :]  # [B,C,X,Z]
                xa = xa.permute(0, 1, 3, 2).contiguous()  # [B,C,Z,X]
                xa = xa.view(B * C, 1, Z, X)
                xa = F.conv2d(xa, kv, padding=(kv.shape[-2] // 2, 0))
                xa = F.conv2d(xa, ku, padding=(0, ku.shape[-1] // 2))
                xa = xa.view(B, C, Z, X).permute(0, 1, 3, 2).contiguous()
                outs.append(xa.unsqueeze(3))
            out = torch.cat(outs, dim=3)
        else:
            ku = _gaussian_kernel1d(float(self._sigma_u), device=dev, dtype=dt).view(1, 1, 1, -1)
            kv = _gaussian_kernel1d(float(self._sigma_v), device=dev, dtype=dt).view(1, 1, -1, 1)
            x = x5.permute(0, 1, 3, 4, 2).contiguous().view(B * C * A, 1, Z, X)
            x = F.conv2d(x, kv, padding=(kv.shape[-2] // 2, 0))
            x = F.conv2d(x, ku, padding=(0, ku.shape[-1] // 2))
            out = x.view(B, C, A, Z, X).permute(0, 1, 4, 2, 3).contiguous()
        return out.squeeze(1) if squeezed else out

    def transpose(self, sino: torch.Tensor) -> torch.Tensor:
        """Transpose PSF operation (identical to forward for symmetric kernels)."""
        return self.forward(sino)
