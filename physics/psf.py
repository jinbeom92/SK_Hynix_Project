# =================================================================================================
# SeparableGaussianPSF2D — Detector-Domain Point Spread Function
# -------------------------------------------------------------------------------------------------
# Purpose
#   Implements a differentiable 2D Gaussian point spread function (PSF) acting on detector-plane
#   sinograms. Models blur along detector u/v directions either as angle-invariant or per-angle
#   variant. Can be used to simulate detector blur, enforce consistency between forward and
#   backward operators, or as a learnable data-fidelity component.
#
# Components
#   • _gaussian_kernel1d(sigma, radius=None, device, dtype)
#       Utility to generate a normalized 1D Gaussian kernel. If sigma<=0, returns δ-kernel [1].
#
#   • SeparableGaussianPSF2D
#       - enabled       : bool, turns PSF on/off.
#       - angle_variant : bool, if True allows per-angle σ_u, σ_v vectors.
#       - base_sigma_u/v: float base widths if no per-angle vectors are provided.
#       - configure(A, device, dtype, sigma_u_vec, sigma_v_vec):
#           Prepares internal σ tensors (scalars or length-A vectors).
#       - forward(sino):
#           Applies separable conv with Gaussian kernels along detector axes.
#           Shape: sino [B, A, V, U] → [B, A, V, U].
#       - transpose(sino):
#           Applies PSFᵀ. Since Gaussian is symmetric, this equals forward().
#
# Modes
#   • Angle-Invariant:
#       Applies the same σ_u, σ_v kernel across all projection angles.
#   • Angle-Variant:
#       Allows specifying per-angle σ vectors for anisotropic blurring.
#
# Usage
#   psf = SeparableGaussianPSF2D(enabled=True, sigma_u=0.7, sigma_v=0.7)
#   psf.configure(A=360, device=torch.device("cuda"))
#   sino_blurred = psf(sino)         # Apply PSF
#   sino_restored = psf.transpose(sino_blurred)  # Symmetric (PSFᵀ)
#
# Notes
#   • Kernels are generated on-the-fly for each forward pass.
#   • Uses torch.nn.functional.conv2d for efficient separable convolution.
#   • Angle-variant mode is more expensive (loops over A).
# =================================================================================================
from typing import Optional
import torch
import torch.nn.functional as F

def _gaussian_kernel1d(sigma: float, radius: int = None, device=None, dtype=None):
    if sigma <= 0:
        k = torch.tensor([1.0], device=device, dtype=dtype)
        return k / k.sum()
    if radius is None:
        radius = int(3.0 * sigma + 0.5)
    x = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    w = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    w = w / w.sum()
    return w

class SeparableGaussianPSF2D(torch.nn.Module):
    def __init__(self, enabled: bool = False, angle_variant: bool = False,
                 sigma_u: float = 0.7, sigma_v: float = 0.7):
        super().__init__()
        self.enabled = bool(enabled)
        self.angle_variant = bool(angle_variant)
        self.base_sigma_u = float(sigma_u)
        self.base_sigma_v = float(sigma_v)
        self.register_buffer("_sigma_u", torch.tensor(0.0))
        self.register_buffer("_sigma_v", torch.tensor(0.0))
        self._A = 0

    @torch.no_grad()
    def configure(self, A: int, device: torch.device, dtype: torch.dtype = torch.float32,
                  sigma_u_vec: Optional[torch.Tensor] = None,
                  sigma_v_vec: Optional[torch.Tensor] = None):
        """Prepare per-angle sigma tensors for the active geometry."""
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

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        """Apply PSF (forward). sino: [B,A,V,U]."""
        if not self.enabled: return sino
        B,A,V,U = sino.shape
        if self.angle_variant:
            outs = []
            for a in range(A):
                ku = _gaussian_kernel1d(float(self._sigma_u[a]), device=sino.device, dtype=sino.dtype).view(1,1,1,-1)
                kv = _gaussian_kernel1d(float(self._sigma_v[a]), device=sino.device, dtype=sino.dtype).view(1,1,-1,1)
                x = sino[:,a:a+1]                                  # [B,1,V,U]
                x = F.conv2d(x, kv, padding=(kv.shape[-2]//2, 0))
                x = F.conv2d(x, ku, padding=(0, ku.shape[-1]//2))
                outs.append(x)
            out = torch.cat(outs, dim=1)                            # [B,A,V,U]
            return out
        else:
            ku = _gaussian_kernel1d(float(self._sigma_u), device=sino.device, dtype=sino.dtype).view(1,1,1,-1)
            kv = _gaussian_kernel1d(float(self._sigma_v), device=sino.device, dtype=sino.dtype).view(1,1,-1,1)
            x = sino.view(B*A, 1, V, U)
            x = F.conv2d(x, kv, padding=(kv.shape[-2]//2, 0))
            x = F.conv2d(x, ku, padding=(0, ku.shape[-1]//2))
            return x.view(B, A, V, U)

    def transpose(self, sino: torch.Tensor) -> torch.Tensor:
        """Apply PSF^T; Gaussian is symmetric so this equals forward()."""
        return self.forward(sino)
