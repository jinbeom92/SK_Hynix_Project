from typing import Optional
import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int = None, device=None, dtype=None):
    """
    Build a normalized 1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian. If ``sigma <= 0``, returns identity kernel ``[1.0]``.
    radius : int, optional
        Kernel radius. If None, uses ``ceil(3*sigma)`` to capture ~99.7% mass.
    device, dtype :
        Torch device/dtype for the returned tensor.

    Returns
    -------
    Tensor
        1D kernel ``[K]`` with ``K = 2*radius + 1``, normalized to sum=1.
    """
    if sigma <= 0:
        k = torch.tensor([1.0], device=device, dtype=dtype)
        return k / k.sum()
    if radius is None:
        radius = int(3.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    w = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    w = w / w.sum()
    return w


class SeparableGaussianPSF2D(torch.nn.Module):
    """
    Separable Gaussian PSF over sinograms in **(x, a, z)** layout.

    Purpose
    -------
    Apply a separable 2D Gaussian blur along **detector‑z** and **detector‑x**
    axes of the sinogram. This can emulate detector/geometry blur while keeping
    the angle axis intact.

    Axis convention (model-facing)
    ------------------------------
    Sinogram: **(x, a, z)** → tensors shaped **[B, C?, X, A, Z]**.
      - Blur is applied along **Z (vertical detector v)** and **X (horizontal detector u)**.
      - The angle axis **A** is never convolved.

    Modes
    -----
    - ``angle_variant=False`` (default):
        Use single ``sigma_u/sigma_v`` for all angles. No configuration call required.
    - ``angle_variant=True``:
        Use per‑angle sigma vectors. **You must call**
        ``configure(A, device, dtype, sigma_u_vec, sigma_v_vec)`` **before** forward.

    Parameters
    ----------
    enabled : bool
        Master enable switch. If False, forward is identity.
    angle_variant : bool
        Use per‑angle sigma vectors if True.
    sigma_u : float
        Base σ along detector‑u (**X** axis).
    sigma_v : float
        Base σ along detector‑v (**Z** axis).

    Attributes
    ----------
    _sigma_u, _sigma_v :
        - angle_variant=False → scalar tensors (device/dtype aware).
        - angle_variant=True  → per‑angle tensors shaped ``[A]``.
    _A : int
        Cached number of angles after ``configure``.

    Notes
    -----
    • Implementation uses two 1D convolutions (separable blur): first **Z**, then **X**.
    • Gaussian is symmetric → under zero‑padding, **PSF^T = PSF** (self‑adjoint).
    • Kernels are built in the input dtype for safe mixed‑precision conv.
    • This PSF is differentiable w.r.t. inputs; σ’s are not learnable here
      because kernels are created outside autograd.
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

        Required when
        -------------
        ``angle_variant=True`` (otherwise optional).

        Parameters
        ----------
        A : int
            Number of projection angles.
        device : torch.device
            Target device for internal buffers.
        dtype : torch.dtype
            Dtype for internal buffers (default float32).
        sigma_u_vec : Tensor, optional
            ``[A]`` σ_u per angle; if None, filled with ``base_sigma_u``.
        sigma_v_vec : Tensor, optional
            ``[A]`` σ_v per angle; if None, filled with ``base_sigma_v``.
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

    def _canon(self, sino: torch.Tensor):
        """
        Canonicalize input to **[B, C, X, A, Z]** and remember if channel was absent.

        Accepted shapes
        ---------------
        - [B, X, A, Z]      → unsqueeze channel dim (C=1)
        - [B, C, X, A, Z]   → returned as-is
        """
        if sino.dim() == 4:   # [B,X,A,Z]
            return sino.unsqueeze(1), True
        if sino.dim() == 5:   # [B,C,X,A,Z]
            return sino, False
        raise ValueError(f"sino must be [B,X,A,Z] or [B,C,X,A,Z], got {tuple(sino.shape)}")

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Apply PSF (forward operator).

        Parameters
        ----------
        sino : Tensor
            Sinogram in **(x, a, z)** layout. Accepted shapes:
            - ``[B, X, A, Z]``
            - ``[B, C, X, A, Z]``

        Returns
        -------
        Tensor
            Blurred sinogram with the **same shape** as input.

        Implementation
        --------------
        - angle_variant=True:
            loop over angles and apply per-angle 1D conv along **Z** then **X**.
        - angle_variant=False:
            vectorized: pack ``(B, C, A)`` into batch and apply shared kernels once.
        """
        if not self.enabled:
            return sino

        x5, squeezed = self._canon(sino)  # x5: [B,C,X,A,Z]
        B, C, X, A, Z = x5.shape
        dev, dt = x5.device, x5.dtype

        if self.angle_variant:
            if self._A != A:
                raise RuntimeError(
                    f"configure(A=...) must be called before forward when angle_variant=True "
                    f"(configured A={self._A}, got A={A})."
                )
            outs = []
            # Process each angle independently: [B,C,X,Z] → conv2d on [N= B*C, 1, Z, X]
            for a in range(A):
                ku = _gaussian_kernel1d(float(self._sigma_u[a]), device=dev, dtype=dt).view(1, 1, 1, -1)
                kv = _gaussian_kernel1d(float(self._sigma_v[a]), device=dev, dtype=dt).view(1, 1, -1, 1)

                xa = x5[:, :, :, a, :]                  # [B,C,X,Z]
                xa = xa.permute(0, 1, 3, 2).contiguous()  # [B,C,Z,X]
                xa = xa.view(B * C, 1, Z, X)              # [B*C,1,Z,X]
                xa = F.conv2d(xa, kv, padding=(kv.shape[-2] // 2, 0))  # blur Z
                xa = F.conv2d(xa, ku, padding=(0, ku.shape[-1] // 2))  # blur X
                xa = xa.view(B, C, Z, X).permute(0, 1, 3, 2).contiguous()  # [B,C,X,Z]
                outs.append(xa.unsqueeze(3))  # add angle dim back at A position
            out = torch.cat(outs, dim=3)  # [B,C,X,A,Z]

        else:
            # Shared kernels (vectorized over B,C,A)
            ku = _gaussian_kernel1d(float(self._sigma_u), device=dev, dtype=dt).view(1, 1, 1, -1)
            kv = _gaussian_kernel1d(float(self._sigma_v), device=dev, dtype=dt).view(1, 1, -1, 1)

            # [B,C,X,A,Z] → [B,C,A,Z,X] → [B*C*A,1,Z,X]
            x = x5.permute(0, 1, 3, 4, 2).contiguous().view(B * C * A, 1, Z, X)
            x = F.conv2d(x, kv, padding=(kv.shape[-2] // 2, 0))  # blur Z
            x = F.conv2d(x, ku, padding=(0, ku.shape[-1] // 2))  # blur X
            out = x.view(B, C, A, Z, X).permute(0, 1, 4, 2, 3).contiguous()  # [B,C,X,A,Z]

        return out.squeeze(1) if squeezed else out

    def transpose(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose operator :math:`PSF^T`. For symmetric Gaussian kernels
        under zero padding, this equals ``forward(sino)``.

        Returns
        -------
        Tensor
            Same as ``forward(sino)``.
        """
        return self.forward(sino)
