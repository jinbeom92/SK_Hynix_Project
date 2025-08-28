"""
This module defines a base interface for 3D projectors and an implementation
using a Joseph voxel-driven forward projector with a scikit-image-compatible
filtered backprojection (FBP).  The backprojection accepts sinograms covering
either 0–180° or 0–360°; when a full 360° scan is provided (i.e., the number
of angles is even), it averages opposing angles so that each ray is used only
once.  The FBP then pads the sinogram to a square, applies a 1D Fourier filter
(ramp, Shepp–Logan, cosine, Hamming or Hann), and performs a differentiable
inverse Radon transform using `grid_sample`.  The result is scaled by π/(2·A)
as described in Kak & Slaney:contentReference[oaicite:1]{index=1}.

Only the backprojection is implemented here; forward projection should be
provided by subclasses.
"""

from typing import Literal
import math
import torch
import torch.nn.functional as F
from .geometry import Parallel3DGeometry


class BaseProjector3D(torch.nn.Module):
    """
    Abstract base class for 3D projectors.

    A subclass must implement:
    - forward(vol):  [B,C,X,Y,Z] → [B,C,X,A,Z] (sino)
    - backproject(s): [B,C,X,A,Z] → [B,C,X,Y,Z] (recon)
    """

    def __init__(self, geom: Parallel3DGeometry) -> None:
        super().__init__()
        self.geom = geom

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class JosephProjector3D(BaseProjector3D):
    """
    Joseph-style voxel-driven projector with differentiable backprojection.

    The forward method must be implemented by subclasses.  The backprojection
    provided here is compatible with scikit-image’s `iradon`: it pads each
    `[X,A]` sinogram slice to a square if `ir_circle=True`, applies a 1D
    Fourier-domain filter, and then backprojects by sampling
    t = y·cosθ − x·sinθ using `grid_sample`.  The result is scaled by
    π/(2·A):contentReference[oaicite:2]{index=2}.  If a full 360° sinogram is provided
    (number of angles A is even), the sinogram is reduced to 180° by averaging
    opposing angles before filtering and backprojection.  This prevents energy
    duplication in full scans.

    Parameters
    ----------
    geom : Parallel3DGeometry
        Geometry describing the detector size and reconstruction size.
    ir_circle : bool, optional
        If True (default), reconstruct inside the inscribed circle and zero
        outside; otherwise reconstruct the full square.
    fbp_filter : str, optional
        Name of the 1D frequency-domain filter: {"ramp","shepp-logan","cosine",
        "hamming","hann","none"}.  Default is "ramp".
    """

    def __init__(
        self,
        geom: Parallel3DGeometry,
        *,
        ir_circle: bool = True,
        fbp_filter: str = "ramp",
    ) -> None:
        super().__init__(geom)
        self.ir_circle = bool(ir_circle)
        self.fbp_filter = str(fbp_filter).lower()

    # -------------------------------------------------------------------------
    # Helper: Fourier filter
    # -------------------------------------------------------------------------
    @staticmethod
    def _fourier_filter_torch(size: int, name: str, device, dtype) -> torch.Tensor:
        """
        Implements the ramp filter and optional Shepp–Logan, cosine,
        Hamming and Hann windows:contentReference[oaicite:3]{index=3}.  Returns a vector
        of length `size` suitable for broadcasting over the angle axis.
        """
        if name in ("none", "", None):
            return torch.ones(size, dtype=dtype, device=device)
        n = torch.cat(
            [
                torch.arange(1, size / 2 + 1, 2, dtype=torch.long, device=device),
                torch.arange(size / 2 - 1, 0, -2, dtype=torch.long, device=device),
            ]
        )
        f = torch.zeros(size, dtype=dtype, device=device)
        f[0] = 0.25
        f[1::2] = -1.0 / (math.pi * n.to(dtype=dtype)) ** 2
        fourier_filter = 2.0 * torch.real(torch.fft.fft(f))
        if name == "ramp":
            pass
        elif name in ("shepp-logan", "shepp"):
            omega = math.pi * torch.fft.fftfreq(size, dtype=dtype, device=device)[1:]
            fourier_filter[1:] *= torch.sin(omega) / omega
        elif name == "cosine":
            freq = torch.linspace(0.0, math.pi, steps=size, dtype=dtype, device=device)
            fourier_filter *= torch.fft.fftshift(torch.sin(freq))
        elif name == "hamming":
            fourier_filter *= torch.fft.fftshift(torch.hamming_window(size, dtype=dtype, device=device))
        elif name == "hann":
            fourier_filter *= torch.fft.fftshift(torch.hann_window(size, dtype=dtype, device=device))
        else:
            raise ValueError(f"Unknown filter: {name}")
        return fourier_filter

    # -------------------------------------------------------------------------
    # Helper: 2D backprojection
    # -------------------------------------------------------------------------
    def _iradon_scikit_grid(self, radon_image: torch.Tensor) -> torch.Tensor:
        """
        Differentiable 2D filtered backprojection using grid_sample.

        Takes a sinogram `[N, A]`, optionally pads it to a square if
        `ir_circle=True`, zero-pads to the next power-of-two along the detector
        axis, applies a 1D frequency-domain filter, and then performs the
        inverse Radon transform by bilinearly sampling at coordinates
        t = y·cosθ − x·sinθ.  The result is scaled by π/(2·A) as in
        Kak & Slaney:contentReference[oaicite:4]{index=4}.
        """
        device = radon_image.device
        dtype = radon_image.dtype

        # Cast to float32 for FFT operations
        sino = radon_image.to(torch.float32)
        N, A = sino.shape

        # Pad detector axis to circumscribed square if circle=True
        if self.ir_circle:
            diag = int(math.ceil(math.sqrt(2.0) * N))
            pad = diag - N
            old_center = N // 2
            new_center = diag // 2
            pad_before = new_center - old_center
            sino = F.pad(sino, (0, 0, pad_before, pad - pad_before))
            N = sino.shape[0]

        # Zero-pad to next power-of-two (>=64) along detector axis for FFT
        P = max(64, 1 << int(math.ceil(math.log2(2 * N))))
        sino_pad = F.pad(sino, (0, 0, 0, P - N))  # shape [P, A]

        # Construct frequency-domain filter and apply
        H = self._fourier_filter_torch(P, self.fbp_filter, device=sino_pad.device, dtype=torch.float32)
        Xf = torch.fft.rfft(sino_pad, dim=0)  # [P_rfft,A]
        Yf = Xf * H[: Xf.shape[0]].unsqueeze(1)
        sino_filt = torch.fft.irfft(Yf, n=P, dim=0)[:N, :]  # [N,A]

        # Determine output size: square of side Y
        if self.ir_circle:
            Y = N
        else:
            Y = int(math.floor((N**2) / 2.0) ** 0.5)
        radius = Y // 2

        # Build sampling grid [A,Y,Y,2]
        yy, xx = torch.meshgrid(
            torch.arange(Y, dtype=torch.float32, device=device) - radius,
            torch.arange(Y, dtype=torch.float32, device=device) - radius,
            indexing="ij",
        )
        theta = torch.arange(A, dtype=torch.float32, device=device) * (math.pi / A)
        t = yy.unsqueeze(0) * torch.cos(theta).view(-1, 1, 1) - xx.unsqueeze(0) * torch.sin(theta).view(-1, 1, 1)
        t_norm = (2.0 * (t + (N - 1) / 2.0) / max(1, N - 1)) - 1.0
        if A > 1:
            a_norm = (2.0 * torch.arange(A, dtype=torch.float32, device=device) / (A - 1)) - 1.0
        else:
            a_norm = torch.zeros(1, dtype=torch.float32, device=device)
        grid = torch.stack(
            [
                a_norm.view(-1, 1, 1).expand(-1, Y, Y),  # x = angle axis
                t_norm,                                    # y = detector axis
            ],
            dim=-1,
        )  # [A,Y,Y,2]

        # Sample sinogram using grid_sample
        inp = (
            sino_filt.permute(1, 0)    # [A, N]
                .unsqueeze(1)       # [A, 1, N]
                .unsqueeze(-1)      # [A, 1, N, 1]
                .expand(-1, 1, N, A)  # [A, 1, N, A]
        )
        recon_per_angle = F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [A,1,Y,Y]
        recon = recon_per_angle.sum(dim=0)[0]  # sum over angles → [Y,Y]

        # Scale by π/(2·A):contentReference[oaicite:5]{index=5}
        recon *= (math.pi / (2.0 * A))

        # Apply circular mask if requested
        if self.ir_circle:
            mask = (xx**2 + yy**2) > (radius**2)
            recon = torch.where(mask, torch.zeros_like(recon), recon)

        return recon.to(dtype)

    # -------------------------------------------------------------------------
    # Helper: deduplicate 360° sinograms
    # -------------------------------------------------------------------------
    @staticmethod
    def _dedup_360_to_180(s_2d: torch.Tensor) -> torch.Tensor:
        """
        Reduce a 360° sinogram to 180° by averaging opposing angles.

        If the number of angles A is even, slice the sinogram `[X,A]` into two
        halves `[X,A/2]` and `[X,A/2]`, flip the second half along the detector
        axis (u→−u), and return 0.5·(s(u,θ) + s(−u,θ+π)).  If A is odd, the
        original sinogram is returned unchanged.
        """
        X, A = s_2d.shape
        if A % 2 != 0:
            return s_2d
        A2 = A // 2
        s1 = s_2d[:, :A2]
        s2 = torch.flip(s_2d[:, A2:], dims=[0])  # detector flip
        return 0.5 * (s1 + s2)

    # -------------------------------------------------------------------------
    # Backprojection
    # -------------------------------------------------------------------------
    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a volume from a sinogram.

        Takes a 5D sinogram `[B,C,X,A,Z]` and returns a reconstructed volume
        `[B,C,X,Y,Z]`.  For each depth slice, if the number of angles A is even
        the sinogram is reduced from 360° to 180° by averaging opposing angles
        before filtering and backprojection.  The 2D backprojection itself is
        computed by `_iradon_scikit_grid`.

        The resulting 2D slice from `_iradon_scikit_grid` is square (shape
        `[Y_rec, Y_rec]`) where `Y_rec` is either the padded detector length or
        `floor((N^2)/2)**0.5`.  To support datasets with different resolutions
        (e.g. 128×128 or 256×256), we always resize this slice to the desired
        geometry dimensions `(X, Y)` using bilinear interpolation and then
        transpose it so that the final output matches `[X,Y]`.  This ensures
        compatibility with varying input sizes.
        """
        if sino.ndim != 5:
            raise ValueError(f"Expected [B,C,X,A,Z], got {tuple(sino.shape)}")
        B, C, X, A, Z = sino.shape

        # Reconstruction height
        Y = X if self.ir_circle else int(math.floor((X**2) / 2.0) ** 0.5)
        out = torch.empty((B, C, X, Y, Z), device=sino.device, dtype=sino.dtype)

        for b in range(B):
            for c in range(C):
                for z in range(Z):
                    s2d = sino[b, c, :, :, z]  # [X,A]
                    # If A is even, treat as full 360° and reduce to 180°
                    s2d_use = self._dedup_360_to_180(s2d)
                    # Backproject to obtain a square slice rec_yy [Y_rec,Y_rec]
                    rec_yy = self._iradon_scikit_grid(s2d_use)
                    # Always resize rec_yy to match geometry size (X,Y) using bilinear interpolation.
                    # This handles cases where the padded detector length differs from the target size.
                    rec_yx = F.interpolate(
                        rec_yy.unsqueeze(0).unsqueeze(0),
                        size=(X, Y),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    # Transpose to produce [X,Y]
                    rec_xy = rec_yx.t()
                    out[b, c, :, :, z] = rec_xy
        return out


def make_projector(method: Literal["joseph3d"], geom: Parallel3DGeometry) -> BaseProjector3D:
    """
    Factory for creating projectors.

    Only 'joseph3d' is supported here.  Returns an instance of
    `JosephProjector3D` initialised with the given geometry.  The caller may
    override its `ir_circle` and `fbp_filter` attributes after construction.
    """
    if method != "joseph3d":
        raise ValueError(f"Unknown projector method: {method}")
    return JosephProjector3D(geom, ir_circle=True, fbp_filter="ramp")
