from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry import Parallel3DGeometry


class BaseProjector3D(torch.nn.Module):
    """
    Abstract base class for 3D projectors (forward/backprojection).

    Axis convention (model-facing)
    ------------------------------
    * Volume      : **(x, y, z)** → ``[B, C, X, Y, Z]``
    * Sinogram    : **(x, a, z)** → ``[B, C, X, A, Z]``

    Concrete implementations may internally permute to their preferred
    layout, but the public API MUST adhere to the above shapes.

    Parameters
    ----------
    geom : Parallel3DGeometry
        Geometry specification (detector, volume, angles, step caps).
    """

    def __init__(self, geom: Parallel3DGeometry):
        super().__init__()
        self.geom = geom

    def reset_geometry(self, geom: Parallel3DGeometry):
        """
        Reset internal buffers when geometry changes.

        Implementations should update any cached tensors that depend on geometry
        (angles, trig caches, detector coordinates, step sizes, etc.).
        """
        raise NotImplementedError

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Forward projection: **volume → sinogram**.

        Parameters
        ----------
        vol : Tensor
            Model-facing volume ``[B, C, X, Y, Z]`` (x,y,z).

        Returns
        -------
        Tensor
            Model-facing sinogram ``[B, C, X, A, Z]`` (x,a,z).
        """
        raise NotImplementedError

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Backprojection: **sinogram → volume**.

        Parameters
        ----------
        sino : Tensor
            Model-facing sinogram ``[B, C, X, A, Z]`` (x,a,z).

        Returns
        -------
        Tensor
            Model-facing volume ``[B, C, X, Y, Z]`` (x,y,z).
        """
        raise NotImplementedError


class JosephProjector3D(BaseProjector3D):
    """
    Joseph 3D projector (voxel-driven / sampled ray integration).

    Backprojection override
    -----------------------
    The `backproject()` method uses a faithful PyTorch port of scikit-image's
    `iradon` (filtered backprojection) **slice-wise along z**. Each input slice
    is ``[X, A]`` (detector-u × angles). The 2D FBP reconstructs ``[Y, Y]`` on
    a square grid; we then **resample the width** to X and transpose to produce
    model-facing ``[X, Y]`` (x,y).
    """

    def __init__(self, geom: Parallel3DGeometry, n_steps: Optional[int] = None):
        super().__init__(geom)
        # Angle/trig caches (used by forward)
        self.register_buffer("angles", geom.angles.clone().detach())
        self.register_buffer("cos_angles", torch.cos(self.angles))
        self.register_buffer("sin_angles", torch.sin(self.angles))

        # Joseph forward parameters
        self.n_steps = int(n_steps if n_steps is not None else geom.n_steps_cap)
        V, U = geom.V, geom.U
        sv, su = geom.det_spacing
        v = (torch.arange(V, dtype=torch.float32) - (V - 1) / 2.0) * sv
        u = (torch.arange(U, dtype=torch.float32) - (U - 1) / 2.0) * su
        self.register_buffer("u_phys", u)
        self.register_buffer("v_phys", v)
        sd, sy, sx = geom.voxel_size
        D, H, W = geom.D, geom.H, geom.W
        self.sx = float(sx); self.sy = float(sy); self.sd = float(sd)
        T = 0.5 * float(((W - 1) * sx) ** 2 + ((H - 1) * sy) ** 2) ** 0.5
        self.T = float(T)
        self.delta_t = (2.0 * self.T) / float(self.n_steps)
        self.step_chunk = max(1, min(8, self.n_steps))
        self.c_chunk = 4

        # FBP options (externally adjustable via config/train loop)
        # Names mirror scikit-image: {'ramp','shepp-logan','cosine','hamming','hann', None}
        self.ir_filter_name: Optional[str] = "ramp"
        self.ir_interpolation: str = "linear"  # {'linear','nearest'}
        self.ir_circle: bool = True            # circular support mask

    @torch.no_grad()
    def reset_geometry(self, geom: Parallel3DGeometry):
        """Update geometry-dependent buffers (angles, detector coords, step sizes)."""
        self.geom = geom
        new_angles = geom.angles.detach().to(self.angles.device, dtype=self.angles.dtype)
        self.angles.resize_(new_angles.shape).copy_(new_angles)
        self.cos_angles.resize_(new_angles.shape).copy_(torch.cos(new_angles))
        self.sin_angles.resize_(new_angles.shape).copy_(torch.sin(new_angles))

        V, U = geom.V, geom.U
        sv, su = geom.det_spacing
        v = (torch.arange(V, dtype=torch.float32, device=self.angles.device) - (V - 1) / 2.0) * sv
        u = (torch.arange(U, dtype=torch.float32, device=self.angles.device) - (U - 1) / 2.0) * su
        if hasattr(self, "u_phys"): self.u_phys.resize_(u.shape).copy_(u)
        else: self.register_buffer("u_phys", u)
        if hasattr(self, "v_phys"): self.v_phys.resize_(v.shape).copy_(v)
        else: self.register_buffer("v_phys", v)

        sd, sy, sx = geom.voxel_size
        D, H, W = geom.D, geom.H, geom.W
        self.sx = float(sx); self.sy = float(sy); self.sd = float(sd)
        T = 0.5 * float(((W - 1) * sx) ** 2 + ((H - 1) * sy) ** 2) ** 0.5
        self.T = float(T)
        self.delta_t = (2.0 * self.T) / float(self.n_steps)
        self.step_chunk = max(1, min(self.step_chunk, self.n_steps))

    # ----------------------------
    # Joseph forward (unchanged)
    # ----------------------------
    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Volume → Sinogram (Joseph forward), model-facing I/O.

        Parameters
        ----------
        vol : Tensor
            [B, C, X, Y, Z] volume (x,y,z).

        Returns
        -------
        Tensor
            [B, C, X, A, Z] sinogram (x,a,z).
        """
        # Convert model layout [B,C,X,Y,Z] → internal [B,C,D,H,W]
        vol = vol.permute(0, 1, 4, 3, 2).contiguous()
        B, C, D, H, W = vol.shape
        A = int(self.angles.numel())
        V, U = self.geom.V, self.geom.U
        device = vol.device

        def _safe_norm(idx: torch.Tensor, size: int) -> torch.Tensor:
            if size <= 1:
                return torch.zeros_like(idx)
            return (2.0 * idx) / (size - 1) - 1.0

        sino_full = torch.empty((B, C, A, V, U), device=device, dtype=vol.dtype)

        for a0 in range(0, A, self.geom.angle_chunk):
            a1 = min(a0 + self.geom.angle_chunk, A)
            cos_t = self.cos_angles[a0:a1].view(-1, 1, 1, 1)
            sin_t = self.sin_angles[a0:a1].view(-1, 1, 1, 1)
            u_phys = self.u_phys.view(1, 1, 1, U)
            v_phys = self.v_phys.view(1, 1, V, 1)
            x0 = -u_phys * sin_t
            y0 =  u_phys * cos_t
            z0 =  v_phys.expand(cos_t.shape[0], 1, V, 1)

            sino_accum = torch.zeros(B, C, cos_t.shape[0], V, U, device=device, dtype=vol.dtype)

            if B == 1:
                v_base = vol.expand(cos_t.shape[0], C, D, H, W)
            else:
                v_base = vol.repeat_interleave(cos_t.shape[0], dim=0)

            for c0 in range(0, C, max(1, self.c_chunk)):
                c1 = min(c0 + max(1, self.c_chunk), C)
                v_in = v_base[:, c0:c1].contiguous()
                accum_c = torch.zeros(B, c1 - c0, cos_t.shape[0], V, U, device=device, dtype=vol.dtype)

                for s0 in range(0, self.n_steps, self.step_chunk):
                    s1 = min(s0 + self.step_chunk, self.n_steps)
                    i = torch.arange(s0, s1, device=device, dtype=vol.dtype).view(1, -1, 1, 1)
                    t = (-self.T + (i + 0.5) * self.delta_t)

                    x = (x0 + t * cos_t).expand(-1, -1, V, -1)
                    y = (y0 + t * sin_t).expand(-1, -1, V, -1)
                    z = (z0 + torch.zeros_like(t)).expand(-1, -1, -1, U)

                    x_idx = x / self.sx + (W - 1) / 2.0
                    y_idx = y / self.sy + (H - 1) / 2.0
                    z_idx = z / self.sd + (D - 1) / 2.0
                    x_n = _safe_norm(x_idx, W); y_n = _safe_norm(y_idx, H); z_n = _safe_norm(z_idx, D)

                    grid = torch.stack([x_n, y_n, z_n], dim=-1)  # [Aa,S,V,U,3]
                    grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1, -1).reshape(
                        B * cos_t.shape[0], s1 - s0, V, U, 3
                    )
                    if grid.dtype != v_in.dtype:
                        grid = grid.to(v_in.dtype)

                    samples = F.grid_sample(v_in, grid, mode="bilinear",
                                            padding_mode="zeros", align_corners=True)
                    part = samples.sum(dim=2).view(B, cos_t.shape[0], c1 - c0, V, U).permute(0, 2, 1, 3, 4)
                    accum_c.add_(part)

                    del i, t, x, y, z, x_idx, y_idx, z_idx, x_n, y_n, z_n, grid, samples, part

                sino_accum[:, c0:c1].add_(accum_c)
                del v_in, accum_c

            sino = sino_accum.mul(self.delta_t).contiguous()
            sino_full[:, :, a0:a1, :, :] = sino
            del sino_accum, sino, v_base

        return sino_full.permute(0, 1, 4, 2, 3).contiguous()  # [B,C,X,A,Z]

    # ----------------------------
    # FBP (scikit-image iradon) BP
    # ----------------------------
    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return max(64, next power-of-two >= 2*n)."""
        return max(64, 1 << ((2 * n - 1).bit_length()))

    @staticmethod
    def _sinogram_circle_to_square_torch(sino: torch.Tensor) -> torch.Tensor:
        """
        Pad rows so that the circle-inscribed width becomes square (skimage-compatible).

        Notes
        -----
        Input is [N, A] where N is detector bins along u.
        """
        import math
        N, A = sino.shape
        diagonal = int(math.ceil(math.sqrt(2.0) * N))
        pad = diagonal - N
        old_center = N // 2
        new_center = diagonal // 2
        pad_before = new_center - old_center
        # pad = (right_left_for_last_dim, ..., left_right_for_first_dim)
        return F.pad(sino, (0, 0, pad_before, pad - pad_before), mode="constant", value=0.0)

    @staticmethod
    def _get_fourier_filter_torch(size: int, filter_name: Optional[str], device, dtype) -> torch.Tensor:
        """
        Build Fourier-domain filters identical to scikit-image's `_get_fourier_filter`.

        Returns
        -------
        Tensor
            Column vector ``[size, 1]`` in **float32** for stable FFT math.
        """
        import math
        # Construct ramp filter core in float32 for numerical stability.
        n1 = torch.arange(1, size // 2 + 1, 2, device=device, dtype=torch.float32)
        n2 = torch.arange(size // 2 - 1, 0, -2, device=device, dtype=torch.float32)
        n = torch.cat([n1, n2])

        f = torch.zeros(size, device=device, dtype=torch.float32)
        f[0] = 0.25
        f[1::2] = -(1.0 / (math.pi * n) ** 2)

        fourier_filter = 2.0 * torch.real(torch.fft.fft(f))  # ramp

        if filter_name == "ramp":
            pass
        elif filter_name == "shepp-logan":
            omega = math.pi * torch.fft.fftfreq(size, d=1.0, device=device, dtype=torch.float32)[1:]
            ff = fourier_filter.clone()
            ff[1:] = ff[1:] * (torch.sin(omega) / omega)
            fourier_filter = ff
        elif filter_name == "cosine":
            freq = torch.linspace(0.0, math.pi, steps=size, device=device, dtype=torch.float32)
            fourier_filter = fourier_filter * torch.fft.fftshift(torch.sin(freq))
        elif filter_name == "hamming":
            w = torch.hamming_window(size, periodic=False, device=device, dtype=torch.float32)
            fourier_filter = fourier_filter * torch.fft.fftshift(w)
        elif filter_name == "hann":
            w = torch.hann_window(size, periodic=False, device=device, dtype=torch.float32)
            fourier_filter = fourier_filter * torch.fft.fftshift(w)
        elif filter_name is None:
            fourier_filter = torch.ones_like(fourier_filter)

        return fourier_filter.view(size, 1).to(dtype=torch.float32)

    @staticmethod
    def _interp1d_linear_torch(col: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """1D linear interpolation with zero outside, x = arange(N) - N//2."""
        N = col.shape[0]
        center = N // 2
        idx = t + center
        i0 = torch.floor(idx).to(torch.long).clamp(0, N - 1)
        i1 = (i0 + 1).clamp(0, N - 1)
        w = (idx - i0.to(idx.dtype)).clamp(0, 1)
        y0 = col[i0]; y1 = col[i1]
        out = (1.0 - w) * y0 + w * y1
        mask = (idx < 0) | (idx > (N - 1))
        return torch.where(mask, torch.zeros((), device=out.device, dtype=out.dtype), out)

    @staticmethod
    def _interp1d_nearest_torch(col: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """1D nearest interpolation with zero outside, x = arange(N) - N//2."""
        N = col.shape[0]
        center = N // 2
        idx = torch.round(t + center).to(torch.long)
        mask = (idx < 0) | (idx >= N)
        idx = idx.clamp(0, N - 1)
        out = col[idx]
        return torch.where(mask, torch.zeros((), device=out.device, dtype=out.dtype), out)

    @staticmethod
    def _iradon_torch_2d(
        radon_image: torch.Tensor,         # [N, A]
        theta_deg: torch.Tensor,           # [A]
        output_size: int,
        filter_name: Optional[str] = "ramp",
        interpolation: str = "linear",
        circle: bool = True,
    ) -> torch.Tensor:
        """
        2D FBP equivalent to scikit-image's `iradon`, implemented in torch.

        **Numerical policy**
        --------------------
        • FFT path is **forced to float32** for stability under AMP/mixed precision.
        • Spatial accumulation uses float32; the final image is cast to the
          caller's dtype.
        """
        if radon_image.ndim != 2:
            raise ValueError("radon_image must be [N, A].")
        if theta_deg.ndim != 1 or theta_deg.numel() != radon_image.shape[1]:
            raise ValueError("theta length must match number of projection columns.")

        device = radon_image.device
        in_dtype = radon_image.dtype
        N, A = radon_image.shape

        # Optional circle-to-square padding (skimage behavior)
        sino = radon_image
        if circle:
            sino = JosephProjector3D._sinogram_circle_to_square_torch(sino)
            N = sino.shape[0]

        # --- FFT filtering in float32 ---
        proj_size_padded = JosephProjector3D._next_pow2(N)
        sino_pad = F.pad(sino, (0, 0, 0, proj_size_padded - N), mode="constant", value=0.0)
        sino_pad_f32 = sino_pad.to(torch.float32)

        Ffilt = JosephProjector3D._get_fourier_filter_torch(
            proj_size_padded, filter_name, device=device, dtype=torch.float32
        )  # [P,1] float32

        proj_fft = torch.fft.fft(sino_pad_f32, dim=0) * Ffilt  # complex64
        radon_filtered = torch.real(torch.fft.ifft(proj_fft, dim=0))[:N, :]  # [N, A], float32

        # --- Spatial backprojection in float32 ---
        recon_acc = torch.zeros((output_size, output_size), device=device, dtype=torch.float32)
        radius = output_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(output_size, device=device, dtype=torch.float32),
            torch.arange(output_size, device=device, dtype=torch.float32),
            indexing="ij",
        )
        ypr = yy - radius
        xpr = xx - radius

        theta_rad = torch.deg2rad(theta_deg.to(torch.float32))
        cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)

        if interpolation not in ("linear", "nearest"):
            raise ValueError("interpolation must be 'linear' or 'nearest'.")

        for i in range(A):
            col = radon_filtered[:, i]  # [N], float32
            t = ypr * cos_t[i] - xpr * sin_t[i]  # float32
            if interpolation == "linear":
                recon_acc += JosephProjector3D._interp1d_linear_torch(col, t)
            else:
                recon_acc += JosephProjector3D._interp1d_nearest_torch(col, t)

        if circle:
            # Mask pixels outside the inscribed disk
            mask = (xpr**2 + ypr**2) > (radius**2)
            recon_acc = torch.where(mask, torch.zeros((), dtype=recon_acc.dtype, device=device), recon_acc)

        # Scale (skimage normalization): pi / (2A)
        import math
        recon_acc = recon_acc * (math.pi / (2.0 * A))

        # Cast back to input dtype for the caller
        return recon_acc.to(in_dtype)

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Differentiable slice-wise filtered backprojection (FBP) in PyTorch.

        Notes
        -----
        • No @torch.no_grad(): autograd must see the whole graph for training.
        • Avoids in-place indexing into a preallocated output tensor. Instead,
        constructs each (X,Y) slice and stacks along Z/C/B to preserve gradients.

        Parameters
        ----------
        sino : Tensor
            Model-facing sinogram [B, C, X, A, Z] in (x, a, z) layout.

        Returns
        -------
        Tensor
            Reconstructed volume [B, C, X, Y, Z] in (x, y, z) layout.
        """
        if sino.ndim != 5:
            raise ValueError(f"Expected [B,C,X,A,Z], got {tuple(sino.shape)}")
        B, C, X, A, Z = sino.shape
        Y = self.geom.H
        device = sino.device
        dtype = sino.dtype

        import math
        theta_deg = (self.geom.angles * (180.0 / math.pi)).to(device, torch.float32)

        outs_B = []
        for b in range(B):
            outs_C = []
            for c in range(C):
                s_bc = sino[b, c]  # [X, A, Z]
                rec_slices = []
                for z in range(Z):
                    s_2d = s_bc[:, :, z].to(dtype)  # [X, A]
                    rec_yy = JosephProjector3D._iradon_torch_2d(
                        s_2d,
                        theta_deg,
                        output_size=Y,
                        filter_name=self.ir_filter_name,
                        interpolation=self.ir_interpolation,
                        circle=self.ir_circle,
                    )  # [Y, Y]
                    rec_xy = rec_yy.t()              # [X, Y]
                    rec_slices.append(rec_xy.unsqueeze(-1))  # [X, Y, 1]
                out_c = torch.cat(rec_slices, dim=-1)       # [X, Y, Z]
                outs_C.append(out_c.unsqueeze(0))           # [1, X, Y, Z]
            out_b = torch.cat(outs_C, dim=0)                # [C, X, Y, Z]
            outs_B.append(out_b.unsqueeze(0))               # [1, C, X, Y, Z]

        out = torch.cat(outs_B, dim=0)                      # [B, C, X, Y, Z]

        # Optional circular support mask in XY (broadcast over B,C,Z).
        if self.ir_circle:
            xx = torch.arange(X, device=device, dtype=torch.float32) - (X - 1) / 2.0
            yy = torch.arange(Y, device=device, dtype=torch.float32) - (Y - 1) / 2.0
            Xg, Yg = torch.meshgrid(xx, yy, indexing="ij")  # [X, Y]
            r = min((X - 1) / 2.0, (Y - 1) / 2.0)
            circ = ((Xg**2 + Yg**2) <= (r**2)).to(dtype)    # [X, Y]
            out = out * circ.view(1, 1, X, Y, 1)

        return out


class SiddonProjector3D(BaseProjector3D):
    """
    Siddon 3D projector (ray-driven, voxel intersection lengths).

    Method
    ------
    • For each ray, computes visited voxels and intersection lengths (XY traversal per Z).
    • Forward: sum voxel values × intersection lengths along the ray.
    • Backward: scatter sinogram ray value × intersection lengths back to voxels.

    Notes
    -----
    • Slower but useful as a reference/validation against the Joseph projector.
    """

    def __init__(self, geom: Parallel3DGeometry):
        super().__init__(geom)
        self.register_buffer("angles", geom.angles.clone().detach())

        # Detector coordinates centered at 0
        V, U = geom.V, geom.U
        sv, su = geom.det_spacing
        v = (torch.arange(V, dtype=torch.float32) - (V - 1) / 2.0) * sv
        u = (torch.arange(U, dtype=torch.float32) - (U - 1) / 2.0) * su
        self.register_buffer("u_phys", u)
        self.register_buffer("v_phys", v)

        # Voxel spacing
        sd, sy, sx = geom.voxel_size
        self.sx = float(sx); self.sy = float(sy); self.sd = float(sd)

    @torch.no_grad()
    def reset_geometry(self, geom: Parallel3DGeometry):
        """
        Update cached geometry-dependent buffers (angles, detector coordinates, spacings).
        """
        self.geom = geom
        new_angles = geom.angles.detach().to(self.angles.device, dtype=self.angles.dtype)
        self.angles.resize_(new_angles.shape).copy_(new_angles)

        V, U = geom.V, geom.U
        sv, su = geom.det_spacing
        v = (torch.arange(V, dtype=torch.float32, device=self.angles.device) - (V - 1) / 2.0) * sv
        u = (torch.arange(U, dtype=torch.float32, device=self.angles.device) - (U - 1) / 2.0) * su
        if hasattr(self, "u_phys"): self.u_phys.resize_(u.shape).copy_(u)
        else: self.register_buffer("u_phys", u)
        if hasattr(self, "v_phys"): self.v_phys.resize_(v.shape).copy_(v)
        else: self.register_buffer("v_phys", v)

        sd, sy, sx = geom.voxel_size
        self.sx = float(sx); self.sy = float(sy); self.sd = float(sd)

    def _trace_ray(self, D, H, W, x0, y0, z0, cos_t, sin_t, device):
        """
        Trace a single XY ray at fixed z0 using Siddon's algorithm.

        Returns
        -------
        idx : LongTensor [N, 3]
            Voxel indices (k, j, i) in internal [D,H,W] indexing.
        wts : FloatTensor [N]
            Segment lengths inside each voxel (same order as idx).
        """
        # Bounds in physical space (centered at 0)
        x_min = -((W - 1) / 2.0) * self.sx; x_max = ((W - 1) / 2.0) * self.sx
        y_min = -((H - 1) / 2.0) * self.sy; y_max = ((H - 1) / 2.0) * self.sy
        dx = cos_t; dy = sin_t

        def _t_for(p0, dp, pmin, pmax):
            if abs(dp) < 1e-12:
                if p0 < pmin or p0 > pmax:
                    return float("inf"), float("-inf")  # never enters
                return float("-inf"), float("inf")     # parallel inside
            t0 = (pmin - p0) / dp
            t1 = (pmax - p0) / dp
            return (min(t0, t1), max(t0, t1))

        tx0, tx1 = _t_for(x0, dx, x_min, x_max)
        ty0, ty1 = _t_for(y0, dy, y_min, y_max)
        t_entry = max(tx0, ty0); t_exit = min(tx1, ty1)
        if (t_entry >= t_exit):
            return (torch.empty(0, 3, device=device, dtype=torch.long),
                    torch.empty(0, device=device, dtype=torch.float32))

        # Entry voxel indices (nearest centers)
        xe = x0 + t_entry * dx; ye = y0 + t_entry * dy; ze = z0
        i = int(round(xe / self.sx + (W - 1) / 2.0))
        j = int(round(ye / self.sy + (H - 1) / 2.0))
        k = int(round(ze / self.sd + (D - 1) / 2.0))
        i = max(0, min(W - 1, i)); j = max(0, min(H - 1, j)); k = max(0, min(D - 1, k))

        inds = []; lens = []; x = xe; y = ye; t = t_entry
        while t < t_exit - 1e-12:
            # Next faces in x and y
            if dx > 0: x_next = ((i + 0.5) * self.sx - (W - 1) / 2.0 * self.sx)
            elif dx < 0: x_next = ((i - 0.5) * self.sx - (W - 1) / 2.0 * self.sx)
            else: x_next = float("inf")
            if dy > 0: y_next = ((j + 0.5) * self.sy - (H - 1) / 2.0 * self.sy)
            elif dy < 0: y_next = ((j - 0.5) * self.sy - (H - 1) / 2.0 * self.sy)
            else: y_next = float("inf")

            tx = (x_next - x) / dx if abs(dx) > 1e-12 else float("inf")
            ty = (y_next - y) / dy if abs(dy) > 1e-12 else float("inf")
            dt = min(tx, ty, t_exit - t)

            seg = max(0.0, float(dt) * (self.sx**2 + self.sy**2)**0.5 / max(abs(dx) + abs(dy), 1e-12))
            inds.append((k, j, i)); lens.append(seg)

            t += dt; x += dx * dt; y += dy * dt
            if tx < ty:
                i += 1 if dx > 0 else -1
                if i < 0 or i >= W: break
            elif ty < tx:
                j += 1 if dy > 0 else -1
                if j < 0 or j >= H: break
            else:
                i += 1 if dx > 0 else -1
                j += 1 if dy > 0 else -1
                if i < 0 or i >= W or j < 0 or j >= H: break

        if len(inds) == 0:
            return (torch.empty(0, 3, device=device, dtype=torch.long),
                    torch.empty(0, device=device, dtype=torch.float32))

        idx = torch.tensor(inds, dtype=torch.long, device=device)
        wts = torch.tensor(lens, dtype=torch.float32, device=device)
        return idx, wts

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Volume → Sinogram (Siddon forward), **model-facing I/O**.

        Parameters
        ----------
        vol : Tensor
            **[B, C, X, Y, Z]** volume (x,y,z).

        Returns
        -------
        Tensor
            **[B, C, X, A, Z]** sinogram (x,a,z).
        """
        # Model [B,C,X,Y,Z] → internal [B,C,D,H,W]
        vol_int = vol.permute(0, 1, 4, 3, 2).contiguous()
        B, C, D, H, W = vol_int.shape
        A = int(self.angles.numel()); V, U = self.geom.V, self.geom.U
        device = vol_int.device

        out = torch.zeros((B, C, A, V, U), device=device, dtype=vol_int.dtype)

        # Angle chunking
        for a0 in range(0, A, self.geom.angle_chunk):
            a1 = min(a0 + self.geom.angle_chunk, A)
            th = self.angles[a0:a1]
            cos_t = torch.cos(th); sin_t = torch.sin(th)

            for ai in range(th.numel()):
                c = cos_t[ai].item(); s = sin_t[ai].item()
                for vv in range(V):
                    z0 = self.v_phys[vv].item()
                    for uu in range(U):
                        u = self.u_phys[uu].item()
                        x0 = -u * s; y0 = u * c
                        idx, wts = self._trace_ray(D, H, W, x0, y0, z0, c, s, device)
                        if idx.numel() == 0:
                            continue
                        if wts.dtype != vol_int.dtype:
                            wts = wts.to(vol_int.dtype)

                        kji = idx.t()
                        vvals = vol_int[:, :, kji[0], kji[1], kji[2]]
                        contrib = (vvals * wts.view(1, 1, -1)).sum(dim=-1)
                        out[:, :, a0 + ai, vv, uu] = contrib

        # Internal [B,C,A,V,U] → model [B,C,X,A,Z] (U→X, V→Z)
        return out.permute(0, 1, 4, 2, 3).contiguous()

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Sinogram → Volume (Siddon backprojection), **model-facing I/O**.

        Parameters
        ----------
        sino : Tensor
            **[B, C, X, A, Z]** sinogram (x,a,z).

        Returns
        -------
        Tensor
            **[B, C, X, Y, Z]** volume (x,y,z).
        """
        # Model [B,C,X,A,Z] → internal [B,C,A,V,U]
        sino_int = sino.permute(0, 1, 3, 4, 2).contiguous()
        B, C, A, V, U = sino_int.shape
        D, H, W = self.geom.D, self.geom.H, self.geom.W
        device = sino_int.device

        vol = torch.zeros((B, C, D, H, W), device=device, dtype=sino_int.dtype)

        for a0 in range(0, A, self.geom.angle_chunk):
            a1 = min(a0 + self.geom.angle_chunk, A)
            th = self.angles[a0:a1]
            cos_t = torch.cos(th); sin_t = torch.sin(th)

            for ai in range(th.numel()):
                c = cos_t[ai].item(); s = sin_t[ai].item()
                for vv in range(V):
                    z0 = self.v_phys[vv].item()
                    for uu in range(U):
                        u = self.u_phys[uu].item()
                        x0 = -u * s; y0 = u * c
                        idx, wts = self._trace_ray(D, H, W, x0, y0, z0, c, s, device)
                        if idx.numel() == 0:
                            continue

                        ray = sino_int[:, :, a0 + ai, vv, uu]
                        if wts.dtype != ray.dtype:
                            wts = wts.to(ray.dtype)

                        kji = idx.t()
                        scatter = (ray.view(B, C, 1) * wts.view(1, 1, -1))
                        flat_idx = (kji[0] * H * W + kji[1] * W + kji[2]).view(-1)
                        vol_flat = vol.view(B * C, D * H * W)
                        vol_flat.index_add_(1, flat_idx, scatter.view(B * C, -1))

        # Internal [B,C,D,H,W] → model [B,C,X,Y,Z] (W→X, H→Y, D→Z)
        return vol.permute(0, 1, 4, 3, 2).contiguous()


def make_projector(method: Literal["joseph3d", "siddon3d"], geom: Parallel3DGeometry) -> BaseProjector3D:
    """
    Factory for 3D projectors with **model-facing (x,y,z)/(x,a,z)** I/O.

    Parameters
    ----------
    method : {"joseph3d", "siddon3d"}
        Choice of projector.
    geom : Parallel3DGeometry
        Geometry configuration.

    Returns
    -------
    BaseProjector3D
        Projector instance.

    Raises
    ------
    ValueError
        If the method is unknown.
    """
    if method == "joseph3d":
        return JosephProjector3D(geom, n_steps=geom.n_steps_cap)
    elif method == "siddon3d":
        return SiddonProjector3D(geom)
    else:
        raise ValueError(f"Unknown projector method: {method}")
