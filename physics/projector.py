from typing import Optional, Literal
import math
import torch
import torch.nn.functional as F
from .geometry import Parallel3DGeometry


# ==========================================================================================
# Base
# ==========================================================================================
class BaseProjector3D(torch.nn.Module):
    """
    Abstract base for 3D projectors.

    Contract
    --------
    • forward(vol):    [B,C,X,Y,Z] → [B,C,X,A,Z]   (sinogram S[x,a,z])
    • backproject(s):  [B,C,X,A,Z] → [B,C,X,Y,Z]   (volume V[x,y,z])
    """
    def __init__(self, geom: Parallel3DGeometry):
        super().__init__()
        self.geom = geom

    def reset_geometry(self, geom: Parallel3DGeometry):
        raise NotImplementedError

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ==========================================================================================
# Joseph (voxel-driven) + BP (unfiltered / filtered by ramp & windows)
# ==========================================================================================
class JosephProjector3D(BaseProjector3D):
    """
    Joseph-style voxel-driven forward projector with **backprojection** that can be:
    • Unfiltered (legacy, scikit-image `iradon(filter=None)`
    • Filtered (ramp/Shepp-Logan/cosine/Hann/Hamming) along detector axis, then BP.

    Switches
    --------
    ir_impl : {"interp","grid"}
    bp_span : {"half","full","auto"}
    dc_mode : {"none","detector","angle","both"}
    ir_interpolation : {"linear","nearest"}
    ir_circle : bool

    FBP
    ---------------
    fbp_filter : {"none","ramp","shepp-logan","cosine","hann","hamming"}
    fbp_cutoff : float in (0,1]
    fbp_pad_mode : {"next_pow2","none"}
    """

    def __init__(
        self,
        geom: Parallel3DGeometry,
        n_steps: Optional[int] = None,
        *,
        ir_impl: Literal["interp", "grid"] = "grid",
        bp_span: Literal["half", "full", "auto"] = "auto",
        dc_mode: Literal["none", "detector", "angle", "both"] = "detector",
        ir_interpolation: Literal["linear", "nearest"] = "linear",
        ir_circle: bool = True,
        # FBP switches
        fbp_filter: Literal["none","ramp","shepp-logan","cosine","hann","hamming"] = "none",
        fbp_cutoff: float = 1.0,
        fbp_pad_mode: Literal["next_pow2","none"] = "next_pow2",
    ):
        super().__init__(geom)

        # Cached angles (radians) and trig for forward projector
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

        # Ray integration extent and step (diagonal of XY in physical units)
        T = 0.5 * float(((W - 1) * sx) ** 2 + ((H - 1) * sy) ** 2) ** 0.5
        self.T = float(T)
        self.delta_t = (2.0 * self.T) / float(self.n_steps)
        self.step_chunk = max(1, min(8, self.n_steps))
        self.c_chunk = 4

        # BP options
        self.ir_impl: str = str(ir_impl)
        self.ir_interpolation: str = str(ir_interpolation)  # for "interp" path
        self.ir_circle: bool = bool(ir_circle)

        # Span/DC switches
        self.bp_span: str = str(bp_span)
        self.dc_mode: str = str(dc_mode)

        # FBP switches
        self.fbp_filter: str = str(fbp_filter).lower()
        self.fbp_cutoff: float = float(fbp_cutoff)
        self.fbp_pad_mode: str = str(fbp_pad_mode).lower()

    @torch.no_grad()
    def reset_geometry(self, geom: Parallel3DGeometry):
        """Update geometry-dependent buffers after shape/angle changes (no autograd)."""
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

    # --------------------------------------------------------------------------------------
    # Forward (voxel-driven Joseph via grid_sample)
    # --------------------------------------------------------------------------------------
    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Compute sinogram via voxel-driven Joseph integration.

        Args
        ----
        vol : Tensor [B,C,X,Y,Z]
            Volume in physics axes (X,Y,Z) = (det-u, img-y, depth-z).

        Returns
        -------
        Tensor [B,C,X,A,Z]
        """
        # [B,C,X,Y,Z] → internal [B,C,D,H,W] = [B,C,Z,Y,X]
        vol = vol.permute(0, 1, 4, 3, 2).contiguous()
        B, C, D, H, W = vol.shape
        A = int(self.angles.numel())
        V, U = self.geom.V, self.geom.U
        device = vol.device

        def _safe_norm(idx: torch.Tensor, size: int) -> torch.Tensor:
            # map pixel index [0..size-1] → normalized [-1,1] for grid_sample
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

            v_base = vol.expand(cos_t.shape[0], C, D, H, W) if (B == 1) else vol.repeat_interleave(cos_t.shape[0], dim=0)

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

                    samples = F.grid_sample(v_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                    # Sum along S (integration over t)
                    part = samples.sum(dim=2).view(B, cos_t.shape[0], c1 - c0, V, U).permute(0, 2, 1, 3, 4)
                    accum_c.add_(part)

                    del i, t, x, y, z, x_idx, y_idx, z_idx, x_n, y_n, z_n, grid, samples, part

                sino_accum[:, c0:c1].add_(accum_c)
                del v_in, accum_c

            sino = sino_accum.mul(self.delta_t).contiguous()
            sino_full[:, :, a0:a1, :, :] = sino
            del sino_accum, sino, v_base

        return sino_full.permute(0, 1, 4, 2, 3).contiguous()  # [B,C,X,A,Z]

    # --------------------------------------------------------------------------------------
    # Helpers (masking, span, filtering)
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _sinogram_circle_to_square_torch(sino: torch.Tensor) -> torch.Tensor:
        """
        Pad detector (height=N) to circumscribed square (enables XY circular support).
        Input [N, A] → Output [N_pad, A]
        """
        N, A = sino.shape
        diagonal = int(math.ceil(math.sqrt(2.0) * N))
        pad = diagonal - N
        old_center = N // 2
        new_center = diagonal // 2
        pad_before = new_center - old_center
        return F.pad(sino, (0, 0, pad_before, pad - pad_before), mode="constant", value=0.0)

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 << (max(0, int(n - 1)).bit_length())

    @staticmethod
    def _fourier_filter_response(P: int, spacing: float, name: str, cutoff: float,
                                 device, dtype) -> torch.Tensor:
        """
        Return 1D |H(f)| for rFFT bins (length P) in cycles/length using spacing.
        """
        # freq: [0 .. 1/(2*spacing)] (rfft bins)
        freq = torch.fft.rfftfreq(P, d=float(spacing)).to(device=device, dtype=dtype).abs()
        omega = 2.0 * math.pi * freq
        # Ramp
        H = 2.0 * freq  # = |f|
        name = (name or "none").lower()
        if name in ("shepp-logan", "shepp"):
            # sinc window on ramp (avoid divide by zero)
            H[1:] = H[1:] * (torch.sin(omega[1:] / 2.0) / (omega[1:] / 2.0))
            H[0] = 0.0
        elif name == "cosine":
            H = H * torch.cos(omega / 2.0)
        elif name == "hann":
            H = H * (0.5 * (1.0 + torch.cos(omega)))
        elif name == "hamming":
            H = H * (0.54 + 0.46 * torch.cos(omega))
        # Hard cutoff (normalized to Nyquist)
        cutoff = float(max(0.0, min(1.0, cutoff)))
        if cutoff < 1.0:
            # Nyquist = 0.5/spacing
            k = freq / (0.5 / float(spacing))
            H = H * (k <= cutoff).to(dtype)
        return H.view(-1, 1)  # [P_rfft, 1] for broadcast over angles

    @staticmethod
    def _apply_filter_1d(sino_NA: torch.Tensor, spacing: float,
                         name: str, cutoff: float, pad_mode: str) -> torch.Tensor:
        """
        Apply 1D frequency filter along detector axis for each angle column.
        sino_NA: [N, A]
        """
        name = (name or "none").lower()
        if name in ("none", ""):
            return sino_NA

        device = sino_NA.device
        dtype_in = sino_NA.dtype
        dtype_acc = torch.float32

        N, A = sino_NA.shape
        if pad_mode == "next_pow2":
            P = max(64, JosephProjector3D._next_pow2(2 * N))
        else:
            P = N

        x = sino_NA.to(dtype_acc)
        if P > N:
            # pad (last two dims order: [N(=H), A(=W)])
            x = F.pad(x, (0, 0, 0, P - N))
        # rFFT over detector axis (dim=0)
        Xf = torch.fft.rfft(x, dim=0)  # [P_rfft, A]

        H = JosephProjector3D._fourier_filter_response(
            P=P, spacing=spacing, name=name, cutoff=cutoff, device=device, dtype=dtype_acc
        )  # [P_rfft,1]

        Yf = Xf * H
        y = torch.fft.irfft(Yf, n=P, dim=0)  # [P, A]
        y = y[:N, :]  # crop back
        return y.to(dtype_in)

    @staticmethod
    def _infer_span(A: int) -> str:
        """Heuristic: typical 360° sets have A≈360 (or ≥300)."""
        return "full" if A >= 300 else "half"

    @staticmethod
    def _dedup_360_to_180(s_2d: torch.Tensor) -> torch.Tensor:
        """
        Reduce 360° redundancy using R(u, θ+π) ≈ R(-u, θ).
        Assumes A is even. Returns [X, A//2].
        """
        X, A = s_2d.shape
        A2 = A // 2
        s1 = s_2d[:, :A2]
        s2 = torch.flip(s_2d[:, A2:], dims=[0])  # detector flip (u→-u)
        return 0.5 * (s1 + s2)

    @staticmethod
    def _theta_for_span(A_eff: int, span: str, device, dtype) -> torch.Tensor:
        """
        Return equally spaced angles [A_eff] in degrees over [0,180) or [0,360).
        Uses endpoint=False to avoid duplicate 0/π or 0/2π.
        """
        total = 180.0 if span == "half" else 360.0
        return torch.linspace(0.0, total, steps=A_eff + 1, device=device, dtype=dtype)[:-1]

    # --------------------------------------------------------------------------------------
    # Unfiltered 2D BP (interp / grid)
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _iradon_unfiltered_2d(
        radon_image: torch.Tensor,   # [N, A_eff]
        theta_deg: torch.Tensor,     # [A_eff]
        output_size: int,
        interpolation: str = "linear",
        circle: bool = True,
    ) -> torch.Tensor:
        if radon_image.ndim != 2:
            raise ValueError("radon_image must be [N, A_eff].")
        if theta_deg.ndim != 1 or theta_deg.numel() != radon_image.shape[1]:
            raise ValueError("theta length must match number of projection columns.")

        device = radon_image.device
        dtype_in = radon_image.dtype
        dtype_acc = torch.float32

        N, A_eff = radon_image.shape
        sino = radon_image.to(dtype_acc)
        if circle:
            sino = JosephProjector3D._sinogram_circle_to_square_torch(sino)
            N = sino.shape[0]

        # Output grid (centered pixel coords)
        recon = torch.zeros((output_size, output_size), device=device, dtype=dtype_acc)
        radius = output_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(output_size, device=device, dtype=dtype_acc),
            torch.arange(output_size, device=device, dtype=dtype_acc),
            indexing="ij",
        )
        ypr = yy - radius
        xpr = xx - radius

        # Angles & trig
        theta_rad = torch.deg2rad(theta_deg.to(device=device, dtype=dtype_acc))
        cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)

        # Accumulate
        if interpolation not in ("linear", "nearest"):
            raise ValueError("interpolation must be 'linear' or 'nearest'.")
        for i in range(A_eff):
            col = sino[:, i]
            t = ypr * cos_t[i] - xpr * sin_t[i]
            if interpolation == "linear":
                recon += JosephProjector3D._interp1d_linear_torch(col, t)
            else:
                recon += JosephProjector3D._interp1d_nearest_torch(col, t)

        # Circle mask
        if circle:
            mask = (xpr**2 + ypr**2) > (radius**2)
            recon = torch.where(mask, torch.zeros((), dtype=dtype_acc, device=device), recon)

        # Scale generalized to arbitrary span/spacing (θ_span/(2·A_eff))
        dtheta = (theta_rad[1] - theta_rad[0]) if A_eff > 1 else torch.tensor(math.pi, dtype=dtype_acc, device=device)
        theta_span = dtheta * A_eff
        scale = float(theta_span) / (2.0 * A_eff)
        return (recon * scale).to(dtype_in)

    @staticmethod
    def _interp1d_linear_torch(col: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        N = col.shape[0]
        center = N // 2
        idx = torch.round(t + center).to(torch.long)
        mask = (idx < 0) | (idx >= N)
        idx = idx.clamp(0, N - 1)
        out = col[idx]
        return torch.where(mask, torch.zeros((), device=out.device, dtype=out.dtype), out)

    @staticmethod
    def _iradon_unfiltered_2d_grid(
        radon_image: torch.Tensor,   # [N, A_eff]
        theta_deg: torch.Tensor,     # [A_eff]
        output_size: int,
        circle: bool = True,
    ) -> torch.Tensor:
        device = radon_image.device
        dtype_in = radon_image.dtype
        dtype_acc = torch.float32

        # Cast + optional circle→square padding
        sino = radon_image.to(dtype_acc)
        N, A_eff = int(sino.shape[0]), int(sino.shape[1])
        if circle:
            sino = JosephProjector3D._sinogram_circle_to_square_torch(sino)
            N = int(sino.shape[0])

        # Angles (deg → rad)
        theta_rad = torch.deg2rad(theta_deg.to(device=device, dtype=dtype_acc))
        cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)

        # Output grid in centered pixel coords
        Y = int(output_size)
        radius = Y // 2
        yy, xx = torch.meshgrid(
            torch.arange(Y, device=device, dtype=dtype_acc),
            torch.arange(Y, device=device, dtype=dtype_acc),
            indexing="ij",
        )
        ypr = yy - radius
        xpr = xx - radius

        # Per-angle sampling coordinates (A_eff, Y, Y)
        t = ypr.unsqueeze(0) * cos_t.view(-1, 1, 1) - xpr.unsqueeze(0) * sin_t.view(-1, 1, 1)
        # Normalize to [-1,1] over detector axis (H-in of grid_sample)
        t_norm = (2.0 * (t + (N - 1) / 2.0) / max(1, (N - 1))) - 1.0
        # Constant angle index per plane, normalized to [-1,1] over angle axis (W-in)
        a_idx = torch.arange(A_eff, device=device, dtype=dtype_acc).view(-1, 1, 1)
        a_norm = (2.0 * a_idx / max(1, (A_eff - 1))) - 1.0 if A_eff > 1 else torch.zeros_like(a_idx)

        # grid: [A_eff, Y, Y, 2] with (x=W=angle, y=H=detector)
        grid = torch.stack([a_norm.expand(-1, Y, Y), t_norm], dim=-1).contiguous()

        # Input replicated across A_eff: [A_eff, 1, H=N, W=A_eff]
        inp = sino.unsqueeze(0).unsqueeze(0).expand(A_eff, 1, N, A_eff).contiguous()

        # Sample and accumulate across angles
        recon_per_angle = torch.nn.functional.grid_sample(
            inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [A_eff,1,Y,Y]
        recon = recon_per_angle.sum(dim=0)[0]  # [Y, Y]

        # XY circular support
        if circle:
            mask = (xpr**2 + ypr**2) <= (radius**2)
            recon = torch.where(mask, recon, torch.zeros((), device=device, dtype=dtype_acc))

        # Scale generalized to arbitrary span/spacing (θ_span/(2·A_eff))
        dtheta = (theta_rad[1] - theta_rad[0]) if A_eff > 1 else torch.tensor(math.pi, dtype=dtype_acc, device=device)
        theta_span = dtheta * A_eff
        scale = float(theta_span) / (2.0 * A_eff)
        return (recon * scale).to(dtype_in)

    # --------------------------------------------------------------------------------------
    # Backprojection (with optional ramp filter)
    # --------------------------------------------------------------------------------------
    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        sino : Tensor [B,C,X,A,Z] (detector=X, angle=A, depth=Z)

        Returns
        -------
        Tensor [B,C,X,Y,Z]
        """
        if sino.ndim != 5:
            raise ValueError(f"Expected [B,C,X,A,Z], got {tuple(sino.shape)}")
        B, C, X, A, Z = sino.shape
        Y = self.geom.H
        device = sino.device
        dtype = sino.dtype

        # detector pixel spacing (t-axis) in physical units
        su, _ = self.geom.det_spacing_xz  # (su, sv)
        su = float(su if su is not None else 1.0)

        outs_B = []
        for b in range(B):
            outs_C = []
            for c in range(C):
                s_bc = sino[b, c]  # [X, A, Z]
                rec_slices = []
                for z in range(Z):
                    s_2d = s_bc[:, :, z].to(dtype)  # [X, A]

                    # --- DC correction (optional) ---
                    if self.dc_mode in ("detector", "both"):
                        s_2d = s_2d - s_2d.mean(dim=0, keepdim=True)  # per-angle detector mean
                    if self.dc_mode in ("angle", "both"):
                        s_2d = s_2d - s_2d.mean(dim=1, keepdim=True)  # per-detector angle mean

                    # --- span selection & angle vector ---
                    span = (self.bp_span if self.bp_span in ("half", "full") else self._infer_span(A))
                    if span == "half":
                        A_eff = (A // 2) if (A >= 2) else A
                        s_use = self._dedup_360_to_180(s_2d) if (A_eff * 2 == A) else s_2d[:, :A_eff]
                    else:
                        A_eff = A
                        s_use = s_2d
                    theta_deg = self._theta_for_span(A_eff, span, device, torch.float32)

                    # --- (optional) FBP filtering along detector axis ---
                    s_pre = self._apply_filter_1d(
                        s_use, spacing=su, name=self.fbp_filter, cutoff=self.fbp_cutoff, pad_mode=self.fbp_pad_mode
                    )

                    # --- BP core (grid or interp) ---
                    if self.ir_impl == "grid":
                        rec_y_y = self._iradon_unfiltered_2d_grid(
                            s_pre, theta_deg, output_size=Y, circle=self.ir_circle
                        )  # [Y, Y]
                    else:
                        rec_y_y = self._iradon_unfiltered_2d(
                            s_pre, theta_deg, output_size=Y,
                            interpolation=self.ir_interpolation, circle=self.ir_circle
                        )  # [Y, Y]

                    # [Y,Y] → [X,Y] (transpose keeps X horizontal)
                    if X != Y:
                        rec_yx = F.interpolate(
                            rec_y_y.unsqueeze(0).unsqueeze(0), size=(Y, X),
                            mode="bilinear", align_corners=False
                        )[0, 0]
                        rec_xy = rec_yx.t()
                    else:
                        rec_xy = rec_y_y.t()
                    rec_slices.append(rec_xy.unsqueeze(-1))  # [X, Y, 1]

                out_c = torch.cat(rec_slices, dim=-1)   # [X, Y, Z]
                outs_C.append(out_c.unsqueeze(0))       # [1, X, Y, Z]
            out_b = torch.cat(outs_C, dim=0)            # [C, X, Y, Z]
            outs_B.append(out_b.unsqueeze(0))           # [1, C, X, Y, Z]
        vol_xyz = torch.cat(outs_B, dim=0)              # [B, C, X, Y, Z]

        if self.ir_circle:
            xx = torch.arange(X, device=device, dtype=torch.float32) - (X - 1) / 2.0
            yy = torch.arange(Y, device=device, dtype=torch.float32) - (Y - 1) / 2.0
            Xg, Yg = torch.meshgrid(xx, yy, indexing="ij")
            r = min((X - 1) / 2.0, (Y - 1) / 2.0)
            circ = ((Xg**2 + Yg**2) <= (r**2)).to(dtype)
            vol_xyz = vol_xyz * circ.view(1, 1, X, Y, 1)
        return vol_xyz


# ==========================================================================================
# Factory
# ==========================================================================================
def make_projector(method: Literal["joseph3d"], geom: Parallel3DGeometry) -> BaseProjector3D:
    if method == "joseph3d":
        # defaults: grid BP, auto span, detector-mean DC removal; FBP off
        return JosephProjector3D(
            geom,
            n_steps=geom.n_steps_cap,
            ir_impl="grid",
            bp_span="auto",
            dc_mode="detector",
            ir_interpolation="linear",
            ir_circle=True,
            fbp_filter="none",
            fbp_cutoff=1.0,
            fbp_pad_mode="next_pow2",
        )
    else:
        raise ValueError(f"Unknown projector method: {method}")
