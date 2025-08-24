from typing import Optional, Literal
import torch
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

    Method
    ------
    • Integrates along rays parameterized by detector coordinates (u, v) and angle θ.
    • Samples the volume on regular parametric steps with trilinear interpolation
      via ``torch.nn.functional.grid_sample``.
    • Accumulates along the ray and scales by the step length ``delta_t``.

    Notes
    -----
    • Angle/channel/step chunking controls memory.
    • Internally uses ``align_corners=True`` because index normalization maps
      voxel centers to ``[-1, 1]``.
    """

    def __init__(self, geom: Parallel3DGeometry, n_steps: Optional[int] = None):
        super().__init__(geom)
        # Cache angles and trig
        self.register_buffer("angles", geom.angles.clone().detach())
        self.register_buffer("cos_angles", torch.cos(self.angles))
        self.register_buffer("sin_angles", torch.sin(self.angles))

        # # of integration steps per ray
        self.n_steps = int(n_steps if n_steps is not None else geom.n_steps_cap)

        # Detector physical coordinates centered at 0 (v: vertical=z, u: horizontal=x)
        V, U = geom.V, geom.U
        sv, su = geom.det_spacing
        v = (torch.arange(V, dtype=torch.float32) - (V - 1) / 2.0) * sv
        u = (torch.arange(U, dtype=torch.float32) - (U - 1) / 2.0) * su
        self.register_buffer("u_phys", u)
        self.register_buffer("v_phys", v)

        # Voxel spacing & integration step
        sd, sy, sx = geom.voxel_size
        D, H, W = geom.D, geom.H, geom.W
        self.sx = float(sx); self.sy = float(sy); self.sd = float(sd)

        # Half diagonal radius in XY plane
        T = 0.5 * float(((W - 1) * sx) ** 2 + ((H - 1) * sy) ** 2) ** 0.5
        self.T = float(T)
        self.delta_t = (2.0 * self.T) / float(self.n_steps)

        # Runtime chunking knobs
        self.step_chunk = max(1, min(8, self.n_steps))
        self.c_chunk = 4

    @torch.no_grad()
    def reset_geometry(self, geom: Parallel3DGeometry):
        """
        Update cached geometry-dependent buffers (angles, trig caches, detector coords).
        """
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

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Volume → Sinogram (Joseph forward), **model-facing I/O**.

        Parameters
        ----------
        vol : Tensor
            **[B, C, X, Y, Z]** volume (x,y,z).

        Returns
        -------
        Tensor
            **[B, C, X, A, Z]** sinogram (x,a,z).
        """
        # Convert model layout [B,C,X,Y,Z] → internal [B,C,D,H,W]
        vol = vol.permute(0, 1, 4, 3, 2).contiguous()
        B, C, D, H, W = vol.shape
        A = int(self.angles.numel())
        V, U = self.geom.V, self.geom.U
        device = vol.device

        def _safe_norm(idx: torch.Tensor, size: int) -> torch.Tensor:
            """Map voxel indices (0..size-1) to grid_sample range [-1, 1]."""
            if size <= 1:
                return torch.zeros_like(idx)
            return (2.0 * idx) / (size - 1) - 1.0

        # Output buffer in internal layout [B,C,A,V,U]
        sino_full = torch.empty((B, C, A, V, U), device=device, dtype=vol.dtype)

        # Angle chunking
        for a0 in range(0, A, self.geom.angle_chunk):
            a1 = min(a0 + self.geom.angle_chunk, A)

            cos_t = self.cos_angles[a0:a1].view(-1, 1, 1, 1)  # [Aa,1,1,1]
            sin_t = self.sin_angles[a0:a1].view(-1, 1, 1, 1)  # [Aa,1,1,1]

            u_phys = self.u_phys.view(1, 1, 1, U)             # [1,1,1,U]
            v_phys = self.v_phys.view(1, 1, V, 1)             # [1,1,V,1]

            # Ray bases at t = 0
            x0 = -u_phys * sin_t                               # [Aa,1,1,U]
            y0 =  u_phys * cos_t                               # [Aa,1,1,U]
            z0 =  v_phys.expand(cos_t.shape[0], 1, V, 1)       # [Aa,1,V,1]

            # Accumulator for this chunk
            sino_accum = torch.zeros(B, C, cos_t.shape[0], V, U, device=device, dtype=vol.dtype)

            # Tile volume across angles into batch dimension: [B*Aa,C,D,H,W]
            if B == 1:
                v_base = vol.expand(cos_t.shape[0], C, D, H, W)
            else:
                v_base = vol.repeat_interleave(cos_t.shape[0], dim=0)

            # Channel chunking
            for c0 in range(0, C, max(1, self.c_chunk)):
                c1 = min(c0 + max(1, self.c_chunk), C)
                v_in = v_base[:, c0:c1].contiguous()
                accum_c = torch.zeros(B, c1 - c0, cos_t.shape[0], V, U, device=device, dtype=vol.dtype)

                # Step chunking along t
                for s0 in range(0, self.n_steps, self.step_chunk):
                    s1 = min(s0 + self.step_chunk, self.n_steps)

                    i = torch.arange(s0, s1, device=device, dtype=vol.dtype).view(1, -1, 1, 1)
                    t = (-self.T + (i + 0.5) * self.delta_t)  # [1,S,1,1]

                    # Ray points in physical coords
                    x = (x0 + t * cos_t).expand(-1, -1, V, -1)      # [Aa,S,V,U]
                    y = (y0 + t * sin_t).expand(-1, -1, V, -1)      # [Aa,S,V,U]
                    z = (z0 + torch.zeros_like(t)).expand(-1, -1, -1, U)  # [Aa,S,V,U]

                    # → voxel indices → normalized for grid_sample
                    x_idx = x / self.sx + (W - 1) / 2.0
                    y_idx = y / self.sy + (H - 1) / 2.0
                    z_idx = z / self.sd + (D - 1) / 2.0
                    x_n = _safe_norm(x_idx, W); y_n = _safe_norm(y_idx, H); z_n = _safe_norm(z_idx, D)

                    # 5D grid for grid_sample: [N, D_out=S, H_out=V, W_out=U, 3]
                    grid = torch.stack([x_n, y_n, z_n], dim=-1)  # [Aa,S,V,U,3]
                    grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1, -1).reshape(
                        B * cos_t.shape[0], s1 - s0, V, U, 3
                    )
                    if grid.dtype != v_in.dtype:
                        grid = grid.to(v_in.dtype)

                    # Trilinear sampling over volume
                    samples = F.grid_sample(v_in, grid, mode="bilinear",
                                            padding_mode="zeros", align_corners=True)  # [N,Cc,S,V,U]

                    # Sum over steps, restore [B,Cc,Aa,V,U]
                    part = samples.sum(dim=2).view(B, cos_t.shape[0], c1 - c0, V, U).permute(0, 2, 1, 3, 4)
                    accum_c.add_(part)

                    # Clean up big temporaries
                    del i, t, x, y, z, x_idx, y_idx, z_idx, x_n, y_n, z_n, grid, samples, part

                sino_accum[:, c0:c1].add_(accum_c)
                del v_in, accum_c

            # Scale accumulated sum by delta_t
            sino = sino_accum.mul(self.delta_t).contiguous()
            sino_full[:, :, a0:a1, :, :] = sino
            del sino_accum, sino, v_base

        # Convert internal [B,C,A,V,U] → model [B,C,X,A,Z] with (U→X, V→Z)
        sino_xaz = sino_full.permute(0, 1, 4, 2, 3).contiguous()
        return sino_xaz

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Sinogram → Volume (Joseph backprojection), **model-facing I/O**.

        Parameters
        ----------
        sino : Tensor
            **[B, C, X, A, Z]** sinogram (x,a,z).

        Returns
        -------
        Tensor
            **[B, C, X, Y, Z]** backprojected volume (x,y,z).
        """
        # Convert model layout [B,C,X,A,Z] → internal [B,C,A,V,U]
        sino = sino.permute(0, 1, 3, 4, 2).contiguous()
        B, C, A, V, U = sino.shape
        D, H, W = self.geom.D, self.geom.H, self.geom.W
        device = sino.device

        su = float(self.geom.det_spacing[1])
        sv = float(self.geom.det_spacing[0])

        def _safe_norm(idx: torch.Tensor, size: int) -> torch.Tensor:
            """Map indices to [-1, 1] for grid_sample."""
            if size <= 1:
                return torch.zeros_like(idx)
            return (2.0 * idx) / (size - 1) - 1.0

        # Physical coordinates of voxel centers
        z_phys = (torch.arange(D, device=device, dtype=torch.float32) - (D - 1) / 2.0) * self.sd
        y_phys = (torch.arange(H, device=device, dtype=torch.float32) - (H - 1) / 2.0) * self.sy
        x_phys = (torch.arange(W, device=device, dtype=torch.float32) - (W - 1) / 2.0) * self.sx

        # XY meshgrid (indexing="xy" → X with W-axis, Y with H-axis)
        Xg, Yg = torch.meshgrid(x_phys, y_phys, indexing="xy")

        # Output accumulator in internal layout [B,C,D,H,W]
        vol = torch.zeros((B, C, D, H, W), device=device, dtype=sino.dtype)

        for a0 in range(0, A, self.geom.angle_chunk):
            a1 = min(a0 + self.geom.angle_chunk, A)
            cos_t = self.cos_angles[a0:a1].view(-1, 1, 1)
            sin_t = self.sin_angles[a0:a1].view(-1, 1, 1)

            # Map (X,Y) plane to detector u for each angle
            # U_phys = -X*sinθ + Y*cosθ
            U_phys_map = (-Xg.T.unsqueeze(0) * sin_t + Yg.T.unsqueeze(0) * cos_t)
            u_idx = U_phys_map / su + (U - 1) / 2.0
            u_norm = _safe_norm(u_idx, U)

            # Reorder sinogram for grid_sample: [Aa*B, C, V, U]
            x_in_all = sino[:, :, a0:a1, :, :].permute(0, 2, 1, 3, 4).reshape(B * cos_t.shape[0], C, V, U)

            # Iterate over Z (internal D)
            for k in range(D):
                v_idx = z_phys[k] / sv + (V - 1) / 2.0
                v_norm = _safe_norm(v_idx, V)  # scalar

                # Build grid for 2D sampling over (U,V): [N, H, W, 2]
                grid = torch.stack([u_norm, torch.full_like(u_norm, v_norm)], dim=-1)
                grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1).view(B * cos_t.shape[0], H, W, 2)

                # Channel chunking
                for c0 in range(0, C, max(1, self.c_chunk)):
                    c1 = min(c0 + max(1, self.c_chunk), C)
                    x_in = x_in_all[:, c0:c1].contiguous()
                    grid_cast = grid.to(x_in.dtype) if grid.dtype != x_in.dtype else grid

                    samp = F.grid_sample(x_in, grid_cast, mode="bilinear",
                                         padding_mode="zeros", align_corners=True)
                    samp = samp.view(B, cos_t.shape[0], c1 - c0, H, W).sum(dim=1)
                    vol[:, c0:c1, k, :, :].add_(samp)
                    del x_in, samp

                del grid

            del x_in_all

        # Scale by step length
        vol = vol * self.delta_t

        # Convert internal [B,C,D,H,W] → model [B,C,X,Y,Z] (W→X, H→Y, D→Z)
        vol_xyz = vol.permute(0, 1, 4, 3, 2).contiguous()
        return vol_xyz


class SiddonProjector3D(BaseProjector3D):
    """
    Siddon 3D projector (ray-driven, analytical voxel intersection lengths).

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
        # Model [B,C,X,Y,Z] → internal [B,C,D,H,,W]
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
