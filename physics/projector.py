from typing import Optional, Literal
import torch
import torch.nn.functional as F
from .geometry import Parallel3DGeometry


class BaseProjector3D(torch.nn.Module):
    """
    Abstract base class for 3D projectors (forward/backprojection).

    Purpose
    -------
    • Provide a common interface for forward projection (volume → sinogram)
      and backprojection (sinogram → volume) in 3D parallel-beam CT.
    • Concrete implementations (e.g., JosephProjector3D, SiddonProjector3D)
      define the specific ray traversal/integration scheme.

    Args
    ----
    geom : Parallel3DGeometry
        Geometry specification including volume/detector shapes and angles.
    """

    def __init__(self, geom: Parallel3DGeometry):
        super().__init__()
        self.geom = geom

    def reset_geometry(self, geom: Parallel3DGeometry):
        """
        Reset internal buffers when geometry changes.

        Implementations should update any cached tensors that depend on geometry,
        such as angles, trigonometric caches, detector coordinates, step sizes, etc.
        """
        raise NotImplementedError

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Forward projection: volume → sinogram.

        Args:
            vol : Tensor
                [B, C, D, H, W] input volume.

        Returns:
            Tensor: [B, C, A, V, U] sinogram.
        """
        raise NotImplementedError

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Backprojection: sinogram → volume.

        Args:
            sino : Tensor
                [B, C, A, V, U] input sinogram.

        Returns:
            Tensor: [B, C, D, H, W] backprojected volume.
        """
        raise NotImplementedError


class JosephProjector3D(BaseProjector3D):
    """
    Joseph 3D projector (voxel-driven / sampled ray integration).

    Method
    ------
    • Integrates along rays parameterized by detector coordinates (u, v) and angle θ.
    • Samples the volume on a regular set of parametric steps using
      `torch.nn.functional.grid_sample` (trilinear interpolation).
    • Accumulates samples along the ray and scales by the step length `delta_t`.

    Notes
    -----
    • Chunking over angles (angle_chunk), channels (c_chunk), and steps (step_chunk)
      reduces peak memory and enables long trajectories.
    • `align_corners=True` is used in `grid_sample` because the index normalization
      maps voxel centers to normalized coordinates in [-1, 1].
    """

    def __init__(self, geom: Parallel3DGeometry, n_steps: Optional[int] = None):
        super().__init__(geom)
        # Cache angle tensor and its trig for speed
        self.register_buffer("angles", geom.angles.clone().detach())
        self.register_buffer("cos_angles", torch.cos(self.angles))
        self.register_buffer("sin_angles", torch.sin(self.angles))

        # # of integration steps per ray (clamped by geometry cap if not provided)
        self.n_steps = int(n_steps if n_steps is not None else geom.n_steps_cap)

        # --- Detector physical coordinates centered at 0 (v: vertical, u: horizontal)
        V, U = geom.V, geom.U
        sv, su = geom.det_spacing
        v = (torch.arange(V, dtype=torch.float32) - (V - 1) / 2.0) * sv
        u = (torch.arange(U, dtype=torch.float32) - (U - 1) / 2.0) * su
        self.register_buffer("u_phys", u)
        self.register_buffer("v_phys", v)

        # --- Voxel spacing & integration step size
        sd, sy, sx = geom.voxel_size
        D, H, W = geom.D, geom.H, geom.W
        self.sx = float(sx); self.sy = float(sy); self.sd = float(sd)

        # Half diagonal radius in the XY plane to span the volume footprint
        T = 0.5 * float(((W - 1) * sx) ** 2 + ((H - 1) * sy) ** 2) ** 0.5
        self.T = float(T)
        self.delta_t = (2.0 * self.T) / float(self.n_steps)  # uniform parametric step

        # --- Runtime chunking knobs (override if needed)
        self.step_chunk = max(1, min(8, self.n_steps))  # steps per inner loop
        self.c_chunk = 4                                # channels per inner loop

    @torch.no_grad()
    def reset_geometry(self, geom: Parallel3DGeometry):
        """
        Update cached geometry-dependent buffers (angles, trig caches, detector coords).

        This allows reusing the projector instance when geometry is changed
        without re-allocating a new module.
        """
        self.geom = geom

        # Refresh angles and trig caches
        new_angles = geom.angles.detach().to(self.angles.device, dtype=self.angles.dtype)
        self.angles.resize_(new_angles.shape).copy_(new_angles)
        self.cos_angles.resize_(new_angles.shape).copy_(torch.cos(new_angles))
        self.sin_angles.resize_(new_angles.shape).copy_(torch.sin(new_angles))

        # Detector axes (u, v) in physical coordinates
        V, U = geom.V, geom.U
        sv, su = geom.det_spacing
        v = (torch.arange(V, dtype=torch.float32, device=self.angles.device) - (V - 1) / 2.0) * sv
        u = (torch.arange(U, dtype=torch.float32, device=self.angles.device) - (U - 1) / 2.0) * su
        if hasattr(self, "u_phys"): self.u_phys.resize_(u.shape).copy_(u)
        else: self.register_buffer("u_phys", u)
        if hasattr(self, "v_phys"): self.v_phys.resize_(v.shape).copy_(v)
        else: self.register_buffer("v_phys", v)

        # Spacing & parametric step
        sd, sy, sx = geom.voxel_size
        D, H, W = geom.D, geom.H, geom.W
        self.sx = float(sx); self.sy = float(sy); self.sd = float(sd)
        T = 0.5 * float(((W - 1) * sx) ** 2 + ((H - 1) * sy) ** 2) ** 0.5
        self.T = float(T)
        self.delta_t = (2.0 * self.T) / float(self.n_steps)
        self.step_chunk = max(1, min(self.step_chunk, self.n_steps))

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Volume → Sinogram (Joseph forward projection).

        Args:
            vol : Tensor
                [B, C, D, H, W] input volume (dtype/precision arbitrary)

        Returns:
            Tensor: [B, C, A, V, U] sinogram
        """
        B, C, D, H, W = vol.shape
        A = int(self.angles.numel())
        V, U = self.geom.V, self.geom.U
        device = vol.device

        def _safe_norm(idx: torch.Tensor, size: int) -> torch.Tensor:
            """Map voxel indices (0..size-1) to grid_sample range [-1, 1]."""
            if size <= 1:
                return torch.zeros_like(idx)
            return (2.0 * idx) / (size - 1) - 1.0

        # Allocate output buffer for all angle chunks
        sino_full = torch.empty((B, C, A, V, U), device=device, dtype=vol.dtype)

        # Process angles in chunks to control memory
        for a0 in range(0, A, self.geom.angle_chunk):
            a1 = min(a0 + self.geom.angle_chunk, A)

            # Trig terms for the chunk
            cos_t = self.cos_angles[a0:a1].view(-1, 1, 1, 1)  # [Aa,1,1,1]
            sin_t = self.sin_angles[a0:a1].view(-1, 1, 1, 1)  # [Aa,1,1,1]

            # Detector planes (broadcastable)
            u_phys = self.u_phys.view(1, 1, 1, U)  # [1,1,1,U]
            v_phys = self.v_phys.view(1, 1, V, 1)  # [1,1,V,1]

            # Ray bases at t = 0 for each (u,v,θ)
            x0 = -u_phys * sin_t
            y0 =  u_phys * cos_t
            z0 =  v_phys.expand(cos_t.shape[0], 1, V, 1)  # copy per-angle

            # Accumulator for this angle chunk
            sino_accum = torch.zeros(B, C, cos_t.shape[0], V, U, device=device, dtype=vol.dtype)

            # Repeat/expand volume across angle chunk in the batch dimension
            v_base = vol.expand(cos_t.shape[0], C, D, H, W) if B == 1 else vol.repeat_interleave(cos_t.shape[0], dim=0)

            # Iterate over channel chunks to limit memory
            for c0 in range(0, C, max(1, self.c_chunk)):
                c1 = min(c0 + max(1, self.c_chunk), C)
                v_in = v_base[:, c0:c1].contiguous()
                accum_c = torch.zeros(B, c1 - c0, cos_t.shape[0], V, U, device=device, dtype=vol.dtype)

                # Integrate along t in small step chunks
                for s0 in range(0, self.n_steps, self.step_chunk):
                    s1 = min(s0 + self.step_chunk, self.n_steps)

                    # Parametric steps centered in each interval
                    i = torch.arange(s0, s1, device=device, dtype=vol.dtype).view(1, -1, 1, 1)
                    t = (-self.T + (i + 0.5) * self.delta_t)  # [1, S, 1, 1]

                    # Ray points in physical coords (broadcast across V/U)
                    x = (x0 + t * cos_t).expand(-1, -1, V, -1)        # [Aa,S,V,U]
                    y = (y0 + t * sin_t).expand(-1, -1, V, -1)        # [Aa,S,V,U]
                    z = (z0 + torch.zeros_like(t)).expand(-1, -1, -1, U)  # [Aa,S,V,U]

                    # Convert to voxel indices, then normalize to [-1, 1]
                    x_idx = x / self.sx + (W - 1) / 2.0
                    y_idx = y / self.sy + (H - 1) / 2.0
                    z_idx = z / self.sd + (D - 1) / 2.0
                    x_n = _safe_norm(x_idx, W); y_n = _safe_norm(y_idx, H); z_n = _safe_norm(z_idx, D)

                    # Build 5D grid for grid_sample: [N, D_out, H_out, W_out, 3]
                    grid = torch.stack([x_n, y_n, z_n], dim=-1)  # [Aa,S,V,U,3]
                    grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1, -1).reshape(
                        B * cos_t.shape[0], s1 - s0, V, U, 3
                    )
                    if grid.dtype != v_in.dtype:
                        grid = grid.to(v_in.dtype)

                    # Trilinear sampling over the volume
                    # v_in: [N, Cc, D, H, W], grid: [N, S, V, U, 3]
                    samples = F.grid_sample(v_in, grid, mode="bilinear",
                                            padding_mode="zeros", align_corners=True)

                    # Sum along steps, reshape back to [B, Cc, Aa, V, U]
                    part = samples.sum(dim=2).view(B, cos_t.shape[0], c1 - c0, V, U).permute(0, 2, 1, 3, 4)
                    accum_c.add_(part)

                    # Help GC on very large runs
                    del i, t, x, y, z, x_idx, y_idx, z_idx, x_n, y_n, z_n, grid, samples, part

                # Merge channel-chunk accumulation
                sino_accum[:, c0:c1].add_(accum_c)
                del v_in, accum_c

            # Convert accumulated sum to integral via delta_t
            sino = sino_accum.mul(self.delta_t).contiguous()
            sino_full[:, :, a0:a1, :, :] = sino

            del sino_accum, sino, v_base

        return sino_full

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Sinogram → Volume (Joseph backprojection).

        Args:
            sino : Tensor
                [B, C, A, V, U] sinogram.

        Returns:
            Tensor: [B, C, D, H, W] backprojected volume.
        """
        B, C, A, V, U = sino.shape
        D, H, W = self.geom.D, self.geom.H, self.geom.W
        device = sino.device

        # Detector spacings
        su = float(self.geom.det_spacing[1])
        sv = float(self.geom.det_spacing[0])

        def _safe_norm(idx: torch.Tensor, size: int) -> torch.Tensor:
            """Map indices to [-1, 1] for grid_sample."""
            if size <= 1:
                return torch.zeros_like(idx)
            return (2.0 * idx) / (size - 1) - 1.0

        # Physical coordinates of voxel centers along each axis
        z_phys = (torch.arange(D, device=device, dtype=torch.float32) - (D - 1) / 2.0) * self.sd
        y_phys = (torch.arange(H, device=device, dtype=torch.float32) - (H - 1) / 2.0) * self.sy
        x_phys = (torch.arange(W, device=device, dtype=torch.float32) - (W - 1) / 2.0) * self.sx

        # XY meshgrid (indexing="xy" yields X with W-axis, Y with H-axis)
        X, Y = torch.meshgrid(x_phys, y_phys, indexing="xy")

        # Output accumulator
        vol = torch.zeros((B, C, D, H, W), device=device, dtype=sino.dtype)

        # Angle chunking
        for a0 in range(0, A, self.geom.angle_chunk):
            a1 = min(a0 + self.geom.angle_chunk, A)
            cos_t = self.cos_angles[a0:a1].view(-1, 1, 1)
            sin_t = self.sin_angles[a0:a1].view(-1, 1, 1)

            # For each angle, map (X,Y) voxel plane to detector u coordinate
            # U_phys = -X*sinθ + Y*cosθ  (ray coordinate orthogonal to beam)
            U_phys_map = (-X.T.unsqueeze(0) * sin_t + Y.T.unsqueeze(0) * cos_t)
            u_idx = U_phys_map / su + (U - 1) / 2.0
            u_norm = _safe_norm(u_idx, U)  # normalized u for grid_sample

            # Reorder sinogram dims for grid_sample consumption
            # x_in_all: [Aa*B, C, V, U]
            x_in_all = sino[:, :, a0:a1, :, :].permute(0, 2, 1, 3, 4).reshape(B * cos_t.shape[0], C, V, U)

            # Iterate over Z-slices; for each z, v_idx is constant over (X,Y)
            for k in range(D):
                v_idx = z_phys[k] / sv + (V - 1) / 2.0
                v_norm = _safe_norm(v_idx, V)  # scalar

                # Build grid for 2D grid_sample over (U,V): [N, H, W, 2]
                grid = torch.stack([u_norm, torch.full_like(u_norm, v_norm)], dim=-1)
                grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1).view(B * cos_t.shape[0], H, W, 2)

                # Channel chunking
                for c0 in range(0, C, max(1, self.c_chunk)):
                    c1 = min(c0 + max(1, self.c_chunk), C)
                    x_in = x_in_all[:, c0:c1].contiguous()
                    grid_cast = grid.to(x_in.dtype) if grid.dtype != x_in.dtype else grid

                    # Sample along detector plane and sum over angles in the chunk
                    samp = F.grid_sample(x_in, grid_cast, mode="bilinear",
                                         padding_mode="zeros", align_corners=True)
                    samp = samp.view(B, cos_t.shape[0], c1 - c0, H, W).sum(dim=1)
                    vol[:, c0:c1, k, :, :].add_(samp)
                    del x_in, samp

                del grid

            del x_in_all

        # Scale by the parametric step length to approximate the line integral
        vol = vol * self.delta_t
        return vol


class SiddonProjector3D(BaseProjector3D):
    """
    Siddon 3D projector (ray-driven, analytical voxel intersection lengths).

    Method
    ------
    • For each ray, computes the sequence of voxels intersected and the
      exact segment length inside each voxel (2D extension in XY; Z handled
      by stacking planes).
    • Forward: sum voxel values × intersection lengths along ray.
    • Backward: scatter sinogram ray value × intersection lengths back to voxels.

    Notes
    -----
    • This implementation focuses on XY traversal per Z, using simple bounds.
    • Slower but more accurate than Joseph for validation/reference.
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

        Args:
            D, H, W : int
                Volume dimensions.
            x0, y0, z0 : float
                Entry point (physical coordinates) at t = t_entry.
            cos_t, sin_t : float
                Direction cosines for the current angle in the XY plane.
            device : torch.device
                Device for returned tensors.

        Returns:
            idx : LongTensor [N, 3]
                Voxel indices (k, j, i) visited along the ray.
            wts : FloatTensor [N]
                Segment lengths inside each voxel (same order as idx).
        """
        # Volume bounds in physical space (centered at 0)
        x_min = -((W - 1) / 2.0) * self.sx; x_max = ((W - 1) / 2.0) * self.sx
        y_min = -((H - 1) / 2.0) * self.sy; y_max = ((H - 1) / 2.0) * self.sy
        dx = cos_t; dy = sin_t

        # Compute parametric entry/exit for x and y slabs
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
        t_entry = max(tx0, ty0)
        t_exit = min(tx1, ty1)
        if (t_entry >= t_exit):
            return (torch.empty(0, 3, device=device, dtype=torch.long),
                    torch.empty(0, device=device, dtype=torch.float32))

        # Entry voxel indices (rounded to nearest voxel center)
        xe = x0 + t_entry * dx; ye = y0 + t_entry * dy; ze = z0
        i = int(round(xe / self.sx + (W - 1) / 2.0))
        j = int(round(ye / self.sy + (H - 1) / 2.0))
        k = int(round(ze / self.sd + (D - 1) / 2.0))
        i = max(0, min(W - 1, i)); j = max(0, min(H - 1, j)); k = max(0, min(D - 1, k))

        inds = []; lens = []; x = xe; y = ye; t = t_entry
        while t < t_exit - 1e-12:
            # Next voxel face in x and y
            if dx > 0: x_next = ((i + 0.5) * self.sx - (W - 1) / 2.0 * self.sx)
            elif dx < 0: x_next = ((i - 0.5) * self.sx - (W - 1) / 2.0 * self.sx)
            else: x_next = float("inf")
            if dy > 0: y_next = ((j + 0.5) * self.sy - (H - 1) / 2.0 * self.sy)
            elif dy < 0: y_next = ((j - 0.5) * self.sy - (H - 1) / 2.0 * self.sy)
            else: y_next = float("inf")

            # Parametric distance to faces
            tx = (x_next - x) / dx if abs(dx) > 1e-12 else float("inf")
            ty = (y_next - y) / dy if abs(dy) > 1e-12 else float("inf")
            dt = min(tx, ty, t_exit - t)

            # Segment length in current voxel (scale to physical distance)
            seg = max(0.0, float(dt) * (self.sx**2 + self.sy**2)**0.5 / max(abs(dx) + abs(dy), 1e-12))
            inds.append((k, j, i)); lens.append(seg)

            # Advance to next voxel
            t += dt; x += dx * dt; y += dy * dt
            if tx < ty:
                i += 1 if dx > 0 else -1
                if i < 0 or i >= W: break
            elif ty < tx:
                j += 1 if dy > 0 else -1
                if j < 0 or j >= H: break
            else:
                # Crossing corner: advance both i and j
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
        Volume → Sinogram (Siddon forward).

        Args:
            vol : Tensor
                [B, C, D, H, W] input volume.

        Returns:
            Tensor: [B, C, A, V, U] sinogram.
        """
        B, C, D, H, W = vol.shape
        A = int(self.angles.numel()); V, U = self.geom.V, self.geom.U
        device = vol.device
        out = torch.zeros((B, C, A, V, U), device=device, dtype=vol.dtype)

        # Angle chunking for memory locality
        for a0 in range(0, A, self.geom.angle_chunk):
            a1 = min(a0 + self.geom.angle_chunk, A)
            th = self.angles[a0:a1]
            cos_t = torch.cos(th); sin_t = torch.sin(th)

            # Iterate rays (v,u) for each angle in chunk
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

                        if wts.dtype != vol.dtype:
                            wts = wts.to(vol.dtype)

                        # Gather voxel values along the ray and integrate
                        kji = idx.t()  # (k, j, i)
                        vvals = vol[:, :, kji[0], kji[1], kji[2]]
                        contrib = (vvals * wts.view(1, 1, -1)).sum(dim=-1)
                        out[:, :, a0 + ai, vv, uu] = contrib
        return out

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Sinogram → Volume (Siddon backprojection).

        Args:
            sino : Tensor
                [B, C, A, V, U] sinogram.

        Returns:
            Tensor: [B, C, D, H, W] backprojected volume.
        """
        B, C, A, V, U = sino.shape
        D, H, W = self.geom.D, self.geom.H, self.geom.W
        device = sino.device

        vol = torch.zeros((B, C, D, H, W), device=device, dtype=sino.dtype)

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

                        ray = sino[:, :, a0 + ai, vv, uu]
                        if wts.dtype != ray.dtype:
                            wts = wts.to(ray.dtype)

                        # Scatter-add along visited voxels
                        kji = idx.t()  # (k, j, i)
                        scatter = (ray.view(B, C, 1) * wts.view(1, 1, -1))
                        flat_idx = (kji[0] * H * W + kji[1] * W + kji[2]).view(-1)
                        vol_flat = vol.view(B * C, D * H * W)
                        vol_flat.index_add_(1, flat_idx, scatter.view(B * C, -1))
        return vol


def make_projector(method: Literal["joseph3d", "siddon3d"], geom: Parallel3DGeometry) -> BaseProjector3D:
    """
    Factory for 3D projectors.

    Args
    ----
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
