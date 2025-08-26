from typing import Optional, Literal
import torch
import torch.nn.functional as F
from .geometry import Parallel3DGeometry


# ------------------------------------------------------------------------------------------
# Base
# ------------------------------------------------------------------------------------------
class BaseProjector3D(torch.nn.Module):
    def __init__(self, geom: Parallel3DGeometry):
        super().__init__()
        self.geom = geom

    def reset_geometry(self, geom: Parallel3DGeometry):
        raise NotImplementedError

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ------------------------------------------------------------------------------------------
# Joseph (voxel-driven) + Unfiltered BP
# ------------------------------------------------------------------------------------------
class JosephProjector3D(BaseProjector3D):
    def __init__(self, geom: Parallel3DGeometry, n_steps: Optional[int] = None):
        super().__init__(geom)

        # Cached angles (radians) and trig
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

        # Ray integration extent and step
        T = 0.5 * float(((W - 1) * sx) ** 2 + ((H - 1) * sy) ** 2) ** 0.5
        self.T = float(T)
        self.delta_t = (2.0 * self.T) / float(self.n_steps)
        self.step_chunk = max(1, min(8, self.n_steps))
        self.c_chunk = 4

        # BP options (no filter anymore)
        self.ir_interpolation: str = "linear"  # {'linear','nearest'}
        self.ir_circle: bool = True            # circular support mask

    @torch.no_grad()
    def reset_geometry(self, geom: Parallel3DGeometry):
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

    # -----------------------------------
    # Joseph forward (unchanged)
    # -----------------------------------
    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        # [B,C,X,Y,Z] → internal [B,C,D,H,W]
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

    # -----------------------------------
    # Helpers for unfiltered BP
    # -----------------------------------
    @staticmethod
    def _next_pow2(n: int) -> int:
        return max(64, 1 << ((2 * n - 1).bit_length()))

    @staticmethod
    def _sinogram_circle_to_square_torch(sino: torch.Tensor) -> torch.Tensor:
        import math
        N, A = sino.shape
        diagonal = int(math.ceil(math.sqrt(2.0) * N))
        pad = diagonal - N
        old_center = N // 2
        new_center = diagonal // 2
        pad_before = new_center - old_center
        return F.pad(sino, (0, 0, pad_before, pad - pad_before), mode="constant", value=0.0)

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
    def _iradon_unfiltered_2d(
        radon_image: torch.Tensor,   # [N, A]
        theta_deg: torch.Tensor,     # [A]
        output_size: int,
        interpolation: str = "linear",
        circle: bool = True,
    ) -> torch.Tensor:

        if radon_image.ndim != 2:
            raise ValueError("radon_image must be [N, A].")
        if theta_deg.ndim != 1 or theta_deg.numel() != radon_image.shape[1]:
            raise ValueError("theta length must match number of projection columns.")

        import math
        device = radon_image.device
        dtype_in = radon_image.dtype
        dtype_acc = torch.float32

        N, A = radon_image.shape
        sino = radon_image
        if circle:
            sino = JosephProjector3D._sinogram_circle_to_square_torch(sino)
            N = sino.shape[0]

        recon = torch.zeros((output_size, output_size), device=device, dtype=dtype_acc)
        radius = output_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(output_size, device=device, dtype=dtype_acc),
            torch.arange(output_size, device=device, dtype=dtype_acc),
            indexing="ij",
        )
        ypr = yy - radius
        xpr = xx - radius

        theta_rad = torch.deg2rad(theta_deg.to(dtype_acc))
        cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)

        if interpolation not in ("linear", "nearest"):
            raise ValueError("interpolation must be 'linear' or 'nearest'.")

        for i in range(A):
            col = sino[:, i].to(dtype_acc)
            t = ypr * cos_t[i] - xpr * sin_t[i]
            if interpolation == "linear":
                recon += JosephProjector3D._interp1d_linear_torch(col, t)
            else:
                recon += JosephProjector3D._interp1d_nearest_torch(col, t)

        if circle:
            mask = (xpr**2 + ypr**2) > (radius**2)
            recon = torch.where(mask, torch.zeros((), dtype=dtype_acc, device=device), recon)

        recon = recon * (math.pi / (2.0 * A))
        return recon.to(dtype_in)

    # Differentiable BP
    def backproject(self, sino: torch.Tensor) -> torch.Tensor:
        if sino.ndim != 5:
            raise ValueError(f"Expected [B,C,X,A,Z], got {tuple(sino.shape)}")
        B, C, X, A, Z = sino.shape
        Y = self.geom.H
        device = sino.device
        dtype = sino.dtype

        import math
        theta_deg = (self.geom.angles * (180.0 / math.pi)).to(torch.float32).to(device)

        outs_B = []
        for b in range(B):
            outs_C = []
            for c in range(C):
                s_bc = sino[b, c]  # [X, A, Z]
                rec_slices = []
                for z in range(Z):
                    s_2d = s_bc[:, :, z].to(dtype)  # [X, A]
                    rec_y_y = JosephProjector3D._iradon_unfiltered_2d(
                        s_2d, theta_deg, output_size=Y,
                        interpolation=self.ir_interpolation, circle=self.ir_circle
                    )  # [Y, Y]

                    # shape == [X, Y]
                    if X != Y:
                        # [Y,Y] → [Y,X] resize → [X,Y]
                        rec_yx = F.interpolate(
                            rec_y_y.unsqueeze(0).unsqueeze(0),  # [1,1,Y,Y]
                            size=(Y, X), mode="bilinear", align_corners=False
                        )[0, 0]                                  # [Y,X]
                        rec_xy = rec_yx.t()                      # [X,Y]
                    else:
                        rec_xy = rec_y_y.t()                     # [Y,Y] == [X,Y]
                    rec_slices.append(rec_xy.unsqueeze(-1))      # [X, Y, 1]
                out_c = torch.cat(rec_slices, dim=-1)       # [X, Y, Z]
                outs_C.append(out_c.unsqueeze(0))           # [1, X, Y, Z]
            out_b = torch.cat(outs_C, dim=0)                # [C, X, Y, Z]
            outs_B.append(out_b.unsqueeze(0))               # [1, C, X, Y, Z]
        vol_xyz = torch.cat(outs_B, dim=0)                  # [B, C, X, Y, Z]

        # Optional final XY circular support
        if self.ir_circle:
            xx = torch.arange(X, device=device, dtype=torch.float32) - (X - 1) / 2.0
            yy = torch.arange(Y, device=device, dtype=torch.float32) - (Y - 1) / 2.0
            Xg, Yg = torch.meshgrid(xx, yy, indexing="ij")
            r = min((X - 1) / 2.0, (Y - 1) / 2.0)
            circ = ((Xg**2 + Yg**2) <= (r**2)).to(dtype)
            vol_xyz = vol_xyz * circ.view(1, 1, X, Y, 1)
        return vol_xyz


# ------------------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------------------
def make_projector(method: Literal["joseph3d", "siddon3d"], geom: Parallel3DGeometry) -> BaseProjector3D:
    if method == "joseph3d":
        return JosephProjector3D(geom, n_steps=geom.n_steps_cap)
    else:
        raise ValueError(f"Unknown projector method: {method}")