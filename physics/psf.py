from typing import Optional
import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int = None, device=None, dtype=None):
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
        if sino.dim() == 4:   # [B,X,A,Z]
            return sino.unsqueeze(1), True
        if sino.dim() == 5:   # [B,C,X,A,Z]
            return sino, False
        raise ValueError(f"sino must be [B,X,A,Z] or [B,C,X,A,Z], got {tuple(sino.shape)}")

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
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
        return self.forward(sino)
