from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class Parallel3DGeometry:
    vol_shape: Tuple[int, int, int]              # (D,H,W) == (Z,Y,X)
    det_shape: Tuple[int, int]                   # (V,U)   == (Z,X)
    angles: torch.Tensor                         # [A], radians
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (sd,sy,sx) == (sz,sy,sx)
    det_spacing: Tuple[float, float] = (1.0, 1.0)             # (sv,su)    == (sz_det,sx_det)
    angle_chunk: int = 16
    n_steps_cap: int = 256

    # ----------------------------
    # Convenience constructors
    # ----------------------------
    @classmethod
    def from_xyz(
        cls,
        vol_shape_xyz: Tuple[int, int, int],
        det_shape_xz: Tuple[int, int],
        angles: torch.Tensor,
        voxel_size_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        det_spacing_xz: Tuple[float, float] = (1.0, 1.0),
        angle_chunk: int = 16,
        n_steps_cap: int = 256,
    ) -> "Parallel3DGeometry":
        
        X, Y, Z = vol_shape_xyz
        U, V = det_shape_xz[0], det_shape_xz[1]
        sx, sy, sz = voxel_size_xyz
        su, sv = det_spacing_xz
        # Store in legacy order for internal consumers:
        return cls(
            vol_shape=(Z, Y, X),
            det_shape=(V, U),
            angles=angles,
            voxel_size=(sz, sy, sx),
            det_spacing=(sv, su),
            angle_chunk=angle_chunk,
            n_steps_cap=n_steps_cap,
        )

    # ----------------------------
    # Legacy-style accessors
    # ----------------------------
    @property
    def A(self) -> int:
        return int(self.angles.numel())

    @property
    def D(self) -> int:
        return int(self.vol_shape[0])

    @property
    def H(self) -> int:
        return int(self.vol_shape[1])

    @property
    def W(self) -> int:
        return int(self.vol_shape[2])

    @property
    def V(self) -> int:
        return int(self.det_shape[0])

    @property
    def U(self) -> int:
        return int(self.det_shape[1])

    # ----------------------------
    # Model-facing convenience
    # ----------------------------
    @property
    def X(self) -> int:
        return self.W

    @property
    def Y(self) -> int:
        return self.H

    @property
    def Z(self) -> int:
        return self.D

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        return (self.W, self.H, self.D)

    @property
    def det_shape_xz(self) -> Tuple[int, int]:
        return (self.U, self.V)

    @property
    def voxel_size_xyz(self) -> Tuple[float, float, float]:
        sd, sy, sx = self.voxel_size
        return (sx, sy, sd)

    @property
    def det_spacing_xz(self) -> Tuple[float, float]:
        sv, su = self.det_spacing
        return (su, sv)

    def device(self) -> torch.device:
        return self.angles.device
