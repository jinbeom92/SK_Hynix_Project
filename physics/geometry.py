"""
This dataclass encapsulates the geometry parameters needed for parallel
projection/backprojection in 3D.  It stores volume and detector shapes,
projection angles, voxel and detector spacings, and optional hyperparameters
(angle_chunk and n_steps_cap) controlling the projector implementation.  The
convenience constructor `from_xyz` accepts intuitive (X,Y,Z) notation and
internally maps to the (Z,Y,X) ordering expected by the underlying physics
engine.  Legacy-style properties (A, D, H, W, U, V) and model-facing
properties (X, Y, Z, shape_xyz, det_shape_xz) are provided for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class Parallel3DGeometry:
    """
    Attributes:
        vol_shape: Tuple (D, H, W) specifying the number of slices (depth),
            height (rows), and width (columns) of the reconstruction volume.
        det_shape: Tuple (V, U) specifying the detector height (v) and width (u).
        angles: 1-D tensor of projection angles in radians.
        voxel_size: Physical spacing (sz, sy, sx) of voxels (depth, height, width).
        det_spacing: Physical spacing (sv, su) of detector pixels (v, u).
        angle_chunk: Number of angles processed per chunk in forward projection.
        n_steps_cap: Maximum number of integration steps along each ray.

    The model-facing convenience properties X, Y, Z map to W, H, D
    respectively, and det_shape_xz maps to (U, V).  This aligns with the
    sinogram convention used in HDN where the sinogram has shape [X,A,Z].
    """

    vol_shape: Tuple[int, int, int]
    det_shape: Tuple[int, int]
    angles: torch.Tensor
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    det_spacing: Tuple[float, float] = (1.0, 1.0)
    angle_chunk: int = 16
    n_steps_cap: int = 256

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
        """Construct geometry using (X,Y,Z) and (U,V) notation.

        Args:
            vol_shape_xyz: (X,Y,Z) dimensions of the reconstruction volume.
            det_shape_xz: (U,V) dimensions of the detector (width, height).
            angles: 1-D tensor of projection angles in radians.
            voxel_size_xyz: Physical spacing of voxels (sx, sy, sz).
            det_spacing_xz: Physical spacing of detector pixels (su, sv).
            angle_chunk: Forward projector angle chunk size.
            n_steps_cap: Maximum integration steps along rays.

        Returns:
            A Parallel3DGeometry instance with internal (D,H,W) and (V,U)
            ordering.  Voxel and detector spacings are reordered to match.
        """
        X, Y, Z = vol_shape_xyz
        U, V = det_shape_xz
        sx, sy, sz = voxel_size_xyz
        su, sv = det_spacing_xz
        return cls(
            vol_shape=(Z, Y, X),
            det_shape=(V, U),
            angles=angles,
            voxel_size=(sz, sy, sx),
            det_spacing=(sv, su),
            angle_chunk=angle_chunk,
            n_steps_cap=n_steps_cap,
        )

    # Legacy-style accessors for backward compatibility
    @property
    def A(self) -> int:
        """Number of projection angles."""
        return int(self.angles.numel())

    @property
    def D(self) -> int:
        """Depth (number of slices) of the volume."""
        return int(self.vol_shape[0])

    @property
    def H(self) -> int:
        """Height (number of rows) of the volume."""
        return int(self.vol_shape[1])

    @property
    def W(self) -> int:
        """Width (number of columns) of the volume."""
        return int(self.vol_shape[2])

    @property
    def V(self) -> int:
        """Detector height."""
        return int(self.det_shape[0])

    @property
    def U(self) -> int:
        """Detector width."""
        return int(self.det_shape[1])

    # Model-facing convenience
    @property
    def X(self) -> int:
        """Alias for W (volume width)."""
        return self.W

    @property
    def Y(self) -> int:
        """Alias for H (volume height)."""
        return self.H

    @property
    def Z(self) -> int:
        """Alias for D (volume depth)."""
        return self.D

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        """Return volume dimensions as (X,Y,Z)."""
        return (self.W, self.H, self.D)

    @property
    def det_shape_xz(self) -> Tuple[int, int]:
        """Return detector dimensions as (U,V)."""
        return (self.U, self.V)

    @property
    def voxel_size_xyz(self) -> Tuple[float, float, float]:
        """Return voxel spacing as (sx, sy, sz)."""
        sd, sy, sx = self.voxel_size
        return (sx, sy, sd)

    @property
    def det_spacing_xz(self) -> Tuple[float, float]:
        """Return detector spacing as (su, sv)."""
        sv, su = self.det_spacing
        return (su, sv)

    def device(self) -> torch.device:
        """Return the device of the angles tensor."""
        return self.angles.device
