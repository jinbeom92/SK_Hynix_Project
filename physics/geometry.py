from dataclasses import dataclass
from typing import Tuple, ClassVar
import torch


@dataclass
class Parallel3DGeometry:
    """
    Geometry specification for 3D parallel-beam CT acquisition.

    Model-facing axis convention
    ----------------------------
    • Volume      : **(x, y, z)** → tensors shaped **[B, C, X, Y, Z]**
    • Sinogram    : **(x, a, z)** → tensors shaped **[B, C, X, A, Z]**
    • Detector    : horizontal **x** has U bins; vertical **z** has V bins.

    Storage vs. convenience
    -----------------------
    Internally we keep legacy fields to preserve compatibility with projector code:
      - `vol_shape`   stores **(D, H, W) = (Z, Y, X)**
      - `det_shape`   stores **(V, U)   = (Z, X)**
      - `voxel_size`  stores **(sd, sy, sx) = (sz, sy, sx)**
      - `det_spacing` stores **(sv, su) = (sz_det, sx_det)**
    while exposing **model-facing** convenience properties:
      - `X, Y, Z`
      - `shape_xyz = (X, Y, Z)`
      - `det_shape_xz = (X, Z)`
      - `voxel_size_xyz = (sx, sy, sz)`
      - `det_spacing_xz = (su, sv)`

    Chunking/steps
    --------------
    `angle_chunk` bounds angles processed per loop; `n_steps_cap` caps integration
    steps for sampled-ray projectors.

    Attributes (stored)
    -------------------
    vol_shape : (D, H, W)   == (Z, Y, X)
    det_shape : (V, U)      == (Z, X)
    angles    : [A] radians
    voxel_size: (sd, sy, sx)== (sz, sy, sx)
    det_spacing: (sv, su)   == (sz_det, sx_det)
    angle_chunk: int
    n_steps_cap: int
    """

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
        """
        Build geometry **from model-facing shapes**.

        Parameters
        ----------
        vol_shape_xyz : (X, Y, Z)
        det_shape_xz  : (X, Z) detector bins along x and z
        angles        : [A] radians
        voxel_size_xyz: (sx, sy, sz)
        det_spacing_xz: (su, sv)
        """
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
        """Number of projection angles."""
        return int(self.angles.numel())

    @property
    def D(self) -> int:
        """Depth (slices) == Z."""
        return int(self.vol_shape[0])

    @property
    def H(self) -> int:
        """Height (voxels) == Y."""
        return int(self.vol_shape[1])

    @property
    def W(self) -> int:
        """Width (voxels) == X."""
        return int(self.vol_shape[2])

    @property
    def V(self) -> int:
        """Detector vertical bins == Z."""
        return int(self.det_shape[0])

    @property
    def U(self) -> int:
        """Detector horizontal bins == X."""
        return int(self.det_shape[1])

    # ----------------------------
    # Model-facing convenience
    # ----------------------------
    @property
    def X(self) -> int:
        """Width in model coords (X)."""
        return self.W

    @property
    def Y(self) -> int:
        """Height in model coords (Y)."""
        return self.H

    @property
    def Z(self) -> int:
        """Depth in model coords (Z)."""
        return self.D

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        """Model-facing volume shape (X, Y, Z)."""
        return (self.W, self.H, self.D)

    @property
    def det_shape_xz(self) -> Tuple[int, int]:
        """Model-facing detector shape (X, Z)."""
        return (self.U, self.V)

    @property
    def voxel_size_xyz(self) -> Tuple[float, float, float]:
        """Model-facing voxel spacing (sx, sy, sz)."""
        sd, sy, sx = self.voxel_size
        return (sx, sy, sd)

    @property
    def det_spacing_xz(self) -> Tuple[float, float]:
        """Model-facing detector spacing (su along x, sv along z)."""
        sv, su = self.det_spacing
        return (su, sv)

    def device(self) -> torch.device:
        """Device where `angles` tensor lives."""
        return self.angles.device
