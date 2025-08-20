from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class Parallel3DGeometry:
    """
    Geometry specification for 3D parallel-beam CT acquisition.

    Purpose
    -------
    • Encapsulates all geometry parameters needed for forward/backward projection.
    • Defines volume dimensions, detector dimensions, projection angles,
      voxel spacing, detector spacing, and optional chunking/step caps.

    Attributes
    ----------
    vol_shape : Tuple[int,int,int]
        Volume shape (D, H, W):
          • D = depth (# slices along z)
          • H = height (# voxels along y)
          • W = width  (# voxels along x)

    det_shape : Tuple[int,int]
        Detector shape (V, U):
          • V = vertical bins
          • U = horizontal bins

    angles : torch.Tensor
        Projection angles [A] in radians.

    voxel_size : Tuple[float,float,float], default=(1.0,1.0,1.0)
        Physical voxel spacing (sd, sy, sx).

    det_spacing : Tuple[float,float], default=(1.0,1.0)
        Detector spacing (sv, su).

    angle_chunk : int, default=16
        Maximum number of projection angles processed per chunk
        (for memory efficiency).

    n_steps_cap : int, default=256
        Maximum number of integration steps per ray (safety cap).

    Properties
    ----------
    A : int   — number of projection angles
    D : int   — depth (slices)
    H : int   — height (voxels)
    W : int   — width (voxels)
    V : int   — detector vertical bins
    U : int   — detector horizontal bins

    Methods
    -------
    device() -> torch.device
        Returns the device where the angle tensor is stored.
    """

    vol_shape: Tuple[int, int, int]  # (D,H,W)
    det_shape: Tuple[int, int]       # (V,U)
    angles: torch.Tensor             # [A], radians
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (sd,sy,sx)
    det_spacing: Tuple[float, float] = (1.0, 1.0)             # (sv,su)
    angle_chunk: int = 16
    n_steps_cap: int = 256

    # ----------------------------
    # Convenience properties
    # ----------------------------
    @property
    def A(self) -> int:
        """Number of projection angles."""
        return int(self.angles.numel())

    @property
    def D(self) -> int:
        """Depth (slices)."""
        return int(self.vol_shape[0])

    @property
    def H(self) -> int:
        """Height (voxels)."""
        return int(self.vol_shape[1])

    @property
    def W(self) -> int:
        """Width (voxels)."""
        return int(self.vol_shape[2])

    @property
    def V(self) -> int:
        """Detector vertical bins."""
        return int(self.det_shape[0])

    @property
    def U(self) -> int:
        """Detector horizontal bins."""
        return int(self.det_shape[1])

    def device(self) -> torch.device:
        """
        Returns:
            torch.device: the device where `angles` tensor is stored.
        """
        return self.angles.device
