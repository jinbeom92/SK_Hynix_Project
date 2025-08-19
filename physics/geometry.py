# =================================================================================================
# Parallel3DGeometry — Rebindable Geometry Descriptor for Parallel-Beam 3D CT
# -------------------------------------------------------------------------------------------------
# Purpose
#   Serves as the canonical container for acquisition geometry parameters in parallel-beam
#   3D computed tomography. Holds both volume- and detector-domain configuration, including
#   sampling resolutions, voxel/detector spacings, and acquisition angles.
#
# Features
#   • Stores geometric metadata:
#       - vol_shape  : (D, H, W) — voxel grid dimensions
#       - det_shape  : (V, U) — detector plane dimensions
#       - angles     : [A] projection angles (in radians, torch.Tensor on target device)
#       - voxel_size : (sd, sy, sx) physical spacing of voxels
#       - det_spacing: (sv, su) physical spacing of detector elements
#   • Training-time controls:
#       - angle_chunk : max number of angles processed per chunk (memory knob)
#       - n_steps_cap : integration step cap for Joseph projector
#   • Provides convenience properties for geometry dimensions (A,D,H,W,V,U).
#   • Device-awareness: `.device()` returns the device of the angles tensor, ensuring
#     geometry and tensors remain colocated on CPU/GPU as needed.
#
# Usage
#   geom = Parallel3DGeometry(
#       vol_shape=(64,64,64),
#       det_shape=(128,128),
#       angles=torch.linspace(0, torch.pi, 360),
#       voxel_size=(1.0,1.0,1.0),
#       det_spacing=(1.0,1.0),
#       angle_chunk=16,
#       n_steps_cap=256
#   )
#   print(geom.A, geom.D, geom.V)  # convenience accessors
#
# Notes
#   • Geometry can be rebound during training (see projector.reset_geometry).
#   • Acts as the single source of truth for projector/backprojector modules.
# =================================================================================================
from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class Parallel3DGeometry:
    vol_shape: Tuple[int, int, int]  # (D,H,W)
    det_shape: Tuple[int, int]       # (V,U)
    angles: torch.Tensor             # [A], radians
    voxel_size: Tuple[float, float, float] = (1.0,1.0,1.0)  # (sd,sy,sx)
    det_spacing: Tuple[float, float] = (1.0,1.0)            # (sv,su)
    angle_chunk: int = 16
    n_steps_cap: int = 256

    @property
    def A(self): return int(self.angles.numel())
    @property
    def D(self): return int(self.vol_shape[0])
    @property
    def H(self): return int(self.vol_shape[1])
    @property
    def W(self): return int(self.vol_shape[2])
    @property
    def V(self): return int(self.det_shape[0])
    @property
    def U(self): return int(self.det_shape[1])

    def device(self):
        return self.angles.device
