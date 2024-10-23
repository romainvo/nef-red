from typing import Iterable, Tuple, Optional, Union

import torch
import numpy as np
import math

class ConeBeamSetup():
    def __init__(self, source_to_obj : int,
                       obj_to_det : int,
                       num_proj : int,
                       num_full_proj : int,
                       angular_range : float,
                       num_detectors : Union[int, Tuple[int, int]],
                       num_voxels : int,
                       pixel_size : Optional[float],
                       detector_size : Optional[float]=None,
                       magnitude : Optional[float]=None,
                       voxel_size : Optional[float]=None,
                       **kwargs
                       ):

        self.source_to_obj = source_to_obj
        self.obj_to_det = obj_to_det
        self.num_proj = num_proj
        self.num_full_proj = num_full_proj
        self.angular_range = angular_range
        self.num_detectors = np.array(num_detectors) if isinstance(num_detectors, Iterable) else np.array([num_detectors, num_detectors])
        self.num_voxels = num_voxels

        self.detector_size = pixel_size * self.num_detectors if detector_size is None else detector_size
        self.detector_size = np.array(self.detector_size) if isinstance(self.detector_size, Iterable) else np.array([self.detector_size, self.detector_size])
        self.pixel_size = self.detector_size / self.num_detectors

        self.magnitude = self.source_to_det / self.source_to_obj if magnitude is None else magnitude
        self.reconstruction_radius = 0.5 * (self.detector_size / self.magnitude)

        self.voxel_size = (2 * self.reconstruction_radius / self.num_voxels)[0] if voxel_size is None else voxel_size
        self.reconstruction_radius = 0.5 * self.voxel_size * self.num_voxels 

        self.roll = kwargs.pop('roll', 0)

        # coordinate system
        source_origin = kwargs.pop('source_origin', None)
        if source_origin is None:
            self.source_origin = np.array([
                [0, -self.source_to_obj, 0]
            ], dtype=np.float32)
        else:
            self.source_origin = np.array(
                [source_origin], dtype=np.float32
            )

        detector_origin = kwargs.pop('detector_origin', None)
        if detector_origin is None:
            self.detector_origin = np.array([
                [0, self.obj_to_det, 0]
            ], dtype=np.float32)
        else:
            self.detector_origin = np.array(
                [detector_origin], dtype=np.float32
            )

        reconstruction_origin = kwargs.pop('reconstruction_origin', None)
        if reconstruction_origin is None:
            self.reconstruction_origin = np.array([
                [0, 0, 0]
            ], dtype=np.float32)
        else:
            self.reconstruction_origin = np.array(
                [reconstruction_origin], dtype=np.float32
            )
            
		# set the AABB to the reconstruction_radius cube
        self.scene_aabb = torch.tensor([-self.reconstruction_radius, -self.reconstruction_radius, -self.reconstruction_radius,
								         self.reconstruction_radius,  self.reconstruction_radius,  self.reconstruction_radius], dtype=torch.float32)
        self.render_step_size = (
            (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
            / self.num_voxels
        ).item()

    def setup_projection_area(self) -> None:

        px_unit_size = self.pixel_size[0]

        ## coordinates correspond to the center of each pixel
        detgrid_z, detgrid_x = np.meshgrid(
            np.linspace(-(self.detector_size[0]-px_unit_size)/2, (self.detector_size[0]-px_unit_size)/2, self.num_detectors[0], dtype=np.float64, endpoint=True),
            np.linspace(-(self.detector_size[1]-px_unit_size)/2, (self.detector_size[1]-px_unit_size)/2, self.num_detectors[1], dtype=np.float64, endpoint=True),
            indexing='ij'
        )

        detgrid_x, detgrid_z = math.cos(self.roll) * detgrid_x - math.sin(self.roll) * detgrid_z \
                             , math.sin(self.roll) * detgrid_x + math.cos(self.roll) * detgrid_z

        center_z, center_x = self.detector_origin[0, [2,0]]

        detgrid_z += center_z
        detgrid_x += center_x

        # projection_center_z, projection_center_x = self.source_origin[0, [2,0]]
        # reconstruction_area = (detgrid_z - projection_center_z)**2 + (detgrid_x - projection_center_x)**2 < (self.detector_size[1] / 2)**2
        # detgrid_z_idx, detgrid_x_idx = np.nonzero(reconstruction_area)
        
        detgrid_z_idx, detgrid_x_idx = np.indices(self.num_detectors).reshape(2,-1)
        
        return detgrid_z, detgrid_x, detgrid_z_idx, detgrid_x_idx

    @property
    def source_to_det(self) -> float:
        return self.source_to_obj + self.obj_to_det

    def __call__(self):
        pass

def nd_linspace(start: torch.Tensor, end: torch.Tensor, num: int):
    dt = (end - start) / (num - 1)
    return start.unsqueeze(-1) + dt.unsqueeze(-1) * torch.arange(num, device=start.device, dtype=start.dtype).unsqueeze(0)