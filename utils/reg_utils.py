from typing import Optional, Iterable, Tuple
from torch import Tensor

import torch
import numpy as np

def sample_random_cube(patch_size: int,
                       patch_z_size: int,
                       resolution: Tuple[int, int, int],
                       device: torch.device,
                       dataset_name: str) -> Tensor:

    res_z, res_y, res_x = resolution
    half_dz, half_dy, half_dx = 0.5 / res_z, 0.5 / res_y, 0.5 / res_x
    dz, dy, dx = 1.0 / res_z, 1.0 / res_y, 1.0 / res_x

    if dataset_name == 'walnut':
        x_offset, y_offset, z_offset = half_dx, half_dy , half_dz
        x_interval, y_interval, z_interval = 1.0 - dx - (patch_size-1)*dx, \
                                             1.0 - dy - (patch_size-1)*dy, \
                                             1.0 - dz - (patch_z_size-1)*dz
    elif dataset_name == 'cork':
        x_offset, y_offset, z_offset = half_dx + 256 * dx, half_dy + 256 * dy, half_dz + 128 * dz
        x_interval, y_interval, z_interval = 0.5 - dx - (patch_size-1)*dx, \
                                             0.5 - dy - (patch_size-1)*dy, \
                                             0.75 - dz - (patch_z_size-1)*dz

    random_x = x_offset + torch.rand(size=(1,)) * x_interval \
             + torch.arange(patch_size) * dx
    random_y = y_offset + torch.rand(size=(1,)) * y_interval \
             + torch.arange(patch_size) * dy

    random_z = z_offset + torch.rand(size=(1,)) * z_interval \
             + torch.arange(patch_z_size) * dz

    yv, xv, zv = torch.meshgrid([random_y, random_x, random_z], indexing='ij')

    xyz = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()]).t().to(device)

    return xyz

def sample_positions(data: np.ndarray,
                     xyz: np.ndarray,
                     resolution: Iterable[int]=(1024,1024,1024),
                     normalized_coords: bool=True,
                     normalization_constant: Optional[float]=None):


    if not normalized_coords:
        xyz = 0.5 + xyz / normalization_constant # normalization_constant = (2*reconstruction_radius)

    xyz = xyz * np.array([resolution[2], resolution[0], resolution[1]]).astype(np.float32)
    indices = xyz.astype(int) # truncates to lower integer
    lerp_weights = xyz - indices.astype(np.float32)

    x0 = indices[:, 0].clip(min=0, max=resolution[2]-1)
    y0 = indices[:, 1].clip(min=0, max=resolution[0]-1)
    z0 = indices[:, 2].clip(min=0, max=resolution[1]-1)
    x1 = (x0 + 1).clip(max=resolution[2]-1)
    y1 = (y0 + 1).clip(max=resolution[0]-1)
    z1 = (z0 + 1).clip(max=resolution[1]-1)

    c00 = data[y0,z0,x0] * (1.0 - lerp_weights[:,0]) \
        + data[y0,z0,x1] * lerp_weights[:,0]
    c01 = data[y0,z1,x0] * (1.0 - lerp_weights[:,0]) \
        + data[y1,z0,x0] * lerp_weights[:,0]
    c10 = data[y1,z0,x0] * (1.0 - lerp_weights[:,0]) \
        + data[y1,z0,x1] * lerp_weights[:,0]
    c11 = data[y1,z1,x0] * (1.0 - lerp_weights[:,0]) \
        + data[y1,z1,x1] * lerp_weights[:,0]

    c0 = c00 * (1.0 - lerp_weights[:,1]) \
    + c10 * lerp_weights[:,1]
    c1 = c01 * (1.0 - lerp_weights[:,1]) \
    + c11 * lerp_weights[:,1]

    c = c0 * (1.0 - lerp_weights[:,2]) + c1 * lerp_weights[:,2]

    return c