from typing import Optional, Mapping, Tuple, Union, Any, List
from torch import Tensor

from torch import nn
import collections

import torch.utils.data as data
import torch

import numpy as np
import pandas as pd
from pathlib import Path
import math

from tqdm import tqdm

from tomography.geometry import ConeBeamSetup
from tomography import Rays

class CorkRayDataset(data.Dataset):
    def __init__(self, input_dir : str,
                       input_file : str,
                       num_proj : int,
                       training : bool=True,
                       test : bool=False,
                       geometry : Optional[Any]=None,
                       **kwargs):
        super(CorkRayDataset, self).__init__()

        split_set = kwargs.pop('split_set', 'validation')

        self.input_dir = Path(input_dir)
        self.df = pd.read_csv(self.input_dir / input_file)
        self.df = self.df.loc[self.df.split_set == split_set]

        if geometry is None:
            self.geometry = ConeBeamSetup(source_to_obj=190,
                                          obj_to_det=447,
                                          num_proj=num_proj,
                                          num_full_proj=720,
                                          angular_range=2*math.pi,
                                          num_detectors=1024,
                                          num_voxels=1024,
                                          pixel_size=0.2,
                                          magnitude=3.27)
        else:
            self.geometry = geometry
        self.detgrid_z, self.detgrid_x, self.detgrid_z_idx, self.detgrid_x_idx = self.geometry.setup_projection_area()

        self.num_rays = kwargs.pop('num_rays', 4096)
        self.num_points = kwargs.pop('num_points', 1024)

        self.batch_over_images = kwargs.pop('batch_over_views', True)

        memmap = kwargs.pop('memmap', True)
        self.create_memmap(memmap=memmap)
        
        self.angles = np.linspace(0, 2*np.pi, num=self.num_full_proj, endpoint=False, dtype=np.float32)[::-1]
        self.angle_indexes = np.linspace(0, self.num_full_proj, num=num_proj, endpoint=False, dtype=int)

        full_index = np.arange(720)
        shuffled_indexes = np.concatenate([full_index[180:], full_index[:180]])
        self.view_indexes = shuffled_indexes[np.linspace(0, 720, num_proj, endpoint=False, dtype=int)]
        
        self.resolution = torch.Size([self.num_voxels,self.num_voxels,self.num_voxels])
        self.slice_shape = torch.Size([self.num_voxels,self.num_voxels,1])

        self.training = training
        self.test = test

        self.field_type = kwargs.pop('field_type', 'attenuation') # 'attenuation' of 'integration'
        self.acquisition_id = kwargs.pop('acquisition_id', 'F39')

    def __getitem__(self, index: int)  -> Mapping[str, 
                                                  Tensor]:

        data = None
        if self.field_type == 'attenuation':
            data = self.fetch_data(index, acquisition_id=self.acquisition_id)
        elif self.field_type == 'integration':
            data = self.fetch_integration_data(index)

        return data

    def __len__(self):
        return 50

    def create_memmap(self, memmap: bool=True) -> None:

        self.radiographies = dict()
        self.reference_rcs = dict()
        print("Initializing memory-mapping for each volume....", end='\n')
        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            sample_id = row.id

            shape = (row.num_full_proj, row.detector_size, row.detector_size) if row.sinogram_format == 'DMM' else (row.detector_size, row.detector_size, row.num_full_proj)

            if memmap:
                self.reference_rcs[sample_id] \
                = np.memmap(self.input_dir / row.reconstruction_file,
                            dtype='float32',
                            mode='r',
                            shape=(row.detector_size, row.detector_size, row.detector_size))
                self.radiographies[row.id] \
                = np.memmap(self.input_dir / row.sinogram_file,
                            dtype='float32',
                            mode='r',
                            shape=shape)

            else:
                with (self.input_dir / row.reconstruction_file).open('rb') as file_in:
                    self.reference_rcs[sample_id] = np.fromfile(file_in, dtype='float32').reshape(row.detector_size, row.detector_size, row.detector_size)

                with (self.input_dir / row.sinogram_file).open('rb') as file_in:
                    self.radiographies[sample_id] = np.fromfile(file_in, dtype='float32').reshape(shape)

    def sample_det_pixels(self):
        indexes = np.random.randint(0, len(self.detgrid_z_idx), size=(self.num_rays,))

        return self.detgrid_z_idx[indexes], self.detgrid_x_idx[indexes]

    def ray_directions(self, z_indexes : np.ndarray, x_indexes : np.ndarray):

        z_target = self.detgrid_z[z_indexes, x_indexes] - self.source_origin[0,2]
        x_target = self.detgrid_x[z_indexes, x_indexes] - self.source_origin[0,0]

        y_target = np.zeros_like(z_target) + self.detector_origin[0,1] - self.source_origin[0,1]

        return np.vstack([x_target, y_target, z_target]).T

    def ray_positions(self, z_indexes : np.ndarray, x_indexes : np.ndarray):

        z_target = self.detgrid_z[z_indexes, x_indexes] - self.source_origin[0,2]
        x_target = self.detgrid_x[z_indexes, x_indexes] - self.source_origin[0,0]

        return np.vstack([x_target, z_target]).T

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    @torch.no_grad()
    def apply_angles(self, ray_origins : Tensor, ray_directions : Tensor, angles : Tensor) -> Tuple[Tensor, Tensor]:
        angles = angles.view(-1)

        if ray_origins.size(1) == 3:
            ray_origins[...,0], ray_origins[...,1] \
                = torch.cos(angles) * ray_origins[...,0] - torch.sin(angles) * ray_origins[...,1], \
                torch.sin(angles) * ray_origins[...,0] + torch.cos(angles) * ray_origins[...,1]

            ray_directions[...,0], ray_directions[...,1] \
                = torch.cos(angles) * ray_directions[...,0] - torch.sin(angles) * ray_directions[...,1], \
                torch.sin(angles) * ray_directions[...,0] + torch.cos(angles) * ray_directions[...,1]

        elif ray_origins.size(1) == 2:
            ray_origins[...,1], ray_origins[...,0] \
                = torch.cos(angles) * ray_origins[...,1] - torch.sin(angles) * ray_origins[...,0], \
                torch.cos(angles) * ray_origins[...,0] + torch.sin(angles) * ray_origins[...,1]

            ray_directions[...,1], ray_directions[...,0] \
                = torch.cos(angles) * ray_directions[...,1] - torch.sin(angles) * ray_directions[...,0], \
                torch.cos(angles) * ray_directions[...,0] + torch.sin(angles) * ray_directions[...,1]

        return ray_origins, ray_directions

    def fetch_data(self, index : int, acquisition_id : str = 'F39') -> Mapping[str, 
                                                                               Tensor]:
        if self.training:
            if self.batch_over_images:
                view_ids = np.random.randint(0, self.num_proj, size=(self.num_rays,))
                z_indexes, x_indexes = self.sample_det_pixels()
                ray_origins = torch.tensor(self.source_origin, dtype=torch.float32).repeat(self.num_rays, 1) # [num_rays, 3]
            else:
                raise NotImplementedError()
        else:
            view_ids = [index]
            y_indexes, x_indexes = torch.meshgrid(
                torch.arange(self.num_detectors),
                torch.arange(self.num_detectors),
                indexing="ij",
            )
            x_indexes = x_indexes.flatten()
            z_indexes = z_indexes.flatten()   
            ray_origins = torch.tensor(self.source_origin, dtype=torch.float32).repeat(self.num_detectors[0], self.num_detectors[1], 1) # [num_rays, 3]   

        # views and angles are offseted view[0] = angle[180]
        angle_indexes = self.angle_indexes[view_ids]
        view_indexes = self.view_indexes[view_ids]

        ray_integrations = np.copy(self.radiographies[acquisition_id][z_indexes, x_indexes, view_indexes])
        ray_integrations = torch.tensor(ray_integrations, dtype=torch.float32).view(-1,1) # [num_rays, 1]

        ray_directions = torch.tensor(self.ray_directions(z_indexes, x_indexes), dtype=torch.float32) # [num_rays, 3]
        angles = torch.tensor(self.angles[angle_indexes], dtype=torch.float32).view(-1,1) # [num_rays, 1]

        if self.training:
            ray_origins = torch.reshape(ray_origins, (self.num_rays, 3))
            ray_integrations = torch.reshape(ray_integrations, (self.num_rays, 1))
            ray_directions = torch.reshape(ray_directions, (self.num_rays, 3))
            angles = torch.reshape(angles, (self.num_rays, 1))

        else:
            ray_origins = torch.reshape(ray_origins, (self.num_detectors[0], self.num_detectors[1], 3))
            ray_integrations = torch.reshape(ray_integrations, (self.num_detectors[0], self.num_detectors[1], 1))
            ray_directions = torch.reshape(ray_directions, (self.num_detectors[0], self.num_detectors[1], 3))
            angles = angles.view(-1,1)
            
        ray_lengths = torch.linalg.norm(ray_directions, ord=2, dim=-1, keepdim=True)
        ray_directions = ray_directions / ray_lengths
        ray_origins, ray_directions = self.apply_angles(ray_origins, ray_directions, angles)
        
        rays = Rays(origins=ray_origins, directions=ray_directions)
            
        return {
            "ray_integrations": ray_integrations,
            "rays": rays,
            "ray_lengths": ray_lengths,
        }

    def fetch_integration_data(self, index : int, acquisition_id : str = 'F39') -> Mapping[str,
                                                                                           Tensor]:
        if self.training:
            if self.batch_over_images:
                view_ids = np.random.randint(
                    0,
                    self.num_proj,
                    size=(self.num_rays,)
                )
                y_indexes, x_indexes = torch.randint(0, self.num_detectors, (self.num_rays, 2)).unbind(1)
            else:
                raise NotImplementedError()
        else:
            view_ids = [index]
            y_indexes, x_indexes = torch.meshgrid(
                torch.arange(self.num_detectors),
                torch.arange(self.num_detectors),
                indexing="ij",
            )
            x_indexes = x_indexes.flatten()
            y_indexes = y_indexes.flatten()   

        # views and angles are offseted view[0] = angle[180]
        angle_indexes = self.angle_indexes[view_ids]
        view_indexes = self.view_indexes[view_ids]

        ray_integrations = self.radiographies[acquisition_id][y_indexes, x_indexes, view_indexes]
        ray_integrations = torch.tensor(ray_integrations, dtype=torch.float32).view(-1,1) # [num_rays, 1]

        ray_positions = torch.tensor(self.ray_positions(y_indexes, x_indexes), dtype=torch.float32) # [num_rays, 2]
        angles = torch.tensor(self.angles[angle_indexes], dtype=torch.float32).view(-1,1) # [num_rays, 1]

        if self.training:
            ray_integrations = torch.reshape(ray_integrations, (self.num_rays, 1))
            ray_positions = torch.reshape(ray_positions, (self.num_rays, 2))
            angles = torch.reshape(angles, (self.num_rays, 1))

        else:
            ray_integrations = torch.reshape(ray_integrations, (self.num_detectors, self.num_detectors, 1))
            ray_positions = torch.reshape(ray_positions, (self.num_detectors, self.num_detectors, 2))
            angles = angles.view(-1,1)
            
        return {
            "ray_integrations": ray_integrations,
            "ray_positions": ray_positions,
            "angles": angles,
        }

    def __getattr__(self, __name):
        return getattr(self.geometry, __name)

class CorkDataset(data.Dataset):
    def __init__(self, input_dir, 
                       input_file='dataset.csv', 
                       patch_size=256,
                       final_activation='ReLU',
                       transforms=None, 
                       outputs=['sparse_rc', 'reference_rc'],
                       training=True,
                       test=False,
                       **kwargs):
        super(CorkDataset, self).__init__()

        self.input_dir = Path(input_dir)
        self.patch_size = patch_size

        self.df = pd.read_csv(self.input_dir / input_file)

        self.final_activation = final_activation
        self.transforms = transforms
        self.training = training
        self.test = test

        self.outputs = outputs
        print(self.df)
        if 'split_set' in self.df:
            if self.training:
                self.df = self.df.loc[self.df.split_set == 'train']
            elif not self.test:
                self.df = self.df.loc[self.df.split_set == 'validation']
            else:
                self.df = self.df.loc[self.df.split_set == 'test']

        print(self.df)
        self.num_voxels = self.df.iloc[0].detector_size

        # Remove upper and bottom slice which contains weird non-dense material
        axial_center_crop = kwargs.pop('axial_center_crop', False)

        self.sample_list = []
        for row in self.df.itertuples():
            if axial_center_crop:
                for slice_index in range(150 , 1024-150):
                    self.sample_list += [(row.id, slice_index)]
            else:
                for slice_index in range(1024):
                    self.sample_list += [(row.id, slice_index)]

        memmap = kwargs.pop('memmap', True)
        self.create_memmap(memmap)

        self.dataset_size = kwargs.pop('dataset_size', 3200)
        self.sample_indexes = self.random_sampler()

        self.center_crop = kwargs.pop('center_crop', False)

        self.pnp = kwargs.pop('pnp', False)
        self.pnp_sigma = kwargs.pop('pnp_sigma', 10.)
        self.fixed_sigma = kwargs.pop('fixed_sigma', False)
        self.noise_level = kwargs.pop('noise_level', 0.0) > 0.

    def random_sampler(self):
        rng = np.random.default_rng()
        while True:
            shuffled_indexes = rng.permutation(list(range(len(self.sample_list))))
            for idx in shuffled_indexes:
                yield idx

    def create_memmap(self, memmap: bool = True):
        self.normalization_factors = dict()
        self.sparse_rcs = dict()
        self.reference_rcs = dict()
        print("Initializing memory-mapping for each volume....", end='\n')
        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            sample_id = row.id

            if memmap:
                self.reference_rcs[sample_id] \
                = np.memmap(self.input_dir / row.reconstruction_file, 
                            dtype='float32', 
                            mode='r', 
                            shape=(self.num_voxels, self.num_voxels, self.num_voxels))

                self.sparse_rcs[sample_id] \
                = np.memmap(self.input_dir / row.sparse_reconstruction_file,
                            dtype='float32', 
                            mode='r', 
                            shape=(self.num_voxels, self.num_voxels, self.num_voxels))

            else:
                with (self.input_dir / row.reconstruction_file).open('rb') as file_in:
                    self.reference_rcs[sample_id] = np.fromfile(file_in, dtype='float32').reshape(self.num_voxels, self.num_voxels, self.num_voxels)

                with (self.input_dir / row.sparse_reconstruction_file).open('rb') as file_in:
                    self.sparse_rcs[sample_id] = np.fromfile(file_in, dtype='float32').reshape(self.num_voxels, self.num_voxels, self.num_voxels)                

    def __getitem__(self, index):
        index = next(self.sample_indexes)
        row_id, slice_index = self.sample_list[index]
        row = self.df.loc[self.df.id == row_id]
        row = row.iloc[0]

        if self.center_crop:
            if row.detector_size - self.patch_size - 256 == 256:
                h_offset, w_offset = 256, 256
            else:
                h_offset = np.random.randint(256, row.detector_size - self.patch_size - 256)
                w_offset = np.random.randint(256, row.detector_size - self.patch_size - 256)
        else:
            h_offset = np.random.randint(row.detector_size - self.patch_size)
            w_offset = np.random.randint(row.detector_size - self.patch_size)

        reference_slice = self.reference_rcs[row_id][slice_index, 
                                                     h_offset:h_offset+self.patch_size,
                                                     w_offset:w_offset+self.patch_size]
        reference_slice = np.copy(reference_slice)

        if self.pnp:
            L_max = 0.1179
            sigma = (L_max * self.pnp_sigma / 255) 
            if not self.fixed_sigma: sigma = sigma * np.random.random()
            noise = sigma * np.random.randn(*reference_slice.shape).astype(np.float32)    
            sparse_slice = reference_slice + noise
            sigma = torch.tensor(sigma, dtype=torch.float32)
        else:
            sparse_slice = self.sparse_rcs[row_id][slice_index, 
                                                h_offset:h_offset+self.patch_size,
                                                w_offset:w_offset+self.patch_size]
            sparse_slice = np.copy(sparse_slice)

        reference_slice = np.clip(reference_slice, 0., None)
        if not self.pnp:
            sparse_slice = np.clip(sparse_slice, 0., None)

        assert (reference_slice.dtype is np.dtype('float32')), \
            print(reference_slice.dtype, reference_slice.max(), row)
        assert (sparse_slice.dtype is np.dtype('float32')), \
            print(sparse_slice.dtype, sparse_slice.max(), row)

        #######################################################################

        inputs = dict(sparse_rc=sparse_slice, reference_rc=reference_slice)

        results = self.transforms(**inputs)

        for key in self.outputs:
            if key in self.df:
                results[key] = row[key]
        
        if self.noise_level:
            return [results[key] for key in self.outputs] + [sigma]
        return [results[key] for key in self.outputs]    

    def __len__(self):

        return self.dataset_size    