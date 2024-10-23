

from .cork_dataset import CorkRayDataset, CorkDataset
from .walnut_dataset import WalnutRayDataset, WalnutDataset

def create_ray_dataset(dataset_name: str,
                   input_dir: str,
                   input_file: str,
                   num_proj: int,
                   num_rays: int,
                   num_points: int,
                   acquisition_id: int,
                   split_set: str,
                   memmap: bool,
                   training: bool=True,
                   **kwargs):

    if dataset_name == 'cork':
        dataset = CorkRayDataset(
            input_dir=input_dir,
            input_file=input_file,
            num_proj=num_proj,
            training=training,
            num_rays=num_rays,
            num_points=num_points,
            acquisition_id=acquisition_id,
            split_set=split_set,
            memmap=memmap,
            **kwargs
        )
    elif dataset_name == 'walnut':
        dataset = WalnutRayDataset(
            input_dir=input_dir,
            input_file=input_file,
            num_proj=num_proj,
            training=training,
            num_rays=num_rays,
            num_points=num_points,
            acquisition_id=acquisition_id,
            split_set=split_set,
            memmap=memmap,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
  
    return dataset

def create_dataset(dataset_name: str,
                   input_dir: str,
                   input_file: str,
                   patch_size: int,
                   training: bool=True,
                   test: bool=False,
                   transforms=None,
                   outputs=['sparse_rc', 'reference_rc'],
                   **kwargs):

    if dataset_name == 'cork':
        dataset = CorkDataset(input_dir,
                                input_file=input_file,
                                patch_size=patch_size,
                                transforms=transforms,
                                outputs=outputs,
                                training=training,
                                test=test,
                                **kwargs)
    elif dataset_name == 'walnut':
        dataset = WalnutDataset(input_dir,
                                input_file=input_file,
                                patch_size=patch_size,
                                transforms=transforms,
                                outputs=outputs,
                                training=training,
                                test=test,
                                **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
  
    return dataset