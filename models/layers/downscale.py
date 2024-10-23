from typing import Tuple, Optional
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_downscaling_layer(in_channels: int, 
                             out_channels: int,
                             kernel_size: int=3,
                             padding: int=1,
                             layer_name: str='strided_conv',
                             normalization_layer: nn.Module=nn.BatchNorm2d,
                             activation: nn.Module=nn.ReLU):
    
    if layer_name == 'strided_conv':
        return nn.Sequential(
                nn.Conv2d(in_channels,out_channels, 
                        kernel_size=kernel_size, 
                        stride=2, 
                        groups=in_channels,
                        padding=padding, 
                        bias=False),
                normalization_layer(out_channels),
                activation()
            )
    else:
        raise ValueError(f"Unknown downscaling layer: {layer_name}")
