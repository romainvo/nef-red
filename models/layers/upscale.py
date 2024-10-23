from typing import Tuple, Optional
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.pre_shuffle = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU()
        
        self.init_weights(scale)

    def forward(self,x):
        x = self.shuffle(self.relu(self.pre_shuffle(x)))
        return x
    
    def init_weights(self, scale):
        """
        Checkerboard artifact free sub-pixel convolution
        https://arxiv.org/abs/1707.02937
        """
        ni,nf,h,w = self.pre_shuffle.weight.shape
        ni2 = int(ni/(scale**2))
        k = nn.init.kaiming_normal_(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale**2)
        k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
        self.pre_shuffle.weight.data.copy_(k)


class UpConv3x3(nn.Module):
    def __init__(self, in_channels, interpolation='bilinear', activation=nn.ReLU):
        super(UpConv3x3, self).__init__()
        assert in_channels % 2 == 0
        
        self.interpolation = interpolation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            activation(),
        )
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode=self.interpolation, align_corners=True)
        x = self.block(x)

        return x

class TransposeConv2d(torch.nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(TransposeConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        height, width = x.size()[-2:]
        return super(TransposeConv2d, self).forward(x, output_size=(2*height, 2*width))

def create_upscaling_layer(in_channels: int, 
                           scale_factor: int, 
                           layer_name: str='upconv', 
                           interpolation: str='bilinear', 
                           normalization_layer: nn.Module=nn.BatchNorm2d,
                           activation: nn.Module=nn.ReLU):

    if layer_name == 'pixelshuffle':
        return PixelShuffle(in_channels, scale_factor)

    elif layer_name == 'transposeconv':
        if scale_factor > 2:
            upscale_blocks = []
            for i in range(scale_factor // 2):
                block = nn.Sequential(*[
                    TransposeConv2d(in_channels, in_channels, 
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1, 
                                    groups=in_channels,
                                    bias=False),
                    normalization_layer(in_channels),
                    activation()
                ])
                upscale_blocks.append(block)
            return nn.Sequential(*upscale_blocks)

        else:
            block = nn.Sequential(*[
                TransposeConv2d(in_channels, in_channels, 
                                kernel_size=3, 
                                stride=2, 
                                padding=1, 
                                groups=in_channels,
                                bias=False),
                normalization_layer(in_channels),
                activation()
            ])
            return block

    elif layer_name == 'transposeconv_nogroup':
        if scale_factor > 2:
            upscale_blocks = []
            for i in range(scale_factor // 2):
                block = nn.Sequential(*[
                    TransposeConv2d(in_channels, in_channels, 
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1, 
                                    bias=False),
                    normalization_layer(in_channels),
                    activation()
                ])
                upscale_blocks.append(block)
            return nn.Sequential(*upscale_blocks)

        else:
            block = nn.Sequential(*[
                TransposeConv2d(in_channels, in_channels, 
                                kernel_size=3, 
                                stride=2, 
                                padding=1, 
                                bias=False),
                normalization_layer(in_channels),
                activation()
            ])
            return block

    elif layer_name == 'upconv':
        if scale_factor > 2:
            upscale_blocks = []
            for i in range(scale_factor // 2):
                upscale_blocks.append(
                    UpConv3x3(in_channels, 
                              interpolation=interpolation,
                              activation=activation))
            return nn.Sequential(*upscale_blocks)

        else:
            return UpConv3x3(in_channels, 
                             interpolation=interpolation,
                             activation=activation)

    elif layer_name == 'interpolation':
        return nn.Upsample(scale_factor=scale_factor, mode=interpolation, align_corners=True)
    
    else:
        raise ValueError(f"Unknown upscaling layer: {layer_name}")
        
