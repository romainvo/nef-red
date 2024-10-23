from typing import Optional, Any, Mapping, Iterable, Union, List
from torch import Tensor

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from collections import OrderedDict
from torch.nn.modules import activation

from .layers import conv_spectral_norm, create_upscaling_layer, create_downscaling_layer
from . import layers
from .layers.normalization import RunningBatchNorm, LayerNorm2d

class EncoderBlock(nn.Module):
    """
    # Parameters
        - in_channels (int): number of channels in input feature map
        - out_channels (int): number of channels in output feature map

    # Keyword arguments:
        - downscaling (bool)=True : False for center block
        - activation (nn.Module)=nn.ReLU: activation function
        - residual (bool)=False : use skip connections or not
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int=3,
                 downscaling: bool=True,
                 downscaling_layer: str='strided_conv',
                 activation: nn.Module=nn.ReLU,
                 normalization_layer: nn.Module=nn.BatchNorm2d,
                 residual: bool=False,
                 residual_scale_factor: float=1.0):
        super(EncoderBlock, self).__init__()

        kernel_size = 3 if downscaling else kernel_size
        padding = kernel_size // 2
        self.residual_scale_factor = residual_scale_factor
        
        self.downscaling = downscaling
        if downscaling:
            self.downscaling_layer = create_downscaling_layer(in_channels=in_channels,
                                                              out_channels=out_channels,
                                                              kernel_size=kernel_size,
                                                              layer_name=downscaling_layer,
                                                              padding=padding,
                                                              normalization_layer=normalization_layer,
                                                              activation=activation)

        self.residual = residual
        if self.residual:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(out_channels if downscaling else in_channels, out_channels, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False))

        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(out_channels if self.downscaling else in_channels, out_channels, 
                      kernel_size=kernel_size, 
                      stride=1, 
                      padding=padding, 
                      bias=False),
            normalization_layer(out_channels),
            activation()
            # activation()
        )
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=kernel_size, 
                               stride=1, 
                               padding=padding, 
                               bias=False)
        self.bn2 = normalization_layer(out_channels)
        self.act = activation()
        # self.act = activation()
        
    def forward(self, x: Tensor):

        if self.downscaling:
            x = self.downscaling_layer(x)

        shortcut = x
        
        x = self.conv_bn1(x)
        
        x = self.conv2(x)
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py#L181
        if self.residual:
            x += self.skip_connection(shortcut)
            x /= self.residual_scale_factor
        x = self.bn2(x)
        x = self.act(x)
        
        return x

class UNetEncoder(nn.Module):
    """
    # Parameters:
        - encoder_channels (list): list of int ordered from highest resolution to lowest
    
    # Keyword arguments:
        - activation (nn.Module): activation function
        - residual (bool)=False : use skip connections or not
    """
    def __init__(self, in_channels: int,
                       encoder_channels: Iterable[int]=[64,128,256,512,1024],
                       stem_size: int=3,
                       activation: nn.Module=nn.ReLU,
                       normalization_layer: nn.Module=nn.BatchNorm2d,
                       downscaling_layer: str='strided_conv',
                       residual: bool=False,
                       block: nn.Module=EncoderBlock,
                       forward_features=True,
                       residual_scale_factor: float=1.0):
        super(UNetEncoder, self).__init__()
        
        self.encoder_channels = encoder_channels 
        self.forward_features = forward_features

        self.stem_block = block(in_channels, encoder_channels[0], 
                                kernel_size=stem_size,
                                activation=activation, 
                                normalization_layer=normalization_layer,
                                downscaling_layer=downscaling_layer,
                                residual=residual,
                                downscaling=False,
                                residual_scale_factor=residual_scale_factor) 
        
        blocks = [block(in_channels, out_channels,
                        activation=activation, 
                        normalization_layer=normalization_layer,
                        downscaling_layer=downscaling_layer,
                        residual=residual,
                        residual_scale_factor=residual_scale_factor) 
                    for in_channels, out_channels
                    in zip(encoder_channels[:-1], encoder_channels[1:])
        ]
        self.encoder_blocks = nn.ModuleList(blocks)

        self.init_weights()
        
    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:

        # ordered from highest resolution to lowest
        x = self.stem_block(x)
        if self.forward_features:
            features_list = [x]
        
        for block in self.encoder_blocks:
            x = block(x)
            if self.forward_features:
                features_list.append(x)

        if self.forward_features:                   
            return features_list
        else:
            return x

    def init_weights(self, nonlinearity : str = 'relu'):
        for name, m in self.named_modules():
            
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
                if 'shuffle' not in name:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DecoderBlock(nn.Module):
    """
    # Parameters
        - in_channels (int): number of channels in input feature map
        - skip_channels (int): number of channels in skip feature map
        - out_channels (int): number of channels in output feature map

    # Keyword arguments:
        - activation (nn.Module)=nn.ReLU: activation function
        - residual (bool)=False : use skip connections or not
        - upscaling_layer=upconv (str): one of ['upconv', 'pixelshuffle', 'interpolation']
        - interpolation='bilinear' (str): interpolation mode in upscaling_layer function
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 scale_skip_connection: bool=True, #encoder-decoder skip connection
                 residual: bool=False,
                 activation: nn.Module=nn.ReLU,
                 normalization_layer: nn.Module=nn.BatchNorm2d,
                 upscaling_layer: str='upconv', 
                 interpolation: str='bilinear',
                 residual_scale_factor: float=1.0):
        super(DecoderBlock, self).__init__()

        self.upscaling_layer = create_upscaling_layer(in_channels, 
                                                      scale_factor=2,
                                                      layer_name=upscaling_layer,
                                                      interpolation=interpolation,
                                                      normalization_layer=normalization_layer,
                                                      activation=activation # this will break older checkpoints if activation is not ReLU
        )

        self.residual_scale_factor = residual_scale_factor
        self.scale_skip_connection = scale_skip_connection
        in_channels = in_channels + skip_channels if self.scale_skip_connection else in_channels

        self.residual = residual
        if self.residual:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False))

        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            normalization_layer(out_channels),
            activation()
            # activation()
        )
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False)
        self.bn2 = normalization_layer(out_channels)
        self.act = activation()
        # self.act = activation()
        
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upscaling_layer(x)
        if self.scale_skip_connection:
            x = torch.cat([x, skip], dim=1)

        shortcut = x

        x = self.conv_bn1(x)
         
        x = self.conv2(x)
        if self.residual:
            x += self.skip_connection(shortcut)
            x /= self.residual_scale_factor
        x = self.bn2(x)
        x = self.act(x)
        
        return x
    
class UNetDecoder(nn.Module):
    """
    # Parameters:
        - encoder_channels (list): list of int ordered from highest resolution to lowest
        - decoder_channels (list): number of channels in decoder path
    
    # Keyword arguments:
        - activation (nn.Module)=nn.ReLU: activation function
        - residual (bool)=False : use skip connections or not
        - upscaling_layer=upconv (str): one of ['upconv', 'pixelshuffle', 'interpolation']
        - interpolation='bilinear' (str): interpolation mode in upscaling_layer function
    """
    def __init__(self, encoder_channels: int, decoder_channels: int, scale_skip_connections: List[bool],
                 upscaling_layer: str='upconv', 
                 interpolation: str='bilinear',
                 activation: nn.Module=nn.ReLU,
                 normalization_layer: nn.Module=nn.BatchNorm2d,
                 residual: bool=False,
                 block: nn.Module=DecoderBlock,
                 residual_scale_factor: float=1.0):
        super(UNetDecoder, self).__init__()
        
        # Reverse list to start loop from lowest resolution block
        # from highest number of channels to lowest
        encoder_channels = encoder_channels[::-1] 

        in_channels_list = [encoder_channels[0]] + list(decoder_channels[:-1])
        
        blocks = [block(in_channels, skip_channels, out_channels,
                        scale_skip_connection=scale_connection,
                        activation=activation,
                        normalization_layer=normalization_layer,
                        residual=residual,
                        upscaling_layer=upscaling_layer, 
                        interpolation=interpolation,
                        residual_scale_factor=residual_scale_factor) 
                    for in_channels, skip_channels, out_channels, scale_connection
                    in zip(in_channels_list, encoder_channels[1:], decoder_channels, scale_skip_connections)
        ]
        self.decoder_blocks = nn.ModuleList(blocks)

        self.init_weights()
        
    # Features ordered from highest resolution to lowest
    def forward(self, features: Iterable[Tensor], 
                      forward_features: bool=False) -> Union[Tensor, List[Tensor]]:

        results = []

        x = features[-1]
        # Reverse list of features to loop from lowest resolution to highest, 
        # and also remove features[-1] (because x = features[-1])
        features = features[-2::-1] 
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, features[i])

            if forward_features:
                results.append(x)
        
        if forward_features:
            return results
                            
        return x

    def init_weights(self, nonlinearity : str = 'relu'):
        for name, m in self.named_modules():
            
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
                if 'shuffle' not in name:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                        
class UNet(nn.Module):
    """
    # Parameters:
        - backbone (str): backbone from timm
    
    # Keyword arguments:
    """
    def __init__(self,
                 in_channels=1, 
                 encoder_channels=[32,32,64,64,128],
                 decoder_channels=[64,64,32,32],
                 scale_skip_connections=[1, 1, 1, 1, 1],
                 residual: bool=True,
                 dropout=0.,
                 activation='ReLU',
                 normalization_layer='LayerNorm',
                 final_activation='Identity', 
                 head_layer='RegressionHead',
                 **kwargs):
        super(UNet, self).__init__()

        self.residual_scale_factor = kwargs.pop('residual_scale_factor', 1.0)
        activation = getattr(nn, activation)
        normalization_bias = not kwargs.pop('normalization_bias_free', False)
        if normalization_layer == 'LayerNorm':
            normalization_layer = LayerNorm2d
            normalization_layer = partial(normalization_layer, bias=normalization_bias)
        elif normalization_layer == 'RunningBatchNorm':
            normalization_layer = RunningBatchNorm
        elif normalization_layer == 'none':
            normalization_layer = lambda x: nn.Identity()
        else:
            normalization_layer = getattr(nn, normalization_layer)
            
        self.input_memory = kwargs.pop('input_memory', False)
        if self.input_memory:
            print("****** INPUT MEMORY ******")
            
        self.noise_level = bool(kwargs.pop('noise_level', 0.0))
        if self.noise_level > 0:
            print("****** NOISE LEVEL ******")
            
        self.additional_input_channels = kwargs.pop('additional_input_channels', 0)
        in_channels = in_channels + self.additional_input_channels

        stem_size = kwargs.pop('stem_size', 3)
        downscaling_layer = kwargs.pop('downscaling_layer', 'strided_conv')
        self.encoder = UNetEncoder(in_channels, encoder_channels,
                                    stem_size=stem_size,
                                    activation=activation,
                                    normalization_layer=normalization_layer,
                                    downscaling_layer=downscaling_layer,
                                    residual=residual,
                                    block=EncoderBlock,
                                    residual_scale_factor=self.residual_scale_factor)
        self.encoder_channels = encoder_channels

        upscaling_layer = kwargs.pop('upscaling_layer', 'upconv')
        interpolation = kwargs.pop('interpolation', 'bilinear')

        scale_skip_connections = [bool(elt) for elt in scale_skip_connections]
        self.decoder = UNetDecoder(self.encoder_channels, 
                                   decoder_channels=decoder_channels,
                                   scale_skip_connections=scale_skip_connections,
                                   upscaling_layer=upscaling_layer, 
                                   interpolation=interpolation,
                                   residual=residual,
                                   activation=activation,
                                   normalization_layer=normalization_layer,
                                   block=DecoderBlock,
                                   residual_scale_factor=self.residual_scale_factor)

        last_channels = decoder_channels[-1]

        head = getattr(layers, head_layer)
        bias_free = kwargs.pop('bias_free', False)
        out_channels = kwargs.pop('out_channels', 1)
        legacy_head = kwargs.pop('legacy_head', False)
        self.head = head(last_channels,
                         num_classes=out_channels,
                         dropout=dropout,
                         inter_channels=last_channels,
                         activation=activation,
                         final_activation=final_activation,
                         bias_free=bias_free,
                         legacy_head=legacy_head) 

        self.final_act = getattr(nn, final_activation)()

        self.spectral_normalization = kwargs.pop('spectral_normalization', False)
        if self.spectral_normalization:
            self.spectral_init()

    def forward(self, x : Tensor, additional_input: Optional[Tensor]=None,
                                  residual_learning: bool=False, 
                                  noise_level: Optional[Tensor]=None,
                                  **kwargs) -> Tensor:


        if self.additional_input_channels:
            if self.noise_level:
                noise_level = torch.einsum('b,bcij->bcij', noise_level, torch.ones_like(x, device=x.device, dtype=x.dtype))
                input = torch.cat([x, noise_level], dim=1)
            elif additional_input is None:
                raise ValueError(f'self.additional_input_channels={self.additional_input_channels} and you forgot to pass argument `additional_input`')
            else:
                input = torch.cat([x, additional_input], dim=1)
        else:
            input = x

        # Features ordered from highest resolution to lowest
        features = self.encoder(input)
        features = self.decoder(features)
                    
        results = self.final_act(self.head(features))
        if residual_learning:
            results = results + x
        
        return results

    def spectral_init(self, sigma=1.0):
        print("****** SPECTRAL NORMALIZATION *******")
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                # P.spectral_norm(m, name='weight', n_power_iterations=1, eps=1e-12)
                conv_spectral_norm(m, sigma=sigma, out_channels=m.out_channels, kernelsize=m.kernel_size[0], padding=m.padding[0])