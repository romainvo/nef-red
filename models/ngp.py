"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union, Mapping, Any, Optional
from torch import Tensor

from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

class NGPNeuralField(nn.Module):
    """Instance-NGP neural Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        config : Mapping[str, Any],
        n_input_dims: int = 3,
		n_output_dims: int=1,
        output_activation: nn.Module=nn.ReLU(),
        **kwargs: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        
        reconstruction_origin = kwargs.pop('reconstruction_origin', [[0.0, 0.0, 0.0]])
        self.register_buffer("reconstruction_origin", 
                             torch.tensor(reconstruction_origin, dtype=torch.float32)[None])

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.config = config

        self.split_arch = kwargs.pop('split_arch', False)
        if self.split_arch:
            self.encoding = tcnn.Encoding(n_input_dims, config['encoding'])
            self.n_encoding_dims = self.encoding.n_output_dims
            self.mlp = tcnn.Network(self.n_encoding_dims,
                                    n_output_dims,
                                    config['network'])

        else:
            self.network = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=config['encoding'],
                network_config=config['network']
            )
            
        self.output_activation = output_activation

    def normalize_positions(self, x):
        aabb_min, aabb_max = torch.split(self.aabb, self.n_input_dims, dim=-1)
        x = x - self.reconstruction_origin
        x = (x - aabb_min) / (aabb_max - aabb_min)

        return x 

    def get_features(self, positions: torch.Tensor) -> Tensor:
        features = self.encoding(positions.view(-1, self.n_input_dims)).to(positions)

        return features

    def get_attenuation(self, positions: Optional[Tensor]=None, features: Optional[Tensor]=None) -> Tensor:
        
        assert (positions is None) or (features is None), "Either positions or features should be provided."
        
        if features is not None:
            attenuation = self.mlp(features.view(-1, self.encoding.n_input_dims))
        if positions is not None:
            if self.split_arch:
                features = self.encoding(positions.view(-1, self.n_input_dims))
                attenuation = self.mlp(features.view(-1, self.mlp.n_input_dims))
            else:
                attenuation = self.network(positions.view(-1, self.n_input_dims))

        return attenuation

    def forward(
        self,
        positions: torch.Tensor,
        normalize_positions: bool=True
    ) -> Tensor:
        if normalize_positions:
            positions = self.normalize_positions(positions)
            selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)

            density_before_activation = (
                self.get_attenuation(positions)
                .view(-1, self.n_output_dims)
                .to(positions)
            ) * selector[...,None]
        else:
            density_before_activation = (
                self.get_attenuation(positions)
                .view(-1, self.n_output_dims)
                .to(positions)
            )

        return self.output_activation(density_before_activation)
