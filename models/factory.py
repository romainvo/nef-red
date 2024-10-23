from torch import Tensor

import math

try:
    import tinycudann as tcnn
except ImportError as e:
    print(e)
    exit()

tcnn_config = {
    'network':{
        "otype": "CutlassMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2    
    },
    'encoding':{
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 16,
        "base_resolution": 16,
        "per_level_scale": 1.5
    }
}

from . import NGPNeuralField

def create_field(n_input_dims,
                 n_output_dims,
                 resolution=None,
                 **kwargs):

    encoding_type = kwargs.pop('encoding_type', 'HashGrid')
    model_type = kwargs.pop('model_type', 'neural_field')

    n_hidden_layers = kwargs.get('n_hidden_layers')
    n_neurons = kwargs.get('n_neurons')

    if model_type == 'ngp':
        if encoding_type == 'HashGrid':
            n_levels, N_min, N_max = kwargs.get('n_levels'), kwargs.get('N_min'), kwargs.get('N_max')
            n_features_per_level = kwargs.get('n_features_per_level')

            tcnn_config['network']['n_neurons'] = n_neurons
            tcnn_config['network']['n_hidden_layers'] = n_hidden_layers
            tcnn_config['encoding']['n_levels'] = n_levels
            tcnn_config['encoding']['base_resolution'] = N_min
            tcnn_config['encoding']['log2_hashmap_size'] = kwargs.get('log2_hashmap_size')
            b = math.exp((math.log(N_max) - math.log(N_min)) / (n_levels - 1))
            print(b)
            tcnn_config['encoding']['per_level_scale'] = b
            tcnn_config['encoding']['n_features_per_level'] = n_features_per_level

            model = NGPNeuralField(
                aabb=kwargs.pop('aabb'),
                config=tcnn_config,
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                bias_free=kwargs.pop('bias_free'),
                **kwargs)


    return model

from .unet import UNet
from .layers import IterativeModule

def create_model(model_name: str,
                 skip_connection: bool=True,
                 **kwargs):
    
    if model_name == 'unet':
        model = UNet(residual=skip_connection, **kwargs)
    
    iterative_model = kwargs.pop('iterative_model', -1)
    jfb = kwargs.pop('jfb', False)
    if iterative_model > 0:
        model = IterativeModule(
            module=model,
            n_iter=iterative_model,
            jfb=jfb,
            **kwargs)
    
    return model 