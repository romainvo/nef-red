from typing import Optional, Callable, Mapping, Union
from torch import Tensor

import torch
import torch.nn as nn

class IterativeModule(nn.Module):
    def __init__(self, module : nn.Module, 
                       n_iter : int=5,
                       jfb: bool=False,
                       **kwargs) -> None:
        super(IterativeModule, self).__init__()

        self._module = module
        self.n_iter = n_iter
        self.jfb = jfb
        if hasattr(module, 'input_memory'):
            self.input_memory = self._module.input_memory 
        else:
            self.input_memory = False

        self.kwargs = kwargs

    def forward_module(self, x: Tensor, 
                             input_memory: Optional[Tensor]=None,
                             **kwargs) -> Tensor:
        output = self._module(x, 
                              additional_input=input_memory,
                              **kwargs)

        return output

    def forward(self, x: Tensor, 
                      forward_iterates: bool=False,
                      **kwargs) -> Tensor:
        outputs = []
        if self.input_memory:
            input_memory = x
        else:
            input_memory = None

        for i in range(self.n_iter):
            with torch.set_grad_enabled((not self.jfb or i >= self.n_iter - 1) and self.training):
                z = self.forward_module(x, input_memory=input_memory,
                                           **kwargs)

                if forward_iterates:
                    outputs.append(z)

        if forward_iterates:
            return outputs
        else:
            return z

    def state_dict(self, *args, **kwargs):
        return self._module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self._module.load_state_dict(*args, **kwargs)