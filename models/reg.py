from typing import Optional, Iterable, Tuple
from torch.autograd import Function
from torch import Tensor

import torch
from torch import nn

from torchvision import transforms as T

from utils import AverageMeter

class TensorHook():
    """
    Statefull hook to remember weight accord to the output
    """
    def __init__(self, tensor: torch.Tensor, weight: float=1.):
        self.hook = tensor.register_hook(self)
        self.weight = weight
        self.grad = 0

    def __call__(self, grad: torch.Tensor):
        self.grad += torch.norm(self.weight * grad, p='fro')

    def get_grad(self) -> torch.Tensor:
        return self.grad

    def reset_grad(self) -> None:
        self.grad = 0   

    def __del__(self) -> None:
        self.hook.remove()
        
class GradientHook():
    def __init__(self, outputs: Iterable[Tuple[float, Tensor]]):

        self.hooks = []
        for output_weight, output_tensor in outputs:
            if output_tensor is not None:
                hook = TensorHook(output_tensor, weight=output_weight)
                self.hooks.append(hook)

    def reg_hook_fn(self, grad: torch.Tensor) -> None:
        self.grad += torch.norm(self.lambda_reg * grad, p='fro') 

    def fidelity_hook_fn(self, grad: torch.Tensor) -> None:
        self.grad += torch.norm(grad, p='fro')

    @property
    def grad(self) -> torch.Tensor:
        return sum([hook.get_grad() for hook in self.hooks])

    def get_grad(self) -> torch.Tensor:
        return self.grad

    def reset_grad(self) -> None:
        for hook in self.hooks:
            hook.reset_grad()

def gaussian_similarity(query: Tensor, key: Tensor, h: float=1., sigma: float=0., dim=1):
    """
    query, key (Tensor): tensors of any shape, measures the similarity along the "dim" axis.

    """

    distance = torch.maximum(torch.sum((query - key)**2, dim=dim, keepdim=True) - 2 * sigma**2, torch.tensor(0.)) / h**2

    return torch.exp(-distance)