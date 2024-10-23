from torch import Tensor

from torch import nn
from torchvision.ops import stochastic_depth

class StochasticDepth(nn.Module):
    def __init__(self, p: float=0.1, mode='row'):
        super(StochasticDepth, self).__init__()
        
        self.p = p
        self.mode = mode
        
    def forward(self, x: Tensor) -> Tensor:
        return stochastic_depth(x, self.p, self.mode, training=self.training)