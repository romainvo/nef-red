import torch
import torch.nn as nn
from collections import OrderedDict
import math
        
class RegressionHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, 
                       final_activation='Sigmoid', 
                       dropout: float=0., 
                       **kwargs):
        bias = not kwargs.pop('bias_free', False)
        
        legacy_head = kwargs.pop('legacy_head', False)
        if legacy_head:
            layers = OrderedDict([
                ('conv1', nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=True)),
                ('conv2', nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=True)),
                ('final_conv', nn.Conv2d(in_channels, num_classes, 5, padding=2, bias=True))
            ])  
        else:
            layers = OrderedDict([
                ('dropout', nn.Dropout2d(p=dropout)),
                ('final_conv', nn.Conv2d(in_channels, num_classes, 1, padding=0, bias=bias))
            ])
    # def __init__(self, in_channels, num_classes, final_activation='Sigmoid', **kwargs):
    #     layers = OrderedDict([
    #         ('conv1', nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=True)),
    #         ('conv2', nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=True)),
    #         ('final_conv', nn.Conv2d(in_channels, num_classes, 5, padding=2, bias=True))
    #     ])
        
        super(RegressionHead, self).__init__(layers)

        init_weights(self, final_activation=final_activation)    
        
def init_weights(module: nn.Module, nonlinearity : str = 'relu', final_activation='Sigmoid'):
    for idx, m in enumerate(module.modules()):
        
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
            # Maybe try nn.init.normal_(m.weight, mean=0.0, std=0.01) for regular Conv2d
    
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if idx == len(list(module.modules())) - 1:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    if final_activation == 'Sigmoid':
                        nn.init.constant_(m.bias, - math.log((1 - 0.01) / 0.01))
                    elif final_activation == 'Tanh':
                        nn.init.constant_(m.bias, 0)
                    elif final_activation == 'Softplus':
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                        nn.init.constant_(m.bias, 1)
                    elif final_activation == 'Identity':
                        nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, torch.nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)