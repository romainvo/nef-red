from typing import Dict, Union, Optional, Any, Tuple
from torch import Tensor

import torch
import torch.nn as nn

import os

import collections
from collections import OrderedDict
import csv

@torch.jit.script
def linspace(start: Tensor, stop: Tensor, num: int):
	"""
	Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
	Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
	"""
	# create a tensor of 'num' steps from 0 to 1
	steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

	# reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
	# - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
	#   "cannot statically infer the expected size of a list in this contex", hence the code below
	# for i in range(start.ndim):
	# 	steps = steps.unsqueeze(0)

	steps = steps.view(1,-1)

	# the output starts at 'start' and increments until 'stop' in each dimension
	out = start + steps*(stop - start)

	return out

@torch.jit.script
def sampling_points(z0 : int , zn : int, t : Tensor, detgrid_x : Tensor, detgrid_y : Tensor, angles: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

		# sample_points = source_pos + t_clip * source_to_det_vect
		x = 0 + t * detgrid_x
		y = 0 + t * detgrid_y
		z = z0 + t * zn

		z, x = torch.cos(angles) * z - torch.sin(angles) * x, torch.cos(angles) * x + torch.sin(angles) * z

		return x, y, z

def save_checkpoint(epoch : int, model : nn.Module, optimizer : torch.optim.Optimizer, 
                    args : Dict[str, Any], metric : Union[float, int], ckpt_name : Optional[str]=None, **kwargs) -> Dict[str, Any]:
    """ Dump the model and optimizer state_dicts, the ConfigArguments and the current best_metric into a Dict.
    Save and returns the Dict """
    save_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
        'metric': metric,
    }

    for key in kwargs.keys():
        save_state[key] = kwargs[key]
    
    if ckpt_name is None:
        ckpt_name = 'checkpoint-{}.pth.tar'.format(epoch)
        
    torch.save(save_state, os.path.join(args.output_dir, ckpt_name))

    return save_state

def update_summary(step : int, 
                train_metrics : Dict[str, Union[float, int]], 
                eval_metrics : Dict[str, Union[float, int]], 
                filename : str, 
                write_header : bool=False) -> None:
    """ Save train and eval metrics for plotting and reviewing """
    rowd = OrderedDict(step=step)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  
            dw.writeheader()
        dw.writerow(rowd)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_parameters(model : nn.Module) -> int:
    """ Counts the number of learnable parameters of a given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)