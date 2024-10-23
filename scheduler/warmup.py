from typing import Union, Sequence, Any
import torch

class WarmUpWrapper():
    def __init__(self, scheduler : torch.optim.lr_scheduler, num_warmup_step, warmup_start):
        
        self.__dict__['_scheduler'] = scheduler

        self.init_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.num_warmup_step = num_warmup_step
        self.warmup_count = 0
        if scheduler.last_epoch > 0:
            self.warmup_count = self.num_warmup_step
        elif num_warmup_step > 0:
            self.warmup_steps = [(init_lr - warmup_start) / self.num_warmup_step for init_lr in self.init_lrs]
            self.init_groups(warmup_start)

    def step(self, warmup, *args, **kwargs) -> None:
        if self.warmup_count < self.num_warmup_step and warmup:
            for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] += self.warmup_steps[group_idx]
            self.warmup_count += 1
        elif self.num_warmup_step <= self.warmup_count and not warmup:
            self._scheduler.step(*args, **kwargs)

    def init_groups(self, values : Union[Sequence[float], float]) -> None:
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value

    def __getattr__(self, __name):
        return getattr(self._scheduler, __name)
    
    def __setattr__(self, __name, __value):
        if __name not in self.__dict__ and hasattr(self._scheduler, __name):
            object.__setattr__(self._scheduler, __name, __value)
        else:
            object.__setattr__(self, __name, __value)       

# class WarmUpWrapper():
#     def __init__(self, scheduler : torch.optim.lr_scheduler, num_warmup_step, warmup_start):
        
#         self._scheduler = scheduler
#         self.optimizer = scheduler.optimizer

#         self.init_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]

#         self.num_warmup_step = num_warmup_step
#         self.warmup_count = 0
#         if num_warmup_step > 0:
#             self.warmup_steps = [(init_lr - warmup_start) / self.num_warmup_step for init_lr in self.init_lrs]
#             self.init_groups(warmup_start)

#     def step(self, warmup, *args, **kwargs) -> None:
#         if self.warmup_count < self.num_warmup_step:
#             for group_idx, param_group in enumerate(self.optimizer.param_groups):
#                     param_group['lr'] += self.warmup_steps[group_idx]
#             self.warmup_count += 1
#         elif not warmup:
#             self._scheduler.step(*args, **kwargs)

#     def init_groups(self, values : Union[Sequence[float], float]) -> None:
#         if not isinstance(values, (list, tuple)):
#             values = [values] * len(self.optimizer.param_groups)
#         for param_group, value in zip(self.optimizer.param_groups, values):
#             param_group['lr'] = value


class CosineAnnealingLRWithWarmup():
    """Adds a WarmUp option to the regular CosineAnnealingLR built-in PyTorch. 
    The warmup and scheduler arguments are directly accessed through the ConfigArguments object.
    """
    def __init__(self, optimizer : torch.optim.Optimizer, args : Any) -> None:

        self.optimizer = optimizer

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_step, eta_min=args.lr_min, last_epoch=-1, verbose=False)
        self.init_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.num_warmup_step = args.num_warmup_step
        self.warmup_count = 0
        if args.num_warmup_epochs > 0:
            self.warmup_steps = [(init_lr - args.warmup_start) / self.num_warmup_step for init_lr in self.init_lrs]
            self.init_groups(args.warmup_start)

    def step(self) -> None:
        if self.warmup_count < self.num_warmup_step:
            for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] += self.warmup_steps[group_idx]
            self.warmup_count += 1
        else:
            self.lr_scheduler.step()

    def init_groups(self, values : Union[Sequence[float], float]) -> None:
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value