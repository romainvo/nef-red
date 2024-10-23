from typing import Callable, Any, Dict, Tuple, Union, Optional
import sys

import os
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEmaV2

from utils import AverageMeter, count_parameters, update_summary, \
    save_checkpoint, Logger, parse_args
from models import create_model
from scheduler import create_scheduler, WarmUpWrapper
from data import create_dataloader, create_transforms, create_dataset
from optim import set_bn_weight_decay, create_optimizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

def main(args, args_text):

    global DEVICE 
    DEVICE = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu' 

    model = create_model(**vars(args))

    model_ema = None
    if args.ema:
        model_ema = ModelEmaV2(model, decay=args.ema_decay, device=DEVICE if args.ema_validation else 'cpu')

    checkpoint = None
    if args.checkpoint_file is not None:
        checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Missing keys :", missing_keys)
        print("Unexpected keys :", unexpected_keys)

        if 'state_dict_ema' in checkpoint:
            try:
                model_ema.load_state_dict(checkpoint['state_dict_ema'])
            except Exception as e:
                print("Iterative matching")
                for model_ema_v, ema_v in zip(model_ema.state_dict().values(), 
                                              checkpoint['state_dict_ema'].values()):
                    model_ema_v.copy_(ema_v)

    model = model.to(DEVICE)
    model = model.train()

    # Initialize the current working directory
    output_dir = ''
    output_base = args.output_base
    exp_name = '-'.join([
        datetime.now().strftime('%Y%m%d-%H%M%S'),
        'PostProcessing',
        type(model).__name__
    ])
    if args.output_suffix is not None:
        exp_name = '_'.join([exp_name, args.output_suffix])
    output_dir = os.path.join(output_base, args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train_imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val_imgs'), exist_ok=True)

    args.output_dir = output_dir

    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        f.write(args_text)

    if args.log_name is not None:
        sys.stdout = Logger(os.path.join(output_dir, args.log_name+'.log'))
    else:
        print("****** NO LOG ******")

    print("Number of model parameters : {}\n".format(count_parameters((model))))

    print(args)

    train_dataset = create_dataset(transforms=create_transforms(training=args.augmentation),
                                   training=True,
                                   test=False,
                                   mode='postprocessing',
                                   outputs=['sparse_rc', 'reference_rc'],
                                   **vars(args))

    args.pnp = False
    val_dataset = create_dataset(transforms=create_transforms(training=False),
                                 training=False,
                                 test=False,
                                 mode='postprocessing',
                                 outputs=['sparse_rc', 'reference_rc'],
                                 **vars(args)); args.pnp=True

    train_dataloader = create_dataloader(train_dataset, args.batch_size,
                                         num_workers=args.num_workers,
                                         trainval=True,
                                         shuffle=True,
                                         drop_last=args.drop_last,
                                         pin_memory=args.pin_memory)
    val_dataloader = create_dataloader(val_dataset, args.batch_size,
                                       num_workers=args.num_workers,
                                       trainval=True,
                                       shuffle=False,
                                       drop_last=args.drop_last,
                                       pin_memory=args.pin_memory)  

    loss_fn = torch.nn.MSELoss(reduction='mean')

    loss_fn = loss_fn.to(DEVICE)

    parameters = set_bn_weight_decay(model, weight_decay=0)
    optimizer = create_optimizer(parameters, **vars(args))

    last_epoch, ckpt_epoch = -1, 0
    if checkpoint is not None and args.resume_training:
        optimizer.load_state_dict(checkpoint['optimizer'])
        ckpt_epoch = checkpoint['epoch']
        last_epoch = ckpt_epoch - args.num_warmup_epochs

        eval_metrics = {
            'mse' : checkpoint['metric'],
            'mae' : checkpoint['metric']
        }

    args.num_step = len(train_dataloader) * args.num_epochs
    args.num_warmup_step = len(train_dataloader) * args.num_warmup_epochs
    args.optimizer = optimizer
    lr_scheduler = create_scheduler(**vars(args)
                                    )
    if checkpoint is not None and args.resume_training:
        for group_idx, closed_form_current_lr in enumerate(lr_scheduler._get_closed_form_lr()):
            optimizer.param_groups[group_idx]['lr'] = closed_form_current_lr
    if args.num_warmup_epochs > 0:
        lr_scheduler = WarmUpWrapper(lr_scheduler, args.num_warmup_step, args.warmup_start)
        args.num_epochs += args.num_warmup_epochs

    scaler = torch.cuda.amp.GradScaler(init_scale=2.0**16, enabled=args.amp) #64536 = 2**16

    best_metric = None
    best_epoch = None
    try:
        for epoch in range(ckpt_epoch, args.num_epochs):
                train_metrics = train_epoch(
                    epoch, 
                    model, 
                    train_dataloader, 
                    optimizer, 
                    args, 
                    loss_fn, 
                    lr_scheduler, 
                    model_ema=model_ema,
                    scaler=scaler)

                if epoch % args.validation_interval == 0 or epoch == args.num_epochs - 1  or epoch == ckpt_epoch:
                    eval_metrics = validation(
                        model if not args.ema else (model_ema.module if args.ema_validation else model), 
                        val_dataloader, 
                        args)

                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(args.output_dir, 'summary.csv'),
                    write_header=best_metric is None)  

                # Update the best_metric and best_epoch, only keep one checkpoint at a time
                if best_metric is None:
                    best_metric, best_epoch = eval_metrics[args.eval_metric], epoch
                    save_checkpoint(epoch, 
                                    model, 
                                    optimizer, 
                                    args,
                                    best_metric, 
                                    scheduler=lr_scheduler.state_dict(),
                                    state_dict_ema=model_ema.state_dict() if args.ema else None,
                                    ckpt_name=None)

                elif eval_metrics[args.eval_metric] < best_metric:
                    if os.path.exists(os.path.join(args.output_dir, 'checkpoint-{}.pth.tar'.format(best_epoch))):
                        os.unlink(os.path.join(args.output_dir, 'checkpoint-{}.pth.tar'.format(best_epoch)))

                    best_metric, best_epoch = eval_metrics[args.eval_metric], epoch
                    save_checkpoint(epoch, 
                                    model, 
                                    optimizer, 
                                    args, 
                                    best_metric, 
                                    scheduler=lr_scheduler.state_dict(),
                                    state_dict_ema=model_ema.state_dict() if args.ema else None,
                                    ckpt_name=None)

                if epoch % args.checkpoint_interval == 0 and args.checkpoint_interval != -1:
                    save_checkpoint(epoch, 
                                    model, 
                                    optimizer, 
                                    args, 
                                    best_metric,
                                    scheduler=lr_scheduler.state_dict(),
                                    state_dict_ema=model_ema.state_dict() if args.ema else None,
                                    ckpt_name='hard_checkpoint-{}.pth.tar'.format(epoch))

                save_checkpoint(epoch, 
                                model, 
                                optimizer, 
                                args, 
                                best_metric, 
                                scheduler=lr_scheduler.state_dict(),
                                state_dict_ema=model_ema.state_dict() if args.ema else None,
                                ckpt_name='last.pth.tar')

                if args.num_warmup_epochs > 0:
                    lr_scheduler.step(False)                
                else:
                    lr_scheduler.step()                


                print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
                sys.stdout.flush()
                if epoch > 3 and args.debug:
                    break

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        print('*** Best metric: {0} (epoch {1}) \n'.format(best_metric, best_epoch))
        save_checkpoint(epoch, 
                        model, 
                        optimizer, 
                        args, 
                        best_metric, 
                        state_dict_ema=model_ema.state_dict() if args.ema else None,
                        scheduler=lr_scheduler.state_dict(),
                        ckpt_name='last.pth.tar')

    if args.log_name is not None:
        sys.stdout.close()

    return best_metric

def train_epoch(epoch : int, 
                model : nn.Module, 
                loader : torch.utils.data.DataLoader, 
                optimizer : torch.optim.Optimizer, 
                args : Any, 
                loss_fn : Callable[[torch.Tensor], torch.Tensor], 
                lr_scheduler : Any, 
                model_ema: Optional[nn.Module]=None,
                scaler: Optional[torch.cuda.amp.GradScaler]=None,
                **kwargs) -> Dict[str, Union[float, int]]:

    loss_m = AverageMeter()
    mse_m = AverageMeter()

    model.train()

    num_step = len(loader) * epoch
    for batch_idx, (batch) in enumerate(loader):
        
        noise_level = None
        if len(batch) == 2:
            input, target = batch
            input = input.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
        elif len(batch) == 3:
            input, target, cond = batch
            input = input.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            if args.noise_level > 0.:
                noise_level = cond.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.amp):
            output = model(input, residual_learning=args.residual_learning,
                                  forward_iterates=args.forward_iterates,
                                  noise_level=noise_level)
            
            if args.forward_iterates and args.iterative_model > 0:
                loss = sum([loss_fn(z, target) for z in output]) / args.iterative_model
            else:
                loss = loss_fn(output, target)         


        optimizer.zero_grad(set_to_none=False)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if args.clip_grad_norm > 0 and args.clip_grad_value <= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        elif args.clip_grad_value > 0 and args.clip_grad_norm <= 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
        elif args.clip_grad_value == 0 and args.clip_grad_norm == 0:
            pass
        else:
            raise ValueError("clip_grad_norm and clip_grad_value should be exclusives. Both cannot be > 0 at the same time", args.clip_grad_value, args.clip_grad_norm )

        scaler.step(optimizer)
        scaler.update()

        if args.ema:
            model_ema.update(model)

        with torch.no_grad():
            if args.forward_iterates and args.iterative_model > 0:
                mse = sum([F.mse_loss(z, target, reduction='mean') for z in output]) / args.iterative_model
            else:
                mse = F.mse_loss(output, target, reduction='mean')
  
        loss_m.update(loss.item(), input.size(0))
        mse_m.update(mse.item(), input.size(0))

        lr = optimizer.param_groups[0]['lr']
        
        if args.num_warmup_epochs > 0:
            lr_scheduler.step(warmup=True)   
        num_step += 1

        if args.debug and batch_idx > 5:
            break

    return_metrics = OrderedDict()
    return_metrics.update({'loss' : loss_m.avg})
    return_metrics.update({'mse' : mse_m.avg})
    return_metrics.update({'lr' : lr})

    return return_metrics


def validation(model : nn.Module, 
               loader : torch.utils.data.DataLoader, 
               args : Any) -> Dict[str, Union[float, int]]:

    mse_m = AverageMeter()
    mae_m = AverageMeter()

    model.eval()

    for batch_idx, (batch) in enumerate(loader):

        noise_level = None
        if len(batch) == 2:
            input, target = batch
            input = input.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            
        elif len(batch) == 3:
            input, target, cond = batch
            input = input.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            if args.noise_level > 0.:
                noise_level = cond.to(DEVICE, non_blocking=True)
        
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.amp):
                noise_level=torch.tensor([0.5], device=DEVICE).repeat(input.size(0))
                output = model(input, residual_learning=args.residual_learning,
                                      noise_level=noise_level) # FIXME hard-coded noise level for eval...


                mse = ((output - target) ** 2).mean()
                mae = (output - target).abs().mean() 

        mse_m.update(mse.item(), input.size(0))
        mae_m.update(mae.item(), input.size(0))

        if args.debug and batch_idx > 5:
            break

    return_metrics = OrderedDict()
    return_metrics.update({'mse' : mse_m.avg})
    return_metrics.update({'mae' : mae_m.avg})

    return return_metrics

if __name__ == '__main__':

    args, args_text = parse_args()
    main(args, args_text)