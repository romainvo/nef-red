from datetime import datetime
from pathlib import Path
from tqdm import trange
import pandas as pd

import torch
import numpy as np
import math

from collections import OrderedDict
from PIL import Image
from matplotlib import cm

import kernelkit as kk
from kernelkit.torch_support import XrayForwardProjection, XrayBackprojection

from models import create_model
from tomography.geometry import ConeBeamSetup

from utils import *

def build_kk_geometry(geometry: ConeBeamSetup,
                      dataset_name):
    
    roll = geometry.roll
    num_detectors = geometry.num_detectors
    pixel_size = geometry.pixel_size
    angular_range = geometry.angular_range
    num_proj = geometry.num_proj
    num_full_proj = geometry.num_full_proj
    num_voxels = geometry.num_voxels
    reconstruction_radius = geometry.reconstruction_radius
    
    if dataset_name == 'walnut':
        source_origin = geometry.source_origin.squeeze()[[2,1,0]]
        detector_origin = geometry.detector_origin.squeeze()[[2,1,0]]
    else:
        source_origin = geometry.source_origin.squeeze()
        detector_origin = geometry.detector_origin.squeeze()
    
    geom_t0 = kk.ProjectionGeometry(
        source_position=source_origin,
        detector_position=detector_origin,
        u=[-math.sin(-roll), 0, math.cos(-roll)],               
        v=[ math.cos(-roll), 0, math.sin(-roll)],
        detector=kk.Detector(rows=num_detectors[0], cols=num_detectors[1], 
                             pixel_height=pixel_size[0], pixel_width=pixel_size[1]),
    )
    angles = np.linspace(0, angular_range, num=num_full_proj, endpoint=False).squeeze()
    
    if dataset_name == 'cork':
        angles = np.hstack([angles[-180:], angles[:-180]])
    elif dataset_name == 'walnut':
        angles = angles[::-1]
        
    sparse_indexes = np.linspace(0, num_full_proj-1, num=num_proj, dtype=int)
    proj_geoms = [kk.rotate(geom_t0, roll=a) for a in angles[sparse_indexes]]

    # cube with random voxels
    vol_geom = kk.resolve_volume_geometry(
        shape=[num_voxels,num_voxels,num_voxels], 
        extent_min=[-reconstruction_radius] * 3, 
        extent_max=[reconstruction_radius] * 3)
    
    return sparse_indexes, proj_geoms, vol_geom

def main(args, args_text):
    
    global DEVICE 
    DEVICE = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu' 

    if args.regularization_mode == 'postp':
        checkpoint = torch.load(args.reg_checkpoint, map_location='cpu')

        postp_args = checkpoint['args']

        if args.reg_n_iter > 1 or postp_args.iterative_model > 0: # allows to wrap non-iterative model in iterative wrapper during call to factory create_model
            postp_args.iterative_model = args.reg_n_iter
            model = create_model(**vars(postp_args))
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.n_iter = args.reg_n_iter
        else:
            model = create_model(**vars(postp_args))
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)	
        print("Unexpected keys :", unexpected_keys)
        print("Missing keys :", missing_keys)
        print("Number of model parameters : {}\n".format(count_parameters((model))))

        assert len(missing_keys) == 0

        if 'state_dict_ema' in checkpoint:
            if checkpoint['state_dict_ema'] is not None:
                try:
                    print('*** New EMA loading ckpt ***')
                    model.load_state_dict(checkpoint['state_dict_ema'], strict=True)
                except Exception as e:
                    print('*** Legacy EMA loading ckpt ***')

                    for model_v, ema_v in zip(model.state_dict().values(), 
                                            checkpoint['state_dict_ema'].values()):
                        model_v.copy_(ema_v)

        model = model.to(DEVICE)
        model = model.eval()

    # Initialize the current working directory
    output_dir = ''
    output_base = args.output_base
    exp_name = '-'.join([
        datetime.now().strftime('%Y%m%d-%H%M%S'),
        'APGD-RED' if args.red_denoising else 'APGD',
    ])
    if args.output_suffix is not None:
        exp_name = '_'.join([exp_name, args.output_suffix])
    output_dir = os.path.join(output_base, args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'imgs'), exist_ok=True)

    args.output_dir = output_dir

    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        f.write(args_text)
        
    if args.dataset_name == 'cork':
        num_proj = args.num_proj

        geometry = ConeBeamSetup(source_to_obj=190,
                                 obj_to_det=447,
                                 num_proj=num_proj,
                                 num_full_proj=720,
                                 angular_range=2*math.pi,
                                 num_detectors=1024,
                                 num_voxels=1024,
                                 pixel_size=0.2,
                                 magnitude=3.27)
        
        sparse_indexes, proj_geoms, vol_geom = build_kk_geometry(geometry, args.dataset_name)

        num_full_proj = geometry.num_full_proj
        num_detectors = geometry.num_detectors
        num_voxels = geometry.num_voxels

        df = pd.read_csv(input_dir / args.input_file)
        df = df.loc[df.split_set == args.split_set]
        row = df.loc[df.id == args.acquisition_id].iloc[0]
        
        sinogram = np.memmap(input_dir / row.sinogram_file, 
                            shape=tuple(num_detectors) + (num_full_proj,),
                            dtype=np.float32, mode='r')
        sparse_sinogram = np.moveaxis(sinogram, -1 ,0)[sparse_indexes]
        
        reference_rc = np.memmap(input_dir / row.reconstruction_file,
                                 shape=(num_voxels, num_voxels, num_voxels),
                                 dtype=np.float32, mode='r')
        slice_index = 300
        convergence_indexes = np.linspace(start=1024//5,
                                            stop=4*1024//5,
                                            num=1024//10, 
                                            dtype=int, endpoint=True)
        
    elif args.dataset_name == 'walnut':
        df = pd.read_csv(input_dir / args.input_file)
        df = df.loc[df.split_set == args.split_set]
        args.acquisition_id = int(args.acquisition_id)
        row = df.loc[df.id == args.acquisition_id].iloc[0]
        trajectory = np.load(input_dir  / row.trajectory_file)

        source_origin = (trajectory[0,0], trajectory[0,1], trajectory[0,2]) 
        detector_origin = (trajectory[0,3], trajectory[0,4], trajectory[0,5])
        pitch = math.asin(trajectory[0,8] / 0.1496)

        source_to_obj = -trajectory[0,1] # 66 mm 
        obj_to_det = trajectory[0,4] # 133 mm
        num_proj = args.num_proj

        geometry = ConeBeamSetup(source_to_obj=source_to_obj,
                                 obj_to_det=obj_to_det,
                                 num_proj=num_proj,
                                 num_full_proj=1200,
                                 angular_range=2*math.pi,
                                 num_detectors=(972,768),
                                 num_voxels=501,
                                 voxel_size=0.1,
                                 pixel_size=0.1496,
                                 magnitude=3.016,
                                 roll=pitch, 
                                 source_origin=source_origin,
                                 detector_origin=detector_origin)

        sparse_indexes, proj_geoms, vol_geom = build_kk_geometry(geometry, args.dataset_name)
        
        num_full_proj = geometry.num_full_proj
        num_detectors = geometry.num_detectors
        num_voxels = geometry.num_voxels
        
        sinogram = np.memmap(input_dir / row.sinogram_file, 
                            shape=(num_full_proj,) + tuple(num_detectors),
                            dtype=np.float32, mode='r')
        sparse_sinogram = sinogram[sparse_indexes]
        
        reference_rc = np.memmap(input_dir / row.reconstruction_file,
                                 shape=(num_voxels, num_voxels, num_voxels),
                                 dtype=np.float32, mode='r')
        slice_index = 250
        convergence_indexes = np.linspace(start=501//5,
                                            stop=4*501//5,
                                            num=501//10, 
                                            dtype=int, endpoint=True)

    ray_trafo = XrayForwardProjection(
        projection_geometry=proj_geoms,
        volume_geometry=vol_geom,
        # projs_per_block=1
    )

    backprojection = XrayBackprojection(
        projection_geometry=proj_geoms,
        volume_geometry=vol_geom,
        bp_kwargs={'projs_per_block': 1, 'voxels_per_block': (4,8,16)}
    )
    
    lambda_reg = args.lambda_reg 
    reg_batch_size = args.reg_batch_size
    reg_patch_size = args.reg_patch_size
    step_size = args.init_lr
    
    with torch.no_grad():
        sinogram_tensor = torch.tensor(sparse_sinogram, device='cuda')
        output = x_prev = torch.zeros(1,1,num_voxels,num_voxels,num_voxels, device='cuda')  
      
    p_bar = trange(args.n_steps)
    t_prev = 1
    txk, xk, tyk, yk = None, None, None, None
    with torch.no_grad():
        for step in p_bar:
            
            metrics = OrderedDict()
            
            # setup stochastic gradient descent
            proj_indexes = list(np.random.choice(len(sinogram_tensor), size=1, replace=False))
            ray_trafo.op.projector.projection_geometry = [proj_geoms[i] for i in proj_indexes]
            ray_trafo.op.backprojector.projection_geometry = [proj_geoms[i] for i in proj_indexes]

            backprojection.op.projector.projection_geometry = [proj_geoms[i] for i in proj_indexes]
            backprojection.op.backprojector.projection_geometry = [proj_geoms[i] for i in proj_indexes]
            
            residual = ray_trafo(output) - sinogram_tensor[proj_indexes][None,None]
            if args.red_denoising and step >= args.reg_start:
                # sample random cube and do coordinate denoising
                if args.dataset_name == 'cork':
                    z_offset = 100 + np.random.randint(num_voxels - reg_batch_size - 100*2) 
                    y_offset = 256 + np.random.randint(num_voxels - reg_patch_size - 256*2) if reg_patch_size > 256*2 else 256
                    x_offset = 256 + np.random.randint(num_voxels - reg_patch_size - 256*2) if reg_patch_size > 256*2 else 256
                elif args.dataset_name == 'walnut':
                    z_offset = np.random.randint(num_voxels - reg_batch_size)
                    y_offset = np.random.randint(num_voxels - reg_patch_size)
                    x_offset = np.random.randint(num_voxels - reg_patch_size)
                    
                patch = output[0,0,z_offset:z_offset+reg_batch_size, y_offset:y_offset+reg_patch_size, x_offset:x_offset+reg_patch_size]
                patch = patch.reshape(args.reg_batch_size,
                                      1,
                                      args.reg_patch_size,
                                      args.reg_patch_size,
                                      )
                
                denoised_patch = model(patch.clamp(0, None), residual_learning=True)
                
                denoising_residual = (patch - denoised_patch).squeeze()
                output[0,0,z_offset:z_offset+reg_batch_size, y_offset:y_offset+reg_patch_size, x_offset:x_offset+reg_patch_size] -= step_size * lambda_reg * denoising_residual
            
            output[:] = output - step_size * backprojection(residual)
            x_next = output.clamp(0, None)
            
            t_next = (1 + math.sqrt(4*t_prev**2 + 1)) / 2.
            lmd = 1 + (t_prev - 1) / t_next
            output[:] = x_prev + lmd * (x_next - x_prev)            
            x_prev[:] = x_next

            ### Metrics
            data_fidelity = torch.sum(residual**2).item()
            regularization = torch.sum(denoising_residual**2).item() if args.red_denoising and step >= args.reg_start else 0
            gradient = torch.norm(residual, p='fro').item()
            if args.red_denoising and step >= args.reg_start:
                gradient += args.lambda_reg * torch.norm(denoising_residual, p='fro').item()

            ### Nonexpansiveness check
            nonexp_constant = 0.
            if args.red_denoising and step >= args.reg_start:
                if xk is None:
                    xk = patch.numpy(force=True).reshape(-1)
                    txk = denoised_patch.numpy(force=True).reshape(-1)
                else:
                    yk = patch.numpy(force=True).reshape(-1)
                    tyk = denoised_patch.numpy(force=True).reshape(-1)

                    nonexp_constant = np.linalg.norm(tyk - txk) / np.linalg.norm(yk - xk)
                    xk, txk = yk, tyk

            ### Monitor convergence
            psnr_s = []
            mse_s = []

            for cvg_slice_index in convergence_indexes:
                cvg_output = x_next[0,0,cvg_slice_index].numpy(force=True)
                cvg_ref = reference_rc[cvg_slice_index]

                if args.dataset_name == 'cork':
                    cvg_ref = cvg_ref[256:256+512, 256:256+512]
                    cvg_output = cvg_output[256:256+512,256:256+512]
                elif args.dataset_name == 'walnut':
                    cvg_ref = cvg_ref[100:400, 100:400]
                    cvg_output = cvg_output[100:400, 100:400]

                mse = np.mean((cvg_ref.reshape(-1) - cvg_output.reshape(-1))**2)
                L = (cvg_ref.max() - cvg_ref.min())
                psnr = 10 * np.log10(L**2 / mse)

                psnr_s.append(psnr)
                mse_s.append(mse)
            psnr = np.mean(psnr)
            mse = np.mean(mse_s)

            ### Save visualization
            right = reference_rc[slice_index]
            left = x_next[0,0,slice_index].clamp(0,None).numpy(force=True); left = left  / right.max() 
            right = right / right.max()
            ckpt_img = np.zeros((left.shape[0], 2*left.shape[1]), dtype=np.float32)
            ckpt_img[:, :left.shape[1]] = left
            ckpt_img[:, left.shape[1]:] = right
            ckpt_img = Image.fromarray(np.uint8(cm.viridis(ckpt_img)*255))
            ckpt_img.save(os.path.join(output_dir, f'imgs/ckpt_img_{step}.png'))
            
            ### Save reg visualization
            if args.red_denoising and step >= args.reg_start:
                patch_check = patch.squeeze()[0].clamp(0,None).numpy(force=True)
                patch_check /= patch_check.max()
                reg_patch_check = denoised_patch.squeeze()[0].clamp(0,1.0).cpu() 
                reg_patch_check /= reg_patch_check.max()
                reg_img = np.zeros((patch_check.shape[0], 2*patch_check.shape[1]), dtype=np.float32)
                
                reg_img[:, :patch_check.shape[1]] = patch_check
                reg_img[:, patch_check.shape[1]:] = reg_patch_check
                reg_img = Image.fromarray(np.uint8(cm.viridis(reg_img)*255))
                reg_img.save(os.path.join(output_dir, f'imgs/reg_img_{step}.png'))
            
            metrics.update({'step': step})
            metrics.update({'data_fidelity': data_fidelity})
            metrics.update({'regularization': regularization})
            metrics.update({'mse': mse})
            metrics.update({'psnr': psnr})
            metrics.update({'gradient': gradient})
            metrics.update({'expansiveness': nonexp_constant})
            metrics.update({'lr' : step_size})
            
            p_bar.set_description(f"Data-fidelity: {data_fidelity:.2e}, Regularization: {regularization:.2e} PSNR: {psnr:.2f}, MSE: {mse:.2e}")
            
            update_summary(
                step, metrics, OrderedDict(), os.path.join(args.output_dir, 'summary.csv'),
                write_header=step == 0)  
        
if __name__ == '__main__':
    
    args, args_text = parse_args()
    input_dir = Path(args.input_dir)
    
    main(args, args_text)