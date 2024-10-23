#!/usr/bin/env python3

import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as D

from PIL import Image
from matplotlib import cm

import time
from pathlib import Path
import math

from tqdm import tqdm
from datetime import datetime

from nerfacc.estimators.occ_grid import OccGridEstimator

from utils import *
from data import create_ray_dataset
from scheduler import create_scheduler
from tomography import namedtuple_map, sample_and_integrate
from models import create_field, create_model
from models.reg import GradientHook

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

if __name__ == "__main__":

	args, args_text = parse_args()

	device = torch.device("cuda") if not args.cpu else torch.device('cpu')

    # Initialize the current working directory
	output_dir = ''
	output_base = args.output_base
	exp_name = '-'.join([
		datetime.now().strftime('%Y%m%d-%H%M%S'),
		f'NGP_3DCT-{args.num_proj}p-{args.n_levels}L-hash{args.log2_hashmap_size}-occ{args.occupancy_threshold}-lambda{args.lambda_reg}',
	])
	if args.output_suffix:
		exp_name = f'{exp_name}_{args.output_suffix}'
	output_dir = os.path.join(output_base, args.output_dir, exp_name)
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(os.path.join(output_dir, 'imgs'), exist_ok=True)
	args.output_dir = output_dir

	with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
		f.write(args_text)

	if args.log_name is not None:
		sys.stdout = Logger(os.path.join(output_dir, args.log_name+'.log'))

	num_rays = args.num_rays = args.target_sample_batch_size // args.num_points
	dataset = create_ray_dataset(training=True, **vars(args))

	train_loader = D.DataLoader(
		dataset,
		batch_size=None,
		batch_sampler=None,
		shuffle=True,
		pin_memory=True,
		num_workers=args.num_workers
	)	

	reconstruction_geometry = dataset.geometry
	reconstruction_radius = reconstruction_geometry.reconstruction_radius
	scene_aabb = reconstruction_geometry.scene_aabb.to(device)
	render_step_size = reconstruction_geometry.render_step_size

	grid_resolution = 128
	occupancy_grid = OccGridEstimator(
		roi_aabb=scene_aabb, resolution=grid_resolution, levels=1
	).to(device)

	neural_field = create_field(
		n_input_dims=3, n_output_dims=1, aabb=scene_aabb, reconstruction_origin=reconstruction_geometry.reconstruction_origin,
		**vars(args)
	)
	if args.checkpoint_file is not None:
		nef_checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
		missing_keys, unexpected_keys = neural_field.load_state_dict(nef_checkpoint['state_dict'], strict=True)
	print(f'[INFO] number of parameters : {count_parameters(neural_field)}')
	neural_field = neural_field.to(device).train()

	try:
		policies = [
			{'params': neural_field.mlp.parameters(), 'weight_decay': args.weight_decay},
			{'params': neural_field.encoding.parameters(), 'weight_decay': 0},
		]
	except AttributeError:
		policies = [
			{'params': neural_field.parameters(), 'weight_decay': 0.}
		]
	optimizer = torch.optim.Adam(policies, 
								 lr=args.init_lr,
								 betas=(0.9,0.99),
								 eps=1e-8)
	lr_scheduler = create_scheduler(
		optimizer=optimizer,
		lr_scheduler='CosineAnnealingLR',
  		min_lr=1e-8,
		num_epochs=args.n_steps,
	)

	##### POSTP CONFIG #####
	print(f'***** REGULARIZATION MODE : {args.regularization_mode}')
	if args.regularization_mode == 'postp':
		checkpoint = torch.load(args.reg_checkpoint, map_location='cpu')

		postp_args = checkpoint['args']

		if args.reg_n_iter > 1 or postp_args.iterative_model > 0:
			postp_args.iterative_model = args.reg_n_iter
			postp_model = create_model(**vars(postp_args))
			missing_keys, unexpected_keys = postp_model.load_state_dict(checkpoint['state_dict'], strict=False)
			postp_model.n_iter = args.reg_n_iter
		else:
			postp_model = create_model(**vars(postp_args))
			missing_keys, unexpected_keys = postp_model.load_state_dict(checkpoint['state_dict'], strict=False)	
		print("Unexpected keys :", unexpected_keys)
		print("Missing keys :", missing_keys)
		print("Number of postp_model parameters : {}\n".format(count_parameters((postp_model))))
  
		assert len(missing_keys) == 0

		if 'state_dict_ema' in checkpoint:
			if checkpoint['state_dict_ema'] is not None:
				try:
					print('*** New EMA loading ckpt ***')
					postp_model.load_state_dict(checkpoint['state_dict_ema'], strict=True)
				except Exception as e:
					print('*** Legacy EMA loading ckpt ***')

					for model_v, ema_v in zip(postp_model.state_dict().values(), 
											checkpoint['state_dict_ema'].values()):
						model_v.copy_(ema_v)

		postp_model = postp_model.to(device)
		postp_model = postp_model.eval()

	##### VISUALIZATION CONFIG #####
	resolution = dataset.resolution
	img_shape = dataset.slice_shape
	print(f'***** resolution : {resolution} ; img_shape : {img_shape}')

	half_dx =  0.5 / resolution[2]
	half_dy =  0.5 / resolution[1]
	half_dz = 0.5 / resolution[0]
	xs = torch.linspace(half_dx, 1-half_dx, resolution[1], device=device)
	ys = torch.linspace(half_dy, 1-half_dy, resolution[2], device=device)
	yv, xv = torch.meshgrid([ys, xs], indexing="ij")

	if args.dataset_name == 'cork':
		slice_index = 300 
		L_max = 0.1179
		convergence_indexes = np.linspace(start=1024//5,
											stop=4*1024//5,
											num=1024//10, 
											dtype=int, endpoint=True)
	elif args.dataset_name == 'walnut':
		slice_index = 250
		L_max = 0.502464 	
		convergence_indexes = np.linspace(start=501//5,
										  stop=4*501//5,
										  num=501//10, 
										  dtype=int, endpoint=True)
		
	z = half_dz + slice_index / resolution[0]
	zv = torch.tensor(z, device=device).repeat(resolution[1:])

	xyz = torch.stack([xv.flatten(),yv.flatten(),zv.flatten()]).t()
	print(f"Grid size : ({resolution[0]},{xyz.shape[0] / resolution[0]})")
	##### VISUALIZATION CONFIG #####
	interval = 10

	print(f"Beginning optimization with {args.n_steps} training steps.")
	args.n_steps = int(args.n_steps)
	loss = 0

	expansiveness = []; txk, xk, tyk, yk = None, None, None, None
	p_bar = tqdm(range(args.n_steps), total=args.n_steps, leave=True, position=0)
	step = 0
	best_metric, best_epoch = None, None
	for epoch in p_bar:
		data_fidelity_m = AverageMeter()
		reg_loss_m = AverageMeter()

		average_gradient_m = AverageMeter()

		expansiveness_m = AverageMeter()

		train_metrics = OrderedDict()
		last_idx = len(train_loader) -1
		for batch_idx, (data) in enumerate(train_loader):			
			loss = 0
			data_fidelity = torch.tensor(0.0).to(device)
			reg = torch.tensor(0.0).to(device)

			torch.cuda.synchronize()

			rays = data['rays']
			rays = namedtuple_map(lambda r: r.to(device), rays)
			targets = data['ray_integrations'].to(device)


			def occ_eval_fn(x):
				step_size = render_step_size
				# compute occupancy
				density = neural_field(x, normalize_positions=True)
				return density * step_size

			# update occupancy grid
			occupancy_grid.update_every_n_steps(
				step=step,
				occ_eval_fn=occ_eval_fn,
				occ_thre=args.occupancy_threshold,
				n=16
			)
   
			output, n_rendering_samples = sample_and_integrate(
				neural_field,
				occupancy_grid,
				rays,
    			t_min=None,	
       			t_max=None,
				render_step_size=render_step_size,			
			)
			if n_rendering_samples == 0:
				continue

			num_rays = len(targets)
			num_rays = int(
				num_rays
				* (args.target_sample_batch_size / float(n_rendering_samples))
			)

			loss = F.mse_loss(output, targets.to(output.dtype), reduction='sum')
			torch.cuda.synchronize()

			##### POSTP REG #####
			reg = 0; neural_patch = None
			if args.regularization_mode != 'none' and epoch >= args.reg_start:
				if args.reg_sampling == 'cube':
					xyz_reg = sample_random_cube(args.reg_patch_size, 
												 patch_z_size=args.reg_batch_size,
												 resolution=resolution,
												 device=output.device,
												 dataset_name=args.dataset_name)

					neural_patch = neural_field(xyz_reg, normalize_positions=False)
					neural_patch = neural_patch.reshape(args.reg_patch_size,
														args.reg_patch_size,
														args.reg_batch_size)
					neural_patch = neural_patch.permute(2,0,1).unsqueeze(1)

				if args.regularization_mode == 'postp':
					if args.red_denoising:
						with torch.no_grad():
							reg_patch = postp_model(neural_patch.clamp(0, None), residual_learning=True)
					else:
						reg_patch = postp_model(neural_patch.clamp(0, None), residual_learning=True)

					reg = F.mse_loss(neural_patch, reg_patch, reduction='sum')

				loss += args.lambda_reg * reg

			##### POSTP REG #####

			gradient_hook = GradientHook(
								[(1., output), 
				 				 (args.lambda_reg, neural_patch)])

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			##
			torch.cuda.synchronize()

			with torch.no_grad():
				data_fidelity_m.update(((output - targets.to(output.dtype))**2).mean().item(), n=targets.size(0))
				if reg > 0:
					reg_loss_m.update(reg.item(), n=neural_patch.size(0))
				average_gradient_m.update(gradient_hook.get_grad().item())

				if args.regularization_mode != 'none' and neural_patch is not None:
					if xk is None:
						xk = neural_patch.detach().cpu().numpy().reshape(-1)
						txk = reg_patch.detach().cpu().numpy().reshape(-1)
					else:
						yk = neural_patch.detach().cpu().numpy().reshape(-1)
						tyk = reg_patch.detach().cpu().numpy().reshape(-1)

						nonexp_constant = np.linalg.norm(tyk - txk) / np.linalg.norm(yk - xk)
						xk, txk = yk, tyk
						expansiveness.append(nonexp_constant)
						expansiveness_m.update(nonexp_constant)

			dataset.update_num_rays(num_rays)
			step +=1

		### Save checkpoint
		lr = optimizer.param_groups[0]['lr']

		loss_val = loss.item()
		torch.cuda.synchronize()

		if best_metric is None:
			best_metric, best_epoch = data_fidelity_m.avg, epoch
			if occupancy_grid is not None:
				save_checkpoint(epoch, neural_field, optimizer, args, loss_val, occupancy_grid=occupancy_grid.state_dict(), ckpt_name=None)
			else:
				save_checkpoint(epoch, neural_field, optimizer, args, loss_val, ckpt_name=None)		
		elif data_fidelity_m.avg < best_metric:
			if os.path.exists(os.path.join(args.output_dir, 'checkpoint-{}.pth.tar'.format(best_epoch))):
				os.unlink(os.path.join(args.output_dir, 'checkpoint-{}.pth.tar'.format(best_epoch)))
			best_metric, best_epoch = data_fidelity_m.avg, epoch
			if occupancy_grid is not None:
				save_checkpoint(epoch, neural_field, optimizer, args, loss_val, occupancy_grid=occupancy_grid.state_dict(), ckpt_name=None)
			else:
				save_checkpoint(epoch, neural_field, optimizer, args, loss_val, ckpt_name=None)

		if occupancy_grid is not None:
			save_checkpoint(epoch, neural_field, optimizer, args, loss_val, occupancy_grid=occupancy_grid.state_dict(), ckpt_name='last.pth.tar')
		else:
			save_checkpoint(epoch, neural_field, optimizer, args, loss_val, ckpt_name='last.pth.tar')
  
		### Monitor convergence
		neural_field = neural_field.eval()
		with torch.no_grad():
			psnr_s = []
			mse_s = []

			for cvg_slice_index in convergence_indexes:
				z = half_dz + cvg_slice_index / resolution[0]
				zv = torch.tensor(z, device=device).repeat(resolution[1:])

				xyz_cvg = torch.stack([xv.flatten(),yv.flatten(),zv.flatten()]).t().to(torch.float32)

				cvg_output = neural_field(xyz_cvg, normalize_positions=False).reshape(resolution[1:]).clamp(0, None).cpu().numpy().squeeze()
				cvg_ref = dataset.reference_rcs[dataset.acquisition_id][cvg_slice_index]

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

		neural_field = neural_field.train()

		### Save visualization
		img_path = f"{args.output_dir}/imgs/tomo_{epoch}.png"
		print(f"Writing '{img_path}'... ", end="")
		neural_field = neural_field.eval()
		with torch.no_grad():
			left = neural_field(xyz, normalize_positions=False).reshape(img_shape).clamp(0, None).cpu().numpy().squeeze()

			right = np.clip(dataset.reference_rcs[dataset.acquisition_id][slice_index], 0, None).squeeze()
			left /=right.max()
			right /= right.max()
			right = abs(left - right) / abs(left - right).max()

			ckpt_img = Image.fromarray(np.uint8(cm.viridis(left)*255))
			ckpt_img.save(img_path)

			ckpt_img = np.zeros((left.shape[0], 2*left.shape[1]), dtype=np.float32)
			ckpt_img[:, :left.shape[1]] = left; ckpt_img[:, left.shape[1]:] = right; 
			ckpt_img = Image.fromarray(np.uint8(cm.viridis(ckpt_img)*255))
			ckpt_img.save(img_path)

		### Save reg visualization
		with torch.no_grad():
			if neural_patch is not None:
				patch_check = neural_patch.squeeze()[0].clamp(0, None).numpy(force=True)
				patch_check /= patch_check.max()
				reg_patch_check = reg_patch.squeeze()[0].clamp(0,None).numpy(force=True)
				reg_patch_check /= reg_patch_check.max()
				reg_img = np.zeros((patch_check.shape[0], 2*patch_check.shape[1]), dtype=np.float32)
                
				reg_img[:, :patch_check.shape[1]] = patch_check
				reg_img[:, patch_check.shape[1]:] = reg_patch_check
				reg_img = Image.fromarray(np.uint8(cm.viridis(reg_img)*255))
				reg_img.save(os.path.join(args.output_dir, f'imgs/reg_img_{epoch}.png'))

		neural_field = neural_field.train()

		### Update metrics
		train_metrics.update({'epoch' : epoch})
		train_metrics.update({'data_fidelity' : data_fidelity_m.avg})
		train_metrics.update({'regularization' : reg_loss_m.avg})
		train_metrics.update({'psnr': psnr})
		train_metrics.update({'mse': mse})
		train_metrics.update({'gradient' : average_gradient_m.avg})
		train_metrics.update({'expansiveness' : expansiveness_m.avg})
		train_metrics.update({'lr' : lr})

		p_bar.set_description(f"Data-fidelity: {data_fidelity_m.avg:.2e}, Regularization: {reg_loss_m.avg:.2e} PSNR: {psnr:.2f}, MSE: {mse:.2e}")

		update_summary(
			step, train_metrics, OrderedDict(), os.path.join(args.output_dir, 'summary.csv'),
			write_header=epoch == 0)  
		np.save(os.path.join(args.output_dir, 'expansiveness.npy'), np.array(expansiveness))

		lr_scheduler.step()
		sys.stdout.flush()
		if args.debug:
			break

	if args.inference_save:

		print(f"Saving reconstruction in raw format... ", end="")
		neural_field = neural_field.eval()

		pred_rc = np.zeros((resolution[0], resolution[1], resolution[2]), dtype=np.float32)
		for idx in tqdm(range(resolution[0])):
			
			z = half_dz + idx / resolution[0]
			zv = torch.tensor(z, device=device).repeat(resolution[1:])
			xyz = torch.stack([xv.flatten(),yv.flatten(),zv.flatten()]).t()	

			with torch.no_grad():
				pred_rc[idx] = neural_field(xyz, normalize_positions=False).reshape(resolution[1:]).clamp(0, None).cpu().numpy().squeeze()

		pred_rc.tofile(os.path.join(args.output_dir, f"{args.acquisition_id}_pred.raw"))

	tcnn.free_temporary_memory()

	if args.log_name is not None:
		sys.stdout.close()