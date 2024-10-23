#! /bin/bash

cd nef-red

# activate your environment

python -m kk_3D_CT  \
    --config='configs/walnut.yaml' \
    --output_base='./train' \
    --output_dir='walnut/red' \
    --output_suffix='Walnut1' \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --split_set='train' \
    --acquisition_id='1' \
    --memmap \
    --num_proj=50 \
    --init_lr=3e-2 \
    --n_steps=1000 \
    --regularization_mode='postp' \
    --reg_checkpoint='/home/users/rvo/nef-red/train/walnut/50p/postp/20241021-142041-PostProcessing-IterativeModule_bs32-256x256-1000e-n=5-FW-biased-BN_stem11-nomemnocrop/last.pth.tar' \
    --lambda_reg=1.0 \
    --reg_start=20 \
    --reg_n_iter=1 \
    --red_denoising \
    --reg_batch_size=64 \
    --reg_patch_size=256 \