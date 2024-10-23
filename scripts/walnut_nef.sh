#! /bin/bash

cd nef-red

# activate your environment

python -m ngp_3D_CT  \
    --config='configs/walnut.yaml' \
    --output_base='./train' \
    --output_dir='walnut/nef' \
    --output_suffix='Walnut1' \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --split_set='train' \
    --acquisition_id='1' \
    --memmap \
    --num_workers=1 \
    --num_proj=50 \
    --init_lr=2e-4 \
    --n_steps=500 \
    --target_sample_batch_size=1048576 \
    --regularization_mode='none' \