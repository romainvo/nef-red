#! /bin/bash

cd nef-red

# activate your environment

python -m kk_3D_CT  \
    --config='configs/walnut.yaml' \
    --output_base='./train' \
    --output_dir='walnut/apgd' \
    --output_suffix='Walnut1' \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --split_set='train' \
    --acquisition_id='1' \
    --memmap \
    --num_proj=50 \
    --init_lr=3e-2 \
    --n_steps=1000 \
    --regularization_mode='none' \