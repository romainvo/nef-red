#! /bin/bash

cd nef-red

# activate your environment

python -m train_postp  \
    --config='configs/walnut.yaml' \
    --output_base='./train' \
    --output_dir='walnut/postp' \
    --output_suffix='' \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --memmap \
    --batch_size=32 \
    --patch_size=256 \
    --num_epochs=1000 \
    --num_warmup_epochs=5 \
    --num_workers=1 \
    --axial_center_crop \
    --iterative_model=5 \
    --forward_iterates \
    --ema \
