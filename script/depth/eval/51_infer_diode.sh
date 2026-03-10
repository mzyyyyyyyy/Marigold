#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
# ckpt=${1:-"prs-eth/marigold-depth-v1-1"}
BASE_DATA_DIR="/mnt/data/dataset/marigold"
subfolder="eval"
n_ensemble=10

python script/depth/infer.py \
    --checkpoint /mnt/data/model/marigold/marigold-depth-v1-1 \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 1 \
    --ensemble_size ${n_ensemble} \
    --dataset_config config/dataset_depth/data_diode_all.yaml \
    --output_dir output/${subfolder}/diode/prediction \
    --processing_res 640 \
    --resample_method bilinear
