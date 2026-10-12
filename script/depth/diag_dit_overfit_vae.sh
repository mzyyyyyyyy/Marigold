#!/bin/bash
#SBATCH --job-name=diag_dit_lumi0
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=0-02:30:00
#SBATCH --account=project_465002934
#SBATCH --exclude=nid005006   # conv2d fails with MIOpen errors on this node (2026-10-08)
#SBATCH --output=/users/mazhanyu/Projects/Marigold/diag_dit_%j.out
#SBATCH --error=/users/mazhanyu/Projects/Marigold/diag_dit_%j.err

# Diagnostics for fm_refiner-D-lumi-0 (1 GPU): VAE floor / noise test + overfit test.
# Usage: sbatch script/depth/diag_dit_overfit_vae.sh [extra args for diag_dit_overfit_vae.py]
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GDAL_CACHEMAX=512
BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
      --bind /scratch/project_465002934:/scratch/project_465002934 \
      --bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
OVERLAY=/flash/project_465002934/env/py_overlay_tf517
cd /users/mazhanyu/Projects/Marigold
singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u script/depth/diag_dit_overfit_vae.py --test both "$@"
