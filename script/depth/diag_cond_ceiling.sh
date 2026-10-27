#!/bin/bash
#SBATCH --job-name=diag_ceil
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=project_465002934
#SBATCH --output=diag_ceil_%j.out
#SBATCH --error=diag_ceil_%j.err

# PS-slot conditions (real / null / pseudo / lsps) for one FM checkpoint, see
# script/depth/diag_cond_ceiling.py (one GPU), e.g.
#   sbatch script/depth/diag_cond_ceiling.sh
OUT=/flash/project_465002934/Marigold_output
EXTRA=${EXTRA:-}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GDAL_CACHEMAX=512

BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
      --bind /scratch/project_465002934:/scratch/project_465002934 \
      --bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
OVERLAY=/flash/project_465002934/env/py_overlay_tf517   # transformers 5.17 (CHMv2)
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/diag_cond_ceiling.py

ulimit -c 0
singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT $EXTRA
