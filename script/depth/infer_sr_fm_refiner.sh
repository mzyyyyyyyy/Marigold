#!/bin/bash
#SBATCH --job-name=marigold_depth_infer
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-04:00:00
#SBATCH --account=project_465002934
#SBATCH --output=infer_%j.out
#SBATCH --error=infer_%j.err

# 8 nodes x 8 GPUs/node = 64 GPUs total.
# Each srun task gets SLURM_PROCID (0..63, used to shard the flattened
# (tile, patch) work list) and SLURM_LOCALID (0..7, used to pick the GPU on
# its node) automatically; infer_sr_fm_refiner.py reads both from the
# environment, no extra flags needed. This matters when there are far fewer
# tiles than GPUs: patches within each tile are split across all 64 ranks
# instead of leaving idle GPUs when tiles run out.

BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
      --bind /scratch/project_465002934:/scratch/project_465002934 \
      --bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/infer_sr_fm_refiner.py
CONFIG=config/sr_fm_refiner_v2_infer_ls.yaml

# Step 1: all 64 ranks compute their patch shard and write partials.
srun singularity exec $BIND $SIF python -u $SCRIPT --config $CONFIG

# Step 2: single task merges all partials into final tifs. Runs after step 1
# completes (srun steps within one job script run sequentially).
srun --ntasks=1 --nodes=1 singularity exec $BIND $SIF python -u $SCRIPT --config $CONFIG --merge
