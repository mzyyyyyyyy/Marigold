#!/bin/bash
#SBATCH --job-name=vis_eval_region
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0-01:00:00
#SBATCH --account=project_465002934
#SBATCH --output=vis_eval_region_%j.out
#SBATCH --error=vis_eval_region_%j.err

# PNGs, year strip, GIFs and LiDAR evaluation of the infer_time_region.py maps (CPU only);
# see script/depth/vis_eval_region.py. Results: <in_dir>/{png,gif,eval}/
#   sbatch script/depth/vis_eval_region.sh
#   sbatch --export=ALL,EXTRA="--in_dir /flash/project_465002934/Marigold_output/_time_region_debug" script/depth/vis_eval_region.sh
EXTRA=${EXTRA:-}

BIND="--bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/vis_eval_region.py

singularity exec $BIND $SIF python -u $SCRIPT $EXTRA
