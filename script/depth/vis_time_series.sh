#!/bin/bash
#SBATCH --job-name=vis_time_series
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --account=project_465002934
#SBATCH --output=vis_time_series_%j.out
#SBATCH --error=vis_time_series_%j.err

# Colour PNGs + per-patch year strips of the infer_time_series.py output (CPU only);
# see script/depth/vis_time_series.py. Results: /flash/project_465002934/result_time_lumi/png/
#   sbatch script/depth/vis_time_series.sh
EXTRA=${EXTRA:-}

BIND="--bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/vis_time_series.py

singularity exec $BIND $SIF python -u $SCRIPT $EXTRA
