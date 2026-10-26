#!/bin/bash
#SBATCH --job-name=eval_std_metrics
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --account=project_465002934
#SBATCH --output=eval_std_metrics_%j.out
#SBATCH --error=eval_std_metrics_%j.err

# Established canopy-height / depth-boundary metrics (Open-Canopy, Tolan et al., iBims-1 DBE,
# SEE) of the four group-meeting models from their saved raw predictions (CPU only);
# see script/depth/eval_std_metrics.py. Needs eval_group_meeting.py --save_dir first.
# Results: /flash/project_465002934/result_group_meeting/metrics_std/
#   sbatch script/depth/eval_std_metrics.sh
EXTRA=${EXTRA:-}

BIND="--bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/eval_std_metrics.py

singularity exec $BIND $SIF python -u $SCRIPT $EXTRA
