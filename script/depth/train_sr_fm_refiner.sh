#!/bin/bash
#SBATCH --job-name=marigold_depth
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-03:00:00
#SBATCH --account=project_465002934
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err


singularity exec \
    --bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
    --bind /scratch/project_465002934:/scratch \
    --bind /flash/project_465002934:/flash \
    /flash/project_465002934/env/marigold_env.sif \
    python /users/mazhanyu/Projects/Marigold/script/depth/train_sr_fm_refiner.py