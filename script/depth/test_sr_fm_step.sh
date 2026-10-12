#!/bin/bash
# 1-GPU functional test of sr_fm_refiner_d_v0-lumi (see test_sr_fm_step.py). Usage: srun -A project_465002934 -p dev-g --gpus=1 -t 20 bash script/depth/test_sr_fm_step.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 --bind /scratch/project_465002934:/scratch/project_465002934 --bind /flash/project_465002934:/flash/project_465002934"
cd /users/mazhanyu/Projects/Marigold
singularity exec $BIND --env PYTHONPATH=/flash/project_465002934/env/py_overlay_tf517 /flash/project_465002934/env/marigold_env.sif python -u script/depth/test_sr_fm_step.py
