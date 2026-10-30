#!/bin/bash
#SBATCH --job-name=precompute_teacher
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00
#SBATCH --account=project_465002934
#SBATCH --output=precompute_teacher_%j.out
#SBATCH --error=precompute_teacher_%j.err

# Distillation targets of a finished teacher run on every train patch, plus its
# train/val R2 (overfitting check); see script/depth/precompute_teacher.py.
#   sbatch --export=ALL,TEACHER_RUN=/flash/project_465002934/Marigold_output/baseline_chmv2-lumi-nd-ps8-h120 \
#     script/depth/precompute_teacher.sh
# Smoke test (one dev-g node, a few patches, temp output dir):
#   sbatch -p dev-g --nodes=1 --time=00:45:00 --export=ALL,TEACHER_RUN=...,EXTRA="--limit 64 --out_dir $(pwd)/_smoke_teacher_targets" \
#     script/depth/precompute_teacher.sh
EXTRA=${EXTRA:-}

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GDAL_CACHEMAX=512

BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
      --bind /scratch/project_465002934:/scratch/project_465002934 \
      --bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
OVERLAY=/flash/project_465002934/env/py_overlay_tf517   # transformers 5.17 (CHMv2)
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/precompute_teacher.py
export BIND SIF SCRIPT OVERLAY EXTRA TEACHER_RUN

srun bash -c '
  ulimit -c 0
  singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT --teacher_run $TEACHER_RUN $EXTRA
'
