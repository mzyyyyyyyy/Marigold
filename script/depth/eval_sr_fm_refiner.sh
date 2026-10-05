#!/bin/bash
#SBATCH --job-name=sr_fm_eval
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00
#SBATCH --account=project_465002934
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# Eval-only: run train_sr_fm_refiner.py's final full validation on one saved
# checkpoint (no training, nothing in the run's checkpoint/ is touched).
# Results go to <run>/eval_<ckpt name>/ (logging.log + metrics.json).
#   sbatch script/depth/eval_sr_fm_refiner.sh                         # v12 latest.pth
#   CKPT=/flash/.../other/checkpoint/best.pth sbatch --export=ALL script/depth/eval_sr_fm_refiner.sh
CKPT=${CKPT:-/flash/project_465002934/Marigold_output/sr_fm_refiner_v12-lumi/checkpoint/latest.pth}

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
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/train_sr_fm_refiner.py
# --config is required-by-default but unused with --eval_ckpt (config.yaml is
# read from the run dir).
CONFIG=config/sr_fm_refiner_v12-lumi.yaml
export BIND SIF SCRIPT CONFIG OVERLAY CKPT

srun bash -c '
  ulimit -c 0
  singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT --config $CONFIG --eval_ckpt $CKPT --no_wandb
'
