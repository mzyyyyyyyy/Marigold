#!/bin/bash
#SBATCH --job-name=diag_tdp_ovf
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-04:00:00
#SBATCH --account=project_465002934
#SBATCH --output=diag_tdp_ovf_%j.out
#SBATCH --error=diag_tdp_ovf_%j.err

# SR overfitting test against a frozen FM (one GRID setting per GPU), see
# script/depth/diag_tdp_overfit.py. Examples:
#   # v13-nd (default), all 8 settings
#   sbatch script/depth/diag_tdp_overfit.sh
#   # smoke test
#   sbatch -p dev-g --gpus-per-node=2 --ntasks-per-node=2 --time=00:30:00 \
#     --export=ALL,EXTRA="--n_train 8 --n_val 2 --iters 20 --eval_every 10 --n_steps 2 --n_avg 1 --settings tdp_lr1e-3_nt4+tdp5pix1_lr1e-3 --out_dir /flash/project_465002934/Marigold_output/_diag_tdp_overfit_debug" \
#     script/depth/diag_tdp_overfit.sh
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
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/diag_tdp_overfit.py
export BIND SIF SCRIPT OVERLAY EXTRA

srun bash -c '
  ulimit -c 0
  singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT $EXTRA
'
