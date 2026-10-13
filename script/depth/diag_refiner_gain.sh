#!/bin/bash
#SBATCH --job-name=diag_gain
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-03:00:00
#SBATCH --account=project_465002934
#SBATCH --output=diag_gain_%j.out
#SBATCH --error=diag_gain_%j.err

# Refiner-gain diagnostics (n-draw curve + stratified errors), see
# script/depth/diag_refiner_gain.py. Examples:
#   # v12 (stochastic, sigma=1): 20 draws, 10 Euler steps, every 2nd val patch
#   sbatch script/depth/diag_refiner_gain.sh
#   # R-lumi-5 (finer_lumi code path, Landsat-only ControlNet)
#   sbatch --export=ALL,EXTRA="--kind plain --run /flash/project_465002934/Marigold_output/fm_refiner-R-lumi-5 --ckpt /flash/project_465002934/Marigold_output/fm_refiner-R-lumi-5/checkpoint/best.pth" script/depth/diag_refiner_gain.sh
#   # smoke test
#   sbatch -p dev-g --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --time=00:30:00 \
#     --export=ALL,EXTRA="--limit 4 --K 4 --n_list 1,2,4 --out_dir /flash/project_465002934/Marigold_output/_diag_refiner_gain_debug" \
#     script/depth/diag_refiner_gain.sh
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
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/diag_refiner_gain.py
export BIND SIF SCRIPT OVERLAY EXTRA

srun bash -c '
  ulimit -c 0
  singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT $EXTRA
'
