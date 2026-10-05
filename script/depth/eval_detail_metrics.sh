#!/bin/bash
#SBATCH --job-name=detail_eval
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-03:00:00
#SBATCH --account=project_465002934
#SBATCH --output=detail_eval_%j.out
#SBATCH --error=detail_eval_%j.err

# Detail-fidelity metrics for baseline_chmv2-lumi / fm_refiner-R-lumi-5 /
# sr_fm_refiner_v12-lumi (latest.pth) on the full val split; see
# script/depth/eval_detail_metrics.py. Results: <out_dir>/detail_metrics.{json,md}
#   sbatch script/depth/eval_detail_metrics.sh
# quick smoke test (2 GPUs, 8 samples):
#   sbatch -p dev-g --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --time=00:30:00 \
#     --export=ALL,EXTRA="--limit 8 --out_dir /flash/project_465002934/Marigold_output/_detail_eval_debug" \
#     script/depth/eval_detail_metrics.sh
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
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/eval_detail_metrics.py
export BIND SIF SCRIPT OVERLAY EXTRA

srun bash -c '
  ulimit -c 0
  singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT $EXTRA
'
