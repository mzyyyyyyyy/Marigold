#!/bin/bash
#SBATCH --job-name=nsteps_navg
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-03:00:00
#SBATCH --account=project_465002934
#SBATCH --output=nsteps_navg_%j.out
#SBATCH --error=nsteps_navg_%j.err

# Sensitivity of sr_fm_refiner_v12-lumi-nd / sr_fm_refiner_v11-lumi (best.pth) to n_steps / n_avg
# on the TEST split; see script/depth/eval_nsteps_navg.py. Results: <out_dir>/{results.json,results.md,plots/,images/}
#   sbatch script/depth/eval_nsteps_navg.sh                       # v12-nd
#   sbatch --export=ALL,MODEL=v11,EXTRA="--vis_all" script/depth/eval_nsteps_navg.sh   # v11, every patch drawn
# quick smoke test (2 GPUs, 4 samples, small grid):
#   sbatch -p dev-g --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --time=00:30:00 \
#     --export=ALL,MODEL=v12-nd,EXTRA="--limit 4 --vis_ids 0 1 --n_steps_grid 1 2 --n_avg_grid 1 2 --out_dir /flash/project_465002934/Marigold_output/_nsteps_navg_debug" \
#     script/depth/eval_nsteps_navg.sh
# If the first conv dies with "MIOpen Error: Cannot open database file:/tmp/gfx90a6e.ukdb"
# (miopenStatusInternalError), that file on the node belongs to another user (mode 600):
# resubmit with --exclude=<that node>.
MODEL=${MODEL:-v12-nd}
EXTRA="--model $MODEL ${EXTRA:-}"

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
# transformers 5.17 overlay for CHMv2 (v12-nd); v11 (DAv2) runs on the plain image's 5.3.0
if [ "$MODEL" = "v12-nd" ]; then
  OVERLAY=/flash/project_465002934/env/py_overlay_tf517
else
  OVERLAY=
fi
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/eval_nsteps_navg.py
export BIND SIF SCRIPT OVERLAY EXTRA

srun bash -c '
  ulimit -c 0
  singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT $EXTRA
'
