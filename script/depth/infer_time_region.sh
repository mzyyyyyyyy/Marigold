#!/bin/bash
#SBATCH --job-name=infer_time_region
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-01:30:00
#SBATCH --account=project_465002934
#SBATCH --output=infer_time_region_%j.out
#SBATCH --error=infer_time_region_%j.err

# Wall-to-wall maps of baseline_sr4ir_reg / sr_fm_refiner_v11-lumi over the Danish LiDAR
# footprint on Landsat tile 13,61, 2014-2025 (one task per year, 12 tasks on 2 nodes);
# see script/depth/infer_time_region.py. Results: /flash/project_465002934/result_time_lumi_denm/
#   sbatch script/depth/infer_time_region.sh
# quick test (1 GPU, one year):
#   sbatch -p dev-g --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --time=00:45:00 \
#     --export=ALL,EXTRA="--years 2019 --out_dir /flash/project_465002934/Marigold_output/_time_region_debug" \
#     script/depth/infer_time_region.sh
# If the first conv dies with "MIOpen Error: Cannot open database file:/tmp/gfx90a6e.ukdb"
# (miopenStatusInternalError), that file on the node belongs to another user (mode 600):
# resubmit with --exclude=<that node>.
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
# plain image (transformers 5.3.0): both models use DAv2, not CHMv2
SIF=/flash/project_465002934/env/marigold_env.sif
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/infer_time_region.py
export BIND SIF SCRIPT EXTRA

srun bash -c '
  ulimit -c 0
  singularity exec $BIND $SIF python -u $SCRIPT $EXTRA
'
