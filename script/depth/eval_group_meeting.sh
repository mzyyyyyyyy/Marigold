#!/bin/bash
#SBATCH --job-name=eval_group_meeting
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00
#SBATCH --account=project_465002934
#SBATCH --output=eval_group_meeting_%j.out
#SBATCH --error=eval_group_meeting_%j.err

# baseline_dav2 / baseline_sr4ir_reg / fm_refiner-R-lumi-4 / sr_fm_refiner_v11-lumi
# on the new dataset's TEST split (split_spatial_v1_test.pkl): metrics + per-patch
# images; see script/depth/eval_group_meeting.py.
# Results: /flash/project_465002934/result_group_meeting/
#   sbatch script/depth/eval_group_meeting.sh
# quick smoke test (2 GPUs, 8 samples):
#   sbatch -p dev-g --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --time=00:30:00 \
#     --export=ALL,EXTRA="--limit 8 --out_dir /flash/project_465002934/Marigold_output/_group_meeting_debug" \
#     script/depth/eval_group_meeting.sh
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
# plain image (transformers 5.3.0), as all four models were trained with DAv2, not CHMv2
SIF=/flash/project_465002934/env/marigold_env.sif
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/eval_group_meeting.py
export BIND SIF SCRIPT EXTRA

srun bash -c '
  ulimit -c 0
  singularity exec $BIND $SIF python -u $SCRIPT $EXTRA
'
