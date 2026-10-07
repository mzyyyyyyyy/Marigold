#!/bin/bash
#SBATCH --job-name=test_dit_omini
#SBATCH --partition=dev-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-01:15:00
#SBATCH --account=project_465002934
#SBATCH --output=/users/mazhanyu/Projects/Marigold/test_dit_omini_%j.out
#SBATCH --error=/users/mazhanyu/Projects/Marigold/test_dit_omini_%j.err

# Tests for fm_refiner-D-lumi-0 (backbone "dit_omini"), 4 nodes x 8 GPUs, same container / overlay /
# environment as train_fm_refiner.sh. Logs: test_dit_omini_<jobid>.out / .err in the project dir.
#   1. backbone sanity checks (script/depth/test_dit_swap.py) on ONE GPU: tiny model (all four checks),
#      then the real DiT-B at 240 px (interface / init / gradients / params / overfit).
#   2. short real-data DDP training smoke run over all 32 GPUs (config/fm_refiner-D-lumi-0-smoke.yaml,
#      300 steps max, stopped by --exit_after): training loop, periodic validation, checkpointing.
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
OVERLAY=/flash/project_465002934/env/py_overlay_tf517
cd /users/mazhanyu/Projects/Marigold
export BIND SIF OVERLAY

echo "=== 1A: tiny model, all four backbone checks (1 GPU) ==="
srun --nodes=1 --ntasks=1 --gpus=1 bash -c \
  'singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u script/depth/test_dit_swap.py --backbone dit_omini --tiny --skip_unet --size 128 --steps 300'
echo "=== 1A exit code: $? ==="

echo "=== 1B: real DiT-B (768 / 12 layers / 12 heads), 240px, 300 steps (1 GPU) ==="
srun --nodes=1 --ntasks=1 --gpus=1 bash -c \
  'singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u script/depth/test_dit_swap.py --backbone dit_omini --skip_unet --size 240 --steps 300 --lr 1e-4 --skip_cond_test'
echo "=== 1B exit code: $? ==="

echo "=== 2: DDP training smoke run, 4 nodes x 8 GPUs ($(date)) ==="
OUTPUT_DIR=/flash/project_465002934/Marigold_output/test_runs
RUN_DIR="$OUTPUT_DIR/fm_refiner-D-lumi-0-smoke"
mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR
# Up to 3 attempts inside this allocation, as in train_fm_refiner.sh: the first 4-node attempt died in
# DDP's initial parameter broadcast (NCCL timeout), the known LUMI Slingshot connection-setup race.
MAX_ATTEMPTS=3
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  rm -rf "$RUN_DIR"                                    # the script refuses to reuse an existing run dir
  echo "=== 2: srun attempt ${attempt}/${MAX_ATTEMPTS} ($(date)) ==="
  srun bash -c '
    ulimit -c 0
    singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u script/depth/train_fm_refiner.py \
      --config config/fm_refiner-D-lumi-0-smoke.yaml --output_dir $OUTPUT_DIR --exit_after 20
  '
  status=$?
  echo "=== 2: attempt ${attempt} exit code: ${status} ==="
  [ "$status" -eq 0 ] && break
done
ls -la "$RUN_DIR/checkpoint" 2>&1
