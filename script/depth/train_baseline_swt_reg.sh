#!/bin/bash
#SBATCH --job-name=marigold_swt
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-08:00:00
#SBATCH --account=project_465002934
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# 8 nodes x 8 GPUs/node = 64 GPUs total (same layout as train_sr_fm_refiner.sh).
# One srun task per GPU. train_baseline_swt_reg.py reads SLURM_PROCID (global
# rank), SLURM_NTASKS (world size) and SLURM_LOCALID (GPU on this node) from
# the environment — no extra flags needed. MASTER_ADDR is the first allocated
# node's hostname, required by torch.distributed for multi-node runs.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500

# Pin RCCL to the Slingshot 11 fabric (hsn0-3); otherwise it can auto-pick the
# management NIC and hang until the watchdog timeout (LUMI's documented fix).
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3   # harmless no-op on ROCm >=6.2, still required on older images

# The CHMv2 snapshot is loaded from a local dir, but transformers/HF hub may still
# do a live HTTP HEAD to huggingface.co from all 64 ranks at once and stall at
# startup; force offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Cap GDAL's per-process block cache (MB): 512 x 6 workers x 8 ranks/node
# = ~24GB worst case per node, well inside the 256G budget (see
# train_sr_fm_refiner.sh for the host-RAM OOM history).
export GDAL_CACHEMAX=512

BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
      --bind /scratch/project_465002934:/scratch/project_465002934 \
      --bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/train_baseline_swt_reg.py
# The sif ships transformers 5.3.0, which has no CHMv2 (the task network). This
# overlay holds transformers 5.17.0, prepended to PYTHONPATH for this job only
# (same as train_baseline_chmv2.sh / train_sr_fm_refiner.sh).
OVERLAY=/flash/project_465002934/env/py_overlay_tf517
CONFIG=config/baseline_swt-lumi-nd.yaml
# Checkpoints are large and the home filesystem (20G quota) fills up; write
# outputs to the project's Flash storage instead.
OUTPUT_DIR=/flash/project_465002934/Marigold_output
# The script does os.makedirs(out_dir_run, exist_ok=False), so a retry of THIS
# run must clear the directory the previous (killed) attempt created.
# JOB_NAME mirrors the script's own naming (config filename without extension).
JOB_NAME=$(basename "$CONFIG" .yaml)
RUN_DIR="$OUTPUT_DIR/$JOB_NAME"
export BIND SIF SCRIPT CONFIG OUTPUT_DIR OVERLAY

# RCCL/NCCL debug logging, one file per rank (64 ranks interleaved on one
# stderr is unreadable), so a collective hang can be traced to the connection
# it is stuck on. Same as train_sr_fm_refiner.sh.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
export NCCL_DEBUG_ROOT=/scratch/project_465002934/nccl_debug
mkdir -p "$NCCL_DEBUG_ROOT"

# Two layers of retry for the intermittent Slingshot/RDMA collective hang seen
# on this cluster (see train_sr_fm_refiner.sh):
# 1. In-allocation retries (no queue wait): a fresh attempt builds new NCCL
#    communicators; dist.init_process_group()'s 2-minute collective timeout
#    makes a stuck attempt abort on its own instead of hanging to the time limit.
# 2. If all of those fail, resubmit as a new sbatch job for a fresh node
#    allocation, up to MAX_RESUBMITS times (tracked via RESUBMIT_COUNT).
MAX_ATTEMPTS=3
MAX_RESUBMITS=2
export RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [ "$attempt" -gt 1 ] || [ "$RESUBMIT_COUNT" -gt 0 ]; then
    echo "=== clearing incomplete output dir from previous retry: $RUN_DIR ==="
    rm -rf "$RUN_DIR"
  fi
  echo "=== srun attempt ${attempt}/${MAX_ATTEMPTS} (resubmit ${RESUBMIT_COUNT}/${MAX_RESUBMITS}, $(date)) ==="
  export ATTEMPT="$attempt"
  srun bash -c '
    # Core dumps default to unlimited here and land in the cwd (home quota);
    # we debug from Python tracebacks, so disable them.
    ulimit -c 0
    export NCCL_DEBUG_FILE="$NCCL_DEBUG_ROOT/rank_${SLURM_PROCID}_resubmit${RESUBMIT_COUNT}_attempt${ATTEMPT}.log"
    singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT --config $CONFIG --output_dir $OUTPUT_DIR
  '
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "=== attempt ${attempt} succeeded ==="
    exit 0
  fi
  echo "=== attempt ${attempt} failed with exit code ${status} ==="
done

echo "=== all ${MAX_ATTEMPTS} in-allocation attempts failed ==="
if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]; then
  echo "=== resubmitting for a fresh node allocation (resubmit $((RESUBMIT_COUNT + 1))/${MAX_RESUBMITS}) ==="
  sbatch --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) "$0"
  # Non-zero on purpose: this job did NOT complete the run, it only handed off.
  exit 1
else
  echo "=== reached MAX_RESUBMITS (${MAX_RESUBMITS}), giving up ==="
  exit "$status"
fi
