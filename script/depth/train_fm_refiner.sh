#!/bin/bash
#SBATCH --job-name=marigold_depth
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-05:00:00
#SBATCH --account=project_465002934
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# 8 nodes x 8 GPUs/node = 64 GPUs total.
# One srun task per GPU. train_fm_refiner.py reads SLURM_PROCID (global
# rank), SLURM_NTASKS (world size), and SLURM_LOCALID (GPU on this node)
# from the environment automatically — no extra flags needed. MASTER_ADDR
# is the first allocated node's hostname, required by
# torch.distributed.init_process_group for multi-node runs.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500

# Without this, RCCL auto-detects a network interface for inter-node
# collectives and can pick the cluster's management NIC instead of the
# Slingshot 11 fabric (hsn0-3) — that's indistinguishable from "the network
# is fine" until the first real collective (DDP's initial parameter
# broadcast) tries to actually move data over it and hangs until NCCL's
# watchdog timeout, exactly the SeqNum=5 OpType=BROADCAST timeout seen
# across 3 separate fresh node allocations on 2026-09-17 (train_22121526/
# 22121824/22121935.err). This is LUMI's own documented fix for that
# failure mode, not something specific to this job — see
# https://lumi-supercomputer.github.io/LUMI-training-materials/2day-20251020/205-Containers/
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3   # harmless no-op on ROCm >=6.2, still required on older images

# The DAv2 backbone (depth-anything/Depth-Anything-V2-Base-hf) is already
# cached locally, but transformers' from_pretrained() still does a live HTTP
# HEAD request to huggingface.co to check for updates unless told not to.
# With many ranks doing that at once, LUMI's compute-node network egress
# becomes a bottleneck and the job stalls at startup. Force cache-only
# loading — no network calls at all.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# See src/util/ps_lazydataset.py's _RASTERIO_CACHE_MAX: the per-worker-process
# rasterio dataset cache is LRU-bounded in code, but also cap GDAL's own
# per-process block cache here as a second, independent ceiling — belt and
# suspenders, and cheap (just an env var) compared to a host-RAM OOM killing
# a multi-hour run. 512 (MB; GDAL_CACHEMAX values under 100000 are
# interpreted as MB) x 6 workers x 8 ranks/node = ~24GB worst case per node,
# well inside the 256G budget.
export GDAL_CACHEMAX=512

BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
      --bind /scratch/project_465002934:/scratch/project_465002934 \
      --bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
# The sif ships transformers 5.3.0, which has no CHMv2 (needed by
# fm_refiner-R-lumi-5's coarse model). This overlay holds transformers 5.17.0
# (+ matching tokenizers/safetensors), prepended to PYTHONPATH for this job
# only; fm_refiner/diffusers 0.37 build fine with it (checked on CPU).
OVERLAY=/flash/project_465002934/env/py_overlay_tf517
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/train_fm_refiner.py
CONFIG=config/fm_refiner-R-lumi-5.yaml
# Checkpoints (best.pth/latest.pth) run tens of GB; the home filesystem
# (/users/mazhanyu, 20G quota) filled up from these and killed a run
# mid-checkpoint-write (see train_sr_fm_refiner.sh). Write outputs to the
# project's Flash storage instead, which has far more headroom.
OUTPUT_DIR=/flash/project_465002934/Marigold_output
# The script does os.makedirs(out_dir_run, exist_ok=False) so it never
# clobbers a genuinely separate completed run — but that means a retry of
# THIS run crashes immediately on the directory the previous (killed)
# attempt already created. RUN_DIR mirrors the script's own naming
# (job_name = config filename without extension, no --add_datetime_prefix
# passed here) so the retry loop below can clear it before each retry.
JOB_NAME=$(basename "$CONFIG" .yaml)
RUN_DIR="$OUTPUT_DIR/$JOB_NAME"
export BIND SIF SCRIPT CONFIG OUTPUT_DIR OVERLAY

# Turn on RCCL/NCCL's own debug logging (one file per rank, since many ranks
# interleaved on one stderr is unreadable) so that a multi-node collective
# hang (see train_sr_fm_refiner.sh for the LUMI Slingshot connection-setup
# race this has shown historically) shows exactly which connection/step it's
# actually stuck on.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
export NCCL_DEBUG_ROOT=/scratch/project_465002934/nccl_debug
mkdir -p "$NCCL_DEBUG_ROOT"

# Two layers of retry, mirroring train_sr_fm_refiner.sh:
# 1. In-allocation retries (cheap, no queue wait): a fresh attempt reuses
#    this same 8-node allocation but builds new NCCL communicators.
#    train_fm_refiner.py's dist.init_process_group() has a 2-minute
#    collective timeout, so a stuck attempt aborts on its own with a clear
#    "Watchdog caught collective timeout" error instead of hanging until
#    the SBATCH time limit. Handles a transient connection-setup race.
# 2. If all in-allocation retries fail, the allocation itself might be the
#    problem (e.g. one bad node/NIC in this particular set) — resubmit as
#    a brand-new sbatch job to get a fresh node allocation, up to
#    MAX_RESUBMITS times, tracked via the RESUBMIT_COUNT env var carried
#    across resubmissions.
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
    # Core dumps are enabled by default on this cluster (ulimit -c unlimited,
    # core_pattern="core" — dumped into the job'"'"'s cwd, this Marigold repo
    # dir, which sits on the 20G-quota home filesystem). NCCL watchdog aborts
    # (and any other SIGABRT/SIGSEGV) trigger one, so disable them outright —
    # debug from Python tracebacks + NCCL_DEBUG logs instead.
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
  # Exit non-zero even though the resubmit itself succeeded: this job did
  # NOT complete the training run, it only handed off to a new one. Without
  # this, sacct shows a misleading "COMPLETED" for a run that never trained
  # anything (last command run was the successful `sbatch` call).
  exit 1
else
  echo "=== reached MAX_RESUBMITS (${MAX_RESUBMITS}), giving up ==="
  exit "$status"
fi
