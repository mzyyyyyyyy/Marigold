#!/bin/bash
#SBATCH --job-name=marigold_depth
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=256G
#SBATCH --time=0-03:00:00
#SBATCH --account=project_465002934
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# 8 nodes x 8 GPUs/node = 64 GPUs total.
# One srun task per GPU. train_sr_fm_refiner.py reads SLURM_PROCID (global
# rank), SLURM_NTASKS (world size), and SLURM_LOCALID (GPU on this node)
# from the environment automatically — no extra flags needed. MASTER_ADDR
# is the first allocated node's hostname, required by
# torch.distributed.init_process_group for multi-node runs.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500

# The DAv2 backbone (depth-anything/Depth-Anything-V2-Base-hf) is already
# cached locally, but transformers' from_pretrained() still does a live HTTP
# HEAD request to huggingface.co to check for updates unless told not to.
# With 64 ranks doing that at once, LUMI's compute-node network egress
# becomes a bottleneck and the job stalls at startup. Force cache-only
# loading — no network calls at all.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Suspected root cause of the host-RAM OOMs killing runs around step 630:
# ps_lazydataset.py's per-worker-process rasterio dataset cache was
# unbounded, so 6 DataLoader workers/rank x 8 ranks/node could each end up
# holding every distinct source raster ever touched open forever (each with
# its own GDAL block cache). The cache is now LRU-bounded in code
# (_RASTERIO_CACHE_MAX), but also cap GDAL's own per-process block cache
# here as a second, independent ceiling — belt and suspenders, and cheap
# (just an env var) compared to another multi-hour test run. 512 (MB;
# GDAL_CACHEMAX values under 100000 are interpreted as MB) x 6 workers x 8
# ranks/node = ~24GB worst case per node, well inside the 256G budget.
export GDAL_CACHEMAX=512

BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
      --bind /scratch/project_465002934:/scratch/project_465002934 \
      --bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/train_sr_fm_refiner.py
CONFIG=config/sr_fm_refiner_v5-1-lumi.yaml
# Checkpoints (best.pth/latest.pth) run tens of GB; the home filesystem
# (/users/mazhanyu, 20G quota) filled up from these and killed a run
# mid-checkpoint-write. Write outputs to the project's Flash storage
# instead, which has far more headroom.
OUTPUT_DIR=/flash/project_465002934/Marigold_output
# The script does os.makedirs(out_dir_run, exist_ok=False) so it never
# clobbers a genuinely separate completed run — but that means a retry of
# THIS run crashes immediately on the directory the previous (killed)
# attempt already created. RUN_DIR mirrors the script's own naming
# (job_name = config filename without extension, no --add_datetime_prefix
# passed here) so the retry loop below can clear it before each retry.
JOB_NAME=$(basename "$CONFIG" .yaml)
RUN_DIR="$OUTPUT_DIR/$JOB_NAME"
export BIND SIF SCRIPT CONFIG OUTPUT_DIR

# Forcing GDR off (NCCL_NET_GDR_LEVEL=0) as a diagnostic only narrowed the
# stall from "all 64 ranks hang" down to "2 of 64 ranks hang" — it didn't
# fix it, so this isn't a GDR/PCIe-path problem. The per-rank NCCL debug
# logs (job 21316942) showed comm init completing cleanly on all 64 ranks
# with no WARN anywhere, then 62/64 ranks sailing through many large
# broadcasts while exactly 2 ranks (different ones each run, on different
# physical nodes each run) silently never get a completion signal for one
# specific ~310MB broadcast. That's consistent with an intermittent
# Slingshot/RDMA completion loss under heavy multi-node collective load,
# not a fixed bad node or a GDR issue. So: leave GDR at the container's
# default (PHB) for full RDMA performance, and instead retry the run — see
# the attempt loop below.
#
# Turn on RCCL/NCCL's own debug logging (one file per rank, since 64 ranks
# interleaved on one stderr is unreadable) so that if it hangs again we can
# see exactly which connection/step it's actually stuck on.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
export NCCL_DEBUG_ROOT=/scratch/project_465002934/nccl_debug
mkdir -p "$NCCL_DEBUG_ROOT"

# Two layers of retry, since the hang has shown up on many different node
# combinations across past jobs (not one consistently bad node) but could
# still, on any given allocation, land on a genuinely bad node:
#
# 1. In-allocation retries (cheap, no queue wait): a fresh attempt reuses
#    this same 8-node allocation but builds new NCCL communicators.
#    train_sr_fm_refiner.py's dist.init_process_group() has a 5-minute
#    collective timeout, so a stuck attempt aborts on its own with a clear
#    "Watchdog caught collective timeout" error instead of hanging until
#    the 3-hour SBATCH limit. Handles a transient connection-setup race.
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
    # (and any other SIGABRT/SIGSEGV) trigger one, and with the process
    # sometimes growing to 90+GB before it crashes (see the memory-leak
    # investigation), a single dump can blow the whole home quota. We debug
    # from Python tracebacks + NCCL_DEBUG logs, not core dumps, so disable
    # them outright.
    ulimit -c 0
    export NCCL_DEBUG_FILE="$NCCL_DEBUG_ROOT/rank_${SLURM_PROCID}_resubmit${RESUBMIT_COUNT}_attempt${ATTEMPT}.log"
    singularity exec $BIND $SIF python -u $SCRIPT --config $CONFIG --output_dir $OUTPUT_DIR
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
