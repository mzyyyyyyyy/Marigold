#!/bin/bash
#SBATCH --job-name=diag_fdiff
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --account=project_465002934
#SBATCH --output=diag_fdiff_%j.out
#SBATCH --error=diag_fdiff_%j.err

# Spatial maps of the real-vs-pseudo PS TDP feature / height difference, see
# script/depth/diag_feat_diff.py. Runs v13-nd, then v12-nd, on one GPU.
#   sbatch script/depth/diag_feat_diff.sh
#   sbatch --export=ALL,RUNS="/flash/project_465002934/Marigold_output/sr_fm_refiner_v13-lumi-nd",EXTRA="--n_patches 4 --n_panels 1" script/depth/diag_feat_diff.sh
OUT=/flash/project_465002934/Marigold_output
RUNS=${RUNS:-"$OUT/sr_fm_refiner_v13-lumi-nd $OUT/sr_fm_refiner_v12-lumi-nd"}
EXTRA=${EXTRA:-}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GDAL_CACHEMAX=512

BIND="--bind /var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
      --bind /scratch/project_465002934:/scratch/project_465002934 \
      --bind /flash/project_465002934:/flash/project_465002934"
SIF=/flash/project_465002934/env/marigold_env.sif
OVERLAY=/flash/project_465002934/env/py_overlay_tf517   # transformers 5.17 (CHMv2)
SCRIPT=/users/mazhanyu/Projects/Marigold/script/depth/diag_feat_diff.py

ulimit -c 0
for RUN in $RUNS; do
  singularity exec $BIND --env PYTHONPATH=$OVERLAY $SIF python -u $SCRIPT --run $RUN $EXTRA || exit 1
done
