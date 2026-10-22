"""
Colour PNGs of the time-series predictions written by infer_time_series.py.

Reads <in_dir>/<model>/<year>/<id>.tif and writes, under <in_dir>/png/:
  <model>/<year>/<id>.png   the prediction, plasma colour map over [0, vmax] m
                            (same scale as eval_group_meeting.py; values clipped),
                            full 480 px resolution, lossless
  strips/<id>.png           one figure per patch: one row per model, one column per
                            year, with the patch mean height under each panel
CPU only; run inside the sif, see script/depth/vis_time_series.sh.
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image

from eval_detail_metrics_nd import _colormap_u8

MODELS = ["baseline_sr4ir_reg", "sr_fm_refiner_v11-lumi"]


def save_strip(path, preds, pid, years, vmax):
    """preds[model][year] -> (H, W) metres."""
    fig, axes = plt.subplots(len(MODELS), len(years), figsize=(1.7 * len(years) + 1.2, 2.05 * len(MODELS) + 0.5),
                             squeeze=False)
    for i, m in enumerate(MODELS):
        for j, yr in enumerate(years):
            ax = axes[i, j]
            p = preds[m][yr]
            ax.imshow(p, cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"mean {np.clip(p, 0, vmax).mean():.1f} m", fontsize=7, labelpad=2)
            if i == 0:
                ax.set_title(str(yr), fontsize=9)
        axes[i, 0].set_ylabel(m, fontsize=8)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, label="height (m)")
    fig.suptitle(pid, fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def process(job):
    pid, in_dir, years, vmax = job
    out = os.path.join(in_dir, "png")
    preds = {}
    for m in MODELS:
        preds[m] = {}
        for yr in years:
            with rasterio.open(os.path.join(in_dir, m, str(yr), pid + ".tif")) as src:
                p = src.read(1)
            preds[m][yr] = p
            Image.fromarray(_colormap_u8(p, vmax)).save(os.path.join(out, m, str(yr), pid + ".png"))
    save_strip(os.path.join(out, "strips", pid + ".png"), preds, pid, years, vmax)
    return pid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="/flash/project_465002934/result_time_lumi")
    ap.add_argument("--vmax", type=float, default=28.0, help="colour-scale top (m); target p99 of training")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N patches")
    args = ap.parse_args()

    with open(os.path.join(args.in_dir, "meta.json")) as f:
        years = json.load(f)["years"]
    with open(os.path.join(args.in_dir, "index.csv")) as f:
        pids = [r["id"] for r in csv.DictReader(f)]
    if args.limit:
        pids = pids[:args.limit]
    for m in MODELS:
        for yr in years:
            os.makedirs(os.path.join(args.in_dir, "png", m, str(yr)), exist_ok=True)
    os.makedirs(os.path.join(args.in_dir, "png", "strips"), exist_ok=True)

    jobs = [(pid, args.in_dir, years, args.vmax) for pid in pids]
    with Pool(args.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(process, jobs), 1):
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} patches", flush=True)
    print(f"wrote {len(jobs)} x {len(years)} x {len(MODELS)} PNGs and {len(jobs)} strips to "
          f"{os.path.join(args.in_dir, 'png')}")
