"""
Height-change maps (later year - earlier year, default 2019 - 2014) of the wall-to-wall
region of infer_time_region.py: the LiDAR CHM and each model's prediction.

Heights are clipped to [0, vmax] m before differencing, as in the change evaluation of
vis_eval_region.py. The LiDAR change is shown only on cells that are valid
(>= --min_valid LiDAR coverage) in BOTH years; elsewhere it is grey, so a difference
in LiDAR coverage is never shown as a height change. The model maps cover the whole
region; the *_lidar_mask.png variants hide the same cells as the LiDAR map, for a
like-for-like comparison. All maps: RdBu over [-dmax, dmax] m (red = lower, blue = higher),
same frame size and ground aspect.

  <in_dir>/png/diff_<y2>-<y1>_lidar.png
  <in_dir>/png/diff_<y2>-<y1>_<model>.png, ..._<model>_lidar_mask.png
  <in_dir>/png/diff_<y2>-<y1>_panel.png       the three side by side (models masked)

  singularity exec --bind /flash/project_465002934:/flash/project_465002934 \
      /flash/project_465002934/env/marigold_env.sif python script/depth/make_region_diff.py
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from vis_eval_region import MODELS, lidar_on_grid


def save_map(path, d, title, dmax, cmap, aspect, figsize):
    fig, (ax, cax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 0.045]})
    im = ax.imshow(d, cmap=cmap, vmin=-dmax, vmax=dmax, interpolation="nearest", aspect=aspect)
    fig.colorbar(im, cax=cax, label="height change (m)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.87, bottom=0.02, top=0.92, wspace=0.04)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="/flash/project_465002934/result_time_lumi_denm")
    ap.add_argument("--years", type=int, nargs=2, default=[2014, 2019], help="earlier later")
    ap.add_argument("--vmax", type=float, default=28.0, help="heights clipped to [0, vmax] before differencing")
    ap.add_argument("--dmax", type=float, default=15.0, help="colour range +-dmax m")
    ap.add_argument("--min_valid", type=float, default=0.9, help="as vis_eval_region.py")
    args = ap.parse_args()
    y1, y2 = args.years

    with open(os.path.join(args.in_dir, "region.json")) as f:
        R = json.load(f)
    Hc, Wc = R["out_shape_hw"]
    out_t, crs = Affine(*R["out_transform"]), CRS.from_string(R["crs"])
    lat_c = out_t.f + out_t.e * Hc / 2
    cell_aspect = abs(out_t.e) / abs(out_t.a) / math.cos(math.radians(lat_c)) if crs.is_geographic else 1.0
    fs = (9.0, 9.0 * Hc * cell_aspect / Wc + 0.6)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#C8C8C8")
    out = os.path.join(args.in_dir, "png")
    os.makedirs(out, exist_ok=True)
    tag = f"diff_{y2}-{y1}"
    clip = lambda a: np.clip(a, 0, args.vmax)

    lidar = {int(os.path.basename(p).split("_")[0]): p for p in R["ref_tifs"]}
    g, ok = {}, {}
    for yr in (y1, y2):
        h, frac = lidar_on_grid(lidar[yr], (Hc, Wc), out_t, crs)
        ok[yr] = (frac >= args.min_valid) & np.isfinite(h)
        g[yr] = clip(np.nan_to_num(h))
    both = ok[y1] & ok[y2]
    d_lidar = np.where(both, g[y2] - g[y1], np.nan)
    print(f"LiDAR valid in both years: {both.mean():.1%} of the region; mean change {np.nanmean(d_lidar):+.2f} m")
    maps = {"lidar": (d_lidar, f"LiDAR CHM  {y2} - {y1}  (grey: not valid in both years)")}
    for m in MODELS:
        p = {}
        for yr in (y1, y2):
            with rasterio.open(os.path.join(args.in_dir, m, f"{yr}.tif")) as src:
                p[yr] = clip(src.read(1))
        d = p[y2] - p[y1]
        maps[m] = (d, f"{m}  {y2} - {y1}")
        maps[f"{m}_lidar_mask"] = (np.where(both, d, np.nan), f"{m}  {y2} - {y1}  (LiDAR cells only)")
        print(f"{m}: mean change {d.mean():+.2f} m (whole region), {d[both].mean():+.2f} m (LiDAR cells)")
    for k, (d, title) in maps.items():
        save_map(os.path.join(out, f"{tag}_{k}.png"), d, title, args.dmax, cmap, cell_aspect, fs)

    # the three side by side, models on the LiDAR cells
    keys = ["lidar"] + [f"{m}_lidar_mask" for m in MODELS]
    labels = [f"LiDAR CHM {y2} - {y1}"] + [f"{m} {y2} - {y1}" for m in MODELS]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6 * Hc * cell_aspect / Wc + 1.0))
    for ax, k, lab in zip(axes, keys, labels):
        im = ax.imshow(maps[k][0], cmap=cmap, vmin=-args.dmax, vmax=args.dmax, interpolation="nearest",
                       aspect=cell_aspect)
        ax.set_title(lab, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.015, label="height change (m)")
    fig.suptitle(f"Height change {y1} -> {y2} on cells with LiDAR in both years (grey: no LiDAR)", fontsize=12)
    fig.savefig(os.path.join(out, f"{tag}_panel.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out, f"{tag}_panel.png"))
