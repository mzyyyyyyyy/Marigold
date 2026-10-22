"""
GIFs that flip between a few years (default: the LiDAR years 2014 and 2019) of the
wall-to-wall region of infer_time_region.py: Landsat, each model's prediction and
the LiDAR CHM, all with the same frame size, timing, colour scale and ground aspect,
so they can be shown side by side.

  <in_dir>/gif/pair_<years>_landsat.gif               Landsat RGB, one 2-98 % stretch for the shown years
  <in_dir>/gif/pair_<years>_<model>.gif               prediction, plasma over [0, vmax] m
  <in_dir>/gif/pair_<years>_lidar.gif                 LiDAR mean-resampled to the prediction grid
                                                      (vis_eval_region.py); grey = no / < --min_valid LiDAR

  singularity exec --bind /flash/project_465002934:/flash/project_465002934 \
      /flash/project_465002934/env/marigold_env.sif python script/depth/make_region_gif.py
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
from rasterio.windows import Window

from make_time_gif import render, save_gif
from vis_eval_region import MODELS, lidar_on_grid

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="/flash/project_465002934/result_time_lumi_denm")
    ap.add_argument("--years", type=int, nargs="+", default=[2014, 2019])
    ap.add_argument("--vmax", type=float, default=28.0)
    ap.add_argument("--min_valid", type=float, default=0.9, help="as vis_eval_region.py")
    ap.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2])
    ap.add_argument("--duration", type=int, default=1200, help="ms per frame")
    args = ap.parse_args()

    with open(os.path.join(args.in_dir, "region.json")) as f:
        R = json.load(f)
    Hc, Wc = R["out_shape_hw"]
    out_t, crs = Affine(*R["out_transform"]), CRS.from_string(R["crs"])
    lat_c = out_t.f + out_t.e * Hc / 2
    cell_aspect = abs(out_t.e) / abs(out_t.a) / math.cos(math.radians(lat_c)) if crs.is_geographic else 1.0
    fs = (9.0, 9.0 * Hc * cell_aspect / Wc + 0.6)
    tag = "pair_" + "_".join(map(str, args.years))
    out_dir = os.path.join(args.in_dir, "gif")
    os.makedirs(out_dir, exist_ok=True)
    vmax, dur = args.vmax, args.duration
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#C8C8C8")

    def write(name, frames):
        path = os.path.join(out_dir, f"{tag}_{name}.gif")
        save_gif(path, frames, dur, dur)
        print("wrote", path)

    # Landsat
    r0, c0, h, w = R["landsat_core_window_rc_hw"]
    ls = []
    for yr in args.years:
        with rasterio.open(R["landsat_paths"][str(yr)]) as src:
            ls.append(src.read([b + 1 for b in args.bands], window=Window(c0, r0, w, h)).astype(np.float32))
    ls = np.stack(ls)
    lo, hi = np.percentile(ls, 2), np.percentile(ls, 98)
    write("landsat", [render(((a - lo) / (hi - lo)).clip(0, 1).transpose(1, 2, 0).repeat(4, 0).repeat(4, 1),
                             f"Landsat  {yr}", figsize=fs, aspect=cell_aspect) for yr, a in zip(args.years, ls)])

    # predictions
    for m in MODELS:
        frames = []
        for yr in args.years:
            with rasterio.open(os.path.join(args.in_dir, m, f"{yr}.tif")) as src:
                p = src.read(1)
            frames.append(render(np.clip(p, 0, vmax), f"{m}  {yr}", cmap=cmap, vmax=vmax,
                                 cbar_label="height (m)", figsize=fs, aspect=cell_aspect))
        write(m, frames)

    # LiDAR (years without a LiDAR file are skipped)
    lidar = {int(os.path.basename(p).split("_")[0]): p for p in R["ref_tifs"]}
    frames = []
    for yr in args.years:
        if yr not in lidar:
            print(f"no LiDAR for {yr}")
            continue
        hgt, frac = lidar_on_grid(lidar[yr], (Hc, Wc), out_t, crs)
        show = np.where((frac >= args.min_valid) & np.isfinite(hgt), np.clip(hgt, 0, vmax), np.nan)
        frames.append(render(show, f"LiDAR CHM  {yr}  (grey: no LiDAR)", cmap=cmap, vmax=vmax,
                             cbar_label="height (m)", figsize=fs, aspect=cell_aspect))
    if frames:
        write("lidar", frames)
