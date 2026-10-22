"""
Build the multi-year (time-series) test set on one Landsat tile that has no GT /
PlanetScope (default: tile 12,53, 2013-2025, /flash/project_465002934/data/Landsat/eur_test2_map).

Same patch layout as the test split of make_spatial_split.py: a non-overlapping
grid of eval_patch x eval_patch patches on the HR (4x Landsat) grid, i.e.
eval_patch/4 Landsat pixels; here --n_patches of the grid cells are drawn at
random (seeded), so no two patches overlap. There is no GT, so the "not >95%
zero" filter of make_spatial_split.py is replaced by: no Landsat pixel is 0
(nodata) in any year. Every patch carries all years.

Each pkl entry has the keys of the spatial-split entries that apply (tile_id,
input_crop_*, input_patch_*, output_crop_* / output_patch_* on the 4x grid,
scale_factor_*) plus 'years', 'input_paths' ({year: Landsat path}) and
'input_offsets' ({year: (dy, dx)}): input_crop_* are on the first year's pixel
grid, and year Y is read at (input_crop_y + dy, input_crop_x + dx).

Needs rasterio: run inside the sif, e.g.
  singularity exec --bind /flash/project_465002934:/flash/project_465002934 \
      /flash/project_465002934/env/marigold_env.sif \
      python script/depth/make_time_test_split.py
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import rasterio
from rasterio.windows import Window

OUT_DIR = "/flash/project_465002934/dataset/CHM_EUR_marigold/train_patch_coord_marigold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/flash/project_465002934/data/Landsat/eur_test2_map")
    ap.add_argument("--tile", default="12,53")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2013, 2026)))
    ap.add_argument("--percentile", default="p50")
    ap.add_argument("--eval_patch", type=int, default=480, help="patch size on the HR (4x Landsat) grid")
    ap.add_argument("--out_scale", type=int, default=4, help="HR / Landsat resolution ratio")
    ap.add_argument("--n_patches", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="time_12-53_v1")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    pkl_path = os.path.join(args.out_dir, f"{args.name}_test.pkl")
    json_path = os.path.join(args.out_dir, f"{args.name}.json")
    existing = [p for p in (pkl_path, json_path) if os.path.exists(p)]
    if existing and not args.force:
        sys.exit(f"Refusing to overwrite (use --force): {existing}")

    paths = {y: os.path.join(args.root, str(y), args.tile, f"landsat_{y}_{args.tile}_{args.percentile}.jp2")
             for y in args.years}
    # Patch coordinates are on the first year's pixel grid. The other years must have
    # the same CRS / resolution and an origin a whole number of pixels away (2024's
    # file starts one column further west); that offset is stored per year, so a
    # patch covers the same ground area in every year.
    shapes, offsets, transform, crs = {}, {}, None, None
    for y, p in paths.items():
        with rasterio.open(p) as src:
            if transform is None:
                transform, crs = src.transform, src.crs
            t = src.transform
            assert src.crs == crs and abs(t.a - transform.a) < 1e-12 and abs(t.e - transform.e) < 1e-12 \
                and t.b == 0 and t.d == 0, f"{y}: CRS / resolution differs"
            dx, dy = (transform.c - t.c) / t.a, (transform.f - t.f) / t.e
            assert abs(dx - round(dx)) < 1e-3 and abs(dy - round(dy)) < 1e-3, f"{y}: grid not pixel-aligned"
            offsets[y] = (int(round(dy)), int(round(dx)))   # year pixel = reference pixel + offset
            shapes[y] = (src.height, src.width)
    # reference-grid area covered by every year
    y_lo = max(max(0, -oy) for oy, _ in offsets.values())
    x_lo = max(max(0, -ox) for _, ox in offsets.values())
    H = min(h - oy for (h, _), (oy, _) in zip(shapes.values(), offsets.values()))
    W = min(w - ox for (_, w), (_, ox) in zip(shapes.values(), offsets.values()))
    print(f"per-year pixel offsets (dy, dx) vs {args.years[0]}: "
          f"{ {y: o for y, o in offsets.items() if o != (0, 0)} or 'none'}")

    P_lr = args.eval_patch // args.out_scale
    assert P_lr * args.out_scale == args.eval_patch
    grid = [(y, x) for y in range(y_lo, H - P_lr + 1, P_lr) for x in range(x_lo, W - P_lr + 1, P_lr)]
    print(f"common Landsat area rows {y_lo}..{H}, cols {x_lo}..{W}; {len(grid)} non-overlapping {P_lr}px cells "
          f"({args.eval_patch}px on the {args.out_scale}x grid)")

    def valid(cell):
        y, x = cell
        for yr, p in paths.items():
            oy, ox = offsets[yr]
            with rasterio.open(p) as src:
                if (src.read(window=Window(x + ox, y + oy, P_lr, P_lr)) == 0).all(axis=0).any():
                    return False
        return True

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(grid))
    chosen, n_rejected = [], 0
    for i in order:
        if len(chosen) == args.n_patches:
            break
        if valid(grid[i]):
            chosen.append(grid[i])
        else:
            n_rejected += 1
    assert len(chosen) == args.n_patches, f"only {len(chosen)} valid cells"
    chosen.sort()

    entries = []
    for y, x in chosen:
        entries.append({
            "tile_id": args.tile, "years": list(args.years), "input_paths": dict(paths),
            "input_offsets": dict(offsets),
            "input_crop_y": y, "input_crop_x": x, "input_patch_height": P_lr, "input_patch_width": P_lr,
            "output_crop_y": y * args.out_scale, "output_crop_x": x * args.out_scale,
            "output_patch_height": args.eval_patch, "output_patch_width": args.eval_patch,
            "scale_factor_x": 1.0 / args.out_scale, "scale_factor_y": 1.0 / args.out_scale,
        })

    # no two patches may overlap
    occ = np.zeros((H, W), dtype=np.uint8)
    for e in entries:
        y, x = e["input_crop_y"], e["input_crop_x"]
        occ[y:y + P_lr, x:x + P_lr] += 1
    assert occ.max() == 1, "patches overlap"

    os.makedirs(args.out_dir, exist_ok=True)
    tmp = pkl_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(entries, f)
    os.replace(tmp, pkl_path)
    info = {"tile": args.tile, "years": list(args.years), "root": args.root, "percentile": args.percentile,
            "landsat_grid_hw": [H, W], "landsat_transform": list(transform)[:6], "crs": str(crs),
            "reference_year": args.years[0], "input_offsets": {str(y): o for y, o in offsets.items()},
            "eval_patch": args.eval_patch, "out_scale": args.out_scale, "landsat_patch": P_lr,
            "n_grid_cells": len(grid), "n_patches": len(entries), "n_rejected_nodata": n_rejected,
            "seed": args.seed, "pkl": pkl_path}
    with open(json_path, "w") as f:
        json.dump(info, f, indent=1)
    print(f"{len(entries)} patches ({n_rejected} cells rejected for nodata); no overlap")
    print(f"wrote {pkl_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
