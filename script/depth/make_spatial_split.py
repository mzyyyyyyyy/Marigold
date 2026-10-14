"""
Build the leak-free spatial train/val/test split for the LUMI canopy-height
data (see src/util/spatial_split.py for the scheme). Run once; training only
loads the result.

Reads the old random-crop pool (dataset.pool_coord_path) and writes, next to
it (or into --out_dir):

  <name>_train.pkl   random (overlapping) crops of the pool inside train blocks
  <name>_val.pkl     non-overlapping eval_patch grid inside val blocks
  <name>_test.pkl    non-overlapping eval_patch grid inside test blocks
  <name>.json        block assignment, counts, parameters

Each pkl is a plain list of patch-coordinate dicts, the same format as the old
pool, so LazyPatchDataset loads them directly. val/test entries carry two extra
keys, 'p75' and 'height_group' (index into dataset.target_range_edges), used
for per-height-range test metrics.

Eval grid patches get the same filters as the training pool, but no height
quota (val/test keep the natural height distribution): inside the tile, and
not >95% zero. NaN counts as zero here, because the dataset maps NaN -> 0 on
load.

Needs rasterio: run inside the sif, e.g.
  singularity exec --bind /flash/project_465002934:/flash/project_465002934 \
      /flash/project_465002934/env/marigold_env.sif \
      python script/depth/make_spatial_split.py --config config/baseline_chmv2-lumi-nd.yaml
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import rasterio

from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset
from src.util.spatial_split import SPLITS, build_split

ZERO_FRAC_MAX = 0.95  # same rule as LazyPatchDataset._process_tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline_chmv2-lumi-nd.yaml")
    parser.add_argument("--name", default="split_spatial_v1")
    parser.add_argument("--out_dir", default=None, help="default: directory of dataset.pool_coord_path")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()

    d = recursive_load_config(args.config).dataset
    out_dir = args.out_dir or os.path.dirname(d.pool_coord_path)
    paths = {s: os.path.join(out_dir, f"{args.name}_{s}.pkl") for s in SPLITS}
    json_path = os.path.join(out_dir, f"{args.name}.json")
    existing = [p for p in [*paths.values(), json_path] if os.path.exists(p)]
    if existing and not args.force:
        sys.exit(f"Refusing to overwrite (use --force): {existing}")

    with open(d.pool_coord_path, "rb") as f:
        pool = pickle.load(f)
    print(f"Loaded pool: {len(pool)} patches from {d.pool_coord_path}")

    edges = list(d.target_range_edges)
    dims_cache = {}

    def tile_dims(t):
        if t["tile_id"] not in dims_cache:
            dims = []
            for key in ("input_path", "output_path", "input_path_hr"):
                with rasterio.open(t[key]) as src:
                    dims.append((src.height, src.width))
            dims_cache[t["tile_id"]] = dims
        return dims_cache[t["tile_id"]]

    def make_eval_patch(t, y, x):
        P = int(d.eval_patch)
        (in_h, in_w), (out_h, out_w), (hr_h, hr_w) = tile_dims(t)
        if y + P > out_h or x + P > out_w or y + P > hr_h or x + P > hr_w:
            return None
        sy, sx = t["scale_factor_y"], t["scale_factor_x"]
        y_in, x_in = round(y * sy), round(x * sx)
        h_in, w_in = round(P * sy), round(P * sx)
        if y_in + h_in > in_h or x_in + w_in > in_w:
            return None
        patch = np.nan_to_num(
            LazyPatchDataset._load_output_patch(t["output_path"], y, x, P, P), nan=0.0, posinf=0.0, neginf=0.0
        )
        if np.mean(patch == 0) > ZERO_FRAC_MAX:
            return None
        p75 = float(np.percentile(patch, 75))
        c = dict(t)
        c.update(
            input_crop_y=y_in, input_crop_x=x_in, input_patch_height=h_in, input_patch_width=w_in,
            input_crop_y_hr=y, input_crop_x_hr=x, input_patch_height_hr=P, input_patch_width_hr=P,
            output_crop_y=y, output_crop_x=x, output_patch_height=P, output_patch_width=P,
            p75=p75, height_group=int(np.digitize([p75], edges[1:-1])[0]),
        )
        return c

    splits, info = build_split(
        pool,
        train_split=float(d.train_split),
        val_split=float(d.val_split),
        seed=int(d.split_seed),
        block_size=int(d.block_size),
        margin=int(d.block_margin),
        eval_patch=int(d.eval_patch),
        make_eval_patch=make_eval_patch,
    )
    info["pool_coord_path"] = d.pool_coord_path
    info["target_range_edges"] = edges
    info["height_group_counts"] = {
        s: np.bincount([p["height_group"] for p in splits[s]], minlength=len(edges) - 1).tolist()
        for s in ("val", "test")
    }

    os.makedirs(out_dir, exist_ok=True)
    for s in SPLITS:
        tmp = paths[s] + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(splits[s], f)
        os.replace(tmp, paths[s])
    with open(json_path, "w") as f:
        json.dump(info, f, indent=1)

    print("Overlap checks passed (across splits; within val; within test).")
    print(f"patches: {info['n_patches']}  blocks: {info['n_blocks']}")
    print(f"dropped from pool (crossing block border): {info['n_pool_dropped_crossing_block_border']}")
    print(f"eval grid patches rejected (out of tile / mostly empty): {info['n_eval_grid_rejected']}")
    print(f"eval height-group counts: {info['height_group_counts']}")
    for s in SPLITS:
        print(f"wrote {paths[s]}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
