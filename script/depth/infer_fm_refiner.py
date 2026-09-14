"""
Inference script for the Flow Matching CHM Refiner.

For each Landsat tile (LR), slides a patch window over it and produces
a high-resolution CHM prediction at out_scale × input resolution.

Pipeline per patch:
  Landsat_LR  →  [-1,1] normalise  →  upsample ×4  →  landsat_hr
  Landsat_LR  →  DAv2             →  h_coarse (HR)  →  normalise to [-1,1]
  fm_refiner.refine(landsat_hr, h_coarse, n_steps=2)  →  h_fine [-1,1]
  h_fine  →  denormalise  →  metres (blended into output GeoTIFF)

Usage (single GPU):
  python script/depth/infer_fm_refiner.py --config config/fm_refiner_pred.yaml

Usage (multi-GPU, e.g. 16 GPUs via srun -n16):
  Every process loads the *full* tile list, then all (tile, patch) work items
  across every tile are flattened and sharded by (global_rank, world_size) —
  read from SLURM_PROCID / SLURM_NTASKS automatically when launched with
  srun (no extra flags needed). This matters when there are far fewer tiles
  than GPUs: sharding by tile alone would leave most GPUs idle, so instead
  the patches *within* each tile are split across all GPUs proportional to
  each tile's patch count (contiguous column-bands, so each rank only needs
  a small local canvas). Each rank writes its local (weighted-sum, weight,
  band-origin) accumulation for the tiles it touched to
  `<output_dir>/.partial/`. The GPU used on each node is derived from
  SLURM_LOCALID mod the number of GPUs visible to the process.

  Once every rank has finished, run a single merge pass (e.g. one more
  `srun -n1` step after the main one) to combine partials into final tifs:
    python script/depth/infer_fm_refiner.py \\
        --config config/fm_refiner_pred.yaml --merge
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import logging
import pickle
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from omegaconf import OmegaConf

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from src.util.config_util import recursive_load_config


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _load_stats(stats_file: str, year: int) -> dict:
    with open(stats_file) as f:
        return json.load(f)[str(year)]


def _minmax_to_neg1_1(x: np.ndarray, p1: np.ndarray, p99: np.ndarray) -> np.ndarray:
    """
    Per-band MinMax → [0,1] → [-1,1], matching LazyPatchDataset.MinMaxScale.

    x   : (C, H, W)  float32
    p1  : (C,)        per-band p1  (indexed by selected_bands, same order as x)
    p99 : (C,)        per-band p99
    """
    p1  = p1.reshape(-1, 1, 1)
    p99 = p99.reshape(-1, 1, 1)
    x = (x - p1) / (p99 - p1 + 1e-8)
    x = np.clip(x, 0.0, 1.0)            # MinMaxScale clips to [0,1] first
    x = x * 2.0 - 1.0                   # scale_input_to_neg1_1
    return x


def _normalize_coarse(h_coarse: torch.Tensor, target_stats: dict) -> torch.Tensor:
    p1  = float(target_stats["p1"][0])
    p99 = float(target_stats["p99"][0])
    h = (h_coarse - p1) / (p99 - p1 + 1e-8)
    h = h * 2.0 - 1.0
    return h.clamp(-1.0, 1.0)


def _denormalize(h_fine: torch.Tensor, target_stats: dict) -> np.ndarray:
    """[-1, 1] → metres using target p1/p99."""
    p1  = float(target_stats["p1"][0])
    p99 = float(target_stats["p99"][0])
    h = (h_fine.float() + 1.0) / 2.0       # [0, 1]
    h = h * (p99 - p1) + p1                 # metres
    return h.cpu().numpy()


def _load_weights(fm_refiner: FMRefiner, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    if fm_refiner.use_controlnet and ckpt.get("controlnet_state") is not None:
        fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    step = ckpt.get("step", -1)
    logging.info(f"Loaded checkpoint: {path}  (step={step})")
    return step


# ---------------------------------------------------------------------------
# Blending helpers  (same Gaussian mask used in predictor_base.py)
# ---------------------------------------------------------------------------

def _create_blending_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    c = size / 2
    mask[int(c), int(c)] = 1.0
    mask = gaussian_filter(mask, sigma=size / 4)
    mask /= mask.max()
    return mask


def _get_offsets(nrows: int, ncols: int, patch_size: int, stride: int):
    row_offs = list(range(0, nrows - patch_size + 1, stride))
    col_offs = list(range(0, ncols - patch_size + 1, stride))
    if nrows % stride != 0:
        row_offs.append(nrows - patch_size)
    if ncols % stride != 0:
        col_offs.append(ncols - patch_size)
    return list(product(col_offs, row_offs))


def _write_tif(arr: np.ndarray, meta: dict, out_path: Path):
    arr = np.clip(arr, 0.0, None)   # CHM cannot be negative
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class FMPredictor:

    def __init__(self, cfg, rank: int = 0, global_rank: int = 0, world_size: int = 1):
        self.cfg = cfg
        self.global_rank = global_rank
        self.world_size = world_size
        # `rank` is the *local* index (e.g. SLURM_LOCALID). Mod by the number
        # of GPUs actually visible to this process so it's correct whether
        # all node GPUs are visible or the launcher already restricts this
        # process to a single GPU.
        if torch.cuda.is_available():
            n_visible = torch.cuda.device_count()
            device_index = rank % n_visible if n_visible > 0 else 0
            self.device = torch.device(f"cuda:{device_index}")
        else:
            self.device = torch.device("cpu")
        logging.info(f"[rank {global_rank}] using device {self.device}")

        patch_size_hr = round(cfg.prediction.patch_size_lr * cfg.model.out_in_scale_factor)
        self.blending_mask = _create_blending_mask(patch_size_hr)

    # ------------------------------------------------------------------
    def build_model(self):
        cfg = self.cfg

        n_ls = len(cfg.dataset.selected_bands)
        n_ps = cfg.model.get("n_ps_bands", 3)

        self.fm_refiner = build_fm_refiner(
            sd_pretrained_path=cfg.model.sd_pretrained_path,
            n_landsat_bands=n_ls,
            n_ps_bands=n_ps,
            ps_dropout_p=cfg.model.get("ps_dropout_p", 0.3),
            device=str(self.device),
        ).to(self.device)
        self.fm_refiner.eval()

        _load_weights(self.fm_refiner, cfg.model.fm_refiner_path)

        self.dav2 = load_dav2(
            dav2_path=cfg.model.dav2_pretrained_path,
            backbone=cfg.model.dav2_backbone,
            out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
        ).to(self.device)
        self.dav2.eval()

        self.input_stats = _load_stats(cfg.dataset.input_stats_file, cfg.dataset.year)
        self.target_stats = _load_stats(cfg.dataset.target_stats_file, cfg.dataset.year)
        logging.info(f"Models ready on {self.device}")

    # ------------------------------------------------------------------
    def load_files(self):
        """Every rank loads the same full tile list — sharding happens at the
        patch level in predict_all(), since tile counts can be far smaller
        than the number of GPUs.

        Directory structure: input_dir/<tile_subdir>/<files>.jp2
        For each tile subdir, pick exactly ONE file at that first level that
        matches selected_percentile — mirrors LazyPatchDataset._gather_tile_ids:
          glob(f'{input_dir}/*/*{selected_percentile[0]}*.{file_type}')
        """
        cfg_data = self.cfg.dataset
        input_dir = Path(cfg_data.input_dir)
        file_ext = cfg_data.file_type_input
        percentile = cfg_data.selected_percentile[0]

        import glob as _glob
        matched = _glob.glob(str(input_dir / "*" / f"*{percentile}*.{file_ext}"))
        # Sorted so every rank derives the same tile order (needed for a
        # deterministic global patch shard).
        self.all_files = sorted((Path(p), Path(p).stem) for p in matched)

        logging.info(f"Found {len(self.all_files)} tiles total (pattern: */{percentile}*.{file_ext}).")
        if self.all_files:
            logging.info(f"Example: {self.all_files[0][0]}")

    # ------------------------------------------------------------------
    def out_dir(self) -> Path:
        return Path(self.cfg.dataset.output_dir)

    # ------------------------------------------------------------------
    def predict_all(self) -> Path:
        """Split each tile's offsets into contiguous column-bands (one band
        per rank assigned to that tile), so each rank only needs a small
        local canvas instead of a full-tile-sized one. Ranks are assigned to
        tiles proportional to each tile's patch count. Each rank writes its
        local (weighted-sum, weight, band-origin) to `<out_dir>/.partial/`.
        Run merge_partials() afterwards (once, from a single process) to
        combine into final tifs.
        """
        cfg = self.cfg
        out_dir = self.out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        partial_dir = out_dir / ".partial"
        partial_dir.mkdir(parents=True, exist_ok=True)

        patch_size = cfg.prediction.patch_size_lr
        stride     = cfg.prediction.stride_lr
        out_scale  = cfg.model.out_in_scale_factor

        # Per-tile offsets for tiles that still need work.
        tiles_meta   = {}   # tile_idx -> (img_path, stem, nrows, ncols)
        tile_offsets = {}   # tile_idx -> offsets list
        for tile_idx, (img_path, stem) in enumerate(self.all_files):
            out_path = out_dir / f"{stem}_fm_refined.tif"
            if out_path.exists():
                continue
            with rasterio.open(img_path) as src:
                nrows, ncols = src.height, src.width
            tile_offsets[tile_idx] = _get_offsets(nrows, ncols, patch_size, stride)
            tiles_meta[tile_idx] = (img_path, stem, nrows, ncols)

        if not tile_offsets:
            logging.info(f"[rank {self.global_rank}] nothing to do (all outputs exist)")
            return out_dir

        # Assign a contiguous range of ranks to each tile, proportional to
        # its patch count (so a big tile gets more ranks than a small one).
        tile_ids     = sorted(tile_offsets.keys())
        total_counts = sum(len(tile_offsets[t]) for t in tile_ids)
        rank_ranges  = {}   # tile_idx -> (rank_start, rank_end)
        cum = 0
        for i, t in enumerate(tile_ids):
            if i == len(tile_ids) - 1:
                n_ranks = self.world_size - cum
            else:
                frac    = len(tile_offsets[t]) / total_counts
                n_ranks = max(1, round(frac * self.world_size))
                n_ranks = min(n_ranks, self.world_size - cum - (len(tile_ids) - 1 - i))
            rank_ranges[t] = (cum, cum + n_ranks)
            cum += n_ranks

        for tile_idx in tile_ids:
            start, end = rank_ranges[tile_idx]
            if not (start <= self.global_rank < end):
                continue
            img_path, stem, nrows, ncols = tiles_meta[tile_idx]
            offsets    = tile_offsets[tile_idx]
            group_size = end - start
            local_idx  = self.global_rank - start
            # Offsets are ordered column-major (product(cols, rows)), so a
            # contiguous chunk is a vertical column-band -> compact bbox.
            chunk_size = -(-len(offsets) // group_size)  # ceil div
            my_offsets = offsets[local_idx * chunk_size:(local_idx + 1) * chunk_size]
            logging.info(
                f"[rank {self.global_rank}] tile {stem}: band {local_idx}/{group_size}, "
                f"{len(my_offsets)} patches"
            )
            if not my_offsets:
                continue
            try:
                result, norm_map, meta, r0, c0 = self._predict_patches(
                    img_path, my_offsets, nrows, ncols, patch_size, out_scale,
                )
                np.savez(
                    partial_dir / f"{stem}__r{self.global_rank}.npz",
                    result=result, norm_map=norm_map, r0=r0, c0=c0,
                )
                meta_path = partial_dir / f"{stem}__meta.pkl"
                if not meta_path.exists():
                    with open(meta_path, "wb") as f:
                        pickle.dump(meta, f)
                logging.info(f"[rank {self.global_rank}] partial saved for {stem}")
            except Exception:
                logging.error(f"Failed on {img_path}")
                traceback.print_exc()

        return out_dir

    # ------------------------------------------------------------------
    def merge_partials(self) -> None:
        """Combine every rank's local (band-shaped) partial into one
        full-size canvas per tile and write final tifs. Run once, from a
        single process, after all compute ranks have finished."""
        out_dir     = self.out_dir()
        partial_dir = out_dir / ".partial"

        if not partial_dir.exists():
            logging.info("No .partial directory found; nothing to merge.")
            return

        stems = sorted({p.stem.split("__r")[0] for p in partial_dir.glob("*__r*.npz")})
        logging.info(f"Merging {len(stems)} tile(s) from {partial_dir}")

        for stem in stems:
            out_path = out_dir / f"{stem}_fm_refined.tif"
            if out_path.exists():
                continue

            parts = sorted(partial_dir.glob(f"{stem}__r*.npz"))
            meta_path = partial_dir / f"{stem}__meta.pkl"
            if not parts or not meta_path.exists():
                logging.warning(f"Skip {stem}: missing partial parts or meta")
                continue

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            out_h, out_w = meta["height"], meta["width"]
            result   = np.zeros((out_h, out_w), dtype=np.float32)
            norm_map = np.zeros((out_h, out_w), dtype=np.float32)

            for part_path in parts:
                with np.load(part_path) as npz:
                    r0, c0 = int(npz["r0"]), int(npz["c0"])
                    h, w   = npz["result"].shape
                    result[r0:r0 + h, c0:c0 + w]   += npz["result"]
                    norm_map[r0:r0 + h, c0:c0 + w] += npz["norm_map"]

            final = np.divide(result, norm_map, out=np.zeros_like(result), where=norm_map != 0)
            _write_tif(final, meta, out_path)
            logging.info(f"Saved: {out_path}")

            for part_path in parts:
                part_path.unlink()
            meta_path.unlink()

        try:
            next(partial_dir.iterdir())
        except StopIteration:
            partial_dir.rmdir()

    # ------------------------------------------------------------------
    def _predict_patches(self, img_path, offsets, nrows, ncols, patch_size, out_scale):
        """Accumulate weighted-sum + weight over the given `offsets` (a
        column-band subset of the tile's full offset list) into a *local*
        canvas sized to just that band's bounding box (plus patch overhang).
        Division is deferred to merge_partials() so bands from different
        ranks can be placed and summed on the full canvas first. Returns
        (result, norm_map, meta, r0, c0) where (r0, c0) is the band's origin
        in full-canvas output pixel coordinates."""
        cfg = self.cfg
        cfg_data = cfg.dataset
        selected_bands = list(cfg_data.selected_bands)
        n_bands = len(selected_bands)
        n_steps = cfg.prediction.n_steps
        method  = cfg.prediction.get("method", "euler")

        p1_in  = np.array([self.input_stats["p1"][b]  for b in selected_bands], dtype=np.float32)
        p99_in = np.array([self.input_stats["p99"][b] for b in selected_bands], dtype=np.float32)

        with rasterio.open(img_path) as src:
            meta   = src.meta.copy()
            bounds = src.bounds
            crs    = src.crs

            out_h = round(nrows * out_scale)
            out_w = round(ncols * out_scale)
            out_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, out_w, out_h)
            meta.update({
                "count": 1, "width": out_w, "height": out_h, "dtype": "float32",
                "crs": crs, "transform": out_transform, "compress": "lzw",
                "driver": "GTiff", "nodata": -9999.0,
            })

            out_patch = round(patch_size * out_scale)
            cols_off  = [c for c, _ in offsets]
            rows_off  = [r for _, r in offsets]
            r0 = round(min(rows_off) * out_scale)
            c0 = round(min(cols_off) * out_scale)
            r1 = min(out_h, round(max(rows_off) * out_scale) + out_patch)
            c1 = min(out_w, round(max(cols_off) * out_scale) + out_patch)
            result   = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
            norm_map = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)

            big_window = windows.Window(0, 0, ncols, nrows)
            bld_mask   = self.blending_mask
            n_offsets  = len(offsets)

            for i, (col_off, row_off) in enumerate(offsets):
                if i % 100 == 0:
                    logging.info(f"[rank {self.global_rank}]   patch {i}/{n_offsets}")
                win = windows.Window(
                    col_off=col_off, row_off=row_off,
                    width=patch_size, height=patch_size,
                ).intersection(big_window)

                data = src.read(
                    indexes=[b + 1 for b in selected_bands],
                    window=win,
                    out_shape=(n_bands, win.height, win.width),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)

                pad_h = patch_size - win.height
                pad_w = patch_size - win.width
                if pad_h > 0 or pad_w > 0:
                    data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

                data = _minmax_to_neg1_1(data, p1_in, p99_in)
                lr_tensor = torch.from_numpy(data).unsqueeze(0).to(self.device)

                H_out = round(win.height * out_scale)
                W_out = round(win.width  * out_scale)
                H_win_hr = round(patch_size * out_scale)
                W_win_hr = round(patch_size * out_scale)

                with torch.no_grad():
                    landsat_hr = F.interpolate(
                        lr_tensor, size=(H_win_hr, W_win_hr),
                        mode="bilinear", align_corners=False,
                    )
                    landsat_for_dav2 = (lr_tensor + 1.0) / 2.0
                    h_coarse = run_dav2(self.dav2, landsat_for_dav2, target_size=(H_win_hr, W_win_hr))
                    h_coarse = _normalize_coarse(h_coarse, self.target_stats)

                    h_fine = self.fm_refiner.refine(landsat_hr, h_coarse, n_steps=n_steps, method=method)

                pred_np = _denormalize(h_fine, self.target_stats).squeeze()[:H_out, :W_out]

                r_local = round(win.row_off * out_scale) - r0
                c_local = round(win.col_off * out_scale) - c0
                bm = bld_mask[:H_out, :W_out]

                result[r_local:r_local + H_out, c_local:c_local + W_out]   += pred_np * bm
                norm_map[r_local:r_local + H_out, c_local:c_local + W_out] += bm

        return result, norm_map, meta, r0, c0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FM Refiner Inference")
    parser.add_argument("--config", type=str, default="config/fm_refiner_pred_v3.yaml")
    parser.add_argument("--rank", type=int, default=None,
                        help="Local GPU index on this node. Defaults to SLURM_LOCALID (or 0).")
    parser.add_argument("--global_rank", type=int, default=None,
                        help="Global process rank, used to shard tiles/patches across all GPUs. "
                             "Defaults to SLURM_PROCID (or 0).")
    parser.add_argument("--world_size", type=int, default=None,
                        help="Total number of parallel processes. Defaults to SLURM_NTASKS (or 1).")
    parser.add_argument("--merge", action="store_true",
                        help="Merge partial results from .partial/ into final tifs. "
                             "Run once, from a single process, after all compute ranks finish.")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = recursive_load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    local_rank  = args.rank        if args.rank        is not None else int(os.environ.get("SLURM_LOCALID", 0))
    global_rank = args.global_rank if args.global_rank  is not None else int(os.environ.get("SLURM_PROCID", 0))
    world_size  = args.world_size  if args.world_size   is not None else int(os.environ.get("SLURM_NTASKS", 1))

    predictor = FMPredictor(cfg, rank=local_rank, global_rank=global_rank, world_size=world_size)

    if args.merge:
        predictor.merge_partials()
        return

    predictor.build_model()
    predictor.load_files()
    out_dir = predictor.predict_all()
    print(f"PATH_OUTPUT: {out_dir}")


if __name__ == "__main__":
    main()
