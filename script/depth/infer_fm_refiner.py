"""
Inference script for the Flow Matching CHM Refiner.

For each Landsat tile (LR), slides a patch window over it and produces
a high-resolution CHM prediction at out_scale × input resolution.

Pipeline per patch:
  Landsat_LR  →  [-1,1] normalise  →  upsample ×4  →  landsat_hr
  Landsat_LR  →  DAv2             →  h_coarse (HR)  →  normalise to [-1,1]
  fm_refiner.refine(landsat_hr, h_coarse, n_steps=2)  →  h_fine [-1,1]
  h_fine  →  denormalise  →  metres (blended into output GeoTIFF)

Usage:
  python script/depth/infer_fm_refiner.py --config config/fm_refiner_pred.yaml
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import logging
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


# ---------------------------------------------------------------------------
# Core tile prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_tile(
    lr_path: Path,
    fm_refiner: FMRefiner,
    dav2_model,
    device: torch.device,
    cfg,
    input_stats: dict,
    target_stats: dict,
    blending_mask: np.ndarray,
) -> tuple:
    """
    Predict a single Landsat tile.

    Returns (output_array [H_hr, W_hr], rasterio meta dict).
    """
    patch_size_lr  = cfg.prediction.patch_size_lr   # LR patch size to read
    stride_lr      = cfg.prediction.stride_lr
    out_scale      = cfg.model.out_in_scale_factor   # e.g. 4
    n_steps        = cfg.prediction.n_steps          # FM integration steps
    method         = cfg.prediction.get("method", "euler")
    selected_bands = list(cfg.dataset.selected_bands)
    n_bands        = len(selected_bands)

    # Per-band input normalisation — matches LazyPatchDataset._transform:
    #   p1 = [input_stats["p1"][b] for b in selected_bands]
    p1_in  = np.array([input_stats["p1"][b]  for b in selected_bands], dtype=np.float32)
    p99_in = np.array([input_stats["p99"][b] for b in selected_bands], dtype=np.float32)

    with rasterio.open(lr_path) as src:
        nols = src.meta["width"]
        nrows = src.meta["height"]
        meta = src.meta.copy()
        bounds = src.bounds
        crs = src.crs

        # HR output dimensions
        H_hr = round(nrows * out_scale)
        W_hr = round(nols  * out_scale)

        output     = np.zeros((H_hr, W_hr), dtype=np.float32)
        weight_map = np.zeros((H_hr, W_hr), dtype=np.float32)

        offsets = _get_offsets(nrows, nols, patch_size_lr, stride_lr)
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

        for col_off, row_off in tqdm(offsets, desc=lr_path.name, leave=False):
            win = windows.Window(
                col_off=col_off, row_off=row_off,
                width=patch_size_lr, height=patch_size_lr,
            ).intersection(big_window)

            # Read LR patch: shape (C, H_win, W_win)
            data = src.read(
                indexes=[b + 1 for b in selected_bands],
                window=win,
                out_shape=(n_bands, win.height, win.width),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

            # Normalize to [-1, 1]
            data = _minmax_to_neg1_1(data, p1_in, p99_in)

            # (1, C, H_win, W_win) tensor
            lr_tensor = torch.from_numpy(data).unsqueeze(0).to(device)

            H_win_hr = round(win.height * out_scale)
            W_win_hr = round(win.width  * out_scale)

            # Upsample Landsat to HR for ControlNet
            landsat_hr = F.interpolate(
                lr_tensor, size=(H_win_hr, W_win_hr),
                mode="bilinear", align_corners=False,
            )

            # DAv2 expects [0,1]; dataset stores [-1,1] → convert back
            landsat_for_dav2 = (lr_tensor + 1.0) / 2.0
            h_coarse = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_win_hr, W_win_hr))
            h_coarse = _normalize_coarse(h_coarse, target_stats)

            # FM refinement
            h_fine = fm_refiner.refine(landsat_hr, h_coarse, n_steps=n_steps, method=method)

            # Denormalize to metres
            pred_np = _denormalize(h_fine, target_stats).squeeze()  # (H_win_hr, W_win_hr)

            # HR output coordinates
            r_hr = round(row_off * out_scale)
            c_hr = round(col_off * out_scale)
            h_out = pred_np.shape[0]
            w_out = pred_np.shape[1]

            bld = blending_mask[:h_out, :w_out]
            output[r_hr:r_hr + h_out, c_hr:c_hr + w_out]     += pred_np * bld
            weight_map[r_hr:r_hr + h_out, c_hr:c_hr + w_out] += bld

    # Weighted average
    final = np.divide(output, weight_map, out=np.zeros_like(output), where=weight_map > 0)

    # Build output meta
    out_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, W_hr, H_hr)
    meta.update({
        "count":     1,
        "width":     W_hr,
        "height":    H_hr,
        "dtype":     "float32",
        "crs":       crs,
        "transform": out_transform,
        "compress":  "lzw",
        "driver":    "GTiff",
        "nodata":    -9999.0,
    })
    return final, meta


def write_tif(arr: np.ndarray, meta: dict, out_path: Path):
    arr = np.clip(arr, 0.0, None)   # CHM cannot be negative
    arr[arr == 0] = meta.get("nodata", -9999.0) if False else arr[arr == 0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FM Refiner Inference")
    parser.add_argument("--config", type=str, default="config/fm_refiner_pred_v3.yaml")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    cfg = recursive_load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")
    logging.info(f"device = {device}")

    # ---- Stats ----
    input_stats  = _load_stats(cfg.dataset.input_stats_file,  cfg.dataset.year)
    target_stats = _load_stats(cfg.dataset.target_stats_file, cfg.dataset.year)

    # ---- Build FM Refiner ----
    n_ls = len(cfg.dataset.selected_bands)
    n_ps = cfg.model.get("n_ps_bands", 3)

    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_ls,
        n_ps_bands=n_ps,
        ps_dropout_p=cfg.model.get("ps_dropout_p", 0.3),
        device=str(device),
    )
    fm_refiner = fm_refiner.to(device)
    fm_refiner.eval()

    # ---- Load fine-tuned checkpoint ----
    ckpt_path = cfg.model.fm_refiner_path
    logging.info(f"Loading FM Refiner checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"].to(device)
    logging.info("FM Refiner loaded.")

    # ---- Load DAv2 ----
    dav2_model = load_dav2(
        dav2_path=cfg.model.dav2_pretrained_path,
        backbone=cfg.model.dav2_backbone,
        out_in_scale_factor=cfg.model.out_in_scale_factor,
    )
    dav2_model = dav2_model.to(device)
    logging.info("DAv2 loaded.")

    # ---- Blending mask (at HR patch size) ----
    patch_size_hr = round(cfg.prediction.patch_size_lr * cfg.model.out_in_scale_factor)
    blending_mask = _create_blending_mask(patch_size_hr)

    # ---- Collect input files ----
    # Directory structure: input_dir/<tile_subdir>/<files>.jp2
    # For each tile subdir, pick exactly ONE file at that first level that
    # matches selected_percentile — mirrors LazyPatchDataset._gather_tile_ids:
    #   glob(f'{input_dir}/*/*{selected_percentile[0]}*.{file_type}')
    import glob as _glob
    input_dir = Path(cfg.dataset.input_dir)
    file_ext  = cfg.dataset.file_type_input
    percentile = cfg.dataset.selected_percentile[0]
    matched = _glob.glob(str(input_dir / "*" / f"*{percentile}*.{file_ext}"))
    all_files = sorted(Path(p) for p in matched)

    logging.info(f"Found {len(all_files)} input tile(s) (pattern: */{percentile}*.{file_ext}).")

    # ---- Output directory ----
    out_dir = Path(cfg.dataset.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lr_path in tqdm(all_files, desc="Tiles"):
        out_path = out_dir / f"{lr_path.stem}_fm_refined.tif"
        if out_path.exists():
            logging.info(f"Skipping (exists): {out_path}")
            continue

        logging.info(f"Processing: {lr_path}")
        pred, meta = predict_tile(
            lr_path=lr_path,
            fm_refiner=fm_refiner,
            dav2_model=dav2_model,
            device=device,
            cfg=cfg,
            input_stats=input_stats,
            target_stats=target_stats,
            blending_mask=blending_mask,
        )
        write_tif(pred, meta, out_path)
        logging.info(f"Saved: {out_path}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
