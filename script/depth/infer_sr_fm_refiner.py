"""
Large-scale tile inference using a trained SR + FM Refiner checkpoint.

Architecture mirrors script/depth/infer_rs.py:
  - load YAML config
  - build predictor
  - load_files / predict_all

Key differences from the Marigold predictor (predictor_base.py):
  - Uses sr_fm_refiner (DAv2 coarse → SR pseudo-PS → FM refinement)
  - n_average=5 stochastic runs are averaged per patch
  - Output is 4× the Landsat input resolution (out_scale=4)

Usage:
  python script/depth/infer_sr_fm_refiner.py \\
      --config config/sr_fm_refiner_v5_infer_ls.yaml \\
      --rank 0
"""

import argparse
import json
import logging
import os
import sys
import traceback
from itertools import product
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from depthfm.sr_module import SRModule, build_sr_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_target_stats(stats_file: str, year: int) -> dict:
    with open(stats_file) as f:
        return json.load(f)[str(year)]


def _normalize_coarse(h_coarse: torch.Tensor, target_stats: dict) -> torch.Tensor:
    p1  = float(target_stats["p1"][0])
    p99 = float(target_stats["p99"][0])
    h   = (h_coarse - p1) / (p99 - p1 + 1e-8)
    return (h * 2.0 - 1.0).clamp(-1.0, 1.0)


def _load_weights(fm_refiner: FMRefiner, sr_module: SRModule, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    sr_module.load_state_dict(ckpt["sr_state"])
    step = ckpt.get("step", -1)
    logging.info(f"Loaded checkpoint: {path}  (step={step})")
    return step


def _create_blending_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    c = size / 2
    mask[int(c), int(c)] = 1.0
    blending = gaussian_filter(mask, sigma=size / 4)
    blending /= blending.max()
    return blending


def _get_offsets(nrows: int, ncols: int, patch_size: int, stride: int):
    rows = list(range(0, nrows - patch_size + 1, stride))
    cols = list(range(0, ncols - patch_size + 1, stride))
    if nrows % stride != 0:
        rows.append(nrows - patch_size)
    if ncols % stride != 0:
        cols.append(ncols - patch_size)
    return list(product(cols, rows))


def _write_tif(mask: np.ndarray, meta: dict, out_path: Path, data_type: str):
    meta = meta.copy()
    meta.update({
        "count":    1,
        "compress": "lzw",
        "driver":   "GTiff",
        "nodata":   65535,
        "dtype":    data_type,
    })
    mask[mask < 0] = 0
    assert mask.ndim == 2
    if data_type == "float32":
        mask = mask.astype(np.float32)
    elif data_type == "uint16":
        mask = mask.astype(np.uint16)
    else:
        raise NotImplementedError(data_type)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask, 1)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class SRFMPredictor:

    def __init__(self, config: dict, rank: int = 1):
        self.config = config
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        data_cfg = config["data"]
        out_patch_size = round(data_cfg["patch_size"] * data_cfg["out_scale"])
        self.blending_mask = _create_blending_mask(out_patch_size)
        logging.info(f"Blending mask shape: {self.blending_mask.shape}")

        # Global min-max for Landsat input normalisation
        if data_cfg.get("use_gb_minmax", False):
            with open(data_cfg["globalnorm_stats_file"]) as f:
                all_stats = json.load(f)
            gb_stats = all_stats[str(data_cfg["year"])]
            self.input_p1  = float(gb_stats["p1"][0])
            self.input_p99 = float(gb_stats["p99"][0])
        else:
            self.input_p1 = None
            self.input_p99 = None

    # ------------------------------------------------------------------
    def build_model(self):
        cfg       = self.config
        model_cfg = cfg["model"]

        train_cfg = OmegaConf.load(model_cfg["config_path"])

        n_landsat = len(cfg["data"]["selected_bands"])
        n_ps      = len(train_cfg.dataset.selected_bands_hr)

        self.fm_refiner = build_fm_refiner(
            sd_pretrained_path=train_cfg.model.sd_pretrained_path,
            n_landsat_bands=n_landsat,
            n_ps_bands=n_ps,
            ps_dropout_p=0.0,
            bridge_sigma=model_cfg["bridge_sigma"],
            concat_z_coarse=train_cfg.trainer.get("concat_z_coarse", False),
            device=str(self.device),
        ).to(self.device)

        self.dav2 = load_dav2(
            dav2_path=model_cfg["dav2_pretrained_path"],
            backbone=model_cfg["dav2_backbone"],
            out_in_scale_factor=model_cfg["dav2_out_in_scale_factor"],
        ).to(self.device)

        self.sr_module = build_sr_module(
            OmegaConf.to_container(train_cfg.sr_module, resolve=True)
        ).to(self.device)

        _load_weights(self.fm_refiner, self.sr_module, model_cfg["ckpt_path"])
        self.fm_refiner.eval()
        self.sr_module.eval()
        self.dav2.eval()

        self.target_stats = _load_target_stats(
            model_cfg["target_stats_file"], cfg["data"]["year"]
        )
        logging.info(f"Models ready on {self.device}")

    # ------------------------------------------------------------------
    def load_files(self):
        data_cfg   = self.config["data"]
        input_dir  = Path(data_cfg["input_dir"])

        with open(data_cfg["grid_id_file"]) as f:
            tile_info = json.load(f)
        geojson_ids = {feat["properties"]["id"] for feat in tile_info["features"]}

        selected = set(data_cfg.get("selected_test_tile") or geojson_ids)

        filtered = []
        for p in input_dir.rglob(f"*.{data_cfg['file_type']}"):
            name = p.stem
            if any(tid in name for tid in selected):
                if (data_cfg["selected_percentile"][0] in name
                        and data_cfg.get("selected_resolution", "") in name
                        and not any(ex in name for ex in data_cfg["exclude_strings"])
                        and not any(ex in p.parts for ex in data_cfg["exclude_list"])):
                    filtered.append(p)

        self.all_files = [(p, p.stem) for p in filtered]
        logging.info(f"Found {len(self.all_files)} tiles to predict")
        if self.all_files:
            logging.info(f"Example: {self.all_files[0][0]}")

    # ------------------------------------------------------------------
    def predict_all(self) -> Path:
        cfg        = self.config
        data_cfg   = cfg["data"]
        model_cfg  = cfg["model"]
        pred_cfg   = cfg["prediction"]

        year_folder = Path(data_cfg["input_dir"]).name
        ckpt_stem   = Path(model_cfg["ckpt_path"]).stem
        suffix      = data_cfg["output_suffix"].strip()
        out_dir     = Path(data_cfg["output_dir"]) / year_folder / f"{ckpt_stem}_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)

        bands        = data_cfg["selected_bands"]
        patch_size   = data_cfg["patch_size"]
        stride       = data_cfg["stride"]
        out_scale    = data_cfg["out_scale"]
        n_average    = model_cfg["n_average"]
        n_steps      = model_cfg["n_steps"]
        method       = model_cfg["method"]

        for img_path, stem in tqdm(self.all_files, desc="Tiles"):
            out_path = out_dir / f"{stem}_pred.{pred_cfg['file_type']}"
            if out_path.exists():
                logging.info(f"Skip (exists): {out_path}")
                continue

            try:
                result, meta = self._predict_tile(
                    img_path, bands, patch_size, stride, out_scale,
                    n_average, n_steps, method, pred_cfg["operator"],
                )
                _write_tif(result, meta, out_path, pred_cfg["data_type"])
                logging.info(f"Saved: {out_path}")
            except Exception:
                logging.error(f"Failed on {img_path}")
                traceback.print_exc()

        return out_dir

    # ------------------------------------------------------------------
    def _normalise_input(self, patch: np.ndarray) -> np.ndarray:
        """Scale to [-1, 1] using global p1/p99 stats."""
        if self.input_p1 is not None:
            patch = (patch - self.input_p1) / (self.input_p99 - self.input_p1 + 1e-8)
        return np.clip(patch * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    def _predict_tile(
        self, img_path, bands, patch_size, stride, out_scale,
        n_average, n_steps, method, operator,
    ):
        with rasterio.open(img_path) as src:
            ncols, nrows = src.meta["width"], src.meta["height"]
            meta         = src.meta.copy()
            bounds       = src.bounds
            crs          = src.crs

            out_h = round(nrows * out_scale)
            out_w = round(ncols * out_scale)
            result   = np.zeros((out_h, out_w), dtype=np.float32)
            norm_map = np.zeros((out_h, out_w), dtype=np.float32)

            output_transform = from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top, out_w, out_h
            )
            meta.update({"width": out_w, "height": out_h,
                         "transform": output_transform, "crs": crs})

            offsets   = _get_offsets(nrows, ncols, patch_size, stride)
            big_win   = windows.Window(0, 0, ncols, nrows)
            bld_mask  = self.blending_mask

            for col_off, row_off in tqdm(offsets, desc="  Patches", leave=False):
                win = windows.Window(
                    col_off=col_off, row_off=row_off,
                    width=patch_size, height=patch_size,
                ).intersection(big_win)

                raw = src.read(
                    indexes=[b + 1 for b in bands],
                    window=win,
                    out_shape=(len(bands), win.height, win.width),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)          # [C, H, W]

                # Pad to patch_size if at edge
                pad_h = patch_size - win.height
                pad_w = patch_size - win.width
                if pad_h > 0 or pad_w > 0:
                    raw = np.pad(raw, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

                patch_norm = self._normalise_input(raw)  # [C, H, W]  in [-1, 1]

                lr = torch.from_numpy(patch_norm).unsqueeze(0).to(self.device)  # [1, C, H, W]

                H_out = round(win.height * out_scale)
                W_out = round(win.width  * out_scale)

                with torch.no_grad():
                    landsat_for_dav2 = (lr + 1.0) / 2.0
                    h_coarse = run_dav2(
                        self.dav2, landsat_for_dav2,
                        target_size=(round(patch_size * out_scale),
                                     round(patch_size * out_scale)),
                    )
                    h_coarse = _normalize_coarse(h_coarse, self.target_stats)
                    pseudo_ps = self.sr_module(
                        lr,
                        target_size=(round(patch_size * out_scale),
                                     round(patch_size * out_scale)),
                    )

                    preds = [
                        self.fm_refiner.refine_with_ps(
                            pseudo_ps, h_coarse, n_steps=n_steps, method=method
                        )
                        for _ in range(n_average)
                    ]
                    h_fine = torch.stack(preds).mean(dim=0)  # [1, 1, H_out, W_out]

                # Crop to actual output window size
                pred_np = h_fine[0, 0, :H_out, :W_out].cpu().float().numpy()

                # Inverse [-1,1] → physical units (p1..p99 range)
                p1  = float(self.target_stats["p1"][0])
                p99 = float(self.target_stats["p99"][0])
                pred_phys = ((pred_np + 1.0) / 2.0) * (p99 - p1) + p1

                # Blending
                r_off = round(win.row_off * out_scale)
                c_off = round(win.col_off * out_scale)
                bm    = bld_mask[:H_out, :W_out]

                if operator == "blend":
                    result[r_off:r_off + H_out, c_off:c_off + W_out]   += pred_phys * bm
                    norm_map[r_off:r_off + H_out, c_off:c_off + W_out] += bm
                else:
                    result[r_off:r_off + H_out, c_off:c_off + W_out] = pred_phys

        if operator == "blend":
            result = np.divide(result, norm_map, out=np.zeros_like(result), where=norm_map != 0)

        return result, meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SR+FM Refiner large-scale inference")
    parser.add_argument("--config", type=str,
                        default="config/sr_fm_refiner_v5_infer_ls.yaml")
    parser.add_argument("--rank",   type=int, default=1)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args      = parse_args()
    config    = load_yaml_config(args.config)
    predictor = SRFMPredictor(config, rank=args.rank)
    predictor.build_model()
    predictor.load_files()
    out_dir = predictor.predict_all()
    print(f"PATH_OUTPUT: {out_dir}")


if __name__ == "__main__":
    main()
