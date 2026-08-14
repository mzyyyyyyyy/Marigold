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

Usage (single GPU):
  python script/depth/infer_sr_fm_refiner.py \\
      --config config/sr_fm_refiner_v5_infer_ls.yaml \\
      --rank 0

Usage (multi-GPU, e.g. 16 GPUs via srun -n16):
  Every process loads the *full* tile list, then all (tile, patch) work items
  across every tile are flattened into one global list and sharded by
  (global_rank, world_size) — read from SLURM_PROCID / SLURM_NTASKS
  automatically when launched with srun (no extra flags needed). This
  matters when there are far fewer tiles than GPUs (e.g. 2 tiles, 16 GPUs):
  sharding by tile would leave most GPUs idle, so instead the patches
  *within* each tile are split across all GPUs. Each rank writes its partial
  (weighted-sum, weight) accumulation for the tiles it touched to
  `<out_dir>/.partial/`. The GPU used on each node is derived from
  SLURM_LOCALID mod the number of GPUs visible to the process, so this works
  whether ROCR_VISIBLE_DEVICES exposes all GPUs on the node or just one.

  Once every rank has finished, run a single merge pass (e.g. one more
  `srun -n1` step after the main one) to combine partials into final tifs:
    python script/depth/infer_sr_fm_refiner.py \\
        --config config/sr_fm_refiner_v5_infer_ls.yaml --merge
"""

import argparse
import json
import logging
import os
import pickle
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

    def __init__(self, config: dict, rank: int = 1, global_rank: int = 0, world_size: int = 1):
        self.config = config
        self.global_rank = global_rank
        self.world_size = world_size
        # `rank` is the *local* index (e.g. SLURM_LOCALID). Mod by the number
        # of GPUs actually visible to this process so it's correct whether
        # all node GPUs are visible or ROCR_VISIBLE_DEVICES already restricts
        # this process to a single GPU.
        if torch.cuda.is_available():
            n_visible = torch.cuda.device_count()
            device_index = rank % n_visible if n_visible > 0 else 0
            self.device = torch.device(f"cuda:{device_index}")
        else:
            self.device = torch.device("cpu")

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
        # Checkpoints trained before sr_fm_refiner_v5 used ControlNet
        # conditioning = concat(landsat_hr, ps) instead of ps-only. Set
        # model.concat_landsat_cond: true in the infer yaml for those.
        self.concat_landsat_cond = bool(model_cfg.get("concat_landsat_cond", False))

        self.fm_refiner = build_fm_refiner(
            sd_pretrained_path=train_cfg.model.sd_pretrained_path,
            n_landsat_bands=n_landsat,
            n_ps_bands=n_ps,
            ps_dropout_p=0.0,
            bridge_sigma=model_cfg["bridge_sigma"],
            concat_z_coarse=train_cfg.trainer.get("concat_z_coarse", False),
            concat_landsat_cond=self.concat_landsat_cond,
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
        """Every rank loads the same full tile list — sharding happens at the
        patch level in predict_all(), since tile counts can be far smaller
        than the number of GPUs."""
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

        # Sorted so every rank derives the same tile order (needed for a
        # deterministic global patch shard).
        self.all_files = sorted(((p, p.stem) for p in filtered), key=lambda x: x[1])
        logging.info(f"Found {len(self.all_files)} tiles total")
        if self.all_files:
            logging.info(f"Example: {self.all_files[0][0]}")

    # ------------------------------------------------------------------
    def out_dir(self) -> Path:
        data_cfg  = self.config["data"]
        model_cfg = self.config["model"]
        year_folder = Path(data_cfg["input_dir"]).name
        ckpt_stem   = Path(model_cfg["ckpt_path"]).stem
        suffix      = data_cfg["output_suffix"].strip()
        return Path(data_cfg["output_dir"]) / year_folder / f"{ckpt_stem}_{suffix}"

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
        cfg        = self.config
        data_cfg   = cfg["data"]
        model_cfg  = cfg["model"]
        pred_cfg   = cfg["prediction"]

        out_dir     = self.out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        partial_dir = out_dir / ".partial"
        partial_dir.mkdir(parents=True, exist_ok=True)

        bands        = data_cfg["selected_bands"]
        patch_size   = data_cfg["patch_size"]
        stride       = data_cfg["stride"]
        out_scale    = data_cfg["out_scale"]
        n_average    = model_cfg["n_average"]
        n_steps      = model_cfg["n_steps"]
        method       = model_cfg["method"]
        operator     = pred_cfg["operator"]

        # Per-tile offsets for tiles that still need work.
        tiles_meta   = {}   # tile_idx -> (img_path, stem, nrows, ncols)
        tile_offsets = {}   # tile_idx -> offsets list
        for tile_idx, (img_path, stem) in enumerate(self.all_files):
            out_path = out_dir / f"{stem}_pred.{pred_cfg['file_type']}"
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
            offsets     = tile_offsets[tile_idx]
            group_size  = end - start
            local_idx   = self.global_rank - start
            # Offsets are ordered column-major (product(cols, rows)), so a
            # contiguous chunk is a vertical column-band -> compact bbox.
            chunk_size  = -(-len(offsets) // group_size)  # ceil div
            my_offsets  = offsets[local_idx * chunk_size:(local_idx + 1) * chunk_size]
            logging.info(
                f"[rank {self.global_rank}] tile {stem}: band {local_idx}/{group_size}, "
                f"{len(my_offsets)} patches"
            )
            if not my_offsets:
                continue
            try:
                result, norm_map, meta, r0, c0 = self._predict_patches(
                    img_path, my_offsets, nrows, ncols, bands, patch_size, out_scale,
                    n_average, n_steps, method, operator,
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
        cfg        = self.config
        pred_cfg   = cfg["prediction"]
        out_dir     = self.out_dir()
        partial_dir = out_dir / ".partial"

        if not partial_dir.exists():
            logging.info("No .partial directory found; nothing to merge.")
            return

        stems = sorted({p.stem.split("__r")[0] for p in partial_dir.glob("*__r*.npz")})
        logging.info(f"Merging {len(stems)} tile(s) from {partial_dir}")

        for stem in stems:
            out_path = out_dir / f"{stem}_pred.{pred_cfg['file_type']}"
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
            _write_tif(final, meta, out_path, pred_cfg["data_type"])
            logging.info(f"Saved: {out_path}")

            for part_path in parts:
                part_path.unlink()
            meta_path.unlink()

        try:
            next(partial_dir.iterdir())
        except StopIteration:
            partial_dir.rmdir()

    # ------------------------------------------------------------------
    def _normalise_input(self, patch: np.ndarray) -> np.ndarray:
        """Scale to [-1, 1] using global p1/p99 stats."""
        if self.input_p1 is not None:
            patch = (patch - self.input_p1) / (self.input_p99 - self.input_p1 + 1e-8)
        return np.clip(patch * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    def _predict_patches(
        self, img_path, offsets, nrows, ncols, bands, patch_size, out_scale,
        n_average, n_steps, method, operator,
    ):
        """Accumulate weighted-sum + weight over the given `offsets` (a
        column-band subset of the tile's full offset list) into a *local*
        canvas sized to just that band's bounding box (plus patch overhang).
        Division is deferred to merge_partials() so bands from different
        ranks can be placed and summed on the full canvas first. Returns
        (result, norm_map, meta, r0, c0) where (r0, c0) is the band's origin
        in full-canvas output pixel coordinates."""
        with rasterio.open(img_path) as src:
            meta   = src.meta.copy()
            bounds = src.bounds
            crs    = src.crs

            out_h = round(nrows * out_scale)
            out_w = round(ncols * out_scale)
            output_transform = from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top, out_w, out_h
            )
            meta.update({"width": out_w, "height": out_h,
                         "transform": output_transform, "crs": crs})

            out_patch = round(patch_size * out_scale)
            cols_off  = [c for c, _ in offsets]
            rows_off  = [r for _, r in offsets]
            r0 = round(min(rows_off) * out_scale)
            c0 = round(min(cols_off) * out_scale)
            r1 = min(out_h, round(max(rows_off) * out_scale) + out_patch)
            c1 = min(out_w, round(max(cols_off) * out_scale) + out_patch)
            result   = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
            norm_map = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)

            big_win  = windows.Window(0, 0, ncols, nrows)
            bld_mask = self.blending_mask
            n_offsets = len(offsets)

            for i, (col_off, row_off) in enumerate(offsets):
                if i % 100 == 0:
                    logging.info(f"[rank {self.global_rank}]   patch {i}/{n_offsets}")
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
                    hr_size = (round(patch_size * out_scale), round(patch_size * out_scale))
                    h_coarse = run_dav2(
                        self.dav2, landsat_for_dav2, target_size=hr_size,
                    )
                    h_coarse = _normalize_coarse(h_coarse, self.target_stats)
                    pseudo_ps = self.sr_module(lr, target_size=hr_size)

                    # Legacy checkpoints need landsat concatenated into the
                    # ControlNet conditioning; current ones ignore it.
                    landsat_hr = (
                        F.interpolate(lr, size=hr_size, mode="bilinear", align_corners=False)
                        if self.concat_landsat_cond else None
                    )

                    preds = [
                        self.fm_refiner.refine_with_ps(
                            pseudo_ps, h_coarse, n_steps=n_steps, method=method,
                            landsat_lr=landsat_hr,
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

                # Local indices within this band's cropped canvas (full-canvas
                # position minus the band's origin r0/c0).
                r_local = round(win.row_off * out_scale) - r0
                c_local = round(win.col_off * out_scale) - c0
                # "blend" uses a gaussian weight per patch; anything else
                # (e.g. "replace") is treated as a uniform weight-1 patch —
                # dividing by the accumulated weight at merge time still
                # recovers the plain overwrite semantics as long as patches
                # for that mode don't overlap.
                bm = bld_mask[:H_out, :W_out] if operator == "blend" else np.ones((H_out, W_out), dtype=np.float32)

                result[r_local:r_local + H_out, c_local:c_local + W_out]   += pred_phys * bm
                norm_map[r_local:r_local + H_out, c_local:c_local + W_out] += bm

        return result, norm_map, meta, r0, c0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SR+FM Refiner large-scale inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the inference YAML config (e.g. config/sr_fm_refiner_v5_infer_ls.yaml).")
    parser.add_argument("--rank", type=int, default=None,
                        help="Local GPU index on this node. Defaults to SLURM_LOCALID (or 0).")
    parser.add_argument("--global_rank", type=int, default=None,
                        help="Global process rank, used to shard tiles across all GPUs. "
                             "Defaults to SLURM_PROCID (or 0).")
    parser.add_argument("--world_size", type=int, default=None,
                        help="Total number of parallel processes. Defaults to SLURM_NTASKS (or 1).")
    parser.add_argument("--merge", action="store_true",
                        help="Merge partial results from .partial/ into final tifs. "
                             "Run once, from a single process, after all compute ranks finish.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    local_rank  = args.rank        if args.rank        is not None else int(os.environ.get("SLURM_LOCALID", 0))
    global_rank = args.global_rank if args.global_rank  is not None else int(os.environ.get("SLURM_PROCID", 0))
    world_size  = args.world_size  if args.world_size   is not None else int(os.environ.get("SLURM_NTASKS", 1))

    config    = load_yaml_config(args.config)
    predictor = SRFMPredictor(config, rank=local_rank, global_rank=global_rank, world_size=world_size)

    if args.merge:
        predictor.merge_partials()
        return

    predictor.build_model()
    predictor.load_files()
    out_dir = predictor.predict_all()
    print(f"PATH_OUTPUT: {out_dir}")


if __name__ == "__main__":
    main()
