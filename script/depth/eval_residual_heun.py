"""
Residual analysis for FM Refiner with 2nd-order Heun integration (4 steps).

Computes:
  residual_pred = H_refined - H_coarse   (using Heun 4-step)
  residual_gt   = H_gt      - H_coarse

Then:
  1. Visualizes 20 samples (residual_pred vs residual_gt side-by-side).
  2. Computes Pearson correlation over the full validation set.

Usage:
  python script/depth/eval_residual_heun.py \
    --config  output/fm_refiner_v3/config.yaml \
    --ckpt    output/fm_refiner_v3/checkpoint/best.pth \
    --save_dir output/fm_refiner_v3/residual_heun
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset
from script.depth.train_fm_refiner import (
    _split_coords, _make_dataset,
    _normalize_coarse, _load_target_stats,
    _to_vis_rgb, _to_vis_depth,
)


def _make_residual_figure(samples: list) -> plt.Figure:
    """
    7 columns × N rows:
      Landsat | PlanetScope | H_coarse | H_refined | H_gt | residual_pred | residual_gt
    """
    n = len(samples)
    col_titles = [
        "Landsat (RGB)", "PlanetScope (RGB)",
        "H_coarse", "H_refined (Heun-4)", "H_gt",
        "residual_pred\n(refined−coarse)", "residual_gt\n(gt−coarse)",
    ]
    fig, axes = plt.subplots(n, 7, figsize=(28, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(samples):
        # --- RGB columns ---
        for col, key in enumerate(("landsat", "ps")):
            ax = axes[row, col]
            ax.imshow(s[key])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            ax.axis("off")

        # --- Depth columns (shared vmin/vmax = [-1, 1]) ---
        for col, key in enumerate(("coarse", "fine", "gt"), start=2):
            ax = axes[row, col]
            im = ax.imshow(s[key], cmap="plasma", vmin=-1.0, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            ax.axis("off")

        # --- Residual columns (symmetric colormap, common scale per sample) ---
        r_pred = s["residual_pred"]
        r_gt   = s["residual_gt"]
        abs_max = max(float(np.abs(r_pred).max()), float(np.abs(r_gt).max()), 1e-6)
        for col, (key, data) in enumerate(
            zip(("residual_pred", "residual_gt"), (r_pred, r_gt)), start=5
        ):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    return fig


def run_residual_eval(
    fm_refiner: FMRefiner,
    dav2_model,
    val_loader: DataLoader,
    device: torch.device,
    n_steps: int,
    method: str,
    target_stats: dict,
    n_landsat_bands: int,
    n_vis_samples: int = 20,
) -> tuple:
    """
    Returns:
      pearson_r  – float, Pearson r over full val set
      pearson_p  – float, p-value
      fig        – matplotlib Figure with n_vis_samples rows
    """
    fm_refiner.eval()

    all_res_pred = []
    all_res_gt   = []
    vis_samples  = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Residual eval [{method}, {n_steps} steps]"):
            inputs_lr, inputs_hr, targets = batch
            if inputs_lr.dim() == 5:
                inputs_lr = inputs_lr.squeeze(0)
            if inputs_hr.dim() == 5:
                inputs_hr = inputs_hr.squeeze(0)
            if targets.dim() == 5:
                targets = targets.squeeze(0)

            inputs_lr = inputs_lr.to(device)
            inputs_hr = inputs_hr.to(device)
            targets   = targets.to(device)

            H_hr, W_hr = targets.shape[2], targets.shape[3]

            landsat    = inputs_lr[:, :n_landsat_bands]
            landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr),
                                       mode="bilinear", align_corners=False)

            landsat_for_dav2 = (landsat + 1.0) / 2.0
            h_coarse = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
            h_coarse = _normalize_coarse(h_coarse, target_stats)

            h_fine = fm_refiner.refine(landsat_hr, h_coarse, n_steps=n_steps, method=method)

            res_pred = (h_fine  - h_coarse).cpu()
            res_gt   = (targets - h_coarse).cpu()

            all_res_pred.append(res_pred.flatten())
            all_res_gt.append(res_gt.flatten())

            if len(vis_samples) < n_vis_samples:
                vis_samples.append({
                    "landsat":       _to_vis_rgb(landsat_hr[0]),
                    "ps":            _to_vis_rgb(inputs_hr[0]),
                    "coarse":        _to_vis_depth(h_coarse[0]),
                    "fine":          _to_vis_depth(h_fine[0]),
                    "gt":            _to_vis_depth(targets[0]),
                    "residual_pred": _to_vis_depth(res_pred[0]),
                    "residual_gt":   _to_vis_depth(res_gt[0]),
                })

    rp = torch.cat(all_res_pred).numpy().astype(np.float32)
    rg = torch.cat(all_res_gt).numpy().astype(np.float32)

    # Drop any NaN/Inf pixels before correlation
    mask = np.isfinite(rp) & np.isfinite(rg)
    rp, rg = rp[mask], rg[mask]
    pearson_r, pearson_p = pearsonr(rp, rg)

    fig = _make_residual_figure(vis_samples) if vis_samples else None
    return float(pearson_r), float(pearson_p), fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM Refiner – Residual Analysis (Heun)")
    parser.add_argument("--config",   type=str, default="output/fm_refiner_v3/config.yaml")
    parser.add_argument("--ckpt",     type=str, default="output/fm_refiner_v3/checkpoint/best.pth")
    parser.add_argument("--n_steps",  type=int, default=4,
                        help="ODE integration steps (default: 4).")
    parser.add_argument("--method",   type=str, default="heun", choices=["euler", "heun"],
                        help="Integration method: euler or heun (default: heun).")
    parser.add_argument("--n_vis",    type=int, default=20,
                        help="Number of samples to visualize.")
    parser.add_argument("--save_dir", type=str, default="output/fm_refiner_v3/residual_heun",
                        help="Directory for output files.")
    parser.add_argument("--no_cuda",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg      = recursive_load_config(args.config)
    cfg_data = cfg.dataset
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"device = {device}  method = {args.method}  n_steps = {args.n_steps}")

    n_landsat_bands = len(cfg_data.selected_bands)
    n_ps_bands      = len(cfg_data.selected_bands_hr)

    # ---- Dataset ----
    base_dataset = LazyPatchDataset({
        'input_dir':                    cfg_data.input_dir,
        'input_dir_hr':                 cfg_data.input_dir_hr,
        'output_dir':                   cfg_data.output_dir,
        'selected_bands':               cfg_data.selected_bands,
        'selected_bands_hr':            cfg_data.selected_bands_hr,
        'file_type_input':              cfg_data.file_type_input,
        'file_type_output':             cfg_data.file_type_output,
        'patch_sizes':                  [(s, s) for s in cfg_data.patch_sizes],
        'patch_coord_path':             cfg_data.patch_coord_path,
        'num_patches_per_tile':         cfg_data.num_patches_per_tile,
        'tile_emphasis':                cfg_data.tile_emphasis,
        'correlation_cleaning':         cfg_data.correlation_cleaning,
        'target_range_edges':           cfg_data.target_range_edges,
        'num_patches_per_target_range': cfg_data.num_patches_per_target_range,
        'selected_percentile':          cfg_data.selected_percentile,
        'year':                         cfg_data.year,
        'batch_size':                   1,
        'mode':                         'val',
        'use_input_minmax':             cfg_data.use_input_minmax,
        'use_input_norm':               cfg_data.use_input_norm,
        'input_stats_file':             cfg_data.input_stats_file,
        'use_input_hr_minmax':          cfg_data.use_input_hr_minmax,
        'use_input_hr_norm':            cfg_data.use_input_hr_norm,
        'input_hr_stats_file':          cfg_data.input_hr_stats_file,
        'use_target_minmax':            cfg_data.use_target_minmax,
        'use_target_norm':              cfg_data.use_target_norm,
        'target_stats_file':            cfg_data.target_stats_file,
        'unit_scale_ratio':             cfg_data.unit_scale_ratio,
        'scale_input_to_neg1_1':        cfg_data.get('scale_input_to_neg1_1', False),
        'scale_input_hr_to_neg1_1':     cfg_data.get('scale_input_hr_to_neg1_1', False),
        'scale_target_to_neg1_1':       cfg_data.get('scale_target_to_neg1_1', False),
    })

    all_coords = base_dataset.patch_coords
    _, val_coords, _ = _split_coords(
        all_coords, cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed
    )
    val_dataset = _make_dataset(base_dataset, val_coords, 'val', 1)
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg_data.workers, pin_memory=True)
    logging.info(f"Val batches: {len(val_loader)}")

    # ---- Models ----
    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=cfg.trainer.ps_dropout_p,
        device=str(device),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    logging.info(f"Loaded checkpoint from {args.ckpt} (step {ckpt.get('step', '?')})")

    dav2_model = load_dav2(
        dav2_path=cfg.model.dav2_pretrained_path,
        backbone=cfg.model.dav2_backbone,
        out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
    ).to(device)

    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ---- Evaluate ----
    pearson_r, pearson_p, fig = run_residual_eval(
        fm_refiner, dav2_model, val_loader, device,
        n_steps=args.n_steps,
        method=args.method,
        target_stats=target_stats,
        n_landsat_bands=n_landsat_bands,
        n_vis_samples=args.n_vis,
    )

    print(f"\n========== Residual Correlation [{args.method}, {args.n_steps} steps] ==========")
    print(f"  Pearson r  : {pearson_r:.6f}")
    print(f"  p-value    : {pearson_p:.3e}")
    print("===========================================================\n")

    if fig is not None:
        fig_path = os.path.join(save_dir, "residual_vis.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        logging.info(f"Residual visualization saved to {fig_path}")
        plt.close(fig)

    results_path = os.path.join(save_dir, "residual_correlation.json")
    with open(results_path, "w") as f:
        json.dump({
            "method": args.method,
            "n_steps": args.n_steps,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        }, f, indent=2)
    logging.info(f"Correlation results saved to {results_path}")
