"""
Evaluation script for Flow Matching CHM Refiner.

Loads the best checkpoint from a training run and evaluates on the full val set.

Usage:
  python script/depth/eval_fm_refiner.py \
    --config  output/fm_refiner_v0/config.yaml \
    --ckpt    output/fm_refiner_v0/checkpoint/best.pth
"""

import copy
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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset

# Reuse helpers from train_fm_refiner
from script.depth.train_fm_refiner import (
    _split_coords, _make_dataset,
    _normalize_coarse, _load_target_stats,
    compute_metrics,
    _to_vis_rgb, _to_vis_depth, _make_vis_figure,
)


def eval_full(
    fm_refiner: FMRefiner,
    dav2_model,
    val_loader: DataLoader,
    device: torch.device,
    n_steps: int,
    target_stats: dict,
    n_landsat_bands: int,
    n_vis_samples: int = 5,
    use_ps: bool = True,           # use real PS as ControlNet conditioning (upper-bound mode)
) -> tuple:
    """Run full evaluation, return (metrics, figure)."""
    fm_refiner.eval()
    all_preds, all_gts = [], []
    vis_samples = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs_lr, inputs_hr, targets = batch
            if inputs_lr.dim() == 5:
                inputs_lr = inputs_lr.squeeze(0)
            if inputs_hr.dim() == 5:
                inputs_hr = inputs_hr.squeeze(0)
            if targets.dim() == 5:
                targets = targets.squeeze(0)

            inputs_lr = inputs_lr.to(device)
            inputs_hr = inputs_hr.to(device)
            targets = targets.to(device)

            H_hr, W_hr = targets.shape[2], targets.shape[3]

            # Landsat to HR for ControlNet
            landsat = inputs_lr[:, :n_landsat_bands]
            landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

            # DAv2 expects [0,1]; dataloader gives [-1,1]
            landsat_for_dav2 = (landsat + 1.0) / 2.0
            h_coarse = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
            h_coarse = _normalize_coarse(h_coarse, target_stats)

            h_fine = fm_refiner.refine(
                landsat_hr, h_coarse, n_steps=n_steps,
                ps_hr=inputs_hr if use_ps else None,
            )

            all_preds.append(h_fine.cpu().flatten())
            all_gts.append(targets.cpu().flatten())

            if len(vis_samples) < n_vis_samples:
                vis_samples.append({
                    "landsat": _to_vis_rgb(landsat_hr[0]),
                    "ps":      _to_vis_rgb(inputs_hr[0]),
                    "coarse":  _to_vis_depth(h_coarse[0]),
                    "fine":    _to_vis_depth(h_fine[0]),
                    "gt":      _to_vis_depth(targets[0]),
                })

    pred_cat = torch.cat(all_preds, dim=0)
    gt_cat   = torch.cat(all_gts,   dim=0)
    metrics  = compute_metrics(pred_cat, gt_cat)
    fig      = _make_vis_figure(vis_samples) if vis_samples else None
    return metrics, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM Refiner – Full Evaluation")
    parser.add_argument("--config", type=str, default="output/fm_refiner/config.yaml",
                        help="Path to config.yaml (usually inside the output run dir).")
    parser.add_argument("--ckpt", type=str, default="output/fm_refiner/checkpoint/best.pth",
                        help="Path to the checkpoint .pth file (e.g. best.pth).")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Euler integration steps (overrides config if given).")
    parser.add_argument("--n_vis", type=int, default=5,
                        help="Number of sample images to save.")
    parser.add_argument("--save_fig", type=str, default=None,
                        help="If given, save the visualization figure to this path (.png).")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ---- Config ----
    cfg = recursive_load_config(args.config)
    cfg_data = cfg.dataset
    n_steps = args.n_steps if args.n_steps is not None else cfg.validation.n_steps

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"device = {device}")

    # ---- Data (val split only) ----
    n_landsat_bands = len(cfg_data.selected_bands)

    base_dataset = LazyPatchDataset({
        'input_dir': cfg_data.input_dir,
        'input_dir_hr': cfg_data.input_dir_hr,
        'output_dir': cfg_data.output_dir,
        'selected_bands': cfg_data.selected_bands,
        'selected_bands_hr': cfg_data.selected_bands_hr,
        'file_type_input': cfg_data.file_type_input,
        'file_type_output': cfg_data.file_type_output,
        'patch_sizes': [(s, s) for s in cfg_data.patch_sizes],
        'patch_coord_path': cfg_data.patch_coord_path,
        'num_patches_per_tile': cfg_data.num_patches_per_tile,
        'tile_emphasis': cfg_data.tile_emphasis,
        'correlation_cleaning': cfg_data.correlation_cleaning,
        'target_range_edges': cfg_data.target_range_edges,
        'num_patches_per_target_range': cfg_data.num_patches_per_target_range,
        'selected_percentile': cfg_data.selected_percentile,
        'year': cfg_data.year,
        'batch_size': 1,
        'mode': 'val',
        'use_input_minmax': cfg_data.use_input_minmax,
        'use_input_norm': cfg_data.use_input_norm,
        'input_stats_file': cfg_data.input_stats_file,
        'use_input_hr_minmax': cfg_data.use_input_hr_minmax,
        'use_input_hr_norm': cfg_data.use_input_hr_norm,
        'input_hr_stats_file': cfg_data.input_hr_stats_file,
        'use_target_minmax': cfg_data.use_target_minmax,
        'use_target_norm': cfg_data.use_target_norm,
        'target_stats_file': cfg_data.target_stats_file,
        'unit_scale_ratio': cfg_data.unit_scale_ratio,
        'scale_input_to_neg1_1': cfg_data.get('scale_input_to_neg1_1', False),
        'scale_input_hr_to_neg1_1': cfg_data.get('scale_input_hr_to_neg1_1', False),
        'scale_target_to_neg1_1': cfg_data.get('scale_target_to_neg1_1', False),
    })

    all_coords = base_dataset.patch_coords
    _, val_coords, _ = _split_coords(
        all_coords, cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed
    )
    val_dataset = _make_dataset(base_dataset, val_coords, 'val', 1)
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg_data.workers, pin_memory=True)
    logging.info(f"Val batches: {len(val_loader)}")

    # ---- Model ----
    n_ps_bands = len(cfg_data.selected_bands_hr)
    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=cfg.trainer.ps_dropout_p,
        device=str(device),
    ).to(device)

    # Load checkpoint weights
    ckpt = torch.load(args.ckpt, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    logging.info(f"Loaded checkpoint from {args.ckpt} (step {ckpt.get('step', '?')})")

    # ---- DAv2 ----
    dav2_model = load_dav2(
        dav2_path=cfg.model.dav2_pretrained_path,
        backbone=cfg.model.dav2_backbone,
        out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
    ).to(device)

    # ---- Target stats ----
    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ---- Evaluate ----
    metrics, fig = eval_full(
        fm_refiner, dav2_model, val_loader, device,
        n_steps=n_steps,
        target_stats=target_stats,
        n_landsat_bands=n_landsat_bands,
        n_vis_samples=args.n_vis,
        use_ps=True,
    )

    print("\n========== Full Val Results ==========")
    for k, v in metrics.items():
        print(f"  {k:>8}: {v:.4f}")
    print("======================================\n")

    if fig is not None:
        save_path = args.save_fig or os.path.join(
            os.path.dirname(args.ckpt), "eval_vis.png"
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Visualization saved to {save_path}")
        plt.close(fig)
