"""
Probe experiment: FM start-point ablation.

The FM model normally integrates from z_coarse → z_gt.
This script runs inference from three different starting points and compares outputs:

  (A) z_coarse          -- normal inference (baseline)
  (B) z_gt + ε·noise    -- start near GT (ε = noise_scale)
  (C) z_gt              -- start exactly at GT (ε = 0, oracle upper bound)

Interpretation:
  If (C) >> (A) in quality → FM network is OK; the trajectory z_coarse→z_gt is wrong.
  If (C) ≈ (A) in quality → FM has a learning problem (underfitting, capacity, etc.).
  (B) interpolates between them to see sensitivity to perturbation magnitude.

Supports both:
  - train_fm_refiner  checkpoints (no SR module, e.g. fm_refiner_v3)
  - train_sr_fm_refiner checkpoints (has SR module, e.g. sr_fm_refiner_v1)
  The script auto-detects which type based on whether 'sr_state' is in the checkpoint.

Usage:
  # fm_refiner_v3 (no SR)
  python script/depth/probe_fm_startpoint.py \
      --config config/fm_refiner_v3.yaml \
      --checkpoint output/fm_refiner_v3/checkpoint/latest.pth \
      --n_samples 100 --noise_scale 0.1 \
      --output_dir output/probe_startpoint_v3

  # sr_fm_refiner_v1 (with SR)
  python script/depth/probe_fm_startpoint.py \
      --config config/sr_fm_refiner_v1.yaml \
      --checkpoint output/sr_fm_refiner_v1/checkpoint/latest.pth \
      --n_samples 100 --noise_scale 0.1 \
      --output_dir output/probe_startpoint_sr_v1
"""

import argparse
import copy
import json
import logging
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from depthfm.chmv2 import load_chmv2, run_chmv2
from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    pred = pred.flatten().float()
    gt   = gt.flatten().float()
    mask = torch.isfinite(gt) & torch.isfinite(pred)
    pred, gt = pred[mask], gt[mask]
    ss_res = ((gt - pred) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    return {
        "r2":   (1.0 - ss_res / (ss_tot + 1e-8)).item(),
        "mae":  (gt - pred).abs().mean().item(),
        "rmse": ((gt - pred) ** 2).mean().sqrt().item(),
    }


def aggregate_metrics(metric_list: list) -> dict:
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


# ---------------------------------------------------------------------------
# Inference from an arbitrary latent start point
# ---------------------------------------------------------------------------

@torch.no_grad()
def refine_from_z(
    fm_refiner: FMRefiner,
    z_start: torch.Tensor,
    control_input: torch.Tensor,
    text_emb: torch.Tensor,
    n_steps: int = 1,
    method: str = "euler",
) -> torch.Tensor:
    z = z_start.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i / n_steps
        v1 = fm_refiner._vel(z, t_val, control_input, text_emb)
        if method == "heun":
            t_next = min(t_val + dt, 1.0)
            v2 = fm_refiner._vel(z + dt * v1, t_next, control_input, text_emb)
            z = z + dt * 0.5 * (v1 + v2)
        else:
            z = z + dt * v1
    return fm_refiner.decode(z)


# ---------------------------------------------------------------------------
# Build dataset (shared between both training script flavours)
# ---------------------------------------------------------------------------

def build_val_loader(cfg_data, val_coords):
    base_dataset = LazyPatchDataset({
        'input_dir':                cfg_data.input_dir,
        'input_dir_hr':             cfg_data.input_dir_hr,
        'output_dir':               cfg_data.output_dir,
        'selected_bands':           cfg_data.selected_bands,
        'selected_bands_hr':        cfg_data.selected_bands_hr,
        'file_type_input':          cfg_data.file_type_input,
        'file_type_output':         cfg_data.file_type_output,
        'patch_sizes':              [(s, s) for s in cfg_data.patch_sizes],
        'patch_coord_path':         cfg_data.patch_coord_path,
        'num_patches_per_tile':     cfg_data.num_patches_per_tile,
        'tile_emphasis':            getattr(cfg_data, 'tile_emphasis', []),
        'min_valid_ratio':          cfg_data.min_valid_ratio,
        'target_name':              cfg_data.target_name,
        'use_input_minmax':         cfg_data.use_input_minmax,
        'use_input_norm':           cfg_data.use_input_norm,
        'input_stats_file':         cfg_data.input_stats_file,
        'use_input_hr_minmax':      cfg_data.use_input_hr_minmax,
        'use_input_hr_norm':        cfg_data.use_input_hr_norm,
        'input_hr_stats_file':      cfg_data.input_hr_stats_file,
        'use_target_minmax':        cfg_data.use_target_minmax,
        'use_target_norm':          cfg_data.use_target_norm,
        'target_stats_file':        cfg_data.target_stats_file,
        'scale_input_to_neg1_1':    cfg_data.scale_input_to_neg1_1,
        'scale_input_hr_to_neg1_1': cfg_data.scale_input_hr_to_neg1_1,
        'scale_target_to_neg1_1':   cfg_data.scale_target_to_neg1_1,
        'correlation_cleaning':     getattr(cfg_data, 'correlation_cleaning', False),
        'unit_scale_ratio':         getattr(cfg_data, 'unit_scale_ratio', 1),
        'year':                     cfg_data.year,
        'selected_percentile':      cfg_data.selected_percentile,
        'target_range_edges':       cfg_data.target_range_edges,
        'num_patches_per_target_range': cfg_data.num_patches_per_target_range,
        'batch_size':               1,
        'mode':                     'val',
    })
    dataset = copy.copy(base_dataset)
    dataset.batch_size   = 1
    dataset.mode         = "val"
    dataset.patch_coords = val_coords
    dataset.patches_by_size = {}
    for patch_size in dataset.patch_sizes:
        dataset.patches_by_size[patch_size] = [
            p for p in val_coords
            if p['output_patch_height'] == patch_size[0]
            and p['output_patch_width'] == patch_size[1]
        ]
    return DataLoader(dataset, batch_size=None, num_workers=2, shuffle=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--n_samples",   type=int,   default=100)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--n_steps",     type=int,   default=1)
    parser.add_argument("--method",      type=str,   default="euler")
    parser.add_argument("--n_vis",       type=int,   default=8)
    parser.add_argument("--output_dir",  type=str,   default="output/probe_startpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg    = recursive_load_config(args.config)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logging.info(f"device = {device}")

    n_ls_bands = len(cfg.dataset.selected_bands)
    n_ps_bands = len(cfg.dataset.selected_bands_hr)

    # ---- Load checkpoint (auto-detect flavour) ----
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    has_sr = "sr_state" in ckpt
    logging.info(f"Checkpoint type: {'sr_fm_refiner' if has_sr else 'fm_refiner'} (sr_state={'yes' if has_sr else 'no'})")

    # ---- Build FM Refiner ----
    fm_refiner = build_fm_refiner(
        cfg.model.sd_pretrained_path,
        n_landsat_bands=n_ls_bands,
        n_ps_bands=n_ps_bands,
        device=str(device),
    )
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    fm_refiner = fm_refiner.to(device).eval()

    # ---- Coarse model ----
    coarse_model_type = getattr(cfg.model, "coarse_model", "dav2")
    if coarse_model_type == "chmv2":
        coarse_model, chmv2_mean, chmv2_std = load_chmv2(cfg.model)
        coarse_model = coarse_model.to(device).eval()
    else:
        coarse_model = load_dav2(
            cfg.model.dav2_pretrained_path,
            cfg.model.dav2_backbone,
            getattr(cfg.model, "dav2_out_in_scale_factor", 1),
        )
        coarse_model = coarse_model.to(device).eval()
        chmv2_mean = chmv2_std = None

    # ---- Target normalisation ----
    with open(cfg.dataset.target_stats_file) as f:
        target_stats = json.load(f)[str(cfg.dataset.year)]

    def normalize_coarse(h):
        p1  = float(target_stats["p1"][0])
        p99 = float(target_stats["p99"][0])
        return ((h - p1) / (p99 - p1 + 1e-8) * 2.0 - 1.0).clamp(-1.0, 1.0)

    # ---- Val split ----
    cfg_data = cfg.dataset
    with open(cfg_data.patch_coord_path, "rb") as f:
        all_coords = pickle.load(f)

    n   = len(all_coords)
    rng = torch.Generator().manual_seed(cfg_data.get("split_seed", 42))
    idx = torch.randperm(n, generator=rng).tolist()
    train_size = int(cfg_data.train_split * n)
    val_size   = int(cfg_data.val_split   * n)
    val_coords = [all_coords[i] for i in idx[train_size: train_size + val_size]]

    loader = build_val_loader(cfg_data, val_coords)

    # ---- Collect metrics ----
    metrics_coarse  = []
    metrics_from_zc = []
    metrics_from_zn = []
    metrics_from_zg = []
    vis_samples     = []
    collected       = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Probing", total=args.n_samples):
            if collected >= args.n_samples:
                break

            inputs_lr, inputs_hr, targets = batch
            if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
            if targets.dim() == 5:   targets   = targets.squeeze(0)

            inputs_lr = inputs_lr.to(device)
            targets   = targets.to(device)

            H_hr, W_hr = targets.shape[2], targets.shape[3]
            landsat    = inputs_lr[:, :n_ls_bands]
            landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr),
                                       mode="bilinear", align_corners=False)

            if coarse_model_type == "chmv2":
                h_coarse = run_chmv2(coarse_model, landsat, (H_hr, W_hr),
                                     chmv2_mean, chmv2_std)
            else:
                h_coarse = run_dav2(coarse_model, (landsat + 1.0) / 2.0,
                                    target_size=(H_hr, W_hr))
            h_coarse = normalize_coarse(h_coarse)
            h_gt     = targets
            B        = h_gt.shape[0]

            z_coarse = fm_refiner.encode(h_coarse)
            z_gt     = fm_refiner.encode(h_gt)

            # All conditions use null PS (consistent with fm_refiner_v3 ps_dropout_p=1.0
            # and with the no-PS inference path of sr_fm_refiner)
            null_ps = fm_refiner._null_ps.expand(B, -1, H_hr, W_hr).to(device)
            control_input = torch.cat([landsat_hr, null_ps], dim=1)
            text_emb = fm_refiner.empty_text_embed.to(device).expand(B, -1, -1)

            # (A) normal inference from z_coarse
            h_from_zc = refine_from_z(fm_refiner, z_coarse, control_input, text_emb,
                                       args.n_steps, args.method)
            # (B) from z_gt + Gaussian noise
            h_from_zn = refine_from_z(fm_refiner, z_gt + torch.randn_like(z_gt) * args.noise_scale,
                                       control_input, text_emb, args.n_steps, args.method)
            # (C) oracle: start exactly at z_gt
            h_from_zg = refine_from_z(fm_refiner, z_gt, control_input, text_emb,
                                       args.n_steps, args.method)

            for b in range(B):
                gt_b = h_gt[b]
                metrics_coarse.append(compute_metrics(h_coarse[b],   gt_b))
                metrics_from_zc.append(compute_metrics(h_from_zc[b], gt_b))
                metrics_from_zn.append(compute_metrics(h_from_zn[b], gt_b))
                metrics_from_zg.append(compute_metrics(h_from_zg[b], gt_b))

                if len(vis_samples) < args.n_vis:
                    def _np(t): return t.squeeze().cpu().float().numpy()
                    vis_samples.append({
                        "gt":      _np(gt_b),
                        "coarse":  _np(h_coarse[b]),
                        "from_zc": _np(h_from_zc[b]),
                        "from_zn": _np(h_from_zn[b]),
                        "from_zg": _np(h_from_zg[b]),
                    })

                collected += 1
                if collected >= args.n_samples:
                    break

    logging.info(f"Collected {collected} samples.")

    # ---- Print results ----
    agg_coarse  = aggregate_metrics(metrics_coarse)
    agg_from_zc = aggregate_metrics(metrics_from_zc)
    agg_from_zn = aggregate_metrics(metrics_from_zn)
    agg_from_zg = aggregate_metrics(metrics_from_zg)

    print(f"\n=== FM Start-point Ablation  [{os.path.basename(args.config)}] ===")
    print(f"{'Condition':<38} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 66)
    for label, agg in [
        ("h_coarse (no FM, baseline)",              agg_coarse),
        ("(A) FM from z_coarse",                    agg_from_zc),
        (f"(B) FM from z_gt + ε={args.noise_scale}", agg_from_zn),
        ("(C) FM from z_gt  [oracle]",              agg_from_zg),
    ]:
        print(f"{label:<38} {agg['r2']:>8.4f} {agg['mae']:>8.4f} {agg['rmse']:>8.4f}")

    gap_mae = agg_from_zg['mae'] - agg_from_zc['mae']
    rel     = abs(gap_mae) / (agg_from_zc['mae'] + 1e-8) * 100
    print()
    if gap_mae < -0.005:
        print(f"→ (C) better than (A) by {rel:.1f}% MAE")
        print("  FM network is OK; the trajectory from z_coarse is the bottleneck.")
    else:
        print(f"→ (C) ≈ (A), gap only {rel:.1f}% MAE")
        print("  FM has a learning problem (underfitting / capacity / training).")

    # ---- Visualisation ----
    n_vis  = len(vis_samples)
    keys   = ["gt", "coarse", "from_zc", "from_zn", "from_zg"]
    titles = ["GT", "h_coarse",
              "(A) FM z_coarse",
              f"(B) FM z_gt+ε={args.noise_scale}",
              "(C) FM z_gt [oracle]"]

    fig, axes = plt.subplots(n_vis, 5, figsize=(20, 4 * n_vis))
    if n_vis == 1:
        axes = axes[np.newaxis, :]
    for row, s in enumerate(vis_samples):
        vmin, vmax = s["gt"].min(), s["gt"].max()
        for col, (key, title) in enumerate(zip(keys, titles)):
            ax = axes[row, col]
            im = ax.imshow(s[key], cmap="plasma", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=9)
            ax.axis("off")
    plt.suptitle(
        f"{os.path.basename(args.config)}  |  "
        f"n={collected}, ε={args.noise_scale}, n_steps={args.n_steps}, method={args.method}",
        fontsize=10, y=1.002,
    )
    plt.tight_layout()
    vis_path = os.path.join(args.output_dir, "startpoint_vis.png")
    plt.savefig(vis_path, dpi=120, bbox_inches="tight")
    logging.info(f"Visualisation saved to {vis_path}")

    # ---- Save metrics ----
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "coarse":   agg_coarse,
            "from_zc":  agg_from_zc,
            "from_zn":  agg_from_zn,
            "from_zg":  agg_from_zg,
            "config": {
                "config_file":    args.config,
                "checkpoint":     args.checkpoint,
                "checkpoint_type": "sr_fm_refiner" if has_sr else "fm_refiner",
                "n_samples":      collected,
                "noise_scale":    args.noise_scale,
                "n_steps":        args.n_steps,
                "method":         args.method,
            },
        }, f, indent=2)
    logging.info(f"Metrics saved to {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
