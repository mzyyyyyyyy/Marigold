"""
Inference / evaluation script for SR + FM Refiner.

Loads a trained checkpoint, runs on the validation or test split, and saves:
  - Quantitative metrics (R², MAE, RMSE) as JSON
  - Qualitative visualisation grid as PNG

Key flags:
  --bridge_sigma 0.0   deterministic ODE at inference (default, recommended for R²)
  --n_avg 1            no averaging needed when bridge_sigma=0

Usage:
  python script/depth/pred_sr_fm_refiner.py \\
      --config  output/sr_fm_refiner_v8/config.yaml \\
      --ckpt    output/sr_fm_refiner_v8/checkpoint/best.pth \\
      --out_dir output/sr_fm_refiner_v8/pred_det \\
      --bridge_sigma 0.0 \\
      --n_avg 1 \\
      --n_steps 4 \\
      --method euler \\
      --split val
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
from depthfm.sr_module import SRModule, build_sr_module
from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _split_coords(all_coords, train_split, val_split, seed):
    n = len(all_coords)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()
    train_size = int(train_split * n)
    val_size   = int(val_split * n)
    return (
        [all_coords[i] for i in indices[:train_size]],
        [all_coords[i] for i in indices[train_size:train_size + val_size]],
        [all_coords[i] for i in indices[train_size + val_size:]],
    )


def _make_dataset(base_dataset, coords, mode, batch_size):
    dataset = copy.copy(base_dataset)
    dataset.batch_size  = batch_size
    dataset.mode        = mode
    dataset.patch_coords = coords
    dataset.patches_by_size = {}
    for patch_size in dataset.patch_sizes:
        dataset.patches_by_size[patch_size] = [
            p for p in coords
            if p['output_patch_height'] == patch_size[0]
            and p['output_patch_width'] == patch_size[1]
        ]
    return dataset


def _normalize_coarse(h_coarse, target_stats):
    p1  = float(target_stats["p1"][0])
    p99 = float(target_stats["p99"][0])
    h   = (h_coarse - p1) / (p99 - p1 + 1e-8)
    return (h * 2.0 - 1.0).clamp(-1.0, 1.0)


def _load_target_stats(stats_file, year):
    with open(stats_file) as f:
        return json.load(f)[str(year)]


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor):
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


def _to_vis_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor[:3].cpu().float()
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _to_vis_depth(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().cpu().float().numpy()


def _make_vis_figure(samples: list) -> plt.Figure:
    n = len(samples)
    col_titles = ["Landsat (RGB)", "pseudo-PS (RGB)", "Real PS (RGB)",
                  "DAv2 Coarse", "FM Refined", "GT"]
    fig, axes = plt.subplots(n, 6, figsize=(24, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, s in enumerate(samples):
        for col, (key, title) in enumerate(zip(
            ["landsat", "pseudo_ps", "real_ps", "coarse", "fine", "gt"], col_titles
        )):
            ax   = axes[row, col]
            data = s[key]
            if key in ("landsat", "pseudo_ps", "real_ps"):
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap="plasma", vmin=-1.0, vmax=1.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    return fig


def load_weights(fm_refiner: FMRefiner, sr_module: SRModule, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    sr_module.load_state_dict(ckpt["sr_state"])
    step = ckpt.get("step", -1)
    logging.info(f"Loaded checkpoint: {path}  (step={step})")
    return step


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, required=True)
    parser.add_argument("--ckpt",         type=str, required=True)
    parser.add_argument("--out_dir",      type=str, required=True)
    parser.add_argument("--split",        type=str, default="val",
                        choices=["val", "test"])
    parser.add_argument("--bridge_sigma", type=float, default=0.0,
                        help="Inference bridge sigma. 0.0 = deterministic ODE.")
    parser.add_argument("--n_avg",        type=int, default=1,
                        help="Stochastic averaging runs. Use 1 when bridge_sigma=0.")
    parser.add_argument("--n_steps",      type=int, default=None,
                        help="ODE steps (overrides config).")
    parser.add_argument("--method",       type=str, default=None,
                        choices=["euler", "heun"],
                        help="ODE method (overrides config).")
    parser.add_argument("--n_vis",        type=int, default=8,
                        help="Number of samples to visualise.")
    parser.add_argument("--no_cuda",      action="store_true")
    parser.add_argument("--gpu",          type=int, default=1,
                        help="CUDA device index (default: 1).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg      = OmegaConf.load(args.config)
    cfg_data = cfg.dataset
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"device = {device}")

    n_steps = args.n_steps if args.n_steps is not None else cfg.validation.n_steps
    method  = args.method  if args.method  is not None else cfg.validation.get("method", "euler")
    logging.info(f"bridge_sigma={args.bridge_sigma}  n_avg={args.n_avg}  "
                 f"n_steps={n_steps}  method={method}")

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
    train_coords, val_coords, test_coords = _split_coords(
        all_coords, cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed
    )
    eval_coords  = val_coords if args.split == "val" else test_coords
    eval_dataset = _make_dataset(base_dataset, eval_coords, args.split, 1)
    eval_loader  = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                              num_workers=cfg_data.workers, pin_memory=True)
    logging.info(f"{args.split} set: {len(eval_loader)} batches")

    # ---- Models ----
    n_landsat_bands = len(cfg_data.selected_bands)
    n_ps_bands      = len(cfg_data.selected_bands_hr)

    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=0.0,
        bridge_sigma=args.bridge_sigma,
        concat_z_coarse=cfg.trainer.get("concat_z_coarse", False),
        device=str(device),
    ).to(device)

    dav2_model = load_dav2(
        dav2_path=cfg.model.dav2_pretrained_path,
        backbone=cfg.model.dav2_backbone,
        out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
    ).to(device)

    sr_module = build_sr_module(
        OmegaConf.to_container(cfg.sr_module, resolve=True)
    ).to(device)

    step = load_weights(fm_refiner, sr_module, args.ckpt)
    fm_refiner.eval()
    sr_module.eval()

    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ---- Inference loop ----
    all_preds, all_gts = [], []
    vis_samples = []

    for batch in tqdm(eval_loader, desc=f"Predicting ({args.split})"):
        inputs_lr, inputs_hr, targets = batch
        if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
        if inputs_hr.dim() == 5: inputs_hr = inputs_hr.squeeze(0)
        if targets.dim()   == 5: targets   = targets.squeeze(0)

        inputs_lr = inputs_lr.to(device)
        inputs_hr = inputs_hr.to(device)
        targets   = targets.to(device)

        H_hr, W_hr = targets.shape[2], targets.shape[3]
        landsat    = inputs_lr[:, :n_landsat_bands]
        landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr),
                                   mode="bilinear", align_corners=False)

        with torch.no_grad():
            landsat_for_dav2 = (landsat + 1.0) / 2.0
            h_coarse  = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
            h_coarse  = _normalize_coarse(h_coarse, target_stats)
            pseudo_ps = sr_module(landsat, target_size=(H_hr, W_hr))

            def _run():
                return fm_refiner.refine_with_ps(
                    pseudo_ps, h_coarse, n_steps=n_steps, method=method
                )

            if args.n_avg > 1:
                h_fine = torch.stack([_run() for _ in range(args.n_avg)]).mean(dim=0)
            else:
                h_fine = _run()

        all_preds.append(h_fine.cpu().flatten())
        all_gts.append(targets.cpu().flatten())

        if len(vis_samples) < args.n_vis:
            vis_samples.append({
                "landsat":   _to_vis_rgb(landsat_hr[0]),
                "pseudo_ps": _to_vis_rgb(pseudo_ps[0]),
                "real_ps":   _to_vis_rgb(inputs_hr[0]),
                "coarse":    _to_vis_depth(h_coarse[0]),
                "fine":      _to_vis_depth(h_fine[0]),
                "gt":        _to_vis_depth(targets[0]),
            })

    # ---- Metrics ----
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_gts))
    logging.info(f"Results: {metrics}")

    result = {
        "split":        args.split,
        "ckpt":         args.ckpt,
        "step":         step,
        "bridge_sigma": args.bridge_sigma,
        "n_avg":        args.n_avg,
        "n_steps":      n_steps,
        "method":       method,
        **metrics,
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Metrics → {metrics_path}")

    # ---- Visualisation ----
    if vis_samples:
        fig      = _make_vis_figure(vis_samples)
        fig_path = os.path.join(args.out_dir, "vis.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Vis     → {fig_path}")
