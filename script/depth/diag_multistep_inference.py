"""
Post-hoc diagnostic (no retraining): sweep the number of ODE integration steps
(and optionally inference-time noise `sample_sigma`) at inference time only,
using an already-trained FM Refiner + SR module checkpoint.

Motivation: v2-R2 (and the other sr_fm_refiner_v2-* configs) were trained with
bridge_sigma=0.0 (deterministic straight-line interpolant) and validated with
n_steps=1 (single Euler step, i.e. z_fine = z_coarse + v(z_coarse, t=0)). If the
true marginal velocity field is not perfectly straight (which it generally
isn't when the coarse->gt mapping is ambiguous), a single tangent-line jump
from t=0 will systematically under/over-shoot, and multi-step integration
(which follows the curved path) could recover accuracy/detail without any
retraining, since the FM loss itself was trained with t~U(0,1) already.

This script does NOT change any training code or config; it only changes how
many times `refine_with_ps()` queries the (frozen, already-trained) network at
inference. Read-only w.r.t. checkpoints.

Usage:
  python script/depth/diag_multistep_inference.py \\
      --config config/sr_fm_refiner_v2-R2.yaml \\
      --checkpoint output/sr_fm_refiner_v2-R2/checkpoint/best.pth \\
      --device cuda:0 \\
      --n_steps 1,2,4,8 \\
      --methods euler,heun \\
      --subset_frac 0.1
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from depthfm.fm_refiner import build_fm_refiner, load_dav2, run_dav2
from depthfm.sr_module import build_sr_module
from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset

import train_sr_fm_refiner as tsfr  # same directory; only helper functions are used


def build_val_loader(cfg, subset_frac=1.0):
    """Truncate the val coord list to `subset_frac` BEFORE building the dataset,
    so only the patches we'll actually evaluate get touched (each patch load is a
    real disk read — materializing the full val set first, then slicing, wastes
    most of the I/O and looks like the script is doing nothing on the GPU)."""
    cfg_data = cfg.dataset
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
        'mode': 'train',
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
    _, val_coords, _ = tsfr._split_coords(all_coords, cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed)
    n_total = len(val_coords)
    subset_size = max(1, int(n_total * subset_frac))
    val_coords = val_coords[:subset_size]
    val_dataset = tsfr._make_dataset(base_dataset, val_coords, 'val', 1)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    return loader, n_total, subset_size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default="config/sr_fm_refiner_v2-R2.yaml")
    ap.add_argument("--checkpoint", type=str, default="output/sr_fm_refiner_v2-R2/checkpoint/best.pth")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--n_steps", type=str, default="1, 8", help="comma-separated list of ODE step counts")
    ap.add_argument("--methods", type=str, default="euler", help="comma-separated: euler,heun")
    ap.add_argument("--sample_sigma", type=float, default=0.0,
                     help="inference-time SDE noise scale (0 = deterministic ODE, matches training)")
    ap.add_argument("--n_avg", type=int, default=1, help="average this many stochastic samples per patch (only useful if sample_sigma>0)")
    ap.add_argument("--subset_frac", type=float, default=0.1, help="fraction of val set to evaluate (1.0 = full)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vis_samples", type=int, default=6,
                     help="if >0, also save a figure comparing this many val patches across all --n_steps "
                          "(using the first method in --methods) to output/diag_multistep_vis.png")
    ap.add_argument("--vis_out", type=str, default="output/diag_multistep_vis.png")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    cfg = recursive_load_config(os.path.join(REPO, args.config))
    cfg_data = cfg.dataset

    val_loader, n_total, subset_size = build_val_loader(cfg, subset_frac=args.subset_frac)
    n_landsat_bands = len(cfg_data.selected_bands)
    n_ps_bands = len(cfg_data.selected_bands_hr)

    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=0.0,
        bridge_sigma=cfg.trainer.get("bridge_sigma", 0.0),
        sample_sigma=args.sample_sigma,
        concat_z_coarse=cfg.trainer.get("concat_z_coarse", False),
        concat_landsat_hr=cfg.trainer.get("concat_landsat_hr", False),
        device=str(device),
    ).to(device)

    sr_module = build_sr_module(OmegaConf.to_container(cfg.sr_module, resolve=True)).to(device)

    coarse_model = load_dav2(
        dav2_path=cfg.model.dav2_pretrained_path,
        backbone=cfg.model.dav2_backbone,
        out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
    ).to(device)

    ckpt_path = os.path.join(REPO, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"].to(device)
    sr_module.load_state_dict(ckpt["sr_state"])
    print(f"Loaded checkpoint (read-only): {ckpt_path} (step={ckpt.get('step')})")
    fm_refiner.eval()
    sr_module.eval()

    target_stats = tsfr._load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    print(f"Loading {subset_size} / {n_total} validation patches (disk I/O, no GPU work yet)...")
    batches = list(tqdm(val_loader, total=subset_size, desc="Loading val patches"))
    print(f"Evaluating on {subset_size} / {n_total} validation patches")

    n_steps_list = [int(x) for x in args.n_steps.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    print(f"\n{'method':<8} {'n_steps':>8} {'r2':>8} {'mae':>8} {'rmse':>8}")
    sweep = [(method, n_steps) for method in methods for n_steps in n_steps_list]
    for method, n_steps in tqdm(sweep, desc="Sweep (method, n_steps)"):
        torch.manual_seed(args.seed)
        preds, gts = [], []
        with torch.no_grad():
            for batch in tqdm(batches, desc=f"{method} n_steps={n_steps}", leave=False):
                inputs_lr, inputs_hr, targets = batch
                if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
                if inputs_hr.dim() == 5: inputs_hr = inputs_hr.squeeze(0)
                if targets.dim() == 5: targets = targets.squeeze(0)
                inputs_lr = inputs_lr.to(device)
                inputs_hr = inputs_hr.to(device)
                targets = targets.to(device)

                H_hr, W_hr = targets.shape[2], targets.shape[3]
                landsat = inputs_lr[:, :n_landsat_bands]
                landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

                landsat_for_dav2 = (landsat + 1.0) / 2.0
                h_coarse = run_dav2(coarse_model, landsat_for_dav2, target_size=(H_hr, W_hr))
                h_coarse = tsfr._normalize_coarse(h_coarse, target_stats)

                pseudo_ps = sr_module(landsat, target_size=(H_hr, W_hr))

                if args.n_avg > 1:
                    h_fine = torch.stack([
                        fm_refiner.refine_with_ps(pseudo_ps, h_coarse, n_steps=n_steps,
                                                   method=method, landsat_hr=landsat_hr)
                        for _ in range(args.n_avg)
                    ]).mean(dim=0)
                else:
                    h_fine = fm_refiner.refine_with_ps(pseudo_ps, h_coarse, n_steps=n_steps,
                                                        method=method, landsat_hr=landsat_hr)

                pred_m = tsfr._denormalize_target(h_fine, target_stats)
                gt_m = tsfr._denormalize_target(targets, target_stats)
                preds.append(pred_m.cpu().flatten())
                gts.append(gt_m.cpu().flatten())

        pred_cat = torch.cat(preds)
        gt_cat = torch.cat(gts)
        m = tsfr.compute_metrics(pred_cat, gt_cat)
        tqdm.write(f"{method:<8} {n_steps:>8} {m['r2']:>8.4f} {m['mae']:>8.4f} {m['rmse']:>8.4f}")

    if args.vis_samples > 0:
        _save_vis_figure(fm_refiner, sr_module, coarse_model, batches[: args.vis_samples],
                          device, n_landsat_bands, target_stats, methods[0], n_steps_list,
                          args.n_avg, os.path.join(REPO, args.vis_out))


@torch.no_grad()
def _save_vis_figure(fm_refiner, sr_module, coarse_model, vis_batches, device,
                      n_landsat_bands, target_stats, method, n_steps_list,
                      n_avg, out_path):
    """Grid: rows = sample patches, columns = [Landsat, pseudo-PS, real-PS, coarse, GT, pred@n_steps=...]."""
    col_titles = ["Landsat", "pseudo-PS", "Real PS", "DAv2 coarse", "GT"] + [f"pred ({method}, n={n})" for n in n_steps_list]
    n_rows = len(vis_batches)
    n_cols = len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, batch in enumerate(vis_batches):
        inputs_lr, inputs_hr, targets = batch
        if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
        if inputs_hr.dim() == 5: inputs_hr = inputs_hr.squeeze(0)
        if targets.dim() == 5: targets = targets.squeeze(0)
        inputs_lr = inputs_lr.to(device)
        inputs_hr = inputs_hr.to(device)
        targets = targets.to(device)

        H_hr, W_hr = targets.shape[2], targets.shape[3]
        landsat = inputs_lr[:, :n_landsat_bands]
        landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

        landsat_for_dav2 = (landsat + 1.0) / 2.0
        h_coarse = run_dav2(coarse_model, landsat_for_dav2, target_size=(H_hr, W_hr))
        h_coarse = tsfr._normalize_coarse(h_coarse, target_stats)
        pseudo_ps = sr_module(landsat, target_size=(H_hr, W_hr))

        gt_m = tsfr._denormalize_target(targets, target_stats)
        vmax = float(gt_m.max().item())

        fixed_panels = [
            (tsfr._to_vis_rgb(landsat_hr[0]), True),
            (tsfr._to_vis_rgb(pseudo_ps[0]), True),
            (tsfr._to_vis_rgb(inputs_hr[0]), True),
            (tsfr._denormalize_target(h_coarse, target_stats)[0, 0].cpu().numpy(), False),
            (gt_m[0, 0].cpu().numpy(), False),
        ]
        for col, (data, is_rgb) in enumerate(fixed_panels):
            ax = axes[row, col]
            if is_rgb:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap="plasma", vmin=0.0, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            ax.axis("off")

        for i, n_steps in enumerate(n_steps_list):
            if n_avg > 1:
                h_fine = torch.stack([
                    fm_refiner.refine_with_ps(pseudo_ps, h_coarse, n_steps=n_steps,
                                               method=method, landsat_hr=landsat_hr)
                    for _ in range(n_avg)
                ]).mean(dim=0)
            else:
                h_fine = fm_refiner.refine_with_ps(pseudo_ps, h_coarse, n_steps=n_steps,
                                                    method=method, landsat_hr=landsat_hr)
            pred_m = tsfr._denormalize_target(h_fine, target_stats)
            col = 5 + i
            ax = axes[row, col]
            im = ax.imshow(pred_m[0, 0].cpu().numpy(), cmap="plasma", vmin=0.0, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nSaved comparison figure: {out_path}")


if __name__ == "__main__":
    main()
