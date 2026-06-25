"""
Diagnostic inference experiment: frozen SR module generating pseudo-PS for ControlNet.

Experiment rules (strict):
  - FM Refiner (DAv2 + UNet + ControlNet): loaded from checkpoint, fully frozen.
  - SwinIR (SR4IR project): loaded independently, fully frozen.
  - For every validation sample:
      1. Generate pseudo-PS from LS via frozen SwinIR.
      2. Feed pseudo-PS into ControlNet branch — 100% pseudo-PS, real PS never used.
      3. DAv2 still takes original LS (not pseudo-PS, per scheme A).
      4. Run FM main trunk → final CHM.
  - No parameter updates. No modality dropout. No real PS.

Normalisation bridge between projects:
  Marigold Landsat  : [-1, 1]   (MinMax → [0,1] → ×2-1)
  SR4IR SwinIR input: [ 0, 1]   (MinMax only)
  => ls_for_sr = (landsat + 1.0) / 2.0

  SwinIR output     : [ 0, 1]
  ControlNet PS     : [-1, 1]   (same as Marigold PS during training)
  => pseudo_ps = sr_out * 2.0 - 1.0

Usage:
  python script/depth/eval_pseudo_ps.py --config config/eval_pseudo_ps.yaml
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
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset
from script.depth.train_fm_refiner import (
    _split_coords, _make_dataset,
    _normalize_coarse, _load_target_stats,
    compute_metrics,
    _to_vis_depth,
)


# ---------------------------------------------------------------------------
# FM Refiner ODE integration with explicit PS (no dropout, no null token)
# ---------------------------------------------------------------------------

@torch.no_grad()
def refine_with_ps(
    fm_refiner: FMRefiner,
    landsat_hr: torch.Tensor,   # (B, C_ls, H_hr, W_hr)  [-1, 1]
    h_coarse: torch.Tensor,      # (B, 1,    H_hr, W_hr)  [-1, 1]
    pseudo_ps: torch.Tensor,     # (B, C_ps, H_hr, W_hr)  [-1, 1]
    n_steps: int = 2,
    method: str = "euler",
) -> torch.Tensor:
    """ODE integration using pseudo_ps — ControlNet branch always active."""
    device = h_coarse.device
    dtype  = h_coarse.dtype
    B      = h_coarse.shape[0]

    z = fm_refiner.encode(h_coarse)                                    # (B, 4, h, w)
    control_input = torch.cat([landsat_hr, pseudo_ps], dim=1)          # (B, C_ls+C_ps, H, W)
    text_emb = fm_refiner.empty_text_embed.to(device, dtype).expand(B, -1, -1)

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

    return fm_refiner.decode(z)   # (B, 1, H_hr, W_hr)


# ---------------------------------------------------------------------------
# SwinIR loader
# ---------------------------------------------------------------------------

def load_swinir(arch_cfg: dict, scale: int, ckpt_path: str) -> nn.Module:
    """
    Load SwinIR from depthfm/swinir_arch.py (Marigold project).

    Checkpoint formats supported:
      - bare state dict
      - dict with 'net_sr' key
    """
    from depthfm.swinir_arch import SwinIR

    opt = dict(arch_cfg)
    opt["upscale"] = scale
    net_sr = SwinIR(**opt)

    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw["net_sr"] if (isinstance(raw, dict) and "net_sr" in raw) else raw
    net_sr.load_state_dict(state, strict=True)
    net_sr.requires_grad_(False)
    net_sr.eval()
    logging.info(f"SwinIR loaded  ← {ckpt_path}")
    return net_sr


# ---------------------------------------------------------------------------
# Visualisation: 5-panel grid
# ---------------------------------------------------------------------------

def _to_rgb01(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) any range → (H, W, 3) float in [0, 1] for imshow."""
    img = t[:3].cpu().float()
    img = img - img.min()
    rng = img.max()
    if rng > 1e-8:
        img = img / rng
    return img.permute(1, 2, 0).numpy()


def _make_vis(samples: list) -> plt.Figure:
    titles = ["Landsat (RGB)", "Real PS (RGB)", "Pseudo-PS (RGB)", "FM Refined CHM", "GT CHM"]
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(22, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, s in enumerate(samples):
        for col, (key, title) in enumerate(
            zip(["landsat", "real_ps", "pseudo_ps", "fine", "gt"], titles)
        ):
            ax = axes[row, col]
            d  = s[key]
            if col < 3:
                ax.imshow(np.clip(d, 0, 1))
            else:
                im = ax.imshow(d, cmap="plasma", vmin=-1.0, vmax=1.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    fm_refiner, dav2_model, net_sr,
    val_loader, device,
    n_steps, target_stats, n_landsat_bands, n_vis,
):
    all_preds, all_gts = [], []
    vis_samples = []

    for batch in tqdm(val_loader, desc="Eval (pseudo-PS)"):
        inputs_lr, inputs_hr, targets = batch
        if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
        if inputs_hr.dim() == 5: inputs_hr = inputs_hr.squeeze(0)
        if targets.dim()   == 5: targets   = targets.squeeze(0)

        inputs_lr = inputs_lr.to(device)
        inputs_hr = inputs_hr.to(device)
        targets   = targets.to(device)

        H_hr, W_hr = targets.shape[2], targets.shape[3]
        landsat    = inputs_lr[:, :n_landsat_bands]              # (B, 3, H_lr, W_lr)

        # ControlNet also needs landsat upsampled to HR
        landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr),
                                   mode="bilinear", align_corners=False)

        # ── Pseudo-PS via SwinIR ─────────────────────────────────────────
        # Marigold Landsat is in [-1,1]; SwinIR expects [0,1]
        ls_for_sr    = (landsat + 1.0) / 2.0          # [-1,1] → [0,1]
        pseudo_ps_01 = net_sr(ls_for_sr.float())       # (B, 3, H_hr, W_hr)  [0,1]
        # ControlNet PS branch was trained with PS in [-1,1]
        pseudo_ps    = pseudo_ps_01 * 2.0 - 1.0        # [0,1] → [-1,1]

        # ── DAv2 coarse from original LS (not pseudo-PS) ─────────────────
        landsat_for_dav2 = (landsat + 1.0) / 2.0
        h_coarse = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
        h_coarse = _normalize_coarse(h_coarse, target_stats)

        # ── FM refinement — ControlNet always active with pseudo-PS ──────
        h_fine = refine_with_ps(fm_refiner, landsat_hr, h_coarse, pseudo_ps,
                                 n_steps=n_steps)

        all_preds.append(h_fine.cpu().flatten())
        all_gts.append(targets.cpu().flatten())

        if len(vis_samples) < n_vis:
            vis_samples.append({
                "landsat":   _to_rgb01(landsat_hr[0]),
                "real_ps":   _to_rgb01(inputs_hr[0]),
                "pseudo_ps": _to_rgb01(pseudo_ps[0]),
                "fine":      _to_vis_depth(h_fine[0]),
                "gt":        _to_vis_depth(targets[0]),
            })

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_gts))
    fig     = _make_vis(vis_samples) if vis_samples else None
    return metrics, fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/eval_pseudo_ps.yaml",
                        help="Path to eval_pseudo_ps.yaml")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    # ── Load experiment config ───────────────────────────────────────────────
    exp_cfg = OmegaConf.load(args.config)

    # ── Load FM Refiner training config ─────────────────────────────────────
    fm_cfg      = recursive_load_config(exp_cfg.fm_refiner.config)
    cfg_data    = fm_cfg.dataset

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"device = {device}")

    # ── Data (val split) ────────────────────────────────────────────────────
    n_landsat_bands = len(cfg_data.selected_bands)
    n_ps_bands      = len(cfg_data.selected_bands_hr)

    base_dataset = LazyPatchDataset({
        "input_dir":          cfg_data.input_dir,
        "input_dir_hr":       cfg_data.input_dir_hr,
        "output_dir":         cfg_data.output_dir,
        "selected_bands":     cfg_data.selected_bands,
        "selected_bands_hr":  cfg_data.selected_bands_hr,
        "file_type_input":    cfg_data.file_type_input,
        "file_type_output":   cfg_data.file_type_output,
        "patch_sizes":        [(s, s) for s in cfg_data.patch_sizes],
        "patch_coord_path":   cfg_data.patch_coord_path,
        "num_patches_per_tile": cfg_data.num_patches_per_tile,
        "tile_emphasis":      cfg_data.tile_emphasis,
        "correlation_cleaning": cfg_data.correlation_cleaning,
        "target_range_edges": cfg_data.target_range_edges,
        "num_patches_per_target_range": cfg_data.num_patches_per_target_range,
        "selected_percentile": cfg_data.selected_percentile,
        "year":               cfg_data.year,
        "batch_size":         1,
        "mode":               "val",
        "use_input_minmax":   cfg_data.use_input_minmax,
        "use_input_norm":     cfg_data.use_input_norm,
        "input_stats_file":   cfg_data.input_stats_file,
        "use_input_hr_minmax": cfg_data.use_input_hr_minmax,
        "use_input_hr_norm":  cfg_data.use_input_hr_norm,
        "input_hr_stats_file": cfg_data.input_hr_stats_file,
        "use_target_minmax":  cfg_data.use_target_minmax,
        "use_target_norm":    cfg_data.use_target_norm,
        "target_stats_file":  cfg_data.target_stats_file,
        "unit_scale_ratio":   cfg_data.unit_scale_ratio,
        "scale_input_to_neg1_1":    cfg_data.get("scale_input_to_neg1_1", False),
        "scale_input_hr_to_neg1_1": cfg_data.get("scale_input_hr_to_neg1_1", False),
        "scale_target_to_neg1_1":   cfg_data.get("scale_target_to_neg1_1", False),
    })

    _, val_coords, _ = _split_coords(
        base_dataset.patch_coords,
        cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed,
    )
    val_dataset = _make_dataset(base_dataset, val_coords, "val", 1)
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             num_workers=cfg_data.workers, pin_memory=True)
    logging.info(f"Val patches: {len(val_loader)}")

    # ── FM Refiner (frozen) ─────────────────────────────────────────────────
    fm_refiner = build_fm_refiner(
        sd_pretrained_path=fm_cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=fm_cfg.trainer.ps_dropout_p,
        device=str(device),
    ).to(device)
    fm_refiner.requires_grad_(False)
    fm_refiner.eval()

    ckpt = torch.load(exp_cfg.fm_refiner.ckpt, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"].to(device)
    logging.info(f"FM Refiner  ← {exp_cfg.fm_refiner.ckpt}  (step {ckpt.get('step','?')})")

    # ── DAv2 (frozen) ───────────────────────────────────────────────────────
    dav2_model = load_dav2(
        dav2_path=fm_cfg.model.dav2_pretrained_path,
        backbone=fm_cfg.model.dav2_backbone,
        out_in_scale_factor=fm_cfg.model.dav2_out_in_scale_factor,
    ).to(device)

    # ── SwinIR from SR4IR (frozen) ──────────────────────────────────────────
    sr_cfg = exp_cfg.sr_module
    net_sr = load_swinir(
        arch_cfg=OmegaConf.to_container(sr_cfg.arch, resolve=True),
        scale=sr_cfg.scale,
        ckpt_path=sr_cfg.ckpt,
    ).to(device)

    logging.info(
        "Normalisation bridge:\n"
        "  Marigold Landsat [-1,1] → SwinIR:      (x+1)/2\n"
        "  SwinIR output   [0,1]  → ControlNet PS: x*2-1"
    )

    # ── Target stats ────────────────────────────────────────────────────────
    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ── Run ─────────────────────────────────────────────────────────────────
    n_steps = exp_cfg.fm_refiner.n_steps
    n_vis   = exp_cfg.output.n_vis
    metrics, fig = evaluate(
        fm_refiner, dav2_model, net_sr,
        val_loader, device,
        n_steps=n_steps,
        target_stats=target_stats,
        n_landsat_bands=n_landsat_bands,
        n_vis=n_vis,
    )

    # ── Print comparison ────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║    Diagnostic: frozen SR pseudo-PS  —  val results   ║")
    print("╠══════════════════════════════════════════════════════╣")
    for b in exp_cfg.baselines:
        print(f"║  baseline  [{b.name}]")
        print(f"║    r2: {b.r2:.4f}")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  this run  (frozen SR pseudo-PS, 100% ControlNet)    ║")
    for k, v in metrics.items():
        arrow = "  ←" if k == "r2" else ""
        print(f"║    {k:>6}: {v:.4f}{arrow}")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Save ────────────────────────────────────────────────────────────────
    save_dir = exp_cfg.output.save_dir or os.path.dirname(exp_cfg.fm_refiner.ckpt)
    os.makedirs(save_dir, exist_ok=True)

    txt_path = os.path.join(save_dir, "eval_pseudo_ps_metrics.txt")
    with open(txt_path, "w") as f:
        for b in exp_cfg.baselines:
            f.write(f"baseline [{b.name}]  r2: {b.r2}\n")
        f.write("this_run (frozen SR pseudo-PS):\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
    logging.info(f"Metrics → {txt_path}")

    if fig is not None:
        fig_path = os.path.join(save_dir, f"eval_pseudo_ps_vis_{n_vis}samples.png")
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        logging.info(f"Visualisation ({n_vis} samples) → {fig_path}")
        plt.close(fig)
