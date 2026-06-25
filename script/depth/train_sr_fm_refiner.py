"""
Alternate-training script: SR module + FM Refiner.

Phase 1 (train SR, freeze FM):
  LS → SwinIR → pseudo-PS
  Loss = L1(pseudo_ps, real_ps)                        [pixel loss]
       + L1(CN_feat(pseudo_ps), CN_feat(real_ps))      [TDP loss via ControlNet hook]

Phase 2 (train FM, freeze SR):
  PS source per sample is drawn from three exclusive outcomes:
    p_null_drop  → null PS token (modality dropout)
    p_pseudo_ps  → SR-generated pseudo-PS (train-test alignment)
    remainder    → real PS
  FM training identical to train_fm_refiner.py.

Key constraint: DAV2 always receives original LS; SR output never enters DAV2.

Usage:
  python script/depth/train_sr_fm_refiner.py --config config/sr_fm_refiner_v0.yaml
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
import wandb
from datetime import datetime, timedelta
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from depthfm.sr_module import SRModule, build_sr_module
from src.util.config_util import recursive_load_config
from src.util.logging_util import config_logging, init_wandb, tb_logger
from src.util.ps_lazydataset import LazyPatchDataset
from src.util.seeding import generate_seed_sequence


# -------------------------------------------------------------------------
# Dataset helpers (identical to train_fm_refiner.py)
# -------------------------------------------------------------------------

def _split_coords(all_coords, train_split, val_split, seed):
    n = len(all_coords)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()
    train_size = int(train_split * n)
    val_size = int(val_split * n)
    return (
        [all_coords[i] for i in indices[:train_size]],
        [all_coords[i] for i in indices[train_size:train_size + val_size]],
        [all_coords[i] for i in indices[train_size + val_size:]],
    )


def _make_dataset(base_dataset, coords, mode, batch_size):
    dataset = copy.copy(base_dataset)
    dataset.batch_size = batch_size
    dataset.mode = mode
    dataset.patch_coords = coords
    dataset.patches_by_size = {}
    for patch_size in dataset.patch_sizes:
        dataset.patches_by_size[patch_size] = [
            p for p in coords
            if p['output_patch_height'] == patch_size[0]
            and p['output_patch_width'] == patch_size[1]
        ]
    print(f"[{mode}] Patches per size after split:")
    for size, patches in dataset.patches_by_size.items():
        print(f"  {size}: {len(patches)} patches")
    return dataset


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor):
    pred = pred.flatten().float()
    gt = gt.flatten().float()
    mask = torch.isfinite(gt) & torch.isfinite(pred)
    pred, gt = pred[mask], gt[mask]
    ss_res = ((gt - pred) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    mae = (gt - pred).abs().mean()
    rmse = ((gt - pred) ** 2).mean().sqrt()
    return {"r2": r2.item(), "mae": mae.item(), "rmse": rmse.item()}


# -------------------------------------------------------------------------
# Visualisation helpers
# -------------------------------------------------------------------------

def _to_vis_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor[:3].cpu().float()
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _to_vis_depth(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().cpu().float().numpy()


def _make_vis_figure(samples: list) -> plt.Figure:
    n = len(samples)
    col_titles = ["Landsat (RGB)", "pseudo-PS (RGB)", "DAv2 Coarse", "FM Refined", "GT"]
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, s in enumerate(samples):
        for col, (key, title) in enumerate(zip(
            ["landsat", "pseudo_ps", "coarse", "fine", "gt"], col_titles
        )):
            ax = axes[row, col]
            data = s[key]
            if key in ("landsat", "pseudo_ps"):
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap="plasma", vmin=-1.0, vmax=1.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------------
# Normalise DAv2 output
# -------------------------------------------------------------------------

def _normalize_coarse(h_coarse: torch.Tensor, target_stats: dict) -> torch.Tensor:
    p1 = float(target_stats["p1"][0])
    p99 = float(target_stats["p99"][0])
    h_norm = (h_coarse - p1) / (p99 - p1 + 1e-8)
    return (h_norm * 2.0 - 1.0).clamp(-1.0, 1.0)


def _load_target_stats(stats_file: str, year: int) -> dict:
    with open(stats_file) as f:
        all_stats = json.load(f)
    return all_stats[str(year)]


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

@torch.no_grad()
def validate(
    fm_refiner: FMRefiner,
    sr_module: SRModule,
    dav2_model,
    val_loader: DataLoader,
    device: torch.device,
    n_steps: int,
    target_stats: dict,
    n_landsat_bands: int,
    val_offset: int = 0,
    n_vis_samples: int = 5,
    full: bool = False,
    method: str = "euler",
    use_sr_pseudo_ps: bool = True,
) -> tuple:
    fm_refiner.eval()
    sr_module.eval()
    all_preds, all_gts = [], []
    vis_samples = []

    val_list = list(val_loader)
    n_total = len(val_list)
    if full:
        batches = val_list
    else:
        subset_size = max(1, n_total // 10)
        indices = [(val_offset + i) % n_total for i in range(subset_size)]
        batches = [val_list[i] for i in indices]

    for batch in tqdm(batches, desc="Validation", leave=False):
        inputs_lr, inputs_hr, targets = batch
        if inputs_lr.dim() == 5:
            inputs_lr = inputs_lr.squeeze(0)
        if inputs_hr.dim() == 5:
            inputs_hr = inputs_hr.squeeze(0)
        if targets.dim() == 5:
            targets = targets.squeeze(0)

        inputs_lr = inputs_lr.to(device)
        targets = targets.to(device)

        H_hr, W_hr = targets.shape[2], targets.shape[3]
        landsat = inputs_lr[:, :n_landsat_bands]
        landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

        landsat_for_dav2 = (landsat + 1.0) / 2.0
        h_coarse = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
        h_coarse = _normalize_coarse(h_coarse, target_stats)

        if use_sr_pseudo_ps:
            pseudo_ps = sr_module(landsat, target_size=(H_hr, W_hr))
            h_fine = fm_refiner.refine_with_ps(
                landsat_hr, pseudo_ps, h_coarse, n_steps=n_steps, method=method
            )
        else:
            h_fine = fm_refiner.refine(landsat_hr, h_coarse, n_steps=n_steps, method=method)

        all_preds.append(h_fine.cpu().flatten())
        all_gts.append(targets.cpu().flatten())

        if len(vis_samples) < n_vis_samples:
            pseudo_ps_vis = (
                sr_module(landsat, target_size=(H_hr, W_hr))[0]
                if use_sr_pseudo_ps else landsat_hr[0]
            )
            vis_samples.append({
                "landsat":   _to_vis_rgb(landsat_hr[0]),
                "pseudo_ps": _to_vis_rgb(pseudo_ps_vis),
                "coarse":    _to_vis_depth(h_coarse[0]),
                "fine":      _to_vis_depth(h_fine[0]),
                "gt":        _to_vis_depth(targets[0]),
            })

    pred_cat = torch.cat(all_preds, dim=0)
    gt_cat = torch.cat(all_gts, dim=0)
    metrics = compute_metrics(pred_cat, gt_cat)
    fig = _make_vis_figure(vis_samples) if vis_samples else None

    fm_refiner.train()
    sr_module.train()
    return metrics, fig


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def save_checkpoint(
    fm_refiner, sr_module, optimizer_fm, optimizer_sr,
    lr_scheduler_fm, lr_scheduler_sr,
    step, phase, steps_in_phase, out_dir, name="latest"
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save({
        "step": step,
        "phase": phase,
        "steps_in_phase": steps_in_phase,
        "unet_state": fm_refiner.unet.state_dict(),
        "controlnet_state": fm_refiner.controlnet.state_dict(),
        "null_ps_state": fm_refiner._null_ps,
        "sr_state": sr_module.state_dict(),
        "optimizer_fm_state": optimizer_fm.state_dict(),
        "optimizer_sr_state": optimizer_sr.state_dict(),
        "lr_scheduler_fm_state": lr_scheduler_fm.state_dict() if lr_scheduler_fm else None,
        "lr_scheduler_sr_state": lr_scheduler_sr.state_dict() if lr_scheduler_sr else None,
    }, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    fm_refiner, sr_module, optimizer_fm, optimizer_sr,
    lr_scheduler_fm, lr_scheduler_sr, path
):
    ckpt = torch.load(path, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    sr_module.load_state_dict(ckpt["sr_state"])
    optimizer_fm.load_state_dict(ckpt["optimizer_fm_state"])
    optimizer_sr.load_state_dict(ckpt["optimizer_sr_state"])
    if lr_scheduler_fm and ckpt.get("lr_scheduler_fm_state"):
        lr_scheduler_fm.load_state_dict(ckpt["lr_scheduler_fm_state"])
    if lr_scheduler_sr and ckpt.get("lr_scheduler_sr_state"):
        lr_scheduler_sr.load_state_dict(ckpt["lr_scheduler_sr_state"])
    return ckpt["step"], ckpt.get("phase", "sr"), ckpt.get("steps_in_phase", 0)


# -------------------------------------------------------------------------
# Phase helpers
# -------------------------------------------------------------------------

def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def _enter_sr_phase(fm_refiner: FMRefiner, sr_module: SRModule):
    """Phase 1: train SR only."""
    _freeze(fm_refiner.unet)
    _freeze(fm_refiner.controlnet)
    _unfreeze(sr_module)
    fm_refiner.eval()
    sr_module.train()


def _enter_task_phase(fm_refiner: FMRefiner, sr_module: SRModule):
    """Phase 2: train FM only."""
    _unfreeze(fm_refiner.unet)
    _unfreeze(fm_refiner.controlnet)
    _freeze(sr_module)
    fm_refiner.train()
    sr_module.eval()


# -------------------------------------------------------------------------
# Phase-1 training step
# -------------------------------------------------------------------------

def _sr_step(
    fm_refiner: FMRefiner,
    sr_module: SRModule,
    landsat: torch.Tensor,       # (B, C_ls, H_lr, W_lr)
    landsat_hr: torch.Tensor,    # (B, C_ls, H_hr, W_hr)
    real_ps: torch.Tensor,       # (B, C_ps, H_hr, W_hr)
    h_coarse: torch.Tensor,      # (B, 1, H_hr, W_hr)
    pixel_weight: float,
    tdp_weight: float,
    use_tdp: bool,               # False during warmup
    device: torch.device,
) -> dict:
    """
    Compute SR Phase-1 loss and return a dict with loss tensors.

    Gradient path (pixel loss):
      pseudo_ps ← SR ← landsat

    Gradient path (TDP loss):
      ControlNet mid-block feature(pseudo_ps) ← ControlNet ← pseudo_ps ← SR ← landsat
    """
    B = landsat.shape[0]
    H_hr, W_hr = real_ps.shape[2], real_ps.shape[3]

    # SR forward (trainable)
    pseudo_ps = sr_module(landsat, target_size=(H_hr, W_hr))  # (B, C_ps, H_hr, W_hr)

    # Pixel loss: L1(pseudo_ps, real_ps)
    l_pix = F.l1_loss(pseudo_ps, real_ps.detach()) * pixel_weight

    l_tdp = torch.tensor(0.0, device=device)
    if use_tdp and tdp_weight > 0:
        # TDP loss via forward hook on ControlNet mid_block.
        # Runs ControlNet twice (pseudo vs real PS) to obtain feature maps at the same
        # timestep, then applies L1. DAV2 coarse latent is used as the 'sample' arg.
        with torch.no_grad():
            z_coarse = fm_refiner.encode(h_coarse)

        text_emb = fm_refiner.empty_text_embed.to(device).expand(B, -1, -1)
        t_val = torch.full((B,), 0.5, device=device)
        t_int = (t_val * 999).long()

        feats: dict = {}

        def _hook(m, inp, out):
            feats["out"] = out

        hook = fm_refiner.controlnet.mid_block.register_forward_hook(_hook)

        # Forward with pseudo-PS (grad flows here)
        cn_fake_input = torch.cat([landsat_hr, pseudo_ps], dim=1)
        fm_refiner.controlnet(
            sample=z_coarse.detach(),
            timestep=t_int,
            encoder_hidden_states=text_emb,
            controlnet_cond=cn_fake_input,
            return_dict=True,
        )
        feat_fake = feats["out"]

        # Forward with real PS (no grad – reference only)
        with torch.no_grad():
            cn_real_input = torch.cat([landsat_hr, real_ps], dim=1)
            fm_refiner.controlnet(
                sample=z_coarse.detach(),
                timestep=t_int,
                encoder_hidden_states=text_emb,
                controlnet_cond=cn_real_input,
                return_dict=True,
            )
            feat_real = feats["out"]

        hook.remove()

        l_tdp = F.l1_loss(feat_fake, feat_real.detach()) * tdp_weight

    return {"l_pix": l_pix, "l_tdp": l_tdp, "l_total": l_pix + l_tdp}


# -------------------------------------------------------------------------
# Phase-2 training step
# -------------------------------------------------------------------------

def _task_step(
    fm_refiner: FMRefiner,
    sr_module: SRModule,
    landsat_hr: torch.Tensor,    # (B, C_ls, H_hr, W_hr)
    real_ps: torch.Tensor,       # (B, C_ps, H_hr, W_hr)
    h_coarse: torch.Tensor,
    h_gt: torch.Tensor,
    p_null_drop: float,
    p_pseudo_ps: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute FM training loss (flow matching MSE on velocities).

    PS source is sampled per-sample from three exclusive outcomes:
      p_null_drop  → null PS token
      p_pseudo_ps  → SR pseudo-PS (frozen)
      remainder    → real PS
    """
    B = real_ps.shape[0]
    H_hr, W_hr = real_ps.shape[2], real_ps.shape[3]

    # Decide PS source per sample
    r = torch.rand(B, device=device)
    null_mask = r < p_null_drop
    pseudo_mask = (r >= p_null_drop) & (r < p_null_drop + p_pseudo_ps)

    # Assemble ps_cond
    ps_cond = real_ps.clone()

    # Null PS
    null_ps = fm_refiner._get_null_ps(real_ps.shape[1], device, real_ps.dtype)
    null_spatial = null_ps.expand(1, -1, H_hr, W_hr)
    for i in range(B):
        if null_mask[i]:
            ps_cond[i] = null_spatial[0]

    # Pseudo-PS from frozen SR
    if pseudo_mask.any():
        with torch.no_grad():
            landsat_lr = F.interpolate(
                landsat_hr, scale_factor=1.0 / sr_module.upscale,
                mode="bilinear", align_corners=False,
            )
            pseudo_batch = sr_module(landsat_lr, target_size=(H_hr, W_hr))
        for i in range(B):
            if pseudo_mask[i]:
                ps_cond[i] = pseudo_batch[i]

    loss = fm_refiner(
        landsat_lr=landsat_hr,
        ps_hr=real_ps,
        h_coarse=h_coarse,
        h_gt=h_gt,
        ps_cond_override=ps_cond,
    )
    return loss


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="SR + FM Refiner Alternate Training")
    parser.add_argument("--config", type=str, default="config/sr_fm_refiner_v1.yaml")
    parser.add_argument("--resume_run", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--exit_after", type=int, default=-1, help="Exit after X minutes.")
    parser.add_argument("--add_datetime_prefix", action="store_true")
    args = parser.parse_args()

    # ---- Config ----
    if args.resume_run is not None:
        out_dir_run = os.path.dirname(os.path.dirname(args.resume_run))
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
        job_name = os.path.basename(out_dir_run)
    else:
        cfg = recursive_load_config(args.config)
        pure_job_name = os.path.basename(args.config).split(".")[0]
        job_name = (
            f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
            if args.add_datetime_prefix else pure_job_name
        )
        out_dir_run = os.path.join(args.output_dir or "./output", job_name)
        os.makedirs(out_dir_run, exist_ok=False)

    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    os.makedirs(out_dir_ckpt, exist_ok=True)

    # ---- Logging ----
    config_logging(cfg.logging, out_dir=out_dir_run)
    if args.resume_run is None:
        with open(os.path.join(out_dir_run, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    if not args.no_wandb:
        wandb_cfg = {
            "config": dict(cfg),
            "name": job_name,
            "mode": "online",
            "dir": out_dir_run,
            **{k: v for k, v in cfg.wandb.items() if k != "name"},
        }
        init_wandb(enable=True, **wandb_cfg)
    else:
        init_wandb(enable=False)

    tb_logger.set_dir(os.path.join(out_dir_run, "tensorboard"))

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"device = {device}")

    # ---- Data ----
    cfg_data = cfg.dataset
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs // cfg.dataloader.max_train_batch_size

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
    train_coords, val_coords, _ = _split_coords(
        all_coords, cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed
    )
    train_dataset = _make_dataset(base_dataset, train_coords, 'train', cfg.dataloader.max_train_batch_size)
    val_dataset   = _make_dataset(base_dataset, val_coords,   'val',   1)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=cfg_data.workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False,
                              num_workers=cfg_data.workers, pin_memory=True)

    # ---- Models ----
    n_landsat_bands = len(cfg_data.selected_bands)
    n_ps_bands      = len(cfg_data.selected_bands_hr)

    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=cfg.trainer.get("ps_dropout_p", 0.0),
        device=str(device),
    ).to(device)

    dav2_model = load_dav2(
        dav2_path=cfg.model.dav2_pretrained_path,
        backbone=cfg.model.dav2_backbone,
        out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
    ).to(device)

    sr_module = build_sr_module(OmegaConf.to_container(cfg.sr_module, resolve=True)).to(device)

    # Load SwinIR pretrained weights if provided
    sr_pretrained_path = cfg.model.get("sr_pretrained_path", None)
    if sr_pretrained_path and os.path.exists(sr_pretrained_path):
        ckpt_sr = torch.load(sr_pretrained_path, map_location="cpu")
        key = cfg.model.get("sr_pretrained_key", None)
        state = ckpt_sr[key] if key and key in ckpt_sr else ckpt_sr
        missing, unexpected = sr_module.swinir.load_state_dict(state, strict=False)
        logging.info(f"SwinIR pretrained loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        logging.info("No SwinIR pretrained weights found; training from scratch.")

    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ---- Optimizers ----
    optimizer_fm = torch.optim.AdamW(
        [
            {"params": fm_refiner.unet.parameters(),       "lr": cfg.optimizer.lr_unet},
            {"params": fm_refiner.controlnet.parameters(), "lr": cfg.optimizer.lr_controlnet},
        ],
        weight_decay=cfg.optimizer.weight_decay,
    )
    optimizer_sr = torch.optim.AdamW(
        sr_module.parameters(),
        lr=cfg.optimizer.lr_sr,
        weight_decay=cfg.optimizer.weight_decay_sr,
    )

    # ---- LR schedulers ----
    lr_scheduler_fm, lr_scheduler_sr = None, None
    if cfg.get("lr_scheduler") is not None:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler_fm = CosineAnnealingLR(optimizer_fm, T_max=cfg.max_iter,
                                             eta_min=cfg.lr_scheduler.eta_min)
        lr_scheduler_sr = CosineAnnealingLR(optimizer_sr, T_max=cfg.max_iter,
                                             eta_min=cfg.lr_scheduler.eta_min)

    # ---- Resume ----
    start_step = 0
    current_phase = cfg.alternate.start_phase   # 'sr' or 'task'
    steps_in_phase = 0
    if args.resume_run is not None:
        start_step, current_phase, steps_in_phase = load_checkpoint(
            fm_refiner, sr_module, optimizer_fm, optimizer_sr,
            lr_scheduler_fm, lr_scheduler_sr, args.resume_run
        )
        logging.info(f"Resumed from step {start_step}, phase={current_phase}, "
                     f"steps_in_phase={steps_in_phase}")

    # ---- Alternate config ----
    alt = cfg.alternate
    sr_phase_steps   = alt.sr_phase_steps
    task_phase_steps = alt.task_phase_steps
    p_null_drop      = float(alt.p_null_drop)
    p_pseudo_ps      = float(alt.p_pseudo_ps)
    warmup_sr_steps  = int(alt.warmup_sr_steps)
    pixel_weight     = float(cfg.sr_loss.pixel_weight)
    tdp_weight       = float(cfg.sr_loss.tdp_weight)

    # Initialise freeze states for starting phase
    if current_phase == "sr":
        _enter_sr_phase(fm_refiner, sr_module)
    else:
        _enter_task_phase(fm_refiner, sr_module)

    # ---- Training loop ----
    t_end = t_start + timedelta(minutes=args.exit_after) if args.exit_after > 0 else None

    step = start_step
    best_r2 = -1e8
    val_offset = 0
    val_subset_size = max(1, len(val_loader) // 10)

    # Accum state per phase
    accum_sr_step = 0
    loss_pix_accum = 0.0
    loss_tdp_accum = 0.0
    loss_fm_accum  = 0.0

    logging.info(f"Starting alternate training. Initial phase: {current_phase}")
    pbar = tqdm(total=cfg.max_iter, initial=step, desc="Training", dynamic_ncols=True)

    for epoch in range(cfg.max_epoch):
        for batch in train_loader:
            if step >= cfg.max_iter:
                break
            if t_end is not None and datetime.now() >= t_end:
                logging.info("Exit after time limit reached.")
                save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                                lr_scheduler_fm, lr_scheduler_sr,
                                step, current_phase, steps_in_phase, out_dir_ckpt, "latest")
                pbar.close()
                sys.exit(0)

            # ---- Phase switching ----
            phase_budget = sr_phase_steps if current_phase == "sr" else task_phase_steps
            if steps_in_phase >= phase_budget:
                current_phase = "task" if current_phase == "sr" else "sr"
                steps_in_phase = 0
                if current_phase == "sr":
                    _enter_sr_phase(fm_refiner, sr_module)
                    logging.info(f"[step {step}] → Phase 1 (SR)")
                else:
                    _enter_task_phase(fm_refiner, sr_module)
                    logging.info(f"[step {step}] → Phase 2 (Task/FM)")

            # ---- Unpack batch ----
            inputs_lr, inputs_hr, targets = batch
            if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
            if inputs_hr.dim() == 5: inputs_hr = inputs_hr.squeeze(0)
            if targets.dim()   == 5: targets   = targets.squeeze(0)

            inputs_lr = inputs_lr.to(device)
            inputs_hr = inputs_hr.to(device)
            targets   = targets.to(device)

            H_hr, W_hr = targets.shape[2], targets.shape[3]
            landsat    = inputs_lr[:, :n_landsat_bands]
            landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

            # DAV2 coarse (always from original LS, always frozen)
            with torch.no_grad():
                landsat_for_dav2 = (landsat + 1.0) / 2.0
                h_coarse = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
                h_coarse = _normalize_coarse(h_coarse, target_stats)

            # ---- Phase 1: train SR ----
            if current_phase == "sr":
                use_tdp = (steps_in_phase >= warmup_sr_steps)
                loss_dict = _sr_step(
                    fm_refiner, sr_module,
                    landsat, landsat_hr, inputs_hr, h_coarse,
                    pixel_weight, tdp_weight, use_tdp, device,
                )
                loss = loss_dict["l_total"] / accumulation_steps
                loss.backward()

                loss_pix_accum += loss_dict["l_pix"].item()
                loss_tdp_accum += loss_dict["l_tdp"].item()
                accum_sr_step  += 1

                if accum_sr_step % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(sr_module.parameters(), max_norm=1.0)
                    optimizer_sr.step()
                    if lr_scheduler_sr is not None:
                        lr_scheduler_sr.step()
                    optimizer_sr.zero_grad()

            # ---- Phase 2: train FM ----
            else:
                loss = _task_step(
                    fm_refiner, sr_module,
                    landsat_hr, inputs_hr, h_coarse, targets,
                    p_null_drop, p_pseudo_ps, device,
                )
                loss = loss / accumulation_steps
                loss.backward()
                loss_fm_accum += loss.item() * accumulation_steps

                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(fm_refiner.unet.parameters()) +
                        list(fm_refiner.controlnet.parameters()),
                        max_norm=1.0,
                    )
                    optimizer_fm.step()
                    if lr_scheduler_fm is not None:
                        lr_scheduler_fm.step()
                    optimizer_fm.zero_grad()

            step += 1
            steps_in_phase += 1
            pbar.update(1)

            # ---- Logging ----
            if step % cfg.trainer.log_period == 0:
                lr_sr = optimizer_sr.param_groups[0]["lr"]
                lr_fm = optimizer_fm.param_groups[0]["lr"]
                log_n = cfg.trainer.log_period
                log_dict = {
                    "phase": 0 if current_phase == "sr" else 1,
                    "lr/sr": lr_sr, "lr/fm": lr_fm,
                    "train/l_pix": loss_pix_accum / max(log_n, 1),
                    "train/l_tdp": loss_tdp_accum / max(log_n, 1),
                    "train/l_fm":  loss_fm_accum  / max(log_n, 1),
                }
                logging.info(
                    f"[step {step}] phase={current_phase} "
                    f"l_pix={log_dict['train/l_pix']:.5f} "
                    f"l_tdp={log_dict['train/l_tdp']:.5f} "
                    f"l_fm={log_dict['train/l_fm']:.5f}"
                )
                pbar.set_postfix(phase=current_phase, r2=f"{best_r2:.4f}")
                wandb.log(log_dict, step=step)
                tb_logger.log_dict(log_dict, global_step=step)
                loss_pix_accum = loss_tdp_accum = loss_fm_accum = 0.0

            # ---- Validation ----
            if step % cfg.trainer.validation_period == 0:
                metrics, fig = validate(
                    fm_refiner, sr_module, dav2_model, val_loader, device,
                    n_steps=cfg.validation.n_steps,
                    target_stats=target_stats,
                    n_landsat_bands=n_landsat_bands,
                    val_offset=val_offset,
                    method=cfg.validation.get("method", "euler"),
                    use_sr_pseudo_ps=cfg.validation.get("use_sr_pseudo_ps", True),
                )
                val_offset = (val_offset + val_subset_size) % len(val_loader)

                logging.info(f"[step {step}] val: {metrics}")
                vlog = {f"val/{k}": v for k, v in metrics.items()}
                if fig is not None:
                    vlog["val/samples"] = wandb.Image(fig)
                    plt.close(fig)
                wandb.log(vlog, step=step)
                tb_logger.log_dict({f"val/{k}": v for k, v in metrics.items()}, global_step=step)

                if metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    pbar.set_postfix(phase=current_phase, r2=f"{best_r2:.4f}")
                    save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                                    lr_scheduler_fm, lr_scheduler_sr,
                                    step, current_phase, steps_in_phase, out_dir_ckpt, "best")
                    logging.info(f"New best R²={best_r2:.4f} at step {step}")

                # Re-enter the correct phase (validate() sets eval mode)
                if current_phase == "sr":
                    _enter_sr_phase(fm_refiner, sr_module)
                else:
                    _enter_task_phase(fm_refiner, sr_module)

            # ---- Periodic checkpoint ----
            if step % cfg.trainer.save_period == 0:
                save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                                lr_scheduler_fm, lr_scheduler_sr,
                                step, current_phase, steps_in_phase, out_dir_ckpt, "latest")

        if step >= cfg.max_iter:
            break

    pbar.close()

    # ---- Final full validation ----
    best_ckpt = os.path.join(out_dir_ckpt, "best.pth")
    if os.path.exists(best_ckpt):
        load_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                        lr_scheduler_fm, lr_scheduler_sr, best_ckpt)
        logging.info(f"Loaded best checkpoint for final validation")

    logging.info("Running full validation on entire val set...")
    final_metrics, final_fig = validate(
        fm_refiner, sr_module, dav2_model, val_loader, device,
        n_steps=cfg.validation.n_steps,
        target_stats=target_stats,
        n_landsat_bands=n_landsat_bands,
        full=True,
        method=cfg.validation.get("method", "euler"),
        use_sr_pseudo_ps=cfg.validation.get("use_sr_pseudo_ps", True),
    )
    logging.info(f"Final val metrics: {final_metrics}")
    flog = {f"val_final/{k}": v for k, v in final_metrics.items()}
    if final_fig is not None:
        flog["val_final/samples"] = wandb.Image(final_fig)
        plt.close(final_fig)
    wandb.log(flog, step=step)

    save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                    lr_scheduler_fm, lr_scheduler_sr,
                    step, current_phase, steps_in_phase, out_dir_ckpt, "final")
    logging.info(f"Training finished at step {step}. Best val R²={best_r2:.4f}")
