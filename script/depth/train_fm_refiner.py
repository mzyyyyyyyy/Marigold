"""
Training script for Flow Matching CHM Refiner.

Pipeline:
  Landsat → DAv2(frozen) → H_coarse (HR)
  [H_coarse, H_gt] → VAE → [z_coarse, z_gt]
  z_t = (1-t)*z_coarse + t*z_gt
  v_pred = UNet(z_t, t) + ControlNet(Landsat_HR, PS_or_null)
  loss = MSE(v_pred, z_gt - z_coarse)

Usage:
  python script/depth/train_fm_refiner.py --config config/fm_refiner.yaml
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
import wandb
from datetime import datetime, timedelta
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from src.util.config_util import recursive_load_config
from src.util.logging_util import config_logging, init_wandb, tb_logger
from src.util.ps_lazydataset import LazyPatchDataset
from src.util.seeding import generate_seed_sequence


# -------------------------------------------------------------------------
# Dataset helpers (identical to train_mm.py)
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
    """
    pred, gt: (N,) or (B, *) tensors in physical units (after inverse norm).
    Returns dict with r2, mae, rmse.
    """
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
# Validation
# -------------------------------------------------------------------------

def _to_vis_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [-1,1] to (H, W, 3) uint8 numpy array."""
    img = tensor[:3].cpu().float()
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _to_vis_depth(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) or (H, W) depth tensor in [-1,1] to (H, W) float numpy."""
    d = tensor.squeeze().cpu().float().numpy()
    return d


def _make_vis_figure(samples: list) -> plt.Figure:
    """
    Draw a grid of 5 columns × N rows.
    Each sample is a dict with keys: landsat, ps, coarse, fine, gt.
    """
    n = len(samples)
    col_titles = ["Landsat (RGB)", "PlanetScope (RGB)", "DAv2 Coarse", "FM Refined", "GT"]
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(samples):
        for col, (key, title) in enumerate(zip(
            ["landsat", "ps", "coarse", "fine", "gt"], col_titles
        )):
            ax = axes[row, col]
            data = s[key]
            if key in ("landsat", "ps"):
                ax.imshow(data)
            else:
                vmin, vmax = -1.0, 1.0
                im = ax.imshow(data, cmap="plasma", vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    return fig


@torch.no_grad()
def validate(
    fm_refiner: FMRefiner,
    dav2_model,
    val_loader: DataLoader,
    device: torch.device,
    n_steps: int,
    target_stats: dict,
    n_landsat_bands: int,
    val_offset: int = 0,           # rotating start index into val_loader
    n_vis_samples: int = 5,        # how many samples to visualize
    full: bool = False,            # if True, iterate entire val_loader
    method: str = "euler",         # integration method: "euler" or "heun"
) -> tuple:
    """
    Returns (metrics_dict, wandb_figure_or_None).

    Subset mode: iterates val_subset_size batches starting from val_offset,
    wrapping around circularly.  full=True ignores offset and iterates all.
    """
    fm_refiner.eval()
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
        inputs_hr = inputs_hr.to(device)
        targets = targets.to(device)

        H_hr, W_hr = targets.shape[2], targets.shape[3]

        landsat = inputs_lr[:, :n_landsat_bands]
        landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

        landsat_for_dav2 = (landsat + 1.0) / 2.0
        h_coarse = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
        h_coarse = _normalize_coarse(h_coarse, target_stats)

        h_fine = fm_refiner.refine(landsat_hr, h_coarse, n_steps=n_steps, method=method)

        all_preds.append(h_fine.cpu().flatten())
        all_gts.append(targets.cpu().flatten())

        # Collect vis samples from the first image of this batch
        if len(vis_samples) < n_vis_samples:
            vis_samples.append({
                "landsat": _to_vis_rgb(landsat_hr[0]),
                "ps":      _to_vis_rgb(inputs_hr[0]),
                "coarse":  _to_vis_depth(h_coarse[0]),
                "fine":    _to_vis_depth(h_fine[0]),
                "gt":      _to_vis_depth(targets[0]),
            })

    pred_cat = torch.cat(all_preds, dim=0)
    gt_cat = torch.cat(all_gts, dim=0)
    metrics = compute_metrics(pred_cat, gt_cat)

    fig = _make_vis_figure(vis_samples) if vis_samples else None

    fm_refiner.train()
    return metrics, fig


def _normalize_coarse(h_coarse: torch.Tensor, target_stats: dict) -> torch.Tensor:
    """
    Normalize DAv2 raw output (meters) to [-1,1] using the same global p1/p99
    as the dataloader's target normalization.
    """
    p1 = float(target_stats["p1"][0])
    p99 = float(target_stats["p99"][0])
    h_norm = (h_coarse - p1) / (p99 - p1 + 1e-8)   # [0, 1]
    h_norm = h_norm * 2.0 - 1.0                       # [-1, 1]
    return h_norm.clamp(-1.0, 1.0)


def _load_target_stats(stats_file: str, year: int) -> dict:
    with open(stats_file) as f:
        all_stats = json.load(f)
    return all_stats[str(year)]


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def save_checkpoint(fm_refiner: FMRefiner, optimizer, lr_scheduler, step, out_dir, name="latest"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save({
        "step": step,
        "use_controlnet": fm_refiner.use_controlnet,
        "unet_state": fm_refiner.unet.state_dict(),
        "controlnet_state": fm_refiner.controlnet.state_dict() if fm_refiner.use_controlnet else None,
        "null_ps_state": fm_refiner._null_ps,
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
    }, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(fm_refiner: FMRefiner, optimizer, lr_scheduler, path):
    ckpt = torch.load(path, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    if fm_refiner.use_controlnet and ckpt.get("controlnet_state") is not None:
        fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if lr_scheduler and ckpt.get("lr_scheduler_state"):
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
    return ckpt["step"]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="FM Refiner Training")
    parser.add_argument("--config", type=str, default="config/fm_refiner_v3-2.yaml")
    parser.add_argument("--resume_run", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--exit_after", type=int, default=-1,
                        help="Exit after X minutes.")
    parser.add_argument("--base_data_dir", type=str, default=None)
    parser.add_argument("--base_ckpt_dir", type=str, default=None)
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

    base_ckpt_dir = args.base_ckpt_dir or os.environ.get("BASE_CKPT_DIR", "")

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
            **cfg.wandb,
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
    val_dataset = _make_dataset(base_dataset, val_coords, 'val', 1)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=cfg_data.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=cfg_data.workers, pin_memory=True)

    # ---- Model ----
    n_landsat_bands = len(cfg_data.selected_bands)
    n_ps_bands = len(cfg_data.selected_bands_hr)

    use_controlnet = cfg.trainer.get("use_controlnet", True)
    controlnet_cond_mode = cfg.trainer.get("controlnet_cond_mode", "landsat_ps")
    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=cfg.trainer.get("ps_dropout_p", 0.0),
        use_controlnet=use_controlnet,
        controlnet_cond_mode=controlnet_cond_mode,
        device=str(device),
    )
    fm_refiner = fm_refiner.to(device)

    # ---- DAv2 (frozen) ----
    dav2_model = load_dav2(
        dav2_path=cfg.model.dav2_pretrained_path,
        backbone=cfg.model.dav2_backbone,
        out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
    )
    dav2_model = dav2_model.to(device)

    # Load target stats for coarse normalization (same p1/p99 as dataloader)
    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ---- Optimizer ----
    param_groups = [{"params": fm_refiner.unet.parameters(), "lr": cfg.optimizer.lr_unet}]
    if use_controlnet:
        param_groups.append({"params": fm_refiner.controlnet.parameters(), "lr": cfg.optimizer.lr_controlnet})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.optimizer.weight_decay)

    lr_scheduler = None
    if cfg.get("lr_scheduler") is not None:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_iter,
            eta_min=cfg.lr_scheduler.eta_min,
        )

    # ---- Resume ----
    start_step = 0
    if args.resume_run is not None:
        start_step = load_checkpoint(fm_refiner, optimizer, lr_scheduler, args.resume_run)
        logging.info(f"Resumed from step {start_step}")

    # ---- Training loop ----
    t_end = t_start + timedelta(minutes=args.exit_after) if args.exit_after > 0 else None

    step = start_step
    best_r2 = -1e8
    val_offset = 0                                          # rotating pointer into val_loader
    val_subset_size = max(1, len(val_loader) // 10)         # 10% of val data per validation

    fm_refiner.train()

    logging.info("Starting FM Refiner training")
    pbar = tqdm(total=cfg.max_iter, initial=step, desc="Training", dynamic_ncols=True)
    loss_val = 0.0
    for epoch in range(cfg.max_epoch):
        for batch in train_loader:
            if step >= cfg.max_iter:
                break
            if t_end is not None and datetime.now() >= t_end:
                logging.info("Exit after time limit reached.")
                save_checkpoint(fm_refiner, optimizer, lr_scheduler, step, out_dir_ckpt, "latest")
                pbar.close()
                sys.exit(0)

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

            # Landsat to HR resolution for ControlNet
            landsat = inputs_lr[:, :n_landsat_bands]
            landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

            # DAv2 coarse prediction (no grad, DAv2 frozen)
            # DAv2 expects [0,1] input; dataloader gives [-1,1] → convert back
            with torch.no_grad():
                landsat_for_dav2 = (landsat + 1.0) / 2.0
                h_coarse = run_dav2(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
                h_coarse = _normalize_coarse(h_coarse, target_stats)

            # FM loss
            loss = fm_refiner(
                landsat_lr=landsat_hr,  # ControlNet sees HR-resolution Landsat
                ps_hr=inputs_hr,
                h_coarse=h_coarse,
                h_gt=targets,
            )
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                all_params = list(fm_refiner.unet.parameters())
                if use_controlnet:
                    all_params += list(fm_refiner.controlnet.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

            step += 1
            pbar.update(1)

            # Logging
            if step % cfg.trainer.log_period == 0:
                loss_val = loss.item() * accumulation_steps
                lr_unet = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{loss_val:.4f}", r2=f"{best_r2:.4f}")
                log_dict = {"train/loss": loss_val, "lr/unet": lr_unet}
                if use_controlnet:
                    lr_cn = optimizer.param_groups[1]["lr"]
                    log_dict["lr/controlnet"] = lr_cn
                    logging.info(f"[step {step}] loss={loss_val:.5f} lr_unet={lr_unet:.2e} lr_cn={lr_cn:.2e}")
                else:
                    logging.info(f"[step {step}] loss={loss_val:.5f} lr_unet={lr_unet:.2e}")
                wandb.log(log_dict, step=step)
                tb_logger.log_dict({"train/loss": loss_val}, global_step=step)

            # Validation (rotating 10% subset)
            if step % cfg.trainer.validation_period == 0:
                metrics, fig = validate(
                    fm_refiner, dav2_model, val_loader, device,
                    n_steps=cfg.validation.n_steps,
                    target_stats=target_stats,
                    n_landsat_bands=n_landsat_bands,
                    val_offset=val_offset,
                    method=cfg.validation.get("method", "euler"),
                )
                val_offset = (val_offset + val_subset_size) % len(val_loader)

                logging.info(f"[step {step}] val: {metrics}")
                log_dict = {f"val/{k}": v for k, v in metrics.items()}
                if fig is not None:
                    log_dict["val/samples"] = wandb.Image(fig)
                    plt.close(fig)
                wandb.log(log_dict, step=step)
                tb_logger.log_dict({f"val/{k}": v for k, v in metrics.items()}, global_step=step)

                # Save best
                if metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    pbar.set_postfix(loss=f"{loss_val:.4f}", r2=f"{best_r2:.4f}")
                    save_checkpoint(fm_refiner, optimizer, lr_scheduler, step, out_dir_ckpt, "best")
                    logging.info(f"New best R²={best_r2:.4f} at step {step}")

            # Periodic checkpoint
            if step % cfg.trainer.save_period == 0:
                save_checkpoint(fm_refiner, optimizer, lr_scheduler, step, out_dir_ckpt, "latest")

        if step >= cfg.max_iter:
            break

    pbar.close()

    # Final full validation on entire val set
    # Final full validation – load best checkpoint first
    best_ckpt_path = os.path.join(out_dir_ckpt, "best.pth")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(fm_refiner, optimizer, lr_scheduler, best_ckpt_path)
        logging.info(f"Loaded best checkpoint from {best_ckpt_path} for final validation")
    else:
        logging.warning("Best checkpoint not found, using final model weights for full validation")

    logging.info("Running full validation on entire val set...")
    final_metrics, final_fig = validate(
        fm_refiner, dav2_model, val_loader, device,
        n_steps=cfg.validation.n_steps,
        target_stats=target_stats,
        n_landsat_bands=n_landsat_bands,
        full=True,
        method=cfg.validation.get("method", "euler"),
    )
    logging.info(f"Final val metrics: {final_metrics}")
    final_log = {f"val_final/{k}": v for k, v in final_metrics.items()}
    if final_fig is not None:
        final_log["val_final/samples"] = wandb.Image(final_fig)
        plt.close(final_fig)
    wandb.log(final_log, step=step)

    save_checkpoint(fm_refiner, optimizer, lr_scheduler, step, out_dir_ckpt, "final")
    logging.info(f"Training finished at step {step}. Best val R²={best_r2:.4f}")
