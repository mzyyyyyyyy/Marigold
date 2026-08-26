"""
Baseline training script: SR4IR-style joint SR + regression training
(no flow-matching diffusion refiner). Ports SR4IR/options/reg/base1_v1.yaml
(models/reg/sr4ir_reg_model.py) into the Marigold training stack so it runs
on the same dataset / patch split / input channels as the sr_fm_refiner and
baseline_dav2 experiments.

Pipeline:
  Phase 1 (update net_sr, net_reg frozen):
    LS → SwinIR → pseudo-PS (img_sr)
    loss = L1(img_sr, real_ps)                                    [pixel]
         + L1(reg_neck_feat(img_sr), reg_neck_feat(real_ps))      [TDP, after warmup]

  Phase 2 (update net_reg neck+head only; backbone frozen; net_sr frozen):
    pred = net_reg(img_sr.detach()).predicted_depth
    loss = Huber(pred, H_gt)

Usage:
  python script/depth/train_baseline_sr4ir_reg.py --config config/baseline_sr4ir_reg.yaml
"""

import copy
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation

from depthfm.sr_module import build_sr_module
from src.util.config_util import recursive_load_config
from src.util.logging_util import config_logging, init_wandb, tb_logger
from src.util.ps_lazydataset import LazyPatchDataset


# -------------------------------------------------------------------------
# Dataset helpers (identical to train_sr_fm_refiner.py)
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
# Visualisation
# -------------------------------------------------------------------------

def _to_vis_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor[:3].cpu().float()
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _to_vis_depth(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().cpu().float().numpy()


def _make_vis_figure(samples: list, vmax: float) -> plt.Figure:
    n = len(samples)
    col_titles = ["Landsat (RGB)", "pseudo-PS (SR out)", "Real PS", "Pred", "GT"]
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, s in enumerate(samples):
        for col, (key, title) in enumerate(zip(
            ["landsat", "img_sr", "img_ps", "pred", "gt"], col_titles
        )):
            ax = axes[row, col]
            data = s[key]
            if key in ("landsat", "img_sr", "img_ps"):
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap="plasma", vmin=0.0, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------------
# TDP feature hook
# -------------------------------------------------------------------------

def _neck_feat(net_reg, x):
    """
    net_reg.neck.fusion_stage returns a list of multi-scale fused feature
    maps (see transformers.models.depth_anything.DepthAnythingFeatureFusionStage),
    not a single tensor.
    """
    feats = {}
    hook = net_reg.neck.fusion_stage.register_forward_hook(
        lambda _m, _inp, out: feats.update({"out": out})
    )
    net_reg(x)
    hook.remove()
    return feats["out"]


def _tdp_loss(net_reg, img_sr_01, img_ps_01):
    feats_sr = _neck_feat(net_reg, img_sr_01)
    feats_ps = _neck_feat(net_reg, img_ps_01)
    l_tdp = 0.0
    for f_sr, f_ps in zip(feats_sr, feats_ps):
        l_tdp = l_tdp + F.l1_loss(f_sr, f_ps)
    return l_tdp


def _reg_predict(net_reg, x, target_size):
    out = net_reg(x).predicted_depth  # (B, H, W)
    pred = out.unsqueeze(1)  # (B, 1, H, W)
    if pred.shape[-2:] != tuple(target_size):
        pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
    return pred


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

@torch.no_grad()
def validate(
    net_sr, net_reg,
    val_loader: DataLoader,
    device: torch.device,
    n_landsat_bands: int,
    target_vmax: float,
    val_offset: int = 0,
    n_vis_samples: int = 5,
    full: bool = False,
) -> tuple:
    net_sr.eval()
    net_reg.eval()
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

        img_sr = net_sr(landsat, target_size=(H_hr, W_hr))
        img_sr_01 = (img_sr + 1.0) / 2.0
        pred = _reg_predict(net_reg, img_sr_01, (H_hr, W_hr))

        all_preds.append(pred.cpu().flatten())
        all_gts.append(targets.cpu().flatten())

        if len(vis_samples) < n_vis_samples:
            landsat_vis = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)
            vis_samples.append({
                "landsat": _to_vis_rgb(landsat_vis[0]),
                "img_sr": _to_vis_rgb(img_sr[0]),
                "img_ps": _to_vis_rgb(inputs_hr[0]),
                "pred": _to_vis_depth(pred[0]),
                "gt": _to_vis_depth(targets[0]),
            })

    pred_cat = torch.cat(all_preds, dim=0)
    gt_cat = torch.cat(all_gts, dim=0)
    metrics = compute_metrics(pred_cat, gt_cat)
    fig = _make_vis_figure(vis_samples, target_vmax) if vis_samples else None

    net_sr.train()
    net_reg.train()
    return metrics, fig


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                     lr_scheduler_sr, lr_scheduler_reg, step, out_dir, name="latest"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save({
        "step": step,
        "sr_state": net_sr.state_dict(),
        "reg_state": net_reg.state_dict(),
        "optimizer_sr_state": optimizer_sr.state_dict(),
        "optimizer_reg_state": optimizer_reg.state_dict(),
        "lr_scheduler_sr_state": lr_scheduler_sr.state_dict() if lr_scheduler_sr else None,
        "lr_scheduler_reg_state": lr_scheduler_reg.state_dict() if lr_scheduler_reg else None,
    }, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                     lr_scheduler_sr, lr_scheduler_reg, path):
    ckpt = torch.load(path, map_location="cpu")
    net_sr.load_state_dict(ckpt["sr_state"])
    net_reg.load_state_dict(ckpt["reg_state"])
    optimizer_sr.load_state_dict(ckpt["optimizer_sr_state"])
    optimizer_reg.load_state_dict(ckpt["optimizer_reg_state"])
    if lr_scheduler_sr and ckpt.get("lr_scheduler_sr_state"):
        lr_scheduler_sr.load_state_dict(ckpt["lr_scheduler_sr_state"])
    if lr_scheduler_reg and ckpt.get("lr_scheduler_reg_state"):
        lr_scheduler_reg.load_state_dict(ckpt["lr_scheduler_reg_state"])
    return ckpt["step"]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="SR4IR-style SR+Reg Baseline Training")
    parser.add_argument("--config", type=str, default="config/baseline_sr4ir_reg.yaml")
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

    n_landsat_bands = len(cfg_data.selected_bands)

    # ---- net_sr (SwinIR) ----
    net_sr = build_sr_module(OmegaConf.to_container(cfg.sr_module, resolve=True)).to(device)
    sr_pretrained_path = cfg.model.get("sr_pretrained_path", None)
    if sr_pretrained_path and os.path.exists(sr_pretrained_path):
        ckpt_sr = torch.load(sr_pretrained_path, map_location="cpu")
        key = cfg.model.get("sr_pretrained_key", None)
        state = ckpt_sr[key] if key and key in ckpt_sr else ckpt_sr
        missing, unexpected = net_sr.swinir.load_state_dict(state, strict=False)
        logging.info(f"SwinIR pretrained loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        logging.info("No SwinIR pretrained weights found; training from scratch.")
    net_sr.train()

    # ---- net_reg (DAv2 HF model, backbone frozen, neck+head trainable) ----
    net_reg = AutoModelForDepthEstimation.from_pretrained(cfg.model.dav2_backbone).to(device)
    for p in net_reg.backbone.parameters():
        p.requires_grad = False
    net_reg.train()

    # ---- Losses ----
    pixel_weight = cfg.sr_loss.pixel_weight
    tdp_weight = cfg.sr_loss.tdp_weight
    warmup_sr_steps = cfg.trainer.get("warmup_sr_steps", 0)
    huber_delta = cfg.trainer.huber_delta
    loss_huber = torch.nn.HuberLoss(delta=huber_delta)
    target_vmax = float(cfg.trainer.get("target_vis_vmax", 30.0))

    # ---- Optimizers ----
    optimizer_sr = torch.optim.AdamW(
        net_sr.parameters(), lr=cfg.optimizer.lr_sr, weight_decay=cfg.optimizer.weight_decay_sr
    )
    optimizer_reg = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_reg.parameters()),
        lr=cfg.optimizer.lr_reg, weight_decay=cfg.optimizer.weight_decay_reg,
    )

    lr_scheduler_sr, lr_scheduler_reg = None, None
    if cfg.get("lr_scheduler") is not None:
        lr_scheduler_sr = CosineAnnealingLR(optimizer_sr, T_max=cfg.max_iter, eta_min=cfg.lr_scheduler.eta_min_sr)
        lr_scheduler_reg = CosineAnnealingLR(optimizer_reg, T_max=cfg.max_iter, eta_min=cfg.lr_scheduler.eta_min_reg)

    # ---- Resume ----
    start_step = 0
    if args.resume_run is not None:
        start_step = load_checkpoint(
            net_sr, net_reg, optimizer_sr, optimizer_reg,
            lr_scheduler_sr, lr_scheduler_reg, args.resume_run,
        )
        logging.info(f"Resumed from step {start_step}")

    # ---- Training loop ----
    t_end = t_start + timedelta(minutes=args.exit_after) if args.exit_after > 0 else None

    step = start_step
    best_r2 = -1e8
    val_offset = 0
    val_subset_size = max(1, len(val_loader) // 10)

    loss_pix_accum = 0.0
    loss_tdp_accum = 0.0
    loss_reg_accum = 0.0
    micro_step = 0

    logging.info("Starting SR4IR-style baseline training (SR + regression, no diffusion refiner).")
    pbar = tqdm(total=cfg.max_iter, initial=step, desc="Training", dynamic_ncols=True)

    optimizer_sr.zero_grad()
    optimizer_reg.zero_grad()
    for epoch in range(cfg.max_epoch):
        for batch in train_loader:
            if step >= cfg.max_iter:
                break
            if t_end is not None and datetime.now() >= t_end:
                logging.info("Exit after time limit reached.")
                save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                                lr_scheduler_sr, lr_scheduler_reg, step, out_dir_ckpt, "latest")
                pbar.close()
                sys.exit(0)

            inputs_lr, inputs_hr, targets = batch
            if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
            if inputs_hr.dim() == 5: inputs_hr = inputs_hr.squeeze(0)
            if targets.dim()   == 5: targets   = targets.squeeze(0)

            inputs_lr = inputs_lr.to(device)
            inputs_hr = inputs_hr.to(device)
            targets   = targets.to(device)

            H_hr, W_hr = targets.shape[2], targets.shape[3]
            landsat = inputs_lr[:, :n_landsat_bands]

            # ============================================================
            # Phase 1: update net_sr (net_reg frozen via no_grad on its use)
            # ============================================================
            use_tdp = step >= warmup_sr_steps
            for p in net_reg.parameters():
                p.requires_grad = False

            img_sr = net_sr(landsat, target_size=(H_hr, W_hr))
            l_pix = F.l1_loss(img_sr, inputs_hr)
            l_total_sr = pixel_weight * l_pix

            if use_tdp:
                img_sr_01 = (img_sr + 1.0) / 2.0
                img_ps_01 = (inputs_hr + 1.0) / 2.0
                l_tdp = _tdp_loss(net_reg, img_sr_01, img_ps_01)
                l_total_sr = l_total_sr + tdp_weight * l_tdp
                loss_tdp_accum += l_tdp.item()

            if micro_step % accumulation_steps == 0:
                optimizer_sr.zero_grad()
            (l_total_sr / accumulation_steps).backward()

            is_accum_boundary = (micro_step + 1) % accumulation_steps == 0
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(net_sr.parameters(), max_norm=1.0)
                optimizer_sr.step()

            loss_pix_accum += l_pix.item()

            # restore backbone freeze (train() calls above don't change requires_grad)
            for p in net_reg.backbone.parameters():
                p.requires_grad = False
            for n, p in net_reg.named_parameters():
                if not n.startswith("backbone."):
                    p.requires_grad = True

            # ============================================================
            # Phase 2: update net_reg neck+head only (net_sr output detached)
            # ============================================================
            with torch.no_grad():
                img_sr_detached = net_sr(landsat, target_size=(H_hr, W_hr))
            img_sr_detached_01 = (img_sr_detached + 1.0) / 2.0

            pred = _reg_predict(net_reg, img_sr_detached_01, (H_hr, W_hr))
            l_reg = loss_huber(pred, targets)

            if micro_step % accumulation_steps == 0:
                optimizer_reg.zero_grad()
            (l_reg / accumulation_steps).backward()

            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in net_reg.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer_reg.step()

            loss_reg_accum += l_reg.item()
            micro_step += 1

            if not is_accum_boundary:
                continue

            if lr_scheduler_sr is not None:
                lr_scheduler_sr.step()
            if lr_scheduler_reg is not None:
                lr_scheduler_reg.step()

            step += 1
            pbar.update(1)

            # ---- Logging ----
            if step % cfg.trainer.log_period == 0:
                log_n = cfg.trainer.log_period
                lr_sr = optimizer_sr.param_groups[0]["lr"]
                lr_reg = optimizer_reg.param_groups[0]["lr"]
                log_dict = {
                    "lr/sr": lr_sr, "lr/reg": lr_reg,
                    "train/l_pix": loss_pix_accum / log_n,
                    "train/l_tdp": loss_tdp_accum / log_n,
                    "train/l_reg": loss_reg_accum / log_n,
                }
                logging.info(
                    f"[step {step}] "
                    f"l_pix={log_dict['train/l_pix']:.5f} "
                    f"l_tdp={log_dict['train/l_tdp']:.5f} "
                    f"l_reg={log_dict['train/l_reg']:.5f}"
                )
                pbar.set_postfix(r2=f"{best_r2:.4f}")
                wandb.log(log_dict, step=step)
                tb_logger.log_dict(log_dict, global_step=step)
                loss_pix_accum = loss_tdp_accum = loss_reg_accum = 0.0

            # ---- Validation ----
            if step % cfg.trainer.validation_period == 0:
                metrics, fig = validate(
                    net_sr, net_reg, val_loader, device,
                    n_landsat_bands=n_landsat_bands,
                    target_vmax=target_vmax,
                    val_offset=val_offset,
                )
                val_offset = (val_offset + val_subset_size) % len(val_loader)

                logging.info(f"[step {step}] val: {metrics}")
                log_dict = {f"val/{k}": v for k, v in metrics.items()}
                if fig is not None:
                    log_dict["val/samples"] = wandb.Image(fig)
                    plt.close(fig)
                wandb.log(log_dict, step=step)
                tb_logger.log_dict({f"val/{k}": v for k, v in metrics.items()}, global_step=step)

                if metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    pbar.set_postfix(r2=f"{best_r2:.4f}")
                    save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                                    lr_scheduler_sr, lr_scheduler_reg, step, out_dir_ckpt, "best")
                    logging.info(f"New best R²={best_r2:.4f} at step {step}")

            # ---- Periodic checkpoint ----
            if step % cfg.trainer.save_period == 0:
                save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                                lr_scheduler_sr, lr_scheduler_reg, step, out_dir_ckpt, "latest")

        if step >= cfg.max_iter:
            break

    pbar.close()

    # Final full validation on entire val set, from the best checkpoint
    best_ckpt_path = os.path.join(out_dir_ckpt, "best.pth")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                        lr_scheduler_sr, lr_scheduler_reg, best_ckpt_path)
        logging.info(f"Loaded best checkpoint from {best_ckpt_path} for final validation")
    else:
        logging.warning("Best checkpoint not found, using final model weights for full validation")

    logging.info("Running full validation on entire val set...")
    final_metrics, final_fig = validate(
        net_sr, net_reg, val_loader, device,
        n_landsat_bands=n_landsat_bands,
        target_vmax=target_vmax,
        full=True,
    )
    logging.info(f"Final val metrics: {final_metrics}")
    final_log = {f"val_final/{k}": v for k, v in final_metrics.items()}
    if final_fig is not None:
        final_log["val_final/samples"] = wandb.Image(final_fig)
        plt.close(final_fig)
    wandb.log(final_log, step=step)

    save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                    lr_scheduler_sr, lr_scheduler_reg, step, out_dir_ckpt, "final")
    logging.info(f"Training finished at step {step}. Best val R²={best_r2:.4f}")
