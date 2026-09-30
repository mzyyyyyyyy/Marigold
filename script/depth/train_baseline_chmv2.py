"""
Baseline training script: direct supervised fine-tuning (frozen DINOv3 backbone, DPT head + learnable upsample head) of CHMv2
(facebook/dinov3-vitl16-chmv2-dpt-head) on Landsat -> canopy height. Same as
train_baseline_dav2.py (dataset, patch split, Huber loss on raw metres) but
with CHMv2 replacing DepthAnythingV2Height, and multi-GPU / multi-node DDP in
the style of train_fm_refiner.py.

Pipeline:
  Landsat (3-band, p50) -> [0,1] -> mean/std norm -> CHMv2 -> H_pred (metres)
  loss = Huber(H_pred, H_gt)

Usage (single GPU):
  python script/depth/train_baseline_chmv2.py --config config/baseline_chmv2-lumi.yaml
Usage (multi-GPU / multi-node): see script/depth/train_baseline_chmv2.sh.
  Reads SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID (or RANK / WORLD_SIZE /
  LOCAL_RANK). The model is DDP-wrapped, data is sharded via
  DistributedSampler, and validation is sharded across ranks and gathered to
  rank0. Real batch per optimizer step =
  max_train_batch_size * accumulation_steps * world_size.
"""

import contextlib
import copy
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datetime import datetime, timedelta
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from depthfm.chmv2 import CHMv2Height
from src.util.config_util import recursive_load_config
from src.util.logging_util import config_logging, init_wandb, tb_logger
from src.util.ps_lazydataset import LazyPatchDataset


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
    col_titles = ["Landsat (RGB)", "Pred", "GT"]
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, s in enumerate(samples):
        for col, (key, title) in enumerate(zip(["landsat", "pred", "gt"], col_titles)):
            ax = axes[row, col]
            data = s[key]
            if key == "landsat":
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
# Validation
# -------------------------------------------------------------------------

def _raw(module: nn.Module) -> nn.Module:
    """Unwrap a DDP-wrapped module (shares parameters; keeps state_dict keys
    unprefixed whether or not distributed training is active)."""
    return module.module if isinstance(module, DDP) else module


@torch.no_grad()
def validate(
    model: nn.Module,
    val_dataset,
    device: torch.device,
    n_landsat_bands: int,
    target_vmax: float,
    val_offset: int = 0,
    n_vis_samples: int = 5,
    full: bool = False,
    global_rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
    num_workers: int = 0,
) -> tuple:
    """
    Sharded validation: every rank forwards a disjoint shard of a
    deterministic index window through the raw (non-DDP) model, and
    predictions are gathered to rank0, which computes the metrics. Non-rank0
    callers get (None, None). Subset mode uses a rotating 10% window starting
    at val_offset; full=True iterates everything.
    """
    raw_model = _raw(model)
    raw_model.eval()
    all_preds, all_gts = [], []
    vis_samples = []

    n_total = len(val_dataset)
    if full:
        indices = list(range(n_total))
    else:
        subset_size = max(1, n_total // 10)
        indices = [(val_offset + i) % n_total for i in range(subset_size)]
    local_indices = indices[global_rank::world_size] if is_distributed else indices
    local_loader = DataLoader(Subset(val_dataset, local_indices), batch_size=1,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    for batch in tqdm(local_loader, desc="Validation", leave=False, disable=(global_rank != 0)):
        inputs_lr, _inputs_hr, targets = batch
        if inputs_lr.dim() == 5:
            inputs_lr = inputs_lr.squeeze(0)
        if targets.dim() == 5:
            targets = targets.squeeze(0)

        inputs_lr = inputs_lr.to(device)
        targets = targets.to(device)

        landsat = inputs_lr[:, :n_landsat_bands]
        pred = raw_model((landsat + 1.0) / 2.0, target_size=tuple(targets.shape[-2:]))

        all_preds.append(pred.cpu().flatten())
        all_gts.append(targets.cpu().flatten())

        if len(vis_samples) < n_vis_samples:
            vis_samples.append({
                "landsat": _to_vis_rgb(landsat[0]),
                "pred": _to_vis_depth(pred[0]),
                "gt": _to_vis_depth(targets[0]),
            })

    pred_local = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0)
    gt_local = torch.cat(all_gts, dim=0) if all_gts else torch.empty(0)

    model.train()

    if is_distributed:
        # NCCL has no gather; all_gather_object is collective on every rank.
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (pred_local, gt_local, vis_samples))
        if global_rank != 0:
            return None, None
        pred_cat = torch.cat([g[0] for g in gathered if g[0].numel() > 0], dim=0)
        gt_cat = torch.cat([g[1] for g in gathered if g[1].numel() > 0], dim=0)
        vis_samples = []
        for g in gathered:
            vis_samples.extend(g[2])
            if len(vis_samples) >= n_vis_samples:
                break
        vis_samples = vis_samples[:n_vis_samples]
    else:
        pred_cat, gt_cat = pred_local, gt_local

    metrics = compute_metrics(pred_cat, gt_cat)
    fig = _make_vis_figure(vis_samples, target_vmax) if vis_samples else None
    return metrics, fig


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def save_checkpoint(model, optimizer, lr_scheduler, step, out_dir, name="latest"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save({
        "step": step,
        "model_state_dict": _raw(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
    }, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, lr_scheduler, path):
    ckpt = torch.load(path, map_location="cpu")
    _raw(model).load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if lr_scheduler and ckpt.get("lr_scheduler_state"):
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
    return ckpt["step"]


@contextlib.contextmanager
def _maybe_no_sync(module, skip_sync: bool):
    """Defer DDP's gradient all-reduce on non-final micro-batches of an
    accumulation window."""
    if skip_sync and isinstance(module, DDP):
        with module.no_sync():
            yield
    else:
        yield


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="CHMv2 Baseline Training")
    parser.add_argument("--config", type=str, default="config/baseline_chmv2-lumi.yaml")
    parser.add_argument("--resume_run", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--exit_after", type=int, default=-1, help="Exit after X minutes.")
    parser.add_argument("--add_datetime_prefix", action="store_true")
    args = parser.parse_args()

    # ---- Distributed setup (same SLURM env reading as train_fm_refiner.py) ----
    global_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_distributed = world_size > 1
    is_main_process = global_rank == 0

    # LazyPatchDataset.__init__ is chatty; keep only rank0's output.
    if not is_main_process:
        sys.stdout = open(os.devnull, "w")
        import functools
        import src.util.ps_lazydataset as _pld
        _pld.tqdm = functools.partial(_pld.tqdm, disable=True)

    if is_distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        backend = "nccl" if (torch.cuda.is_available() and not args.no_cuda) else "gloo"
        dist.init_process_group(
            backend=backend, rank=global_rank, world_size=world_size,
            timeout=timedelta(minutes=2),
        )

    # ---- Device ----
    if torch.cuda.is_available() and not args.no_cuda:
        n_visible = torch.cuda.device_count()
        device_index = local_rank % n_visible if n_visible > 0 else 0
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        gpu_name = torch.cuda.get_device_name(device_index)
    else:
        n_visible = 0
        device = torch.device("cpu")
        gpu_name = "cpu"
    print(
        f"[rank {global_rank}/{world_size}] SLURM_LOCALID={local_rank} "
        f"visible_gpus={n_visible} -> using {device} ({gpu_name})",
        file=sys.stderr, flush=True,
    )

    def _barrier():
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)

    if is_distributed:
        # Warm up rank-pair connections before DDP's construction broadcast
        # (LUMI Slingshot connection-setup race; see train_fm_refiner.sh).
        _barrier()

    # ---- Config ----
    if args.resume_run is not None:
        out_dir_run = os.path.dirname(os.path.dirname(args.resume_run))
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
        job_name = os.path.basename(out_dir_run)
    else:
        cfg = recursive_load_config(args.config)

        # Same step scaling as train_fm_refiner.py, so that a config with the
        # same max_iter/period values sees the same total data as
        # fm_refiner-R-lumi*.yaml regardless of GPU count: a step consumes
        # max_train_batch_size * accumulation_steps * world_size samples, and
        # the reference those configs were tuned against is real batch =
        # max_train_batch_size * 4 (fm_refiner-R.yaml's accumulation of 4).
        # Applied once on a fresh run; a resumed run reloads the scaled config.
        _REFERENCE_ACCUM_STEPS = 4
        _eff_bs = cfg.dataloader.get("effective_batch_size", cfg.dataloader.max_train_batch_size)
        _accum_steps = max(1, _eff_bs // cfg.dataloader.max_train_batch_size)
        total_scale = max(1, (_accum_steps * (world_size if is_distributed else 1)) // _REFERENCE_ACCUM_STEPS)
        if total_scale > 1:
            cfg.max_iter = max(1, cfg.max_iter // total_scale)
            cfg.trainer.validation_period = max(1, cfg.trainer.validation_period // total_scale)
            cfg.trainer.save_period = max(1, cfg.trainer.save_period // total_scale)
            cfg.trainer.log_period = max(1, cfg.trainer.log_period // total_scale)

        pure_job_name = os.path.basename(args.config).split(".")[0]
        job_name = (
            f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
            if args.add_datetime_prefix else pure_job_name
        )
        out_dir_run = os.path.join(args.output_dir or "./output", job_name)
        if is_main_process:
            os.makedirs(out_dir_run, exist_ok=False)

    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")

    # ---- Logging / wandb (rank0 only) ----
    if is_main_process:
        os.makedirs(out_dir_ckpt, exist_ok=True)
        config_logging(cfg.logging, out_dir=out_dir_run)
        if args.resume_run is None:
            with open(os.path.join(out_dir_run, "config.yaml"), "w") as f:
                OmegaConf.save(cfg, f)
            if total_scale > 1:
                logging.info(
                    f"[scale] (accumulation_steps={_accum_steps} x world_size="
                    f"{world_size if is_distributed else 1}) / reference_accum_steps="
                    f"{_REFERENCE_ACCUM_STEPS} = {total_scale}: max_iter/validation_period/"
                    f"save_period/log_period scaled to {cfg.max_iter}/{cfg.trainer.validation_period}/"
                    f"{cfg.trainer.save_period}/{cfg.trainer.log_period}."
                )

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
    else:
        logging.basicConfig(level=logging.INFO)

    if is_distributed:
        _barrier()

    # ---- Seeding ----
    base_seed = int(cfg.dataloader.seed)

    def _worker_init_fn(worker_id: int):
        # LazyPatchDataset uses python `random` / numpy for patch sampling,
        # which DataLoader does not reseed per worker.
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # ---- Data ----
    cfg_data = cfg.dataset
    eff_bs = cfg.dataloader.get("effective_batch_size", cfg.dataloader.max_train_batch_size)
    accumulation_steps = max(1, eff_bs // cfg.dataloader.max_train_batch_size)
    if is_main_process:
        logging.info(
            f"Batch size: {cfg.dataloader.max_train_batch_size} per micro-batch "
            f"x {accumulation_steps} accumulation step(s) x {world_size} rank(s) "
            f"= {cfg.dataloader.max_train_batch_size * accumulation_steps * world_size} "
            f"real batch per optimizer step."
        )

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

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank,
                           shuffle=True, seed=base_seed)
        if is_distributed else None
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=cfg_data.workers, pin_memory=True,
                              worker_init_fn=_worker_init_fn,
                              generator=torch.Generator().manual_seed(base_seed))
    # validate() builds its own on-demand DataLoader over a per-rank Subset.

    # ---- Model ----
    n_landsat_bands = len(cfg_data.selected_bands)

    # CHMv2's input mean/std come from the same stats file / year the
    # dataloader minmax-scales with (matches run_chmv2 in sr_fm_refiner).
    with open(cfg_data.input_stats_file) as f:
        input_stats = json.load(f)[str(cfg_data.year)]

    model = CHMv2Height(
        model_id=cfg.model.chmv2_model_id,
        mean=input_stats["mean"],
        std=input_stats["std"],
        pretrained=cfg.model.get("pretrained", True),
        freeze_backbone=cfg.model.get("freeze_backbone", True),
        upsample_input=cfg.model.get("upsample_input_to_target", False),
        out_in_scale_factor=cfg.model.get("out_in_scale_factor", 1),
    ).to(device)
    model.train()
    # DDP with find_unused_parameters=False raises if any trainable param gets
    # no gradient. Find such params with a dummy backward (the graph is static,
    # so this is identical on every rank) and freeze them: they cannot affect
    # the output anyway.
    unused = model.find_unused_trainable_params()
    for n, p in model.named_parameters():
        if n in unused:
            p.requires_grad_(False)
    if is_main_process:
        logging.info(f"Froze {len(unused)} trainable params that receive no gradient: {unused}")
    model_raw = model
    trainable_params = [p for p in model_raw.parameters() if p.requires_grad]
    if is_main_process:
        n_train = sum(p.numel() for p in trainable_params)
        n_all = sum(p.numel() for p in model_raw.parameters())
        logging.info(f"Trainable params: {n_train / 1e6:.1f}M / {n_all / 1e6:.1f}M "
                     f"(freeze_backbone={cfg.model.get('freeze_backbone', True)})")
    if is_distributed:
        ddp_ids = [device.index] if device.type == "cuda" else None
        model = DDP(model_raw, device_ids=ddp_ids,
                    output_device=ddp_ids[0] if ddp_ids else None,
                    find_unused_parameters=cfg.trainer.get("ddp_find_unused_parameters", False))

    # ---- Loss ----
    loss_fn = torch.nn.HuberLoss(delta=cfg.trainer.huber_delta)
    target_vmax = float(cfg.trainer.get("target_vis_vmax", 30.0))

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )

    lr_scheduler = None
    if cfg.get("lr_scheduler") is not None:
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.max_iter, eta_min=cfg.lr_scheduler.eta_min,
        )

    # ---- Resume ----
    start_step = 0
    if args.resume_run is not None:
        start_step = load_checkpoint(model, optimizer, lr_scheduler, args.resume_run)
        if is_main_process:
            logging.info(f"Resumed from step {start_step}")

    # ---- Training loop ----
    t_end = t_start + timedelta(minutes=args.exit_after) if args.exit_after > 0 else None

    step = start_step
    micro_step = 0  # position within the accumulation window
    best_r2 = -1e8
    val_offset = 0
    val_subset_size = max(1, len(val_dataset) // 10)

    if is_main_process:
        logging.info("Starting CHMv2 baseline training")
    pbar = tqdm(total=cfg.max_iter, initial=step, desc="Training", dynamic_ncols=True,
                disable=not is_main_process)
    loss_val = 0.0

    optimizer.zero_grad()
    for epoch in range(cfg.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= cfg.max_iter:
                break
            if t_end is not None and datetime.now() >= t_end:
                if is_main_process:
                    logging.info("Exit after time limit reached.")
                    save_checkpoint(model, optimizer, lr_scheduler, step, out_dir_ckpt, "latest")
                if is_distributed:
                    _barrier()
                    dist.destroy_process_group()
                pbar.close()
                sys.exit(0)

            inputs_lr, _inputs_hr, targets = batch
            if inputs_lr.dim() == 5:
                inputs_lr = inputs_lr.squeeze(0)
            if targets.dim() == 5:
                targets = targets.squeeze(0)

            inputs_lr = inputs_lr.to(device)
            targets = targets.to(device)

            landsat = inputs_lr[:, :n_landsat_bands]

            is_last_micro = (micro_step == accumulation_steps - 1)
            with _maybe_no_sync(model, is_distributed and not is_last_micro):
                pred = model((landsat + 1.0) / 2.0, target_size=tuple(targets.shape[-2:]))
                loss = loss_fn(pred, targets)
                (loss / accumulation_steps).backward()

            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
            else:
                micro_step += 1
                continue
            micro_step = 0

            step += 1
            pbar.update(1)

            if is_main_process and step % cfg.trainer.log_period == 0:
                loss_val = loss.item()
                lr_cur = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{loss_val:.4f}", r2=f"{best_r2:.4f}")
                logging.info(f"[step {step}] loss={loss_val:.5f} lr={lr_cur:.2e}")
                log_dict = {"train/loss": loss_val, "lr": lr_cur}
                wandb.log(log_dict, step=step)
                tb_logger.log_dict(log_dict, global_step=step)

            if step % cfg.trainer.validation_period == 0:
                metrics, fig = validate(
                    model, val_dataset, device,
                    n_landsat_bands=n_landsat_bands,
                    target_vmax=target_vmax,
                    val_offset=val_offset,
                    global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
                )
                val_offset = (val_offset + val_subset_size) % len(val_dataset)

                if is_main_process:
                    logging.info(f"[step {step}] val: {metrics}")
                    log_dict = {f"val/{k}": v for k, v in metrics.items()}
                    if fig is not None:
                        log_dict["val/samples"] = wandb.Image(fig)
                        plt.close(fig)
                    wandb.log(log_dict, step=step)
                    tb_logger.log_dict({f"val/{k}": v for k, v in metrics.items()}, global_step=step)

                    if metrics["r2"] > best_r2:
                        best_r2 = metrics["r2"]
                        pbar.set_postfix(loss=f"{loss_val:.4f}", r2=f"{best_r2:.4f}")
                        save_checkpoint(model, optimizer, lr_scheduler, step, out_dir_ckpt, "best")
                        logging.info(f"New best R²={best_r2:.4f} at step {step}")

                if is_distributed:
                    _barrier()

            if is_main_process and step % cfg.trainer.save_period == 0:
                save_checkpoint(model, optimizer, lr_scheduler, step, out_dir_ckpt, "latest")

        if step >= cfg.max_iter:
            break

    pbar.close()

    # Final full validation from the best checkpoint. Every rank loads it so
    # the sharded validate() combines consistent weights.
    if is_distributed:
        _barrier()
    best_ckpt_path = os.path.join(out_dir_ckpt, "best.pth")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(model, optimizer, lr_scheduler, best_ckpt_path)
        if is_main_process:
            logging.info(f"Loaded best checkpoint from {best_ckpt_path} for final validation")
    elif is_main_process:
        logging.warning("Best checkpoint not found, using final model weights for full validation")

    if is_main_process:
        logging.info("Running full validation on entire val set...")
    final_metrics, final_fig = validate(
        model, val_dataset, device,
        n_landsat_bands=n_landsat_bands,
        target_vmax=target_vmax,
        full=True,
        global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
        num_workers=cfg_data.workers,
    )
    if is_main_process:
        logging.info(f"Final val metrics: {final_metrics}")
        final_log = {f"val_final/{k}": v for k, v in final_metrics.items()}
        if final_fig is not None:
            final_log["val_final/samples"] = wandb.Image(final_fig)
            plt.close(final_fig)
        wandb.log(final_log, step=step)

        save_checkpoint(model, optimizer, lr_scheduler, step, out_dir_ckpt, "final")
        logging.info(f"Training finished at step {step}. Best val R²={best_r2:.4f}")

    if is_distributed:
        _barrier()
        dist.destroy_process_group()
