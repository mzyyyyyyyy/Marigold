"""
Training script for Flow Matching CHM Refiner.

Pipeline:
  Landsat → DAv2(frozen) → H_coarse (HR)
  [H_coarse, H_gt] → VAE → [z_coarse, z_gt]
  z_t = (1-t)*z_coarse + t*z_gt
  v_pred = UNet(z_t, t) + ControlNet(Landsat_HR, PS_or_null)
  loss = MSE(v_pred, z_gt - z_coarse)

Usage (single GPU):
  python script/depth/train_fm_refiner.py --config config/fm_refiner.yaml

Usage (multi-GPU, standard DDP, e.g. 8 GPUs via srun -n8):
  Reads SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID from the environment
  automatically (mirroring infer_fm_refiner.py), no extra flags needed — see
  script/depth/train_fm_refiner.sh. unet/controlnet are DDP-wrapped; data is
  sharded via DistributedSampler; validation runs sharded across all ranks
  (parallel inference) and is gathered to rank0, which alone handles
  logging/wandb/checkpointing.
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
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from depthfm.chmv2 import CHMv2Height, load_chmv2_baseline
from depthfm.flux_refiner import FluxRefiner, build_flux_refiner
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
    val_dataset,
    device: torch.device,
    n_steps: int,
    target_stats: dict,
    n_landsat_bands: int,
    val_offset: int = 0,           # rotating start index into val_dataset
    n_vis_samples: int = 5,        # how many samples to visualize
    full: bool = False,            # if True, iterate entire val_dataset
    method: str = "euler",         # integration method: "euler" or "heun"
    sampling_fn: str = "uniform",  # step-warp for velocity_parameterization="indi"
    ensemble_size: int = 1,        # average this many independent refine() draws per sample
    global_rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
    num_workers: int = 0,
) -> tuple:
    """
    Runs validation in parallel across all ranks: every rank forwards its own
    shard of a deterministic index window through its local raw (non-DDP)
    modules — no gradients, so no DDP participation is needed — and
    per-sample predictions are gathered to rank0, which computes the
    aggregate metrics. Non-rank0 callers get back (None, None).

    Subset mode: iterates a val_subset_size window starting from val_offset,
    wrapping around circularly.  full=True ignores offset and iterates all.
    """
    # Always run validation through the raw (non-DDP) modules: DDP's forward
    # hooks assume every rank calls it in lockstep for gradient sync, which
    # doesn't apply here (no backward pass, independent per-rank shards).
    is_flux = isinstance(fm_refiner, FluxRefiner)   # LoRA refiner: no DDP-wrapped modules
    if not is_flux:
        orig_unet, orig_controlnet = fm_refiner.unet, fm_refiner.controlnet
        fm_refiner.unet = _raw(orig_unet)
        if orig_controlnet is not None:
            fm_refiner.controlnet = _raw(orig_controlnet)

    fm_refiner.eval()
    all_preds, all_gts = [], []
    vis_samples = []

    # Every rank computes the identical (deterministic) index window, then
    # takes a disjoint shard of it — this keeps ranks in sync without
    # needing to broadcast anything, and without duplicating/missing
    # samples. Indices are resolved to actual data via a small on-demand
    # DataLoader over just this shard's Subset, so only the handful of
    # samples this rank actually needs get read off disk — not the full
    # val_dataset (which materializing a val_loader up front would force
    # on every rank, every validation call).
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
        h_coarse = _run_coarse(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
        h_coarse = _normalize_coarse(h_coarse, target_stats)

        # Each call is an independent stochastic draw when sample_sigma>0
        # (fresh SDE noise per call); averaging reduces sampling variance at
        # ensemble_size x the inference cost. A stack+mean of a single
        # element (ensemble_size=1, the default) is a no-op.
        h_fine_draws = [
            fm_refiner.refine(landsat_hr, h_coarse, n_steps=n_steps, method=method, sampling_fn=sampling_fn)
            for _ in range(ensemble_size)
        ]
        h_fine = torch.stack(h_fine_draws, dim=0).mean(dim=0)

        pred_m = _denormalize_target(h_fine, target_stats)
        gt_m = _denormalize_target(targets, target_stats)
        all_preds.append(pred_m.cpu().flatten())
        all_gts.append(gt_m.cpu().flatten())

        # Collect vis samples from the first image of this batch
        if len(vis_samples) < n_vis_samples:
            vis_samples.append({
                "landsat": _to_vis_rgb(landsat_hr[0]),
                "ps":      _to_vis_rgb(inputs_hr[0]),
                "coarse":  _to_vis_depth(h_coarse[0]),
                "fine":    _to_vis_depth(h_fine[0]),
                "gt":      _to_vis_depth(targets[0]),
            })

    pred_local = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0)
    gt_local   = torch.cat(all_gts, dim=0)   if all_gts   else torch.empty(0)

    fm_refiner.train()
    # Restore the (possibly DDP-wrapped) modules used for training.
    if not is_flux:
        fm_refiner.unet, fm_refiner.controlnet = orig_unet, orig_controlnet

    if is_distributed:
        # NCCL doesn't implement the point-to-point gather primitive, so use
        # all_gather_object (backed by all_gather, which NCCL does support)
        # even though only rank0 ends up using the result — every rank must
        # still pre-allocate the output list and call this collectively.
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (pred_local, gt_local, vis_samples))
        if global_rank != 0:
            return None, None
        pred_cat = torch.cat([g[0] for g in gathered if g[0].numel() > 0], dim=0)
        gt_cat   = torch.cat([g[1] for g in gathered if g[1].numel() > 0], dim=0)
        vis_samples = []
        for g in gathered:
            vis_samples.extend(g[2])
            if len(vis_samples) >= n_vis_samples:
                break
        vis_samples = vis_samples[:n_vis_samples]
    else:
        pred_cat, gt_cat = pred_local, gt_local

    metrics = compute_metrics(pred_cat, gt_cat)
    fig = _make_vis_figure(vis_samples) if vis_samples else None
    return metrics, fig


def _run_coarse(coarse_model, landsat_01: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """Coarse height (metres, (B,1,*target_size)) from Landsat in [0,1], from either
    the fine-tuned DAv2 (run_dav2) or the fine-tuned CHMv2 baseline (CHMv2Height)."""
    if isinstance(coarse_model, CHMv2Height):
        with torch.no_grad():
            return coarse_model(landsat_01, target_size=target_size)
    return run_dav2(coarse_model, landsat_01, target_size=target_size)


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


def _denormalize_target(h: torch.Tensor, target_stats: dict) -> torch.Tensor:
    """[-1, 1] -> metres, inverse of _normalize_coarse (same as infer_fm_refiner._denormalize)."""
    p1 = float(target_stats["p1"][0])
    p99 = float(target_stats["p99"][0])
    h = (h.float() + 1.0) / 2.0        # [0, 1]
    h = h * (p99 - p1) + p1             # metres
    return h


def _load_target_stats(stats_file: str, year: int) -> dict:
    with open(stats_file) as f:
        all_stats = json.load(f)
    return all_stats[str(year)]


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def _raw(module: nn.Module) -> nn.Module:
    """Unwrap a DDP-wrapped module to the underlying module (shares the same
    parameter tensors), so state_dict() keys stay unprefixed regardless of
    whether distributed training is active."""
    return module.module if isinstance(module, DDP) else module


def save_checkpoint(fm_refiner: FMRefiner, optimizer, lr_scheduler, step, out_dir, name="latest"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    if isinstance(fm_refiner, FluxRefiner):
        # LoRA-only checkpoint (the 12B base is never modified).
        torch.save({
            "step": step,
            "refiner_type": "flux",
            "lora_state": fm_refiner.lora_state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
        }, path)
        logging.info(f"Checkpoint saved to {path}")
        return
    torch.save({
        "step": step,
        "use_controlnet": fm_refiner.use_controlnet,
        "unet_state": _raw(fm_refiner.unet).state_dict(),
        "controlnet_state": _raw(fm_refiner.controlnet).state_dict() if fm_refiner.use_controlnet else None,
        "null_ps_state": fm_refiner._null_ps,
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
    }, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(fm_refiner: FMRefiner, optimizer, lr_scheduler, path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(fm_refiner, FluxRefiner):
        fm_refiner.load_lora_state_dict(ckpt["lora_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if lr_scheduler and ckpt.get("lr_scheduler_state"):
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        return ckpt["step"]
    _raw(fm_refiner.unet).load_state_dict(ckpt["unet_state"])
    if fm_refiner.use_controlnet and ckpt.get("controlnet_state") is not None:
        _raw(fm_refiner.controlnet).load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if lr_scheduler and ckpt.get("lr_scheduler_state"):
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
    return ckpt["step"]


@contextlib.contextmanager
def _maybe_no_sync(modules, skip_sync: bool):
    """Suppress DDP's gradient all-reduce on non-final micro-batches of a
    gradient-accumulation window.

    DDP triggers an all-reduce on every .backward() call by default, which
    would average+sync gradients after each individual micro-batch instead
    of once per real optimizer step — wasteful (accumulation_steps-1 extra
    rounds of communication per step) and, if zero_grad() isn't also called
    each time, redundant on top of the local accumulation. `.no_sync()`
    defers the all-reduce to the next backward() outside this context, so
    gradients accumulate locally across the window and get averaged across
    ranks exactly once, on the final micro-batch.
    """
    if not skip_sync:
        yield
        return
    with contextlib.ExitStack() as stack:
        for m in modules:
            stack.enter_context(m.no_sync())
        yield


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="FM Refiner Training")
    parser.add_argument("--config", type=str, default="config/fm_refiner-R.yaml")
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

    # ---- Distributed setup (mirrors infer_fm_refiner.py's SLURM env reading) ----
    # global_rank/world_size come from SLURM_PROCID/SLURM_NTASKS (srun sets one
    # task per GPU); local_rank (SLURM_LOCALID) selects the GPU on this node.
    global_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size  = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local_rank  = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_distributed  = world_size > 1
    is_main_process = global_rank == 0

    # Every rank builds its own full copy of LazyPatchDataset (needed for its
    # DataLoader), whose __init__ is chatty (print() + several tqdm bars over
    # every patch). Left unguarded, world_size copies of that output interleave
    # into one stderr/stdout stream. Since the dataset construction is
    # identical on every rank, only rank0's copy of the output is useful —
    # silence print() and disable tqdm rendering everywhere else.
    if not is_main_process:
        sys.stdout = open(os.devnull, "w")
        import functools
        import src.util.ps_lazydataset as _pld
        _pld.tqdm = functools.partial(_pld.tqdm, disable=True)

    if is_distributed:
        # MASTER_ADDR should be set by the launch script for multi-node runs
        # (e.g. the first node's hostname); defaults to localhost for a
        # single-node, multi-GPU srun/torchrun launch.
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
    # print (not logging.info — no handler is configured yet at this point,
    # and Python's logging silently drops INFO records without one) to
    # sys.stderr (not stdout — non-rank0 stdout is redirected to devnull
    # above) so every rank's GPU binding is visible for sanity-checking that
    # DDP is actually spread across distinct physical GPUs.
    print(
        f"[rank {global_rank}/{world_size}] SLURM_LOCALID={local_rank} "
        f"visible_gpus={n_visible} -> using {device} ({gpu_name})",
        file=sys.stderr, flush=True,
    )

    def _barrier():
        # Passing device_ids avoids NCCL's "devices used by this process are
        # currently unknown" warning on every barrier() call.
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)

    if is_distributed:
        # Warm up every rank-pair connection on a trivial call before DDP's
        # construction broadcast gets to it — see train_sr_fm_refiner.sh for
        # the LUMI Slingshot connection-setup race this mitigates.
        _barrier()

    # ---- Config ----
    if args.resume_run is not None:
        out_dir_run = os.path.dirname(os.path.dirname(args.resume_run))
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
        job_name = os.path.basename(out_dir_run)
    else:
        cfg = recursive_load_config(args.config)

        # A "step" consumes max_train_batch_size * accumulation_steps *
        # world_size samples: accumulation_steps from this config's own
        # local gradient accumulation (see dataloader.effective_batch_size),
        # times world_size again from DDP averaging gradients across ranks
        # on top of that. Both multiply total data-per-step independently
        # of each other, so both must be divided out — not just world_size
        # — to keep max_iter (and the other step-counted knobs below)
        # anchored to the same total sample count these numbers were
        # actually tuned against.
        #
        # That reference point is real batch = max_train_batch_size *
        # _REFERENCE_ACCUM_STEPS, NOT max_train_batch_size alone:
        # fm_refiner-R.yaml (this config's single-GPU basis) was tuned with
        # effective_batch_size=8 / max_train_batch_size=2, i.e.
        # accumulation_steps=4 already in effect. A *-lumi config typically
        # drops effective_batch_size back to max_train_batch_size (relying
        # on DDP's world_size multiplier instead of local accumulation), so
        # this config's own accumulation_steps no longer carries that
        # factor — it has to be reinstated explicitly here or max_iter
        # silently ends up scaled for 1/_REFERENCE_ACCUM_STEPS of the
        # intended total data. Applied once, here, on a fresh run — a
        # resumed run reloads the already-scaled config.yaml saved by the
        # run being resumed.
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

    base_ckpt_dir = args.base_ckpt_dir or os.environ.get("BASE_CKPT_DIR", "")

    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")

    # ---- Logging / wandb (rank0 only: avoids racing directory creation and
    # duplicate wandb runs/log files across ranks) ----
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
                    f"{_REFERENCE_ACCUM_STEPS} = {total_scale}: scaled "
                    f"max_iter/validation_period/save_period/log_period down by "
                    f"1/{total_scale} (now {cfg.max_iter}/{cfg.trainer.validation_period}/"
                    f"{cfg.trainer.save_period}/{cfg.trainer.log_period}) so this run "
                    f"covers the same total data (and wall-clock time) as a config "
                    f"training at real batch = max_train_batch_size * "
                    f"{_REFERENCE_ACCUM_STEPS}."
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

    # ---- Seeding (reproducibility) ----
    base_seed = int(cfg.dataloader.seed)

    def _worker_init_fn(worker_id: int):
        # DataLoader auto-reseeds torch's RNG per worker, but NOT Python's
        # `random` or numpy — LazyPatchDataset uses `random.randint`/`random.choice`
        # for patch sampling, so without this, worker processes (forked) can
        # share correlated `random` state across runs/machines.
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # ---- Data ----
    cfg_data = cfg.dataset
    # effective_batch_size is optional and defaults to max_train_batch_size
    # (accumulation_steps=1, i.e. no local accumulation): it only exists for
    # configs that need to simulate a larger batch on a single GPU (see
    # fm_refiner-R.yaml). A multi-GPU config doesn't need it — DDP's
    # cross-rank gradient averaging already provides that multiplier for
    # free, so the real per-step batch there is simply
    # max_train_batch_size * world_size, and max_train_batch_size is the
    # only batch-size knob such a config needs to set (see
    # fm_refiner-R-lumi.yaml).
    eff_bs = cfg.dataloader.get("effective_batch_size", cfg.dataloader.max_train_batch_size)
    # accumulation_steps micro-batches (each max_train_batch_size samples,
    # per rank) are accumulated into one real optimizer step, so the true
    # per-step batch a gradient update is computed over is
    # max_train_batch_size * accumulation_steps * world_size (DDP averages
    # gradients across ranks on top of this).
    accumulation_steps = max(1, eff_bs // cfg.dataloader.max_train_batch_size)
    if is_main_process:
        real_batch = cfg.dataloader.max_train_batch_size * accumulation_steps * world_size
        logging.info(
            f"Batch size: {cfg.dataloader.max_train_batch_size} per micro-batch "
            f"x {accumulation_steps} accumulation step(s) x {world_size} rank(s) "
            f"= {real_batch} real batch per optimizer step."
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
                           shuffle=True, seed=cfg.dataloader.seed)
        if is_distributed else None
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=cfg_data.workers, pin_memory=True,
                              worker_init_fn=_worker_init_fn,
                              generator=torch.Generator().manual_seed(base_seed))
    # validate() builds its own small on-demand DataLoader per call, over a
    # Subset of just the sample indices it actually needs (see validate()) —
    # so no val_loader is built here. Materializing a full DataLoader over
    # val_dataset up front is fine on 1 rank, but at high rank counts each
    # validation call would otherwise force every process to eagerly
    # read+transform the *entire* val set from shared storage before
    # subsetting down to the handful of samples actually used.

    # ---- Model ----
    n_landsat_bands = len(cfg_data.selected_bands)
    n_ps_bands = len(cfg_data.selected_bands_hr)

    use_controlnet = cfg.trainer.get("use_controlnet", True)
    controlnet_cond_mode = cfg.trainer.get("controlnet_cond_mode", "landsat_ps")
    velocity_parameterization = cfg.trainer.get("velocity_parameterization", "fixed")
    # refiner_type: "sd" (default: SD2.1 UNet + ControlNet, FMRefiner) or
    # "flux" (FLUX.1 transformer + LoRA + token-concat conditioning, FluxRefiner).
    is_flux = cfg.model.get("refiner_type", "sd") == "flux"
    if is_flux:
        if velocity_parameterization != "fixed":
            raise NotImplementedError("refiner_type='flux' supports only velocity_parameterization='fixed'.")
        use_controlnet = False   # conditioning is token concatenation, no ControlNet
        fm_refiner = build_flux_refiner(
            model_id=cfg.model.flux_model_id,
            text_cache_path=cfg.model.flux_text_cache,
            lora_rank=cfg.model.get("lora_rank", 64),
            lora_alpha=cfg.model.get("lora_alpha", 64),
            ps_dropout_p=cfg.trainer.get("ps_dropout_p", 0.0),
            noise_sigma=cfg.trainer.get("noise_sigma", 0.0),
            sample_sigma=cfg.trainer.get("sample_sigma", None),
            sample_noise_mode=cfg.trainer.get("sample_noise_mode", "sde"),
            guidance_scale=cfg.model.get("flux_guidance", 1.0),
            grad_checkpointing=cfg.model.get("grad_checkpointing", True),
        )
        fm_refiner = fm_refiner.to(device)
        if is_distributed:
            fm_refiner.broadcast_lora(0)   # identical LoRA init on every rank
        if is_main_process:
            n_lora = sum(p.numel() for p in fm_refiner.lora_parameters())
            logging.info(f"FLUX refiner: {fm_refiner.n_lora_layers} LoRA-wrapped linears, "
                         f"{n_lora / 1e6:.1f}M trainable LoRA params "
                         f"(rank {cfg.model.get('lora_rank', 64)}); base transformer frozen.")
    else:
        fm_refiner = build_fm_refiner(
            sd_pretrained_path=cfg.model.sd_pretrained_path,
            n_landsat_bands=n_landsat_bands,
            n_ps_bands=n_ps_bands,
            ps_dropout_p=cfg.trainer.get("ps_dropout_p", 0.0),
            use_controlnet=use_controlnet,
            controlnet_cond_mode=controlnet_cond_mode,
            velocity_parameterization=velocity_parameterization,
            noise_sigma=cfg.trainer.get("noise_sigma", 0.0),
            sample_sigma=cfg.trainer.get("sample_sigma", None),
            sample_noise_mode=cfg.trainer.get("sample_noise_mode", "sde"),
            device=str(device),
        )
        fm_refiner = fm_refiner.to(device)

    # ---- Coarse-height model (frozen): fine-tuned DAv2 (default) or CHMv2 baseline ----
    # (variable keeps its historical name dav2_model; it may hold either.)
    coarse_model_type = cfg.model.get("coarse_model", "dav2")
    if coarse_model_type == "chmv2":
        with open(cfg_data.input_stats_file) as f:
            _in_stats = json.load(f)[str(cfg_data.year)]
        dav2_model = load_chmv2_baseline(
            model_dir=cfg.model.chmv2_model_id,
            ckpt_path=cfg.model.chmv2_ckpt_path,
            mean=_in_stats["mean"], std=_in_stats["std"],
            out_in_scale_factor=cfg.model.chmv2_out_in_scale_factor,
        )
    else:
        dav2_model = load_dav2(
            dav2_path=cfg.model.dav2_pretrained_path,
            backbone=cfg.model.dav2_backbone,
            out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
        )
    dav2_model = dav2_model.to(device)

    # Load target stats for coarse normalization (same p1/p99 as dataloader)
    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ---- DDP wrapping ----
    # (FLUX/LoRA path: no DDP -- the frozen 12B base would be broadcast to all
    # ranks for nothing; only the small LoRA grads are all-reduced by hand.)
    unet_raw       = None if is_flux else fm_refiner.unet
    controlnet_raw = fm_refiner.controlnet if (use_controlnet and not is_flux) else None
    if is_distributed and not is_flux:
        ddp_ids = [device.index] if device.type == "cuda" else None
        fm_refiner.unet = DDP(unet_raw, device_ids=ddp_ids, output_device=ddp_ids[0] if ddp_ids else None)
        if use_controlnet:
            fm_refiner.controlnet = DDP(controlnet_raw, device_ids=ddp_ids, output_device=ddp_ids[0] if ddp_ids else None)

    # ---- Optimizer ----
    if is_flux:
        param_groups = [{"params": fm_refiner.lora_parameters(), "lr": cfg.optimizer.lr_lora}]
    else:
        param_groups = [{"params": unet_raw.parameters(), "lr": cfg.optimizer.lr_unet}]
    if use_controlnet and not is_flux:
        param_groups.append({"params": controlnet_raw.parameters(), "lr": cfg.optimizer.lr_controlnet})
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
        if is_main_process:
            logging.info(f"Resumed from step {start_step}")

    # ---- Training loop ----
    t_end = t_start + timedelta(minutes=args.exit_after) if args.exit_after > 0 else None

    step = start_step
    # Position within the current accumulation window (0..accumulation_steps-1)
    # — a real optimizer step only happens, and `step` only advances, when
    # this reaches accumulation_steps-1. Always starts a fresh window on
    # resume: a resumed run doesn't know which micro-batches the checkpointed
    # step already included, so starting mid-window would silently apply a
    # partial-batch gradient update.
    micro_step = 0
    best_r2 = -1e8
    val_offset = 0                                          # rotating pointer into val_dataset
    val_subset_size = max(1, len(val_dataset) // 10)        # 10% of val data per validation

    fm_refiner.train()

    if is_main_process:
        logging.info("Starting FM Refiner training")
    pbar = tqdm(total=cfg.max_iter, initial=step, desc="Training", dynamic_ncols=True,
                disable=not is_main_process)
    loss_val = 0.0
    for epoch in range(cfg.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= cfg.max_iter:
                break
            if t_end is not None and datetime.now() >= t_end:
                if is_main_process:
                    logging.info("Exit after time limit reached.")
                    save_checkpoint(fm_refiner, optimizer, lr_scheduler, step, out_dir_ckpt, "latest")
                if is_distributed:
                    _barrier()
                    dist.destroy_process_group()
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
                h_coarse = _run_coarse(dav2_model, landsat_for_dav2, target_size=(H_hr, W_hr))
                h_coarse = _normalize_coarse(h_coarse, target_stats)

            # Only the final micro-batch of a window should let DDP all-reduce
            # gradients; earlier ones accumulate locally (see _maybe_no_sync).
            is_first_micro = (micro_step == 0)
            is_last_micro  = (micro_step == accumulation_steps - 1)
            skip_sync = is_distributed and not is_last_micro

            if is_first_micro:
                optimizer.zero_grad()

            ddp_modules = [] if is_flux else [
                m for m in (fm_refiner.unet, fm_refiner.controlnet if use_controlnet else None)
                if isinstance(m, DDP)]
            with _maybe_no_sync(ddp_modules, skip_sync):
                # FM loss
                loss = fm_refiner(
                    landsat_lr=landsat_hr,  # ControlNet sees HR-resolution Landsat
                    ps_hr=inputs_hr,
                    h_coarse=h_coarse,
                    h_gt=targets,
                )
                (loss / accumulation_steps).backward()

            if is_last_micro:
                if is_flux:
                    if is_distributed:
                        fm_refiner.all_reduce_grads(world_size)
                    all_params = fm_refiner.lora_parameters()
                else:
                    all_params = list(unet_raw.parameters())
                    if use_controlnet:
                        all_params += list(controlnet_raw.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

            if not is_last_micro:
                micro_step += 1
                continue
            micro_step = 0

            step += 1
            pbar.update(1)

            # Logging (rank0 only)
            if is_main_process and step % cfg.trainer.log_period == 0:
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

            # Validation (rotating 10% subset). Every rank participates
            # (sharded, parallel inference — see validate()'s docstring);
            # only rank0 gets real metrics back and acts on them.
            if step % cfg.trainer.validation_period == 0:
                metrics, fig = validate(
                    fm_refiner, dav2_model, val_dataset, device,
                    n_steps=cfg.validation.n_steps,
                    target_stats=target_stats,
                    n_landsat_bands=n_landsat_bands,
                    val_offset=val_offset,
                    method=cfg.validation.get("method", "euler"),
                    sampling_fn=cfg.validation.get("sampling_fn", "uniform"),
                    ensemble_size=cfg.validation.get("ensemble_size", 1),
                    global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
                )
                # Deterministic given val_offset/val_subset_size/len(val_dataset),
                # which are identical on every rank — safe to update everywhere.
                val_offset = (val_offset + val_subset_size) % len(val_dataset)

                if is_main_process:
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

                fm_refiner.train()

                if is_distributed:
                    # rank0's checkpoint write above must finish before other
                    # ranks resume training and possibly overwrite "latest".
                    _barrier()

            # Periodic checkpoint (rank0 only)
            if is_main_process and step % cfg.trainer.save_period == 0:
                save_checkpoint(fm_refiner, optimizer, lr_scheduler, step, out_dir_ckpt, "latest")

        if step >= cfg.max_iter:
            break

    pbar.close()

    # Final full validation on entire val set — every rank must load the
    # same best checkpoint (not just rank0) since the sharded validate()
    # call below combines predictions across ranks — mixing a stale-weight
    # rank into that would corrupt the aggregate metric.
    best_ckpt_path = os.path.join(out_dir_ckpt, "best.pth")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(fm_refiner, optimizer, lr_scheduler, best_ckpt_path)
        if is_main_process:
            logging.info(f"Loaded best checkpoint from {best_ckpt_path} for final validation")
    elif is_main_process:
        logging.warning("Best checkpoint not found, using final model weights for full validation")

    if is_main_process:
        logging.info("Running full validation on entire val set...")
    final_metrics, final_fig = validate(
        fm_refiner, dav2_model, val_dataset, device,
        # Final full-val pass may use a heavier sampler than the periodic
        # subset validations (final_n_steps / final_ensemble_size, default =
        # the periodic values) -- see fm_refiner-R-lumi-6.yaml.
        n_steps=cfg.validation.get("final_n_steps", cfg.validation.n_steps),
        target_stats=target_stats,
        n_landsat_bands=n_landsat_bands,
        full=True,
        method=cfg.validation.get("method", "euler"),
        sampling_fn=cfg.validation.get("sampling_fn", "uniform"),
        ensemble_size=cfg.validation.get("final_ensemble_size", cfg.validation.get("ensemble_size", 1)),
        global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
        num_workers=cfg_data.workers,  # full=True shards the whole val set,
        # worth the worker-process cost here (unlike the periodic call above).
    )
    if is_main_process:
        logging.info(f"Final val metrics: {final_metrics}")
        final_log = {f"val_final/{k}": v for k, v in final_metrics.items()}
        if final_fig is not None:
            final_log["val_final/samples"] = wandb.Image(final_fig)
            plt.close(final_fig)
        wandb.log(final_log, step=step)

        save_checkpoint(fm_refiner, optimizer, lr_scheduler, step, out_dir_ckpt, "final")
        logging.info(f"Training finished at step {step}. Best val R²={best_r2:.4f}")

    if is_distributed:
        _barrier()
        dist.destroy_process_group()
