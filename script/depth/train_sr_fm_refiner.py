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

Usage (single GPU):
  python script/depth/train_sr_fm_refiner.py --config config/sr_fm_refiner_v0.yaml

Usage (multi-GPU, standard DDP, e.g. 8 GPUs via srun -n8):
  Reads SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID from the environment
  automatically (mirroring infer_sr_fm_refiner.py), no extra flags needed —
  see script/depth/train_sr_fm_refiner.sh. unet/controlnet/sr_module are
  DDP-wrapped; data is sharded via DistributedSampler; validation runs
  sharded across all ranks (parallel inference) and is gathered to rank0,
  which alone handles logging/wandb/checkpointing.
"""

import contextlib
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
from depthfm.chmv2 import load_chmv2, run_chmv2
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
    col_titles = ["Landsat (RGB)", "pseudo-PS (RGB)", "Real PS (RGB)", "DAv2 Coarse", "FM Refined", "GT"]
    fig, axes = plt.subplots(n, 6, figsize=(24, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, s in enumerate(samples):
        for col, (key, title) in enumerate(zip(
            ["landsat", "pseudo_ps", "real_ps", "coarse", "fine", "gt"], col_titles
        )):
            ax = axes[row, col]
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
    coarse_model,
    val_dataset,
    device: torch.device,
    n_steps: int,
    target_stats: dict,
    n_landsat_bands: int,
    val_offset: int = 0,
    n_vis_samples: int = 5,
    full: bool = False,
    method: str = "euler",
    use_sr_pseudo_ps: bool = True,
    use_real_ps: bool = False,
    coarse_model_type: str = "dav2",
    chmv2_mean: list = None,
    chmv2_std: list = None,
    n_avg: int = 1,
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
    """
    # Always run validation through the raw (non-DDP) modules: DDP's forward
    # hooks assume every rank calls it in lockstep for gradient sync, which
    # doesn't apply here (no backward pass, independent per-rank shards).
    orig_unet, orig_controlnet = fm_refiner.unet, fm_refiner.controlnet
    fm_refiner.unet, fm_refiner.controlnet = _raw(orig_unet), _raw(orig_controlnet)
    sr_module = _raw(sr_module)

    fm_refiner.eval()
    sr_module.eval()
    all_preds, all_gts = [], []
    vis_samples = []

    # Every rank computes the identical (deterministic) index window, then
    # takes a disjoint shard of it — this keeps ranks in sync without
    # needing to broadcast anything, and without duplicating/missing
    # samples. Indices are resolved to actual data via a small on-demand
    # DataLoader over just this shard's Subset, so only the handful of
    # samples this rank actually needs get read off disk — not the full
    # val_dataset (which materializing a val_loader up front would force
    # on every one of the 64 ranks, every validation call).
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

        if coarse_model_type == "chmv2":
            h_coarse = run_chmv2(coarse_model, landsat, (H_hr, W_hr), chmv2_mean, chmv2_std)
        else:
            landsat_for_dav2 = (landsat + 1.0) / 2.0
            h_coarse = run_dav2(coarse_model, landsat_for_dav2, target_size=(H_hr, W_hr))
        h_coarse = _normalize_coarse(h_coarse, target_stats)

        if use_real_ps:
            ps_input = inputs_hr
            ps_vis = inputs_hr[0]
        elif use_sr_pseudo_ps:
            ps_input = sr_module(landsat, target_size=(H_hr, W_hr))
            ps_vis = ps_input[0]
        else:
            ps_input = None
            ps_vis = landsat_hr[0]

        def _run_refine():
            if ps_input is not None:
                return fm_refiner.refine_with_ps(ps_input, h_coarse, n_steps=n_steps, method=method,
                                                  landsat_lr=landsat_hr)
            return fm_refiner.refine(h_coarse, n_steps=n_steps, method=method, landsat_lr=landsat_hr)

        if n_avg > 1:
            h_fine = torch.stack([_run_refine() for _ in range(n_avg)]).mean(dim=0)
        else:
            h_fine = _run_refine()

        all_preds.append(h_fine.cpu().flatten())
        all_gts.append(targets.cpu().flatten())

        if len(vis_samples) < n_vis_samples:
            pseudo_ps_vis = ps_vis
            vis_samples.append({
                "landsat":   _to_vis_rgb(landsat_hr[0]),
                "pseudo_ps": _to_vis_rgb(pseudo_ps_vis),
                "real_ps":   _to_vis_rgb(inputs_hr[0]),
                "coarse":    _to_vis_depth(h_coarse[0]),
                "fine":      _to_vis_depth(h_fine[0]),
                "gt":        _to_vis_depth(targets[0]),
            })

    pred_local = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0)
    gt_local   = torch.cat(all_gts, dim=0)   if all_gts   else torch.empty(0)

    fm_refiner.train()
    sr_module.train()
    # Restore the (possibly DDP-wrapped) modules used for training.
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


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def save_checkpoint(
    fm_refiner, sr_module, optimizer_fm, optimizer_sr,
    lr_scheduler_fm, lr_scheduler_sr,
    step, out_dir, name="latest"
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save({
        "step": step,
        "unet_state": _raw(fm_refiner.unet).state_dict(),
        "controlnet_state": _raw(fm_refiner.controlnet).state_dict(),
        "null_ps_state": fm_refiner._null_ps,
        "sr_state": _raw(sr_module).state_dict(),
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
    _raw(fm_refiner.unet).load_state_dict(ckpt["unet_state"])
    _raw(fm_refiner.controlnet).load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    _raw(sr_module).load_state_dict(ckpt["sr_state"])
    optimizer_fm.load_state_dict(ckpt["optimizer_fm_state"])
    optimizer_sr.load_state_dict(ckpt["optimizer_sr_state"])
    if lr_scheduler_fm and ckpt.get("lr_scheduler_fm_state"):
        lr_scheduler_fm.load_state_dict(ckpt["lr_scheduler_fm_state"])
    if lr_scheduler_sr and ckpt.get("lr_scheduler_sr_state"):
        lr_scheduler_sr.load_state_dict(ckpt["lr_scheduler_sr_state"])
    return ckpt["step"]


# -------------------------------------------------------------------------
# Phase helpers
# -------------------------------------------------------------------------

def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def _raw(module: nn.Module) -> nn.Module:
    """Unwrap a DDP-wrapped module to the underlying module (shares the same
    parameter tensors), so state_dict() keys stay unprefixed regardless of
    whether distributed training is active."""
    return module.module if isinstance(module, DDP) else module


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
    tdp_hook: str,               # e.g. "controlnet.cond_embedding" / "controlnet.mid_block" / "unet.up_blocks.2"
    device: torch.device,
    concat_landsat_cond: bool = False,
) -> dict:
    """
    Compute SR Phase-1 loss and return a dict with loss tensors.

    Gradient path (pixel loss):
      pseudo_ps ← SR ← landsat

    Gradient path (TDP loss):
      feature at tdp_hook(pseudo_ps) ← pseudo_ps ← SR ← landsat

    tdp_hook options:
      "controlnet.cond_embedding" – directly call controlnet_cond_embedding (no z_t context,
                                    cleanest option: features depend only on PS input)
      "controlnet.mid_block"      – hook ControlNet mid_block (requires full CN forward with z_coarse)
      "unet.up_blocks.N"          – hook UNet up_blocks[N], runs ControlNet + UNet forward

    Note: always operates on the raw (non-DDP-wrapped) unet/controlnet, since
    these are frozen during the SR step (no gradient sync required) — this
    also lets the SR step run every iteration without tripping DDP's
    freeze/unfreeze bookkeeping (see _raw()).
    """
    B = landsat.shape[0]
    H_hr, W_hr = real_ps.shape[2], real_ps.shape[3]
    controlnet = _raw(fm_refiner.controlnet)
    unet = _raw(fm_refiner.unet)

    # SR forward (trainable)
    pseudo_ps = sr_module(landsat, target_size=(H_hr, W_hr))  # (B, C_ps, H_hr, W_hr)

    # Pixel loss: L1(pseudo_ps, real_ps)
    l_pix = F.l1_loss(pseudo_ps, real_ps.detach()) * pixel_weight

    l_tdp = torch.tensor(0.0, device=device)
    if use_tdp and tdp_weight > 0:
        if concat_landsat_cond:
            cond_fake = torch.cat([landsat_hr, pseudo_ps], dim=1)
            cond_real = torch.cat([landsat_hr, real_ps], dim=1)
        else:
            cond_fake = pseudo_ps
            cond_real = real_ps

        if tdp_hook == "controlnet.cond_embedding":
            # Directly call controlnet_cond_embedding — no z_t context, no full CN forward.
            # Grad flows: l_tdp → cond_embedding(pseudo_ps) → pseudo_ps → SR
            feat_fake = controlnet.controlnet_cond_embedding(cond_fake)
            with torch.no_grad():
                feat_real = controlnet.controlnet_cond_embedding(cond_real)
        else:
            # Original hook-based path (controlnet.mid_block or unet.up_blocks.N)
            with torch.no_grad():
                z_coarse = fm_refiner.encode(h_coarse)

            text_emb = fm_refiner.empty_text_embed.to(device).expand(B, -1, -1)
            t_val = torch.full((B,), 0.5, device=device)
            t_int = (t_val * 999).long()

            use_unet = tdp_hook.startswith("unet.")
            if use_unet:
                up_idx = int(tdp_hook.split(".")[-1])
                hook_module = unet.up_blocks[up_idx]
            else:
                hook_module = controlnet.mid_block

            feats: dict = {}

            def _hook(m, _inp, out):
                feats["out"] = out

            hook = hook_module.register_forward_hook(_hook)

            cn_fake_out = controlnet(
                sample=z_coarse.detach(),
                timestep=t_int,
                encoder_hidden_states=text_emb,
                controlnet_cond=cond_fake,
                return_dict=True,
            )
            if use_unet:
                unet(
                    sample=z_coarse.detach(),
                    timestep=t_int,
                    encoder_hidden_states=text_emb,
                    down_block_additional_residuals=cn_fake_out.down_block_res_samples,
                    mid_block_additional_residual=cn_fake_out.mid_block_res_sample,
                    return_dict=True,
                )
            feat_fake = feats["out"]

            with torch.no_grad():
                cn_real_out = controlnet(
                    sample=z_coarse.detach(),
                    timestep=t_int,
                    encoder_hidden_states=text_emb,
                    controlnet_cond=cond_real,
                    return_dict=True,
                )
                if use_unet:
                    unet(
                        sample=z_coarse.detach(),
                        timestep=t_int,
                        encoder_hidden_states=text_emb,
                        down_block_additional_residuals=cn_real_out.down_block_res_samples,
                        mid_block_additional_residual=cn_real_out.mid_block_res_sample,
                        return_dict=True,
                    )
                feat_real = feats["out"]

            hook.remove()

        l_tdp = F.l1_loss(feat_fake, feat_real.detach()) * tdp_weight

    return {"l_pix": l_pix, "l_tdp": l_tdp, "l_total": l_pix + l_tdp,
            "pseudo_ps": pseudo_ps}


# -------------------------------------------------------------------------
# Phase-2 training step
# -------------------------------------------------------------------------

def _task_step(
    fm_refiner: FMRefiner,
    real_ps: torch.Tensor,       # (B, C_ps, H_hr, W_hr)
    h_coarse: torch.Tensor,
    h_gt: torch.Tensor,
    pseudo_ps_detached: torch.Tensor,  # (B, C_ps, H_hr, W_hr) — SR output already detached
    p_null_drop: float,
    p_pseudo_ps: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute FM training loss (flow matching MSE on velocities).

    PS source is sampled per-sample from three exclusive outcomes:
      p_null_drop  → null PS token
      p_pseudo_ps  → SR pseudo-PS (passed in detached, no second SR forward)
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

    # Pseudo-PS — reuse detached output from SR step (no extra forward pass)
    if pseudo_mask.any():
        for i in range(B):
            if pseudo_mask[i]:
                ps_cond[i] = pseudo_ps_detached[i]

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
    parser.add_argument("--config", type=str, default="config/sr_fm_refiner_v10.yaml")
    parser.add_argument("--resume_run", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--exit_after", type=int, default=-1, help="Exit after X minutes.")
    parser.add_argument("--add_datetime_prefix", action="store_true")
    args = parser.parse_args()

    # ---- Distributed setup (mirrors infer_sr_fm_refiner.py's SLURM env reading) ----
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
            timeout=timedelta(minutes=5),
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
        # construction broadcast (the first real bulk transfer, ~300MB x
        # ~15 back-to-back) gets to it. At 64 ranks that broadcast has
        # intermittently lost its completion signal on one rank (LUMI
        # Slingshot fabric, not a code bug — see
        # script/depth/train_sr_fm_refiner.sh); doing so on this cheap
        # barrier first, while nothing else is competing for the fabric,
        # is a common mitigation for that class of connection-setup race.
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
        # anchored to "how many steps at max_train_batch_size", which is
        # what these numbers were originally tuned against (2026-08-18:
        # confirmed against the single-GPU baseline that predates the
        # gradient-accumulation fix, i.e. accumulation_steps was always 1
        # then regardless of effective_batch_size — so that's the correct
        # zero point to scale from). Applied once, here, on a fresh run — a
        # resumed run reloads the already-scaled config.yaml saved by the
        # run being resumed.
        _eff_bs = cfg.dataloader.get("effective_batch_size", cfg.dataloader.max_train_batch_size)
        _accum_steps = max(1, _eff_bs // cfg.dataloader.max_train_batch_size)
        total_scale = _accum_steps * (world_size if is_distributed else 1)
        if total_scale > 1:
            cfg.max_iter = max(1, cfg.max_iter // total_scale)
            cfg.trainer.warmup_sr_steps = max(1, cfg.trainer.warmup_sr_steps // total_scale)
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
                    f"[scale] accumulation_steps={_accum_steps} x world_size="
                    f"{world_size if is_distributed else 1} = {total_scale}: scaled "
                    f"max_iter/warmup_sr_steps/validation_period/save_period/log_period "
                    f"down by 1/{total_scale} (now {cfg.max_iter}/"
                    f"{cfg.trainer.warmup_sr_steps}/{cfg.trainer.validation_period}/"
                    f"{cfg.trainer.save_period}/{cfg.trainer.log_period}) so this run "
                    f"covers the same total data (and wall-clock time) as a config "
                    f"training at real batch = max_train_batch_size alone."
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

    # ---- Data ----
    cfg_data = cfg.dataset
    # effective_batch_size is optional and defaults to max_train_batch_size
    # (accumulation_steps=1, i.e. no local accumulation): it only exists for
    # configs that need to simulate a larger batch on a single GPU (see
    # v5-1.yaml). A multi-GPU config doesn't need it — DDP's cross-rank
    # gradient averaging already provides that multiplier for free, so the
    # real per-step batch there is simply max_train_batch_size * world_size,
    # and max_train_batch_size is the only batch-size knob such a config
    # needs to set (see v5-1-lumi.yaml).
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
    val_dataset   = _make_dataset(base_dataset, val_coords,   'val',   1)

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank,
                           shuffle=True, seed=cfg.dataloader.seed)
        if is_distributed else None
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=cfg_data.workers, pin_memory=True)
    # validate() builds its own small on-demand DataLoader per call, over a
    # Subset of just the sample indices it actually needs (see validate()) —
    # so no val_loader is built here. Materializing a full DataLoader over
    # val_dataset up front is fine on 1 rank, but at 64 ranks each validation
    # call would otherwise force all 64 processes to eagerly read+transform
    # the *entire* val set from shared Lustre storage before subsetting down
    # to the handful of samples actually used — a 64-way concurrent I/O
    # storm that stalled training for many minutes when this ran unmodified.

    # ---- Models ----
    n_landsat_bands = len(cfg_data.selected_bands)
    n_ps_bands      = len(cfg_data.selected_bands_hr)
    concat_landsat_cond = bool(cfg.trainer.get("concat_landsat_cond", False))

    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=cfg.trainer.get("ps_dropout_p", 0.0),
        bridge_sigma=cfg.trainer.get("bridge_sigma", 0.0),
        concat_z_coarse=cfg.trainer.get("concat_z_coarse", False),
        concat_landsat_cond=concat_landsat_cond,
        refine_threshold=cfg.trainer.get("refine_threshold", 0.0),
        device=str(device),
    ).to(device)

    coarse_model_type = cfg.model.get("coarse_model", "dav2")
    _chmv2_mean = _chmv2_std = None
    if coarse_model_type == "chmv2":
        coarse_model = load_chmv2(cfg.model.chmv2_model_id).to(device)
        input_stats = json.load(open(cfg_data.input_stats_file))
        _chmv2_mean = input_stats[str(cfg_data.year)]["mean"]
        _chmv2_std  = input_stats[str(cfg_data.year)]["std"]
    else:
        coarse_model = load_dav2(
            dav2_path=cfg.model.dav2_pretrained_path,
            backbone=cfg.model.dav2_backbone,
            out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
        ).to(device)

    sr_module = build_sr_module(OmegaConf.to_container(cfg.sr_module, resolve=True)).to(device)

    # Load SwinIR pretrained weights if provided.
    # Only rank 0 reads the checkpoint from disk — DDP(sr_module_raw, ...)
    # below broadcasts rank 0's parameters to every other rank as part of its
    # normal construction, so having all `world_size` ranks independently
    # read this (large, shared-Lustre) file is both redundant and, at high
    # rank counts, slow enough to blow NCCL's 5-minute collective timeout
    # while DDP is being constructed (observed at 64 ranks / 8 nodes).
    sr_pretrained_path = cfg.model.get("sr_pretrained_path", None)
    if sr_pretrained_path and os.path.exists(sr_pretrained_path):
        if is_main_process:
            ckpt_sr = torch.load(sr_pretrained_path, map_location="cpu")
            key = cfg.model.get("sr_pretrained_key", None)
            state = ckpt_sr[key] if key and key in ckpt_sr else ckpt_sr
            missing, unexpected = sr_module.swinir.load_state_dict(state, strict=False)
            logging.info(f"SwinIR pretrained loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    elif is_main_process:
        logging.info("No SwinIR pretrained weights found; training from scratch.")

    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ---- DDP wrapping ----
    # Only unet/controlnet/sr_module are wrapped (not the whole FMRefiner):
    # the SR step freezes unet/controlnet and calls a few of their submodules
    # directly for the TDP loss (see _sr_step / _raw()), which must bypass DDP
    # since DDP's gradient-sync hooks assume requires_grad stays static after
    # construction — incompatible with this per-iteration freeze/unfreeze.
    # unet_raw/controlnet_raw/sr_module_raw keep referring to the underlying
    # modules (same parameter tensors) for that path, for validation, and for
    # checkpointing (via _raw()).
    unet_raw       = fm_refiner.unet
    controlnet_raw = fm_refiner.controlnet
    sr_module_raw  = sr_module
    if is_distributed:
        ddp_ids = [device.index] if device.type == "cuda" else None
        fm_refiner.unet       = DDP(unet_raw, device_ids=ddp_ids, output_device=ddp_ids[0] if ddp_ids else None)
        fm_refiner.controlnet = DDP(controlnet_raw, device_ids=ddp_ids, output_device=ddp_ids[0] if ddp_ids else None)
        sr_module              = DDP(sr_module_raw, device_ids=ddp_ids, output_device=ddp_ids[0] if ddp_ids else None)

    # ---- Optimizers ----
    optimizer_fm = torch.optim.AdamW(
        [
            {"params": unet_raw.parameters(),       "lr": cfg.optimizer.lr_unet},
            {"params": controlnet_raw.parameters(), "lr": cfg.optimizer.lr_controlnet},
        ],
        weight_decay=cfg.optimizer.weight_decay,
    )
    optimizer_sr = torch.optim.AdamW(
        sr_module_raw.parameters(),
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
    if args.resume_run is not None:
        start_step = load_checkpoint(
            fm_refiner, sr_module, optimizer_fm, optimizer_sr,
            lr_scheduler_fm, lr_scheduler_sr, args.resume_run
        )
        if is_main_process:
            logging.info(f"Resumed from step {start_step}")

    # ---- Training config ----
    p_null_drop     = float(cfg.trainer.p_null_drop)
    p_pseudo_ps     = float(cfg.trainer.p_pseudo_ps)
    warmup_sr_steps = int(cfg.trainer.warmup_sr_steps)
    pixel_weight    = float(cfg.sr_loss.pixel_weight)
    tdp_weight      = float(cfg.sr_loss.tdp_weight)
    tdp_hook        = str(cfg.sr_loss.get("tdp_hook", "unet.up_blocks.2"))

    # SR trains every batch; FM trains every batch with detached SR output.
    # Both modules stay in train mode throughout.
    _unfreeze(sr_module)
    _unfreeze(fm_refiner.unet)
    _unfreeze(fm_refiner.controlnet)
    fm_refiner.train()
    sr_module.train()

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
    val_offset = 0
    val_subset_size = max(1, len(val_dataset) // 10)

    loss_pix_accum = 0.0
    loss_tdp_accum = 0.0
    loss_fm_accum  = 0.0
    _pix_check_buf = []

    if is_main_process:
        logging.info("Starting joint SR + FM training (every batch updates both).")
    pbar = tqdm(total=cfg.max_iter, initial=step, desc="Training", dynamic_ncols=True,
                disable=not is_main_process)

    for epoch in range(cfg.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= cfg.max_iter:
                break
            if t_end is not None and datetime.now() >= t_end:
                if is_main_process:
                    logging.info("Exit after time limit reached.")
                    save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                                    lr_scheduler_fm, lr_scheduler_sr,
                                    step, out_dir_ckpt, "latest")
                if is_distributed:
                    _barrier()
                    dist.destroy_process_group()
                pbar.close()
                sys.exit(0)

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

            # Coarse depth (always frozen)
            with torch.no_grad():
                if coarse_model_type == "chmv2":
                    h_coarse = run_chmv2(coarse_model, landsat, (H_hr, W_hr), _chmv2_mean, _chmv2_std)
                else:
                    landsat_for_dav2 = (landsat + 1.0) / 2.0
                    h_coarse = run_dav2(coarse_model, landsat_for_dav2, target_size=(H_hr, W_hr))
                h_coarse = _normalize_coarse(h_coarse, target_stats)

            # ================================================================
            # Step 1: update SR  (FM frozen via no_grad on its forward pass)
            # ================================================================
            is_first_micro = (micro_step == 0)
            is_last_micro  = (micro_step == accumulation_steps - 1)
            # Only the final micro-batch of a window should let DDP all-reduce
            # gradients; earlier ones accumulate locally (see _maybe_no_sync).
            skip_sync = is_distributed and not is_last_micro

            skip_sr_training = cfg.trainer.get("skip_sr_training", False)
            if not skip_sr_training:
                use_tdp = (step >= warmup_sr_steps)
                _freeze(fm_refiner.unet)
                _freeze(fm_refiner.controlnet)

                if is_first_micro:
                    optimizer_sr.zero_grad()

                with _maybe_no_sync([sr_module] if is_distributed else [], skip_sync):
                    loss_dict = _sr_step(
                        fm_refiner, sr_module,
                        landsat, landsat_hr, inputs_hr, h_coarse,
                        pixel_weight, tdp_weight, use_tdp, tdp_hook, device,
                        concat_landsat_cond=concat_landsat_cond,
                    )
                    # Scale down so accumulation_steps backward() calls (summed
                    # into .grad by autograd) land on the *average* gradient
                    # over the effective batch, not its sum.
                    (loss_dict["l_total"] / accumulation_steps).backward()

                if is_last_micro:
                    torch.nn.utils.clip_grad_norm_(sr_module_raw.parameters(), max_norm=1.0)
                    optimizer_sr.step()

                # detach SR output before releasing the computation graph
                pseudo_ps_detached = loss_dict["pseudo_ps"].detach().clone()
                _l_pix_val = loss_dict["l_pix"].item()
                loss_pix_accum += _l_pix_val
                loss_tdp_accum += loss_dict["l_tdp"].item()
                _pix_check_buf.append(_l_pix_val)
                del loss_dict
                torch.cuda.empty_cache()
            else:
                pseudo_ps_detached = None

            # ================================================================
            # Step 2: update FM  (SR output detached, no second SR forward)
            # ================================================================
            _unfreeze(fm_refiner.unet)
            _unfreeze(fm_refiner.controlnet)

            if is_first_micro:
                optimizer_fm.zero_grad()

            fm_modules = [fm_refiner.unet, fm_refiner.controlnet] if is_distributed else []
            with _maybe_no_sync(fm_modules, skip_sync):
                loss_fm = _task_step(
                    fm_refiner,
                    inputs_hr, h_coarse, targets,
                    pseudo_ps_detached,
                    p_null_drop, p_pseudo_ps, device,
                )
                (loss_fm / accumulation_steps).backward()

            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(
                    list(unet_raw.parameters()) +
                    list(controlnet_raw.parameters()),
                    max_norm=1.0,
                )
                optimizer_fm.step()

            loss_fm_accum += loss_fm.item()

            if not is_last_micro:
                micro_step += 1
                continue
            micro_step = 0

            if lr_scheduler_sr is not None:
                lr_scheduler_sr.step()
            if lr_scheduler_fm is not None:
                lr_scheduler_fm.step()

            step += 1
            pbar.update(1)

            # ---- Logging (rank0 only) ----
            # loss_*_accum accumulates every micro-batch (not just every real
            # step), so the averaging window is log_period real steps' worth
            # of micro-batches, not log_period micro-batches.
            if is_main_process and step % cfg.trainer.log_period == 0:
                log_n = cfg.trainer.log_period * accumulation_steps
                lr_sr = optimizer_sr.param_groups[0]["lr"]
                lr_fm = optimizer_fm.param_groups[0]["lr"]
                log_dict = {
                    "lr/sr": lr_sr, "lr/fm": lr_fm,
                    "train/l_pix": loss_pix_accum / log_n,
                    "train/l_tdp": loss_tdp_accum / log_n,
                    "train/l_fm":  loss_fm_accum  / log_n,
                }
                logging.info(
                    f"[step {step}] "
                    f"l_pix={log_dict['train/l_pix']:.5f} "
                    f"l_tdp={log_dict['train/l_tdp']:.5f} "
                    f"l_fm={log_dict['train/l_fm']:.5f}"
                )
                pbar.set_postfix(r2=f"{best_r2:.4f}")
                wandb.log(log_dict, step=step)
                tb_logger.log_dict(log_dict, global_step=step)
                loss_pix_accum = loss_tdp_accum = loss_fm_accum = 0.0
            elif step % cfg.trainer.log_period == 0:
                loss_pix_accum = loss_tdp_accum = loss_fm_accum = 0.0

            # ---- Validation ----
            # Every rank participates (sharded, parallel inference — see
            # validate()'s docstring); only rank0 gets real metrics back and
            # acts on them (logging / checkpointing).
            if step % cfg.trainer.validation_period == 0:
                metrics, fig = validate(
                    fm_refiner, sr_module, coarse_model, val_dataset, device,
                    n_steps=cfg.validation.n_steps,
                    target_stats=target_stats,
                    n_landsat_bands=n_landsat_bands,
                    val_offset=val_offset,
                    method=cfg.validation.get("method", "euler"),
                    use_sr_pseudo_ps=cfg.validation.get("use_sr_pseudo_ps", True),
                    use_real_ps=cfg.validation.get("use_real_ps", False),
                    coarse_model_type=coarse_model_type,
                    chmv2_mean=_chmv2_mean if coarse_model_type == "chmv2" else None,
                    chmv2_std=_chmv2_std if coarse_model_type == "chmv2" else None,
                    n_avg=cfg.validation.get("n_avg", 1),
                    global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
                    # num_workers left at the validate() default (0): each
                    # rank only loads ~n_total/10/world_size samples here, too
                    # few to be worth the worker-process spin-up cost.
                )
                # Deterministic given val_offset/val_subset_size/len(val_dataset),
                # which are identical on every rank — safe to update everywhere.
                val_offset = (val_offset + val_subset_size) % len(val_dataset)

                if is_main_process:
                    logging.info(f"[step {step}] val: {metrics}")
                    vlog = {f"val/{k}": v for k, v in metrics.items()}
                    if fig is not None:
                        vlog["val/samples"] = wandb.Image(fig)
                        plt.close(fig)
                    wandb.log(vlog, step=step)
                    tb_logger.log_dict({f"val/{k}": v for k, v in metrics.items()}, global_step=step)

                    if metrics["r2"] > best_r2:
                        best_r2 = metrics["r2"]
                        pbar.set_postfix(r2=f"{best_r2:.4f}")
                        save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                                        lr_scheduler_fm, lr_scheduler_sr,
                                        step, out_dir_ckpt, "best")
                        logging.info(f"New best R²={best_r2:.4f} at step {step}")

                # restore train mode after validate() on every rank
                fm_refiner.train()
                sr_module.train()
                _unfreeze(fm_refiner.unet)
                _unfreeze(fm_refiner.controlnet)
                _unfreeze(sr_module)

                if is_distributed:
                    # rank0's checkpoint write above must finish before other
                    # ranks resume training and possibly overwrite "latest".
                    _barrier()

            # ---- Periodic checkpoint (rank0 only) ----
            if is_main_process and step % cfg.trainer.save_period == 0:
                save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                                lr_scheduler_fm, lr_scheduler_sr,
                                step, out_dir_ckpt, "latest")

        if step >= cfg.max_iter:
            break

    pbar.close()

    # ---- Final full validation ----
    # Every rank must load the same best checkpoint (not just rank0) since
    # the sharded validate() call below combines predictions across ranks —
    # mixing a stale-weight rank into that would corrupt the aggregate metric.
    best_ckpt = os.path.join(out_dir_ckpt, "best.pth")
    if os.path.exists(best_ckpt):
        load_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                        lr_scheduler_fm, lr_scheduler_sr, best_ckpt)
        if is_main_process:
            logging.info(f"Loaded best checkpoint for final validation")

    if is_main_process:
        logging.info("Running full validation on entire val set...")
    final_metrics, final_fig = validate(
        fm_refiner, sr_module, coarse_model, val_dataset, device,
        n_steps=cfg.validation.n_steps,
        target_stats=target_stats,
        n_landsat_bands=n_landsat_bands,
        full=True,
        method=cfg.validation.get("method", "euler"),
        use_sr_pseudo_ps=cfg.validation.get("use_sr_pseudo_ps", True),
        use_real_ps=cfg.validation.get("use_real_ps", False),
        coarse_model_type=coarse_model_type,
        chmv2_mean=_chmv2_mean if coarse_model_type == "chmv2" else None,
        chmv2_std=_chmv2_std if coarse_model_type == "chmv2" else None,
        n_avg=cfg.validation.get("n_avg", 1),
        global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
        num_workers=cfg_data.workers,  # full=True shards the whole val set,
        # worth the worker-process cost here (unlike the periodic call above).
    )

    if is_main_process:
        logging.info(f"Final val metrics: {final_metrics}")
        flog = {f"val_final/{k}": v for k, v in final_metrics.items()}
        if final_fig is not None:
            flog["val_final/samples"] = wandb.Image(final_fig)
            plt.close(final_fig)
        wandb.log(flog, step=step)

        save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                        lr_scheduler_fm, lr_scheduler_sr,
                        step, out_dir_ckpt, "final")
        logging.info(f"Training finished at step {step}. Best val R²={best_r2:.4f}")

    if is_distributed:
        _barrier()
        dist.destroy_process_group()
