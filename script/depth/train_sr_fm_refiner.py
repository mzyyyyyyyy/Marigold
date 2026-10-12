"""
Joint SR + FM-refiner training for the DiT refiner (sr_fm_refiner_d_v0-lumi).

This is fm_refiner-D-lumi-0 (backbone "dit_omini", frozen CHMv2-baseline coarse model, sigma=1 / init_only
flow) plus a trainable SwinIR super-resolution module that turns Landsat into a pseudo-PlanetScope image,
ported from the UNet version (sr_fm_refiner_v12-lumi, sr_finer_lumi branch, train_sr_fm_refiner.py):

  every iteration, on the same batch:
    1. SR step (FM backbone frozen):
         pseudo_ps = SR(landsat)
         l_pix = L1(pseudo_ps, real_ps)
         l_tdp = L1(F(pseudo_ps), F(real_ps))   [after warmup_sr_steps]
       where F = the X-token output of DiT block `tdp_block` (default: the last block) given a
       t-sampled, noised z_t and the condition [landsat, PS] (the DiT analogue of v12's
       "unet.up_blocks.3" hook). The condition images go through the frozen VAE encoder, which
       therefore has to be differentiable w.r.t. its input here (DiTVAECondEncoder.cond_grad).
    2. FM step (SR output detached): standard flow-matching loss with the condition
       [landsat, ps_cond], ps_cond = pseudo-PS for p_pseudo_ps of the samples, the null PS for
       p_null_drop, and the real PS otherwise (v12 / this config: pseudo-PS always).
  The real PS is therefore only seen through the SR losses; inference uses pseudo-PS only.

Everything about the FM part (data, noise schedule, optimiser, scheduler, budget, coarse model) is
unchanged from fm_refiner-D-lumi-0; the SR part (SwinIR x4 + DIV2K weights, lr_sr, loss weights) is v12's.
Differences from v12 that are deliberate: TDP feature location (DiT), TDP warmup length (config).

Usage: see script/depth/train_fm_refiner.sh (SCRIPT/CONFIG env overrides).
"""

import contextlib
import copy
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

from depthfm.fm_refiner import FMRefiner, build_fm_refiner
from depthfm.chmv2 import CHMv2Height, load_chmv2_baseline
from depthfm.sr_module import SRModule, build_sr_module
from src.util.config_util import recursive_load_config
from src.util.logging_util import config_logging, init_wandb, tb_logger
from src.util.ps_lazydataset import LazyPatchDataset
from src.util.seeding import generate_seed_sequence

# Helpers shared verbatim with the plain FM-refiner training script.
from train_fm_refiner import (
    _split_coords, _make_dataset, compute_metrics, _to_vis_rgb, _to_vis_depth,
    _run_coarse, _normalize_coarse, _denormalize_target, _load_target_stats,
    _raw, _maybe_no_sync,
)


# -------------------------------------------------------------------------
# Validation (pseudo-PS from the SR module, or real PS / null PS for comparison)
# -------------------------------------------------------------------------

def _make_vis_figure_sr(samples: list) -> plt.Figure:
    keys = ["landsat", "pseudo_ps", "real_ps", "coarse", "fine", "gt"]
    titles = ["Landsat (RGB)", "Pseudo-PS (SR)", "Real PS", "Coarse", "Refined", "GT"]
    n = len(samples)
    fig, axes = plt.subplots(n, len(keys), figsize=(4 * len(keys), 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for r, s in enumerate(samples):
        for c, (k, title) in enumerate(zip(keys, titles)):
            ax = axes[r, c]
            if k in ("landsat", "pseudo_ps", "real_ps"):
                ax.imshow(s[k])
            else:
                im = ax.imshow(s[k], cmap="plasma", vmin=-1.0, vmax=1.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if r == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    return fig


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
    ps_source: str = "pseudo",     # "pseudo" (SR), "real" (real PS), "null" (null PS)
    ensemble_size: int = 1,
    global_rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
    num_workers: int = 0,
) -> tuple:
    """Same sharded validation as train_fm_refiner.validate(), with the PS condition chosen by
    ps_source. Non-rank0 callers get (None, None)."""
    orig_unet, orig_controlnet = fm_refiner.unet, fm_refiner.controlnet
    fm_refiner.unet, fm_refiner.controlnet = _raw(orig_unet), _raw(orig_controlnet)
    sr_raw = _raw(sr_module)

    fm_refiner.eval()
    sr_raw.eval()
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

        h_coarse = _run_coarse(coarse_model, (landsat + 1.0) / 2.0, target_size=(H_hr, W_hr))
        h_coarse = _normalize_coarse(h_coarse, target_stats)

        pseudo_ps = sr_raw(landsat, target_size=(H_hr, W_hr))
        if ps_source == "pseudo":
            ps_input = pseudo_ps
        elif ps_source == "real":
            ps_input = inputs_hr
        elif ps_source == "null":
            ps_input = None          # refine() falls back to the null PS token
        else:
            raise ValueError(f"ps_source must be 'pseudo', 'real' or 'null', got {ps_source!r}")

        h_fine_draws = [
            fm_refiner.refine(landsat_hr, h_coarse, n_steps=n_steps, method=method,
                              **({"ps_hr": ps_input} if ps_input is not None else {}))
            for _ in range(ensemble_size)
        ]
        h_fine = torch.stack(h_fine_draws, dim=0).mean(dim=0)

        all_preds.append(_denormalize_target(h_fine, target_stats).cpu().flatten())
        all_gts.append(_denormalize_target(targets, target_stats).cpu().flatten())

        if len(vis_samples) < n_vis_samples:
            vis_samples.append({
                "landsat":   _to_vis_rgb(landsat_hr[0]),
                "pseudo_ps": _to_vis_rgb(pseudo_ps[0]),
                "real_ps":   _to_vis_rgb(inputs_hr[0]),
                "coarse":    _to_vis_depth(h_coarse[0]),
                "fine":      _to_vis_depth(h_fine[0]),
                "gt":        _to_vis_depth(targets[0]),
            })

    pred_local = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0)
    gt_local   = torch.cat(all_gts, dim=0)   if all_gts   else torch.empty(0)

    fm_refiner.train()
    sr_raw.train()
    fm_refiner.unet, fm_refiner.controlnet = orig_unet, orig_controlnet

    if is_distributed:
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
    fig = _make_vis_figure_sr(vis_samples) if vis_samples else None
    return metrics, fig


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr, sched_fm, sched_sr, step, out_dir, name="latest"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    torch.save({
        "step": step,
        "use_controlnet": fm_refiner.use_controlnet,
        "unet_state": _raw(fm_refiner.unet).state_dict(),
        "controlnet_state": _raw(fm_refiner.controlnet).state_dict(),
        "null_ps_state": fm_refiner._null_ps,
        "sr_state": _raw(sr_module).state_dict(),
        "optimizer_fm_state": optimizer_fm.state_dict(),
        "optimizer_sr_state": optimizer_sr.state_dict(),
        "lr_scheduler_fm_state": sched_fm.state_dict() if sched_fm else None,
        "lr_scheduler_sr_state": sched_sr.state_dict() if sched_sr else None,
    }, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr, sched_fm, sched_sr, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _raw(fm_refiner.unet).load_state_dict(ckpt["unet_state"])
    _raw(fm_refiner.controlnet).load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    _raw(sr_module).load_state_dict(ckpt["sr_state"])
    optimizer_fm.load_state_dict(ckpt["optimizer_fm_state"])
    optimizer_sr.load_state_dict(ckpt["optimizer_sr_state"])
    if sched_fm and ckpt.get("lr_scheduler_fm_state"):
        sched_fm.load_state_dict(ckpt["lr_scheduler_fm_state"])
    if sched_sr and ckpt.get("lr_scheduler_sr_state"):
        sched_sr.load_state_dict(ckpt["lr_scheduler_sr_state"])
    return ckpt["step"]


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


# -------------------------------------------------------------------------
# SR step: pixel L1 + TDP feature loss through the frozen DiT
# -------------------------------------------------------------------------

def _sr_step(
    fm_refiner: FMRefiner,
    sr_module: SRModule,
    landsat: torch.Tensor,       # (B, C_ls, H_lr, W_lr)
    landsat_hr: torch.Tensor,    # (B, C_ls, H_hr, W_hr)
    real_ps: torch.Tensor,       # (B, C_ps, H_hr, W_hr)
    h_coarse: torch.Tensor,
    h_gt: torch.Tensor,
    pixel_weight: float,
    tdp_weight: float,
    use_tdp: bool,
    tdp_block: int,              # index into the DiT blocks (-1 = last)
    device: torch.device,
    tdp_t_min: float = 0.0,
    tdp_t_max: float = 1.0,
) -> dict:
    """Port of v12's _sr_step. The FM backbone/condition encoder are frozen by the caller
    (requires_grad False) but gradients still flow through them to the pseudo-PS. Always operates on
    the raw (non-DDP) modules."""
    B = landsat.shape[0]
    H_hr, W_hr = real_ps.shape[2], real_ps.shape[3]
    cond_enc = _raw(fm_refiner.controlnet)
    backbone = _raw(fm_refiner.unet)

    pseudo_ps = sr_module(landsat, target_size=(H_hr, W_hr))
    l_pix = F.l1_loss(pseudo_ps, real_ps.detach()) * pixel_weight

    l_tdp = torch.tensor(0.0, device=device)
    if use_tdp and tdp_weight > 0:
        with torch.no_grad():
            z_coarse = fm_refiner.encode(h_coarse)
            z_gt = fm_refiner.encode(h_gt)
        # t-sampled, noised z_t exactly as FMRefiner.forward builds it (see v12's _sr_step for why)
        t_val = tdp_t_min + (tdp_t_max - tdp_t_min) * torch.rand(B, device=device)
        t4 = t_val.view(B, 1, 1, 1)
        z_t = (1.0 - t4) * z_coarse + t4 * z_gt
        if fm_refiner.noise_sigma > 0.0:
            z_t = z_t + (1.0 - t4) * fm_refiner.noise_sigma * torch.randn_like(z_coarse)
        z_t = z_t.detach()
        t_int = (t_val * 999).long()
        text_emb = fm_refiner.empty_text_embed.to(device).expand(B, -1, -1)
        grid = z_t.shape[-1] // backbone.patch_size
        n_x = grid * grid                       # number of latent (X) tokens; condition tokens follow

        cond_fake = torch.cat([landsat_hr, pseudo_ps], dim=1)
        cond_real = torch.cat([landsat_hr, real_ps], dim=1)

        feats = {}
        hook = backbone.blocks[tdp_block].register_forward_hook(lambda m, i, o: feats.__setitem__("out", o))

        def _run(cond):
            co = cond_enc(sample=z_t, timestep=t_int, encoder_hidden_states=text_emb,
                          controlnet_cond=cond, return_dict=True)
            backbone(sample=z_t, timestep=t_int, encoder_hidden_states=text_emb,
                     down_block_additional_residuals=co.down_block_res_samples,
                     mid_block_additional_residual=co.mid_block_res_sample, return_dict=True)
            return feats["out"][:, :n_x].float()

        cond_enc.cond_grad = True               # gradients through the frozen VAE encoder to pseudo_ps
        try:
            feat_fake = _run(cond_fake)
        finally:
            cond_enc.cond_grad = False
        with torch.no_grad():
            feat_real = _run(cond_real)
        hook.remove()
        l_tdp = F.l1_loss(feat_fake, feat_real.detach()) * tdp_weight

    return {"l_pix": l_pix, "l_tdp": l_tdp, "l_total": l_pix + l_tdp, "pseudo_ps": pseudo_ps}


# -------------------------------------------------------------------------
# FM step
# -------------------------------------------------------------------------

def _fm_step(fm_refiner, real_ps, landsat_hr, h_coarse, h_gt, pseudo_ps_detached,
             p_null_drop: float, p_pseudo_ps: float, device) -> torch.Tensor:
    """FM loss; per-sample PS source: null (p_null_drop) / pseudo-PS (p_pseudo_ps) / real (remainder)."""
    B = real_ps.shape[0]
    H_hr, W_hr = real_ps.shape[2], real_ps.shape[3]
    r = torch.rand(B, device=device)
    null_mask = r < p_null_drop
    pseudo_mask = (r >= p_null_drop) & (r < p_null_drop + p_pseudo_ps)

    ps_cond = real_ps.clone()
    null_spatial = fm_refiner._get_null_ps(real_ps.shape[1], device, real_ps.dtype).expand(1, -1, H_hr, W_hr)
    for i in range(B):
        if null_mask[i]:
            ps_cond[i] = null_spatial[0]
        elif pseudo_mask[i]:
            ps_cond[i] = pseudo_ps_detached[i]
    # fm_refiner.ps_dropout_p must be 0 here: the PS source is decided above, not inside forward()
    return fm_refiner(landsat_lr=landsat_hr, ps_hr=ps_cond, h_coarse=h_coarse, h_gt=h_gt)


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
        if not cfg.get("scale_steps_by_batch", True):
            # step counts in this config are REAL optimizer steps (from-scratch DiT needs far more
            # steps than the ~3k the batch-anchored rule above leaves after dividing by 16)
            total_scale = 1
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

    if cfg.model.get("refiner_type", "sd") != "sd":
        raise NotImplementedError("train_sr_fm_refiner.py supports only the FMRefiner (UNet/DiT backbones).")
    if not cfg.trainer.get("use_controlnet", True):
        raise ValueError("SR training needs the condition path (trainer.use_controlnet: True).")
    if cfg.trainer.get("ps_dropout_p", 0.0) != 0.0:
        raise ValueError("trainer.ps_dropout_p must be 0: the PS source is chosen by p_null_drop / p_pseudo_ps.")
    if cfg.model.get("backbone", "unet") != "dit_omini":
        raise NotImplementedError("the TDP feature hook is implemented for backbone 'dit_omini' only.")
    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_landsat_bands,
        n_ps_bands=n_ps_bands,
        ps_dropout_p=0.0,
        use_controlnet=True,
        controlnet_cond_mode=cfg.trainer.get("controlnet_cond_mode", "landsat_ps"),
        velocity_parameterization=cfg.trainer.get("velocity_parameterization", "fixed"),
        noise_sigma=cfg.trainer.get("noise_sigma", 0.0),
        sample_sigma=cfg.trainer.get("sample_sigma", None),
        sample_noise_mode=cfg.trainer.get("sample_noise_mode", "sde"),
        device=str(device),
        pretrained_unet=cfg.model.get("pretrained_unet", True),
        backbone=cfg.model.get("backbone", "unet"),
        dit_kwargs=dict(cfg.model.get("dit", {}) or {}),
        loss_pixel_weight=cfg.trainer.get("loss_pixel_weight", 0.0),
        pixel_loss_t_min=cfg.trainer.get("pixel_loss_t_min", 0.5),
    )
    fm_refiner = fm_refiner.to(device)

    # ---- Coarse-height model (frozen CHMv2 baseline) ----
    if cfg.model.get("coarse_model", "dav2") != "chmv2":
        raise NotImplementedError("train_sr_fm_refiner.py supports coarse_model: 'chmv2' only.")
    with open(cfg_data.input_stats_file) as f:
        _in_stats = json.load(f)[str(cfg_data.year)]
    coarse_model = load_chmv2_baseline(
        model_dir=cfg.model.chmv2_model_id,
        ckpt_path=cfg.model.chmv2_ckpt_path,
        mean=_in_stats["mean"], std=_in_stats["std"],
        out_in_scale_factor=cfg.model.chmv2_out_in_scale_factor,
    ).to(device)

    # ---- SR module (SwinIR) ----
    sr_module = build_sr_module(OmegaConf.to_container(cfg.sr_module, resolve=True)).to(device)
    # Only rank 0 reads the (large, shared-Lustre) pretrained file; DDP(sr_module) below broadcasts
    # rank 0's parameters to every other rank (see train_sr_fm_refiner.py on sr_finer_lumi).
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
    if is_main_process:
        n_bb = sum(p.numel() for p in fm_refiner.unet.parameters()) / 1e6
        n_ce = sum(p.numel() for p in fm_refiner.controlnet.parameters()) / 1e6
        n_sr = sum(p.numel() for p in sr_module.parameters()) / 1e6
        logging.info(f"Params: DiT backbone {n_bb:.1f}M + condition encoder {n_ce:.1f}M (trainable), SR {n_sr:.1f}M")

    # ---- DDP wrapping (unet / controlnet / sr_module separately, as in v12: the SR step freezes the
    # FM modules and calls them through their raw (non-DDP) versions, see _sr_step) ----
    unet_raw       = fm_refiner.unet
    controlnet_raw = fm_refiner.controlnet
    sr_module_raw  = sr_module
    if is_distributed:
        ddp_ids = [device.index] if device.type == "cuda" else None
        _kw = dict(device_ids=ddp_ids, output_device=ddp_ids[0] if ddp_ids else None)
        fm_refiner.unet       = DDP(unet_raw, **_kw)
        fm_refiner.controlnet = DDP(controlnet_raw, **_kw)
        sr_module             = DDP(sr_module_raw, **_kw)

    # ---- Optimizers / schedulers ----
    # FM: identical to fm_refiner-D-lumi-0. SR: v12's lr_sr / weight_decay_sr.
    optimizer_fm = torch.optim.AdamW(
        [{"params": unet_raw.parameters(), "lr": cfg.optimizer.lr_unet},
         {"params": controlnet_raw.parameters(), "lr": cfg.optimizer.lr_controlnet}],
        weight_decay=cfg.optimizer.weight_decay)
    optimizer_sr = torch.optim.AdamW(sr_module_raw.parameters(), lr=cfg.optimizer.lr_sr,
                                     weight_decay=cfg.optimizer.weight_decay_sr)

    sched_fm, sched_sr = None, None
    if cfg.get("lr_scheduler") is not None:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        # FM: as fm_refiner-D-lumi-0. warmup_steps counts optimizer steps AFTER the [scale] adjustment of max_iter.
        warmup_steps = int(cfg.lr_scheduler.get("warmup_steps", 0))
        cosine_fm = CosineAnnealingLR(optimizer_fm, T_max=cfg.max_iter - warmup_steps,
                                      eta_min=cfg.lr_scheduler.eta_min)
        sched_fm = cosine_fm if warmup_steps <= 0 else SequentialLR(
            optimizer_fm, [LinearLR(optimizer_fm, start_factor=1e-3, total_iters=warmup_steps), cosine_fm],
            milestones=[warmup_steps])
        # SR: as sr_fm_refiner_v12-lumi, which has no warmup (plain cosine over max_iter), independent of
        # the FM warmup above.
        sched_sr = CosineAnnealingLR(optimizer_sr, T_max=cfg.max_iter, eta_min=cfg.lr_scheduler.eta_min)

    # ---- Resume ----
    start_step = 0
    if args.resume_run is not None:
        start_step = load_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr,
                                     sched_fm, sched_sr, args.resume_run)
        if is_main_process:
            logging.info(f"Resumed from step {start_step}")

    # ---- Training config ----
    p_null_drop     = float(cfg.trainer.p_null_drop)
    p_pseudo_ps     = float(cfg.trainer.p_pseudo_ps)
    warmup_sr_steps = int(cfg.trainer.warmup_sr_steps)   # optimizer steps, NOT rescaled by [scale]
    pixel_weight    = float(cfg.sr_loss.pixel_weight)
    tdp_weight      = float(cfg.sr_loss.tdp_weight)
    tdp_block       = int(cfg.sr_loss.get("tdp_block", -1))
    tdp_t_min       = float(cfg.sr_loss.get("tdp_t_min", 0.0))
    tdp_t_max       = float(cfg.sr_loss.get("tdp_t_max", 1.0))
    val_ps_source   = "pseudo" if cfg.validation.get("use_sr_pseudo_ps", True) else (
        "real" if cfg.validation.get("use_real_ps", False) else "null")

    _unfreeze(sr_module)
    _unfreeze(fm_refiner.unet)
    _unfreeze(fm_refiner.controlnet)
    fm_refiner.train()
    sr_module.train()

    # ---- Training loop ----
    t_end = t_start + timedelta(minutes=args.exit_after) if args.exit_after > 0 else None

    step = start_step
    micro_step = 0
    best_r2 = -1e8
    val_offset = 0
    val_subset_size = max(1, len(val_dataset) // 10)

    loss_pix_accum = loss_tdp_accum = loss_fm_accum = 0.0

    def _ckpt(name):
        save_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr, sched_fm, sched_sr,
                        step, out_dir_ckpt, name)

    def _val(full, ps_source=val_ps_source):
        return validate(
            fm_refiner, sr_module, coarse_model, val_dataset, device,
            n_steps=cfg.validation.get("final_n_steps", cfg.validation.n_steps) if full else cfg.validation.n_steps,
            target_stats=target_stats, n_landsat_bands=n_landsat_bands,
            val_offset=val_offset, full=full,
            method=cfg.validation.get("method", "euler"), ps_source=ps_source,
            ensemble_size=(cfg.validation.get("final_ensemble_size", cfg.validation.get("ensemble_size", 1))
                           if full else cfg.validation.get("ensemble_size", 1)),
            global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
            num_workers=cfg_data.workers if full else 0,
        )

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
                    _ckpt("latest")
                if is_distributed:
                    _barrier()
                    dist.destroy_process_group()
                pbar.close()
                sys.exit(0)

            inputs_lr, inputs_hr, targets = batch
            if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
            if inputs_hr.dim() == 5: inputs_hr = inputs_hr.squeeze(0)
            if targets.dim() == 5: targets = targets.squeeze(0)
            inputs_lr, inputs_hr, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)

            H_hr, W_hr = targets.shape[2], targets.shape[3]
            landsat = inputs_lr[:, :n_landsat_bands]
            landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

            with torch.no_grad():
                h_coarse = _run_coarse(coarse_model, (landsat + 1.0) / 2.0, target_size=(H_hr, W_hr))
                h_coarse = _normalize_coarse(h_coarse, target_stats)

            is_first_micro = (micro_step == 0)
            is_last_micro  = (micro_step == accumulation_steps - 1)
            skip_sync = is_distributed and not is_last_micro

            # ---- Step 1: SR (FM modules frozen) ----
            use_tdp = (step >= warmup_sr_steps)
            _freeze(fm_refiner.unet)
            _freeze(fm_refiner.controlnet)
            if is_first_micro:
                optimizer_sr.zero_grad()
            with _maybe_no_sync([sr_module] if is_distributed else [], skip_sync):
                loss_dict = _sr_step(
                    fm_refiner, sr_module, landsat, landsat_hr, inputs_hr, h_coarse, targets,
                    pixel_weight, tdp_weight, use_tdp, tdp_block, device,
                    tdp_t_min=tdp_t_min, tdp_t_max=tdp_t_max)
                (loss_dict["l_total"] / accumulation_steps).backward()
            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(sr_module_raw.parameters(), max_norm=1.0)
                optimizer_sr.step()
            pseudo_ps_detached = loss_dict["pseudo_ps"].detach().clone()
            loss_pix_accum += loss_dict["l_pix"].item()
            loss_tdp_accum += loss_dict["l_tdp"].item()
            del loss_dict
            torch.cuda.empty_cache()

            # ---- Step 2: FM (SR output detached) ----
            _unfreeze(fm_refiner.unet)
            _unfreeze(fm_refiner.controlnet)
            if is_first_micro:
                optimizer_fm.zero_grad()
            fm_modules = [fm_refiner.unet, fm_refiner.controlnet] if is_distributed else []
            with _maybe_no_sync(fm_modules, skip_sync):
                loss_fm = _fm_step(fm_refiner, inputs_hr, landsat_hr, h_coarse, targets,
                                   pseudo_ps_detached, p_null_drop, p_pseudo_ps, device)
                (loss_fm / accumulation_steps).backward()
            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(
                    list(unet_raw.parameters()) + list(controlnet_raw.parameters()), max_norm=1.0)
                optimizer_fm.step()
            loss_fm_accum += loss_fm.item()

            if not is_last_micro:
                micro_step += 1
                continue
            micro_step = 0

            if sched_sr is not None:
                sched_sr.step()
            if sched_fm is not None:
                sched_fm.step()

            step += 1
            pbar.update(1)

            # Logging (rank0 only). The accumulators hold every micro-batch of the window.
            if is_main_process and step % cfg.trainer.log_period == 0:
                log_n = cfg.trainer.log_period * accumulation_steps
                log_dict = {
                    "lr/sr": optimizer_sr.param_groups[0]["lr"], "lr/fm": optimizer_fm.param_groups[0]["lr"],
                    "train/l_pix": loss_pix_accum / log_n, "train/l_tdp": loss_tdp_accum / log_n,
                    "train/l_fm": loss_fm_accum / log_n,
                }
                logging.info(f"[step {step}] l_pix={log_dict['train/l_pix']:.5f} "
                             f"l_tdp={log_dict['train/l_tdp']:.5f} l_fm={log_dict['train/l_fm']:.5f} "
                             f"tdp={'on' if use_tdp else 'warmup'}")
                pbar.set_postfix(r2=f"{best_r2:.4f}")
                wandb.log(log_dict, step=step)
                tb_logger.log_dict(log_dict, global_step=step)
            if step % cfg.trainer.log_period == 0:
                loss_pix_accum = loss_tdp_accum = loss_fm_accum = 0.0

            # Validation (rotating 10% subset, sharded over all ranks)
            if step % cfg.trainer.validation_period == 0:
                metrics, fig = _val(full=False)
                val_offset = (val_offset + val_subset_size) % len(val_dataset)
                if is_main_process:
                    logging.info(f"[step {step}] val ({val_ps_source} PS): {metrics}")
                    log_dict = {f"val/{k}": v for k, v in metrics.items()}
                    if fig is not None:
                        log_dict["val/samples"] = wandb.Image(fig)
                        plt.close(fig)
                    wandb.log(log_dict, step=step)
                    tb_logger.log_dict({f"val/{k}": v for k, v in metrics.items()}, global_step=step)
                    if metrics["r2"] > best_r2:
                        best_r2 = metrics["r2"]
                        pbar.set_postfix(r2=f"{best_r2:.4f}")
                        _ckpt("best")
                        logging.info(f"New best R²={best_r2:.4f} at step {step}")
                fm_refiner.train()
                sr_module.train()
                _unfreeze(fm_refiner.unet); _unfreeze(fm_refiner.controlnet); _unfreeze(sr_module)
                if is_distributed:
                    _barrier()

            if is_main_process and step % cfg.trainer.save_period == 0:
                _ckpt("latest")

        if step >= cfg.max_iter:
            break

    pbar.close()

    # Final full validation on best.pth (every rank loads it: the sharded validate() combines ranks).
    best_ckpt_path = os.path.join(out_dir_ckpt, "best.pth")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(fm_refiner, sr_module, optimizer_fm, optimizer_sr, sched_fm, sched_sr, best_ckpt_path)
        if is_main_process:
            logging.info(f"Loaded best checkpoint from {best_ckpt_path} for final validation")
    elif is_main_process:
        logging.warning("Best checkpoint not found, using final model weights for full validation")

    if is_main_process:
        logging.info("Running full validation on entire val set...")
    final_metrics, final_fig = _val(full=True)
    if is_main_process:
        logging.info(f"Final val metrics ({val_ps_source} PS): {final_metrics}")
        final_log = {f"val_final/{k}": v for k, v in final_metrics.items()}
        if final_fig is not None:
            final_log["val_final/samples"] = wandb.Image(final_fig)
            plt.close(final_fig)
        wandb.log(final_log, step=step)
    if is_main_process:
        _ckpt("final")
        logging.info(f"Training finished at step {step}. Best val R²={best_r2:.4f}")

    if is_distributed:
        _barrier()
        dist.destroy_process_group()
