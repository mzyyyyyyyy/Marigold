"""
Baseline training script: SwT-style (SR with Task, joint training) SR + regression
(no flow-matching diffusion refiner). SwT is the SR4IR paper's "SwT" comparison
setting (SR4IR/options/*/013_SwT_*.yml, models/*/sr_*_model.py with both
sr_is_trainable and seg_is_trainable) applied to the regression task, on the same
dataset / patch split / input channels as baseline_sr4ir_reg.

Differences from baseline_sr4ir_reg: no TDP loss, no warmup, no alternate
(two-phase) training, single joint step.

Pipeline (one joint step, one backward):
    LS → SwinIR → pseudo-PS (img_sr)
    pred = CHMv2(img_sr)  # DINOv3 encoder frozen, neck + DPT head trained
    loss = L1(img_sr, real_ps) + Huber(pred, H_gt)
    update net_sr and net_reg (neck + head only; encoder frozen) together.

Usage (single GPU):
  python script/depth/train_baseline_swt_reg.py --config config/baseline_swt-lumi.yaml

Usage (multi-GPU, standard DDP, e.g. 64 GPUs via srun):
  Reads SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID from the environment
  automatically (same as train_sr_fm_refiner.py), no extra flags needed —
  see script/depth/train_baseline_swt_reg.sh. net_sr / net_reg are
  DDP-wrapped; data is sharded via DistributedSampler; validation runs
  sharded across all ranks and is gathered to rank0, which alone handles
  logging/wandb/checkpointing.
"""

import contextlib
import copy
import json
import logging
import os
import pickle
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
from depthfm.sr_module import build_sr_module
from src.util.config_util import recursive_load_config
from src.util.logging_util import config_logging, init_wandb, tb_logger
from src.util.ps_lazydataset import LazyPatchDataset
from src.util.seeding import seed_all


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


def _load_coords(path):
    """Patch-coordinate list from script/depth/make_spatial_split.py."""
    with open(path, "rb") as f:
        coords = pickle.load(f)
    missing = [k for k in ("input_path", "input_path_hr", "output_path") if not os.path.exists(coords[0][k])]
    assert not missing, f"{path}: source files for {missing} do not exist (paths stale?)"
    return coords


def _make_dataset(base_dataset, coords, mode, batch_size, deterministic_eval=False):
    dataset = copy.copy(base_dataset)
    dataset.batch_size = batch_size
    dataset.mode = mode
    dataset.patch_coords = coords
    if deterministic_eval:
        # base_dataset built its transforms with mode='train' (random
        # brightness/contrast/gamma); rebuild them for eval, and make item idx
        # return exactly patch_coords[idx] (every patch evaluated once).
        dataset.transform_input, dataset.transform_input_hr, dataset.transform_output = dataset._transform()
        dataset.deterministic_eval = True
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
# Regression head helper
# -------------------------------------------------------------------------

def _reg_predict(net_reg, x01, target_size):
    """x01: (B, 3, H, W) in [0, 1]. Returns (B, 1, *target_size) height in metres
    (CHMv2Height normalizes, pads to the DINOv3 patch size and resizes internally)."""
    return net_reg(x01, target_size=tuple(target_size))


# -------------------------------------------------------------------------
# DDP helpers
# -------------------------------------------------------------------------

def _raw(module: nn.Module) -> nn.Module:
    """Unwrap a DDP-wrapped module to the underlying module (shares the same
    parameter tensors), so state_dict() keys stay unprefixed regardless of
    whether distributed training is active."""
    return module.module if isinstance(module, DDP) else module


@contextlib.contextmanager
def _maybe_no_sync(modules, skip_sync: bool):
    """Suppress DDP's gradient all-reduce on non-final micro-batches of a
    gradient-accumulation window, so gradients accumulate locally and get
    averaged across ranks exactly once, on the final micro-batch."""
    if not skip_sync:
        yield
        return
    with contextlib.ExitStack() as stack:
        for m in modules:
            stack.enter_context(m.no_sync())
        yield


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

@torch.no_grad()
def validate(
    net_sr, net_reg,
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
    range_edges=None,
) -> tuple:
    """
    Runs validation in parallel across all ranks: every rank forwards its own
    shard of a deterministic index window through its local raw (non-DDP)
    modules — no gradients, so no DDP participation is needed — and the
    R²/MAE/RMSE sufficient statistics are all-reduced (exact, and tiny compared
    to gathering every per-pixel prediction). Every rank gets the metrics back;
    only rank0 gets a figure (None elsewhere). With range_edges, the metrics
    also include per-height-range entries: a patch belongs to the range of its
    GT 75th percentile (same grouping as the training pool's quota).
    """
    # Raw (non-DDP) modules: DDP's forward hooks assume every rank calls it in
    # lockstep for gradient sync, which doesn't apply here (no backward pass,
    # independent per-rank shards).
    net_sr = _raw(net_sr)
    net_reg = _raw(net_reg)
    net_sr.eval()
    net_reg.eval()

    # Every rank computes the identical (deterministic) index window, then
    # takes a disjoint shard of it. Indices are resolved to data via a small
    # on-demand DataLoader over just this shard's Subset, so only the samples
    # this rank needs get read off disk.
    n_total = len(val_dataset)
    if full:
        indices = list(range(n_total))
    else:
        subset_size = max(1, n_total // 10)
        indices = [(val_offset + i) % n_total for i in range(subset_size)]
    local_indices = indices[global_rank::world_size] if is_distributed else indices
    local_loader = DataLoader(Subset(val_dataset, local_indices), batch_size=1,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    # Sufficient statistics: [n, sum(gt), sum(gt^2), sum(res^2), sum(|res|)]
    stats = torch.zeros(5, dtype=torch.float64, device=device)
    # Same statistics (+ patch count in column 5) per height range.
    n_groups = len(range_edges) - 1 if range_edges is not None else 0
    stats_g = torch.zeros(n_groups, 6, dtype=torch.float64, device=device)
    # Only the first n_vis_samples ranks keep one sample each, so the gather
    # below moves at most n_vis_samples images instead of one set per rank.
    collect_vis = global_rank < n_vis_samples
    vis_samples = []

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

        img_sr = net_sr(landsat, target_size=(H_hr, W_hr))
        img_sr_01 = (img_sr + 1.0) / 2.0
        pred = _reg_predict(net_reg, img_sr_01, (H_hr, W_hr))

        p = pred.flatten().double()
        g = targets.flatten().double()
        mask = torch.isfinite(g) & torch.isfinite(p)
        p, g = p[mask], g[mask]
        res = g - p
        s5 = torch.stack([
            torch.tensor(float(g.numel()), dtype=torch.float64, device=device),
            g.sum(), (g * g).sum(), (res * res).sum(), res.abs().sum(),
        ])
        stats += s5
        if range_edges is not None:
            p75 = float(np.percentile(targets.cpu().numpy(), 75))
            gi = int(np.digitize([p75], range_edges[1:-1])[0])
            stats_g[gi, :5] += s5
            stats_g[gi, 5] += 1

        if collect_vis and len(vis_samples) < 1:
            landsat_vis = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)
            vis_samples.append({
                "landsat": _to_vis_rgb(landsat_vis[0]),
                "img_sr": _to_vis_rgb(img_sr[0]),
                "img_ps": _to_vis_rgb(inputs_hr[0]),
                "pred": _to_vis_depth(pred[0]),
                "gt": _to_vis_depth(targets[0]),
            })

    net_sr.train()
    net_reg.train()

    if is_distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(stats_g, op=dist.ReduceOp.SUM)
        # NCCL has no point-to-point gather, so use all_gather_object (backed
        # by all_gather) even though only rank0 uses the result.
        gathered = [None] * world_size
        dist.all_gather_object(gathered, vis_samples)
        vis_samples = [s for g in gathered for s in g][:n_vis_samples]

    def _metrics(st):
        n, sum_g, sum_gg, ss_res, sum_abs = st
        ss_tot = sum_gg - sum_g ** 2 / max(n, 1.0)
        return {
            "r2": 1.0 - ss_res / (ss_tot + 1e-8),
            "mae": sum_abs / max(n, 1.0),
            "rmse": (ss_res / max(n, 1.0)) ** 0.5,
        }

    metrics = _metrics(stats.tolist())
    for gi, row in enumerate(stats_g.tolist()):
        if row[5] == 0:
            continue
        # np.digitize(edges[1:-1]) puts everything >= edges[-2] in the last group
        tag = f"h{range_edges[gi]}+m" if gi == n_groups - 1 else f"h{range_edges[gi]}-{range_edges[gi + 1]}m"
        metrics.update({f"{k}_{tag}": v for k, v in _metrics(row[:5]).items()})
        metrics[f"n_patches_{tag}"] = int(row[5])
    fig = _make_vis_figure(vis_samples, target_vmax) if (vis_samples and global_rank == 0) else None
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
        "sr_state": _raw(net_sr).state_dict(),
        "reg_state": _raw(net_reg).state_dict(),
        "optimizer_sr_state": optimizer_sr.state_dict(),
        "optimizer_reg_state": optimizer_reg.state_dict(),
        "lr_scheduler_sr_state": lr_scheduler_sr.state_dict() if lr_scheduler_sr else None,
        "lr_scheduler_reg_state": lr_scheduler_reg.state_dict() if lr_scheduler_reg else None,
    }, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                     lr_scheduler_sr, lr_scheduler_reg, path):
    ckpt = torch.load(path, map_location="cpu")
    _raw(net_sr).load_state_dict(ckpt["sr_state"])
    _raw(net_reg).load_state_dict(ckpt["reg_state"])
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

    parser = argparse.ArgumentParser(description="SwT-style SR+Reg Baseline Training")
    parser.add_argument("--config", type=str, default="config/baseline_swt-lumi.yaml")
    parser.add_argument("--resume_run", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--exit_after", type=int, default=-1, help="Exit after X minutes.")
    parser.add_argument("--add_datetime_prefix", action="store_true")
    args = parser.parse_args()

    # ---- Distributed setup (same SLURM env reading as train_sr_fm_refiner.py) ----
    # global_rank/world_size come from SLURM_PROCID/SLURM_NTASKS (srun sets one
    # task per GPU); local_rank (SLURM_LOCALID) selects the GPU on this node.
    global_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size  = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local_rank  = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_distributed  = world_size > 1
    is_main_process = global_rank == 0

    # Every rank builds its own full LazyPatchDataset, whose __init__ is chatty
    # (print() + tqdm bars). Only rank0's copy of the output is useful — silence
    # print() and disable tqdm rendering everywhere else.
    if not is_main_process:
        sys.stdout = open(os.devnull, "w")
        import functools
        import src.util.ps_lazydataset as _pld
        _pld.tqdm = functools.partial(_pld.tqdm, disable=True)

    if is_distributed:
        # MASTER_ADDR should be set by the launch script for multi-node runs;
        # defaults to localhost for a single-node, multi-GPU srun launch.
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
    # stderr, not stdout: non-rank0 stdout is redirected to devnull above, and
    # no logging handler exists yet. Makes every rank's GPU binding visible.
    print(
        f"[rank {global_rank}/{world_size}] SLURM_LOCALID={local_rank} "
        f"visible_gpus={n_visible} -> using {device} ({gpu_name})",
        file=sys.stderr, flush=True,
    )

    def _barrier():
        # device_ids avoids NCCL's "devices used by this process are currently
        # unknown" warning on every barrier() call.
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)

    if is_distributed:
        # Warm up every rank-pair connection on a cheap call before DDP's
        # construction broadcast (see script/depth/train_sr_fm_refiner.sh).
        _barrier()

    # ---- Config ----
    if args.resume_run is not None:
        out_dir_run = os.path.dirname(os.path.dirname(args.resume_run))
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
        job_name = os.path.basename(out_dir_run)
    else:
        cfg = recursive_load_config(args.config)

        # Same step scaling as train_sr_fm_refiner.py (so v12-lumi's max_iter /
        # period values carry over unchanged): max_iter and the log / save /
        # validation periods in the config are in units of a reference run with
        # _REFERENCE_ACCUM_STEPS accumulation steps and a single rank. A step
        # here consumes max_train_batch_size * accumulation_steps * world_size
        # samples, so divide those knobs by (accumulation_steps * world_size) /
        # _REFERENCE_ACCUM_STEPS. Applied once, on a fresh run — a resumed run
        # reloads the already-scaled config.yaml.
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
                    f"[scale] (accumulation_steps={_accum_steps} x world_size={world_size}) / {_REFERENCE_ACCUM_STEPS} = "
                    f"{total_scale}: scaled max_iter/validation_period/save_period/log_period "
                    f"down by 1/{total_scale} (now {cfg.max_iter}/{cfg.trainer.validation_period}/"
                    f"{cfg.trainer.save_period}/{cfg.trainer.log_period}) so this run covers "
                    f"the same total data as a reference run (same convention as train_sr_fm_refiner.py)."
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

    logging.info(f"device = {device}")

    # ---- Seeding (reproducibility) ----
    base_seed = int(cfg.dataloader.seed)
    seed_all(base_seed)
    # Deterministic cuDNN/MIOpen convs are opt-in here: on LUMI (MIOpen) the
    # deterministic backward solver for CHMv2's DPT head at 240/480 px is ~95x
    # slower (13.4s vs 0.14s per backward at 240px, batch 2; measured), which
    # made a step take 15s/54s for 240/480 patches. Seeds are still fixed.
    cudnn_deterministic = bool(cfg.dataloader.get("cudnn_deterministic", False))
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    if is_main_process:
        logging.info(f"Global seed = {base_seed} (cudnn deterministic={cudnn_deterministic}, benchmark=False)")

    def _worker_init_fn(worker_id: int):
        # DataLoader auto-reseeds torch's RNG per worker, but NOT Python's
        # `random` or numpy — LazyPatchDataset uses `random.choice` for patch
        # size selection.
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # ---- Data ----
    cfg_data = cfg.dataset
    # effective_batch_size is optional and defaults to max_train_batch_size
    # (no local accumulation); see baseline_swt-lumi.yaml.
    eff_bs = cfg.dataloader.get("effective_batch_size", cfg.dataloader.max_train_batch_size)
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

    # Spatial split (make_spatial_split.py, on finer_lumi_nd): patch_coord_path
    # is the *train* list, val/test lists are given separately; they are
    # disjoint from each other in space, and val/test are non-overlapping,
    # non-augmented grids. Without val_coord_path: legacy random patch-level
    # split of one pool.
    spatial_split = cfg_data.get("val_coord_path") is not None
    test_dataset, range_edges = None, None
    if spatial_split:
        train_coords = base_dataset.patch_coords
        val_coords = _load_coords(cfg_data.val_coord_path)
        test_coords = _load_coords(cfg_data.test_coord_path)
        # Fixed shuffle so the rotating 10% val window during training mixes
        # tiles/blocks (the list is otherwise ordered by block).
        random.Random(cfg_data.split_seed).shuffle(val_coords)
        range_edges = list(cfg_data.target_range_edges)
        if is_main_process:
            logging.info(f"Patches: train={len(train_coords)} val={len(val_coords)} test={len(test_coords)}")
        test_dataset = _make_dataset(base_dataset, test_coords, 'test', 1, deterministic_eval=True)
    else:
        all_coords = base_dataset.patch_coords
        train_coords, val_coords, _ = _split_coords(
            all_coords, cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed
        )
    train_dataset = _make_dataset(base_dataset, train_coords, 'train', cfg.dataloader.max_train_batch_size)
    val_dataset = _make_dataset(base_dataset, val_coords, 'val', 1, deterministic_eval=spatial_split)

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
    # No val_loader: validate() builds a small on-demand DataLoader over just
    # this rank's shard of the validation indices (materializing the full val
    # set on all 64 ranks would be a 64-way concurrent I/O storm on Lustre).

    n_landsat_bands = len(cfg_data.selected_bands)

    # ---- net_sr (SwinIR) ----
    net_sr = build_sr_module(OmegaConf.to_container(cfg.sr_module, resolve=True)).to(device)
    # Only rank 0 reads the checkpoint — DDP's construction broadcasts rank 0's
    # parameters to every other rank, so having all ranks read this file from
    # shared Lustre is redundant (and slow at high rank counts).
    sr_pretrained_path = cfg.model.get("sr_pretrained_path", None)
    if sr_pretrained_path and os.path.exists(sr_pretrained_path):
        if is_main_process:
            ckpt_sr = torch.load(sr_pretrained_path, map_location="cpu")
            key = cfg.model.get("sr_pretrained_key", None)
            state = ckpt_sr[key] if key and key in ckpt_sr else ckpt_sr
            missing, unexpected = net_sr.swinir.load_state_dict(state, strict=False)
            logging.info(f"SwinIR pretrained loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    elif is_main_process:
        logging.info("No SwinIR pretrained weights found; training from scratch.")
    net_sr.train()

    # ---- net_reg (CHMv2: DINOv3 encoder frozen, neck + DPT head trainable) ----
    # Input is the SR output (pseudo-PS), so the mean/std CHMv2Height normalizes
    # with are the PS statistics, reordered to match selected_bands_hr (the
    # channel order the dataloader/SR emit). out_in_scale_factor=1: the SR
    # module already upsamples to the label resolution, so no upsample_head.
    with open(cfg_data.input_hr_stats_file) as f:
        ps_stats = json.load(f)[str(cfg_data.year)]
    net_reg = CHMv2Height(
        model_id=cfg.model.chmv2_model_id,
        mean=[ps_stats["mean"][i] for i in cfg_data.selected_bands_hr],
        std=[ps_stats["std"][i] for i in cfg_data.selected_bands_hr],
        pretrained=cfg.model.get("pretrained", True),
        freeze_backbone=cfg.model.get("freeze_backbone", True),
        out_in_scale_factor=1,
    ).to(device)
    net_reg.train()  # CHMv2Height.train() keeps the frozen encoder in eval mode
    # DDP with  raises if any trainable param gets
    # no gradient. Find such params with a dummy backward (static graph, so
    # identical on every rank) and freeze them: they can't affect the output.
    unused = net_reg.find_unused_trainable_params()
    for n, p in net_reg.named_parameters():
        if n in unused:
            p.requires_grad_(False)
    if is_main_process:
        n_train = sum(p.numel() for p in net_reg.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in net_reg.parameters())
        logging.info(f"net_reg (CHMv2): froze {len(unused)} trainable params that receive no "
                     f"gradient: {unused}; trainable {n_train / 1e6:.1f}M / {n_all / 1e6:.1f}M")

    # ---- DDP wrapping ----
    # net_sr_raw / net_reg_raw keep referring to the underlying modules (same
    # parameter tensors) for optimizers, grad clipping and checkpointing.
    # DDP only syncs parameters with requires_grad=True, so net_reg's frozen
    # backbone costs no communication.
    net_sr_raw  = net_sr
    net_reg_raw = net_reg
    if is_distributed:
        ddp_ids = [device.index] if device.type == "cuda" else None
        out_dev = ddp_ids[0] if ddp_ids else None
        net_sr  = DDP(net_sr_raw,  device_ids=ddp_ids, output_device=out_dev)
        net_reg = DDP(net_reg_raw, device_ids=ddp_ids, output_device=out_dev,
                      find_unused_parameters=False)

    # ---- Losses ----
    pixel_weight = cfg.sr_loss.pixel_weight
    task_weight = cfg.sr_loss.task_weight
    huber_delta = cfg.trainer.huber_delta
    loss_huber = torch.nn.HuberLoss(delta=huber_delta)
    target_vmax = float(cfg.trainer.get("target_vis_vmax", 30.0))

    # ---- Optimizers ----
    reg_params = [p for p in net_reg_raw.parameters() if p.requires_grad]
    optimizer_sr = torch.optim.AdamW(
        net_sr_raw.parameters(), lr=cfg.optimizer.lr_sr, weight_decay=cfg.optimizer.weight_decay_sr
    )
    optimizer_reg = torch.optim.AdamW(
        reg_params, lr=cfg.optimizer.lr_reg, weight_decay=cfg.optimizer.weight_decay_reg,
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
        if is_main_process:
            logging.info(f"Resumed from step {start_step}")

    # ---- Training loop ----
    t_end = t_start + timedelta(minutes=args.exit_after) if args.exit_after > 0 else None

    step = start_step
    best_r2 = -1e8
    val_offset = 0
    val_subset_size = max(1, len(val_dataset) // 10)

    loss_pix_accum = 0.0
    loss_reg_accum = 0.0
    micro_step = 0

    if is_main_process:
        logging.info("Starting SwT-style baseline training (SR + regression, no diffusion refiner).")
    pbar = tqdm(total=cfg.max_iter, initial=step, desc="Training", dynamic_ncols=True,
                disable=not is_main_process)

    optimizer_sr.zero_grad()
    optimizer_reg.zero_grad()
    for epoch in range(cfg.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= cfg.max_iter:
                break
            if t_end is not None and datetime.now() >= t_end:
                if is_main_process:
                    logging.info("Exit after time limit reached.")
                    save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                                    lr_scheduler_sr, lr_scheduler_reg, step, out_dir_ckpt, "latest")
                if is_distributed:
                    _barrier()
                    dist.destroy_process_group()
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

            # Only the final micro-batch of a window should let DDP all-reduce
            # gradients; earlier ones accumulate locally (see _maybe_no_sync).
            is_last_micro = (micro_step == accumulation_steps - 1)
            skip_sync = is_distributed and not is_last_micro

            # ============================================================
            # Joint step: SR + net_reg (neck+head) updated together
            # ============================================================
            if micro_step == 0:
                optimizer_sr.zero_grad()
                optimizer_reg.zero_grad()

            with _maybe_no_sync([net_sr, net_reg] if is_distributed else [], skip_sync):
                img_sr = net_sr(landsat, target_size=(H_hr, W_hr))
                l_pix = F.l1_loss(img_sr, inputs_hr)

                img_sr_01 = (img_sr + 1.0) / 2.0
                pred = _reg_predict(net_reg, img_sr_01, (H_hr, W_hr))
                l_reg = loss_huber(pred, targets)

                l_total = pixel_weight * l_pix + task_weight * l_reg
                (l_total / accumulation_steps).backward()

            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(net_sr_raw.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(reg_params, max_norm=1.0)
                optimizer_sr.step()
                optimizer_reg.step()

            loss_pix_accum += l_pix.item()
            loss_reg_accum += l_reg.item()

            if not is_last_micro:
                micro_step += 1
                continue
            micro_step = 0

            if lr_scheduler_sr is not None:
                lr_scheduler_sr.step()
            if lr_scheduler_reg is not None:
                lr_scheduler_reg.step()

            step += 1
            pbar.update(1)

            # ---- Logging (rank0 only) ----
            # loss_*_accum accumulates every micro-batch, so the averaging
            # window is log_period real steps' worth of micro-batches.
            if is_main_process and step % cfg.trainer.log_period == 0:
                log_n = cfg.trainer.log_period * accumulation_steps
                lr_sr = optimizer_sr.param_groups[0]["lr"]
                lr_reg = optimizer_reg.param_groups[0]["lr"]
                log_dict = {
                    "lr/sr": lr_sr, "lr/reg": lr_reg,
                    "train/l_pix": loss_pix_accum / log_n,
                    "train/l_reg": loss_reg_accum / log_n,
                }
                logging.info(
                    f"[step {step}] "
                    f"l_pix={log_dict['train/l_pix']:.5f} "
                    f"l_reg={log_dict['train/l_reg']:.5f}"
                )
                pbar.set_postfix(r2=f"{best_r2:.4f}")
                wandb.log(log_dict, step=step)
                tb_logger.log_dict(log_dict, global_step=step)
                loss_pix_accum = loss_reg_accum = 0.0
            elif step % cfg.trainer.log_period == 0:
                loss_pix_accum = loss_reg_accum = 0.0

            # ---- Validation ----
            # Every rank participates (sharded, parallel inference — see
            # validate()'s docstring) and gets identical metrics back, so the
            # best-checkpoint decision is consistent; only rank0 writes.
            if step % cfg.trainer.validation_period == 0:
                metrics, fig = validate(
                    net_sr, net_reg, val_dataset, device,
                    n_landsat_bands=n_landsat_bands,
                    target_vmax=target_vmax,
                    val_offset=val_offset,
                    global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
                )
                # Deterministic given val_offset/val_subset_size/len(val_dataset),
                # identical on every rank — safe to update everywhere.
                val_offset = (val_offset + val_subset_size) % len(val_dataset)

                if metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    is_best = True
                else:
                    is_best = False

                if is_main_process:
                    logging.info(f"[step {step}] val: {metrics}")
                    vlog = {f"val/{k}": v for k, v in metrics.items()}
                    if fig is not None:
                        vlog["val/samples"] = wandb.Image(fig)
                        plt.close(fig)
                    wandb.log(vlog, step=step)
                    tb_logger.log_dict({f"val/{k}": v for k, v in metrics.items()}, global_step=step)

                    if is_best:
                        pbar.set_postfix(r2=f"{best_r2:.4f}")
                        save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                                        lr_scheduler_sr, lr_scheduler_reg, step, out_dir_ckpt, "best")
                        logging.info(f"New best R²={best_r2:.4f} at step {step}")

                if is_distributed:
                    # rank0's checkpoint write above must finish before other
                    # ranks resume training and possibly overwrite "latest".
                    _barrier()

            # ---- Periodic checkpoint (rank0 only) ----
            if is_main_process and step % cfg.trainer.save_period == 0:
                save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                                lr_scheduler_sr, lr_scheduler_reg, step, out_dir_ckpt, "latest")

        if step >= cfg.max_iter:
            break

    pbar.close()

    # ---- Final full validation, from the best checkpoint ----
    # Every rank must load the same best checkpoint (not just rank0) since the
    # sharded validate() below combines statistics across ranks.
    best_ckpt_path = os.path.join(out_dir_ckpt, "best.pth")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                        lr_scheduler_sr, lr_scheduler_reg, best_ckpt_path)
        if is_main_process:
            logging.info(f"Loaded best checkpoint from {best_ckpt_path} for final validation")
    elif is_main_process:
        logging.warning("Best checkpoint not found, using final model weights for full validation")

    if is_main_process:
        logging.info("Running full validation on entire val set...")
    final_metrics, final_fig = validate(
        net_sr, net_reg, val_dataset, device,
        n_landsat_bands=n_landsat_bands,
        target_vmax=target_vmax,
        full=True,
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

        save_checkpoint(net_sr, net_reg, optimizer_sr, optimizer_reg,
                        lr_scheduler_sr, lr_scheduler_reg, step, out_dir_ckpt, "final")
        logging.info(f"Training finished at step {step}. Best val R²={best_r2:.4f}")

    # Held-out test set: the number to report. Never used for model selection
    # (best.pth is picked by val R2). validate() returns metrics on every rank.
    if test_dataset is not None:
        if is_main_process:
            logging.info("Running full evaluation on the test set...")
        test_metrics, test_fig = validate(
            net_sr, net_reg, test_dataset, device,
            n_landsat_bands=n_landsat_bands,
            target_vmax=target_vmax,
            full=True,
            global_rank=global_rank, world_size=world_size, is_distributed=is_distributed,
            num_workers=cfg_data.workers,
            range_edges=range_edges,
        )
        if is_main_process:
            logging.info(f"Final TEST metrics: {test_metrics}")
            tlog = {f"test_final/{k}": v for k, v in test_metrics.items()}
            if test_fig is not None:
                tlog["test_final/samples"] = wandb.Image(test_fig)
                plt.close(test_fig)
            wandb.log(tlog, step=step)
            with open(os.path.join(out_dir_run, "test_metrics.json"), "w") as f:
                json.dump({"step": step, "n_test_patches": len(test_dataset), "metrics": test_metrics}, f, indent=1)

    if is_distributed:
        _barrier()
        dist.destroy_process_group()
