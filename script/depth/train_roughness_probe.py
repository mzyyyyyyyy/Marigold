"""
Stage 0 of the latent-guided generative refiner: is local canopy roughness
predictable from Landsat beyond what the coarse height mu already explains?

Frozen: baseline_chmv2-lumi-nd-up8-h120 (DINOv3 backbone + DPT head) -> mu.
Target, from the GT height at label resolution, averaged over each ViT token
cell (cell = 16 / input_scale label px, ~30 m for up8):
  HF = GT - G_hf(GT)
  r  = log(1 + sqrt(G_rough(HF^2)))
Three heads, trained jointly (separate parameters, L1 on r over valid cells):
  L-head : frozen DINOv3 token features of the (upsampled) Landsat input,
           i.e. the same features mu is decoded from
  PS-head: frozen DINOv3 token features of PS (reference: what PS adds)
  mu-head: small CNN on mu alone (control: what mu already explains)
Each head is model-selected on its own val R2. Reported on test: R2(r) per
head over all valid cells and over forest cells (cell-mean mu > forest_mu),
the gain L - mu, and corr(r(mu), r) (how much roughness mu itself carries).

Usage (multi-GPU / multi-node), with script/depth/train_baseline_chmv2.sh:
  sbatch --export=ALL,SCRIPT=$(pwd)/script/depth/train_roughness_probe.py,\
CONFIG=config/roughness_probe-lumi-nd-up8.yaml script/depth/train_baseline_chmv2.sh
"""

import copy
import json
import logging
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from depthfm.chmv2 import load_chmv2_baseline
from src.util.config_util import recursive_load_config
from src.util.logging_util import config_logging, init_wandb, tb_logger
from src.util.ps_lazydataset import LazyPatchDataset
from train_baseline_chmv2 import _load_coords, _make_dataset, _raw

HEADS = ("L", "PS", "mu")


# -------------------------------------------------------------------------
# Roughness target
# -------------------------------------------------------------------------

def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, 1, H, W), zero padding."""
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))


def _masked_blur(x, m, sigma):
    """Normalized convolution: ignores invalid pixels and the zero padding."""
    return _blur(x * m, sigma) / _blur(m, sigma).clamp(min=1e-6)


@torch.no_grad()
def roughness_target(h: torch.Tensor, hf_sigma: float, rough_sigma: float, cell: int):
    """h: (B, 1, H, W) metres (NaN = invalid). Returns r (B, 1, H/cell, W/cell)
    and a bool mask of cells with > 50% valid pixels."""
    m = torch.isfinite(h).float()
    g = torch.nan_to_num(h)
    hf = g - _masked_blur(g, m, hf_sigma)
    rough = _masked_blur(hf ** 2, m, rough_sigma).clamp(min=0).sqrt()
    num = F.avg_pool2d(rough * m, cell)
    den = F.avg_pool2d(m, cell)
    return torch.log1p(num / den.clamp(min=1e-6)), den > 0.5


# -------------------------------------------------------------------------
# Frozen feature extractor and probe heads
# -------------------------------------------------------------------------

class FrozenFeatures(nn.Module):
    """mu plus the DINOv3 token features (all 4 DPT-tapped layers) of Landsat
    and of PS, from the frozen up8-h120 CHMv2. The Landsat backbone pass is
    the one that also produces mu (captured with a forward hook); PS gets its
    own backbone pass, resized by the same input_scale so both feature maps
    sit on the same token grid."""

    def __init__(self, coarse: nn.Module, ps_mean: list, ps_std: list):
        super().__init__()
        self.coarse = coarse
        self.register_buffer("ps_mean", torch.tensor(ps_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("ps_std", torch.tensor(ps_std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self._feats = None
        coarse.model.backbone.register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        self._feats = list(output.feature_maps)

    @torch.no_grad()
    def forward(self, landsat, ps, target_size):
        """landsat, ps in [-1, 1]; ps at target_size. Returns mu (B,1,*target_size)
        and two lists of (B, C, h, w) token feature maps."""
        mu = self.coarse((landsat + 1.0) / 2.0, target_size=target_size)
        feats_l = self._feats
        x = ((ps + 1.0) / 2.0 - self.ps_mean) / self.ps_std.clamp(min=1e-8)
        x = F.interpolate(x, size=tuple(s * self.coarse.input_scale for s in target_size),
                          mode="bilinear", align_corners=False)
        self.coarse.model.backbone(x)
        return mu, feats_l, self._feats


def _body(width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.GELU(), nn.Conv2d(width, width, 3, padding=1),
        nn.GELU(), nn.Conv2d(width, width, 3, padding=1),
        nn.GELU(), nn.Conv2d(width, 1, 1),
    )


class FeatHead(nn.Module):
    """Concatenated token features -> per-layer GroupNorm -> 1x1 -> two 3x3 -> r."""

    def __init__(self, ch: int, n_layers: int, width: int):
        super().__init__()
        self.norm = nn.GroupNorm(n_layers, ch * n_layers)
        self.proj = nn.Conv2d(ch * n_layers, width, 1)
        self.body = _body(width)

    def forward(self, feats):
        return self.body(self.proj(self.norm(torch.cat(feats, 1))))


class MuHead(nn.Module):
    """CNN on mu (label resolution) down to the cell grid, then the same body."""

    def __init__(self, cell: int, width: int):
        super().__init__()
        layers, ch = [], 1
        for i in range(int(math.log2(cell))):
            out = min(width, 64 * 2 ** i)
            layers += [nn.Conv2d(ch, out, 3, stride=2, padding=1), nn.GELU(),
                       nn.Conv2d(out, out, 3, padding=1), nn.GELU()]
            ch = out
        self.down = nn.Sequential(*layers)
        self.proj = nn.Conv2d(ch, width, 1)
        self.body = _body(width)

    def forward(self, mu):
        return self.body(self.proj(self.down(mu / 30.0)))


class RoughnessProbe(nn.Module):
    def __init__(self, ch: int, n_layers: int, cell: int, width: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            "L": FeatHead(ch, n_layers, width),
            "PS": FeatHead(ch, n_layers, width),
            "mu": MuHead(cell, width),
        })

    def forward(self, feats_l, feats_ps, mu):
        return {"L": self.heads["L"](feats_l), "PS": self.heads["PS"](feats_ps), "mu": self.heads["mu"](mu)}


# -------------------------------------------------------------------------
# Batch / evaluation
# -------------------------------------------------------------------------

def _prepare(batch, frozen, device, n_landsat_bands, cfg_p, cell):
    # Squeeze each tensor on its own: LazyPatchDataset .squeeze()s the PS batch,
    # so at batch_size 1 (eval) PS arrives 4-D while Landsat / GT are 5-D.
    inputs_lr, inputs_hr, targets = (t.squeeze(0) if t.dim() == 5 else t for t in batch)
    landsat = inputs_lr[:, :n_landsat_bands].to(device)
    ps = inputs_hr[:, :3].to(device)
    targets = targets.to(device)
    target_size = tuple(targets.shape[-2:])
    mu, feats_l, feats_ps = frozen(landsat, ps, target_size)
    r, valid = roughness_target(targets, cfg_p.hf_sigma, cfg_p.rough_sigma, cell)
    return dict(landsat=landsat, ps=ps, targets=targets, mu=mu, feats_l=feats_l, feats_ps=feats_ps,
                r=r, valid=valid)


def _summarize(stats: torch.Tensor) -> dict:
    out = {}
    for si, tag in enumerate(("all", "forest")):
        n, sy, syy, *ssres, sx, sxx, sxy = stats[si].tolist()
        n = max(n, 1.0)
        sstot = max(syy - sy * sy / n, 1e-12)
        out[f"{tag}/n_cells"] = int(n)
        out[f"{tag}/mean_r"] = sy / n
        for h, ss in zip(HEADS, ssres):
            out[f"{tag}/r2_{h}"] = 1.0 - ss / sstot
        out[f"{tag}/gain_L_over_mu"] = out[f"{tag}/r2_L"] - out[f"{tag}/r2_mu"]
        out[f"{tag}/gain_PS_over_mu"] = out[f"{tag}/r2_PS"] - out[f"{tag}/r2_mu"]
        vx = max(sxx - sx * sx / n, 1e-12)
        out[f"{tag}/corr_r(mu)"] = (sxy - sx * sy / n) / math.sqrt(vx * sstot)
    return out


@torch.no_grad()
def evaluate(probe, frozen, dataset, device, n_landsat_bands, cfg_p, cell, full=False, offset=0,
             global_rank=0, world_size=1, is_distributed=False, num_workers=0):
    """Sharded evaluation; the summed statistics are all-reduced, so every
    rank returns the same metrics. Subset mode = rotating 10% window."""
    raw = _raw(probe)
    raw.eval()
    n_total = len(dataset)
    if full:
        indices = list(range(n_total))
    else:
        indices = [(offset + i) % n_total for i in range(max(1, n_total // 10))]
    local = indices[global_rank::world_size] if is_distributed else indices
    loader = DataLoader(Subset(dataset, local), batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    # per subset (all, forest): n, sum y, sum y^2, SSres x3 heads, sum x, sum x^2, sum xy (x = r(mu))
    stats = torch.zeros(2, 9, dtype=torch.float64, device=device)
    for batch in tqdm(loader, desc="Eval", leave=False, disable=(global_rank != 0)):
        b = _prepare(batch, frozen, device, n_landsat_bands, cfg_p, cell)
        preds = raw(b["feats_l"], b["feats_ps"], b["mu"])
        r_mu, _ = roughness_target(b["mu"], cfg_p.hf_sigma, cfg_p.rough_sigma, cell)
        forest = F.avg_pool2d(b["mu"], cell) > cfg_p.forest_mu
        for si, sel in enumerate((b["valid"], b["valid"] & forest)):
            y, x = b["r"][sel].double(), r_mu[sel].double()
            stats[si, 0] += y.numel()
            stats[si, 1] += y.sum()
            stats[si, 2] += (y * y).sum()
            for hi, h in enumerate(HEADS):
                stats[si, 3 + hi] += ((preds[h][sel].double() - y) ** 2).sum()
            stats[si, 6] += x.sum()
            stats[si, 7] += (x * x).sum()
            stats[si, 8] += (x * y).sum()
    raw.train()
    if is_distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return _summarize(stats)


@torch.no_grad()
def save_vis(probe, frozen, dataset, device, n_landsat_bands, cfg_p, cell, path, n=4):
    raw = _raw(probe)
    raw.eval()
    idx = [int(i * len(dataset) / n) for i in range(n)]
    rows = []
    for i in idx:
        batch = [t.unsqueeze(0) if torch.is_tensor(t) else t for t in dataset[i]]
        b = _prepare(batch, frozen, device, n_landsat_bands, cfg_p, cell)
        preds = raw(b["feats_l"], b["feats_ps"], b["mu"])
        rows.append([b["targets"][0, 0], b["mu"][0, 0], b["r"][0, 0]] + [preds[h][0, 0] for h in HEADS])
    titles = ["GT height", "mu (up8-h120)", "r (GT)", "r L-head", "r PS-head", "r mu-head"]
    fig, axes = plt.subplots(n, 6, figsize=(24, 4 * n))
    for ri, row in enumerate(rows):
        rmax = float(row[2].max())
        for ci, img in enumerate(row):
            ax = axes[ri, ci]
            kw = dict(vmin=0, vmax=30) if ci < 2 else dict(vmin=0, vmax=rmax)
            im = ax.imshow(img.cpu().float().numpy(), cmap="plasma", **kw)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if ri == 0:
                ax.set_title(titles[ci])
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="Roughness probe (stage 0)")
    parser.add_argument("--config", type=str, default="config/roughness_probe-lumi-nd-up8.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # ---- Distributed setup (as train_baseline_chmv2.py) ----
    global_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_distributed = world_size > 1
    is_main_process = global_rank == 0

    if not is_main_process:
        sys.stdout = open(os.devnull, "w")
        import functools
        import src.util.ps_lazydataset as _pld
        _pld.tqdm = functools.partial(_pld.tqdm, disable=True)

    if is_distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size,
                                timeout=timedelta(minutes=2))

    device_index = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    def _barrier():
        dist.barrier(device_ids=[device.index])

    if is_distributed:
        _barrier()

    # ---- Config (same step scaling as train_baseline_chmv2.py) ----
    cfg = recursive_load_config(args.config)
    _REFERENCE_ACCUM_STEPS = 4
    _eff_bs = cfg.dataloader.get("effective_batch_size", cfg.dataloader.max_train_batch_size)
    assert _eff_bs == cfg.dataloader.max_train_batch_size, "gradient accumulation not supported"
    total_scale = max(1, world_size // _REFERENCE_ACCUM_STEPS)
    if total_scale > 1:
        cfg.max_iter = max(1, cfg.max_iter // total_scale)
        cfg.trainer.validation_period = max(1, cfg.trainer.validation_period // total_scale)
        cfg.trainer.log_period = max(1, cfg.trainer.log_period // total_scale)

    job_name = os.path.basename(args.config).split(".")[0]
    out_dir_run = os.path.join(args.output_dir or "./output", job_name)
    if is_main_process:
        os.makedirs(out_dir_run, exist_ok=False)
        config_logging(cfg.logging, out_dir=out_dir_run)
        from omegaconf import OmegaConf
        with open(os.path.join(out_dir_run, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
        logging.info(f"[scale] world_size={world_size} / {_REFERENCE_ACCUM_STEPS} = {total_scale}: "
                     f"max_iter/validation_period/log_period = {cfg.max_iter}/"
                     f"{cfg.trainer.validation_period}/{cfg.trainer.log_period}")
        if not args.no_wandb:
            init_wandb(enable=True, config=dict(cfg), name=job_name, mode="online", dir=out_dir_run,
                       **{k: v for k, v in cfg.wandb.items() if k != "name"})
        else:
            init_wandb(enable=False)
        tb_logger.set_dir(os.path.join(out_dir_run, "tensorboard"))
    else:
        logging.basicConfig(level=logging.INFO)
    if is_distributed:
        _barrier()

    base_seed = int(cfg.dataloader.seed)

    def _worker_init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # ---- Data (same dataset / split as train_baseline_chmv2.py) ----
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
    train_coords = base_dataset.patch_coords
    val_coords = _load_coords(cfg_data.val_coord_path)
    test_coords = _load_coords(cfg_data.test_coord_path)
    random.Random(cfg_data.split_seed).shuffle(val_coords)
    if is_main_process:
        logging.info(f"Patches: train={len(train_coords)} val={len(val_coords)} test={len(test_coords)}")
    train_dataset = _make_dataset(base_dataset, train_coords, 'train', cfg.dataloader.max_train_batch_size)
    val_dataset = _make_dataset(base_dataset, val_coords, 'val', 1)
    test_dataset = _make_dataset(base_dataset, test_coords, 'test', 1)
    train_sampler = (DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank,
                                        shuffle=True, seed=base_seed) if is_distributed else None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=cfg_data.workers, pin_memory=True,
                              worker_init_fn=_worker_init_fn,
                              # Fork the workers once, before this process reads any raster
                              # itself (evaluation runs with num_workers=0 in this process).
                              # Workers forked after that segfaulted in JP2 decoding on every
                              # rank (smoke_probe 23222994 / 23223151).
                              persistent_workers=True,
                              generator=torch.Generator().manual_seed(base_seed))

    # ---- Frozen mu / features ----
    n_landsat_bands = len(cfg_data.selected_bands)
    with open(cfg_data.input_stats_file) as f:
        input_stats = json.load(f)[str(cfg_data.year)]
    with open(cfg_data.input_hr_stats_file) as f:
        ps_stats = json.load(f)[str(cfg_data.year)]
    cfg_p = cfg.probe
    coarse = load_chmv2_baseline(
        cfg.model.chmv2_model_id, cfg_p.coarse_ckpt, input_stats["mean"], input_stats["std"],
        out_in_scale_factor=cfg.model.get("out_in_scale_factor", 1),
        upsample_input=cfg.model.get("upsample_input_to_target", False),
        input_scale=cfg.model.get("input_scale", 1),
    )
    frozen = FrozenFeatures(
        coarse,
        ps_mean=[ps_stats["mean"][i] for i in cfg_data.selected_bands_hr],
        ps_std=[ps_stats["std"][i] for i in cfg_data.selected_bands_hr],
    ).to(device).eval()
    cell = coarse.PATCH // coarse.input_scale  # label px per ViT token
    assert all(s % cell == 0 for s in cfg_data.patch_sizes), (cfg_data.patch_sizes, cell)

    # Feature width / number of tapped layers from a dummy pass.
    _, _f, _ = frozen(torch.zeros(1, 3, 4, 4, device=device), torch.zeros(1, 3, 16, 16, device=device), (16, 16))
    n_layers, feat_ch = len(_f), _f[0].shape[1]

    probe = RoughnessProbe(feat_ch, n_layers, cell, int(cfg_p.head_width)).to(device)
    trainable_params = list(probe.parameters())
    if is_main_process:
        logging.info(f"cell={cell} label px ({cell * 3.75:.0f} m N-S), {n_layers} layers x {feat_ch} ch; "
                     f"probe params {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    if is_distributed:
        probe = DDP(probe, device_ids=[device.index], output_device=device.index)

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_iter, eta_min=cfg.lr_scheduler.eta_min)

    eval_kw = dict(n_landsat_bands=n_landsat_bands, cfg_p=cfg_p, cell=cell, global_rank=global_rank,
                   world_size=world_size, is_distributed=is_distributed)

    # ---- Training ----
    step, val_offset = 0, 0
    val_subset_size = max(1, len(val_dataset) // 10)
    best = {h: (-1e8, None) for h in HEADS}
    pbar = tqdm(total=cfg.max_iter, desc="Training", dynamic_ncols=True, disable=not is_main_process)
    for epoch in range(cfg.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= cfg.max_iter:
                break
            b = _prepare(batch, frozen, device, n_landsat_bands, cfg_p, cell)
            preds = probe(b["feats_l"], b["feats_ps"], b["mu"])
            v = b["valid"].float()
            losses = {h: ((preds[h] - b["r"]).abs() * v).sum() / v.sum().clamp(min=1) for h in HEADS}
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            step += 1
            pbar.update(1)

            if is_main_process and step % cfg.trainer.log_period == 0:
                log_dict = {f"train/l1_{h}": l.item() for h, l in losses.items()}
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
                logging.info(f"[step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in log_dict.items()))
                wandb.log(log_dict, step=step)
                tb_logger.log_dict(log_dict, global_step=step)

            if step % cfg.trainer.validation_period == 0:
                m = evaluate(probe, frozen, val_dataset, device, offset=val_offset, **eval_kw)
                val_offset = (val_offset + val_subset_size) % len(val_dataset)
                for h in HEADS:  # identical on every rank (all-reduced stats)
                    if m[f"all/r2_{h}"] > best[h][0]:
                        best[h] = (m[f"all/r2_{h}"], copy.deepcopy(_raw(probe).heads[h].state_dict()))
                if is_main_process:
                    logging.info(f"[step {step}] val: {m}")
                    wandb.log({f"val/{k}": v for k, v in m.items()}, step=step)
                    tb_logger.log_dict({f"val/{k}": v for k, v in m.items()}, global_step=step)
                if is_distributed:
                    _barrier()
        if step >= cfg.max_iter:
            break
    pbar.close()

    # ---- Final: each head at its own best val R2, full test ----
    for h in HEADS:
        if best[h][1] is not None:
            _raw(probe).heads[h].load_state_dict(best[h][1])
    if is_main_process:
        logging.info(f"Best val R2 per head (10% subsets): { {h: round(best[h][0], 4) for h in HEADS} }")
    # Test only (model selection already used the rotating val subsets).
    # num_workers=0: no fork after this process has read rasters (see train_loader).
    test_m = evaluate(probe, frozen, test_dataset, device, full=True, **eval_kw)
    if is_main_process:
        logging.info(f"Final TEST: {test_m}")
        wandb.log({f"test_final/{k}": v for k, v in test_m.items()}, step=step)
        with open(os.path.join(out_dir_run, "probe_metrics.json"), "w") as f:
            json.dump({"step": step, "cell_px": cell,
                       "best_val_subset_r2": {h: best[h][0] for h in HEADS}, "test": test_m}, f, indent=1)
        os.makedirs(os.path.join(out_dir_run, "checkpoint"), exist_ok=True)
        torch.save({"step": step, "probe_state_dict": _raw(probe).state_dict()},
                   os.path.join(out_dir_run, "checkpoint", "probe_best.pth"))
        save_vis(probe, frozen, test_dataset, device, n_landsat_bands, cfg_p, cell,
                 os.path.join(out_dir_run, "test_vis.png"))
        logging.info(f"Done at step {step}")
    if is_distributed:
        _barrier()
        dist.destroy_process_group()
