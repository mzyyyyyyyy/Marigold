"""
How different are the frozen CHMv2 baseline's errors on the data the refiner was
TRAINED on vs the data it is evaluated on? Only the baseline forward is run.

Four conditions, same number of patches each (--n_per_cond), baseline preds and GT
clipped to [p1, p99] (same convention as diag_refiner_gain.py):

  val_clean   : val split,   dataset mode "val"   (no input augmentation)  <- inference-like
  val_aug     : val split,   dataset mode "train" (brightness/contrast/gamma augmentation of the inputs)
  train_clean : train split, dataset mode "val"
  train_aug   : train split, dataset mode "train"                           <- what training sees

Why both: LazyPatchDataset builds its transform pipeline in __init__ from the mode it was
built with; the training scripts build ONE base dataset with mode "train" and derive the val
dataset by copy.copy + `.mode = "val"`, which does NOT rebuild the transforms. So the val
metrics logged during training (and the "Final val metrics") may be computed on AUGMENTED
inputs. val_aug vs val_clean measures exactly that effect; train_* vs val_* measures the
train/val generalisation gap of the baseline (what the refiner's residuals are calibrated to).

Reported per condition: R2 / MAE / RMSE, bias, std of the baseline error, and the
distribution of |error| (pixel fraction and share of the squared error per |e| bin).

Launch (one task per GPU): script/depth/diag_baseline_trainval_gap.sh
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import eval_detail_metrics as edm
from depthfm.chmv2 import load_chmv2_baseline

OUT_ROOT = edm.OUT_ROOT
ABS_EDGES = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]       # |error| bins: [0,.5) ... [12,inf)
CONDS = {                                              # name -> (split, dataset mode)
    "val_clean": ("val", "val"),
    "val_aug": ("val", "train"),
    "train_clean": ("train", "val"),
    "train_aug": ("train", "train"),
}


class Acc:
    def __init__(self, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.g = z(5)                              # n, sum_g, sum_gg, sse, sae
        self.e = z(2)                              # sum_e, sum_ee  (e = base - gt)
        self.bins = z(len(ABS_EDGES) + 1, 2)       # n, sse per |e| bin
        self.edges = torch.tensor(ABS_EDGES, device=device)

    def update(self, base, gt):
        b, g = base.flatten().double(), gt.flatten().double()
        e = b - g
        self.g += torch.stack([torch.tensor(float(g.numel()), dtype=torch.float64, device=g.device),
                               g.sum(), (g * g).sum(), (e * e).sum(), e.abs().sum()])
        self.e += torch.stack([e.sum(), (e * e).sum()])
        bi = torch.bucketize(e.abs(), self.edges, right=True)
        self.bins[:, 0].index_add_(0, bi, torch.ones_like(e))
        self.bins[:, 1].index_add_(0, bi, e * e)

    def state(self):
        return {"g": self.g.cpu(), "e": self.e.cpu(), "bins": self.bins.cpu()}


def finalize(states):
    S = {k: sum(s[k] for s in states) for k in states[0]}
    n, sg, sgg, sse, sae = S["g"].tolist()
    sst = sgg - sg ** 2 / n
    bias = S["e"][0].item() / n
    out = {"n_pixels": n, "r2": 1 - sse / sst, "mae": sae / n, "rmse": (sse / n) ** 0.5,
           "bias": bias, "err_std": (S["e"][1].item() / n - bias ** 2) ** 0.5, "gt_std": (sst / n) ** 0.5}
    labels = [f"[0,{ABS_EDGES[0]:g})"] + [f"[{ABS_EDGES[i]:g},{ABS_EDGES[i + 1]:g})" for i in range(len(ABS_EDGES) - 1)] \
             + [f"[{ABS_EDGES[-1]:g},inf)"]
    tot = S["bins"][:, 1].sum().item()
    out["abs_err_bins"] = {lab: {"pixel_frac": r[0] / n, "sse_share": r[1] / tot}
                           for lab, r in zip(labels, S["bins"].tolist())}
    big = S["bins"][4:]                                 # |e| >= 3
    out["frac_abs_err_ge3"] = big[:, 0].sum().item() / n
    out["sse_share_abs_err_ge3"] = big[:, 1].sum().item() / tot
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=f"{OUT_ROOT}/sr_fm_refiner_v12-lumi",
                    help="run whose config.yaml gives the dataset + the frozen baseline checkpoint")
    ap.add_argument("--out_dir", default=f"{OUT_ROOT}/_diag_baseline_trainval_gap")
    ap.add_argument("--n_per_cond", type=int, default=3600)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N patches per condition")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_main, dist_on = rank == 0, world > 1
    if not is_main:
        sys.stdout = open(os.devnull, "w")
        import functools
        import src.util.ps_lazydataset as _pld
        _pld.tqdm = functools.partial(_pld.tqdm, disable=True)
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    if use_cuda:
        idx = local % max(1, torch.cuda.device_count())
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = OmegaConf.load(os.path.join(args.run, "config.yaml"))
    cfg_d = cfg.dataset
    in_stats = edm._load_target_stats(cfg_d.input_stats_file, cfg_d.year)
    chm = load_chmv2_baseline(
        model_dir=cfg.model.chmv2_model_id, ckpt_path=cfg.model.chmv2_ckpt_path,
        mean=in_stats["mean"], std=in_stats["std"],
        out_in_scale_factor=cfg.model.chmv2_out_in_scale_factor,
    ).to(device).eval()
    stats = edm._load_target_stats(cfg_d.target_stats_file, cfg_d.year)
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    n_ls = len(cfg_d.selected_bands)
    if is_main:
        logging.info(f"baseline: {cfg.model.chmv2_ckpt_path}")

    states = {}
    for name in args.conds.split(","):
        split, mode = CONDS[name]
        # the augmentation (mode "train") is random: fix all RNGs, different per rank
        random.seed(1234 + rank); np.random.seed(1234 + rank); torch.manual_seed(1234 + rank)
        ds = edm._make_val_dataset(cfg_d, split=split, mode=mode)
        stride = max(1, len(ds) // args.n_per_cond)
        sel = list(range(0, len(ds), stride))[: args.n_per_cond]
        if args.limit:
            sel = sel[: args.limit]
        local_idx = sel[rank::world]
        if is_main:
            logging.info(f"[{name}] split={split} mode={mode}: {len(ds)} patches, using {len(sel)} (stride {stride})")
        loader = DataLoader(Subset(ds, local_idx), batch_size=1, shuffle=False,
                            num_workers=cfg_d.workers, pin_memory=True,
                            worker_init_fn=lambda w, r=rank: (random.seed(777 + 100 * r + w), np.random.seed(777 + 100 * r + w)))
        acc = Acc(device)
        for batch in tqdm(loader, disable=not is_main, desc=name):
            inputs_lr, _hr, targets = batch
            inputs_lr = inputs_lr.squeeze(0) if inputs_lr.dim() == 5 else inputs_lr
            targets = targets.squeeze(0) if targets.dim() == 5 else targets
            inputs_lr, targets = inputs_lr.to(device), targets.to(device)
            H, W = targets.shape[-2:]
            with torch.no_grad():
                coarse = chm((inputs_lr[:, :n_ls] + 1.0) / 2.0, target_size=(H, W)).clamp(p1, p99)
                acc.update(coarse.float(), edm._denormalize(targets, stats).float())
        states[name] = acc.state()

    # No NCCL needed: every rank dumps its sufficient statistics to a file, rank 0 collects them.
    os.makedirs(args.out_dir, exist_ok=True)
    jid = os.environ.get("SLURM_JOB_ID", "local")   # keeps files of different launches apart
    tmp = os.path.join(args.out_dir, "_rank_states")
    os.makedirs(tmp, exist_ok=True)
    torch.save(states, os.path.join(tmp, f"{jid}_rank{rank}.pt.tmp"))
    os.replace(os.path.join(tmp, f"{jid}_rank{rank}.pt.tmp"), os.path.join(tmp, f"{jid}_rank{rank}.pt"))
    if is_main:
        import time
        t0 = time.time()
        while not all(os.path.exists(os.path.join(tmp, f"{jid}_rank{r}.pt")) for r in range(world)):
            if time.time() - t0 > 1800:
                raise TimeoutError("timed out waiting for the other ranks' state files")
            time.sleep(5)
        gathered = [torch.load(os.path.join(tmp, f"{jid}_rank{r}.pt")) for r in range(world)]
    if is_main:
        names = list(states)
        res = {n: finalize([g[n] for g in gathered]) for n in names}
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "gap.json"), "w") as f:
            json.dump({"meta": {"run": args.run, "baseline_ckpt": cfg.model.chmv2_ckpt_path,
                                "n_per_cond": args.n_per_cond, "range_m": [p1, p99]}, "results": res}, f, indent=2)
        L = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for lab, key, fmt in (("pixels", "n_pixels", "{:.3e}"), ("R2", "r2", "{:.4f}"), ("MAE m", "mae", "{:.3f}"),
                              ("RMSE m", "rmse", "{:.3f}"), ("bias m (base-gt)", "bias", "{:+.3f}"),
                              ("error std m", "err_std", "{:.3f}"), ("GT std m", "gt_std", "{:.3f}"),
                              ("frac pixels |e|>=3m", "frac_abs_err_ge3", "{:.3f}"),
                              ("share of SSE from |e|>=3m", "sse_share_abs_err_ge3", "{:.3f}")):
            L.append(f"| {lab} | " + " | ".join(fmt.format(res[n][key]) for n in names) + " |")
        L += ["", "|e| distribution (pixel frac / share of squared error):", "",
              "| |e| bin (m) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for lab in res[names[0]]["abs_err_bins"]:
            L.append(f"| {lab} | " + " | ".join(
                f"{res[n]['abs_err_bins'][lab]['pixel_frac']:.3f} / {res[n]['abs_err_bins'][lab]['sse_share']:.3f}"
                for n in names) + " |")
        md = "\n".join(L) + "\n"
        with open(os.path.join(args.out_dir, "gap.md"), "w") as f:
            f.write(md)
        logging.info("\n" + md)
