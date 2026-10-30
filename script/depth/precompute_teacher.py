"""
Precompute output-distillation targets from a finished teacher run of
train_baseline_chmv2.py (e.g. baseline_chmv2-lumi-nd-ps8-h120): the teacher's
prediction (metres, clamped to [0, 50] like the GT) for every TRAIN patch, at
the patch's full size and without augmentation, saved as
<out_dir>/train/<LazyPatchDataset.patch_key>.npy (float16). The student run
reads them through LazyPatchDataset.teacher_target_dir (distill.teacher_run).

Also reports the teacher's pixel-pooled R2 / MAE / RMSE on the train patches
and on the full val split (<out_dir>/summary.json): a train R2 well above val
means the teacher memorised its training patches, so its targets there would
be ~GT and the distillation would carry nothing (then cross-fit instead).

Launch (one task per GPU): see script/depth/precompute_teacher.sh
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from depthfm.chmv2 import load_chmv2_baseline
from src.util.ps_lazydataset import LazyPatchDataset
from train_baseline_chmv2 import _input_mean_std, _load_coords, _make_dataset, _model_input


def make_base_dataset(d):
    return LazyPatchDataset({
        'input_dir': d.input_dir, 'input_dir_hr': d.input_dir_hr, 'output_dir': d.output_dir,
        'selected_bands': d.selected_bands, 'selected_bands_hr': d.selected_bands_hr,
        'file_type_input': d.file_type_input, 'file_type_output': d.file_type_output,
        'patch_sizes': [(s, s) for s in d.patch_sizes], 'patch_coord_path': d.patch_coord_path,
        'num_patches_per_tile': d.num_patches_per_tile, 'tile_emphasis': d.tile_emphasis,
        'correlation_cleaning': d.correlation_cleaning, 'target_range_edges': d.target_range_edges,
        'num_patches_per_target_range': d.num_patches_per_target_range,
        'selected_percentile': d.selected_percentile, 'year': d.year, 'batch_size': 1, 'mode': 'train',
        'use_input_minmax': d.use_input_minmax, 'use_input_norm': d.use_input_norm,
        'input_stats_file': d.input_stats_file,
        'use_input_hr_minmax': d.use_input_hr_minmax, 'use_input_hr_norm': d.use_input_hr_norm,
        'input_hr_stats_file': d.input_hr_stats_file,
        'use_target_minmax': d.use_target_minmax, 'use_target_norm': d.use_target_norm,
        'target_stats_file': d.target_stats_file, 'unit_scale_ratio': d.unit_scale_ratio,
        'scale_input_to_neg1_1': d.get('scale_input_to_neg1_1', False),
        'scale_input_hr_to_neg1_1': d.get('scale_input_hr_to_neg1_1', False),
        'scale_target_to_neg1_1': d.get('scale_target_to_neg1_1', False),
    })


def finalize(s):
    ss_tot = s["sum_gt2"] - s["sum_gt"] ** 2 / s["n"]
    return {"r2": 1.0 - s["sum_sq"] / ss_tot, "mae": s["sum_abs"] / s["n"], "rmse": math.sqrt(s["sum_sq"] / s["n"]),
            "n_patches": int(s["n_patches"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_run", required=True, help="finished train_baseline_chmv2.py run dir")
    ap.add_argument("--out_dir", default=None, help="default: <teacher_run>/teacher_targets")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="targets are saved for train only")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N patches per split")
    ap.add_argument("--no_clamp", action="store_true",
                    help="score the raw predictions (as the training script's test does), not clamped to [0, 50]")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.teacher_run, "teacher_targets")

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_main = rank == 0
    if not is_main:
        sys.stdout = open(os.devnull, "w")
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=60))
    idx = local % max(1, torch.cuda.device_count())
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}")

    cfg = OmegaConf.load(os.path.join(args.teacher_run, "config.yaml"))
    d = cfg.dataset
    source = cfg.model.get("input_source", "landsat")
    mean, std = _input_mean_std(cfg)
    teacher = load_chmv2_baseline(
        model_dir=cfg.model.chmv2_model_id, ckpt_path=os.path.join(args.teacher_run, "checkpoint", "best.pth"),
        mean=mean, std=std, out_in_scale_factor=cfg.model.get("out_in_scale_factor", 1),
        upsample_input=cfg.model.get("upsample_input_to_target", False),
        input_scale=cfg.model.get("input_scale", 1), sampled_head=cfg.model.get("sampled_head", False),
    ).to(device)
    n_ls = len(d.selected_bands)
    base = make_base_dataset(d)
    coord_paths = {"train": d.patch_coord_path, "val": d.val_coord_path, "test": d.test_coord_path}
    if is_main:
        logging.info(f"teacher {args.teacher_run} (input {source}); splits {args.splits}; out {out_dir}")

    summary = {}
    for split in args.splits:
        coords = _load_coords(coord_paths[split])
        ds = _make_dataset(base, coords, "test", 1)  # eval mode: no augmentation, item idx == patch idx
        n_total = min(args.limit, len(ds)) if args.limit else len(ds)
        local_idx = list(range(n_total))[rank::world]
        loader = DataLoader(Subset(ds, local_idx), batch_size=1, shuffle=False,
                            num_workers=d.workers, pin_memory=True)
        save_dir = os.path.join(out_dir, split)
        if split == "train":
            os.makedirs(save_dir, exist_ok=True)
        acc = torch.zeros(8, dtype=torch.float64, device=device)  # n, sum_gt, sum_gt2, sum_sq, sum_abs, n_patches, n_neg, n_over
        with torch.no_grad():
            for i, batch in zip(local_idx, tqdm(loader, disable=not is_main, desc=f"teacher-{split}")):
                inputs_lr, inputs_hr, targets = (t.squeeze(0) if t.dim() == 5 else t for t in batch)
                inputs_lr, inputs_hr, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
                raw = teacher(_model_input(inputs_lr, inputs_hr, source, n_ls), target_size=tuple(targets.shape[-2:]))
                pred = raw if args.no_clamp else raw.clamp(0.0, 50.0)
                if split == "train":
                    np.save(os.path.join(save_dir, LazyPatchDataset.patch_key(coords[i]) + ".npy"),
                            pred[0, 0].cpu().numpy().astype(np.float16))
                g, e = targets.double(), (pred - targets).double()
                acc += torch.stack([torch.tensor(g.numel(), dtype=torch.float64, device=device), g.sum(),
                                    (g ** 2).sum(), (e ** 2).sum(), e.abs().sum(),
                                    torch.tensor(1.0, dtype=torch.float64, device=device),
                                    (raw < 0).sum().double(), (raw > 50).sum().double()])
        if world > 1:
            dist.all_reduce(acc)
        s = dict(zip(["n", "sum_gt", "sum_gt2", "sum_sq", "sum_abs", "n_patches", "n_neg", "n_over"], acc.tolist()))
        summary[split] = {**finalize(s), "clamped": not args.no_clamp,
                          "raw_frac_below_0": s["n_neg"] / s["n"], "raw_frac_above_50": s["n_over"] / s["n"]}
        if is_main:
            logging.info(f"teacher on {split}: {summary[split]}")

    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({"teacher_run": args.teacher_run, "limit": args.limit, "metrics": summary}, f, indent=1)
        print(json.dumps(summary, indent=1))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
