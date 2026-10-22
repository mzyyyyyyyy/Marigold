"""
Height predictions of baseline_sr4ir_reg and sr_fm_refiner_v11-lumi on the
multi-year test set of make_time_test_split.py (default: tile 12,53, 2013-2025,
500 non-overlapping 480 px patches), stored per model, per year, per patch.

Inputs are preprocessed exactly as in training (LazyPatchDataset): bands
dataset.selected_bands, min-max scaled with the TRAINING statistics
(dataset.input_stats_file at dataset.year = 2019, also for the other years),
then mapped to [-1, 1]. Inference is that of eval_group_meeting.py:
  baseline_sr4ir_reg      SwinIR SR -> DAv2 regression, one forward pass
  sr_fm_refiner_v11-lumi  frozen DAv2 coarse + pseudo-PS ControlNet refiner,
                          n_steps Euler steps, mean of n_avg draws; draw k of
                          patch i is seeded i*1000 + k in EVERY year, so the
                          years of a patch share their sampling noise and
                          differences between years come from the input only.

Output, under --out_dir:
  <model>/<year>/<id>.tif   float32 GeoTIFF, height in metres (raw model output,
                            not clipped), 480 x 480 on the 4x Landsat grid,
                            georeferenced (CRS / transform of the Landsat tile)
  index.csv                 id, test index, Landsat / HR crop, patch centre lon / lat
  meta.json                 checkpoints, inference settings, normalisation
<id> = <test-list index>_<tile>_y<row>_x<col> (row / col on the 4x grid of the
first year), the naming of eval_group_meeting.py.

Launch (one task per GPU): see script/depth/infer_time_series.sh
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
import torch
import torch.distributed as dist
from affine import Affine
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation

import eval_detail_metrics as old            # _load_target_stats, _normalize_coarse, _denormalize, _torch_load
import eval_nsteps_navg as nsteps            # _load_sr_fm
from eval_group_meeting import DAV2_BACKBONE, MODEL_ROOT, OUT_ROOT, SR4IR_SR_CFG
from depthfm import fm_refiner as sr_fm_mod
from depthfm.sr_module import build_sr_module
from src.util.ps_lazydataset import LazyPatchDataset, MinMaxScale

SPLIT = "/flash/project_465002934/dataset/CHM_EUR_marigold/train_patch_coord_marigold/time_12-53_v1_test.pkl"
V11_RUN = "sr_fm_refiner_v11-lumi"
MODELS = ["baseline_sr4ir_reg", V11_RUN]


class TimeSeriesPatches(Dataset):
    """Item i -> (years, C, h, w) Landsat patch of entry i in every year, in [-1, 1]."""

    def __init__(self, entries, years, bands, p1, p99):
        self.entries, self.years, self.bands = entries, years, list(bands)
        self.minmax = MinMaxScale(p1, p99)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        out = []
        for yr in self.years:
            dy, dx = e["input_offsets"][yr]
            img = LazyPatchDataset._load_multiband_patch(
                e["input_paths"][yr], e["input_crop_y"] + dy, e["input_crop_x"] + dx,
                e["input_patch_height"], e["input_patch_width"], self.bands).astype(np.float32)
            img = self.minmax.apply(img) * 2.0 - 1.0          # LazyPatchDataset: min-max, then [-1, 1]
            out.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float())
        x = torch.nan_to_num(torch.stack(out), nan=0.0, posinf=0.0, neginf=0.0)
        return x.clamp(-1.0, 1.0)


def load_sr4ir(device, log):
    sr = build_sr_module(SR4IR_SR_CFG)
    reg = AutoModelForDepthEstimation.from_pretrained(DAV2_BACKBONE)
    ck = old._torch_load(os.path.join(MODEL_ROOT, "baseline_sr4ir_reg", "best.pth"))
    sr.load_state_dict(ck["sr_state"])
    reg.load_state_dict(ck["reg_state"])
    log(f"loaded baseline_sr4ir_reg (step={ck.get('step')})")
    del ck
    return sr.to(device).eval(), reg.to(device).eval()


def write_tif(path, arr, transform, crs):
    tmp = path + ".tmp"
    with rasterio.open(tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(arr.astype(np.float32), 1)
    os.replace(tmp, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--run_root", default=OUT_ROOT)
    ap.add_argument("--ckpt", default="best", help="v11 checkpoint name (best|final|latest)")
    ap.add_argument("--out_dir", default="/flash/project_465002934/result_time_lumi")
    ap.add_argument("--years", type=int, nargs="*", default=None, help="default: every year of the split")
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_avg", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N patches")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_main = rank == 0
    dist_on = world > 1
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    def log(msg):
        if is_main:
            logging.info(msg)

    if dist_on:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        use_cuda = torch.cuda.is_available() and not args.no_cuda
        dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=world,
                                timeout=timedelta(minutes=60))
    if torch.cuda.is_available() and not args.no_cuda:
        idx = local % max(1, torch.cuda.device_count())
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    # Same as the training scripts (MIOpen find-db write otherwise kills the first conv).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- models ----
    run = os.path.join(args.run_root, V11_RUN)
    cfg_v11 = OmegaConf.load(os.path.join(run, "config.yaml"))
    fm, sr_v11, n_ls = nsteps._load_sr_fm(cfg_v11, run, args, device, is_main)
    m = cfg_v11.model
    coarse = sr_fm_mod.load_dav2(dav2_path=m.dav2_pretrained_path, backbone=m.dav2_backbone,
                                 out_in_scale_factor=m.dav2_out_in_scale_factor).to(device).eval()
    sr4ir_sr, sr4ir_reg = load_sr4ir(device, log)

    # ---- data: training normalisation (2019 statistics) for every year ----
    cfg_d = cfg_v11.dataset
    in_stats = old._load_target_stats(cfg_d.input_stats_file, cfg_d.year)
    bands = list(cfg_d.selected_bands)
    p1 = [in_stats["p1"][b] for b in bands]
    p99 = [in_stats["p99"][b] for b in bands]
    target_stats = old._load_target_stats(cfg_d.target_stats_file, cfg_d.year)
    with open(args.split, "rb") as f:
        entries = pickle.load(f)
    years = args.years or list(entries[0]["years"])
    n_total = len(entries) if not args.limit else min(args.limit, len(entries))
    ref_year = entries[0]["years"][0]
    with rasterio.open(entries[0]["input_paths"][ref_year]) as src:
        ls_transform, crs = src.transform, src.crs
    log(f"{n_total} patches x {len(years)} years ({years[0]}-{years[-1]}); world {world}; "
        f"v11 {args.ckpt}.pth n_steps={args.n_steps} n_avg={args.n_avg}; "
        f"input min-max from {cfg_d.input_stats_file} [{cfg_d.year}]")

    ds = TimeSeriesPatches(entries, years, bands, p1, p99)
    local_idx = list(range(n_total))[rank::world]
    loader = DataLoader(Subset(ds, local_idx), batch_size=1, shuffle=False,
                        num_workers=cfg_d.workers, pin_memory=True)
    for name in MODELS:
        for yr in years:
            os.makedirs(os.path.join(args.out_dir, name, str(yr)), exist_ok=True)

    index_rows = []
    for sample_i, x in zip(local_idx, tqdm(loader, disable=not is_main, desc="time-series")):
        e = entries[sample_i]
        P, scale = e["output_patch_height"], e["output_patch_height"] / e["input_patch_height"]
        H = W = P
        pid = f"{sample_i:05d}_{e['tile_id'].replace(',', '-')}_y{e['output_crop_y']}_x{e['output_crop_x']}"
        hr_transform = ls_transform * Affine.translation(e["input_crop_x"], e["input_crop_y"]) \
            * Affine.scale(1.0 / scale)
        lon, lat = hr_transform * (W / 2, H / 2)
        index_rows.append({"id": pid, "test_idx": sample_i, "tile": e["tile_id"],
                           "landsat_y": e["input_crop_y"], "landsat_x": e["input_crop_x"],
                           "hr_y": e["output_crop_y"], "hr_x": e["output_crop_x"], "size": P,
                           "center_lon": lon, "center_lat": lat})
        x = x[0].to(device)                                     # (years, C, h, w)
        with torch.no_grad():
            for j, yr in enumerate(years):
                landsat = x[j:j + 1, :n_ls]
                landsat_01 = (landsat + 1.0) / 2.0
                landsat_hr = torch.nn.functional.interpolate(landsat, size=(H, W), mode="bilinear",
                                                             align_corners=False)

                # baseline_sr4ir_reg
                img_sr = sr4ir_sr(landsat, target_size=(H, W))
                pred = sr4ir_reg((img_sr + 1.0) / 2.0).predicted_depth.unsqueeze(1)
                if pred.shape[-2:] != (H, W):
                    pred = torch.nn.functional.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
                write_tif(os.path.join(args.out_dir, "baseline_sr4ir_reg", str(yr), pid + ".tif"),
                          pred[0, 0].float().cpu().numpy(), hr_transform, crs)

                # sr_fm_refiner_v11-lumi (same noise in every year: seed depends on the patch only)
                coarse_n = old._normalize_coarse(sr_fm_mod.run_dav2(coarse, landsat_01, target_size=(H, W)),
                                                 target_stats)
                pseudo_ps = sr_v11(landsat, target_size=(H, W))
                draws = []
                for k in range(args.n_avg):
                    torch.manual_seed(sample_i * 1000 + k)
                    draws.append(fm.refine_with_ps(pseudo_ps, coarse_n, n_steps=args.n_steps, method="euler",
                                                   landsat_hr=landsat_hr))
                pred = old._denormalize(torch.stack(draws).mean(0), target_stats)
                write_tif(os.path.join(args.out_dir, V11_RUN, str(yr), pid + ".tif"),
                          pred[0, 0].float().cpu().numpy(), hr_transform, crs)

    if dist_on:
        gathered = [None] * world
        dist.all_gather_object(gathered, index_rows)
    else:
        gathered = [index_rows]
    if is_main:
        rows = sorted((r for g in gathered for r in g), key=lambda r: r["test_idx"])
        with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        meta = {"split": args.split, "n_patches": n_total, "years": years, "tile": entries[0]["tile_id"],
                "models": {"baseline_sr4ir_reg": f"{MODEL_ROOT}/baseline_sr4ir_reg/best.pth",
                           V11_RUN: f"{run}/checkpoint/{args.ckpt}.pth"},
                "v11_inference": {"n_steps": args.n_steps, "n_avg": args.n_avg, "method": "euler",
                                  "seed_rule": "torch.manual_seed(patch_idx*1000 + draw_idx), same in every year",
                                  "coarse": m.dav2_pretrained_path},
                "input_normalisation": {"stats_file": cfg_d.input_stats_file, "stats_year": int(cfg_d.year),
                                        "bands": bands, "p1": p1, "p99": p99,
                                        "rule": "(x - p1) / (p99 - p1) clipped to [0, 1], then * 2 - 1"},
                "output": "float32 GeoTIFF, metres, raw model output (not clipped), 4x Landsat grid",
                "crs": str(crs)}
        with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log(f"wrote {len(rows)} patches x {len(years)} years x {len(MODELS)} models to {args.out_dir}")

    if dist_on:
        dist.barrier()
        dist.destroy_process_group()
