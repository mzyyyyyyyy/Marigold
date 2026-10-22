"""
Wall-to-wall height maps of baseline_sr4ir_reg and sr_fm_refiner_v11-lumi over one
region, for every year of a multi-year Landsat tile (default: the footprint of the
Danish LiDAR CHMs in /flash/project_465002934/dataset/CHM_Denm, on Landsat tile
13,61, 2014-2025).

Region: the Landsat pixels covering the union of the --ref_tifs footprints
(reprojected to the Landsat CRS), plus --context Landsat px on every side that are
read as model input but cropped from the output, so the border of the map is not
predicted from truncated windows.
Mosaic: --window px windows (on the 4x grid, i.e. window/4 Landsat px) every
--stride px, blended with the Gaussian weight of infer_sr_fm_refiner.py.
Inference per window is that of infer_time_series.py: training preprocessing
(min-max with the 2019 training statistics, in every year); v11 n_steps Euler
steps, mean of n_avg draws seeded window_idx*1000 + k, the same in every year.

Each task handles the years years[rank::world] alone (no communication).
Output, under --out_dir:
  <model>/<year>.tif   float32 GeoTIFF, metres (raw model output, not clipped), on the
                       4x Landsat grid (CRS of the Landsat tile)
  region.json          Landsat window, context, windows, settings
Then run script/depth/vis_eval_region.py (PNGs, strips, GIFs, LiDAR evaluation).
Launch: see script/depth/infer_time_region.sh
"""

import argparse
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from affine import Affine
from omegaconf import OmegaConf
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from tqdm import tqdm

import eval_detail_metrics as old
import eval_nsteps_navg as nsteps
from eval_group_meeting import OUT_ROOT
from infer_sr_fm_refiner import _create_blending_mask
from infer_time_series import V11_RUN, load_sr4ir
from depthfm import fm_refiner as sr_fm_mod
from src.util.ps_lazydataset import MinMaxScale

MODELS = ["baseline_sr4ir_reg", V11_RUN]


def landsat_region(ls_path, ref_tifs, context):
    """Landsat pixel window (row0, col0, h, w) covering the ref tifs, and the same window grown by context."""
    with rasterio.open(ls_path) as src:
        t, crs, H, W = src.transform, src.crs, src.height, src.width
    l = b = math.inf
    r = tp = -math.inf
    for p in ref_tifs:
        with rasterio.open(p) as ref:
            bb = transform_bounds(ref.crs, crs, *ref.bounds, densify_pts=101)
        l, b, r, tp = min(l, bb[0]), min(b, bb[1]), max(r, bb[2]), max(tp, bb[3])
    c0, c1 = math.floor((l - t.c) / t.a), math.ceil((r - t.c) / t.a)
    r0, r1 = math.floor((tp - t.f) / t.e), math.ceil((b - t.f) / t.e)
    core = (r0, c0, r1 - r0, c1 - c0)
    padded = (r0 - context, c0 - context, r1 - r0 + 2 * context, c1 - c0 + 2 * context)
    assert padded[0] >= 0 and padded[1] >= 0 and padded[0] + padded[2] <= H and padded[1] + padded[3] <= W, \
        f"region + context leaves the Landsat tile: {padded} vs {H}x{W}"
    return core, padded, t, crs


def starts(n, win, stride):
    s = list(range(0, n - win + 1, stride))
    if s[-1] != n - win:
        s.append(n - win)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--landsat_root", default="/flash/project_465002934/data/Landsat/eur_test2_map")
    ap.add_argument("--tile", default="13,61")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2014, 2026)))
    ap.add_argument("--ref_tifs", nargs="+", default=["/flash/project_465002934/dataset/CHM_Denm/2014_mosaic.tif",
                                                      "/flash/project_465002934/dataset/CHM_Denm/2019_mosaic.tif"])
    ap.add_argument("--context", type=int, default=60, help="extra Landsat px read on every side (cropped)")
    ap.add_argument("--window", type=int, default=480, help="window on the 4x grid")
    ap.add_argument("--stride", type=int, default=240, help="window stride on the 4x grid")
    ap.add_argument("--run_root", default=OUT_ROOT)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_avg", type=int, default=5)
    ap.add_argument("--out_dir", default="/flash/project_465002934/result_time_lumi_denm")
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [rank {rank}] %(message)s")
    my_years = args.years[rank::world]
    if not my_years:
        logging.info("no year for this task")
        sys.exit(0)
    torch.cuda.set_device(local % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device())
    # Same as the training scripts (MIOpen find-db write otherwise kills the first conv).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    paths = {y: os.path.join(args.landsat_root, str(y), args.tile, f"landsat_{y}_{args.tile}_p50.jp2")
             for y in args.years}
    core, padded, ls_t, crs = landsat_region(paths[args.years[0]], args.ref_tifs, args.context)
    for y in args.years:
        with rasterio.open(paths[y]) as src:
            assert src.transform.almost_equals(ls_t) and src.crs == crs, f"{y}: Landsat grid differs"
    S = 4
    assert args.window % S == 0 and args.stride % S == 0
    win_lr, stride_lr = args.window // S, args.stride // S
    _, _, Hp, Wp = padded
    wins = [(y, x) for y in starts(Hp, win_lr, stride_lr) for x in starts(Wp, win_lr, stride_lr)]
    blend = torch.from_numpy(_create_blending_mask(args.window)).to(device)
    # output = core region on the 4x grid
    off_r, off_c = (core[0] - padded[0]) * S, (core[1] - padded[1]) * S
    Hc, Wc = core[2] * S, core[3] * S
    out_t = ls_t * Affine.translation(core[1], core[0]) * Affine.scale(1.0 / S)
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "region.json"), "w") as f:
            json.dump({"tile": args.tile, "years": args.years, "landsat_paths": {str(k): v for k, v in paths.items()},
                       "ref_tifs": args.ref_tifs, "landsat_core_window_rc_hw": core,
                       "landsat_padded_window_rc_hw": padded, "context_landsat_px": args.context,
                       "window_hr": args.window, "stride_hr": args.stride, "n_windows": len(wins),
                       "out_shape_hw": [Hc, Wc], "out_transform": list(out_t)[:6], "crs": str(crs),
                       "models": {"baseline_sr4ir_reg": "/flash/project_465002934/model/baseline_sr4ir_reg/best.pth",
                                  V11_RUN: f"{args.run_root}/{V11_RUN}/checkpoint/{args.ckpt}.pth"},
                       "v11_inference": {"n_steps": args.n_steps, "n_avg": args.n_avg, "method": "euler",
                                         "seed_rule": "torch.manual_seed(window_idx*1000 + draw_idx), same in every year"},
                       "output": "float32 GeoTIFF, metres, raw model output (not clipped)"}, f, indent=1)
    logging.info(f"years {my_years}; Landsat core {core}, padded {padded}; {len(wins)} windows of {args.window} px; "
                 f"output {Hc}x{Wc}")

    # ---- models ----
    run = os.path.join(args.run_root, V11_RUN)
    cfg_v11 = OmegaConf.load(os.path.join(run, "config.yaml"))
    fm, sr_v11, n_ls = nsteps._load_sr_fm(cfg_v11, run, args, device, rank == 0)
    m = cfg_v11.model
    coarse = sr_fm_mod.load_dav2(dav2_path=m.dav2_pretrained_path, backbone=m.dav2_backbone,
                                 out_in_scale_factor=m.dav2_out_in_scale_factor).to(device).eval()
    sr4ir_sr, sr4ir_reg = load_sr4ir(device, logging.info)
    cfg_d = cfg_v11.dataset
    in_stats = old._load_target_stats(cfg_d.input_stats_file, cfg_d.year)
    bands = list(cfg_d.selected_bands)
    minmax = MinMaxScale([in_stats["p1"][b] for b in bands], [in_stats["p99"][b] for b in bands])
    target_stats = old._load_target_stats(cfg_d.target_stats_file, cfg_d.year)

    for name in MODELS:
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)
    win = args.window
    for yr in my_years:
        with rasterio.open(paths[yr]) as src:
            img = src.read([b + 1 for b in bands], window=Window(padded[1], padded[0], Wp, Hp))
        img = minmax.apply(img.transpose(1, 2, 0).astype(np.float32)) * 2.0 - 1.0   # training preprocessing
        x_all = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float()
        x_all = torch.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1, 1).to(device)
        acc = {n: torch.zeros(Hp * S, Wp * S, device=device) for n in MODELS}
        wsum = torch.zeros(Hp * S, Wp * S, device=device)
        for wi, (y0, x0) in enumerate(tqdm(wins, desc=f"{yr}", disable=rank != 0)):
            landsat = x_all[..., y0:y0 + win_lr, x0:x0 + win_lr]
            landsat_01 = (landsat + 1.0) / 2.0
            landsat_hr = F.interpolate(landsat, size=(win, win), mode="bilinear", align_corners=False)
            with torch.no_grad():
                img_sr = sr4ir_sr(landsat, target_size=(win, win))
                p_sr4ir = sr4ir_reg((img_sr + 1.0) / 2.0).predicted_depth.unsqueeze(1)
                if p_sr4ir.shape[-2:] != (win, win):
                    p_sr4ir = F.interpolate(p_sr4ir, size=(win, win), mode="bilinear", align_corners=False)
                coarse_n = old._normalize_coarse(sr_fm_mod.run_dav2(coarse, landsat_01, target_size=(win, win)),
                                                 target_stats)
                pseudo_ps = sr_v11(landsat, target_size=(win, win))
                draws = []
                for k in range(args.n_avg):
                    torch.manual_seed(wi * 1000 + k)
                    draws.append(fm.refine_with_ps(pseudo_ps, coarse_n, n_steps=args.n_steps, method="euler",
                                                   landsat_hr=landsat_hr))
                p_v11 = old._denormalize(torch.stack(draws).mean(0), target_stats)
            ys, xs = y0 * S, x0 * S
            for n, p in ((MODELS[0], p_sr4ir), (MODELS[1], p_v11)):
                acc[n][ys:ys + win, xs:xs + win] += p[0, 0].float() * blend
            wsum[ys:ys + win, xs:xs + win] += blend
        for n in MODELS:
            mosaic = (acc[n] / wsum)[off_r:off_r + Hc, off_c:off_c + Wc].cpu().numpy()
            path = os.path.join(args.out_dir, n, f"{yr}.tif")
            tmp = path + ".tmp"
            with rasterio.open(tmp, "w", driver="GTiff", height=Hc, width=Wc, count=1, dtype="float32", crs=crs,
                               transform=out_t, compress="deflate", tiled=True) as dst:
                dst.write(mosaic.astype(np.float32), 1)
            os.replace(tmp, path)
            logging.info(f"wrote {path}")
