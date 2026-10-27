"""
How well does a trained FM refiner use the PS slot? Refinement metrics of one
FM checkpoint (default: fm_refiner-R-lumi-5-nd final.pth, trained on the
finer_lumi_nd branch with real PS / null-PS dropout) under several conditions
in the PS slot, on evenly spaced test patches:

  real    real PlanetScope                 (ceiling of this ControlNet)
  null    null-PS token                    (what the run itself reports)
  pseudo  SR output of --sr_run's SR module (e.g. v12-nd's SwinIR)
  lsps    landsat_hr in the PS slot

The FM is rebuilt with depthfm/fm_refiner.py (the train_sr_fm_refiner.py code
path, concat_landsat_hr=True), whose ControlNet input cat([landsat_hr, ps]),
z_t / velocity / timestep conventions match fm_refiner_finer.py's
controlnet_cond_mode="landsat_ps" -- so this also checks that such a
checkpoint can be loaded as the frozen FM of an SR4IR-style stage 2.

Output: <out_dir>/cond_ceiling.md / .json. Launch: script/depth/diag_cond_ceiling.sh
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import eval_detail_metrics as edm
from depthfm import fm_refiner as sr_fm_mod
from depthfm.chmv2 import load_chmv2_baseline
from depthfm.sr_module import build_sr_module
from eval_detail_metrics_nd import make_test_dataset

OUT_ROOT = edm.OUT_ROOT
CONDS = ["real", "null", "pseudo", "lsps"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm_run", default=f"{OUT_ROOT}/fm_refiner-R-lumi-5-nd")
    ap.add_argument("--fm_ckpt", default="final.pth")
    ap.add_argument("--sr_run", default=f"{OUT_ROOT}/sr_fm_refiner_v12-lumi-nd", help="run whose SR makes the pseudo-PS")
    ap.add_argument("--sr_ckpt", default="best.pth")
    ap.add_argument("--n_patches", type=int, default=128)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_avg", type=int, default=5)
    ap.add_argument("--out_dir", default=None, help="default: <OUT_ROOT>/_diag_cond_ceiling/<fm run>")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda:0")

    cfg = OmegaConf.load(os.path.join(args.fm_run, "config.yaml"))
    cfg_sr = OmegaConf.load(os.path.join(args.sr_run, "config.yaml"))
    d, t = cfg.dataset, cfg.trainer
    n_ls, n_ps = len(d.selected_bands), len(d.selected_bands_hr)
    in_stats = edm._load_target_stats(d.input_stats_file, d.year)
    stats = edm._load_target_stats(d.target_stats_file, d.year)
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    assert t.get("controlnet_cond_mode", "landsat_ps") == "landsat_ps", "only landsat_ps checkpoints map onto concat_landsat_hr"

    chm = load_chmv2_baseline(model_dir=cfg.model.chmv2_model_id, ckpt_path=cfg.model.chmv2_ckpt_path,
                              mean=in_stats["mean"], std=in_stats["std"],
                              out_in_scale_factor=cfg.model.chmv2_out_in_scale_factor).to(device).eval()
    fm = sr_fm_mod.build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path, n_landsat_bands=n_ls, n_ps_bands=n_ps, ps_dropout_p=0.0,
        noise_sigma=t.get("noise_sigma", 0.0), sample_sigma=t.get("sample_sigma", None),
        sample_noise_mode=t.get("sample_noise_mode", "sde"), concat_landsat_hr=True, device=str(device)).to(device)
    ck = edm._torch_load(os.path.join(args.fm_run, "checkpoint", args.fm_ckpt))
    fm.unet.load_state_dict(ck["unet_state"])
    fm.controlnet.load_state_dict(ck["controlnet_state"])
    fm._null_ps = ck["null_ps_state"]
    fm_step = ck.get("step")
    del ck
    fm.eval()
    sr = build_sr_module(OmegaConf.to_container(cfg_sr.sr_module, resolve=True)).to(device)
    ck = edm._torch_load(os.path.join(args.sr_run, "checkpoint", args.sr_ckpt))
    sr.load_state_dict(ck["sr_state"])
    del ck
    sr.eval()
    logging.info(f"FM {args.fm_run}/{args.fm_ckpt} (step {fm_step}); SR from {args.sr_run}/{args.sr_ckpt}")

    test_ds, _ = make_test_dataset(d)
    sel = sorted(set(np.linspace(0, len(test_ds) - 1, args.n_patches).round().astype(int).tolist()))
    acc = {c: edm.DetailAccumulator(device, 2.0, 1, 2.0) for c in CONDS}
    with torch.no_grad():
        for j, i in enumerate(sel):
            inputs_lr, inputs_hr, targets = [x.unsqueeze(0) if x.dim() == 3 else x for x in test_ds[i]]
            inputs_lr, real_ps, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
            H, W = targets.shape[-2:]
            landsat = inputs_lr[:, :n_ls]
            landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)
            h_coarse = edm._normalize_coarse(chm((landsat + 1.0) / 2.0, target_size=(H, W)), stats)
            gt_m = edm._denormalize(targets, stats).clamp(p1, p99)
            ps_by = {"real": real_ps, "null": fm._null_ps.to(device, real_ps.dtype).expand_as(real_ps),
                     "pseudo": sr(landsat, target_size=(H, W)), "lsps": landsat_hr}
            for c in CONDS:
                torch.manual_seed(i)          # identical init noise for every condition
                hs = [fm.refine_with_ps(ps_by[c], h_coarse, n_steps=args.n_steps, method="euler",
                                        landsat_hr=landsat_hr) for _ in range(args.n_avg)]
                acc[c].update(edm._denormalize(torch.stack(hs).mean(0), stats).clamp(p1, p99).float(), gt_m.float())
            if (j + 1) % 16 == 0:
                logging.info(f"{j + 1}/{len(sel)} patches")

    res = {c: edm.finalize([acc[c].state()]) for c in CONDS}
    fm_name = os.path.basename(args.fm_run.rstrip("/"))
    out_dir = args.out_dir or f"{OUT_ROOT}/_diag_cond_ceiling/{fm_name}"
    os.makedirs(out_dir, exist_ok=True)
    L = [f"# PS-slot conditions for {fm_name} ({args.fm_ckpt}, step {fm_step})", "",
         f"{len(sel)} test patches; refine {args.n_steps} steps x {args.n_avg} draws; pseudo-PS from "
         f"{os.path.basename(args.sr_run.rstrip('/'))} ({args.sr_ckpt})", "",
         "| PS slot | R2 | MAE | RMSE | HF corr | HF energy ratio | Edge F1 tol1 |", "|---|---|---|---|---|---|---|"]
    for c in CONDS:
        m = res[c]
        L.append(f"| {c} | {m['r2']:.4f} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['hf_corr']:.4f} | "
                 f"{m['hf_energy_ratio']:.4f} | {m['edge_f1_tol1']:.4f} |")
    with open(os.path.join(out_dir, "cond_ceiling.json"), "w") as f:
        json.dump({"args": vars(args), "fm_step": fm_step, "metrics": res}, f, indent=2)
    with open(os.path.join(out_dir, "cond_ceiling.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    logging.info("\n" + "\n".join(L))
