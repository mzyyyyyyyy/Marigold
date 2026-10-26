"""
Group-meeting comparison of four models on the full TEST split of the new
(leak-free spatial) dataset, split_spatial_v1_test.pkl (= baseline_swt-lumi-nd's
test set, 2206 non-overlapping 480 px patches), plus per-patch result images.

  baseline_dav2           : DAv2 regression, Landsat -> height (sr_finer branch)
  baseline_sr4ir_reg      : SR4IR, SwinIR SR (Landsat -> pseudo-PS) + DAv2 regression (sr_finer branch)
  fm_refiner-R-lumi-4     : FM refiner, Landsat-only ControlNet, DAv2 coarse (finer_lumi branch)
  sr_fm_refiner_v11-lumi  : SR + FM refiner, pseudo-PS ControlNet, DAv2 coarse (sr_finer_lumi branch)

All four were trained on the OLD random split of save_20260609*.pkl, so part of
this test set overlaps their training patches (same for all four models).

Code: depthfm/fm_refiner.py on this branch is byte-identical to the one v11 was
trained with (74f1eee), depthfm/fm_refiner_finer.py to finer_lumi's
depthfm/fm_refiner.py at R-lumi-4 (21709c2), and depthfm/depth_anything.py /
sr_module.py are unchanged since the baselines (sr_finer f671788).

Inference mirrors each model's own validation: baselines are single forward
passes; refiners use the shared frozen DAv2 coarse model (ls_da_model_r2.pth),
n_steps Euler steps, mean of n_avg stochastic draws (seeded per patch).
Metrics are those of eval_detail_metrics(_nd).py, pixel-pooled over the split:
GT and predictions clipped to the target range [p1, p99] = [0, 28] m (as in the
earlier detail evals); an unclipped R2 / MAE / RMSE is reported as a sanity check.
Image-quality metrics on the same clipped maps, data range R = p99 - p1: PSNR (mean of
per-patch PSNR, and from the pooled MSE) and SSIM (see PsnrSsimAccumulator).

Output (<out_dir>, one PNG per test patch, same file name <id>.png in every folder):
  landsat/                    Landsat RGB input (nearest x4 to the 480 px grid)
  ps/                         real PlanetScope RGB
  baseline_dav2/              height predictions, plasma colour map over [0, 28] m
  baseline_sr4ir_reg/         (same colour scale for every model and the GT)
  sr_fm_refiner_v11-lumi/
  fm_refiner-R-lumi-4/
  baseline_sr4ir_reg_sr/      baseline_sr4ir_reg's SwinIR super-resolved image
  sr_fm_refiner_v11-lumi_sr/  v11's pseudo-PS
  gt/                         ground-truth height, same colour map
  panel/                      all of the above side by side, with per-patch MAE
  detail_metrics.{md,json}, index.csv (per-patch GT p75 and per-model MAE)
<id> = <test-list index>_<tile>_y<row>_x<col>.

Launch (one task per GPU): see script/depth/eval_group_meeting.sh
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation

import eval_detail_metrics as old            # DetailAccumulator, finalize, helpers
import eval_detail_metrics_nd as nd          # make_test_dataset, EdgeAccumulator, image helpers
from depthfm import fm_refiner as sr_fm_mod            # == v11's code (sr_finer_lumi 74f1eee)
from depthfm import fm_refiner_finer as plain_fm_mod   # == R-lumi-4's code (finer_lumi 21709c2)
from depthfm.depth_anything import DepthAnythingV2Height
from depthfm.sr_module import build_sr_module

OUT_ROOT = "/flash/project_465002934/Marigold_output"
MODEL_ROOT = "/flash/project_465002934/model"
NAMES = ["baseline_dav2", "baseline_sr4ir_reg", "fm_refiner-R-lumi-4", "sr_fm_refiner_v11-lumi"]
DAV2_BACKBONE = "depth-anything/Depth-Anything-V2-Base-hf"
# sr_module block of sr_finer:config/baseline_sr4ir_reg.yaml
SR4IR_SR_CFG = dict(in_chans=3, out_chans=3, upscale=4, img_size=48, window_size=8,
                    depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2.0,
                    upsampler="pixelshuffle", resi_connection="1conv")


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------

def load_models(args, device, log):
    cfg_v11 = OmegaConf.load(os.path.join(args.run_root, "sr_fm_refiner_v11-lumi", "config.yaml"))
    cfg_r4 = OmegaConf.load(os.path.join(args.run_root, "fm_refiner-R-lumi-4", "config.yaml"))
    ck_name = f"{args.ckpt}.pth"

    # --- frozen DAv2 coarse model shared by both refiners ---
    for k in ("dav2_pretrained_path", "dav2_backbone", "dav2_out_in_scale_factor"):
        assert cfg_v11.model[k] == cfg_r4.model[k], f"refiners use a different coarse model ({k})"
    coarse = sr_fm_mod.load_dav2(
        dav2_path=cfg_v11.model.dav2_pretrained_path, backbone=cfg_v11.model.dav2_backbone,
        out_in_scale_factor=cfg_v11.model.dav2_out_in_scale_factor,
    ).to(device).eval()
    log(f"loaded coarse DAv2 ({cfg_v11.model.dav2_pretrained_path})")

    # --- baseline_dav2 ---
    dav2 = DepthAnythingV2Height(backbone=DAV2_BACKBONE, use_geo_encoder=False,
                                 pretrained=False, out_in_scale_factor=1)
    ck = old._torch_load(os.path.join(MODEL_ROOT, "baseline_dav2", "best.pth"))
    dav2.load_state_dict(ck["model_state_dict"])
    log(f"loaded baseline_dav2 (step={ck.get('step')})")
    del ck
    dav2 = dav2.to(device).eval()

    # --- baseline_sr4ir_reg: SwinIR SR + DAv2 (HF) regression ---
    sr4ir_sr = build_sr_module(SR4IR_SR_CFG)
    sr4ir_reg = AutoModelForDepthEstimation.from_pretrained(DAV2_BACKBONE)
    ck = old._torch_load(os.path.join(MODEL_ROOT, "baseline_sr4ir_reg", "best.pth"))
    sr4ir_sr.load_state_dict(ck["sr_state"])
    sr4ir_reg.load_state_dict(ck["reg_state"])
    log(f"loaded baseline_sr4ir_reg (step={ck.get('step')})")
    del ck
    sr4ir_sr, sr4ir_reg = sr4ir_sr.to(device).eval(), sr4ir_reg.to(device).eval()

    n_ls = len(cfg_v11.dataset.selected_bands)
    n_ps = len(cfg_v11.dataset.selected_bands_hr)

    # --- fm_refiner-R-lumi-4 (finer_lumi code) ---
    t = cfg_r4.trainer
    fm_r4 = plain_fm_mod.build_fm_refiner(
        sd_pretrained_path=cfg_r4.model.sd_pretrained_path,
        n_landsat_bands=n_ls, n_ps_bands=n_ps, ps_dropout_p=0.0,
        use_controlnet=t.get("use_controlnet", True),
        controlnet_cond_mode=t.get("controlnet_cond_mode", "landsat_ps"),
        velocity_parameterization=t.get("velocity_parameterization", "fixed"),
        noise_sigma=t.get("noise_sigma", 0.0), sample_sigma=t.get("sample_sigma", None),
        sample_noise_mode=t.get("sample_noise_mode", "sde"),
        device=str(device),
    ).to(device)
    ck = old._torch_load(os.path.join(args.run_root, "fm_refiner-R-lumi-4", "checkpoint", ck_name))
    fm_r4.unet.load_state_dict(ck["unet_state"])
    if fm_r4.use_controlnet and ck.get("controlnet_state") is not None:
        fm_r4.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm_r4._null_ps = ck["null_ps_state"]
    log(f"loaded fm_refiner-R-lumi-4 (step={ck.get('step')})")
    del ck
    fm_r4.eval()

    # --- sr_fm_refiner_v11-lumi (this branch's code) ---
    t = cfg_v11.trainer
    fm_v11 = sr_fm_mod.build_fm_refiner(
        sd_pretrained_path=cfg_v11.model.sd_pretrained_path,
        n_landsat_bands=n_ls, n_ps_bands=n_ps, ps_dropout_p=0.0,
        bridge_sigma=t.get("bridge_sigma", 0.0), noise_sigma=t.get("noise_sigma", 0.0),
        sample_sigma=t.get("sample_sigma", None), sample_noise_mode=t.get("sample_noise_mode", "sde"),
        concat_z_coarse=t.get("concat_z_coarse", False), refine_threshold=t.get("refine_threshold", 0.0),
        concat_landsat_hr=t.get("concat_landsat_hr", False),
        device=str(device),
    ).to(device)
    sr_v11 = build_sr_module(OmegaConf.to_container(cfg_v11.sr_module, resolve=True)).to(device)
    ck = old._torch_load(os.path.join(args.run_root, "sr_fm_refiner_v11-lumi", "checkpoint", ck_name))
    fm_v11.unet.load_state_dict(ck["unet_state"])
    fm_v11.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm_v11._null_ps = ck["null_ps_state"]
    sr_v11.load_state_dict(ck["sr_state"])
    log(f"loaded sr_fm_refiner_v11-lumi (step={ck.get('step')})")
    del ck
    fm_v11.eval()
    sr_v11.eval()

    # the refiners' coarse normalisation and output range must match the eval range
    for c in (cfg_v11, cfg_r4):
        assert c.dataset.target_stats_file == cfg_v11.dataset.target_stats_file
    return dict(coarse=coarse, dav2=dav2, sr4ir_sr=sr4ir_sr, sr4ir_reg=sr4ir_reg,
                fm_r4=fm_r4, fm_v11=fm_v11, sr_v11=sr_v11, n_ls=n_ls,
                target_stats_file=cfg_v11.dataset.target_stats_file, ck_name=ck_name)


# -------------------------------------------------------------------------
# Unclipped R2 / MAE / RMSE (sanity check against the clipped numbers)
# -------------------------------------------------------------------------

class RawAccumulator:
    def __init__(self):
        self.s = {k: 0.0 for k in ("n", "sum_gt", "sum_gt2", "sum_abs", "sum_sq")}

    @torch.no_grad()
    def update(self, pred, gt):
        p, g = pred.double(), gt.double()
        e = p - g
        self.s["n"] += g.numel()
        self.s["sum_gt"] += g.sum().item()
        self.s["sum_gt2"] += (g ** 2).sum().item()
        self.s["sum_abs"] += e.abs().sum().item()
        self.s["sum_sq"] += (e ** 2).sum().item()

    @staticmethod
    def finalize(states):
        s = {k: sum(st[k] for st in states) for k in states[0]}
        ss_tot = s["sum_gt2"] - s["sum_gt"] ** 2 / s["n"]
        return {"r2": 1.0 - s["sum_sq"] / ss_tot, "mae": s["sum_abs"] / s["n"],
                "rmse": math.sqrt(s["sum_sq"] / s["n"])}


# -------------------------------------------------------------------------
# PSNR / SSIM (image-quality metrics on the clipped height maps)
# -------------------------------------------------------------------------

class PsnrSsimAccumulator:
    """PSNR and SSIM with data range R = p99 - p1 (the clipped height range), as for images.

    PSNR = 10 log10(R^2 / MSE): reported as the mean over patches of the per-patch PSNR
    (the usual super-resolution convention) and from the pooled MSE ("global", a monotone
    function of RMSE). SSIM: Wang et al. 2004 with an 11x11 Gaussian window (sigma 1.5),
    K1 = 0.01, K2 = 0.03, valid filtering (no padding); mean over patches of the mean SSIM
    map (all test patches have the same size, so this is also the pixel-pooled mean)."""

    def __init__(self, device, data_range, win=11, sigma=1.5, k1=0.01, k2=0.03):
        x = torch.arange(win, dtype=torch.float32) - (win - 1) / 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        self.gh, self.gv = g.view(1, 1, 1, -1).to(device), g.view(1, 1, -1, 1).to(device)
        self.R = float(data_range)
        self.c1, self.c2 = (k1 * self.R) ** 2, (k2 * self.R) ** 2
        self.s = {"n_patches": 0.0, "sum_psnr": 0.0, "n_px": 0.0, "sum_sq": 0.0, "sum_ssim": 0.0}

    def _blur(self, x):
        return F.conv2d(F.conv2d(x, self.gh), self.gv)

    @torch.no_grad()
    def update(self, pred, gt):
        """pred, gt: (1, 1, H, W) metres, already clipped to the range. Returns (psnr, ssim) of the patch."""
        p, g = pred.double(), gt.double()
        sq = ((p - g) ** 2).sum().item()
        mse = sq / g.numel()
        psnr = 10.0 * math.log10(self.R ** 2 / max(mse, 1e-12))
        p, g = pred.float(), gt.float()
        mp, mg = self._blur(p), self._blur(g)
        vp = self._blur(p * p) - mp ** 2
        vg = self._blur(g * g) - mg ** 2
        cpg = self._blur(p * g) - mp * mg
        smap = ((2 * mp * mg + self.c1) * (2 * cpg + self.c2)) / ((mp ** 2 + mg ** 2 + self.c1) * (vp + vg + self.c2))
        ssim = smap.double().mean().item()
        s = self.s
        s["n_patches"] += 1
        s["sum_psnr"] += psnr
        s["n_px"] += g.numel()
        s["sum_sq"] += sq
        s["sum_ssim"] += ssim
        return psnr, ssim

    @staticmethod
    def finalize(states, data_range):
        s = {k: sum(st[k] for st in states) for k in states[0]}
        return {"psnr": s["sum_psnr"] / s["n_patches"],
                "psnr_global": 10.0 * math.log10(data_range ** 2 / (s["sum_sq"] / s["n_px"])),
                "ssim": s["sum_ssim"] / s["n_patches"]}


# -------------------------------------------------------------------------
# Images
# -------------------------------------------------------------------------

def save_panel(path, imgs, preds, gt, maes, title, vmax):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.0))
    row1 = [("Landsat", imgs["landsat"]), ("Real PlanetScope", imgs["ps"]),
            ("SR (baseline_sr4ir_reg)", imgs["sr_sr4ir"]), ("SR (sr_fm_refiner_v11)", imgs["sr_v11"])]
    for ax, (ttl, im) in zip(axes[0], row1):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(ttl, fontsize=10)
    axes[0, 4].imshow(gt, cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
    axes[0, 4].set_title("GT height (m)", fontsize=10)
    for ax, n in zip(axes[1], NAMES):
        ax.imshow(preds[n], cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{n}\nMAE {maes[n]:.2f} m", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, vmax))
    cax = axes[1, 4]
    cax.axis("off")
    fig.colorbar(sm, ax=cax, fraction=0.5, label="height (m)")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(h_pad=3.0)
    fig.savefig(path, dpi=100)
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default=OUT_ROOT)
    ap.add_argument("--data_cfg", default=f"{OUT_ROOT}/baseline_swt-lumi-nd/config.yaml",
                    help="run config whose dataset block defines the new dataset / test split")
    ap.add_argument("--ckpt", default="best", help="refiner checkpoint name (best|final|latest)")
    ap.add_argument("--out_dir", default="/flash/project_465002934/result_group_meeting")
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_avg", type=int, default=5)
    ap.add_argument("--edge_thr", type=float, default=2.0, help="edge = Sobel gradient >= this (m/px)")
    ap.add_argument("--edge_thrs", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                    help="thresholds (m/px) for the multi-threshold edge F1 table")
    ap.add_argument("--edge_tol", type=int, default=1, help="edge match tolerance in px (also reports 2x)")
    ap.add_argument("--hf_sigma", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N test samples")
    ap.add_argument("--no_vis", action="store_true", help="metrics only, no images")
    ap.add_argument("--save_dir", default=None,
                    help="also save <id>.npz per patch: raw GT and each model's raw prediction (float32 metres), "
                         "for offline metrics (script/depth/eval_std_metrics.py)")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_main = rank == 0
    dist_on = world > 1
    if not is_main:
        sys.stdout = open(os.devnull, "w")
        import functools
        import src.util.ps_lazydataset as _pld
        _pld.tqdm = functools.partial(_pld.tqdm, disable=True)
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

    M = load_models(args, device, log)
    cfg_d = OmegaConf.load(args.data_cfg).dataset
    # inputs must be normalised exactly as during the four models' training
    cfg_v11_d = OmegaConf.load(os.path.join(args.run_root, "sr_fm_refiner_v11-lumi", "config.yaml")).dataset
    for k in ("input_stats_file", "input_hr_stats_file", "selected_bands", "selected_bands_hr",
              "use_input_minmax", "use_input_norm", "use_input_hr_minmax", "use_input_hr_norm",
              "scale_input_to_neg1_1", "scale_input_hr_to_neg1_1", "year"):
        assert cfg_d[k] == cfg_v11_d[k], f"input preprocessing differs from training: {k}"
    assert not cfg_d.use_target_minmax and not cfg_d.use_target_norm, "expect raw-metre targets"
    target_stats = old._load_target_stats(M["target_stats_file"], cfg_d.year)
    p1, p99 = float(target_stats["p1"][0]), float(target_stats["p99"][0])

    test_ds, test_coords = nd.make_test_dataset(cfg_d)
    n_total = len(test_ds) if not args.limit else min(args.limit, len(test_ds))
    log(f"test samples: {n_total}; world size {world}; n_steps={args.n_steps} n_avg={args.n_avg}; "
        f"refiner ckpt={M['ck_name']}; range [{p1}, {p99}] m")
    local_idx = list(range(n_total))[rank::world]
    loader = DataLoader(Subset(test_ds, local_idx), batch_size=1, shuffle=False,
                        num_workers=cfg_d.workers, pin_memory=True)

    folders = {"landsat": "landsat", "ps": "ps", "gt": "gt", "panel": "panel",
               "sr_sr4ir": "baseline_sr4ir_reg_sr", "sr_v11": "sr_fm_refiner_v11-lumi_sr",
               **{n: n for n in NAMES}}
    if not args.no_vis:
        for sub in folders.values():
            os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    acc = {n: old.DetailAccumulator(device, args.edge_thr, args.edge_tol, args.hf_sigma) for n in NAMES}
    tols = (args.edge_tol, 2 * args.edge_tol)
    edge_acc = {n: nd.EdgeAccumulator(device, args.edge_thrs, tols) for n in NAMES}
    raw_acc = {n: RawAccumulator() for n in NAMES}
    iq_acc = {n: PsnrSsimAccumulator(device, p99 - p1) for n in NAMES}
    n_ls = M["n_ls"]
    index_rows = []

    def _avg(fn, seed):
        draws = []
        for k in range(args.n_avg):
            torch.manual_seed(seed * 1000 + k)
            draws.append(fn())
        return torch.stack(draws).mean(0)

    for sample_i, batch in zip(local_idx, tqdm(loader, disable=not is_main, desc="group-meeting-eval")):
        inputs_lr, inputs_hr, targets = batch
        inputs_lr = inputs_lr.squeeze(0) if inputs_lr.dim() == 5 else inputs_lr
        inputs_hr = inputs_hr.squeeze(0) if inputs_hr.dim() == 5 else inputs_hr
        targets = targets.squeeze(0) if targets.dim() == 5 else targets
        inputs_lr, inputs_hr, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
        H, W = targets.shape[-2:]
        landsat = inputs_lr[:, :n_ls]
        landsat_01 = (landsat + 1.0) / 2.0
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)

        with torch.no_grad():
            gt_raw = targets.float()                 # metres (dataset clamps to [0, 50])
            gt_m = gt_raw.clamp(p1, p99)

            # 1. baseline_dav2: Landsat -> height (raw metres), bilinear to the HR grid
            pred = M["dav2"](landsat_01)
            raw_dav2 = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)

            # 2. baseline_sr4ir_reg: SwinIR SR -> DAv2 regression (raw metres)
            img_sr4ir = M["sr4ir_sr"](landsat, target_size=(H, W))
            pred = M["sr4ir_reg"]((img_sr4ir + 1.0) / 2.0).predicted_depth.unsqueeze(1)
            raw_sr4ir = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False) \
                if pred.shape[-2:] != (H, W) else pred

            # shared coarse height of both refiners
            coarse_m = sr_fm_mod.run_dav2(M["coarse"], landsat_01, target_size=(H, W))
            coarse_n = old._normalize_coarse(coarse_m, target_stats)

            # 3. fm_refiner-R-lumi-4 (Landsat-only ControlNet, null PS)
            h = _avg(lambda: M["fm_r4"].refine(landsat_hr, coarse_n, n_steps=args.n_steps, method="euler"),
                     sample_i)
            raw_r4 = old._denormalize(h, target_stats)

            # 4. sr_fm_refiner_v11-lumi (pseudo-PS ControlNet, concat Landsat)
            pseudo_ps = M["sr_v11"](landsat, target_size=(H, W))
            h = _avg(lambda: M["fm_v11"].refine_with_ps(pseudo_ps, coarse_n, n_steps=args.n_steps,
                                                        method="euler", landsat_hr=landsat_hr),
                     sample_i)
            raw_v11 = old._denormalize(h, target_stats)

            raws = dict(zip(NAMES, (raw_dav2, raw_sr4ir, raw_r4, raw_v11)))
            preds = {n: r.float().clamp(p1, p99) for n, r in raws.items()}
            maes, psnrs, ssims = {}, {}, {}
            for n in NAMES:
                acc[n].update(preds[n], gt_m)
                edge_acc[n].update(preds[n], gt_m)
                raw_acc[n].update(raws[n].float(), gt_raw)
                maes[n] = (preds[n] - gt_m).abs().mean().item()
                psnrs[n], ssims[n] = iq_acc[n].update(preds[n], gt_m)

        c = test_coords[sample_i]
        pid = f"{sample_i:05d}_{c['tile_id'].replace(',', '-')}_y{c['output_crop_y']}_x{c['output_crop_x']}"
        index_rows.append({
            "id": pid, "test_idx": sample_i, "tile": c["tile_id"],
            "y": c["output_crop_y"], "x": c["output_crop_x"], "size": c["output_patch_height"],
            "gt_p75_m": float(np.percentile(gt_m.cpu().numpy(), 75)),
            **{f"mae_{n}": maes[n] for n in NAMES},
            **{f"psnr_{n}": psnrs[n] for n in NAMES},
            **{f"ssim_{n}": ssims[n] for n in NAMES},
        })
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            tmp = os.path.join(args.save_dir, pid + ".tmp.npz")
            np.savez_compressed(tmp, gt=gt_raw[0, 0].cpu().numpy().astype(np.float32),
                                **{n: raws[n][0, 0].float().cpu().numpy() for n in NAMES})
            os.replace(tmp, os.path.join(args.save_dir, pid + ".npz"))
        if not args.no_vis:
            landsat_nn = F.interpolate(landsat, size=(H, W), mode="nearest")
            imgs = {"landsat": nd._rgb_u8(landsat_nn[0]), "ps": nd._rgb_u8(inputs_hr[0]),
                    "sr_sr4ir": nd._rgb_u8(img_sr4ir[0]), "sr_v11": nd._rgb_u8(pseudo_ps[0])}
            for k, im in imgs.items():
                Image.fromarray(im).save(os.path.join(args.out_dir, folders[k], pid + ".png"))
            pred_np = {n: preds[n][0, 0].cpu().numpy() for n in NAMES}
            gt_np = gt_m[0, 0].cpu().numpy()
            Image.fromarray(nd._colormap_u8(gt_np, p99)).save(os.path.join(args.out_dir, "gt", pid + ".png"))
            for n in NAMES:
                Image.fromarray(nd._colormap_u8(pred_np[n], p99)).save(os.path.join(args.out_dir, n, pid + ".png"))
            save_panel(os.path.join(args.out_dir, "panel", pid + ".png"), imgs=imgs, preds=pred_np, gt=gt_np,
                       maes=maes, title=f"{pid}   (GT p75 = {index_rows[-1]['gt_p75_m']:.1f} m)", vmax=p99)

    payload = ({n: acc[n].state() for n in NAMES}, index_rows,
               {n: edge_acc[n].state() for n in NAMES}, {n: raw_acc[n].s for n in NAMES},
               {n: iq_acc[n].s for n in NAMES})
    if dist_on:
        gathered = [None] * world
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    if is_main:
        results = {n: old.finalize([g[0][n] for g in gathered]) for n in NAMES}
        for n in NAMES:
            results[n]["edge_multi_thr"] = nd.EdgeAccumulator.finalize([g[2][n] for g in gathered],
                                                                       args.edge_thrs, tols)
            results[n]["unclipped"] = RawAccumulator.finalize([g[3][n] for g in gathered])
            results[n].update(PsnrSsimAccumulator.finalize([g[4][n] for g in gathered], p99 - p1))
        os.makedirs(args.out_dir, exist_ok=True)
        meta = {"split": f"test ({cfg_d.test_coord_path})", "n_test_samples": n_total,
                "n_steps": args.n_steps, "n_avg": args.n_avg,
                "edge_thr_m_per_px": args.edge_thr, "edge_thrs_m_per_px": args.edge_thrs,
                "edge_tol_px": args.edge_tol, "hf_sigma_px": args.hf_sigma, "range_m": [p1, p99],
                "ckpts": {"baseline_dav2": f"{MODEL_ROOT}/baseline_dav2/best.pth",
                          "baseline_sr4ir_reg": f"{MODEL_ROOT}/baseline_sr4ir_reg/best.pth",
                          "fm_refiner-R-lumi-4": f"{args.run_root}/fm_refiner-R-lumi-4/checkpoint/{M['ck_name']}",
                          "sr_fm_refiner_v11-lumi": f"{args.run_root}/sr_fm_refiner_v11-lumi/checkpoint/{M['ck_name']}"}}
        with open(os.path.join(args.out_dir, "detail_metrics.json"), "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

        rows_all = sorted((r for g in gathered for r in g[1]), key=lambda r: r["test_idx"])
        with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
            w.writeheader()
            w.writerows(rows_all)

        t1, t2 = tols
        rows = [
            ("R2 (higher)", "r2", "{:.4f}"), ("MAE m (lower)", "mae", "{:.3f}"), ("RMSE m (lower)", "rmse", "{:.3f}"),
            ("Grad mag error m/px (lower)", "grad_mag_error", "{:.4f}"),
            ("Grad energy ratio (->1)", "grad_energy_ratio", "{:.4f}"),
            ("Grad magnitude ratio (->1)", "grad_mag_ratio", "{:.4f}"),
            (f"Edge F1 tol{t1}px (higher)", f"edge_f1_tol{t1}", "{:.4f}"),
            (f"Edge precision tol{t1}px", f"edge_precision_tol{t1}", "{:.4f}"),
            (f"Edge recall tol{t1}px", f"edge_recall_tol{t1}", "{:.4f}"),
            (f"Edge F1 tol{t2}px (higher)", f"edge_f1_tol{t2}", "{:.4f}"),
            ("HF energy ratio (->1)", "hf_energy_ratio", "{:.4f}"),
            ("HF correlation (higher)", "hf_corr", "{:.4f}"),
            (f"PSNR dB, mean over patches (higher; R = {p99 - p1:g} m)", "psnr", "{:.2f}"),
            ("PSNR dB, from pooled MSE (higher)", "psnr_global", "{:.2f}"),
            ("SSIM (higher)", "ssim", "{:.4f}"),
        ]
        lines = [f"Test split: {n_total} patches of 480 px; GT and predictions clipped to [{p1:g}, {p99:g}] m", "",
                 "| metric | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
        for label, key, fmt in rows:
            lines.append(f"| {label} | " + " | ".join(fmt.format(results[n][key]) for n in NAMES) + " |")
        lines += ["", "Unclipped (raw GT, raw predictions) R2 / MAE / RMSE", "",
                  "| metric | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
        for label, key, fmt in rows[:3]:
            lines.append(f"| {label} | " + " | ".join(fmt.format(results[n]["unclipped"][key]) for n in NAMES) + " |")
        lines += ["", "Height-binned error (by GT height): MAE / bias (pred-gt) / RMSE in m", "",
                  "| bin | pixel frac | " + " | ".join(NAMES) + " |", "|---|---|" + "---|" * len(NAMES)]
        for lab in results[NAMES[0]]["height_bins"]:
            fr = results[NAMES[0]]["height_bins"][lab]["pixel_frac"]
            cells = []
            for n in NAMES:
                b = results[n]["height_bins"][lab]
                cells.append(f"{b['mae']:.2f} / {b['bias']:+.2f} / {b['rmse']:.2f}")
            lines.append(f"| {lab} | {fr:.3f} | " + " | ".join(cells) + " |")
        lines += ["", "Edge F1 / precision / recall by gradient threshold (m/px), tolerance in px", "",
                  "| threshold | tol | GT edge pixels | " + " | ".join(NAMES) + " |",
                  "|---|---|---|" + "---|" * len(NAMES)]
        for thr in args.edge_thrs:
            for t in tols:
                gt_px = results[NAMES[0]]["edge_multi_thr"][f"thr{thr:g}_gt_edge_pixels"]
                cells = []
                for n in NAMES:
                    e = results[n]["edge_multi_thr"][f"thr{thr:g}_tol{t}"]
                    cells.append(f"{e['f1']:.3f} / {e['precision']:.3f} / {e['recall']:.3f}")
                lines.append(f"| {thr:g} | {t} | {gt_px:,.0f} | " + " | ".join(cells) + " |")
        lines += ["", f"GT mean gradient: {results[NAMES[0]]['gt_mean_grad']:.4f} m/px; GT edge pixel fraction "
                      f"(thr {args.edge_thr} m/px): {results[NAMES[0]]['edge_gt_frac']:.4f}"]
        table = "\n".join(lines)
        with open(os.path.join(args.out_dir, "detail_metrics.md"), "w") as f:
            f.write(table + "\n")
        print("\n" + table)

    if dist_on:
        dist.barrier()
        dist.destroy_process_group()
