"""
Detail-fidelity evaluation of run A (fm_refiner-R-lumi-5-nd-up8) on the TEST
split of the spatial split (split_spatial_v1_test.pkl), against its own frozen
coarse model, a sharpening control and two references. Ported from
sr_finer_lumi's eval_detail_metrics{,_nd}.py (same metric definitions).

Methods (all scored on the same test patches against the same GT, in metres,
predictions clipped to the target [p1, p99]):
  mu            baseline_chmv2-lumi-nd-up8-h120: run A's frozen coarse model
  mu_usm_a<a>   mu + a * (mu - G_usm_sigma(mu)): unsharp mask, adds detail
                energy but no information (the "free detail" control)
  runA_k<K>     run A, n_steps Euler steps, mean of the first K of max(K)
                draws (draw k seeded with test_idx * 1000 + k)
  fm_old        fm_refiner-R-lumi-5-nd (native-res coarse model, null PS)
  swt           baseline_swt-lumi-nd (SwinIR SR + CHMv2 head)

Metrics (pixel-pooled over the whole split):
  R2 / MAE / RMSE, and MAE / bias / RMSE per GT height bin
  gradient (Sobel/8, m/px, valid conv): |grad| error, energy and magnitude
      ratios pred/GT (amount of texture, 1 = GT's)
  edge precision / recall / F1: edge = |grad| >= thr; a predicted edge pixel
      is a hit if a GT edge pixel lies within tol px (precision) and vice versa
      (recall); several thresholds, tol = edge_tol and 2*edge_tol
  high frequency HF(x) = x - G_hf_sigma(x) (border of 3*sigma dropped):
      energy ratio (amount) and correlation with HF(GT) (location/phase,
      unchanged by sharpening when usm_sigma == hf_sigma); also per height bin

Outputs under --out_dir: detail_metrics.{json,md}, detail_tradeoff.png
(detail metric vs R2 per method), index.csv (per-patch MAE), and
images/<id>.png panels for every vis_every-th test patch.

Launch (one task per GPU): see script/depth/eval_detail_metrics_nd.sh
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from depthfm.chmv2 import CHMv2Height, load_chmv2_baseline
from depthfm.fm_refiner import build_fm_refiner
from depthfm.sr_module import build_sr_module
from src.util.ps_lazydataset import LazyPatchDataset
from train_fm_refiner import (_denormalize_target, _load_coords, _load_target_stats, _make_dataset,
                              _normalize_coarse)

OUT_ROOT = "/flash/project_465002934/Marigold_output"
RUN_MU = "baseline_chmv2-lumi-nd-up8-h120"
RUN_A = "fm_refiner-R-lumi-5-nd-up8"
RUN_FM_OLD = "fm_refiner-R-lumi-5-nd"
RUN_SWT = "baseline_swt-lumi-nd"
HEIGHT_BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # last bin is [30, inf)


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

def make_test_dataset(cfg_data):
    """Test split in file order, no augmentation, item idx == patch idx
    (the same dataset train_fm_refiner.py's final test uses)."""
    base = LazyPatchDataset({
        'input_dir': cfg_data.input_dir, 'input_dir_hr': cfg_data.input_dir_hr,
        'output_dir': cfg_data.output_dir,
        'selected_bands': cfg_data.selected_bands, 'selected_bands_hr': cfg_data.selected_bands_hr,
        'file_type_input': cfg_data.file_type_input, 'file_type_output': cfg_data.file_type_output,
        'patch_sizes': [(s, s) for s in cfg_data.patch_sizes],
        'patch_coord_path': cfg_data.patch_coord_path,
        'num_patches_per_tile': cfg_data.num_patches_per_tile, 'tile_emphasis': cfg_data.tile_emphasis,
        'correlation_cleaning': cfg_data.correlation_cleaning,
        'target_range_edges': cfg_data.target_range_edges,
        'num_patches_per_target_range': cfg_data.num_patches_per_target_range,
        'selected_percentile': cfg_data.selected_percentile, 'year': cfg_data.year,
        'batch_size': 1, 'mode': 'train',
        'use_input_minmax': cfg_data.use_input_minmax, 'use_input_norm': cfg_data.use_input_norm,
        'input_stats_file': cfg_data.input_stats_file,
        'use_input_hr_minmax': cfg_data.use_input_hr_minmax, 'use_input_hr_norm': cfg_data.use_input_hr_norm,
        'input_hr_stats_file': cfg_data.input_hr_stats_file,
        'use_target_minmax': cfg_data.use_target_minmax, 'use_target_norm': cfg_data.use_target_norm,
        'target_stats_file': cfg_data.target_stats_file,
        'unit_scale_ratio': cfg_data.unit_scale_ratio,
        'scale_input_to_neg1_1': cfg_data.get('scale_input_to_neg1_1', False),
        'scale_input_hr_to_neg1_1': cfg_data.get('scale_input_hr_to_neg1_1', False),
        'scale_target_to_neg1_1': cfg_data.get('scale_target_to_neg1_1', False),
    })
    coords = _load_coords(cfg_data.test_coord_path)
    return _make_dataset(base, coords, 'test', 1, deterministic_eval=True), coords


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------

def _torch_load(path):
    """mmap (if supported) so the multi-GB optimizer state is never read."""
    for kw in ({"mmap": True, "weights_only": False}, {"weights_only": False}, {}):
        try:
            return torch.load(path, map_location="cpu", **kw)
        except TypeError:
            continue
    raise RuntimeError("torch.load failed")


def _load_coarse(cfg, device):
    """The frozen CHMv2 coarse model a refiner run was trained with."""
    in_stats = _load_target_stats(cfg.dataset.input_stats_file, cfg.dataset.year)
    return load_chmv2_baseline(
        model_dir=cfg.model.chmv2_model_id, ckpt_path=cfg.model.chmv2_ckpt_path,
        mean=in_stats["mean"], std=in_stats["std"],
        out_in_scale_factor=cfg.model.chmv2_out_in_scale_factor,
        upsample_input=cfg.model.get("chmv2_upsample_input", False),
        input_scale=cfg.model.get("chmv2_input_scale", 1),
    ).to(device).eval()


def _load_refiner(run_dir, ck_name, device):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    t = cfg.trainer
    fm = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=len(cfg.dataset.selected_bands), n_ps_bands=len(cfg.dataset.selected_bands_hr),
        ps_dropout_p=0.0,
        use_controlnet=t.get("use_controlnet", True),
        controlnet_cond_mode=t.get("controlnet_cond_mode", "landsat_ps"),
        velocity_parameterization=t.get("velocity_parameterization", "fixed"),
        noise_sigma=t.get("noise_sigma", 0.0), sample_sigma=t.get("sample_sigma", None),
        sample_noise_mode=t.get("sample_noise_mode", "sde"),
        unet_init=cfg.model.get("unet_init", "pretrained"),
        unet_config=(OmegaConf.to_container(cfg.model.unet_config, resolve=True)
                     if cfg.model.get("unet_config") is not None else None),
        device=str(device),
    ).to(device)
    ck = _torch_load(os.path.join(run_dir, "checkpoint", ck_name))
    fm.unet.load_state_dict(ck["unet_state"])
    if fm.use_controlnet and ck.get("controlnet_state") is not None:
        fm.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm._null_ps = ck["null_ps_state"]
    step = ck.get("step")
    del ck
    return fm.eval(), cfg, step


def load_models(args, device, log):
    run = lambda n: os.path.join(args.run_root, n)
    ck_name = f"{args.ckpt}.pth"

    # mu, and run A whose frozen coarse model it must be
    cfg_mu = OmegaConf.load(os.path.join(run(RUN_MU), "config.yaml"))
    fm_a, cfg_a, step = _load_refiner(run(RUN_A), ck_name, device)
    mu_ckpt = os.path.join(run(RUN_MU), "checkpoint", "best.pth")
    assert cfg_a.model.chmv2_ckpt_path == mu_ckpt, f"run A's coarse model is {cfg_a.model.chmv2_ckpt_path}"
    m = cfg_mu.model
    assert (cfg_a.model.chmv2_out_in_scale_factor, cfg_a.model.chmv2_upsample_input,
            cfg_a.model.chmv2_input_scale) == (m.out_in_scale_factor, m.upsample_input_to_target, m.input_scale)
    mu = _load_coarse(cfg_a, device)
    log(f"loaded {RUN_A} (step={step}) and its coarse model {RUN_MU}")

    fm_old, cfg_old, step = _load_refiner(run(RUN_FM_OLD), ck_name, device)
    mu_old = _load_coarse(cfg_old, device)
    log(f"loaded {RUN_FM_OLD} (step={step}) and its coarse model {cfg_old.model.chmv2_ckpt_path}")

    cfg_swt = OmegaConf.load(os.path.join(run(RUN_SWT), "config.yaml"))
    sr = build_sr_module(OmegaConf.to_container(cfg_swt.sr_module, resolve=True)).to(device)
    ps_stats = _load_target_stats(cfg_swt.dataset.input_hr_stats_file, cfg_swt.dataset.year)
    reg = CHMv2Height(
        model_id=cfg_swt.model.chmv2_model_id,
        mean=[ps_stats["mean"][i] for i in cfg_swt.dataset.selected_bands_hr],
        std=[ps_stats["std"][i] for i in cfg_swt.dataset.selected_bands_hr],
        pretrained=False, freeze_backbone=False, out_in_scale_factor=1,
    ).to(device)
    ck = _torch_load(os.path.join(run(RUN_SWT), "checkpoint", ck_name))
    sr.load_state_dict(ck["sr_state"])
    reg.load_state_dict(ck["reg_state"])
    log(f"loaded {RUN_SWT} (step={ck.get('step')})")
    del ck
    return dict(cfg=cfg_a, mu=mu, fm_a=fm_a, mu_old=mu_old, fm_old=fm_old,
                sr=sr.eval(), reg=reg.eval(), ck_name=ck_name)


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0


class GaussBlur:
    """Separable Gaussian blur, reflect padding, same size out."""

    def __init__(self, sigma, device):
        self.r = int(math.ceil(3 * sigma))
        x = torch.arange(-self.r, self.r + 1, dtype=torch.float32)
        k = torch.exp(-x ** 2 / (2 * sigma ** 2))
        k = (k / k.sum()).to(device)
        self.kh, self.kv = k.view(1, 1, 1, -1), k.view(1, 1, -1, 1)

    def __call__(self, x):
        r = self.r
        return F.conv2d(F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), self.kh), self.kv)


class DetailAccumulator:
    """Streaming pixel-pooled sums for one method, kept on the device in float64."""

    BIN_KEYS = ("cnt", "sum_abs", "sum_err", "sum_sq", "n_hf", "sum_hp2", "sum_hg2", "sum_hpg")

    def __init__(self, device, edge_thrs, edge_tols, hf_sigma):
        self.edge_thrs, self.edge_tols = list(edge_thrs), list(edge_tols)
        self.kx = _SOBEL_X.view(1, 1, 3, 3).to(device)
        self.ky = _SOBEL_X.t().contiguous().view(1, 1, 3, 3).to(device)
        self.blur = GaussBlur(hf_sigma, device)
        self.bin_edges = torch.tensor(HEIGHT_BIN_EDGES, dtype=torch.float64, device=device)
        self.s = defaultdict(lambda: torch.zeros((), dtype=torch.float64, device=device))
        self.bins = torch.zeros(len(self.BIN_KEYS), len(HEIGHT_BIN_EDGES), dtype=torch.float64, device=device)

    def _grad(self, x):
        return torch.sqrt(F.conv2d(x, self.kx) ** 2 + F.conv2d(x, self.ky) ** 2)

    def _hf(self, x):
        r = self.blur.r
        return (x - self.blur(x))[..., r:-r, r:-r]

    @staticmethod
    def _dilate(m, r):
        return F.max_pool2d(m, kernel_size=2 * r + 1, stride=1, padding=r) > 0

    def _bincount(self, row, b, w):
        self.bins[row] += torch.bincount(b, weights=w, minlength=len(HEIGHT_BIN_EDGES))

    @torch.no_grad()
    def update(self, pred, gt):
        """pred, gt: (1, 1, H, W) float32 metres."""
        s = self.s
        p, g = pred.double(), gt.double()
        err = p - g
        s["n"] += g.numel()
        s["sum_gt"] += g.sum()
        s["sum_gt2"] += (g ** 2).sum()
        s["sum_abs"] += err.abs().sum()
        s["sum_sq"] += (err ** 2).sum()

        gp, gg = self._grad(pred), self._grad(gt)
        gp64, gg64 = gp.double(), gg.double()
        s["n_g"] += gp.numel()
        s["sum_gp"] += gp64.sum()
        s["sum_gg"] += gg64.sum()
        s["sum_gdiff"] += (gp64 - gg64).abs().sum()
        s["sum_gp2"] += (gp64 ** 2).sum()
        s["sum_gg2"] += (gg64 ** 2).sum()

        for thr in self.edge_thrs:
            ep, eg = gp >= thr, gg >= thr
            s[f"pred_edges_{thr:g}"] += ep.sum()
            s[f"gt_edges_{thr:g}"] += eg.sum()
            epf, egf = ep.float(), eg.float()
            for t in self.edge_tols:
                s[f"prec_hit_{thr:g}_{t}"] += (ep & self._dilate(egf, t)).sum()
                s[f"rec_hit_{thr:g}_{t}"] += (eg & self._dilate(epf, t)).sum()

        hp, hg = self._hf(pred).double(), self._hf(gt).double()
        s["n_hf"] += hp.numel()
        s["sum_hp2"] += (hp ** 2).sum()
        s["sum_hg2"] += (hg ** 2).sum()
        s["sum_hpg"] += (hp * hg).sum()

        # per GT-height bin (the HF terms binned on the cropped GT)
        b = (torch.bucketize(g.flatten(), self.bin_edges, right=True) - 1).clamp(min=0)
        e = err.flatten()
        self._bincount(0, b, None)
        self._bincount(1, b, e.abs())
        self._bincount(2, b, e)
        self._bincount(3, b, e ** 2)
        r = self.blur.r
        bh = (torch.bucketize(g[..., r:-r, r:-r].flatten(), self.bin_edges, right=True) - 1).clamp(min=0)
        hp, hg = hp.flatten(), hg.flatten()
        self._bincount(4, bh, None)
        self._bincount(5, bh, hp ** 2)
        self._bincount(6, bh, hg ** 2)
        self._bincount(7, bh, hp * hg)

    def state(self):
        return {"s": {k: v.item() for k, v in self.s.items()}, "bins": self.bins.cpu().tolist()}


def finalize(states, edge_thrs, edge_tols):
    """Sum per-rank states and compute the reported metrics."""
    s = {k: sum(st["s"][k] for st in states) for k in states[0]["s"]}
    bins = torch.tensor([st["bins"] for st in states], dtype=torch.float64).sum(0)
    out = {}
    ss_tot = s["sum_gt2"] - s["sum_gt"] ** 2 / s["n"]
    out["r2"] = 1.0 - s["sum_sq"] / ss_tot
    out["mae"] = s["sum_abs"] / s["n"]
    out["rmse"] = math.sqrt(s["sum_sq"] / s["n"])
    out["grad_mag_error"] = s["sum_gdiff"] / s["n_g"]
    out["grad_energy_ratio"] = s["sum_gp2"] / s["sum_gg2"]
    out["grad_mag_ratio"] = s["sum_gp"] / s["sum_gg"]
    out["gt_mean_grad"] = s["sum_gg"] / s["n_g"]
    for thr in edge_thrs:
        out[f"edge_gt_frac_thr{thr:g}"] = s[f"gt_edges_{thr:g}"] / s["n_g"]
        out[f"edge_pred_frac_thr{thr:g}"] = s[f"pred_edges_{thr:g}"] / s["n_g"]
        for t in edge_tols:
            prec = s[f"prec_hit_{thr:g}_{t}"] / max(s[f"pred_edges_{thr:g}"], 1.0)
            rec = s[f"rec_hit_{thr:g}_{t}"] / max(s[f"gt_edges_{thr:g}"], 1.0)
            out[f"edge_precision_thr{thr:g}_tol{t}"] = prec
            out[f"edge_recall_thr{thr:g}_tol{t}"] = rec
            out[f"edge_f1_thr{thr:g}_tol{t}"] = 2 * prec * rec / max(prec + rec, 1e-12)
    out["hf_energy_ratio"] = s["sum_hp2"] / s["sum_hg2"]
    out["hf_corr"] = s["sum_hpg"] / math.sqrt(s["sum_hp2"] * s["sum_hg2"])
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip(HEIGHT_BIN_EDGES[:-1], HEIGHT_BIN_EDGES[1:])]
    labels.append(f"[{HEIGHT_BIN_EDGES[-1]:g},inf)")
    out["height_bins"] = {}
    for i, lab in enumerate(labels):
        c, ch = bins[0, i].item(), bins[4, i].item()
        if not c:
            continue
        hp2, hg2, hpg = bins[5, i].item(), bins[6, i].item(), bins[7, i].item()
        out["height_bins"][lab] = {
            "pixel_frac": c / s["n"],
            "mae": bins[1, i].item() / c,
            "bias": bins[2, i].item() / c,
            "rmse": math.sqrt(bins[3, i].item() / c),
            "hf_energy_ratio": hp2 / hg2 if ch and hg2 else float("nan"),
            "hf_corr": hpg / math.sqrt(hp2 * hg2) if ch and hp2 and hg2 else float("nan"),
        }
    return out


# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

# categorical slots 1-4 of the dataviz reference palette; mu in secondary ink
C_RUNA, C_USM, C_OLD, C_SWT, C_MU = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e"


def plot_tradeoff(results, names, args, path):
    """Each detail metric against R2: the unsharp-mask curve (mu at a=0) is what
    sharpening alone buys; run A's K curve is compared with it at equal R2."""
    thr, tol = f"{args.edge_thr:g}", args.edge_tol
    panels = [("hf_corr", "HF correlation with GT (higher)"),
              ("hf_energy_ratio", "HF energy ratio pred/GT (→1)"),
              ("grad_energy_ratio", "Gradient energy ratio pred/GT (→1)"),
              (f"edge_f1_thr{thr}_tol{tol}", f"Edge F1, |∇| ≥ {thr} m/px, tol {tol} px (higher)"),
              (f"edge_precision_thr{thr}_tol{tol}", f"Edge precision, {thr} m/px, tol {tol} px (higher)"),
              ("grad_mag_error", "Gradient magnitude error, m/px (lower)")]
    usm = ["mu"] + [n for n in names if n.startswith("mu_usm_")]
    runa = [n for n in names if n.startswith("runA_k")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, title) in zip(axes.flat, panels):
        xy = lambda ns: ([results[n]["r2"] for n in ns], [results[n][key] for n in ns])
        ax.plot(*xy(usm), "-o", color=C_USM, lw=2, ms=8, label="μ + unsharp mask (α sweep)")
        for n, a in zip(usm[1:], args.usm_alphas):
            ax.annotate(f"α={a:g}", (results[n]["r2"], results[n][key]), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=C_MU)
        ax.plot(*xy(runa), "-o", color=C_RUNA, lw=2, ms=8, label="run A, mean of K draws")
        for n, k in zip(runa, args.ks):
            ax.annotate(f"K={k}", (results[n]["r2"], results[n][key]), textcoords="offset points",
                        xytext=(5, -12), fontsize=8, color=C_MU)
        for n, mk, c, lab in (("mu", "s", C_MU, "μ (up8-h120)"), ("fm_old", "D", C_OLD, RUN_FM_OLD),
                              ("swt", "^", C_SWT, RUN_SWT)):
            ax.plot(results[n]["r2"], results[n][key], mk, color=c, ms=10, mec="white", mew=1.5,
                    label=lab, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("R² (higher)", fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle(f"Detail vs accuracy on TEST ({args.n_steps} Euler steps; HF σ={args.hf_sigma:g} px, "
                 f"unsharp σ={args.usm_sigma:g} px)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_tables(results, names, args, path):
    thr, t1, t2 = f"{args.edge_thr:g}", args.edge_tol, 2 * args.edge_tol
    cols = [("R2", "r2", "{:.4f}"), ("MAE", "mae", "{:.3f}"), ("RMSE", "rmse", "{:.3f}"),
            ("HF corr ↑", "hf_corr", "{:.4f}"), ("HF energy →1", "hf_energy_ratio", "{:.4f}"),
            ("grad energy →1", "grad_energy_ratio", "{:.4f}"), ("grad mag →1", "grad_mag_ratio", "{:.4f}"),
            ("grad err ↓", "grad_mag_error", "{:.4f}"),
            (f"edge F1 tol{t1} ↑", f"edge_f1_thr{thr}_tol{t1}", "{:.4f}"),
            (f"edge P tol{t1}", f"edge_precision_thr{thr}_tol{t1}", "{:.4f}"),
            (f"edge R tol{t1}", f"edge_recall_thr{thr}_tol{t1}", "{:.4f}"),
            (f"edge F1 tol{t2} ↑", f"edge_f1_thr{thr}_tol{t2}", "{:.4f}")]
    L = [f"Edge threshold {thr} m/px for the edge columns; HF sigma {args.hf_sigma:g} px; "
         f"unsharp sigma {args.usm_sigma:g} px", "",
         "| method | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(fmt.format(results[n][k]) for _, k, fmt in cols) + " |")

    bins = list(results[names[0]]["height_bins"])
    fr = " | ".join(f"{results[names[0]]['height_bins'][b]['pixel_frac']:.3f}" for b in bins)
    for title, key, fmt in (("RMSE (m)", "rmse", "{:.2f}"), ("bias pred-GT (m)", "bias", "{:+.2f}"),
                            ("HF correlation", "hf_corr", "{:.3f}"), ("HF energy ratio", "hf_energy_ratio", "{:.3f}")):
        L += ["", f"{title} by GT height bin", "", "| method | " + " | ".join(bins) + " |",
              "|---|" + "---|" * len(bins), f"| pixel frac | {fr} |"]
        for n in names:
            L.append(f"| {n} | " + " | ".join(fmt.format(results[n]["height_bins"][b][key]) for b in bins) + " |")

    L += ["", "Edge F1 / precision / recall by gradient threshold (m/px) and tolerance (px)", "",
          "| method | " + " | ".join(f"thr {h:g} tol {t}" for h in args.edge_thrs for t in (t1, t2)) + " |",
          "|---|" + "---|" * (2 * len(args.edge_thrs))]
    for n in names:
        cells = []
        for h in args.edge_thrs:
            for t in (t1, t2):
                k = f"thr{h:g}_tol{t}"
                cells.append(f"{results[n]['edge_f1_' + k]:.3f} / {results[n]['edge_precision_' + k]:.3f} / "
                             f"{results[n]['edge_recall_' + k]:.3f}")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    r0 = results[names[0]]
    L += ["", f"GT mean gradient {r0['gt_mean_grad']:.4f} m/px; GT edge pixel fraction: " +
          ", ".join(f"{r0[f'edge_gt_frac_thr{h:g}']:.4f} (thr {h:g})" for h in args.edge_thrs)]
    table = "\n".join(L)
    with open(path, "w") as f:
        f.write(table + "\n")
    return table


def _rgb(t):
    """(C,H,W) tensor in [-1,1] -> (H,W,3) float in [0,1]."""
    return ((t[:3].float().cpu() + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy()


def save_panel(path, landsat, ps, gt, preds, maes, title, vmax):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
    axes[0, 0].imshow(landsat, interpolation="nearest")
    axes[0, 0].set_title("Landsat (bilinear)", fontsize=10)
    axes[0, 1].imshow(ps, interpolation="nearest")
    axes[0, 1].set_title("PlanetScope (not used)", fontsize=10)
    axes[0, 2].imshow(gt, cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
    axes[0, 2].set_title("GT height (m)", fontsize=10)
    for ax, n in zip(list(axes[0, 3:]) + list(axes[1, :4]), preds):
        ax.imshow(preds[n], cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{n}\nMAE {maes[n]:.2f} m", fontsize=9)
    axes[1, 4].axis("off")
    fig.colorbar(plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, vmax)), ax=axes[1, 4],
                 fraction=0.5, label="height (m)")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(h_pad=3.0)
    fig.savefig(path, dpi=90)
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default=OUT_ROOT)
    ap.add_argument("--ckpt", default="best", help="checkpoint name in each run's checkpoint/ dir")
    ap.add_argument("--out_dir", default=f"{OUT_ROOT}/_detail_eval_nd_up8")
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10],
                    help="run A: report the mean of the first K draws for each K (max(K) draws are run)")
    ap.add_argument("--n_avg_old", type=int, default=5, help="draws averaged for fm_old")
    ap.add_argument("--usm_alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--usm_sigma", type=float, default=2.0, help="unsharp-mask Gaussian sigma (px)")
    ap.add_argument("--edge_thr", type=float, default=2.0, help="edge threshold (m/px) of the main table")
    ap.add_argument("--edge_thrs", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--edge_tol", type=int, default=1, help="edge match tolerance in px (also reports 2x)")
    ap.add_argument("--hf_sigma", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N test samples")
    ap.add_argument("--vis_every", type=int, default=20, help="save a panel for every N-th test patch (0: none)")
    args = ap.parse_args()
    args.ks = sorted(set(args.ks))
    if args.edge_thr not in args.edge_thrs:
        args.edge_thrs = sorted(args.edge_thrs + [args.edge_thr])

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_main = rank == 0
    if not is_main:
        sys.stdout = open(os.devnull, "w")
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.info if is_main else (lambda msg: None)

    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=30))
    idx = local % max(1, torch.cuda.device_count())
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}")
    # Same as the training scripts (MIOpen find-db write otherwise kills the first conv).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    M = load_models(args, device, log)
    cfg_d = M["cfg"].dataset
    target_stats = _load_target_stats(cfg_d.target_stats_file, cfg_d.year)
    p1, p99 = float(target_stats["p1"][0]), float(target_stats["p99"][0])
    n_ls = len(cfg_d.selected_bands)
    test_ds, test_coords = make_test_dataset(cfg_d)
    n_total = min(args.limit, len(test_ds)) if args.limit else len(test_ds)
    log(f"test samples: {n_total}; world size {world}; n_steps={args.n_steps} ks={args.ks} "
        f"n_avg_old={args.n_avg_old}; ckpt={M['ck_name']}")
    local_idx = list(range(n_total))[rank::world]
    loader = DataLoader(Subset(test_ds, local_idx), batch_size=1, shuffle=False,
                        num_workers=cfg_d.workers, pin_memory=True)

    names = (["mu"] + [f"mu_usm_a{a:g}" for a in args.usm_alphas] + [f"runA_k{k}" for k in args.ks]
             + ["fm_old", "swt"])
    vis_names = ["mu", f"mu_usm_a{args.usm_alphas[len(args.usm_alphas) // 2]:g}",
                 f"runA_k{args.ks[0]}", f"runA_k{args.ks[-1]}", "fm_old", "swt"]
    img_dir = os.path.join(args.out_dir, "images")
    if args.vis_every and is_main:
        os.makedirs(img_dir, exist_ok=True)
    tols = (args.edge_tol, 2 * args.edge_tol)
    acc = {n: DetailAccumulator(device, args.edge_thrs, tols, args.hf_sigma) for n in names}
    usm_blur = GaussBlur(args.usm_sigma, device)
    index_rows = []

    for sample_i, batch in zip(local_idx, tqdm(loader, disable=not is_main, desc="detail-eval-test")):
        inputs_lr, inputs_hr, targets = (t.squeeze(0) if t.dim() == 5 else t for t in batch)
        inputs_lr, inputs_hr, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
        H, W = targets.shape[-2:]
        landsat = inputs_lr[:, :n_ls]
        landsat01 = (landsat + 1.0) / 2.0
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)
        preds = {}

        with torch.no_grad():
            gt_m = _denormalize_target(targets, target_stats)

            mu_m = M["mu"](landsat01, target_size=(H, W))
            mu_c = mu_m.clamp(p1, p99)
            preds["mu"] = mu_c
            for a in args.usm_alphas:
                preds[f"mu_usm_a{a:g}"] = (mu_c + a * (mu_c - usm_blur(mu_c))).clamp(p1, p99)

            # run A: nested means of one sequence of draws
            coarse_n = _normalize_coarse(mu_m, target_stats)
            h_sum = 0.0
            for k in range(args.ks[-1]):
                torch.manual_seed(sample_i * 1000 + k)
                h_sum = h_sum + M["fm_a"].refine(landsat_hr, coarse_n, n_steps=args.n_steps, method="euler")
                if k + 1 in args.ks:
                    preds[f"runA_k{k + 1}"] = _denormalize_target(h_sum / (k + 1), target_stats).clamp(p1, p99)

            coarse_n = _normalize_coarse(M["mu_old"](landsat01, target_size=(H, W)), target_stats)
            h_sum = 0.0
            for k in range(args.n_avg_old):
                torch.manual_seed(sample_i * 1000 + 500 + k)
                h_sum = h_sum + M["fm_old"].refine(landsat_hr, coarse_n, n_steps=args.n_steps, method="euler")
            preds["fm_old"] = _denormalize_target(h_sum / args.n_avg_old, target_stats).clamp(p1, p99)

            img_sr = M["sr"](landsat, target_size=(H, W))
            preds["swt"] = M["reg"]((img_sr + 1.0) / 2.0, target_size=(H, W)).clamp(p1, p99)

            maes = {}
            for n in names:
                acc[n].update(preds[n].float(), gt_m.float())
                maes[n] = (preds[n].float() - gt_m.float()).abs().mean().item()

        c = test_coords[sample_i]
        pid = f"{sample_i:05d}_{c['tile_id'].replace(',', '-')}_y{c['output_crop_y']}_x{c['output_crop_x']}"
        gt_p75 = float(np.percentile(gt_m.cpu().numpy(), 75))
        index_rows.append({"id": pid, "test_idx": sample_i, "tile": c["tile_id"], "y": c["output_crop_y"],
                           "x": c["output_crop_x"], "size": c["output_patch_height"], "gt_p75_m": gt_p75,
                           **{f"mae_{n}": maes[n] for n in names}})
        if args.vis_every and sample_i % args.vis_every == 0:
            save_panel(os.path.join(img_dir, pid + ".png"), _rgb(landsat_hr[0]), _rgb(inputs_hr[0]),
                       gt_m[0, 0].cpu().numpy(), {n: preds[n][0, 0].cpu().numpy() for n in vis_names},
                       maes, f"{pid}   (GT p75 = {gt_p75:.1f} m)", vmax=p99)

    payload = ({n: acc[n].state() for n in names}, index_rows)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    if is_main:
        results = {n: finalize([g[0][n] for g in gathered], args.edge_thrs, tols) for n in names}
        os.makedirs(args.out_dir, exist_ok=True)
        meta = {"split": "test (split_spatial_v1_test.pkl)", "n_test_samples": n_total, "ckpt": M["ck_name"],
                "n_steps": args.n_steps, "ks": args.ks, "n_avg_old": args.n_avg_old,
                "usm_alphas": args.usm_alphas, "usm_sigma_px": args.usm_sigma, "hf_sigma_px": args.hf_sigma,
                "edge_thr_m_per_px": args.edge_thr, "edge_thrs_m_per_px": args.edge_thrs,
                "edge_tols_px": list(tols), "range_m": [p1, p99],
                "runs": {"mu": RUN_MU, "runA": RUN_A, "fm_old": RUN_FM_OLD, "swt": RUN_SWT}}
        with open(os.path.join(args.out_dir, "detail_metrics.json"), "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        rows = sorted((r for g in gathered for r in g[1]), key=lambda r: r["test_idx"])
        with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        plot_tradeoff(results, names, args, os.path.join(args.out_dir, "detail_tradeoff.png"))
        print("\n" + write_tables(results, names, args, os.path.join(args.out_dir, "detail_metrics.md")))

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
