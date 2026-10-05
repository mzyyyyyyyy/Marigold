"""
Detail-fidelity evaluation of three models on the FULL validation split:

  baseline_chmv2-lumi   : fine-tuned CHMv2 (also the frozen coarse model of both refiners)
  fm_refiner-R-lumi-5   : FM refiner (finer_lumi code, depthfm/fm_refiner_finer.py), Landsat-only ControlNet
  sr_fm_refiner_v12-lumi: SR + FM refiner (this branch's depthfm/fm_refiner.py), pseudo-PS ControlNet

FM models are run exactly like their final validation (10 Euler steps, mean of
5 independent stochastic draws). All models see the same patches (one dataset,
same split as training), and are scored against the same ground truth, in metres.

Metrics (pixel-pooled over the whole split; patches are 240 or 480 px):
  1. gradient magnitude error  GME  = mean | |grad pred| - |grad gt| |   (m/px)
  2. gradient energy ratio     GER  = sum |grad pred|^2 / sum |grad gt|^2  (1 = same sharpness)
     (also the plain magnitude ratio, and the gt mean gradient for reference)
  3. edge F1: edge = Sobel gradient magnitude >= edge_thr (m/px); a predicted edge
     pixel is correct if a gt edge pixel lies within edge_tol px (precision), and
     vice versa (recall); reported for tol = edge_tol and 2*edge_tol
  4. high-frequency energy ratio HFR = sum HF(pred)^2 / sum HF(gt)^2, plus the
     correlation of HF(pred) with HF(gt); HF(x) = x - GaussianBlur(x, hf_sigma)
  5. height-binned error (bins by GT height): MAE / bias / RMSE per bin
  (plus overall R2 / MAE / RMSE as a sanity check against the training logs)

Gradient = Sobel/8 (metres per pixel), computed with valid convolution (no border).
GT is the dataset target (minmax-scaled in training => clipped to [p1, p99] metres),
predictions are clipped to the same range, so all models share one value range.

Launch (one task per GPU): see script/depth/eval_detail_metrics.sh
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from depthfm import fm_refiner as sr_fm_mod            # this branch (SR + pseudo-PS ControlNet)
from depthfm import fm_refiner_finer as plain_fm_mod   # finer_lumi branch (Landsat-only ControlNet)
from depthfm.chmv2 import load_chmv2_baseline
from depthfm.sr_module import build_sr_module
from src.util.ps_lazydataset import LazyPatchDataset

OUT_ROOT = "/flash/project_465002934/Marigold_output"
HEIGHT_BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # last bin is [30, inf)


# -------------------------------------------------------------------------
# Data helpers (same split logic as train_*_fm_refiner.py)
# -------------------------------------------------------------------------

def _split_coords(all_coords, train_split, val_split, seed):
    n = len(all_coords)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()
    train_size = int(train_split * n)
    val_size = int(val_split * n)
    return indices[train_size:train_size + val_size], all_coords


def _make_val_dataset(cfg_data):
    base = LazyPatchDataset({
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
        'mode': 'val',
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
    all_coords = base.patch_coords
    idx, _ = _split_coords(all_coords, cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed)
    coords = [all_coords[i] for i in idx]
    ds = copy.copy(base)
    ds.batch_size = 1
    ds.mode = 'val'
    ds.patch_coords = coords
    ds.patches_by_size = {}
    for ps in ds.patch_sizes:
        ds.patches_by_size[ps] = [
            p for p in coords
            if p['output_patch_height'] == ps[0] and p['output_patch_width'] == ps[1]
        ]
    return ds


def _load_target_stats(path, year):
    with open(path) as f:
        return json.load(f)[str(year)]


def _normalize_coarse(h, stats):
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    return (((h - p1) / (p99 - p1 + 1e-8)) * 2.0 - 1.0).clamp(-1.0, 1.0)


def _denormalize(h, stats):
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    return (h.float() + 1.0) / 2.0 * (p99 - p1) + p1


def _torch_load(path):
    """Load a checkpoint; mmap (if supported) so the multi-GB optimizer state is never read."""
    for kw in ({"mmap": True, "weights_only": False}, {"weights_only": False}, {}):
        try:
            return torch.load(path, map_location="cpu", **kw)
        except TypeError:
            continue
    raise RuntimeError("torch.load failed")


# -------------------------------------------------------------------------
# Metric accumulation
# -------------------------------------------------------------------------

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0


def _gaussian_kernel(sigma):
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum(), r


class DetailAccumulator:
    """Streaming pixel-pooled sums for one model; all sums kept in float64."""

    def __init__(self, device, edge_thr, edge_tol, hf_sigma):
        self.device = device
        self.edge_thr = edge_thr
        self.tols = (edge_tol, 2 * edge_tol)
        self.kx = _SOBEL_X.view(1, 1, 3, 3).to(device)
        self.ky = _SOBEL_X.t().contiguous().view(1, 1, 3, 3).to(device)
        k1, self.hf_r = _gaussian_kernel(hf_sigma)
        self.gk_h = k1.view(1, 1, 1, -1).to(device)
        self.gk_v = k1.view(1, 1, -1, 1).to(device)
        n_bins = len(HEIGHT_BIN_EDGES)
        self.s = {k: 0.0 for k in (
            "n", "sum_gt", "sum_gt2", "sum_abs", "sum_sq",
            "n_g", "sum_gp", "sum_gg", "sum_gdiff", "sum_gp2", "sum_gg2",
            "pred_edges", "gt_edges", "n_hf", "sum_hp2", "sum_hg2", "sum_hpg")}
        for t in self.tols:
            self.s[f"prec_hit_{t}"] = 0.0
            self.s[f"rec_hit_{t}"] = 0.0
        self.bins = torch.zeros(4, n_bins, dtype=torch.float64)  # cnt, sum_abs, sum_err, sum_sq

    def _grad(self, x):
        return torch.sqrt(F.conv2d(x, self.kx) ** 2 + F.conv2d(x, self.ky) ** 2)

    def _hf(self, x):
        r = self.hf_r
        xp = F.pad(x, (r, r, r, r), mode="reflect")
        blur = F.conv2d(F.conv2d(xp, self.gk_h), self.gk_v)
        return (x - blur)[..., r:-r, r:-r]

    @staticmethod
    def _dilate(m, r):
        return F.max_pool2d(m, kernel_size=2 * r + 1, stride=1, padding=r) > 0

    @torch.no_grad()
    def update(self, pred, gt):
        """pred, gt: (1, 1, H, W) float32 metres."""
        p, g = pred.double(), gt.double()
        err = p - g
        s = self.s
        s["n"] += g.numel()
        s["sum_gt"] += g.sum().item()
        s["sum_gt2"] += (g ** 2).sum().item()
        s["sum_abs"] += err.abs().sum().item()
        s["sum_sq"] += (err ** 2).sum().item()

        # gradient-based
        gp, gg = self._grad(pred), self._grad(gt)
        gp64, gg64 = gp.double(), gg.double()
        s["n_g"] += gp.numel()
        s["sum_gp"] += gp64.sum().item()
        s["sum_gg"] += gg64.sum().item()
        s["sum_gdiff"] += (gp64 - gg64).abs().sum().item()
        s["sum_gp2"] += (gp64 ** 2).sum().item()
        s["sum_gg2"] += (gg64 ** 2).sum().item()

        # edges
        ep, eg = gp >= self.edge_thr, gg >= self.edge_thr
        s["pred_edges"] += ep.sum().item()
        s["gt_edges"] += eg.sum().item()
        epf, egf = ep.float(), eg.float()
        for t in self.tols:
            s[f"prec_hit_{t}"] += (ep & self._dilate(egf, t)).sum().item()
            s[f"rec_hit_{t}"] += (eg & self._dilate(epf, t)).sum().item()

        # high frequency
        hp, hg = self._hf(pred).double(), self._hf(gt).double()
        s["n_hf"] += hp.numel()
        s["sum_hp2"] += (hp ** 2).sum().item()
        s["sum_hg2"] += (hg ** 2).sum().item()
        s["sum_hpg"] += (hp * hg).sum().item()

        # height bins (by GT height)
        edges = torch.tensor(HEIGHT_BIN_EDGES, dtype=torch.float64, device=g.device)
        b = (torch.bucketize(g.flatten(), edges, right=True) - 1).clamp(min=0)
        nb = len(HEIGHT_BIN_EDGES)
        e = err.flatten()
        self.bins[0] += torch.bincount(b, minlength=nb).double().cpu()
        self.bins[1] += torch.bincount(b, weights=e.abs(), minlength=nb).cpu()
        self.bins[2] += torch.bincount(b, weights=e, minlength=nb).cpu()
        self.bins[3] += torch.bincount(b, weights=e ** 2, minlength=nb).cpu()

    def state(self):
        return {"s": self.s, "bins": self.bins.tolist()}


def finalize(states):
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
    out["edge_pred_frac"] = s["pred_edges"] / s["n_g"]
    out["edge_gt_frac"] = s["gt_edges"] / s["n_g"]
    for key in [k for k in s if k.startswith("prec_hit_")]:
        t = key.split("_")[-1]
        prec = s[key] / max(s["pred_edges"], 1.0)
        rec = s[f"rec_hit_{t}"] / max(s["gt_edges"], 1.0)
        out[f"edge_precision_tol{t}"] = prec
        out[f"edge_recall_tol{t}"] = rec
        out[f"edge_f1_tol{t}"] = 2 * prec * rec / max(prec + rec, 1e-12)
    out["hf_energy_ratio"] = s["sum_hp2"] / s["sum_hg2"]
    out["hf_corr"] = s["sum_hpg"] / math.sqrt(s["sum_hp2"] * s["sum_hg2"])
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip(HEIGHT_BIN_EDGES[:-1], HEIGHT_BIN_EDGES[1:])]
    labels.append(f"[{HEIGHT_BIN_EDGES[-1]:g},inf)")
    out["height_bins"] = {}
    for i, lab in enumerate(labels):
        c = bins[0, i].item()
        out["height_bins"][lab] = {
            "pixel_frac": c / s["n"],
            "mae": bins[1, i].item() / c if c else float("nan"),
            "bias": bins[2, i].item() / c if c else float("nan"),
            "rmse": math.sqrt(bins[3, i].item() / c) if c else float("nan"),
        }
    return out


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------

def load_models(args, device, is_main):
    """Returns dict with coarse CHMv2 baseline and the two refiners (all eval, frozen)."""
    cfg_fm = OmegaConf.load(os.path.join(args.fm_run, "config.yaml"))      # fm_refiner-R-lumi-5
    cfg_sr = OmegaConf.load(os.path.join(args.sr_run, "config.yaml"))      # sr_fm_refiner_v12-lumi

    def log(msg):
        if is_main:
            logging.info(msg)

    # --- baseline == coarse model of both refiners (checked: same chmv2_ckpt_path) ---
    for c, name in ((cfg_fm, "fm"), (cfg_sr, "sr")):
        assert c.model.chmv2_ckpt_path == cfg_fm.model.chmv2_ckpt_path, \
            f"{name} run uses a different coarse CHMv2 checkpoint"
    cfg_d = cfg_fm.dataset
    in_stats = _load_target_stats(cfg_d.input_stats_file, cfg_d.year)
    chm = load_chmv2_baseline(
        model_dir=cfg_fm.model.chmv2_model_id,
        ckpt_path=args.baseline_ckpt or cfg_fm.model.chmv2_ckpt_path,
        mean=in_stats["mean"], std=in_stats["std"],
        out_in_scale_factor=cfg_fm.model.chmv2_out_in_scale_factor,
    ).to(device).eval()
    log("loaded CHMv2 baseline")

    n_ls = len(cfg_d.selected_bands)
    n_ps = len(cfg_d.selected_bands_hr)

    # --- fm_refiner-R-lumi-5 (finer_lumi code) ---
    t = cfg_fm.trainer
    fm_plain = plain_fm_mod.build_fm_refiner(
        sd_pretrained_path=cfg_fm.model.sd_pretrained_path,
        n_landsat_bands=n_ls, n_ps_bands=n_ps,
        ps_dropout_p=0.0,
        use_controlnet=t.get("use_controlnet", True),
        controlnet_cond_mode=t.get("controlnet_cond_mode", "landsat_ps"),
        velocity_parameterization=t.get("velocity_parameterization", "fixed"),
        noise_sigma=t.get("noise_sigma", 0.0),
        sample_sigma=t.get("sample_sigma", None),
        sample_noise_mode=t.get("sample_noise_mode", "sde"),
        device=str(device),
    ).to(device)
    ck = _torch_load(args.fm_ckpt or os.path.join(args.fm_run, "checkpoint", "best.pth"))
    fm_plain.unet.load_state_dict(ck["unet_state"])
    if fm_plain.use_controlnet and ck.get("controlnet_state") is not None:
        fm_plain.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm_plain._null_ps = ck["null_ps_state"]
    log(f"loaded fm_refiner-R-lumi-5 (step={ck.get('step')})")
    del ck
    fm_plain.eval()

    # --- sr_fm_refiner_v12-lumi (this branch) ---
    t = cfg_sr.trainer
    fm_sr = sr_fm_mod.build_fm_refiner(
        sd_pretrained_path=cfg_sr.model.sd_pretrained_path,
        n_landsat_bands=n_ls, n_ps_bands=n_ps,
        ps_dropout_p=0.0,
        bridge_sigma=t.get("bridge_sigma", 0.0),
        noise_sigma=t.get("noise_sigma", 0.0),
        sample_sigma=t.get("sample_sigma", None),
        sample_noise_mode=t.get("sample_noise_mode", "sde"),
        concat_z_coarse=t.get("concat_z_coarse", False),
        refine_threshold=t.get("refine_threshold", 0.0),
        concat_landsat_hr=t.get("concat_landsat_hr", False),
        device=str(device),
    ).to(device)
    sr_module = build_sr_module(OmegaConf.to_container(cfg_sr.sr_module, resolve=True)).to(device)
    ck = _torch_load(args.sr_ckpt or os.path.join(args.sr_run, "checkpoint", "latest.pth"))
    fm_sr.unet.load_state_dict(ck["unet_state"])
    fm_sr.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm_sr._null_ps = ck["null_ps_state"]
    sr_module.load_state_dict(ck["sr_state"])
    log(f"loaded sr_fm_refiner_v12-lumi (step={ck.get('step')})")
    del ck
    fm_sr.eval()
    sr_module.eval()
    return dict(cfg=cfg_fm, chm=chm, fm_plain=fm_plain, fm_sr=fm_sr, sr_module=sr_module,
                n_ls=n_ls, cfg_sr=cfg_sr)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm_run", default=f"{OUT_ROOT}/fm_refiner-R-lumi-5")
    ap.add_argument("--sr_run", default=f"{OUT_ROOT}/sr_fm_refiner_v12-lumi")
    ap.add_argument("--baseline_ckpt", default=None, help="default: coarse ckpt of the fm run's config (baseline best.pth)")
    ap.add_argument("--fm_ckpt", default=None, help="default: <fm_run>/checkpoint/best.pth")
    ap.add_argument("--sr_ckpt", default=None, help="default: <sr_run>/checkpoint/latest.pth")
    ap.add_argument("--out_dir", default=f"{OUT_ROOT}/_detail_eval")
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_avg", type=int, default=5)
    ap.add_argument("--edge_thr", type=float, default=2.0, help="edge = Sobel gradient >= this (m/px)")
    ap.add_argument("--edge_tol", type=int, default=1, help="edge match tolerance in px (also reports 2x)")
    ap.add_argument("--hf_sigma", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N val samples")
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

    if dist_on:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        use_cuda = torch.cuda.is_available() and not args.no_cuda
        dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=world,
                                timeout=timedelta(minutes=30))
    if torch.cuda.is_available() and not args.no_cuda:
        idx = local % max(1, torch.cuda.device_count())
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")

    # Same as the training scripts: without these, MIOpen's conv auto-tuning tries to
    # write its find-db and the first conv dies with miopenStatusInternalError.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    M = load_models(args, device, is_main)
    cfg_d = M["cfg"].dataset
    target_stats = _load_target_stats(cfg_d.target_stats_file, cfg_d.year)
    p1, p99 = float(target_stats["p1"][0]), float(target_stats["p99"][0])
    val_ds = _make_val_dataset(cfg_d)
    n_total = len(val_ds) if not args.limit else min(args.limit, len(val_ds))
    if is_main:
        logging.info(f"val samples: {n_total}; world size {world}; n_steps={args.n_steps} n_avg={args.n_avg}")
    local_idx = list(range(n_total))[rank::world]
    loader = DataLoader(Subset(val_ds, local_idx), batch_size=1, shuffle=False,
                        num_workers=cfg_d.workers, pin_memory=True)

    names = ["baseline_chmv2", "fm_refiner-R-lumi-5", "sr_fm_refiner_v12"]
    acc = {n: DetailAccumulator(device, args.edge_thr, args.edge_tol, args.hf_sigma) for n in names}
    n_ls = M["n_ls"]

    def _avg(fn, seed):
        draws = []
        for k in range(args.n_avg):
            torch.manual_seed(seed * 1000 + k)
            draws.append(fn())
        return torch.stack(draws).mean(0)

    for sample_i, batch in zip(local_idx, tqdm(loader, disable=not is_main, desc="detail-eval")):
        inputs_lr, inputs_hr, targets = batch
        inputs_lr = inputs_lr.squeeze(0) if inputs_lr.dim() == 5 else inputs_lr
        targets = targets.squeeze(0) if targets.dim() == 5 else targets
        inputs_lr, targets = inputs_lr.to(device), targets.to(device)
        H, W = targets.shape[-2:]
        landsat = inputs_lr[:, :n_ls]
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)

        with torch.no_grad():
            coarse_m = M["chm"]((landsat + 1.0) / 2.0, target_size=(H, W))       # metres
            coarse_n = _normalize_coarse(coarse_m, target_stats)
            gt_m = _denormalize(targets, target_stats)

            pred_base = coarse_m.clamp(p1, p99)

            h = _avg(lambda: M["fm_plain"].refine(landsat_hr, coarse_n, n_steps=args.n_steps, method="euler"),
                     sample_i)
            pred_fm = _denormalize(h, target_stats).clamp(p1, p99)

            pseudo_ps = M["sr_module"](landsat, target_size=(H, W))
            h = _avg(lambda: M["fm_sr"].refine_with_ps(pseudo_ps, coarse_n, n_steps=args.n_steps,
                                                       method="euler", landsat_hr=landsat_hr),
                     sample_i)
            pred_sr = _denormalize(h, target_stats).clamp(p1, p99)

            for n, pr in zip(names, (pred_base, pred_fm, pred_sr)):
                acc[n].update(pr.float(), gt_m.float())

    states = {n: acc[n].state() for n in names}
    if dist_on:
        gathered = [None] * world
        dist.all_gather_object(gathered, states)
    else:
        gathered = [states]

    if is_main:
        results = {n: finalize([g[n] for g in gathered]) for n in names}
        os.makedirs(args.out_dir, exist_ok=True)
        meta = {"n_val_samples": n_total, "n_steps": args.n_steps, "n_avg": args.n_avg,
                "edge_thr_m_per_px": args.edge_thr, "edge_tol_px": args.edge_tol,
                "hf_sigma_px": args.hf_sigma, "range_m": [p1, p99],
                "ckpts": {"baseline": args.baseline_ckpt or str(M["cfg"].model.chmv2_ckpt_path),
                          "fm_refiner-R-lumi-5": args.fm_ckpt or f"{args.fm_run}/checkpoint/best.pth",
                          "sr_fm_refiner_v12": args.sr_ckpt or f"{args.sr_run}/checkpoint/latest.pth"}}
        with open(os.path.join(args.out_dir, "detail_metrics.json"), "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

        t1, t2 = args.edge_tol, 2 * args.edge_tol
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
        ]
        lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for label, key, fmt in rows:
            lines.append(f"| {label} | " + " | ".join(fmt.format(results[n][key]) for n in names) + " |")
        lines += ["", "Height-binned error (by GT height): MAE / bias (pred-gt) / RMSE in m", "",
                  "| bin | pixel frac | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for lab in results[names[0]]["height_bins"]:
            fr = results[names[0]]["height_bins"][lab]["pixel_frac"]
            cells = []
            for n in names:
                b = results[n]["height_bins"][lab]
                cells.append(f"{b['mae']:.2f} / {b['bias']:+.2f} / {b['rmse']:.2f}")
            lines.append(f"| {lab} | {fr:.3f} | " + " | ".join(cells) + " |")
        gt_ref = results[names[0]]["gt_mean_grad"]
        lines += ["", f"GT mean gradient: {gt_ref:.4f} m/px; GT edge pixel fraction (thr {args.edge_thr} m/px): "
                      f"{results[names[0]]['edge_gt_frac']:.4f}"]
        table = "\n".join(lines)
        with open(os.path.join(args.out_dir, "detail_metrics.md"), "w") as f:
            f.write(table + "\n")
        logging.info("\n" + table)

    if dist_on:
        dist.barrier()
        dist.destroy_process_group()
