"""
Detail-fidelity evaluation of four models on the held-out TEST split of the
leak-free spatial split (split_spatial_v1_test.pkl), plus result images.

  baseline_chmv2-lumi-nd     : fine-tuned CHMv2 (also the frozen coarse model of both refiners)
  fm_refiner-R-lumi-5-nd     : FM refiner (finer_lumi_nd code, depthfm/fm_refiner_finer.py), Landsat-only ControlNet
  baseline_swt-lumi-nd       : SwinIR SR (Landsat -> pseudo-PS) + CHMv2 regression head, trained jointly
  sr_fm_refiner_v12-lumi-nd  : SR + FM refiner (this branch's depthfm/fm_refiner.py), pseudo-PS ControlNet

All four are scored on the same test patches (non-overlapping 480 px grid, no
augmentation) against the same ground truth, in metres. Metrics are those of
eval_detail_metrics.py (gradient error / energy ratio, edge F1, high-frequency
ratio / correlation, height-binned error, plus R2 / MAE / RMSE), pixel-pooled.
FM refiners run like their final validation: n_steps Euler steps, mean of
n_avg independent stochastic draws (seeded per patch).

Images (--no_vis to skip), under <out_dir>/images/:
  panel/<id>.png   per test patch: Landsat | real PS | SR (SwT) | SR (v12) | GT on the
                   first row, baseline | fm_refiner | swt | v12 predictions on the second
  sr_swt/<id>.png  the SwT model's super-resolved image (full res, lossless)
  sr_v12/<id>.png  v12's pseudo-PS (full res, lossless)
  pred_<model>/<id>.png  each model's height prediction, colour-mapped (plasma, 0..p99 m,
                   same scale for all models and the GT), full res, lossless; <model> is
                   one of baseline_chmv2, fm_refiner-R-lumi-5, baseline_swt, sr_fm_refiner_v12
  gt/<id>.png      the ground-truth height in the same colour map
  index.csv        id, tile, crop position, GT p75 height, per-model MAE of that patch
<id> = <test-list index>_<tile>_y<row>_x<col>.

Launch (one task per GPU): see script/depth/eval_detail_metrics_nd.sh
"""

import argparse
import copy
import csv
import json
import logging
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

import eval_detail_metrics as old  # DetailAccumulator, finalize, helpers (rasters/metrics unchanged)
from depthfm import fm_refiner as sr_fm_mod            # this branch (SR + pseudo-PS ControlNet)
from depthfm import fm_refiner_finer as plain_fm_mod   # finer_lumi_nd code (Landsat-only ControlNet)
from depthfm.chmv2 import CHMv2Height, load_chmv2_baseline
from depthfm.sr_module import build_sr_module
from src.util.ps_lazydataset import LazyPatchDataset

OUT_ROOT = "/flash/project_465002934/Marigold_output"
NAMES = ["baseline_chmv2", "fm_refiner-R-lumi-5", "baseline_swt", "sr_fm_refiner_v12"]


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

def make_test_dataset(cfg_data):
    """Test split of the spatial split: cfg_data.test_coord_path, in file order,
    no augmentation, item idx == patch idx."""
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
    import pickle
    with open(cfg_data.test_coord_path, "rb") as f:
        coords = pickle.load(f)
    ds = copy.copy(base)
    ds.batch_size = 1
    ds.mode = 'test'
    ds.patch_coords = coords
    # base built its transforms in 'train' mode (random brightness/gamma): rebuild without.
    ds.transform_input, ds.transform_input_hr, ds.transform_output = ds._transform()
    ds.deterministic_eval = True
    return ds, coords


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------

def load_models(args, device, is_main):
    run = lambda n: os.path.join(args.run_root, n)
    ck_name = f"{args.ckpt}.pth"

    def log(msg):
        if is_main:
            logging.info(msg)

    cfg_chm = OmegaConf.load(os.path.join(run("baseline_chmv2-lumi-nd"), "config.yaml"))
    cfg_fm = OmegaConf.load(os.path.join(run("fm_refiner-R-lumi-5-nd"), "config.yaml"))
    cfg_swt = OmegaConf.load(os.path.join(run("baseline_swt-lumi-nd"), "config.yaml"))
    cfg_sr = OmegaConf.load(os.path.join(run("sr_fm_refiner_v12-lumi-nd"), "config.yaml"))
    cfg_d = cfg_fm.dataset

    # --- baseline == coarse model of both refiners ---
    chm_ckpt = os.path.join(run("baseline_chmv2-lumi-nd"), "checkpoint", ck_name)
    for c, name in ((cfg_fm, "fm"), (cfg_sr, "sr")):
        assert c.model.chmv2_ckpt_path == os.path.join(run("baseline_chmv2-lumi-nd"), "checkpoint", "best.pth"), \
            f"{name} run uses a different coarse CHMv2 checkpoint: {c.model.chmv2_ckpt_path}"
    in_stats = old._load_target_stats(cfg_d.input_stats_file, cfg_d.year)
    chm = load_chmv2_baseline(
        model_dir=cfg_chm.model.chmv2_model_id, ckpt_path=chm_ckpt,
        mean=in_stats["mean"], std=in_stats["std"],
        out_in_scale_factor=cfg_chm.model.out_in_scale_factor,
    ).to(device).eval()
    log(f"loaded baseline_chmv2-lumi-nd ({chm_ckpt})")
    # The refiners' frozen coarse model is their best.pth-pinned baseline; if a
    # different --ckpt is evaluated for the baseline, the refiners still use
    # their own configured coarse model, which is what they were trained with.
    if ck_name != "best.pth":
        chm_coarse = load_chmv2_baseline(
            model_dir=cfg_chm.model.chmv2_model_id, ckpt_path=cfg_fm.model.chmv2_ckpt_path,
            mean=in_stats["mean"], std=in_stats["std"],
            out_in_scale_factor=cfg_chm.model.out_in_scale_factor,
        ).to(device).eval()
    else:
        chm_coarse = chm

    n_ls = len(cfg_d.selected_bands)
    n_ps = len(cfg_d.selected_bands_hr)

    # --- fm_refiner-R-lumi-5-nd (finer_lumi_nd code) ---
    t = cfg_fm.trainer
    fm_plain = plain_fm_mod.build_fm_refiner(
        sd_pretrained_path=cfg_fm.model.sd_pretrained_path,
        n_landsat_bands=n_ls, n_ps_bands=n_ps, ps_dropout_p=0.0,
        use_controlnet=t.get("use_controlnet", True),
        controlnet_cond_mode=t.get("controlnet_cond_mode", "landsat_ps"),
        velocity_parameterization=t.get("velocity_parameterization", "fixed"),
        noise_sigma=t.get("noise_sigma", 0.0), sample_sigma=t.get("sample_sigma", None),
        sample_noise_mode=t.get("sample_noise_mode", "sde"),
        device=str(device),
    ).to(device)
    ck = old._torch_load(os.path.join(run("fm_refiner-R-lumi-5-nd"), "checkpoint", ck_name))
    fm_plain.unet.load_state_dict(ck["unet_state"])
    if fm_plain.use_controlnet and ck.get("controlnet_state") is not None:
        fm_plain.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm_plain._null_ps = ck["null_ps_state"]
    log(f"loaded fm_refiner-R-lumi-5-nd (step={ck.get('step')})")
    del ck
    fm_plain.eval()

    # --- baseline_swt-lumi-nd: SR (Landsat -> pseudo-PS) + CHMv2 regression head ---
    sr_swt = build_sr_module(OmegaConf.to_container(cfg_swt.sr_module, resolve=True)).to(device)
    ps_stats = old._load_target_stats(cfg_swt.dataset.input_hr_stats_file, cfg_swt.dataset.year)
    reg = CHMv2Height(
        model_id=cfg_swt.model.chmv2_model_id,
        mean=[ps_stats["mean"][i] for i in cfg_swt.dataset.selected_bands_hr],
        std=[ps_stats["std"][i] for i in cfg_swt.dataset.selected_bands_hr],
        pretrained=False, freeze_backbone=False, out_in_scale_factor=1,
    ).to(device)
    ck = old._torch_load(os.path.join(run("baseline_swt-lumi-nd"), "checkpoint", ck_name))
    sr_swt.load_state_dict(ck["sr_state"])
    reg.load_state_dict(ck["reg_state"])
    log(f"loaded baseline_swt-lumi-nd (step={ck.get('step')})")
    del ck
    sr_swt.eval()
    reg.eval()

    # --- sr_fm_refiner_v12-lumi-nd (this branch) ---
    t = cfg_sr.trainer
    fm_sr = sr_fm_mod.build_fm_refiner(
        sd_pretrained_path=cfg_sr.model.sd_pretrained_path,
        n_landsat_bands=n_ls, n_ps_bands=n_ps, ps_dropout_p=0.0,
        bridge_sigma=t.get("bridge_sigma", 0.0), noise_sigma=t.get("noise_sigma", 0.0),
        sample_sigma=t.get("sample_sigma", None), sample_noise_mode=t.get("sample_noise_mode", "sde"),
        concat_z_coarse=t.get("concat_z_coarse", False), refine_threshold=t.get("refine_threshold", 0.0),
        concat_landsat_hr=t.get("concat_landsat_hr", False),
        device=str(device),
    ).to(device)
    sr_v12 = build_sr_module(OmegaConf.to_container(cfg_sr.sr_module, resolve=True)).to(device)
    ck = old._torch_load(os.path.join(run("sr_fm_refiner_v12-lumi-nd"), "checkpoint", ck_name))
    fm_sr.unet.load_state_dict(ck["unet_state"])
    fm_sr.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm_sr._null_ps = ck["null_ps_state"]
    sr_v12.load_state_dict(ck["sr_state"])
    log(f"loaded sr_fm_refiner_v12-lumi-nd (step={ck.get('step')})")
    del ck
    fm_sr.eval()
    sr_v12.eval()
    return dict(cfg=cfg_fm, chm=chm, chm_coarse=chm_coarse, fm_plain=fm_plain, sr_swt=sr_swt, reg=reg,
                fm_sr=fm_sr, sr_v12=sr_v12, n_ls=n_ls, ck_name=ck_name)


# -------------------------------------------------------------------------
# Edge F1 at several gradient thresholds
# -------------------------------------------------------------------------

class EdgeAccumulator:
    """Edge precision / recall / F1 for several Sobel-gradient thresholds (m/px)
    at once. Same definition as DetailAccumulator's edge metric (edge = Sobel/8
    gradient magnitude >= thr; a predicted edge pixel is a hit if a GT edge pixel
    lies within tol px, and vice versa), pixel-pooled, so the thr == edge_thr
    entry reproduces its edge_*_tol* numbers. A single fixed threshold is
    arbitrary (smooth predictions have few edges above any absolute value), so
    several are reported."""

    def __init__(self, device, thrs, tols):
        self.thrs, self.tols = list(thrs), list(tols)
        self.kx = old._SOBEL_X.view(1, 1, 3, 3).to(device)
        self.ky = old._SOBEL_X.t().contiguous().view(1, 1, 3, 3).to(device)
        keys = ["pred", "gt"] + [f"prec_hit_{t}" for t in self.tols] + [f"rec_hit_{t}" for t in self.tols]
        self.s = {thr: {k: 0.0 for k in keys} for thr in self.thrs}

    def _grad(self, x):
        return torch.sqrt(F.conv2d(x, self.kx) ** 2 + F.conv2d(x, self.ky) ** 2)

    @torch.no_grad()
    def update(self, pred, gt):
        gp, gg = self._grad(pred), self._grad(gt)
        for thr in self.thrs:
            ep, eg = gp >= thr, gg >= thr
            s = self.s[thr]
            s["pred"] += ep.sum().item()
            s["gt"] += eg.sum().item()
            epf, egf = ep.float(), eg.float()
            for t in self.tols:
                s[f"prec_hit_{t}"] += (ep & old.DetailAccumulator._dilate(egf, t)).sum().item()
                s[f"rec_hit_{t}"] += (eg & old.DetailAccumulator._dilate(epf, t)).sum().item()

    def state(self):
        return self.s

    @staticmethod
    def finalize(states, thrs, tols):
        out = {}
        for thr in thrs:
            s = {k: sum(st[thr][k] for st in states) for k in states[0][thr]}
            for t in tols:
                prec = s[f"prec_hit_{t}"] / max(s["pred"], 1.0)
                rec = s[f"rec_hit_{t}"] / max(s["gt"], 1.0)
                out[f"thr{thr:g}_tol{t}"] = {"precision": prec, "recall": rec,
                                             "f1": 2 * prec * rec / max(prec + rec, 1e-12)}
            out[f"thr{thr:g}_gt_edge_pixels"] = s["gt"]
        return out


# -------------------------------------------------------------------------
# Images
# -------------------------------------------------------------------------

def _rgb_u8(t):
    """(C,H,W) tensor in [-1,1] -> (H,W,3) uint8."""
    x = ((t[:3].float().cpu() + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def _colormap_u8(h, vmax):
    """(H,W) height in metres -> (H,W,3) uint8, plasma over [0, vmax]."""
    rgba = plt.get_cmap("plasma")(np.clip(h / vmax, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def save_panel(path, imgs, preds, gt, maes, title, vmax):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.0))
    row1 = [("Landsat (bilinear)", imgs["landsat"]), ("Real PlanetScope", imgs["ps"]),
            ("SR (SwT)", imgs["sr_swt"]), ("SR (v12 pseudo-PS)", imgs["sr_v12"])]
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
    ap.add_argument("--ckpt", default="best", help="checkpoint name in each run's checkpoint/ dir (best|final|latest)")
    ap.add_argument("--out_dir", default=f"{OUT_ROOT}/_detail_eval_nd")
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n_avg", type=int, default=5)
    ap.add_argument("--edge_thr", type=float, default=2.0, help="edge = Sobel gradient >= this (m/px)")
    ap.add_argument("--edge_thrs", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                    help="thresholds (m/px) for the multi-threshold edge F1 table")
    ap.add_argument("--edge_tol", type=int, default=1, help="edge match tolerance in px (also reports 2x)")
    ap.add_argument("--hf_sigma", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N test samples")
    ap.add_argument("--no_vis", action="store_true", help="metrics only, no images")
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

    # Same as the training scripts (MIOpen find-db write otherwise kills the first conv).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    M = load_models(args, device, is_main)
    cfg_d = M["cfg"].dataset
    target_stats = old._load_target_stats(cfg_d.target_stats_file, cfg_d.year)
    p1, p99 = float(target_stats["p1"][0]), float(target_stats["p99"][0])
    test_ds, test_coords = make_test_dataset(cfg_d)
    n_total = len(test_ds) if not args.limit else min(args.limit, len(test_ds))
    if is_main:
        logging.info(f"test samples: {n_total}; world size {world}; n_steps={args.n_steps} n_avg={args.n_avg}; "
                     f"ckpt={M['ck_name']}")
    local_idx = list(range(n_total))[rank::world]
    loader = DataLoader(Subset(test_ds, local_idx), batch_size=1, shuffle=False,
                        num_workers=cfg_d.workers, pin_memory=True)

    img_dir = os.path.join(args.out_dir, "images")
    if not args.no_vis:
        for sub in ("panel", "sr_swt", "sr_v12", "gt", *[f"pred_{n}" for n in NAMES]):
            os.makedirs(os.path.join(img_dir, sub), exist_ok=True)

    acc = {n: old.DetailAccumulator(device, args.edge_thr, args.edge_tol, args.hf_sigma) for n in NAMES}
    tols = (args.edge_tol, 2 * args.edge_tol)
    edge_acc = {n: EdgeAccumulator(device, args.edge_thrs, tols) for n in NAMES}
    n_ls = M["n_ls"]
    index_rows = []

    def _avg(fn, seed):
        draws = []
        for k in range(args.n_avg):
            torch.manual_seed(seed * 1000 + k)
            draws.append(fn())
        return torch.stack(draws).mean(0)

    for sample_i, batch in zip(local_idx, tqdm(loader, disable=not is_main, desc="detail-eval-test")):
        inputs_lr, inputs_hr, targets = batch
        inputs_lr = inputs_lr.squeeze(0) if inputs_lr.dim() == 5 else inputs_lr
        inputs_hr = inputs_hr.squeeze(0) if inputs_hr.dim() == 5 else inputs_hr
        targets = targets.squeeze(0) if targets.dim() == 5 else targets
        inputs_lr, inputs_hr, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
        H, W = targets.shape[-2:]
        landsat = inputs_lr[:, :n_ls]
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)

        with torch.no_grad():
            gt_m = old._denormalize(targets, target_stats)

            # 1. baseline_chmv2-lumi-nd
            pred_base = M["chm"]((landsat + 1.0) / 2.0, target_size=(H, W)).clamp(p1, p99)

            # coarse height shared by both refiners (their own configured coarse model)
            coarse_m = M["chm_coarse"]((landsat + 1.0) / 2.0, target_size=(H, W))
            coarse_n = old._normalize_coarse(coarse_m, target_stats)

            # 2. fm_refiner-R-lumi-5-nd
            h = _avg(lambda: M["fm_plain"].refine(landsat_hr, coarse_n, n_steps=args.n_steps, method="euler"),
                     sample_i)
            pred_fm = old._denormalize(h, target_stats).clamp(p1, p99)

            # 3. baseline_swt-lumi-nd: SR image -> CHMv2 regression head
            img_sr_swt = M["sr_swt"](landsat, target_size=(H, W))
            pred_swt = M["reg"]((img_sr_swt + 1.0) / 2.0, target_size=(H, W)).clamp(p1, p99)

            # 4. sr_fm_refiner_v12-lumi-nd
            pseudo_ps = M["sr_v12"](landsat, target_size=(H, W))
            h = _avg(lambda: M["fm_sr"].refine_with_ps(pseudo_ps, coarse_n, n_steps=args.n_steps,
                                                       method="euler", landsat_hr=landsat_hr),
                     sample_i)
            pred_sr = old._denormalize(h, target_stats).clamp(p1, p99)

            preds = dict(zip(NAMES, (pred_base, pred_fm, pred_swt, pred_sr)))
            maes = {}
            for n, pr in preds.items():
                acc[n].update(pr.float(), gt_m.float())
                edge_acc[n].update(pr.float(), gt_m.float())
                maes[n] = (pr.float() - gt_m.float()).abs().mean().item()

        c = test_coords[sample_i]
        pid = f"{sample_i:05d}_{c['tile_id'].replace(',', '-')}_y{c['output_crop_y']}_x{c['output_crop_x']}"
        index_rows.append({
            "id": pid, "test_idx": sample_i, "tile": c["tile_id"],
            "y": c["output_crop_y"], "x": c["output_crop_x"], "size": c["output_patch_height"],
            "gt_p75_m": float(np.percentile(gt_m.cpu().numpy(), 75)),
            **{f"mae_{n}": maes[n] for n in NAMES},
        })
        if not args.no_vis:
            Image.fromarray(_rgb_u8(img_sr_swt[0])).save(os.path.join(img_dir, "sr_swt", pid + ".png"))
            Image.fromarray(_rgb_u8(pseudo_ps[0])).save(os.path.join(img_dir, "sr_v12", pid + ".png"))
            Image.fromarray(_colormap_u8(gt_m[0, 0].cpu().numpy(), p99)).save(os.path.join(img_dir, "gt", pid + ".png"))
            for n in NAMES:
                Image.fromarray(_colormap_u8(preds[n][0, 0].cpu().numpy(), p99)).save(
                    os.path.join(img_dir, f"pred_{n}", pid + ".png"))
            save_panel(
                os.path.join(img_dir, "panel", pid + ".png"),
                imgs={"landsat": _rgb_u8(landsat_hr[0]), "ps": _rgb_u8(inputs_hr[0]),
                      "sr_swt": _rgb_u8(img_sr_swt[0]), "sr_v12": _rgb_u8(pseudo_ps[0])},
                preds={n: preds[n][0, 0].cpu().numpy() for n in NAMES},
                gt=gt_m[0, 0].cpu().numpy(), maes=maes,
                title=f"{pid}   (GT p75 = {index_rows[-1]['gt_p75_m']:.1f} m)", vmax=p99,
            )

    states = {n: acc[n].state() for n in NAMES}
    edge_states = {n: edge_acc[n].state() for n in NAMES}
    payload = (states, index_rows, edge_states)
    if dist_on:
        gathered = [None] * world
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    if is_main:
        results = {n: old.finalize([g[0][n] for g in gathered]) for n in NAMES}
        for n in NAMES:
            results[n]["edge_multi_thr"] = EdgeAccumulator.finalize([g[2][n] for g in gathered],
                                                                    args.edge_thrs, tols)
        os.makedirs(args.out_dir, exist_ok=True)
        meta = {"split": "test (split_spatial_v1_test.pkl)", "n_test_samples": n_total,
                "n_steps": args.n_steps, "n_avg": args.n_avg,
                "edge_thr_m_per_px": args.edge_thr, "edge_thrs_m_per_px": args.edge_thrs,
                "edge_tol_px": args.edge_tol,
                "hf_sigma_px": args.hf_sigma, "range_m": [p1, p99], "ckpt": M["ck_name"],
                "runs": ["baseline_chmv2-lumi-nd", "fm_refiner-R-lumi-5-nd",
                         "baseline_swt-lumi-nd", "sr_fm_refiner_v12-lumi-nd"]}
        with open(os.path.join(args.out_dir, "detail_metrics.json"), "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

        rows_all = sorted((r for g in gathered for r in g[1]), key=lambda r: r["test_idx"])
        with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
            w.writeheader()
            w.writerows(rows_all)

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
        lines = ["| metric | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
        for label, key, fmt in rows:
            lines.append(f"| {label} | " + " | ".join(fmt.format(results[n][key]) for n in NAMES) + " |")
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
