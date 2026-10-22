"""
Images and LiDAR evaluation of the wall-to-wall maps of infer_time_region.py.

Reads <in_dir>/region.json and <in_dir>/<model>/<year>.tif and writes:
  png/landsat/<year>.png     Landsat RGB of the region, one 2-98 % stretch for all years,
                             nearest x4 onto the prediction grid
  png/<model>/<year>.png     prediction, plasma over [0, vmax] m (scale of eval_group_meeting.py)
  png/strip.png              rows Landsat / models, one column per year
  gif/landsat.gif, gif/<model>.gif   one frame per year (make_time_gif.py style)
  eval/lidar_<year>.png      LiDAR (resampled) next to both predictions, for the LiDAR years
  (figures: cells drawn with their ground aspect, 1/cos(lat) for a degree grid)
  eval/metrics.{md,json}     accuracy against the LiDAR CHMs (--ref_tifs, year = file-name prefix)

LiDAR -> prediction grid: rasterio reproject with Resampling.average (mean of the
valid 0.5 m cells in each prediction cell); a cell is evaluated only if at least
--min_valid of it is covered by valid LiDAR. As in eval_group_meeting.py, GT and
predictions are clipped to [0, vmax] m (an unclipped R2 / MAE / RMSE is reported too).
Detail metrics follow eval_detail_metrics.py (Sobel/8 gradient in m/px of the
prediction grid, edge thr / tol, HF = x - GaussianBlur(sigma)); a pixel counts only if
the whole neighbourhood the metric uses is valid LiDAR. Change: LiDAR 2nd year - 1st
year against the same difference of the predictions, on cells valid in both years.
CPU only; see script/depth/vis_eval_region.sh.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from affine import Affine
from PIL import Image
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from eval_detail_metrics import HEIGHT_BIN_EDGES, _SOBEL_X, _gaussian_kernel
from eval_detail_metrics_nd import _colormap_u8
from make_time_gif import render, save_gif

MODELS = ["baseline_sr4ir_reg", "sr_fm_refiner_v11-lumi"]


# -------------------------------------------------------------------------
# LiDAR on the prediction grid
# -------------------------------------------------------------------------

def lidar_on_grid(path, shape, transform, crs):
    """mean LiDAR height and valid fraction per prediction cell."""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nod = src.nodata
        valid = np.isfinite(a) if nod is None else (a != nod) & np.isfinite(a)
        a = np.where(valid, a, np.nan)
        h = np.full(shape, np.nan, np.float32)
        reproject(a, h, src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        frac = np.zeros(shape, np.float32)
        reproject(valid.astype(np.float32), frac, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.average)
    return h, frac


# -------------------------------------------------------------------------
# Masked metrics (definitions of eval_detail_metrics.py)
# -------------------------------------------------------------------------

def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]


def _erode(m, r):
    """valid only where the (2r+1)^2 neighbourhood is valid (same size as m)."""
    return -F.max_pool2d(-m, 2 * r + 1, stride=1, padding=r) > 0.5


def _grad(x):
    kx = _SOBEL_X.view(1, 1, 3, 3)
    ky = _SOBEL_X.t().contiguous().view(1, 1, 3, 3)
    return torch.sqrt(F.conv2d(x, kx) ** 2 + F.conv2d(x, ky) ** 2)


def _hf(x, sigma):
    k, r = _gaussian_kernel(sigma)
    xp = F.pad(x, (r, r, r, r), mode="reflect")
    blur = F.conv2d(F.conv2d(xp, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
    return (x - blur)[..., r:-r, r:-r], r


def metrics(pred, gt, valid, vmax, edge_thr, tols, hf_sigma, pred_raw=None, gt_raw=None):
    """pred, gt: (H, W) metres clipped to [0, vmax]; valid: (H, W) bool."""
    out = {}
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    e = p - g
    out["n_pixels"] = int(valid.sum())
    out["r2"] = 1 - (e ** 2).sum() / ((g - g.mean()) ** 2).sum()
    out["mae"] = np.abs(e).mean()
    out["rmse"] = math.sqrt((e ** 2).mean())
    if pred_raw is not None:
        pr, gr = pred_raw[valid].astype(np.float64), gt_raw[valid].astype(np.float64)
        er = pr - gr
        out["unclipped"] = {"r2": 1 - (er ** 2).sum() / ((gr - gr.mean()) ** 2).sum(),
                            "mae": np.abs(er).mean(), "rmse": math.sqrt((er ** 2).mean())}
    edges = np.array(HEIGHT_BIN_EDGES)
    b = np.clip(np.digitize(g, edges) - 1, 0, len(edges) - 1)
    out["height_bins"] = {}
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip(edges[:-1], edges[1:])] + [f"[{edges[-1]:g},inf)"]
    for i, lab in enumerate(labels):
        s = b == i
        out["height_bins"][lab] = {"pixel_frac": float(s.mean()),
                                   "mae": float(np.abs(e[s]).mean()) if s.any() else float("nan"),
                                   "bias": float(e[s].mean()) if s.any() else float("nan"),
                                   "rmse": float(math.sqrt((e[s] ** 2).mean())) if s.any() else float("nan")}

    P, G, V = _t(pred), _t(np.where(valid, gt, 0.0)), _t(valid.astype(np.float32))
    gp, gg = _grad(P), _grad(G)                                   # (H-2, W-2)
    mg = _erode(V, 1)[..., 1:-1, 1:-1]
    gp64, gg64 = gp[mg].double(), gg[mg].double()
    out["grad_mag_error"] = (gp64 - gg64).abs().mean().item()
    out["grad_energy_ratio"] = ((gp64 ** 2).sum() / (gg64 ** 2).sum()).item()
    out["grad_mag_ratio"] = (gp64.sum() / gg64.sum()).item()
    ep, eg = gp >= edge_thr, gg >= edge_thr
    for t in tols:
        m = _erode(V, 1 + t)[..., 1:-1, 1:-1]               # the gradient AND the tolerance window are valid
        dil = lambda x: F.max_pool2d(x.float(), 2 * t + 1, stride=1, padding=t) > 0
        prec = (ep & dil(eg) & m).sum().item() / max((ep & m).sum().item(), 1)
        rec = (eg & dil(ep) & m).sum().item() / max((eg & m).sum().item(), 1)
        out[f"edge_precision_tol{t}"], out[f"edge_recall_tol{t}"] = prec, rec
        out[f"edge_f1_tol{t}"] = 2 * prec * rec / max(prec + rec, 1e-12)
    hp, r = _hf(P, hf_sigma)
    hg, _ = _hf(G, hf_sigma)
    mh = _erode(V, r)[..., r:-r, r:-r]
    hp64, hg64 = hp[mh].double(), hg[mh].double()
    out["hf_energy_ratio"] = ((hp64 ** 2).sum() / (hg64 ** 2).sum()).item()
    out["hf_corr"] = ((hp64 * hg64).sum() / torch.sqrt((hp64 ** 2).sum() * (hg64 ** 2).sum())).item()
    return out


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="/flash/project_465002934/result_time_lumi_denm")
    ap.add_argument("--vmax", type=float, default=28.0)
    ap.add_argument("--min_valid", type=float, default=0.9, help="min valid LiDAR fraction of an evaluated cell")
    ap.add_argument("--edge_thr", type=float, default=2.0)
    ap.add_argument("--edge_tol", type=int, default=1)
    ap.add_argument("--hf_sigma", type=float, default=2.0)
    ap.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2], help="Landsat bands shown as R G B")
    ap.add_argument("--no_images", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(args.in_dir, "region.json")) as f:
        R = json.load(f)
    years = R["years"]
    Hc, Wc = R["out_shape_hw"]
    out_t = Affine(*R["out_transform"])
    crs = CRS.from_string(R["crs"])
    preds = {m: {} for m in MODELS}
    for m in MODELS:
        for yr in years:
            with rasterio.open(os.path.join(args.in_dir, m, f"{yr}.tif")) as src:
                preds[m][yr] = src.read(1)
    vmax = args.vmax
    # Display aspect of one grid cell (height / width on the ground): the grid is in degrees, so at
    # latitude phi a cell is 1/cos(phi) times taller than wide. Figures use it; the per-year PNGs
    # keep the raw grid (pixel-for-pixel with the GeoTIFFs).
    lat_c = out_t.f + out_t.e * Hc / 2
    cell_aspect = abs(out_t.e) / abs(out_t.a) / math.cos(math.radians(lat_c)) if crs.is_geographic else 1.0
    aspect = Hc * cell_aspect / Wc

    if not args.no_images:
        # ---- Landsat of the core window, shared stretch ----
        r0, c0, h, w = R["landsat_core_window_rc_hw"]
        ls = []
        for yr in years:
            with rasterio.open(R["landsat_paths"][str(yr)]) as src:
                ls.append(src.read([b + 1 for b in args.bands], window=Window(c0, r0, w, h)).astype(np.float32))
        ls = np.stack(ls)
        lo, hi = np.percentile(ls, 2), np.percentile(ls, 98)
        rgb = {yr: ((a - lo) / (hi - lo)).clip(0, 1).transpose(1, 2, 0).repeat(4, 0).repeat(4, 1)
               for yr, a in zip(years, ls)}
        for sub in ["landsat", *MODELS]:
            os.makedirs(os.path.join(args.in_dir, "png", sub), exist_ok=True)
        os.makedirs(os.path.join(args.in_dir, "gif"), exist_ok=True)
        for yr in years:
            Image.fromarray((rgb[yr] * 255).astype(np.uint8)).save(os.path.join(args.in_dir, "png", "landsat", f"{yr}.png"))
            for m in MODELS:
                Image.fromarray(_colormap_u8(preds[m][yr], vmax)).save(os.path.join(args.in_dir, "png", m, f"{yr}.png"))

        # ---- strip: rows Landsat / models, columns years ----
        rows = [("Landsat", None)] + [(m, m) for m in MODELS]
        cw = 2.0
        fig, axes = plt.subplots(len(rows), len(years), figsize=(cw * len(years) + 1.2, cw * aspect * len(rows) + 1.0),
                                 squeeze=False)
        for i, (lab, m) in enumerate(rows):
            for j, yr in enumerate(years):
                ax = axes[i, j]
                if m is None:
                    ax.imshow(rgb[yr], interpolation="bilinear", aspect=cell_aspect)
                else:
                    ax.imshow(np.clip(preds[m][yr], 0, vmax), cmap="plasma", vmin=0, vmax=vmax, interpolation="bilinear",
                              aspect=cell_aspect)
                    ax.set_xlabel(f"mean {np.clip(preds[m][yr], 0, vmax).mean():.1f} m", fontsize=7, labelpad=2)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(str(yr), fontsize=9)
            axes[i, 0].set_ylabel(lab, fontsize=8)
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, vmax))
        fig.colorbar(sm, ax=axes[1:, :], fraction=0.012, pad=0.01, label="height (m)")
        fig.suptitle(f"Landsat tile {R['tile']}, LiDAR footprint (CHM_Denm)", fontsize=10)
        fig.savefig(os.path.join(args.in_dir, "png", "strip.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ---- GIFs ----
        fs = (9.0, 9.0 * aspect + 0.6)
        save_gif(os.path.join(args.in_dir, "gif", "landsat.gif"),
                 [render(rgb[yr], f"Landsat  {yr}", figsize=fs, aspect=cell_aspect) for yr in years], 700, 1500)
        for m in MODELS:
            save_gif(os.path.join(args.in_dir, "gif", f"{m}.gif"),
                     [render(np.clip(preds[m][yr], 0, vmax), f"{m}  {yr}", cmap="plasma", vmax=vmax,
                             cbar_label="height (m)", figsize=fs, aspect=cell_aspect) for yr in years], 700, 1500)
        print("images written")

    # ---- LiDAR evaluation ----
    os.makedirs(os.path.join(args.in_dir, "eval"), exist_ok=True)
    tols = (args.edge_tol, 2 * args.edge_tol)
    gt, valid, res = {}, {}, {}
    for p in R["ref_tifs"]:
        yr = int(os.path.basename(p).split("_")[0])
        if yr not in years:
            print(f"skip LiDAR {yr}: not predicted")
            continue
        h, frac = lidar_on_grid(p, (Hc, Wc), out_t, crs)
        valid[yr] = (frac >= args.min_valid) & np.isfinite(h)
        gt[yr] = h
        res[yr] = {"lidar": p, "valid_cell_frac": float(valid[yr].mean())}
        g = np.clip(np.nan_to_num(h), 0, vmax)
        for m in MODELS:
            pr = preds[m][yr]
            res[yr][m] = metrics(np.clip(pr, 0, vmax), g, valid[yr], vmax, args.edge_thr, tols, args.hf_sigma,
                                 pred_raw=pr, gt_raw=np.nan_to_num(h))
        if not args.no_images:
            fig, axes = plt.subplots(1, 3, figsize=(18, 18 * aspect / 3 + 1.2))
            show = np.where(valid[yr], g, np.nan)
            for ax, (ttl, im) in zip(axes, [(f"LiDAR {yr} (mean to grid)", show)] +
                                     [(f"{m} {yr}", np.clip(preds[m][yr], 0, vmax)) for m in MODELS]):
                ax.imshow(im, cmap="plasma", vmin=0, vmax=vmax, interpolation="bilinear", aspect=cell_aspect)
                ax.set_title(ttl, fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.colorbar(plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, vmax)), ax=axes,
                         fraction=0.015, label="height (m)")
            fig.savefig(os.path.join(args.in_dir, "eval", f"lidar_{yr}.png"), dpi=110, bbox_inches="tight")
            plt.close(fig)

    # change between the first two LiDAR years (needs both predicted)
    change = None
    if len(gt) >= 2:
        y1, y2 = sorted(gt)[:2]
        both = valid[y1] & valid[y2]
        dg = np.clip(gt[y2], 0, vmax) - np.clip(gt[y1], 0, vmax)
        change = {"years": [y1, y2], "n_pixels": int(both.sum()),
                  "lidar_mean_change_m": float(dg[both].mean()),
                  "lidar_loss_gt5m_frac": float((dg[both] < -5).mean())}
        for m in MODELS:
            dp = np.clip(preds[m][y2], 0, vmax) - np.clip(preds[m][y1], 0, vmax)
            a, b_ = dp[both].astype(np.float64), dg[both].astype(np.float64)
            loss_p, loss_g = a < -5, b_ < -5
            tp = (loss_p & loss_g).sum()
            change[m] = {"corr": float(np.corrcoef(a, b_)[0, 1]), "mae_of_change": float(np.abs(a - b_).mean()),
                         "mean_change_m": float(a.mean()),
                         "loss_gt5m_precision": float(tp / max(loss_p.sum(), 1)),
                         "loss_gt5m_recall": float(tp / max(loss_g.sum(), 1))}
    with open(os.path.join(args.in_dir, "eval", "metrics.json"), "w") as f:
        json.dump({"settings": vars(args), "results": {str(k): v for k, v in res.items()}, "change": change},
                  f, indent=2)

    t1, t2 = tols
    keys = [("R2 (higher)", "r2", "{:.4f}"), ("MAE m (lower)", "mae", "{:.3f}"), ("RMSE m (lower)", "rmse", "{:.3f}"),
            ("Grad energy ratio (->1)", "grad_energy_ratio", "{:.4f}"),
            (f"Edge F1 thr{args.edge_thr:g} tol{t1}px (higher)", f"edge_f1_tol{t1}", "{:.4f}"),
            (f"Edge precision tol{t1}px", f"edge_precision_tol{t1}", "{:.4f}"),
            (f"Edge recall tol{t1}px", f"edge_recall_tol{t1}", "{:.4f}"),
            ("HF energy ratio (->1)", "hf_energy_ratio", "{:.4f}"), ("HF correlation (higher)", "hf_corr", "{:.4f}"),
            ("Grad mag error m/px (lower)", "grad_mag_error", "{:.4f}")]
    lines = [f"LiDAR CHM (mean-resampled to the prediction grid, cells >= {args.min_valid:.0%} valid), "
             f"GT and predictions clipped to [0, {vmax:g}] m", ""]
    for yr in sorted(res):
        r = res[yr]
        lines += [f"### {yr}  ({r['valid_cell_frac']:.1%} of the region evaluated)", "",
                  "| metric | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        for lab, k, fmt in keys:
            lines.append(f"| {lab} | " + " | ".join(fmt.format(r[m][k]) for m in MODELS) + " |")
        lines.append("| unclipped R2 / MAE / RMSE | " + " | ".join(
            f"{r[m]['unclipped']['r2']:.3f} / {r[m]['unclipped']['mae']:.2f} / {r[m]['unclipped']['rmse']:.2f}"
            for m in MODELS) + " |")
        lines += ["", "| GT height bin | pixel frac | " + " | ".join(f"{m} MAE / bias" for m in MODELS) + " |",
                  "|---|---|" + "---|" * len(MODELS)]
        for lab in r[MODELS[0]]["height_bins"]:
            hb = r[MODELS[0]]["height_bins"][lab]
            if hb["pixel_frac"] == 0:
                continue
            lines.append(f"| {lab} | {hb['pixel_frac']:.3f} | " + " | ".join(
                f"{r[m]['height_bins'][lab]['mae']:.2f} / {r[m]['height_bins'][lab]['bias']:+.2f}" for m in MODELS) + " |")
        lines.append("")
    if change is not None:
        y1, y2 = change["years"]
        lines += [f"### Change {y1} -> {y2} (cells valid in both years: {change['n_pixels']:,})", "",
                  f"LiDAR: mean change {change['lidar_mean_change_m']:+.2f} m, "
                  f"{change['lidar_loss_gt5m_frac']:.1%} of cells lost > 5 m", "",
                  "| metric | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        for lab, k, fmt in (("corr(pred change, LiDAR change)", "corr", "{:.3f}"),
                            ("MAE of the change (m)", "mae_of_change", "{:.2f}"),
                            ("mean predicted change (m)", "mean_change_m", "{:+.2f}"),
                            ("loss > 5 m: precision", "loss_gt5m_precision", "{:.3f}"),
                            ("loss > 5 m: recall", "loss_gt5m_recall", "{:.3f}")):
            lines.append(f"| {lab} | " + " | ".join(fmt.format(change[m][k]) for m in MODELS) + " |")
    md = "\n".join(lines)
    with open(os.path.join(args.in_dir, "eval", "metrics.md"), "w") as f:
        f.write(md + "\n")
    print(md)
