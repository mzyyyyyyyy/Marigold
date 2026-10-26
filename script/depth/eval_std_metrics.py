"""
Established (published) evaluation metrics for canopy height / depth maps, computed
offline from the raw predictions saved by eval_group_meeting.py --save_dir
(<id>.npz: 'gt' and one array per model, float32 metres, NOT clipped).

Every metric follows a published reference implementation; deviations needed for
canopy height maps are listed as ADAPTATION and also written to the output.

Canopy height (Open-Canopy, Fogel et al. 2024, src/metrics/compute_metrics.py):
  pixels with 0 <= GT < 60 m (np.digitize on bins [0,2,5,10,15,20,30,60]), pixel-pooled
  MAE, RMSE, bias (mean of pred - GT), nMAE = mean |pred - GT| / (1 + |GT|) over GT >= 2 m,
  tree cover IoU at 2 m = |GT>=2 & pred>=2| / (|GT>=2 | pred>=2| + 1),
  the same per GT height bin. R2 (pixel-pooled, as torchmetrics.R2Score).
  ADAPTATION: Open-Canopy evaluates inside a vegetation mask; we have none -> all pixels.
Canopy height (Tolan et al. 2024, RSE; facebookresearch/HighResCanopyHeight inference.py):
  Block R2: drop a 3 px border, AvgPool2d(block), R2 of block means, pooled.
    ADAPTATION: block = 10 px (~30 m on our ~3 m grid) instead of 50 px (= 30 m at 0.6 m).
  Edge Error (paper Appendix C.3, Algorithm 1; no public code):
    EE = sum|E(pred) - E(GT)| / (sum|E(pred)| + sum|E(GT)|), E = Sobel gradient magnitude.
    Unspecified in the paper, our choice: 3x3 Sobel, 1 px border removed, per patch, then
    the mean over patches (the pooled ratio is also reported). EE is invariant to the
    Sobel kernel scale.
Depth boundaries (iBims-1, Koch et al. 2018; official compute_depth_boundary_error by
  T. Koch, as shipped in xuefeng-cvr/BS-Net evaluate_ibims_error_metrics.py):
  per patch: prediction min-max normalised to [0,1], skimage Canny (sigma sqrt(2), low 0.15,
  high 0.3); Euclidean distance transforms; theta = 10 px;
  eps_acc = mean distance of the predicted edges within theta of a GT edge to the GT edges
            (theta if no such edge);
  eps_comp = (sum of truncated dist(pred->GT) + truncated dist(GT->pred)) / (#pred + #GT edges)
            (the symmetric form of the official code, not the one-sided formula of the paper);
  nan if the GT has no edge; mean over patches (nanmean).
  ADAPTATION: (1) iBims GT edges are hand-curated; we extract GT edges with the same Canny
  pipeline on the GT; (2) the official code treats depth 0 as missing (set to NaN before
  normalising); in a CHM 0 m is valid ground, so zeros are kept.
Depth boundaries (Soft Edge Error, SMD-Nets, Tosi et al. 2021, lib/evaluation_utils.py; also
  used by PatchFusion): GT boundaries = pixels whose |difference| to a 4-neighbour > th
  (both sides; get_boundaries with dilation 0); SEE_k = mean over boundary pixels of
  min over the k x k shifts of |shift(GT) - pred| (shift fills with 0, as shift_2d_replace);
  per image, then the mean over images. k = 3 and 5 (as in the paper).
  ADAPTATION: th = 1 disparity px in the paper -> th in metres (2 m and 5 m); patches
  without a boundary pixel are skipped (the official loop would count them as 0).

Output: <out_dir>/std_metrics.{json,md}, per_patch.csv
"""

import argparse
import csv
import glob
import json
import math
import os
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from skimage import feature

MODELS = ["baseline_dav2", "baseline_sr4ir_reg", "fm_refiner-R-lumi-4", "sr_fm_refiner_v11-lumi"]
OC_BINS = [0, 2, 5, 10, 15, 20, 30, 60]      # Open-Canopy default bins (m)
TREE_THR = 2.0                               # Open-Canopy tree cover threshold (m)
BLOCK, BLOCK_BD = 10, 3                      # Tolan block R2 (block adapted, border as in the code)
IBIMS_THETA = 10.0                           # iBims-1 max_dist_thr (px)
SEE_THS = (2.0, 5.0)                         # SEE boundary thresholds (m), adapted from 1 disparity px
SEE_KS = (3, 5)


# -------------------------------------------------------------------------
# Ports of the reference implementations
# -------------------------------------------------------------------------

def ibims_dbe(edges_gt, pred, theta=IBIMS_THETA, zero_is_missing=False):
    """Port of compute_depth_boundary_error (T. Koch, iBims-1). Returns (acc, comp, edges_est)."""
    if np.sum(edges_gt) == 0:
        return np.nan, np.nan, np.zeros(pred.shape, bool)
    pn = pred.copy().astype("f")
    if zero_is_missing:                       # official behaviour; off for CHMs (ADAPTATION)
        pn[pn == 0] = np.nan
    pn = pn - np.nanmin(pn)
    mx = np.nanmax(pn)
    if not mx > 0:                            # flat prediction: no edges (guard against 0/0)
        edges_est = np.zeros(pred.shape, bool)
    else:
        pn = pn / mx
        edges_est = feature.canny(pn, sigma=np.sqrt(2), low_threshold=0.15, high_threshold=0.3)
    D_gt = ndimage.distance_transform_edt(1 - edges_gt)
    D_est = ndimage.distance_transform_edt(1 - edges_est)
    mask_D_gt = D_gt < theta
    E_fin_est_filt = edges_est * mask_D_gt
    if np.sum(E_fin_est_filt) == 0:
        return theta, theta, edges_est
    acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)
    ch1 = D_gt * edges_est
    ch1[ch1 > theta] = theta
    ch2 = D_est * edges_gt
    ch2[ch2 > theta] = theta
    comp = np.nansum(ch1 + ch2) / (np.nansum(edges_est) + np.nansum(edges_gt))
    return acc, comp, edges_est


def ibims_gt_edges(gt):
    """ADAPTATION: GT edges with the iBims prediction pipeline (min-max + Canny) instead of hand curation."""
    g = gt.astype("f") - gt.min()
    if not g.max() > 0:
        return np.zeros(gt.shape, bool)
    return feature.canny(g / g.max(), sigma=np.sqrt(2), low_threshold=0.15, high_threshold=0.3)


def shift_2d_replace(data, dx, dy, constant=0):
    """Port of SMD-Nets lib/utils.py shift_2d_replace."""
    s = np.roll(data, dx, axis=1)
    if dx < 0:
        s[:, dx:] = constant
    elif dx > 0:
        s[:, 0:dx] = constant
    s = np.roll(s, dy, axis=0)
    if dy < 0:
        s[dy:, :] = constant
    elif dy > 0:
        s[0:dy, :] = constant
    return s


def get_boundaries(d, th):
    """Port of SMD-Nets lib/utils.py get_boundaries with dilation = 0."""
    ey = np.logical_or(np.pad(np.abs(d[1:, :] - d[:-1, :]) > th, ((1, 0), (0, 0))),
                       np.pad(np.abs(d[:-1, :] - d[1:, :]) > th, ((0, 1), (0, 0))))
    ex = np.logical_or(np.pad(np.abs(d[:, 1:] - d[:, :-1]) > th, ((0, 0), (1, 0))),
                       np.pad(np.abs(d[:, :-1] - d[:, 1:]) > th, ((0, 0), (0, 1))))
    return np.logical_or(ey, ex)


def soft_edge_error(pred, gt, radius):
    """Port of SMD-Nets lib/evaluation_utils.py soft_edge_error."""
    return np.minimum.reduce([np.abs(shift_2d_replace(gt, i, j, 0) - pred)
                              for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)])


def sobel_mag(x):
    gx = ndimage.sobel(x, axis=1, mode="nearest")
    gy = ndimage.sobel(x, axis=0, mode="nearest")
    return np.hypot(gx, gy)[1:-1, 1:-1]        # 1 px border removed (border handling unspecified)


def block_means(x, b=BLOCK, bd=BLOCK_BD):
    """As nn.AvgPool2d(b) on x[bd:, bd:] (incomplete blocks dropped)."""
    x = x[bd:, bd:]
    h, w = (x.shape[0] // b) * b, (x.shape[1] // b) * b
    return x[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


# -------------------------------------------------------------------------
# Per patch
# -------------------------------------------------------------------------

def zero_acc():
    nb = len(OC_BINS) - 1
    return {"n": 0.0, "se": 0.0, "ae": 0.0, "e": 0.0, "g": 0.0, "g2": 0.0,
            "nmae_sum": 0.0, "nmae_n": 0.0, "iou_i": 0.0, "iou_u": 0.0,
            "bin_n": [0.0] * nb, "bin_ae": [0.0] * nb, "bin_se": [0.0] * nb, "bin_e": [0.0] * nb,
            "bin_nae": [0.0] * nb, "bin_iou_i": [0.0] * nb, "bin_iou_u": [0.0] * nb,
            "blk_n": 0.0, "blk_se": 0.0, "blk_g": 0.0, "blk_g2": 0.0,
            "ee_num": 0.0, "ee_den": 0.0}


def process(path):
    z = np.load(path)
    gt = z["gt"].astype(np.float64)
    pid = os.path.basename(path)[:-4]
    valid = (gt >= OC_BINS[0]) & (gt < OC_BINS[-1])
    bin_idx = np.digitize(gt, OC_BINS) - 1                    # 0..6 inside the valid range
    gt_edges = ibims_gt_edges(gt)
    see_masks = {th: get_boundaries(gt, th) for th in SEE_THS}
    Eg = sobel_mag(gt)
    gblk = block_means(gt)
    out = {"id": pid, "models": {}}
    for m in MODELS:
        p = z[m].astype(np.float64)
        a = zero_acc()
        e = (p - gt)[valid]
        g = gt[valid]
        pv = p[valid]
        a["n"], a["se"], a["ae"], a["e"] = e.size, float((e ** 2).sum()), float(np.abs(e).sum()), float(e.sum())
        a["g"], a["g2"] = float(g.sum()), float((g ** 2).sum())
        tall = g >= OC_BINS[1]
        a["nmae_sum"] = float((np.abs(e[tall]) / (1 + np.abs(g[tall]))).sum())
        a["nmae_n"] = float(tall.sum())
        tg, tp = g >= TREE_THR, pv >= TREE_THR
        a["iou_i"], a["iou_u"] = float((tg & tp).sum()), float((tg | tp).sum())
        bi = bin_idx[valid]
        for k in range(len(OC_BINS) - 1):
            s = bi == k
            if s.any():
                es, gs = e[s], g[s]
                a["bin_n"][k] = float(s.sum())
                a["bin_ae"][k] = float(np.abs(es).sum())
                a["bin_se"][k] = float((es ** 2).sum())
                a["bin_e"][k] = float(es.sum())
                a["bin_nae"][k] = float((np.abs(es) / (1 + np.abs(gs))).sum())
                a["bin_iou_i"][k] = float((tg[s] & tp[s]).sum())
                a["bin_iou_u"][k] = float((tg[s] | tp[s]).sum())
        pb = block_means(p)
        a["blk_n"], a["blk_se"] = float(pb.size), float(((pb - gblk) ** 2).sum())
        a["blk_g"], a["blk_g2"] = float(gblk.sum()), float((gblk ** 2).sum())
        Ep = sobel_mag(p)
        a["ee_num"], a["ee_den"] = float(np.abs(Ep - Eg).sum()), float(np.abs(Ep).sum() + np.abs(Eg).sum())
        ee_patch = a["ee_num"] / a["ee_den"] if a["ee_den"] > 0 else 0.0
        acc, comp, _ = ibims_dbe(gt_edges, p)
        see = {}
        for th in SEE_THS:
            msk = see_masks[th]
            for k in SEE_KS:
                see[f"see{k}_th{th:g}"] = float(soft_edge_error(p, gt, k // 2)[msk].mean()) if msk.any() else np.nan
        out["models"][m] = {"acc": a, "ee": ee_patch, "dbe_acc": float(acc), "dbe_comp": float(comp), **see}
    out["n_see_px"] = {f"th{th:g}": int(see_masks[th].sum()) for th in SEE_THS}
    out["n_gt_edges"] = int(gt_edges.sum())
    return out


# -------------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------------

def finalize(rows, m):
    A = zero_acc()
    for r in rows:
        a = r["models"][m]["acc"]
        for k, v in a.items():
            if isinstance(v, list):
                A[k] = [x + y for x, y in zip(A[k], v)]
            else:
                A[k] += v
    res = {"mae": A["ae"] / A["n"], "rmse": math.sqrt(A["se"] / A["n"]), "bias": A["e"] / A["n"],
           "r2": 1 - A["se"] / (A["g2"] - A["g"] ** 2 / A["n"]),
           "nmae": A["nmae_sum"] / A["nmae_n"], "tree_iou": A["iou_i"] / (A["iou_u"] + 1),
           "block_r2": 1 - A["blk_se"] / (A["blk_g2"] - A["blk_g"] ** 2 / A["blk_n"]),
           "edge_error": float(np.mean([r["models"][m]["ee"] for r in rows])),
           "edge_error_pooled": A["ee_num"] / A["ee_den"],
           "dbe_acc": float(np.nanmean([r["models"][m]["dbe_acc"] for r in rows])),
           "dbe_comp": float(np.nanmean([r["models"][m]["dbe_comp"] for r in rows])),
           "n_patches_dbe": int(np.sum(~np.isnan([r["models"][m]["dbe_acc"] for r in rows])))}
    for th in SEE_THS:
        for k in SEE_KS:
            key = f"see{k}_th{th:g}"
            v = np.array([r["models"][m][key] for r in rows], float)
            res[key] = float(np.nanmean(v))
            res[f"n_patches_{key}"] = int((~np.isnan(v)).sum())
    res["bins"] = {}
    for k in range(len(OC_BINS) - 1):
        n = A["bin_n"][k]
        lab = f"[{OC_BINS[k]},{OC_BINS[k + 1]})"
        res["bins"][lab] = {"pixel_frac": n / A["n"],
                            "mae": A["bin_ae"][k] / n if n else float("nan"),
                            "rmse": math.sqrt(A["bin_se"][k] / n) if n else float("nan"),
                            "bias": A["bin_e"][k] / n if n else float("nan"),
                            "nmae": A["bin_nae"][k] / n if n else float("nan"),
                            "tree_iou": A["bin_iou_i"][k] / (A["bin_iou_u"][k] + 1)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="/flash/project_465002934/result_group_meeting/raw_preds")
    ap.add_argument("--psnr_json", default="/flash/project_465002934/result_group_meeting/metrics_psnr_ssim/detail_metrics.json",
                    help="PSNR / SSIM of eval_group_meeting.py, added to the summary table")
    ap.add_argument("--out_dir", default="/flash/project_465002934/result_group_meeting/metrics_std")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} patches from {args.pred_dir}; {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(process, files, chunksize=4), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"{i}/{len(files)}", flush=True)
    results = {m: finalize(rows, m) for m in MODELS}
    psnr = {}
    if args.psnr_json and os.path.exists(args.psnr_json):
        with open(args.psnr_json) as f:
            pj = json.load(f)["results"]
        psnr = {m: {k: pj[m][k] for k in ("psnr", "psnr_global", "ssim")} for m in MODELS if m in pj}
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"n_patches": len(rows), "pred_dir": args.pred_dir, "values": "raw metres, not clipped",
            "open_canopy_bins": OC_BINS, "tree_threshold_m": TREE_THR, "block_px": BLOCK, "block_border_px": BLOCK_BD,
            "ibims_theta_px": IBIMS_THETA, "see_thresholds_m": list(SEE_THS), "see_k": list(SEE_KS),
            "adaptations": [
                "Open-Canopy metrics on all pixels (no vegetation mask available)",
                "Block R2 with 10 px blocks (~30 m, the ground size of Tolan's 50 px at 0.6 m)",
                "Edge Error: no public code; 3x3 Sobel magnitude, 1 px border removed, per-patch mean",
                "iBims DBE: GT edges by the same Canny pipeline (not hand-curated); 0 m kept as valid",
                "SEE: boundary threshold in metres (2, 5 m) instead of 1 disparity px; edgeless patches skipped"],
            "psnr_ssim_source": args.psnr_json if psnr else None}
    with open(os.path.join(args.out_dir, "std_metrics.json"), "w") as f:
        json.dump({"meta": meta, "results": results, "psnr_ssim": psnr}, f, indent=2)
    with open(os.path.join(args.out_dir, "per_patch.csv"), "w", newline="") as f:
        keys = ["ee", "dbe_acc", "dbe_comp"] + [f"see{k}_th{th:g}" for th in SEE_THS for k in SEE_KS]
        w = csv.writer(f)
        w.writerow(["id", "n_gt_edges"] + [f"{k}_{m}" for m in MODELS for k in keys])
        for r in rows:
            w.writerow([r["id"], r["n_gt_edges"]] + [r["models"][m][k] for m in MODELS for k in keys])

    R = results
    rows_md = [
        ("**Canopy height (Open-Canopy / Tolan)**", None, None),
        ("MAE m (lower)", "mae", "{:.3f}"), ("RMSE m (lower)", "rmse", "{:.3f}"), ("Bias m (->0)", "bias", "{:+.3f}"),
        ("R2 (higher)", "r2", "{:.4f}"), ("nMAE, GT >= 2 m (lower)", "nmae", "{:.4f}"),
        ("Tree cover IoU @ 2 m (higher)", "tree_iou", "{:.4f}"),
        (f"Block R2, {BLOCK} px blocks (higher)", "block_r2", "{:.4f}"),
        ("Edge Error, Tolan et al. (lower)", "edge_error", "{:.4f}"),
        ("**Depth boundaries**", None, None),
        (f"iBims DBE accuracy px (lower)", "dbe_acc", "{:.3f}"),
        (f"iBims DBE completeness px (lower)", "dbe_comp", "{:.3f}"),
    ] + [(f"SEE_{k}, boundary th {th:g} m (lower, m)", f"see{k}_th{th:g}", "{:.3f}") for th in SEE_THS for k in SEE_KS]
    lines = [f"Test split, {len(rows)} patches of 480 px; raw (unclipped) heights; GT in [0, 60) m", "",
             "| metric | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for lab, key, fmt in rows_md:
        if key is None:
            lines.append(f"| {lab} |" + " |" * len(MODELS))
            continue
        lines.append(f"| {lab} | " + " | ".join(fmt.format(R[m][key]) for m in MODELS) + " |")
    if psnr:
        lines.append("| **Super-resolution (clipped to [0, 28] m, R = 28 m)** |" + " |" * len(MODELS))
        for lab, key, fmt in (("PSNR dB, mean over patches (higher)", "psnr", "{:.2f}"),
                              ("SSIM (higher)", "ssim", "{:.4f}")):
            lines.append(f"| {lab} | " + " | ".join(fmt.format(psnr[m][key]) for m in MODELS) + " |")
    lines += ["", "Open-Canopy height bins (by GT): MAE / bias / nMAE / tree IoU", "",
              "| bin m | pixel frac | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for lab in R[MODELS[0]]["bins"]:
        b0 = R[MODELS[0]]["bins"][lab]
        lines.append(f"| {lab} | {b0['pixel_frac']:.3f} | " + " | ".join(
            f"{R[m]['bins'][lab]['mae']:.2f} / {R[m]['bins'][lab]['bias']:+.2f} / {R[m]['bins'][lab]['nmae']:.3f} / "
            f"{R[m]['bins'][lab]['tree_iou']:.3f}" for m in MODELS) + " |")
    lines += ["", f"Patches with GT edges (iBims): {R[MODELS[0]]['n_patches_dbe']}; " + ", ".join(
        f"SEE th {th:g} m: {R[MODELS[0]][f'n_patches_see3_th{th:g}']}" for th in SEE_THS),
              "", "Adaptations: " + "; ".join(meta["adaptations"])]
    md = "\n".join(lines)
    with open(os.path.join(args.out_dir, "std_metrics.md"), "w") as f:
        f.write(md + "\n")
    print(md)
