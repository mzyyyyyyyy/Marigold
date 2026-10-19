"""
Where do real PS and pseudo-PS make the FM differ? Spatial maps of the TDP
feature difference (same construction as diag_tdp.py / _sr_step).

For --n_patches evenly spaced test patches and t in T_LIST (fixed eps):
  feature diff   ||f(real) - f(pseudo)||_2 over channels at the tdp_hook,
                 at latent resolution, relative to the mean ||f(pseudo)||
  height diff    |x1(real) - x1(pseudo)| in metres, x1 = decode(z_t + (1-t) v)
and, for reference, the image-level difference |real PS - pseudo-PS| and the
GT height gradient (Sobel, m/px). Quantitative summary pooled over patches:
  - edge enrichment of the height diff: share of |dh| on GT edge pixels
    (gradient >= 2 m/px) / share of edge pixels (1 = no preference)
  - Pearson correlation of |dh| with the GT gradient magnitude and with the
    image-level PS difference; same for the feature diff at latent resolution
    (GT gradient / image diff average-pooled 8x8)

Output: <out_dir>/panels/*.png, <out_dir>/feat_diff.md, feat_diff.json.
Launch: script/depth/diag_feat_diff.sh (one GPU per --run).
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import eval_detail_metrics as edm
from diag_refiner_gain import load as load_run
from diag_tdp import TDPProbe, make_eps
from eval_detail_metrics_nd import make_test_dataset

OUT_ROOT = edm.OUT_ROOT
T_LIST = [0.0, 0.3, 0.6, 0.9]
EDGE_THR = 2.0   # m/px, as in eval_detail_metrics


def sobel_mag(h):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=h.device).view(1, 1, 3, 3) / 8.0
    hp = F.pad(h, (1, 1, 1, 1), mode="replicate")
    return torch.sqrt(F.conv2d(hp, kx) ** 2 + F.conv2d(hp, kx.transpose(2, 3)) ** 2)


class Corr:
    """Pooled Pearson correlation from streaming sums."""

    def __init__(self):
        self.s = np.zeros(6)   # n, sx, sy, sxx, syy, sxy

    def add(self, x, y):
        x, y = x.flatten().double(), y.flatten().double()
        self.s += np.array([x.numel(), x.sum().item(), y.sum().item(), (x * x).sum().item(),
                            (y * y).sum().item(), (x * y).sum().item()])

    def value(self):
        n, sx, sy, sxx, syy, sxy = self.s
        cov = sxy / n - sx * sy / n ** 2
        return cov / np.sqrt(max(sxx / n - (sx / n) ** 2, 1e-30) * max(syy / n - (sy / n) ** 2, 1e-30))


def _rgb(x):
    return ((x[:3].float().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


def _np(x):
    return x.squeeze().float().cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=f"{OUT_ROOT}/sr_fm_refiner_v13-lumi-nd")
    ap.add_argument("--ckpt", default=None, help="default: <run>/checkpoint/best.pth")
    ap.add_argument("--out_dir", default=None, help="default: <OUT_ROOT>/_diag_feat_diff/<run name>")
    ap.add_argument("--n_patches", type=int, default=32, help="test patches for the quantitative summary")
    ap.add_argument("--n_panels", type=int, default=8, help="first n of those get a figure")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = torch.device("cuda:0")
    run_name = os.path.basename(args.run.rstrip("/"))
    out_dir = args.out_dir or f"{OUT_ROOT}/_diag_feat_diff/{run_name}"
    os.makedirs(os.path.join(out_dir, "panels"), exist_ok=True)
    ckpt = args.ckpt or os.path.join(args.run, "checkpoint", "best.pth")

    cfg, chm, fm, sr, n_ls = load_run("sr", args.run, ckpt, device, True)
    sr.eval()
    probe = TDPProbe(fm, str(cfg.sr_loss.get("tdp_hook", "unet.up_blocks.3")))
    cfg_d = cfg.dataset
    stats = edm._load_target_stats(cfg_d.target_stats_file, cfg_d.year)
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    test_ds, _ = make_test_dataset(cfg_d)
    sel = sorted(set(np.linspace(0, len(test_ds) - 1, args.n_patches).round().astype(int).tolist()))

    corr = {f"{k}/{t}": Corr() for t in T_LIST for k in
            ("dh_vs_gtgrad", "dh_vs_imgdiff", "feat_vs_gtgrad", "feat_vs_imgdiff")}
    S = {f"{k}/{t}": 0.0 for t in T_LIST for k in ("dh_edge", "dh_all", "feat_rel", "dh_mean")}
    S.update(edge_px=0.0, all_px=0.0, n=0)

    with torch.no_grad():
        for j, i in enumerate(sel):
            inputs_lr, inputs_hr, targets = [x.unsqueeze(0) if x.dim() == 3 else x for x in test_ds[i]]
            inputs_lr, real_ps, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
            H, W = targets.shape[-2:]
            landsat = inputs_lr[:, :n_ls]
            landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)
            h_coarse = edm._normalize_coarse(chm((landsat + 1.0) / 2.0, target_size=(H, W)), stats)
            z_c, z_gt = fm.encode(h_coarse), fm.encode(targets)
            gt_m = edm._denormalize(targets, stats).clamp(p1, p99)
            pseudo = sr(landsat, target_size=(H, W))
            gt_grad = sobel_mag(gt_m)
            edge = (gt_grad >= EDGE_THR).float()
            img_diff = (real_ps - pseudo).abs().mean(1, keepdim=True)
            gt_grad_lat = F.avg_pool2d(gt_grad, 8)
            img_diff_lat = F.avg_pool2d(img_diff, 8)
            S["edge_px"] += edge.sum().item()
            S["all_px"] += edge.numel()
            S["n"] += 1

            maps = {}
            for k, t in enumerate(T_LIST):
                eps = make_eps(z_c.shape, device, i * 100 + k)
                zt = probe.z_t(z_c, z_gt, t, eps)
                f_r, v_r = probe.run(fm._maybe_concat_landsat(real_ps, landsat_hr), zt, z_c, t)
                f_f, v_f = probe.run(fm._maybe_concat_landsat(pseudo, landsat_hr), zt, z_c, t)
                fdiff = (f_r - f_f).norm(dim=1, keepdim=True) / f_f.norm(dim=1).mean()
                x1_r = edm._denormalize(fm.decode(zt + (1 - t) * v_r), stats).clamp(p1, p99)
                x1_f = edm._denormalize(fm.decode(zt + (1 - t) * v_f), stats).clamp(p1, p99)
                dh = (x1_r - x1_f).abs()
                corr[f"dh_vs_gtgrad/{t}"].add(dh, gt_grad)
                corr[f"dh_vs_imgdiff/{t}"].add(dh, img_diff)
                corr[f"feat_vs_gtgrad/{t}"].add(fdiff, gt_grad_lat)
                corr[f"feat_vs_imgdiff/{t}"].add(fdiff, img_diff_lat)
                S[f"dh_edge/{t}"] += (dh * edge).sum().item()
                S[f"dh_all/{t}"] += dh.sum().item()
                S[f"dh_mean/{t}"] += dh.mean().item()
                S[f"feat_rel/{t}"] += fdiff.mean().item()
                maps[t] = (fdiff, dh)

            if j < args.n_panels:
                fig, ax = plt.subplots(3, 5, figsize=(17, 10.4))
                vmax_f = max(_np(maps[t][0]).max() for t in T_LIST)
                vmax_h = max(float(np.percentile(_np(maps[t][1]), 99.5)) for t in T_LIST)
                top = [("Landsat (bilinear)", _rgb(landsat_hr[0]), None), ("pseudo-PS (SR)", _rgb(pseudo[0]), None),
                       ("real PS", _rgb(real_ps[0]), None),
                       ("|real - pseudo| (image)", _np(img_diff), "magma"), ("GT height (m)", _np(gt_m), "viridis")]
                for a, (name, im, cm) in zip(ax[0], top):
                    a.imshow(im, cmap=cm, **({"vmin": 0, "vmax": p99} if name.startswith("GT") else {}))
                    a.set_title(name, fontsize=9)
                for a, t in zip(ax[1], T_LIST):
                    a.imshow(_np(maps[t][0]), cmap="magma", vmin=0, vmax=vmax_f, interpolation="nearest")
                    a.set_title(f"feature diff, t={t} (latent res)", fontsize=9)
                ax[1][4].imshow(_np(gt_grad), cmap="gray", vmin=0, vmax=4)
                ax[1][4].set_title("GT gradient (m/px)", fontsize=9)
                for a, t in zip(ax[2], T_LIST):
                    a.imshow(_np(maps[t][1]), cmap="magma", vmin=0, vmax=vmax_h)
                    a.set_title(f"|x1(real) - x1(pseudo)| (m), t={t}", fontsize=9)
                ax[2][4].imshow(_np(maps[T_LIST[0]][1]), cmap="magma", vmin=0, vmax=vmax_h)
                ax[2][4].contour(_np(edge), levels=[0.5], colors="cyan", linewidths=0.4)
                ax[2][4].set_title(f"height diff t=0 + GT edges (>= {EDGE_THR:g} m/px)", fontsize=9)
                for a in ax.flat:
                    a.axis("off")
                fig.suptitle(f"{run_name}  test patch {i}", fontsize=11)
                plt.tight_layout()
                fig.savefig(os.path.join(out_dir, "panels", f"patch{i:05d}.png"), dpi=100)
                plt.close(fig)
            logging.info(f"patch {i} done ({j + 1}/{len(sel)})")

    n, edge_share = S["n"], S["edge_px"] / S["all_px"]
    res = {"run": args.run, "ckpt": ckpt, "n_patches": n, "edge_pixel_share": edge_share, "per_t": {}}
    L = [f"# Real-vs-pseudo PS feature/height difference: {run_name}", "",
         f"{n} test patches; hook `{cfg.sr_loss.get('tdp_hook')}`; GT edge pixels (>= {EDGE_THR:g} m/px): "
         f"{edge_share:.3f} of all pixels", "",
         "| t | mean feature diff (rel.) | mean |dh| (m) | share of |dh| on GT edges | edge enrichment | "
         "corr(|dh|, GT grad) | corr(|dh|, image diff) | corr(feat diff, GT grad) | corr(feat diff, image diff) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t in T_LIST:
        share = S[f"dh_edge/{t}"] / S[f"dh_all/{t}"]
        r = {"feat_rel": S[f"feat_rel/{t}"] / n, "dh_mean_m": S[f"dh_mean/{t}"] / n, "dh_edge_share": share,
             "edge_enrichment": share / edge_share,
             **{k: corr[f"{k}/{t}"].value() for k in ("dh_vs_gtgrad", "dh_vs_imgdiff", "feat_vs_gtgrad", "feat_vs_imgdiff")}}
        res["per_t"][str(t)] = r
        L.append(f"| {t} | {r['feat_rel']:.3f} | {r['dh_mean_m']:.3f} | {r['dh_edge_share']:.3f} | "
                 f"{r['edge_enrichment']:.2f} | {r['dh_vs_gtgrad']:+.3f} | {r['dh_vs_imgdiff']:+.3f} | "
                 f"{r['feat_vs_gtgrad']:+.3f} | {r['feat_vs_imgdiff']:+.3f} |")
    with open(os.path.join(out_dir, "feat_diff.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, "feat_diff.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    logging.info("\n" + "\n".join(L))
