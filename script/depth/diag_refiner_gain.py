"""
Diagnostics for "why does the FM refiner only slightly improve accuracy over the
frozen CHMv2 baseline?" (see the discussion around sr_fm_refiner_v12-lumi).

For one refiner run (SR+pseudo-PS refiner "sr", or the Landsat-only refiner
"plain" of finer_lumi) it draws K independent samples per validation patch and
reports, all in metres, with prediction and GT clipped to [p1, p99] (same
convention as eval_detail_metrics.py):

  1. n-draw curve: R2 / MAE / RMSE of the mean of the first n draws, n in --n_list,
     plus the baseline. For an (ideal) posterior sampler
        E[MSE(mean of n draws)] = M_inf + V / n,
     V = mean across-draw variance, M_inf = MSE of the posterior-mean prediction.
     We report V (unbiased) and the extrapolated M_inf = MSE(K) - V/K, i.e. the
     accuracy the refiner would reach with infinitely many draws, and the
     single-draw penalty V.
  2. stratified error, baseline vs refiner (single draw / mean of K draws):
       a. by GT height bin:        MSE, MAE, bias, across-draw variance
       b. by signed baseline error (baseline - gt): MSE and the fraction of the
          baseline error the refiner corrects (-mean(pred-coarse)/mean(base err)).

For a deterministic refiner (noise_sigma = 0) use --K 1. --n_steps may be a
comma list (e.g. "1,4,10"); each is evaluated on the same patches.

Launch (one task per GPU): script/depth/diag_refiner_gain.sh
"""

import argparse
import json
import logging
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import eval_detail_metrics as edm
from depthfm import fm_refiner as sr_fm_mod
from depthfm import fm_refiner_finer as plain_fm_mod
from depthfm.chmv2 import load_chmv2_baseline
from depthfm.sr_module import build_sr_module

OUT_ROOT = edm.OUT_ROOT
GT_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]      # bins [0,5) ... [25, inf) (GT is clipped at p99=28)
ERR_EDGES = [-6.0, -3.0, -1.5, -0.5, 0.5, 1.5, 3.0, 6.0]   # signed baseline error (m), 8 inner edges -> 9 bins


def _bin_labels(edges, lo="-inf", hi="inf"):
    e = [lo] + [f"{x:g}" for x in edges] + [hi]
    return [f"[{e[i]},{e[i + 1]})" for i in range(len(e) - 1)]


class Acc:
    """Sufficient statistics for the pooled metrics and the two stratifications."""

    def __init__(self, device, n_list):
        self.n_list = n_list
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.dev = device
        # global
        self.N = z(1)
        self.sum_g, self.sum_gg = z(1), z(1)
        self.base = z(2)                       # sse, sae
        self.nd = z(len(n_list), 2)            # sse, sae of mean-of-n draws
        self.var_sum = z(1)                    # sum of unbiased across-draw variance
        # stratified. columns documented in update()
        self.gt_bins = z(len(GT_EDGES), 7)
        self.err_bins = z(len(ERR_EDGES) + 1, 7)
        self.gt_edges_t = torch.tensor(GT_EDGES, device=device)
        self.err_edges_t = torch.tensor(ERR_EDGES, device=device)

    @staticmethod
    def _scatter(tab, idx, cols):
        for j, c in enumerate(cols):
            tab[:, j].index_add_(0, idx, c.double())

    def update(self, base, draws, gt):
        """base, gt: (1,1,H,W) metres; draws: (K,1,H,W) metres."""
        K = draws.shape[0]
        g = gt.flatten().double()
        b = base.flatten().double()
        d = draws.flatten(1).double()                               # (K, P)
        cum = d.cumsum(0) / torch.arange(1, K + 1, device=d.device).view(-1, 1)
        self.N += g.numel()
        self.sum_g += g.sum()
        self.sum_gg += (g * g).sum()
        self.base += torch.stack([((b - g) ** 2).sum(), (b - g).abs().sum()])
        for i, n in enumerate(self.n_list):
            e = cum[n - 1] - g
            self.nd[i] += torch.stack([(e ** 2).sum(), e.abs().sum()])
        var = d.var(0, unbiased=True) if K > 1 else torch.zeros_like(g)
        self.var_sum += var.sum()

        single, meanK = d[0], cum[-1]
        # columns: count, sse_base, sse_single, sse_meanK, sum(base-gt), sum(meanK-gt), sum var
        # (for err bins, column 5 is replaced by sum(meanK-base) so the "fraction corrected" is available)
        cols_common = [torch.ones_like(g), (b - g) ** 2, (single - g) ** 2, (meanK - g) ** 2, b - g]
        gi = torch.bucketize(g, self.gt_edges_t, right=True) - 1
        self._scatter(self.gt_bins, gi.clamp(min=0), cols_common + [meanK - g, var])
        ei = torch.bucketize(b - g, self.err_edges_t, right=True)
        self._scatter(self.err_bins, ei, cols_common + [meanK - b, var])

    def state(self):
        return {k: getattr(self, k).cpu() for k in
                ("N", "sum_g", "sum_gg", "base", "nd", "var_sum", "gt_bins", "err_bins")}


def finalize(states, n_list, K):
    S = {k: sum(s[k] for s in states) for k in states[0]}
    N = S["N"].item()
    sst = S["sum_gg"].item() - S["sum_g"].item() ** 2 / N

    def met(sse, sae):
        return {"r2": 1.0 - sse / sst, "mae": sae / N, "rmse": (sse / N) ** 0.5, "mse": sse / N}

    out = {"N_pixels": N, "K": K, "baseline": met(*S["base"].tolist()), "mean_of_n": {}}
    for i, n in enumerate(n_list):
        out["mean_of_n"][str(n)] = met(*S["nd"][i].tolist())
    V = S["var_sum"].item() / N
    mseK = out["mean_of_n"][str(n_list[-1])]["mse"]
    out["across_draw_var_V"] = V
    out["mse_inf_extrapolated"] = mseK - V / K
    out["rmse_inf_extrapolated"] = max(mseK - V / K, 0.0) ** 0.5
    out["r2_inf_extrapolated"] = 1.0 - (mseK - V / K) * N / sst
    if K > 1:
        # mean MSE of a single draw implied by the fit M_inf + V (cross-check vs measured n=1)
        out["single_draw_mse_model"] = out["mse_inf_extrapolated"] + V

    def strat(tab, labels, shift_is_vs_base):
        rows = {}
        for lab, r in zip(labels, tab.tolist()):
            n = r[0]
            if n == 0:
                continue
            row = {"pixel_frac": n / N, "mse_base": r[1] / n, "mse_single": r[2] / n, "mse_meanK": r[3] / n,
                   "bias_base": r[4] / n, "across_draw_var": r[6] / n}
            if shift_is_vs_base:
                row["mean_shift_meanK_minus_base"] = r[5] / n
                row["frac_base_error_corrected"] = (-r[5] / r[4]) if abs(r[4]) > 1e-9 else float("nan")
            else:
                row["bias_meanK"] = r[5] / n
            rows[lab] = row
        # contribution to total SSE change (negative = refiner helps), single draw and mean of K
        tot = tab[:, 1].sum().item()
        for lab, r in zip(labels, tab.tolist()):
            if lab in rows:
                rows[lab]["dSSE_single_share"] = (r[2] - r[1]) / tot
                rows[lab]["dSSE_meanK_share"] = (r[3] - r[1]) / tot
        return rows

    out["by_gt_height"] = strat(S["gt_bins"], [f"[{GT_EDGES[i]:g},{GT_EDGES[i + 1]:g})" for i in range(len(GT_EDGES) - 1)]
                                + [f"[{GT_EDGES[-1]:g},inf)"], False)
    out["by_baseline_error"] = strat(S["err_bins"], _bin_labels(ERR_EDGES), True)
    return out


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------

def load(kind, run, ckpt, device, is_main):
    cfg = OmegaConf.load(os.path.join(run, "config.yaml"))
    d = cfg.dataset
    in_stats = edm._load_target_stats(d.input_stats_file, d.year)
    chm = load_chmv2_baseline(
        model_dir=cfg.model.chmv2_model_id, ckpt_path=cfg.model.chmv2_ckpt_path,
        mean=in_stats["mean"], std=in_stats["std"],
        out_in_scale_factor=cfg.model.chmv2_out_in_scale_factor,
    ).to(device).eval()
    n_ls, n_ps = len(d.selected_bands), len(d.selected_bands_hr)
    t = cfg.trainer
    sr_module = None
    if kind == "sr":
        fm = sr_fm_mod.build_fm_refiner(
            sd_pretrained_path=cfg.model.sd_pretrained_path, n_landsat_bands=n_ls, n_ps_bands=n_ps,
            ps_dropout_p=0.0, bridge_sigma=t.get("bridge_sigma", 0.0), noise_sigma=t.get("noise_sigma", 0.0),
            sample_sigma=t.get("sample_sigma", None), sample_noise_mode=t.get("sample_noise_mode", "sde"),
            concat_z_coarse=t.get("concat_z_coarse", False), refine_threshold=t.get("refine_threshold", 0.0),
            concat_landsat_hr=t.get("concat_landsat_hr", False), device=str(device),
        ).to(device)
        sr_module = build_sr_module(OmegaConf.to_container(cfg.sr_module, resolve=True)).to(device)
    else:
        fm = plain_fm_mod.build_fm_refiner(
            sd_pretrained_path=cfg.model.sd_pretrained_path, n_landsat_bands=n_ls, n_ps_bands=n_ps,
            ps_dropout_p=0.0, use_controlnet=t.get("use_controlnet", True),
            controlnet_cond_mode=t.get("controlnet_cond_mode", "landsat_ps"),
            velocity_parameterization=t.get("velocity_parameterization", "fixed"),
            noise_sigma=t.get("noise_sigma", 0.0), sample_sigma=t.get("sample_sigma", None),
            sample_noise_mode=t.get("sample_noise_mode", "sde"), device=str(device),
        ).to(device)
    ck = edm._torch_load(ckpt)
    fm.unet.load_state_dict(ck["unet_state"])
    if fm.controlnet is not None and ck.get("controlnet_state") is not None:
        fm.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm._null_ps = ck["null_ps_state"]
    if sr_module is not None:
        sr_module.load_state_dict(ck["sr_state"])
        sr_module.eval()
    if is_main:
        logging.info(f"loaded {kind} refiner from {ckpt} (step={ck.get('step')}; "
                     f"noise_sigma={fm.noise_sigma} sample_sigma={fm.sample_sigma})")
    del ck
    fm.eval()
    return cfg, chm, fm, sr_module, n_ls


def fmt_md(res, tag):
    L = [f"### {tag}", ""]
    b = res["baseline"]
    L += [f"pixels={res['N_pixels']:.3e}  K={res['K']}", "",
          "| prediction | R2 | MAE | RMSE |", "|---|---|---|---|",
          f"| baseline_chmv2 | {b['r2']:.4f} | {b['mae']:.3f} | {b['rmse']:.3f} |"]
    for n, m in res["mean_of_n"].items():
        L.append(f"| refiner, mean of {n} draw(s) | {m['r2']:.4f} | {m['mae']:.3f} | {m['rmse']:.3f} |")
    if res["K"] > 1:
        L.append(f"| refiner, n->inf (extrapolated MSE_K - V/K) | {res['r2_inf_extrapolated']:.4f} | - | "
                 f"{res['rmse_inf_extrapolated']:.3f} |")
        L += ["", f"across-draw variance V = {res['across_draw_var_V']:.3f} m^2  "
                  f"(single draw MSE implied by fit: {res['single_draw_mse_model']:.3f}; "
                  f"measured n=1: {res['mean_of_n']['1']['mse']:.3f})"]
    L += ["", "By GT height (m): MSE base / single / mean-of-K, bias base, across-draw var, share of total baseline SSE "
              "added by refiner (single / meanK; negative = refiner helps)", "",
          "| bin | pix frac | MSE base | MSE single | MSE meanK | bias base | var | dSSE share single | dSSE share meanK |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lab, r in res["by_gt_height"].items():
        L.append(f"| {lab} | {r['pixel_frac']:.3f} | {r['mse_base']:.2f} | {r['mse_single']:.2f} | {r['mse_meanK']:.2f} | "
                 f"{r['bias_base']:+.2f} | {r['across_draw_var']:.2f} | {r['dSSE_single_share']:+.3f} | "
                 f"{r['dSSE_meanK_share']:+.3f} |")
    L += ["", "By signed baseline error (baseline - gt, m): fraction of that error the refiner (mean of K) removes", "",
          "| bin | pix frac | MSE base | MSE single | MSE meanK | mean base err | mean shift | frac corrected | "
          "dSSE share single | dSSE share meanK |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in res["by_baseline_error"].items():
        L.append(f"| {lab} | {r['pixel_frac']:.3f} | {r['mse_base']:.2f} | {r['mse_single']:.2f} | {r['mse_meanK']:.2f} | "
                 f"{r['bias_base']:+.2f} | {r['mean_shift_meanK_minus_base']:+.2f} | "
                 f"{r['frac_base_error_corrected']:.2f} | {r['dSSE_single_share']:+.3f} | {r['dSSE_meanK_share']:+.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sr", "plain"], default="sr",
                    help="sr: depthfm/fm_refiner.py + SwinIR pseudo-PS (v12 family); "
                         "plain: depthfm/fm_refiner_finer.py Landsat-only ControlNet (R-lumi-5)")
    ap.add_argument("--run", default=f"{OUT_ROOT}/sr_fm_refiner_v12-lumi")
    ap.add_argument("--ckpt", default=None, help="default: <run>/checkpoint/latest.pth")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default=None, help="default: <OUT_ROOT>/_diag_refiner_gain/<tag>")
    ap.add_argument("--n_steps", default="10", help="comma list, e.g. 1,4,10")
    ap.add_argument("--K", type=int, default=20, help="independent draws per patch (use 1 for deterministic)")
    ap.add_argument("--n_list", default="1,2,3,5,10,20", help="n for the mean-of-n curve (must be <= K)")
    ap.add_argument("--draw_chunk", type=int, default=5, help="draws processed in one batch")
    ap.add_argument("--stride", type=int, default=2, help="evaluate every stride-th val patch")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N selected patches")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    n_steps_list = [int(x) for x in args.n_steps.split(",")]
    n_list = sorted({n for n in (int(x) for x in args.n_list.split(",")) if n <= args.K} | {1, args.K})
    tag = args.tag or os.path.basename(args.run.rstrip("/"))
    out_dir = args.out_dir or f"{OUT_ROOT}/_diag_refiner_gain/{tag}"
    ckpt = args.ckpt or os.path.join(args.run, "checkpoint", "latest.pth")

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_main, dist_on = rank == 0, world > 1
    if not is_main:
        sys.stdout = open(os.devnull, "w")
        import functools
        import src.util.ps_lazydataset as _pld
        _pld.tqdm = functools.partial(_pld.tqdm, disable=True)
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    if dist_on:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=world,
                                timeout=timedelta(minutes=30))
    if use_cuda:
        idx = local % max(1, torch.cuda.device_count())
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg, chm, fm, sr_module, n_ls = load(args.kind, args.run, ckpt, device, is_main)
    cfg_d = cfg.dataset
    stats = edm._load_target_stats(cfg_d.target_stats_file, cfg_d.year)
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    val_ds = edm._make_val_dataset(cfg_d)
    sel = list(range(0, len(val_ds), args.stride))
    if args.limit:
        sel = sel[: args.limit]
    local_idx = sel[rank::world]
    if is_main:
        logging.info(f"val patches: {len(val_ds)}, evaluating {len(sel)} (stride {args.stride}); K={args.K} "
                     f"n_steps={n_steps_list} n_list={n_list}")
    loader = DataLoader(Subset(val_ds, local_idx), batch_size=1, shuffle=False,
                        num_workers=cfg_d.workers, pin_memory=True)
    accs = {s: Acc(device, n_list) for s in n_steps_list}

    for sample_i, batch in zip(local_idx, tqdm(loader, disable=not is_main, desc="diag")):
        inputs_lr, _inputs_hr, targets = batch
        inputs_lr = inputs_lr.squeeze(0) if inputs_lr.dim() == 5 else inputs_lr
        targets = targets.squeeze(0) if targets.dim() == 5 else targets
        inputs_lr, targets = inputs_lr.to(device), targets.to(device)
        H, W = targets.shape[-2:]
        landsat = inputs_lr[:, :n_ls]
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)
        with torch.no_grad():
            coarse_m = chm((landsat + 1.0) / 2.0, target_size=(H, W))
            coarse_n = edm._normalize_coarse(coarse_m, stats)
            gt_m = edm._denormalize(targets, stats)
            base = coarse_m.clamp(p1, p99)
            pseudo_ps = sr_module(landsat, target_size=(H, W)) if args.kind == "sr" else None

            for s in n_steps_list:
                torch.manual_seed(sample_i * 1000 + s)
                draws = []
                for c0 in range(0, args.K, args.draw_chunk):
                    c = min(args.draw_chunk, args.K - c0)
                    cn = coarse_n.expand(c, -1, -1, -1)
                    lh = landsat_hr.expand(c, -1, -1, -1)
                    if args.kind == "sr":
                        h = fm.refine_with_ps(pseudo_ps.expand(c, -1, -1, -1), cn, n_steps=s, method="euler",
                                              landsat_hr=lh)
                    else:
                        h = fm.refine(lh, cn, n_steps=s, method="euler")
                    draws.append(edm._denormalize(h, stats).clamp(p1, p99))
                accs[s].update(base.float(), torch.cat(draws, 0).float(), gt_m.float())

    states = {s: accs[s].state() for s in n_steps_list}
    gathered = [None] * world if dist_on else [states]
    if dist_on:
        dist.all_gather_object(gathered, states)
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        results, md = {}, []
        for s in n_steps_list:
            res = finalize([g[s] for g in gathered], n_list, args.K)
            results[f"n_steps={s}"] = res
            md.append(fmt_md(res, f"{tag}  ({args.kind}, n_steps={s}, ckpt={os.path.basename(ckpt)}, "
                                  f"{len(sel)} patches, stride {args.stride})"))
        with open(os.path.join(out_dir, "diag.json"), "w") as f:
            json.dump({"meta": {"run": args.run, "ckpt": ckpt, "kind": args.kind, "K": args.K, "stride": args.stride,
                                "n_patches": len(sel), "range_m": [p1, p99]}, "results": results}, f, indent=2)
        with open(os.path.join(out_dir, "diag.md"), "w") as f:
            f.write("\n".join(md))
        logging.info("\n" + "\n".join(md))
    if dist_on:
        dist.barrier()
        dist.destroy_process_group()
