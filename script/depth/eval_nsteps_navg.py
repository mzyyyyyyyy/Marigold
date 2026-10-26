"""
Sensitivity of an SR + FM refiner (--model v12-nd: sr_fm_refiner_v12-lumi-nd,
--model v11: sr_fm_refiner_v11-lumi) to the two inference parameters
n_steps (Euler steps) and n_avg (number of independent noise draws averaged),
on the TEST split (split_spatial_v1_test.pkl), accuracy and detail metrics.

--model v11: v11 was trained on the old random split with the frozen DAv2 coarse
model; data / GT handling are those of eval_group_meeting.py (new dataset given by
--data_cfg, raw-metre GT clipped to [p1, p99]), so the (n_steps=10, n_avg=5) cell
reproduces its sr_fm_refiner_v11-lumi column. The other models of that evaluation
(--ref_json) are drawn as reference lines / points, and plots/tradeoff.png shows
accuracy (RMSE) against detail (edge F1, grad. energy ratio) for every cell.
Image names are the same <id>.png as in eval_group_meeting.py.

One pass per n_steps: for every patch, n_avg_max independent draws are
generated (seeded patch_idx*1000 + k, the same rule as eval_detail_metrics_nd.py,
so the (n_steps=10, n_avg=5) cell reproduces that evaluation and the same noise
is used across n_steps), and the n_avg = k result is the mean of the first k
draws. A "coarse" reference row (the frozen coarse model: CHMv2 for v12, DAv2
for v11; no refinement) is added. Both use sample_noise_mode=init_only: n_avg only averages different
initial noises, n_steps only changes the ODE discretisation.

Metrics per (n_steps, n_avg): those of eval_detail_metrics_nd.py (R2/MAE/RMSE,
gradient error / energy ratio, edge F1 at several thresholds, HF ratio /
correlation, height-binned error). Predictions and GT are clipped to the
dataset's [p1, p99] metres range, as in that script.

Outputs under --out_dir:
  results.json, results.md      all metrics per cell / the main metrics as steps x avg tables
  plots/metrics_vs_steps.png    main metrics vs n_steps, one line per n_avg (dashed = coarse)
  plots/metrics_vs_avg.png      main metrics vs n_avg, one line per n_steps
  plots/heatmaps.png            main metrics as n_steps x n_avg heatmaps
  plots/tradeoff.png            (with --ref_json) RMSE vs edge F1 / grad. energy ratio per cell
  images/<id>.png               for the selected patches (--vis_all: every patch): reference row
                                (Landsat, real PS, pseudo-PS, coarse, GT) + the n_steps x n_avg
                                grid of predictions;
                                <id> = <test-list index>_<tile>_y<row>_x<col>

Launch (one task per GPU): see script/depth/eval_nsteps_navg.sh
"""

import argparse
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import eval_detail_metrics as old
import eval_detail_metrics_nd as nd
from depthfm import fm_refiner as sr_fm_mod
from depthfm.chmv2 import load_chmv2_baseline
from depthfm.sr_module import build_sr_module

OUT_ROOT = "/flash/project_465002934/Marigold_output"

# (key, label, higher_is_better or None) -- the "main" metrics plotted / tabulated
MAIN = [
    ("r2", "R2", True), ("mae", "MAE (m)", False), ("rmse", "RMSE (m)", False),
    ("grad_mag_error", "Grad. magnitude error (m/px)", False),
    ("grad_energy_ratio", "Grad. energy ratio (->1)", None),
    ("edge_f1", "Edge F1 (thr 2 m/px, tol 1 px)", True),
    ("hf_energy_ratio", "HF energy ratio (->1)", None),
    ("hf_corr", "HF correlation", True),
]


def cell_key(s, k):
    return f"s{s}_k{k}"


RUN_NAMES = {"v12-nd": "sr_fm_refiner_v12-lumi-nd", "v11": "sr_fm_refiner_v11-lumi"}


def _load_sr_fm(cfg_sr, run, args, device, is_main):
    """FM refiner + SR module of an sr_fm_refiner run, from <run>/checkpoint/<ckpt>.pth."""
    cfg_d = cfg_sr.dataset
    n_ls, n_ps = len(cfg_d.selected_bands), len(cfg_d.selected_bands_hr)
    t = cfg_sr.trainer
    fm = sr_fm_mod.build_fm_refiner(
        sd_pretrained_path=cfg_sr.model.sd_pretrained_path,
        n_landsat_bands=n_ls, n_ps_bands=n_ps, ps_dropout_p=0.0,
        bridge_sigma=t.get("bridge_sigma", 0.0), noise_sigma=t.get("noise_sigma", 0.0),
        sample_sigma=t.get("sample_sigma", None), sample_noise_mode=t.get("sample_noise_mode", "sde"),
        concat_z_coarse=t.get("concat_z_coarse", False), refine_threshold=t.get("refine_threshold", 0.0),
        concat_landsat_hr=t.get("concat_landsat_hr", False), device=str(device),
    ).to(device)
    sr = build_sr_module(OmegaConf.to_container(cfg_sr.sr_module, resolve=True)).to(device)
    ck = old._torch_load(os.path.join(run, "checkpoint", f"{args.ckpt}.pth"))
    fm.unet.load_state_dict(ck["unet_state"])
    fm.controlnet.load_state_dict(ck["controlnet_state"])
    if ck.get("null_ps_state") is not None:
        fm._null_ps = ck["null_ps_state"]
    sr.load_state_dict(ck["sr_state"])
    if is_main:
        logging.info(f"loaded {os.path.basename(run)} {args.ckpt}.pth (step={ck.get('step')}); "
                     f"noise mode {t.get('sample_noise_mode')}, sample_sigma {t.get('sample_sigma')}")
    del ck
    fm.eval()
    sr.eval()
    return fm, sr, n_ls


def load_model(args, device, is_main):
    """Returns (cfg_data, coarse_fn, fm, sr, n_ls, target_stats_file, gt_fn):
    coarse_fn(landsat [-1,1], (H, W)) -> coarse height in metres;
    gt_fn(targets, target_stats, p1, p99) -> GT in metres, clipped to [p1, p99]."""
    run = os.path.join(args.run_root, RUN_NAMES[args.model])
    cfg_sr = OmegaConf.load(os.path.join(run, "config.yaml"))
    fm, sr, n_ls = _load_sr_fm(cfg_sr, run, args, device, is_main)

    if args.model == "v12-nd":
        # frozen coarse CHMv2 it was trained with; its own (new) dataset, minmax-scaled targets
        cfg_d = cfg_sr.dataset
        in_stats = old._load_target_stats(cfg_d.input_stats_file, cfg_d.year)
        chm = load_chmv2_baseline(
            model_dir=cfg_sr.model.chmv2_model_id, ckpt_path=cfg_sr.model.chmv2_ckpt_path,
            mean=in_stats["mean"], std=in_stats["std"],
            out_in_scale_factor=cfg_sr.model.chmv2_out_in_scale_factor,
        ).to(device).eval()
        coarse_fn = lambda landsat, size: chm((landsat + 1.0) / 2.0, target_size=size)
        gt_fn = lambda targets, stats, p1, p99: old._denormalize(targets, stats)
        return cfg_d, coarse_fn, fm, sr, n_ls, cfg_d.target_stats_file, gt_fn

    # v11: frozen coarse DAv2; new dataset from --data_cfg (raw-metre targets),
    # exactly as in eval_group_meeting.py
    m = cfg_sr.model
    dav2 = sr_fm_mod.load_dav2(dav2_path=m.dav2_pretrained_path, backbone=m.dav2_backbone,
                               out_in_scale_factor=m.dav2_out_in_scale_factor).to(device).eval()
    coarse_fn = lambda landsat, size: sr_fm_mod.run_dav2(dav2, (landsat + 1.0) / 2.0, target_size=size)
    cfg_d = OmegaConf.load(args.data_cfg).dataset
    for k in ("input_stats_file", "input_hr_stats_file", "selected_bands", "selected_bands_hr",
              "use_input_minmax", "use_input_norm", "use_input_hr_minmax", "use_input_hr_norm",
              "scale_input_to_neg1_1", "scale_input_hr_to_neg1_1", "year"):
        assert cfg_d[k] == cfg_sr.dataset[k], f"input preprocessing differs from training: {k}"
    assert not cfg_d.use_target_minmax and not cfg_d.use_target_norm, "expect raw-metre targets"
    gt_fn = lambda targets, stats, p1, p99: targets.float().clamp(p1, p99)
    return cfg_d, coarse_fn, fm, sr, n_ls, cfg_sr.dataset.target_stats_file, gt_fn


def select_vis_ids(coords, n_per_group, seed):
    """n_per_group random test patches per height group (GT p75 range from
    make_spatial_split.py), deterministic."""
    rng = np.random.RandomState(seed)
    ids = []
    groups = sorted({c.get("height_group", 0) for c in coords})
    for g in groups:
        pool = [i for i, c in enumerate(coords) if c.get("height_group", 0) == g]
        if pool:
            ids += sorted(rng.choice(pool, size=min(n_per_group, len(pool)), replace=False).tolist())
    return ids


def save_grid_image(path, ref, preds, gt, steps, avgs, pid, title_extra, vmax):
    """ref: list of (title, rgb-or-height array, is_height); preds[(s,k)] -> (H,W) metres."""
    n_rows, n_cols = 1 + len(steps), max(len(avgs), len(ref))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows + 0.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (ttl, arr, is_h) in zip(axes[0], ref):
        if is_h:
            ax.imshow(arr, cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
        else:
            ax.imshow(arr, interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
    for i, s in enumerate(steps):
        for j, k in enumerate(avgs):
            ax = axes[1 + i, j]
            p = preds[(s, k)]
            ax.imshow(p, cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"n_steps={s}, n_avg={k}\nMAE {np.abs(p - gt).mean():.2f} m", fontsize=8)
    fig.suptitle(f"{pid}   {title_extra}", fontsize=10)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=70)
    plt.close(fig)


REF_STYLE = {"baseline_dav2": ("tab:red", "s"), "baseline_sr4ir_reg": ("tab:green", "D"),
             "fm_refiner-R-lumi-4": ("tab:orange", "^")}


def plot_results(out_dir, results, coarse, steps, avgs, model_name, refs=None, edge_tol=1):
    """refs: {model name: metrics dict of eval_group_meeting.py} drawn as reference lines / points."""
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    refs = refs or {}

    def val(s, k, key):
        r = results[cell_key(s, k)]
        return r["edge_f1"] if key == "edge_f1" else r[key]

    def cval(key):
        return coarse["edge_f1"] if key == "edge_f1" else coarse[key]

    def rval(name, key):
        return refs[name][f"edge_f1_tol{edge_tol}" if key == "edge_f1" else key]

    def draw_refs(ax, key):
        for name in refs:
            color, _ = REF_STYLE.get(name, ("k", "o"))
            ax.axhline(rval(name, key), color=color, ls="-.", lw=1, label=name)

    for fname, xs, xlabel, lines, line_label, lookup in (
        ("metrics_vs_steps.png", steps, "n_steps", avgs, "n_avg", lambda x, l, key: val(x, l, key)),
        ("metrics_vs_avg.png", avgs, "n_avg", steps, "n_steps", lambda x, l, key: val(l, x, key)),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        cmap = plt.get_cmap("viridis")
        for ax, (key, label, _) in zip(axes.flat, MAIN):
            for li, l in enumerate(lines):
                ax.plot(xs, [lookup(x, l, key) for x in xs], marker="o", ms=3,
                        color=cmap(li / max(len(lines) - 1, 1)), label=f"{line_label}={l}")
            ax.axhline(cval(key), color="gray", ls="--", lw=1, label="coarse (no refine)")
            draw_refs(ax, key)
            if key in ("grad_energy_ratio", "hf_energy_ratio"):
                ax.axhline(1.0, color="k", ls=":", lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            ax.set_xlabel(xlabel)
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{model_name}, test set: metrics vs {xlabel}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plots", fname), dpi=110)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (key, label, _) in zip(axes.flat, MAIN):
        m = np.array([[val(s, k, key) for k in avgs] for s in steps])
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(avgs)))
        ax.set_xticklabels(avgs)
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels(steps)
        ax.set_xlabel("n_avg")
        ax.set_ylabel("n_steps")
        ax.set_title(label, fontsize=10)
        for i in range(len(steps)):
            for j in range(len(avgs)):
                ax.text(j, i, f"{m[i, j]:.3g}", ha="center", va="center", fontsize=7, color="w")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{model_name}, test set: n_steps x n_avg", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plots", "heatmaps.png"), dpi=110)
    plt.close(fig)

    if not refs:
        return
    # accuracy vs detail: one point per (n_steps, n_avg) cell, joined along n_avg
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    cmap = plt.get_cmap("viridis")
    for ax, (ykey, ylabel) in zip(axes, (("edge_f1", f"Edge F1 (thr 2 m/px, tol {edge_tol} px)  higher = more detail"),
                                         ("grad_energy_ratio", "Grad. energy ratio (->1 = GT sharpness)"))):
        for i, s in enumerate(steps):
            color = cmap(i / max(len(steps) - 1, 1))
            xs = [val(s, k, "rmse") for k in avgs]
            ys = [val(s, k, ykey) for k in avgs]
            ax.plot(xs, ys, "-o", ms=4, color=color, label=f"n_steps={s} (n_avg {avgs[0]}->{avgs[-1]})")
            ax.annotate(f"k={avgs[0]}", (xs[0], ys[0]), fontsize=6, color=color,
                        xytext=(3, 3), textcoords="offset points")
            ax.annotate(f"k={avgs[-1]}", (xs[-1], ys[-1]), fontsize=6, color=color,
                        xytext=(3, -8), textcoords="offset points")
        ax.scatter([cval("rmse")], [cval(ykey)], marker="X", s=90, color="gray", zorder=5,
                   label="coarse DAv2 (no refine)")
        for name in refs:
            color, marker = REF_STYLE.get(name, ("k", "o"))
            ax.scatter([rval(name, "rmse")], [rval(name, ykey)], marker=marker, s=80, color=color,
                       edgecolor="k", zorder=5, label=name)
        ax.set_xlabel("RMSE (m)  lower = more accurate")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{model_name}, test set: accuracy vs detail per (n_steps, n_avg)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plots", "tradeoff.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(RUN_NAMES), default="v12-nd")
    ap.add_argument("--run_root", default=OUT_ROOT)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--data_cfg", default=f"{OUT_ROOT}/baseline_swt-lumi-nd/config.yaml",
                    help="v11 only: run config whose dataset block defines the new dataset / test split")
    ap.add_argument("--ref_json", default=None,
                    help="eval_group_meeting.py's detail_metrics.json: its other models are drawn as references "
                         "(default for v11: /flash/project_465002934/result_group_meeting/detail_metrics.json)")
    ap.add_argument("--out_dir", default=None, help="default: <run_root>/_nsteps_navg_v12_nd or _nsteps_navg_v11")
    ap.add_argument("--n_steps_grid", type=int, nargs="+", default=[1, 2, 4, 6, 10, 20])
    ap.add_argument("--n_avg_grid", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--edge_thr", type=float, default=2.0)
    ap.add_argument("--edge_thrs", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--edge_tol", type=int, default=1)
    ap.add_argument("--hf_sigma", type=float, default=2.0)
    ap.add_argument("--vis_ids", type=int, nargs="*", default=None,
                    help="test-list indices to draw (default: --vis_per_group random patches per GT height group)")
    ap.add_argument("--vis_per_group", type=int, default=2)
    ap.add_argument("--vis_seed", type=int, default=0)
    ap.add_argument("--vis_all", action="store_true", help="draw every test patch")
    ap.add_argument("--no_vis", action="store_true")
    ap.add_argument("--save_npz", action="store_true",
                    help="also save GT and every (n_steps, n_avg) prediction of drawn patches to images/<pid>.npz")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N test samples")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()
    model_name = RUN_NAMES[args.model]
    if args.out_dir is None:
        args.out_dir = os.path.join(args.run_root, "_nsteps_navg_" + args.model.replace("-", "_"))
    if args.ref_json is None and args.model == "v11":
        args.ref_json = "/flash/project_465002934/result_group_meeting/detail_metrics.json"
    steps, avgs = sorted(args.n_steps_grid), sorted(args.n_avg_grid)
    k_max = max(avgs)

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg_d, coarse_fn, fm, sr, n_ls, target_stats_file, gt_fn = load_model(args, device, is_main)
    target_stats = old._load_target_stats(target_stats_file, cfg_d.year)
    p1, p99 = float(target_stats["p1"][0]), float(target_stats["p99"][0])
    test_ds, test_coords = nd.make_test_dataset(cfg_d)
    n_total = len(test_ds) if not args.limit else min(args.limit, len(test_ds))
    vis_ids = set()
    if not args.no_vis:
        if args.vis_all:
            vis_ids = set(range(n_total))
        elif args.vis_ids is not None:
            vis_ids = set(args.vis_ids)
        else:
            vis_ids = set(select_vis_ids(test_coords[:n_total], args.vis_per_group, args.vis_seed))
        vis_ids = {i for i in vis_ids if i < n_total}
    if is_main:
        logging.info(f"{model_name}; test samples: {n_total}; world {world}; n_steps {steps}; n_avg {avgs}; "
                     f"{len(vis_ids)} patches drawn" + ("" if args.vis_all else f": {sorted(vis_ids)}"))
        os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    local_idx = list(range(n_total))[rank::world]
    loader = DataLoader(Subset(test_ds, local_idx), batch_size=1, shuffle=False,
                        num_workers=cfg_d.workers, pin_memory=True)

    tols = (args.edge_tol, 2 * args.edge_tol)
    def new_acc():
        return (old.DetailAccumulator(device, args.edge_thr, args.edge_tol, args.hf_sigma),
                nd.EdgeAccumulator(device, args.edge_thrs, tols))
    acc = {cell_key(s, k): new_acc() for s in steps for k in avgs}
    acc["coarse"] = new_acc()

    for sample_i, batch in zip(local_idx, tqdm(loader, disable=not is_main, desc="nsteps-navg")):
        inputs_lr, inputs_hr, targets = batch
        inputs_lr = inputs_lr.squeeze(0) if inputs_lr.dim() == 5 else inputs_lr
        inputs_hr = inputs_hr.squeeze(0) if inputs_hr.dim() == 5 else inputs_hr
        targets = targets.squeeze(0) if targets.dim() == 5 else targets
        inputs_lr, inputs_hr, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
        H, W = targets.shape[-2:]
        landsat = inputs_lr[:, :n_ls]
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)
        want_vis = sample_i in vis_ids
        preds_vis = {}

        with torch.no_grad():
            gt_m = gt_fn(targets, target_stats, p1, p99)
            coarse_m = coarse_fn(landsat, (H, W))
            coarse_n = old._normalize_coarse(coarse_m, target_stats)
            coarse_c = coarse_m.clamp(p1, p99)
            acc["coarse"][0].update(coarse_c.float(), gt_m.float())
            acc["coarse"][1].update(coarse_c.float(), gt_m.float())
            pseudo_ps = sr(landsat, target_size=(H, W))

            for s in steps:
                draws = []
                for kk in range(k_max):
                    torch.manual_seed(sample_i * 1000 + kk)
                    draws.append(fm.refine_with_ps(pseudo_ps, coarse_n, n_steps=s, method="euler",
                                                   landsat_hr=landsat_hr))
                draws = torch.stack(draws)
                for k in avgs:
                    pred = old._denormalize(draws[:k].mean(0), target_stats).clamp(p1, p99)
                    a, e = acc[cell_key(s, k)]
                    a.update(pred.float(), gt_m.float())
                    e.update(pred.float(), gt_m.float())
                    if want_vis:
                        preds_vis[(s, k)] = pred[0, 0].cpu().numpy()

        if want_vis:
            c = test_coords[sample_i]
            pid = f"{sample_i:05d}_{c['tile_id'].replace(',', '-')}_y{c['output_crop_y']}_x{c['output_crop_x']}"
            gt_np = gt_m[0, 0].cpu().numpy()
            ref = [("Landsat (bilinear)", nd._rgb_u8(landsat_hr[0]), False),
                   ("Real PlanetScope", nd._rgb_u8(inputs_hr[0]), False),
                   (f"{args.model} pseudo-PS (SR)", nd._rgb_u8(pseudo_ps[0]), False),
                   (f"coarse (no refine)\nMAE {np.abs(coarse_c[0, 0].cpu().numpy() - gt_np).mean():.2f} m",
                    coarse_c[0, 0].cpu().numpy(), True),
                   ("GT height (m)", gt_np, True)]
            save_grid_image(os.path.join(args.out_dir, "images", pid + ".png"), ref, preds_vis, gt_np,
                            steps, avgs, pid, f"(GT p75 = {c.get('p75', float('nan')):.1f} m)", p99)
            if args.save_npz:
                np.savez_compressed(os.path.join(args.out_dir, "images", pid + ".npz"), gt=gt_np,
                                    **{f"s{s}_k{k}": p for (s, k), p in preds_vis.items()})

    states = {k: (v[0].state(), v[1].state()) for k, v in acc.items()}
    if dist_on:
        gathered = [None] * world
        dist.all_gather_object(gathered, states)
    else:
        gathered = [states]

    if is_main:
        def fin(key):
            r = old.finalize([g[key][0] for g in gathered])
            ed = nd.EdgeAccumulator.finalize([g[key][1] for g in gathered], args.edge_thrs, tols)
            r["edge_multi_thr"] = ed
            r["edge_f1"] = r[f"edge_f1_tol{args.edge_tol}"]  # at --edge_thr
            return r

        results = {cell_key(s, k): fin(cell_key(s, k)) for s in steps for k in avgs}
        coarse = fin("coarse")
        os.makedirs(args.out_dir, exist_ok=True)
        refs = {}
        if args.ref_json:
            with open(args.ref_json) as f:
                refs = {n: r for n, r in json.load(f)["results"].items() if n != model_name}
        meta = {"model": model_name, "ckpt": f"{args.ckpt}.pth", "split": f"test ({cfg_d.test_coord_path})",
                "ref_json": args.ref_json,
                "n_test_samples": n_total, "n_steps_grid": steps, "n_avg_grid": avgs,
                "seed_rule": "torch.manual_seed(patch_idx*1000 + draw_idx) before each draw",
                "edge_thr_m_per_px": args.edge_thr, "edge_thrs_m_per_px": args.edge_thrs,
                "edge_tol_px": args.edge_tol, "hf_sigma_px": args.hf_sigma, "range_m": [p1, p99],
                "vis_ids": sorted(vis_ids)}
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump({"meta": meta, "coarse": coarse, "cells": results}, f, indent=2)

        lines = []
        for key, label, _ in MAIN:
            lines += [f"### {label}", "", "| n_steps \\ n_avg | " + " | ".join(str(k) for k in avgs) + " |",
                      "|---|" + "---|" * len(avgs)]
            for s in steps:
                lines.append(f"| {s} | " + " | ".join(
                    f"{results[cell_key(s, k)]['edge_f1' if key == 'edge_f1' else key]:.4f}" for k in avgs) + " |")
            lines += [f"| coarse (no refine) | {coarse['edge_f1' if key == 'edge_f1' else key]:.4f} |"]
            for n, r in refs.items():
                lines.append(f"| {n} | {r[f'edge_f1_tol{args.edge_tol}' if key == 'edge_f1' else key]:.4f} |")
            lines.append("")
        with open(os.path.join(args.out_dir, "results.md"), "w") as f:
            f.write("\n".join(lines))
        plot_results(args.out_dir, results, coarse, steps, avgs, model_name, refs, args.edge_tol)
        print("\n".join(lines))

    if dist_on:
        dist.barrier()
        dist.destroy_process_group()
