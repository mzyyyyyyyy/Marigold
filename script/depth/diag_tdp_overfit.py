"""
Can the SR module drive the TDP loss down at all? Overfitting test against a
frozen trained FM (default: sr_fm_refiner_v13-lumi-nd best.pth).

The SR module (init: the run's trained SR) is optimised on a tiny FIXED train
subset (--n_train 240px train patches, no augmentation) for --iters steps with
the training-time TDP loss (hook / noisy z_t of _sr_step, t ~ U(t_min, t_max)).
Every --eval_every steps l_tdp (fixed t/eps set) and l_pix are measured on that
train subset and on --n_val fixed val patches; at the start and the end the
refinement metrics (pseudo-PS input) are measured on both subsets.

Reading the result:
  train l_tdp does not go down          -> optimisation problem (TDP gradient
                                           through the frozen FM is too noisy /
                                           ill-conditioned), not capacity/info
  train l_tdp goes down, val does not   -> SR can memorise but not generalise:
                                           information limit, a bigger SR won't help
  both go down (slowly)                 -> capacity may matter

Each GPU (rank) runs one setting of GRID independently; rank0 gathers and
writes <out_dir>/overfit.md + overfit.json. Launch: script/depth/diag_tdp_overfit.sh
"""

import argparse
import copy
import json
import logging
import os
import pickle
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import eval_detail_metrics as edm
from diag_refiner_gain import load as load_run
from eval_detail_metrics_nd import make_test_dataset

OUT_ROOT = edm.OUT_ROOT
T_EVAL = [0.0, 0.2, 0.4, 0.6]

# One setting per rank. w_tdp / w_pix: loss weights; n_t: t-samples averaged per
# step (variance reduction of the TDP gradient).
GRID = [
    {"name": "tdp_lr1e-4",       "w_tdp": 1.0, "w_pix": 0.0, "lr": 1e-4, "n_t": 1},
    {"name": "tdp_lr3e-4",       "w_tdp": 1.0, "w_pix": 0.0, "lr": 3e-4, "n_t": 1},
    {"name": "tdp_lr1e-3",       "w_tdp": 1.0, "w_pix": 0.0, "lr": 1e-3, "n_t": 1},
    {"name": "tdp_lr3e-3",       "w_tdp": 1.0, "w_pix": 0.0, "lr": 3e-3, "n_t": 1},
    {"name": "tdp_lr1e-3_nt4",   "w_tdp": 1.0, "w_pix": 0.0, "lr": 1e-3, "n_t": 4},
    {"name": "tdp5pix1_lr3e-4",  "w_tdp": 5.0, "w_pix": 1.0, "lr": 3e-4, "n_t": 1},
    {"name": "tdp5pix1_lr1e-3",  "w_tdp": 5.0, "w_pix": 1.0, "lr": 1e-3, "n_t": 1},
    {"name": "pix_lr1e-3",       "w_tdp": 0.0, "w_pix": 1.0, "lr": 1e-3, "n_t": 1},
]


def subset(ds, coords):
    d = copy.copy(ds)
    d.patch_coords = coords
    return d


def pick(coords, n, size=None):
    c = [p for p in coords if size is None or p["output_patch_height"] == size]
    idx = np.linspace(0, len(c) - 1, n).round().astype(int)
    return [c[i] for i in sorted(set(idx.tolist()))]


@torch.no_grad()
def prepare(ds, chm, fm, stats, n_ls, device):
    """Load every patch once and precompute everything the loops need."""
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    out = []
    for i in range(len(ds)):
        inputs_lr, inputs_hr, targets = [x.unsqueeze(0) if x.dim() == 3 else x for x in ds[i]]
        inputs_lr, real_ps, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
        H, W = targets.shape[-2:]
        landsat = inputs_lr[:, :n_ls]
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)
        h_coarse = edm._normalize_coarse(chm((landsat + 1.0) / 2.0, target_size=(H, W)), stats)
        out.append({"landsat": landsat, "landsat_hr": landsat_hr, "real_ps": real_ps, "h_coarse": h_coarse,
                    "z_c": fm.encode(h_coarse), "z_gt": fm.encode(targets),
                    "gt_m": edm._denormalize(targets, stats).clamp(p1, p99)})
    return out


def collate(items, key):
    return torch.cat([it[key] for it in items], 0)


class Feat:
    """Hook feature at cfg.sr_loss.tdp_hook for a (B,) vector of t, as in _sr_step."""

    def __init__(self, fm, hook):
        assert hook.startswith("unet.up_blocks.")
        self.fm, self.mod = fm, fm.unet.up_blocks[int(hook.split(".")[-1])]

    def z_t(self, z_c, z_gt, t, eps):
        t4 = t.view(-1, 1, 1, 1)
        z = (1 - t4) * z_c + t4 * z_gt
        return z + (1 - t4) * self.fm.noise_sigma * eps if self.fm.noise_sigma > 0 else z

    def __call__(self, cond, zt, z_c, t):
        fm = self.fm
        B = zt.shape[0]
        t_int = (t * 999).long()
        sample = torch.cat([zt, z_c], 1) if fm.concat_z_coarse else zt
        text = fm.empty_text_embed.to(zt.device).expand(B, -1, -1)
        box = {}
        h = self.mod.register_forward_hook(lambda m, i, o: box.__setitem__("f", o))
        try:
            cn = fm.controlnet(sample=sample, timestep=t_int, encoder_hidden_states=text,
                               controlnet_cond=cond, return_dict=True)
            fm.unet(sample=sample, timestep=t_int, encoder_hidden_states=text,
                    down_block_additional_residuals=cn.down_block_res_samples,
                    mid_block_additional_residual=cn.mid_block_res_sample, return_dict=True)
        finally:
            h.remove()
        return box["f"]


@torch.no_grad()
def eval_losses(items, sr, fm, feat, seed0):
    """Mean l_tdp over T_EVAL (fixed eps per patch/t) and l_pix, pseudo-PS from the current SR."""
    tdp, pix = 0.0, 0.0
    for j, it in enumerate(items):
        H, W = it["real_ps"].shape[-2:]
        ps = sr(it["landsat"], target_size=(H, W))
        pix += F.l1_loss(ps, it["real_ps"]).item()
        for k, t in enumerate(T_EVAL):
            g = torch.Generator(device=ps.device).manual_seed(seed0 + 1000 * j + k)
            eps = torch.randn(it["z_c"].shape, generator=g, device=ps.device)
            tt = torch.full((1,), t, device=ps.device)
            zt = feat.z_t(it["z_c"], it["z_gt"], tt, eps)
            f_fake = feat(fm._maybe_concat_landsat(ps, it["landsat_hr"]), zt, it["z_c"], tt)
            f_real = feat(fm._maybe_concat_landsat(it["real_ps"], it["landsat_hr"]), zt, it["z_c"], tt)
            tdp += F.l1_loss(f_fake, f_real).item() / len(T_EVAL)
    return tdp / len(items), pix / len(items)


@torch.no_grad()
def eval_refine(items, sr, fm, n_steps, n_avg, stats, use_real=False):
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    acc = edm.DetailAccumulator(items[0]["gt_m"].device, 2.0, 1, 2.0)
    for j, it in enumerate(items):
        H, W = it["real_ps"].shape[-2:]
        ps = it["real_ps"] if use_real else sr(it["landsat"], target_size=(H, W))
        torch.manual_seed(j)
        hs = [fm.refine_with_ps(ps, it["h_coarse"], n_steps=n_steps, method="euler", landsat_hr=it["landsat_hr"])
              for _ in range(n_avg)]
        acc.update(edm._denormalize(torch.stack(hs).mean(0), stats).clamp(p1, p99).float(), it["gt_m"].float())
    m = edm.finalize([acc.state()])
    return {k: m[k] for k in ("r2", "mae", "rmse", "hf_corr", "edge_f1_tol1")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=f"{OUT_ROOT}/sr_fm_refiner_v13-lumi-nd")
    ap.add_argument("--ckpt", default=None, help="default: <run>/checkpoint/best.pth")
    ap.add_argument("--out_dir", default=None, help="default: <OUT_ROOT>/_diag_tdp_overfit/<run name>")
    ap.add_argument("--n_train", type=int, default=64, help="fixed 240px train patches")
    ap.add_argument("--n_val", type=int, default=32, help="fixed 480px val patches")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--t_min", type=float, default=0.0)
    ap.add_argument("--t_max", type=float, default=0.6)
    ap.add_argument("--n_steps", type=int, default=10, help="refinement Euler steps for R2 eval")
    ap.add_argument("--n_avg", type=int, default=2, help="refinement draws for R2 eval")
    ap.add_argument("--settings", default="",
                    help="'+'-separated GRID names to run (default: all, one per rank). Not ',': "
                         "sbatch --export=ALL,EXTRA=... splits on commas")
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    is_main, dist_on = rank == 0, world > 1
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [r{rank}] %(message)s")
    if not is_main:
        sys.stdout = open(os.devnull, "w")
    if dist_on:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(hours=4))
    idx = local % max(1, torch.cuda.device_count())
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}")

    grid = [g for g in GRID if not args.settings or g["name"] in args.settings.split("+")]
    setting = grid[rank] if rank < len(grid) else None
    run_name = os.path.basename(args.run.rstrip("/"))
    out_dir = args.out_dir or f"{OUT_ROOT}/_diag_tdp_overfit/{run_name}"
    ckpt = args.ckpt or os.path.join(args.run, "checkpoint", "best.pth")

    result = None
    if setting is not None:
        cfg, chm, fm, sr, n_ls = load_run("sr", args.run, ckpt, device, is_main)
        fm.requires_grad_(False)
        sr.eval()                                   # no drop_path: deterministic, easiest to overfit
        sr.requires_grad_(True)
        feat = Feat(fm, str(cfg.sr_loss.get("tdp_hook", "unet.up_blocks.3")))
        cfg_d = cfg.dataset
        stats = edm._load_target_stats(cfg_d.target_stats_file, cfg_d.year)
        base, _ = make_test_dataset(cfg_d)          # deterministic-eval dataset, coords swapped below
        with open(cfg_d.patch_coord_path, "rb") as f:
            train_coords = pickle.load(f)
        with open(cfg_d.val_coord_path, "rb") as f:
            val_coords = pickle.load(f)
        train = prepare(subset(base, pick(train_coords, args.n_train, 240)), chm, fm, stats, n_ls, device)
        val = prepare(subset(base, pick(val_coords, args.n_val)), chm, fm, stats, n_ls, device)
        logging.info(f"{setting} | train {len(train)} x 240px, val {len(val)} x 480px")

        hist = []

        def log_eval(step, run_loss):
            tr_tdp, tr_pix = eval_losses(train, sr, fm, feat, seed0=11)
            va_tdp, va_pix = eval_losses(val, sr, fm, feat, seed0=22)
            hist.append({"step": step, "train_l_tdp": tr_tdp, "train_l_pix": tr_pix,
                         "val_l_tdp": va_tdp, "val_l_pix": va_pix, "running_loss": run_loss})
            logging.info(f"{setting['name']} step {step}: train l_tdp {tr_tdp:.4f} l_pix {tr_pix:.4f} | "
                         f"val l_tdp {va_tdp:.4f} l_pix {va_pix:.4f} | running {run_loss}")

        ref = {"train_real": eval_refine(train, sr, fm, args.n_steps, args.n_avg, stats, use_real=True),
               "val_real": eval_refine(val, sr, fm, args.n_steps, args.n_avg, stats, use_real=True),
               "train_pseudo_start": eval_refine(train, sr, fm, args.n_steps, args.n_avg, stats),
               "val_pseudo_start": eval_refine(val, sr, fm, args.n_steps, args.n_avg, stats)}
        log_eval(0, None)

        opt = torch.optim.Adam(sr.parameters(), lr=setting["lr"])
        g = torch.Generator(device=device).manual_seed(1234 + rank)
        rng = np.random.default_rng(1234 + rank)
        run_sum, run_n, t0 = 0.0, 0, time.time()
        for step in range(1, args.iters + 1):
            items = [train[i] for i in rng.choice(len(train), args.batch, replace=False)]
            landsat, real_ps = collate(items, "landsat"), collate(items, "real_ps")
            landsat_hr, z_c, z_gt = collate(items, "landsat_hr"), collate(items, "z_c"), collate(items, "z_gt")
            H, W = real_ps.shape[-2:]
            ps = sr(landsat, target_size=(H, W))
            loss = setting["w_pix"] * F.l1_loss(ps, real_ps) if setting["w_pix"] > 0 else ps.sum() * 0.0
            if setting["w_tdp"] > 0:
                for _ in range(setting["n_t"]):
                    t = args.t_min + (args.t_max - args.t_min) * torch.rand(len(items), generator=g, device=device)
                    eps = torch.randn(z_c.shape, generator=g, device=device)
                    zt = feat.z_t(z_c, z_gt, t, eps)
                    with torch.no_grad():
                        f_real = feat(fm._maybe_concat_landsat(real_ps, landsat_hr), zt, z_c, t)
                    f_fake = feat(fm._maybe_concat_landsat(ps, landsat_hr), zt, z_c, t)
                    loss = loss + setting["w_tdp"] * F.l1_loss(f_fake, f_real) / setting["n_t"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sr.parameters(), max_norm=1.0)
            opt.step()
            run_sum, run_n = run_sum + loss.item(), run_n + 1
            if step % args.eval_every == 0 or step == args.iters:
                log_eval(step, run_sum / run_n)
                run_sum, run_n = 0.0, 0
                logging.info(f"{setting['name']}: {(time.time() - t0) / step:.2f} s/step")

        ref["train_pseudo_end"] = eval_refine(train, sr, fm, args.n_steps, args.n_avg, stats)
        ref["val_pseudo_end"] = eval_refine(val, sr, fm, args.n_steps, args.n_avg, stats)
        result = {"setting": setting, "hist": hist, "refine": ref}

    gathered = [None] * world if dist_on else [result]
    if dist_on:
        dist.all_gather_object(gathered, result)
    if is_main:
        res = [r for r in gathered if r is not None]
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "overfit.json"), "w") as f:
            json.dump({"args": vars(args), "ckpt": ckpt, "results": res}, f, indent=2)
        L = [f"# SR overfitting test against frozen FM: {run_name}", "",
             f"ckpt `{ckpt}`; train = {args.n_train} fixed 240px train patches, val = {args.n_val} fixed 480px val "
             f"patches; {args.iters} iters, batch {args.batch}, t ~ U({args.t_min}, {args.t_max}); l_tdp evaluated "
             f"at t in {T_EVAL} with fixed eps", "",
             "### l_tdp (train / val)", "",
             "| setting | " + " | ".join(f"step {h['step']}" for h in res[0]["hist"]) + " | end/start train | end/start val |",
             "|---" * (len(res[0]["hist"]) + 3) + "|"]
        for r in res:
            h = r["hist"]
            cells = [f"{x['train_l_tdp']:.3f} / {x['val_l_tdp']:.3f}" for x in h]
            L.append(f"| {r['setting']['name']} | " + " | ".join(cells) +
                     f" | {h[-1]['train_l_tdp'] / h[0]['train_l_tdp']:.3f} | {h[-1]['val_l_tdp'] / h[0]['val_l_tdp']:.3f} |")
        L += ["", "### l_pix (train / val)", "", "| setting | step 0 | end |", "|---|---|---|"]
        for r in res:
            h = r["hist"]
            L.append(f"| {r['setting']['name']} | {h[0]['train_l_pix']:.4f} / {h[0]['val_l_pix']:.4f} | "
                     f"{h[-1]['train_l_pix']:.4f} / {h[-1]['val_l_pix']:.4f} |")
        L += ["", f"### Refinement with pseudo-PS ({args.n_steps} steps x {args.n_avg} draws): R2 / HF corr", "",
              "| setting | train start | train end | val start | val end |", "|---|---|---|---|---|"]
        r0 = res[0]["refine"]
        L.append(f"| (real PS reference) | {r0['train_real']['r2']:.4f} / {r0['train_real']['hf_corr']:.3f} | | "
                 f"{r0['val_real']['r2']:.4f} / {r0['val_real']['hf_corr']:.3f} | |")
        for r in res:
            f = r["refine"]
            L.append(f"| {r['setting']['name']} | " + " | ".join(
                f"{f[k]['r2']:.4f} / {f[k]['hf_corr']:.3f}"
                for k in ("train_pseudo_start", "train_pseudo_end", "val_pseudo_start", "val_pseudo_end")) + " |")
        with open(os.path.join(out_dir, "overfit.md"), "w") as f:
            f.write("\n".join(L) + "\n")
        logging.info("\n" + "\n".join(L))
    if dist_on:
        dist.barrier()
        dist.destroy_process_group()
