"""
Why does the TDP loss not help? Internal diagnostics of a trained SR + FM refiner
(sr_fm_refiner_v12-lumi-nd / v13-lumi-nd family), on the spatial test split.

All FM forwards use the training-time TDP construction of _sr_step exactly:
  z_t = (1-t) z_coarse + t z_gt + (1-t) noise_sigma eps,   feature = cfg.sr_loss.tdp_hook
  l_tdp = L1(feat(pseudo-PS), feat(real PS)),             l_pix = L1(pseudo-PS, real PS)
Conditions (PS slot of the ControlNet input; landsat_hr is always concatenated):
  real   real PlanetScope            fake   SR pseudo-PS
  null   null-PS token               lsps   landsat_hr in the PS slot
  fake_nols  pseudo-PS with landsat_hr channels zeroed (exp 2 only)

Exp 1  gradient diagnosis (per patch, t in T_GRAD, fixed eps)
  - ||dl_tdp/dtheta_SR|| / ||dl_pix/dtheta_SR|| and their cosine
  - spectrum of dl/d(pseudo-PS) by spatial-period band, and per band the cosine
    between the descent direction (-grad) and d = real - pseudo
    ("does the loss push pseudo-PS towards real PS in this band?")
Exp 2  sensitivity of the hook feature f and the output v to the PS condition,
  per t in T_SENS: ||f_a - f_b|| / ||f_fake|| for several pairs, plus the
  one-step height estimate x1 = decode(z_t + (1-t) v) (metres): MAE vs GT per
  condition and mean |x1_a - x1_b|.
Exp 3  TDP-optimal input: the pseudo-PS of the first --n_opt patches is replaced
  by a free pixel tensor (init = SR output) optimised to minimise l_tdp with
  random t/eps per iteration, as in training. Reports l_tdp before/after on a
  fixed (t, eps) set, the band spectrum of the change vs real - pseudo, and the
  full refinement metrics (final_n_steps x final_n_avg) with the optimised input.
Exp 0 (shared with exp 3)  refinement metrics on all patches for fake / real /
  null / lsps conditions — answers "does real PS help *this* FM at all?".

Launch: script/depth/diag_tdp.sh (one task per GPU). Output: <out_dir>/diag.md,
diag.json, panels/*.png.
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import eval_detail_metrics as edm
from diag_refiner_gain import load as load_run
from eval_detail_metrics_nd import make_test_dataset

OUT_ROOT = edm.OUT_ROOT
T_GRAD = [0.1, 0.3, 0.5, 0.7, 0.9]
T_SENS = [round(0.1 * i, 1) for i in range(10)]
BAND_EDGES = [4.0, 8.0, 16.0, 32.0]          # spatial period in px (3 m/px)
BAND_LABELS = ["<4px (<12m)", "4-8px", "8-16px", "16-32px", ">=32px (>=96m, incl. DC)"]
INFER_CONDS = ["fake", "real", "null", "lsps"]
SENS_PAIRS = [("real", "fake"), ("real", "null"), ("fake", "null"), ("lsps", "null"),
              ("fake_nols", "fake")]


# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------

class Sums(dict):
    """dict of float sums that merges across ranks by key-wise addition."""

    def add(self, key, val):
        self[key] = self.get(key, 0.0) + float(val)


def merge(dicts):
    out = Sums()
    for d in dicts:
        for k, v in d.items():
            out.add(k, v)
    return out


_MASKS = {}


def band_masks(H, W, device):
    key = (H, W, str(device))
    if key not in _MASKS:
        fy = torch.fft.fftfreq(H, device=device).view(-1, 1)
        fx = torch.fft.fftfreq(W, device=device).view(1, -1)
        r = torch.sqrt(fy ** 2 + fx ** 2)
        period = torch.where(r > 0, 1.0 / r.clamp(min=1e-12), torch.full_like(r, float("inf")))
        edges = [0.0] + BAND_EDGES + [float("inf")]
        _MASKS[key] = [(period >= lo) & (period < hi) if hi != float("inf") else (period >= lo)
                       for lo, hi in zip(edges[:-1], edges[1:])]
    return _MASKS[key]


def band_accumulate(acc, prefix, a, b=None):
    """Per period band: energy of a (and b), and cross term Re<A, B> (for pooled cosine)."""
    A = torch.fft.fft2(a.double())
    pa = (A.abs() ** 2).sum(dim=(0, 1))
    if b is not None:
        B = torch.fft.fft2(b.double())
        pb = (B.abs() ** 2).sum(dim=(0, 1))
        cross = (A.conj() * B).real.sum(dim=(0, 1))
    for i, m in enumerate(band_masks(a.shape[-2], a.shape[-1], a.device)):
        acc.add(f"{prefix}/E_a/{i}", pa[m].sum())
        if b is not None:
            acc.add(f"{prefix}/E_b/{i}", pb[m].sum())
            acc.add(f"{prefix}/X/{i}", cross[m].sum())


def band_report(S, prefix, name_a, name_b=None):
    Ea = [S.get(f"{prefix}/E_a/{i}", 0.0) for i in range(len(BAND_LABELS))]
    rows = []
    for i, lab in enumerate(BAND_LABELS):
        r = {"band": lab, f"energy_frac_{name_a}": Ea[i] / max(sum(Ea), 1e-30)}
        if name_b is not None:
            Eb = [S.get(f"{prefix}/E_b/{j}", 0.0) for j in range(len(BAND_LABELS))]
            r[f"energy_frac_{name_b}"] = Eb[i] / max(sum(Eb), 1e-30)
            r[f"energy_ratio_{name_a}_over_{name_b}"] = Ea[i] / max(Eb[i], 1e-30)
            r["cos"] = S.get(f"{prefix}/X/{i}", 0.0) / max((Ea[i] * Eb[i]) ** 0.5, 1e-30)
        rows.append(r)
    return rows


# -------------------------------------------------------------------------
# FM forward with the TDP hook (mirrors train_sr_fm_refiner._sr_step)
# -------------------------------------------------------------------------

class TDPProbe:
    def __init__(self, fm, hook_name):
        assert hook_name.startswith("unet.up_blocks."), f"only unet.up_blocks.N hooks supported, got {hook_name}"
        self.fm = fm
        self.hook_module = fm.unet.up_blocks[int(hook_name.split(".")[-1])]

    def z_t(self, z_c, z_gt, t, eps):
        t4 = torch.full((z_c.shape[0], 1, 1, 1), t, device=z_c.device, dtype=z_c.dtype)
        z = (1.0 - t4) * z_c + t4 * z_gt
        if self.fm.noise_sigma > 0.0:
            z = z + (1.0 - t4) * self.fm.noise_sigma * eps
        return z

    def run(self, cond, z_t, z_c, t):
        """cond: full ControlNet input (landsat_hr + PS). Returns (hook feature, v)."""
        fm = self.fm
        B = z_t.shape[0]
        t_int = (torch.full((B,), t, device=z_t.device) * 999).long()
        sample = torch.cat([z_t, z_c], dim=1) if fm.concat_z_coarse else z_t
        text = fm.empty_text_embed.to(z_t.device).expand(B, -1, -1)
        feats = {}
        h = self.hook_module.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
        try:
            cn = fm.controlnet(sample=sample, timestep=t_int, encoder_hidden_states=text,
                               controlnet_cond=cond, return_dict=True)
            v = fm.unet(sample=sample, timestep=t_int, encoder_hidden_states=text,
                        down_block_additional_residuals=cn.down_block_res_samples,
                        mid_block_additional_residual=cn.mid_block_res_sample, return_dict=True).sample
        finally:
            h.remove()
        return feats["f"], v


def make_eps(shape, device, seed):
    g = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(shape, generator=g, device=device)


# -------------------------------------------------------------------------
# Visualisation
# -------------------------------------------------------------------------

def _rgb(x):
    return ((x[:3].detach().float().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


def save_panel(path, imgs, hmaps, title, vmax):
    n = len(imgs) + len(hmaps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    for ax, (name, im) in zip(axes, imgs):
        ax.imshow(im)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax, (name, hm) in zip(axes[len(imgs):], hmaps):
        ax.imshow(hm.squeeze().float().cpu().numpy(), cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=f"{OUT_ROOT}/sr_fm_refiner_v12-lumi-nd")
    ap.add_argument("--ckpt", default=None, help="default: <run>/checkpoint/best.pth")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default=None, help="default: <OUT_ROOT>/_diag_tdp/<tag>")
    ap.add_argument("--n_patches", type=int, default=128, help="test patches (evenly spaced) for exp 0/1/2")
    ap.add_argument("--n_opt", type=int, default=48, help="first n of those used for exp 3")
    ap.add_argument("--opt_iters", type=int, default=200)
    ap.add_argument("--opt_lr", type=float, default=1e-2)
    ap.add_argument("--n_steps", type=int, default=None, help="default: cfg.validation.final_n_steps")
    ap.add_argument("--n_avg", type=int, default=None, help="default: cfg.validation.final_n_avg")
    ap.add_argument("--n_panels", type=int, default=6)
    ap.add_argument("--skip", default="", help="comma list of exps to skip: 1,2,3 (3 also skips exp 0)")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}

    tag = args.tag or os.path.basename(args.run.rstrip("/"))
    out_dir = args.out_dir or f"{OUT_ROOT}/_diag_tdp/{tag}"
    ckpt = args.ckpt or os.path.join(args.run, "checkpoint", "best.pth")

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
                                timeout=timedelta(minutes=60))
    if use_cuda:
        idx = local % max(1, torch.cuda.device_count())
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")

    cfg, chm, fm, sr_module, n_ls = load_run("sr", args.run, ckpt, device, is_main)
    fm.requires_grad_(False)
    sr_module.eval()                     # drop_path off (training ran SwinIR in train mode)
    sr_params = [p for p in sr_module.parameters()]
    for p in sr_params:
        p.requires_grad_(True)
    probe = TDPProbe(fm, str(cfg.sr_loss.get("tdp_hook", "unet.up_blocks.3")))
    n_steps = args.n_steps or int(cfg.validation.get("final_n_steps", cfg.validation.n_steps))
    n_avg = args.n_avg or int(cfg.validation.get("final_n_avg", cfg.validation.get("n_avg", 1)))

    cfg_d = cfg.dataset
    stats = edm._load_target_stats(cfg_d.target_stats_file, cfg_d.year)
    p1, p99 = float(stats["p1"][0]), float(stats["p99"][0])
    test_ds, _ = make_test_dataset(cfg_d)
    sel = sorted(set(np.linspace(0, len(test_ds) - 1, args.n_patches).round().astype(int).tolist()))
    opt_set = set(sel[: args.n_opt])
    local_idx = sel[rank::world]
    if is_main:
        logging.info(f"test patches {len(test_ds)}; using {len(sel)} (exp3 on {len(opt_set)}); "
                     f"hook={cfg.sr_loss.get('tdp_hook')} noise_sigma={fm.noise_sigma} "
                     f"refine n_steps={n_steps} n_avg={n_avg}; null_ps={'trained' if fm._null_ps is not None else 'none'}")
        os.makedirs(os.path.join(out_dir, "panels"), exist_ok=True)
    loader = DataLoader(Subset(test_ds, local_idx), batch_size=1, shuffle=False,
                        num_workers=cfg_d.workers, pin_memory=True)

    S = Sums()
    det_all = {c: edm.DetailAccumulator(device, 2.0, 1, 2.0) for c in INFER_CONDS}
    det_opt = {c: edm.DetailAccumulator(device, 2.0, 1, 2.0) for c in INFER_CONDS + ["opt"]}
    n_panel_done = 0

    for sample_i, batch in zip(local_idx, tqdm(loader, disable=not is_main, desc="diag_tdp")):
        inputs_lr, inputs_hr, targets = [x.squeeze(0) if x.dim() == 5 else x for x in batch]
        inputs_lr, real_ps, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
        H, W = targets.shape[-2:]
        landsat = inputs_lr[:, :n_ls]
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)
        with torch.no_grad():
            h_coarse = edm._normalize_coarse(chm((landsat + 1.0) / 2.0, target_size=(H, W)), stats)
            z_c, z_gt = fm.encode(h_coarse), fm.encode(targets)
            gt_m = edm._denormalize(targets, stats).clamp(p1, p99)
            null_ps = (fm._null_ps.to(device, real_ps.dtype) if fm._null_ps is not None
                       else torch.zeros(1, real_ps.shape[1], 1, 1, device=device)).expand_as(real_ps)
        cat = lambda ps, ls=landsat_hr: fm._maybe_concat_landsat(ps, ls)

        # ---------------- Exp 1: gradients ----------------
        with torch.set_grad_enabled("1" not in skip):
            pseudo_g = sr_module(landsat, target_size=(H, W))      # graph to SR params (exp 1 only)
        pseudo = pseudo_g.detach()
        if "1" not in skip:
            p_leaf = pseudo.clone().requires_grad_(True)
            l_pix = F.l1_loss(p_leaf, real_ps)
            g_pix_img, = torch.autograd.grad(l_pix, p_leaf)
            g_tdp_img = torch.zeros_like(pseudo)
            l_tdp_sum = 0.0
            for ti, t in enumerate(T_GRAD):
                eps = make_eps(z_c.shape, device, sample_i * 100 + ti)
                zt = probe.z_t(z_c, z_gt, t, eps)
                with torch.no_grad():
                    f_real, _ = probe.run(cat(real_ps), zt, z_c, t)
                f_fake, _ = probe.run(cat(p_leaf), zt, z_c, t)
                l = F.l1_loss(f_fake, f_real) / len(T_GRAD)
                g, = torch.autograd.grad(l, p_leaf)
                g_tdp_img += g
                l_tdp_sum += l.item()
            # vector-Jacobian products through the SR network
            gp = torch.autograd.grad(pseudo_g, sr_params, grad_outputs=g_pix_img, retain_graph=True)
            gt_ = torch.autograd.grad(pseudo_g, sr_params, grad_outputs=g_tdp_img)
            vp = torch.cat([x.flatten() for x in gp]).double()
            vt = torch.cat([x.flatten() for x in gt_]).double()
            S.add("e1/n", 1)
            S.add("e1/l_pix", l_pix.item())
            S.add("e1/l_tdp", l_tdp_sum)
            S.add("e1/ratio_param", (vt.norm() / vp.norm().clamp(min=1e-30)).item())
            S.add("e1/log10_ratio_param", torch.log10(vt.norm() / vp.norm().clamp(min=1e-30)).item())
            S.add("e1/cos_param", F.cosine_similarity(vt, vp, dim=0).item())
            S.add("e1/ratio_img", (g_tdp_img.norm() / g_pix_img.norm().clamp(min=1e-30)).item())
            S.add("e1/cos_img", F.cosine_similarity(g_tdp_img.flatten(), g_pix_img.flatten(), dim=0).item())
            d = (real_ps - pseudo)
            band_accumulate(S, "e1/tdp_vs_d", -g_tdp_img, d)
            band_accumulate(S, "e1/pix_vs_d", -g_pix_img, d)
            del gp, gt_, vp, vt
        del pseudo_g

        with torch.no_grad():
            # ---------------- Exp 2: sensitivity ----------------
            if "2" not in skip:
                conds = {"real": cat(real_ps), "fake": cat(pseudo), "null": cat(null_ps),
                         "lsps": cat(landsat_hr), "fake_nols": cat(pseudo, torch.zeros_like(landsat_hr))}
                for ti, t in enumerate(T_SENS):
                    eps = make_eps(z_c.shape, device, sample_i * 100 + 50 + ti)
                    zt = probe.z_t(z_c, z_gt, t, eps)
                    out = {k: probe.run(c, zt, z_c, t) for k, c in conds.items()}
                    S.add(f"e2/{t}/n", 1)
                    S.add(f"e2/{t}/f_norm2", (out["fake"][0].double() ** 2).sum())
                    S.add(f"e2/{t}/v_norm2", (out["fake"][1].double() ** 2).sum())
                    for a, b in SENS_PAIRS:
                        S.add(f"e2/{t}/f/{a}-{b}", ((out[a][0] - out[b][0]).double() ** 2).sum())
                        S.add(f"e2/{t}/v/{a}-{b}", ((out[a][1] - out[b][1]).double() ** 2).sum())
                    S.add(f"e2/{t}/l_tdp", F.l1_loss(out["fake"][0], out["real"][0]))
                    x1 = {k: edm._denormalize(fm.decode(zt + (1.0 - t) * out[k][1]), stats).clamp(p1, p99)
                          for k in ("real", "fake", "null", "lsps")}
                    S.add(f"e2/{t}/npix", gt_m.numel())
                    for k, h in x1.items():
                        S.add(f"e2/{t}/x1_abs/{k}", (h - gt_m).abs().sum())
                        S.add(f"e2/{t}/x1_sq/{k}", ((h - gt_m) ** 2).sum())
                    S.add(f"e2/{t}/x1_diff/real-fake", (x1["real"] - x1["fake"]).abs().sum())
                    S.add(f"e2/{t}/x1_diff/fake-null", (x1["fake"] - x1["null"]).abs().sum())
                    S.add(f"e2/{t}/x1_diff/real-null", (x1["real"] - x1["null"]).abs().sum())
                    del out, x1

        if "3" in skip:
            continue

        # ---------------- Exp 3: optimise the PS input against l_tdp ----------------
        x_opt = None
        if sample_i in opt_set:
            eval_eps = [make_eps(z_c.shape, device, sample_i * 100 + 80 + ti) for ti in range(len(T_SENS))]

            def eval_tdp(ps):
                with torch.no_grad():
                    tot = 0.0
                    for t, eps in zip(T_SENS, eval_eps):
                        zt = probe.z_t(z_c, z_gt, t, eps)
                        tot += F.l1_loss(probe.run(cat(ps), zt, z_c, t)[0],
                                         probe.run(cat(real_ps), zt, z_c, t)[0]).item()
                    return tot / len(T_SENS)

            x = pseudo.clone().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=args.opt_lr)
            g = torch.Generator(device=device).manual_seed(sample_i * 7 + 1)
            for it in range(args.opt_iters):
                t = float(torch.rand(1, generator=g, device=device))
                eps = torch.randn(z_c.shape, generator=g, device=device)
                zt = probe.z_t(z_c, z_gt, t, eps)
                with torch.no_grad():
                    f_real, _ = probe.run(cat(real_ps), zt, z_c, t)
                loss = F.l1_loss(probe.run(cat(x), zt, z_c, t)[0], f_real)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(-1.0, 1.0)
            x_opt = x.detach()
            S.add("e3/n", 1)
            S.add("e3/l_tdp/fake", eval_tdp(pseudo))
            S.add("e3/l_tdp/opt", eval_tdp(x_opt))
            S.add("e3/l_tdp/lsps", eval_tdp(landsat_hr))
            S.add("e3/l_tdp/null", eval_tdp(null_ps))
            S.add("e3/l1_to_real/fake", F.l1_loss(pseudo, real_ps))
            S.add("e3/l1_to_real/opt", F.l1_loss(x_opt, real_ps))
            band_accumulate(S, "e3/delta_vs_d", x_opt - pseudo, real_ps - pseudo)

        # ---------------- Exp 0 (+3): refinement metrics per condition ----------------
        with torch.no_grad():
            ps_by = {"fake": pseudo, "real": real_ps, "null": null_ps, "lsps": landsat_hr}
            if x_opt is not None:
                ps_by["opt"] = x_opt
            preds = {}
            for c, ps in ps_by.items():
                torch.manual_seed(sample_i)          # identical init noise for every condition
                hs = [fm.refine_with_ps(ps, h_coarse, n_steps=n_steps, method="euler", landsat_hr=landsat_hr)
                      for _ in range(n_avg)]
                preds[c] = edm._denormalize(torch.stack(hs).mean(0), stats).clamp(p1, p99)
            for c in INFER_CONDS:
                det_all[c].update(preds[c].float(), gt_m.float())
            if x_opt is not None:
                for c in INFER_CONDS + ["opt"]:
                    det_opt[c].update(preds[c].float(), gt_m.float())
                if is_main and n_panel_done < args.n_panels:
                    diff = (x_opt - pseudo) * 2.5           # x2.5 amplified; _rgb maps 0 to mid-grey
                    save_panel(os.path.join(out_dir, "panels", f"patch{sample_i:05d}.png"),
                               [("Landsat (bilinear)", _rgb(landsat_hr[0])), ("pseudo-PS (SR)", _rgb(pseudo[0])),
                                ("TDP-optimised", _rgb(x_opt[0])), ("optimised - pseudo (x2.5)", _rgb(diff[0])),
                                ("real PS", _rgb(real_ps[0]))],
                               [("pred: pseudo", preds["fake"][0]), ("pred: optimised", preds["opt"][0]),
                                ("pred: real PS", preds["real"][0]), ("GT", gt_m[0])],
                               f"patch {sample_i}", vmax=p99)
                    n_panel_done += 1

    # ---------------- gather + report ----------------
    payload = (dict(S), {c: a.state() for c, a in det_all.items()}, {c: a.state() for c, a in det_opt.items()})
    gathered = [None] * world if dist_on else [payload]
    if dist_on:
        dist.all_gather_object(gathered, payload)
    if is_main:
        S = merge([g[0] for g in gathered])
        res = {"meta": {"run": args.run, "ckpt": ckpt, "n_patches": len(sel), "n_opt": len(opt_set),
                        "opt_iters": args.opt_iters, "opt_lr": args.opt_lr, "n_steps": n_steps, "n_avg": n_avg,
                        "tdp_hook": str(cfg.sr_loss.get("tdp_hook")), "noise_sigma": fm.noise_sigma}}
        L = [f"# TDP diagnostics: {tag}", "", f"ckpt `{ckpt}`; {len(sel)} test patches "
             f"(exp 3 on {len(opt_set)}); hook `{res['meta']['tdp_hook']}`; refine {n_steps} steps x {n_avg} draws", ""]

        def det_table(states_by_cond, conds, title):
            L.extend([f"### {title}", "", "| PS condition | R2 | MAE | RMSE | HF corr | HF energy ratio | Edge F1 tol1 |",
                      "|---|---|---|---|---|---|---|"])
            out = {}
            for c in conds:
                m = edm.finalize([st[c] for st in states_by_cond])
                out[c] = m
                L.append(f"| {c} | {m['r2']:.4f} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['hf_corr']:.4f} | "
                         f"{m['hf_energy_ratio']:.4f} | {m['edge_f1_tol1']:.4f} |")
            L.append("")
            return out

        if "3" not in skip:
            res["exp0"] = det_table([g[1] for g in gathered], INFER_CONDS,
                                    "Exp 0: refinement with each PS condition (all patches)")
            n3 = S.get("e3/n", 0)
            if n3:
                res["exp3_refine"] = det_table([g[2] for g in gathered], INFER_CONDS + ["opt"],
                                               f"Exp 3: refinement incl. TDP-optimised input ({int(n3)} patches)")
                e3 = {k: S[f"e3/{k}"] / n3 for k in ("l_tdp/fake", "l_tdp/opt", "l_tdp/lsps", "l_tdp/null",
                                                     "l1_to_real/fake", "l1_to_real/opt")}
                res["exp3"] = e3
                res["exp3_bands"] = band_report(S, "e3/delta_vs_d", "opt_minus_pseudo", "real_minus_pseudo")
                L += ["### Exp 3: l_tdp on a fixed (t, eps) set (real PS = 0)", "",
                      "| input | l_tdp | L1 to real PS |", "|---|---|---|",
                      f"| pseudo-PS (SR) | {e3['l_tdp/fake']:.4f} | {e3['l1_to_real/fake']:.4f} |",
                      f"| TDP-optimised | {e3['l_tdp/opt']:.4f} | {e3['l1_to_real/opt']:.4f} |",
                      f"| landsat_hr as PS | {e3['l_tdp/lsps']:.4f} | - |",
                      f"| null PS | {e3['l_tdp/null']:.4f} | - |", "",
                      "Change made by the optimisation vs the real-pseudo gap, per spatial-period band "
                      "(cos > 0: the optimiser moved pseudo-PS towards real PS in that band)", "",
                      "| band | energy frac (opt-pseudo) | energy frac (real-pseudo) | energy ratio | cos |",
                      "|---|---|---|---|---|"]
                for r in res["exp3_bands"]:
                    L.append(f"| {r['band']} | {r['energy_frac_opt_minus_pseudo']:.3f} | "
                             f"{r['energy_frac_real_minus_pseudo']:.3f} | "
                             f"{r['energy_ratio_opt_minus_pseudo_over_real_minus_pseudo']:.3f} | {r['cos']:+.3f} |")
                L.append("")

        n1 = S.get("e1/n", 0)
        if n1:
            e1 = {k.split("/", 1)[1]: S[k] / n1 for k in S if k.startswith("e1/") and k.count("/") == 1 and k != "e1/n"}
            e1["geomean_ratio_param"] = 10 ** e1.pop("log10_ratio_param")
            res["exp1"] = e1
            res["exp1_bands_tdp"] = band_report(S, "e1/tdp_vs_d", "neg_grad_tdp", "real_minus_pseudo")
            res["exp1_bands_pix"] = band_report(S, "e1/pix_vs_d", "neg_grad_pix", "real_minus_pseudo")
            L += [f"### Exp 1: gradients (t in {T_GRAD}, mean over {int(n1)} patches)", "",
                  f"l_pix = {e1['l_pix']:.4f}, l_tdp = {e1['l_tdp']:.4f}", "",
                  "| quantity | value |", "|---|---|",
                  f"| ‖∇θ l_tdp‖ / ‖∇θ l_pix‖ (SR params; mean / geo-mean) | {e1['ratio_param']:.4f} / "
                  f"{e1['geomean_ratio_param']:.4f} |",
                  f"| cos(∇θ l_tdp, ∇θ l_pix) | {e1['cos_param']:+.4f} |",
                  f"| ‖∇x l_tdp‖ / ‖∇x l_pix‖ (pseudo-PS pixels) | {e1['ratio_img']:.4f} |",
                  f"| cos(∇x l_tdp, ∇x l_pix) | {e1['cos_img']:+.4f} |", "",
                  "Descent direction (-grad wrt pseudo-PS) per spatial-period band; cos = alignment with real - pseudo", "",
                  "| band | energy frac -∇l_tdp | cos(-∇l_tdp, d) | energy frac -∇l_pix | cos(-∇l_pix, d) | "
                  "energy frac d=real-pseudo |", "|---|---|---|---|---|---|"]
            for rt, rp in zip(res["exp1_bands_tdp"], res["exp1_bands_pix"]):
                L.append(f"| {rt['band']} | {rt['energy_frac_neg_grad_tdp']:.3f} | {rt['cos']:+.3f} | "
                         f"{rp['energy_frac_neg_grad_pix']:.3f} | {rp['cos']:+.3f} | "
                         f"{rt['energy_frac_real_minus_pseudo']:.3f} |")
            L.append("")

        if S.get(f"e2/{T_SENS[0]}/n", 0):
            res["exp2"] = {}
            pairs = [f"{a}-{b}" for a, b in SENS_PAIRS]
            L += ["### Exp 2: sensitivity of the hook feature f to the PS condition: ‖f_a − f_b‖ / ‖f_fake‖", "",
                  "| t | " + " | ".join(pairs) + " | l_tdp |", "|---" * (len(pairs) + 2) + "|"]
            for t in T_SENS:
                fn = S[f"e2/{t}/f_norm2"] ** 0.5
                row = {p: S[f"e2/{t}/f/{p}"] ** 0.5 / fn for p in pairs}
                row["l_tdp"] = S[f"e2/{t}/l_tdp"] / S[f"e2/{t}/n"]
                vn = S[f"e2/{t}/v_norm2"] ** 0.5
                row["v"] = {p: S[f"e2/{t}/v/{p}"] ** 0.5 / vn for p in pairs}
                npx = S[f"e2/{t}/npix"]
                row["x1_mae"] = {k: S[f"e2/{t}/x1_abs/{k}"] / npx for k in ("real", "fake", "null", "lsps")}
                row["x1_rmse"] = {k: (S[f"e2/{t}/x1_sq/{k}"] / npx) ** 0.5 for k in ("real", "fake", "null", "lsps")}
                row["x1_diff"] = {k: S[f"e2/{t}/x1_diff/{k}"] / npx for k in ("real-fake", "fake-null", "real-null")}
                res["exp2"][str(t)] = row
                L.append(f"| {t} | " + " | ".join(f"{row[p]:.4f}" for p in pairs) + f" | {row['l_tdp']:.4f} |")
            L += ["", "Same for the output velocity v: ‖v_a − v_b‖ / ‖v_fake‖", "",
                  "| t | " + " | ".join(pairs) + " |", "|---" * (len(pairs) + 1) + "|"]
            for t in T_SENS:
                r = res["exp2"][str(t)]["v"]
                L.append(f"| {t} | " + " | ".join(f"{r[p]:.4f}" for p in pairs) + " |")
            L += ["", "One-step height estimate x1 = decode(z_t + (1-t) v), metres: MAE vs GT per condition, "
                      "and mean |x1_a − x1_b|", "",
                  "| t | MAE real | MAE fake | MAE null | MAE lsps | |real−fake| | |fake−null| | |real−null| |",
                  "|---|---|---|---|---|---|---|---|"]
            for t in T_SENS:
                r = res["exp2"][str(t)]
                L.append(f"| {t} | {r['x1_mae']['real']:.3f} | {r['x1_mae']['fake']:.3f} | {r['x1_mae']['null']:.3f} | "
                         f"{r['x1_mae']['lsps']:.3f} | {r['x1_diff']['real-fake']:.3f} | "
                         f"{r['x1_diff']['fake-null']:.3f} | {r['x1_diff']['real-null']:.3f} |")
            L.append("")

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "diag.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(out_dir, "diag.md"), "w") as f:
            f.write("\n".join(L))
        logging.info("\n" + "\n".join(L))
    if dist_on:
        dist.barrier()
        dist.destroy_process_group()
