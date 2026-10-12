"""
Diagnostics for fm_refiner-D-lumi-0 (backbone "dit_omini"), single GPU.

  --test vae      VAE floor + latent statistics + effect of the noise sigma, and the trained
                  D-lumi-0 best.pth evaluated on the same samples (sample_sigma sweep).
  --test overfit  Fresh D-lumi-0 model trained on a tiny fixed set (is the model / loss / sampler
                  correct and able to fit?). Two runs: (A) sigma=0, no PS dropout -> pure
                  implementation/capacity check; (B) the real recipe (sigma=1, PS dropout 0.3).
  --test both     both of the above.

All samples are 240 px val-split patches, cached after the first load (same samples in every test).
Metrics are in metres (R2 / MAE / RMSE over all pixels), as in train_fm_refiner.py's validate().
"""

import argparse
import json
import math
import os
import random
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import train_fm_refiner as T            # helpers only (the training entry point is __main__-guarded)
from depthfm.chmv2 import load_chmv2_baseline
from depthfm.fm_refiner import build_fm_refiner
from src.util.ps_lazydataset import LazyPatchDataset


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------

def load_samples(cfg, n, device, workers, cache_path):
    """n val-split 240px samples: landsat_hr, ps, h_coarse (normalised [-1,1]), gt (normalised)."""
    if os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        if d["gt"].shape[0] >= n:
            print(f"[data] loaded {d['gt'].shape[0]} cached samples from {cache_path}", flush=True)
            return {k: v[:n] for k, v in d.items() if torch.is_tensor(v)} | {"target_stats": d["target_stats"]}

    cd = cfg.dataset
    base = LazyPatchDataset({
        'input_dir': cd.input_dir, 'input_dir_hr': cd.input_dir_hr, 'output_dir': cd.output_dir,
        'selected_bands': cd.selected_bands, 'selected_bands_hr': cd.selected_bands_hr,
        'file_type_input': cd.file_type_input, 'file_type_output': cd.file_type_output,
        'patch_sizes': [(s, s) for s in cd.patch_sizes], 'patch_coord_path': cd.patch_coord_path,
        'num_patches_per_tile': cd.num_patches_per_tile, 'tile_emphasis': cd.tile_emphasis,
        'correlation_cleaning': cd.correlation_cleaning, 'target_range_edges': cd.target_range_edges,
        'num_patches_per_target_range': cd.num_patches_per_target_range,
        'selected_percentile': cd.selected_percentile, 'year': cd.year, 'batch_size': 1, 'mode': 'train',
        'use_input_minmax': cd.use_input_minmax, 'use_input_norm': cd.use_input_norm,
        'input_stats_file': cd.input_stats_file, 'use_input_hr_minmax': cd.use_input_hr_minmax,
        'use_input_hr_norm': cd.use_input_hr_norm, 'input_hr_stats_file': cd.input_hr_stats_file,
        'use_target_minmax': cd.use_target_minmax, 'use_target_norm': cd.use_target_norm,
        'target_stats_file': cd.target_stats_file, 'unit_scale_ratio': cd.unit_scale_ratio,
        'scale_input_to_neg1_1': cd.get('scale_input_to_neg1_1', False),
        'scale_input_hr_to_neg1_1': cd.get('scale_input_hr_to_neg1_1', False),
        'scale_target_to_neg1_1': cd.get('scale_target_to_neg1_1', False),
    })
    _, val_coords, _ = T._split_coords(base.patch_coords, cd.train_split, cd.val_split, cd.split_seed)
    val_ds = T._make_dataset(base, val_coords, 'val', 1)

    n_ls = len(cd.selected_bands)
    with open(cd.input_stats_file) as f:
        in_stats = json.load(f)[str(cd.year)]
    coarse_model = load_chmv2_baseline(
        model_dir=cfg.model.chmv2_model_id, ckpt_path=cfg.model.chmv2_ckpt_path,
        mean=in_stats["mean"], std=in_stats["std"],
        out_in_scale_factor=cfg.model.chmv2_out_in_scale_factor).to(device)
    target_stats = T._load_target_stats(cd.target_stats_file, cd.year)

    out = {k: [] for k in ("landsat_hr", "ps", "h_coarse", "gt")}
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=workers,
                        worker_init_fn=lambda w: (random.seed(1000 + w), np.random.seed(1000 + w)))
    t0 = time.time()
    for inputs_lr, inputs_hr, targets in loader:
        if inputs_lr.dim() == 5: inputs_lr = inputs_lr.squeeze(0)
        if inputs_hr.dim() == 5: inputs_hr = inputs_hr.squeeze(0)
        if targets.dim() == 5: targets = targets.squeeze(0)
        H, W = targets.shape[2:]
        if H != 240:
            continue
        inputs_lr, inputs_hr, targets = inputs_lr.to(device), inputs_hr.to(device), targets.to(device)
        landsat = inputs_lr[:, :n_ls]
        landsat_hr = F.interpolate(landsat, size=(H, W), mode="bilinear", align_corners=False)
        with torch.no_grad():
            hc = T._run_coarse(coarse_model, (landsat + 1.0) / 2.0, target_size=(H, W))
            hc = T._normalize_coarse(hc, target_stats)
        for k, v in (("landsat_hr", landsat_hr), ("ps", inputs_hr), ("h_coarse", hc), ("gt", targets)):
            out[k].append(v.cpu())
        if sum(x.shape[0] for x in out["gt"]) >= n:
            break
    out = {k: torch.cat(v, 0)[:n] for k, v in out.items()}
    print(f"[data] loaded {out['gt'].shape[0]} samples in {time.time() - t0:.0f}s", flush=True)
    out["target_stats"] = target_stats
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(out, cache_path)
    del coarse_model
    torch.cuda.empty_cache()
    return out


def batches(data, idx, bs, device):
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield {k: data[k][j].to(device) for k in ("landsat_hr", "ps", "h_coarse", "gt")}


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------

def build_model(cfg, device, noise_sigma, sample_sigma, ps_dropout):
    n_ls, n_ps = len(cfg.dataset.selected_bands), len(cfg.dataset.selected_bands_hr)
    fm = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path, n_landsat_bands=n_ls, n_ps_bands=n_ps,
        ps_dropout_p=ps_dropout, use_controlnet=cfg.trainer.get("use_controlnet", True),
        controlnet_cond_mode=cfg.trainer.get("controlnet_cond_mode", "landsat_ps"),
        noise_sigma=noise_sigma, sample_sigma=sample_sigma,
        sample_noise_mode=cfg.trainer.get("sample_noise_mode", "init_only"), device=str(device),
        pretrained_unet=cfg.model.get("pretrained_unet", True), backbone=cfg.model.get("backbone", "unet"),
        dit_kwargs=OmegaConf.to_container(cfg.model.get("dit", {}) or {}, resolve=True),
        loss_pixel_weight=cfg.trainer.get("loss_pixel_weight", 0.0),
        pixel_loss_t_min=cfg.trainer.get("pixel_loss_t_min", 0.5))
    fm = fm.to(device)
    fm._get_null_ps(n_ps, device, torch.float32)       # lazily created in training; needed by refine()
    return fm


def metrics_m(pred_norm, gt_norm, stats):
    return T.compute_metrics(T._denormalize_target(pred_norm, stats).cpu(), T._denormalize_target(gt_norm, stats).cpu())


@torch.no_grad()
def refine_all(fm, data, idx, device, n_steps, ens=1, sample_sigma=None, bs=16):
    old = fm.sample_sigma
    if sample_sigma is not None:
        fm.sample_sigma = sample_sigma
    fm.eval()
    preds = []
    for b in batches(data, idx, bs, device):
        d = torch.stack([fm.refine(b["landsat_hr"], b["h_coarse"], n_steps=n_steps) for _ in range(ens)]).mean(0)
        preds.append(d.cpu())
    fm.sample_sigma = old
    return torch.cat(preds, 0)


@torch.no_grad()
def tf_loss(fm, data, idx, device, null_ps, reps=4, bs=16):
    """Teacher-forced FM loss with fixed (t, eps) draws; PS real or null; no PS dropout."""
    old_p, old_mode = fm.ps_dropout_p, fm.training
    fm.ps_dropout_p = 0.0
    fm.train()
    tot, n = 0.0, 0
    for r in range(reps):
        torch.manual_seed(1234 + r)
        for b in batches(data, idx, bs, device):
            ps = torch.zeros_like(b["ps"]) if null_ps else b["ps"]
            tot += fm(landsat_lr=b["landsat_hr"], ps_hr=ps, h_coarse=b["h_coarse"], h_gt=b["gt"]).item() * len(b["gt"])
            n += len(b["gt"])
    fm.ps_dropout_p = old_p
    fm.train(old_mode)
    return tot / n


# --------------------------------------------------------------------------------------------
# test 2: VAE floor / noise
# --------------------------------------------------------------------------------------------

@torch.no_grad()
def run_vae_test(cfg, data, device, args, out_dir):
    stats = data["target_stats"]
    N = data["gt"].shape[0]
    idx = list(range(N))
    fm = build_model(cfg, device, cfg.trainer.noise_sigma, cfg.trainer.sample_sigma, 0.0)
    fm.eval()
    gt, hc = data["gt"], data["h_coarse"]
    res = {"n_samples": N}

    def enc_dec(x, s=0.0, bs=32):
        outs, zs = [], []
        for i in range(0, len(x), bs):
            z = fm.encode(x[i:i + bs].to(device))
            zs.append(z.cpu())
            if s > 0:
                z = z + s * torch.randn_like(z)
            outs.append(fm.decode(z).cpu())
        return torch.cat(outs), torch.cat(zs)

    # 1) VAE round trip
    rt_gt, z_gt = enc_dec(gt)
    rt_hc, z_hc = enc_dec(hc)
    res["coarse_vs_gt"] = metrics_m(hc, gt, stats)
    res["vae(gt)_vs_gt"] = metrics_m(rt_gt, gt, stats)
    res["vae(coarse)_vs_coarse"] = metrics_m(rt_hc, hc, stats)
    res["vae(coarse)_vs_gt"] = metrics_m(rt_hc, gt, stats)

    # 2) latent statistics (units of the scaled latent, i.e. the units sigma is expressed in)
    dz = z_gt - z_hc
    def st(z):
        return {"std_all": z.std().item(), "mean_all": z.mean().item(),
                "std_per_ch": [round(z[:, c].std().item(), 4) for c in range(z.shape[1])],
                "mean_per_ch": [round(z[:, c].mean().item(), 4) for c in range(z.shape[1])]}
    res["latent_stats"] = {"z_gt": st(z_gt), "z_coarse": st(z_hc), "z_gt-z_coarse": st(dz),
                           "per_image_std_of_z_coarse_median": z_hc.flatten(1).std(1).median().item()}

    # 3) noise: decode(z_coarse + s*eps) against coarse / gt, and SNR
    zc_std = z_hc.std().item()
    sweep = {}
    torch.manual_seed(0)
    for s in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        p, _ = enc_dec(hc, s=s)
        sweep[str(s)] = {"snr_db": (20 * math.log10(zc_std / s)) if s > 0 else None,
                         "vs_coarse": metrics_m(p, hc, stats), "vs_gt": metrics_m(p, gt, stats)}
    res["decode(z_coarse+s*eps)"] = sweep
    # per-t noise level of the real recipe (sigma = noise_sigma): std of the noise at time t vs signal std
    sig = float(cfg.trainer.noise_sigma)
    res["recipe_noise_vs_t"] = {f"t={t}": {"noise_std": (1 - t) * sig,
                                           "snr_db": (20 * math.log10(zc_std / ((1 - t) * sig))) if t < 1 else None}
                                for t in (0.0, 0.25, 0.5, 0.75, 0.9)}

    # 4) trained D-lumi-0 best.pth on the same samples
    ckpt_path = args.ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        fm.unet.load_state_dict(ck["unet_state"])
        fm.controlnet.load_state_dict(ck["controlnet_state"])
        if ck.get("null_ps_state") is not None:
            fm._null_ps = torch.nn.Parameter(ck["null_ps_state"].detach().to(device))
        res["ckpt"] = {"path": ckpt_path, "step": ck.get("step")}
        runs = {}
        for name, kw in {"sigma=1.0,10 steps,1 draw": dict(sample_sigma=1.0, n_steps=10, ens=1),
                         "sigma=1.0,10 steps,5 draws": dict(sample_sigma=1.0, n_steps=10, ens=5),
                         "sigma=1.0,4 steps,1 draw": dict(sample_sigma=1.0, n_steps=4, ens=1),
                         "sigma=0.5,10 steps,1 draw": dict(sample_sigma=0.5, n_steps=10, ens=1),
                         "sigma=0.0,10 steps,1 draw": dict(sample_sigma=0.0, n_steps=10, ens=1)}.items():
            torch.manual_seed(0)
            p = refine_all(fm, data, idx, device, **kw)
            runs[name] = {"vs_gt": metrics_m(p, gt, stats), "vs_coarse(self-consistency)": metrics_m(p, hc, stats)}
        res["trained_D-lumi-0"] = runs
        # teacher-forced loss of the trained model, with/without PS
        res["trained_D-lumi-0_tf_loss"] = {"ps_real": tf_loss(fm, data, idx, device, False),
                                           "ps_null": tf_loss(fm, data, idx, device, True)}
    return res


# --------------------------------------------------------------------------------------------
# test 1: overfit
# --------------------------------------------------------------------------------------------

def run_overfit(cfg, data, device, args, out_dir, tag, noise_sigma, ps_dropout):
    stats = data["target_stats"]
    n_tr = args.n_overfit
    tr_idx = list(range(n_tr))
    ho_idx = list(range(n_tr, n_tr + args.n_heldout))
    torch.manual_seed(0)
    random.seed(0)
    fm = build_model(cfg, device, noise_sigma, noise_sigma, ps_dropout)
    n_par = sum(p.numel() for p in list(fm.unet.parameters()) + list(fm.controlnet.parameters())) / 1e6
    params = list(fm.unet.parameters()) + list(fm.controlnet.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warm = min(100, args.steps // 10)
    sched = SequentialLR(opt, [LinearLR(opt, start_factor=1e-3, total_iters=warm),
                               CosineAnnealingLR(opt, T_max=args.steps - warm, eta_min=1e-7)], milestones=[warm])
    log = {"params_M": n_par, "steps": [], "loss": [], "evals": []}
    print(f"[overfit {tag}] sigma={noise_sigma} ps_dropout={ps_dropout} params={n_par:.1f}M "
          f"train_set={n_tr} heldout={len(ho_idx)} steps={args.steps} bs={args.bs}", flush=True)

    coarse_tr = metrics_m(data["h_coarse"][tr_idx], data["gt"][tr_idx], stats)
    coarse_ho = metrics_m(data["h_coarse"][ho_idx], data["gt"][ho_idx], stats)
    log["coarse_baseline"] = {"train_set": coarse_tr, "heldout": coarse_ho}

    def evaluate(step):
        torch.manual_seed(0)
        p_tr = refine_all(fm, data, tr_idx, device, n_steps=10, ens=1)
        p_ho = refine_all(fm, data, ho_idx, device, n_steps=10, ens=1)
        e = {"step": step,
             "train_set_R2/MAE/RMSE": metrics_m(p_tr, data["gt"][tr_idx], stats),
             "heldout_R2/MAE/RMSE": metrics_m(p_ho, data["gt"][ho_idx], stats),
             "tf_loss_train_set(ps_null)": tf_loss(fm, data, tr_idx, device, True),
             "tf_loss_train_set(ps_real)": tf_loss(fm, data, tr_idx, device, False),
             "tf_loss_heldout(ps_null)": tf_loss(fm, data, ho_idx, device, True)}
        log["evals"].append(e)
        print(f"[overfit {tag}] eval@{step}: train R2={e['train_set_R2/MAE/RMSE']['r2']:.4f} "
              f"RMSE={e['train_set_R2/MAE/RMSE']['rmse']:.3f} | heldout R2={e['heldout_R2/MAE/RMSE']['r2']:.4f} "
              f"RMSE={e['heldout_R2/MAE/RMSE']['rmse']:.3f} | tf_loss train(null)={e['tf_loss_train_set(ps_null)']:.4f} "
              f"held(null)={e['tf_loss_heldout(ps_null)']:.4f}  (coarse: train R2={coarse_tr['r2']:.4f} held R2={coarse_ho['r2']:.4f})",
              flush=True)
        fm.train()
        return p_tr

    fm.train()
    acc, t0 = [], time.time()
    eval_steps = set(range(args.eval_every, args.steps + 1, args.eval_every)) | {args.steps}
    for step in range(1, args.steps + 1):
        sel = random.sample(tr_idx, args.bs)
        b = {k: data[k][sel].to(device) for k in ("landsat_hr", "ps", "h_coarse", "gt")}
        opt.zero_grad(set_to_none=True)
        loss = fm(landsat_lr=b["landsat_hr"], ps_hr=b["ps"], h_coarse=b["h_coarse"], h_gt=b["gt"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        acc.append(loss.item())
        if step % 25 == 0:
            log["steps"].append(step)
            log["loss"].append(float(np.mean(acc)))
            acc = []
            if step % 100 == 0:
                print(f"[overfit {tag}] step {step} loss(avg last 25)={log['loss'][-1]:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if step in eval_steps:
            p_tr = evaluate(step)
    # qualitative figure on the final model
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for r in range(4):
        for c, (name, arr) in enumerate((("coarse", data["h_coarse"][r]), ("refined", p_tr[r]), ("GT", data["gt"][r]),
                                          ("refined-GT", p_tr[r] - data["gt"][r]))):
            ax = axes[r, c]
            im = ax.imshow(arr[0].numpy(), cmap="plasma" if c < 3 else "coolwarm",
                           vmin=-1 if c < 3 else -0.5, vmax=1 if c < 3 else 0.5)
            ax.axis("off")
            if r == 0:
                ax.set_title(name)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"overfit_{tag}_samples.png"), dpi=80)
    plt.close(fig)
    del fm, opt
    torch.cuda.empty_cache()
    return log


def plot_loss(logs, out_dir):
    plt.figure(figsize=(7, 4))
    for tag, lg in logs.items():
        plt.plot(lg["steps"], lg["loss"], label=tag)
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("train FM loss (avg of 25 steps)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overfit_loss.png"), dpi=100)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["vae", "overfit", "both"], default="both")
    ap.add_argument("--run_dir", default="/flash/project_465002934/Marigold_output/fm_refiner-D-lumi-0")
    ap.add_argument("--ckpt", default=None, help="default: <run_dir>/checkpoint/best.pth")
    ap.add_argument("--out_dir", default="/flash/project_465002934/Marigold_output/_diag_dit_lumi0")
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--n_overfit", type=int, default=32)
    ap.add_argument("--n_heldout", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.ckpt = args.ckpt or os.path.join(args.run_dir, "checkpoint", "best.pth")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda")
    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    data = load_samples(cfg, args.n_samples, device, args.workers,
                        os.path.join(args.out_dir, f"samples_{args.n_samples}.pt"))

    if args.test in ("vae", "both"):
        res = run_vae_test(cfg, data, device, args, args.out_dir)
        with open(os.path.join(args.out_dir, "vae_noise_test.json"), "w") as f:
            json.dump(res, f, indent=1)
        print("==== VAE / noise test ====\n" + json.dumps(res, indent=1), flush=True)

    if args.test in ("overfit", "both"):
        logs = {}
        sig = float(cfg.trainer.noise_sigma)
        for tag, ns, pd in (("A_sigma0_nopsdrop", 0.0, 0.0), (f"B_sigma{sig:g}_psdrop0.3_recipe", sig, 0.3)):
            logs[tag] = run_overfit(cfg, data, device, args, args.out_dir, tag, ns, pd)
            with open(os.path.join(args.out_dir, "overfit_test.json"), "w") as f:
                json.dump(logs, f, indent=1)
        plot_loss(logs, args.out_dir)
        print("==== overfit done ====", flush=True)


if __name__ == "__main__":
    main()
