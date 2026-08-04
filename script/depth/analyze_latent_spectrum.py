"""
Latent-space spectral analysis: zc vs z0.

Goal: determine whether the FM model fails to learn high-frequency velocity
fields, or whether the latent space (VAE) simply suppresses high-frequency
signal in the first place.

We compute, over a batch of validation samples:
  - FFT of (z_gt - z_coarse)   -- "target velocity" in latent space
  - FFT of v_pred @ t=0        -- model's predicted velocity (null PS, t=0)
  - FFT of z_gt                 -- high-freq content of GT latent alone
  - FFT of z_coarse             -- high-freq content of coarse latent alone
  - FFT of (h_gt - h_coarse)   -- same difference but in *pixel* space

For each, we plot the radially-averaged power spectral density (PSD) and
save mean/std to a .npz for further analysis.

Usage:
  python script/depth/analyze_latent_spectrum.py \
      --config config/sr_fm_refiner_v4.yaml \
      --checkpoint output/sr_fm_refiner_v4/checkpoint/latest.pth \
      --n_samples 200 \
      --output_dir output/spectrum_analysis
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from depthfm.chmv2 import load_chmv2, run_chmv2
from depthfm.sr_module import build_sr_module
from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def radial_psd(x: torch.Tensor) -> np.ndarray:
    """
    Compute radially averaged PSD for a single 2-D map.

    Args:
        x: (H, W) float tensor
    Returns:
        1-D np.ndarray of length min(H,W)//2, power indexed by radial frequency bin
    """
    x_np = x.float().cpu().numpy()
    H, W = x_np.shape
    fft = np.fft.fft2(x_np)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    # Build radial frequency grid
    cy, cx = H // 2, W // 2
    y_idx = np.arange(H) - cy
    x_idx = np.arange(W) - cx
    xg, yg = np.meshgrid(x_idx, y_idx)
    r = np.sqrt(xg**2 + yg**2).astype(int)

    n_bins = min(cy, cx)
    psd = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = r == b
        counts[b] = mask.sum()
        psd[b] = power[mask].mean() if counts[b] > 0 else 0.0

    return psd


def batch_psd(maps: torch.Tensor) -> np.ndarray:
    """
    Radially averaged PSD over a batch of 2-D maps.

    Args:
        maps: (N, H, W) tensor
    Returns:
        (N, n_bins) np.ndarray
    """
    results = []
    for i in range(maps.shape[0]):
        results.append(radial_psd(maps[i]))
    # Truncate to shortest (in case shapes differ)
    min_len = min(r.shape[0] for r in results)
    return np.stack([r[:min_len] for r in results], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sr_fm_refiner_v4.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (loads VAE weights). "
                             "If omitted, uses pretrained VAE from config.")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of patches to analyse.")
    parser.add_argument("--output_dir", type=str, default="output/spectrum_analysis")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = recursive_load_config(args.config)
    device = torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"device = {device}")

    # ---- Build models (VAE only strictly needed, but we need encode()) ----
    logging.info("Building FM Refiner (VAE) …")
    n_ls_bands = len(cfg.dataset.selected_bands)
    n_ps_bands = len(cfg.dataset.selected_bands_hr)
    fm_refiner = build_fm_refiner(
        cfg.model.sd_pretrained_path,
        n_landsat_bands=n_ls_bands,
        n_ps_bands=n_ps_bands,
        device=str(device),
    )
    fm_refiner = fm_refiner.to(device)
    fm_refiner.eval()

    # Optionally restore checkpoint (not strictly needed for VAE since it's frozen,
    # but keeps the setup consistent with training)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logging.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        fm_refiner.unet.load_state_dict(ckpt["unet_state"])
        fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
        logging.info("Checkpoint loaded.")

    # ---- Coarse model ----
    coarse_model_type = getattr(cfg.model, "coarse_model", "dav2")
    if coarse_model_type == "chmv2":
        coarse_model, chmv2_mean, chmv2_std = load_chmv2(cfg.model)
        coarse_model = coarse_model.to(device).eval()
    else:
        coarse_model = load_dav2(
            cfg.model.dav2_pretrained_path,
            cfg.model.dav2_backbone,
            getattr(cfg.model, "dav2_out_in_scale_factor", 1),
        )
        coarse_model = coarse_model.to(device).eval()
        chmv2_mean = chmv2_std = None

    # ---- Target stats (for normalising coarse output) ----
    with open(cfg.dataset.target_stats_file) as f:
        all_stats = json.load(f)
    target_stats = all_stats[str(cfg.dataset.year)]

    def normalize_coarse(h):
        p1 = float(target_stats["p1"][0])
        p99 = float(target_stats["p99"][0])
        h_norm = (h - p1) / (p99 - p1 + 1e-8)
        return (h_norm * 2.0 - 1.0).clamp(-1.0, 1.0)

    # ---- Dataset (val split) ----
    cfg_data = cfg.dataset
    import pickle
    with open(cfg_data.patch_coord_path, "rb") as f:
        all_coords = pickle.load(f)

    n = len(all_coords)
    rng = torch.Generator().manual_seed(cfg_data.get("split_seed", 42))
    indices = torch.randperm(n, generator=rng).tolist()
    train_size = int(cfg_data.train_split * n)
    val_size = int(cfg_data.val_split * n)
    val_coords = [all_coords[i] for i in indices[train_size: train_size + val_size]]
    logging.info(f"Val patches available: {len(val_coords)}")

    base_dataset = LazyPatchDataset({
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
        'tile_emphasis': getattr(cfg_data, 'tile_emphasis', []),
        'min_valid_ratio': cfg_data.min_valid_ratio,
        'target_name': cfg_data.target_name,
        'use_input_minmax': cfg_data.use_input_minmax,
        'use_input_norm': cfg_data.use_input_norm,
        'input_stats_file': cfg_data.input_stats_file,
        'use_input_hr_minmax': cfg_data.use_input_hr_minmax,
        'use_input_hr_norm': cfg_data.use_input_hr_norm,
        'input_hr_stats_file': cfg_data.input_hr_stats_file,
        'use_target_minmax': cfg_data.use_target_minmax,
        'use_target_norm': cfg_data.use_target_norm,
        'target_stats_file': cfg_data.target_stats_file,
        'scale_input_to_neg1_1': cfg_data.scale_input_to_neg1_1,
        'scale_input_hr_to_neg1_1': cfg_data.scale_input_hr_to_neg1_1,
        'scale_target_to_neg1_1': cfg_data.scale_target_to_neg1_1,
        'correlation_cleaning': getattr(cfg_data, 'correlation_cleaning', False),
        'unit_scale_ratio': getattr(cfg_data, 'unit_scale_ratio', 1),
        'year': cfg_data.year,
        'selected_percentile': cfg_data.selected_percentile,
        'target_range_edges': cfg_data.target_range_edges,
        'num_patches_per_target_range': cfg_data.num_patches_per_target_range,
        'batch_size': 1,
        'mode': 'val',
        'min_valid_ratio': cfg_data.min_valid_ratio,
        'target_name': cfg_data.target_name,
    })
    dataset = copy.copy(base_dataset)
    dataset.batch_size = 1
    dataset.mode = "val"
    dataset.patch_coords = val_coords
    dataset.patches_by_size = {}
    for patch_size in dataset.patch_sizes:
        dataset.patches_by_size[patch_size] = [
            p for p in val_coords
            if p['output_patch_height'] == patch_size[0]
            and p['output_patch_width'] == patch_size[1]
        ]

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=None, num_workers=2, shuffle=False)

    # ---- Accumulate PSDs ----
    psd_vel = []        # z_gt - z_coarse  (velocity target)
    psd_vpred = []      # v_pred from model @ t=0, null PS
    psd_z_gt = []       # z_gt
    psd_z_coarse = []   # z_coarse
    psd_pixel_diff = [] # h_gt - h_coarse  (pixel space)
    vis_pixel_diff = [] # saved for spatial visualisation (up to 16 samples)

    fm_refiner.eval()
    collected = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Analysing", total=args.n_samples):
            if collected >= args.n_samples:
                break

            inputs_lr, inputs_hr, targets = batch
            if inputs_lr.dim() == 5:
                inputs_lr = inputs_lr.squeeze(0)
            if inputs_hr.dim() == 5:
                inputs_hr = inputs_hr.squeeze(0)
            if targets.dim() == 5:
                targets = targets.squeeze(0)

            inputs_lr = inputs_lr.to(device)
            inputs_hr = inputs_hr.to(device)
            targets = targets.to(device)

            H_hr, W_hr = targets.shape[2], targets.shape[3]
            landsat = inputs_lr[:, :n_ls_bands]
            landsat_hr = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

            # Coarse prediction
            if coarse_model_type == "chmv2":
                h_coarse = run_chmv2(coarse_model, landsat, (H_hr, W_hr), chmv2_mean, chmv2_std)
            else:
                landsat_01 = (landsat + 1.0) / 2.0
                h_coarse = run_dav2(coarse_model, landsat_01, target_size=(H_hr, W_hr))
            h_coarse = normalize_coarse(h_coarse)  # (B,1,H,W) in [-1,1]

            h_gt = targets  # (B,1,H,W)
            B = h_gt.shape[0]

            # Encode to latent
            z_gt = fm_refiner.encode(h_gt)          # (B,4,h,w)
            z_coarse = fm_refiner.encode(h_coarse)  # (B,4,h,w)
            vel = z_gt - z_coarse                   # velocity target

            # Model prediction at t=0, null PS conditioning
            # z_t = z_coarse when t=0 (straight-path interpolation)
            null_ps = fm_refiner._get_null_ps(n_ps_bands, device, h_gt.dtype)
            null_spatial = null_ps.expand(B, -1, H_hr, W_hr)
            control_input = torch.cat([landsat_hr, null_spatial], dim=1)
            t_int = torch.zeros(B, dtype=torch.long, device=device)
            text_emb = fm_refiner.empty_text_embed.to(device).expand(B, -1, -1)

            cn_out = fm_refiner.controlnet(
                sample=z_coarse,
                timestep=t_int,
                encoder_hidden_states=text_emb,
                controlnet_cond=control_input,
                return_dict=True,
            )
            unet_out = fm_refiner.unet(
                sample=z_coarse,
                timestep=t_int,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=cn_out.down_block_res_samples,
                mid_block_additional_residual=cn_out.mid_block_res_sample,
                return_dict=True,
            )
            v_pred = unet_out.sample  # (B,4,h,w)

            # Per-sample, per-channel PSD
            for b in range(B):
                psd_vel.append(batch_psd(vel[b]))           # (4, n_bins)
                psd_vpred.append(batch_psd(v_pred[b]))      # (4, n_bins)
                psd_z_gt.append(batch_psd(z_gt[b]))         # (4, n_bins)
                psd_z_coarse.append(batch_psd(z_coarse[b])) # (4, n_bins)
                diff_px = (h_gt[b] - h_coarse[b]).squeeze(0)
                psd_pixel_diff.append(radial_psd(diff_px)[np.newaxis, :])
                if len(vis_pixel_diff) < 16:
                    vis_pixel_diff.append({
                        "h_gt":    h_gt[b].squeeze(0).cpu().float().numpy(),
                        "h_coarse":h_coarse[b].squeeze(0).cpu().float().numpy(),
                        "diff":    diff_px.cpu().float().numpy(),
                    })
                collected += 1
                if collected >= args.n_samples:
                    break

    logging.info(f"Collected {collected} samples.")

    # ---- Stack and aggregate ----
    # psd_vel[i]: (4, n_bins)  → stack → (N, 4, n_bins)
    def _stack_truncate(lst):
        min_bins = min(a.shape[-1] for a in lst)
        return np.stack([a[..., :min_bins] for a in lst], axis=0)

    psd_vel_arr = _stack_truncate(psd_vel)            # (N,4,n_bins)
    psd_vpred_arr = _stack_truncate(psd_vpred)        # (N,4,n_bins)
    psd_z_gt_arr = _stack_truncate(psd_z_gt)
    psd_z_coarse_arr = _stack_truncate(psd_z_coarse)
    psd_pixel_diff_arr = _stack_truncate(psd_pixel_diff)  # (N,1,n_bins)

    # Mean across samples and channels
    def _mean_std(arr):
        flat = arr.reshape(-1, arr.shape[-1])
        return flat.mean(0), flat.std(0)

    vel_mean, vel_std = _mean_std(psd_vel_arr)
    vpred_mean, vpred_std = _mean_std(psd_vpred_arr)
    z_gt_mean, z_gt_std = _mean_std(psd_z_gt_arr)
    z_coarse_mean, z_coarse_std = _mean_std(psd_z_coarse_arr)
    px_mean, px_std = _mean_std(psd_pixel_diff_arr)

    # Save raw arrays
    np.savez(
        os.path.join(args.output_dir, "psd_data.npz"),
        psd_vel=psd_vel_arr,
        psd_vpred=psd_vpred_arr,
        psd_z_gt=psd_z_gt_arr,
        psd_z_coarse=psd_z_coarse_arr,
        psd_pixel_diff=psd_pixel_diff_arr,
    )
    logging.info(f"Raw PSD arrays saved to {args.output_dir}/psd_data.npz")

    # ---- Pixel diff visualisation ----
    n_vis = len(vis_pixel_diff)
    fig_vis, axes_vis = plt.subplots(n_vis, 3, figsize=(12, 4 * n_vis))
    if n_vis == 1:
        axes_vis = axes_vis[np.newaxis, :]
    col_titles = ["h_gt", "h_coarse", "h_gt − h_coarse"]
    for row, s in enumerate(vis_pixel_diff):
        for col, (key, title) in enumerate(zip(["h_gt", "h_coarse", "diff"], col_titles)):
            ax = axes_vis[row, col]
            data = s[key]
            if col < 2:
                im = ax.imshow(data, cmap="plasma", vmin=-1.0, vmax=1.0)
            else:
                # Symmetric colormap centred at 0 for the difference
                vabs = max(np.abs(data).max(), 1e-6)
                im = ax.imshow(data, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    vis_path = os.path.join(args.output_dir, "pixel_diff_vis.png")
    plt.savefig(vis_path, dpi=120)
    plt.close(fig_vis)
    logging.info(f"Pixel diff visualisation saved to {vis_path}")

    # ---- Plots ----
    freq_lat = np.arange(vel_mean.shape[0])
    freq_px = np.arange(px_mean.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: latent space --
    ax = axes[0]
    ax.set_title("Latent space PSD (log scale)")
    for arr, label, color, ls in [
        (z_gt_mean,    "z_gt",                        "steelblue",  "-"),
        (z_coarse_mean,"z_coarse",                    "darkorange", "-"),
        (vel_mean,     "v_target = z_gt − z_coarse",  "crimson",    "-"),
        (vpred_mean,   "v_pred (model, t=0, null PS)","forestgreen","--"),
    ]:
        ax.semilogy(freq_lat, arr + 1e-12, label=label, color=color, linestyle=ls)
    ax.set_xlabel("Radial frequency bin")
    ax.set_ylabel("Mean power")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Right: pixel space diff --
    ax = axes[1]
    ax.set_title("Pixel space PSD: h_gt − h_coarse (log scale)")
    ax.semilogy(freq_px, px_mean + 1e-12, color="purple", label="h_gt − h_coarse")
    ax.fill_between(
        freq_px,
        np.maximum(px_mean - px_std, 1e-12),
        px_mean + px_std,
        alpha=0.2, color="purple"
    )
    ax.set_xlabel("Radial frequency bin")
    ax.set_ylabel("Mean power")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "psd_comparison.png")
    plt.savefig(plot_path, dpi=150)
    logging.info(f"Plot saved to {plot_path}")

    # ---- Per-channel breakdown (latent) ----
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 4))
    for ch in range(4):
        ax = axes2[ch]
        ax.set_title(f"Latent ch {ch}")
        for arr_full, label, color, ls in [
            (psd_z_gt_arr,    "z_gt",    "steelblue",   "-"),
            (psd_z_coarse_arr,"z_coarse","darkorange",  "-"),
            (psd_vel_arr,     "v_target","crimson",     "-"),
            (psd_vpred_arr,   "v_pred",  "forestgreen", "--"),
        ]:
            m = arr_full[:, ch, :].mean(0)
            ax.semilogy(m + 1e-12, label=label, color=color, linestyle=ls)
        ax.set_xlabel("Radial freq bin")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_path = os.path.join(args.output_dir, "psd_per_channel.png")
    plt.savefig(plot2_path, dpi=150)
    logging.info(f"Per-channel plot saved to {plot2_path}")

    # ---- Print HF/LF energy ratio ----
    def hf_ratio(mean_psd, cutoff_frac=0.5):
        """Fraction of total energy above cutoff_frac * nyquist."""
        n = len(mean_psd)
        cut = int(n * cutoff_frac)
        lf = mean_psd[:cut].sum()
        hf = mean_psd[cut:].sum()
        return hf / (lf + hf + 1e-12)

    hf_vel   = hf_ratio(vel_mean)
    hf_vpred = hf_ratio(vpred_mean)
    hf_z_gt  = hf_ratio(z_gt_mean)
    hf_zc    = hf_ratio(z_coarse_mean)
    hf_px    = hf_ratio(px_mean)

    print("\n=== High-frequency energy ratio (above 50% Nyquist) ===")
    print(f"  z_gt                    : {hf_z_gt:.4f}")
    print(f"  z_coarse                : {hf_zc:.4f}")
    print(f"  v_target (z_gt - z_c)   : {hf_vel:.4f}")
    print(f"  v_pred   (model, t=0)   : {hf_vpred:.4f}")
    print(f"  pixel diff (h_gt - h_c) : {hf_px:.4f}")

    gap = hf_vel - hf_vpred
    print(f"\n  HF gap (v_target - v_pred): {gap:.4f}  ({gap/hf_vel*100:.1f}% of target HF energy)")
    print()
    print("Interpretation guide:")
    print("  gap ≈ 0     → model predicts high-freq velocity correctly; problem lies elsewhere.")
    print("  gap > 0     → model under-predicts high-freq velocity (spectral bias).")
    print("  gap < 0     → model hallucinates high-freq (over-sharpening).")
    print("  v_target HF ≪ z_gt HF  → velocity itself is smooth; VAE bottleneck unlikely.")


if __name__ == "__main__":
    main()
