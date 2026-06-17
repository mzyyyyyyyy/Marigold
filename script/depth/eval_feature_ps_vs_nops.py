"""
Analyze the effect of PlanetScope (PS) conditioning on intermediate features.

For each of the first 20 validation samples, runs two forward passes through the
trained FMRefiner at t=0:
  - With PS:        ControlNet input = [landsat | ps]
  - Without PS:     ControlNet input = [landsat | null_token]

Hooks capture the output of every UNet block (down / mid / up) and every
ControlNet down block.  Per layer we compute:
  - L2 norm of the feature difference  (||f_ps - f_nops||)
  - Relative difference                (||diff|| / ||f_nops|| + eps)
  - Per-channel mean absolute difference (spatial average)

Outputs (saved to --output_dir):
  feature_diff_heatmap.png  – bar chart: relative L2 diff per layer, averaged over 20 samples
  layer_XX_sample_YY.png    – spatial heatmaps for the N_VIS largest-diff layers

Usage:
  python script/depth/eval_feature_ps_vs_nops.py \
      --config config/fm_refiner_v4.yaml \
      --checkpoint output/fm_refiner_v4/checkpoint/best.pth \
      --output_dir output/feature_analysis \
      [--n_samples 20] [--t_val 0.0] [--n_vis_layers 5]
"""

import argparse
import copy
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from depthfm.fm_refiner import FMRefiner, build_fm_refiner, load_dav2, run_dav2
from src.util.config_util import recursive_load_config
from src.util.ps_lazydataset import LazyPatchDataset


# ---------------------------------------------------------------------------
# Dataset helpers (mirror train_fm_refiner.py)
# ---------------------------------------------------------------------------

def _split_coords(all_coords, train_split, val_split, seed):
    n = len(all_coords)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()
    train_size = int(train_split * n)
    val_size = int(val_split * n)
    return (
        [all_coords[i] for i in indices[:train_size]],
        [all_coords[i] for i in indices[train_size:train_size + val_size]],
        [all_coords[i] for i in indices[train_size + val_size:]],
    )


def _make_dataset(base_dataset, coords, mode, batch_size):
    dataset = copy.copy(base_dataset)
    dataset.batch_size = batch_size
    dataset.mode = mode
    dataset.patch_coords = coords
    dataset.patches_by_size = {}
    for patch_size in dataset.patch_sizes:
        dataset.patches_by_size[patch_size] = [
            p for p in coords
            if p['output_patch_height'] == patch_size[0]
            and p['output_patch_width'] == patch_size[1]
        ]
    return dataset


def _normalize_coarse(h_coarse: torch.Tensor, target_stats: dict) -> torch.Tensor:
    p1  = float(target_stats["p1"][0])
    p99 = float(target_stats["p99"][0])
    h_norm = (h_coarse - p1) / (p99 - p1 + 1e-8)
    return (h_norm * 2.0 - 1.0).clamp(-1.0, 1.0)


def _load_target_stats(stats_file: str, year: int) -> dict:
    with open(stats_file) as f:
        return json.load(f)[str(year)]


# ---------------------------------------------------------------------------
# Hook-based feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Register forward hooks on UNet and ControlNet blocks."""

    def __init__(self, unet, controlnet):
        self._hooks = []
        self.features: dict[str, torch.Tensor] = {}
        self._register(unet, controlnet)

    def _register(self, unet, controlnet):
        # ControlNet down blocks
        for i, blk in enumerate(controlnet.down_blocks):
            name = f"cn_down_{i}"
            self._hooks.append(
                blk.register_forward_hook(self._make_hook(name))
            )
        # ControlNet mid block
        if hasattr(controlnet, "mid_block") and controlnet.mid_block is not None:
            self._hooks.append(
                controlnet.mid_block.register_forward_hook(self._make_hook("cn_mid"))
            )
        # UNet down blocks
        for i, blk in enumerate(unet.down_blocks):
            name = f"unet_down_{i}"
            self._hooks.append(
                blk.register_forward_hook(self._make_hook(name))
            )
        # UNet mid block
        if hasattr(unet, "mid_block") and unet.mid_block is not None:
            self._hooks.append(
                unet.mid_block.register_forward_hook(self._make_hook("unet_mid"))
            )
        # UNet up blocks
        for i, blk in enumerate(unet.up_blocks):
            name = f"unet_up_{i}"
            self._hooks.append(
                blk.register_forward_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # output may be a tuple (hidden_states, ...) – take first tensor
            if isinstance(output, (tuple, list)):
                feat = output[0]
            else:
                feat = output
            if isinstance(feat, torch.Tensor):
                self.features[name] = feat.detach().cpu()
        return hook

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Single forward pass returning feature snapshot
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_with_features(
    fm_refiner: FMRefiner,
    z_t: torch.Tensor,
    t_int: torch.Tensor,
    control_input: torch.Tensor,
    text_emb: torch.Tensor,
    extractor: FeatureExtractor,
) -> dict[str, torch.Tensor]:
    extractor.clear()

    cn_out = fm_refiner.controlnet(
        sample=z_t,
        timestep=t_int,
        encoder_hidden_states=text_emb,
        controlnet_cond=control_input,
        return_dict=True,
    )
    fm_refiner.unet(
        sample=z_t,
        timestep=t_int,
        encoder_hidden_states=text_emb,
        down_block_additional_residuals=cn_out.down_block_res_samples,
        mid_block_additional_residual=cn_out.mid_block_res_sample,
        return_dict=True,
    )
    return dict(extractor.features)


# ---------------------------------------------------------------------------
# Per-layer statistics
# ---------------------------------------------------------------------------

def layer_stats(f_ps: torch.Tensor, f_nops: torch.Tensor) -> dict:
    """Compute diff statistics between two feature tensors (same shape)."""
    diff = f_ps - f_nops
    l2_diff    = diff.norm().item()
    l2_nops    = f_nops.norm().item()
    rel_diff   = l2_diff / (l2_nops + 1e-8)
    # Spatial mean-absolute diff per channel: (C,)
    if diff.dim() == 4:                        # (B, C, H, W)
        mad_per_channel = diff.abs().mean(dim=(0, 2, 3))  # (C,)
        # Spatial map of mean-abs-diff across channels: (H, W)
        spatial_mad = diff.abs().mean(dim=(0, 1))         # (H, W)
    else:
        mad_per_channel = diff.abs().mean(dim=0).flatten()
        spatial_mad = None
    return {
        "l2_diff":          l2_diff,
        "l2_nops":          l2_nops,
        "rel_diff":         rel_diff,
        "mad_per_channel":  mad_per_channel.numpy(),
        "spatial_mad":      spatial_mad.numpy() if spatial_mad is not None else None,
    }


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = recursive_load_config(args.config)
    cfg_data = cfg.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---- Dataset ----
    base_dataset = LazyPatchDataset({
        'input_dir':             cfg_data.input_dir,
        'input_dir_hr':          cfg_data.input_dir_hr,
        'output_dir':            cfg_data.output_dir,
        'selected_bands':        cfg_data.selected_bands,
        'selected_bands_hr':     cfg_data.selected_bands_hr,
        'file_type_input':       cfg_data.file_type_input,
        'file_type_output':      cfg_data.file_type_output,
        'patch_sizes':           [(s, s) for s in cfg_data.patch_sizes],
        'patch_coord_path':      cfg_data.patch_coord_path,
        'num_patches_per_tile':  cfg_data.num_patches_per_tile,
        'tile_emphasis':         cfg_data.tile_emphasis,
        'correlation_cleaning':  cfg_data.correlation_cleaning,
        'target_range_edges':    cfg_data.target_range_edges,
        'num_patches_per_target_range': cfg_data.num_patches_per_target_range,
        'selected_percentile':   cfg_data.selected_percentile,
        'year':                  cfg_data.year,
        'batch_size':            1,
        'mode':                  'val',
        'use_input_minmax':      cfg_data.use_input_minmax,
        'use_input_norm':        cfg_data.use_input_norm,
        'input_stats_file':      cfg_data.input_stats_file,
        'use_input_hr_minmax':   cfg_data.use_input_hr_minmax,
        'use_input_hr_norm':     cfg_data.use_input_hr_norm,
        'input_hr_stats_file':   cfg_data.input_hr_stats_file,
        'use_target_minmax':     cfg_data.use_target_minmax,
        'use_target_norm':       cfg_data.use_target_norm,
        'target_stats_file':     cfg_data.target_stats_file,
        'unit_scale_ratio':      cfg_data.unit_scale_ratio,
        'scale_input_to_neg1_1':    cfg_data.get('scale_input_to_neg1_1', False),
        'scale_input_hr_to_neg1_1': cfg_data.get('scale_input_hr_to_neg1_1', False),
        'scale_target_to_neg1_1':   cfg_data.get('scale_target_to_neg1_1', False),
    })

    all_coords = base_dataset.patch_coords
    _, val_coords, _ = _split_coords(
        all_coords, cfg_data.train_split, cfg_data.val_split, cfg_data.split_seed
    )
    val_dataset = _make_dataset(base_dataset, val_coords, 'val', 1)
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # ---- Model ----
    n_ls  = len(cfg_data.selected_bands)
    n_ps  = len(cfg_data.selected_bands_hr)

    fm_refiner = build_fm_refiner(
        sd_pretrained_path=cfg.model.sd_pretrained_path,
        n_landsat_bands=n_ls,
        n_ps_bands=n_ps,
        ps_dropout_p=cfg.trainer.ps_dropout_p,
    ).to(device)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    fm_refiner.unet.load_state_dict(ckpt["unet_state"])
    fm_refiner.controlnet.load_state_dict(ckpt["controlnet_state"])
    if ckpt.get("null_ps_state") is not None:
        fm_refiner._null_ps = ckpt["null_ps_state"]
    fm_refiner.eval()
    logging.info(f"Loaded checkpoint: {args.checkpoint}")

    # ---- DAv2 ----
    dav2_model = load_dav2(
        cfg.model.dav2_pretrained_path,
        backbone=cfg.model.dav2_backbone,
        out_in_scale_factor=cfg.model.dav2_out_in_scale_factor,
    ).to(device)
    target_stats = _load_target_stats(cfg_data.target_stats_file, cfg_data.year)

    # ---- Hooks ----
    extractor = FeatureExtractor(fm_refiner.unet, fm_refiner.controlnet)

    # ---- Accumulate per-layer stats over N samples ----
    # layer_name -> list of rel_diff values (one per sample)
    accum: dict[str, list] = {}
    # For spatial visualisation keep (f_ps, f_nops) of the first sample per layer
    first_sample_feats: dict[str, tuple] = {}

    n_done = 0
    for batch in tqdm(val_loader, desc="Samples"):
        if n_done >= args.n_samples:
            break

        inputs_lr, inputs_hr, targets = batch
        if inputs_lr.dim() == 5:  inputs_lr  = inputs_lr.squeeze(0)
        if inputs_hr.dim() == 5:  inputs_hr  = inputs_hr.squeeze(0)
        if targets.dim()   == 5:  targets    = targets.squeeze(0)

        inputs_lr  = inputs_lr.to(device)
        inputs_hr  = inputs_hr.to(device)
        targets    = targets.to(device)

        H_hr, W_hr  = targets.shape[2], targets.shape[3]
        c_ps        = inputs_hr.shape[1]

        landsat     = inputs_lr[:, :n_ls]
        landsat_hr  = F.interpolate(landsat, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

        # DAv2 coarse
        with torch.no_grad():
            landsat01 = (landsat + 1.0) / 2.0
            h_coarse  = run_dav2(dav2_model, landsat01, target_size=(H_hr, W_hr))
            h_coarse  = _normalize_coarse(h_coarse, target_stats)
            z_coarse  = fm_refiner.encode(h_coarse)

        # Fixed t = args.t_val
        # z_t = (1-t)*z_coarse + t*z_gt; for t>0 we need z_gt from targets
        B = z_coarse.shape[0]
        t_val = args.t_val
        t_int = torch.full((B,), int(t_val * 999), device=device, dtype=torch.long)
        if t_val > 0.0:
            z_gt = fm_refiner.encode(targets)
            t4   = torch.full((B, 1, 1, 1), t_val, device=device, dtype=z_coarse.dtype)
            z_t  = (1.0 - t4) * z_coarse + t4 * z_gt
        else:
            z_t = z_coarse
        text_emb = fm_refiner.empty_text_embed.to(device).expand(B, -1, -1)

        # Null PS token
        null_ps = fm_refiner._get_null_ps(c_ps, device, z_coarse.dtype)
        null_ps_spatial = null_ps.expand(B, -1, H_hr, W_hr)

        # --- With PS ---
        ctrl_with_ps    = torch.cat([landsat_hr, inputs_hr], dim=1)
        feats_with_ps   = forward_with_features(
            fm_refiner, z_t, t_int, ctrl_with_ps, text_emb, extractor
        )

        # --- Without PS ---
        ctrl_no_ps      = torch.cat([landsat_hr, null_ps_spatial], dim=1)
        feats_no_ps     = forward_with_features(
            fm_refiner, z_t, t_int, ctrl_no_ps, text_emb, extractor
        )

        # Accumulate stats
        for layer_name in feats_with_ps:
            if layer_name not in feats_no_ps:
                continue
            f_ps   = feats_with_ps[layer_name]
            f_nops = feats_no_ps[layer_name]
            if f_ps.shape != f_nops.shape:
                continue
            stats = layer_stats(f_ps, f_nops)
            accum.setdefault(layer_name, []).append(stats["rel_diff"])
            if n_done == 0:
                first_sample_feats[layer_name] = (f_ps, f_nops, stats["spatial_mad"])

        n_done += 1

    extractor.remove()

    # ---- Summary statistics ----
    layer_names  = sorted(accum.keys())
    mean_rel     = np.array([np.mean(accum[n]) for n in layer_names])
    std_rel      = np.array([np.std(accum[n])  for n in layer_names])

    # Save CSV
    csv_path = os.path.join(args.output_dir, "layer_rel_diff.csv")
    with open(csv_path, "w") as f:
        f.write("layer,mean_rel_diff,std_rel_diff\n")
        for name, mu, sig in zip(layer_names, mean_rel, std_rel):
            f.write(f"{name},{mu:.6f},{sig:.6f}\n")
    logging.info(f"Saved CSV: {csv_path}")

    # ---- Bar chart: relative diff per layer ----
    fig, ax = plt.subplots(figsize=(max(12, len(layer_names) * 0.6), 5))
    colors = ["steelblue" if "cn_" in n else "coral" for n in layer_names]
    bars   = ax.bar(range(len(layer_names)), mean_rel, color=colors, alpha=0.85)
    ax.errorbar(range(len(layer_names)), mean_rel, yerr=std_rel,
                fmt="none", color="black", capsize=3, linewidth=1)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative L2 diff  ||f_ps - f_nops|| / ||f_nops||")
    ax.set_title(f"PS vs. No-PS feature difference per layer\n"
                 f"(mean ± std over {n_done} val samples, t={args.t_val})\n"
                 "Blue = ControlNet blocks, Coral = UNet blocks")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(args.output_dir, "feature_diff_heatmap.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved bar chart: {bar_path}")

    # ---- Spatial heatmaps for all layers (sorted by rel_diff descending) ----
    top_names = [layer_names[i] for i in np.argsort(mean_rel)[::-1]]

    for layer_name in top_names:
        if layer_name not in first_sample_feats:
            continue
        f_ps, f_nops, spatial_mad = first_sample_feats[layer_name]

        if spatial_mad is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        def _feat_mean(t):
            if t.dim() == 4:
                return t.squeeze(0).abs().mean(0).numpy()
            return t.abs().numpy()

        im0 = axes[0].imshow(_feat_mean(f_ps),   cmap="viridis")
        axes[0].set_title("With PS\n(mean|feat| across channels)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(_feat_mean(f_nops),  cmap="viridis")
        axes[1].set_title("Without PS\n(mean|feat| across channels)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(spatial_mad, cmap="hot")
        axes[2].set_title("|f_ps - f_nops|\n(mean across channels)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        for ax in axes:
            ax.axis("off")

        fig.suptitle(f"Layer: {layer_name}  |  rel_diff={mean_rel[layer_names.index(layer_name)]:.4f}",
                     fontsize=11)
        plt.tight_layout()
        sp_path = os.path.join(args.output_dir, f"spatial_{layer_name}.png")
        fig.savefig(sp_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved spatial heatmap: {sp_path}")

    # ---- Channel-wise MAD for all layers (first sample) ----
    for layer_name in top_names:
        if layer_name not in first_sample_feats:
            continue
        f_ps, f_nops, _ = first_sample_feats[layer_name]
        diff = f_ps - f_nops
        if diff.dim() == 4:
            mad_ch = diff.abs().mean(dim=(0, 2, 3)).numpy()   # (C,)
        else:
            mad_ch = diff.abs().mean(dim=0).flatten().numpy()

        fig, ax = plt.subplots(figsize=(max(8, len(mad_ch) * 0.05), 4))
        ax.bar(range(len(mad_ch)), mad_ch, color="teal", alpha=0.7)
        ax.set_xlabel("Channel index")
        ax.set_ylabel("Mean absolute difference")
        ax.set_title(f"Channel-wise MAD: {layer_name}")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        ch_path = os.path.join(args.output_dir, f"channel_mad_{layer_name}.png")
        fig.savefig(ch_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved channel MAD: {ch_path}")

    logging.info(f"\nAll layers ranked by PS sensitivity:")
    for rank, name in enumerate(top_names, 1):
        idx = layer_names.index(name)
        logging.info(f"  {rank:2d}. {name:25s}  rel_diff={mean_rel[idx]:.4f} ± {std_rel[idx]:.4f}")

    logging.info(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PS vs. No-PS feature analysis")
    parser.add_argument("--config",      type=str, default="output/fm_refiner/config.yaml",
                        help="Path to fm_refiner config yaml")
    parser.add_argument("--checkpoint",  type=str, default="output/fm_refiner/checkpoint/best.pth",
                        help="Path to trained .pth checkpoint (best.pth or latest.pth)")
    parser.add_argument("--output_dir",  type=str, default="output/fm_refiner/feature_analysis_ps_vs_nops",
                        help="Directory for output figures and CSV")
    parser.add_argument("--n_samples",   type=int, default=20,
                        help="Number of validation samples to analyze")
    parser.add_argument("--t_val",       type=float, default=0.0,
                        help="Flow-matching timestep t in [0,1] for the forward pass")
    args = parser.parse_args()
    run_analysis(args)
