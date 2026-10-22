"""
Animated GIFs (one frame per year) of one time-series test patch: the Landsat
input and each model's predicted height, from the infer_time_series.py output.

  <out_dir>/<id>_landsat.gif                 Landsat RGB (bands dataset.selected_bands), one
                                             2-98 % stretch shared by all years, so brightness
                                             changes between frames are real; nearest x4 to 480 px
  <out_dir>/<id>_<model>.gif                 predicted height, plasma over [0, vmax] m (the
                                             colour scale of eval_group_meeting.py)
All three have the same frame size and timing, so they can be shown side by side.

  singularity exec --bind /flash/project_465002934:/flash/project_465002934 \
      /flash/project_465002934/env/marigold_env.sif \
      python script/depth/make_time_gif.py --idx 17
"""

import argparse
import csv
import io
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

MODELS = {"sr_fm_refiner_v11-lumi": "sr_fm_refiner_v11-lumi", "baseline_sr4ir_reg": "baseline_sr4ir_reg"}


def render(img, title, cmap=None, vmax=None, cbar_label=None, figsize=(6.0, 5.4), aspect="equal"):
    """One frame: image + title, with a colour bar slot (empty for RGB) so all GIFs match in size."""
    fig, (ax, cax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 0.045]})
    if cmap is None:
        ax.imshow(img, interpolation="nearest", aspect=aspect)
        cax.axis("off")
    else:
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect=aspect)
        fig.colorbar(im, cax=cax, label=cbar_label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)
    # fixed margins (not tight_layout): the image box is then identical with and without a
    # colour bar, so RGB and height GIFs line up when shown side by side
    fig.subplots_adjust(left=0.02, right=0.87, bottom=0.02, top=0.92, wspace=0.04)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(path, frames, duration_ms, hold_last_ms):
    durations = [duration_ms] * (len(frames) - 1) + [hold_last_ms]
    # one shared palette for all frames, so colours do not flicker between years
    pal = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True, help="test-list index of the patch (e.g. 17)")
    ap.add_argument("--in_dir", default="/flash/project_465002934/result_time_lumi")
    ap.add_argument("--out_dir", default=None, help="default: <in_dir>/gif")
    ap.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2], help="Landsat bands shown as R G B")
    ap.add_argument("--vmax", type=float, default=28.0)
    ap.add_argument("--duration", type=int, default=700, help="ms per frame")
    ap.add_argument("--hold_last", type=int, default=1500, help="ms on the last frame")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.in_dir, "gif")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.in_dir, "meta.json")) as f:
        meta = json.load(f)
    years = meta["years"]
    with open(meta["split"], "rb") as f:
        e = pickle.load(f)[args.idx]
    with open(os.path.join(args.in_dir, "index.csv")) as f:
        pid = next(r["id"] for r in csv.DictReader(f) if int(r["test_idx"]) == args.idx)

    # Landsat RGB, one shared stretch over all years
    ls = []
    for yr in years:
        dy, dx = e["input_offsets"][yr]
        with rasterio.open(e["input_paths"][yr]) as src:
            ls.append(src.read([b + 1 for b in args.bands], window=Window(
                e["input_crop_x"] + dx, e["input_crop_y"] + dy,
                e["input_patch_width"], e["input_patch_height"])).astype(np.float32))
    ls = np.stack(ls)                                           # (years, 3, h, w)
    lo, hi = np.percentile(ls, 2), np.percentile(ls, 98)
    scale = e["output_patch_height"] // e["input_patch_height"]
    frames = []
    for yr, a in zip(years, ls):
        rgb = ((a - lo) / (hi - lo)).clip(0, 1).transpose(1, 2, 0)
        rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)   # nearest x4: same grid as the predictions
        frames.append(render(rgb, f"Landsat  {yr}"))
    path = os.path.join(out_dir, f"{pid}_landsat.gif")
    save_gif(path, frames, args.duration, args.hold_last)
    print("wrote", path)

    for m, label in MODELS.items():
        frames = []
        for yr in years:
            with rasterio.open(os.path.join(args.in_dir, m, str(yr), pid + ".tif")) as src:
                h = src.read(1)
            frames.append(render(np.clip(h, 0, args.vmax), f"{label}  {yr}", cmap="plasma", vmax=args.vmax,
                                 cbar_label="height (m)"))
        path = os.path.join(out_dir, f"{pid}_{m}.gif")
        save_gif(path, frames, args.duration, args.hold_last)
        print("wrote", path)
