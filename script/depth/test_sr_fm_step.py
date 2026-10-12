"""
Functional test (1 GPU, no DDP) of the sr_fm_refiner_d_v0-lumi building blocks in train_sr_fm_refiner.py:
SR step (pixel + TDP), FM step, gradient paths, a few real optimiser steps, validation with pseudo-PS.
Uses the cached samples written by diag_dit_overfit_vae.py. Run via script/depth/test_sr_fm_step.sh.
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from omegaconf import OmegaConf
import train_sr_fm_refiner as S
from depthfm.fm_refiner import build_fm_refiner
from depthfm.sr_module import build_sr_module

dev = torch.device("cuda")
cfg = OmegaConf.load("config/sr_fm_refiner_d_v0-lumi.yaml")
d = torch.load("/flash/project_465002934/Marigold_output/_diag_dit_lumi0/samples_256.pt", weights_only=False)
n_ls, n_ps = 3, 3
fm = build_fm_refiner(sd_pretrained_path=cfg.model.sd_pretrained_path, n_landsat_bands=n_ls, n_ps_bands=n_ps,
                      ps_dropout_p=0.0, use_controlnet=True, controlnet_cond_mode="landsat_ps",
                      noise_sigma=1.0, sample_sigma=1.0, sample_noise_mode="init_only", device="cuda",
                      backbone="dit_omini", dit_kwargs=dict(OmegaConf.to_container(cfg.model.dit))).to(dev)
sr = build_sr_module(OmegaConf.to_container(cfg.sr_module)).to(dev)
st = torch.load(cfg.model.sr_pretrained_path, map_location="cpu")["params"]
print("swinir load:", sr.swinir.load_state_dict(st, strict=False).missing_keys[:3])
fm.train(); sr.train()
opt_sr = torch.optim.AdamW(sr.parameters(), lr=8e-4, weight_decay=0.0)
opt_fm = torch.optim.AdamW(list(fm.unet.parameters()) + list(fm.controlnet.parameters()), lr=1e-4, weight_decay=0.0)

B = 2
def batch(i):
    sl = slice(i * B, (i + 1) * B)
    gt = d["gt"][sl].to(dev); ps = d["ps"][sl].to(dev); hc = d["h_coarse"][sl].to(dev); lh = d["landsat_hr"][sl].to(dev)
    landsat = F.interpolate(lh, size=(48, 48), mode="bilinear", align_corners=False)   # low-res Landsat for SR
    return landsat, lh, ps, hc, gt

for p in list(fm.unet.parameters()) + list(fm.controlnet.parameters()):
    p.requires_grad_(True)
for step in range(6):
    landsat, lh, ps, hc, gt = batch(step)
    use_tdp = step >= 2
    S._freeze(fm.unet); S._freeze(fm.controlnet)
    opt_fm.zero_grad(set_to_none=True)   # so any FM grad seen after the SR step comes from the SR step itself
    opt_sr.zero_grad()
    out = S._sr_step(fm, sr, landsat, lh, ps, hc, gt, 1.0, 1.0, use_tdp, -1, dev)
    out["l_total"].backward()
    g_sr = sum(p.grad.abs().sum().item() for p in sr.parameters() if p.grad is not None)
    g_fm = [p for p in list(fm.unet.parameters()) + list(fm.controlnet.parameters()) if p.grad is not None]
    assert not g_fm, "FM params must not receive grads in the SR step"
    torch.nn.utils.clip_grad_norm_(sr.parameters(), 1.0); opt_sr.step()
    pseudo = out["pseudo_ps"].detach().clone()
    S._unfreeze(fm.unet); S._unfreeze(fm.controlnet)
    opt_fm.zero_grad()
    l_fm = S._fm_step(fm, ps, lh, hc, gt, pseudo, 0.0, 1.0, dev)
    l_fm.backward()
    g_fm = sum(p.grad.abs().sum().item() for p in list(fm.unet.parameters()) + list(fm.controlnet.parameters()) if p.grad is not None)
    n_none = sum(1 for p in list(fm.unet.parameters()) + list(fm.controlnet.parameters()) if p.grad is None)
    torch.nn.utils.clip_grad_norm_(list(fm.unet.parameters()) + list(fm.controlnet.parameters()), 1.0); opt_fm.step()
    print(f"step {step} tdp={use_tdp} l_pix={out['l_pix'].item():.4f} l_tdp={out['l_tdp'].item():.4f} l_fm={l_fm.item():.4f} "
          f"|g_sr|={g_sr:.3e} |g_fm|={g_fm:.3e} fm_params_without_grad={n_none}", flush=True)
    if use_tdp:
        assert out["l_tdp"].item() > 0
        assert not fm.controlnet.cond_grad
# the TDP gradient must reach the SR module even if the pixel loss is off
S._freeze(fm.unet); S._freeze(fm.controlnet); opt_sr.zero_grad(); opt_fm.zero_grad(set_to_none=True)
landsat, lh, ps, hc, gt = batch(7)
out = S._sr_step(fm, sr, landsat, lh, ps, hc, gt, 0.0, 1.0, True, -1, dev)
out["l_total"].backward()
print("TDP-only grad on SR:", sum(p.grad.abs().sum().item() for p in sr.parameters() if p.grad is not None))
S._unfreeze(fm.unet); S._unfreeze(fm.controlnet)
# refine with pseudo-PS
fm.eval(); sr.eval()
with torch.no_grad():
    landsat, lh, ps, hc, gt = batch(8)
    pseudo = sr(landsat, target_size=(240, 240))
    h = fm.refine(lh, hc, n_steps=4, ps_hr=pseudo)
print("refine ok", tuple(h.shape), float(h.min()), float(h.max()))
print("peak mem GB", torch.cuda.max_memory_allocated() / 2**30)
print("ALL OK")
