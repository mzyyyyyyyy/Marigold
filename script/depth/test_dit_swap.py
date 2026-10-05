"""
Sanity checks for the DiT variants (fm_refiner-R-lumi-11 / -12 / -13), no real data needed.

  --backbone dit         -11: DiT + DiT-ControlNet analogue (zero-init control branch)
  --backbone dit_concat  -12: condition encoded to latent resolution, concatenated along CHANNELS
  --backbone dit_token   -13: condition encoded to tokens, concatenated along the SEQUENCE

  1. interface: the variant and a random-init UNet, built through the same build_fm_refiner(), give the
     same shapes through FMRefiner.forward / refine; at init the output is exactly 0 (zero-init output
     layer) so the loss is ~E|v|^2; for -11 the control residuals are exact no-ops as well.
  2. gradients: after a few steps the condition path and the backbone get non-zero gradients.
  3. overfit one fixed synthetic batch (noise_sigma=0 -> deterministic target): loss must drop by >95%,
     the predicted velocity must point the right way (cosine with the target > 0.9) and refine() from
     the coarse latent must end up closer to the ground truth than the coarse map.
  4. the condition is actually USED: a fresh model is trained on a batch whose samples share ONE coarse
     map but have different ground truths (and Landsat/PS carrying them). At t=0 the input z_t is then
     identical across samples, so only the condition can tell them apart: cosine(v_pred, v_true) must
     be high with the true condition and clearly lower when the conditions are shuffled between samples.

  singularity exec ... python script/depth/test_dit_swap.py --backbone dit_token [--tiny] [--steps 300]
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from depthfm.fm_refiner import build_fm_refiner  # noqa: E402

SD = "/flash/project_465002934/model/stable-diffusion-2"


def synth_batch(B, size, device, g):
    """smooth random height maps in [-1,1]; coarse = blurred + biased gt; landsat/ps = noisy functions of gt."""
    raw = torch.randn(B, 1, size // 8, size // 8, device=device, generator=g)
    gt = F.interpolate(raw, size=(size, size), mode="bicubic", align_corners=False).tanh()
    coarse = (F.avg_pool2d(gt, 9, 1, 4) * 0.8 - 0.1).clamp(-1, 1)
    landsat = (gt.repeat(1, 3, 1, 1) * 0.5 + 0.1 * torch.randn(B, 3, size, size, device=device, generator=g)).clamp(-1, 1)
    ps = (gt.repeat(1, 3, 1, 1) + 0.05 * torch.randn(B, 3, size, size, device=device, generator=g)).clamp(-1, 1)
    return landsat, ps, coarse, gt


def build(backbone, tiny, noise_sigma, device):
    dit_kwargs = dict(hidden_size=128, depth=4, num_heads=4, n_control=2) if tiny else {}
    m = build_fm_refiner(
        sd_pretrained_path=SD, n_landsat_bands=3, n_ps_bands=3, ps_dropout_p=0.0, use_controlnet=True,
        controlnet_cond_mode="landsat_ps", noise_sigma=noise_sigma, sample_sigma=0.0,
        sample_noise_mode="init_only", pretrained_unet=False, backbone=backbone, dit_kwargs=dit_kwargs)
    return m.to(device)


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def velocity(m, zt, t_int, cond):
    cn = m.controlnet(sample=zt, timestep=t_int, encoder_hidden_states=None, controlnet_cond=cond)
    return m.unet(sample=zt, timestep=t_int, encoder_hidden_states=None,
                  down_block_additional_residuals=cn.down_block_res_samples,
                  mid_block_additional_residual=cn.mid_block_res_sample).sample


def overfit(m, landsat, ps, coarse, gt, steps, tiny, tag):
    m.train()
    torch.manual_seed(2)
    opt = torch.optim.AdamW(
        [{"params": m.unet.parameters(), "lr": 4e-5 * (5 if tiny else 1)},
         {"params": m.controlnet.parameters(), "lr": 4e-4 * (2 if tiny else 1)}], weight_decay=0.0)
    for gp in opt.param_groups:
        gp["base_lr"] = gp["lr"]
    t0, hist = time.time(), []
    for it in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = gp["base_lr"] * min(1.0, (it + 1) / 50)
        opt.zero_grad()
        loss = m(landsat, ps, coarse, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(m.unet.parameters()) + list(m.controlnet.parameters()), 1.0)
        opt.step()
        hist.append(loss.item())
        if it % max(1, steps // 10) == 0 or it == steps - 1:
            print(f"[{tag}] step {it:4d} loss={loss.item():.5f}  ({time.time() - t0:.0f}s)")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dit", choices=["dit", "dit_concat", "dit_token"])
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_unet", action="store_true")
    ap.add_argument("--skip_cond_test", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(0)
    B = 2 if a.tiny else 4
    landsat, ps, coarse, gt = synth_batch(B, a.size, dev, g)
    ok_all = True

    # ---------------- 1. interface / init ----------------
    models = {a.backbone: build(a.backbone, a.tiny, 0.0, dev)}
    if not a.skip_unet:
        models["unet"] = build("unet", a.tiny, 0.0, dev)
    for name, m in models.items():
        m.train()
        with torch.no_grad():
            torch.manual_seed(1)
            loss = m(landsat, ps, coarse, gt).item()
            ref = m.refine(landsat, coarse, n_steps=2)
            z_c, z_g = m.encode(coarse), m.encode(gt)
            e_v2 = ((z_g - z_c) ** 2).mean().item()
        print(f"[1] {name:10s} params(unet+cn)={n_params(m.unet) + n_params(m.controlnet):8.1f}M  "
              f"(backbone {n_params(m.unet):7.1f}M)  init loss={loss:.4f} (E|v|^2={e_v2:.4f})  "
              f"refine out={tuple(ref.shape)} gt={tuple(gt.shape)}")
        assert ref.shape == gt.shape
    m = models[a.backbone]
    with torch.no_grad():
        t_int = torch.tensor([100, 800] * (B // 2) or [100], device=dev)[:B]
        z = m.encode(coarse)
        cond = torch.cat([landsat, ps], 1)
        cn = m.controlnet(sample=z, timestep=t_int, encoder_hidden_states=None, controlnet_cond=cond)
        out = velocity(m, z, t_int, cond)
        shapes = [tuple(r.shape) for r in cn.down_block_res_samples][:2]
        print(f"[1] {a.backbone}: condition output shapes {shapes}{'...' if len(cn.down_block_res_samples) > 2 else ''}; "
              f"backbone out {tuple(out.shape)} max|v|={out.abs().max().item():.2e}")
        assert out.abs().max().item() == 0.0, "output layer must start as an exact no-op"
        if a.backbone == "dit":
            mx = max(r.abs().max().item() for r in cn.down_block_res_samples)
            assert mx == 0.0, "control residuals must start as exact no-ops"

    # ---------------- 2. gradients ----------------
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
    for _ in range(3):
        opt.zero_grad()
        m(landsat, ps, coarse, gt).backward()
        opt.step()
    opt.zero_grad()
    m(landsat, ps, coarse, gt).backward()
    gn_cond = sum(p.grad.norm() ** 2 for p in m.controlnet.parameters() if p.grad is not None).sqrt().item()
    gn_blk = sum(p.grad.norm() ** 2 for p in m.unet.blocks.parameters() if p.grad is not None).sqrt().item()
    print(f"[2] grad norms after 3 steps: condition path={gn_cond:.3e} backbone blocks={gn_blk:.3e}")
    assert gn_cond > 0 and gn_blk > 0

    # ---------------- 3. overfit one batch ----------------
    m = build(a.backbone, a.tiny, 0.0, dev)      # fresh model
    hist = overfit(m, landsat, ps, coarse, gt, a.steps, a.tiny, "3")
    first, tail = hist[0], sum(hist[-10:]) / 10
    print(f"[3] loss {first:.4f} -> {tail:.4f} (x{tail / first:.3f})")
    m.eval()
    with torch.no_grad():
        z_c, z_g = m.encode(coarse), m.encode(gt)
        cond = torch.cat([landsat, ps], 1)
        v_true = z_g - z_c
        for tval in (0.0, 0.5):
            t_int = torch.full((B,), int(tval * 999), device=dev)
            zt = (1 - tval) * z_c + tval * z_g
            v = velocity(m, zt, t_int, cond)
            cos = F.cosine_similarity(v.flatten(1), v_true.flatten(1), dim=1).mean().item()
            print(f"[3] t={tval}: cos(v_pred, z_gt - z_coarse)={cos:.3f}  |v_pred|/|v_true|="
                  f"{(v.norm() / v_true.norm()).item():.3f}")
            if tval == 0.0:
                cos0 = cos
        err_c = (coarse - gt).abs().mean().item()
        errs = {n: (m.refine(landsat, coarse, n_steps=n) - gt).abs().mean().item() for n in (1, 4)}
        print(f"[3] mean|h-gt|: coarse={err_c:.4f}  refined 1 step={errs[1]:.4f}  4 steps={errs[4]:.4f}")
    ok3 = tail < 0.05 * first and cos0 > 0.9 and errs[4] < err_c
    print("[3]", "PASS" if ok3 else "FAIL")
    ok_all &= ok3

    # ---------------- 4. the condition is used ----------------
    if not a.skip_cond_test:
        same_coarse = coarse[:1].expand_as(coarse).contiguous()
        m = build(a.backbone, a.tiny, 0.0, dev)
        overfit(m, landsat, ps, same_coarse, gt, a.steps, a.tiny, "4")
        m.eval()
        with torch.no_grad():
            z_c, z_g = m.encode(same_coarse), m.encode(gt)
            v_true = z_g - z_c
            t_int = torch.zeros(B, dtype=torch.long, device=dev)
            cond = torch.cat([landsat, ps], 1)
            cos_true = F.cosine_similarity(velocity(m, z_c, t_int, cond).flatten(1), v_true.flatten(1), dim=1).mean().item()
            cond_sh = torch.roll(cond, 1, dims=0)
            cos_sh = F.cosine_similarity(velocity(m, z_c, t_int, cond_sh).flatten(1), v_true.flatten(1), dim=1).mean().item()
        ok4 = cos_true > 0.8 and cos_true - cos_sh > 0.3
        print(f"[4] t=0 (z_t identical across samples): cos with TRUE cond={cos_true:.3f}  SHUFFLED cond={cos_sh:.3f}  "
              f"-> {'PASS' if ok4 else 'FAIL'}")
        ok_all &= ok4

    print(f"RESULT[{a.backbone}]:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
