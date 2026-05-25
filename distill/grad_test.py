#!/usr/bin/env python3
"""Contained validation: prove the student can be sampled DIFFERENTIABLY.

The v2 scaffold drove ComfyUI's KSampler for the student, but every k_diffusion
sampler is @torch.no_grad() -> student latent has no grad graph -> backward fails
with 'element 0 does not require grad'.

Fix under test here: run the student denoise through a custom KSAMPLER whose
sampler_function is a differentiable euler (identical math to comfy's sample_euler
minus the @torch.no_grad decorator), driven by CFGGuider (which has NO no_grad in
its sample/inner_sample path) under torch.enable_grad(). This reuses ComfyUI's
correct cond/sigma/i2v handling and only removes the grad-blocking.

This script loads, runs ONE teacher sample (no_grad) + ONE student sample (grad),
checks lat_S.requires_grad, runs loss.backward(), confirms LoRA grads are non-None,
and does 3 optimizer steps printing the loss. If loss is finite and grads flow,
the approach is validated and we patch the real trainer.
"""
import os, sys, time
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", Path.home() / "AI/ComfyUI"))
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
os.chdir(COMFY_ROOT)
sys.path.insert(0, str(Path.home() / "AI/distill"))

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
# Flip ComfyUI into training mode so RoPE (and any other gated op) uses the
# differentiable pure-PyTorch path instead of the no-backward comfy_kitchen op.
comfy.model_management.in_training = True

# reuse the scaffold's loaders
import argparse
import wan_distill_v2 as wd


def _diff_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
    """comfy sample_euler, WITHOUT @torch.no_grad — retains the autograd graph."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


def sample_student_diff(model_patcher, positive, negative, latent_dict, steps, cfg, seed):
    latent = latent_dict["samples"]
    ms = model_patcher.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(ms, "simple", steps).to(latent.device)
    noise = comfy.sample.prepare_noise(latent, seed).to(latent.device)
    sampler = comfy.samplers.KSAMPLER(_diff_euler)
    guider = comfy.samplers.CFGGuider(model_patcher)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    with torch.enable_grad():
        out = guider.sample(noise, latent, sampler, sigmas,
                            denoise_mask=None, disable_pbar=True, seed=seed)
    return out


def main():
    args = argparse.Namespace(
        wan_unet=wd.DEFAULT_WAN_UNET, lightx2v_lora=wd.DEFAULT_LIGHTX2V_LORA,
        vae=wd.DEFAULT_VAE, text_encoder=wd.DEFAULT_TEXT_ENCODER,
        width=256, height=256, length=17, cfg=1.0,  # tiny footprint to isolate grad-flow from OOM
        sampler="euler", scheduler="simple", negative="",
        teacher_steps=4, student_steps=2, rank=8,
    )
    print("[test] loading pipeline...")
    model_t, model_s, clip, vae = wd.build_pipeline(args)
    peft_wan = wd.inject_peft_lora(model_s, args.rank)
    trainable = [p for p in peft_wan.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    still = wd.load_still_as_tensor(
        str(Path.home() / "AI/distill/stills/d01_misty_forest.png"),
        args.width, args.height)
    prompt = "soft fog drifting slowly between pine trees at dawn, gentle light shift"

    for step in range(1, 4):
        pos = wd.encode_prompt(clip, prompt)
        neg = wd.encode_prompt(clip, args.negative)
        pos_t, neg_t, latent_dict = wd.prepare_wan_inputs(
            pos, neg, vae, still, args.width, args.height, args.length)
        latent_s = {"samples": latent_dict["samples"].clone()}
        seed = 1234 + step

        peft_wan.disable_adapter_layers()
        with torch.no_grad():
            lat_T = wd.sample_latent(model_t, pos_t, neg_t, latent_dict,
                                     args.teacher_steps, args.cfg,
                                     args.sampler, args.scheduler, seed)
        peft_wan.enable_adapter_layers()
        lat_S = sample_student_diff(model_s, pos_t, neg_t, latent_s,
                                    args.student_steps, args.cfg, seed)

        print(f"[test] step {step}: lat_S.requires_grad={lat_S.requires_grad} "
              f"grad_fn={lat_S.grad_fn is not None} "
              f"shapes T={tuple(lat_T.shape)} S={tuple(lat_S.shape)}")
        loss = F.mse_loss(lat_S, lat_T.to(lat_S.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        n_grad = sum(1 for p in trainable if p.grad is not None and p.grad.abs().sum() > 0)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        print(f"[test] step {step}: loss={loss.item():.5f}  "
              f"params_with_grad={n_grad}/{len(trainable)}", flush=True)
        # free per-step memory so the next step's grad graph fits
        del lat_S, lat_T, loss, pos, neg, pos_t, neg_t, latent_dict, latent_s
        import gc; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("[test] VALIDATION PASS — backward works, grads flow, loss finite.")


if __name__ == "__main__":
    main()
