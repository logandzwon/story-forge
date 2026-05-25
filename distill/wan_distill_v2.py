#!/usr/bin/env python3
"""Wan 2.2 i2v 4-step -> 2-step distillation harness (v2).

Goal: train a small LoRA on top of Wan 2.2 i2v 14B + the existing lightx2v
4-step LoRA (teacher) so that the student collapses 4 denoising steps into 2
with minimal quality loss. Output drops into the existing ComfyUI workflow.

Approach (LCM-style consistency distillation, latent-space):
  - Teacher path: Wan + lightx2v LoRA @ strength 1.0, KSampler 4 steps  -> latent_T
  - Student path: Wan + lightx2v + this new LoRA (rank 16), KSampler 2 steps -> latent_S
  - Loss: MSE(latent_S, latent_T) on the final denoised latent
  - Backprop only updates the student's PEFT LoRA adapters

Integration:
  - Imports ComfyUI's own python modules directly (no HTTP queue). This is the
    "genuine hard part" per the handoff -- we drive Wan via:
        comfy.sd.load_diffusion_model      -> ModelPatcher
        comfy.sd.load_text_encoder         -> CLIP
        comfy.sd.VAE                       -> VAE
        comfy.sd.load_lora_for_models      -> applies lightx2v LoRA (teacher)
        comfy_extras.nodes_wan.WanImageToVideo -> conditioning + empty latent
        nodes.KSampler -> sampling (same path the workflow uses)

Usage:
  source ~/AI/ComfyUI/venv/bin/activate
  cd ~/AI/ComfyUI
  python ~/AI/distill/wan_distill_v2.py \\
      --dataset ~/AI/distill/dataset.json \\
      --out-dir ~/AI/distill/checkpoints/ \\
      --steps 300 --save-every 50

Hardware target: M4 Pro 64GB unified. Wan 14B fp16 = ~27GB; teacher+student
share the same base weights (LoRA-only delta), peak ~32-36GB during sampling.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# ---- Pin ComfyUI on the import path BEFORE importing comfy.* ----------------
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", Path.home() / "AI/ComfyUI"))
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
os.chdir(COMFY_ROOT)  # folder_paths.py reads CWD for model dirs

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from safetensors.torch import save_file

# ComfyUI internals
import comfy.sd
import comfy.utils
import comfy.model_management as mm
import comfy.samplers
import comfy.sample
import folder_paths
from comfy_extras import nodes_wan
import nodes as comfy_nodes
from peft import LoraConfig, get_peft_model

# Flip ComfyUI into training mode: RoPE (comfy/ldm/flux/math.py) and any other
# gated op then use the differentiable pure-PyTorch path instead of the
# comfy_kitchen custom op that has no autograd backward formula.
mm.in_training = True


# ---------------------------------------------------------------------------- #
# Config                                                                       #
# ---------------------------------------------------------------------------- #
DEFAULT_WAN_UNET = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
DEFAULT_LIGHTX2V_LORA = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
DEFAULT_VAE = "wan_2.1_vae.safetensors"
DEFAULT_TEXT_ENCODER = "umt5_xxl_fp16.safetensors"  # M5: fp8 dtype unsupported on MPS

# LoRA target modules for Wan's WanAttentionBlock (q/k/v + cross-attn + ffn)
# Confirmed by inspecting comfy/ldm/wan/model.py WanAttentionBlock.
WAN_TARGET_MODULES = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="JSON list of {still, prompt}")
    p.add_argument("--out-dir", default=str(Path.home() / "AI/distill/checkpoints"))
    p.add_argument("--wan-unet", default=DEFAULT_WAN_UNET)
    p.add_argument("--lightx2v-lora", default=DEFAULT_LIGHTX2V_LORA)
    p.add_argument("--vae", default=DEFAULT_VAE)
    p.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--rank", type=int, default=8)  # M5 lean default
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--teacher-steps", type=int, default=4)
    p.add_argument("--student-steps", type=int, default=2)
    p.add_argument("--width", type=int, default=256)   # grad graph OOMs at 480x480x33
    p.add_argument("--height", type=int, default=256)  # 256x256x17 validated to fit
    p.add_argument("--length", type=int, default=17)
    p.add_argument("--cfg", type=float, default=1.0)  # lightx2v is CFG=1
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--negative", default="")
    p.add_argument("--dry-run", action="store_true", help="load only, no train")
    return p.parse_args()


# ---------------------------------------------------------------------------- #
# Loaders                                                                       #
# ---------------------------------------------------------------------------- #
def resolve(folder: str, name: str) -> str:
    """Resolve a model filename through ComfyUI's folder_paths."""
    p = folder_paths.get_full_path(folder, name)
    if p is None:
        raise FileNotFoundError(f"{folder}/{name} not found via folder_paths")
    return p


def load_lora_sd(path: str):
    return comfy.utils.load_torch_file(path, safe_load=True)


def build_pipeline(args):
    """Load Wan UNet + VAE + text encoder, apply lightx2v LoRA (teacher).

    Returns: (model_teacher, model_student_base, clip, vae, lightx2v_sd)
    model_student_base is the patcher BEFORE PEFT injection; caller wraps it.
    """
    print("[load] Wan UNet:", args.wan_unet)
    unet_path = resolve("diffusion_models", args.wan_unet)
    model_teacher = comfy.sd.load_diffusion_model(unet_path)

    # Student shares weights -> reload separately so PEFT wrapping doesn't leak
    # into the teacher's transformer. Two ModelPatcher instances, one base UNet
    # loaded twice (memory cost: 2x 27GB ~= 54GB which is too much). Trick:
    # share the underlying nn.Module and toggle LoRA via a flag.
    # See toggle_student_lora() below for the runtime swap.
    model_student = model_teacher  # SAME weights; PEFT adapters are extra

    print("[load] VAE:", args.vae)
    vae_path = resolve("vae", args.vae)
    vae_sd = comfy.utils.load_torch_file(vae_path, safe_load=True)
    vae = comfy.sd.VAE(sd=vae_sd)

    print("[load] text encoder:", args.text_encoder)
    te_path = resolve("text_encoders", args.text_encoder)
    clip = comfy.sd.load_text_encoder_state_dicts(
        [comfy.utils.load_torch_file(te_path, safe_load=True)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.WAN,
    )

    print("[load] lightx2v 4-step LoRA:", args.lightx2v_lora)
    lora_path = resolve("loras", args.lightx2v_lora)
    lightx2v_sd = load_lora_sd(lora_path)

    # Apply lightx2v at strength 1.0 to teacher (this is the 4-step accelerator)
    model_teacher, _ = comfy.sd.load_lora_for_models(
        model_teacher, None, lightx2v_sd, strength_model=1.0, strength_clip=0.0
    )
    # Student also gets lightx2v baseline; our trainable LoRA stacks on top
    model_student, _ = comfy.sd.load_lora_for_models(
        model_student, None, lightx2v_sd, strength_model=1.0, strength_clip=0.0
    )

    return model_teacher, model_student, clip, vae


def inject_peft_lora(model_patcher, rank: int) -> torch.nn.Module:
    """Wrap the WanModel transformer with a PEFT LoRA. Returns the wrapped nn.Module.

    NOTE: model_patcher.model is the BaseModel; .model.diffusion_model is WanModel.
    """
    wan = model_patcher.model.diffusion_model
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=WAN_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    peft_wan = get_peft_model(wan, cfg)
    model_patcher.model.diffusion_model = peft_wan
    n_train = sum(p.numel() for p in peft_wan.parameters() if p.requires_grad)
    print(f"[peft] student LoRA: rank={rank}, trainable params={n_train:,}")
    return peft_wan


# ---------------------------------------------------------------------------- #
# Sampling                                                                      #
# ---------------------------------------------------------------------------- #
def encode_prompt(clip, text: str):
    """Returns conditioning list like the CLIPTextEncode node does."""
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    return cond


def prepare_wan_inputs(positive, negative, vae, still_image, width, height, length):
    """Drive comfy_extras.nodes_wan.WanImageToVideo to build conditioning+latent.

    Returns: (positive_cond, negative_cond, latent_dict)
    """
    node = nodes_wan.WanImageToVideo()
    # node.execute returns a NodeOutput in newer ComfyUI; older returns tuple.
    # Either way, contents are (positive, negative, latent).
    out = node.execute(
        positive=positive, negative=negative, vae=vae,
        width=width, height=height, length=length, batch_size=1,
        start_image=still_image, clip_vision_output=None,
    )
    if hasattr(out, "result"):
        pos, neg, latent = out.result
    else:
        pos, neg, latent = out
    return pos, neg, latent


def sample_latent(model_patcher, positive, negative, latent_dict,
                  steps: int, cfg: float, sampler: str, scheduler: str,
                  seed: int, with_grad: bool = False):
    """Run KSampler in-process. Returns the denoised latent tensor [B,C,T,H,W]."""
    ksampler = comfy_nodes.KSampler()
    # KSampler.sample signature:
    #   (model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
    #    latent_image, denoise=1.0)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        out = ksampler.sample(
            model_patcher, seed, steps, cfg, sampler, scheduler,
            positive, negative, latent_dict, denoise=1.0,
        )
    if hasattr(out, "result"):
        (latent_out,) = out.result
    else:
        (latent_out,) = out
    return latent_out["samples"]


# --- Differentiable student sampler -----------------------------------------
# ComfyUI's k_diffusion samplers are all @torch.no_grad() (inference-only), so
# the student latent produced by KSampler has no autograd graph and loss.backward()
# fails. We run the student denoise through a custom KSAMPLER whose sampler_function
# is a differentiable euler (comfy's sample_euler minus the @no_grad), driven by
# CFGGuider (which has NO no_grad in its sample/inner_sample path) under enable_grad.
def _diff_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


def sample_student_diff(model_patcher, positive, negative, latent_dict,
                        steps, cfg, seed):
    """Differentiable 2-step student denoise. Returns a latent WITH grad graph."""
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


# ---------------------------------------------------------------------------- #
# Data                                                                          #
# ---------------------------------------------------------------------------- #
def load_dataset(path: str):
    data = json.loads(Path(path).read_text())
    print(f"[data] {len(data)} (still, prompt) pairs from {path}")
    return data


def load_still_as_tensor(path: str, width: int, height: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,3] like ComfyUI IMAGE


# ---------------------------------------------------------------------------- #
# Train                                                                         #
# ---------------------------------------------------------------------------- #
def save_lora_checkpoint(peft_wan, out_path: Path, metadata: dict):
    """Extract only the LoRA adapter weights and save in safetensors."""
    sd = {k: v.detach().cpu() for k, v in peft_wan.state_dict().items()
          if "lora_" in k}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out_path), metadata={k: str(v) for k, v in metadata.items()})
    print(f"[ckpt] {len(sd)} tensors -> {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)

    print(f"[boot] torch={torch.__version__} mps={torch.backends.mps.is_available()}")
    print(f"[boot] comfy root={COMFY_ROOT}")

    model_t, model_s, clip, vae = build_pipeline(args)
    peft_wan = inject_peft_lora(model_s, args.rank)

    if args.dry_run:
        print("[dry-run] loaders + peft injection succeeded, exiting.")
        return

    # Adam over LoRA params only
    trainable = [p for p in peft_wan.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        sample = dataset[(step - 1) % len(dataset)]
        still = load_still_as_tensor(sample["still"], args.width, args.height)
        prompt = sample["prompt"]
        seed = 1_000_000 + step  # deterministic per step

        # Encode (CPU/GPU mix — comfy handles)
        pos = encode_prompt(clip, prompt)
        neg = encode_prompt(clip, args.negative)

        # Build i2v conditioning + empty latent (same for teacher & student)
        pos_t, neg_t, latent_dict = prepare_wan_inputs(
            pos, neg, vae, still, args.width, args.height, args.length)
        # Reset the latent each iter so teacher/student see the same noise seed
        latent_dict_s = {"samples": latent_dict["samples"].clone()}

        # --- Teacher: 4 steps, no grad, LoRA adapters OFF (lightx2v only) ----
        peft_wan.disable_adapter_layers()
        with torch.no_grad():
            lat_T = sample_latent(model_t, pos_t, neg_t, latent_dict,
                                  args.teacher_steps, args.cfg,
                                  args.sampler, args.scheduler, seed,
                                  with_grad=False)
        # --- Student: 2 steps, grad on, LoRA adapters ON (differentiable) ---
        peft_wan.enable_adapter_layers()
        lat_S = sample_student_diff(model_s, pos_t, neg_t, latent_dict_s,
                                    args.student_steps, args.cfg, seed)

        loss = F.mse_loss(lat_S, lat_T.to(lat_S.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        elapsed = time.time() - t0
        print(f"[step {step:04d}/{args.steps}] loss={loss.item():.5f}  "
              f"prompt={prompt[:48]!r}  elapsed={elapsed/60:.1f}m")

        if step % args.save_every == 0:
            save_lora_checkpoint(
                peft_wan, out_dir / f"student_step_{step:04d}.safetensors",
                metadata={"step": step, "rank": args.rank,
                          "teacher_steps": args.teacher_steps,
                          "student_steps": args.student_steps,
                          "loss": float(loss.item())},
            )

        # MPS GC
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"[done] total {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
