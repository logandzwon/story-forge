# Wan 2-step distillation — trained on Apple Silicon

> Teaching Wan 2.2 i2v to do in **2 denoising steps** what it normally needs 4 for, by training a small LoRA — **on a MacBook, not a data center.** As far as we can find, this is the first distillation *training* of a video diffusion model done on Apple Silicon / MPS.

## What this is

We already run Wan 2.2 image-to-video with the `lightx2v` 4-step LoRA (≈4× over base). This trains our **own** rank-8 LoRA on top that collapses those 4 steps into 2 — a perpetual ~2× speedup that lives on the same machine that runs inference.

It's LCM-style consistency distillation in latent space:

- **Teacher:** Wan + lightx2v, 4-step sample → `lat_T` (the target, no gradient)
- **Student:** Wan + lightx2v + our trainable LoRA, 2-step sample → `lat_S` (with gradient)
- **Loss:** `MSE(lat_S, lat_T)` — backprop updates only the student LoRA

## The hard part: ComfyUI is built for inference, not training

Most people distill in diffusers/native training rigs on CUDA. We drive Wan through ComfyUI on MPS, which fights training at every turn. Four walls, four fixes — this is the reproducible recipe:

1. **Memory (OOM).** fp16 dual-stage Wan + the gradient graph won't fit a 64 GB Mac mini. → Train on the 128 GB M5; keep the footprint modest (`192×192×17f`) so the autograd graph fits in RAM without swap-thrash. A step-distill LoRA's behavior transfers across resolution, so small training frames are fine.

2. **`@torch.no_grad()` samplers.** Every ComfyUI k-diffusion sampler (`sample_euler`, etc.) is decorated `@torch.no_grad()` — the student latent comes back with **no autograd graph** and `loss.backward()` fails with *"element 0 does not require grad."* → Hand-roll a differentiable euler (identical math, no decorator) and drive it through `comfy.samplers.CFGGuider` (whose `sample`/`inner_sample` path has **no** no_grad), under `torch.enable_grad()`. The teacher can stay on the normal no_grad KSampler. See `sample_student_diff()` + `_diff_euler()` in `wan_distill_v2.py`.

3. **Device split.** The teacher latent can land on CPU while the student is on `mps:0` → `mse_loss` device mismatch. → `lat_T.to(lat_S.device)` before the loss.

4. **Non-differentiable fused ops.** ComfyUI's RoPE (`comfy/ldm/flux/math.py: apply_rope1`) dispatches to a `comfy_kitchen` custom op with **no autograd backward** → *"Trying to backward through comfy_kitchen.apply_rope1 ... no autograd formula."* → ComfyUI has a built-in switch: set `comfy.model_management.in_training = True` and RoPE uses the pure-PyTorch differentiable path. (It's the only op gated this way; the rest of Wan's path is plain SDPA, already differentiable.)

## Validate before you commit hours

`grad_test.py` is a contained harness: load the stack, run one teacher + one student sample, assert `lat_S.requires_grad`, run `loss.backward()`, confirm gradients reach the LoRA params, do a few optimizer steps. Run it (ComfyUI stopped, so it owns the memory) **before** firing the full run:

```bash
cd ~/AI/ComfyUI
COMFY_ROOT=~/AI/ComfyUI PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -u grad_test.py
# expect: [test] step N: loss=...  params_with_grad>0
```

## Run it

```bash
cd ~/AI/ComfyUI   # comfy.* must be importable; folder_paths reads CWD for model dirs
COMFY_ROOT=~/AI/ComfyUI PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -u \
  wan_distill_v2.py \
    --dataset dataset.json \
    --out-dir ~/AI/distill/checkpoints \
    --rank 8 --steps 300 --save-every 25 \
    --width 192 --height 192 --length 17
```

- `dataset.json` is a list of `{"still": "/abs/path.png", "prompt": "motion description"}`. ~10 varied stills cycling is enough — the LoRA learns the per-timestep denoising-collapse correction, which is largely content/resolution-agnostic.
- Use the **fp16** text encoder (`umt5_xxl_fp16.safetensors`) — MPS does not support fp8.
- ~24s/step at this footprint → ~2 hr for 300 steps on an M5 Max. Checkpoints (LoRA-only safetensors) drop every 25 steps.

## Caveats / honest notes

- Per-step loss is noisy because each step draws a different still/prompt; judge the trend across checkpoints, not single steps.
- Caching (TeaCache/MagCache/EasyCache) gives ~nothing at ≤4 steps — don't bother stacking it; spend the budget here and on GGUF quant instead.
- Quality at production resolution (480p+) is the open validation: a LoRA trained at 192² should transfer, but verify with the LPIPS harness (`bin/measure-render`) before trusting the 2× in production. Higher-res training would want gradient checkpointing.

## To advance (not copy)

- Port the score-regularized consistency loss from [NVlabs/rcm](https://github.com/NVlabs/rcm) (released, Wan-specific) — skip their 8-GPU scale, keep the JVP loss.
- Asymmetric MoE distillation: Wan's high-noise stage tolerates fewer steps than low-noise; a split rank is untried.
- Seed from [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) causal-consistency init to stabilize the 4→2 collapse.
