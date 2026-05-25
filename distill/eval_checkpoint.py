#!/usr/bin/env python3
"""eval_checkpoint.py — score a v3 distill checkpoint's 1-step render against the
4-step teacher, in LPIPS, vs the 1.09 'wall' that 1-step lightx2v alone hits.

Reuses wan_distill_v3.py's exact loaders + samplers, so the eval render path is
identical to the one training optimized (same M5 fp16 stack, same model files).

Per (still, prompt, seed):
    teacher = base + lightx2v, 4 steps         (adapters OFF)  -> reference
    student = base + lightx2v + ckpt LoRA, 1 step (adapters ON) -> candidate
    decode both via VAE -> frames; LPIPS(student, teacher) per frame.
Lower mean LPIPS = the student's single step matches the teacher's four.
The bar: beat 1.09 (what raw 1-step lightx2v scores — catastrophic).

Modes:
    --smoke-decode   load ONLY the VAE + lpips and exercise the decode->LPIPS
                     path on a random latent (no 14B UNet -> safe to run while
                     training holds the GPU). Validates the new code cheaply.
    (default)        full eval of --ckpt over the eval set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Importing the trainer wires ComfyUI onto sys.path + chdir, and exposes the
# proven loaders/samplers. It is guarded by __main__, so nothing runs on import.
import wan_distill_v3 as W
import comfy.model_management as mm
mm.in_training = False  # eval: use the fast inference op path (no autograd needed)

import comfy.utils
import comfy.sd
import lpips as lpips_lib


# ---------------------------------------------------------------------------- #
# decode + perceptual metric                                                    #
# ---------------------------------------------------------------------------- #
def decode_to_frames(vae, latent: torch.Tensor) -> torch.Tensor:
    """VAE-decode a Wan latent [B,C,T,h,w] -> frames [N,3,H,W] in [0,1]."""
    imgs = vae.decode(latent)
    # comfy VAE.decode returns IMAGE layout with channels-last. Wan video VAE
    # gives [B,T,H,W,3] (5D) or [T,H,W,3] (4D). Flatten any leading dims to N.
    if imgs.dim() == 5:
        b, t, h, w, c = imgs.shape
        imgs = imgs.reshape(b * t, h, w, c)
    elif imgs.dim() == 4 and imgs.shape[-1] in (1, 3):
        pass  # [N,H,W,3]
    else:
        raise RuntimeError(f"unexpected decode shape {tuple(imgs.shape)}")
    imgs = imgs.clamp(0, 1).permute(0, 3, 1, 2).contiguous()  # [N,3,H,W]
    return imgs


def lpips_mean_p95(metric, a: torch.Tensor, b: torch.Tensor):
    """Per-frame LPIPS between two [N,3,H,W] tensors in [0,1]. Returns (mean,p95,max)."""
    a = (a * 2 - 1).to(torch.float32)  # lpips expects [-1,1]
    b = (b * 2 - 1).to(torch.float32)
    with torch.no_grad():
        d = metric(a, b).flatten()  # [N]
    d = d.detach().cpu()
    k = min(d.numel(), max(1, int(round(0.95 * d.numel()))))
    p95 = float(d.kthvalue(k).values)
    return float(d.mean()), p95, float(d.max())


# ---------------------------------------------------------------------------- #
# smoke test — decode + lpips path only (no 14B UNet)                            #
# ---------------------------------------------------------------------------- #
def smoke_decode(args):
    print("[smoke] loading VAE only (no UNet) ...")
    vae_path = W.resolve("vae", args.vae)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path, safe_load=True))
    # Wan 2.1 VAE: 16 latent channels, /8 spatial, /4 (+1) temporal.
    t_lat = (args.length - 1) // 4 + 1
    h_lat, w_lat = args.height // 8, args.width // 8
    lat = torch.randn(1, 16, t_lat, h_lat, w_lat)
    print(f"[smoke] decoding random latent {tuple(lat.shape)} ...")
    frames = decode_to_frames(vae, lat)
    print(f"[smoke] decoded -> frames {tuple(frames.shape)}")
    metric = lpips_lib.LPIPS(net="alex", verbose=False)
    m, p95, mx = lpips_mean_p95(metric, frames, frames.flip(0))
    print(f"[smoke] LPIPS self-vs-shuffled mean={m:.4f} p95={p95:.4f} max={mx:.4f}")
    print("[smoke] OK — decode + LPIPS path works.")


# ---------------------------------------------------------------------------- #
# full eval                                                                     #
# ---------------------------------------------------------------------------- #
def load_ckpt_into_student(peft_wan, ckpt_path: str):
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    # checkpoint holds only "lora_" keys; load non-strict onto the PEFT module.
    missing, unexpected = peft_wan.load_state_dict(sd, strict=False)
    loaded = len(sd)
    print(f"[eval] loaded {loaded} LoRA tensors from {Path(ckpt_path).name} "
          f"(unexpected={len(unexpected)})")
    if loaded == 0 or len(unexpected) == loaded:
        raise RuntimeError("checkpoint keys did not match the PEFT module")


def eval_one(args, model_t, model_s, peft_wan, clip, vae, metric, eval_set, ckpt):
    """Render teacher(4-step) vs student(1-step) for the eval set, return result."""
    load_ckpt_into_student(peft_wan, ckpt)
    rows = []
    for i, sample in enumerate(eval_set):
        still = W.load_still_as_tensor(sample["still"], args.width, args.height)
        prompt = sample["prompt"]
        pos = W.encode_prompt(clip, prompt)
        neg = W.encode_prompt(clip, args.negative)
        pos_t, neg_t, latent_dict = W.prepare_wan_inputs(
            pos, neg, vae, still, args.width, args.height, args.length)
        latent_s = {"samples": latent_dict["samples"].clone()}

        peft_wan.disable_adapter_layers()
        with torch.no_grad():
            lat_T = W.sample_latent(model_t, pos_t, neg_t, latent_dict,
                                    args.teacher_steps, args.cfg,
                                    args.sampler, args.scheduler, args.seed)
        peft_wan.enable_adapter_layers()
        with torch.no_grad():
            lat_S = W.sample_latent(model_s, pos_t, neg_t, latent_s,
                                    args.student_steps, args.cfg,
                                    args.sampler, args.scheduler, args.seed)

        fr_T = decode_to_frames(vae, lat_T)
        fr_S = decode_to_frames(vae, lat_S.to(fr_T.device))
        m, p95, mx = lpips_mean_p95(metric, fr_S, fr_T)
        print(f"[eval]   sample {i} lpips_mean={m:.4f} p95={p95:.4f} ({prompt[:36]!r})")
        rows.append({"sample": i, "prompt": prompt, "lpips_mean": m,
                     "lpips_p95": p95, "lpips_max": mx, "pass": m < args.wall})

    mean_overall = sum(r["lpips_mean"] for r in rows) / len(rows)
    beats = mean_overall < args.wall
    print(f"[eval] {Path(ckpt).name}: mean LPIPS {mean_overall:.4f} vs wall "
          f"{args.wall} -> {'BEATS WALL' if beats else 'above wall'}")
    return {"ckpt": Path(ckpt).name, "mean_lpips": mean_overall,
            "beats_wall": beats, "rows": rows}


def measure_wall(args, model_t, model_s, peft_wan, clip, vae, metric, eval_set):
    """The actual 1-step wall AT THIS RESOLUTION: raw lightx2v (no student LoRA),
    1 step, vs the 4-step teacher. The student must beat this same-res baseline."""
    peft_wan.disable_adapter_layers()
    vals = []
    for sample in eval_set:
        still = W.load_still_as_tensor(sample["still"], args.width, args.height)
        pos = W.encode_prompt(clip, sample["prompt"])
        neg = W.encode_prompt(clip, args.negative)
        pos_t, neg_t, latent_dict = W.prepare_wan_inputs(
            pos, neg, vae, still, args.width, args.height, args.length)
        latent_r = {"samples": latent_dict["samples"].clone()}
        with torch.no_grad():
            lat_T = W.sample_latent(model_t, pos_t, neg_t, latent_dict,
                                    args.teacher_steps, args.cfg,
                                    args.sampler, args.scheduler, args.seed)
            lat_R = W.sample_latent(model_s, pos_t, neg_t, latent_r,
                                    args.student_steps, args.cfg,
                                    args.sampler, args.scheduler, args.seed)
        fr_T = decode_to_frames(vae, lat_T)
        fr_R = decode_to_frames(vae, lat_R.to(fr_T.device))
        m, _, _ = lpips_mean_p95(metric, fr_R, fr_T)
        vals.append(m)
    wall = sum(vals) / len(vals)
    print(f"[eval] MEASURED 1-step wall (raw lightx2v vs {args.teacher_steps}-step teacher) "
          f"@ {args.width}x{args.height}: {wall:.4f}")
    return wall


def write_markdown(results, args, md_path):
    lines = [
        "# v3 1-step distill — checkpoint eval",
        "",
        f"Metric: per-frame LPIPS, student 1-step render vs teacher {args.teacher_steps}-step "
        f"render (same still/prompt/seed). Eval set: first {args.n_samples} prompts.",
        f"**Measured 1-step wall @ {args.width}×{args.height}: {args.wall:.4f}** "
        f"(raw lightx2v 1-step vs {args.teacher_steps}-step teacher, same res; "
        f"literature ref {getattr(args, 'literature_wall', 1.09)}). Lower is better.",
        "",
        "| checkpoint | mean LPIPS | beats wall? |",
        "|---|---|---|",
    ]
    best = None
    for r in results:
        mark = "✅" if r["beats_wall"] else "—"
        lines.append(f"| {r['ckpt']} | {r['mean_lpips']:.4f} | {mark} |")
        if best is None or r["mean_lpips"] < best["mean_lpips"]:
            best = r
    lines += ["", f"**Best: {best['ckpt']} @ LPIPS {best['mean_lpips']:.4f}** "
              f"({'beats the wall' if best['beats_wall'] else 'above the wall'}), "
              f"vs measured wall {args.wall:.4f}."]
    Path(md_path).write_text("\n".join(lines) + "\n")
    print(f"[eval] wrote {md_path}")
    return best


def full_eval(args):
    dataset = W.load_dataset(args.dataset)
    eval_set = dataset[: args.n_samples]

    if args.ckpt_dir:
        ckpts = sorted(Path(args.ckpt_dir).glob("student_step_*.safetensors"))
        ckpts = [str(c) for c in ckpts]
    else:
        ckpts = [args.ckpt]
    if not ckpts:
        raise SystemExit(f"no checkpoints found in {args.ckpt_dir}")
    print(f"[eval] evaluating {len(ckpts)} checkpoint(s)")

    model_t, model_s, clip, vae = W.build_pipeline(args)
    peft_wan = W.inject_peft_lora(model_s, args.rank)
    metric = lpips_lib.LPIPS(net="alex", verbose=False)

    # Measure the real wall at this resolution, then compare every checkpoint to IT
    # (not the literature 1.09, which was measured at a different resolution).
    args.literature_wall = args.wall
    args.wall = measure_wall(args, model_t, model_s, peft_wan, clip, vae, metric, eval_set)

    results = []
    for ckpt in ckpts:
        results.append(eval_one(args, model_t, model_s, peft_wan, clip, vae,
                                metric, eval_set, ckpt))
        if args.json_out:  # write incrementally so a crash still leaves partial data
            Path(args.json_out).write_text(json.dumps(results, indent=2))

    if args.md_out:
        write_markdown(results, args, args.md_out)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="single student_step_NNNN.safetensors to evaluate")
    p.add_argument("--ckpt-dir", default=None, help="eval every student_step_*.safetensors in dir")
    p.add_argument("--md-out", default=None, help="write a RESULTS markdown table here")
    p.add_argument("--dataset", default=str(Path.home() / "AI/distill/dataset_m5.json"))
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--wall", type=float, default=1.09, help="the 1-step lightx2v LPIPS to beat")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--teacher-steps", type=int, default=4)
    p.add_argument("--student-steps", type=int, default=1)
    p.add_argument("--width", type=int, default=192)   # match training resolution
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--length", type=int, default=17)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--negative", default="")
    p.add_argument("--wan-unet", default=W.DEFAULT_WAN_UNET)
    p.add_argument("--lightx2v-lora", default=W.DEFAULT_LIGHTX2V_LORA)
    p.add_argument("--vae", default=W.DEFAULT_VAE)
    p.add_argument("--text-encoder", dest="text_encoder", default=W.DEFAULT_TEXT_ENCODER)
    p.add_argument("--json-out", default=None)
    p.add_argument("--smoke-decode", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_decode:
        smoke_decode(args)
    elif args.ckpt or args.ckpt_dir:
        full_eval(args)
    else:
        raise SystemExit("provide --ckpt, --ckpt-dir, or --smoke-decode")
