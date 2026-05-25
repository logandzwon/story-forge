#!/usr/bin/env python3
"""Convert a PEFT-saved distill LoRA into a ComfyUI-loadable LoRA.

wan_distill_v2.py saves PEFT-native keys:
    base_model.model.blocks.0.cross_attn.k.lora_A.default.weight
ComfyUI's LoraLoaderModelOnly expects diffusion-model keys:
    diffusion_model.blocks.0.cross_attn.k.lora_A.weight

Transform: strip the PEFT wrapper prefix -> 'diffusion_model.', drop the
'.default' adapter-name infix. Cast to fp16 (halves file size; weights are
fp16 at inference anyway).

Usage:
    python convert_peft_lora.py IN.safetensors OUT.safetensors
"""
import sys
from safetensors.torch import load_file, save_file


def convert(in_path: str, out_path: str) -> tuple[int, int]:
    sd = load_file(in_path)
    out = {}
    for k, v in sd.items():
        if "lora_" not in k:
            continue
        nk = k.replace("base_model.model.", "diffusion_model.")
        nk = nk.replace(".default.weight", ".weight")
        out[nk] = v.half()
    save_file(out, out_path)
    return len(sd), len(out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: convert_peft_lora.py IN.safetensors OUT.safetensors")
    n_in, n_out = convert(sys.argv[1], sys.argv[2])
    print(f"converted {n_in} -> {n_out} keys -> {sys.argv[2]}")
