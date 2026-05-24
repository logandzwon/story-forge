"""verify_flash_attn.py — correctness + perf gate for flash_attn_mps.py.

Compares vs F.scaled_dot_product_attention on:  (1, 40, 4096, 128) fp16, MPS.
PASS gate: PSNR >= 35 dB AND speedup > 1.5x.
"""
import math
import statistics
import time

import torch
import torch.nn.functional as F

from flash_attn_mps import flash_attn_fwd

assert torch.backends.mps.is_available(), "MPS not available"

# Wan 2.2 14B self-attention shape per call (single chunk).
B, H, S, D = 1, 40, 4096, 128
SCALE = D ** -0.5
N_WARMUP = 3
N_RUNS = 10


def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """PSNR in dB for fp tensors. Reference peak = max|ref|."""
    diff = (test.float() - ref.float())
    mse = (diff * diff).mean().item()
    if mse == 0.0:
        return float("inf")
    peak = max(ref.abs().max().item(), 1e-6)
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def trimmed_mean(xs):
    xs = sorted(xs)
    if len(xs) >= 4:
        xs = xs[1:-1]  # drop fastest + slowest
    return statistics.mean(xs)


def time_call(fn, *args, n_warm=N_WARMUP, n=N_RUNS):
    for _ in range(n_warm):
        out = fn(*args)
        torch.mps.synchronize()
    samples = []
    for _ in range(n):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        torch.mps.synchronize()
        samples.append(time.perf_counter() - t0)
    return out, trimmed_mean(samples)


def main():
    g = torch.Generator(device="mps").manual_seed(0xBEEF)
    q = torch.randn((B, H, S, D), dtype=torch.float16, device="mps", generator=g)
    k = torch.randn((B, H, S, D), dtype=torch.float16, device="mps", generator=g)
    v = torch.randn((B, H, S, D), dtype=torch.float16, device="mps", generator=g)
    print(f"Shape: Q,K,V = {tuple(q.shape)} {q.dtype}  scale={SCALE:.5f}")

    # Reference: PyTorch MPS SDPA.
    ref, ref_t = time_call(
        lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_, is_causal=False, scale=SCALE),
        q, k, v,
    )

    # Fused Metal kernel.
    fused, fused_t = time_call(flash_attn_fwd, q, k, v)

    # Correctness.
    max_abs = (fused.float() - ref.float()).abs().max().item()
    db = psnr(ref, fused)

    # Perf.
    speedup = ref_t / fused_t if fused_t > 0 else float("inf")

    print(f"max_abs_diff = {max_abs:.4e}")
    print(f"PSNR         = {db:.2f} dB   (gate >= 35)")
    print(f"fused wall   = {fused_t*1000:.3f} ms (trimmed mean of {N_RUNS-2})")
    print(f"ref   wall   = {ref_t*1000:.3f} ms")
    print(f"speedup      = {speedup:.2f}x   (gate > 1.5x)")

    psnr_ok = db >= 35.0
    perf_ok = speedup > 1.5
    gate = psnr_ok and perf_ok
    print("GATE: " + ("PASS" if gate else "FAIL")
          + f"  (psnr_ok={psnr_ok}, perf_ok={perf_ok})")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
