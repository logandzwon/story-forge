"""verify_rmsnorm_linear.py — correctness + perf gate for the fused kernel.

Input shape: (1, 4096, 5120) fp16 — representative of a Wan 2.2 14B i2v block
(B=1, seq=4096 image tokens, dim=5120). Compares the fused kernel vs the eager
F.rms_norm -> F.linear reference path, asserts fp16-precision agreement, then
benchmarks both with mps.synchronize() between runs.
"""
import math
import time

import torch
import torch.nn.functional as F

from metal_rmsnorm_linear import fused_rmsnorm_linear


DEVICE = "mps"
DTYPE = torch.float16
B, S, DIM_IN = 1, 4096, 5120
DIM_OUT = 5120
EPS = 1e-6
N_WARMUP = 3
N_RUNS = 10


def eager_ref(x, w_rms, w_lin, b_lin, eps):
    # rms_norm in fp32 for stability, cast back to input dtype (matches what
    # most production stacks do, incl. HuggingFace's RMSNorm path).
    normed = F.rms_norm(x.float(), (x.shape[-1],), weight=w_rms.float(), eps=eps)
    return F.linear(normed.to(x.dtype), w_lin, b_lin)


def bench(fn, n_runs=N_RUNS, n_warmup=N_WARMUP):
    for _ in range(n_warmup):
        out = fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n_runs):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    # trimmed mean (drop best & worst)
    trimmed = ts[1:-1] if len(ts) >= 4 else ts
    return sum(trimmed) / len(trimmed) * 1000.0, out  # ms


def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    diff = (ref.float() - test.float())
    mse = (diff * diff).mean().item()
    if mse == 0:
        return float("inf")
    peak = ref.float().abs().max().item()
    if peak == 0:
        peak = 1.0
    return 20.0 * math.log10(peak / math.sqrt(mse))


def main():
    assert torch.backends.mps.is_available()
    torch.manual_seed(0)

    x = torch.randn(B, S, DIM_IN, dtype=DTYPE, device=DEVICE) * 0.5
    w_rms = torch.randn(DIM_IN, dtype=DTYPE, device=DEVICE) * 0.1 + 1.0
    w_lin = torch.randn(DIM_OUT, DIM_IN, dtype=DTYPE, device=DEVICE) * (1.0 / math.sqrt(DIM_IN))
    b_lin = torch.randn(DIM_OUT, dtype=DTYPE, device=DEVICE) * 0.01

    # ---- correctness ----
    out_fused = fused_rmsnorm_linear(x, w_rms, w_lin, b_lin, eps=EPS)
    out_ref = eager_ref(x, w_rms, w_lin, b_lin, eps=EPS)

    max_abs = (out_fused.float() - out_ref.float()).abs().max().item()
    p = psnr(out_ref, out_fused)
    ok = torch.allclose(out_fused, out_ref, rtol=1e-3, atol=1e-3)

    print(f"shape in={tuple(x.shape)}  out={tuple(out_fused.shape)}  dtype={DTYPE}")
    print(f"max_abs_diff = {max_abs:.6f}")
    print(f"PSNR(fused vs ref) = {p:.2f} dB")
    print(f"allclose(rtol=1e-3, atol=1e-3): {'PASS' if ok else 'FAIL'}")
    if not ok:
        # print a few diffs for debugging
        diff = (out_fused.float() - out_ref.float()).abs()
        idx = torch.argsort(diff.flatten(), descending=True)[:5]
        for i in idx.tolist():
            r, c = divmod(i, DIM_OUT)
            print(f"  row={r:5d} col={c:5d}  ref={out_ref.flatten()[i].item():.4f}  fused={out_fused.flatten()[i].item():.4f}  diff={diff.flatten()[i].item():.4f}")

    # ---- perf ----
    t_fused_ms, _ = bench(lambda: fused_rmsnorm_linear(x, w_rms, w_lin, b_lin, eps=EPS))
    t_ref_ms, _   = bench(lambda: eager_ref(x, w_rms, w_lin, b_lin, eps=EPS))
    speedup = t_ref_ms / t_fused_ms if t_fused_ms > 0 else float("inf")

    print()
    print(f"fused (Metal)  : {t_fused_ms:7.3f} ms  (trimmed mean of {N_RUNS} runs)")
    print(f"reference path : {t_ref_ms:7.3f} ms")
    print(f"speedup        : {speedup:.2f}x")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
