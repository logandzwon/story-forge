"""hello_metal.py — verify torch.mps.compile_shader wiring on M5 Max.

Compiles a trivial 'add 1.0' Metal kernel, dispatches it on a fp16 tensor,
and checks the result. PASS/FAIL + wall time printed to stdout.
"""
import time
import torch

assert torch.backends.mps.is_available(), "MPS not available"
assert hasattr(torch.mps, "compile_shader"), "torch.mps.compile_shader missing (need PyTorch >= 2.5)"

SRC = r"""
#include <metal_stdlib>
using namespace metal;

kernel void add_one_half(device half* inout [[buffer(0)]],
                         uint idx [[thread_position_in_grid]]) {
    inout[idx] = inout[idx] + half(1.0);
}
"""

def main():
    lib = torch.mps.compile_shader(SRC)
    x = torch.randn((128, 256), dtype=torch.float16, device="mps")
    ref = x + 1.0
    y = x.clone()

    torch.mps.synchronize()
    t0 = time.perf_counter()
    lib.add_one_half(y)
    torch.mps.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    max_abs = (y - ref).abs().max().item()
    ok = torch.allclose(y, ref, rtol=1e-3, atol=1e-3)
    print(f"shape={tuple(x.shape)} dtype={x.dtype}")
    print(f"max_abs_diff={max_abs:.6f}  wall={dt_ms:.3f} ms")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
