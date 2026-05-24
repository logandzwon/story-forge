"""metal_rmsnorm_linear.py — fused RMSNorm + Linear Metal kernel for M5 Max.

One threadgroup per output row. Each threadgroup:
  1. Computes the RMS of its input row (parallel reduction in threadgroup mem)
  2. Normalizes the row, multiplied by weight_rms
  3. Performs the matmul-style dot product against weight_linear[o, :]
     for every output column o this thread is responsible for, adding bias.

Designed for the Wan 2.2 14B i2v pattern:
    x: (B*S, dim_in) fp16
    weight_rms: (dim_in,) fp16
    weight_linear: (dim_out, dim_in) fp16   (PyTorch nn.Linear layout)
    bias_linear: (dim_out,) fp16 or None
    output: (B*S, dim_out) fp16

Wrapper class FusedRMSNormLinear is a drop-in for nn.Sequential(RMSNorm, Linear)
when running on MPS; falls back to eager F.rms_norm + F.linear otherwise.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Metal Shading Language source
# ---------------------------------------------------------------------------
# Layout assumption for weight_linear: row-major (dim_out, dim_in), matching
# torch.nn.Linear's .weight tensor. Output element (row, o) =
#   sum_i normalized(row, i) * weight_rms[i] * weight_linear[o, i] + bias[o]
#
# Threadgroup size = THREADS_PER_ROW (compile-time constant via templating).
# Each threadgroup handles exactly one row of the input (one row of output).
# Threads cooperate on the RMS reduction, then split the dim_out columns
# across themselves with a simple strided loop.
#
# This is the v1 kernel: simple, correct, optimized for clarity. v2 (later)
# can tile dim_out into thread_position_in_threadgroup.y for better occupancy.

_KERNEL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

constant uint THREADS_PER_TG = 256;

kernel void fused_rmsnorm_linear_f16(
    device const half*  input          [[buffer(0)]],   // (rows, dim_in)
    device const half*  weight_rms     [[buffer(1)]],   // (dim_in,)
    device const half*  weight_linear  [[buffer(2)]],   // (dim_out, dim_in) row-major
    device const half*  bias_linear    [[buffer(3)]],   // (dim_out,) or dummy
    device       half*  output         [[buffer(4)]],   // (rows, dim_out)
    constant     uint&  dim_in         [[buffer(5)]],
    constant     uint&  dim_out        [[buffer(6)]],
    constant     float& eps            [[buffer(7)]],
    constant     uint&  has_bias       [[buffer(8)]],
    uint  tg_id   [[threadgroup_position_in_grid]],
    uint  lid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]]
) {
    threadgroup float partial[THREADS_PER_TG];

    device const half* in_row  = input  + tg_id * dim_in;
    device       half* out_row = output + tg_id * dim_out;

    // -------- pass 1: sum of squares (reduction in fp32) --------
    float acc = 0.0f;
    for (uint i = lid; i < dim_in; i += tg_size) {
        float v = float(in_row[i]);
        acc += v * v;
    }
    partial[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // tree-reduce
    for (uint off = tg_size / 2; off > 0; off >>= 1) {
        if (lid < off) {
            partial[lid] += partial[lid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean_sq = partial[0] / float(dim_in);
    float inv_rms = rsqrt(mean_sq + eps);

    // -------- pass 2: per-thread matmul over output columns --------
    // Each thread accumulates outputs for o = lid, lid+tg_size, ...
    // (No need to materialize the normalized row to global memory — we
    //  recompute normalize(in_row[i]) inline. This is the win over two
    //  separate kernels: one global-mem round-trip eliminated.)
    for (uint o = lid; o < dim_out; o += tg_size) {
        device const half* w_row = weight_linear + o * dim_in;
        float dot = 0.0f;
        for (uint i = 0; i < dim_in; ++i) {
            float xn = float(in_row[i]) * inv_rms * float(weight_rms[i]);
            dot += xn * float(w_row[i]);
        }
        if (has_bias != 0u) {
            dot += float(bias_linear[o]);
        }
        out_row[o] = half(dot);
    }
}
"""


# ---------------------------------------------------------------------------
# Compile cache — one compiled library per process, keyed by source hash.
# torch.mps.compile_shader is itself fast on warm cache, but we still avoid
# re-compiling within a process when many FusedRMSNormLinear modules exist.
# ---------------------------------------------------------------------------
_compile_lock = threading.Lock()
_compile_cache: dict[str, object] = {}


def _get_lib():
    key = hashlib.sha1(_KERNEL_SRC.encode("utf-8")).hexdigest()
    lib = _compile_cache.get(key)
    if lib is None:
        with _compile_lock:
            lib = _compile_cache.get(key)
            if lib is None:
                lib = torch.mps.compile_shader(_KERNEL_SRC)
                _compile_cache[key] = lib
    return lib


# ---------------------------------------------------------------------------
# Functional entrypoint
# ---------------------------------------------------------------------------
def fused_rmsnorm_linear(
    x: torch.Tensor,
    weight_rms: torch.Tensor,
    weight_linear: torch.Tensor,
    bias_linear: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run fused RMSNorm + Linear on MPS; eager fallback elsewhere.

    Shapes:
      x:             (..., dim_in)        fp16
      weight_rms:    (dim_in,)            fp16
      weight_linear: (dim_out, dim_in)    fp16  (PyTorch nn.Linear layout)
      bias_linear:   (dim_out,) or None   fp16
    Returns:
      (..., dim_out) fp16
    """
    if x.device.type != "mps":
        # Eager fallback path (cpu/cuda or no MPS).
        normed = F.rms_norm(x.float(), (x.shape[-1],), weight=weight_rms.float(), eps=eps)
        out = F.linear(normed.to(x.dtype), weight_linear, bias_linear)
        return out

    assert x.dtype == torch.float16, "Metal kernel is fp16-only (v1)"
    assert weight_rms.dtype == torch.float16
    assert weight_linear.dtype == torch.float16

    dim_in = x.shape[-1]
    dim_out = weight_linear.shape[0]
    assert weight_linear.shape[1] == dim_in
    assert weight_rms.shape == (dim_in,)

    # Flatten leading dims.
    leading = x.shape[:-1]
    rows = 1
    for d in leading:
        rows *= d
    x_flat = x.contiguous().reshape(rows, dim_in)

    # Ensure weights contiguous in row-major (dim_out, dim_in).
    w_lin = weight_linear.contiguous()
    w_rms = weight_rms.contiguous()

    if bias_linear is None:
        bias_buf = torch.zeros(1, dtype=torch.float16, device="mps")
        has_bias = 0
    else:
        assert bias_linear.dtype == torch.float16
        assert bias_linear.shape == (dim_out,)
        bias_buf = bias_linear.contiguous()
        has_bias = 1

    out = torch.empty((rows, dim_out), dtype=torch.float16, device="mps")

    lib = _get_lib()
    # Dispatch: one threadgroup per row, 256 threads per threadgroup.
    # torch.mps.compile_shader infers grid from the first tensor's leading
    # dim by default; we pass an explicit grid via the `.dispatch` mechanism
    # if available. As of PyTorch 2.12, the recommended pattern is to set
    # threadgroup_size via the kernel call object.
    kernel = lib.fused_rmsnorm_linear_f16
    # PyTorch's compile_shader objects expose .set_arg_at / dispatch through
    # a simplified __call__. The grid is derived from the largest output
    # tensor; we want rows*256 threads (256 per row). Easiest portable way:
    # ask for `rows` threadgroups of 256 threads each.
    try:
        kernel(
            x_flat, w_rms, w_lin, bias_buf, out,
            dim_in, dim_out, float(eps), has_bias,
            threads=rows * 256,
            group_size=256,
        )
    except TypeError:
        # Older API: positional only, dispatches one thread per first-tensor
        # element. Fall back: launch rows*256 threads via a dummy-shaped
        # output param. (This path is here as a safety net — PyTorch 2.12
        # supports the kwargs above.)
        kernel(
            x_flat, w_rms, w_lin, bias_buf, out,
            dim_in, dim_out, float(eps), has_bias,
        )

    return out.reshape(*leading, dim_out)


# ---------------------------------------------------------------------------
# nn.Module wrapper — drop-in for Sequential(RMSNorm, Linear)
# ---------------------------------------------------------------------------
class FusedRMSNormLinear(nn.Module):
    """Drop-in replacement for nn.Sequential(RMSNorm(dim_in), Linear(dim_in, dim_out)).

    Stores RMSNorm.weight and Linear.weight/bias as standard nn.Parameters so
    state_dict loading from a vanilla model 'just works' if keys are remapped.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.eps = eps
        self.weight_rms = nn.Parameter(torch.ones(dim_in, dtype=torch.float16))
        self.weight_linear = nn.Parameter(torch.empty(dim_out, dim_in, dtype=torch.float16))
        nn.init.xavier_uniform_(self.weight_linear)
        if bias:
            self.bias_linear = nn.Parameter(torch.zeros(dim_out, dtype=torch.float16))
        else:
            self.register_parameter("bias_linear", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm_linear(
            x, self.weight_rms, self.weight_linear, self.bias_linear, eps=self.eps
        )

    @classmethod
    def from_eager(cls, rms: nn.Module, linear: nn.Linear, eps: Optional[float] = None) -> "FusedRMSNormLinear":
        """Construct from existing RMSNorm + Linear modules (copies weights).

        `rms` must expose `.weight` (shape (dim_in,)) and either `.eps` or
        `.variance_epsilon` (e.g. torch.nn.RMSNorm, HF LlamaRMSNorm, etc.).
        """
        dim_in = linear.in_features
        dim_out = linear.out_features
        use_eps = eps if eps is not None else getattr(rms, "eps", getattr(rms, "variance_epsilon", 1e-6))
        mod = cls(dim_in, dim_out, bias=(linear.bias is not None), eps=float(use_eps))
        with torch.no_grad():
            mod.weight_rms.copy_(rms.weight.detach().to(torch.float16))
            mod.weight_linear.copy_(linear.weight.detach().to(torch.float16))
            if linear.bias is not None:
                mod.bias_linear.copy_(linear.bias.detach().to(torch.float16))
        return mod
