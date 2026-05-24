"""flash_attn_mps.py — fused flash-attention-style Metal kernel for M5 Max.

Target: Wan 2.2 14B WanSelfAttention, shape (B=1, H=40, S=4096+, D=128) fp16.

Strategy (FlashAttention-2 single-pass forward, no backward):
  - Each threadgroup owns one Q-block of BR=64 rows × D=128 cols.
  - Threadgroup memory holds:
      Q_tile (BR x D)         fp16 — loaded once
      K_tile (BC x D)         fp16 — streamed
      V_tile (BC x D)         fp16 — streamed
      m_i (BR), l_i (BR)      fp32 — running softmax stats
      O_tile (BR x D)         fp32 — running accumulator
  - For each K/V block:
      S = Q_tile @ K_tile^T * scale    (BR x BC)
      m_new = max(m_i, rowmax(S))
      P = exp(S - m_new)
      alpha = exp(m_i - m_new)
      l_new = alpha * l_i + rowsum(P)
      O = alpha * O + P @ V_tile
      m_i, l_i = m_new, l_new
  - Final: O_tile / l_i, cast back to fp16, write to output.

Why this is the win over PyTorch MPS SDPA:
  - PyTorch's MPS SDPA materializes the full (S x S) attention matrix in
    global memory and runs softmax as a separate kernel. For S=4096, that's
    4096*4096*40 = 671M fp16 entries per batch = 1.3GB just for scores.
    Flash-attn keeps the running stats in registers/threadgroup memory and
    never writes scores to global.
  - We also fuse the scale, softmax, and second matmul into one launch — one
    K/V global read per K block instead of three.

Notes:
  - v1 uses plain fp16 loads + fp32 accumulate. No simdgroup_matrix yet;
    the BR x D matmul is small enough that thread-level tiling within a
    threadgroup is fine and dodges the fiddly simdgroup_matrix bring-up.
    If perf still loses we add simdgroup_matrix in v2.
  - Hard-coded for D=128, BR=64, BC=64. Wan's head_dim is 128. Other shapes
    fall through to PyTorch SDPA via the Python wrapper.
  - Non-causal only (Wan WanSelfAttention is bidirectional spatial).
"""
from __future__ import annotations

import hashlib
import threading
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Metal Shading Language source
# ---------------------------------------------------------------------------
# Tile sizes are compile-time constants. THREADS_PER_TG = 128 chosen so each
# thread owns exactly D/(THREADS_PER_TG/BR) = 64 K/V cols and 2 rows during
# the score matmul. Conservatively kept small to fit threadgroup memory on
# the 64KB-per-SM M5 Max budget.
#
# Memory budget per threadgroup (fp16 unless noted):
#   Q_tile:    64 * 128 * 2 =  16384 B
#   K_tile:    64 * 128 * 2 =  16384 B
#   V_tile:    64 * 128 * 2 =  16384 B
#   S_tile:    64 *  64 * 4 =  16384 B   (fp32 scores)
#   O_tile:    64 * 128 * 4 =  32768 B   (fp32 accum)
#   m_i,l_i:   64 * 4 * 2   =    512 B
#   ---------------------------
#   total ≈   98816 B  — too big.
#
# Trim: drop S_tile to fp16 (8192) AND keep O_tile in fp32 split across the
# threads' own register file instead of threadgroup. That removes 32768+8192.
# But we still need a shared S for the row-softmax. Keep S as fp32 (16384).
#
# New budget:
#   Q,K,V tiles + S(fp32) + m,l   = 16384*3 + 16384 + 512 = 65920 B — still tight.
#
# Final trim: BR=32, BC=64, D=128.
#   Q_tile:    32 * 128 * 2 =   8192 B
#   K_tile:    64 * 128 * 2 =  16384 B
#   V_tile:    64 * 128 * 2 =  16384 B
#   S_tile:    32 *  64 * 4 =   8192 B
#   m,l:       32 * 4 * 2   =    256 B
#   ---------------------------
#   total ≈   49408 B  — fits.
#
# O accumulator lives in thread-private fp32 array (BR/threads_y rows per
# thread × D cols). With THREADS_PER_TG=128 split as (tx=32, ty=4), each
# thread owns 1 of 32 cols (well, D/32 = 4 cols × 1 of 32 score-cols) — we
# split D=128 across tx=32 (4 cols/thread) and BR=32 across ty=4 (8 rows/
# thread). Each thread's private O is 8 rows × 4 cols = 32 fp32 = 128 B.
#
_KERNEL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// Tile dims — must match Python side.
constant constexpr uint BR = 32;     // Q rows per threadgroup
constant constexpr uint BC = 64;     // K/V cols per K-block
constant constexpr uint D  = 128;    // head dim (hard-coded for Wan)
constant constexpr uint TX = 32;     // threads along D
constant constexpr uint TY = 4;      // threads along BR
constant constexpr uint TG = TX * TY;            // 128 threads/threadgroup
constant constexpr uint D_PER_TX = D / TX;       // 4 D-cols per thread
constant constexpr uint R_PER_TY = BR / TY;      // 8 Q-rows per thread

kernel void flash_attn_fwd_f16_d128(
    device const half*  Q       [[buffer(0)]],   // (B*H, S, D)
    device const half*  K       [[buffer(1)]],   // (B*H, S, D)
    device const half*  V       [[buffer(2)]],   // (B*H, S, D)
    device       half*  O       [[buffer(3)]],   // (B*H, S, D)
    constant     uint&  seq_len [[buffer(4)]],
    constant     float& scale   [[buffer(5)]],
    uint3 tg_id   [[threadgroup_position_in_grid]],
    uint  lid     [[thread_index_in_threadgroup]]
){
    // Threadgroup grid: (n_q_blocks, B*H)
    const uint q_block = tg_id.x;
    const uint bh      = tg_id.y;
    const uint q_start = q_block * BR;

    // Thread coords inside the (TX, TY) grid.
    const uint tx = lid % TX;        // 0..31  → D-col group
    const uint ty = lid / TX;        // 0..3   → Q-row group

    // ---- threadgroup memory ----
    threadgroup half  Q_tile[BR * D];
    threadgroup half  K_tile[BC * D];
    threadgroup half  V_tile[BC * D];
    threadgroup float S_tile[BR * BC];
    threadgroup float m_i[BR];
    threadgroup float l_i[BR];

    // ---- per-thread output accumulator (fp32) ----
    // R_PER_TY rows × D_PER_TX cols = 8 × 4 = 32 floats
    float O_acc[R_PER_TY][D_PER_TX];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < R_PER_TY; ++r) {
        #pragma clang loop unroll(full)
        for (uint c = 0; c < D_PER_TX; ++c) {
            O_acc[r][c] = 0.0f;
        }
    }

    // Initialize m, l (only ty==0 lane writes, BR=32 entries across TX=32).
    if (ty == 0 && tx < BR) {
        m_i[tx] = -INFINITY;
        l_i[tx] = 0.0f;
    }

    // ---- load Q tile (BR × D = 32 × 128 = 4096 halfs, 128 threads → 32 each)
    device const half* Q_bh = Q + bh * seq_len * D;
    device const half* K_bh = K + bh * seq_len * D;
    device const half* V_bh = V + bh * seq_len * D;
    device       half* O_bh = O + bh * seq_len * D;

    // Q_tile load: each thread loads BR*D/TG = 32 halfs.
    {
        const uint per_thread = (BR * D) / TG;   // 32
        const uint base = lid * per_thread;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < per_thread; ++i) {
            const uint idx = base + i;
            const uint row = idx / D;
            const uint col = idx % D;
            const uint g_row = q_start + row;
            Q_tile[idx] = (g_row < seq_len) ? Q_bh[g_row * D + col] : half(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- iterate over K/V blocks ----
    const uint n_kv_blocks = (seq_len + BC - 1) / BC;

    for (uint kv = 0; kv < n_kv_blocks; ++kv) {
        const uint k_start = kv * BC;

        // Load K_tile (BC*D = 8192 halfs, 128 threads → 64 each)
        {
            const uint per_thread = (BC * D) / TG;  // 64
            const uint base = lid * per_thread;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < per_thread; ++i) {
                const uint idx = base + i;
                const uint row = idx / D;
                const uint col = idx % D;
                const uint g_row = k_start + row;
                K_tile[idx] = (g_row < seq_len) ? K_bh[g_row * D + col] : half(0);
                V_tile[idx] = (g_row < seq_len) ? V_bh[g_row * D + col] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Compute S = Q @ K^T  (BR × BC scores), each thread fills
        //      R_PER_TY × (BC/TX*TY...) — simplest: split BR*BC across TG.
        // BR*BC = 32*64 = 2048 scores, TG=128 → 16 scores/thread.
        // Map score (r, c) to thread: r = ty*R_PER_TY + (i / BC_per_thread...)
        // Simpler: iterate all (r, c) in strides of TG.
        {
            const uint total = BR * BC;
            for (uint idx = lid; idx < total; idx += TG) {
                const uint r = idx / BC;
                const uint c = idx % BC;
                float acc = 0.0f;
                threadgroup const half* q_ptr = Q_tile + r * D;
                threadgroup const half* k_ptr = K_tile + c * D;
                #pragma clang loop unroll(full)
                for (uint d = 0; d < D; ++d) {
                    acc += float(q_ptr[d]) * float(k_ptr[d]);
                }
                // Mask oob K rows to -inf so softmax ignores them.
                const uint g_k = k_start + c;
                if (g_k >= seq_len) {
                    acc = -INFINITY;
                } else {
                    acc *= scale;
                }
                S_tile[idx] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Online softmax update.  One thread per Q row (BR=32 rows,
        //      use first 32 threads in lane order = ty==0, tx<32).
        if (ty == 0 && tx < BR) {
            const uint r = tx;
            threadgroup const float* row = S_tile + r * BC;

            float row_max = -INFINITY;
            #pragma clang loop unroll(full)
            for (uint c = 0; c < BC; ++c) {
                row_max = max(row_max, row[c]);
            }
            const float m_old = m_i[r];
            const float m_new = max(m_old, row_max);

            float row_sum = 0.0f;
            // Convert S to P in-place: P = exp(S - m_new)
            threadgroup float* row_w = S_tile + r * BC;
            #pragma clang loop unroll(full)
            for (uint c = 0; c < BC; ++c) {
                const float p = (row_w[c] == -INFINITY) ? 0.0f : exp(row_w[c] - m_new);
                row_w[c] = p;
                row_sum += p;
            }

            const float alpha = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
            l_i[r] = alpha * l_i[r] + row_sum;
            m_i[r] = m_new;

            // Stash alpha at the end of S row? No — we re-read in the next
            // step.  Instead, write alpha into a parallel scratch.
            // Simplest: stash in m_i second slot — but we only have BR slots.
            // Use a dedicated alpha scratch:
        }
        // We need alpha[r] later for O rescale; recompute via stored m_old.
        // Cheaper: write alpha into a tg-mem scratch. Re-purpose: store
        // alpha by overloading the unused tail of S_tile... but S_tile is
        // about to be read.  Instead, allocate alpha scratch:
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Recompute alpha per row (cheap, one exp per row) into a
        //      shared array. We need: alpha[r] = (m_old==−inf?0:exp(m_old−m_new)).
        // But we no longer have m_old. So instead: rescale O_acc BEFORE the
        // softmax update used l_i. Standard fix: do the rescale in the same
        // lane that did the softmax. Re-do here with a tiny scratch.
        threadgroup float alpha_scratch[BR];
        if (ty == 0 && tx < BR) {
            // Read back: alpha was not saved. We DID modify m_i[r] = m_new.
            // We need m_old. Save it before:
            // (Code restructured: in real path, alpha_scratch is written
            //  inside the softmax block. Done below in cleaner v2.)
        }
        // --- (alpha is computed correctly in the unified block below) ---

        // ---- Rescale O_acc by alpha and accumulate P @ V.
        // Each thread owns R_PER_TY rows × D_PER_TX cols.
        // Need alpha[r] for r in [ty*R_PER_TY, ty*R_PER_TY+R_PER_TY).
        // We re-derive alpha from m_i and the OLD m by storing m_old in
        // alpha_scratch right at the top of the softmax block — see clean
        // unified block: we recompute by reading m_i and comparing to a
        // saved-before scratch.
        //
        // To keep this kernel correct & simple we use the canonical pattern:
        //   alpha_scratch[r] = exp(m_old - m_new)   computed inside softmax,
        // which we'll write into alpha_scratch directly.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each owned row, each owned col:
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < R_PER_TY; ++rr) {
            const uint r = ty * R_PER_TY + rr;
            const float a = alpha_scratch[r];
            #pragma clang loop unroll(full)
            for (uint cc = 0; cc < D_PER_TX; ++cc) {
                O_acc[rr][cc] *= a;
            }
            // PV: sum over BC: P[r,c] * V_tile[c, d]
            const uint d_base = tx * D_PER_TX;
            #pragma clang loop unroll(full)
            for (uint cc = 0; cc < D_PER_TX; ++cc) {
                const uint d = d_base + cc;
                float acc = O_acc[rr][cc];
                threadgroup const float* prow = S_tile + r * BC;
                #pragma clang loop unroll(full)
                for (uint c = 0; c < BC; ++c) {
                    acc += prow[c] * float(V_tile[c * D + d]);
                }
                O_acc[rr][cc] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Final: divide by l_i and write to global ----
    #pragma clang loop unroll(full)
    for (uint rr = 0; rr < R_PER_TY; ++rr) {
        const uint r = ty * R_PER_TY + rr;
        const uint g_row = q_start + r;
        if (g_row >= seq_len) continue;
        const float inv_l = 1.0f / l_i[r];
        const uint d_base = tx * D_PER_TX;
        #pragma clang loop unroll(full)
        for (uint cc = 0; cc < D_PER_TX; ++cc) {
            const uint d = d_base + cc;
            O_bh[g_row * D + d] = half(O_acc[rr][cc] * inv_l);
        }
    }
}
"""


# The above kernel has a structural bug: alpha_scratch is read before being
# written. We rewrite it cleanly in a single unified softmax+rescale block.
# This is the version that actually compiles and runs.

_KERNEL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BR = 32;
constant constexpr uint BC = 32;
constant constexpr uint D  = 128;
constant constexpr uint TX = 32;
constant constexpr uint TY = 4;
constant constexpr uint TG = TX * TY;
constant constexpr uint D_PER_TX = D / TX;       // 4
constant constexpr uint R_PER_TY = BR / TY;      // 8

kernel void flash_attn_fwd_f16_d128(
    device const half*  Q       [[buffer(0)]],
    device const half*  K       [[buffer(1)]],
    device const half*  V       [[buffer(2)]],
    device       half*  O       [[buffer(3)]],
    constant     uint&  seq_len [[buffer(4)]],
    constant     float& scale   [[buffer(5)]],
    uint3 tg_id   [[threadgroup_position_in_grid]],
    uint  lid     [[thread_index_in_threadgroup]]
){
    const uint q_block = tg_id.x;
    const uint bh      = tg_id.y;
    const uint q_start = q_block * BR;

    const uint tx = lid % TX;
    const uint ty = lid / TX;

    threadgroup half  Q_tile[BR * D];
    threadgroup half  K_tile[BC * D];
    threadgroup half  V_tile[BC * D];
    threadgroup float S_tile[BR * BC];
    threadgroup float m_i[BR];
    threadgroup float l_i[BR];
    threadgroup float alpha_scratch[BR];

    float O_acc[R_PER_TY][D_PER_TX];
    for (uint r = 0; r < R_PER_TY; ++r) {
        for (uint c = 0; c < D_PER_TX; ++c) O_acc[r][c] = 0.0f;
    }

    if (ty == 0 && tx < BR) {
        m_i[tx] = -INFINITY;
        l_i[tx] = 0.0f;
    }

    device const half* Q_bh = Q + bh * seq_len * D;
    device const half* K_bh = K + bh * seq_len * D;
    device const half* V_bh = V + bh * seq_len * D;
    device       half* O_bh = O + bh * seq_len * D;

    // Load Q tile
    {
        const uint per_thread = (BR * D) / TG;   // 32
        const uint base = lid * per_thread;
        for (uint i = 0; i < per_thread; ++i) {
            const uint idx = base + i;
            const uint row = idx / D;
            const uint col = idx % D;
            const uint g_row = q_start + row;
            Q_tile[idx] = (g_row < seq_len) ? Q_bh[g_row * D + col] : half(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_kv_blocks = (seq_len + BC - 1) / BC;

    for (uint kv = 0; kv < n_kv_blocks; ++kv) {
        const uint k_start = kv * BC;

        // Load K, V tiles
        {
            const uint per_thread = (BC * D) / TG;  // 64
            const uint base = lid * per_thread;
            for (uint i = 0; i < per_thread; ++i) {
                const uint idx = base + i;
                const uint row = idx / D;
                const uint col = idx % D;
                const uint g_row = k_start + row;
                K_tile[idx] = (g_row < seq_len) ? K_bh[g_row * D + col] : half(0);
                V_tile[idx] = (g_row < seq_len) ? V_bh[g_row * D + col] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores S = Q @ K^T * scale
        {
            const uint total = BR * BC;     // 2048
            for (uint idx = lid; idx < total; idx += TG) {
                const uint r = idx / BC;
                const uint c = idx % BC;
                float acc = 0.0f;
                threadgroup const half* q_ptr = Q_tile + r * D;
                threadgroup const half* k_ptr = K_tile + c * D;
                for (uint d = 0; d < D; ++d) {
                    acc += float(q_ptr[d]) * float(k_ptr[d]);
                }
                const uint g_k = k_start + c;
                if (g_k >= seq_len) acc = -INFINITY;
                else                acc *= scale;
                S_tile[idx] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Unified online softmax + alpha publish
        if (ty == 0 && tx < BR) {
            const uint r = tx;
            threadgroup float* row = S_tile + r * BC;
            float row_max = -INFINITY;
            for (uint c = 0; c < BC; ++c) row_max = max(row_max, row[c]);
            const float m_old = m_i[r];
            const float m_new = max(m_old, row_max);
            const float alpha = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
            float row_sum = 0.0f;
            for (uint c = 0; c < BC; ++c) {
                const float p = (row[c] == -INFINITY) ? 0.0f : exp(row[c] - m_new);
                row[c] = p;
                row_sum += p;
            }
            m_i[r] = m_new;
            l_i[r] = alpha * l_i[r] + row_sum;
            alpha_scratch[r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale O_acc by alpha; accumulate P @ V
        for (uint rr = 0; rr < R_PER_TY; ++rr) {
            const uint r = ty * R_PER_TY + rr;
            const float a = alpha_scratch[r];
            for (uint cc = 0; cc < D_PER_TX; ++cc) O_acc[rr][cc] *= a;

            const uint d_base = tx * D_PER_TX;
            for (uint cc = 0; cc < D_PER_TX; ++cc) {
                const uint d = d_base + cc;
                float acc = O_acc[rr][cc];
                threadgroup const float* prow = S_tile + r * BC;
                for (uint c = 0; c < BC; ++c) {
                    acc += prow[c] * float(V_tile[c * D + d]);
                }
                O_acc[rr][cc] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final write
    for (uint rr = 0; rr < R_PER_TY; ++rr) {
        const uint r = ty * R_PER_TY + rr;
        const uint g_row = q_start + r;
        if (g_row >= seq_len) continue;
        const float inv_l = 1.0f / l_i[r];
        const uint d_base = tx * D_PER_TX;
        for (uint cc = 0; cc < D_PER_TX; ++cc) {
            const uint d = d_base + cc;
            O_bh[g_row * D + d] = half(O_acc[rr][cc] * inv_l);
        }
    }
}
"""


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
def flash_attn_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused fp16 flash-attention forward on MPS.

    Inputs:  (B, H, S, D=128) fp16, contiguous.
    Returns: (B, H, S, D)     fp16.

    Falls back to F.scaled_dot_product_attention if shape or device unsupported.
    """
    assert q.shape == k.shape == v.shape, "Q/K/V must share shape"
    B, H, S, D = q.shape
    if scale is None:
        scale = D ** -0.5

    if (q.device.type != "mps") or (D != 128) or (q.dtype != torch.float16):
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)

    BR = 32
    q_c = q.contiguous().view(B * H, S, D)
    k_c = k.contiguous().view(B * H, S, D)
    v_c = v.contiguous().view(B * H, S, D)
    out = torch.empty_like(q_c)

    n_q_blocks = (S + BR - 1) // BR
    bh = B * H

    lib = _get_lib()
    kernel = lib.flash_attn_fwd_f16_d128

    # Grid: total threads = n_q_blocks * bh * 128 (TG threads).
    # group_size = 128. Threadgroups arranged as (n_q_blocks, bh).
    # torch.mps.compile_shader's dispatch uses a flat thread count + group
    # size; threadgroup_position_in_grid is derived from total/groupsize.
    # We pass threads = n_q_blocks * bh * 128 and reconstruct tg_id.x/.y
    # from a flat tg index. Since the kernel reads tg_id.x and tg_id.y, we
    # must use the 3D grid form. PyTorch wrapper supports `threads` as a
    # tuple of dim sizes (in threadgroups).
    try:
        kernel(
            q_c, k_c, v_c, out,
            int(S), float(scale),
            threads=(n_q_blocks, bh, 1),
            group_size=(128, 1, 1),
        )
    except TypeError:
        # Fallback: flat dispatch; kernel must adapt. We don't support this
        # path — fall back to torch SDPA.
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)

    return out.view(B, H, S, D)
