# Attention Kernel Examples

Production-quality Triton implementations of attention mechanisms. Covers Flash Attention v2, block-pointer variants, Grouped Query Attention, and Split-KV decode attention.

---

## Table of Contents

1. [Flash Attention v2 Forward](#1-flash-attention-v2-forward) -- Online softmax, causal masking, tiled QKV
2. [Flash Attention with Block Pointers](#2-flash-attention-with-block-pointers) -- Cleaner memory access via `tl.make_block_ptr`
3. [Grouped Query Attention (GQA)](#3-grouped-query-attention-gqa) -- Multiple Q heads sharing KV heads
4. [Split-KV Attention for Decode](#4-split-kv-attention-for-decode) -- Parallel KV splits for autoregressive inference

---

## 1. Flash Attention v2 Forward

**Pattern overview:** See [PATTERNS.md §2.3 — Online Softmax](PATTERNS.md#pattern-3-online-softmax-flash-attention-style) for when/why to use this pattern and the core recurrence.

Standard Flash Attention v2 with online softmax. The outer loop iterates over KV blocks;
the inner body computes QK^T, applies online softmax rescaling, and accumulates P @ V.
Causal masking skips fully-masked KV blocks and applies a triangular mask on partial blocks.

**Online softmax recurrence:**
```
m_i_new = max(m_i, rowmax(qk))
alpha   = exp(m_i - m_i_new)       # rescale factor for old accumulator
p       = exp(qk - m_i_new)        # attention weights for current block
acc     = acc * alpha + p @ v       # rescaled accumulator + new contribution
l_i     = l_i * alpha + rowsum(p)   # rescaled normalizer + new weights
```

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr, offs_n: tl.constexpr, N_CTX: tl.constexpr,
):
    """Inner loop over KV blocks.
    STAGE 1: causal, before diagonal (fully unmasked)
    STAGE 2: causal, on the diagonal (triangular mask)
    STAGE 3: non-causal, all KV blocks
    """
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)  # [HEAD_DIM, BLOCK_N]

        # QK^T: [BLOCK_M, HEAD_DIM] @ [HEAD_DIM, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * qk_scale

        if STAGE == 2:  # Triangular causal mask on diagonal block
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(V_block_ptr)  # [BLOCK_N, HEAD_DIM]
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    """Flash Attention v2 forward. Grid: (cdiv(N_CTX, BLOCK_M), H * Z)
    Q, K, V: [Z, H, N_CTX, HEAD_DIM].  STAGE=1 for causal, STAGE=3 for non-causal.
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Block pointers for Q, K (transposed), V, and output
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize online softmax state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Pre-multiply by log2(e) so we use exp2 (faster on GPU)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)

    # Causal: run stage 1 (before diagonal) then stage 2 (on diagonal)
    # Non-causal: run stage 3 only
    if STAGE == 1 or STAGE == 3:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, qk_scale, BLOCK_M, BLOCK_N, HEAD_DIM,
            STAGE if STAGE == 3 else 1, offs_m, offs_n, N_CTX)
    if STAGE == 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, qk_scale, BLOCK_M, BLOCK_N, HEAD_DIM,
            2, offs_m, offs_n, N_CTX)

    # Normalize and store
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def flash_attention_v2(q, k, v, causal=False):
    """Launch Flash Attention v2. q/k/v: [batch, heads, seq_len, head_dim] float16."""
    Z, H, N_CTX, HEAD_DIM = q.shape
    out = torch.empty_like(q)
    M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    grid = (triton.cdiv(N_CTX, 128), H * Z, 1)
    _attn_fwd[grid](
        q, k, v, sm_scale, M, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=128, BLOCK_N=64, HEAD_DIM=HEAD_DIM,
        STAGE=1 if causal else 3)
    return out
```

---

## 2. Flash Attention with Block Pointers

Same algorithm as Section 1, but uses `tl.make_block_ptr` and `tl.advance` throughout for
cleaner pointer management. Block pointers encode shape, strides, and offsets in a single
object, enabling the compiler to emit optimized TMA instructions on Hopper GPUs.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_block_ptr(
    Q, K, V, Out, sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    """Flash Attention forward with block pointers. Grid: (cdiv(N_CTX, BLOCK_M), H * Z)"""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    base_off = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    q_ptr = tl.make_block_ptr(
        base=Q + base_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    k_ptr = tl.make_block_ptr(
        base=K + base_off, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    v_ptr = tl.make_block_ptr(
        base=V + base_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    o_ptr = tl.make_block_ptr(
        base=Out + base_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    q = tl.load(q_ptr, boundary_check=(0, 1))
    q = (q * qk_scale).to(tl.float16)

    kv_len = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, kv_len, BLOCK_N):
        k = tl.load(k_ptr, boundary_check=(0, 1))
        v = tl.load(v_ptr, boundary_check=(0, 1))
        qk = tl.dot(q, k)

        if IS_CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)

        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij

        k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(o_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))


def flash_attention_block_ptr(q, k, v, causal=False):
    """Wrapper for block-pointer Flash Attention. q/k/v: [B, H, N, D] float16."""
    Z, H, N_CTX, HEAD_DIM = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    grid = (triton.cdiv(N_CTX, 128), H * Z)
    _attn_fwd_block_ptr[grid](
        q, k, v, out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=128, BLOCK_N=64, HEAD_DIM=HEAD_DIM, IS_CAUSAL=causal)
    return out
```

---

## 3. Grouped Query Attention (GQA)

Multiple query heads share one KV head (e.g., 32 Q heads, 8 KV heads = groups of 4).
The key modification: `kv_head_idx = q_head_idx // num_queries_per_kv`. K/V pointers
use the KV head index while Q/Out use the Q head index.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_attn_fwd(
    Q, K, V, Out, sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H_Q, H_KV, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    """GQA forward. Grid: (cdiv(N_CTX, BLOCK_M), H_Q * Z)"""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_hq = off_hz % H_Q

    # Head remapping: map Q head -> shared KV head
    num_queries_per_kv = H_Q // H_KV
    off_hkv = off_hq // num_queries_per_kv

    # Q/Out use Q-head index; K/V use KV-head index
    q_off = off_z.to(tl.int64) * stride_qz + off_hq.to(tl.int64) * stride_qh
    o_off = off_z.to(tl.int64) * stride_oz + off_hq.to(tl.int64) * stride_oh
    kv_off = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh

    q_ptr = tl.make_block_ptr(
        base=Q + q_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    k_ptr = tl.make_block_ptr(
        base=K + kv_off, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    v_ptr = tl.make_block_ptr(
        base=V + kv_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    o_ptr = tl.make_block_ptr(
        base=Out + o_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    q = tl.load(q_ptr, boundary_check=(0, 1))
    q = (q * qk_scale).to(tl.float16)

    kv_len = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, kv_len, BLOCK_N):
        k = tl.load(k_ptr, boundary_check=(0, 1))
        v = tl.load(v_ptr, boundary_check=(0, 1))
        qk = tl.dot(q, k)

        if IS_CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij

        k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(o_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))


def gqa_attention(q, k, v, causal=False):
    """Launch GQA. q: [B, H_Q, N, D], k/v: [B, H_KV, N, D]. H_Q % H_KV == 0."""
    Z, H_Q, N_CTX, HEAD_DIM = q.shape
    H_KV = k.shape[1]
    assert H_Q % H_KV == 0, f"Q heads ({H_Q}) must be divisible by KV heads ({H_KV})"
    out = torch.empty_like(q)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    grid = (triton.cdiv(N_CTX, 128), H_Q * Z)
    _gqa_attn_fwd[grid](
        q, k, v, out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H_Q, H_KV, N_CTX,
        BLOCK_M=128, BLOCK_N=64, HEAD_DIM=HEAD_DIM, IS_CAUSAL=causal)
    return out
```

---

## 4. Split-KV Attention for Decode

During autoregressive decoding, each step has a single query token attending over a long
KV cache. Standard Flash Attention gives poor GPU utilization (only one Q "block").
Split-KV partitions the KV sequence into chunks processed by separate thread blocks, then
a reduction kernel combines partial results using the log-sum-exp trick.

**Two-kernel approach:**
1. `_splitkv_attn_kernel` -- each program handles one KV split, produces partial output + LSE
2. `_splitkv_reduce_kernel` -- reduces partial outputs across splits via log-sum-exp

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _splitkv_attn_kernel(
    Q, K, V, Out_partial, LSE_partial, sm_scale,
    stride_qz, stride_qh, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_opz, stride_oph, stride_ops, stride_opd,
    stride_lz, stride_lh, stride_ls,
    H, KV_LEN,
    BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    """Partial attention over one KV split. Grid: (NUM_SPLITS, H * Z)
    Q: [Z, H, HEAD_DIM] (seq_len=1 squeezed), K/V: [Z, H, KV_LEN, HEAD_DIM]
    """
    split_idx = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Determine this split's KV range
    split_size = tl.cdiv(KV_LEN, NUM_SPLITS)
    kv_start = split_idx * split_size
    kv_end = tl.minimum(kv_start + split_size, KV_LEN)

    # Load query vector [HEAD_DIM]
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_offset + offs_d * stride_qd) * sm_scale

    # Online softmax state (scalar max for single query token)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh

    for start_n in range(kv_start, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_end

        # Load K block [BLOCK_N, HEAD_DIM]
        k_ptrs = K + kv_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # QK^T via element-wise multiply + reduce: [BLOCK_N]
        qk = tl.sum(q[None, :] * k, 1)
        qk = tl.where(mask_n, qk, -float("inf"))

        # Online softmax
        m_ij = tl.max(qk, 0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
        p = tl.math.exp2((qk - m_new) * 1.44269504)

        # Load V block [BLOCK_N, HEAD_DIM] and accumulate
        v_ptrs = V + kv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new

    # Normalize partial output and compute partial LSE
    acc = acc / l_i
    partial_lse = m_i / 1.44269504 + tl.log(l_i)  # convert log2 -> ln

    # Store partial output [HEAD_DIM] and partial LSE (scalar)
    op_offset = (off_z.to(tl.int64) * stride_opz + off_h.to(tl.int64) * stride_oph
                 + split_idx.to(tl.int64) * stride_ops)
    tl.store(Out_partial + op_offset + offs_d * stride_opd, acc)
    lse_offset = (off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh
                  + split_idx.to(tl.int64) * stride_ls)
    tl.store(LSE_partial + lse_offset, partial_lse)


@triton.jit
def _splitkv_reduce_kernel(
    Out_partial, LSE_partial, Out,
    stride_opz, stride_oph, stride_ops, stride_opd,
    stride_lz, stride_lh, stride_ls,
    stride_oz, stride_oh, stride_od,
    H,
    NUM_SPLITS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Reduce partial outputs across splits. Grid: (H * Z,)
    Uses log-sum-exp trick: weight_i = exp(lse_i - global_lse), out = sum(w_i * partial_i)
    """
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_d = tl.arange(0, HEAD_DIM)

    # Load all partial LSE values [NUM_SPLITS]
    lse_base = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh
    offs_s = tl.arange(0, NUM_SPLITS)
    lse_vals = tl.load(LSE_partial + lse_base + offs_s * stride_ls)

    # Compute weights from LSE values
    max_lse = tl.max(lse_vals, 0)
    exp_lse = tl.exp(lse_vals - max_lse)
    weights = exp_lse / tl.sum(exp_lse, 0)

    # Weighted sum of partial outputs
    op_base = off_z.to(tl.int64) * stride_opz + off_h.to(tl.int64) * stride_oph
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        partial = tl.load(Out_partial + op_base + s * stride_ops + offs_d * stride_opd)
        acc = acc + partial * weights[s]

    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    tl.store(Out + o_offset + offs_d * stride_od, acc)


def splitkv_decode_attention(q, k, v, num_splits=8):
    """Split-KV attention for single-token decode.
    q: [B, H, 1, D], k/v: [B, H, kv_len, D]. Returns [B, H, 1, D].
    """
    Z, H, _, HEAD_DIM = q.shape
    KV_LEN = k.shape[2]
    q_squeezed = q.squeeze(2)  # [Z, H, HEAD_DIM]

    out_partial = torch.empty((Z, H, num_splits, HEAD_DIM), device=q.device, dtype=torch.float32)
    lse_partial = torch.empty((Z, H, num_splits), device=q.device, dtype=torch.float32)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    # Kernel 1: partial attention per KV split
    _splitkv_attn_kernel[(num_splits, H * Z)](
        q_squeezed, k, v, out_partial, lse_partial, sm_scale,
        q_squeezed.stride(0), q_squeezed.stride(1), q_squeezed.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out_partial.stride(0), out_partial.stride(1), out_partial.stride(2), out_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        H, KV_LEN,
        BLOCK_N=128, HEAD_DIM=HEAD_DIM, NUM_SPLITS=num_splits)

    # Kernel 2: reduce partial results
    out_squeezed = torch.empty((Z, H, HEAD_DIM), device=q.device, dtype=q.dtype)
    _splitkv_reduce_kernel[(H * Z,)](
        out_partial, lse_partial, out_squeezed,
        out_partial.stride(0), out_partial.stride(1), out_partial.stride(2), out_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        out_squeezed.stride(0), out_squeezed.stride(1), out_squeezed.stride(2),
        H,
        NUM_SPLITS=num_splits, HEAD_DIM=HEAD_DIM)

    return out_squeezed.unsqueeze(2)  # Restore seq_len=1 dimension
```
