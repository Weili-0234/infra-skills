# Compute-Intensive Operations

This file contains examples of compute-intensive operations including batch matrix multiplication and Flash Attention implementations.

## Table of Contents
- [Example 6: Batch Matrix Multiply](#example-6-batch-matrix-multiply-3d-indexing)
- [Example 11: Flash Attention (Online Softmax)](#example-11-flash-attention-online-softmax)
- [Example 12: Flash Attention (Production)](#example-12-flash-attention-with-running-max-production)
- [Example 14: Matrix Multiplication (Optimized)](#example-14-matrix-multiplication-optimized)

**See also**: [EXAMPLES.md](EXAMPLES.md) for basic patterns

---

## Example 6: Batch Matrix Multiply (3D Indexing)

**Purpose**: Compute batched matrix multiplication: `C[b] = A[b] @ B[b]` for all batches.

**Key Techniques**: 3D indexing, `ct.mma`, batch parallelism, tile reshaping

```python
import torch
import cuda.tile as ct
from math import ceil

@ct.kernel
def batch_matmul_kernel(
    A: ct.Array, B: ct.Array, C: ct.Array,
    tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]
):
    """Batch matrix multiplication: C[b, i, j] = sum_k A[b, i, k] * B[b, k, j]."""
    batch_idx = ct.bid(0)
    tile_m = ct.bid(1)
    tile_n = ct.bid(2)

    # Initialize accumulator
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)

    # K-dimension tiling loop
    num_k_tiles = ct.cdiv(A.shape[2], tk)
    for k in range(num_k_tiles):
        # Load A tile: shape (1, tm, tk) -> reshape to (tm, tk)
        tile_a = ct.load(
            A, index=(batch_idx, tile_m, k),
            shape=(1, tm, tk),
            padding_mode=ct.PaddingMode.ZERO
        )
        tile_a = tile_a.reshape((tm, tk))

        # Load B tile: shape (1, tk, tn) -> reshape to (tk, tn)
        tile_b = ct.load(
            B, index=(batch_idx, k, tile_n),
            shape=(1, tk, tn),
            padding_mode=ct.PaddingMode.ZERO
        )
        tile_b = tile_b.reshape((tk, tn))

        # Matrix multiply-accumulate
        acc = ct.mma(tile_a, tile_b, acc)

    # Store result: reshape to (1, tm, tn)
    result = acc.astype(C.dtype).reshape((1, tm, tn))
    ct.store(C, index=(batch_idx, tile_m, tile_n), tile=result)

def batch_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batch matrix multiplication wrapper."""
    Batch, M, K = A.shape
    _, K_b, N = B.shape
    assert K == K_b, f"K dimension mismatch: {K} vs {K_b}"

    # Tile sizes optimized for Tensor Cores
    tm, tn, tk = 128, 256, 64

    C = torch.empty((Batch, M, N), device=A.device, dtype=A.dtype)

    grid = (Batch, ceil(M / tm), ceil(N / tn))

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        batch_matmul_kernel,
        (A, B, C, tm, tn, tk)
    )

    return C

# Usage
Batch, M, K, N = 8, 512, 256, 1024
A = torch.randn(Batch, M, K, device="cuda", dtype=torch.float16)
B = torch.randn(Batch, K, N, device="cuda", dtype=torch.float16)

C = batch_matmul(A, B)

# Validation
C_torch = torch.bmm(A, B)
torch.testing.assert_close(C, C_torch, rtol=1e-2, atol=1e-2)
```

---

## Example 11: Flash Attention (Online Softmax)

**Purpose**: Memory-efficient attention using online softmax algorithm (without running max).

**Key Techniques**: Online softmax, `ct.mma`, numerical stability, attention computation

```python
import torch
import cuda.tile as ct
import math

INV_LOG_2 = 1.0 / math.log(2)

@ct.kernel
def flash_attention_kernel(
    Q: ct.Array, K: ct.Array, V: ct.Array, output: ct.Array,
    kv_len: int, head_dim: ct.Constant[int],
    TILE_Q: ct.Constant[int], TILE_KV: ct.Constant[int]
):
    """Flash Attention with online softmax (no running max)."""
    q_idx = ct.bid(0)
    head_idx = ct.bid(1)

    # Load query tile
    tile_q = ct.load(
        Q, (q_idx, head_idx, 0),
        (TILE_Q, 1, head_dim),
        padding_mode=ct.PaddingMode.ZERO
    )
    tile_q = tile_q * INV_LOG_2  # Convert to log2 space
    tile_q = tile_q.reshape((TILE_Q, head_dim))

    # Initialize accumulators
    acc = ct.full((TILE_Q, head_dim), 0.0, dtype=ct.float32)
    logits_sum = ct.full((TILE_Q, 1), 0.0, dtype=ct.float32)

    # Iterate over K/V tiles
    num_kv_tiles = ct.cdiv(kv_len, TILE_KV)
    for kv_iter in range(num_kv_tiles):
        # Compute QK
        tile_k = ct.load(
            K, (kv_iter, head_idx, 0),
            (TILE_KV, 1, head_dim),
            padding_mode=ct.PaddingMode.ZERO
        )
        tile_k = tile_k.transpose(0, 2).reshape((head_dim, TILE_KV))

        logits = ct.full((TILE_Q, TILE_KV), 0.0, dtype=ct.float32)
        logits = ct.mma(tile_q, tile_k, logits)

        # Mask out-of-range KV positions
        kv_positions = ct.arange(TILE_KV, dtype=ct.int32) + kv_iter * TILE_KV
        mask = ct.where(kv_positions < kv_len, 0.0, -10000000.0)
        logits = logits + mask.reshape((1, TILE_KV))

        # Compute attention weights (exp without max subtraction)
        logits = ct.exp2(logits, flush_to_zero=True)
        logits_sum = logits_sum + ct.sum(logits, axis=1, keepdims=True)

        # Accumulate weighted values
        tile_v = ct.load(
            V, (kv_iter, head_idx, 0),
            (TILE_KV, 1, head_dim),
            padding_mode=ct.PaddingMode.ZERO
        ).reshape((TILE_KV, head_dim))

        acc = ct.mma(logits.astype(tile_v.dtype), tile_v, acc)

    # Final normalization
    acc = ct.truediv(acc, logits_sum, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)

    # Store output
    acc = acc.reshape((TILE_Q, 1, head_dim))
    ct.store(output, (q_idx, head_idx, 0), acc.astype(output.dtype))

def flash_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    tile_q: int = 64, tile_kv: int = 128
) -> torch.Tensor:
    """
    Flash Attention wrapper.

    Args:
        Q: Query tensor [q_len, num_heads, head_dim]
        K: Key tensor [kv_len, num_heads, head_dim]
        V: Value tensor [kv_len, num_heads, head_dim]

    Returns:
        Output tensor [q_len, num_heads, head_dim]
    """
    q_len, num_heads, head_dim = Q.shape
    kv_len = K.shape[0]

    output = torch.empty_like(Q)

    grid = (ct.cdiv(q_len, tile_q), num_heads)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        flash_attention_kernel,
        (Q, K, V, output, kv_len, head_dim, tile_q, tile_kv)
    )

    return output

# Usage
num_heads, q_len, kv_len, head_dim = 32, 128, 512, 64
Q = torch.randn(q_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
K = torch.randn(kv_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
V = torch.randn(kv_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)

output = flash_attention(Q, K, V)

print(f"Flash Attention output shape: {output.shape}")
```

---

## Example 12: Flash Attention with Running Max (Production)

**Purpose**: Numerically stable Flash Attention with online max tracking and rescaling.

**Key Techniques**: Running max, exponential rescaling, causal masking, online softmax

```python
import torch
import cuda.tile as ct
import math
import numpy as np

INV_LOG_2 = 1.0 / math.log(2)

@ct.kernel
def fmha_kernel(
    Q: ct.Array, K: ct.Array, V: ct.Array, output: ct.Array,
    qk_scale: float, input_pos: int,
    TILE_D: ct.Constant[int], TILE_M: ct.Constant[int], TILE_N: ct.Constant[int],
    CAUSAL: ct.Constant[bool]
):
    """Flash Attention with running max and causal masking."""
    bid_x = ct.bid(0)
    head_idx = ct.bid(1)

    # Adjust scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize running statistics
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)  # Running max
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)      # Running sum
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32) # Output accumulator

    # Load query tile
    q = ct.load(
        Q, index=(head_idx, bid_x, 0),
        shape=(1, TILE_M, TILE_D)
    ).reshape((TILE_M, TILE_D))

    # Compute iteration bounds
    k_seqlen = K.shape[1]
    m_end = input_pos + (bid_x + 1) * TILE_M

    if CAUSAL:
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Iterate over K/V tiles
    for j in range(Tc):
        # Load K tile
        k = ct.load(
            K, index=(head_idx, 0, j),
            shape=(1, TILE_D, TILE_N),
            order=(0, 2, 1)
        ).reshape((TILE_D, TILE_N))

        # Compute QK
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
        qk = ct.mma(q, k, qk)

        # Apply causal mask if needed
        if CAUSAL and j >= mask_start:
            offs_m = (bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32) + input_pos)[:, None]
            offs_n = (j * TILE_N + ct.arange(TILE_N, dtype=np.int32))[None, :]
            mask = ct.where(offs_m >= offs_n, 0.0, -np.inf)
            qk = qk + mask

        # Update running max
        m_ij = ct.max(ct.max(qk, axis=-1, keepdims=True) * qk_scale, m_i)
        qk = qk * qk_scale - m_ij

        # Compute attention weights
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)

        # Rescale previous accumulator
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V and accumulate
        v = ct.load(
            V, index=(head_idx, j, 0),
            shape=(1, TILE_N, TILE_D)
        ).reshape((TILE_N, TILE_D))

        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)

        # Update running max
        m_i = m_ij

    # Final normalization
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)
    acc = acc.reshape((1, TILE_M, TILE_D)).astype(output.dtype)
    ct.store(output, index=(head_idx, bid_x, 0), tile=acc)

def fmha(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    qk_scale: float = None, input_pos: int = 0,
    tile_m: int = 128, tile_n: int = 128, causal: bool = False
) -> torch.Tensor:
    """
    Flash Multi-Head Attention.

    Args:
        Q: [num_heads, seq_len_q, head_dim]
        K: [num_heads, seq_len_kv, head_dim]
        V: [num_heads, seq_len_kv, head_dim]
        qk_scale: Attention scale (default: 1/sqrt(head_dim))
        input_pos: Starting position for causal masking
        causal: Apply causal masking
    """
    num_heads, seq_len_q, head_dim = Q.shape

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(head_dim)

    output = torch.empty_like(Q)

    grid = (ct.cdiv(seq_len_q, tile_m), num_heads)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fmha_kernel,
        (Q, K, V, output, qk_scale, input_pos, head_dim, tile_m, tile_n, causal)
    )

    return output

# Usage
num_heads, seq_len, head_dim = 32, 1024, 64
Q = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
K = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
V = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)

# Causal attention
output = fmha(Q, K, V, causal=True)
print(f"FMHA output shape: {output.shape}")
```

---

## Example 14: Matrix Multiplication (Optimized)

**Purpose**: High-performance matrix multiplication with tiling and Tensor Core optimization.

**Key Techniques**: 2D tiling, `ct.mma`, K-dimension loop, padding

```python
import torch
import cuda.tile as ct
from math import ceil

@ct.kernel
def matmul_kernel(
    A: ct.Array, B: ct.Array, C: ct.Array,
    tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]
):
    """Optimized matrix multiplication kernel."""
    tile_m = ct.bid(0)
    tile_n = ct.bid(1)

    # Initialize accumulator in float32 for precision
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)

    # K-dimension tiling
    num_k_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    for k in range(num_k_tiles):
        # Load A tile: (tm, tk)
        a = ct.load(
            A, index=(tile_m, k),
            shape=(tm, tk),
            padding_mode=ct.PaddingMode.ZERO
        )

        # Load B tile: (tk, tn)
        b = ct.load(
            B, index=(k, tile_n),
            shape=(tk, tn),
            padding_mode=ct.PaddingMode.ZERO
        )

        # Matrix multiply-accumulate using Tensor Cores
        acc = ct.mma(a, b, acc)

    # Store result
    result = acc.astype(C.dtype)
    ct.store(C, index=(tile_m, tile_n), tile=result)

def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Optimized matrix multiplication."""
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, f"K dimension mismatch: {K} vs {K_b}"

    # Tile sizes optimized for Tensor Cores
    tm, tn, tk = 128, 256, 64

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = (ceil(M / tm), ceil(N / tn))

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        matmul_kernel,
        (A, B, C, tm, tn, tk)
    )

    return C

# Usage
M, K, N = 2048, 1024, 4096
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

C = matmul(A, B)

# Validation
C_torch = A @ B
torch.testing.assert_close(C, C_torch, rtol=1e-2, atol=1e-2)

print(f"MatMul completed: {M}x{K} @ {K}x{N} = {M}x{N}")
```

---

**See also**:
- [EXAMPLES.md](EXAMPLES.md) - Quick start guide
- [EXAMPLES_NORMALIZATION.md](EXAMPLES_NORMALIZATION.md) - Normalization operations
- [EXAMPLES_ELEMENTWISE.md](EXAMPLES_ELEMENTWISE.md) - Element-wise operations
- [EXAMPLES_TRAINING.md](EXAMPLES_TRAINING.md) - Training operations
