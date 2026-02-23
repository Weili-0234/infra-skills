# Advanced Examples

Advanced Triton kernel examples: Hopper+ features, multi-problem batching, FP8, and cross-block coordination. Hardware requirements annotated per section.

---

## Table of Contents

1. [Persistent Matrix Multiplication with TMA](#1-persistent-matrix-multiplication-with-tma) -- Hopper+ persistent kernel with TMA
2. [Grouped GEMM](#2-grouped-gemm) -- Multiple variable-size matmuls in one launch
3. [FP8 / Block-Scaled MatMul](#3-fp8--block-scaled-matmul) -- FP8 types and microscaling
4. [Warp Specialization](#4-warp-specialization-hopper) -- Producer-consumer on Hopper+
5. [Associative Scan](#5-associative-scan) -- Parallel prefix operations
6. [Lock-Based Cross-Block Reduction](#6-lock-based-cross-block-reduction) -- Split-K and inter-block coordination

---

## 1. Persistent Matrix Multiplication with TMA

**Hardware:** Hopper+ (compute capability >= 9.0)

A persistent kernel launches fewer programs than total tiles -- each program loops over
multiple tiles. Combined with TMA (Tensor Memory Accelerator), memory transfers are
handled by dedicated hardware, freeing warps for computation.

**Key concepts:**
- Persistent scheduling: `tl.num_programs(0)` programs share all tiles via strided loop
- TMA descriptors via `tl.make_tensor_descriptor`; `desc.load()` / `desc.store()` for transfers
- `tl.range(..., num_stages=N)` for software pipelining
- Grid sized to device SM count

Based on official Triton tutorial 09.

### Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_persistent_tma(
    a_desc, b_desc, c_desc,  # TMA descriptors for A [M,K], B [K,N], C [M,N]
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_tiles_m * num_tiles_n

    # Persistent loop: each program handles multiple tiles in strided fashion
    for tile_id in tl.range(pid, num_tiles, tl.num_programs(0)):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # num_stages=3: overlap 3 iterations of loads with compute via TMA
        for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=3):
            a = a_desc.load([tile_m * BLOCK_M, k * BLOCK_K])
            b = b_desc.load([k * BLOCK_K, tile_n * BLOCK_N])
            acc = tl.dot(a, b, acc)

        c_desc.store([tile_m * BLOCK_M, tile_n * BLOCK_N], acc.to(tl.float16))
```

### Wrapper

```python
def matmul_persistent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

    # TMA descriptors encode pointer, shape, strides, and block shape
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        a.data_ptr(), M, K, BLOCK_M, BLOCK_K, a.element_size(),
    )
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        b.data_ptr(), K, N, BLOCK_K, BLOCK_N, b.element_size(),
    )
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        c.data_ptr(), M, N, BLOCK_M, BLOCK_N, c.element_size(),
    )

    # One persistent program per SM
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    matmul_persistent_tma[(NUM_SMS,)](
        desc_a, desc_b, desc_c, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
```

**Why persistent + TMA?** Standard kernels suffer wave-quantization when tile counts
don't evenly divide SM counts. Persistent kernels reuse SMs. TMA offloads address
computation and data movement to dedicated hardware.

---

## 2. Grouped GEMM

**Hardware:** Any CUDA GPU (benefits from Hopper+ for large problem counts)

Process multiple independent matmuls of different sizes in one kernel launch, avoiding
per-problem launch overhead. Each program iterates over a global tile index space with
device-side scheduling. Based on official Triton tutorial 08.

### Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def grouped_gemm_kernel(
    a_ptrs, b_ptrs, c_ptrs,  # Arrays of base pointers (one per problem)
    ms, ns, ks,               # Arrays of M, N, K dimensions per problem
    tile_starts,               # Cumulative tile counts (exclusive prefix sum)
    num_problems, total_tiles,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_sms = tl.num_programs(0)

    tile_idx = pid
    while tile_idx < total_tiles:
        # Linear scan to find which problem owns this tile
        problem_id = 0
        while problem_id < num_problems - 1:
            next_start = tl.load(tile_starts + problem_id + 1)
            if tile_idx < next_start:
                break
            problem_id += 1

        # Load problem-specific dimensions and pointers from device arrays
        M = tl.load(ms + problem_id)
        N = tl.load(ns + problem_id)
        K = tl.load(ks + problem_id)
        a_ptr = tl.load(a_ptrs + problem_id).to(tl.pointer_type(tl.float16))
        b_ptr = tl.load(b_ptrs + problem_id).to(tl.pointer_type(tl.float16))
        c_ptr = tl.load(c_ptrs + problem_id).to(tl.pointer_type(tl.float16))

        # Global tile -> local tile coordinates within this problem
        local_tile = tile_idx - tl.load(tile_starts + problem_id)
        num_tiles_n = tl.cdiv(N, BLOCK_N)
        tile_m = local_tile // num_tiles_n
        tile_n = local_tile % num_tiles_n

        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + (offs_k[:, None] * N + offs_n[None, :]),
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)

        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                 acc.to(tl.float16), mask=c_mask)
        tile_idx += num_sms
```

### Wrapper

```python
def grouped_gemm(a_list, b_list):
    num_problems = len(a_list)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    device = a_list[0].device

    c_list, tile_counts = [], []
    for a, b in zip(a_list, b_list):
        M, K = a.shape; _, N = b.shape
        c_list.append(torch.empty(M, N, device=device, dtype=torch.float16))
        tile_counts.append(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))

    # Build device arrays
    import itertools
    a_ptrs_d = torch.tensor([a.data_ptr() for a in a_list], dtype=torch.int64, device=device)
    b_ptrs_d = torch.tensor([b.data_ptr() for b in b_list], dtype=torch.int64, device=device)
    c_ptrs_d = torch.tensor([c.data_ptr() for c in c_list], dtype=torch.int64, device=device)
    ms_d = torch.tensor([a.shape[0] for a in a_list], dtype=torch.int32, device=device)
    ns_d = torch.tensor([b.shape[1] for b in b_list], dtype=torch.int32, device=device)
    ks_d = torch.tensor([a.shape[1] for a in a_list], dtype=torch.int32, device=device)
    tile_starts_d = torch.tensor(
        [0] + list(itertools.accumulate(tile_counts[:-1])), dtype=torch.int32, device=device)
    total_tiles = sum(tile_counts)

    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    grouped_gemm_kernel[(min(NUM_SMS, total_tiles),)](
        a_ptrs_d, b_ptrs_d, c_ptrs_d, ms_d, ns_d, ks_d,
        tile_starts_d, num_problems, total_tiles,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c_list
```

---

## 3. FP8 / Block-Scaled MatMul

**Hardware:**
- FP8 matmul: Hopper+ (compute capability >= 9.0)
- `tl.dot_scaled`: Blackwell (compute capability >= 10.0)

FP8 halves memory traffic vs FP16. Two formats:
- **E4M3** (`torch.float8_e4m3fn` / `tl.float8e4nv`): more precision, smaller range
- **E5M2** (`torch.float8_e5m2` / `tl.float8e5`): less precision, larger range

Based on official Triton tutorial 10.

### FP8 Matmul (Hopper+)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_fp8_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    scale_a, scale_b,  # Per-tensor dequantization scales
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=3):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        # FP8 data stays in low precision until tl.dot upcasts internally
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)

    # Dequantize: C = scale_a * scale_b * (A_fp8 @ B_fp8)
    acc = acc * scale_a * scale_b
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float16), mask=c_mask)


def matmul_fp8(a, b, scale_a: float, scale_b: float):
    """a, b: float8_e4m3fn tensors. Returns float16."""
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    matmul_fp8_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        scale_a, scale_b, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
```

### Block-Scaled MatMul with `tl.dot_scaled` (Blackwell)

`tl.dot_scaled` performs hardware-accelerated scaled dot products with per-block
microscaling factors. Supports mixed FP8/FP4 formats.

```python
@triton.jit
def matmul_block_scaled_kernel(
    a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,  # Scale granularity along K (e.g. 128)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_scale_blocks = BLOCK_K // SCALE_BLOCK_K

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Per-block scales: one scale per SCALE_BLOCK_K elements along K
        scale_k_idx = k * num_scale_blocks + tl.arange(0, num_scale_blocks)
        a_scale = tl.load(a_scale_ptr + offs_m[:, None] * tl.cdiv(K, SCALE_BLOCK_K) + scale_k_idx[None, :])
        b_scale = tl.load(b_scale_ptr + scale_k_idx[:, None] * N + offs_n[None, :])

        # Hardware-accelerated scaled dot (Blackwell SM 100+)
        acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float16), mask=c_mask)
```

**Notes:** Format strings for `tl.dot_scaled`: `"e4m3"`, `"e5m2"`, `"e2m1"` (FP4).
The API may evolve across Triton versions.

---

## 4. Warp Specialization (Hopper+)

**Hardware:** Hopper+ (compute capability >= 9.0)

Warp specialization splits warps into **producers** (memory loads) and **consumers**
(computation). Enabled via `tl.range(..., warp_specialize=True)`. The compiler
partitions warps automatically.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_warp_specialized(
    a_desc, b_desc, c_desc,  # TMA descriptors
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    num_tiles = tl.cdiv(M, BLOCK_M) * num_tiles_n

    for tile_id in tl.range(pid, num_tiles, tl.num_programs(0)):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # warp_specialize=True: producer warps prefetch next iteration's
        # data via TMA while consumer warps compute tl.dot on current data
        for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=3, warp_specialize=True):
            a = a_desc.load([tile_m * BLOCK_M, k * BLOCK_K])
            b = b_desc.load([k * BLOCK_K, tile_n * BLOCK_N])
            acc = tl.dot(a, b, acc)

        c_desc.store([tile_m * BLOCK_M, tile_n * BLOCK_N], acc.to(tl.float16))
```

**When to use:** Persistent matmul with TMA where loads can be decoupled from compute;
large tile sizes with enough warps to split between roles.

**When NOT to use:** Simple kernels where `num_stages` suffices; very few warps per
block; non-Hopper hardware (flag is silently ignored).

---

## 5. Associative Scan

**Hardware:** Any CUDA GPU

Parallel prefix scan in O(log n) steps. `tl.associative_scan(input, axis, combine_fn)`
for arbitrary associative operators; `tl.cumsum` / `tl.cumprod` as built-in shorthands.

### Cumulative Sum

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cumsum_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    cumsum = tl.cumsum(x, axis=0)  # Inclusive prefix sum
    tl.store(out_ptr + offsets, cumsum, mask=mask)
```

### Custom Scan: Running Maximum

```python
@triton.jit
def max_combine_fn(a, b):
    return tl.maximum(a, b)

@triton.jit
def running_max_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    running_max = tl.associative_scan(x, axis=0, combine_fn=max_combine_fn)
    tl.store(out_ptr + offsets, running_max, mask=mask)
```

### Multi-Element Scan: Parallel Linear Recurrence

Associative scan supports tuples for multi-element state. Enables parallel computation
of recurrences like `h[t] = a[t] * h[t-1] + b[t]`.

```python
@triton.jit
def linear_recurrence_combine(a1, b1, a2, b2):
    """Composition: (a2*a1, a2*b1 + b2)"""
    return a2 * a1, a2 * b1 + b2

@triton.jit
def linear_recurrence_kernel(
    a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    _, result = tl.associative_scan(
        (a, b), axis=0, combine_fn=linear_recurrence_combine,
    )
    tl.store(out_ptr + offsets, result, mask=mask)
```

**Note:** `tl.associative_scan` operates within a single block. For cross-block scans,
use a multi-pass approach: scan within blocks, scan block totals, propagate prefixes.

---

## 6. Lock-Based Cross-Block Reduction

**Hardware:** Any CUDA GPU

Patterns for when multiple blocks contribute partial results to the same output.

**Key concepts:**
- `tl.atomic_add`: simple accumulation, no locks needed
- `tl.atomic_cas`: spin-locks for read-modify-write sequences
- `tl.debug_barrier()`: CTA-level sync before releasing locks
- Atomic counter to detect the last completing block

### Split-K Matmul with Atomic Accumulation

The K dimension is split across `SPLIT_K` blocks. Each computes a partial sum, then
atomically accumulates into the output.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_split_k_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)  # Which K-split

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(k_end - k_start, BLOCK_K)):
        offs_k = k_start + k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_end), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < k_end) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Multiple blocks write to same output tile -- atomic_add for safe accumulation
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, acc.to(tl.float32), mask=c_mask)
    else:
        tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


def matmul_split_k(a, b, split_k=4):
    M, K = a.shape; _, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)  # Zero-init for accumulation
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
    matmul_split_k_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=split_k,
    )
    return c.to(torch.float16)
```

### Counter-Based Last-Block Reduction

When one block must finalize results after all others complete (e.g., normalization backward).

```python
@triton.jit
def counter_reduction_kernel(
    partial_ptr, output_ptr, counter_ptr,  # counter: single int32, init to 0
    N, num_blocks,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Each block stores its partial result to its own slot
    partial = tl.load(partial_ptr + pid * N + offs, mask=mask, other=0.0)
    # ... compute partial result ...

    # Sync within block, then atomically signal completion
    tl.debug_barrier()
    count = tl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")

    # Last block performs final reduction
    if count == num_blocks - 1:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for i in range(num_blocks):
            acc += tl.load(partial_ptr + i * N + offs, mask=mask, other=0.0)
        tl.store(output_ptr + offs, acc, mask=mask)
```

### Spin-Lock with `tl.atomic_cas`

For non-trivial read-modify-write sequences where `atomic_add` is insufficient.
The spin-lock pattern (acquire via `tl.atomic_cas`, critical section, release via `tl.atomic_xchg`)
is identical to the lock-based weight gradient pattern. See [EXAMPLES_TRAINING.md §3 — LayerNorm Backward](EXAMPLES_TRAINING.md#3-layernorm-backward) for a complete implementation with spin-locks, partial buffers, and count tracking.

**When to use each pattern:**
- **`tl.atomic_add`**: Simple accumulation (split-K). Fastest, no locks.
- **Counter-based last-block**: One block finalizes after all complete (norm backward).
- **Spin-lock**: Non-trivial read-modify-write. Use sparingly -- serializes blocks.

**Important:** Zero-initialize output buffers for `atomic_add`. Lock/counter variables
must be in global memory. `tl.debug_barrier()` syncs within a block, not across blocks.

---

*Reference covers Triton 3.x APIs. Hardware requirements annotated per section.
API signatures for `tl.dot_scaled` and TMA descriptors may evolve across versions.*
