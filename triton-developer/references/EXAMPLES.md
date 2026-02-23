# Essential Examples

Complete, runnable Triton kernel examples with detailed commentary. These three examples cover the fundamental patterns needed for most GPU kernel development.

---

## Table of Contents

1. [Vector Addition](#1-vector-addition) — Pointer arithmetic, masking, 1D kernel
2. [Fused Softmax](#2-fused-softmax) — Row reduction, numerical stability, persistent kernel
3. [Matrix Multiplication](#3-matrix-multiplication) — 2D tiling, autotuning, L2 optimization
4. [Pattern Cheat Sheet](#4-pattern-cheat-sheet)

---

## 1. Vector Addition

The simplest Triton kernel. Demonstrates the foundational pattern: pointer arithmetic + masking.

**Key concepts:**
- `tl.program_id(0)` identifies which block we are
- `tl.arange(0, BLOCK_SIZE)` creates offset vector
- `mask = offsets < n_elements` guards out-of-bounds access
- `tl.load/store` with mask for safe memory access
- Grid as lambda for autotune compatibility

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,       # Pointer to first input vector
    y_ptr,       # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Elements per program (compile-time constant)
):
    # Identify which block this program is
    pid = tl.program_id(axis=0)

    # Compute the range of elements this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard against out-of-bounds access
    mask = offsets < n_elements

    # Load inputs (masked: out-of-bounds reads return 0)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute and store (masked: out-of-bounds writes are ignored)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper: allocate output, compute grid, launch kernel."""
    output = torch.empty_like(x)
    n_elements = output.numel()

    # Grid = number of blocks needed to cover all elements
    # Lambda form allows @triton.autotune to inject BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# --- Usage & Validation ---
x = torch.rand(98432, device=DEVICE)
y = torch.rand(98432, device=DEVICE)
output = add(x, y)
torch.testing.assert_close(output, x + y)
print("Vector addition: PASSED")
```

**Why this matters:**
- Every Triton kernel uses this pointer + mask pattern
- `BLOCK_SIZE` is `tl.constexpr` — compiler knows it at compile time, enabling optimizations
- Grid lambda pattern is compatible with `@triton.autotune`

---

## 2. Fused Softmax

Row-wise softmax with numerical stability. Demonstrates reductions and persistent kernel pattern.

**Key concepts:**
- One row per iteration (or multiple rows with persistent pattern)
- `other=-float('inf')` for masked elements in max reduction
- Max subtraction before `tl.exp` prevents overflow
- `tl.range()` with `num_stages` for software pipelining
- Persistent kernel: fewer programs than rows, each handles multiple

```python
import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Persistent kernel: each program processes multiple rows
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Compute pointer to start of this row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        # Load row with -inf padding (won't affect max or contribute to sum)
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # NUMERICAL STABILITY: subtract max before exp
        row_minus_max = row - tl.max(row, axis=0)

        # Compute softmax
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # Write back
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    # BLOCK_SIZE must be power-of-2 and >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Heuristic: more warps for wider rows
    num_warps = 8
    num_stages = 4

    y = torch.empty_like(x)

    # Persistent kernel: use fewer programs than rows
    # Production code would calculate occupancy from device properties
    num_programs = min(n_rows, 128)

    softmax_kernel[(num_programs, 1, 1)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y


# --- Usage & Validation ---
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
torch.testing.assert_close(y_triton, y_torch, atol=1e-4, rtol=1e-4)
print("Fused softmax: PASSED")
```

**Why this matters:**
- **Kernel fusion**: Single pass over data instead of 5 PyTorch ops (max, sub, exp, sum, div)
- **Numerical stability**: Max subtraction is essential — without it, `exp()` overflows for large values
- **Persistent pattern**: `tl.range(start, end, step, num_stages=N)` enables software pipelining
- **Padding**: `other=-float('inf')` ensures masked elements don't affect max/sum

---

## 3. Matrix Multiplication

Production-quality FP16 matmul with autotuning and L2 cache optimization. This is the canonical Triton kernel.

**Key concepts:**
- 2D pointer arithmetic: `offs_m[:, None] * stride_m + offs_k[None, :] * stride_k`
- K-loop with **fp32 accumulator** (CRITICAL for numerical precision)
- Grouped ordering for L2 cache hit rate
- `@triton.autotune` with multiple configurations
- Modular pointer offset wrapping: `(pid_m * BLOCK_M + arange) % M`

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # Retune when problem dimensions change
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (how much to add to pointer to move by 1 element in that dim)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters (constexpr = compile-time known)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[K,N] with grouped ordering and fp32 accumulation."""

    # ============================================================
    # STEP 1: Map program ID to (pid_m, pid_n) with grouped ordering
    # This promotes L2 cache reuse by processing nearby blocks together
    # ============================================================
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ============================================================
    # STEP 2: Create pointers for first blocks of A and B
    # A block: [BLOCK_M, BLOCK_K], B block: [BLOCK_K, BLOCK_N]
    # ============================================================
    # Wrapping with % M/N handles edge blocks cleanly
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # 2D pointer blocks: outer product of row offsets and col offsets
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ============================================================
    # STEP 3: K-loop accumulation in fp32 (CRITICAL for precision)
    # Even with fp16 inputs, accumulate in fp32 to avoid overflow
    # ============================================================
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Mask K dimension for partial tiles at the end
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)

        k_mask_b = offs_k[:, None] < K - k * BLOCK_K
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)

        # Matrix multiply-accumulate using Tensor Cores
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Cast back to output dtype AFTER all accumulation is done
    c = accumulator.to(tl.float16)

    # ============================================================
    # STEP 4: Store result with boundary mask
    # ============================================================
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for Triton matmul."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1D grid: total blocks = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# --- Usage & Validation ---
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=0)
print("Matrix multiplication: PASSED")
```

**Why this matters:**
- **fp32 accumulator**: Without it, fp16 matmul produces wrong results (overflow/precision loss). This is the #1 Triton bug.
- **Grouped ordering**: ~10% speedup on A100 by improving L2 cache hit rate
- **Autotuning**: Different block sizes work best for different problem sizes; `@triton.autotune` finds the best
- **2D pointer arithmetic**: `offs_m[:, None] * stride_m + offs_k[None, :] * stride_k` creates a 2D block of pointers via broadcasting

---

## 4. Pattern Cheat Sheet

| Pattern | Key Code | When to Use |
|---------|----------|-------------|
| **1D element-wise** | `pid * BLOCK + arange(0, BLOCK)` | Vector ops, reductions |
| **2D tiling** | `offs_m[:, None] * stride_m + offs_n[None, :] * stride_n` | Matmul, 2D operations |
| **Boundary guard** | `mask = offsets < n_elements` | Always (prevent illegal access) |
| **fp32 accumulate** | `acc = tl.zeros(..., dtype=tl.float32)` | Any `tl.dot` operation |
| **Max subtraction** | `x - tl.max(x, axis=0)` | Before `tl.exp()` (softmax, attention) |
| **Persistent kernel** | `tl.range(start, n, step, num_stages=N)` | Many work items, want pipelining |
| **L2 swizzle** | Grouped ordering with GROUP_SIZE_M | Matmul with large matrices |
| **Autotuning** | `@triton.autotune(configs=[...], key=[...])` | Any kernel needing peak performance |

**Tolerance guidelines for validation:**

| Data Type | rtol | atol | Notes |
|-----------|------|------|-------|
| float32 | 1e-4 | 1e-4 | Tight tolerance |
| float16 | 1e-3 | 1e-3 | Standard for fp16 |
| bfloat16 | 1e-2 | 1e-2 | Lower precision mantissa |
| Attention | 0 | 1e-2 | Many ops compound error |
| fp8 | 0 | 0.125 | Very low precision |

**For more examples, see:**
- [EXAMPLES_ATTENTION.md](EXAMPLES_ATTENTION.md) — Flash Attention, GQA, Split-KV
- [EXAMPLES_NORMALIZATION.md](EXAMPLES_NORMALIZATION.md) — LayerNorm, RMSNorm
- [EXAMPLES_LLM.md](EXAMPLES_LLM.md) — RoPE, SwiGLU, Cross-Entropy, Dropout
- [EXAMPLES_TRAINING.md](EXAMPLES_TRAINING.md) — Backward passes, gradient kernels
- [EXAMPLES_ADVANCED.md](EXAMPLES_ADVANCED.md) — Persistent matmul, Grouped GEMM, FP8
