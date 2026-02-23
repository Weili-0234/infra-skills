---
name: triton-developer
description: |
  GPU kernel development with OpenAI Triton, a Python DSL for high-performance GPU programming.

  Use this skill when:
  - User mentions "triton", "triton kernel", "@triton.jit", "tl.load", "tl.store", or "triton.language"
  - User asks to write GPU kernels using Triton language
  - User wants Hopper/H100/Blackwell GPU kernel optimization with TMA or warp specialization
  - User mentions triton autotuning, persistent kernels, or fused attention in Triton
  - User asks about matrix multiplication, attention, normalization, or LLM operators using Triton
  - User needs to optimize existing Triton kernels (memory coalescing, L2 swizzle, pipelining)
  - User references Flash Attention, online softmax, or Triton-based ML kernels
  - User asks to write backward/gradient kernels in Triton
---

# triton-developer

## Overview

**Triton** is OpenAI's Python DSL for writing high-performance GPU kernels. It uses an **SPMD block-level programming model** — each kernel instance (called a "program") operates on a tile of data, and Triton's compiler handles thread-level parallelism, shared memory, and instruction scheduling automatically.

**Key characteristics:**
- **Pythonic syntax**: `@triton.jit` decorator, standard Python control flow
- **Pointer arithmetic + masking**: `tl.load(ptr + offsets, mask=mask)` for safe memory access
- **Hardware Tensor Cores**: `tl.dot()` maps to hardware matrix engines
- **Automatic optimization**: Compiler handles shared memory, coalescing, pipelining
- **Built-in autotuning**: `@triton.autotune` searches over configurations automatically
- **Cross-platform**: Compiles to NVIDIA PTX and AMD AMDGCN

### System Requirements

- **GPU**: NVIDIA Ampere+ (A100, H100, B200) or AMD MI250/MI300X
- **Python**: 3.9+
- **Installation**: `pip install triton`

## Core Workflow

### Step 1: Import and Setup

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
```

### Step 2: Define Kernel

Use `@triton.jit` decorator. Mark tile sizes as `tl.constexpr` so they become compile-time constants:

```python
@triton.jit
def my_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant → enables static shapes
):
    pid = tl.program_id(axis=0)  # Which block am I?
    # ... kernel logic ...
```

### Step 3: Implement Logic

**Core pattern — pointer arithmetic + masking:**
```python
# Compute offsets for this block's elements
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements  # Guard out-of-bounds

# Load with mask (out-of-bounds → 0)
x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

# Compute
y = x * 2.0

# Store with mask
tl.store(output_ptr + offsets, y, mask=mask)
```

### Step 4: Launch Kernel

```python
# Grid = number of blocks (can be a lambda for autotune compatibility)
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

# Launch: kernel[grid](*args, **constexpr_kwargs)
my_kernel[grid](input_ptr, output_ptr, n_elements, BLOCK_SIZE=1024)
```

### Step 5: Validate Results

```python
reference = input_tensor * 2.0
torch.testing.assert_close(output_tensor, reference, rtol=1e-3, atol=1e-3)
```

## API Quick Reference

### Decorators & Types
```python
@triton.jit                              # JIT-compile kernel
@triton.autotune(configs=[...], key=[...])  # Auto-select best config
triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3)

tl.constexpr  # Compile-time constant (used for tile sizes, flags)
```

### Memory Operations
```python
tl.load(pointer, mask=None, other=0.0)   # Load with boundary guard
tl.store(pointer, value, mask=None)       # Store with boundary guard
```

### Compute
```python
tl.dot(a, b, acc)                         # Matrix multiply (Tensor Cores), ALWAYS use fp32 acc
tl.sum(x, axis=0)                         # Reduction sum
tl.max(x, axis=0)                         # Reduction max
tl.min(x, axis=0)                         # Reduction min
```

### Math
```python
tl.exp(x)    tl.exp2(x)    tl.log(x)    tl.log2(x)
tl.sqrt(x)   tl.rsqrt(x)   tl.abs(x)    tl.sigmoid(x)
tl.where(cond, x, y)        tl.maximum(a, b)   tl.minimum(a, b)
```

### Tensors & Grid
```python
tl.arange(0, BLOCK_SIZE)                  # [0, 1, 2, ..., BLOCK_SIZE-1]
tl.zeros((M, N), dtype=tl.float32)        # Zero tensor
tl.full((M, N), val, dtype=tl.float32)    # Constant tensor
tl.program_id(axis=0)                     # Block index (0, 1, or 2)
tl.num_programs(axis=0)                   # Grid size
triton.cdiv(a, b)                         # Ceiling division
x.to(tl.float32)                          # Type cast
```

## Essential Examples

### Example 1: Vector Addition

Demonstrates pointer arithmetic, masking, and basic kernel launch.

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# Validate
x = torch.rand(98432, device=DEVICE)
y = torch.rand(98432, device=DEVICE)
torch.testing.assert_close(add(x, y), x + y)
```

### Example 2: Fused Softmax

Demonstrates row-wise reduction, numerical stability (max subtraction), and persistent kernel pattern.

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        # Load row, pad out-of-bounds with -inf (won't affect max/sum)
        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        # Numerical stability: subtract max before exp
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back
        output_row_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_ptr + col_offsets, softmax_output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4
    y = torch.empty_like(x)
    # Use persistent kernel: fewer blocks than rows, each processes multiple rows
    num_programs = min(n_rows, 128)  # Simplified; production code uses occupancy calc
    softmax_kernel[(num_programs, 1, 1)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps,
    )
    return y

# Validate
x = torch.randn(1823, 781, device=DEVICE)
torch.testing.assert_close(softmax(x), torch.softmax(x, dim=1), atol=1e-4, rtol=1e-4)
```

### Example 3: Matrix Multiplication (Autotuned)

Demonstrates 2D tiling, `tl.dot` with fp32 accumulator, L2 cache optimization via grouped ordering, and autotuning.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # -- L2 cache optimization: grouped ordering --
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -- Pointer setup --
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -- K-loop with fp32 accumulator (CRITICAL for precision) --
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float16)

    # -- Store with boundary mask --
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0] and a.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c

# Validate
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
torch.testing.assert_close(matmul(a, b), torch.matmul(a, b), atol=1e-2, rtol=0)
```

## Common Patterns

### Pattern 1: Pointer Arithmetic + Masking
Every Triton kernel uses this. Compute `offsets = block_start + tl.arange(0, BLOCK)`, create `mask = offsets < n_elements`, then `tl.load(ptr + offsets, mask=mask, other=0.0)`.

### Pattern 2: K-loop Tiling with fp32 Accumulator
For matmul/attention: iterate over K dimension in BLOCK_K tiles. **Always** use `tl.zeros(..., dtype=tl.float32)` accumulator even with fp16 inputs. Cast to output dtype only at the end.

### Pattern 3: Row-wise Reductions
Process one row per program (or multiple rows with persistent pattern). Load entire row, apply `tl.max/sum/min(row, axis=0)`. Pad with `-inf` for max, `0` for sum.

### Pattern 4: Autotuning
Use `@triton.autotune` with multiple `triton.Config` objects varying `BLOCK_SIZE`, `num_warps`, `num_stages`. Set `key=[...]` to retune when problem dimensions change.

### Pattern 5: Persistent Kernels
Launch fewer programs than work items. Each program loops over work:
```python
for idx in tl.range(start, total, step, num_stages=num_stages):
    # process work item idx
```
Benefits: better SM utilization, software pipelining via `num_stages`.

## Debugging Quick Tips

- **`TRITON_INTERPRET=1`**: Run kernel on CPU with Python debugger — set breakpoints inside kernels!
- **`tl.device_print("x =", x)`**: Print values from GPU at runtime
- **`tl.static_print(BLOCK_SIZE)`**: Print compile-time values during compilation
- **Common pitfalls**: Missing mask → illegal memory access; fp16 accumulator → NaN/wrong results; wrong stride → garbled output

## Advanced Topics

For deeper coverage, see the reference files:

**Core References (Layer 1) — patterns & API for 90% of kernels:**
- **[PATTERNS.md](references/PATTERNS.md)** — Pattern catalog: when/why to use each pattern, key snippets, cross-refs to full implementations
- **[API_ESSENTIALS.md](references/API_ESSENTIALS.md)** — Core API: decorators, load/store, dot, reductions, math
- **[EXAMPLES.md](references/EXAMPLES.md)** — Complete vector add, softmax, matmul with full commentary

**Domain-Specific Examples (Layer 2) — full implementations by domain:**
- **[EXAMPLES_ATTENTION.md](references/EXAMPLES_ATTENTION.md)** — Flash Attention v2, GQA, Split-KV decode
- **[EXAMPLES_NORMALIZATION.md](references/EXAMPLES_NORMALIZATION.md)** — LayerNorm, RMSNorm, fused add+norm
- **[EXAMPLES_LLM.md](references/EXAMPLES_LLM.md)** — RoPE, SwiGLU, fused cross-entropy, dropout
- **[EXAMPLES_TRAINING.md](references/EXAMPLES_TRAINING.md)** — Backward passes, gradient kernels
- **[EXAMPLES_ADVANCED.md](references/EXAMPLES_ADVANCED.md)** — Persistent matmul, grouped GEMM, FP8, warp specialization

**Performance & Debugging (Layer 3-4) — optimization, testing, advanced API:**
- **[API_ADVANCED.md](references/API_ADVANCED.md)** — Block pointers, TMA, atomics, RNG, FP8, compiler hints
- **[OPTIMIZATION.md](references/OPTIMIZATION.md)** — Hardware tuning, autotuning deep dive, TMA, pipelining
- **[TESTING.md](references/TESTING.md)** — PyTest patterns, tolerance guidelines, benchmarking
- **[DEBUGGING.md](references/DEBUGGING.md)** — Environment variables, common errors, profiling
