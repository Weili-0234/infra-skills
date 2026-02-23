# Triton API Essentials

Core API reference for OpenAI Triton covering 90% of kernel development needs.

```python
import triton
import triton.language as tl
```

---

## Table of Contents

1. [Decorators & Compilation](#1-decorators--compilation)
2. [Data Types](#2-data-types)
3. [Memory Operations](#3-memory-operations)
4. [Compute Operations](#4-compute-operations)
5. [Math Functions](#5-math-functions)
6. [Tensor Creation & Manipulation](#6-tensor-creation--manipulation)
7. [Grid & Program Info](#7-grid--program-info)
8. [Type Promotion Rules](#8-type-promotion-rules)

---

## 1. Decorators & Compilation

### `@triton.jit`

```python
@triton.jit
def my_kernel(arg0, arg1, N, BLOCK_SIZE: tl.constexpr):
    ...
```

JIT-compiles a Python function into a GPU kernel. Each invocation operates on a tile of data.

- Positional args become kernel arguments (pointers, scalars, strides)
- `tl.constexpr`-annotated parameters are compile-time constants (tile sizes, flags)
- Kernel is recompiled for each unique set of `constexpr` values

**Launching:**
```python
grid = (num_blocks_x,)                              # 1D grid
grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),) # dynamic grid
my_kernel[grid](arg0, arg1, N, BLOCK_SIZE=128)
# num_warps, num_stages can be passed at launch
```

### `@triton.autotune`

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N'],
    prune_configs_by=None,    # {'early_config_prune': fn} to eliminate configs
    reset_to_zero=['c_ptr'],  # Zero output ptrs before each benchmark run
    pre_hook=None,            # Callable run before each benchmark
)
@triton.jit
def my_kernel(...): ...
```

Benchmarks all configs and selects the fastest. `key` lists arg names (problem dims) that trigger re-tuning. Grid **must** be `lambda meta:` when using autotune.

### `triton.Config`

```python
triton.Config(
    kwargs: dict,          # constexpr values: {'BLOCK_M': 128, 'BLOCK_N': 64}
    num_warps: int = 4,    # Warps per program (power of 2, typically 1-16)
    num_stages: int = 2,   # Software pipelining stages (2-5 typical)
    num_ctas: int = 1,     # Cooperative thread arrays (cluster launches)
    maxnreg: int = None,   # Max registers per thread (None = unlimited)
)
```

- `num_warps`: 4 for small tiles, 8 for large tiles
- `num_stages`: 2-3 for compute-bound, 4-5 for memory-bound
- `maxnreg`: Lower = higher occupancy but may spill to local memory

---

## 2. Data Types

### Floating-Point

| Type | Bits | Use |
|---|---|---|
| `tl.float16` | 16 | Inference, Tensor Core inputs |
| `tl.bfloat16` | 16 | Training (better dynamic range) |
| `tl.float32` | 32 | Accumulators, reductions |
| `tl.float64` | 64 | Rarely used in ML |

### Integer

| Type | Bits | Use |
|---|---|---|
| `tl.int8` / `tl.uint8` | 8 | Quantized inference |
| `tl.int16` / `tl.uint16` | 16 | |
| `tl.int32` / `tl.uint32` | 32 | Indices, offsets, strides |
| `tl.int64` / `tl.uint64` | 64 | Large tensor offsets |
| `tl.int1` | 1 | Boolean (comparison results) |

### Special

- `tl.constexpr` -- compile-time constant for tile sizes and flags

### Casting

```python
x_fp32 = x_fp16.to(tl.float32)   # Explicit cast via .to()
mask = (offsets < N)              # Produces tl.int1
```

Pointer types are inferred from the PyTorch tensor dtype at launch.

---

## 3. Memory Operations

### `tl.load`

```python
tl.load(
    pointer,                    # Pointer or tensor of pointers
    mask=None,                  # Boolean; False lanes get `other`
    other=0.0,                  # Default for masked-out lanes
    cache_modifier="",          # "", ".ca", ".cg" (L2 only), ".cs" (streaming)
    eviction_policy="",         # "", "evict_first", "evict_last"
    volatile=False,             # Force reload (no caching)
)
```

Loads from global memory. Returns tensor shaped like `pointer`.

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### `tl.store`

```python
tl.store(
    pointer,                    # Pointer or tensor of pointers
    value,                      # Values to write
    mask=None,                  # Only writes where True
    cache_modifier="",          # Same options as load
    eviction_policy="",         # Same options as load
)
```

```python
tl.store(output_ptr + offsets, result, mask=mask)
```

### Pointer Arithmetic

**1D -- contiguous block:**
```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Shape: (BLOCK,)
mask = offsets < n_elements
x = tl.load(base_ptr + offsets, mask=mask, other=0.0)
```

**2D -- matrix tile:**
```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

# Broadcasting: (BLOCK_M,1) * stride + (1,BLOCK_N) -> (BLOCK_M, BLOCK_N)
ptrs = base_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tile = tl.load(ptrs, mask=mask, other=0.0)
```

### Boundary Masking

Always mask when blocks may exceed bounds:
```python
mask = offsets < n_elements                              # 1D
mask = (offs_row[:, None] < M) & (offs_col[None, :] < N) # 2D
```

Common `other` values: `0.0` (additions/dot), `-float('inf')` (max), `float('inf')` (min).

---

## 4. Compute Operations

### `tl.dot`

```python
tl.dot(
    a,                          # Shape (M, K)
    b,                          # Shape (K, N)
    acc=None,                   # Accumulator (M, N), must be fp32
    input_precision="tf32",     # "tf32" | "ieee" | "tf32x3"
    max_num_imprecise_acc=None, # Max low-precision accumulations before rounding
)
```

Block-level matrix multiply via Tensor Cores. Returns `(M, N)`.

**CRITICAL -- always accumulate in fp32:**
```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K_TILES):
    a = tl.load(a_ptrs, ...)  # fp16/bf16
    b = tl.load(b_ptrs, ...)  # fp16/bf16
    acc = tl.dot(a, b, acc)   # fp32 accumulation
result = acc.to(tl.float16)   # Cast only at end
```

- Min K dimension: 16 elements. Min tile: 16x16
- Both operands must be same dtype; cast explicitly if they differ

### `tl.sum`

```python
tl.sum(input, axis=None)  # Sum reduction; axis=None -> scalar
```

```python
row_sum = tl.sum(x, axis=1)  # (M, N) -> (M,)
col_sum = tl.sum(x, axis=0)  # (M, N) -> (N,)
```

### `tl.max`

```python
tl.max(input, axis=None)  # Max reduction
```

Tip: Use `other=-float('inf')` in `tl.load` so masked elements don't affect max.

### `tl.min`

```python
tl.min(input, axis=None)  # Min reduction
```

Tip: Use `other=float('inf')` in `tl.load` so masked elements don't affect min.

### `tl.reduce`

```python
tl.reduce(input, axis, combine_fn)  # Custom reduction
```

`combine_fn` must be `@triton.jit`-decorated, associative, and commutative.

```python
@triton.jit
def max_combine(val_a, idx_a, val_b, idx_b):
    gt = val_a > val_b
    return tl.where(gt, val_a, val_b), tl.where(gt, idx_a, idx_b)

max_val, max_idx = tl.reduce(
    (values, tl.arange(0, BLOCK)), axis=0, combine_fn=max_combine
)
```

Supports reducing multiple tensors simultaneously (pass as tuple).

### `tl.softmax` (Deprecated)

```python
tl.softmax(input, axis)
```

Prefer manual softmax (subtract max, exp, divide by sum) for numerical stability.

---

## 5. Math Functions

### Exponential & Logarithmic

```python
tl.exp(x)       # e^x         -- softmax, attention
tl.exp2(x)      # 2^x         -- faster on GPU
tl.log(x)       # ln(x)       -- natural log
tl.log2(x)      # log2(x)     -- base-2 log
```

For softmax stability: always subtract row max before `tl.exp`.

### Power & Root

```python
tl.sqrt(x)      # Square root
tl.rsqrt(x)     # 1/sqrt(x)   -- faster for normalization (LayerNorm, RMSNorm)
```

### Trigonometric & Special

```python
tl.abs(x)       # Absolute value
tl.sin(x)       # Sine
tl.cos(x)       # Cosine
tl.erf(x)       # Gauss error function -- used in GELU
tl.sigmoid(x)   # 1/(1+exp(-x)) -- used in SiLU/Swish
```

### Fused & Rounding

```python
tl.fma(a, b, c) # Fused multiply-add: a*b + c (single rounding)
tl.fdiv(a, b)   # Floating-point division
tl.floor(x)     # Round toward -inf
tl.ceil(x)      # Round toward +inf
```

### Conditional & Comparison

```python
tl.where(cond, x, y)     # Element-wise select: x if cond else y
tl.maximum(a, b)          # Element-wise max
tl.minimum(a, b)          # Element-wise min
tl.clamp(x, min, max)     # Clamp to [min, max]
```

```python
output = tl.where(x > 0, x, 0.0)                                     # ReLU
scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -1e9)   # Causal mask
x_clip = tl.clamp(x, min=-1.0, max=1.0)
```

---

## 6. Tensor Creation & Manipulation

### `tl.arange`

```python
tl.arange(start, end)  # -> 1D int tensor [start, ..., end-1]
```

Both `start` and `end` must be `tl.constexpr`. Shape: `(end - start,)`.

```python
offsets = tl.arange(0, BLOCK_SIZE)           # [0, 1, ..., BLOCK_SIZE-1]
offsets = start + tl.arange(0, BLOCK_SIZE)   # shifted range
```

### `tl.zeros`

```python
tl.zeros(shape, dtype)  # Tensor of zeros
```

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

### `tl.full`

```python
tl.full(shape, value, dtype)  # Tensor filled with constant
```

```python
neg_inf = tl.full((BLOCK_M, BLOCK_N), float('-inf'), dtype=tl.float32)
```

### `tl.broadcast_to`

```python
tl.broadcast_to(x, shape)  # Broadcast to larger shape (NumPy rules)
```

Implicit in most ops; explicit form rarely needed.

### `tl.expand_dims`

```python
tl.expand_dims(x, axis)  # Add size-1 dimension
```

Equivalent to indexing syntax:
```python
col_vec = offsets[:, None]   # (N,) -> (N, 1), same as tl.expand_dims(offsets, 1)
row_vec = offsets[None, :]   # (N,) -> (1, N), same as tl.expand_dims(offsets, 0)
```

### `tl.reshape`

```python
tl.reshape(x, shape)  # Reshape (preserves total elements)
```

### `tl.permute`

```python
tl.permute(x, dims)  # Permute dimensions
```

```python
x_T = tl.permute(x, (1, 0))  # (M, N) -> (N, M)
```

### `tl.trans`

```python
tl.trans(x)  # Transpose 2D tensor; shorthand for tl.permute(x, (1, 0))
```

### `tl.view`

```python
tl.view(x, dtype)  # Bitcast to different dtype (total bits must match)
```

```python
x_bits = tl.view(x_fp16, tl.int16)  # Reinterpret bits
```

### `tl.cat`

```python
tl.cat(lhs, rhs)  # Concatenate two 1D tensors
```

### `.to(dtype)`

```python
x.to(dtype)  # Cast tensor to new dtype
```

```python
x_fp32 = x_fp16.to(tl.float32)    # Upcast for accumulation
result = acc.to(tl.float16)       # Downcast for storage
```

---

## 7. Grid & Program Info

### `tl.program_id`

```python
tl.program_id(axis)  # -> int32: current block index along axis (0, 1, or 2)
```

```python
pid_m = tl.program_id(0)   # Row tile index
pid_n = tl.program_id(1)   # Column tile index
pid_b = tl.program_id(2)   # Batch index
```

### `tl.num_programs`

```python
tl.num_programs(axis)  # -> int32: grid size along axis
```

```python
# Persistent kernel pattern
for idx in tl.range(tl.program_id(0), total_work, tl.num_programs(0)):
    ...  # process work item
```

### `triton.cdiv`

```python
triton.cdiv(a, b)  # -> ceil(a / b)
```

Standard grid dimension computation. Also available as `tl.cdiv()` inside kernels.

```python
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
```

### `triton.next_power_of_2`

```python
triton.next_power_of_2(n)  # -> smallest power of 2 >= n
```

```python
BLOCK_SIZE = triton.next_power_of_2(n_cols)  # 781 -> 1024
```

---

## 8. Type Promotion Rules

### Implicit Promotion

Operations on mixed types promote to the wider type:
```
int8  + int16  -> int16       float16  + float32 -> float32
int16 + int32  -> int32       bfloat16 + float32 -> float32
int32 + int64  -> int64       float32  + float64 -> float64
```

Integer-float mixed: `int32 + float16 -> float16`, `int32 + float32 -> float32`.

### Precision Best Practices

1. **Accumulators must be fp32.** fp16/bf16 accumulators cause overflow and NaN:
   ```python
   acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # CORRECT
   acc = tl.dot(a_fp16, b_fp16, acc)
   ```

2. **Cast to output dtype only at the end:**
   ```python
   result = acc.to(tl.float16)
   tl.store(out_ptr + offsets, result, mask=mask)
   ```

3. **Reductions in fp32** even with fp16 inputs:
   ```python
   x_sum = tl.sum(x.to(tl.float32), axis=0)
   ```

4. **Normalization in fp32** (LayerNorm, Softmax):
   ```python
   mean = tl.sum(x.to(tl.float32), axis=0) / N
   var = tl.sum((x.to(tl.float32) - mean) ** 2, axis=0) / N
   rstd = tl.rsqrt(var + eps)
   out = ((x.to(tl.float32) - mean) * rstd).to(tl.float16)
   ```

5. **`tl.dot` operands must match type.** Cast explicitly if they differ:
   ```python
   a = tl.load(a_ptr + ...).to(tl.float16)
   b = tl.load(b_ptr + ...).to(tl.float16)
   acc = tl.dot(a, b, acc)
   ```
