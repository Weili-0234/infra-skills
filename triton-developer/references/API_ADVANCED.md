# Triton Advanced API Reference

Advanced features for optimized kernels targeting Hopper+ architectures.

## Table of Contents
1. [Block Pointers](#block-pointers)
2. [TMA Descriptors - Hopper+](#tma-descriptors---hopper)
3. [Atomic Operations](#atomic-operations)
4. [Random Number Generation](#random-number-generation)
5. [Scans & Cumulative Operations](#scans--cumulative-operations)
6. [Sorting & Selection](#sorting--selection)
7. [Compiler Hints & Optimization](#compiler-hints--optimization)
8. [Debug Primitives](#debug-primitives)
9. [Loop Constructs](#loop-constructs)
10. [Heuristics](#heuristics)
11. [FP8 & Microscaling](#fp8--microscaling)
12. [Cache Control](#cache-control)

---

## Block Pointers

Block pointers provide a structured abstraction for loading/storing 2D (or N-D) tiles.
They encode base pointer, tensor shape, strides, offsets, and tile shape, enabling the
compiler to emit efficient bulk-copy instructions.

### `tl.make_block_ptr`
```python
tl.make_block_ptr(
    base,          # Base pointer to tensor in global memory
    shape,         # Tuple of tensor dimensions, e.g. (M, N)
    strides,       # Tuple of strides in elements, e.g. (stride_m, stride_n)
    offsets,       # Tuple of initial offsets, e.g. (off_m, off_n)
    block_shape,   # Tile size to load, e.g. (BLOCK_M, BLOCK_N)
    order,         # Memory layout order, default (1, 0) = row-major
)
```

### `tl.load` / `tl.store` (block pointer overloads)
```python
tl.load(block_ptr, boundary_check=(), padding_option="")
tl.store(block_ptr, value, boundary_check=())
```
Axes in `boundary_check` are guarded; out-of-bounds elements filled per `padding_option`
(`"zero"` or `"nan"`). `value` shape must match `block_shape`.

### `tl.advance`
```python
tl.advance(block_ptr, offsets)   # Returns new block pointer with offsets incremented
```

### Example: 2D Matmul Tile Loading
```python
# Inside a matmul kernel:
a_block = tl.make_block_ptr(
    base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
)
b_block = tl.make_block_ptr(
    base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
    offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
    b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
    acc = tl.dot(a, b, acc)
    a_block = tl.advance(a_block, (0, BLOCK_K))
    b_block = tl.advance(b_block, (BLOCK_K, 0))
c_block = tl.make_block_ptr(
    base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
)
tl.store(c_block, acc.to(tl.float16), boundary_check=(0, 1))
```

---

## TMA Descriptors - Hopper+

Tensor Memory Accelerator (TMA) descriptors leverage dedicated Hopper hardware for
asynchronous bulk copies between global and shared memory.

> **Requires compute capability >= 9.0** (H100, H200, B200 and newer).

### `tl.make_tensor_descriptor`
```python
desc = tl.make_tensor_descriptor(
    base,          # Base pointer to global-memory tensor
    shape,         # Tuple of tensor dimensions, e.g. (M, N)
    strides,       # Tuple of strides in elements
    block_shape,   # Tile dimensions for each TMA transfer
)
```
Returns an opaque descriptor that can issue hardware TMA loads and stores.

### `desc.load` / `desc.store`
```python
tile = desc.load(offsets)          # TMA load; returns tensor of shape block_shape
desc.store(offsets, value)         # TMA store; value must match block_shape
```

### Example: TMA Matmul Tile Load
```python
# Inside a Hopper matmul kernel:
a_desc = tl.make_tensor_descriptor(
    a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    block_shape=(BLOCK_M, BLOCK_K),
)
b_desc = tl.make_tensor_descriptor(
    b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
    block_shape=(BLOCK_K, BLOCK_N),
)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = a_desc.load((pid_m * BLOCK_M, k))
    b = b_desc.load((k, pid_n * BLOCK_N))
    acc = tl.dot(a, b, acc)
c_desc = tl.make_tensor_descriptor(
    c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
    block_shape=(BLOCK_M, BLOCK_N),
)
c_desc.store((pid_m * BLOCK_M, pid_n * BLOCK_N), acc.to(tl.float16))
```

---

## Atomic Operations

All atomics operate element-wise on pointers with optional `mask`, memory ordering
(`sem`), and visibility scope (`scope`).

### Core Atomics
```python
tl.atomic_add(pointer, val, mask=None, sem="relaxed", scope="gpu")
tl.atomic_max(pointer, val, mask=None, sem="relaxed", scope="gpu")
tl.atomic_min(pointer, val, mask=None, sem="relaxed", scope="gpu")
```
`atomic_add` supports integer and floating-point types. `atomic_max`/`atomic_min`
support integers; float support depends on GPU architecture.

### Bitwise Atomics
```python
tl.atomic_and(pointer, val, mask=None, sem="relaxed", scope="gpu")
tl.atomic_or(pointer, val, mask=None, sem="relaxed", scope="gpu")
tl.atomic_xor(pointer, val, mask=None, sem="relaxed", scope="gpu")
```

### Exchange and Compare-and-Swap
```python
tl.atomic_xchg(pointer, val, mask=None, sem="relaxed", scope="gpu")
```
Atomically replaces `*pointer` with `val`, returns old value.

```python
tl.atomic_cas(pointer, cmp, val, mask=None, sem="relaxed", scope="gpu")
```
If `*pointer == cmp`, atomically sets `*pointer = val`. Returns old value regardless.

### Memory Ordering (`sem`)

| Semantic    | Description                                        |
|-------------|----------------------------------------------------|
| `"relaxed"` | No ordering guarantees beyond atomicity            |
| `"acquire"` | Subsequent reads/writes cannot be reordered before |
| `"release"` | Prior reads/writes cannot be reordered after       |
| `"acq_rel"` | Combined acquire and release semantics             |

### Visibility Scope (`scope`)

| Scope   | Description                       |
|---------|-----------------------------------|
| `"gpu"` | Visible to all threads on the GPU |
| `"cta"` | Visible within the thread block   |
| `"sys"` | System-wide (including host/PCIe) |

### Example: Spin-Lock
```python
@triton.jit
def locked_add_kernel(lock_ptr, data_ptr, value, N):
    # Spin until this CTA acquires the lock
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="gpu") != 0:
        pass
    # Critical section
    offs = tl.arange(0, N)
    x = tl.load(data_ptr + offs)
    tl.store(data_ptr + offs, x + value)
    # Release the lock
    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="gpu")
```

---

## Random Number Generation

Triton uses the **Philox** counter-based PRNG. Given a `(seed, offset)` pair, output
is fully deterministic and reproducible regardless of grid scheduling.

### Uniform, Normal, and Integer
```python
tl.rand(seed, offset)       # Uniform float32 in [0, 1)
tl.randn(seed, offset)      # Standard normal (mean=0, std=1) float32
tl.randint(seed, offset)    # Random int32
```
`seed` is an int32/int64 scalar. `offset` is typically a per-element tensor.

### 4-Stream Variants
```python
tl.rand4x(seed, offset)     # Returns 4 independent uniform tensors
tl.randn4x(seed, offset)    # Returns 4 independent normal tensors
tl.randint4x(seed, offset)  # Returns 4 independent int32 tensors
```
The `4x` variants extract all four Philox outputs per round for higher throughput.

### Example: Dropout
```python
@triton.jit
def dropout_kernel(x_ptr, out_ptr, N, p, seed, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    rand_vals = tl.rand(seed, offs)
    keep = rand_vals > p
    out = tl.where(keep, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offs, out, mask=mask)
```

---

## Scans & Cumulative Operations

### `tl.associative_scan`
```python
tl.associative_scan(input, axis, combine_fn)
```
Parallel inclusive prefix scan. `combine_fn` must be a `@triton.jit` function
taking `(a, b)` and returning the combined result. The operator must be associative.

```python
@triton.jit
def add_fn(a, b):
    return a + b

@triton.jit
def scan_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    result = tl.associative_scan(x, axis=0, combine_fn=add_fn)
    tl.store(out_ptr + offs, result, mask=offs < N)
```

### `tl.cumsum` / `tl.cumprod`
```python
tl.cumsum(input, axis=0)    # Inclusive cumulative sum along axis
tl.cumprod(input, axis=0)   # Inclusive cumulative product along axis
```

---

## Sorting & Selection

### `tl.sort`
```python
tl.sort(x, dim=0, descending=False)
```
Returns tensor sorted along `dim` using a bitonic sort network.

### `tl.flip`
```python
tl.flip(x, dim=0)           # Reverse elements along dim
```

### `tl.argmax` / `tl.argmin`
```python
tl.argmax(x, axis=0)        # Index of maximum value along axis
tl.argmin(x, axis=0)        # Index of minimum value along axis
```
Return the index (not the value) of the extreme element.

### `tl.histogram`
```python
tl.histogram(input, num_bins)
```
Values in `input` must be non-negative integers in `[0, num_bins)`.
Returns a 1D tensor of length `num_bins` with counts.

---

## Compiler Hints & Optimization

These intrinsics communicate invariants to the compiler for more aggressive
optimization (vectorization, coalescing, etc.).

### `tl.multiple_of`
```python
tl.multiple_of(input, values)
```
Asserts every element is a multiple of the corresponding value. Enables wider loads.
```python
offs = tl.multiple_of(tl.arange(0, BLOCK) * stride, [16])
```

### `tl.max_contiguous`
```python
tl.max_contiguous(input, values)
```
Asserts the first `values[i]` elements along dimension `i` are contiguous.
```python
offs = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), [8]), [8])
```

### `tl.assume` / `tl.debug_barrier`
```python
tl.assume(condition)     # Tell compiler condition is always True (UB if False)
tl.debug_barrier()       # CTA synchronization barrier (__syncthreads equivalent)
```

---

## Debug Primitives

### Runtime (Device)
```python
tl.device_print(prefix, *args)       # GPU runtime print; very slow, debug only
tl.device_assert(condition, message="")  # GPU runtime assert; controlled by TRITON_DEBUG
```

### Compile-Time
```python
tl.static_print(*values)             # Print during JIT compilation
tl.static_assert(condition, message="")  # Assert during compilation
```

```python
@triton.jit
def kernel(..., BLOCK: tl.constexpr):
    tl.static_assert(BLOCK % 16 == 0, "BLOCK must be a multiple of 16")
    tl.static_print("Compiling with BLOCK =", BLOCK)
```

---

## Loop Constructs

### `tl.range`
```python
tl.range(start, end, step=1, num_stages=1, loop_schedule=None)
```
Drop-in replacement for `range()` with **software pipelining**. `num_stages` controls
how many iterations overlap to hide memory latency.
```python
for k in tl.range(0, K, BLOCK_K, num_stages=3):
    a = tl.load(a_ptr + offsets_a)
    b = tl.load(b_ptr + offsets_b)
    acc = tl.dot(a, b, acc)
```

### `tl.static_range`
```python
tl.static_range(start, end)   # Fully unrolls at compile time; args must be constexpr
```
```python
for i in tl.static_range(0, 4):
    x = tl.load(ptr + i * stride)
```
Standard Python `range()` also works but does **not** enable pipelining.

---

## Heuristics

### `@triton.heuristics`
```python
@triton.heuristics({
    'NAME': lambda args: expression,
})
```
Auto-computes meta-parameters from runtime arguments. Heuristic functions receive
the full argument dictionary and return the meta-parameter value.

```python
@triton.heuristics({
    'BLOCK_M': lambda args: 128 if args['M'] >= 1024 else 64,
    'num_stages': lambda args: 4 if args['K'] >= 2048 else 2,
})
@triton.jit
def adaptive_kernel(..., BLOCK_M: tl.constexpr, num_stages: tl.constexpr):
    ...
```
Heuristics evaluate **before** autotune configs. Stack `@triton.heuristics` above
`@triton.autotune` in decorator order.

---

## FP8 & Microscaling

### FP8 Types
```
tl.float8e4nv      # E4M3 (4-bit exponent, 3-bit mantissa) - NVIDIA format
tl.float8e5        # E5M2 (5-bit exponent, 2-bit mantissa) - IEEE format
tl.float8e4b15     # E4M3 with bias 15 (older Triton variant)
```
Supported in `tl.load`, `tl.store`, and `tl.dot`. Convert via `.to()`:
```python
x_fp8 = x_fp16.to(tl.float8e4nv)
x_fp16 = x_fp8.to(tl.float16)
acc = tl.dot(a_fp8, b_fp8, acc)   # acc remains float32
```

### `tl.dot_scaled` (Blackwell / SM 100+)
```python
tl.dot_scaled(
    lhs, lhs_scale, lhs_format,   # e.g. lhs_format="e4m3"
    rhs, rhs_scale, rhs_format,   # e.g. rhs_format="e2m1"
    acc=None,                      # Optional accumulator
)
```
Scaled matrix multiply with per-block scaling factors for microscaling (MX) hardware.
Requires compute capability >= 10.0 (Blackwell).

> The `dot_scaled` API may evolve across Triton versions. Consult the latest source.

---

## Cache Control

### Load/Store Cache Modifiers
The `cache_modifier` parameter on `tl.load`/`tl.store` controls L1/L2 caching.

| Modifier | PTX     | Description                            |
|----------|---------|----------------------------------------|
| `".ca"`  | `ld.ca` | Cache at all levels (default)          |
| `".cg"`  | `ld.cg` | Cache in L2 only, bypass L1           |
| `".cs"`  | `ld.cs` | Cache streaming (evict first)          |
| `".lu"`  | `st.lu` | Last use (won't be reused)             |
| `".cv"`  | `ld.cv` | Don't cache, volatile read             |

### Eviction Policy
| Policy          | Description                                        |
|-----------------|----------------------------------------------------|
| `"evict_first"` | Prefer evicting this data before other cached data |
| `"evict_last"`  | Prefer keeping this data cached as long as possible|

```python
x = tl.load(ptr + offs, eviction_policy="evict_first")
tl.store(out_ptr + offs, result, eviction_policy="evict_last")
```
These hints are advisory; the hardware cache controller makes final decisions.

---

*Reference covers Triton 3.x APIs. Signatures may differ between versions.*
