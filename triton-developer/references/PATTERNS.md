# Triton GPU Kernel Development: Core Patterns Reference

Essential programming patterns for high-performance GPU kernels with OpenAI Triton.
Each pattern includes when to use it, a key snippet, and cross-references to full implementations.

---

## 1. Memory Access Patterns

### Pattern 1: 1D Pointer Arithmetic

The most fundamental access pattern. Each program handles a contiguous block of elements.

**When to use:** Element-wise ops (add, multiply, activations) where each element is independent.

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

- `BLOCK_SIZE` must be a power of two. Always mask the final block.
- Grid size: `triton.cdiv(n_elements, BLOCK_SIZE)`.

> **Full example:** [EXAMPLES.md §1](EXAMPLES.md#1-vector-addition)

---

### Pattern 2: 2D Tiling

Broadcast row and column offsets into a 2D pointer grid via `[:, None]` and `[None, :]`.

**When to use:** Matrix operations, any kernel needing a 2D tile, tensors with non-trivial strides.

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
```

- Always pass strides from the wrapper. Mask must check both dimensions for edge tiles.

> **Full example:** [EXAMPLES.md §3](EXAMPLES.md#3-matrix-multiplication)

---

### Pattern 3: Block Pointers

`tl.make_block_ptr` defines a sliding window; `tl.advance` moves it. Compiler maps to hardware-accelerated instructions.

**When to use:** Matmul-style K-loop iteration, Hopper TMA, code clarity over manual pointers.

```python
a_bptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                            offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
a = tl.load(a_bptr, boundary_check=(0, 1))
a_bptr = tl.advance(a_bptr, (0, BLOCK_K))
```

- `order=(1, 0)` = row-major. `boundary_check` replaces manual masking.
- `tl.advance` returns a new pointer; does not mutate in place.

> **Full example:** [API_ADVANCED.md §1](API_ADVANCED.md#block-pointers)

---

### Pattern 4: TMA Descriptors (Hopper+)

Tensor Memory Accelerator enables hardware-managed bulk transfers on sm_90+.

**When to use:** Targeting H100+ exclusively, highly regular access, maximum bandwidth.

```python
desc = tl.make_tensor_descriptor(input_ptr, shape=[T, D], strides=[stride_t, stride_d],
                                  block_shape=[BLOCK_T, BLOCK_D])
data = desc.load([pid_t * BLOCK_T, pid_d * BLOCK_D])   # No masks needed
```

- Requires Hopper+. TMA handles boundary checks in hardware.

> **Full example:** [API_ADVANCED.md §2](API_ADVANCED.md#tma-descriptors---hopper)

---

### Pattern 5: Memory Coalescing

GPU threads access memory most efficiently when reading contiguous addresses. Triton
provides hints to help the compiler generate coalesced patterns.

**When to use:** Always -- coalescing is fundamental. Especially with non-trivial strides.

```python
@triton.jit
def coalesced_load_kernel(
    input_ptr, output_ptr, n_rows, n_cols, stride_row, stride_col,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offs = tl.arange(0, BLOCK_COLS)
    # Contiguity hints: compiler can generate wider vectorized loads
    col_offs = tl.max_contiguous(tl.multiple_of(col_offs, BLOCK_COLS), BLOCK_COLS)
    ptrs = input_ptr + row_offs[:, None] * stride_row + col_offs[None, :] * stride_col
    mask = (row_offs[:, None] < n_rows) & (col_offs[None, :] < n_cols)
    data = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(output_ptr + row_offs[:, None] * stride_row + col_offs[None, :] * stride_col,
             data, mask=mask)
```

**Key Considerations:**
- Innermost tensor dimension should be contiguous (stride = 1).
- `tl.max_contiguous(x, N)` tells the compiler `x` has N contiguous values.
- `tl.multiple_of(x, N)` tells the compiler values are multiples of N.
- Strided (non-coalesced) access can reduce effective bandwidth by 10-30x.

---

### Pattern 6: L2 Cache Optimization (Grouped Ordering)

Reorder tile execution so spatially nearby tiles share L2 cache lines.

**When to use:** Matmul and 2D-tiled ops when profiling shows low L2 hit rates.

```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

- `GROUP_SIZE_M = 8` is a common starting point. Typical L2 hit rate improvement: 20-40%.

> **Full examples:** [EXAMPLES.md §3](EXAMPLES.md#3-matrix-multiplication), [OPTIMIZATION.md §6](OPTIMIZATION.md#6-l2-cache-optimization)

---

## 2. Reduction Patterns

### Pattern 1: Tile-Local Reductions

Built-in reductions collapse data within a single program's tile.

**When to use:** Row/column reductions where each row fits in one tile; per-row statistics.

```python
@triton.jit
def row_sum_kernel(input_ptr, output_ptr, M, N, stride_m, stride_n, BLOCK_N: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_N)
    data = tl.load(input_ptr + row_idx * stride_m + col_offs * stride_n,
                   mask=col_offs < N, other=0.0)
    tl.store(output_ptr + row_idx, tl.sum(data, axis=0))

@triton.jit
def row_max_2d_kernel(input_ptr, out_ptr, M, N, stride_m, stride_n,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)
    ptrs = input_ptr + row_offs[:, None] * stride_m + col_offs[None, :] * stride_n
    data = tl.load(ptrs, mask=(row_offs[:, None] < M) & (col_offs[None, :] < N),
                   other=float('-inf'))
    # axis=1 reduces columns, result shape: (BLOCK_M,)
    tl.store(out_ptr + row_offs, tl.max(data, axis=1), mask=row_offs < M)
```

- `axis=0` reduces first dim, `axis=1` second. Use `other=float('-inf')` for max, `0.0` for sum.
- Available: `tl.sum`, `tl.max`, `tl.min`, `tl.argmax`, `tl.argmin`.

---

### Pattern 2: Custom Reductions

`tl.reduce` with an arbitrary associative combine function for non-standard reductions.

**When to use:** Argmax-with-value, products, or any reduction not in built-ins.

```python
@triton.jit
def _argmax_combine(val_a, idx_a, val_b, idx_b):
    prefer_a = (val_a > val_b) | ((val_a == val_b) & (idx_a < idx_b))
    return tl.where(prefer_a, val_a, val_b), tl.where(prefer_a, idx_a, idx_b)

@triton.jit
def argmax_kernel(input_ptr, val_out_ptr, idx_out_ptr, M, N, stride_m, stride_n,
                  BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    data = tl.load(input_ptr + row * stride_m + cols * stride_n,
                   mask=cols < N, other=float('-inf'))
    max_val, max_idx = tl.reduce((data, cols), axis=0, combine_fn=_argmax_combine)
    tl.store(val_out_ptr + row, max_val)
    tl.store(idx_out_ptr + row, max_idx)
```

- Combine function must be associative and commutative. Compiles to warp shuffle instructions.

---

### Pattern 3: Online Softmax (Flash Attention Style)

Maintains running max and sum to compute softmax across a dimension too large for one tile.

**When to use:** Flash Attention, softmax over long sequences, fused attention + weighted sum.

```python
m_i = float('-inf')               # Running max
l_i = 0.0                         # Running sum of exp
acc = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
for start_kv in range(0, seq_len, BLOCK_SEQ):
    # ... load K block, compute scores ...
    m_i_new = tl.maximum(m_i, tl.max(scores, axis=0))
    correction = tl.exp(m_i - m_i_new)      # Exact rescale factor
    l_i = l_i * correction + tl.sum(tl.exp(scores - m_i_new), axis=0)
    acc = acc * correction + ...             # Rescaled V accumulation
    m_i = m_i_new
```

- Keep `m_i`, `l_i`, `acc` in float32. Reduces memory from O(N^2) to O(N).
- For causal: `scores = tl.where(kv_offs <= pid, scores, float('-inf'))`.

> **Full example:** [EXAMPLES_ATTENTION.md §1](EXAMPLES_ATTENTION.md#1-flash-attention-v2-forward)

---

### Pattern 4: Welford's Algorithm for Variance

Online mean and variance in a single pass with numerical stability.

**When to use:** LayerNorm, GroupNorm, any kernel needing mean + variance.

```python
@triton.jit
def welford_kernel(input_ptr, mean_ptr, var_ptr, M, N, stride_m, stride_n,
                   BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    data = tl.load(input_ptr + row * stride_m + cols * stride_n,
                   mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(data, axis=0) / N
    centered = tl.where(mask, data - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, var)
```

- Use float32 accumulators even with float16 inputs.
- For rows wider than BLOCK_N, use the online Welford variant with block-wise merging.
- For LayerNorm: `rstd = 1 / sqrt(variance + eps)`. Save mean and rstd for backward.

> **Full examples:** [EXAMPLES_NORMALIZATION.md §1](EXAMPLES_NORMALIZATION.md#1-layer-normalization-forward)

---

### Pattern 5: Cross-Block Reduction with Atomics

Multiple programs compute partial results and combine them via atomic ops.

**When to use:** Column sums, histograms, any reduction requiring multiple programs.

```python
@triton.jit
def column_sum_atomic_kernel(
    input_ptr, output_ptr, M, N, stride_m, stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs = input_ptr + row_offs[:, None] * stride_m + col_offs[None, :] * stride_n
    data = tl.load(ptrs, mask=(row_offs[:, None] < M) & (col_offs[None, :] < N), other=0.0)
    partial_sum = tl.sum(data, axis=0)  # shape (BLOCK_N,)
    tl.atomic_add(output_ptr + col_offs, partial_sum, mask=col_offs < N)
```

- Output must be zeroed before launch. Atomics serialize at the same address.
- Available: `tl.atomic_add`, `tl.atomic_max`, `tl.atomic_min`, `tl.atomic_cas`, `tl.atomic_xchg`.

---

### Pattern 6: Two-Pass Reduction

First pass: per-block partials. Second pass: reduce partials. Deterministic, no atomics.

**When to use:** Large reductions with high contention, when determinism is required.

```python
@triton.jit
def reduction_pass1(input_ptr, partial_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = tl.load(input_ptr + offs, mask=offs < N, other=0.0)
    tl.store(partial_ptr + pid, tl.sum(data, axis=0))

@triton.jit
def reduction_pass2(partial_ptr, output_ptr, num_partials, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    partials = tl.load(partial_ptr + offs, mask=offs < num_partials, other=0.0)
    tl.store(output_ptr, tl.sum(partials, axis=0))
# Launch: pass1[(num_blocks,)](...), then pass2[(1,)](...)
```

- Partial buffer has one entry per first-pass program. Deterministic ordering.

---

## 3. Compute Patterns

### Pattern 1: K-Loop Tiling for Matmul

Accumulate partial dot products over K in tiles with fp32 accumulator.

**When to use:** Any matrix multiplication, convolution as implicit GEMM, linear layers.

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # ALWAYS fp32
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc = tl.dot(a, b, acc)   # Tensor cores when fp16/bf16, blocks multiple of 16
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
tl.store(c_ptrs, acc.to(tl.float16), mask=...)  # Cast only at the end
```

- Always fp32 accumulator. Typical: `BLOCK_M=BLOCK_N=128`, `BLOCK_K=32` or `64`.

> **Full example:** [EXAMPLES.md §3](EXAMPLES.md#3-matrix-multiplication)

---

### Pattern 2: Software Pipelining

Overlap memory loads with computation via `num_stages` on `tl.range`.

**When to use:** Memory-bound kernels with a dominant inner loop.

```python
for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
    a = tl.load(a_bptr, boundary_check=(0, 1))
    b = tl.load(b_bptr, boundary_check=(0, 1))
    acc = tl.dot(a, b, acc)
    a_bptr = tl.advance(a_bptr, (0, BLOCK_K))
    b_bptr = tl.advance(b_bptr, (BLOCK_K, 0))
```

- More stages = more shared memory. Typical: `num_stages=2` compute-bound, `3-5` memory-bound.
- Most effective with block pointers (compiler schedules async copies).

> **Full details:** [OPTIMIZATION.md §5](OPTIMIZATION.md#5-software-pipelining)

---

### Pattern 3: Persistent Kernels

Launch fewer programs than tiles; each iterates over multiple tiles.

**When to use:** Small problems, cross-tile coordination, better SM scheduling.

```python
@triton.jit
def persistent_matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    total_tiles = tl.cdiv(M, BLOCK_M) * num_tiles_n

    for tile_idx in tl.range(pid, total_tiles, NUM_SMS, num_stages=2):
        tile_m = tile_idx // num_tiles_n
        tile_n = tile_idx % num_tiles_n
        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
# Launch: persistent_matmul_kernel[(NUM_SMS,)](..., NUM_SMS=132)  # 132 for H100
```

- `NUM_SMS` = number of SMs (108 for A100, 132 for H100).
- Use `torch.cuda.get_device_properties(0).multi_processor_count`.

---

### Pattern 4: Fused Element-wise Operations

Combine multiple ops in a single kernel to eliminate intermediate stores.

**When to use:** Chains of element-wise ops (bias+activation, normalize+scale+shift).

```python
@triton.jit
def fused_bias_activation_kernel(
    input_ptr, bias_ptr, output_ptr, n_rows, n_cols, stride_row, stride_col,
    BLOCK_COLS: tl.constexpr, ACTIVATION: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    mask = cols < n_cols
    x = tl.load(input_ptr + row * stride_row + cols * stride_col, mask=mask, other=0.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    x = x + b  # Fuse: no intermediate store
    if ACTIVATION == "relu":
        x = tl.where(x > 0, x, 0.0)
    elif ACTIVATION == "gelu":
        x = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    elif ACTIVATION == "silu":
        x = x * tl.sigmoid(x)
    tl.store(output_ptr + row * stride_row + cols * stride_col, x, mask=mask)
```

- N fused ops reduces memory traffic by ~(N-1)/N.
- `ACTIVATION` is constexpr: compiler generates separate optimized kernels per type.

---

### Pattern 5: Compile-time Branching (constexpr)

`tl.constexpr` parameters are resolved at compile time for zero-cost branching.

**When to use:** Optional features (causal mask, bias, dropout), activation selection.

```python
@triton.jit
def flexible_kernel(
    x_ptr, bias_ptr, output_ptr, N, stride,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,     # Zero-cost compile-time flag
    IS_CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs * stride, mask=mask, other=0.0)
    if HAS_BIAS:  # Compiled away when False -- zero overhead
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        x = x + b
    if IS_CAUSAL:
        x = tl.where(offs <= pid, x, 0.0)
    tl.store(output_ptr + offs * stride, x, mask=mask)
```

- Each unique constexpr combination produces a separate compiled binary.
- Dead branches are eliminated entirely during compilation.
- Use `@triton.heuristics` to set flags from runtime args (see Autotuning Pattern 4).

---

## 4. Autotuning Patterns

### Pattern 1: Basic Autotuning

Run the kernel with multiple configs and select the fastest.

**When to use:** Any production kernel. When optimal block size depends on problem dimensions.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def autotuned_kernel(...): pass
# grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
```

- `key` determines when cached results are invalidated. Use `lambda META:` for config-dependent grids.

> **Full example:** [EXAMPLES.md §3](EXAMPLES.md#3-matrix-multiplication)

---

### Pattern 2: Config Pruning

Eliminate invalid/suboptimal configs before benchmarking.

**When to use:** Shared memory limits, many configs, problem-specific constraints.

Pass a pruning function via `prune_configs_by={'early_config_prune': fn}`. The function
receives `(configs, named_args, **kwargs)` and filters based on problem size, SRAM budget, etc.

> **Full example:** [OPTIMIZATION.md §3](OPTIMIZATION.md#config-pruning)

---

### Pattern 3: Pre/Post Hooks

Pre-hooks run before each trial (e.g., zeroing output for atomic kernels).

**When to use:** Atomic reduction kernels requiring zeroed outputs.

```python
def init_hook(nargs):
    nargs['output_ptr'].zero_()

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, pre_hook=init_hook),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, pre_hook=init_hook),
    ],
    key=['M', 'N'],
)
@triton.jit
def atomic_kernel(input_ptr, output_ptr, M, N, ...): pass
```

---

### Pattern 4: Heuristics

Compute constexpr values from runtime arguments at launch time.

**When to use:** Setting flags from optional args, derived constants (e.g., EVEN_K).

```python
@triton.autotune(configs=[...], key=['M', 'N', 'K'])
@triton.heuristics({
    'HAS_BIAS': lambda args: args['bias_ptr'] is not None,
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
})
@triton.jit
def heuristic_kernel(..., HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr): pass
```

- `@triton.heuristics` goes between `@triton.autotune` and `@triton.jit`.
- `EVEN_K` eliminates mask overhead in K-loop: 5-15% speedup.

> **Full example:** [API_ADVANCED.md §10](API_ADVANCED.md#heuristics)

---

### Pattern 5: Search Strategy Guidelines

| Parameter      | Values                 | Notes                                  |
|----------------|------------------------|----------------------------------------|
| `BLOCK_M/N`    | 16, 32, 64, 128, 256  | Larger = more reuse                    |
| `BLOCK_K`      | 16, 32, 64, 128       | Affects register usage                 |
| `num_warps`    | 2, 4, 8               | 4 for small blocks, 8 for large        |
| `num_stages`   | 2, 3, 4, 5            | Higher for memory-bound                |
| `num_ctas`     | 1 (Ampere), 1-2 (H+)  | Cooperative thread arrays              |
| `GROUP_SIZE_M` | 1, 4, 8, 16           | L2 cache grouping                      |

Start with 5-10 configs. Scale `num_warps` with block size. Include one small safe config.

---

## 5. Backward Pass Patterns

### Pattern 1: Recomputation Strategy

Save inputs, recompute forward activations in backward. Trades compute for memory.

**When to use:** Cheap activations (SiLU, GELU, ReLU), memory-constrained training.

```python
# In backward kernel: reload inputs, recompute intermediate values
gate_sig = tl.sigmoid(gate)       # Cheap: one exp + one div
silu_gate = gate * gate_sig        # Recomputed, not loaded from saved tensor
grad_x = dy * silu_gate
grad_gate = dy * x * (gate_sig + gate * gate_sig * (1.0 - gate_sig))
```

**Rule of thumb:** Cache if tensor is small (scalars, statistics) or expensive to recompute.
Recompute if tensor is activation-sized and the operation is elementwise.

> **Full example:** [EXAMPLES_TRAINING.md §1](EXAMPLES_TRAINING.md#1-swiglu-backward)

---

### Pattern 2: Saved Statistics Strategy

Save compact statistics (mean, rstd) from forward; reuse in backward.

**When to use:** Normalization layers where 2 scalars/row beats saving full normalized tensor.

```python
mean = tl.load(mean_ptr + row)
rstd = tl.load(rstd_ptr + row)
x_hat = (x - mean) * rstd  # Reconstructed from saved stats
wdy = dy * w
dx = rstd * (wdy - tl.sum(wdy) / N - x_hat * tl.sum(wdy * x_hat) / N)
```

> **Full examples:** [EXAMPLES_TRAINING.md §2](EXAMPLES_TRAINING.md#2-rmsnorm-backward), [EXAMPLES_TRAINING.md §3](EXAMPLES_TRAINING.md#3-layernorm-backward)

---

### Pattern 3: Fused Gradient Computation

Compute multiple gradients (dx, dw, db) loading each element once.

**When to use:** Ops with multiple learnable params; shared subexpressions across gradients.

```python
@triton.jit
def fused_linear_relu_bwd_kernel(
    grad_out_ptr, pre_relu_ptr, grad_x_ptr, grad_bias_ptr,
    M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    grad_out = tl.load(grad_out_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n,
                       mask=mask, other=0.0)
    pre_relu = tl.load(pre_relu_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n,
                       mask=mask, other=0.0)
    # Fuse: ReLU backward + bias gradient in one kernel
    grad_through = tl.where(pre_relu > 0, grad_out, 0.0)
    tl.store(grad_x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n,
             grad_through, mask=mask)
    tl.atomic_add(grad_bias_ptr + offs_n, tl.sum(grad_through, axis=0), mask=offs_n < N)
```

---

### Pattern 4: Parallel Reduction with Locks

Lock-based accumulation for weight gradients across many rows.

**When to use:** Weight gradients in normalization; when atomic contention is too high.

```python
# Acquire lock, accumulate into partial buffer, release
while tl.atomic_cas(lock_ptr + group_id, 0, 1) != 0:
    pass
count = tl.load(count_ptr + group_id)
if count == 0:
    tl.store(partial_ptr + group_id * N + offs_n, partial, mask=col_mask)
else:
    prev = tl.load(partial_ptr + group_id * N + offs_n, mask=col_mask, other=0.0)
    tl.store(partial_ptr + group_id * N + offs_n, prev + partial, mask=col_mask)
tl.store(count_ptr + group_id, count + 1)
tl.atomic_xchg(lock_ptr + group_id, 0)  # Release lock
```

- `GROUP_SIZE_M` typical values: 4, 8, 16. Lock and count buffers must be zeroed.

> **Full example:** [EXAMPLES_TRAINING.md §3](EXAMPLES_TRAINING.md#3-layernorm-backward)

---

### Pattern 5: torch.autograd.Function Integration

Wrap Triton kernels for PyTorch autograd compatibility.

**When to use:** Any Triton kernel participating in backpropagation.

```python
class TritonOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        output = torch.empty_like(x)
        _fwd_kernel[grid](x, w, output, ...)
        ctx.save_for_backward(x, w)   # Save inputs, not intermediates
        return output
    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        dx, dw = torch.empty_like(x), torch.empty_like(w)
        _bwd_kernel[grid](grad_output, x, w, dx, dw, ...)
        return dx, dw                  # Same order as forward args
```

- Return `None` for inputs not requiring gradients. Test with `torch.autograd.gradcheck`.

> **Full example:** [EXAMPLES_TRAINING.md §5](EXAMPLES_TRAINING.md#5-backward-pass-best-practices)

---

### Common Gradient Formulas

```
Sigmoid:     dx = dy * sigmoid(x) * (1 - sigmoid(x))
SiLU:        dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
GELU:        dx = dy * (0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi))
ReLU:        dx = dy * (x > 0)
Softmax:     dx_i = y_i * (dy_i - sum(dy_j * y_j))
LayerNorm:   dx = rstd/N * (N*dy*w - sum(dy*w) - x_hat*sum(dy*w*x_hat))
             dw = sum_rows(dy * x_hat),  db = sum_rows(dy)
RMSNorm:     dx = rstd * w * (dy - x * mean(dy*w*x) * rstd^2)
             dw = sum_rows(dy * x * rstd)
CrossEntropy: dx_i = softmax(x)_i - (i == target)
```

---

## Pattern Selection Summary

| Scenario                  | Recommended Patterns                            |
|---------------------------|------------------------------------------------|
| Element-wise op           | 1D Pointer Arithmetic + Fused Element-wise      |
| Matrix multiply           | K-loop Tiling + Grouped Ordering + Autotuning   |
| Row-wise reduction        | Tile-local Reductions                           |
| Full-tensor reduction     | Two-Pass or Cross-Block with Atomics            |
| Attention mechanism       | Online Softmax + 2D Tiling + constexpr Branching|
| Normalization forward     | Welford's + Saved Statistics                    |
| Normalization backward    | Saved Statistics + Parallel Reduction with Locks|
| Training-ready custom op  | autograd.Function + Recomputation/Saved Stats   |
| Performance optimization  | Autotuning + Pipelining + Coalescing            |
| Hopper-specific           | Block Pointers + TMA + Persistent Kernels       |
