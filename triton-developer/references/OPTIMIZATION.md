# Triton GPU Kernel Performance Optimization Guide

Performance optimization guide for Triton GPU kernels with Hopper+ focus.

---

## 1. Hardware Overview

### Architecture Comparison

| Feature | A100 (SM80) | H100 (SM90) | B200 (SM100) |
|---|---|---|---|
| FP16 TFLOPS | 312 | 990 | ~2000+ |
| BF16 TFLOPS | 312 | 990 | ~2000+ |
| FP8 TFLOPS | N/A | 1979 | ~4000+ |
| HBM Bandwidth | 2 TB/s | 3.35 TB/s | 8 TB/s |
| HBM Capacity | 80 GB (HBM2e) | 80 GB (HBM3) | 192 GB (HBM3e) |
| Shared Memory / SM | 164 KB | 228 KB | 228+ KB |
| L2 Cache | 40 MB | 50 MB | 64+ MB |
| SMs | 108 | 132 | 192 |
| Tensor Core Generation | 3rd Gen | 4th Gen | 5th Gen |
| TMA Support | No | Yes | Yes |
| Warp Specialization | No | Yes | Yes |
| FP8 (`dot_scaled`) | No | Partial | Yes |
| `num_ctas > 1` | No | Yes | Yes |
| Max Threads / SM | 2048 | 2048 | 2048 |
| Warp Size | 32 | 32 | 32 |
| Max Registers / Thread | 255 | 255 | 255 |

### Key Architectural Differences

**A100 (SM80) -- Ampere:**
- Baseline architecture for most Triton kernels.
- 3rd-gen Tensor Cores: FP16, BF16, TF32, INT8.
- No hardware TMA; all memory movement via `tl.load`/`tl.store`.
- Software pipelining via `num_stages` uses `cp.async` instructions.

**H100 (SM90) -- Hopper:**
- 4th-gen Tensor Cores add FP8 (E4M3, E5M2).
- Hardware TMA: asynchronous bulk copies without occupying compute resources.
- Warp specialization: dedicate warps to producer (memory) or consumer (compute).
- Distributed shared memory: CTAs in a cluster access each other's SRAM (`num_ctas > 1`).

**B200 (SM100) -- Blackwell:**
- 5th-gen Tensor Cores with native `dot_scaled` for mixed FP8/FP6/FP4.
- 8 TB/s HBM bandwidth significantly relaxes memory-bound limits.
- All Hopper features carry forward and are refined.

### Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes Moved

A100: 312 TFLOPS / 2 TB/s    = 156 FLOPs/byte  (FP16)
H100: 990 TFLOPS / 3.35 TB/s = 295 FLOPs/byte  (FP16)
B200: 2000 TFLOPS / 8 TB/s   = 250 FLOPs/byte  (FP16)
```

Kernels below these thresholds are memory-bound; above, compute-bound.
Most element-wise/reduction kernels are memory-bound. Matmul and attention
are typically compute-bound for large sizes.

---

## 2. Memory Hierarchy

```
                    Bandwidth        Latency       Size
                    ---------        -------       ----
  HBM (Global)     2-8 TB/s         ~400 cycles   80-192 GB
       |
       v
  L2 Cache          ~6-12 TB/s      ~200 cycles   40-64 MB
       |
       v
  SRAM (Shared)     ~19 TB/s        ~30 cycles    164-228 KB / SM
       |
       v
  Registers          Unlimited*      0 cycles      256 KB / SM
       |
       v
  Tensor Cores       ---             ---           Compute
```

### HBM (Global Memory)
- **Bandwidth:** 2 TB/s (A100) to 8 TB/s (B200). **Latency:** ~400-600 cycles.
- All `tl.load()` / `tl.store()` access HBM.
- Coalesced accesses (consecutive threads, consecutive addresses) are critical.
  Triton handles this automatically for standard pointer arithmetic patterns.
- Use `tl.multiple_of(ptr, 16)` to hint 16-byte alignment for wider loads.

### L2 Cache
- **Size:** 40 MB (A100), 50 MB (H100), 64+ MB (B200). Shared across all SMs.
- Managed by hardware, not the programmer.
- Grouped ordering (Section 6) improves L2 hit rate by scheduling spatially
  related thread blocks close together in time.

### Shared Memory (SRAM)
- **Size:** 164 KB/SM (A100), 228 KB/SM (H100/B200). **Latency:** ~30 cycles.
- Triton manages SRAM automatically (unlike CUDA `__shared__`).
- Controlled via `num_stages`: more stages = more SRAM buffers for prefetching.
- **SRAM budget estimate:**
  ```
  SRAM_per_stage = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * element_size
  Total_SRAM = SRAM_per_stage * num_stages
  ```
  Example: FP16, BLOCK_M=128, BLOCK_K=64, BLOCK_N=256, num_stages=3:
  `(128*64 + 64*256) * 2 * 3 = 144 KB`

### Registers
- Zero-latency. 255 per thread, 256 KB per SM.
- Accumulator tensors (`tl.zeros(...)` for `tl.dot`) live in registers.
- **Register spilling** occurs when the compiler needs more registers than
  available, falling back to local memory (HBM speed, ~400-cycle penalty).
- Symptoms: high "local memory" traffic in Nsight Compute.
- Fix: reduce BLOCK_M/BLOCK_N or use fewer intermediate variables.

---

## 3. Autotuning Deep Dive

### Block Size Selection

**BLOCK_M and BLOCK_N** (output tile dimensions):
- Values: 16, 32, 64, 128, 256. Must be powers of 2 for Tensor Cores.
- Larger blocks = better Tensor Core utilization, but more register/SRAM pressure.

**BLOCK_K** (reduction dimension): 16, 32, 64, 128.
- Must be >= 16 for FP16/BF16 Tensor Cores. 32 is a safe default.

**Starting points by architecture:**

| Architecture | BLOCK_M | BLOCK_N | BLOCK_K |
|---|---|---|---|
| A100 (SM80) | 128 | 128 | 32 |
| H100 (SM90) | 128 | 256 | 64 |
| B200 (SM100) | 128 | 256 | 64 |

### num_warps Selection

| num_warps | Threads | Best For |
|---|---|---|
| 4 | 128 | Small-medium blocks (64x64), memory-bound kernels |
| 8 | 256 | Large blocks (128x128+), compute-bound kernels |
| 16 | 512 | Very large blocks (256x256), Hopper warp specialization |

**Rule of thumb:**
```python
num_warps = max(4, min(8, BLOCK_M * BLOCK_N // 256))
```

### num_stages Selection

Controls software pipeline depth (how many K-blocks prefetched while computing).

| num_stages | Buffering | Best For |
|---|---|---|
| 2 | Double | Minimum for pipelining. SRAM-tight configs. |
| 3 | Triple | Good default. Hides most memory latency. |
| 4-5 | Deep | Memory-bound kernels with small blocks. Requires more SRAM. |

**Constraint:** `num_stages * tile_sram_bytes <= available_sram_per_sm`

If the compiler cannot fit the pipeline, compilation fails with "out of shared
memory." Reduce `num_stages` or block sizes.

### num_ctas (Hopper+ Only)

Controls Thread Block Cluster size. CTAs in a cluster share SRAM via DSMEM.
- `num_ctas=1`: Default. Each CTA is independent.
- `num_ctas=2`: Two CTAs share SRAM. Try for very large matmuls on H100.
- Measure carefully -- can hurt if cluster scheduling reduces occupancy.

### Config Pruning

**Pattern overview:** See [PATTERNS.md §4 — Autotuning Patterns](PATTERNS.md#4-autotuning-patterns) for the full catalog of autotuning patterns (basic, pruning, hooks, heuristics).

```python
def prune_configs(configs, named_args, **kwargs):
    """Remove configs that exceed SRAM or are too large for the problem."""
    M = named_args.get('M', kwargs.get('M', 1024))
    N = named_args.get('N', kwargs.get('N', 1024))
    pruned = []
    for cfg in configs:
        bm = cfg.kwargs['BLOCK_M']
        bn = cfg.kwargs['BLOCK_N']
        bk = cfg.kwargs['BLOCK_K']
        ns = cfg.num_stages
        # Skip if block is larger than problem dimension
        if bm > M or bn > N:
            continue
        # Estimate SRAM usage (FP16 = 2 bytes per element)
        sram = (bm * bk + bk * bn) * 2 * ns
        if sram > 200_000:  # Conservative limit (200 KB)
            continue
        pruned.append(cfg)
    return pruned

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
                      num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
    prune_configs_by={'early_config_prune': prune_configs},
)
@triton.jit
def matmul_kernel(...):
    ...
```

**Autotuning tips:**
- Always include `key=[...]` with problem dimensions that affect performance.
- Start with 4-8 configs spanning small-to-large blocks.
- Use `prune_configs_by` to eliminate infeasible configs before benchmarking.
- Cache lives in `.triton/cache/`; delete when changing hardware or Triton version.
- Use `triton.testing.do_bench()` for standalone microbenchmarking.

---

## 4. Tensor Memory Accelerator (TMA) on Hopper+

TMA is a dedicated hardware unit on H100+ that performs asynchronous bulk
data movement between global and shared memory, offloading address computation
and boundary handling from the SM.

### When TMA Helps
- Matmul and attention kernels with regular, dense, strided access patterns.
- Large block sizes (128+ per dimension) where descriptor overhead is amortized.
- Pipelined kernels where TMA copies overlap with Tensor Core compute.
- Irregular boundaries -- TMA handles out-of-bounds automatically with zero-fill.

### When TMA Does NOT Help
- Gather/scatter with non-contiguous indices.
- Very small blocks where descriptor setup overhead dominates.
- Element-wise kernels with simple coalesced linear access.

### How to Use TMA in Triton

**Step 1: Create a TMA descriptor (host-side).**
```python
from triton.tools.experimental_descriptor import create_2d_tma_descriptor

desc_a = create_2d_tma_descriptor(
    ptr=a.data_ptr(),       # Raw pointer to the tensor
    dim1=M,                  # First logical dimension
    dim0=K,                  # Second logical dimension (contiguous)
    block_dim1=BLOCK_M,      # Tile size in dim1
    block_dim0=BLOCK_K,      # Tile size in dim0
    element_ty=tl.float16,
)
```

**Step 2: Pass descriptor to kernel (replaces pointer + strides).**
```python
matmul_kernel[grid](desc_a, desc_b, c_ptr, stride_cm, stride_cn, M, N, K)
```

**Step 3: Use `desc.load()` in kernel instead of `tl.load()`.**
```python
@triton.jit
def matmul_kernel(
    desc_a, desc_b,
    c_ptr, stride_cm, stride_cn,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, K, BLOCK_K, num_stages=3):
        # TMA loads -- no mask, no pointer arithmetic
        a = desc_a.load([pid_m * BLOCK_M, k])
        b = desc_b.load([k, pid_n * BLOCK_N])
        accumulator = tl.dot(a, b, acc=accumulator)

    # Store result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### TMA Limitations
- **Descriptor block shape is compile-time:** `block_dim0`/`block_dim1` must
  match the kernel's constexpr block sizes.
- **Autotuning with TMA:** If block sizes vary, recreate descriptors in `pre_hook`:
  ```python
  def tma_pre_hook(nargs):
      BLOCK_M = nargs['BLOCK_M']
      BLOCK_K = nargs['BLOCK_K']
      nargs['desc_a'] = create_2d_tma_descriptor(
          nargs['a_ptr'], nargs['M'], nargs['K'], BLOCK_M, BLOCK_K, tl.float16
      )
  ```
- **Requires Hopper or later.** Will fail on A100 or earlier.
- **Column-major tensors:** Swap dim0/dim1 and block_dim0/block_dim1.

---

## 5. Software Pipelining

Overlaps memory loads with computation across loop iterations. The single
most impactful optimization for reduction loops (e.g., K-loop in matmul).

### How It Works

Without pipelining (num_stages=1):
```
Iter 0: [Load tile] [Compute on tile]
Iter 1:             [Load tile] [Compute on tile]
```

With double buffering (num_stages=2):
```
Iter 0: [Load 0] [Compute 0]
Iter 1: [Load 1] [Compute 0] [Compute 1]
Iter 2: [Load 2] [Compute 1] [Compute 2]   <-- load/compute overlap
```

With triple buffering (num_stages=3):
```
        [Load 0][Load 1][Load 2][Load 3]...
                [Comp 0][Comp 1][Comp 2][Comp 3]...
                         ^^^ Two loads ahead, fully hiding latency
```

### Triton Syntax

Use `tl.range` with the `num_stages` parameter:
```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in tl.range(0, K, BLOCK_K, num_stages=3):
    a = tl.load(a_ptrs + k * stride_ak)
    b = tl.load(b_ptrs + k * stride_bk)
    accumulator = tl.dot(a, b, acc=accumulator)
```

The compiler automatically allocates `num_stages` SRAM buffers, issues async
copies (`cp.async` on A100, TMA on H100) for future iterations, and inserts
barriers to ensure data readiness.

### When to Use
- **K-loops in matmul** (canonical use case).
- **Persistent kernels** iterating over rows (streaming softmax).
- **Any loop** where each iteration independently loads and computes.

### When NOT to Use
- Loops with data dependencies between iterations (sequential scans).
- Single-iteration loops.
- Already compute-bound kernels where memory latency is fully hidden.

### Trade-offs

| More Stages | Fewer Stages |
|---|---|
| Better latency hiding | Less SRAM consumption |
| Higher SRAM usage | May stall on memory |
| Risk of SRAM overflow | Safer for large blocks |

SRAM budget: `num_stages * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * dtype_bytes`

---

## 6. L2 Cache Optimization

### The Problem with Naive Ordering

With row-major launch order, blocks in the same row share A tiles but differ
in B tiles. By the time the next row starts, A tiles are evicted from L2.

### Grouped (Swizzled) Ordering

```python
GROUP_SIZE_M: tl.constexpr = 8

pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)

# Number of programs in a group
num_pid_in_group = GROUP_SIZE_M * num_pid_n

# Which group this program belongs to
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

# Column-major order within the group
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

**Why it works:** Within a group of `GROUP_SIZE_M` rows, all blocks iterate
over the same K tiles of B. These stay in L2 while the group processes them.

**Impact and guidelines:**
- **Typical speedup:** 10-15% on large matmuls (M, N >= 2048).
- **GROUP_SIZE_M:** 8 is a good default.
- **When to skip:** Small matrices fitting entirely in L2, or 1D kernels
  (element-wise, reductions) with no 2D reuse pattern.

---

## 7. Performance Checklist

### Correctness First
- [ ] **fp32 accumulator** for `tl.dot` (MOST IMPORTANT). Tensor Cores require
      `tl.float32` accumulators; fp16 falls back to non-Tensor-Core paths.
- [ ] **Boundary masking** on all `tl.load()`/`tl.store()` where indices may
      exceed tensor dimensions. Use `other=0.0` for loads.

### Block and Launch Configuration
- [ ] **Power-of-2 block sizes** (16, 32, 64, 128, 256). Non-power-of-2
      prevents Tensor Core usage.
- [ ] **Autotune** with 4-8 configs spanning BLOCK_M/N/K, num_warps, num_stages.
- [ ] **num_warps:** 4 for small blocks (64x64), 8 for large blocks (128x128+).
- [ ] **num_stages >= 2** for any loop that loads and computes.

### Memory Access
- [ ] **Grouped ordering** for 2D tiled kernels (matmul, attention).
- [ ] `tl.multiple_of()` / `tl.max_contiguous()` hints for known alignments.
      Enables wider (128-bit) memory transactions.
- [ ] **Coalesced access:** consecutive threads access consecutive addresses.

### Hopper+ Specific
- [ ] **TMA descriptors** for regular access patterns on H100/B200.
- [ ] **num_ctas=2** for very large matmuls on Hopper+.

### Profiling
- [ ] **Nsight Compute** (`ncu`) to identify bottleneck:
      - Compute-bound: high SM utilization, high Tensor Core active cycles.
      - Memory-bound: high HBM utilization, low SM utilization.
      - Latency-bound: low utilization everywhere (increase occupancy/stages).
- [ ] Compare against **cuBLAS / Flash Attention** baselines via `do_bench()`.
- [ ] Check for **register spilling** (local memory metrics in Nsight Compute).
