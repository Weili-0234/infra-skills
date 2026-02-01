# cutile Optimization Guide

Performance optimization techniques for cutile kernels.

## Table of Contents

1. [TMA (Tensor Memory Accelerator)](#tma-tensor-memory-accelerator)
2. [Persistent Kernels](#persistent-kernels)
3. [Memory Layout Optimization](#memory-layout-optimization)
4. [Tile Size Selection](#tile-size-selection)

---

## TMA (Tensor Memory Accelerator)

TMA is a Hopper/Blackwell hardware feature that asynchronously transfers data between global and shared memory.

### What is TMA?

- Hardware unit separate from compute cores
- Async data movement (compute + memory overlap)
- Better bandwidth utilization than manual loads
- Supports multi-dimensional transfers

### Using Latency Hints

The `latency` parameter in `ct.load()` controls TMA pipeline depth:

```python
# Latency values and their meaning:
tile = ct.load(array, index, shape, latency=2)   # Use immediately (next few instructions)
tile = ct.load(array, index, shape, latency=3)   # Use in ~10 instructions
tile = ct.load(array, index, shape, latency=4)   # Use in next loop iteration
tile = ct.load(array, index, shape, latency=10)  # Prefetch for future use
```

**Guidelines:**
1. **Immediate use** (`latency=2`): Data needed right away
2. **Next iteration** (`latency=3-4`): Load for upcoming computation
3. **Prefetch** (`latency=10`): Load early, use much later

### TMA Example: Flash Attention

```python
@ct.kernel(occupancy=2)
def fmha_kernel(Q, K, V, Out, TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bidx = ct.bid(0)

    # Q loaded with low latency (used immediately)
    q = ct.load(Q, index=(batch, head, bidx, 0),
                shape=(1, 1, TILE_M, TILE_D),
                latency=2)

    for k_idx in range(num_kv_tiles):
        # K loaded with medium latency (used soon in mma)
        k = ct.load(K, index=(batch, head, 0, k_idx),
                    shape=(1, 1, TILE_D, TILE_N),
                    order=(0, 1, 3, 2),  # Transpose
                    latency=2)

        # Compute QK while V is loading
        qk = ct.mma(q, k, qk_acc)

        # V loaded with higher latency (used after softmax)
        v = ct.load(V, index=(batch, head, k_idx, 0),
                    shape=(1, 1, TILE_N, TILE_D),
                    latency=4)

        # Softmax computation (overlaps with V load)
        p = online_softmax(qk, m_i, l_i)

        # Use V (should be ready by now)
        acc = ct.mma(p, v, acc)
```

### When TMA Helps Most

✅ **Good for TMA:**
- Sequential access patterns
- Predictable memory access
- Memory-bound kernels
- Attention mechanisms

❌ **Less beneficial:**
- Compute-bound kernels
- Random access patterns
- Very small tiles
- Single load per kernel

### Controlling TMA

```python
# Disable TMA if causing issues
tile = ct.load(array, index, shape, allow_tma=False)

# TMA is enabled by default
tile = ct.load(array, index, shape, allow_tma=True)
```

**When to disable:**
- Very small tiles (<16 elements)
- Complex indexing patterns
- Debugging memory issues

---

## Persistent Kernels

Persistent kernels process multiple tiles per thread block for better SM utilization.

### Standard vs Persistent

**Standard kernel:**
```python
# One block = one tile
# Grid size = total tiles
grid = (ct.cdiv(M, tm) * ct.cdiv(N, tn), 1, 1)
```

**Persistent kernel:**
```python
# One block = multiple tiles
# Grid size = number of SMs (or fewer)
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
grid = (NUM_SMS, 1, 1)
```

### Implementation Pattern

```python
@ct.kernel
def persistent_matmul(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    bid = ct.bid(0)
    num_tile_blocks = ct.num_blocks(0)

    M, N = A.shape[0], B.shape[1]
    total_tiles = ct.cdiv(M, tm) * ct.cdiv(N, tn)

    # Process multiple tiles per block
    for tile_id in range(bid, total_tiles, num_tile_blocks):
        # Compute tile coordinates
        tile_m = tile_id // ct.cdiv(N, tn)
        tile_n = tile_id % ct.cdiv(N, tn)

        # Standard tile processing
        accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
        for k in range(ct.cdiv(A.shape[1], tk)):
            a = ct.load(A, (tile_m, k), (tm, tk), padding_mode=ct.PaddingMode.ZERO)
            b = ct.load(B, (k, tile_n), (tk, tn), padding_mode=ct.PaddingMode.ZERO)
            accumulator = ct.mma(a, b, accumulator)

        ct.store(C, (tile_m, tile_n), accumulator.astype(C.dtype))
```

### When to Use Persistent Kernels

✅ **Use when:**
- Total tiles >> number of SMs (>4x)
- Unbalanced workload across tiles
- Want to maximize SM occupancy
- Large matrices (M, N > 2048)

❌ **Avoid when:**
- Small problem sizes (total tiles < 100)
- Tiles have vastly different costs
- Each tile is very expensive (>1ms)

### Determining Grid Size

```python
def get_persistent_grid_size(total_tiles, num_sms):
    """Calculate optimal grid size for persistent kernel."""
    if total_tiles <= num_sms:
        return total_tiles  # Standard launch

    # Use all SMs, but don't exceed tiles
    return min(num_sms, total_tiles)

# Example
M, N = 4096, 4096
tm, tn = 128, 128
total_tiles = ct.cdiv(M, tm) * ct.cdiv(N, tn)  # 32 * 32 = 1024 tiles

NUM_SMS = 144  # B200
grid_size = get_persistent_grid_size(total_tiles, NUM_SMS)  # 144
```

### Performance Considerations

**Benefits:**
- Better SM utilization (fewer idle periods)
- Reduced launch overhead
- Dynamic load balancing
- Better for irregular workloads

**Costs:**
- More complex indexing
- Potential load imbalance if tiles vary in cost
- May reduce cache locality

---

## Memory Layout Optimization

### 2D Swizzling for L2 Cache

Reorder block execution to improve L2 cache hit rate.

```python
def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    """Compute swizzled block IDs for better L2 locality."""
    bid = ct.bid(0)

    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n

    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)

    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m

    return bid_m, bid_n
```

**How it works:**
- Group adjacent rows together
- Process all columns for group before moving to next
- Improves reuse of B matrix tiles in A@B

**GROUP_SIZE_M selection:**
- **8**: Good default for most cases
- **4**: For very large N (wide matrices)
- **16**: For square matrices with good L2

### Transposed Storage

Use `order` parameter to avoid explicit transpose kernels.

```python
# Instead of:
K_T = torch.transpose(K, -2, -1)  # Separate kernel
k_tile = ct.load(K_T, index, shape)

# Do this:
k_tile = ct.load(K, index, shape, order=(0, 1, 3, 2))  # Transpose during load
```

**Benefits:**
- No separate transpose kernel
- Better memory efficiency
- Reduced kernel launch overhead

**Common use cases:**
- Attention K matrix for QK^T
- Transpose in MoE expert routing
- MLA decoding

---

## Tile Size Selection

Choosing optimal tile sizes with power-of-2 constraint.

### Tile Size Constraints

1. **Must be power of 2**: 16, 32, 64, 128, 256, 512, 1024
2. **Total elements**: ~256-1024 for good occupancy
3. **Register pressure**: FP32 uses 2x registers of FP16
4. **K alignment**: 16, 32, or 64 for Tensor Core efficiency

### Selection Heuristics

```python
def select_matmul_tiles(M, N, K, dtype, gpu_arch):
    """Select optimal tile sizes for matrix multiplication."""

    itemsize = dtype.itemsize  # 2 for FP16, 4 for FP32

    if gpu_arch == "B200":  # Blackwell
        if itemsize == 2:  # FP16
            if M >= 4096 and N >= 4096:
                return 256, 256, 64  # Large tiles
            elif M >= 2048 or N >= 2048:
                return 128, 256, 64  # Medium-large tiles
            else:
                return 128, 128, 64  # Medium tiles
        else:  # FP32
            if M >= 2048 and N >= 2048:
                return 128, 128, 32  # Medium tiles
            else:
                return 64, 64, 32  # Small-medium tiles

    elif gpu_arch == "RTX5090":
        if itemsize == 2:
            return 128, 128, 64  # Conservative for consumer GPU
        else:
            return 64, 64, 32

    elif gpu_arch == "RTX5080":
        if itemsize == 2:
            return 64, 128, 64  # Smaller tiles
        else:
            return 32, 64, 32

# Usage
tm, tn, tk = select_matmul_tiles(M, N, K, torch.float16, "B200")
```

### Problem-Specific Tuning

**Attention (FMHA):**
```python
# Prefill (square attention)
TILE_M = 128  # Query tile
TILE_N = 128  # KV tile
TILE_D = 64   # Head dimension (often fixed)

# Decode (narrow attention)
TILE_M = 64   # Smaller query tile (single token)
TILE_N = 256  # Larger KV tile (full context)
```

**Normalization (LayerNorm/RMSNorm):**
```python
# Row-wise operations
TILE_M = 1    # One row per block
TILE_N = 1024  # Power-of-2 covering row length
```

**Element-wise:**
```python
# Maximize parallelism
TILE = 1024   # Largest power-of-2 up to 1024
```

### Occupancy vs Tile Size Trade-off

**Small tiles (32x32):**
- ✅ Higher occupancy (more blocks)
- ✅ Less register pressure
- ❌ More kernel launches overhead
- ❌ Less work per block

**Large tiles (256x256):**
- ✅ More work per block
- ✅ Better instruction-level parallelism
- ❌ Lower occupancy
- ❌ Higher register pressure

**Sweet spot:** 128x128 or 128x256 for FP16 on Blackwell

### Autotuning

Use experimental autotuner to find optimal configuration:

```python
from types import SimpleNamespace

result = ct_experimental.autotune_launch(
    torch.cuda.current_stream(),
    grid_fn=lambda cfg: (ct.cdiv(M, cfg.tm), ct.cdiv(N, cfg.tn), 1),
    kernel=matmul_kernel,
    args_fn=lambda cfg: (A, B, C, cfg.tm, cfg.tn, cfg.tk),
    hints_fn=lambda cfg: {
        "num_ctas": cfg.num_ctas,
        "occupancy": cfg.occupancy
    },
    search_space=[
        SimpleNamespace(tm=256, tn=256, tk=64, num_ctas=1, occupancy=1),
        SimpleNamespace(tm=128, tn=256, tk=64, num_ctas=1, occupancy=2),
        SimpleNamespace(tm=128, tn=128, tk=64, num_ctas=2, occupancy=2),
        SimpleNamespace(tm=64, tn=128, tk=64, num_ctas=1, occupancy=4),
        SimpleNamespace(tm=64, tn=64, tk=32, num_ctas=2, occupancy=4),
    ]
)

best_config = result.tuned_config
```

---

## Performance Checklist

Before deploying a kernel, verify:

- [ ] Tile dimensions are powers of 2
- [ ] Float32 accumulators for matrix operations
- [ ] K dimension aligned to 16/32 for Tensor Cores
- [ ] TMA latency hints for memory-bound kernels
- [ ] 2D swizzling for large matrix multiplications
- [ ] Persistent scheduling for large problem sizes
- [ ] Transposed loads instead of separate transpose kernels
- [ ] Appropriate tile sizes for target GPU
- [ ] Occupancy hint set (1-4, typically 2)
- [ ] Flush-to-zero enabled for divisions/exp

## Profiling with Nsight Compute

```bash
# Profile kernel
ncu --set full --target-processes all python script.py

# Key metrics to check:
# - SM occupancy (target: >50%)
# - Memory throughput (target: >60% of theoretical)
# - Tensor Core utilization (for mma operations)
# - L2 cache hit rate (target: >80% for matmul)
```
