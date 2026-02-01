---
name: cutile-developer
description: |
  GPU kernel development with NVIDIA cutile, a tile-level Pythonic DSL for Blackwell GPUs.

  Use this skill when:
  - User mentions "cutile", "CUDA Tile", "cuda.tile", "ct.kernel", or "@ct.kernel"
  - User asks to write GPU kernels for Blackwell/B200/RTX 5080/RTX 5090
  - User wants to optimize kernels using TMA (Tensor Memory Accelerator), persistent scheduling, or tile-based programming
  - User references cutile examples, TileGym kernels, or NVIDIA tile-level programming
  - User asks about matrix multiplication, attention (FMHA/FlashAttention), normalization (LayerNorm/RMSNorm), or other ML operators on GPU
  - User needs to port Triton or TileLang kernels to cutile
---

# cutile-developer

## Overview

**cutile** (CUDA Tile) is NVIDIA's Pythonic DSL for tile-based GPU programming on Blackwell architecture. It compiles Python through HIR → IR → Bytecode → MLIR → GPU cubin, enabling high-performance kernel development with a simpler API than raw CUDA.

**Key characteristics:**
- **Pythonic syntax**: `@ct.kernel` decorator, familiar Python operations
- **Tile-based programming**: Explicit tiling with power-of-2 constraint
- **Two memory access patterns**: Load/store (structured) vs gather/scatter (indexed)
- **TMA support**: Tensor Memory Accelerator hints for Blackwell/Hopper
- **Immutable tiles**: Functional programming style

**Comparison with similar frameworks:**

| Feature | cutile | Triton | TileLang |
|---------|--------|--------|----------|
| Decorator | `@ct.kernel` | `@triton.jit` | `@tilelang.jit` |
| Memory access | load/store + gather/scatter | load/store | T.copy |
| Tile constraints | Power-of-2 dims | Flexible | Flexible |
| Launch | `ct.launch(stream, grid, kernel, args)` | `kernel[grid](*args)` | `func(*args)` |
| Target GPUs | Blackwell+ (SM 10.0+) | NVIDIA GPUs | NVIDIA/AMD/CPU |

### System Requirements

- **GPU**: NVIDIA Blackwell (B200, RTX 5080, RTX 5090) - Compute capability 10.0+
- **Driver**: r580 or later
- **CUDA Toolkit**: 13.1+
- **Python**: 3.10-3.13

**Installation:**
```bash
pip install cuda-tile  # PyPI installation
# OR
pip install -e /path/to/cutile-python  # Development install
```

## Core Workflow

### Step 1: Import and Setup

```python
import cuda.tile as ct
import torch
import math

ConstInt = ct.Constant[int]  # Type hint for compile-time constants
```

### Step 2: Define Kernel

Use `@ct.kernel` decorator with optional hints (`num_ctas`, `occupancy`):

```python
@ct.kernel
def my_kernel(input_array, output_array, TILE_SIZE: ConstInt):
    # Kernel implementation
    pass
```

### Step 3: Implement Logic

**Core operations:**
- Block indexing: `ct.bid(0)`, `ct.num_blocks(0)`, `ct.num_tiles(array, axis, shape)`
- Memory access: `ct.load()`, `ct.store()`, `ct.gather()`, `ct.scatter()`
- Compute: `ct.mma()`, `ct.sum()`, `ct.max()`, `ct.exp2()`, `ct.where()`
- Tile creation: `ct.full()`, `ct.arange()`

### Step 4: Launch Kernel

```python
# Calculate grid dimensions
grid = (math.ceil(N / TILE_SIZE), 1, 1)

# Launch on current CUDA stream
ct.launch(torch.cuda.current_stream(), grid, my_kernel,
          (input_tensor, output_tensor, TILE_SIZE))
```

### Step 5: Validate Results

```python
# Compare with reference implementation
torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)
```

## API Quick Reference

### Types

**Array** - Global memory (HBM), DLPack/CUDA Array Interface compatible
```python
array.dtype  # Data type
array.shape  # Dimensions
array.ndim   # Number of dimensions
array.slice(axis, start, stop)  # Create view
```

**Tile** - On-chip data (registers/shared memory), immutable, **power-of-2 dimensions**
```python
tile.dtype  # Data type
tile.shape  # Dimensions (must be powers of 2)
tile.astype(dtype)  # Type conversion
tile.reshape(shape)  # Reshape
tile.permute(axes)   # Permute axes
```

**Constant[T]** - Compile-time constant embedding
```python
TILE_SIZE: ct.Constant[int]
CAUSAL: ct.Constant[bool]
```

### Decorators

```python
@ct.kernel(num_ctas=2, occupancy=2)  # Kernel with hints
def kernel_name(arrays, constants):
    pass

@ct.function  # Reusable device function
def helper(tile):
    return tile * 2
```

### Memory Operations

**Load/Store (Structured Access)**
```python
# Load tile from array
tile = ct.load(array, index=(block_x, block_y), shape=(TILE_M, TILE_N),
               padding_mode=ct.PaddingMode.ZERO, latency=2)

# Store tile to array
ct.store(array, index=(block_x, block_y), tile=result_tile)
```

**Gather/Scatter (Indexed Access)**
```python
# Compute indices
indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

# Gather with automatic boundary handling
tile = ct.gather(array, indices)  # Out-of-bounds → zero

# Scatter with automatic boundary masking
ct.scatter(array, indices, tile)  # Out-of-bounds ignored
```

### Compute Operations

```python
# Matrix multiply-accumulate (uses Tensor Cores)
accumulator = ct.mma(a_tile, b_tile, accumulator)

# Reductions
max_val = ct.max(tile, axis=-1, keepdims=True)
sum_val = ct.sum(tile, axis=1)

# Math functions
y = ct.exp2(x, flush_to_zero=True)
y = ct.log(x)
y = ct.sqrt(x)
y = ct.tanh(x)

# Element-wise
result = ct.where(mask, true_val, false_val)
result = a + b - c * d / e

# Atomic operations (for global reductions)
ct.atomic_add(output_array, (0,), scalar_tile)
```

### Utilities

```python
# Grid/block info
block_id = ct.bid(0)  # Block index (axis: 0, 1, 2)
num_blocks = ct.num_blocks(0)
num_tiles = ct.num_tiles(array, axis=1, shape=(TILE_M, TILE_K))

# Tile creation
zeros = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
range_tile = ct.arange(TILE_SIZE, dtype=torch.int32)

# Math
cdiv_result = ct.cdiv(N, TILE_SIZE)  # Ceiling division
```

## Essential Examples

### Example 1: Vector Addition

Demonstrates gather/scatter pattern with boundary handling.

```python
import cuda.tile as ct
import torch
import math

ConstInt = ct.Constant[int]

@ct.kernel
def vec_add_kernel(a, b, c, TILE: ConstInt):
    """Element-wise vector addition using gather/scatter."""
    bid = ct.bid(0)

    # Compute indices for this block's tile
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

    # Gather elements (auto-handles boundaries)
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)

    # Compute
    result = a_tile + b_tile

    # Scatter result (auto-masks out-of-bounds)
    ct.scatter(c, indices, result)

def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper function for vector addition."""
    N = a.shape[0]
    TILE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1

    c = torch.empty_like(a)
    grid = (math.ceil(N / TILE), 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, vec_add_kernel,
              (a, b, c, TILE))
    return c

# Usage
a = torch.randn(10000, dtype=torch.float32, device='cuda')
b = torch.randn(10000, dtype=torch.float32, device='cuda')
c = vec_add(a, b)

# Validate
torch.testing.assert_close(c, a + b)
```

### Example 2: Matrix Multiplication

Shows K-loop tiling, mma operation, and persistent scheduling.

```python
import cuda.tile as ct
import torch
from math import ceil

ConstInt = ct.Constant[int]

@ct.kernel
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """Matrix multiplication: C = A @ B using tiled approach."""
    bidx, bidy = ct.bid(0), ct.bid(1)

    # Initialize accumulator (use float32 for precision)
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)

    # K-dimension tiling loop
    num_k_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    for k in range(num_k_tiles):
        # Load tiles
        a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                    padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                    padding_mode=ct.PaddingMode.ZERO)

        # Matrix multiply-accumulate
        accumulator = ct.mma(a, b, accumulator)

    # Convert to output dtype and store
    result = accumulator.astype(C.dtype)
    ct.store(C, index=(bidx, bidy), tile=result)

def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for matrix multiplication."""
    m, k = A.shape
    _, n = B.shape

    # Tile sizes (power of 2, optimized for dtype)
    if A.dtype.itemsize == 2:  # float16/bfloat16
        tm, tn, tk = 128, 256, 64
    else:  # float32
        tm, tn, tk = 32, 32, 32

    C = torch.empty((m, n), device=A.device, dtype=A.dtype)
    grid = (ceil(m / tm), ceil(n / tn), 1)

    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel,
              (A, B, C, tm, tn, tk))
    return C

# Usage
A = torch.randn(512, 768, dtype=torch.float16, device='cuda')
B = torch.randn(768, 512, dtype=torch.float16, device='cuda')
C = matmul(A, B)

# Validate
torch.testing.assert_close(C, A @ B, rtol=1e-3, atol=1e-3)
```

**Persistent variant** for better SM utilization:
```python
@ct.kernel
def persistent_matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    bid = ct.bid(0)
    num_tile_blocks = ct.num_blocks(0)

    M, N = A.shape[0], B.shape[1]
    num_tiles = ct.cdiv(M, tm) * ct.cdiv(N, tn)

    # Each block processes multiple tiles
    for tile_id in range(bid, num_tiles, num_tile_blocks):
        tile_m = tile_id // ct.cdiv(N, tn)
        tile_n = tile_id % ct.cdiv(N, tn)

        accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
        num_k_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))

        for k in range(num_k_tiles):
            a = ct.load(A, index=(tile_m, k), shape=(tm, tk),
                        padding_mode=ct.PaddingMode.ZERO)
            b = ct.load(B, index=(k, tile_n), shape=(tk, tn),
                        padding_mode=ct.PaddingMode.ZERO)
            accumulator = ct.mma(a, b, accumulator)

        result = accumulator.astype(C.dtype)
        ct.store(C, index=(tile_m, tile_n), tile=result)
```

### Example 3: Softmax

Demonstrates reduction pattern with numerical stability.

```python
import cuda.tile as ct
import torch
import math
import numpy as np

ConstInt = ct.Constant[int]
INV_LOG_2 = 1.0 / math.log(2)

@ct.kernel
def softmax_kernel(x, output, TILE_M: ConstInt, TILE_N: ConstInt):
    """Row-wise softmax with numerical stability."""
    bid_x = ct.bid(0)

    # Load row tile
    x_tile = ct.load(x, index=(bid_x, 0), shape=(TILE_M, TILE_N),
                     padding_mode=ct.PaddingMode.ZERO)

    # Numerical stability: subtract max
    max_val = ct.max(x_tile, axis=-1, keepdims=True)
    x_shifted = x_tile - max_val

    # Compute exp (use exp2 for performance)
    exp_vals = ct.exp2(x_shifted * INV_LOG_2, flush_to_zero=True)

    # Normalize
    sum_exp = ct.sum(exp_vals, axis=-1, keepdims=True)
    result = ct.truediv(exp_vals, sum_exp, flush_to_zero=True,
                        rounding_mode=ct.RoundingMode.APPROX)

    ct.store(output, index=(bid_x, 0), tile=result)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for row-wise softmax."""
    M, N = x.shape
    TILE_M = 1  # Process one row per block
    TILE_N = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1

    output = torch.empty_like(x)
    grid = (M, 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, softmax_kernel,
              (x, output, TILE_M, TILE_N))
    return output

# Usage
x = torch.randn(256, 512, dtype=torch.float32, device='cuda')
result = softmax(x)

# Validate
reference = torch.softmax(x, dim=-1)
torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)
```

## Common Patterns

### Pattern 1: Element-wise Operations
Use **gather/scatter** for flexible, non-contiguous access with automatic boundary handling.

### Pattern 2: K-loop Tiling
For matrix operations, iterate over the shared dimension (K) in tiles. Always use **float32 accumulators** for precision even with float16 inputs.

### Pattern 3: Reductions
- **Tile-local**: Use `ct.sum/max/min(tile, axis=...)`
- **Global (cross-block)**: Use `ct.atomic_add()` for final aggregation

### Pattern 4: Persistent Kernels
Launch fewer blocks than tiles. Each block processes multiple tiles via a loop:
```python
for tile_id in range(bid, total_tiles, num_tile_blocks):
    # Process tile_id
```
Benefits: Better SM utilization, reduced launch overhead.

### Pattern 5: TMA Optimization
Use `latency` hints (2, 3, 4, 10) in `ct.load()` to help pipeline memory transfers:
```python
tile = ct.load(array, index=(i, j), shape=(M, N), latency=2)
```
Critical for attention kernels and complex memory access patterns.

## Debugging Quick Tips

**Environment variables:**
- `CUDA_TILE_DUMP_BYTECODE=1` - Save bytecode to disk
- `CUDA_TILE_DUMP_TILEIR=1` - Save MLIR module
- `CUDA_TILE_COMPILER_TIMEOUT_SEC=120` - Extend timeout (default 60s)

**Common errors:**
- *"Tile dimensions must be power of 2"* → Use 16, 32, 64, 128, 256, etc.
- *"Shape mismatch in mma"* → Verify (M, K) @ (K, N) → (M, N) alignment
- *"Out of shared memory"* → Reduce tile sizes or use smaller power-of-2 dims

**Validation techniques:**
```python
# FP16 tolerance
torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)

# FP32 tolerance
torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)
```

## Advanced Topics

For more complex patterns and optimization techniques, see the reference files:

**Examples by Category:**
- **[EXAMPLES.md](references/EXAMPLES.md)** - Quick start (3 essential examples: VectorSum, ReLU, MatMul)
- **[EXAMPLES_COMPUTE.md](references/EXAMPLES_COMPUTE.md)** - Compute operations (MatMul, BatchMatMul, FlashAttention)
- **[EXAMPLES_NORMALIZATION.md](references/EXAMPLES_NORMALIZATION.md)** - Normalization (Softmax, LayerNorm, RMSNorm)
- **[EXAMPLES_ELEMENTWISE.md](references/EXAMPLES_ELEMENTWISE.md)** - Element-wise ops (Quantization, RoPE, SiLU&Mul, Img2Patch)
- **[EXAMPLES_TRAINING.md](references/EXAMPLES_TRAINING.md)** - Training ops (AdamW, 4 backward kernel examples)

**Reference Documentation:**
- **[API_REFERENCE.md](references/API_REFERENCE.md)** - Complete API documentation (types, operations, parameters)
- **[PATTERNS.md](references/PATTERNS.md)** - Memory access patterns, reduction strategies, attention patterns, performance patterns, backward pass patterns
- **[OPTIMIZATION.md](references/OPTIMIZATION.md)** - TMA deep dive, persistent kernels, memory layout, tile size selection
- **[TESTING.md](references/TESTING.md)** - Correctness testing with PyTest, performance benchmarking with triton.testing, forward+backward validation
- **[DEBUGGING.md](references/DEBUGGING.md)** - Detailed error diagnosis, debugging workflow, testing strategies
