# CuTile Code Examples - Quick Start

This is the main entry point for CuTile code examples. Start here to understand basic patterns, then explore specialized topics.

## Table of Contents

**This File (Quick Start)**:
- [Example 1: Vector Sum](#example-1-vector-sum-global-reduction) - Global reduction with atomics
- [Example 2: ReLU Activation](#example-2-relu-activation) - Element-wise operations
- [Example 14: Matrix Multiplication](#example-14-matrix-multiplication-optimized) - Optimized matmul with tiling
- [Summary of Key Patterns](#summary-of-key-patterns) - Common patterns and best practices

**Specialized Topics** (see separate files):
- [EXAMPLES_COMPUTE.md](EXAMPLES_COMPUTE.md) - Compute-intensive operations (MatMul, FlashAttention)
- [EXAMPLES_NORMALIZATION.md](EXAMPLES_NORMALIZATION.md) - Normalization operations (Softmax, LayerNorm, RMSNorm)
- [EXAMPLES_ELEMENTWISE.md](EXAMPLES_ELEMENTWISE.md) - Element-wise operations (Quantization, RoPE, SiLU, Img2Patch)
- [EXAMPLES_TRAINING.md](EXAMPLES_TRAINING.md) - Training operations (AdamW, Backward passes)

---

## Example 1: Vector Sum (Global Reduction)

**Purpose**: Compute sum of all elements in a vector using tile-level reduction and atomic operations.

**Key Techniques**: `ct.load`, `ct.sum`, `ct.atomic_add`, reduction patterns

```python
import torch
import cuda.tile as ct

@ct.kernel
def vector_sum_kernel(x: ct.Array, output: ct.Array, TILE: ct.Constant[int]):
    """Sum all elements of 1D array x using atomic reduction."""
    bid = ct.bid(0)

    # Load tile from global memory
    tile = ct.load(x, index=(bid,), shape=(TILE,))

    # Reduce tile to scalar (local reduction)
    tile = tile.astype(ct.float32)
    local_sum = ct.sum(tile)

    # Global reduction using atomic add
    ct.atomic_add(output, (0,), local_sum.astype(output.dtype))

def vector_sum(x: torch.Tensor) -> torch.Tensor:
    """Compute sum of vector x."""
    TILE = 256
    num_blocks = ct.cdiv(x.numel(), TILE)

    output = torch.zeros(1, device=x.device, dtype=x.dtype)

    ct.launch(
        torch.cuda.current_stream(),
        (num_blocks,),
        vector_sum_kernel,
        (x, output, TILE)
    )

    return output

# Usage
x = torch.randn(1024 * 1024, device="cuda", dtype=torch.float16)
result = vector_sum(x)
print(f"CuTile sum: {result.item():.4f}, PyTorch sum: {x.sum().item():.4f}")
```

**Validation**:
```python
torch.testing.assert_close(result, x.sum(), rtol=1e-3, atol=1e-3)
```

---

## Example 2: ReLU Activation

**Purpose**: Apply ReLU activation element-wise: `y = max(0, x)`.

**Key Techniques**: `ct.gather`, `ct.scatter`, `ct.max`, element-wise operations

```python
import torch
import cuda.tile as ct

@ct.kernel
def relu_kernel(x: ct.Array, y: ct.Array, TILE: ct.Constant[int]):
    """Apply ReLU activation: y = max(0, x)."""
    bid = ct.bid(0)

    # Compute indices for this block
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

    # Gather input elements
    tile_x = ct.gather(x, indices)

    # Apply ReLU: max(0, x)
    zero = ct.full(tile_x.shape, 0.0, dtype=tile_x.dtype)
    tile_y = ct.max(tile_x, zero)

    # Scatter result to output
    ct.scatter(y, indices, tile_y)

def relu(x: torch.Tensor) -> torch.Tensor:
    """Apply ReLU activation."""
    TILE = 1024
    num_blocks = ct.cdiv(x.numel(), TILE)

    y = torch.empty_like(x)

    ct.launch(
        torch.cuda.current_stream(),
        (num_blocks,),
        relu_kernel,
        (x.flatten(), y.flatten(), TILE)
    )

    return y.reshape(x.shape)

# Usage
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = relu(x)

# Validation
torch.testing.assert_close(y, torch.relu(x))
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

## Summary of Key Patterns

### Memory Access Patterns
1. **Structured Access**: Use `ct.load/store` with index tuples for tiled access
2. **Indexed Access**: Use `ct.gather/scatter` for irregular or masked access
3. **Padding**: Use `padding_mode=ct.PaddingMode.ZERO` for boundary handling

### Computation Patterns
1. **Element-wise**: Direct tile arithmetic (`+`, `-`, `*`, `/`)
2. **Reductions**: `ct.sum`, `ct.max` with `axis` and `keepdims`
3. **Matrix Ops**: `ct.mma` for Tensor Core acceleration
4. **Atomic Ops**: `ct.atomic_add` for global reductions

### Optimization Techniques
1. **Tile Sizes**: Powers of 2, typically 64-1024
2. **FP32 Accumulators**: Use float32 for accumulation, convert at store
3. **exp2 vs exp**: Use `ct.exp2` with `INV_LOG_2 = 1/log(2)` for speed
4. **Numerical Stability**: Subtract max before exp, use online algorithms
5. **Fusion**: Combine operations to reduce memory traffic

### Grid Calculation
```python
# 1D: blocks along single dimension
grid = (ct.cdiv(N, TILE),)

# 2D: blocks along two dimensions
grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N))

# 3D: batch + 2D tiling
grid = (Batch, ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N))
```

---

## Next Steps

Explore specialized topics:
- **Compute Operations**: See [EXAMPLES_COMPUTE.md](EXAMPLES_COMPUTE.md) for BatchMatMul and FlashAttention
- **Normalization**: See [EXAMPLES_NORMALIZATION.md](EXAMPLES_NORMALIZATION.md) for Softmax, LayerNorm, RMSNorm
- **Element-wise**: See [EXAMPLES_ELEMENTWISE.md](EXAMPLES_ELEMENTWISE.md) for Quantization, RoPE, SiLU
- **Training**: See [EXAMPLES_TRAINING.md](EXAMPLES_TRAINING.md) for AdamW and backward passes
