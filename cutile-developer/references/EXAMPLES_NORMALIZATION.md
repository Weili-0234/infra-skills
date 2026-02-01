# Normalization Operations

This file contains examples of normalization operations commonly used in neural networks.

## Table of Contents
- [Example 3: Softmax](#example-3-softmax)
- [Example 4: Layer Normalization](#example-4-layer-normalization)
- [Example 7: RMSNorm](#example-7-rmsnorm-root-mean-square-normalization)

**See also**: [EXAMPLES.md](EXAMPLES.md) for basic patterns

---

## Example 3: Softmax

**Purpose**: Compute softmax along last dimension: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.

**Key Techniques**: `ct.max`, `ct.exp2`, numerical stability, log2 optimization

```python
import torch
import cuda.tile as ct
import math

INV_LOG_2 = 1.0 / math.log(2)

@ct.kernel
def softmax_kernel(x: ct.Array, y: ct.Array, TILE: ct.Constant[int]):
    """Compute softmax along rows (last dimension)."""
    bid = ct.bid(0)

    # Load entire row
    tile_x = ct.load(x, (bid, 0), (1, TILE))

    # Convert to log2 space for exp2 (faster than exp)
    tile_x = tile_x.astype(ct.float32) * INV_LOG_2

    # Subtract max for numerical stability
    max_val = ct.max(tile_x)
    tile_x = tile_x - max_val

    # Compute exp and sum
    tile_exp = ct.exp2(tile_x, flush_to_zero=True)
    exp_sum = ct.sum(tile_exp)

    # Normalize
    tile_y = ct.truediv(tile_exp, exp_sum, flush_to_zero=True)

    ct.store(y, (bid, 0), tile_y.astype(y.dtype))

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax along specified dimension."""
    if dim != -1 and dim != x.ndim - 1:
        raise ValueError("Only last dimension softmax is supported")

    M, N = x.shape
    y = torch.empty_like(x)

    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        softmax_kernel,
        (x, y, N)
    )

    return y

# Usage
x = torch.randn(128, 256, device="cuda", dtype=torch.float16)
y = softmax(x)

# Validation
torch.testing.assert_close(y, torch.softmax(x, dim=-1), rtol=1e-3, atol=1e-3)
```

---

## Example 4: Layer Normalization

**Purpose**: Normalize activations: `y = (x - mean) / sqrt(var + eps) * weight + bias`.

**Key Techniques**: Mean/variance computation, `ct.rsqrt`, affine transformation

```python
import torch
import cuda.tile as ct

@ct.kernel
def layernorm_kernel(
    x: ct.Array, weight: ct.Array, bias: ct.Array,
    output: ct.Array, eps: float, TILE: ct.Constant[int]
):
    """Layer normalization with affine transformation."""
    bid = ct.bid(0)

    # Load input row
    tile_x = ct.load(x, (bid, 0), (1, TILE)).astype(ct.float32)

    # Compute mean
    mean = ct.sum(tile_x) / TILE
    tile_x = tile_x - mean

    # Compute variance and normalize
    variance = ct.sum(tile_x * tile_x) / TILE
    rsqrt = ct.rsqrt(variance + eps)
    tile_x = tile_x * rsqrt

    # Apply affine transformation
    tile_w = ct.load(weight, (0,), (TILE,)).astype(ct.float32).reshape((1, TILE))
    tile_b = ct.load(bias, (0,), (TILE,)).astype(ct.float32).reshape((1, TILE))
    tile_x = tile_x * tile_w + tile_b

    ct.store(output, (bid, 0), tile_x.astype(output.dtype))

def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    """Apply layer normalization."""
    M, N = x.shape
    output = torch.empty_like(x)

    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        layernorm_kernel,
        (x, weight, bias, output, eps, N)
    )

    return output

# Usage
M, N = 1024, 512
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(N, device="cuda", dtype=torch.bfloat16)
bias = torch.zeros(N, device="cuda", dtype=torch.bfloat16)

y = layernorm(x, weight, bias)

# Validation
y_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias)
torch.testing.assert_close(y, y_torch, rtol=1e-2, atol=1e-2)
```

---

## Example 7: RMSNorm (Root Mean Square Normalization)

**Purpose**: Compute RMSNorm: `y = x / sqrt(mean(x^2) + eps) * weight`.

**Key Techniques**: RMS computation, `ct.rsqrt`, no mean subtraction

```python
import torch
import cuda.tile as ct

@ct.kernel
def rmsnorm_kernel(
    x: ct.Array, weight: ct.Array, output: ct.Array,
    eps: float, TILE: ct.Constant[int]
):
    """RMS normalization."""
    bid = ct.bid(0)

    # Load input and weight
    tile_x = ct.load(x, (bid, 0), (1, TILE)).astype(ct.float32)
    tile_w = ct.load(weight, (0,), (TILE,)).astype(ct.float32).reshape((1, TILE))

    # Compute RMS: sqrt(mean(x^2) + eps)
    square_mean = ct.sum(tile_x * tile_x) / TILE + eps
    rms_inv = ct.rsqrt(square_mean)

    # Normalize and scale
    tile_y = tile_x * rms_inv * tile_w

    ct.store(output, (bid, 0), tile_y.astype(output.dtype))

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm."""
    M, N = x.shape
    output = torch.empty_like(x)

    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        rmsnorm_kernel,
        (x, weight, output, eps, N)
    )

    return output

# Usage
M, N = 2048, 1024
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(N, device="cuda", dtype=torch.bfloat16)

y = rmsnorm(x, weight)

# Validation (manual RMSNorm implementation)
def torch_rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight

y_torch = torch_rmsnorm(x.float(), weight.float()).to(x.dtype)
torch.testing.assert_close(y, y_torch, rtol=1e-2, atol=1e-2)
```

---

**See also**:
- [EXAMPLES.md](EXAMPLES.md) - Quick start guide
- [EXAMPLES_COMPUTE.md](EXAMPLES_COMPUTE.md) - Compute-intensive operations
- [EXAMPLES_ELEMENTWISE.md](EXAMPLES_ELEMENTWISE.md) - Element-wise operations
- [EXAMPLES_TRAINING.md](EXAMPLES_TRAINING.md) - Training operations
