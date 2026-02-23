# Normalization Kernel Examples

Complete Triton implementations for normalization operations used in transformers.
Each example includes the kernel, a Python wrapper, and validation code.

```python
import triton
import triton.language as tl
```

---

## Table of Contents

1. [Layer Normalization Forward](#1-layer-normalization-forward) -- Two-pass mean/variance, weight+bias, save stats
2. [RMS Normalization Forward + Backward](#2-rms-normalization-forward--backward) -- Simpler norm, full backward pass
3. [Fused Add + RMSNorm](#3-fused-add--rmsnorm) -- Residual addition fused with normalization
4. [Tolerance and Testing Notes](#4-tolerance-and-testing-notes) -- Numerical precision, validation strategies

---

## 1. Layer Normalization Forward

**Pattern overview:** See [PATTERNS.md §2.4 — Welford's Algorithm](PATTERNS.md#pattern-4-welfords-algorithm-for-variance) for the variance computation pattern.

One program per row. Two-pass approach: first compute mean, then variance. Internally
casts to fp32 for accumulation regardless of input dtype, then casts back on store.
Saves `mean` and `rstd` (1/sqrt(var+eps)) for the backward pass.

**Key concepts:**
- `tl.program_id(0)` maps each program to one row
- Two-pass reduction: mean first, then variance (stable for large N)
- Masking handles N not a multiple of BLOCK_SIZE

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fwd_kernel(
    X, Y, W, B, Mean, Rstd,
    stride,  # row stride of X and Y
    N,       # number of columns (normalization dimension)
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    # Load row, cast to fp32 for stable accumulation
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    # Pass 1: mean
    mean = tl.sum(x, axis=0) / N
    # Pass 2: variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    # Save statistics for backward
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply affine: y = x_hat * weight + bias
    x_hat = x_centered * rstd
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * w + b
    tl.store(Y + row * stride + cols, y, mask=mask)


def layer_norm_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                       eps: float = 1e-5) -> tuple:
    """Layer normalization forward pass.
    Returns (y, mean, rstd) where y has same shape as x, mean and rstd have shape (M,).
    """
    assert x.ndim == 2, "Input must be 2D (M, N)"
    M, N = x.shape
    assert weight.shape == (N,) and bias.shape == (N,)
    y = torch.empty_like(x)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _layer_norm_fwd_kernel[(M,)](
        x, y, weight, bias, mean, rstd,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, mean, rstd


# --- Validation ---
def test_layer_norm_forward():
    torch.manual_seed(0)
    M, N = 128, 512
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(N, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
    y_triton, mean_triton, rstd_triton = layer_norm_forward(x, weight, bias)
    # Reference: PyTorch LayerNorm
    ln = torch.nn.LayerNorm(N, device=DEVICE, dtype=torch.float16)
    ln.weight.data = weight
    ln.bias.data = bias
    y_ref = ln(x)
    torch.testing.assert_close(y_triton, y_ref, rtol=1e-3, atol=1e-3)
    print("LayerNorm forward: PASSED")
```

---

## 2. RMS Normalization Forward + Backward

RMSNorm is simpler than LayerNorm: no mean subtraction. Used in LLaMA, Gemma, and other
modern architectures. Forward: `y = x / rms * w` where `rms = sqrt(mean(x^2) + eps)`.

**Key concepts:**
- Forward: single reduction for mean-of-squares, no centering needed
- Backward dx: uses saved rstd to recompute the correction term
- Backward dw: sums across all rows (two-pass approach avoids atomics)

### Forward Kernel

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _rms_norm_fwd_kernel(
    X, Y, W, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    # RMS = sqrt(mean(x^2) + eps)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    tl.store(Rstd + row, rstd)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + row * stride + cols, y, mask=mask)
```

### Backward Kernels

Gradient formulas:
- `dx = (dy * w - x * rstd^2 * mean(dy * w * x)) * rstd`
- `dw = sum_over_rows(dy * x * rstd)`

The dx kernel processes one row per program. The dw kernel uses a two-pass approach:
each program accumulates a partial sum over a chunk of rows, then a host-side reduction
combines them.

```python
@triton.jit
def _rms_norm_bwd_dx_kernel(
    DY, X, W, Rstd, DX,
    stride, N,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute dx for one row."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + row)
    dy_w = dy * w
    # Correction term: mean(dy * w * x)
    inner = tl.sum(dy_w * x, axis=0) / N
    # dx = (dy * w - x * rstd^2 * mean(dy * w * x)) * rstd
    dx = (dy_w - x * rstd * rstd * inner) * rstd
    tl.store(DX + row * stride + cols, dx, mask=mask)


@triton.jit
def _rms_norm_bwd_dw_kernel(
    DY, X, Rstd,
    DW_partial,  # (num_groups, N) buffer for partial sums
    stride, N, M,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute partial dw sums over a chunk of rows."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    row_start = pid * ROWS_PER_PROG
    dw_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(ROWS_PER_PROG):
        row = row_start + i
        if row < M:
            dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd + row)
            dw_acc += dy * x * rstd
    tl.store(DW_partial + pid * N + cols, dw_acc, mask=mask)


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor,
                     eps: float = 1e-6) -> tuple:
    """RMSNorm forward: y = x / rms * weight."""
    M, N = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rms_norm_fwd_kernel[(M,)](
        x, y, weight, rstd,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, rstd


def rms_norm_backward(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                      rstd: torch.Tensor) -> tuple:
    """RMSNorm backward: compute dx and dw."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    # dx: one program per row
    dx = torch.empty_like(x)
    _rms_norm_bwd_dx_kernel[(M,)](
        dy, x, weight, rstd, dx,
        x.stride(0), N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # dw: partition rows into groups, accumulate partial sums
    ROWS_PER_PROG = 32
    num_groups = triton.cdiv(M, ROWS_PER_PROG)
    dw_partial = torch.empty(num_groups, N, dtype=torch.float32, device=x.device)
    _rms_norm_bwd_dw_kernel[(num_groups,)](
        dy, x, rstd, dw_partial,
        x.stride(0), N, M,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    dw = dw_partial.sum(dim=0).to(weight.dtype)
    return dx, dw


# --- Validation ---
def test_rms_norm():
    torch.manual_seed(0)
    M, N = 256, 768
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, device=DEVICE, dtype=torch.float32, requires_grad=True)
    eps = 1e-6
    # Forward
    y_triton, rstd = rms_norm_forward(x, weight, eps)
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = x / rms * weight
    torch.testing.assert_close(y_triton, y_ref, rtol=1e-5, atol=1e-5)
    print("RMSNorm forward: PASSED")
    # Backward
    dy = torch.randn_like(y_triton)
    y_ref.backward(dy)
    dx_triton, dw_triton = rms_norm_backward(dy, x, weight, rstd)
    torch.testing.assert_close(dx_triton, x.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw_triton, weight.grad, rtol=1e-4, atol=1e-4)
    print("RMSNorm backward: PASSED")
```

---

## 3. Fused Add + RMSNorm

Common transformer pattern: after each attention or MLP block, the residual connection
is added and then normalized. Fusing `residual = x + residual` with
`y = rmsnorm(residual)` into one kernel eliminates an extra memory round-trip and a
separate kernel launch.

**Key concepts:**
- Load both `x` (current output) and `residual` (running residual stream)
- Add in registers, write updated residual back (needed by next layer)
- Normalize the summed value and store final output `y`
- One kernel replaces two: elementwise add + RMSNorm

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _fused_add_rms_norm_kernel(
    X,         # current layer output, shape (M, N)
    Residual,  # running residual stream, shape (M, N) -- updated in-place
    Y,         # normalized output, shape (M, N)
    W,         # RMSNorm weight, shape (N,)
    Rstd,      # output: 1/rms per row, shape (M,)
    stride,    # row stride for X, Residual, Y
    N,         # normalization dimension
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Residual + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    # Fused add
    x = x + res
    # Store updated residual back
    tl.store(Residual + row * stride + cols, x, mask=mask)
    # RMSNorm
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    tl.store(Rstd + row, rstd)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + row * stride + cols, y.to(tl.float16), mask=mask)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, eps: float = 1e-6) -> tuple:
    """Fused residual add + RMSNorm.

    Args:
        x: Current layer output, shape (M, N), fp16
        residual: Running residual stream, shape (M, N), fp16. Modified in-place.
        weight: RMSNorm weight, shape (N,)
        eps: Epsilon for numerical stability
    Returns:
        (y, rstd) where y is normalized output (fp16) and rstd has shape (M,)
    """
    assert x.shape == residual.shape
    M, N = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _fused_add_rms_norm_kernel[(M,)](
        x, residual, y, weight, rstd,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, rstd


# --- Validation ---
def test_fused_add_rms_norm():
    torch.manual_seed(0)
    M, N = 64, 1024
    eps = 1e-6
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    residual = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(N, device=DEVICE, dtype=torch.float16)
    residual_ref = residual.clone()
    residual_triton = residual.clone()
    # Triton
    y_triton, rstd_triton = fused_add_rms_norm(x, residual_triton, weight, eps)
    # Reference: separate add + RMSNorm in fp32
    added = x.float() + residual_ref.float()
    rms = torch.sqrt(added.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = (added / rms * weight.float()).half()
    # Check updated residual
    torch.testing.assert_close(residual_triton, (x + residual_ref).half(),
                               rtol=1e-3, atol=1e-3)
    # Check normalized output
    torch.testing.assert_close(y_triton, y_ref, rtol=1e-3, atol=1e-3)
    print("Fused Add + RMSNorm: PASSED")
```

---

## 4. Tolerance and Testing Notes

Normalization kernels are sensitive to numerical precision because they involve
reductions, divisions, and square roots.

### Recommended Tolerances by Dtype

| Input Dtype | Internal Compute | `rtol` | `atol` | Notes |
|---|---|---|---|---|
| `float16` | `float32` | `1e-3` | `1e-3` | Cast to fp32 inside kernel, cast back on store |
| `bfloat16` | `float32` | `1e-2` | `1e-2` | BF16 has only 8 mantissa bits vs fp16's 11 |
| `float32` | `float32` | `1e-5` | `1e-5` | Native precision, tightest tolerances |

### Always Cast to FP32 Internally

Every normalization kernel should cast inputs to `tl.float32` before arithmetic.
Accumulating sums in fp16 loses precision for large N (4096+), causing incorrect
means and variances.

```python
# Good: cast on load
x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
mean = tl.sum(x, axis=0) / N  # accurate sum in fp32

# Bad: accumulate in fp16
x = tl.load(X + offsets, mask=mask, other=0.0)  # stays in fp16
mean = tl.sum(x, axis=0) / N  # large rounding errors for big N
```

### Backward Pass Validation

Use `torch.autograd.gradcheck` with double precision for rigorous gradient verification:

```python
def test_backward_gradcheck():
    """Rigorous gradient check using finite differences in float64."""
    M, N = 4, 16  # small sizes for gradcheck (it is slow)
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float64, requires_grad=True)
    w = torch.randn(N, device=DEVICE, dtype=torch.float64, requires_grad=True)
    def func(x, w):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x / rms * w
    torch.autograd.gradcheck(func, (x, w), eps=1e-6, atol=1e-4)
    print("gradcheck: PASSED")
```

For practical testing, compare Triton backward output against PyTorch autograd:

```python
def test_backward_against_pytorch():
    """Compare Triton backward output against PyTorch autograd."""
    torch.manual_seed(42)
    M, N = 128, 512
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32, requires_grad=True)
    w = torch.randn(N, device=DEVICE, dtype=torch.float32, requires_grad=True)
    # Triton forward + backward
    y_triton, rstd = rms_norm_forward(x.detach(), w.detach())
    dy = torch.randn_like(y_triton)
    dx_triton, dw_triton = rms_norm_backward(dy, x.detach(), w.detach(), rstd)
    # PyTorch reference
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    y_ref = x / rms * w
    y_ref.backward(dy)
    torch.testing.assert_close(dx_triton, x.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw_triton, w.grad, rtol=1e-4, atol=1e-4)
    print("Backward against PyTorch: PASSED")
```

### Common Pitfalls

**Weight gradient accumulation:** `dw = sum_over_rows(dy * x_hat)` requires summing
across all M rows. Three approaches:
1. **Two-pass (recommended):** Partition rows into groups, each program accumulates a
   partial sum, then reduce on host with `dw_partial.sum(dim=0)`.
2. **Atomic adds:** Use `tl.atomic_add` to accumulate directly, but this serializes
   writes and is slower for large M. Only viable when M is small.
3. **Separate reduction kernel:** Write per-row contributions to an (M, N) buffer,
   then launch a second kernel or use PyTorch's `.sum(dim=0)`.

**BLOCK_SIZE selection:** Use `triton.next_power_of_2(N)` to ensure BLOCK_SIZE >= N.
The entire row must fit in a single tile since `tl.sum` operates within one program's
data. For very large N (8192+), consider a loop inside the kernel to accumulate partial
statistics across multiple loads.

**Epsilon placement:** Always add epsilon inside the square root: `1/sqrt(var + eps)`,
not `1/(sqrt(var) + eps)`. The former prevents negative values under the sqrt when
variance is exactly zero.

**Store dtype:** When output is fp16 but computation is fp32, explicitly cast before
storing: `tl.store(ptr, result.to(tl.float16), mask=mask)`. Being explicit avoids
surprises and documents intent.
