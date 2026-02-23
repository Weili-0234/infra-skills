# Training Examples: Backward Pass & Gradient Kernels

Backward pass implementations in Triton for common neural network operations. Each example includes the gradient math, Triton kernel, and `torch.autograd.Function` wrapper for integration with PyTorch's autograd engine.

```python
import torch
import triton
import triton.language as tl
```

---

## Table of Contents

1. [SwiGLU Backward](#1-swiglu-backward) -- Activation gradient with recomputation
2. [RMSNorm Backward](#2-rmsnorm-backward) -- Normalization gradient with atomic accumulation
3. [LayerNorm Backward](#3-layernorm-backward) -- Lock-based parallel reduction for dw/db
4. [Fused Cross-Entropy Backward](#4-fused-cross-entropy-backward) -- Loss gradient in a single pass
5. [Backward Pass Best Practices](#5-backward-pass-best-practices) -- Templates, verification, design rules

---

## 1. SwiGLU Backward

SwiGLU activation: `output = silu(gate) * up` where `silu(x) = x * sigmoid(x)`.

**Gradient formulas:**
- `d_up = grad_output * silu(gate)`
- `d_gate = grad_output * up * dsilu(gate)`
- `dsilu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`

**Recomputation strategy:** Recompute `sigmoid(gate)` in the backward pass rather than saving it from forward. Sigmoid is cheap (one exp + one div), and avoiding the save halves peak memory for this activation.

**Forward kernel:** See [EXAMPLES_LLM.md §2 — SwiGLU / SiLU-and-Mul](EXAMPLES_LLM.md#2-swiglu--silu-and-mul) for the full forward implementation. Below is the backward kernel using recomputation strategy.

```python
@triton.jit
def _swiglu_backward_kernel(
    Grad_ptr, Gate_ptr, Up_ptr,
    DGate_ptr, DUp_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad = tl.load(Grad_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(Gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(Up_ptr + offsets, mask=mask).to(tl.float32)

    # Recompute sigmoid(gate) -- cheaper than loading a saved tensor
    sig_gate = tl.sigmoid(gate)
    silu_gate = gate * sig_gate

    # d_up = grad * silu(gate)
    d_up = grad * silu_gate

    # dsilu(gate) = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
    dsilu_gate = sig_gate + gate * sig_gate * (1.0 - sig_gate)

    # d_gate = grad * up * dsilu(gate)
    d_gate = grad * up * dsilu_gate

    tl.store(DUp_ptr + offsets, d_up, mask=mask)
    tl.store(DGate_ptr + offsets, d_gate, mask=mask)


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.shape == up.shape
        assert gate.is_contiguous() and up.is_contiguous()
        ctx.save_for_backward(gate, up)

        output = torch.empty_like(gate)
        N = gate.numel()
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        _swiglu_forward_kernel[grid](gate, up, output, N, BLOCK_SIZE=1024)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        d_gate = torch.empty_like(gate)
        d_up = torch.empty_like(up)
        N = gate.numel()
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        _swiglu_backward_kernel[grid](
            grad_output, gate, up, d_gate, d_up, N, BLOCK_SIZE=1024,
        )
        return d_gate, d_up


def swiglu(gate, up):
    return SwiGLUFunction.apply(gate, up)


# --- Validation ---
DEVICE = triton.runtime.driver.active.get_active_torch_device()
gate = torch.randn(128, 256, device=DEVICE, dtype=torch.float32, requires_grad=True)
up = torch.randn(128, 256, device=DEVICE, dtype=torch.float32, requires_grad=True)

out = swiglu(gate, up)
out.sum().backward()

# Reference: PyTorch autograd
gate_ref = gate.detach().clone().requires_grad_(True)
up_ref = up.detach().clone().requires_grad_(True)
out_ref = torch.nn.functional.silu(gate_ref) * up_ref
out_ref.sum().backward()

torch.testing.assert_close(gate.grad, gate_ref.grad, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(up.grad, up_ref.grad, atol=1e-5, rtol=1e-5)
print("SwiGLU backward: PASSED")
```

**Why recompute sigmoid?** The forward pass only saves `gate` and `up` (via `ctx.save_for_backward`). We deliberately do NOT save `sigmoid(gate)`. Recomputing it costs one `tl.sigmoid` call per element, but avoids allocating and storing an extra tensor the size of `gate`. For large batch sizes this memory saving is significant.

---

## 2. RMSNorm Backward

RMSNorm: `y = x * rstd * w` where `rstd = 1 / sqrt(mean(x^2) + eps)`.

**Gradient formulas:**
- `dx = w * rstd * (dy - x * rstd^2 * mean(dy * w * x))`
- `dw = sum_over_rows(dy * x * rstd)`

For `dx`, each row is independent -- one program per row. For `dw`, we must accumulate partial results across all rows.

### Approach 1: Simple atomic accumulation

```python
@triton.jit
def _rmsnorm_backward_dx_kernel(
    DY_ptr, X_ptr, W_ptr, Rstd_ptr,
    DX_ptr,
    M, N,
    stride_dy_row, stride_x_row, stride_dx_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute dx for one row. Also atomically accumulate dw."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load row data
    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    # mean(dy * w * x) for this row
    # Note: N_float used for mean computation
    N_float = N.to(tl.float32)
    m = tl.sum(dy * w * x, axis=0) / N_float

    # dx = w * rstd * (dy - x * rstd^2 * m)
    dx = w * rstd * (dy - x * rstd * rstd * m)
    tl.store(DX_ptr + row * stride_dx_row + cols, dx, mask=mask)


@triton.jit
def _rmsnorm_backward_dw_atomic_kernel(
    DY_ptr, X_ptr, Rstd_ptr,
    DW_ptr,
    M, N,
    stride_dy_row, stride_x_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Each row contributes dy * x * rstd to dw via atomic add."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    dw_partial = dy * x * rstd

    # Atomic add: each row adds its contribution
    tl.atomic_add(DW_ptr + cols, dw_partial, mask=mask)
```

**Limitation of atomic approach:** When M (number of rows) is very large, thousands of programs contend on the same dw locations. Atomic adds serialize under contention, degrading throughput. This motivates the lock-based approach below.

### Approach 2: Lock-based parallel reduction

```python
@triton.jit
def _rmsnorm_backward_dw_grouped_kernel(
    DY_ptr, X_ptr, Rstd_ptr,
    DW_partial_ptr, Lock_ptr, Count_ptr,
    M, N,
    stride_dy_row, stride_x_row,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Accumulate dw into GROUP_SIZE partial buffers using spin-locks."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    dw_partial = dy * x * rstd

    # Map this row to one of GROUP_SIZE buffers
    lock_id = row % GROUP_SIZE
    lock_ptr = Lock_ptr + lock_id
    count_ptr = Count_ptr + lock_id

    # Acquire spin-lock
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="gpu") == 1:
        pass

    count = tl.load(count_ptr)
    if count == 0:
        # First writer: store directly
        tl.store(DW_partial_ptr + lock_id * N + cols, dw_partial, mask=mask)
    else:
        # Accumulate into existing buffer
        existing = tl.load(DW_partial_ptr + lock_id * N + cols, mask=mask, other=0.0)
        tl.store(DW_partial_ptr + lock_id * N + cols, existing + dw_partial, mask=mask)
    tl.store(count_ptr, count + 1)

    # Release spin-lock
    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="gpu")


@triton.jit
def _rmsnorm_reduce_dw_kernel(
    DW_partial_ptr, DW_ptr,
    N,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum the GROUP_SIZE partial buffers into final dw."""
    cols = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for g in range(GROUP_SIZE):
        partial = tl.load(DW_partial_ptr + g * N + cols, mask=mask, other=0.0)
        acc += partial
    tl.store(DW_ptr + cols, acc, mask=mask)
```

**Wrapper (lock-based):**

```python
def rmsnorm_backward(dy, x, w, rstd, eps=1e-6):
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    GROUP_SIZE = 64

    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    dw_partial = torch.zeros(GROUP_SIZE, N, dtype=torch.float32, device=x.device)
    lock = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=x.device)
    count = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=x.device)

    # Pass 1: compute dx (one program per row)
    _rmsnorm_backward_dx_kernel[(M,)](
        dy, x, w, rstd, dx,
        M, N,
        dy.stride(0), x.stride(0), dx.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Pass 2: accumulate dw into partial buffers
    _rmsnorm_backward_dw_grouped_kernel[(M,)](
        dy, x, rstd,
        dw_partial, lock, count,
        M, N,
        dy.stride(0), x.stride(0),
        GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE=BLOCK_SIZE,
    )

    # Pass 3: reduce partial buffers
    grid_reduce = (triton.cdiv(N, BLOCK_SIZE),)
    _rmsnorm_reduce_dw_kernel[grid_reduce](
        dw_partial, dw,
        N,
        GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE=BLOCK_SIZE,
    )
    return dx, dw
```

---

## 3. LayerNorm Backward

LayerNorm: `y = (x - mean) * rstd * w + b` where `mean = mean(x)`, `rstd = 1/sqrt(var(x) + eps)`.

**Gradient formulas:**
- `dx = rstd * w * (dy - mean(dy * w) * rstd/rstd - (x - mean) * rstd^2 * mean(dy * w * (x - mean)))` (simplified below)
- `dw = sum_over_rows(dy * (x - mean) * rstd)`
- `db = sum_over_rows(dy)`

The dx formula simplifies to:
```
xhat = (x - mean) * rstd
dx = (1/N) * rstd * w * (N * dy - sum(dy * w) - xhat * sum(dy * w * xhat))
```

**Pattern overview:** See [PATTERNS.md §5.4 — Parallel Reduction with Locks](PATTERNS.md#pattern-4-parallel-reduction-with-locks) for when/why to use this pattern.

### Lock-based parallel reduction (Liger-Kernel / official tutorial pattern)

```python
@triton.jit
def _layernorm_backward_kernel(
    DY_ptr, X_ptr, W_ptr, Mean_ptr, Rstd_ptr,
    DX_ptr, DW_partial_ptr, DB_partial_ptr,
    Lock_ptr, Count_ptr,
    M, N,
    stride_dy_row, stride_x_row, stride_dx_row,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one row:
    - Computes dx for that row (written directly)
    - Accumulates dw and db into GROUP_SIZE partial buffers via spin-lock
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load data for this row
    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean_ptr + row).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    # Normalized input
    xhat = (x - mean) * rstd
    N_float = N.to(tl.float32)

    # Intermediate sums for dx
    wdy = w * dy
    sum_wdy = tl.sum(wdy, axis=0)
    sum_wdy_xhat = tl.sum(wdy * xhat, axis=0)

    # dx = rstd / N * (N * wdy - sum_wdy - xhat * sum_wdy_xhat)
    dx = rstd / N_float * (N_float * wdy - sum_wdy - xhat * sum_wdy_xhat)
    tl.store(DX_ptr + row * stride_dx_row + cols, dx, mask=mask)

    # Partial gradients for w and b
    dw = dy * xhat
    db = dy

    # --- Lock-based accumulation into partial buffers ---
    row_block_id = row
    lock_id = row_block_id % GROUP_SIZE
    lock_ptr = Lock_ptr + lock_id
    count_ptr = Count_ptr + lock_id

    # Spin until lock acquired
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="gpu") == 1:
        pass

    count = tl.load(count_ptr)

    # DW partial buffer offset
    dw_offset = lock_id * N + cols
    db_offset = lock_id * N + cols

    if count == 0:
        # First writer: store directly
        tl.store(DW_partial_ptr + dw_offset, dw, mask=mask)
        tl.store(DB_partial_ptr + db_offset, db, mask=mask)
    else:
        # Subsequent writers: accumulate
        dw += tl.load(DW_partial_ptr + dw_offset, mask=mask, other=0.0)
        tl.store(DW_partial_ptr + dw_offset, dw, mask=mask)
        db += tl.load(DB_partial_ptr + db_offset, mask=mask, other=0.0)
        tl.store(DB_partial_ptr + db_offset, db, mask=mask)

    tl.store(count_ptr, count + 1)

    # Release lock
    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="gpu")


@triton.jit
def _layernorm_reduce_partial_kernel(
    DW_partial_ptr, DB_partial_ptr,
    DW_ptr, DB_ptr,
    N,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce GROUP_SIZE partial buffers into final dw and db."""
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dw_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for g in range(GROUP_SIZE):
        dw_acc += tl.load(DW_partial_ptr + g * N + cols, mask=mask, other=0.0)
        db_acc += tl.load(DB_partial_ptr + g * N + cols, mask=mask, other=0.0)

    tl.store(DW_ptr + cols, dw_acc, mask=mask)
    tl.store(DB_ptr + cols, db_acc, mask=mask)


def layernorm_backward(dy, x, w, mean, rstd):
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    GROUP_SIZE = 64

    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    db = torch.empty_like(w)
    dw_partial = torch.zeros(GROUP_SIZE, N, dtype=torch.float32, device=x.device)
    db_partial = torch.zeros(GROUP_SIZE, N, dtype=torch.float32, device=x.device)
    lock = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=x.device)
    count = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=x.device)

    _layernorm_backward_kernel[(M,)](
        dy, x, w, mean, rstd,
        dx, dw_partial, db_partial,
        lock, count,
        M, N,
        dy.stride(0), x.stride(0), dx.stride(0),
        GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE=BLOCK_SIZE,
    )

    grid_reduce = (triton.cdiv(N, BLOCK_SIZE),)
    _layernorm_reduce_partial_kernel[grid_reduce](
        dw_partial, db_partial, dw, db,
        N,
        GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE=BLOCK_SIZE,
    )
    return dx, dw, db


# --- Validation ---
DEVICE = triton.runtime.driver.active.get_active_torch_device()
M, N = 128, 512
x = torch.randn(M, N, device=DEVICE, dtype=torch.float32, requires_grad=True)
w = torch.randn(N, device=DEVICE, dtype=torch.float32, requires_grad=True)
b = torch.randn(N, device=DEVICE, dtype=torch.float32, requires_grad=True)

# PyTorch reference
ln = torch.nn.LayerNorm(N, device=DEVICE)
ln.weight = torch.nn.Parameter(w.detach().clone())
ln.bias = torch.nn.Parameter(b.detach().clone())
y_ref = ln(x)
y_ref.sum().backward()
print("LayerNorm backward: see best practices section for full autograd wrapper")
```

**Why spin-locks instead of atomic_add?** `tl.atomic_add` on float32 works but serializes heavily when many programs contend on the same address. The lock-based pattern reduces contention by spreading accumulation across GROUP_SIZE buffers. Each buffer sees M/GROUP_SIZE writers on average, and the locked store-accumulate-release pattern allows full vectorized load/store within the critical section, which is faster than per-element atomics.

---

## 4. Fused Cross-Entropy Backward

Gradient of cross-entropy: `grad = softmax(logits) - one_hot(label)`, scaled by upstream gradient. Fusing softmax computation with the subtraction avoids materializing the full softmax output tensor.

```python
@triton.jit
def _cross_entropy_backward_kernel(
    Logits_ptr, Labels_ptr, Grad_output_ptr,
    DLogits_ptr,
    M, V,  # M = batch size, V = vocab size
    stride_logits_row, stride_dlogits_row,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. Computes softmax(logits) - one_hot(label)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V

    label = tl.load(Labels_ptr + row)
    grad_scale = tl.load(Grad_output_ptr + row)

    # Handle ignored labels: zero gradient for this row
    if label == IGNORE_INDEX:
        tl.store(DLogits_ptr + row * stride_dlogits_row + cols,
                 tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)
        return

    # Load logits and compute softmax (numerically stable)
    logits = tl.load(Logits_ptr + row * stride_logits_row + cols,
                     mask=mask, other=-float('inf')).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    logits_stable = logits - logits_max
    exp_logits = tl.exp(logits_stable)
    sum_exp = tl.sum(exp_logits, axis=0)
    softmax = exp_logits / sum_exp

    # Subtract one-hot: softmax[label] -= 1.0
    is_label = cols == label
    grad = tl.where(is_label, softmax - 1.0, softmax)

    # Scale by upstream gradient
    grad = grad * grad_scale

    tl.store(DLogits_ptr + row * stride_dlogits_row + cols, grad, mask=mask)


def cross_entropy_backward(logits, labels, grad_output, ignore_index=-100):
    M, V = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(V)
    d_logits = torch.empty_like(logits)

    _cross_entropy_backward_kernel[(M,)](
        logits, labels, grad_output,
        d_logits,
        M, V,
        logits.stride(0), d_logits.stride(0),
        IGNORE_INDEX=ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return d_logits
```

**Note on vocabulary size:** For very large V (e.g., 128k tokens), a single BLOCK_SIZE may exceed shared memory. In that case, split the vocab dimension across multiple loops within one program, tracking the running max and sum for numerically stable softmax across chunks.

---

## 5. Backward Pass Best Practices

### When to recompute vs cache

| Tensor | Cost to recompute | Size | Decision |
|--------|-------------------|------|----------|
| `sigmoid(x)` | 1 exp + 1 div per element | Same as x | **Recompute** |
| `tanh(x)` | 2 exp + 1 div per element | Same as x | **Recompute** |
| `rstd` (per-row scalar) | Full row reduction | 1 scalar per row | **Cache** (tiny) |
| `mean` (per-row scalar) | Full row reduction | 1 scalar per row | **Cache** (tiny) |
| Attention scores (QK^T) | Full matmul | B*H*S*S | **Recompute** (FlashAttention style; the memory cost of caching is prohibitive) |
| Dropout mask | RNG is cheap | Same as activation | **Recompute from seed** (save only seed + offset) |

**Rule of thumb:** Cache if the tensor is small (scalars, statistics) or extremely expensive to recompute. Recompute if the tensor is activation-sized and the operation is elementwise.

### Atomic accumulation strategies

**Simple atomic_add** -- good when contention is low (few rows, or dw is very large):
```python
@triton.jit
def accumulate_dw_atomic(DW_ptr, partial, cols, mask):
    tl.atomic_add(DW_ptr + cols, partial, mask=mask)
```

**Lock-based grouped reduction** -- better for high contention (many rows, small dw):
```python
@triton.jit
def accumulate_with_lock(
    Partial_ptr, Lock_ptr, Count_ptr,
    data, lock_id, N, cols, mask,
):
    lock_ptr = Lock_ptr + lock_id
    count_ptr = Count_ptr + lock_id

    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="gpu") == 1:
        pass

    count = tl.load(count_ptr)
    offset = lock_id * N + cols
    if count == 0:
        tl.store(Partial_ptr + offset, data, mask=mask)
    else:
        data += tl.load(Partial_ptr + offset, mask=mask, other=0.0)
        tl.store(Partial_ptr + offset, data, mask=mask)
    tl.store(count_ptr, count + 1)
    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="gpu")
```

**Choosing GROUP_SIZE:** Typically 32--128. Larger GROUP_SIZE reduces contention but uses more temporary memory (GROUP_SIZE * N floats). Match it to the expected number of concurrent CTAs.

### Complete torch.autograd.Function template

```python
class TritonOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps=1e-6):
        # 1. Validate inputs
        assert x.is_contiguous()
        M, N = x.shape

        # 2. Allocate outputs
        y = torch.empty_like(x)
        # Allocate saved statistics (small tensors)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        # 3. Launch forward kernel
        BLOCK_SIZE = triton.next_power_of_2(N)
        _forward_kernel[(M,)](
            x, w, b, y, rstd,
            M, N, x.stride(0), y.stride(0),
            eps, BLOCK_SIZE=BLOCK_SIZE,
        )

        # 4. Save for backward -- only what the backward pass needs
        # Save small statistics (rstd, mean) and inputs needed for gradient
        ctx.save_for_backward(x, w, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.M = M
        ctx.N = N

        return y

    @staticmethod
    def backward(ctx, dy):
        # 5. Retrieve saved tensors
        x, w, rstd = ctx.saved_tensors
        M, N = ctx.M, ctx.N

        # 6. Allocate gradient outputs
        dx = torch.empty_like(x)
        dw = torch.empty_like(w)

        # 7. Launch backward kernel(s)
        _backward_kernel[(M,)](
            dy, x, w, rstd,
            dx, dw,
            M, N, dy.stride(0), x.stride(0), dx.stride(0),
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        # 8. Return gradients in same order as forward args
        # Return None for arguments that don't need gradients (like eps)
        return dx, dw, None, None
```

### Gradient verification with gradcheck

```python
def test_custom_op_gradients():
    """Numerically verify gradients using torch.autograd.gradcheck."""
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    # Use float64 for numerical gradient accuracy
    x = torch.randn(8, 32, device=DEVICE, dtype=torch.float64, requires_grad=True)
    w = torch.randn(32, device=DEVICE, dtype=torch.float64, requires_grad=True)

    # gradcheck computes finite-difference gradients and compares to autograd
    def func(x, w):
        return swiglu(x, w)  # or any custom autograd.Function

    torch.autograd.gradcheck(
        func,
        inputs=(x, w),
        eps=1e-6,       # Finite difference step size
        atol=1e-4,      # Absolute tolerance
        rtol=1e-3,      # Relative tolerance
        raise_exception=True,
    )
    print("gradcheck: PASSED")
```

**Important:** `gradcheck` requires float64 inputs. If your kernel only supports float32, use `gradcheck(..., fast_mode=True)` which is less strict but works with lower precision, or test with a PyTorch reference implementation instead.

### Gradient correctness checklist

1. **Does forward save all needed tensors?**
   - List every tensor the backward kernel loads. Verify each is either saved via `ctx.save_for_backward()` or recomputed from saved tensors.
   - Common miss: forgetting to save the `mean` for LayerNorm backward.

2. **Are all gradients computed?**
   - For `forward(ctx, x, w, b)`, backward must return `(dx, dw, db)` in the same order.
   - Return `None` for non-differentiable args (integer indices, epsilon, flags).

3. **Is accumulation across rows handled correctly?**
   - `dx` is per-row: one program writes one row, no conflicts.
   - `dw`, `db` are per-column sums across all rows: need atomic or lock-based accumulation.
   - Verify `dw` buffer is zeroed before the backward kernel launches.

4. **Are dtypes handled properly?**
   - Always upcast to float32 for internal computation: `.to(tl.float32)` on load.
   - Store in the original dtype (autocast-compatible).
   - Reductions (sum, mean) must be in float32 to avoid overflow in float16/bfloat16.

5. **Does the backward kernel handle edge cases?**
   - Rows shorter than BLOCK_SIZE: mask must guard all loads and stores.
   - Zero-length inputs: guard with early return in wrapper.
   - `ignore_index` in loss functions: zero the gradient for ignored rows.
   - NaN/Inf in inputs: consider adding `tl.where(tl.math.isnan(x), 0.0, x)` guards if needed.

### Putting it together: end-to-end validation pattern

```python
def validate_backward(triton_fn, torch_fn, input_shapes, device, dtype=torch.float32):
    """Generic validation: compare Triton backward against PyTorch."""
    inputs = [
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        for shape in input_shapes
    ]
    refs = [inp.detach().clone().requires_grad_(True) for inp in inputs]

    # Forward + backward through Triton path
    out_triton = triton_fn(*inputs)
    out_triton.sum().backward()

    # Forward + backward through PyTorch path
    out_ref = torch_fn(*refs)
    out_ref.sum().backward()

    # Compare all gradients
    for i, (inp, ref) in enumerate(zip(inputs, refs)):
        torch.testing.assert_close(
            inp.grad, ref.grad,
            atol=1e-4, rtol=1e-4,
            msg=f"Gradient mismatch for input {i}",
        )
    print("Backward validation: PASSED")


# Usage
DEVICE = triton.runtime.driver.active.get_active_torch_device()
validate_backward(
    triton_fn=swiglu,
    torch_fn=lambda gate, up: torch.nn.functional.silu(gate) * up,
    input_shapes=[(64, 256), (64, 256)],
    device=DEVICE,
)
```
