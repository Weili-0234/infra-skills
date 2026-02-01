# Training Operations

This file contains examples of training-related operations including optimizers and backward pass implementations.

## Table of Contents
- [Example 13: Fused AdamW Optimizer](#example-13-fused-adamw-optimizer)
- [Example 15: ReLU Backward](#example-15-relu-backward-beginner)
- [Example 16: SwiGLU Backward](#example-16-swiglu-backward-intermediate)
- [Example 17: RMSNorm Backward](#example-17-rmsnorm-backward-advanced)
- [Example 18: SiLU & Mul Backward](#example-18-silu--mul-with-input-splitting-intermediate)

**See also**: [EXAMPLES.md](EXAMPLES.md) for basic patterns

---

## Example 13: Fused AdamW Optimizer

**Purpose**: Fused AdamW optimizer step combining momentum, variance, and weight update.

**Key Techniques**: Optimizer fusion, `ct.rsqrt`, in-place updates, decoupled weight decay

```python
import torch
import cuda.tile as ct

@ct.kernel
def fused_adamw_kernel(
    param: ct.Array, grad: ct.Array, exp_avg: ct.Array, exp_avg_sq: ct.Array,
    lr: float, beta1: float, beta2: float, eps: float, weight_decay: float,
    TILE: ct.Constant[int]
):
    """Fused AdamW optimizer step."""
    bid = ct.bid(0)

    # Load current states
    tile_g = ct.load(grad, (bid,), (TILE,)).astype(ct.float32)
    tile_m = ct.load(exp_avg, (bid,), (TILE,)).astype(ct.float32)
    tile_v = ct.load(exp_avg_sq, (bid,), (TILE,)).astype(ct.float32)

    # Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    tile_m = beta1 * tile_m + (1.0 - beta1) * tile_g

    # Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    tile_v = beta2 * tile_v + (1.0 - beta2) * (tile_g * tile_g)

    # Store updated moments
    ct.store(exp_avg, (bid,), tile_m.astype(exp_avg.dtype))
    ct.store(exp_avg_sq, (bid,), tile_v.astype(exp_avg_sq.dtype))

    # Compute bias-corrected moments
    # Note: In practice, bias correction factors should be passed as arguments
    # For simplicity, we assume they're incorporated into lr

    # Compute update: m_t / (sqrt(v_t) + eps)
    update = tile_m * ct.rsqrt(tile_v + eps)

    # Load parameter and apply update with weight decay
    tile_p = ct.load(param, (bid,), (TILE,)).astype(ct.float32)
    tile_p = tile_p - lr * (update + weight_decay * tile_p)

    # Store updated parameter
    ct.store(param, (bid,), tile_p.astype(param.dtype))

def fused_adamw_step(
    param: torch.Tensor, grad: torch.Tensor,
    exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor,
    lr: float, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.01
):
    """Perform single AdamW optimization step."""
    TILE = 1024
    num_blocks = ct.cdiv(param.numel(), TILE)

    ct.launch(
        torch.cuda.current_stream(),
        (num_blocks,),
        fused_adamw_kernel,
        (param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, TILE)
    )

# Usage
param_size = 10_000_000
param = torch.randn(param_size, device="cuda", dtype=torch.float32)
grad = torch.randn(param_size, device="cuda", dtype=torch.float32)
exp_avg = torch.zeros(param_size, device="cuda", dtype=torch.float32)
exp_avg_sq = torch.zeros(param_size, device="cuda", dtype=torch.float32)

# Perform optimizer step
fused_adamw_step(param, grad, exp_avg, exp_avg_sq, lr=1e-3)

print(f"Parameter updated, mean: {param.mean():.6f}, std: {param.std():.6f}")
```

---

## Example 15: ReLU Backward (Beginner)

**Purpose**: Simplest backward kernel - introduction to autograd.Function pattern.

**Mathematics**:
```
Forward: y = max(0, x)
Backward: dy/dx = 1 if x > 0 else 0
```

**Implementation**:

```python
import torch
import cuda.tile as ct
from math import ceil

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """Forward: compute ReLU."""
        output = torch.empty_like(input)

        @ct.kernel
        def relu_forward(x, y, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
            tile_x = ct.gather(x, indices).astype(ct.float32)
            tile_y = ct.max(tile_x, 0.0)
            ct.scatter(y, indices, tile_y.astype(x.dtype))

        N = input.numel()
        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, relu_forward, (input, output, TILE))

        # Save input for backward
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: gradient is 1 where input > 0, else 0."""
        (input,) = ctx.saved_tensors
        grad_input = torch.empty_like(input)

        @ct.kernel
        def relu_backward(x, grad_out, grad_in, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

            tile_x = ct.gather(x, indices).astype(ct.float32)
            tile_grad_out = ct.gather(grad_out, indices).astype(ct.float32)

            # Gradient: 1 if x > 0, else 0
            mask = (tile_x > 0.0).astype(ct.float32)
            tile_grad_in = tile_grad_out * mask

            ct.scatter(grad_in, indices, tile_grad_in.astype(x.dtype))

        N = input.numel()
        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, relu_backward, (input, grad_output, grad_input, TILE))

        return grad_input


# Wrapper function
def relu_cutile(x):
    """Apply ReLU activation."""
    return ReLUFunction.apply(x)


# Test
if __name__ == "__main__":
    device = torch.device("cuda")

    # Forward + backward test
    x = torch.randn(1024, dtype=torch.float32, device=device, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    # Forward
    y = relu_cutile(x)
    y_ref = torch.relu(x_ref)

    # Backward
    grad_output = torch.ones_like(y)
    y.backward(grad_output)
    y_ref.backward(grad_output)

    # Validate
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    print("✓ ReLU forward and backward pass validated")

    # Numerical gradient check
    x_check = torch.randn(128, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(ReLUFunction.apply, (x_check,), eps=1e-6, atol=1e-4)
    print("✓ ReLU gradient check passed")
```

**Key Points**:
- Simplest pattern: mask-based gradient
- Save input for backward to determine mask
- Use `.astype(ct.float32)` for boolean mask conversion
- Validate with `torch.autograd.gradcheck()`

---

## Example 16: SwiGLU Backward (Intermediate)

**Purpose**: Recomputation strategy with fused gradient computation.

**Mathematics**:
```
Forward: output = silu(gate) * up = gate * sigmoid(gate) * up

Backward:
  dL/d(up) = grad_output * silu(gate)
  dL/d(gate) = grad_output * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
```

**Implementation**:

```python
import torch
import cuda.tile as ct
from math import ceil

class SiLUMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        """Forward: compute silu(gate) * up."""
        output = torch.empty_like(gate)

        @ct.kernel
        def swiglu_forward(gate, up, output, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

            g = ct.gather(gate, indices).astype(ct.float32)
            u = ct.gather(up, indices).astype(ct.float32)

            # silu(g) = g * sigmoid(g)
            sig_g = 1.0 / (1.0 + ct.exp(-g))
            silu_g = g * sig_g

            result = (silu_g * u).astype(gate.dtype)
            ct.scatter(output, indices, result)

        N = gate.numel()
        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, swiglu_forward, (gate, up, output, TILE))

        # Save inputs for recomputation in backward
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: recompute sigmoid and compute both gradients."""
        gate, up = ctx.saved_tensors

        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)

        @ct.kernel
        def swiglu_backward(grad_out, gate, up, grad_gate, grad_up, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

            dout = ct.gather(grad_out, indices).astype(ct.float32)
            g = ct.gather(gate, indices).astype(ct.float32)
            u = ct.gather(up, indices).astype(ct.float32)

            # Recompute sigmoid and silu
            sig_g = 1.0 / (1.0 + ct.exp(-g))
            silu_g = g * sig_g

            # Fused gradient computation
            dgrad_up = dout * silu_g

            # d/dg[silu(g)] = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
            dsilu_dg = sig_g * (1.0 + g * (1.0 - sig_g))
            dgrad_gate = dout * u * dsilu_dg

            ct.scatter(grad_gate, indices, dgrad_gate.astype(gate.dtype))
            ct.scatter(grad_up, indices, dgrad_up.astype(up.dtype))

        N = gate.numel()
        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, swiglu_backward,
                  (grad_output, gate, up, grad_gate, grad_up, TILE))

        return grad_gate, grad_up


# Wrapper
def swiglu_cutile(gate, up):
    """Apply SwiGLU: silu(gate) * up."""
    return SiLUMulFunction.apply(gate, up)


# Test
if __name__ == "__main__":
    device = torch.device("cuda")

    # Forward + backward test
    gate = torch.randn(1024, dtype=torch.float32, device=device, requires_grad=True)
    up = torch.randn(1024, dtype=torch.float32, device=device, requires_grad=True)
    gate_ref = gate.clone().detach().requires_grad_(True)
    up_ref = up.clone().detach().requires_grad_(True)

    # Forward
    output = swiglu_cutile(gate, up)
    output_ref = torch.nn.functional.silu(gate_ref) * up_ref

    # Backward
    grad_output = torch.ones_like(output)
    output.backward(grad_output)
    output_ref.backward(grad_output)

    # Validate
    torch.testing.assert_close(output, output_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gate.grad, gate_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(up.grad, up_ref.grad, rtol=1e-4, atol=1e-4)
    print("✓ SwiGLU forward and backward validated")

    # Numerical gradient check
    gate_check = torch.randn(128, dtype=torch.float64, device=device, requires_grad=True)
    up_check = torch.randn(128, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(SiLUMulFunction.apply, (gate_check, up_check), eps=1e-6, atol=1e-4)
    print("✓ SwiGLU gradient check passed")
```

**Key Points**:
- Recomputation strategy: save inputs, recompute sigmoid in backward
- Fused gradients: compute both `grad_gate` and `grad_up` in single kernel
- Use float32 for sigmoid/exp to maintain numerical stability
- Complex derivative: carefully derive and validate `dsilu_dg`

---

## Example 17: RMSNorm Backward (Advanced)

**Purpose**: Saved statistics pattern with row-parallel reduction.

**Mathematics**:
```
Forward: y = weight * x / sqrt(mean(x^2) + eps)

Backward:
  dL/dx = dL/dy * weight * inv_rms - x * inv_rms^3 / N * sum(dL/dy * weight * x)
  dL/d(weight) = sum_rows(dL/dy * x * inv_rms)
```

**Implementation**:

```python
import torch
import cuda.tile as ct

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        """Forward: compute RMSNorm and save statistics."""
        M, N = x.shape[0], normalized_shape[0]
        output = torch.empty_like(x)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        @ct.kernel
        def rms_norm_forward(x, weight, output, rstd, N: ct.Constant[int], eps: ct.Constant[float], TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = ct.arange(N, dtype=torch.int32)

            # Load row
            tile_x = ct.gather(x, bid * N + indices).astype(ct.float32)

            # Compute RMS
            variance = (tile_x * tile_x).sum() / N
            inv_rms = 1.0 / ct.sqrt(variance + eps)

            # Save inv_rms for backward
            ct.scatter(rstd, ct.full((1,), bid, dtype=torch.int32), ct.full((1,), inv_rms, dtype=ct.float32))

            # Normalize
            norm_x = tile_x * inv_rms

            # Apply weight
            tile_w = ct.gather(weight, indices).astype(ct.float32)
            result = (norm_x * tile_w).astype(x.dtype)

            ct.scatter(output, bid * N + indices, result)

        grid = (M, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, rms_norm_forward, (x, weight, output, rstd, N, eps, N))

        # Save for backward
        ctx.save_for_backward(x, weight, rstd)
        ctx.N = N

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: use saved rstd for gradient computation."""
        x, weight, rstd = ctx.saved_tensors
        N = ctx.N
        M = x.shape[0]

        grad_x = torch.empty_like(x)
        temp_buffer = torch.empty(M, N, dtype=torch.float32, device=x.device)

        @ct.kernel
        def rms_norm_backward(grad_out, x, weight, rstd, grad_x, temp_buffer, N: ct.Constant[int], TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = ct.arange(N, dtype=torch.int32)

            # Load
            dout = ct.gather(grad_out, bid * N + indices).astype(ct.float32)
            tile_x = ct.gather(x, bid * N + indices).astype(ct.float32)
            tile_w = ct.gather(weight, indices).astype(ct.float32)
            inv_rms = ct.gather(rstd, ct.full((1,), bid, dtype=torch.int32)).astype(ct.float32)

            # Direct term
            direct_term = dout * tile_w * inv_rms

            # Correction term
            weighted_grad_prod = (dout * tile_w * tile_x).sum()
            correction_term = tile_x * inv_rms * inv_rms * inv_rms * weighted_grad_prod / N

            # Final gradient
            grad_input = direct_term - correction_term

            ct.scatter(grad_x, bid * N + indices, grad_input.astype(x.dtype))

            # Store intermediate for weight gradient
            ct.scatter(temp_buffer, bid * N + indices, dout * tile_x * inv_rms)

        grid = (M, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, rms_norm_backward,
                  (grad_output, x, weight, rstd, grad_x, temp_buffer, N, N))

        # Weight gradient: sum across batch dimension
        grad_weight = temp_buffer.to(torch.float32).sum(dim=0).to(weight.dtype)

        return grad_x, None, grad_weight, None


# Wrapper
def rms_norm_cutile(x, normalized_shape, weight, eps=1e-5):
    """Apply RMSNorm."""
    return RMSNormFunction.apply(x, normalized_shape, weight, eps)


# Test
if __name__ == "__main__":
    device = torch.device("cuda")
    M, N = 256, 512

    # Forward + backward test
    x = torch.randn(M, N, dtype=torch.float32, device=device, requires_grad=True)
    weight = torch.randn(N, dtype=torch.float32, device=device, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)

    # Forward
    output = rms_norm_cutile(x, (N,), weight)

    # Reference
    variance = x_ref.pow(2).mean(-1, keepdim=True)
    output_ref = weight_ref * x_ref * torch.rsqrt(variance + 1e-5)

    # Backward
    grad_output = torch.ones_like(output)
    output.backward(grad_output)
    output_ref.backward(grad_output)

    # Validate
    torch.testing.assert_close(output, output_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-4)
    print("✓ RMSNorm forward and backward validated")
```

**Key Points**:
- Saved statistics: pre-compute `inv_rms` in forward, reuse in backward
- Row-parallel: each block processes one row independently
- Correction term: accounts for normalization constraint
- Weight gradient: use temp buffer, then sum reduction across batch

---

## Example 18: SiLU & Mul with Input Splitting (Intermediate)

**Purpose**: Gradient concatenation pattern for split inputs.

**Mathematics**:
```
Input: [gate, up] concatenated (shape: batch × 2*hidden_size)
Forward: output = silu(gate) * up

Backward:
  grad_input = [grad_gate, grad_up] concatenated
```

**Implementation**:

```python
import torch
import cuda.tile as ct

class SiLUAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Input shape: (batch, 2 * hidden_size)
        Split into gate and up, compute silu(gate) * up
        """
        batch, doubled_hidden = input.shape
        hidden_size = doubled_hidden // 2
        output = torch.empty(batch, hidden_size, dtype=input.dtype, device=input.device)

        @ct.kernel
        def forward_kernel(input, output, hidden_size: ct.Constant[int]):
            bid = ct.bid(0)
            indices = ct.arange(hidden_size, dtype=torch.int32)

            # Load gate and up from different halves
            gate = ct.gather(input, bid * (2 * hidden_size) + indices).astype(ct.float32)
            up = ct.gather(input, bid * (2 * hidden_size) + hidden_size + indices).astype(ct.float32)

            # Compute silu(gate) * up
            sig = 1.0 / (1.0 + ct.exp(-gate))
            result = (gate * sig * up).astype(input.dtype)

            ct.scatter(output, bid * hidden_size + indices, result)

        grid = (batch, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, forward_kernel, (input, output, hidden_size))

        ctx.save_for_backward(input)
        ctx.hidden_size = hidden_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute both grad_gate and grad_up in single kernel."""
        (input,) = ctx.saved_tensors
        hidden_size = ctx.hidden_size
        batch = grad_output.shape[0]

        grad_gate = torch.empty(batch, hidden_size, dtype=input.dtype, device=input.device)
        grad_up = torch.empty(batch, hidden_size, dtype=input.dtype, device=input.device)

        @ct.kernel
        def backward_kernel(grad_out, input, grad_gate, grad_up, hidden_size: ct.Constant[int]):
            bid = ct.bid(0)
            indices = ct.arange(hidden_size, dtype=torch.int32)

            # Load
            dout = ct.gather(grad_out, bid * hidden_size + indices).astype(ct.float32)
            gate = ct.gather(input, bid * (2 * hidden_size) + indices).astype(ct.float32)
            up = ct.gather(input, bid * (2 * hidden_size) + hidden_size + indices).astype(ct.float32)

            # Recompute sigmoid and silu
            sig = 1.0 / (1.0 + ct.exp(-gate))
            silu = gate * sig

            # Fused gradient computation
            dgrad_up = dout * silu
            dsilu_dgate = sig * (1.0 + gate * (1.0 - sig))
            dgrad_gate = dout * up * dsilu_dgate

            # Store both gradients
            ct.scatter(grad_gate, bid * hidden_size + indices, dgrad_gate.astype(input.dtype))
            ct.scatter(grad_up, bid * hidden_size + indices, dgrad_up.astype(input.dtype))

        grid = (batch, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, backward_kernel,
                  (grad_output, input, grad_gate, grad_up, hidden_size))

        # Concatenate gradients to match input layout
        grad_input = torch.cat([grad_gate, grad_up], dim=-1)

        return grad_input


# Wrapper
def silu_and_mul_cutile(input):
    """Apply SiLU and mul with input splitting."""
    return SiLUAndMulFunction.apply(input)


# Test
if __name__ == "__main__":
    device = torch.device("cuda")
    batch, hidden_size = 32, 512

    # Forward + backward test
    input = torch.randn(batch, 2 * hidden_size, dtype=torch.float32, device=device, requires_grad=True)
    input_ref = input.clone().detach().requires_grad_(True)

    # Forward
    output = silu_and_mul_cutile(input)

    # Reference
    gate_ref = input_ref[:, :hidden_size]
    up_ref = input_ref[:, hidden_size:]
    output_ref = torch.nn.functional.silu(gate_ref) * up_ref

    # Backward
    grad_output = torch.ones_like(output)
    output.backward(grad_output)
    output_ref.backward(grad_output)

    # Validate
    torch.testing.assert_close(output, output_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(input.grad, input_ref.grad, rtol=1e-4, atol=1e-4)
    print("✓ SiLU & Mul forward and backward validated")

    # Numerical gradient check
    input_check = torch.randn(4, 64, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(SiLUAndMulFunction.apply, (input_check,), eps=1e-6, atol=1e-4)
    print("✓ SiLU & Mul gradient check passed")
```

**Key Points**:
- Input splitting: load gate and up from different halves of input
- Gradient concatenation: `torch.cat([grad_gate, grad_up], dim=-1)`
- Fused backward: compute both gradients in single kernel
- Useful pattern for MLP blocks (gate + up projections)

---

**See also**:
- [EXAMPLES.md](EXAMPLES.md) - Quick start guide
- [EXAMPLES_COMPUTE.md](EXAMPLES_COMPUTE.md) - Compute-intensive operations
- [EXAMPLES_NORMALIZATION.md](EXAMPLES_NORMALIZATION.md) - Normalization operations
- [EXAMPLES_ELEMENTWISE.md](EXAMPLES_ELEMENTWISE.md) - Element-wise operations
