# LLM-Specific Kernel Examples

Complete, runnable Triton kernels for operations common in large language model inference and training. Each example includes the kernel, a PyTorch wrapper, and validation code.

```python
import triton
import triton.language as tl
```

---

## Table of Contents

1. [Rotary Position Embedding (RoPE)](#1-rotary-position-embedding-rope) -- Frequency computation, paired rotation, two layout variants
2. [SwiGLU / SiLU-and-Mul](#2-swiglu--silu-and-mul) -- Fused MLP activation, avoids intermediate materialization
3. [Fused Cross-Entropy Loss](#3-fused-cross-entropy-loss) -- Online log-softmax + NLL, chunked row processing
4. [Low-Memory Dropout](#4-low-memory-dropout) -- Philox PRNG, seedable, no stored mask

---

## 1. Rotary Position Embedding (RoPE)

Rotary embeddings encode absolute position by rotating pairs of elements in Q and K
by angles proportional to their position in the sequence. For each consecutive pair
`(x0, x1)` at frequency index `i`:

- `theta = position * freq`, where `freq = 1 / (10000 ^ (2i / dim))`
- `x0_new = x0 * cos(theta) - x1 * sin(theta)`
- `x1_new = x0 * sin(theta) + x1 * cos(theta)`

Two layout conventions exist for how pairs are organized within `head_dim`.

### Variant 1: Non-interleaved (GPT-NeoX style)

Pairs are formed from the first half and second half of `head_dim`:
pair `i` consists of elements at indices `i` and `i + head_dim/2`.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def rope_non_interleaved_kernel(
    QK,             # Pointer to Q or K tensor [batch, n_heads, seq_len, head_dim]
    cos_ptr,        # Pointer to cos table [seq_len, head_dim // 2]
    sin_ptr,        # Pointer to sin table [seq_len, head_dim // 2]
    seq_len,        # Sequence length
    half_dim,       # head_dim // 2
    stride_b,       # Stride for batch dimension
    stride_h,       # Stride for head dimension
    stride_s,       # Stride for sequence dimension
    stride_d,       # Stride for head_dim dimension (typically 1)
    stride_cos_s,   # Stride for cos/sin sequence dim
    stride_cos_d,   # Stride for cos/sin head_dim dim
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (batch, head, seq_pos) triple
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Base pointer for this (batch, head, seq_pos)
    base = QK + b * stride_b + h * stride_h + s * stride_s

    # Process the half_dim pairs in blocks
    for block_start in range(0, half_dim, BLOCK_SIZE):
        d = block_start + tl.arange(0, BLOCK_SIZE)
        mask = d < half_dim

        # Load the pair: x0 from first half, x1 from second half
        x0 = tl.load(base + d * stride_d, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(base + (d + half_dim) * stride_d, mask=mask, other=0.0).to(tl.float32)

        # Load cos and sin for this position and frequency index
        cos_offset = s * stride_cos_s + d * stride_cos_d
        cos = tl.load(cos_ptr + cos_offset, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)

        # Apply rotation
        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos

        # Store back in-place
        tl.store(base + d * stride_d, x0_new.to(tl.float16), mask=mask)
        tl.store(base + (d + half_dim) * stride_d, x1_new.to(tl.float16), mask=mask)
```

### Variant 2: Interleaved (LLaMA style)

Pairs are adjacent: pair `i` consists of elements at indices `2*i` and `2*i+1`.

```python
@triton.jit
def rope_interleaved_kernel(
    QK,             # Pointer to Q or K tensor [batch, n_heads, seq_len, head_dim]
    cos_ptr,        # Pointer to cos table [seq_len, head_dim // 2]
    sin_ptr,        # Pointer to sin table [seq_len, head_dim // 2]
    seq_len,
    half_dim,       # head_dim // 2
    stride_b, stride_h, stride_s, stride_d,
    stride_cos_s, stride_cos_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base = QK + b * stride_b + h * stride_h + s * stride_s

    for block_start in range(0, half_dim, BLOCK_SIZE):
        d = block_start + tl.arange(0, BLOCK_SIZE)
        mask = d < half_dim

        # Interleaved: pair i is at positions 2*i and 2*i+1
        x0 = tl.load(base + (2 * d) * stride_d, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(base + (2 * d + 1) * stride_d, mask=mask, other=0.0).to(tl.float32)

        cos_offset = s * stride_cos_s + d * stride_cos_d
        cos = tl.load(cos_ptr + cos_offset, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)

        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos

        tl.store(base + (2 * d) * stride_d, x0_new.to(tl.float16), mask=mask)
        tl.store(base + (2 * d + 1) * stride_d, x1_new.to(tl.float16), mask=mask)
```

### Wrapper and Validation

```python
def precompute_freqs(seq_len, head_dim, base=10000.0, device="cuda"):
    """Precompute cos/sin tables [seq_len, head_dim//2] for RoPE."""
    half_dim = head_dim // 2
    freq = 1.0 / (base ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    angles = torch.outer(torch.arange(seq_len, device=device).float(), freq)
    return angles.cos().contiguous(), angles.sin().contiguous()


def apply_rope_triton(Q, style="non-interleaved"):
    """Apply RoPE in-place. Q shape: [batch, n_heads, seq_len, head_dim]."""
    batch, n_heads, seq_len, head_dim = Q.shape
    half_dim = head_dim // 2
    cos, sin = precompute_freqs(seq_len, head_dim, device=Q.device)
    BLOCK_SIZE = min(triton.next_power_of_2(half_dim), 1024)
    grid = (batch, n_heads, seq_len)
    kernel = rope_non_interleaved_kernel if style == "non-interleaved" else rope_interleaved_kernel
    kernel[grid](Q, cos, sin, seq_len, half_dim,
                 Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                 cos.stride(0), cos.stride(1), BLOCK_SIZE=BLOCK_SIZE)
    return Q


# --- Validation ---
if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, S, D = 2, 8, 128, 64
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    Q_ref = Q.clone()
    apply_rope_triton(Q, style="non-interleaved")
    cos, sin = precompute_freqs(S, D, device="cuda")
    x0, x1 = Q_ref[:, :, :, :D//2].float(), Q_ref[:, :, :, D//2:].float()
    cos_exp, sin_exp = cos[None, None, :, :], sin[None, None, :, :]
    Q_ref[:, :, :, :D//2] = (x0 * cos_exp - x1 * sin_exp).half()
    Q_ref[:, :, :, D//2:] = (x0 * sin_exp + x1 * cos_exp).half()
    print(f"RoPE max error: {(Q - Q_ref).abs().max().item():.6f}")
```

---

## 2. SwiGLU / SiLU-and-Mul

SwiGLU is the standard MLP activation in modern LLMs (LLaMA, Mistral, etc.). Given
two projections -- `gate` and `up` -- the output is:

```
output = SiLU(gate) * up
```

where `SiLU(x) = x * sigmoid(x)` (also called the Swish activation). Fusing this
into a single kernel avoids writing the intermediate SiLU result to global memory.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    gate_ptr,       # Pointer to gate projection output
    up_ptr,         # Pointer to up projection output
    output_ptr,     # Pointer to output tensor
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load in float32 for numerical precision
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) = gate * sigmoid(gate)
    silu_gate = gate * tl.sigmoid(gate)
    output = silu_gate * up

    # Store result back in float16
    tl.store(output_ptr + offsets, output.to(tl.float16), mask=mask)


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: SiLU(gate) * up. gate and up must have the same shape."""
    assert gate.shape == up.shape, "gate and up must have the same shape"
    assert gate.is_contiguous() and up.is_contiguous()
    output = torch.empty_like(gate)
    n_elements = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    swiglu_kernel[grid](gate, up, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


# --- Validation ---
if __name__ == "__main__":
    torch.manual_seed(0)
    gate = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
    up = torch.randn(4, 2048, device="cuda", dtype=torch.float16)

    output_triton = swiglu_forward(gate, up)

    # Reference implementation
    silu_gate_ref = torch.nn.functional.silu(gate.float())
    output_ref = (silu_gate_ref * up.float()).half()

    print(f"SwiGLU max error: {(output_triton - output_ref).abs().max().item():.6f}")
```

---

## 3. Fused Cross-Entropy Loss

Fuses log-softmax and NLL into a single kernel per row, avoiding a `[batch, vocab]`
softmax tensor. Algorithm per row: iterate vocabulary in chunks, track running max
(online softmax), compute `log(sum(exp(x - max)))`, then `loss = log_sum_exp - logit[target]`.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr,         # [n_rows, n_cols] logits (float32 or float16)
    labels_ptr,         # [n_rows] integer labels
    loss_ptr,           # [n_rows] output per-token loss
    n_cols,             # Vocabulary size
    logits_row_stride,  # Stride between rows in logits
    ignore_index,       # Label value to ignore (e.g., -100)
    label_smoothing: tl.constexpr,  # Label smoothing factor (0.0 = none)
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row (token)
    row = tl.program_id(0)

    # Load the target label for this row
    label = tl.load(labels_ptr + row)

    # If this token should be ignored, write zero loss and return
    if label == ignore_index:
        tl.store(loss_ptr + row, 0.0)
        return

    # Base pointer for this row of logits
    row_start = logits_ptr + row * logits_row_stride

    # Online max + sum-of-exp (chunked over vocabulary)
    running_max = -float("inf")
    running_sum = 0.0
    target_logit = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        logits = tl.load(row_start + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

        # Online softmax: correct running sum when max changes
        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(logits - new_max), axis=0)
        running_max = new_max

        # Pick target logit if label falls in this block
        target_mask = col_offsets == label
        target_logit += tl.sum(tl.where(target_mask, logits, 0.0), axis=0)

    log_sum_exp = running_max + tl.log(running_sum)
    nll_loss = log_sum_exp - target_logit

    if label_smoothing > 0.0:
        # Blend NLL with uniform distribution loss (log_sum_exp approximation)
        nll_loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * log_sum_exp

    tl.store(loss_ptr + row, nll_loss)


def fused_cross_entropy(logits, labels, ignore_index=-100, label_smoothing=0.0):
    """Fused cross-entropy. logits: [N, V], labels: [N]. Returns [N] per-token loss."""
    n_rows, n_cols = logits.shape
    loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4096)
    cross_entropy_kernel[(n_rows,)](
        logits, labels, loss, n_cols, logits.stride(0),
        ignore_index, label_smoothing, BLOCK_SIZE=BLOCK_SIZE)
    return loss


# --- Validation ---
if __name__ == "__main__":
    torch.manual_seed(42)
    N, V = 128, 32000
    logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, V, (N,), device="cuda")
    labels[0], labels[10] = -100, -100  # test ignore_index

    loss_triton = fused_cross_entropy(logits, labels, ignore_index=-100)
    loss_ref = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=-100, reduction="none")

    valid = labels != -100
    print(f"Cross-entropy max error: {(loss_triton[valid] - loss_ref[valid]).abs().max().item():.6f}")
    print(f"  Triton mean: {loss_triton[valid].mean().item():.4f}")
    print(f"  PyTorch mean: {loss_ref[valid].mean().item():.4f}")
```

---

## 4. Low-Memory Dropout

Uses Triton's Philox PRNG (`tl.rand`) to deterministically generate dropout masks
from a `(seed, offset)` pair. The same mask is regenerated in the backward pass,
so no mask tensor is stored. Based on Triton tutorial 04.

- **Forward:** `output = x * mask / (1 - p)` where `mask = (rand(seed, offset) > p)`
- **Backward:** `grad_input = grad_output * mask / (1 - p)` using same `(seed, offset)`

```python
import torch
import triton
import triton.language as tl


@triton.jit
def dropout_forward_kernel(
    x_ptr,          # Input tensor pointer
    output_ptr,     # Output tensor pointer
    n_elements,     # Total number of elements
    p,              # Dropout probability (fraction to zero out)
    seed,           # Random seed for Philox PRNG
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Philox PRNG: deterministic random from (seed, offset) pair
    random = tl.rand(seed, offsets)
    keep_mask = random > p
    output = tl.where(keep_mask, x / (1.0 - p), 0.0)

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def dropout_backward_kernel(
    grad_output_ptr,   # Upstream gradient pointer
    grad_input_ptr,    # Output gradient pointer
    n_elements,
    p,                 # Same dropout probability as forward
    seed,              # Same seed as forward -- regenerates the same mask
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)

    # Regenerate the exact same mask using the same (seed, offsets)
    random = tl.rand(seed, offsets)
    keep_mask = random > p
    grad_input = tl.where(keep_mask, grad_output / (1.0 - p), 0.0)

    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


class SeededDropout(torch.autograd.Function):
    """Autograd wrapper: carries seed through forward/backward, stores no mask."""
    @staticmethod
    def forward(ctx, x, p, seed):
        output = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 1024),)
        dropout_forward_kernel[grid](x, output, n, p, seed, BLOCK_SIZE=1024)
        ctx.save_for_backward(torch.tensor([seed], device=x.device))
        ctx.p, ctx.n = p, n
        return output

    @staticmethod
    def backward(ctx, grad_output):
        seed = ctx.saved_tensors[0].item()
        grad_input = torch.empty_like(grad_output)
        grid = (triton.cdiv(ctx.n, 1024),)
        dropout_backward_kernel[grid](grad_output, grad_input, ctx.n, ctx.p, seed,
                                      BLOCK_SIZE=1024)
        return grad_input, None, None


def seeded_dropout(x, p=0.1, seed=0):
    """Drop elements with probability p using a deterministic seed."""
    return x if p == 0.0 else SeededDropout.apply(x, p, seed)


# --- Validation ---
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 4096, device="cuda", dtype=torch.float32, requires_grad=True)
    p = 0.1
    seed = 12345

    # Forward
    y = seeded_dropout(x, p=p, seed=seed)

    # Check dropout fraction and scaling
    zero_frac = (y == 0).float().mean().item()
    nonzero_mask = y != 0
    scale_err = (y[nonzero_mask] / x[nonzero_mask] - 1.0 / (1.0 - p)).abs().max().item()
    print(f"Dropout fraction (expected ~{p:.2f}): {zero_frac:.4f}")
    print(f"Scale error (should be ~0): {scale_err:.6f}")

    # Backward: gradient mask must match forward mask
    y.sum().backward()
    print(f"Grad zeros match dropped: {(x.grad[~nonzero_mask] == 0).all().item()}")

    # Reproducibility
    y2 = seeded_dropout(x.detach(), p=p, seed=seed)
    print(f"Reproducible with same seed: {torch.equal(y.detach(), y2)}")
```
