# cutile Programming Patterns

Common patterns and best practices for cutile kernel development.

## Table of Contents

1. [Memory Access Patterns](#memory-access-patterns)
2. [Reduction Patterns](#reduction-patterns)
3. [Attention Patterns](#attention-patterns)
4. [Performance Patterns](#performance-patterns)

---

## Memory Access Patterns

### Pattern 1: Structured Access (load/store)

Use for contiguous, tiled memory access.

```python
@ct.kernel
def structured_access(A, B, C, tm: ConstInt, tn: ConstInt):
    bidx, bidy = ct.bid(0), ct.bid(1)

    # Load contiguous tiles
    a_tile = ct.load(A, index=(bidx, 0), shape=(tm, tn),
                     padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(B, index=(bidx, 0), shape=(tm, tn),
                     padding_mode=ct.PaddingMode.ZERO)

    # Compute
    result = a_tile + b_tile

    # Store result
    ct.store(C, index=(bidx, 0), tile=result)
```

**When to use:**
- Accessing contiguous memory regions
- Matrix operations (matmul, bmm)
- Tile-aligned dimensions

**Benefits:**
- Automatic coalescing
- TMA support for latency hiding
- Efficient shared memory usage

### Pattern 2: Indexed Access (gather/scatter)

Use for non-contiguous or element-wise access.

```python
@ct.kernel
def indexed_access(input, output, TILE: ConstInt):
    bid = ct.bid(0)

    # Compute indices
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

    # Gather elements (auto-handles boundaries)
    values = ct.gather(input, indices)

    # Process
    result = values * 2.0

    # Scatter results (auto-masks out-of-bounds)
    ct.scatter(output, indices, result)
```

**When to use:**
- Element-wise operations
- Non-contiguous access patterns
- Sparse operations
- Flexible boundary handling

**Benefits:**
- Automatic boundary masking
- Flexible indexing
- Simpler for irregular patterns

### Pattern 3: Transposed Loads

Use `order` parameter for efficient transposed access.

```python
@ct.kernel
def transposed_load(K, V, output, TILE_D: ConstInt, TILE_N: ConstInt):
    bidx = ct.bid(0)

    # Load K transposed: (D, N) → (N, D)
    k_tile = ct.load(K, index=(bidx, 0), shape=(TILE_D, TILE_N),
                     order=(1, 0))  # Swap dims

    # Load V with 4D transposition: (B, H, N, D) → (B, H, D, N)
    v_tile = ct.load(V, index=(batch, head, 0, k_idx),
                     shape=(1, 1, TILE_N, TILE_D),
                     order=(0, 1, 3, 2))  # Swap last two dims
```

**When to use:**
- Attention K matrix (transpose for QK^T)
- Avoid explicit transpose operations
- MLA/FlashAttention kernels

**Benefits:**
- No separate transpose kernel needed
- Better memory efficiency
- Reduced kernel launch overhead

### Pattern 4: Hybrid Access (MoE)

Combine load/store for structured data with gather for routing.

```python
@ct.kernel
def moe_kernel(tokens, weights, expert_ids, sorted_ids, output,
               TILE_M: ConstInt, TILE_K: ConstInt):
    bid = ct.bid(0)

    # Gather token indices (irregular)
    token_idx = ct.gather(sorted_ids, bid)

    # Load expert weights (structured)
    expert_id = ct.gather(expert_ids, bid)
    w_tile = ct.load(weights, index=(expert_id, 0), shape=(TILE_M, TILE_K))

    # Load input tile (structured)
    input_tile = ct.load(tokens, index=(token_idx, 0), shape=(1, TILE_K))

    # Compute
    result = ct.mma(input_tile, w_tile.transpose(), accumulator)

    # Store (structured)
    ct.store(output, index=(token_idx, 0), tile=result)
```

**When to use:**
- Mixture of Experts
- Sparse operations with dense computation
- Routing-based kernels

---

## Reduction Patterns

### Pattern 1: Tile-Local Reduction

Reduce within a single tile.

```python
@ct.kernel
def tile_local_reduction(x, output, TILE_M: ConstInt, TILE_N: ConstInt):
    bid = ct.bid(0)

    # Load tile
    tile = ct.load(x, index=(bid, 0), shape=(TILE_M, TILE_N))

    # Row-wise reduction
    row_max = ct.max(tile, axis=-1, keepdims=True)  # Shape: (M, 1)
    row_sum = ct.sum(tile, axis=1)  # Shape: (M,)

    # Column-wise reduction
    col_max = ct.max(tile, axis=0, keepdims=True)  # Shape: (1, N)

    # Global tile reduction
    total = ct.sum(tile)  # Scalar (0D tile)
```

**When to use:**
- Softmax (row-wise max/sum)
- Layer normalization (mean/variance)
- Any per-row/column operation

### Pattern 2: Cross-Block Reduction

Use atomic operations for global reduction across blocks.

```python
@ct.kernel
def global_reduction(input, output, TILE: ConstInt):
    bid = ct.bid(0)

    # Each block processes a tile
    tile = ct.load(input, index=(bid,), shape=(TILE,))

    # Tile-local reduction
    local_sum = ct.sum(tile)

    # Atomic add for cross-block reduction
    ct.atomic_add(output, (0,), local_sum)
```

**When to use:**
- Total loss computation
- Global statistics (mean, variance across dataset)
- Any reduction requiring results from all blocks

**Pattern:**
1. Tile-local reduction (parallel within block)
2. Atomic operation for final aggregation

### Pattern 3: Online Reduction (Welford's Algorithm)

Update running statistics iteratively for numerically stable reductions.

```python
@ct.kernel
def online_softmax(Q, K, V, output,
                   TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bidx = ct.bid(0)

    # Initialize running max and sum
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load query (fixed for this block)
    q = ct.load(Q, index=(bidx, 0), shape=(TILE_M, TILE_D))

    # Iterate over K, V tiles
    num_kv_tiles = ct.num_tiles(K, axis=1, shape=(TILE_D, TILE_N))
    for k_idx in range(num_kv_tiles):
        # Load K, V tiles
        k = ct.load(K, index=(0, k_idx), shape=(TILE_D, TILE_N),
                    order=(1, 0))  # Transpose
        v = ct.load(V, index=(k_idx, 0), shape=(TILE_N, TILE_D))

        # Compute QK^T
        qk = ct.mma(q, k, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))

        # Online max update
        m_ij = ct.max(ct.max(qk, axis=-1, keepdims=True), m_i)

        # Subtract max (numerical stability)
        qk = qk - m_ij

        # Compute exp
        p = ct.exp2(qk * INV_LOG_2, flush_to_zero=True)

        # Online sum update
        l_ij = ct.sum(p, axis=-1, keepdims=True)

        # Rescale previous accumulator
        alpha = ct.exp2((m_i - m_ij) * INV_LOG_2, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Accumulate P @ V
        acc = ct.mma(p, v, acc)

        # Update running max
        m_i = m_ij

    # Final normalization
    output_tile = ct.truediv(acc, l_i, flush_to_zero=True)
    ct.store(output, index=(bidx, 0), tile=output_tile)
```

**When to use:**
- Flash Attention (online softmax)
- Streaming algorithms
- Numerically stable reductions over large sequences

**Benefits:**
- O(1) memory instead of O(N)
- Numerical stability (avoids overflow)
- Single-pass algorithm

### Pattern 4: Two-Stage Reduction

Split reduction into local reduction + final aggregation.

```python
# Stage 1: Local reduction per block
@ct.kernel
def reduce_stage1(input, partial_sums, TILE: ConstInt):
    bid = ct.bid(0)
    tile = ct.load(input, index=(bid,), shape=(TILE,))
    local_sum = ct.sum(tile)
    ct.store(partial_sums, index=(bid,), tile=local_sum.reshape((1,)))

# Stage 2: Final aggregation
@ct.kernel
def reduce_stage2(partial_sums, output, NUM_BLOCKS: ConstInt):
    partial_tile = ct.load(partial_sums, index=(0,), shape=(NUM_BLOCKS,))
    total = ct.sum(partial_tile)
    ct.store(output, index=(0,), tile=total.reshape((1,)))
```

**When to use:**
- Very large reductions (>10M elements)
- When atomic operations cause contention
- Split-KV attention reduction

---

## Attention Patterns

### Pattern 1: Online Softmax (Flash Attention)

Numerically stable attention without materializing full attention matrix.

**Key equations:**
- Running max: `m_i = max(m_i, max(QK_i))`
- Rescale factor: `alpha = exp(m_prev - m_new)`
- Running sum: `l_i = l_i * alpha + sum(exp(QK_i - m_i))`
- Accumulator: `acc = acc * alpha + P @ V`

See Pattern 3 in Reduction Patterns for full implementation.

### Pattern 2: Causal Masking

Apply triangular mask for autoregressive generation.

```python
@ct.kernel
def causal_attention(Q, K, V, output, CAUSAL: ConstBool,
                     TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt):
    bidx, head = ct.bid(0), ct.bid(1)

    q = ct.load(Q, index=(bidx, head, 0), shape=(TILE_M, TILE_D))

    num_kv_tiles = ct.num_tiles(K, axis=2, shape=(TILE_D, TILE_N))

    # ... setup accumulators ...

    for k_idx in range(num_kv_tiles):
        # Load K, V
        k = ct.load(K, index=(head, 0, k_idx), shape=(TILE_D, TILE_N))
        v = ct.load(V, index=(head, k_idx, 0), shape=(TILE_N, TILE_D))

        # Compute QK^T
        qk = ct.mma(q, k, qk_acc)

        # Apply causal mask
        if CAUSAL:
            # Compute position indices
            q_pos = bidx * TILE_M + ct.arange(TILE_M, dtype=torch.int32)[:, None]
            k_pos = k_idx * TILE_N + ct.arange(TILE_N, dtype=torch.int32)[None, :]

            # Causal mask: q_pos >= k_pos
            mask = q_pos >= k_pos
            qk = ct.where(mask, qk, -10000000.0)  # Large negative for exp

        # ... continue with online softmax ...
```

**When to use:**
- GPT-style autoregressive models
- Decoder-only transformers
- Any left-to-right processing

### Pattern 3: Grouped Query Attention (GQA)

Multiple query heads share single KV head.

```python
@ct.kernel
def gqa_attention(Q, K, V, output,
                  TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt,
                  QUERY_GROUP_SIZE: ConstInt):
    bid_q_tile, bid_head = ct.bid(0), ct.bid(1)

    # Map query head to KV head
    kv_head = bid_head // QUERY_GROUP_SIZE

    # Load query for this head
    q = ct.load(Q, index=(batch, bid_head, bid_q_tile, 0),
                shape=(1, 1, TILE_M, TILE_D))

    # Iterate over KV tiles (shared across query group)
    for k_idx in range(num_kv_tiles):
        k = ct.load(K, index=(batch, kv_head, 0, k_idx),
                    shape=(1, 1, TILE_D, TILE_N), order=(0, 1, 3, 2))
        v = ct.load(V, index=(batch, kv_head, k_idx, 0),
                    shape=(1, 1, TILE_N, TILE_D))

        # ... standard attention computation ...
```

**When to use:**
- Llama-style GQA models (8 Q heads : 1 KV head)
- Multi-Query Attention (MQA) - all queries share one KV
- Memory-efficient attention

### Pattern 4: Split-KV Parallelization

Parallelize attention across KV sequence dimension.

```python
@ct.kernel
def split_kv_attention(Q, K, V, partial_out, partial_lse,
                       TILE_M: ConstInt, TILE_N: ConstInt,
                       NUM_SPLITS: ConstInt):
    bid_q, bid_head, bid_split = ct.bid(0), ct.bid(1), ct.bid(2)

    # Compute this split's KV range
    kv_len = K.shape[2]
    split_size = ct.cdiv(kv_len, NUM_SPLITS)
    kv_start = bid_split * split_size
    kv_end = min(kv_start + split_size, kv_len)

    # Load query
    q = ct.load(Q, index=(batch, bid_head, bid_q, 0),
                shape=(1, 1, TILE_M, TILE_D))

    # Online softmax over this split's KV range
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    for k_idx in range(kv_start // TILE_N, ct.cdiv(kv_end, TILE_N)):
        # ... standard online softmax ...
        pass

    # Store partial results
    ct.store(partial_out, index=(batch, bid_head, bid_split, bid_q, 0),
             tile=acc)
    ct.store(partial_lse, index=(batch, bid_head, bid_split, bid_q, 0),
             tile=m_i)  # Store log-sum-exp

# Separate reduction kernel combines splits
@ct.kernel
def split_kv_reduce(partial_out, partial_lse, output,
                    NUM_SPLITS: ConstInt, TILE_M: ConstInt, TILE_D: ConstInt):
    bid_q, bid_head = ct.bid(0), ct.bid(1)

    # Load all splits
    max_lse = ct.full((TILE_M, 1), -np.inf, dtype=ct.float32)
    result = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    for split_idx in range(NUM_SPLITS):
        lse = ct.load(partial_lse, index=(batch, bid_head, split_idx, bid_q, 0),
                      shape=(1, 1, 1, TILE_M, 1))
        out = ct.load(partial_out, index=(batch, bid_head, split_idx, bid_q, 0),
                      shape=(1, 1, 1, TILE_M, TILE_D))

        # Combine using log-sum-exp
        new_max = ct.max(max_lse, lse)
        alpha = ct.exp2((lse - new_max) * INV_LOG_2, flush_to_zero=True)
        result = result * ct.exp2((max_lse - new_max) * INV_LOG_2) + out * alpha
        max_lse = new_max

    ct.store(output, index=(batch, bid_head, bid_q, 0), tile=result)
```

**When to use:**
- Long sequences (>2048 tokens)
- Decoding phase with large KV cache
- Improve GPU utilization

**Benefits:**
- Better parallelism for decode
- Reduces latency for long contexts

---

## Performance Patterns

### Pattern 1: Persistent Scheduling

Each block processes multiple tiles for better SM utilization.

```python
@ct.kernel
def persistent_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    bid = ct.bid(0)
    num_tile_blocks = ct.num_blocks(0)

    M, N, K = A.shape[0], B.shape[1], A.shape[1]
    total_tiles = ct.cdiv(M, tm) * ct.cdiv(N, tn)

    # Each block processes multiple output tiles
    for tile_id in range(bid, total_tiles, num_tile_blocks):
        tile_m = tile_id // ct.cdiv(N, tn)
        tile_n = tile_id % ct.cdiv(N, tn)

        # Process tile (tile_m, tile_n)
        accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)

        for k_idx in range(ct.cdiv(K, tk)):
            a = ct.load(A, index=(tile_m, k_idx), shape=(tm, tk),
                        padding_mode=ct.PaddingMode.ZERO)
            b = ct.load(B, index=(k_idx, tile_n), shape=(tk, tn),
                        padding_mode=ct.PaddingMode.ZERO)
            accumulator = ct.mma(a, b, accumulator)

        result = accumulator.astype(C.dtype)
        ct.store(C, index=(tile_m, tile_n), tile=result)

# Launch with fewer blocks than tiles
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
grid_size = min(NUM_SMS, total_tiles)
grid = (grid_size, 1, 1)
```

**When to use:**
- Large matrices with many tiles
- Unbalanced workloads
- Maximize GPU utilization

**Benefits:**
- Better SM occupancy
- Reduced launch overhead
- Dynamic load balancing

### Pattern 2: TMA Pipeline Hints

Use latency hints to hide memory transfer latency.

```python
@ct.kernel
def tma_optimized(Q, K, V, output, TILE_M: ConstInt, TILE_N: ConstInt):
    bidx = ct.bid(0)

    # Load with latency hints
    q = ct.load(Q, index=(bidx, 0), shape=(TILE_M, TILE_D),
                latency=2)  # Short latency for immediate use

    for k_idx in range(num_kv_tiles):
        # Prefetch K with higher latency
        k = ct.load(K, index=(0, k_idx), shape=(TILE_D, TILE_N),
                    latency=4, order=(1, 0))

        # Prefetch V even earlier
        v = ct.load(V, index=(k_idx, 0), shape=(TILE_N, TILE_D),
                    latency=10)

        # Compute (while next tiles are loading)
        qk = ct.mma(q, k, qk_acc)
        # ...
```

**Latency guidelines:**
- `latency=2`: Immediate use (current iteration)
- `latency=3-4`: Next iteration
- `latency=10`: Prefetch for future iterations

**When to use:**
- Memory-bound kernels
- Attention mechanisms
- Any kernel with predictable access patterns

### Pattern 3: 2D Swizzling

Reorder block execution for better L2 cache locality.

```python
def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    """Compute swizzled block indices."""
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

@ct.kernel
def swizzled_matmul(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    GROUP_SIZE_M = 8

    M, N = A.shape[0], B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    # ... standard matmul with swizzled indices ...
```

**When to use:**
- Large matrix multiplications
- When L2 cache reuse is important
- Production kernels

**Benefits:**
- Improved L2 cache hit rate
- Better memory bandwidth utilization

### Pattern 4: Power-of-2 Tile Size Selection

Choose optimal tile sizes within power-of-2 constraint.

```python
def select_tile_sizes(M, N, K, dtype):
    """Select optimal tile sizes for given problem."""

    if dtype.itemsize == 2:  # FP16/BF16
        # Larger tiles for Tensor Core efficiency
        if M >= 2048 and N >= 2048:
            tm, tn, tk = 256, 256, 64
        elif M >= 1024 or N >= 1024:
            tm, tn, tk = 128, 256, 64
        else:
            tm, tn, tk = 64, 128, 64
    else:  # FP32
        # Smaller tiles due to register pressure
        if M >= 1024 and N >= 1024:
            tm, tn, tk = 64, 64, 32
        else:
            tm, tn, tk = 32, 32, 32

    return tm, tn, tk
```

**Constraints:**
- Tile dims must be powers of 2
- Total tile elements ~ 256-1024 for good occupancy
- Consider dtype size (FP16 vs FP32)
- K dimension: 16/32/64 for Tensor Core alignment

**GPU-specific guidelines:**
- **B200**: Can use larger tiles (256x256), more registers
- **RTX 5090**: Medium tiles (128x128), balance occupancy
- **RTX 5080**: Smaller tiles (64x64), more conservative

---

## Backward Pass Patterns

### Overview

Backward pass kernels compute gradients for training deep learning models. cutile kernels integrate with PyTorch's autograd via `torch.autograd.Function`, enabling custom backward implementations.

**Key Concepts:**
- **Forward pass**: Computes output from inputs, saves tensors needed for backward
- **Backward pass**: Computes input gradients from output gradients
- **Tensor saving strategies**: Recomputation vs saved statistics
- **Gradient computation**: Apply chain rule with kernel-specific derivatives

---

### Pattern 1: Recomputation Strategy

**When to use:** Simple activations where recomputing is cheaper than storing.

**Advantages:**
- Lower memory footprint
- Fewer tensors saved in ctx
- Good for memory-constrained scenarios

**Example: SwiGLU Backward**

SwiGLU computes: `output = silu(gate) * up = gate * sigmoid(gate) * up`

```python
class SiLUMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        """Forward: compute silu(gate) * up"""
        output = torch.empty_like(gate)

        @ct.kernel
        def swiglu_forward_kernel(gate, up, output, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

            # Load inputs
            g = ct.gather(gate, indices).astype(ct.float32)
            u = ct.gather(up, indices).astype(ct.float32)

            # Compute silu(g) = g * sigmoid(g)
            sig_g = 1.0 / (1.0 + ct.exp(-g))
            silu_g = g * sig_g

            # Multiply
            result = (silu_g * u).astype(gate.dtype)
            ct.scatter(output, indices, result)

        N = gate.numel()
        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, swiglu_forward_kernel, (gate, up, output, TILE))

        # Save inputs for recomputation in backward
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: recompute sigmoid and compute gradients"""
        gate, up = ctx.saved_tensors

        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)

        @ct.kernel
        def swiglu_backward_kernel(grad_output, gate, up, grad_gate, grad_up, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

            # Load
            dout = ct.gather(grad_output, indices).astype(ct.float32)
            g = ct.gather(gate, indices).astype(ct.float32)
            u = ct.gather(up, indices).astype(ct.float32)

            # Recompute sigmoid and silu
            sig_g = 1.0 / (1.0 + ct.exp(-g))
            silu_g = g * sig_g

            # Gradient for up: dL/d(up) = grad_output * silu(gate)
            dgrad_up = dout * silu_g

            # Gradient for gate: dL/d(gate) = grad_output * up * d/dg[silu(g)]
            # d/dg[silu(g)] = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
            dsilu_dg = sig_g * (1.0 + g * (1.0 - sig_g))
            dgrad_gate = dout * u * dsilu_dg

            # Store
            ct.scatter(grad_gate, indices, dgrad_gate.astype(gate.dtype))
            ct.scatter(grad_up, indices, dgrad_up.astype(up.dtype))

        N = gate.numel()
        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, swiglu_backward_kernel,
                  (grad_output, gate, up, grad_gate, grad_up, TILE))

        return grad_gate, grad_up
```

**Gradient Mathematics:**
```
Forward: output = silu(g) * u = g * sigmoid(g) * u

Backward:
  dL/du = dL/doutput * silu(g)
        = grad_output * g * sigmoid(g)

  dL/dg = dL/doutput * u * d/dg[silu(g)]
        = grad_output * u * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
```

---

### Pattern 2: Saved Statistics

**When to use:** Normalization layers where pre-computed stats are reused.

**Advantages:**
- Avoid recomputing expensive operations (variance, mean)
- More efficient for complex layers
- Time-space tradeoff favoring speed

**Example: RMSNorm Backward**

RMSNorm computes: `output = weight * input / sqrt(mean(input^2) + eps)`

```python
class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        """Forward: compute RMSNorm and save statistics"""
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

            # Save rstd for backward
            ct.scatter(rstd, ct.full((1,), bid, dtype=torch.int32), ct.full((1,), inv_rms, dtype=ct.float32))

            # Normalize
            norm_x = tile_x * inv_rms

            # Apply weight
            tile_w = ct.gather(weight, indices).astype(ct.float32)
            result = (norm_x * tile_w).astype(x.dtype)

            # Store
            ct.scatter(output, bid * N + indices, result)

        grid = (M, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, rms_norm_forward, (x, weight, output, rstd, N, eps, N))

        # Save inputs and statistics
        ctx.save_for_backward(x, weight, rstd)
        ctx.N = N
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: use saved rstd for efficient gradient computation"""
        x, weight, rstd = ctx.saved_tensors
        N = ctx.N
        M = x.shape[0]

        grad_x = torch.empty_like(x)
        temp_buffer = torch.empty(M, N, dtype=torch.float32, device=x.device)

        @ct.kernel
        def rms_norm_backward(grad_output, x, weight, rstd, grad_x, temp_buffer, N: ct.Constant[int], TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = ct.arange(N, dtype=torch.int32)

            # Load
            dout = ct.gather(grad_output, bid * N + indices).astype(ct.float32)
            tile_x = ct.gather(x, bid * N + indices).astype(ct.float32)
            tile_w = ct.gather(weight, indices).astype(ct.float32)
            inv_rms = ct.gather(rstd, ct.full((1,), bid, dtype=torch.int32)).astype(ct.float32)

            # Direct term: dL/dx = grad_output * weight * inv_rms
            direct_term = dout * tile_w * inv_rms

            # Correction term: accounts for normalization constraint
            weighted_grad_prod = (dout * tile_w * tile_x).sum()
            correction_term = tile_x * inv_rms * inv_rms * inv_rms * weighted_grad_prod / N

            # Final gradient
            grad_input = direct_term - correction_term

            # Store gradient
            ct.scatter(grad_x, bid * N + indices, grad_input.astype(x.dtype))

            # Store intermediate for weight gradient
            ct.scatter(temp_buffer, bid * N + indices, dout * tile_x * inv_rms)

        grid = (M, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, rms_norm_backward,
                  (grad_output, x, weight, rstd, grad_x, temp_buffer, N, N))

        # Weight gradient: sum across batch dimension
        grad_weight = temp_buffer.to(torch.float32).sum(dim=0).to(weight.dtype)

        return grad_x, None, grad_weight, None
```

**Gradient Mathematics:**
```
Forward: y = w * x * inv_rms, where inv_rms = 1 / sqrt(mean(x^2) + eps)

Backward:
  dL/dx = dL/dy * w * inv_rms - x * inv_rms^3 / N * sum(dL/dy * w * x)
  dL/dw = sum_rows(dL/dy * x * inv_rms)
```

---

### Pattern 3: Fused Gradients

**When to use:** Multiple input gradients computed in single kernel.

**Advantages:**
- Fewer kernel launches
- Better memory locality
- Shared intermediate computations

**Example: Input Splitting with Gradient Concatenation**

```python
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
        def forward_kernel(input, output, TILE: ct.Constant[int], hidden_size: ct.Constant[int]):
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
        ct.launch(torch.cuda.current_stream(), grid, forward_kernel, (input, output, hidden_size, hidden_size))

        ctx.save_for_backward(input)
        ctx.hidden_size = hidden_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute both grad_gate and grad_up in single kernel"""
        (input,) = ctx.saved_tensors
        hidden_size = ctx.hidden_size
        batch = grad_output.shape[0]

        grad_gate = torch.empty(batch, hidden_size, dtype=input.dtype, device=input.device)
        grad_up = torch.empty(batch, hidden_size, dtype=input.dtype, device=input.device)

        @ct.kernel
        def backward_kernel(grad_output, input, grad_gate, grad_up, TILE: ct.Constant[int], hidden_size: ct.Constant[int]):
            bid = ct.bid(0)
            indices = ct.arange(hidden_size, dtype=torch.int32)

            # Load
            dout = ct.gather(grad_output, bid * hidden_size + indices).astype(ct.float32)
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
                  (grad_output, input, grad_gate, grad_up, hidden_size, hidden_size))

        # Concatenate gradients to match input layout
        grad_input = torch.cat([grad_gate, grad_up], dim=-1)

        return grad_input
```

---

### Common Gradient Formulas

#### Activation Functions

**Sigmoid:**
```
Forward: y = 1 / (1 + exp(-x))
Backward: dy/dx = y * (1 - y)
```

**SiLU (Swish):**
```
Forward: y = x * sigmoid(x)
Backward: dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
```

**ReLU:**
```
Forward: y = max(0, x)
Backward: dy/dx = 1 if x > 0 else 0
```

**GELU:**
```
Forward: y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
Backward: Complex, use numerical gradcheck for validation
```

#### Normalization

**LayerNorm / RMSNorm:**
```
Key insight: Normalization constraint requires correction term
Direct term: grad * weight * inv_std
Correction: -x * inv_std^3 / N * sum(grad * weight * x)
```

**Softmax:**
```
Forward: y_i = exp(x_i) / sum(exp(x_j))
Backward: dy_i/dx_j = y_i * (δ_ij - y_j)
```

---

### Backward Pass Checklist

**Design Phase:**
- [ ] Identify what needs to be saved (`ctx.save_for_backward`)
- [ ] Choose strategy: recomputation vs saved statistics
- [ ] Derive gradient formulas (chain rule)
- [ ] Consider numerical stability (use float32 for intermediate computations)

**Implementation Phase:**
- [ ] Load saved tensors from ctx
- [ ] Recompute or load statistics
- [ ] Compute gradients following chain rule
- [ ] Cast to appropriate dtype before storing
- [ ] Return gradients in same order as forward inputs (use `None` for non-differentiable inputs)

**Validation Phase:**
- [ ] Test forward-backward correctness against PyTorch reference
- [ ] Use `torch.autograd.gradcheck()` for numerical validation
- [ ] Test with different dtypes (FP16, FP32)
- [ ] Verify gradient shapes match input shapes
- [ ] Check for NaN/Inf in gradients

---

### Best Practices

1. **Use float32 for intermediate computations** - Even with FP16 inputs, use float32 for sigmoid, exp, sqrt to maintain numerical stability

2. **Recompute cheap operations** - sigmoid, exp are cheaper to recompute than storing activations

3. **Save expensive statistics** - Variance, mean, softmax normalizers should be saved

4. **Fuse when possible** - Compute multiple gradients in single kernel to reduce kernel launches

5. **Test numerically** - Always use `torch.autograd.gradcheck()` with FP64 for rigorous validation

6. **Handle edge cases** - Test with zero inputs, very large/small values, NaN propagation
