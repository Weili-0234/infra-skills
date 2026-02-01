# cutile Debugging Guide

Troubleshooting and debugging cutile kernels.

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Common Errors](#common-errors)
3. [Debugging Workflow](#debugging-workflow)
4. [Testing Strategies](#testing-strategies)

---

## Environment Variables

### Compilation Debug Variables

```bash
# Save bytecode to disk for inspection
export CUDA_TILE_DUMP_BYTECODE=1

# Save MLIR module
export CUDA_TILE_DUMP_TILEIR=1

# Extend compiler timeout (default: 60 seconds)
export CUDA_TILE_COMPILER_TIMEOUT_SEC=120

# Example usage
CUDA_TILE_DUMP_BYTECODE=1 CUDA_TILE_DUMP_TILEIR=1 python my_kernel.py
```

**Output locations:**
- Bytecode: `./cutile_bytecode_*.bin`
- MLIR: `./cutile_tileir_*.mlir`

---

## Common Errors

### Compilation Errors

#### Error: "Tile dimensions must be power of 2"

**Cause:** Tile shape contains non-power-of-2 dimension.

**Example:**
```python
# ❌ Wrong
tile = ct.load(array, (0, 0), shape=(100, 200))

# ✅ Correct
tile = ct.load(array, (0, 0), shape=(128, 256))
```

**Fix:** Use 16, 32, 64, 128, 256, 512, or 1024.

#### Error: "Shape mismatch in mma"

**Cause:** Incompatible dimensions for matrix multiplication.

**Example:**
```python
# ❌ Wrong: K dimension mismatch
a = ct.load(A, (bidx, k), shape=(128, 32))  # (M=128, K=32)
b = ct.load(B, (k, bidy), shape=(64, 128))  # (K=64, N=128) - K mismatch!
accumulator = ct.mma(a, b, accumulator)

# ✅ Correct
a = ct.load(A, (bidx, k), shape=(128, 64))  # (M=128, K=64)
b = ct.load(B, (k, bidy), shape=(64, 128))  # (K=64, N=128) - K matches!
accumulator = ct.mma(a, b, accumulator)
```

**Fix:** Ensure `a.shape[-1] == b.shape[0]` (K dimension must match).

#### Error: "Out of shared memory"

**Cause:** Tile sizes too large or too many pipeline stages.

**Example:**
```python
# ❌ Too large
tile = ct.load(array, (0, 0), shape=(512, 512))  # 512KB for FP16!

# ✅ Smaller tiles
tile = ct.load(array, (0, 0), shape=(128, 256))  # 64KB for FP16
```

**Fix:**
1. Reduce tile sizes (256→128, 128→64)
2. Use smaller data types (FP32→FP16)
3. Reduce pipeline stages if using pipelining

**Shared memory limits:**
- B200: ~230KB
- H100: ~228KB
- RTX 5090: ~100KB
- RTX 5080: ~64KB

#### Error: "Compilation timeout"

**Cause:** Complex kernel exceeds 60-second compile time.

**Fix:**
```bash
export CUDA_TILE_COMPILER_TIMEOUT_SEC=180
```

### Runtime Errors

#### Results are all zeros

**Causes:**
1. Accumulator not initialized
2. Incorrect grid dimensions
3. Wrong tile indexing

**Debug checklist:**
```python
# ✅ Check 1: Initialize accumulator
accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)  # Initialize!

# ✅ Check 2: Verify grid covers all data
grid = (math.ceil(M / tm), math.ceil(N / tn), 1)  # Use ceiling division

# ✅ Check 3: Verify tile indices
print(f"Grid: {grid}, M={M}, N={N}, tm={tm}, tn={tn}")

# ✅ Check 4: Print intermediate values
# (This requires debugging outside kernel - use small test case)
```

#### Results contain NaN or Inf

**Causes:**
1. Division by zero
2. Overflow in exp
3. Wrong accumulator dtype

**Fixes:**
```python
# ✅ Use float32 accumulators
accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)  # Not FP16!

# ✅ Add epsilon to divisions
epsilon = 1e-8
result = ct.truediv(tile, sum_tile + epsilon, flush_to_zero=True)

# ✅ Use exp2 with numerical stability
max_val = ct.max(tile, axis=-1, keepdims=True)
exp_vals = ct.exp2((tile - max_val) * INV_LOG_2, flush_to_zero=True)

# ✅ Clamp extreme values
clamped = ct.max(ct.min(tile, 1e4), -1e4)
```

#### Incorrect results (but not NaN)

**Causes:**
1. Indexing error
2. Incorrect boundary handling
3. Missing padding
4. Wrong memory layout (order parameter)

**Debug strategy:**
```python
# Test with small, known inputs
A = torch.ones(64, 64, dtype=torch.float16, device='cuda')
B = torch.ones(64, 64, dtype=torch.float16, device='cuda')
C = matmul(A, B)

# Expected: C should be 64 * ones(64, 64)
print(f"C[0,0] = {C[0,0]}, expected = 64.0")
print(f"C min/max = {C.min()}/{C.max()}")

# Check a specific tile
bidx, bidy = 0, 0
expected_tile = A[bidx*tm:(bidx+1)*tm, :] @ B[:, bidy*tn:(bidy+1)*tn]
actual_tile = C[bidx*tm:(bidx+1)*tm, bidy*tn:(bidy+1)*tn]
torch.testing.assert_close(actual_tile, expected_tile)
```

#### CUDA errors (invalid configuration, memory errors)

**Cause:** Grid dimensions exceed hardware limits.

**Fix:**
```python
# ✅ Check grid dimensions
max_grid_x = 2**31 - 1  # Usually not an issue
max_grid_y = 65535
max_grid_z = 65535

assert grid[0] <= max_grid_x
assert grid[1] <= max_grid_y
assert grid[2] <= max_grid_z
```

### Performance Issues

#### Kernel is slower than expected

**Diagnose:**
1. Profile with Nsight Compute
2. Check SM occupancy
3. Verify Tensor Core usage
4. Check memory bandwidth

```bash
# Profile with Nsight Compute
ncu --set full --target-processes all python script.py

# Key metrics:
# - Achieved Occupancy (target: >50%)
# - Memory Throughput (target: >60% of peak)
# - Compute Throughput (for compute-bound kernels)
```

**Common fixes:**
```python
# ✅ Use TMA latency hints
tile = ct.load(array, index, shape, latency=2)

# ✅ Use persistent scheduling for large problems
# (See OPTIMIZATION.md)

# ✅ Add 2D swizzling for matmul
bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

# ✅ Use appropriate tile sizes
# (See OPTIMIZATION.md)
```

---

## Debugging Workflow

### Step 1: Verify Kernel Compiles

```python
# Test compilation without running
@ct.kernel
def my_kernel(a, b, TILE: ConstInt):
    bid = ct.bid(0)
    # ... kernel code ...

# If this runs without error, kernel compiles
try:
    print("Kernel compiled successfully")
except Exception as e:
    print(f"Compilation error: {e}")
```

### Step 2: Test with Minimal Input

```python
# Start with very small inputs
A = torch.ones(16, 16, dtype=torch.float16, device='cuda')
B = torch.ones(16, 16, dtype=torch.float16, device='cuda')

# Run kernel
C = my_kernel_wrapper(A, B)

# Check shape
assert C.shape == (16, 16), f"Wrong shape: {C.shape}"

# Check values make sense
print(f"C min/max/mean: {C.min():.4f}/{C.max():.4f}/{C.mean():.4f}")
```

### Step 3: Compare Against Reference

```python
# Compute reference result
reference = compute_reference(A, B)

# Compare
try:
    torch.testing.assert_close(C, reference, rtol=1e-3, atol=1e-3)
    print("✅ Correctness check passed!")
except AssertionError as e:
    print(f"❌ Mismatch: {e}")

    # Print detailed mismatch
    diff = torch.abs(C - reference)
    print(f"Max absolute error: {diff.max()}")
    print(f"Mean absolute error: {diff.mean()}")

    # Find worst error location
    worst_idx = torch.argmax(diff)
    i, j = worst_idx // C.shape[1], worst_idx % C.shape[1]
    print(f"Worst error at ({i},{j}): got {C[i,j]}, expected {reference[i,j]}")
```

### Step 4: Test Edge Cases

```python
# Non-square matrices
test_shapes = [
    (128, 256),  # Wide
    (256, 128),  # Tall
    (17, 19),    # Prime dimensions
    (1024, 1024),  # Large square
]

for M, N in test_shapes:
    A = torch.randn(M, 64, dtype=torch.float16, device='cuda')
    B = torch.randn(64, N, dtype=torch.float16, device='cuda')
    C = matmul(A, B)
    reference = A @ B
    torch.testing.assert_close(C, reference, rtol=1e-3, atol=1e-3)
    print(f"✅ Shape ({M}, {N}) passed")
```

### Step 5: Profile and Optimize

```bash
# Profile with Nsight Compute
ncu --set full --target-processes all python script.py

# Or use PyTorch profiler
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    result = my_kernel_wrapper(A, B)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Testing Strategies

### Unit Test Structure

```python
import pytest
import torch
import cuda.tile as ct

class TestMyKernel:
    """Test my_kernel correctness."""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    def test_small_square(self, device):
        """Test with small square matrices."""
        A = torch.randn(64, 64, dtype=torch.float16, device=device)
        B = torch.randn(64, 64, dtype=torch.float16, device=device)

        result = my_kernel_wrapper(A, B)
        reference = A @ B

        torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)

    def test_non_square(self, device):
        """Test with non-square matrices."""
        A = torch.randn(128, 256, dtype=torch.float16, device=device)
        B = torch.randn(256, 512, dtype=torch.float16, device=device)

        result = my_kernel_wrapper(A, B)
        reference = A @ B

        torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)

    def test_fp32(self, device):
        """Test with float32 dtype."""
        A = torch.randn(128, 128, dtype=torch.float32, device=device)
        B = torch.randn(128, 128, dtype=torch.float32, device=device)

        result = my_kernel_wrapper(A, B)
        reference = A @ B

        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)
```

### Tolerance Selection

| Data Type | Relative Tolerance | Absolute Tolerance | Use Case |
|-----------|-------------------|-------------------|----------|
| float16 | 1e-3 | 1e-3 | Standard FP16 kernels |
| float32 | 1e-4 | 1e-4 | Standard FP32 kernels |
| bfloat16 | 5e-3 | 5e-3 | BF16 (lower precision) |
| Attention | 1e-2 | 1e-2 | Online softmax (accumulated error) |

```python
# FP16 matmul
torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)

# FP32 matmul
torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)

# Flash Attention (more tolerant due to online algorithm)
torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
```

### Property-Based Testing

```python
import hypothesis
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

@hypothesis.given(
    M=st.integers(min_value=16, max_value=512).filter(lambda x: x % 16 == 0),
    N=st.integers(min_value=16, max_value=512).filter(lambda x: x % 16 == 0),
    K=st.integers(min_value=16, max_value=512).filter(lambda x: x % 16 == 0),
)
def test_matmul_properties(M, N, K):
    """Test matmul with random dimensions."""
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')

    result = matmul(A, B)
    reference = A @ B

    assert result.shape == (M, N)
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)
```

### Benchmark Template

```python
import time

def benchmark_kernel(kernel_fn, A, B, num_warmup=10, num_iters=100):
    """Benchmark kernel performance."""

    # Warmup
    for _ in range(num_warmup):
        result = kernel_fn(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        result = kernel_fn(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    # Compute TFLOPS
    M, K = A.shape
    K, N = B.shape
    flops = 2 * M * N * K  # 2 for multiply-add
    tflops = (flops / (avg_time_ms / 1000)) / 1e12

    return avg_time_ms, tflops

# Usage
A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')

time_ms, tflops = benchmark_kernel(matmul, A, B)
print(f"Average time: {time_ms:.3f} ms")
print(f"Performance: {tflops:.2f} TFLOPS")
```

---

## Debugging Checklist

Before asking for help, verify:

- [ ] Kernel compiles without errors
- [ ] Tile dimensions are all powers of 2
- [ ] Grid dimensions are correct (cover all data)
- [ ] Tested with minimal input (16x16 or smaller)
- [ ] Compared against reference implementation
- [ ] Checked for NaN/Inf in output
- [ ] Used float32 accumulators for matrix ops
- [ ] Added epsilon to divisions where needed
- [ ] Tested multiple input sizes (small, medium, large)
- [ ] Checked environment variables (CUDA_TILE_DUMP_*)
- [ ] Profiled with Nsight Compute (if performance issue)

---

## Getting Help

If stuck after debugging:

1. **Reduce to minimal example** - Simplify kernel to smallest reproducible case
2. **Check cutile GitHub issues** - Search for similar problems
3. **Provide context**:
   - cutile version (`pip show cuda-tile`)
   - CUDA version (`nvcc --version`)
   - GPU model
   - Minimal code to reproduce
   - Expected vs actual behavior
   - Error messages (full stack trace)
