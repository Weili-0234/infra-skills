# Testing and Benchmarking cutile Kernels

This guide covers correctness testing and performance benchmarking for cutile kernels, based on proven patterns from TileGym and Liger-Kernel.

---

## Part 1: Correctness Testing

### Overview

Correctness testing validates that your cutile kernel produces the same results as a reference implementation (typically PyTorch). Testing both forward and backward passes ensures your kernel integrates correctly with autograd.

**Key Principles:**
- Test against a trusted reference implementation
- Use appropriate tolerances for floating-point comparisons
- Test both regular and irregular shapes
- Validate forward and backward passes separately
- Use parametrization for comprehensive coverage

---

### PyTest Setup

#### Installation

```bash
pip install pytest torch
```

#### pytest.ini Configuration

Create `pytest.ini` in your project root:

```ini
[pytest]
markers =
    cuda: Tests requiring CUDA GPU (Blackwell)
    integration: Integration tests running actual kernels
    compilation: Compilation-only tests (no GPU needed)
    slow: Long-running tests
    blackwell: Tests requiring Blackwell architecture specifically (SM 10.0+)

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Verbose output by default
addopts = -v --tb=short

# Ignore warnings from external libraries
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

#### conftest.py Fixtures

Create `tests/conftest.py`:

```python
import pytest
import torch

@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

@pytest.fixture
def blackwell_available():
    """Check if Blackwell GPU is available (compute capability 10.0+)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 10

@pytest.fixture
def cutile_available():
    """Check if cutile is installed."""
    try:
        import cuda.tile as ct
        return True
    except ImportError:
        return False

@pytest.fixture
def skip_if_no_cuda(cuda_available):
    """Skip test if CUDA is not available."""
    if not cuda_available:
        pytest.skip("CUDA not available")

@pytest.fixture
def skip_if_no_blackwell(blackwell_available):
    """Skip test if Blackwell GPU is not available."""
    if not blackwell_available:
        pytest.skip("Blackwell GPU (SM 10.0+) not available")

@pytest.fixture
def skip_if_no_cutile(cutile_available):
    """Skip test if cutile is not installed."""
    if not cutile_available:
        pytest.skip("cutile not installed (pip install cuda-tile)")

@pytest.fixture
def device():
    """Get CUDA device for tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")
```

---

### Test Structure Patterns

#### File Organization

```
tests/
├── conftest.py              # Shared fixtures
├── pytest.ini               # PyTest configuration
└── cutile-kernels/
    ├── test_elementwise.py  # Vector operations
    ├── test_matmul.py       # Matrix multiplication
    ├── test_attention.py    # Attention kernels
    └── test_normalization.py # LayerNorm, RMSNorm, etc.
```

#### Test Class Structure

Each test file should follow this pattern:

```python
import pytest
import torch
import cuda.tile as ct
from math import ceil

class TestMyKernelCompilation:
    """Compilation tests - no GPU required."""

    @pytest.mark.compilation
    def test_kernel_compiles(self, cutile_available):
        """Test that kernel definition compiles."""
        if not cutile_available:
            pytest.skip("cutile not installed")

        @ct.kernel
        def my_kernel(input, output, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
            tile = ct.gather(input, indices)
            result = tile * 2.0  # Example operation
            ct.scatter(output, indices, result)

        assert callable(my_kernel), "Kernel should be callable"


class TestMyKernelIntegration:
    """Integration tests - requires Blackwell GPU."""

    @staticmethod
    def reference(input):
        """Reference implementation using PyTorch."""
        return input * 2.0

    @pytest.mark.cuda
    @pytest.mark.integration
    @pytest.mark.parametrize("N", [1024, 4096, 16384])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_correctness(self, skip_if_no_blackwell, skip_if_no_cutile, device, N, dtype):
        """Test kernel correctness."""
        # Create test data
        input = torch.randn(N, dtype=dtype, device=device)
        output = torch.empty_like(input)

        # Run kernel
        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, my_kernel, (input, output, TILE))

        # Validate against reference
        expected = self.reference(input)
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)
```

#### Reference Implementation Pattern

Always define a static reference method:

```python
@staticmethod
def reference(input, param1, param2):
    """
    Reference implementation using vanilla PyTorch.

    Best practices:
    1. Use stable, well-tested PyTorch functions
    2. Match exact semantics of custom kernel
    3. Handle edge cases (NaN, infinity, dtype casting)
    4. Keep simple and readable
    """
    return torch.nn.functional.some_operation(input, param1, param2)
```

---

### Forward + Backward Testing

#### Forward-Only Testing

For kernels without backward pass:

```python
@pytest.mark.cuda
@pytest.mark.integration
def test_forward_only(self, skip_if_no_blackwell, skip_if_no_cutile, device):
    """Test forward pass only."""
    input = torch.randn(1024, dtype=torch.float32, device=device)

    # No requires_grad needed
    output = my_kernel_wrapper(input)
    expected = self.reference(input)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
```

#### Forward + Backward Testing

For kernels with backward pass (using `torch.autograd.Function`):

```python
@pytest.mark.cuda
@pytest.mark.integration
def test_forward_and_backward(self, skip_if_no_blackwell, skip_if_no_cutile, device):
    """Test forward and backward passes."""
    # Create inputs WITH requires_grad
    input = torch.randn(1024, dtype=torch.float32, device=device, requires_grad=True)
    input_ref = input.clone().detach().requires_grad_(True)

    # Forward pass
    output = MyKernelFunction.apply(input)
    output_ref = self.reference(input_ref)

    # Validate forward
    torch.testing.assert_close(output, output_ref, rtol=1e-5, atol=1e-5)

    # Backward pass
    grad_output = torch.ones_like(output)
    output.backward(grad_output)
    output_ref.backward(grad_output)

    # Validate backward (gradients)
    torch.testing.assert_close(input.grad, input_ref.grad, rtol=1e-4, atol=1e-4)
```

#### Numerical Gradient Validation with gradcheck

Use `torch.autograd.gradcheck()` for rigorous numerical validation:

```python
@pytest.mark.cuda
@pytest.mark.integration
def test_backward_gradcheck(self, skip_if_no_blackwell, skip_if_no_cutile, device):
    """Numerical gradient validation using finite differences."""
    # Use small inputs and float64 for gradcheck
    input = torch.randn(128, dtype=torch.float64, device=device, requires_grad=True)

    # gradcheck uses finite differences to validate gradients
    assert torch.autograd.gradcheck(
        MyKernelFunction.apply,
        (input,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    ), "Gradient check failed"
```

**gradcheck parameters:**
- `eps`: Step size for finite differences (default: 1e-6)
- `atol`: Absolute tolerance for comparison
- `rtol`: Relative tolerance for comparison
- `raise_exception`: If True, raises error on failure (recommended)

---

### Parametrization Strategies

#### Basic Parametrization

Test multiple shapes and dtypes:

```python
@pytest.mark.parametrize("N", [1024, 4096, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_multiple_configs(self, device, N, dtype):
    """Test runs for all combinations (3 shapes × 3 dtypes = 9 tests)."""
    # Test implementation
    pass
```

#### Irregular Shapes

Test non-power-of-2 shapes to ensure proper boundary handling:

```python
@pytest.mark.parametrize("N", [
    997,    # Prime number
    1000,   # Round number, not power of 2
    1023,   # 2^10 - 1
    4100,   # Slightly over 4096
])
def test_irregular_shapes(self, device, N):
    """Test with irregular (non-power-of-2) shapes."""
    # Kernel should handle boundaries correctly
    pass
```

#### Multi-Dimensional Parametrization

For matrix operations:

```python
@pytest.mark.parametrize(
    "M, K, N",
    [
        (256, 256, 256),    # Square, small
        (512, 256, 512),    # Rectangular
        (1024, 512, 256),   # Tall × wide
        (4096, 4096, 4096), # Large square
    ],
)
def test_matmul_shapes(self, device, M, K, N):
    """Test matrix multiplication with various shapes."""
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    # Test implementation
    pass
```

#### Conditional Parametrization

Skip certain parameter combinations:

```python
@pytest.mark.parametrize("M, N", [(256, 256), (4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_with_conditional_skip(self, device, M, N, dtype):
    """Skip large shapes with FP32 to avoid OOM."""
    if M >= 4096 and dtype == torch.float32:
        pytest.skip("Skip large FP32 to avoid OOM")

    # Test implementation
    pass
```

---

### Tolerance Guidelines

Different data types require different tolerances due to precision limitations.

#### Tolerance Table

| Data Type | Relative Tolerance (rtol) | Absolute Tolerance (atol) | Notes |
|-----------|---------------------------|---------------------------|-------|
| `float32` | 1e-5 | 1e-8 | Standard precision |
| `float16` | 1e-3 | 1e-3 | Half precision (lower accuracy) |
| `bfloat16` | 5e-3 | 5e-3 | Brain float (even lower accuracy) |
| `float64` | 1e-12 | 1e-15 | High precision (for gradcheck) |
| Attention | 1e-2 | 1e-2 | Numerical instability in softmax |

#### Using torch.testing.assert_close

```python
# FP32 (tight tolerance)
torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-8)

# FP16 (looser tolerance)
torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)

# BF16 (loosest tolerance)
torch.testing.assert_close(result, reference, rtol=5e-3, atol=5e-3)

# Attention kernels (numerical instability)
torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
```

#### Tolerance Formula

The comparison uses: `|result - reference| ≤ atol + rtol * |reference|`

- **Absolute tolerance (atol)**: Maximum allowed difference
- **Relative tolerance (rtol)**: Maximum allowed relative error

---

### Edge Case Coverage

#### Boundary Conditions

```python
@pytest.mark.parametrize("N", [
    1,      # Single element
    16,     # Minimum tile size
    1024,   # Regular size
    1023,   # Just below 1024
    1025,   # Just above 1024
])
def test_boundary_conditions(self, device, N):
    """Test edge cases and boundaries."""
    # Ensure kernel handles all sizes correctly
    pass
```

#### Empty Tensors

```python
def test_empty_tensor(self, device):
    """Test with empty tensor (0 elements)."""
    input = torch.empty(0, device=device, dtype=torch.float32)
    # Kernel should handle gracefully (no crash)
    try:
        output = my_kernel_wrapper(input)
        assert output.numel() == 0
    except RuntimeError as e:
        # Document expected behavior
        assert "empty tensor" in str(e).lower()
```

#### NaN and Inf Handling

```python
def test_nan_and_inf(self, device):
    """Test behavior with NaN and infinity."""
    input = torch.tensor([1.0, float('nan'), float('inf'), -float('inf')], device=device)
    output = my_kernel_wrapper(input)

    # Verify NaN/Inf propagation matches reference
    expected = self.reference(input)
    assert torch.isnan(output[1]) == torch.isnan(expected[1])
    assert torch.isinf(output[2]) == torch.isinf(expected[2])
```

---

### Complete Example: RMSNorm Test

```python
import pytest
import torch
import cuda.tile as ct
from math import ceil

class TestRMSNormCompilation:
    """Compilation tests for RMSNorm - no GPU required."""

    @pytest.mark.compilation
    def test_rmsnorm_kernel_compiles(self, cutile_available):
        """Test that RMSNorm kernel compiles."""
        if not cutile_available:
            pytest.skip("cutile not installed")

        @ct.kernel
        def rms_norm_kernel(x, w, out, N: ct.Constant[int], eps: ct.Constant[float], TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = ct.arange(N, dtype=torch.int32)

            # Load row
            tile_x = ct.gather(x, bid * N + indices).astype(ct.float32)

            # Compute RMS
            variance = (tile_x * tile_x).sum() / N
            inv_rms = 1.0 / ct.sqrt(variance + eps)

            # Normalize
            norm_x = tile_x * inv_rms

            # Apply weight
            tile_w = ct.gather(w, indices).astype(ct.float32)
            result = (norm_x * tile_w).astype(x.dtype)

            # Store
            ct.scatter(out, bid * N + indices, result)

        assert callable(rms_norm_kernel), "Kernel should be callable"


class TestRMSNormIntegration:
    """Integration tests for RMSNorm - requires Blackwell GPU."""

    @staticmethod
    def reference(x, normalized_shape, weight, eps):
        """Reference RMSNorm implementation."""
        dims = tuple(range(-len(normalized_shape), 0))
        variance = x.to(torch.float32).pow(2).mean(dims, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        if weight is None:
            return x_norm
        return weight * x_norm.to(weight.dtype)

    @pytest.mark.cuda
    @pytest.mark.integration
    @pytest.mark.parametrize("M, N", [(256, 512), (1024, 2048), (4096, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_correctness_fp16(self, skip_if_no_blackwell, skip_if_no_cutile, device, M, N, dtype):
        """Test RMSNorm correctness with various shapes and dtypes."""
        # Create test data
        x = torch.randn(M, N, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        output = torch.empty_like(x)
        eps = 1e-5

        # Launch kernel
        grid = (M, 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            rms_norm_kernel,
            (x, weight, output, N, eps, N),
        )

        # Validate
        expected = self.reference(x, (N,), weight, eps)

        # Use dtype-appropriate tolerance
        if dtype == torch.float16:
            torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_backward(self, skip_if_no_blackwell, skip_if_no_cutile, device):
        """Test RMSNorm backward pass (if implemented)."""
        M, N = 256, 512
        x = torch.randn(M, N, dtype=torch.float32, device=device, requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)
        weight = torch.randn(N, dtype=torch.float32, device=device, requires_grad=True)
        weight_ref = weight.clone().detach().requires_grad_(True)

        # Forward
        output = RMSNormFunction.apply(x, weight, 1e-5)
        output_ref = self.reference(x_ref, (N,), weight_ref, 1e-5)

        # Backward
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        output_ref.backward(grad_output)

        # Validate gradients
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-4, atol=1e-4)
```

---

## Part 2: Performance Benchmarking

### Overview

Performance benchmarking measures kernel throughput (TFLOPS or GB/s) and compares against baselines (PyTorch, cuBLAS). Use `triton.testing` for accurate GPU timing.

**Key Principles:**
- Validate correctness before benchmarking
- Use appropriate metrics (TFLOPS for compute-bound, GB/s for memory-bound)
- Sweep realistic shapes (exponential ranges)
- Compare against baselines
- Use CUDA graphs for reduced overhead

---

### Timing Methodology

#### Using triton.testing.do_bench_cudagraph

```python
import triton.testing

# Basic timing
fn = lambda: my_kernel(input)
ms = triton.testing.do_bench_cudagraph(fn)

# With quantiles (median, 20th percentile, 80th percentile)
ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])
```

**Advantages:**
- Automatic warmup handling
- CUDA graph capture reduces launch overhead
- Quantile-based measurements capture variance
- No manual synchronization needed

#### When to Use Standard do_bench

For simpler kernels without CUDA graph support:

```python
ms = triton.testing.do_bench(fn)
```

---

### Performance Metrics

#### TFLOPS (Compute-Bound Kernels)

Use for matrix multiplication, convolution, attention:

```python
def benchmark_matmul(M, N, K, dtype):
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    fn = lambda: matmul_kernel(A, B)
    ms = triton.testing.do_bench_cudagraph(fn)

    # FLOPS = 2 * M * N * K (one multiply-add per output element)
    flops = 2 * M * N * K
    tflops = flops * 1e-12 / (ms * 1e-3)

    return tflops
```

**Formula**: `TFLOPS = (FLOPs * 1e-12) / (time_ms * 1e-3)`

#### GB/s (Memory-Bound Kernels)

Use for element-wise operations, normalization, activation functions:

```python
def benchmark_rmsnorm(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")

    fn = lambda: rms_norm_kernel(x, weight)
    ms = triton.testing.do_bench_cudagraph(fn)

    # Bytes = read input + read weight + write output
    bytes_per_element = x.element_size()
    total_bytes = (M * N + N + M * N) * bytes_per_element
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s
```

**Formula**: `GB/s = (total_bytes * 1e-9) / (time_ms * 1e-3)`

---

### Shape Sweeps and Configuration

#### Using triton.testing.Benchmark

```python
import triton.testing

def create_benchmark_config(dtype):
    return triton.testing.Benchmark(
        x_names=["M", "N", "K"],                    # Parameter names
        x_vals=[2**i for i in range(10, 15)],      # 1024 to 16384
        line_arg="implementation",                  # Comparison axis
        line_vals=["cutile", "pytorch"],            # Implementations
        line_names=["CuTile", "PyTorch"],           # Display names
        styles=[("orange", "-"), ("green", "-")],   # Plot styles
        xlabel="M/N/K",
        ylabel="TFLOPS",
        plot_name=f"matmul-{dtype}",
        args={"dtype": dtype},                      # Fixed parameters
    )

@triton.testing.perf_report([
    create_benchmark_config(torch.float16),
    create_benchmark_config(torch.float32),
])
def benchmark_matmul(M, N, K, implementation, dtype):
    """Benchmark matrix multiplication."""
    A = torch.randn(M, K, dtype=dtype, device="cuda")
    B = torch.randn(K, N, dtype=dtype, device="cuda")

    if implementation == "cutile":
        fn = lambda: cutile_matmul(A, B)
    else:
        fn = lambda: torch.matmul(A, B)

    # Validate correctness first
    if implementation == "cutile":
        result = fn()
        expected = torch.matmul(A, B)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    ms = triton.testing.do_bench_cudagraph(fn)
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return tflops

# Run benchmark
benchmark_matmul.run(print_data=True)
```

#### Exponential Shape Ranges

```python
# Small to medium (1K to 32K)
x_vals = [2**i for i in range(10, 16)]

# Large shapes (32K to 128K)
x_vals = [2**i for i in range(15, 18)]

# Custom ranges
x_vals = [1024, 2048, 4096, 8192, 16384]
```

---

### Baseline Comparisons

#### PyTorch Reference

```python
def pytorch_reference(A, B):
    """PyTorch baseline for comparison."""
    return torch.matmul(A, B)

# Always validate correctness before benchmarking
cutile_result = cutile_matmul(A, B)
pytorch_result = pytorch_reference(A, B)
torch.testing.assert_close(cutile_result, pytorch_result, rtol=1e-3, atol=1e-3)
```

#### Multi-Backend Comparison

```python
BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")),
    ("pytorch", "PyTorch", ("green", "-")),
]

# Filter based on dtype support
def get_supported_backends(dtype):
    if dtype == torch.float8_e5m2:
        # PyTorch doesn't support FP8
        return [BACKENDS[0]]
    else:
        return BACKENDS
```

---

### Orchestration Scripts

#### run_benchmarks.sh

Create `scripts/run_benchmarks.sh`:

```bash
#!/bin/bash
# Run all benchmark scripts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-.}"

echo "Running cutile kernel benchmarks..."
echo "Output directory: $OUTPUT_DIR"

# Find all benchmark scripts
for file in "$SCRIPT_DIR"/bench_*.py; do
    if [ -f "$file" ]; then
        benchmark_name=$(basename "$file" .py)
        echo "Running $benchmark_name..."

        python3 "$file" 2>&1 | tee "$OUTPUT_DIR/${benchmark_name}_results.txt"

        if [ $? -eq 0 ]; then
            echo "✓ $benchmark_name completed"
        else
            echo "✗ $benchmark_name failed"
        fi
    fi
done

echo "All benchmarks completed"
```

Make executable: `chmod +x scripts/run_benchmarks.sh`

---

### Complete Benchmark Example

```python
# bench_softmax.py
import torch
import triton.testing
import cuda.tile as ct

def pytorch_softmax(x):
    """PyTorch reference implementation."""
    return torch.softmax(x, dim=-1)

def create_benchmark_config(M):
    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 15)],  # 1K to 16K
        line_arg="implementation",
        line_vals=["cutile", "pytorch"],
        line_names=["CuTile Softmax", "PyTorch Softmax"],
        styles=[("orange", "-"), ("green", "-")],
        xlabel="N (sequence length)",
        ylabel="GB/s",
        plot_name=f"softmax-M{M}",
        args={"M": M},
    )

@triton.testing.perf_report([
    create_benchmark_config(M)
    for M in [1024, 4096]
])
def benchmark_softmax(M, N, implementation):
    """Benchmark softmax kernel."""
    x = torch.randn(M, N, dtype=torch.float32, device="cuda")

    if implementation == "cutile":
        fn = lambda: cutile_softmax(x)
    else:
        fn = lambda: pytorch_softmax(x)

    # Validate correctness
    if implementation == "cutile":
        result = fn()
        expected = pytorch_softmax(x)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    # Benchmark
    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory bandwidth: read input + write output
    total_bytes = 2 * x.numel() * x.element_size()
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s

if __name__ == "__main__":
    benchmark_softmax.run(print_data=True)
```

---

## Running Tests and Benchmarks

### Run Tests

```bash
# All tests
pytest tests/

# Only compilation tests (no GPU needed)
pytest tests/ -m compilation

# Only integration tests (requires Blackwell GPU)
pytest tests/ -m integration

# Specific test file
pytest tests/cutile-kernels/test_matmul.py

# Verbose output
pytest tests/ -v

# Skip slow tests
pytest tests/ -m "not slow"
```

### Run Benchmarks

```bash
# Single benchmark
python scripts/bench_matmul.py

# All benchmarks
bash scripts/run_benchmarks.sh

# Save results to directory
bash scripts/run_benchmarks.sh ./results
```

---

## Best Practices Summary

### Testing
- ✅ Always test against PyTorch reference implementation
- ✅ Use appropriate tolerances for each dtype
- ✅ Parametrize shapes (regular + irregular)
- ✅ Test both forward and backward passes
- ✅ Use `torch.autograd.gradcheck()` for numerical validation
- ✅ Test edge cases (empty tensors, NaN, Inf, boundaries)

### Benchmarking
- ✅ Validate correctness before benchmarking
- ✅ Use `triton.testing.do_bench_cudagraph()` for accurate timing
- ✅ Choose appropriate metrics (TFLOPS vs GB/s)
- ✅ Sweep exponential shape ranges
- ✅ Compare against PyTorch baseline
- ✅ Report median with variance (quantiles)

### Organization
- ✅ Separate compilation tests from integration tests
- ✅ Use pytest markers for test categorization
- ✅ Provide reference implementations as static methods
- ✅ Use fixtures for device detection and skipping
- ✅ Document expected behavior for edge cases
