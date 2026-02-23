# Testing and Benchmarking Triton Kernels

This document covers testing methodology, correctness verification, tolerance guidelines,
edge case coverage, and performance benchmarking for Triton GPU kernels.

---

## 1. PyTest Setup

### pytest.ini Configuration

```ini
[pytest]
markers =
    cuda: Tests requiring CUDA GPU
    hopper: Tests requiring Hopper GPU (H100+)
    integration: Integration tests running actual kernels
    compilation: Compilation-only tests
    slow: Long-running tests

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short

filterwarnings =
    ignore::DeprecationWarning:triton.*
```

### conftest.py with Fixtures

```python
import pytest
import torch
import triton

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return triton.runtime.driver.active.get_active_torch_device()

@pytest.fixture
def cuda_available():
    return torch.cuda.is_available()

@pytest.fixture
def hopper_available():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 9  # SM 9.0+

@pytest.fixture
def skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

@pytest.fixture
def skip_if_no_hopper():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    props = torch.cuda.get_device_properties(0)
    if props.major < 9:
        pytest.skip("Hopper+ GPU required (SM 9.0+)")

@pytest.fixture(autouse=True)
def clear_cuda_cache():
    yield
    torch.cuda.empty_cache()

@pytest.fixture
def seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    return 42
```

### Running Tests by Marker

```bash
pytest -m cuda             # GPU tests only
pytest -m hopper           # Hopper-specific tests
pytest -m "not slow"       # Skip slow tests
pytest -m compilation      # Compilation tests (no GPU needed)
```

---

## 2. Test Structure

### Compilation Tests (No GPU Needed)

Compilation tests verify that kernels parse, type-check, and compile without errors.
These run on CPU-only CI nodes using the Triton interpreter mode.

```python
import os
import pytest
import torch
import triton
import triton.language as tl
from my_kernels import my_kernel, my_kernel_autotune


class TestMyKernelCompilation:
    """Tests that verify kernel compilation without requiring a GPU."""

    def test_kernel_compiles(self):
        """Verify the kernel compiles without errors using interpreter mode."""
        os.environ["TRITON_INTERPRET"] = "1"
        try:
            x = torch.randn(16, 16)
            output = torch.empty(16, 16)
            grid = (1,)
            my_kernel[grid](x, output, 16, 16, BLOCK_SIZE=16)
        finally:
            os.environ.pop("TRITON_INTERPRET", None)

    def test_constexpr_values(self):
        """Verify that constexpr parameters are validated."""
        os.environ["TRITON_INTERPRET"] = "1"
        try:
            x = torch.randn(16, 16)
            output = torch.empty(16, 16)
            grid = (1,)
            with pytest.raises(Exception):
                my_kernel[grid](x, output, 16, 16, BLOCK_SIZE=17)
        finally:
            os.environ.pop("TRITON_INTERPRET", None)

    def test_kernel_signature(self):
        """Verify the kernel is a proper Triton JIT function."""
        assert hasattr(my_kernel, "run")
        assert isinstance(my_kernel, triton.runtime.JITFunction)

    def test_autotune_configs_valid(self):
        """Verify all autotune configurations are well-formed."""
        if hasattr(my_kernel_autotune, "configs"):
            for config in my_kernel_autotune.configs:
                assert "BLOCK_M" in config.kwargs
                assert "BLOCK_N" in config.kwargs
                for key, val in config.kwargs.items():
                    if "BLOCK" in key:
                        assert val > 0 and (val & (val - 1)) == 0, (
                            f"{key}={val} is not a power of 2"
                        )
```

### Integration Tests (Requires GPU)

Integration tests run actual kernels on the GPU and compare outputs to reference
implementations.

```python
@pytest.mark.cuda
class TestMyKernelIntegration:
    """Integration tests that launch kernels on the GPU."""

    @pytest.mark.parametrize("M,N,K", [
        (128, 128, 128),      # Small, aligned
        (512, 512, 512),      # Medium
        (1024, 1024, 1024),   # Large
        (127, 255, 513),      # Non-power-of-2
        (1, 1024, 1024),      # Edge: single row
        (1024, 1, 1024),      # Edge: single column
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_matmul_correctness(self, M, N, K, dtype, device):
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)

        result = my_triton_matmul(a, b)
        reference = torch.matmul(a, b)

        tols = {
            torch.float16:  (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float32:  (1e-4, 1e-4),
        }
        rtol, atol = tols[dtype]
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_output_shape(self, dtype, device):
        M, N, K = 256, 512, 384
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)
        result = my_triton_matmul(a, b)
        assert result.shape == (M, N)
        assert result.dtype == dtype

    def test_deterministic(self, device):
        a = torch.randn(256, 256, device=device, dtype=torch.float32)
        b = torch.randn(256, 256, device=device, dtype=torch.float32)
        r1 = my_triton_matmul(a, b)
        r2 = my_triton_matmul(a, b)
        torch.testing.assert_close(r1, r2, rtol=0, atol=0)
```

---

## 3. Forward + Backward Testing

### Testing the Forward Pass

Compare Triton kernel output against a known-correct reference (usually PyTorch).
Test multiple shapes, dtypes, and boundary conditions.

```python
@pytest.mark.cuda
class TestRMSNormForward:

    @pytest.mark.parametrize("shape", [
        (1, 128), (32, 256), (64, 1024), (128, 4096),
        (1, 1), (256, 127),  # Non-power-of-2
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_forward_correctness(self, shape, dtype, device):
        x = torch.randn(shape, device=device, dtype=dtype)
        w = torch.randn(shape[-1], device=device, dtype=dtype)
        result = triton_rms_norm(x, w, eps=1e-6)
        # PyTorch reference
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        reference = (x.float() / rms * w.float()).to(dtype)
        tols = {
            torch.float16:  {"rtol": 1e-3, "atol": 1e-3},
            torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
            torch.float32:  {"rtol": 1e-5, "atol": 1e-5},
        }
        torch.testing.assert_close(result, reference, **tols[dtype])
```

### Testing the Backward Pass with gradcheck

Use `torch.autograd.gradcheck` for rigorous gradient verification via finite
differences. This requires **float64** inputs for numerical stability.

```python
@pytest.mark.cuda
class TestRMSNormBackward:

    def test_backward_gradcheck(self, device):
        """IMPORTANT: Use float64 for gradcheck -- lower precision is insufficient."""
        x = torch.randn(32, 128, device=device, dtype=torch.float64, requires_grad=True)
        w = torch.randn(128, device=device, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x, w: triton_rms_norm(x, w, eps=1e-6), (x, w),
            eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_gradgradcheck(self, device):
        """Verify second-order gradients if the kernel supports them."""
        x = torch.randn(16, 64, device=device, dtype=torch.float64, requires_grad=True)
        w = torch.randn(64, device=device, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(
            lambda x, w: triton_rms_norm(x, w, eps=1e-6), (x, w),
            eps=1e-6, atol=1e-4, rtol=1e-3)
```

### Testing Gradient Values Directly

When `gradcheck` is too slow for large tensors or you need dtype-specific tests,
compare gradient tensors directly against a PyTorch reference backward pass.

```python
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_backward_values(self, dtype, device):
        M, N, eps = 128, 256, 1e-6

        # --- Triton path ---
        x_tri = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
        w_tri = torch.randn(N, device=device, dtype=dtype, requires_grad=True)
        out_tri = triton_rms_norm(x_tri, w_tri, eps)
        out_tri.sum().backward()

        # --- Reference path ---
        x_ref = x_tri.detach().clone().requires_grad_(True)
        w_ref = w_tri.detach().clone().requires_grad_(True)
        rms = torch.sqrt(x_ref.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        out_ref = (x_ref.float() / rms * w_ref.float()).to(dtype)
        out_ref.sum().backward()

        tols = {
            torch.float16:  {"rtol": 1e-2, "atol": 1e-2},
            torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
            torch.float32:  {"rtol": 1e-4, "atol": 1e-4},
        }
        torch.testing.assert_close(x_tri.grad, x_ref.grad, **tols[dtype])
        torch.testing.assert_close(w_tri.grad, w_ref.grad, **tols[dtype])
```

---

## 4. Tolerance Guidelines

### Recommended Tolerances by Dtype and Operation Type

| Operation       | float32              | float16             | bfloat16             | Notes                        |
|-----------------|----------------------|---------------------|----------------------|------------------------------|
| Element-wise    | rtol=1e-5, atol=1e-5 | rtol=1e-3, atol=1e-3 | rtol=1e-2, atol=1e-2 | Simple ops (add, mul, relu) |
| MatMul          | rtol=1e-4, atol=1e-4 | rtol=0, atol=1e-2   | rtol=0, atol=5e-2    | Compound rounding error      |
| Softmax         | rtol=1e-5, atol=1e-5 | rtol=1e-3, atol=1e-3 | rtol=1e-2, atol=1e-2 | Numerical stability matters  |
| Attention       | rtol=0, atol=1e-2    | rtol=0, atol=1e-2   | rtol=0, atol=5e-2    | Many ops compound            |
| Normalization   | rtol=1e-5, atol=1e-5 | rtol=1e-3, atol=1e-3 | rtol=1e-2, atol=1e-2 | Reduction sensitivity        |
| gradcheck       | eps=1e-6, rtol=1e-3  | N/A                 | N/A                  | Always use float64           |

### Why Different Tolerances Are Needed

- **float32**: 23-bit mantissa, approximately 7 decimal digits of precision.
- **float16**: 10-bit mantissa, approximately 3 decimal digits.
  Accumulation across many operations amplifies rounding.
- **bfloat16**: 7-bit mantissa, approximately 2 decimal digits.
  Same exponent range as float32, but reduced mantissa means wider tolerances.
- **TF32** (Tensor Float 32): 10-bit mantissa, used automatically by NVIDIA
  tensor cores for float32 matmul on Ampere+. Use
  `torch.backends.cuda.matmul.allow_tf32 = False` for bit-exact float32 comparisons.

### Practical Tolerance Selection

```python
def get_tolerances(dtype, op_type="elementwise"):
    """Return (rtol, atol) for a given dtype and operation type."""
    table = {
        "elementwise": {
            torch.float32: (1e-5, 1e-5), torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
        },
        "matmul": {
            torch.float32: (1e-4, 1e-4), torch.float16: (0, 1e-2),
            torch.bfloat16: (0, 5e-2),
        },
        "attention": {
            torch.float32: (0, 1e-2), torch.float16: (0, 1e-2),
            torch.bfloat16: (0, 5e-2),
        },
        "normalization": {
            torch.float32: (1e-5, 1e-5), torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
        },
    }
    return table.get(op_type, table["elementwise"]).get(dtype, (1e-3, 1e-3))
```

### When to Use `rtol=0`

For matmul and attention, setting `rtol=0` and relying solely on `atol` is
recommended because relative tolerance breaks down when reference values are
near zero. A small absolute error on a near-zero value produces a huge relative
error, causing false test failures.

---

## 5. Edge Case Coverage

Every Triton kernel test suite should cover these edge cases. Many kernel bugs
manifest only at boundary conditions.

```python
@pytest.mark.cuda
class TestEdgeCases:

    # --- Shape edge cases ---

    @pytest.mark.parametrize("shape", [
        (127, 255),    # Non-power-of-2 dimensions
        (513, 1023),   # Just above power-of-2
        (1, 1),        # Single-element tensor
        (1, 4096),     # Single row
        (4096, 1),     # Single column
    ])
    def test_non_standard_shapes(self, shape, device):
        x = torch.randn(shape, device=device, dtype=torch.float32)
        result = my_triton_op(x)
        reference = torch_reference_op(x)
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

    def test_very_large_tensor(self, device):
        """Stress test to verify grid/block calculations at scale."""
        x = torch.randn(8192, 8192, device=device, dtype=torch.float16)
        result = my_triton_op(x)
        reference = torch_reference_op(x)
        torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)

    # --- Memory layout edge cases ---

    def test_non_contiguous_input(self, device):
        x = torch.randn(256, 512, device=device, dtype=torch.float32).t()
        assert not x.is_contiguous()
        result = my_triton_op(x)
        reference = torch_reference_op(x)
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

    def test_stride_not_1(self, device):
        x = torch.randn(512, 512, device=device, dtype=torch.float32)
        x_sliced = x[::2, ::2]  # Every other element
        result = my_triton_op(x_sliced.contiguous())
        reference = torch_reference_op(x_sliced.contiguous())
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

    # --- Special value edge cases ---

    def test_zeros(self, device):
        x = torch.zeros(128, 256, device=device, dtype=torch.float32)
        result = my_triton_op(x)
        reference = torch_reference_op(x)
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

    def test_nan_propagation(self, device):
        x = torch.randn(128, 256, device=device, dtype=torch.float32)
        x[0, 0] = float("nan")
        result = my_triton_op(x)
        assert torch.isnan(result).any(), "NaN should propagate through the kernel"

    def test_inf_handling(self, device):
        x = torch.randn(128, 256, device=device, dtype=torch.float32)
        x[0, 0] = float("inf")
        x[1, 0] = float("-inf")
        result = my_triton_op(x)
        reference = torch_reference_op(x)
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_empty_tensor(self, device):
        x = torch.randn(0, 256, device=device, dtype=torch.float32)
        try:
            result = my_triton_op(x)
            assert result.shape[0] == 0
        except RuntimeError:
            pass  # Acceptable to raise an error for empty input
```

---

## 6. Performance Benchmarking

### Basic Benchmarking with `triton.testing.do_bench`

The `do_bench` function handles GPU warmup, synchronization, and timing. It returns
wall-clock time in milliseconds.

```python
import torch
import triton

# Basic timing
ms = triton.testing.do_bench(lambda: my_kernel(x), warmup=25, rep=100)
print(f"Kernel time: {ms:.3f} ms")

# With quantile reporting (median, 20th percentile, 80th percentile)
ms, min_ms, max_ms = triton.testing.do_bench(
    lambda: my_kernel(x), quantiles=[0.5, 0.2, 0.8])
print(f"Median: {ms:.3f} ms, P20: {min_ms:.3f} ms, P80: {max_ms:.3f} ms")
```

### Computing Performance Metrics

```python
def benchmark_memory_bound(M, N, dtype=torch.float32):
    """Report GB/s for memory-bound kernels (element-wise, softmax, normalization)."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    ms = triton.testing.do_bench(lambda: my_elementwise_kernel(x), warmup=25, rep=100)
    # Total bytes: read input + write output
    gbps = 2 * M * N * element_size * 1e-9 / (ms * 1e-3)
    return gbps

def benchmark_compute_bound(M, N, K, dtype=torch.float16):
    """Report TFLOPS for compute-bound kernels (matmul)."""
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    ms = triton.testing.do_bench(lambda: my_triton_matmul(a, b), warmup=25, rep=100)
    # FLOPs for matmul: 2 * M * N * K (multiply + add per output element)
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops
```

### Full Benchmark with Plots Using `triton.testing.Benchmark`

The `Benchmark` class and `perf_report` decorator automate sweeping over problem
sizes, comparing multiple implementations, and generating plots.

```python
import torch
import triton
from triton.testing import do_bench

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],                                    # Parameter to sweep
        x_vals=[128 * i for i in range(2, 33)],           # Values to sweep over
        line_arg="provider",                               # Selects implementation
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch (cuBLAS)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={"N": 4096, "K": 4096},
    ))
def benchmark_matmul(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == "triton":
        ms, min_ms, max_ms = do_bench(lambda: my_triton_matmul(a, b), quantiles=quantiles)
    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)

benchmark_matmul.run(show_plots=True, print_data=True)
# Save to file: benchmark_matmul.run(save_path="./benchmarks/", print_data=True)
```

### CUDA Graph Benchmarking

CUDA graphs capture GPU operations and replay them with reduced CPU overhead.
Use `do_bench_cudagraph` when launch overhead is significant.

```python
ms = triton.testing.do_bench_cudagraph(lambda: my_kernel(x))

# Compare eager vs graph
ms_eager = triton.testing.do_bench(lambda: my_kernel(x))
ms_graph = triton.testing.do_bench_cudagraph(lambda: my_kernel(x))
print(f"Eager: {ms_eager:.3f} ms, Graph: {ms_graph:.3f} ms, Speedup: {ms_eager/ms_graph:.2f}x")
```

### Benchmarking Best Practices

1. **Always warm up**: The first launch compiles PTX and initializes caches.
   `do_bench` handles this automatically.
2. **Use GPU-side timing**: `torch.cuda.Event` is more accurate than `time.time()`.
3. **Sync before measuring**: Call `torch.cuda.synchronize()` if timing manually.
4. **Disable TF32 for fair comparison**: PyTorch matmul uses TF32 on Ampere+ by default.
5. **Report quantiles**: Median (P50) is more stable than mean. Report P20/P80 for variability.
6. **Test at realistic sizes**: Tiny sizes are misleading because launch overhead dominates.

---

## 7. Complete Test Examples

### Example: RMSNorm Full Test Suite

All testing patterns combined into a complete, runnable test module.

```python
"""
Complete test suite for Triton RMSNorm kernel.

    pytest test_rmsnorm.py -v
    pytest test_rmsnorm.py -v -m cuda
    pytest test_rmsnorm.py -v -m compilation
"""
import os
import pytest
import torch
import triton

def torch_rms_norm(x, weight, eps=1e-6):
    """PyTorch reference implementation of RMSNorm."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)


@pytest.mark.compilation
class TestRMSNormCompilation:

    def test_kernel_is_jit_function(self):
        assert isinstance(rms_norm_kernel, triton.runtime.JITFunction)

    def test_interpreter_mode(self):
        os.environ["TRITON_INTERPRET"] = "1"
        try:
            x = torch.randn(4, 16)
            w = torch.ones(16)
            output = torch.empty_like(x)
            rms_norm_kernel[(4,)](x, w, output, 4, 16, 1e-6, BLOCK_SIZE=16)
        finally:
            os.environ.pop("TRITON_INTERPRET", None)


@pytest.mark.cuda
class TestRMSNormForward:

    @pytest.mark.parametrize("M,N", [
        (1, 128), (32, 256), (64, 1024), (128, 4096),
        (1, 1), (256, 127), (512, 513),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_correctness(self, M, N, dtype, device):
        x = torch.randn(M, N, device=device, dtype=dtype)
        w = torch.randn(N, device=device, dtype=dtype)
        result = triton_rms_norm(x, w, 1e-6)
        reference = torch_rms_norm(x, w, 1e-6)
        tols = {
            torch.float16:  {"rtol": 1e-3, "atol": 1e-3},
            torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
            torch.float32:  {"rtol": 1e-5, "atol": 1e-5},
        }
        torch.testing.assert_close(result, reference, **tols[dtype])

    def test_output_dtype_matches_input(self, device):
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            x = torch.randn(32, 256, device=device, dtype=dtype)
            w = torch.randn(256, device=device, dtype=dtype)
            assert triton_rms_norm(x, w, 1e-6).dtype == dtype

    def test_zeros(self, device):
        x = torch.zeros(32, 256, device=device, dtype=torch.float32)
        w = torch.ones(256, device=device, dtype=torch.float32)
        result = triton_rms_norm(x, w, 1e-6)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-3)


@pytest.mark.cuda
class TestRMSNormBackward:

    def test_gradcheck_float64(self, device):
        x = torch.randn(16, 64, device=device, dtype=torch.float64, requires_grad=True)
        w = torch.randn(64, device=device, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(
            lambda x, w: triton_rms_norm(x, w, 1e-6), (x, w),
            eps=1e-6, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_backward_values(self, dtype, device):
        M, N, eps = 64, 256, 1e-6
        x_tri = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
        w_tri = torch.randn(N, device=device, dtype=dtype, requires_grad=True)
        triton_rms_norm(x_tri, w_tri, eps).sum().backward()

        x_ref = x_tri.detach().clone().requires_grad_(True)
        w_ref = w_tri.detach().clone().requires_grad_(True)
        torch_rms_norm(x_ref, w_ref, eps).sum().backward()

        tol = {"rtol": 1e-2, "atol": 1e-2} if dtype == torch.float16 else {"rtol": 1e-4, "atol": 1e-4}
        torch.testing.assert_close(x_tri.grad, x_ref.grad, **tol)
        torch.testing.assert_close(w_tri.grad, w_ref.grad, **tol)


@pytest.mark.cuda
class TestRMSNormEdgeCases:

    def test_single_element(self, device):
        x = torch.randn(1, 1, device=device, dtype=torch.float32)
        w = torch.randn(1, device=device, dtype=torch.float32)
        torch.testing.assert_close(
            triton_rms_norm(x, w, 1e-6), torch_rms_norm(x, w, 1e-6),
            rtol=1e-5, atol=1e-5)

    def test_large_batch(self, device):
        x = torch.randn(4096, 1024, device=device, dtype=torch.float16)
        w = torch.randn(1024, device=device, dtype=torch.float16)
        torch.testing.assert_close(
            triton_rms_norm(x, w, 1e-6), torch_rms_norm(x, w, 1e-6),
            rtol=1e-3, atol=1e-3)

    def test_nan_propagation(self, device):
        x = torch.randn(32, 256, device=device, dtype=torch.float32)
        w = torch.randn(256, device=device, dtype=torch.float32)
        x[0, 0] = float("nan")
        result = triton_rms_norm(x, w, 1e-6)
        assert torch.isnan(result[0]).any()
```

### Example: Complete Matmul Benchmark

```python
"""Benchmark comparing Triton matmul against cuBLAS. Run: python benchmark_matmul.py"""
import torch
import triton
from triton.testing import do_bench

def run_matmul_benchmark():
    # --- Single-size quick validation ---
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ms_torch = do_bench(lambda: torch.matmul(a, b), warmup=25, rep=100)
    ms_triton = do_bench(lambda: triton_matmul(a, b), warmup=25, rep=100)
    flops = 2 * M * N * K
    print(f"cuBLAS: {ms_torch:.3f} ms = {flops*1e-12/(ms_torch*1e-3):.1f} TFLOPS")
    print(f"Triton: {ms_triton:.3f} ms = {flops*1e-12/(ms_triton*1e-3):.1f} TFLOPS")

    # --- Square matrix sweep ---
    print(f"{'Size':>8}  {'Torch (ms)':>12}  {'Triton (ms)':>12}  {'Speedup':>8}")
    for size in [256, 512, 1024, 2048, 4096, 8192]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        ms_t = do_bench(lambda: torch.matmul(a, b))
        ms_tr = do_bench(lambda: triton_matmul(a, b))
        print(f"{size:>8}  {ms_t:>12.3f}  {ms_tr:>12.3f}  {ms_t/ms_tr:>8.3f}x")

if __name__ == "__main__":
    run_matmul_benchmark()
```

### Example: Roofline Analysis

```python
def roofline_analysis(M, N, K, ms, dtype=torch.float16):
    """Determine whether a kernel is memory-bound or compute-bound."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    flops = 2 * M * N * K
    bytes_loaded = (M * K + K * N) * element_size
    arithmetic_intensity = flops / bytes_loaded  # FLOPs/byte
    # Machine limits (example: H100 SXM)
    peak_tflops, peak_bw = 989.5, 3.35e3  # FP16 TFLOPS, GB/s HBM
    ridge_point = peak_tflops * 1e3 / peak_bw  # FLOPs/byte

    achieved_tflops = flops * 1e-12 / (ms * 1e-3)
    print(f"Intensity: {arithmetic_intensity:.1f} FLOPs/byte, Ridge: {ridge_point:.1f}")
    print(f"Achieved: {achieved_tflops:.1f} TFLOPS")
    print("MEMORY-BOUND" if arithmetic_intensity < ridge_point else "COMPUTE-BOUND")
```
