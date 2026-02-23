"""
Template for writing Triton kernel tests.

This template demonstrates best practices for testing Triton kernels:
- Compilation tests (no GPU needed with TRITON_INTERPRET)
- Integration tests (requires CUDA GPU)
- Forward and backward pass testing
- Parametrization for comprehensive coverage
- Proper tolerance handling per dtype
"""

import pytest
import torch
import triton
import triton.language as tl


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def hopper_available():
    """Check if Hopper GPU is available (compute capability 9.0+)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 9


@pytest.fixture
def triton_available():
    """Check if triton is installed."""
    try:
        import triton
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_no_hopper():
    """Skip test if Hopper GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    props = torch.cuda.get_device_properties(0)
    if props.major < 9:
        pytest.skip("Hopper+ GPU (SM 9.0+) required")


@pytest.fixture
def device():
    """Get Triton device for tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return triton.runtime.driver.active.get_active_torch_device()


# ============================================================================
# Tolerance Constants
# ============================================================================

TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    torch.float16: {"rtol": 1e-3, "atol": 1e-3},
    torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
}


def get_tolerance(dtype):
    """Get appropriate tolerance for a given dtype."""
    return TOLERANCES.get(dtype, {"rtol": 1e-3, "atol": 1e-3})


# ============================================================================
# Example Kernel Under Test
# ============================================================================

@triton.jit
def scale_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Example kernel: scale input by 2.0"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * 2.0, mask=mask)


def triton_scale(input: torch.Tensor) -> torch.Tensor:
    """Wrapper for Triton scale kernel."""
    output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scale_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output


class ScaleFunction(torch.autograd.Function):
    """Autograd wrapper for scale kernel (forward + backward)."""

    @staticmethod
    def forward(ctx, input):
        return triton_scale(input)

    @staticmethod
    def backward(ctx, grad_output):
        # d/dx(2x) = 2
        return grad_output * 2.0


# ============================================================================
# Compilation Tests
# ============================================================================

class TestScaleKernelCompilation:
    """Compilation tests - verify kernel compiles without errors."""

    @pytest.mark.compilation
    def test_kernel_compiles(self):
        """Test that kernel definition compiles without error."""
        assert callable(scale_kernel), "Kernel should be callable"

    @pytest.mark.compilation
    def test_kernel_jit_attributes(self):
        """Test that JIT attributes are accessible."""
        assert hasattr(scale_kernel, 'fn')


# ============================================================================
# Integration Tests
# ============================================================================

class TestScaleKernelIntegration:
    """Integration tests - requires CUDA GPU."""

    @staticmethod
    def reference(input: torch.Tensor) -> torch.Tensor:
        """PyTorch reference implementation."""
        return input * 2.0

    @pytest.mark.cuda
    @pytest.mark.integration
    @pytest.mark.parametrize("N", [1024, 4096, 16384, 65536])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_correctness(self, skip_if_no_cuda, device, N, dtype):
        """Test kernel correctness with various shapes and dtypes."""
        input = torch.randn(N, dtype=dtype, device=device)

        result = triton_scale(input)
        expected = self.reference(input)

        tol = get_tolerance(dtype)
        torch.testing.assert_close(result, expected, **tol)

    @pytest.mark.cuda
    @pytest.mark.integration
    @pytest.mark.parametrize("N", [1, 7, 127, 997, 1023, 4100, 65537])
    def test_irregular_shapes(self, skip_if_no_cuda, device, N):
        """Test with non-power-of-2 shapes for proper boundary handling."""
        input = torch.randn(N, dtype=torch.float32, device=device)

        result = triton_scale(input)
        expected = self.reference(input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_large_input(self, skip_if_no_cuda, device):
        """Stress test with large input."""
        N = 1 << 24  # 16M elements
        input = torch.randn(N, dtype=torch.float16, device=device)

        result = triton_scale(input)
        expected = self.reference(input)

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


# ============================================================================
# Backward Pass Tests
# ============================================================================

class TestScaleKernelBackward:
    """Tests for backward pass / gradient computation."""

    @pytest.mark.cuda
    @pytest.mark.integration
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_backward_correctness(self, skip_if_no_cuda, device, dtype):
        """Test backward pass produces correct gradients."""
        N = 1024
        input_triton = torch.randn(N, dtype=dtype, device=device, requires_grad=True)
        input_ref = input_triton.clone().detach().requires_grad_(True)

        # Forward pass
        output_triton = ScaleFunction.apply(input_triton)
        output_ref = input_ref * 2.0

        # Validate forward
        tol = get_tolerance(dtype)
        torch.testing.assert_close(output_triton, output_ref, **tol)

        # Backward pass
        grad_output = torch.randn_like(output_triton)
        output_triton.backward(grad_output)
        output_ref.backward(grad_output)

        # Validate gradients
        torch.testing.assert_close(input_triton.grad, input_ref.grad, **tol)

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_backward_gradcheck(self, skip_if_no_cuda, device):
        """
        Numerical gradient validation using finite differences.

        IMPORTANT: Always use float64 for gradcheck — finite differences
        need high precision to produce meaningful results.
        """
        input = torch.randn(128, dtype=torch.float64, device=device, requires_grad=True)

        assert torch.autograd.gradcheck(
            ScaleFunction.apply,
            (input,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True,
        ), "Gradient check failed"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestScaleKernelEdgeCases:
    """Tests for edge cases and special values."""

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_zeros(self, skip_if_no_cuda, device):
        """Test with zero-filled input."""
        input = torch.zeros(1024, dtype=torch.float32, device=device)
        result = triton_scale(input)
        expected = torch.zeros_like(input)
        torch.testing.assert_close(result, expected)

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_large_values(self, skip_if_no_cuda, device):
        """Test with large values (near dtype limits)."""
        input = torch.tensor([1e4, -1e4, 1e-4, -1e-4],
                             dtype=torch.float32, device=device)
        result = triton_scale(input)
        expected = input * 2.0
        torch.testing.assert_close(result, expected)

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_nan_propagation(self, skip_if_no_cuda, device):
        """Test that NaN propagates correctly."""
        input = torch.tensor([1.0, float('nan'), 3.0],
                             dtype=torch.float32, device=device)
        result = triton_scale(input)
        assert torch.isnan(result[1]), "NaN should propagate"
        assert not torch.isnan(result[0]), "Non-NaN should not become NaN"

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_single_element(self, skip_if_no_cuda, device):
        """Test with single element tensor."""
        input = torch.tensor([3.14], dtype=torch.float32, device=device)
        result = triton_scale(input)
        expected = input * 2.0
        torch.testing.assert_close(result, expected)
