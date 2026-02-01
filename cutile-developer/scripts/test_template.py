# SPDX-FileCopyrightText: Copyright (c) 2025 cutile-developer
# SPDX-License-Identifier: Apache-2.0

"""
Template for writing cutile kernel tests.

This template demonstrates best practices for testing cutile kernels:
- Compilation tests (no GPU needed)
- Integration tests (requires Blackwell GPU)
- Forward and backward pass testing
- Parametrization for comprehensive coverage
"""

import pytest
import torch
import cuda.tile as ct
from math import ceil

# ============================================================================
# Fixtures
# ============================================================================

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


# ============================================================================
# Compilation Tests
# ============================================================================

class TestMyKernelCompilation:
    """Compilation tests - no GPU required."""

    @pytest.mark.compilation
    def test_kernel_compiles(self, cutile_available):
        """Test that kernel definition compiles."""
        if not cutile_available:
            pytest.skip("cutile not installed")

        @ct.kernel
        def my_kernel(input, output, TILE: ct.Constant[int]):
            """Example kernel: scale input by 2.0"""
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
            tile = ct.gather(input, indices)
            result = tile * 2.0
            ct.scatter(output, indices, result)

        assert callable(my_kernel), "Kernel should be callable"


# ============================================================================
# Integration Tests
# ============================================================================

class TestMyKernelIntegration:
    """Integration tests - requires Blackwell GPU."""

    @staticmethod
    def reference(input):
        """
        Reference implementation using vanilla PyTorch.

        Best practices:
        1. Use stable, well-tested PyTorch functions
        2. Match exact semantics of custom kernel
        3. Handle edge cases (NaN, infinity, dtype casting)
        4. Keep simple and readable
        """
        return input * 2.0

    @pytest.mark.cuda
    @pytest.mark.integration
    @pytest.mark.parametrize("N", [1024, 4096, 16384])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_correctness(self, skip_if_no_blackwell, skip_if_no_cutile, device, N, dtype):
        """Test kernel correctness with various shapes and dtypes."""
        # Create test data
        input = torch.randn(N, dtype=dtype, device=device)
        output = torch.empty_like(input)

        # Define kernel
        @ct.kernel
        def my_kernel(input, output, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
            tile = ct.gather(input, indices)
            result = tile * 2.0
            ct.scatter(output, indices, result)

        # Launch kernel
        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, my_kernel, (input, output, TILE))

        # Validate against reference
        expected = self.reference(input)

        # Use dtype-appropriate tolerance
        if dtype == torch.float16:
            torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.cuda
    @pytest.mark.integration
    @pytest.mark.parametrize("N", [997, 1023, 4100])  # Irregular shapes
    def test_irregular_shapes(self, skip_if_no_blackwell, skip_if_no_cutile, device, N):
        """Test with non-power-of-2 shapes to ensure proper boundary handling."""
        input = torch.randn(N, dtype=torch.float32, device=device)
        output = torch.empty_like(input)

        @ct.kernel
        def my_kernel(input, output, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
            tile = ct.gather(input, indices)  # Auto-handles boundaries
            result = tile * 2.0
            ct.scatter(output, indices, result)  # Auto-masks out-of-bounds

        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, my_kernel, (input, output, TILE))

        expected = self.reference(input)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)


# ============================================================================
# Backward Pass Testing (if kernel has autograd.Function wrapper)
# ============================================================================

class MyKernelFunction(torch.autograd.Function):
    """Example autograd.Function wrapper for backward pass support."""

    @staticmethod
    def forward(ctx, input):
        """Forward pass."""
        output = torch.empty_like(input)

        @ct.kernel
        def my_kernel(input, output, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
            tile = ct.gather(input, indices)
            result = tile * 2.0
            ct.scatter(output, indices, result)

        TILE = 256
        N = input.numel()
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, my_kernel, (input, output, TILE))

        # Save tensors for backward (if needed)
        # ctx.save_for_backward(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        # For this simple example: d/dx(2x) = 2
        grad_input = grad_output * 2.0
        return grad_input


class TestMyKernelBackward:
    """Tests for backward pass."""

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_backward(self, skip_if_no_blackwell, skip_if_no_cutile, device):
        """Test backward pass correctness."""
        N = 1024
        input = torch.randn(N, dtype=torch.float32, device=device, requires_grad=True)
        input_ref = input.clone().detach().requires_grad_(True)

        # Forward pass
        output = MyKernelFunction.apply(input)
        output_ref = input_ref * 2.0

        # Validate forward
        torch.testing.assert_close(output, output_ref, rtol=1e-5, atol=1e-8)

        # Backward pass
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        output_ref.backward(grad_output)

        # Validate backward (gradients)
        torch.testing.assert_close(input.grad, input_ref.grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    @pytest.mark.integration
    def test_backward_gradcheck(self, skip_if_no_blackwell, skip_if_no_cutile, device):
        """Numerical gradient validation using finite differences."""
        # Use small input and float64 for gradcheck
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


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestMyKernelEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.cuda
    @pytest.mark.integration
    @pytest.mark.parametrize("N", [1, 16, 1023, 1025])
    def test_boundary_conditions(self, skip_if_no_blackwell, skip_if_no_cutile, device, N):
        """Test edge cases and boundaries."""
        input = torch.randn(N, dtype=torch.float32, device=device)
        output = torch.empty_like(input)

        @ct.kernel
        def my_kernel(input, output, TILE: ct.Constant[int]):
            bid = ct.bid(0)
            indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
            tile = ct.gather(input, indices)
            result = tile * 2.0
            ct.scatter(output, indices, result)

        TILE = 256
        grid = (ceil(N / TILE), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, my_kernel, (input, output, TILE))

        expected = input * 2.0
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-8)
