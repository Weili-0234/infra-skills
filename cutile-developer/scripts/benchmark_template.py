# SPDX-FileCopyrightText: Copyright (c) 2025 cutile-developer
# SPDX-License-Identifier: Apache-2.0

"""
Template for benchmarking cutile kernels.

This template demonstrates best practices for performance benchmarking:
- Using triton.testing for accurate GPU timing
- Comparing against PyTorch baseline
- Computing appropriate performance metrics (TFLOPS or GB/s)
- Shape sweeps with exponential ranges
- Correctness validation before benchmarking
"""

import torch
import triton
import triton.testing
import cuda.tile as ct
from math import ceil

DEVICE = "cuda"

# ============================================================================
# Kernel Implementation
# ============================================================================

@ct.kernel
def my_kernel(input, output, TILE: ct.Constant[int]):
    """Example kernel: scale input by 2.0"""
    bid = ct.bid(0)
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
    tile = ct.gather(input, indices)
    result = tile * 2.0
    ct.scatter(output, indices, result)


def my_cutile_kernel(input):
    """Wrapper for cutile kernel."""
    output = torch.empty_like(input)
    TILE = 256
    N = input.numel()
    grid = (ceil(N / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, my_kernel, (input, output, TILE))
    return output


# ============================================================================
# Reference Implementation
# ============================================================================

def pytorch_reference(input):
    """PyTorch reference implementation for baseline comparison."""
    return input * 2.0


# ============================================================================
# Benchmark Configuration
# ============================================================================

def create_benchmark_config(dtype):
    """
    Create benchmark configuration for triton.testing.

    Parameters:
    - x_names: Parameter names for sweep
    - x_vals: Parameter values (exponential range)
    - line_arg: Variable to compare (e.g., implementation, backend)
    - line_vals: Values for line_arg
    - line_names: Display names for each line
    - styles: Plot styles (color, linestyle)
    - ylabel: Performance metric (TFLOPS or GB/s)
    """
    return triton.testing.Benchmark(
        x_names=["N"],                              # Parameter names
        x_vals=[2**i for i in range(10, 16)],      # 1K to 32K (exponential)
        line_arg="implementation",                  # Comparison axis
        line_vals=["cutile", "pytorch"],            # Implementations to compare
        line_names=["CuTile", "PyTorch"],           # Display names
        styles=[("orange", "-"), ("green", "-")],   # Plot styles
        xlabel="N (tensor size)",
        ylabel="GB/s",                              # Memory bandwidth metric
        plot_name=f"my-kernel-{dtype}-GBps",
        args={"dtype": dtype},                      # Fixed parameters
    )


# ============================================================================
# Benchmark Function
# ============================================================================

@triton.testing.perf_report([
    create_benchmark_config(dtype)
    for dtype in [torch.float16, torch.float32]  # Benchmark multiple dtypes
])
def benchmark_my_kernel(N, implementation, dtype):
    """
    Benchmark kernel performance.

    Args:
        N: Tensor size
        implementation: "cutile" or "pytorch"
        dtype: Data type (torch.float16, torch.float32, etc.)

    Returns:
        Performance metric (GB/s in this case)
    """
    # Create test data
    input = torch.randn(N, dtype=dtype, device=DEVICE)

    # Select implementation
    if implementation == "cutile":
        fn = lambda: my_cutile_kernel(input)
    else:
        fn = lambda: pytorch_reference(input)

    # Validate correctness before benchmarking
    if implementation == "cutile":
        result = fn()
        expected = pytorch_reference(input)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    # Benchmark using CUDA graphs (reduces kernel launch overhead)
    ms = triton.testing.do_bench_cudagraph(fn)

    # Compute performance metric
    # For memory-bound kernels: use GB/s
    total_bytes = 2 * input.numel() * input.element_size()  # Read + write
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


# ============================================================================
# Alternative: TFLOPS Benchmark (for compute-bound kernels)
# ============================================================================

def benchmark_matmul_example():
    """
    Example of compute-bound benchmark using TFLOPS.

    For matrix multiplication: C = A @ B
    FLOPs = 2 * M * N * K (one multiply-add per output element)
    """

    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[2**i for i in range(10, 14)],   # 1K to 8K
            line_arg="implementation",
            line_vals=["cutile", "pytorch"],
            line_names=["CuTile MatMul", "PyTorch MatMul"],
            styles=[("orange", "-"), ("green", "-")],
            xlabel="M/N/K",
            ylabel="TFLOPS",
            plot_name="matmul-performance-TFLOPS",
            args={},
        )
    ])
    def benchmark_matmul(M, N, K, implementation):
        A = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
        B = torch.randn(K, N, dtype=torch.float16, device=DEVICE)

        if implementation == "cutile":
            fn = lambda: cutile_matmul(A, B)  # Your cutile matmul
        else:
            fn = lambda: torch.matmul(A, B)

        # Validate correctness
        if implementation == "cutile":
            result = fn()
            expected = torch.matmul(A, B)
            torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        # Benchmark
        ms = triton.testing.do_bench_cudagraph(fn)

        # TFLOPS = (FLOPs * 1e-12) / (time_ms * 1e-3)
        flops = 2 * M * N * K
        tflops = flops * 1e-12 / (ms * 1e-3)

        return tflops

    # Run benchmark
    benchmark_matmul.run(print_data=True)


# ============================================================================
# Quantile-Based Measurement (captures variance)
# ============================================================================

def benchmark_with_quantiles():
    """
    Example of using quantiles to capture measurement variance.

    Quantiles: [0.5, 0.2, 0.8]
    - 0.5: median (typical performance)
    - 0.2: 20th percentile (best case)
    - 0.8: 80th percentile (worst case)
    """
    input = torch.randn(4096, dtype=torch.float32, device=DEVICE)
    fn = lambda: my_cutile_kernel(input)

    # Returns: median, min (20th %ile), max (80th %ile)
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])

    # Compute metrics for all quantiles
    def compute_gbps(ms_val):
        total_bytes = 2 * input.numel() * input.element_size()
        return total_bytes * 1e-9 / (ms_val * 1e-3)

    median_gbps = compute_gbps(ms)
    best_gbps = compute_gbps(min_ms)
    worst_gbps = compute_gbps(max_ms)

    print(f"Performance: {median_gbps:.2f} GB/s (median)")
    print(f"Range: {worst_gbps:.2f} - {best_gbps:.2f} GB/s")


# ============================================================================
# Multi-Parameter Sweep
# ============================================================================

def benchmark_multi_parameter():
    """
    Example of sweeping multiple parameters.

    Useful for exploring:
    - Different tile sizes
    - Different precision modes
    - Different algorithm variants
    """

    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(10, 16)],
            line_arg="config",
            line_vals=list(range(4)),
            line_names=[
                "CuTile FP16",
                "CuTile FP32",
                "PyTorch FP16",
                "PyTorch FP32",
            ],
            styles=[
                ("orange", "-"),
                ("orange", "--"),
                ("green", "-"),
                ("green", "--"),
            ],
            xlabel="N",
            ylabel="GB/s",
            plot_name="multi-config-comparison",
            args={},
        )
    ])
    def benchmark(N, config):
        # Configuration mapping
        configs = [
            ("cutile", torch.float16),
            ("cutile", torch.float32),
            ("pytorch", torch.float16),
            ("pytorch", torch.float32),
        ]
        implementation, dtype = configs[config]

        input = torch.randn(N, dtype=dtype, device=DEVICE)

        if implementation == "cutile":
            fn = lambda: my_cutile_kernel(input)
        else:
            fn = lambda: pytorch_reference(input)

        ms = triton.testing.do_bench_cudagraph(fn)
        total_bytes = 2 * input.numel() * input.element_size()
        return total_bytes * 1e-9 / (ms * 1e-3)

    benchmark.run(print_data=True)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running Benchmark: My Kernel")
    print("=" * 60)

    # Run main benchmark
    benchmark_my_kernel.run(print_data=True)

    print("\n" + "=" * 60)
    print("Additional Examples")
    print("=" * 60)

    # Uncomment to run other examples:
    # benchmark_with_quantiles()
    # benchmark_multi_parameter()
    # benchmark_matmul_example()
