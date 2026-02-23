"""
Template for benchmarking Triton kernels.

This template demonstrates best practices for performance benchmarking:
- Using triton.testing for accurate GPU timing
- Comparing against PyTorch baseline
- Computing appropriate performance metrics (TFLOPS or GB/s)
- Shape sweeps with exponential ranges
- Correctness validation before benchmarking
"""

import torch
import triton
import triton.language as tl
import triton.testing

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ============================================================================
# Kernel Implementation
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
    """Wrapper for Triton kernel."""
    output = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scale_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Reference Implementation
# ============================================================================

def pytorch_reference(input: torch.Tensor) -> torch.Tensor:
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
    - x_vals: Parameter values (exponential range recommended)
    - line_arg: Variable to compare (e.g., implementation)
    - line_vals: Values for line_arg
    - line_names: Display names for each line
    - styles: Plot styles (color, linestyle)
    - ylabel: Performance metric (TFLOPS or GB/s)
    """
    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(12, 28, 1)],   # 4K to 128M
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        xlabel="N (tensor size)",
        ylabel="GB/s",
        plot_name=f"scale-kernel-{dtype}-GBps",
        args={"dtype": dtype},
    )


# ============================================================================
# Memory-Bound Benchmark (GB/s)
# ============================================================================

@triton.testing.perf_report([
    create_benchmark_config(dtype)
    for dtype in [torch.float16, torch.float32]
])
def benchmark_scale_kernel(N, provider, dtype):
    """
    Benchmark kernel performance.

    Args:
        N: Tensor size
        provider: "triton" or "torch"
        dtype: Data type

    Returns:
        Performance metric (GB/s)
    """
    input = torch.randn(N, dtype=dtype, device=DEVICE)

    if provider == "triton":
        fn = lambda: triton_scale(input)
    else:
        fn = lambda: pytorch_reference(input)

    # Validate correctness before benchmarking
    if provider == "triton":
        result = fn()
        expected = pytorch_reference(input)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    # Benchmark with quantiles for variance
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # GB/s = total bytes / time
    total_bytes = 2 * input.numel() * input.element_size()  # Read + write
    gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ============================================================================
# Compute-Bound Benchmark (TFLOPS) - MatMul Example
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
            x_vals=[128 * i for i in range(2, 33)],   # 256 to 4096
            line_arg="provider",
            line_vals=["triton", "cublas"],
            line_names=["Triton", "cuBLAS"],
            styles=[("blue", "-"), ("green", "-")],
            xlabel="M/N/K (square matrix)",
            ylabel="TFLOPS",
            plot_name="matmul-performance-TFLOPS",
            args={},
        )
    ])
    def benchmark_matmul(M, N, K, provider):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "cublas":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(a, b), quantiles=quantiles)
        else:
            # Replace with your Triton matmul
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(a, b), quantiles=quantiles)

        # TFLOPS = FLOPs / time
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark_matmul.run(print_data=True, show_plots=True)


# ============================================================================
# CUDA Graph Benchmark (lower overhead)
# ============================================================================

def benchmark_with_cuda_graph():
    """
    Use CUDA graphs for benchmarking when kernel launch overhead matters.

    triton.testing.do_bench_cudagraph reduces launch overhead by
    capturing the kernel into a CUDA graph.
    """
    input = torch.randn(4096, dtype=torch.float32, device=DEVICE)
    fn = lambda: triton_scale(input)

    # Standard benchmark
    ms_standard = triton.testing.do_bench(fn, warmup=25, rep=100)

    # CUDA graph benchmark (lower overhead)
    ms_cudagraph = triton.testing.do_bench_cudagraph(fn)

    print(f"Standard bench: {ms_standard:.3f} ms")
    print(f"CUDA graph:     {ms_cudagraph:.3f} ms")


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
    input = torch.randn(1 << 20, dtype=torch.float32, device=DEVICE)
    fn = lambda: triton_scale(input)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

    def compute_gbps(ms_val):
        total_bytes = 2 * input.numel() * input.element_size()
        return total_bytes * 1e-9 / (ms_val * 1e-3)

    print(f"Performance: {compute_gbps(ms):.2f} GB/s (median)")
    print(f"Range: {compute_gbps(max_ms):.2f} - {compute_gbps(min_ms):.2f} GB/s")


# ============================================================================
# Multi-Dtype Comparison
# ============================================================================

def benchmark_multi_dtype():
    """
    Compare performance across multiple data types.

    Useful for understanding:
    - FP16 vs BF16 vs FP32 throughput
    - Tensor Core utilization (FP16/BF16 can use TC, FP32 uses TF32)
    """

    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(12, 26)],
            line_arg="config",
            line_vals=list(range(6)),
            line_names=[
                "Triton FP16", "Triton BF16", "Triton FP32",
                "PyTorch FP16", "PyTorch BF16", "PyTorch FP32",
            ],
            styles=[
                ("blue", "-"), ("blue", "--"), ("blue", ":"),
                ("green", "-"), ("green", "--"), ("green", ":"),
            ],
            xlabel="N",
            ylabel="GB/s",
            plot_name="multi-dtype-comparison",
            args={},
        )
    ])
    def benchmark(N, config):
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        providers = ["triton", "triton", "triton", "torch", "torch", "torch"]
        dtype = dtypes[config % 3]
        provider = providers[config]

        input = torch.randn(N, dtype=dtype, device=DEVICE)

        if provider == "triton":
            fn = lambda: triton_scale(input)
        else:
            fn = lambda: pytorch_reference(input)

        ms = triton.testing.do_bench(fn)
        total_bytes = 2 * input.numel() * input.element_size()
        return total_bytes * 1e-9 / (ms * 1e-3)

    benchmark.run(print_data=True, show_plots=True)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running Benchmark: Scale Kernel")
    print("=" * 60)

    # Run main benchmark
    benchmark_scale_kernel.run(print_data=True, show_plots=True)

    print("\n" + "=" * 60)
    print("Additional Examples")
    print("=" * 60)

    # Uncomment to run other examples:
    # benchmark_with_quantiles()
    # benchmark_with_cuda_graph()
    # benchmark_matmul_example()
    # benchmark_multi_dtype()
