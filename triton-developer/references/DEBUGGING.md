# Triton Debugging and Troubleshooting Guide

Complete reference for debugging, profiling, and troubleshooting Triton GPU kernels.

---

## 1. Environment Variables

Triton exposes a set of environment variables that control compilation behavior,
enable IR inspection, and activate debugging modes. These are the primary knobs
for diagnosing compilation and correctness issues.

### Interpreter Mode

The single most powerful debugging tool for Triton kernels.

| Variable | Value | Description |
|----------|-------|-------------|
| `TRITON_INTERPRET` | `1` | Run kernels on the CPU through a Python interpreter instead of compiling to GPU code. |

Usage:

```bash
TRITON_INTERPRET=1 python my_kernel.py
```

When interpreter mode is active:

- Every `@triton.jit` kernel executes as plain Python/NumPy on the CPU.
- You can set **breakpoints** inside `@triton.jit` functions using `breakpoint()` or
  your IDE debugger. Step through kernel logic line by line.
- All `tl.*` operations are emulated, including loads, stores, masks, and reductions.
- Atomic operations are serialized and deterministic.

Limitations:

- **Single-threaded execution** -- no GPU parallelism. All program IDs run sequentially.
- **Significantly slower** than GPU execution, especially for large inputs.
- Some hardware-specific behavior (e.g., warp-level effects, exact floating-point
  rounding) is not perfectly replicated.
- `tl.inline_asm_elementwise` and other inline assembly is not supported.

Best for:

- Logic bugs and incorrect numerical results.
- Algorithm debugging with breakpoints and print statements.
- Verifying mask logic and boundary handling.
- Stepping through reduction and scan operations.

### Compilation Debugging

| Variable | Value | Description |
|----------|-------|-------------|
| `TRITON_ALWAYS_COMPILE` | `1` | Skip the compilation cache and force a full recompile on every launch. Useful when modifying Triton internals or when the cache is suspected to be stale. |
| `TRITON_PRINT_AUTOTUNING` | `1` | Print the best autotuning configuration selected and the total time spent tuning. Helps verify that autotuning is selecting reasonable parameters. |
| `TRITON_DEBUG` | `1` | Enable runtime debug features. Required for `tl.device_assert` to be active. Without this variable, device asserts are silently compiled out. |
| `TRITON_DISABLE_LINE_INFO` | `1` | Disable generation of line info in compiled code. Reduces compilation overhead slightly. |

### Float32 Precision Control

| Variable | Value | Description |
|----------|-------|-------------|
| `TRITON_F32_DEFAULT` | `ieee` | Use full IEEE fp32 precision in `tl.dot`. Slowest but most accurate. |
| `TRITON_F32_DEFAULT` | `tf32` | Use TF32 precision in `tl.dot` (default). Mantissa truncated to 10 bits for operands. Fast on Ampere and later. |
| `TRITON_F32_DEFAULT` | `tf32x3` | Use 3xTF32 algorithm for `tl.dot`. Computes three TF32 matrix multiplications and sums them for higher precision than single TF32 while remaining faster than full IEEE fp32. |

Usage:

```bash
# Debug numerical differences in matmul
TRITON_F32_DEFAULT=ieee python test_matmul.py   # reference precision
TRITON_F32_DEFAULT=tf32 python test_matmul.py   # default, fast
```

### IR Dumping

Triton compiles kernels through several intermediate representation stages:
TTIR (Triton IR) -> TTGIR (Triton GPU IR) -> LLVM IR -> PTX/AMDGCN -> cubin/hsaco.

| Variable | Value | Description |
|----------|-------|-------------|
| `TRITON_KERNEL_DUMP` | `1` | Enable dumping of all compilation stages. Must be combined with `TRITON_DUMP_DIR`. |
| `TRITON_DUMP_DIR` | `/path/to/dir` | Directory where IR files are written. Created automatically if it does not exist. |
| `MLIR_ENABLE_DUMP` | `1` | Dump MLIR IR before and after each compiler pass. Extremely verbose -- produces large output. Best redirected to a file. |

Usage:

```bash
# Dump all compilation stages
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_ir python my_kernel.py

# Inspect the results
ls /tmp/triton_ir/
# You will see files like: *.ttir, *.ttgir, *.llir, *.ptx

# Full MLIR pass dump (very verbose)
MLIR_ENABLE_DUMP=1 python my_kernel.py 2> /tmp/mlir_dump.log
```

### IR Override (Advanced)

You can modify intermediate IR and feed it back to the compiler. This is useful
for experimenting with optimizations or diagnosing compiler bugs.

Workflow:

```bash
# Step 1: Dump all compilation stages
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_ir python my_kernel.py

# Step 2: Inspect and modify the IR files in /tmp/triton_ir/
#   Edit the .ttir, .ttgir, or .llir files as needed

# Step 3: Run again with override enabled
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=/tmp/triton_ir python my_kernel.py
```

| Variable | Value | Description |
|----------|-------|-------------|
| `TRITON_KERNEL_OVERRIDE` | `1` | Enable IR override mode. The compiler will load IR from the override directory instead of compiling from source. |
| `TRITON_OVERRIDE_DIR` | `/path/to/dir` | Directory from which to load overridden IR files. Typically the same as `TRITON_DUMP_DIR`. |

The compiler matches override files by kernel name and hash. Only stages that
have corresponding files in the override directory are replaced; other stages
compile normally.

### Cache Control

| Variable | Value | Description |
|----------|-------|-------------|
| `TRITON_CACHE_DIR` | `/path/to/dir` | Override the default compilation cache directory (default: `~/.triton/cache`). |
| `TRITON_ALWAYS_COMPILE` | `1` | Bypass the cache entirely and recompile every kernel from scratch. |

---

## 2. In-Kernel Debugging

Triton provides built-in primitives for printing and asserting both at compile
time and at runtime on the GPU.

### Runtime Debugging (Runs on GPU)

```python
@triton.jit
def my_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Print a scalar value from GPU -- one line per program instance
    tl.device_print("pid", pid)

    # Print a tensor value -- prints all elements in the block
    tl.device_print("x_val", x)

    # Assert a condition on GPU -- kernel aborts if any element is false
    # IMPORTANT: requires TRITON_DEBUG=1 environment variable
    tl.device_assert(x > 0, "x must be positive")

    output = x * 2
    tl.store(output_ptr + offsets, output, mask=mask)
```

### Compile-Time Debugging

```python
@triton.jit
def my_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr, DTYPE: tl.constexpr):
    # Print during compilation -- output appears once at compile time, not at runtime
    tl.static_print("BLOCK_SIZE is", BLOCK_SIZE)
    tl.static_print("DTYPE is", DTYPE)

    # Assert during compilation -- compilation fails if condition is false
    tl.static_assert(BLOCK_SIZE >= 16, "BLOCK_SIZE too small")
    tl.static_assert(BLOCK_SIZE & (BLOCK_SIZE - 1) == 0, "BLOCK_SIZE must be power of 2")
```

### Tips for In-Kernel Debugging

- **`tl.device_print` can severely slow down kernel execution.** Every thread in
  every block prints, which serializes output and floods stdout. Use only during
  debugging, never in production.
- **For large grids, guard prints with a program ID check:**
  ```python
  if pid == 0:
      tl.device_print("debug_value", value)
  ```
- **`tl.device_assert` is compiled out by default.** You must set `TRITON_DEBUG=1`
  for device asserts to be active. Without it, they are silently removed during
  compilation and have zero runtime cost.
- **`tl.static_print` is invaluable for autotuning.** It lets you see which
  constexpr values the compiler is specializing on for each configuration.
- **Combine interpreter mode with Python's pdb** for the most interactive
  debugging experience:
  ```python
  @triton.jit
  def my_kernel(...):
      pid = tl.program_id(0)
      if pid == 0:
          breakpoint()  # Only works with TRITON_INTERPRET=1
  ```

---

## 3. Common Errors and Solutions

### Compilation Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `incompatible types: 'fp16' and 'fp32'` | Operands to an operation have mismatched dtypes. Triton does not auto-promote in all contexts. | Add explicit `.to(tl.float32)` or `.to(tl.float16)` cast before the operation. |
| `Shape mismatch in tl.dot: (M, K) and (K2, N)` | The inner dimensions of `tl.dot(a, b)` do not match. `a` has K columns but `b` has K2 rows. | Verify that `a` is shape `(M, K)` and `b` is shape `(K, N)`. Check your load offsets and reshaping. |
| `shared memory size exceeded` | The combined BLOCK_SIZE and `num_stages` require more shared memory than the GPU provides. | Reduce `BLOCK_SIZE`, lower `num_stages`, or reduce `num_warps`. Check `nvidia-smi` for your GPU's shared memory limit. |
| `compilation timeout` | Kernel is too complex for the compiler to optimize within the default time limit. | Simplify the kernel, split into multiple kernels, or increase the compilation timeout. |
| `triton.next_power_of_2 assertion error` | A constexpr parameter that must be a power of 2 is not. | Ensure all `BLOCK_SIZE` and similar parameters are powers of 2 (16, 32, 64, 128, ...). |
| `cannot use operator on constexpr and non-constexpr` | Mixing compile-time and runtime values in an unsupported way. | Separate constexpr logic from runtime tensor operations. Use `tl.constexpr` annotation. |
| `"Trying to load from an invalid memory location"` during compile | Pointer arithmetic error in the kernel source. | Review pointer offset calculations. Check that base pointers and strides are correct. |
| `MLIR verification failed` | Internal IR is malformed, often due to unsupported operations or Triton bugs. | Check Triton GitHub issues. Try simplifying the kernel. Update Triton version. |

### Runtime Errors

| Error / Symptom | Cause | Solution |
|-----------------|-------|----------|
| `CUDA error: an illegal memory access was encountered` | Missing or incorrect `mask` in `tl.load` or `tl.store`. Out-of-bounds threads access invalid memory. | Add proper mask: `mask = offsets < n_elements`. Ensure every `tl.load` and `tl.store` with potentially OOB access has a mask. |
| `CUDA error: misaligned address` | Tensor pointer is not aligned to the required boundary, often because the tensor is a non-contiguous view. | Call `.contiguous()` on PyTorch tensors before passing to the kernel. Check that strides are correct. |
| All zeros in output | Accumulator initialized to zeros and never updated, OR mask is always False, OR store offset is wrong. | Verify accumulator initialization with `tl.zeros`. Check mask logic. Print offsets in interpreter mode. Verify store pointer arithmetic. |
| NaN or Inf in output | fp16 accumulator overflow in matmul, missing max subtraction before `tl.exp` in softmax/attention, or division by zero. | Use fp32 accumulator: `acc = tl.zeros((M, N), dtype=tl.float32)`. Subtract max before exp: `x = x - tl.max(x, axis=1)[:, None]`. Add epsilon to denominators. |
| Wrong results only with autotuning | Race condition when multiple blocks write to the same output location (e.g., atomic adds in split-K matmul). Different configs change the number of blocks, exposing the race. | Add `reset_to_zero=['output_ptr']` in `triton.Config(...)` for any output that is accumulated via atomics. |
| Results differ across runs | Non-deterministic ordering of atomic operations. Floating-point addition is not associative, so different orderings produce different results. | Use lock-based accumulation, fixed ordering, or accept small numerical differences. For exact reproducibility, avoid atomics. |
| Wrong results with fp16 inputs | Intermediate computations overflow fp16 range (max ~65504). | Cast inputs to fp32 early: `x = tl.load(ptr, mask=mask).to(tl.float32)`. Accumulate in fp32. Cast back to fp16 only at the final store. |
| Kernel produces correct results for small inputs but wrong for large | Grid does not cover all elements, or block boundary logic is incorrect. | Verify grid: `grid = (triton.cdiv(n, BLOCK_SIZE),)`. Check that the last partial block is handled correctly via masking. |
| `CUDA error: launch failed` | Kernel requires more resources (registers, shared memory) than the GPU can provide for the requested grid. | Reduce `num_warps`, `num_stages`, or `BLOCK_SIZE`. Profile with Nsight Compute to check resource usage. |
| Silent wrong results (no error) | Pointer arithmetic bug -- kernel reads/writes to wrong locations but within allocated memory. | Test with sequential input values (0, 1, 2, ...) and verify output positions. Use interpreter mode to trace pointer calculations. |

### Autotuning Issues

| Problem | Solution |
|---------|----------|
| Autotuning produces different results per config | Race condition in output accumulation. Add `reset_to_zero=['output_ptr']` to every `triton.Config`. |
| Autotuning is very slow | Reduce the number of configs. Use `prune_configs_by` with a `perf_model` or custom filter function to eliminate clearly bad configs before benchmarking. |
| Best config changes between runs | Increase warmup reps in autotuning. Small kernels may have high timing variance. Consider pinning a config for production. |
| Autotuning crashes on some configs | Some configs may exceed hardware limits. Add a `pre_hook` that validates config feasibility, or use `prune_configs_by` to filter. |
| Config works alone but fails in autotuning | The `reset_to_zero` parameter is not set, so output tensors retain stale values from previous config evaluations. |

---

## 4. Debugging Workflow

A systematic approach to debugging Triton kernels, from quick checks to deep
investigation.

### Step 1: Compilation Check

Verify that the kernel compiles without error before worrying about correctness.

```python
# Compile without running on real data
# Use warmup to trigger compilation with specific dtypes and constexpr values
grid = (1,)
kernel.warmup(
    torch.float32,    # dtype of first pointer arg
    torch.float32,    # dtype of second pointer arg
    32,               # integer arg
    BLOCK_SIZE=128,   # constexpr
    grid=grid,
)
print("Compilation succeeded")
```

If compilation fails, the error message will point to the problematic operation.
Fix compilation errors before proceeding.

### Step 2: Minimal Input Test

Start with the smallest valid input to make debugging tractable.

```python
# Use tiny, human-readable inputs
N = 16  # Small enough to print and inspect
x = torch.arange(N, dtype=torch.float32, device='cuda')
# Or use known values:
# x = torch.ones(N, device='cuda')
# x = torch.eye(N, device='cuda')  # for matrix ops

output = torch.empty_like(x)
my_kernel[(1,)](x, output, N, BLOCK_SIZE=16)
print("Input: ", x)
print("Output:", output)
```

For matrix operations, use identity matrices, all-ones matrices, or sequential
values where you can compute the expected result by hand.

### Step 3: Reference Comparison

Always compare Triton output against a known-correct reference implementation.

```python
def torch_reference(x):
    """Pure PyTorch reference implementation."""
    return x * 2 + 1  # whatever your kernel does

# Run both implementations
ref = torch_reference(x)
out = triton_impl(x)

# Compare with detailed diagnostics
diff = (ref - out).abs()
print(f"Max absolute diff:  {diff.max().item():.2e}")
print(f"Mean absolute diff: {diff.mean().item():.2e}")
print(f"Max relative diff:  {(diff / (ref.abs() + 1e-8)).max().item():.2e}")

# Indices of largest differences
worst = diff.argmax()
print(f"Worst element at index {worst}: ref={ref.flat[worst]:.6f}, out={out.flat[worst]:.6f}")

# Formal assertion with tolerances
torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
```

Choose tolerances based on precision:
- fp32 with IEEE: `atol=1e-5, rtol=1e-5`
- fp32 with TF32: `atol=1e-3, rtol=1e-3`
- fp16/bf16: `atol=1e-2, rtol=1e-2`

### Step 4: Interpreter Mode

If results are wrong, switch to interpreter mode for full Python-level debugging.

```bash
TRITON_INTERPRET=1 python test_my_kernel.py
```

In interpreter mode you can:
- Set breakpoints inside `@triton.jit` functions with `breakpoint()` or your IDE.
- Inspect intermediate tensor values at every step.
- Verify mask logic by printing masks and checking which elements are True.
- Step through reduction loops and check accumulator state.
- Verify pointer arithmetic by printing offsets and comparing to expected positions.

### Step 5: Edge Case Testing

Test the boundaries that most commonly trigger bugs.

```python
test_sizes = [
    1,              # single element
    15,             # not a multiple of BLOCK_SIZE
    16,             # exactly one block
    17,             # one block + 1 element
    128,            # multiple blocks, power of 2
    1000,           # non-power-of-2, multi-block
    100000,         # large input
]
for n in test_sizes:
    x = torch.randn(n, device='cuda')
    ref = torch_reference(x)
    out = triton_impl(x)
    try:
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
        print(f"  N={n:>8d}: PASS")
    except AssertionError as e:
        print(f"  N={n:>8d}: FAIL -- {e}")
```

For 2D kernels, also test:
- Non-square matrices (M != N).
- Tall-skinny and short-wide shapes.
- K dimension equal to 1.
- Dimensions smaller than BLOCK_SIZE.

### Step 6: Profiling

Once correctness is established, profile for performance.

```bash
# Quick timing
python -c "
import torch, triton
from my_kernel import my_triton_kernel
x = torch.randn(1024*1024, device='cuda')
ms = triton.testing.do_bench(lambda: my_triton_kernel(x), warmup=25, rep=100)
print(f'Kernel time: {ms:.3f} ms')
"

# Detailed profiling with Nsight Compute (see Section 5)
ncu --set full python my_kernel.py
```

---

## 5. Profiling

### Triton Built-In Benchmarking

The simplest way to measure kernel performance.

```python
import triton

# Basic timing
ms = triton.testing.do_bench(lambda: my_kernel(x), warmup=25, rep=100)
print(f"Kernel time: {ms:.3f} ms")

# With quantiles for variance analysis
ms, min_ms, max_ms = triton.testing.do_bench(
    lambda: my_kernel(x),
    warmup=25,
    rep=100,
    quantiles=[0.5, 0.2, 0.8],
)
print(f"Median: {ms:.3f} ms, P20: {min_ms:.3f} ms, P80: {max_ms:.3f} ms")
```

### PyTorch Profiler Integration

```python
import torch

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    for _ in range(10):
        output = my_triton_kernel(x)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export Chrome trace for visualization
prof.export_chrome_trace("trace.json")
# Open in chrome://tracing or https://ui.perfetto.dev
```

### NVIDIA Nsight Compute

The most detailed GPU profiling tool. Provides hardware counter data.

```bash
# Full analysis (collects all metrics -- slow but thorough)
ncu --set full --target-processes all -o profile_output python my_kernel.py

# Profile a specific kernel, skip warmup launches
ncu --kernel-name my_kernel_kernel \
    --launch-skip 5 \
    --launch-count 1 \
    --set full \
    -o profile_output \
    python my_kernel.py

# Quick roofline analysis
ncu --set roofline -o roofline_output python my_kernel.py
```

Key metrics to examine:

| Metric | What It Tells You |
|--------|-------------------|
| Achieved Occupancy | Fraction of maximum warps active. Low occupancy may indicate register pressure or shared memory limits. |
| Memory Throughput (%) | How close to peak memory bandwidth. High = memory-bound kernel. |
| Compute Throughput (%) | How close to peak compute. High = compute-bound kernel. |
| L2 Hit Rate | Cache efficiency. Low hit rate = memory access pattern is not cache-friendly. |
| Shared Memory Usage | How much SRAM the kernel uses per block. Near the limit = cannot increase BLOCK_SIZE. |
| Registers Per Thread | High register count reduces occupancy. |

Interpretation:
- **Memory-bound** (high memory %, low compute %): Increase arithmetic intensity,
  use tiling, improve data reuse.
- **Compute-bound** (low memory %, high compute %): Already efficient at data
  movement. Optimize arithmetic or accept current performance.
- **Latency-bound** (both low): Increase occupancy (`num_warps`), add `num_stages`
  for pipelining, increase block size.

---

## 6. Debugging Checklist

Quick-reference checklist to review before concluding that a Triton kernel is correct.

### Memory Access
- [ ] Every `tl.load` that may access out-of-bounds addresses has a `mask` parameter.
- [ ] Every `tl.store` that may write out-of-bounds has a `mask` parameter.
- [ ] The `other` parameter in `tl.load` is set appropriately:
  - `other=0.0` for sum reductions and general arithmetic.
  - `other=-float('inf')` for max reductions (so padding does not affect the max).
  - `other=float('inf')` for min reductions.
- [ ] Pointer offsets account for tensor strides: `ptr + row * stride_row + col * stride_col`.
- [ ] Input tensors are contiguous, or strides are passed and used correctly.

### Numerical Stability
- [ ] Accumulator for `tl.dot` is `tl.float32`, not `tl.float16` or `tl.bfloat16`.
- [ ] In softmax/attention: max is subtracted before `tl.exp` to prevent overflow.
- [ ] Division denominators have an epsilon guard to prevent division by zero.
- [ ] fp16/bf16 intermediate values are checked for overflow (max ~65504 for fp16).

### Grid and Block Configuration
- [ ] Grid dimensions cover all elements: use `triton.cdiv(n, BLOCK_SIZE)`.
- [ ] `BLOCK_SIZE` and related constexpr values are powers of 2.
- [ ] `num_warps` and `num_stages` are reasonable for the target GPU.
- [ ] For 2D kernels, both dimensions of the grid are computed correctly.

### Autotuning
- [ ] `reset_to_zero` is specified for output tensors accumulated via atomics.
- [ ] All configs in `@triton.autotune` are valid (do not exceed hardware limits).
- [ ] `key` parameter in `@triton.autotune` lists all arguments that affect optimal config.

### Testing
- [ ] Tested with `TRITON_INTERPRET=1` for new or modified kernels.
- [ ] Compared against a PyTorch reference implementation.
- [ ] Tested with non-power-of-2 input sizes.
- [ ] Tested with minimum valid input size (e.g., single element, single row).
- [ ] Tested with large inputs to verify grid coverage.
- [ ] Tolerances in `assert_close` are appropriate for the precision used.

### Before Production
- [ ] All `tl.device_print` calls removed.
- [ ] All `breakpoint()` calls removed.
- [ ] `TRITON_DEBUG=1` is not required for correct operation (asserts are for debugging only).
- [ ] Kernel is benchmarked against PyTorch/cuBLAS baseline.
- [ ] Memory usage is validated (no unnecessary allocations in the wrapper).
