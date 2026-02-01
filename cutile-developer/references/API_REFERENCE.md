# cutile API Reference

Complete API documentation for NVIDIA cutile (CUDA Tile) Python DSL.

## Table of Contents

1. [Type System](#type-system)
2. [Decorators](#decorators)
3. [Memory Operations](#memory-operations)
4. [Compute Operations](#compute-operations)
5. [Tile Creation & Manipulation](#tile-creation--manipulation)
6. [Utilities](#utilities)
7. [Enumerations](#enumerations)

---

## Type System

### Array

Global memory arrays compatible with DLPack and CUDA Array Interface (PyTorch, CuPy).

```python
class Array:
    """A global array stored in GPU memory (HBM)."""

    @property
    def dtype -> DType
        """Data type of array elements."""

    @property
    def shape -> tuple[int, ...]
        """Number of elements in each dimension."""

    @property
    def strides -> tuple[int, ...]
        """Number of elements to step in each dimension."""

    @property
    def ndim -> int
        """Number of dimensions."""

    def slice(axis: int, start: int | Tile, stop: int | Tile) -> Array
        """Create view sliced along single axis.

        Args:
            axis: Dimension to slice (constant, supports negative indexing)
            start: Starting index (inclusive, can be dynamic)
            stop: Ending index (exclusive, can be dynamic)

        Returns:
            View of original array (no copy)

        Example:
            >>> segment = A.slice(axis=0, start=offset, stop=offset + length)
            >>> tile = ct.load(segment, (0, 0), shape=(M, N))
        """
```

**Key points:**
- Arrays can be used in both host code and tile code
- Can be kernel parameters
- Copying an array does not copy underlying data
- Compatible with PyTorch tensors, CuPy arrays

### Tile

Immutable multidimensional collection of values local to a block.

```python
class Tile:
    """Tile array - immutable, block-local data."""

    @property
    def dtype -> DType
        """Data type of tile elements."""

    @property
    def shape -> tuple[int, ...]
        """Number of elements in each dimension (all must be powers of 2)."""

    @property
    def ndim -> int
        """Number of dimensions."""

    def item() -> Tile
        """Extract scalar (0D Tile) from single-element tile.

        Returns:
            0D Tile usable as scalar

        Example:
            >>> tx = ct.full((1,), 0, dtype=ct.int32)
            >>> x = tx.item()  # Use as scalar
            >>> ty = ct.load(array, (0, x), shape=(4, 4))
        """

    def extract(index: tuple[int, ...], shape: tuple[int, ...]) -> Tile
        """Extract sub-tile from tile.

        Args:
            index: Starting index for extraction
            shape: Shape of sub-tile to extract

        Example:
            >>> sub = tile.extract((0, 0), (16, 16))
        """

    def reshape(shape: tuple[int, ...]) -> Tile
        """Reshape tile (must preserve total elements).

        Example:
            >>> reshaped = tile.reshape((64, 16))  # From (32, 32)
        """

    def permute(axes: tuple[int, ...]) -> Tile
        """Permute dimensions.

        Example:
            >>> transposed = tile.permute((1, 0))  # Swap dims
        """

    def transpose(axis0: int = None, axis1: int = None) -> Tile
        """Transpose two axes (defaults to last two).

        Example:
            >>> t = tile.transpose()  # Swap last two dims
            >>> t = tile.transpose(0, 2)  # Swap dims 0 and 2
        """

    def astype(dtype: DType) -> Tile
        """Convert tile to different data type.

        Example:
            >>> fp32_tile = fp16_tile.astype(ct.float32)
        """

    # Arithmetic operators
    def __add__(other) -> Tile  # tile + other
    def __sub__(other) -> Tile  # tile - other
    def __mul__(other) -> Tile  # tile * other
    def __truediv__(other) -> Tile  # tile / other
    def __floordiv__(other) -> Tile  # tile // other
    def __mod__(other) -> Tile  # tile % other
    def __pow__(other) -> Tile  # tile ** other

    # Bitwise operators
    def __and__(other) -> Tile  # tile & other
    def __or__(other) -> Tile  # tile | other
    def __xor__(other) -> Tile  # tile ^ other

    # Comparison operators
    def __eq__(other) -> Tile  # tile == other
    def __ne__(other) -> Tile  # tile != other
    def __lt__(other) -> Tile  # tile < other
    def __le__(other) -> Tile  # tile <= other
    def __gt__(other) -> Tile  # tile > other
    def __ge__(other) -> Tile  # tile >= other

    # Indexing
    def __getitem__(index) -> Tile  # Syntax sugar for expand_dim
```

**Critical constraints:**
- **All tile dimensions must be powers of 2** (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
- Tiles are immutable (operations return new tiles)
- Cannot be used in host code (tile code only)
- Cannot be kernel parameters

### Constant[T]

Type hint for compile-time constant embedding.

```python
from typing import Annotated

Constant = Annotated[T, ConstantAnnotation()]

# Common usage
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]
```

**Usage in kernels:**
```python
@ct.kernel
def my_kernel(array, TILE_SIZE: ConstInt, CAUSAL: ConstBool, SCALE: ConstFloat):
    # TILE_SIZE, CAUSAL, SCALE are embedded at compile time
    pass
```

**Why use Constant:**
- Enables loop unrolling
- Allows conditional compilation
- Required for tile shape parameters

---

## Decorators

### @ct.kernel

Mark a function as a GPU kernel entry point.

```python
@ct.kernel(num_ctas: int | ByTarget = None, occupancy: int = None)
def kernel_name(arrays, constants):
    """Kernel implementation."""
    pass
```

**Parameters:**
- `num_ctas`: Number of CTAs (Cooperative Thread Arrays) to launch per SM
  - Integer: Fixed number for all GPUs
  - `ct.ByTarget(sm_100=2, sm_90=1)`: GPU-specific configuration
- `occupancy`: Number of thread blocks per SM (1-4 typical)

**Examples:**
```python
# Basic kernel
@ct.kernel
def simple_kernel(a, b):
    pass

# With hints
@ct.kernel(num_ctas=2, occupancy=2)
def optimized_kernel(a, b):
    pass

# GPU-specific hints
@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=4))
def adaptive_kernel(a, b):
    pass
```

### @ct.function

Mark a function as a reusable device function.

```python
@ct.function
def helper(tile: Tile, scale: float) -> Tile:
    """Reusable tile operation."""
    return tile * scale + 1.0
```

**Usage:**
```python
@ct.kernel
def my_kernel(array, SCALE: ConstFloat):
    tile = ct.load(array, (0, 0), (16, 16))
    result = helper(tile, SCALE)  # Call device function
    ct.store(array, (0, 0), result)
```

---

## Memory Operations

### ct.load

Load a tile from global memory.

```python
ct.load(
    array: Array,
    index: tuple[int | Tile, ...],
    shape: Constant[tuple[int, ...]],
    padding_mode: PaddingMode = PaddingMode.ZERO,
    order: Constant[str | tuple[int, ...]] = "C",
    latency: int = None,
    allow_tma: bool = True
) -> Tile
```

**Parameters:**
- `array`: Source array in global memory
- `index`: Tile indices (block coordinates), can be dynamic
- `shape`: Tile dimensions (**must be powers of 2**, compile-time constant)
- `padding_mode`: How to handle out-of-bounds access
  - `ct.PaddingMode.ZERO`: Fill with zeros (default)
  - `ct.PaddingMode.REPEAT`: Repeat edge values
  - `ct.PaddingMode.EDGE`: Clamp to array bounds
- `order`: Memory layout
  - `"C"`: Row-major (default)
  - `"F"`: Column-major
  - `(0, 2, 1)`: Custom permutation for transposed loads
- `latency`: TMA pipeline hint (2, 3, 4, 10) - higher for prefetching
- `allow_tma`: Enable Tensor Memory Accelerator (default True)

**Examples:**
```python
# Basic load
tile = ct.load(A, index=(bidx, bidy), shape=(16, 16))

# With padding
tile = ct.load(A, index=(bidx, k), shape=(128, 64),
               padding_mode=ct.PaddingMode.ZERO)

# Transposed load (swap last two dims)
tile = ct.load(K, index=(batch, head, 0, k), shape=(1, 1, 64, 128),
               order=(0, 1, 3, 2))

# With latency hint for prefetching
tile = ct.load(V, index=(batch, head, k, 0), shape=(1, 1, 128, 64),
               latency=4)
```

### ct.store

Store a tile to global memory.

```python
ct.store(
    array: Array,
    index: tuple[int | Tile, ...],
    tile: Tile
) -> None
```

**Parameters:**
- `array`: Destination array in global memory
- `index`: Tile indices (block coordinates)
- `tile`: Tile to store

**Example:**
```python
ct.store(C, index=(bidx, bidy), tile=result_tile)
```

**Note:** Store automatically handles tile-to-memory conversion. No need to manually distribute across threads.

### ct.gather

Gather elements from array using index tensor.

```python
ct.gather(
    array: Array,
    indices: Tile | tuple[Tile, ...],
    axis: int = None
) -> Tile
```

**Parameters:**
- `array`: Source array
- `indices`: Index tile(s) for gathering
  - Single tile: 1D indexing
  - Tuple of tiles: Multi-dimensional indexing
- `axis`: Optional axis for 1D gather

**Boundary handling:** Out-of-bounds indices return zero (automatic masking).

**Examples:**
```python
# 1D gather
indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
values = ct.gather(array, indices)

# 2D gather
x = bid_x * TILE_X + ct.arange(TILE_X, dtype=torch.int32)[:, None]
y = bid_y * TILE_Y + ct.arange(TILE_Y, dtype=torch.int32)[None, :]
values = ct.gather(array, (x, y))
```

### ct.scatter

Scatter elements to array using index tensor.

```python
ct.scatter(
    array: Array,
    indices: Tile | tuple[Tile, ...],
    tile: Tile,
    axis: int = None
) -> None
```

**Parameters:**
- `array`: Destination array
- `indices`: Index tile(s) for scattering
- `tile`: Values to scatter
- `axis`: Optional axis for 1D scatter

**Boundary handling:** Out-of-bounds indices are ignored (automatic masking).

**Examples:**
```python
# 1D scatter
indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
ct.scatter(output, indices, result_tile)

# 2D scatter
ct.scatter(output, (x_indices, y_indices), result_tile)
```

### ct.atomic_add

Atomic addition for global reductions.

```python
ct.atomic_add(
    array: Array,
    index: tuple[int | Tile, ...],
    value: Tile
) -> None
```

**Example:**
```python
# Reduce tile to scalar, then atomic add to global sum
local_sum = ct.sum(tile)
ct.atomic_add(output, (0,), local_sum)
```

---

## Compute Operations

### ct.mma

Matrix multiply-accumulate using Tensor Cores.

```python
ct.mma(
    a: Tile,
    b: Tile,
    accumulator: Tile,
    transpose_a: bool = False,
    transpose_b: bool = False
) -> Tile
```

**Operation:** `accumulator += a @ b`

**Parameters:**
- `a`: Left operand tile (M, K)
- `b`: Right operand tile (K, N)
- `accumulator`: Accumulator tile (M, N)
- `transpose_a`: Transpose a before multiplication
- `transpose_b`: Transpose b before multiplication

**Shape constraints:**
- `a.shape[-1]` must equal `b.shape[0]` (K dimension)
- Result shape: `(a.shape[0], b.shape[1])`

**Best practices:**
- Use `float32` accumulator even with `float16` inputs
- K dimension should be multiple of 16 or 32 for Tensor Core efficiency

**Example:**
```python
accumulator = ct.full((128, 128), 0.0, dtype=ct.float32)
for k in range(num_k_tiles):
    a = ct.load(A, (bidx, k), (128, 64))
    b = ct.load(B, (k, bidy), (64, 128))
    accumulator = ct.mma(a, b, accumulator)
```

### Reductions

Reduce along specified axis.

```python
ct.sum(tile: Tile, axis: int = None, keepdims: bool = False) -> Tile
ct.max(tile: Tile, axis: int = None, keepdims: bool = False) -> Tile
ct.min(tile: Tile, axis: int = None, keepdims: bool = False) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis to reduce (None = reduce all)
- `keepdims`: Keep reduced dimension as size 1

**Examples:**
```python
# Row-wise sum
row_sums = ct.sum(tile, axis=-1, keepdims=True)  # Shape: (M, 1)

# Max per row
row_maxes = ct.max(tile, axis=1)  # Shape: (M,)

# Global reduction
total = ct.sum(tile)  # Scalar (0D tile)
```

### Math Functions

```python
ct.exp2(x: Tile, flush_to_zero: bool = False) -> Tile
    """Compute 2^x (faster than exp for GPU)."""

ct.log(x: Tile) -> Tile
    """Natural logarithm."""

ct.sqrt(x: Tile) -> Tile
    """Square root."""

ct.tanh(x: Tile) -> Tile
    """Hyperbolic tangent."""

ct.abs(x: Tile) -> Tile
    """Absolute value."""

ct.ceil(x: Tile) -> Tile
    """Ceiling function."""

ct.floor(x: Tile) -> Tile
    """Floor function."""
```

**Example - exp2 for softmax:**
```python
INV_LOG_2 = 1.0 / math.log(2)

# Convert exp(x) to exp2(x * log2(e))
exp_vals = ct.exp2(x * INV_LOG_2, flush_to_zero=True)
```

### Element-wise Operations

```python
ct.where(condition: Tile, x: Tile | Scalar, y: Tile | Scalar) -> Tile
    """Conditional selection: condition ? x : y"""

ct.max(a: Tile, b: Tile | Scalar) -> Tile
    """Element-wise maximum."""

ct.min(a: Tile, b: Tile | Scalar) -> Tile
    """Element-wise minimum."""

ct.truediv(
    a: Tile,
    b: Tile | Scalar,
    flush_to_zero: bool = False,
    rounding_mode: RoundingMode = RoundingMode.DEFAULT
) -> Tile
    """True division with control over flushing and rounding."""
```

**Examples:**
```python
# Conditional masking
masked = ct.where(mask, value, -np.inf)

# Clamping
clamped = ct.max(ct.min(tile, upper_bound), lower_bound)

# Safe division
normalized = ct.truediv(tile, sum_tile, flush_to_zero=True,
                        rounding_mode=ct.RoundingMode.APPROX)
```

---

## Tile Creation & Manipulation

### Tile Creation

```python
ct.full(
    shape: Constant[tuple[int, ...]],
    value: Scalar,
    dtype: DType
) -> Tile
    """Create tile filled with constant value.

    Example:
        >>> zeros = ct.full((16, 16), 0.0, dtype=ct.float32)
        >>> ones = ct.full((32, 64), 1.0, dtype=ct.float16)
    """

ct.arange(
    size: Constant[int],
    dtype: DType = None
) -> Tile
    """Create 1D tile with range [0, size).

    Example:
        >>> indices = ct.arange(128, dtype=torch.int32)  # [0, 1, ..., 127]
    """
```

### Tile Manipulation

```python
ct.reshape(tile: Tile, shape: Constant[tuple[int, ...]]) -> Tile
    """Reshape tile (total elements must match).

    Example:
        >>> reshaped = ct.reshape(tile, (64, 32))  # From (128, 16)
    """

ct.permute(tile: Tile, axes: Constant[tuple[int, ...]]) -> Tile
    """Permute tile dimensions.

    Example:
        >>> transposed = ct.permute(tile, (1, 0))  # Swap dimensions
        >>> permuted = ct.permute(tile, (2, 0, 1))  # 3D permutation
    """

ct.transpose(tile: Tile, axis0: int = None, axis1: int = None) -> Tile
    """Transpose two axes (defaults to last two).

    Example:
        >>> t = ct.transpose(tile)  # Swap last two dims
        >>> t = ct.transpose(tile, 0, 2)  # Swap dims 0 and 2
    """

ct.astype(tile: Tile, dtype: DType) -> Tile
    """Convert tile data type.

    Example:
        >>> fp32 = ct.astype(fp16_tile, ct.float32)
        >>> tf32 = ct.astype(tile, ct.tfloat32)  # For Tensor Cores
    """
```

---

## Utilities

### Grid & Block Info

```python
ct.bid(axis: int) -> int
    """Get block index along axis (0, 1, or 2).

    Example:
        >>> bidx = ct.bid(0)  # Block ID in X dimension
        >>> bidy = ct.bid(1)  # Block ID in Y dimension
    """

ct.num_blocks(axis: int) -> int
    """Get total number of blocks along axis.

    Example:
        >>> total_blocks = ct.num_blocks(0)
    """

ct.num_tiles(
    array: Array,
    axis: int,
    shape: Constant[tuple[int, ...]],
    order: Constant[str | tuple[int, ...]] = "C"
) -> int
    """Get number of tiles along axis.

    Example:
        >>> # For array (1024, 2048) with tile shape (128, 64):
        >>> num_k_tiles = ct.num_tiles(A, axis=1, shape=(128, 64))
        >>> # Result: ceil(2048 / 64) = 32
    """
```

### Math Utilities

```python
ct.cdiv(a: int, b: int) -> int
    """Ceiling division: ceil(a / b).

    Example:
        >>> grid_x = ct.cdiv(M, tile_m)  # Number of tiles in M dimension
    """
```

### Kernel Launch

```python
ct.launch(
    stream,
    grid: tuple[int, int, int],
    kernel: Callable,
    args: tuple
) -> None
    """Launch kernel on CUDA stream.

    Parameters:
        stream: PyTorch CUDA stream (torch.cuda.current_stream())
        grid: (grid_x, grid_y, grid_z) - number of blocks
        kernel: Kernel function decorated with @ct.kernel
        args: Tuple of arguments to pass to kernel

    Example:
        >>> grid = (math.ceil(M / tm), math.ceil(N / tn), 1)
        >>> ct.launch(torch.cuda.current_stream(), grid, my_kernel,
        ...           (A, B, C, tm, tn, tk))
    """
```

### Experimental: Autotuning

```python
ct_experimental.autotune_launch(
    stream,
    grid_fn: Callable[[Config], tuple[int, int, int]],
    kernel: Callable,
    args_fn: Callable[[Config], tuple],
    hints_fn: Callable[[Config], dict] = None,
    search_space: list[Config] = None
) -> AutotuneResult
    """Launch kernel with automatic performance tuning.

    Example:
        >>> from types import SimpleNamespace
        >>> result = ct_experimental.autotune_launch(
        ...     torch.cuda.current_stream(),
        ...     grid_fn=lambda cfg: (math.ceil(M / cfg.tm), math.ceil(N / cfg.tn), 1),
        ...     kernel=matmul_kernel,
        ...     args_fn=lambda cfg: (A, B, C, cfg.tm, cfg.tn, cfg.tk),
        ...     hints_fn=lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
        ...     search_space=[
        ...         SimpleNamespace(tm=128, tn=256, tk=64, num_ctas=1, occupancy=2),
        ...         SimpleNamespace(tm=64, tn=64, tk=32, num_ctas=2, occupancy=4),
        ...     ]
        ... )
        >>> print(result.tuned_config)  # Best configuration
    """
```

---

## Enumerations

### DType

Data types for arrays and tiles.

```python
ct.float16    # 16-bit floating point
ct.float32    # 32-bit floating point
ct.float64    # 64-bit floating point
ct.bfloat16   # Brain float 16
ct.tfloat32   # TensorFloat-32 (for Tensor Cores)

ct.int8       # 8-bit signed integer
ct.int16      # 16-bit signed integer
ct.int32      # 32-bit signed integer
ct.int64      # 64-bit signed integer

ct.uint8      # 8-bit unsigned integer
ct.uint16     # 16-bit unsigned integer
ct.uint32     # 32-bit unsigned integer
ct.uint64     # 64-bit unsigned integer
```

### PaddingMode

Out-of-bounds handling for ct.load().

```python
ct.PaddingMode.ZERO     # Fill with zeros (default)
ct.PaddingMode.REPEAT   # Repeat edge values
ct.PaddingMode.EDGE     # Clamp to array bounds
```

### RoundingMode

Rounding modes for ct.truediv().

```python
ct.RoundingMode.DEFAULT  # Default rounding
ct.RoundingMode.APPROX   # Approximate rounding (faster)
```

### MemoryOrder

Memory ordering for advanced synchronization.

```python
ct.MemoryOrder.RELAXED
ct.MemoryOrder.ACQUIRE
ct.MemoryOrder.RELEASE
ct.MemoryOrder.ACQ_REL
ct.MemoryOrder.SEQ_CST
```

### MemoryScope

Memory scope for synchronization.

```python
ct.MemoryScope.CTA      # Thread block
ct.MemoryScope.GPU      # Entire GPU
ct.MemoryScope.SYSTEM   # Entire system
```

---

## Type Conversions

### Explicit Casting

```python
# Using astype method
fp32_tile = fp16_tile.astype(ct.float32)

# Using ct.astype function
fp32_tile = ct.astype(fp16_tile, ct.float32)

# TensorFloat-32 for Tensor Cores
tf32_tile = tile.astype(ct.tfloat32)
```

### Automatic Conversions

Tiles automatically broadcast scalars:
```python
tile = tile * 2.0  # Scalar broadcasts to tile shape
tile = tile + 1    # Integer broadcasts
```

---

## Performance Tips

1. **Use float32 accumulators** for mma even with float16 inputs
2. **Align K dimension** to 16 or 32 for Tensor Core efficiency
3. **Use exp2 instead of exp** for ~2x speedup: `ct.exp2(x * INV_LOG_2)`
4. **Add latency hints** for TMA pipeline hiding in memory-bound kernels
5. **Use tfloat32** for float32 data with Tensor Cores
6. **Flush to zero** in divisions for better performance: `flush_to_zero=True`
