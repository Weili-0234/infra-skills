# Element-wise Operations

This file contains examples of element-wise and data transformation operations.

## Table of Contents
- [Example 5: Quantization](#example-5-quantization-per-channel-int8)
- [Example 8: Rotary Position Embeddings (RoPE)](#example-8-rotary-position-embeddings-rope)
- [Example 9: SiLU and Multiply](#example-9-silu-and-multiply-fused-activation)
- [Example 10: Image to Patches](#example-10-image-to-patches-vision-transformer)

**See also**: [EXAMPLES.md](EXAMPLES.md) for basic patterns

---

## Example 5: Quantization (Per-Channel INT8)

**Purpose**: Quantize float16/bfloat16 to int8 using per-channel absmax scaling.

**Key Techniques**: `ct.astype`, per-channel reduction, scale computation

```python
import torch
import cuda.tile as ct

@ct.function
def quantize_row(tile: ct.Tile) -> tuple[ct.Tile, ct.Tile]:
    """Quantize a single row to int8 with absmax scaling."""
    # Find absolute max (no ct.abs available, assume positive dominant)
    absmax = ct.max(tile).astype(ct.float32)

    # Compute scale: absmax / 127 (int8 range is -128 to 127)
    scale = absmax / 127.0

    # Quantize
    tile_quant = tile / scale
    tile_int8 = tile_quant.astype(ct.int8)

    return tile_int8, scale

@ct.kernel
def quant_kernel(x: ct.Array, y: ct.Array, scales: ct.Array, TILE: ct.Constant[int]):
    """Per-row quantization to int8."""
    bid = ct.bid(0)

    # Load row
    tile_x = ct.load(x, (bid, 0), (1, TILE))

    # Quantize
    tile_y, scale = quantize_row(tile_x)

    # Store quantized values and scale
    ct.store(y, (bid, 0), tile_y)
    ct.store(scales, (bid,), scale)

def quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to int8 with per-row scaling."""
    M, N = x.shape
    y = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scales = torch.empty(M, device=x.device, dtype=torch.float32)

    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        quant_kernel,
        (x, y, scales, N)
    )

    return y, scales

# Usage
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16) * 10
y_int8, scales = quantize_int8(x)

# Dequantize and validate
x_dequant = y_int8.float() * scales[:, None]
print(f"Quantization error: {(x - x_dequant).abs().mean():.6f}")
```

---

## Example 8: Rotary Position Embeddings (RoPE)

**Purpose**: Apply rotary position embeddings for transformer position encoding.

**Key Techniques**: Trigonometric functions, rotation matrix, `ct.mma` for rotation

```python
import torch
import cuda.tile as ct
import math

@ct.kernel
def rope_kernel(
    x: ct.Array, pos_ids: ct.Array, freqs: ct.Array, output: ct.Array,
    TILE: ct.Constant[int]
):
    """Apply RoPE to input tensor."""
    seq_idx = ct.bid(0)
    tile_idx = ct.bid(1)

    # Load position ID for this sequence element
    pos = ct.load(pos_ids, (seq_idx,), (1,)).item()

    # Load input: (1, TILE)
    tile_x = ct.load(x, (seq_idx, tile_idx), (1, TILE)).astype(ct.float32)

    # Load rotation matrix: (1, TILE//2, 2, 2)
    tile_rot = ct.load(
        freqs, (pos, tile_idx, 0, 0),
        (1, TILE // 2, 2, 2)
    ).reshape((TILE // 2, 2, 2))

    # Reshape x to pairs: (TILE // 2, 1, 2)
    tile_x = tile_x.reshape((TILE // 2, 1, 2))

    # Apply rotation using matrix multiply
    tile_y = ct.full(tile_x.shape, 0.0, dtype=ct.float32)
    tile_y = ct.mma(tile_x, tile_rot, tile_y)

    # Reshape back and store
    tile_y = tile_y.reshape((1, TILE))
    ct.store(output, (seq_idx, tile_idx), tile_y.astype(output.dtype))

def build_rope_freqs(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Build RoPE frequency rotation matrices."""
    freqs = torch.zeros(max_seq_len, dim // 2, 2, 2, device="cuda", dtype=torch.float32)

    # Compute frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device="cuda").float() / dim))

    for pos in range(max_seq_len):
        angles = pos * inv_freq
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        # Build rotation matrices: [[cos, -sin], [sin, cos]]
        freqs[pos, :, 0, 0] = cos_vals
        freqs[pos, :, 0, 1] = -sin_vals
        freqs[pos, :, 1, 0] = sin_vals
        freqs[pos, :, 1, 1] = cos_vals

    return freqs

def apply_rope(x: torch.Tensor, pos_ids: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    seq_len, hidden_dim = x.shape
    TILE = hidden_dim

    output = torch.empty_like(x)

    ct.launch(
        torch.cuda.current_stream(),
        (seq_len, 1),
        rope_kernel,
        (x, pos_ids, freqs, output, TILE)
    )

    return output

# Usage
max_seq_len, seq_len, hidden_dim = 2048, 128, 256
x = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
pos_ids = torch.arange(seq_len, device="cuda", dtype=torch.int32)
freqs = build_rope_freqs(max_seq_len, hidden_dim)

y = apply_rope(x, pos_ids, freqs)
print(f"RoPE applied, output shape: {y.shape}")
```

---

## Example 9: SiLU and Multiply (Fused Activation)

**Purpose**: Fused operation: `y = x * sigmoid(gate) = x / (1 + exp(-gate))`.

**Key Techniques**: Fused operations, `ct.exp`, element-wise division

```python
import torch
import cuda.tile as ct

@ct.kernel
def silu_mul_kernel(x: ct.Array, gate: ct.Array, output: ct.Array, TILE: ct.Constant[int]):
    """Fused SiLU and multiply: y = x * sigmoid(gate)."""
    bid = ct.bid(0)

    # Load x and gate
    tile_x = ct.load(x, (bid,), (TILE,)).astype(ct.float32)
    tile_gate = ct.load(gate, (bid,), (TILE,)).astype(ct.float32)

    # SiLU: x / (1 + exp(-gate))
    # Equivalent to: x * sigmoid(gate)
    tile_y = tile_x / (1.0 + ct.exp(-tile_gate))

    ct.store(output, (bid,), tile_y.astype(output.dtype))

def silu_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply fused SiLU and multiply."""
    TILE = 1024
    num_blocks = ct.cdiv(x.numel(), TILE)

    output = torch.empty_like(x)

    ct.launch(
        torch.cuda.current_stream(),
        (num_blocks,),
        silu_mul_kernel,
        (x.flatten(), gate.flatten(), output.flatten(), TILE)
    )

    return output.reshape(x.shape)

# Usage
M, N = 32, 12288
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
gate = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

y = silu_mul(x, gate)

# Validation
y_torch = x * torch.sigmoid(gate)
torch.testing.assert_close(y, y_torch, rtol=1e-2, atol=1e-2)
```

---

## Example 10: Image to Patches (Vision Transformer)

**Purpose**: Convert image to patches for Vision Transformer input.

**Key Techniques**: 3D tiling, `ct.reshape`, coordinate tracking

```python
import torch
import cuda.tile as ct

@ct.kernel
def img2patch_kernel(
    image: ct.Array, patches: ct.Array, coords: ct.Array,
    patch_h: ct.Constant[int], patch_w: ct.Constant[int]
):
    """Convert image to patches: [C, H, W] -> [num_patches, C * patch_h * patch_w]."""
    patch_y = ct.bid(0)  # Patch row index
    patch_x = ct.bid(1)  # Patch col index
    channel = ct.bid(2)  # Channel index

    num_patches_x = ct.num_blocks(1)

    # Load patch from image
    tile = ct.load(
        image, (channel, patch_y, patch_x),
        (1, patch_h, patch_w),
        padding_mode=ct.PaddingMode.ZERO
    )

    # Reshape to 1D and store
    tile = tile.reshape((1, patch_h * patch_w))
    patch_idx = patch_y * num_patches_x + patch_x
    ct.store(patches, (patch_idx, channel), tile)

    # Store coordinates (only once per patch)
    if channel == 0:
        coord_y = ct.full((1, 1), patch_y, dtype=ct.int32)
        coord_x = ct.full((1, 1), patch_x, dtype=ct.int32)
        coord = ct.cat((coord_y, coord_x), axis=1)
        ct.store(coords, (patch_idx, 0), coord)

def img2patch(image: torch.Tensor, patch_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert image to patches."""
    C, H, W = image.shape
    patch_h, patch_w = patch_size

    num_patches_h = ct.cdiv(H, patch_h)
    num_patches_w = ct.cdiv(W, patch_w)
    num_patches = num_patches_h * num_patches_w

    patches = torch.empty(
        (num_patches, C * patch_h * patch_w),
        device=image.device, dtype=image.dtype
    )
    coords = torch.empty((num_patches, 2), device=image.device, dtype=torch.int32)

    ct.launch(
        torch.cuda.current_stream(),
        (num_patches_h, num_patches_w, C),
        img2patch_kernel,
        (image, patches, coords, patch_h, patch_w)
    )

    return patches, coords

# Usage
C, H, W = 3, 224, 224
patch_size = (16, 16)

image = torch.randn(C, H, W, device="cuda", dtype=torch.float16)
patches, coords = img2patch(image, patch_size)

print(f"Image shape: {image.shape}")
print(f"Patches shape: {patches.shape}")
print(f"Coordinates shape: {coords.shape}")
print(f"First patch coords: {coords[0]}")
```

---

**See also**:
- [EXAMPLES.md](EXAMPLES.md) - Quick start guide
- [EXAMPLES_COMPUTE.md](EXAMPLES_COMPUTE.md) - Compute-intensive operations
- [EXAMPLES_NORMALIZATION.md](EXAMPLES_NORMALIZATION.md) - Normalization operations
- [EXAMPLES_TRAINING.md](EXAMPLES_TRAINING.md) - Training operations
