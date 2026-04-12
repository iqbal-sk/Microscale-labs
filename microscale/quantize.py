# microscale/quantize.py
"""From-scratch quantization: naive 4-bit, NF4, and K-quant Q4_K_M.

Each scheme implements quantize() and dequantize() so you can measure
the round-trip error and compare approaches.
"""

import numpy as np

# The exact 16 NF4 bin centers from bitsandbytes source code.
# These are the quantiles of N(0,1) that divide it into 16
# equal-probability bins — information-theoretically optimal
# for Gaussian-distributed weights.
NF4_VALUES = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Naive Uniform 4-bit
# ---------------------------------------------------------------------------


def quantize_naive_4bit(
    tensor: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Naive uniform 4-bit quantization.

    Divides the [min, max] range into 16 equal bins.

    Returns:
        (indices, scale, zero_point) where indices are uint8 [0..15].
    """
    t_min = tensor.min()
    t_max = tensor.max()
    scale = (t_max - t_min) / 15.0
    if scale == 0:
        return np.zeros_like(tensor, dtype=np.uint8), 1.0, 0.0
    indices = np.clip(np.round((tensor - t_min) / scale), 0, 15).astype(
        np.uint8,
    )
    return indices, float(scale), float(t_min)


def dequantize_naive_4bit(
    indices: np.ndarray,
    scale: float,
    zero_point: float,
) -> np.ndarray:
    """Dequantize naive 4-bit back to float32."""
    return indices.astype(np.float32) * scale + zero_point


# ---------------------------------------------------------------------------
# NF4 (Normal Float 4) — from QLoRA (Dettmers et al., 2023)
# ---------------------------------------------------------------------------


def quantize_nf4(
    tensor: np.ndarray,
    block_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """NF4 quantization with per-block absmax scaling.

    1. Divide tensor into blocks
    2. Per block: scale to [-1, 1] using absmax
    3. Map each value to the nearest NF4 bin center

    Returns:
        (indices, scales) where indices are uint8 [0..15]
        and scales are fp32 per-block absmax values.
    """
    flat = tensor.flatten()
    n = len(flat)
    # Pad to block_size multiple
    pad_n = (block_size - n % block_size) % block_size
    if pad_n > 0:
        flat = np.concatenate([flat, np.zeros(pad_n, dtype=flat.dtype)])

    n_blocks = len(flat) // block_size
    blocks = flat.reshape(n_blocks, block_size)

    # Per-block absmax
    scales = np.abs(blocks).max(axis=1)
    scales = np.where(scales == 0, 1.0, scales)  # avoid div-by-zero

    # Normalize to [-1, 1]
    normalized = blocks / scales[:, None]

    # Find nearest NF4 bin for each value
    # Broadcast: (n_blocks, block_size, 1) vs (16,)
    diffs = np.abs(
        normalized[:, :, None] - NF4_VALUES[None, None, :],
    )
    indices = diffs.argmin(axis=2).astype(np.uint8)

    return indices.flatten()[:n], scales


def dequantize_nf4(
    indices: np.ndarray,
    scales: np.ndarray,
    block_size: int = 64,
    original_shape: tuple | None = None,
) -> np.ndarray:
    """Dequantize NF4 back to float32."""
    flat = indices.flatten()
    n = len(flat)
    pad_n = (block_size - n % block_size) % block_size
    if pad_n > 0:
        flat = np.concatenate([flat, np.zeros(pad_n, dtype=np.uint8)])

    n_blocks = len(flat) // block_size
    blocks = flat.reshape(n_blocks, block_size)

    # Look up NF4 values and scale back
    values = NF4_VALUES[blocks]  # (n_blocks, block_size)
    result = values * scales[:n_blocks, None]

    result = result.flatten()[:n]
    if original_shape:
        result = result.reshape(original_shape)
    return result


# ---------------------------------------------------------------------------
# K-quant Q4_K style (inspired by llama.cpp)
# ---------------------------------------------------------------------------


def quantize_q4k(
    tensor: np.ndarray,
    super_block_size: int = 256,
    sub_block_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q4_K-style quantization with sub-block scales.

    Two-level scaling hierarchy:
    - Super-block (256 weights): fp16 scale-of-scales
    - Sub-block (32 weights): 6-bit scale relative to super-block

    Returns:
        (indices, sub_scales, sub_mins, super_scales) where:
        - indices: uint8 [0..15] per weight
        - sub_scales: float32 per sub-block scale
        - sub_mins: float32 per sub-block minimum
        - super_scales: float32 per super-block (informational)
    """
    flat = tensor.flatten()
    n = len(flat)
    pad_n = (super_block_size - n % super_block_size) % super_block_size
    if pad_n > 0:
        flat = np.concatenate([flat, np.zeros(pad_n, dtype=flat.dtype)])

    n_super = len(flat) // super_block_size
    subs_per_super = super_block_size // sub_block_size
    n_sub = n_super * subs_per_super

    all_indices = np.zeros(len(flat), dtype=np.uint8)
    sub_scales = np.zeros(n_sub, dtype=np.float32)
    sub_mins = np.zeros(n_sub, dtype=np.float32)
    super_scales = np.zeros(n_super, dtype=np.float32)

    for si in range(n_super):
        super_start = si * super_block_size
        for sbi in range(subs_per_super):
            sub_idx = si * subs_per_super + sbi
            sub_start = super_start + sbi * sub_block_size
            sub_end = sub_start + sub_block_size
            sub_block = flat[sub_start:sub_end]

            s_min = sub_block.min()
            s_max = sub_block.max()
            scale = (s_max - s_min) / 15.0
            sub_scales[sub_idx] = scale
            sub_mins[sub_idx] = s_min

            if scale > 0:
                q = np.clip(
                    np.round((sub_block - s_min) / scale),
                    0,
                    15,
                ).astype(np.uint8)
            else:
                q = np.zeros(sub_block_size, dtype=np.uint8)
            all_indices[sub_start:sub_end] = q

        # Super-block scale is the max sub-block scale
        sb_start = si * subs_per_super
        sb_end = sb_start + subs_per_super
        super_scales[si] = sub_scales[sb_start:sb_end].max()

    return all_indices[:n], sub_scales, sub_mins, super_scales


def dequantize_q4k(
    indices: np.ndarray,
    sub_scales: np.ndarray,
    sub_mins: np.ndarray,
    original_shape: tuple | None = None,
    super_block_size: int = 256,
    sub_block_size: int = 32,
) -> np.ndarray:
    """Dequantize Q4_K back to float32."""
    flat = indices.flatten().copy()
    n = len(flat)
    pad_n = (super_block_size - n % super_block_size) % super_block_size
    if pad_n > 0:
        flat = np.concatenate([flat, np.zeros(pad_n, dtype=np.uint8)])

    result = np.zeros(len(flat), dtype=np.float32)

    for sub_idx in range(len(sub_scales)):
        start = sub_idx * sub_block_size
        end = start + sub_block_size
        # Dequant: value = index * scale + min
        result[start:end] = (
            flat[start:end].astype(np.float32) * sub_scales[sub_idx] + sub_mins[sub_idx]
        )

    result = result[:n]
    if original_shape:
        result = result.reshape(original_shape)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def quantization_error(original: np.ndarray, dequantized: np.ndarray) -> dict:
    """Compute quantization error metrics.

    Returns dict with: mse, rmse, max_error, sqnr_db
    """
    diff = original - dequantized
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(np.abs(diff)))

    signal_power = float(np.mean(original**2))
    sqnr = 10 * np.log10(signal_power / mse) if mse > 0 else float("inf")

    return {
        "mse": mse,
        "rmse": rmse,
        "max_error": max_err,
        "sqnr_db": float(sqnr),
    }


def bits_per_weight(
    method: str,
    n_weights: int,
    block_size: int = 64,
    sub_block_size: int = 32,
    super_block_size: int = 256,
) -> float:
    """Estimate effective bits per weight for each method."""
    if method == "naive":
        # 4 bits per weight + 2 floats (scale, zero) for the whole tensor
        return 4.0 + (2 * 32) / n_weights
    elif method == "nf4":
        # 4 bits per weight + 1 fp32 scale per block
        n_blocks = (n_weights + block_size - 1) // block_size
        overhead_bits = n_blocks * 32
        return 4.0 + overhead_bits / n_weights
    elif method == "q4k":
        # 4 bits per weight + sub-block scales + super-block scales
        n_sub = (n_weights + sub_block_size - 1) // sub_block_size
        n_super = (n_weights + super_block_size - 1) // super_block_size
        # 6 bits per sub-block scale + 16 bits per super-block scale
        overhead = n_sub * 6 + n_super * 16
        return 4.0 + overhead / n_weights
    return 4.0
