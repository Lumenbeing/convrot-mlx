# SPDX-FileCopyrightText: 2025 Comfy Org. All rights reserved.
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0
"""ConvRot W4A4 primitives for MLX on Apple Silicon.

The quantization contract follows Comfy-Org/comfy-kitchen's Apache-2.0 plain
ConvRot reference: a normalized regular-Hadamard rotation, symmetric signed
int4 quantization in [-7, 7], one scale per activation row / weight row, and
low-nibble-first row-major packing. The implementation here is an independent
MLX/Metal port with group-64 and group-256 kernels.

The legacy Metal simdgroup matrix API has floating matrix fragments but no
integer or int4 fragment type.  The original packed backend therefore expands
nibbles into half fragments before its dot product.  Metal 4's
MetalPerformancePrimitives TensorOps API adds native half/bfloat x signed-int4
and signed-int8 x signed-int4 operations.  The MPP backends below use those
operations with a bit-exact K x N packed-weight layout.  W4A4 expands only the
packed activation to signed int8 (the TensorOp's supported left operand), while
W4A16 feeds the rotated floating activation directly.  Row scales and bias are
applied after the TensorOp dot product.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import mlx.core as mx
from mlx import nn

CONVROT_GROUP_SIZE = 256
SUPPORTED_FUSED_GROUP_SIZES = frozenset({64, 256})
INT4_MAX = 7
SUPPORTED_BACKENDS = frozenset(
    {
        "metal",
        "metal-k32",
        "metal-k64",
        "metal-m32n32-k64",
        "mpp-w4a4",
        "mpp-w4a16",
        "mlx-expanded-fp32",
        "reference",
    }
)
SUPPORTED_PREPROCESSORS = frozenset({"mlx-compiled", "metal-fused"})


def _validate_group_size(group_size: int) -> None:
    if group_size < 4:
        raise ValueError(
            f"Regular Hadamard size must be a power of 4, got {group_size}"
        )
    value = group_size
    while value > 1 and value % 4 == 0:
        value //= 4
    if value != 1:
        raise ValueError(
            f"Regular Hadamard size must be a power of 4, got {group_size}"
        )


def regular_hadamard(x: mx.array, group_size: int = CONVROT_GROUP_SIZE) -> mx.array:
    """Right-multiply each feature group by the normalized regular Hadamard.

    The radix-4 butterfly is the matrix-free equivalent of multiplication by
    H4 kron H4 ... in comfy-kitchen.  H4 is symmetric, so this same routine is
    used for the reference's activation ``x @ H`` and weight ``W @ H.T``.
    """

    _validate_group_size(group_size)
    features = x.shape[-1]
    if features % group_size:
        raise ValueError(
            f"features {features} not divisible by ConvRot group size {group_size}"
        )

    groups = features // group_size
    prefix = (*x.shape[:-1], groups)
    y = x.reshape(*prefix, group_size)
    stride = 1
    while stride < group_size:
        y = y.reshape(*prefix, group_size // (4 * stride), 4, stride)
        a = y[..., 0, :]
        b = y[..., 1, :]
        c = y[..., 2, :]
        d = y[..., 3, :]
        y = mx.stack(
            [a + b + c - d, a + b - c + d, a - b + c + d, -a + b + c + d],
            axis=-2,
        ).reshape(*prefix, group_size)
        stride *= 4
    return (y / math.sqrt(group_size)).reshape(x.shape)


def pack_signed_int4(values: mx.array) -> mx.array:
    """Pack signed nibbles low-column first, matching comfy-kitchen."""

    if values.shape[-1] % 2:
        raise ValueError(f"last dimension must be even, got {values.shape[-1]}")
    values = values.astype(mx.int32)
    lo = values[..., 0::2] & 0x0F
    hi = values[..., 1::2] & 0x0F
    return (lo | (hi << 4)).astype(mx.uint8)


def unpack_signed_int4(packed: mx.array) -> mx.array:
    """Unpack two's-complement nibbles to int8 in [-8, 7]."""

    data = packed.astype(mx.int32)
    lo = data & 0x0F
    hi = (data >> 4) & 0x0F
    lo = mx.where(lo >= 8, lo - 16, lo)
    hi = mx.where(hi >= 8, hi - 16, hi)
    return (
        mx.stack([lo, hi], axis=-1)
        .reshape(*packed.shape[:-1], packed.shape[-1] * 2)
        .astype(mx.int8)
    )


def repack_nk_weight_for_mpp(qweight: mx.array) -> mx.array:
    """Bit-exact N x packed-K to K x packed-N conversion for MPP TensorOps."""

    if qweight.ndim != 2:
        raise ValueError(f"MPP weight repack expects a 2D tensor, got {qweight.shape}")
    unpacked = unpack_signed_int4(qweight)
    if unpacked.shape[0] % 2:
        raise ValueError(
            f"MPP int4 packing requires even out_features, got {unpacked.shape[0]}"
        )
    return pack_signed_int4(unpacked.T)


def repack_mpp_weight_to_nk(qweight: mx.array) -> mx.array:
    """Reverse MPP's K x packed-N layout back to N x packed-K."""

    if qweight.ndim != 2:
        raise ValueError(f"MPP weight repack expects a 2D tensor, got {qweight.shape}")
    unpacked = unpack_signed_int4(qweight)
    if unpacked.shape[0] % 2:
        raise ValueError(
            f"row-major int4 packing requires even in_features, got {unpacked.shape[0]}"
        )
    return pack_signed_int4(unpacked.T)


def quantize_signed_int4_rowwise(x: mx.array) -> tuple[mx.array, mx.array]:
    """Symmetric absmax/7 row quantization used by ConvRot W4A4."""

    if x.ndim != 2:
        raise ValueError(f"rowwise int4 quantization expects 2D input, got {x.shape}")
    absmax = mx.max(mx.abs(x), axis=-1, keepdims=True)
    absmax = mx.maximum(absmax, mx.array(1e-10, dtype=absmax.dtype))
    scales = absmax / INT4_MAX
    q = mx.clip(mx.round(x / scales), -INT4_MAX, INT4_MAX).astype(mx.int8)
    return pack_signed_int4(q), scales.reshape(-1).astype(mx.float32)


def prepare_convrot_activation(
    x: mx.array,
    group_size: int = CONVROT_GROUP_SIZE,
) -> tuple[mx.array, mx.array]:
    """Rotate and row-quantize an activation under the ConvRot contract."""

    return quantize_signed_int4_rowwise(regular_hadamard(x, group_size))


# This graph is pure and shape-specialized by MLX. Keeping the rotation,
# reduction, rounding, and nibble packing behind one compiled boundary removes
# avoidable dispatch/materialization work without changing either returned
# tensor. MLX caches the small set of FLUX activation shapes after first use.
compiled_prepare_convrot_activation = mx.compile(prepare_convrot_activation)
compiled_regular_hadamard = mx.compile(regular_hadamard)


_FUSED_PREPARE_HEADER = r"""
template <typename T>
METAL_FUNC T convrot_mlx_hadamard_lane(T a, T b, T c, T d, uint lane) {
    T value;
    if (lane == 0u) {
        value = static_cast<T>(a + b);
        value = static_cast<T>(value + c);
        return static_cast<T>(value - d);
    }
    if (lane == 1u) {
        value = static_cast<T>(a + b);
        value = static_cast<T>(value - c);
        return static_cast<T>(value + d);
    }
    if (lane == 2u) {
        value = static_cast<T>(a - b);
        value = static_cast<T>(value + c);
        return static_cast<T>(value + d);
    }
    value = static_cast<T>(-a);
    value = static_cast<T>(value + b);
    value = static_cast<T>(value + c);
    return static_cast<T>(value + d);
}
"""


_FUSED_PREPARE_SOURCE = r"""
    constexpr uint GROUP_SIZE = 256u;
    constexpr float NORMALIZATION = 0.0625f;

    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    const uint rows = uint(x_shape[0]);
    const uint features = uint(x_shape[1]);
    if (row >= rows) return;

    threadgroup T stage_a[GROUP_SIZE];
    threadgroup T stage_b[GROUP_SIZE];
    threadgroup float reductions[GROUP_SIZE];
    threadgroup T shared_scale[1];

    float local_absmax = 0.0f;
    for (uint group0 = 0u; group0 < features; group0 += GROUP_SIZE) {
        stage_a[tid] = x[row * features + group0 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup T* read_tile = stage_a;
        threadgroup T* write_tile = stage_b;
        for (uint stride = 1u; stride < GROUP_SIZE; stride *= 4u) {
            const uint block_width = 4u * stride;
            const uint within = tid % block_width;
            const uint lane = within / stride;
            const uint offset = within % stride;
            const uint base = (tid / block_width) * block_width + offset;
            const T a = read_tile[base];
            const T b = read_tile[base + stride];
            const T c = read_tile[base + 2u * stride];
            const T d = read_tile[base + 3u * stride];
            write_tile[tid] = convrot_mlx_hadamard_lane(a, b, c, d, lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup T* swap_tile = read_tile;
            read_tile = write_tile;
            write_tile = swap_tile;
        }

        const T rotated = static_cast<T>(static_cast<float>(read_tile[tid]) * NORMALIZATION);
        local_absmax = metal::max(local_absmax, metal::abs(static_cast<float>(rotated)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    reductions[tid] = local_absmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = GROUP_SIZE / 2u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            reductions[tid] = metal::max(reductions[tid], reductions[tid + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const T floor_value = static_cast<T>(1.0e-10f);
        const T maximum = static_cast<T>(metal::max(reductions[0], static_cast<float>(floor_value)));
        shared_scale[0] = static_cast<T>(maximum / static_cast<T>(7.0f));
        scales[row] = static_cast<float>(shared_scale[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint packed_features = features / 2u;
    for (uint group0 = 0u; group0 < features; group0 += GROUP_SIZE) {
        stage_a[tid] = x[row * features + group0 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup T* read_tile = stage_a;
        threadgroup T* write_tile = stage_b;
        for (uint stride = 1u; stride < GROUP_SIZE; stride *= 4u) {
            const uint block_width = 4u * stride;
            const uint within = tid % block_width;
            const uint lane = within / stride;
            const uint offset = within % stride;
            const uint base = (tid / block_width) * block_width + offset;
            const T a = read_tile[base];
            const T b = read_tile[base + stride];
            const T c = read_tile[base + 2u * stride];
            const T d = read_tile[base + 3u * stride];
            write_tile[tid] = convrot_mlx_hadamard_lane(a, b, c, d, lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup T* swap_tile = read_tile;
            read_tile = write_tile;
            write_tile = swap_tile;
        }

        if ((tid & 1u) == 0u) {
            const T lo_rotated = static_cast<T>(static_cast<float>(read_tile[tid]) * NORMALIZATION);
            const T hi_rotated = static_cast<T>(static_cast<float>(read_tile[tid + 1u]) * NORMALIZATION);
            const T lo_ratio = static_cast<T>(lo_rotated / shared_scale[0]);
            const T hi_ratio = static_cast<T>(hi_rotated / shared_scale[0]);
            int lo = int(metal::rint(static_cast<float>(lo_ratio)));
            int hi = int(metal::rint(static_cast<float>(hi_ratio)));
            lo = metal::clamp(lo, -7, 7);
            hi = metal::clamp(hi, -7, 7);
            const uchar packed = uchar((lo & 0x0f) | ((hi & 0x0f) << 4));
            qact[row * packed_features + ((group0 + tid) >> 1u)] = packed;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


_FUSED_PREPARE_I8_SOURCE = _FUSED_PREPARE_SOURCE.replace(
    r"""        if ((tid & 1u) == 0u) {
            const T lo_rotated = static_cast<T>(static_cast<float>(read_tile[tid]) * NORMALIZATION);
            const T hi_rotated = static_cast<T>(static_cast<float>(read_tile[tid + 1u]) * NORMALIZATION);
            const T lo_ratio = static_cast<T>(lo_rotated / shared_scale[0]);
            const T hi_ratio = static_cast<T>(hi_rotated / shared_scale[0]);
            int lo = int(metal::rint(static_cast<float>(lo_ratio)));
            int hi = int(metal::rint(static_cast<float>(hi_ratio)));
            lo = metal::clamp(lo, -7, 7);
            hi = metal::clamp(hi, -7, 7);
            const uchar packed = uchar((lo & 0x0f) | ((hi & 0x0f) << 4));
            qact[row * packed_features + ((group0 + tid) >> 1u)] = packed;
        }
""",
    r"""        const T rotated = static_cast<T>(static_cast<float>(read_tile[tid]) * NORMALIZATION);
        const T ratio = static_cast<T>(rotated / shared_scale[0]);
        int quantized = int(metal::rint(static_cast<float>(ratio)));
        quantized = metal::clamp(quantized, -7, 7);
        qact[row * features + group0 + tid] = static_cast<char>(quantized);
""",
)


_FUSED_ROTATE_SOURCE = r"""
    constexpr uint GROUP_SIZE = 256u;
    constexpr float NORMALIZATION = 0.0625f;

    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    const uint rows = uint(x_shape[0]);
    const uint features = uint(x_shape[1]);
    if (row >= rows) return;

    threadgroup T stage_a[GROUP_SIZE];
    threadgroup T stage_b[GROUP_SIZE];
    for (uint group0 = 0u; group0 < features; group0 += GROUP_SIZE) {
        stage_a[tid] = x[row * features + group0 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup T* read_tile = stage_a;
        threadgroup T* write_tile = stage_b;
        for (uint stride = 1u; stride < GROUP_SIZE; stride *= 4u) {
            const uint block_width = 4u * stride;
            const uint within = tid % block_width;
            const uint lane = within / stride;
            const uint offset = within % stride;
            const uint base = (tid / block_width) * block_width + offset;
            const T a = read_tile[base];
            const T b = read_tile[base + stride];
            const T c = read_tile[base + 2u * stride];
            const T d = read_tile[base + 3u * stride];
            write_tile[tid] = convrot_mlx_hadamard_lane(a, b, c, d, lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup T* swap_tile = read_tile;
            read_tile = write_tile;
            write_tile = swap_tile;
        }

        out[row * features + group0 + tid] = static_cast<T>(
            static_cast<float>(read_tile[tid]) * NORMALIZATION
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def _specialize_fused_source(source: str, group_size: int) -> str:
    _validate_group_size(group_size)
    if group_size not in SUPPORTED_FUSED_GROUP_SIZES:
        raise ValueError(
            f"fused ConvRot supports groups {sorted(SUPPORTED_FUSED_GROUP_SIZES)}, got {group_size}"
        )
    normalization = 1.0 / math.sqrt(group_size)
    return source.replace(
        "constexpr uint GROUP_SIZE = 256u;",
        f"constexpr uint GROUP_SIZE = {group_size}u;",
    ).replace(
        "constexpr float NORMALIZATION = 0.0625f;",
        f"constexpr float NORMALIZATION = {normalization:.10g}f;",
    )


@lru_cache(maxsize=len(SUPPORTED_FUSED_GROUP_SIZES))
def _fused_prepare_convrot_kernel(group_size: int):
    return mx.fast.metal_kernel(
        name=f"convrot_mlx_convrot_fused_hadamard_absmax_a4_pack_g{group_size}",
        input_names=["x"],
        output_names=["qact", "scales"],
        header=_FUSED_PREPARE_HEADER,
        source=_specialize_fused_source(_FUSED_PREPARE_SOURCE, group_size),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=len(SUPPORTED_FUSED_GROUP_SIZES))
def _fused_prepare_convrot_i8_kernel(group_size: int):
    return mx.fast.metal_kernel(
        name=f"convrot_mlx_convrot_fused_hadamard_absmax_a4_i8_g{group_size}",
        input_names=["x"],
        output_names=["qact", "scales"],
        header=_FUSED_PREPARE_HEADER,
        source=_specialize_fused_source(_FUSED_PREPARE_I8_SOURCE, group_size),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=len(SUPPORTED_FUSED_GROUP_SIZES))
def _fused_rotate_convrot_kernel(group_size: int):
    return mx.fast.metal_kernel(
        name=f"convrot_mlx_convrot_fused_hadamard_rotate_g{group_size}",
        input_names=["x"],
        output_names=["out"],
        header=_FUSED_PREPARE_HEADER,
        source=_specialize_fused_source(_FUSED_ROTATE_SOURCE, group_size),
        ensure_row_contiguous=True,
    )


def metal_fused_prepare_convrot_activation(
    x: mx.array,
    group_size: int = CONVROT_GROUP_SIZE,
) -> tuple[mx.array, mx.array]:
    """Two-pass threadgroup-local Hadamard, row absmax, A4 quantize, and pack."""

    if x.ndim != 2:
        raise ValueError(
            f"fused ConvRot activation preparation expects 2D input, got {x.shape}"
        )
    if group_size not in SUPPORTED_FUSED_GROUP_SIZES:
        raise ValueError(
            f"fused ConvRot activation preparation supports groups {sorted(SUPPORTED_FUSED_GROUP_SIZES)}, got {group_size}"
        )
    if x.shape[1] % group_size:
        raise ValueError(
            f"features {x.shape[1]} not divisible by ConvRot group size {group_size}"
        )
    rows, features = x.shape
    return tuple(
        _fused_prepare_convrot_kernel(group_size)(
            inputs=[x],
            template=[("T", x.dtype)],
            output_shapes=[(rows, features // 2), (rows,)],
            output_dtypes=[mx.uint8, mx.float32],
            grid=(rows * group_size, 1, 1),
            threadgroup=(group_size, 1, 1),
        )
    )


def metal_fused_prepare_convrot_activation_i8(
    x: mx.array,
    group_size: int = CONVROT_GROUP_SIZE,
) -> tuple[mx.array, mx.array]:
    """Rotate and A4-quantize directly into MPP's signed-int8 left operand."""

    if x.ndim != 2:
        raise ValueError(
            f"fused ConvRot activation preparation expects 2D input, got {x.shape}"
        )
    if group_size not in SUPPORTED_FUSED_GROUP_SIZES:
        raise ValueError(
            f"fused ConvRot activation preparation supports groups {sorted(SUPPORTED_FUSED_GROUP_SIZES)}, got {group_size}"
        )
    if x.shape[1] % group_size:
        raise ValueError(
            f"features {x.shape[1]} not divisible by ConvRot group size {group_size}"
        )
    rows, features = x.shape
    return tuple(
        _fused_prepare_convrot_i8_kernel(group_size)(
            inputs=[x],
            template=[("T", x.dtype)],
            output_shapes=[(rows, features), (rows,)],
            output_dtypes=[mx.int8, mx.float32],
            grid=(rows * group_size, 1, 1),
            threadgroup=(group_size, 1, 1),
        )
    )


def metal_fused_rotate_convrot_activation(
    x: mx.array,
    group_size: int = CONVROT_GROUP_SIZE,
) -> mx.array:
    """Rotate an activation in one Metal pass for the MPP W4A16 path."""

    if x.ndim != 2:
        raise ValueError(f"fused ConvRot rotation expects 2D input, got {x.shape}")
    if group_size not in SUPPORTED_FUSED_GROUP_SIZES:
        raise ValueError(
            f"fused ConvRot rotation supports groups {sorted(SUPPORTED_FUSED_GROUP_SIZES)}, got {group_size}"
        )
    if x.shape[1] % group_size:
        raise ValueError(
            f"features {x.shape[1]} not divisible by ConvRot group size {group_size}"
        )
    rows, features = x.shape
    return _fused_rotate_convrot_kernel(group_size)(
        inputs=[x],
        template=[("T", x.dtype)],
        output_shapes=[(rows, features)],
        output_dtypes=[x.dtype],
        grid=(rows * group_size, 1, 1),
        threadgroup=(group_size, 1, 1),
    )[0]


def quantize_convrot_weight(
    weight: mx.array,
    group_size: int = CONVROT_GROUP_SIZE,
) -> tuple[mx.array, mx.array]:
    """Rotate a dense ``(out, in)`` weight and store it as packed signed W4."""

    if weight.ndim != 2:
        raise ValueError(f"ConvRot weight must be 2D, got {weight.shape}")
    if weight.shape[-1] % group_size:
        raise ValueError(
            f"in_features {weight.shape[-1]} not divisible by ConvRot group size {group_size}"
        )
    rotated = regular_hadamard(weight, group_size)
    return quantize_signed_int4_rowwise(rotated)


_METAL_HEADER = r"""
#include <metal_simdgroup_matrix>

METAL_FUNC half convrot_mlx_signed_nibble(uchar packed, uint column) {
    uchar nibble = (column & 1u) ? ((packed >> 4u) & 0x0fu) : (packed & 0x0fu);
    char value = nibble >= 8u ? char(int(nibble) - 16) : char(nibble);
    return half(value);
}
"""


_METAL_SOURCE = r"""
    constexpr uint TILE_M = 16;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 8;

    const uint tid = thread_position_in_threadgroup.x;
    const uint sgid = simdgroup_index_in_threadgroup;
    const uint block_m = threadgroup_position_in_grid.y * TILE_M;
    const uint block_n = threadgroup_position_in_grid.x * TILE_N;
    const uint sub_m = sgid / 4u;
    const uint sub_n = sgid % 4u;
    const uint M = uint(qact_shape[0]);
    const uint K = uint(qact_shape[1]) * 2u;
    const uint N = uint(qweight_shape[0]);
    const uint packed_k = K / 2u;

    threadgroup half a_tile[TILE_M * TILE_K];
    threadgroup half b_tile[TILE_K * TILE_N];
    threadgroup float c_tile[TILE_M * TILE_N];

    simdgroup_float8x8 accum = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < K; k0 += TILE_K) {
        if (tid < TILE_M * TILE_K) {
            const uint row = tid / TILE_K;
            const uint kk = tid % TILE_K;
            const uint global_m = block_m + row;
            const uint global_k = k0 + kk;
            half value = 0.0h;
            if (global_m < M && global_k < K) {
                const uchar packed = qact[global_m * packed_k + (global_k >> 1u)];
                value = convrot_mlx_signed_nibble(packed, global_k);
            }
            a_tile[row * TILE_K + kk] = value;
        }

        {
            const uint kk = tid / TILE_N;
            const uint col = tid % TILE_N;
            const uint global_n = block_n + col;
            const uint global_k = k0 + kk;
            half value = 0.0h;
            if (global_n < N && global_k < K) {
                const uchar packed = qweight[global_n * packed_k + (global_k >> 1u)];
                value = convrot_mlx_signed_nibble(packed, global_k);
            }
            b_tile[kk * TILE_N + col] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 a_frag;
        simdgroup_half8x8 b_frag;
        simdgroup_load(a_frag, a_tile + sub_m * 8u * TILE_K, TILE_K);
        simdgroup_load(b_frag, b_tile + sub_n * 8u, TILE_N);
        simdgroup_multiply_accumulate(accum, a_frag, b_frag, accum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accum, c_tile + sub_m * 8u * TILE_N + sub_n * 8u, TILE_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < TILE_M * TILE_N; index += 256u) {
        const uint row = index / TILE_N;
        const uint col = index % TILE_N;
        const uint global_m = block_m + row;
        const uint global_n = block_n + col;
        if (global_m < M && global_n < N) {
            const float value = c_tile[index] * xscales[global_m] * wscales[global_n] + bias[global_n];
            out[global_m * N + global_n] = static_cast<T>(value);
        }
    }
"""


_METAL_SOURCE_STAGED = r"""
    constexpr uint TILE_M = 16;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 32;
    constexpr uint PACKED_TILE_K = TILE_K / 2;

    const uint tid = thread_position_in_threadgroup.x;
    const uint sgid = simdgroup_index_in_threadgroup;
    const uint block_m = threadgroup_position_in_grid.y * TILE_M;
    const uint block_n = threadgroup_position_in_grid.x * TILE_N;
    const uint sub_m = sgid / 4u;
    const uint sub_n = sgid % 4u;
    const uint M = uint(qact_shape[0]);
    const uint K = uint(qact_shape[1]) * 2u;
    const uint N = uint(qweight_shape[0]);
    const uint packed_k = K / 2u;

    threadgroup half a_tile[TILE_M * TILE_K];
    threadgroup half b_tile[TILE_K * TILE_N];
    threadgroup float c_tile[TILE_M * TILE_N];

    simdgroup_float8x8 accum = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < K; k0 += TILE_K) {
        for (uint packed_index = tid; packed_index < TILE_M * PACKED_TILE_K; packed_index += 256u) {
            const uint row = packed_index / PACKED_TILE_K;
            const uint packed_kk = packed_index % PACKED_TILE_K;
            const uint global_m = block_m + row;
            const uint global_k = k0 + packed_kk * 2u;
            uchar packed = 0u;
            if (global_m < M && global_k < K) {
                packed = qact[global_m * packed_k + (global_k >> 1u)];
            }
            const uint tile_k = packed_kk * 2u;
            a_tile[row * TILE_K + tile_k] = convrot_mlx_signed_nibble(packed, 0u);
            a_tile[row * TILE_K + tile_k + 1u] = convrot_mlx_signed_nibble(packed, 1u);
        }

        for (uint packed_index = tid; packed_index < PACKED_TILE_K * TILE_N; packed_index += 256u) {
            const uint packed_kk = packed_index / TILE_N;
            const uint col = packed_index % TILE_N;
            const uint global_n = block_n + col;
            const uint global_k = k0 + packed_kk * 2u;
            uchar packed = 0u;
            if (global_n < N && global_k < K) {
                packed = qweight[global_n * packed_k + (global_k >> 1u)];
            }
            const uint tile_k = packed_kk * 2u;
            b_tile[tile_k * TILE_N + col] = convrot_mlx_signed_nibble(packed, 0u);
            b_tile[(tile_k + 1u) * TILE_N + col] = convrot_mlx_signed_nibble(packed, 1u);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub_k = 0u; sub_k < TILE_K; sub_k += 8u) {
            simdgroup_half8x8 a_frag;
            simdgroup_half8x8 b_frag;
            simdgroup_load(a_frag, a_tile + sub_m * 8u * TILE_K + sub_k, TILE_K);
            simdgroup_load(b_frag, b_tile + sub_k * TILE_N + sub_n * 8u, TILE_N);
            simdgroup_multiply_accumulate(accum, a_frag, b_frag, accum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(accum, c_tile + sub_m * 8u * TILE_N + sub_n * 8u, TILE_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < TILE_M * TILE_N; index += 256u) {
        const uint row = index / TILE_N;
        const uint col = index % TILE_N;
        const uint global_m = block_m + row;
        const uint global_n = block_n + col;
        if (global_m < M && global_n < N) {
            const float value = c_tile[index] * xscales[global_m] * wscales[global_n] + bias[global_n];
            out[global_m * N + global_n] = static_cast<T>(value);
        }
    }
"""


@lru_cache(maxsize=1)
def _packed_w4a4_kernel():
    return mx.fast.metal_kernel(
        name="convrot_mlx_convrot_packed_w4a4_16x32",
        input_names=["qact", "qweight", "xscales", "wscales", "bias"],
        output_names=["out"],
        header=_METAL_HEADER,
        source=_METAL_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def _packed_w4a4_kernel_staged(tile_m: int, tile_n: int, tile_k: int):
    if (
        tile_m not in {16, 32}
        or tile_n not in {32, 64}
        or tile_k not in {32, 64, 128, 256}
    ):
        raise ValueError(f"Unsupported staged tile {tile_m}x{tile_n}x{tile_k}")
    if tile_m % 8 or tile_n % 8:
        raise ValueError(
            "Staged M and N tiles must be divisible by the 8x8 simdgroup fragment"
        )
    subtiles_n = tile_n // 8
    threadgroup_size = (tile_m // 8) * subtiles_n * 32
    if threadgroup_size not in {256, 512}:
        raise ValueError(f"Unsupported staged threadgroup size {threadgroup_size}")
    source = (
        _METAL_SOURCE_STAGED.replace(
            "constexpr uint TILE_M = 16;", f"constexpr uint TILE_M = {tile_m};"
        )
        .replace("constexpr uint TILE_N = 32;", f"constexpr uint TILE_N = {tile_n};")
        .replace("constexpr uint TILE_K = 32;", f"constexpr uint TILE_K = {tile_k};")
        .replace(
            "const uint sub_m = sgid / 4u;", f"const uint sub_m = sgid / {subtiles_n}u;"
        )
        .replace(
            "const uint sub_n = sgid % 4u;", f"const uint sub_n = sgid % {subtiles_n}u;"
        )
        .replace("packed_index += 256u", f"packed_index += {threadgroup_size}u")
        .replace("index += 256u", f"index += {threadgroup_size}u")
    )
    return mx.fast.metal_kernel(
        name=f"convrot_mlx_convrot_packed_w4a4_{tile_m}x{tile_n}x{tile_k}",
        input_names=["qact", "qweight", "xscales", "wscales", "bias"],
        output_names=["out"],
        header=_METAL_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def packed_w4a4_matmul(
    qact: mx.array,
    qweight: mx.array,
    xscales: mx.array,
    wscales: mx.array,
    bias: mx.array,
    *,
    output_dtype,
) -> mx.array:
    """Run the packed W4A4 dot product through the tiled Metal kernel."""

    if qact.ndim != 2 or qweight.ndim != 2:
        raise ValueError("packed W4A4 expects 2D activation and weight tensors")
    if qact.shape[1] != qweight.shape[1]:
        raise ValueError(
            f"packed K mismatch: activation {qact.shape}, weight {qweight.shape}"
        )
    m, n = qact.shape[0], qweight.shape[0]
    grid = (math.ceil(n / 32) * 256, math.ceil(m / 16), 1)
    return _packed_w4a4_kernel()(
        inputs=[qact, qweight, xscales, wscales, bias],
        template=[("T", output_dtype)],
        output_shapes=[(m, n)],
        output_dtypes=[output_dtype],
        grid=grid,
        threadgroup=(256, 1, 1),
    )[0]


def packed_w4a4_matmul_staged(
    qact: mx.array,
    qweight: mx.array,
    xscales: mx.array,
    wscales: mx.array,
    bias: mx.array,
    *,
    output_dtype,
    tile_m: int = 16,
    tile_n: int = 32,
    tile_k: int,
) -> mx.array:
    """Packed W4A4 kernel with paired-byte loads and configurable K staging."""

    if qact.ndim != 2 or qweight.ndim != 2:
        raise ValueError("packed W4A4 expects 2D activation and weight tensors")
    if qact.shape[1] != qweight.shape[1]:
        raise ValueError(
            f"packed K mismatch: activation {qact.shape}, weight {qweight.shape}"
        )
    m, n = qact.shape[0], qweight.shape[0]
    threadgroup_size = (tile_m // 8) * (tile_n // 8) * 32
    grid = (math.ceil(n / tile_n) * threadgroup_size, math.ceil(m / tile_m), 1)
    return _packed_w4a4_kernel_staged(tile_m, tile_n, tile_k)(
        inputs=[qact, qweight, xscales, wscales, bias],
        template=[("T", output_dtype)],
        output_shapes=[(m, n)],
        output_dtypes=[output_dtype],
        grid=grid,
        threadgroup=(threadgroup_size, 1, 1),
    )[0]


def packed_w4a4_matmul_k32(*args, **kwargs) -> mx.array:
    return packed_w4a4_matmul_staged(*args, **kwargs, tile_k=32)


def packed_w4a4_matmul_k64(*args, **kwargs) -> mx.array:
    return packed_w4a4_matmul_staged(*args, **kwargs, tile_k=64)


def packed_w4a4_matmul_k128(*args, **kwargs) -> mx.array:
    return packed_w4a4_matmul_staged(*args, **kwargs, tile_k=128)


def packed_w4a4_matmul_k256(*args, **kwargs) -> mx.array:
    return packed_w4a4_matmul_staged(*args, **kwargs, tile_k=256)


def packed_w4a4_matmul_m32n32_k64(*args, **kwargs) -> mx.array:
    return packed_w4a4_matmul_staged(*args, **kwargs, tile_m=32, tile_n=32, tile_k=64)


def packed_w4a4_matmul_m16n64_k64(*args, **kwargs) -> mx.array:
    return packed_w4a4_matmul_staged(*args, **kwargs, tile_m=16, tile_n=64, tile_k=64)


_MPP_HEADER = r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
"""


_MPP_W4A4_SOURCE = r"""
    constexpr int M = __M__;
    constexpr int N = __N__;
    constexpr int K = __K__;
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 32;

    using a8_tensor = tensor<device int8_t, dextents<int, 2>, tensor_inline>;
    using b4_tensor = tensor<device int4b_format, dextents<int, 2>, tensor_inline>;
    using c32_tensor = tensor<device int32_t, dextents<int, 2>, tensor_inline>;

    a8_tensor ta((device int8_t*)a, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    b4_tensor tb((device uint8_t*)b, dextents<int, 2>{N, K}, array<int, 2>{1, N});
    c32_tensor tc((device int32_t*)dots, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
        TILE_M,
        TILE_N,
        TILE_K,
        false,
        false,
        false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );
    tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;

    const int block_n = int(threadgroup_position_in_grid.x) * TILE_N;
    const int block_m = int(threadgroup_position_in_grid.y) * TILE_M;
    auto mc = tc.slice(block_n, block_m);
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        auto ma = ta.slice(k0, block_m);
        auto mb = tb.slice(block_n, k0);
        operation.run(ma, mb, mc);
    }
"""


_MPP_W4A16_SOURCE = r"""
    constexpr int M = __M__;
    constexpr int N = __N__;
    constexpr int K = __K__;
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 32;

    using activation_tensor = tensor<device T, dextents<int, 2>, tensor_inline>;
    using b4_tensor = tensor<device int4b_format, dextents<int, 2>, tensor_inline>;
    using output_tensor = tensor<device float, dextents<int, 2>, tensor_inline>;

    activation_tensor ta((device T*)a, dextents<int, 2>{K, M}, array<int, 2>{1, K});
    b4_tensor tb((device uint8_t*)b, dextents<int, 2>{N, K}, array<int, 2>{1, N});
    output_tensor tc((device float*)dots, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
        TILE_M,
        TILE_N,
        TILE_K,
        false,
        false,
        false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );
    tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;

    const int block_n = int(threadgroup_position_in_grid.x) * TILE_N;
    const int block_m = int(threadgroup_position_in_grid.y) * TILE_M;
    auto mc = tc.slice(block_n, block_m);
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        auto ma = ta.slice(k0, block_m);
        auto mb = tb.slice(block_n, k0);
        operation.run(ma, mb, mc);
    }
"""


_UNPACK_INT4_SOURCE = r"""
    const uint index = thread_position_in_grid.x;
    const uint total = uint(packed_shape[0]) * uint(packed_shape[1]) * 2u;
    if (index >= total) return;
    const uchar byte = packed[index >> 1u];
    const uchar nibble = (index & 1u) ? ((byte >> 4u) & 0x0fu) : (byte & 0x0fu);
    unpacked[index] = nibble >= 8u ? static_cast<char>(int(nibble) - 16) : static_cast<char>(nibble);
"""


_MPP_W4A4_EPILOGUE_SOURCE = r"""
    const uint index = thread_position_in_grid.x;
    const uint M = uint(dots_shape[0]);
    const uint N = uint(dots_shape[1]);
    if (index >= M * N) return;
    const uint row = index / N;
    const uint column = index % N;
    const float value = static_cast<float>(dots[index])
        * xscales[row]
        * wscales[column]
        + bias[column];
    out[index] = static_cast<T>(value);
"""


def _specialize_mpp_source(source: str, m: int, n: int, k: int) -> str:
    return (
        source.replace("__M__", str(m))
        .replace("__N__", str(n))
        .replace("__K__", str(k))
    )


@lru_cache(maxsize=128)
def _mpp_w4a4_kernel(m: int, n: int, k: int):
    return mx.fast.metal_kernel(
        name=f"convrot_mlx_mpp_w4a4_{m}_{n}_{k}",
        input_names=["a", "b"],
        output_names=["dots"],
        header=_MPP_HEADER,
        source=_specialize_mpp_source(_MPP_W4A4_SOURCE, m, n, k),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=128)
def _mpp_w4a16_kernel(m: int, n: int, k: int):
    return mx.fast.metal_kernel(
        name=f"convrot_mlx_mpp_w4a16_{m}_{n}_{k}",
        input_names=["a", "b"],
        output_names=["dots"],
        header=_MPP_HEADER,
        source=_specialize_mpp_source(_MPP_W4A16_SOURCE, m, n, k),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _unpack_signed_int4_kernel():
    return mx.fast.metal_kernel(
        name="convrot_mlx_unpack_signed_int4_to_i8",
        input_names=["packed"],
        output_names=["unpacked"],
        source=_UNPACK_INT4_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _mpp_w4a4_epilogue_kernel():
    return mx.fast.metal_kernel(
        name="convrot_mlx_mpp_w4a4_scale_bias_epilogue",
        input_names=["dots", "xscales", "wscales", "bias"],
        output_names=["out"],
        source=_MPP_W4A4_EPILOGUE_SOURCE,
        ensure_row_contiguous=True,
    )


def _mpp_w4a4_epilogue(
    dots: mx.array,
    xscales: mx.array,
    wscales: mx.array,
    bias: mx.array,
    *,
    output_dtype,
) -> mx.array:
    m, n = dots.shape
    total = m * n
    return _mpp_w4a4_epilogue_kernel()(
        inputs=[dots, xscales, wscales, bias],
        template=[("T", output_dtype)],
        output_shapes=[(m, n)],
        output_dtypes=[output_dtype],
        grid=(math.ceil(total / 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


def metal_unpack_signed_int4(packed: mx.array) -> mx.array:
    """Expand packed signed nibbles directly to int8 without int32 graph temporaries."""

    if packed.ndim != 2:
        raise ValueError(f"Metal int4 unpack expects a 2D tensor, got {packed.shape}")
    rows, packed_columns = packed.shape
    total = rows * packed_columns * 2
    return _unpack_signed_int4_kernel()(
        inputs=[packed],
        output_shapes=[(rows, packed_columns * 2)],
        output_dtypes=[mx.int8],
        grid=(math.ceil(total / 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


def _validate_mpp_weight_layout(qweight: mx.array, k: int) -> int:
    if qweight.ndim != 2:
        raise ValueError(f"MPP W4 weight must be 2D, got {qweight.shape}")
    if qweight.shape[0] != k:
        raise ValueError(
            f"MPP W4 weight must use K x packed-N layout; activation K={k}, weight={qweight.shape}"
        )
    if k % 32:
        raise ValueError(f"MPP W4 TensorOp requires K divisible by 32, got {k}")
    return qweight.shape[1] * 2


def _validate_mpp_epilogue(
    m: int,
    n: int,
    xscales: mx.array | None,
    wscales: mx.array,
    bias: mx.array,
) -> None:
    if xscales is not None and xscales.shape != (m,):
        raise ValueError(
            f"MPP activation scales must have shape {(m,)}, got {xscales.shape}"
        )
    if wscales.shape != (n,):
        raise ValueError(
            f"MPP weight scales must have shape {(n,)}, got {wscales.shape}"
        )
    if bias.shape != (n,):
        raise ValueError(f"MPP bias must have shape {(n,)}, got {bias.shape}")


def mpp_w4a4_matmul(
    qact: mx.array,
    qweight: mx.array,
    xscales: mx.array,
    wscales: mx.array,
    bias: mx.array,
    *,
    output_dtype,
) -> mx.array:
    """Run ConvRot's A4 values through native MPP signed-int8 x signed-int4."""

    if qact.ndim != 2:
        raise ValueError(f"MPP W4A4 activation must be 2D, got {qact.shape}")
    m, k = qact.shape[0], qact.shape[1] * 2
    n = _validate_mpp_weight_layout(qweight, k)
    _validate_mpp_epilogue(m, n, xscales, wscales, bias)
    activations = metal_unpack_signed_int4(qact)
    return mpp_w4a4_int8_matmul(
        activations,
        qweight,
        xscales,
        wscales,
        bias,
        output_dtype=output_dtype,
    )


def mpp_w4a4_int8_matmul(
    activations: mx.array,
    qweight: mx.array,
    xscales: mx.array,
    wscales: mx.array,
    bias: mx.array,
    *,
    output_dtype,
) -> mx.array:
    """Run already prepared A4 values held in int8 through MPP's native dot."""

    if activations.ndim != 2 or activations.dtype != mx.int8:
        raise ValueError(
            f"MPP W4A4 requires a 2D int8 activation, got {activations.shape} {activations.dtype}"
        )
    m, k = activations.shape
    n = _validate_mpp_weight_layout(qweight, k)
    _validate_mpp_epilogue(m, n, xscales, wscales, bias)
    dots = _mpp_w4a4_kernel(m, n, k)(
        inputs=[activations, qweight],
        output_shapes=[(m, n)],
        output_dtypes=[mx.int32],
        grid=(math.ceil(n / 32) * 32, math.ceil(m / 32), 1),
        threadgroup=(32, 1, 1),
        init_value=0,
    )[0]
    return _mpp_w4a4_epilogue(
        dots,
        xscales,
        wscales,
        bias,
        output_dtype=output_dtype,
    )


def mpp_w4a16_matmul(
    activations: mx.array,
    qweight: mx.array,
    wscales: mx.array,
    bias: mx.array,
    *,
    output_dtype,
) -> mx.array:
    """Run rotated FP16/BF16 activations through native MPP float x signed-int4."""

    if activations.ndim != 2:
        raise ValueError(f"MPP W4A16 activation must be 2D, got {activations.shape}")
    if activations.dtype not in {mx.float16, mx.bfloat16}:
        raise ValueError(
            f"MPP W4A16 requires float16 or bfloat16 activation, got {activations.dtype}"
        )
    m, k = activations.shape
    n = _validate_mpp_weight_layout(qweight, k)
    _validate_mpp_epilogue(m, n, None, wscales, bias)
    dots = _mpp_w4a16_kernel(m, n, k)(
        inputs=[activations, qweight],
        template=[("T", activations.dtype)],
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
        grid=(math.ceil(n / 32) * 32, math.ceil(m / 32), 1),
        threadgroup=(32, 1, 1),
        init_value=0,
    )[0]
    output = dots * wscales[None, :] + bias[None, :]
    return output.astype(output_dtype)


def reference_w4a4_matmul(
    qact: mx.array,
    qweight: mx.array,
    xscales: mx.array,
    wscales: mx.array,
    bias: mx.array,
    *,
    output_dtype,
) -> mx.array:
    """Portable eager oracle; intended for tests, not full-model inference."""

    activations = unpack_signed_int4(qact).astype(mx.float32)
    weights = unpack_signed_int4(qweight).astype(mx.float32)
    output = (activations @ weights.T) * xscales[:, None] * wscales[None, :] + bias[
        None, :
    ]
    return output.astype(output_dtype)


def expanded_fp32_w4a4_matmul(
    qact: mx.array,
    qweight: mx.array,
    xscales: mx.array,
    wscales: mx.array,
    bias: mx.array,
    *,
    output_dtype,
) -> mx.array:
    """Use MLX's tuned FP32 matmul after transient exact int4 expansion.

    Signed int4 values are exactly representable in FP32, and every possible
    FLUX linear dot remains below FP32's consecutive-integer limit.  The
    expanded operands are graph temporaries rather than persistent weights, so
    the on-disk and steady-state checkpoint remains packed W4.
    """

    if qact.ndim != 2 or qweight.ndim != 2:
        raise ValueError("expanded W4A4 expects 2D activation and weight tensors")
    if qact.shape[1] != qweight.shape[1]:
        raise ValueError(
            f"packed K mismatch: activation {qact.shape}, weight {qweight.shape}"
        )
    feature_count = qact.shape[1] * 2
    if feature_count * INT4_MAX * INT4_MAX >= 2**24:
        raise ValueError(
            f"K={feature_count} can exceed FP32's consecutive-integer dot range"
        )
    activations = unpack_signed_int4(qact).astype(mx.float32)
    weights = unpack_signed_int4(qweight).astype(mx.float32)
    output = (activations @ weights.T) * xscales[:, None] * wscales[None, :] + bias[
        None, :
    ]
    return output.astype(output_dtype)


class ConvRotLinear(nn.Module):
    """An MLX linear layer stored and executed under the ConvRot W4A4 contract."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        group_size: int = CONVROT_GROUP_SIZE,
        backend: str | None = None,
        preprocessor: str | None = None,
    ):
        super().__init__()
        if in_features % group_size:
            raise ValueError(
                f"in_features {in_features} not divisible by ConvRot group size {group_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.backend = backend or os.getenv("CONVROT_MLX_BACKEND", "mpp-w4a4")
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown ConvRot backend {self.backend!r}; expected one of {sorted(SUPPORTED_BACKENDS)}"
            )
        self.preprocessor = preprocessor or os.getenv(
            "CONVROT_MLX_PREPROCESSOR", "metal-fused"
        )
        if self.preprocessor not in SUPPORTED_PREPROCESSORS:
            raise ValueError(
                f"Unknown ConvRot preprocessor {self.preprocessor!r}; "
                f"expected one of {sorted(SUPPORTED_PREPROCESSORS)}"
            )
        if self.backend.startswith("mpp-"):
            if out_features % 2:
                raise ValueError(
                    f"MPP packed weights require even out_features, got {out_features}"
                )
            self.qweight = mx.zeros((in_features, out_features // 2), dtype=mx.uint8)
            self.weight_layout = "packed-kn-low-nibble-even-output"
        else:
            self.qweight = mx.zeros((out_features, in_features // 2), dtype=mx.uint8)
            self.weight_layout = "packed-nk-low-nibble-even-feature"
        self.scales = mx.ones((out_features,), dtype=mx.float32)
        # A zero vector gives the kernel one fixed signature for biasless and
        # biased linears. Conversion preserves real biases and emits zeros only
        # for a genuinely biasless source layer.
        self.bias = mx.zeros((out_features,), dtype=mx.float32)
        self.source_has_bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"ConvRotLinear expected K={self.in_features}, got {x.shape[-1]}"
            )
        original_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        if self.backend == "mpp-w4a16":
            rotated = metal_fused_rotate_convrot_activation(x2d, self.group_size)
            output = mpp_w4a16_matmul(
                rotated,
                self.qweight,
                self.scales,
                self.bias,
                output_dtype=x.dtype,
            )
            return output.reshape(*original_shape[:-1], self.out_features)
        if self.backend == "mpp-w4a4":
            qact_i8, xscales = metal_fused_prepare_convrot_activation_i8(
                x2d, self.group_size
            )
            output = mpp_w4a4_int8_matmul(
                qact_i8,
                self.qweight,
                xscales,
                self.scales,
                self.bias,
                output_dtype=x.dtype,
            )
            return output.reshape(*original_shape[:-1], self.out_features)

        prepare = {
            "mlx-compiled": compiled_prepare_convrot_activation,
            "metal-fused": metal_fused_prepare_convrot_activation,
        }[self.preprocessor]
        qact, xscales = prepare(x2d, self.group_size)
        matmul = {
            "metal": packed_w4a4_matmul,
            "metal-k32": packed_w4a4_matmul_k32,
            "metal-k64": packed_w4a4_matmul_k64,
            "metal-m32n32-k64": packed_w4a4_matmul_m32n32_k64,
            "mlx-expanded-fp32": expanded_fp32_w4a4_matmul,
            "reference": reference_w4a4_matmul,
        }[self.backend]
        output = matmul(
            qact,
            self.qweight,
            xscales,
            self.scales,
            self.bias,
            output_dtype=x.dtype,
        )
        return output.reshape(*original_shape[:-1], self.out_features)


def eligible_linear(module: object, group_size: int = CONVROT_GROUP_SIZE) -> bool:
    return (
        isinstance(module, nn.Linear)
        and module.weight.ndim == 2
        and module.weight.shape[1] % group_size == 0
    )
