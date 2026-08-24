#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Metal 4 MPP W4A4 and W4A16 ConvRot paths at FLUX shapes.

Each result includes the kernel-only cost, the complete activation-preparation
plus linear cost, and both total and incremental MLX peak memory. The dense
BF16 control uses the same M x K x N geometry without rotation or quantization.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx

from convrot_mlx import (
    compiled_regular_hadamard,
    metal_fused_prepare_convrot_activation,
    metal_fused_prepare_convrot_activation_i8,
    metal_fused_rotate_convrot_activation,
    mpp_w4a4_int8_matmul,
    mpp_w4a4_matmul,
    mpp_w4a16_matmul,
    pack_signed_int4,
    packed_w4a4_matmul_m32n32_k64,
    reference_w4a4_matmul,
    repack_nk_weight_for_mpp,
    unpack_signed_int4,
)

GIB = 1024**3
MIB = 1024**2


def random_packed(rows: int, columns: int) -> mx.array:
    values = mx.random.randint(-7, 8, shape=(rows, columns), dtype=mx.int32).astype(
        mx.int8
    )
    packed = pack_signed_int4(values)
    mx.eval(packed)
    del values
    mx.clear_cache()
    return packed


def timed(function, *, iterations: int) -> dict[str, object]:
    warmup = function()
    mx.eval(*warmup) if isinstance(warmup, (tuple, list)) else mx.eval(warmup)
    del warmup
    mx.clear_cache()

    seconds: list[float] = []
    total_peaks: list[float] = []
    incremental_peaks: list[float] = []
    for _ in range(iterations):
        mx.clear_cache()
        active_before = mx.get_active_memory()
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = function()
        mx.eval(*output) if isinstance(output, (tuple, list)) else mx.eval(output)
        elapsed = time.perf_counter() - started
        peak = mx.get_peak_memory()
        seconds.append(elapsed)
        total_peaks.append(peak / GIB)
        incremental_peaks.append(max(0, peak - active_before) / MIB)
        del output
        mx.clear_cache()
    return {
        "seconds": [round(value, 6) for value in seconds],
        "bestSeconds": round(min(seconds), 6),
        "medianSeconds": round(statistics.median(seconds), 6),
        "totalPeakGiB": round(max(total_peaks), 6),
        "incrementalPeakMiB": round(max(incremental_peaks), 3),
    }


def correctness_gate() -> dict[str, object]:
    m, k, n = 37, 256, 46
    qact = random_packed(m, k)
    qweight_nk = random_packed(n, k)
    qweight_mpp = repack_nk_weight_for_mpp(qweight_nk)
    xscales = mx.random.uniform(low=0.01, high=0.2, shape=(m,)).astype(mx.float32)
    wscales = mx.random.uniform(low=0.01, high=0.2, shape=(n,)).astype(mx.float32)
    bias = mx.random.uniform(low=-1, high=1, shape=(n,)).astype(mx.float32)
    expected_a4 = reference_w4a4_matmul(
        qact,
        qweight_nk,
        xscales,
        wscales,
        bias,
        output_dtype=mx.float32,
    )
    actual_a4 = mpp_w4a4_matmul(
        qact,
        qweight_mpp,
        xscales,
        wscales,
        bias,
        output_dtype=mx.float32,
    )
    unit_xscales = mx.ones((m,), dtype=mx.float32)
    unit_wscales = mx.ones((n,), dtype=mx.float32)
    zero_bias = mx.zeros((n,), dtype=mx.float32)
    expected_integer = reference_w4a4_matmul(
        qact,
        qweight_nk,
        unit_xscales,
        unit_wscales,
        zero_bias,
        output_dtype=mx.float32,
    )
    actual_integer = mpp_w4a4_matmul(
        qact,
        qweight_mpp,
        unit_xscales,
        unit_wscales,
        zero_bias,
        output_dtype=mx.float32,
    )

    activation = mx.random.normal(shape=(m, k)).astype(mx.bfloat16)
    unpacked_weight = unpack_signed_int4(qweight_nk).astype(mx.float32)
    expected_a16 = (activation.astype(mx.float32) @ unpacked_weight.T) * wscales[
        None, :
    ] + bias[None, :]
    actual_a16 = mpp_w4a16_matmul(
        activation,
        qweight_mpp,
        wscales,
        bias,
        output_dtype=mx.float32,
    )
    mx.eval(
        expected_a4,
        actual_a4,
        expected_integer,
        actual_integer,
        expected_a16,
        actual_a16,
    )
    integer_exact = bool(mx.all(expected_integer == actual_integer).item())
    a4_max = float(mx.max(mx.abs(expected_a4 - actual_a4)).item())
    a16_max = float(mx.max(mx.abs(expected_a16 - actual_a16)).item())
    if not integer_exact:
        raise RuntimeError("MPP W4A4 failed the exact integer-dot gate")
    if a4_max > 1e-5:
        raise RuntimeError(
            f"MPP W4A4 failed the FP32 epilogue gate: maximum error {a4_max}"
        )
    if a16_max > 1e-5:
        raise RuntimeError(f"MPP W4A16 failed the FP32 gate: maximum error {a16_max}")
    return {
        "mppW4A4ExactIntegerDot": integer_exact,
        "mppW4A4Float32EpilogueMaximumAbsoluteError": a4_max,
        "mppW4A4Float32EpilogueTolerance": 1e-5,
        "mppW4A16MaximumAbsoluteError": a16_max,
    }


def benchmark_shape(m: int, k: int, n: int, *, iterations: int) -> dict[str, object]:
    activation = mx.random.normal(shape=(m, k)).astype(mx.bfloat16)
    dense_weight = mx.random.normal(shape=(n, k)).astype(mx.bfloat16)
    qweight_nk = random_packed(n, k)
    qweight_mpp = repack_nk_weight_for_mpp(qweight_nk)
    wscales = mx.full((n,), 0.125, dtype=mx.float32)
    bias = mx.zeros((n,), dtype=mx.float32)
    mx.eval(activation, dense_weight, qweight_mpp, wscales, bias)

    qact, xscales = metal_fused_prepare_convrot_activation(activation)
    qact_i8, xscales_i8 = metal_fused_prepare_convrot_activation_i8(activation)
    rotated = metal_fused_rotate_convrot_activation(activation, 256)
    mx.eval(qact, xscales, qact_i8, xscales_i8, rotated)

    dense = timed(
        lambda: activation @ dense_weight.T + bias.astype(mx.bfloat16)[None, :],
        iterations=iterations,
    )
    legacy_kernel = timed(
        lambda: packed_w4a4_matmul_m32n32_k64(
            qact,
            qweight_nk,
            xscales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        ),
        iterations=iterations,
    )
    mpp_a4_kernel = timed(
        lambda: mpp_w4a4_int8_matmul(
            qact_i8,
            qweight_mpp,
            xscales_i8,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        ),
        iterations=iterations,
    )
    mpp_a16_kernel = timed(
        lambda: mpp_w4a16_matmul(
            rotated,
            qweight_mpp,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        ),
        iterations=iterations,
    )
    a4_preparation = timed(
        lambda: metal_fused_prepare_convrot_activation(activation),
        iterations=iterations,
    )
    rotation = timed(
        lambda: compiled_regular_hadamard(activation, 256),
        iterations=iterations,
    )
    fused_rotation = timed(
        lambda: metal_fused_rotate_convrot_activation(activation, 256),
        iterations=iterations,
    )
    mpp_a4_preparation = timed(
        lambda: metal_fused_prepare_convrot_activation_i8(activation),
        iterations=iterations,
    )

    def legacy_complete_path():
        prepared, scales = metal_fused_prepare_convrot_activation(activation)
        return packed_w4a4_matmul_m32n32_k64(
            prepared,
            qweight_nk,
            scales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )

    legacy_path = timed(legacy_complete_path, iterations=iterations)

    def mpp_a4_path():
        prepared, scales = metal_fused_prepare_convrot_activation_i8(activation)
        return mpp_w4a4_int8_matmul(
            prepared,
            qweight_mpp,
            scales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )

    def mpp_a16_path():
        prepared = metal_fused_rotate_convrot_activation(activation, 256)
        return mpp_w4a16_matmul(
            prepared,
            qweight_mpp,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )

    mpp_a4_path_result = timed(mpp_a4_path, iterations=iterations)
    mpp_a16_path_result = timed(mpp_a16_path, iterations=iterations)

    for result in (
        legacy_kernel,
        mpp_a4_kernel,
        mpp_a16_kernel,
        legacy_path,
        mpp_a4_path_result,
        mpp_a16_path_result,
    ):
        result["speedupVsDenseBf16"] = round(
            dense["medianSeconds"] / result["medianSeconds"], 3
        )
    mpp_a4_kernel["speedupVsLegacyKernel"] = round(
        legacy_kernel["medianSeconds"] / mpp_a4_kernel["medianSeconds"],
        3,
    )
    mpp_a16_kernel["speedupVsLegacyKernel"] = round(
        legacy_kernel["medianSeconds"] / mpp_a16_kernel["medianSeconds"],
        3,
    )
    mpp_a4_path_result["speedupVsLegacyPath"] = round(
        legacy_path["medianSeconds"] / mpp_a4_path_result["medianSeconds"],
        3,
    )
    mpp_a16_path_result["speedupVsLegacyPath"] = round(
        legacy_path["medianSeconds"] / mpp_a16_path_result["medianSeconds"],
        3,
    )
    return {
        "shape": {"m": m, "k": k, "n": n},
        "denseBf16": dense,
        "activationPreparation": {
            "fusedHadamardAndA4Pack": a4_preparation,
            "fusedHadamardAndA4Int8": mpp_a4_preparation,
            "compiledHadamardOnlyControl": rotation,
            "fusedHadamardOnly": fused_rotation,
        },
        "kernelOnly": {
            "legacyPackedMetalM32N32K64": legacy_kernel,
            "mppW4A4": mpp_a4_kernel,
            "mppW4A16": mpp_a16_kernel,
        },
        "completeLinearPath": {
            "legacyPackedMetalM32N32K64": legacy_path,
            "mppW4A4": mpp_a4_path_result,
            "mppW4A16": mpp_a16_path_result,
        },
    }


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        m, k, n = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must be MxKxN") from error
    if min(m, k, n) < 1 or k % 256 or n % 2:
        raise argparse.ArgumentTypeError(
            "dimensions must be positive, K divisible by 256, and N even"
        )
    return m, k, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.output and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f"Refusing to overwrite benchmark receipt: {args.output}")
    shapes = args.shape or [
        (1536, 3072, 3072),
        (1536, 3072, 21504),
        (1536, 21504, 3072),
    ]
    mx.random.seed(20260823)
    receipt = {
        "schema": "convrot-mlx.convrot-mpp-tensorops-benchmark.v2",
        "platform": platform.platform(),
        "mlxVersion": mx.__version__,
        "iterations": args.iterations,
        "correctness": correctness_gate(),
        "benchmarks": [
            benchmark_shape(*shape, iterations=args.iterations) for shape in shapes
        ],
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
