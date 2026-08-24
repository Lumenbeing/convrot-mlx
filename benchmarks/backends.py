#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare exact ConvRot W4A4 execution strategies on representative shapes.

The transient-expanded candidate keeps the checkpoint and activations packed
at W4A4, expands one layer at a time to exactly represented FP32 integers, and
delegates the dot product to MLX's tuned matrix kernel. FP32 preserves every
possible signed int4 dot for the tested feature widths before row scaling.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from convrot_mlx import (
    expanded_fp32_w4a4_matmul,
    pack_signed_int4,
    packed_w4a4_matmul,
    packed_w4a4_matmul_k32,
    packed_w4a4_matmul_k64,
    packed_w4a4_matmul_k128,
    packed_w4a4_matmul_k256,
    packed_w4a4_matmul_m16n64_k64,
    packed_w4a4_matmul_m32n32_k64,
    reference_w4a4_matmul,
)


def random_packed(rows: int, columns: int) -> mx.array:
    values = mx.random.randint(-7, 8, shape=(rows, columns), dtype=mx.int32).astype(
        mx.int8
    )
    packed = pack_signed_int4(values)
    mx.eval(packed)
    return packed


def timed(function, arguments: tuple, *, iterations: int) -> dict[str, object]:
    warmup = function(*arguments, output_dtype=mx.bfloat16)
    mx.eval(warmup)
    mx.clear_cache()

    samples = []
    peaks = []
    for _ in range(iterations):
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = function(*arguments, output_dtype=mx.bfloat16)
        mx.eval(output)
        samples.append(time.perf_counter() - started)
        peaks.append(mx.get_peak_memory() / 1024**3)
        del output
        mx.clear_cache()
    return {
        "seconds": [round(sample, 6) for sample in samples],
        "bestSeconds": round(min(samples), 6),
        "peakGiB": round(max(peaks), 6),
    }


def benchmark_shape(m: int, k: int, n: int, *, iterations: int) -> dict[str, object]:
    qact = random_packed(m, k)
    qweight = random_packed(n, k)
    xscales = mx.full((m,), 0.125, dtype=mx.float32)
    wscales = mx.full((n,), 0.125, dtype=mx.float32)
    bias = mx.zeros((n,), dtype=mx.float32)
    mx.eval(xscales, wscales, bias)

    metal = timed(
        packed_w4a4_matmul,
        (qact, qweight, xscales, wscales, bias),
        iterations=iterations,
    )
    expanded = timed(
        expanded_fp32_w4a4_matmul,
        (qact, qweight, xscales, wscales, bias),
        iterations=iterations,
    )
    metal_k32 = timed(
        packed_w4a4_matmul_k32,
        (qact, qweight, xscales, wscales, bias),
        iterations=iterations,
    )
    metal_k64 = timed(
        packed_w4a4_matmul_k64,
        (qact, qweight, xscales, wscales, bias),
        iterations=iterations,
    )
    metal_k128 = timed(
        packed_w4a4_matmul_k128,
        (qact, qweight, xscales, wscales, bias),
        iterations=iterations,
    )
    metal_k256 = timed(
        packed_w4a4_matmul_k256,
        (qact, qweight, xscales, wscales, bias),
        iterations=iterations,
    )
    metal_m32n32_k64 = timed(
        packed_w4a4_matmul_m32n32_k64,
        (qact, qweight, xscales, wscales, bias),
        iterations=iterations,
    )
    metal_m16n64_k64 = timed(
        packed_w4a4_matmul_m16n64_k64,
        (qact, qweight, xscales, wscales, bias),
        iterations=iterations,
    )
    expanded["speedupVsPackedMetal"] = round(
        metal["bestSeconds"] / expanded["bestSeconds"], 3
    )
    metal_k32["speedupVsPackedMetal"] = round(
        metal["bestSeconds"] / metal_k32["bestSeconds"], 3
    )
    metal_k64["speedupVsPackedMetal"] = round(
        metal["bestSeconds"] / metal_k64["bestSeconds"], 3
    )
    metal_k128["speedupVsPackedMetal"] = round(
        metal["bestSeconds"] / metal_k128["bestSeconds"], 3
    )
    metal_k256["speedupVsPackedMetal"] = round(
        metal["bestSeconds"] / metal_k256["bestSeconds"], 3
    )
    metal_m32n32_k64["speedupVsPackedMetal"] = round(
        metal["bestSeconds"] / metal_m32n32_k64["bestSeconds"], 3
    )
    metal_m32n32_k64["speedupVsK64"] = round(
        metal_k64["bestSeconds"] / metal_m32n32_k64["bestSeconds"], 3
    )
    metal_m16n64_k64["speedupVsPackedMetal"] = round(
        metal["bestSeconds"] / metal_m16n64_k64["bestSeconds"], 3
    )
    metal_m16n64_k64["speedupVsK64"] = round(
        metal_k64["bestSeconds"] / metal_m16n64_k64["bestSeconds"], 3
    )
    return {
        "shape": {"m": m, "k": k, "n": n},
        "packedMetal": metal,
        "packedMetalK32": metal_k32,
        "packedMetalK64": metal_k64,
        "packedMetalK128": metal_k128,
        "packedMetalK256": metal_k256,
        "packedMetalM32N32K64": metal_m32n32_k64,
        "packedMetalM16N64K64": metal_m16n64_k64,
        "expandedFp32": expanded,
    }


def correctness_gate() -> dict[str, object]:
    m, k, n = 19, 256, 37
    qact = random_packed(m, k)
    qweight = random_packed(n, k)
    xscales = mx.random.uniform(low=0.01, high=0.2, shape=(m,)).astype(mx.float32)
    wscales = mx.random.uniform(low=0.01, high=0.2, shape=(n,)).astype(mx.float32)
    bias = mx.random.uniform(low=-1, high=1, shape=(n,)).astype(mx.float32)
    expected = reference_w4a4_matmul(
        qact,
        qweight,
        xscales,
        wscales,
        bias,
        output_dtype=mx.bfloat16,
    )
    actual = expanded_fp32_w4a4_matmul(
        qact,
        qweight,
        xscales,
        wscales,
        bias,
        output_dtype=mx.bfloat16,
    )
    mx.eval(expected, actual)
    equal = bool(mx.all(expected == actual).item())
    maximum_error = float(
        mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))).item()
    )
    if not equal:
        raise RuntimeError(
            f"Expanded FP32 backend failed the exact BF16 output gate (max error {maximum_error})"
        )
    return {"exactBfloat16Output": equal, "maximumAbsoluteError": maximum_error}


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        m, k, n = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must be MxKxN") from error
    if min(m, k, n) < 1 or k % 256:
        raise argparse.ArgumentTypeError(
            "dimensions must be positive and K must be divisible by 256"
        )
    return m, k, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    shapes = args.shape or [(1536, 3072, 3072), (1536, 3072, 21504)]
    if args.iterations < 1:
        raise ValueError("iterations must be positive")

    receipt = {
        "schema": "convrot-mlx.convrot-backend-benchmark.v1",
        "mlxVersion": mx.__version__,
        "correctness": correctness_gate(),
        "benchmarks": [
            benchmark_shape(*shape, iterations=args.iterations) for shape in shapes
        ],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
