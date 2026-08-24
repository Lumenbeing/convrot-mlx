#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure ConvRot activation preparation separately from its packed dot.

The breakdown separates Hadamard rotation, row quantization, and the packed
dot so kernel work can target the dominant cost.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from convrot_mlx import (
    compiled_prepare_convrot_activation,
    metal_fused_prepare_convrot_activation,
    pack_signed_int4,
    packed_w4a4_matmul_k64,
    prepare_convrot_activation,
    quantize_signed_int4_rowwise,
    regular_hadamard,
)


def preprocess(x: mx.array) -> tuple[mx.array, mx.array]:
    return prepare_convrot_activation(x)


def timed(function, *arguments: mx.array, iterations: int) -> dict[str, object]:
    warmup = function(*arguments)
    mx.eval(warmup)
    mx.clear_cache()
    samples = []
    peaks = []
    for _ in range(iterations):
        mx.reset_peak_memory()
        started = time.perf_counter()
        result = function(*arguments)
        mx.eval(result)
        samples.append(time.perf_counter() - started)
        peaks.append(mx.get_peak_memory() / 1024**3)
        del result
        mx.clear_cache()
    return {
        "seconds": [round(sample, 6) for sample in samples],
        "bestSeconds": round(min(samples), 6),
        "peakGiB": round(max(peaks), 6),
    }


def benchmark(m: int, k: int, n: int, iterations: int) -> dict[str, object]:
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    qweight = pack_signed_int4(
        mx.random.randint(-7, 8, shape=(n, k), dtype=mx.int32).astype(mx.int8),
    )
    wscales = mx.full((n,), 0.125, dtype=mx.float32)
    bias = mx.zeros((n,), dtype=mx.float32)
    mx.eval(x, qweight, wscales, bias)

    rotated = regular_hadamard(x)
    mx.eval(rotated)
    qact, xscales = preprocess(x)
    mx.eval(qact, xscales)

    def quantize_only(value: mx.array):
        return quantize_signed_int4_rowwise(value)

    def dot_only(qvalues: mx.array, scales: mx.array):
        return packed_w4a4_matmul_k64(
            qvalues,
            qweight,
            scales,
            wscales,
            bias,
            output_dtype=mx.bfloat16,
        )

    def full_layer(value: mx.array):
        packed, scales = preprocess(value)
        return dot_only(packed, scales)

    result = {
        "shape": {"m": m, "k": k, "n": n},
        "hadamard": timed(regular_hadamard, x, iterations=iterations),
        "quantizeRotated": timed(quantize_only, rotated, iterations=iterations),
        "preprocessEager": timed(preprocess, x, iterations=iterations),
        "preprocessCompiled": timed(
            compiled_prepare_convrot_activation, x, iterations=iterations
        ),
        "preprocessMetalFused": timed(
            metal_fused_prepare_convrot_activation, x, iterations=iterations
        ),
        "packedDotK64": timed(dot_only, qact, xscales, iterations=iterations),
        "fullLayer": timed(full_layer, x, iterations=iterations),
    }
    eager = result["preprocessEager"]["bestSeconds"]
    compiled = result["preprocessCompiled"]["bestSeconds"]
    fused = result["preprocessMetalFused"]["bestSeconds"]
    full = result["fullLayer"]["bestSeconds"]
    result["derived"] = {
        "compiledPreprocessSpeedup": round(eager / compiled, 3),
        "metalFusedSpeedupVsCompiled": round(compiled / fused, 3),
        "preprocessShareOfFullPercent": round(eager / full * 100, 3),
        "packedDotShareOfFullPercent": round(
            result["packedDotK64"]["bestSeconds"] / full * 100, 3
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1536)
    parser.add_argument("--k", type=int, default=3072)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if min(args.m, args.k, args.n, args.iterations) < 1 or args.k % 256:
        raise ValueError(
            "dimensions and iterations must be positive; K must be divisible by 256"
        )
    print(
        json.dumps(
            {
                "schema": "convrot-mlx.convrot-preprocessing-benchmark.v1",
                "mlxVersion": mx.__version__,
                "benchmark": benchmark(args.m, args.k, args.n, args.iterations),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
