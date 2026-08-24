#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0
"""Losslessly repack a receipted ConvRot checkpoint for MPP TensorOps.

The source stores each signed-int4 weight as N x packed-K.  Metal 4 MPP's
int4 matmul consumes a logical K x N right operand packed along N.  This script
transposes the unpacked signed nibbles and repacks them without requantizing or
changing scales, biases, or any unconverted tensor.  It never overwrites input
or output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from convrot_mlx import (
    CONVROT_GROUP_SIZE,
    repack_mpp_weight_to_nk,
    repack_nk_weight_for_mpp,
)

from .adapter import (
    CONVROT_MPP_SCHEMA,
    CONVROT_MPP_WEIGHT_LAYOUT,
    CONVROT_SCHEMA,
    SOURCE_SHA256,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(path: Path, metadata: dict[str, str]) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"Expected a regular, non-symlinked ConvRot checkpoint: {path}"
        )
    required = {
        "schema": CONVROT_SCHEMA,
        "source_sha256": SOURCE_SHA256,
        "convrot_group_size": str(CONVROT_GROUP_SIZE),
        "weight_bits": "4",
        "activation_bits": "4",
        "packing": "signed_twos_complement_low_nibble_even_column",
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"Source metadata {name!r} is {metadata.get(name)!r}; expected {expected!r}"
            )


def inspect(path: Path) -> None:
    _, metadata = mx.load(str(path), return_metadata=True)
    required = {
        "schema": CONVROT_MPP_SCHEMA,
        "source_sha256": SOURCE_SHA256,
        "convrot_group_size": str(CONVROT_GROUP_SIZE),
        "weight_bits": "4",
        "weight_layout": CONVROT_MPP_WEIGHT_LAYOUT,
        "lossless_layout_repack": "true",
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"{path} metadata {name!r} is {metadata.get(name)!r}; expected {expected!r}"
            )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def repack(
    source: Path, output: Path, receipt: Path, *, skip_source_hash: bool
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite MPP checkpoint: {output}")
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite MPP repack receipt: {receipt}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 8 * 1024**3:
        raise RuntimeError("MPP repack requires at least 8 GiB free beside the output")

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    print("CONVROT_MLX_PROGRESS 2 opening packed ConvRot checkpoint", flush=True)
    source_weights, source_metadata = mx.load(str(source), return_metadata=True)
    validate_source(source, source_metadata)
    print("CONVROT_MLX_PROGRESS 4 hashing packed ConvRot source", flush=True)
    source_hash = sha256(source) if not skip_source_hash else "skipped"

    qweight_keys = sorted(name for name in source_weights if name.endswith(".qweight"))
    expected_count = int(source_metadata.get("converted_linear_count", "-1"))
    if len(qweight_keys) != expected_count:
        raise ValueError(
            f"Found {len(qweight_keys)} qweights; metadata expects {expected_count}"
        )

    output_weights: dict[str, mx.array] = {}
    shapes: dict[str, dict[str, list[int]]] = {}
    for index, name in enumerate(qweight_keys, start=1):
        packed_nk = source_weights.pop(name)
        packed_kn = repack_nk_weight_for_mpp(packed_nk)
        round_trip = repack_mpp_weight_to_nk(packed_kn)
        exact = mx.all(round_trip == packed_nk)
        mx.eval(packed_kn, exact)
        if not bool(exact.item()):
            raise RuntimeError(f"Bit-exact MPP repack gate failed for {name}")
        output_weights[name] = packed_kn
        shapes[name] = {
            "sourcePackedNK": [int(value) for value in packed_nk.shape],
            "mppPackedKN": [int(value) for value in packed_kn.shape],
        }
        del packed_nk, round_trip, exact
        mx.clear_cache()
        progress = 5 + round(index / len(qweight_keys) * 80)
        print(
            f"CONVROT_MLX_PROGRESS {progress} MPP repack {index}/{len(qweight_keys)} {name}",
            flush=True,
        )

    output_weights.update(source_weights)
    source_weights = None
    mx.clear_cache()
    metadata = dict(source_metadata)
    metadata.update(
        {
            "schema": CONVROT_MPP_SCHEMA,
            "source_convrot_sha256": source_hash,
            "weight_layout": CONVROT_MPP_WEIGHT_LAYOUT,
            "packing": "signed_twos_complement_low_nibble_even_output",
            "activation_bits": "4-or-16-runtime-selected",
            "supported_activation_modes": "a4-values-in-int8,bfloat16,float16",
            "metal_execution": "Metal 4 MPP TensorOps int8-or-float x packed signed-int4",
            "native_int4_mma": "true",
            "lossless_layout_repack": "true",
        }
    )

    print("CONVROT_MLX_PROGRESS 88 writing MPP-layout checkpoint", flush=True)
    mx.save_safetensors(str(output), output_weights, metadata=metadata)
    mx.synchronize()
    output_weights = None
    mx.clear_cache()

    print("CONVROT_MLX_PROGRESS 95 hashing MPP-layout checkpoint", flush=True)
    output_hash = sha256(output)
    completed_at = datetime.now(UTC).isoformat()
    evidence = {
        "schema": "convrot-mlx.flux1-convrot-mpp-repack.v1",
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": source_hash,
        },
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": output_hash,
        },
        "algorithm": metadata,
        "repackedLinearCount": len(qweight_keys),
        "bitExactRoundTrip": True,
        "repackedShapes": shapes,
        "startedAt": started_at,
        "completedAt": completed_at,
        "elapsedSeconds": round(time.perf_counter() - started, 3),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print("CONVROT_MLX_PROGRESS 100 MPP repack complete", flush=True)
    print(json.dumps(evidence, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--skip-source-hash", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        inspect(args.output.resolve())
        return
    if args.source is None:
        parser.error("--source is required unless --inspect-only is used")
    output = args.output.resolve()
    receipt = (args.receipt or output.with_name("repack.json")).resolve()
    repack(
        args.source.resolve(), output, receipt, skip_source_hash=args.skip_source_hash
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FLUX ConvRot MPP repack failed: {error}", file=sys.stderr)
        raise
