#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Comfy Org. All rights reserved.
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0
"""Convert an authorized FLUX.1 Dev transformer to packed ConvRot W4A4."""

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
from flux.model import Flux
from flux.utils import configs
from mlx import nn
from mlx.utils import tree_flatten

from convrot_mlx import CONVROT_GROUP_SIZE, quantize_convrot_weight

from .adapter import CONVROT_SCHEMA, SOURCE_SHA256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_for(path: Path) -> dict[str, str]:
    _, metadata = mx.load(str(path), return_metadata=True)
    return metadata


def inspect(path: Path) -> None:
    metadata = metadata_for(path)
    required = {
        "schema": CONVROT_SCHEMA,
        "source_sha256": SOURCE_SHA256,
        "convrot_group_size": str(CONVROT_GROUP_SIZE),
        "weight_bits": "4",
        "activation_bits": "4",
        "weight_scale": "rowwise_absmax_div_7",
        "activation_scale": "rowwise_absmax_div_7",
        "packing": "signed_twos_complement_low_nibble_even_column",
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"{path} metadata {name!r} is {metadata.get(name)!r}; expected {expected!r}"
            )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def convert(
    source: Path,
    output: Path,
    receipt: Path,
    runtime_revision: str,
    skip_source_hash: bool,
    logical_output: Path | None,
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing ConvRot checkpoint: {output}"
        )
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing conversion receipt: {receipt}"
        )
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(
            f"Expected a regular, non-symlinked FLUX source: {source}"
        )
    if source.stat().st_size != 23_802_932_552:
        raise ValueError(f"FLUX source byte count changed: {source.stat().st_size}")
    if not skip_source_hash:
        print("CONVROT_MLX_PROGRESS 2 hashing exact FLUX.1 Dev source", flush=True)
        observed = sha256(source)
        if observed != SOURCE_SHA256:
            raise ValueError(f"FLUX source SHA-256 mismatch: {observed}")

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if free < 9 * 1024**3:
        raise RuntimeError(
            f"ConvRot conversion needs at least 9 GiB free beside {output}; found {free / 1024**3:.2f} GiB"
        )

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    print("CONVROT_MLX_PROGRESS 5 opening original BFL safetensors", flush=True)
    flow = Flux(configs["flux-dev"].params)
    source_weights = flow.sanitize(mx.load(str(source)))
    expected = {name for name, _ in tree_flatten(flow.parameters())}
    missing = sorted(expected - source_weights.keys())
    unexpected = sorted(source_weights.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"Source topology differs from Apple's pinned FLUX Dev model; missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    linears: list[tuple[str, nn.Linear]] = [
        (name, module)
        for name, module in flow.named_modules()
        if isinstance(module, nn.Linear)
    ]
    selected = [
        (name, module)
        for name, module in linears
        if module.weight.shape[1] % CONVROT_GROUP_SIZE == 0
    ]
    output_weights: dict[str, mx.array] = {}
    converted_shapes: dict[str, list[int]] = {}

    for index, (name, module) in enumerate(selected, start=1):
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        weight = source_weights.pop(weight_key)
        qweight, scales = quantize_convrot_weight(weight)
        mx.eval(qweight, scales)
        output_weights[f"{name}.qweight"] = qweight
        output_weights[f"{name}.scales"] = scales
        if bias_key in source_weights:
            bias = source_weights.pop(bias_key).astype(mx.float32)
        else:
            bias = mx.zeros((weight.shape[0],), dtype=mx.float32)
        mx.eval(bias)
        output_weights[bias_key] = bias
        converted_shapes[name] = [int(weight.shape[0]), int(weight.shape[1])]
        del weight
        mx.clear_cache()
        progress = 5 + round(index / len(selected) * 82)
        print(
            f"CONVROT_MLX_PROGRESS {progress} ConvRot {index}/{len(selected)} {name}",
            flush=True,
        )

    # The few ineligible projections (notably img_in's 64-wide input) stay in
    # their exact source dtype. Every source key must be consumed exactly once.
    output_weights.update(source_weights)
    flow = None
    source_weights = None
    mx.clear_cache()

    metadata = {
        "schema": CONVROT_SCHEMA,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": "23802932552",
        "source_provenance": "User-supplied authorized FLUX.1 Dev source; canonical digest also matches argmaxinc/mlx-FLUX.1-dev@08a867181c83019013dd0f991a76853fe772f16b",
        "runtime_repository": "ml-explore/mlx-examples",
        "runtime_revision": runtime_revision,
        "reference_repository": "Comfy-Org/comfy-kitchen",
        "reference_revision": "7d86acf60c88fd6c3c733c0e54db22ef74b8d77f",
        "reference_algorithm": "plain ConvRot W4A4 eager contract",
        "convrot_group_size": str(CONVROT_GROUP_SIZE),
        "quant_group_size": "64",
        "weight_bits": "4",
        "activation_bits": "4",
        "weight_scale": "rowwise_absmax_div_7",
        "activation_scale": "rowwise_absmax_div_7",
        "packing": "signed_twos_complement_low_nibble_even_column",
        "converted_linear_count": str(len(selected)),
        "linear_count": str(len(linears)),
        "metal_execution": "packed operands; per-tile int4-to-half expansion; simdgroup half MMA; float accumulation",
        "native_int4_mma": "false",
    }
    print("CONVROT_MLX_PROGRESS 90 writing packed checkpoint", flush=True)
    mx.save_safetensors(str(output), output_weights, metadata=metadata)
    mx.synchronize()
    output_weights = None
    mx.clear_cache()

    print("CONVROT_MLX_PROGRESS 95 hashing packed checkpoint", flush=True)
    output_hash = sha256(output)
    completed_at = datetime.now(UTC).isoformat()
    evidence = {
        "schema": "convrot-mlx.flux1-convrot-conversion.v1",
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": SOURCE_SHA256,
        },
        "output": {
            "path": str(logical_output or output),
            "bytes": output.stat().st_size,
            "sha256": output_hash,
        },
        "algorithm": metadata,
        "convertedLayers": converted_shapes,
        "startedAt": started_at,
        "completedAt": completed_at,
        "elapsedSeconds": round(time.perf_counter() - started, 3),
    }
    receipt.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print("CONVROT_MLX_PROGRESS 100 conversion complete", flush=True)
    print(json.dumps(evidence, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--runtime-revision", default="796f5b53cab69a3d48a44233ce21aae889e94a08"
    )
    parser.add_argument("--logical-output", type=Path)
    parser.add_argument("--skip-source-hash", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        inspect(args.output)
        return
    if args.source is None:
        parser.error("--source is required unless --inspect-only is used")
    receipt = args.receipt or args.output.with_name("conversion.json")
    convert(
        args.source.resolve(),
        args.output.resolve(),
        receipt.resolve(),
        args.runtime_revision,
        args.skip_source_hash,
        args.logical_output.resolve() if args.logical_output else None,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FLUX ConvRot conversion failed: {error}", file=sys.stderr)
        raise
