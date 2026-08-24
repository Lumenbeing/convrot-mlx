#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 Apple Inc.
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: MIT
"""Generate a FLUX.1 Dev image through dense BF16 or ConvRot W4A4 MLX."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .adapter import build_pipeline

MLX_EXAMPLES_REVISION = "796f5b53cab69a3d48a44233ce21aae889e94a08"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def progress(value: int, phase: str) -> None:
    print(f"CONVROT_MLX_PROGRESS {value} {phase}", flush=True)


def require_local_files(paths: dict[str, Path]) -> None:
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing local {role}: {path}")


def tensor_stats(label: str, value: mx.array) -> dict[str, float | int | list[int]]:
    """Evaluate a tensor and refuse silent NaN-to-black image conversion."""

    value_f32 = value.astype(mx.float32)
    finite = mx.all(mx.isfinite(value_f32))
    mx.eval(finite)
    if not bool(finite.item()):
        raise FloatingPointError(f"{label} contains NaN or infinity")
    minimum = mx.min(value_f32)
    maximum = mx.max(value_f32)
    mean = mx.mean(value_f32)
    standard_deviation = mx.std(value_f32)
    mx.eval(minimum, maximum, mean, standard_deviation)
    return {
        "shape": [int(dimension) for dimension in value.shape],
        "minimum": float(minimum.item()),
        "maximum": float(maximum.item()),
        "mean": float(mean.item()),
        "standardDeviation": float(standard_deviation.item()),
    }


def generate(args: argparse.Namespace) -> dict[str, object]:
    if args.width % 16 or args.height % 16:
        raise ValueError("FLUX width and height must each be divisible by 16")
    if args.width < 256 or args.height < 256:
        raise ValueError("FLUX proof canvases must be at least 256 pixels per side")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    if args.receipt and (args.receipt.exists() or args.receipt.is_symlink()):
        raise FileExistsError(f"Refusing to overwrite receipt: {args.receipt}")
    if not mx.metal.is_available():
        raise RuntimeError(
            "FLUX ConvRot requires the MLX Metal backend on Apple Silicon"
        )

    transformer = args.transformer
    paths = {
        "transformer": transformer,
        "vae": args.vae,
        "CLIP-L": args.clip,
        "T5-XXL": args.t5,
        "CLIP config": args.clip_config,
        "CLIP vocabulary": args.clip_vocab,
        "CLIP merges": args.clip_merges,
        "T5 config": args.t5_config,
        "T5 SentencePiece model": args.t5_spiece,
    }
    require_local_files(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    progress(3, f"loading FLUX.1 Dev {args.transformer_kind}")
    pipeline, transformer_receipt = build_pipeline(
        transformer_path=transformer,
        transformer_kind=args.transformer_kind,
        vae_path=args.vae,
        clip_path=args.clip,
        clip_config_path=args.clip_config,
        clip_vocab_path=args.clip_vocab,
        clip_merges_path=args.clip_merges,
        t5_path=args.t5,
        t5_config_path=args.t5_config,
        t5_spiece_path=args.t5_spiece,
        backend=args.backend,
        preprocessor=args.preprocessor,
    )

    latent_size = (args.height // 8, args.width // 8)
    latents = pipeline.generate_latents(
        args.prompt,
        n_images=1,
        num_steps=args.steps,
        guidance=args.guidance,
        latent_size=latent_size,
        seed=args.seed,
    )

    progress(8, "encoding prompt with local T5 and CLIP")
    conditioning = next(latents)
    mx.eval(conditioning)
    conditioning_stats = {
        "initialLatent": tensor_stats("initial latent", conditioning[0]),
        "t5": tensor_stats("T5 conditioning", conditioning[2]),
        "clip": tensor_stats("CLIP conditioning", conditioning[4]),
    }
    conditioning_peak = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()
    del pipeline.t5
    del pipeline.clip
    mx.clear_cache()

    step_seconds = []
    x_t = None
    for index, x_t in enumerate(latents, start=1):
        step_started = time.perf_counter()
        mx.eval(x_t)
        finite = mx.all(mx.isfinite(x_t))
        mx.eval(finite)
        if not bool(finite.item()):
            raise FloatingPointError(
                f"FLUX denoise step {index}/{args.steps} contains NaN or infinity"
            )
        step_seconds.append(time.perf_counter() - step_started)
        progress(12 + round(index / args.steps * 76), f"denoise {index}/{args.steps}")
    if x_t is None:
        raise RuntimeError("FLUX denoising loop produced no latent")
    final_latent_stats = tensor_stats("final latent", x_t)

    generation_peak = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()
    del pipeline.flow
    mx.clear_cache()

    progress(92, "decoding FLUX VAE")
    decoded = pipeline.decode(x_t, latent_size)
    mx.eval(decoded)
    decoded_stats = tensor_stats("decoded image", decoded)
    if decoded_stats["maximum"] - decoded_stats["minimum"] < 1e-4:
        raise RuntimeError("FLUX decoder produced a constant image")
    decoding_peak = mx.get_peak_memory() / 1024**3
    pixels = (decoded[0] * 255).astype(mx.uint8)
    mx.eval(pixels)
    pixel_stats = tensor_stats("8-bit image", pixels)
    if pixel_stats["maximum"] - pixel_stats["minimum"] < 2:
        raise RuntimeError("FLUX 8-bit output has no usable dynamic range")
    Image.fromarray(np.array(pixels)).save(args.output)

    output_hash = sha256(args.output)
    completed_at = datetime.now(UTC).isoformat()
    receipt = {
        "schema": "convrot-mlx.flux1-convrot-runtime-proof.v1",
        "route": "flux1-dev-mlx-convrot-w4a4"
        if args.transformer_kind == "convrot"
        else "flux1-dev-bf16-mlx-control",
        "status": "rendered_pending_visual_review"
        if args.transformer_kind == "convrot"
        else "control_rendered",
        "artifact": {
            "path": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "sha256": output_hash,
            "mediaType": "image/png",
        },
        "generation": {
            "prompt": args.prompt,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance": args.guidance,
            "sampler": "FlowMatch Euler",
            "scheduler": "Apple mlx-examples FLUX shifted linear schedule",
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "transformer": {
            "path": str(transformer.resolve()),
            **transformer_receipt,
        },
        "runtime": {
            "repository": "ml-explore/mlx-examples",
            "revision": MLX_EXAMPLES_REVISION,
            "mlxVersion": mx.__version__,
            "platform": platform.platform(),
            "metalAvailable": True,
            "nativeInt4Mma": args.backend.startswith("mpp-")
            if args.transformer_kind == "convrot"
            else None,
            "kernelSemantics": (
                {
                    "metal": "packed W4A4 operands, tile-local int4-to-half expansion, float accumulation, row scales after dot",
                    "metal-k32": "packed W4A4 operands, paired-nibble loads, K=32 tile-local int4-to-half expansion, float accumulation, row scales after dot",
                    "metal-k64": "packed W4A4 operands; paired-nibble loads; K=64 tile-local int4-to-half expansion; float accumulation; row scales after dot",
                    "metal-m32n32-k64": "packed W4A4 operands; paired-nibble loads; M=32, N=32, K=64 tile-local int4-to-half expansion; float accumulation; row scales after dot",
                    "mpp-w4a4": "Metal 4 MPP TensorOps signed-int8 x signed-int4 to int32; A4 values prepared directly in int8; K x packed-N weights; float row-scale and bias epilogue",
                    "mpp-w4a16": "Metal 4 MPP TensorOps bfloat16 x signed-int4 to float; one-pass Metal Hadamard rotation; K x packed-N weights; float row-scale and bias epilogue",
                    "mlx-expanded-fp32": "packed W4A4 storage and activations, transient per-layer int4-to-FP32 expansion, tuned MLX matmul, row scales after dot",
                    "reference": "packed W4A4 operands, eager int4-to-FP32 expansion, reference MLX matmul, row scales after dot",
                }[args.backend]
                if args.transformer_kind == "convrot"
                else "dense BF16 MLX transformer operations"
            ),
            "activationPreparation": (
                (
                    "fused two-pass Metal Hadamard rotation, row absmax, and signed A4 values written directly as int8"
                    if args.backend == "mpp-w4a4"
                    else "one-pass fused Metal Hadamard rotation with floating activations retained"
                    if args.backend == "mpp-w4a16"
                    else {
                        "mlx-compiled": "MLX-compiled Hadamard rotation, row absmax, signed A4 quantization, and nibble packing",
                        "metal-fused": "fused two-pass Metal Hadamard rotation, row absmax, signed A4 quantization, and nibble packing",
                    }[args.preprocessor]
                )
                if args.transformer_kind == "convrot"
                else None
            ),
            "t5SourceDtype": "float16",
            "t5ExecutionDtype": "bfloat16",
            "t5DtypeReason": "The exact FP16 repack is cast in memory to BF16 to prevent MLX attention overflow; source bytes are unchanged.",
        },
        "performance": {
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "conditioningPeakGiB": round(conditioning_peak, 3),
            "generationPeakGiB": round(generation_peak, 3),
            "decodingPeakGiB": round(decoding_peak, 3),
            "peakGiB": round(max(conditioning_peak, generation_peak, decoding_peak), 3),
            "stepSeconds": [round(value, 3) for value in step_seconds],
        },
        "numerics": {
            "conditioning": conditioning_stats,
            "finalLatent": final_latent_stats,
            "decoded": decoded_stats,
            "pixels": pixel_stats,
        },
        "visualReview": "pending" if args.transformer_kind == "convrot" else "control",
    }
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    progress(100, "image ready")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--prompt", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--receipt", type=Path)
    result.add_argument(
        "--transformer-kind", choices=["convrot", "bf16"], default="convrot"
    )
    result.add_argument("--transformer", type=Path, required=True)
    result.add_argument(
        "--backend",
        choices=[
            "metal",
            "metal-k32",
            "metal-k64",
            "metal-m32n32-k64",
            "mpp-w4a4",
            "mpp-w4a16",
            "mlx-expanded-fp32",
            "reference",
        ],
        default="mpp-w4a4",
    )
    result.add_argument(
        "--preprocessor", choices=["mlx-compiled", "metal-fused"], default="metal-fused"
    )
    result.add_argument("--width", type=int, default=768)
    result.add_argument("--height", type=int, default=768)
    result.add_argument("--steps", type=int, default=20)
    result.add_argument("--guidance", type=float, default=3.5)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--vae", type=Path, required=True)
    result.add_argument("--clip", type=Path, required=True)
    result.add_argument("--t5", type=Path, required=True)
    result.add_argument("--clip-config", type=Path, required=True)
    result.add_argument("--clip-vocab", type=Path, required=True)
    result.add_argument("--clip-merges", type=Path, required=True)
    result.add_argument("--t5-config", type=Path, required=True)
    result.add_argument("--t5-spiece", type=Path, required=True)
    return result


if __name__ == "__main__":
    generate(parser().parse_args())
