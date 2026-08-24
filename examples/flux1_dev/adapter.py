# SPDX-FileCopyrightText: Copyright (c) 2023 Apple Inc.
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: MIT
"""Offline FLUX.1 Dev adapter for Apple's pinned MLX example runtime."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from flux.autoencoder import AutoEncoder
from flux.clip import CLIPTextModel, CLIPTextModelConfig
from flux.flux import FluxPipeline
from flux.model import Flux
from flux.sampler import FluxSampler
from flux.t5 import T5Config, T5Encoder
from flux.tokenizers import CLIPTokenizer, T5Tokenizer
from flux.utils import configs
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from convrot_mlx import CONVROT_GROUP_SIZE, ConvRotLinear, eligible_linear

CONVROT_SCHEMA = "convrot-mlx.flux1-convrot-w4a4.v1"
CONVROT_MPP_SCHEMA = "convrot-mlx.flux1-convrot-w4-mpp.v1"
CONVROT_MPP_WEIGHT_LAYOUT = (
    "signed_twos_complement_low_nibble_even_output_k_by_packed_n"
)
SOURCE_SHA256 = "4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7"


def _parameter_names(model: nn.Module) -> set[str]:
    return {name for name, _ in tree_flatten(model.parameters())}


def _load_filtered(
    model: nn.Module, weights: dict[str, mx.array], *, strict: bool = True
) -> None:
    expected = _parameter_names(model)
    filtered = {name: value for name, value in weights.items() if name in expected}
    model.load_weights(list(filtered.items()), strict=strict)


def replace_convrot_linears(
    flow: Flux,
    *,
    backend: str = "mpp-w4a4",
    preprocessor: str = "metal-fused",
) -> list[str]:
    replacements = []
    names = []
    for name, module in flow.named_modules():
        if eligible_linear(module):
            replacements.append(
                (
                    name,
                    ConvRotLinear(
                        module.weight.shape[1],
                        module.weight.shape[0],
                        bias=getattr(module, "bias", None) is not None,
                        group_size=CONVROT_GROUP_SIZE,
                        backend=backend,
                        preprocessor=preprocessor,
                    ),
                )
            )
            names.append(name)
    flow.update_modules(tree_unflatten(replacements))
    return names


def load_flow_convrot(
    path: str | Path,
    *,
    backend: str = "mpp-w4a4",
    preprocessor: str = "metal-fused",
) -> tuple[Flux, dict[str, str], list[str]]:
    weights, metadata = mx.load(str(path), return_metadata=True)
    uses_mpp = backend.startswith("mpp-")
    expected_schema = CONVROT_MPP_SCHEMA if uses_mpp else CONVROT_SCHEMA
    if metadata.get("schema") != expected_schema:
        raise ValueError(
            f"ConvRot backend {backend!r} requires checkpoint schema {expected_schema!r}; "
            f"found {metadata.get('schema')!r}"
        )
    if uses_mpp and metadata.get("weight_layout") != CONVROT_MPP_WEIGHT_LAYOUT:
        raise ValueError(
            f"MPP ConvRot checkpoint has unsupported weight layout: {metadata.get('weight_layout')!r}"
        )
    if metadata.get("source_sha256") != SOURCE_SHA256:
        raise ValueError(
            "ConvRot checkpoint does not derive from the supported FLUX.1 Dev source digest."
        )
    if int(metadata.get("convrot_group_size", "0")) != CONVROT_GROUP_SIZE:
        raise ValueError("ConvRot checkpoint group size does not match this runtime.")

    flow = Flux(configs["flux-dev"].params)
    converted = replace_convrot_linears(
        flow, backend=backend, preprocessor=preprocessor
    )
    if int(metadata.get("converted_linear_count", "-1")) != len(converted):
        raise ValueError(
            "ConvRot checkpoint layer count does not match the pinned Apple FLUX topology."
        )
    flow.load_weights(list(weights.items()), strict=True)
    return flow, metadata, converted


def load_flow_bf16(path: str | Path) -> Flux:
    flow = Flux(configs["flux-dev"].params)
    weights = flow.sanitize(mx.load(str(path)))
    flow.load_weights(list(weights.items()), strict=True)
    return flow


def load_autoencoder(path: str | Path) -> AutoEncoder:
    ae = AutoEncoder(configs["flux-dev"].ae_params)
    _load_filtered(ae, ae.sanitize(mx.load(str(path))), strict=True)
    return ae


def load_clip(path: str | Path, config_path: str | Path) -> CLIPTextModel:
    config = json.loads(Path(config_path).read_text())
    # openai/clip-vit-large-patch14 stores the text config below `text_config`;
    # the gated FLUX diffusers repository stores an equivalent flat config.
    config = config.get("text_config", config)
    clip = CLIPTextModel(CLIPTextModelConfig.from_dict(config))
    _load_filtered(clip, clip.sanitize(mx.load(str(path))), strict=True)
    return clip


def load_t5(path: str | Path, config_path: str | Path) -> T5Encoder:
    config = T5Config.from_dict(json.loads(Path(config_path).read_text()))
    t5 = T5Encoder(config)
    sanitized = t5.sanitize(mx.load(str(path)))
    # ComfyUI's T5 repack carries both `shared.weight` and
    # `encoder.embed_tokens.weight`; the Apple encoder has one canonical wte.
    # Filtering by the constructed topology keeps `shared.weight -> wte.weight`
    # and refuses to invent a second embedding parameter.
    #
    # This exact local repack is FP16. MLX's T5 attention overflows in that
    # dtype and silently produces non-finite conditioning; the official Apple
    # FLUX path is stable with BF16 text weights. Cast in memory at load time so
    # the receipted source bytes remain untouched and the wider exponent range
    # is retained through prompt encoding.
    sanitized = {name: value.astype(mx.bfloat16) for name, value in sanitized.items()}
    _load_filtered(t5, sanitized, strict=True)
    return t5


def load_clip_tokenizer(
    vocab_path: str | Path, merges_path: str | Path
) -> CLIPTokenizer:
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    lines = Path(merges_path).read_text(encoding="utf-8").strip().split("\n")
    merges = [tuple(line.split()) for line in lines[1 : 49152 - 256 - 2 + 1]]
    ranks = dict(map(reversed, enumerate(merges)))
    return CLIPTokenizer(ranks, vocab, max_length=77)


def build_pipeline(
    *,
    transformer_path: str | Path,
    transformer_kind: str,
    vae_path: str | Path,
    clip_path: str | Path,
    clip_config_path: str | Path,
    clip_vocab_path: str | Path,
    clip_merges_path: str | Path,
    t5_path: str | Path,
    t5_config_path: str | Path,
    t5_spiece_path: str | Path,
    backend: str = "mpp-w4a4",
    preprocessor: str = "metal-fused",
) -> tuple[FluxPipeline, dict[str, object]]:
    """Construct a pipeline without any Hugging Face network resolution."""

    pipeline = FluxPipeline.__new__(FluxPipeline)
    pipeline.dtype = mx.bfloat16
    pipeline.name = "flux-dev"
    pipeline.t5_padding = True
    pipeline.ae = load_autoencoder(vae_path)
    if transformer_kind == "convrot":
        pipeline.flow, metadata, converted = load_flow_convrot(
            transformer_path,
            backend=backend,
            preprocessor=preprocessor,
        )
        transformer_receipt = {
            "kind": "convrot-w4a4-mpp"
            if backend == "mpp-w4a4"
            else ("convrot-w4a16-mpp" if backend == "mpp-w4a16" else "convrot-w4a4"),
            "backend": backend,
            "preprocessor": preprocessor,
            "convertedLinearCount": len(converted),
            "metadata": metadata,
        }
    elif transformer_kind == "bf16":
        pipeline.flow = load_flow_bf16(transformer_path)
        transformer_receipt = {"kind": "bf16-control", "backend": "mlx-dense"}
    else:
        raise ValueError(f"Unknown transformer kind {transformer_kind!r}")
    pipeline.clip = load_clip(clip_path, clip_config_path)
    pipeline.clip_tokenizer = load_clip_tokenizer(clip_vocab_path, clip_merges_path)
    pipeline.t5 = load_t5(t5_path, t5_config_path)
    pipeline.t5_tokenizer = T5Tokenizer(str(t5_spiece_path), 512)
    pipeline.sampler = FluxSampler("flux-dev")
    return pipeline, transformer_receipt
