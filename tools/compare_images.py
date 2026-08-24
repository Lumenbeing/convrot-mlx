#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0
"""Small deterministic image metrics for the ConvRot/BF16 proof pair."""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def edge_energy(image: np.ndarray) -> float:
    luminance = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    dx = np.abs(np.diff(luminance, axis=1)).mean()
    dy = np.abs(np.diff(luminance, axis=0)).mean()
    return float((dx + dy) / 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("convrot")
    parser.add_argument("bf16")
    args = parser.parse_args()
    convrot = (
        np.asarray(Image.open(args.convrot).convert("RGB"), dtype=np.float32) / 255.0
    )
    bf16 = np.asarray(Image.open(args.bf16).convert("RGB"), dtype=np.float32) / 255.0
    if convrot.shape != bf16.shape:
        raise ValueError(f"Proof shapes differ: {convrot.shape} versus {bf16.shape}")
    delta = np.abs(convrot - bf16)
    convrot_edge = edge_energy(convrot)
    bf16_edge = edge_energy(bf16)
    metrics = {
        "shape": list(convrot.shape),
        "meanAbsoluteDifference255": round(float(delta.mean() * 255), 4),
        "p95AbsoluteDifference255": round(float(np.percentile(delta, 95) * 255), 4),
        "rmse255": round(float(np.sqrt(np.square(convrot - bf16).mean()) * 255), 4),
        "channelCorrelations": [
            None
            if (value := correlation(convrot[..., channel], bf16[..., channel])) is None
            else round(value, 6)
            for channel in range(3)
        ],
        "convrotEdgeEnergy": round(convrot_edge, 7),
        "bf16EdgeEnergy": round(bf16_edge, 7),
        "edgeEnergyRatio": None
        if bf16_edge == 0
        else round(convrot_edge / bf16_edge, 6),
        "interpretation": "Sampling is iterative, so pixel deltas are descriptive only; route promotion depends on side-by-side visual review for structure, detail, text, anatomy, and quantization artefacts.",
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
