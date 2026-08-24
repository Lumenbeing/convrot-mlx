#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Lumenbeing contributors
# SPDX-License-Identifier: Apache-2.0
"""Fail if the public tree contains private paths, secrets, weights, or media."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".gguf",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".onnx",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".webp",
}
FORBIDDEN_TEXT = {
    "absolute macOS volume path": "/" + "Volumes" + "/",
    "former local username": "chris" + "chambless",
    "former disk name": "Black" + "Mamba",
    "private proof directory": "." + "lumen-proofs",
    "private progress protocol": "LUMEN" + "_PROGRESS",
    "former model root": "MLX-" + "Video-Models",
    "former ComfyUI path": "Documents/" + "ComfyUI",
}
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style secret": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "bearer token": re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}"),
}


def files() -> list[Path]:
    result = []
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if not path.is_file() or any(
            part in SKIP_DIRECTORIES or part.endswith(".egg-info") for part in parts
        ):
            continue
        result.append(path)
    return sorted(result)


def main() -> int:
    failures: list[str] = []
    checked = files()
    for path in checked:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact: {relative}")
        if path.stat().st_size > 1_000_000:
            failures.append(f"file exceeds 1 MB public-source limit: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"unexpected binary file: {relative}")
            continue
        for label, needle in FORBIDDEN_TEXT.items():
            if needle in content:
                failures.append(f"{label}: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                failures.append(f"{label}: {relative}")

    if failures:
        print("Public-tree audit failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"Public-tree audit passed for {len(checked)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
