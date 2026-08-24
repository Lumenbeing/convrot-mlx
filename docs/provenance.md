# Provenance and licensing

This repository was extracted as a clean public project on 2026-08-24. It does
not preserve private application history, local proof artifacts, or model
files.

## ConvRot algorithm

The algorithm is described in:

- Feice Huang, Zuliang Han, Xing Zhou, Yihuang Chen, Lifei Zhu, and Haoqian
  Wang, “ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion
  Transformers,” arXiv:2512.03673, 2025.
- Official project: <https://github.com/feice-huang/ConvRot>

At extraction time, the official repository did not contain an explicit
license file. Its source was therefore treated as citation-only and was not
copied into this project.

## Comfy Kitchen reference contract

The quantization semantics and eager reference used for this MLX/Metal port
come from Comfy Kitchen at:

- Repository: <https://github.com/Comfy-Org/comfy-kitchen>
- Revision: `7d86acf60c88fd6c3c733c0e54db22ef74b8d77f`
- Relevant reference: `comfy_kitchen/backends/eager/convrot_w4a4.py`
- License: Apache-2.0

`src/convrot_mlx/core.py` retains Comfy Org's copyright attribution and adds
the Apple-specific implementation copyright. The Apache license and required
notice are retained at repository root and in `third_party/`.

## Apple MLX example

The optional FLUX adapter is based in part on:

- Repository: <https://github.com/ml-explore/mlx-examples>
- Revision: `796f5b53cab69a3d48a44233ce21aae889e94a08`
- Relevant sources: `flux/flux/utils.py`, `flux/txt2image.py`
- License: MIT

The affected files carry MIT SPDX headers, and Apple's license is retained in
`third_party/MLX-EXAMPLES-LICENSE`.

MLX itself is an external runtime dependency and is not vendored.

## Model boundary

No weights or converted checkpoints are included. The FLUX.1 Dev adapter
accepts a user-supplied model only after checking the supported canonical
source digest. FLUX.1 Dev and its converted derivatives remain governed by
Black Forest Labs' FLUX.1 Dev Non-Commercial License; the software license in
this repository does not alter that model license.

The generic core and synthetic tests do not require FLUX or any gated model.
