# convrot-mlx

Unofficial ConvRot W4A4 kernels, conversion tools, and reproducible benchmarks
for Apple Silicon and [MLX](https://github.com/ml-explore/mlx).

This project ports the plain ConvRot contract to MLX, implements fused Metal
Hadamard/activation preparation, and executes packed signed INT4 weights with
Metal 4 MPP TensorOps. It is early research software, not an official project
of the ConvRot authors, Apple, Comfy Org, or Black Forest Labs.

## Why this repository exists

The official ConvRot implementation targets CUDA and NVFP4-capable Blackwell
GPUs. This repository explores the same algorithmic family on Apple unified
memory, where fitting a model can matter as much as raw throughput.

On a 48 GB M4 Pro, the verified FLUX.1 Dev 512×512, 20-step comparison was:

| Path | End-to-end | Peak MLX memory | Relative to dense BF16 |
|---|---:|---:|---:|
| Dense BF16 | 98.640 s | 24.603 GiB | baseline |
| Legacy MLX ConvRot | 132.260 s | 9.296 GiB | 34.1% slower, 62.2% less memory |
| Metal 4 MPP ConvRot | 119.848 s | 9.296 GiB | 21.5% slower, 62.2% less memory |

The MPP path is 9.4% faster than the first MLX ConvRot kernel while preserving
its output byte-for-byte. It does **not** yet beat dense BF16. Closing that gap
without giving back the memory saving is the main optimization target.

See [the complete M4 Pro record](docs/benchmarks/m4-pro.md) and the
[video-model transfer study](docs/benchmarks/video-transfer.md).

## What is included

- Normalized regular-Hadamard rotation for group sizes 64 and 256.
- Symmetric signed W4/A4 rowwise absmax quantization in `[-7, 7]`.
- Reversible `N × packed-K` and Metal MPP `K × packed-N` layouts.
- A portable eager oracle, MLX graph path, custom Metal kernels, and Metal 4
  MPP W4A4/W4A16 TensorOps paths.
- `ConvRotLinear`, which can replace eligible `mlx.nn.Linear` modules.
- Correctness gates for packing, rotation, fused preparation, integer edges,
  full-shape products, and lossless MPP repacking.
- Reproducible microbenchmarks and an optional FLUX.1 Dev adapter.

The core package does not download models and contains no model weights.

## Requirements

- Apple Silicon Mac.
- Python 3.11 or newer.
- MLX 0.32.x; the exact proven environment is in `requirements.lock`.
- macOS 26 and Metal 4 for the MPP TensorOps backends.

The eager and older custom-Metal paths are useful as correctness oracles and
fallbacks, but only the MPP paths perform native signed INT8×INT4 or
BF16×INT4 TensorOps.

## Install and test

```bash
git clone https://github.com/Lumenbeing/convrot-mlx.git
cd convrot-mlx
uv venv --python 3.13
uv pip install -e '.[test]'
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Small core example:

```python
import mlx.core as mx
from convrot_mlx import ConvRotLinear, quantize_convrot_weight, repack_nk_weight_for_mpp

weight = mx.random.normal((1024, 1024)).astype(mx.bfloat16)
qweight_nk, scales = quantize_convrot_weight(weight)

layer = ConvRotLinear(1024, 1024, backend="mpp-w4a4")
layer.qweight = repack_nk_weight_for_mpp(qweight_nk)
layer.scales = scales

x = mx.random.normal((64, 1024)).astype(mx.bfloat16)
y = layer(x)
mx.eval(y)
```

Run `python benchmarks/tensorops.py --help` for benchmark shapes and controls.
Benchmark contributions must include the complete linear path—rotation,
activation quantization, TensorOp, scaling, and bias—not only the integer dot.

## FLUX.1 Dev example

The optional example converts a locally supplied, authorized FLUX.1 Dev
transformer and uses Apple's pinned MLX FLUX runtime. No weights, tokens, or
model files are distributed here. FLUX.1 Dev and converted derivatives remain
subject to the [FLUX.1 Dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev).

See [examples/flux1_dev](examples/flux1_dev/README.md) before using it.

## Contributing

Kernel tuning for M3, M4, and M5 hardware; additional Apple-friendly model
adapters; numerical tests; and independently reproduced benchmarks are
welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Provenance and license

The MLX/Metal core follows Comfy Kitchen's Apache-2.0 plain ConvRot contract.
The FLUX adapter is based in part on Apple's MIT-licensed MLX example. The
official ConvRot paper and implementation are cited but their unlicensed source
code was not imported. See [docs/provenance.md](docs/provenance.md) and
[`NOTICE`](NOTICE).

Unless a file says otherwise, this repository is licensed under Apache-2.0.
The specifically marked MLX-derived FLUX files remain under MIT.

## Citation

If you use the algorithm, cite the ConvRot authors:

```bibtex
@article{huang2025convrot,
  title={ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers},
  author={Huang, Feice and Han, Zuliang and Zhou, Xing and Chen, Yihuang and Zhu, Lifei and Wang, Haoqian},
  journal={arXiv preprint arXiv:2512.03673},
  year={2025}
}
```
