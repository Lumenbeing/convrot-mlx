# FLUX.1 Dev example

This optional example converts and runs a locally supplied FLUX.1 Dev
transformer. It does not download or redistribute model weights.

## License boundary

FLUX.1 Dev is gated and governed by Black Forest Labs' non-commercial model
license. A converted checkpoint remains a model derivative under that license.
Do not upload converted files here or assume this repository's Apache license
changes the model terms.

## Runtime

Clone the exact MLX example revision and expose its `flux` directory:

```bash
git clone https://github.com/ml-explore/mlx-examples.git /path/to/mlx-examples
git -C /path/to/mlx-examples checkout 796f5b53cab69a3d48a44233ce21aae889e94a08
export PYTHONPATH="$PWD/src:/path/to/mlx-examples/flux"
```

Install the optional dependencies with `uv pip install -e '.[flux]'`.

## Convert and repack

The converter requires the supported canonical FLUX.1 Dev transformer digest
and refuses to overwrite existing outputs:

```bash
python -m examples.flux1_dev.convert \
  --source /path/to/authorized/flux1-dev.safetensors \
  --output /path/to/work/transformer-convrot-nk.safetensors \
  --receipt /path/to/work/conversion.json

python -m examples.flux1_dev.repack_mpp \
  --source /path/to/work/transformer-convrot-nk.safetensors \
  --output /path/to/work/transformer-convrot-mpp.safetensors \
  --receipt /path/to/work/repack.json
```

The second step is lossless: every packed weight is unpacked, transposed into
MPP's logical right-operand layout, repacked, and reverse-checked bit-for-bit.

## Generate

The generator is intentionally explicit about every local component:

```bash
python -m examples.flux1_dev.generate \
  --prompt 'A brass observatory above a stormy coast at blue hour' \
  --output /path/to/output.png \
  --receipt /path/to/output.json \
  --transformer /path/to/work/transformer-convrot-mpp.safetensors \
  --vae /path/to/ae.safetensors \
  --clip /path/to/clip_l.safetensors \
  --t5 /path/to/t5xxl_fp16.safetensors \
  --clip-config /path/to/clip/config.json \
  --clip-vocab /path/to/clip/vocab.json \
  --clip-merges /path/to/clip/merges.txt \
  --t5-config /path/to/t5/config.json \
  --t5-spiece /path/to/t5/spiece.model \
  --backend mpp-w4a4 \
  --width 512 --height 512 --steps 20 --seed 0
```

Receipts contain local paths and prompts. Keep them outside the repository.
