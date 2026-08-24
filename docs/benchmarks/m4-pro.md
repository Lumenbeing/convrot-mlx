# M4 Pro benchmark record

Date: 2026-08-23

## Environment

- Apple M4 Pro, 48 GB unified memory
- macOS 26.5.2
- MLX 0.32.0
- FLUX.1 Dev transformer
- 512×512 output, 20 denoise steps
- Identical prompt, seed, sampler, and local component bytes across controls

The runs measured the complete local generation path. The per-step figure
includes activation rotation and quantization, the linear operations, scaling,
and bias. Peak memory is the maximum MLX allocator peak across conditioning,
denoising, and decode phases.

## Results

| Path | End-to-end | Median step | Generation peak | Total peak |
|---|---:|---:|---:|---:|
| Dense BF16 | 98.640 s | 4.834 s | not separately retained | 24.603 GiB |
| Legacy M32/N32/K64 ConvRot | 132.260 s | 6.522 s | 7.352 GiB | 9.296 GiB |
| Metal 4 MPP ConvRot | 119.848 s | 5.900 s | 7.553 GiB | 9.296 GiB |

Derived comparisons:

- MPP versus legacy ConvRot: 9.4% faster end to end.
- MPP versus dense BF16: 1.215× as long, or 21.5% slower.
- MPP versus dense BF16 peak: 62.2% less MLX memory.
- Packed transformer: 5,974,123,747 bytes versus 23,802,932,552 source bytes.

## Correctness and quality gates

- Seventeen unit gates passed.
- Signed INT8×INT4 edge products matched the exact integer-dot oracle; the
  separate FP32 scale/bias epilogue passed a `1e-5` absolute-error gate.
- Representative full-shape projections passed.
- All 313 converted qweights survived a bit-exact layout round trip.
- MPP and legacy ConvRot produced the same 20-step PNG SHA-256 and identical
  recorded latent/pixel statistics.
- A paired dense BF16 image and the ConvRot image both passed visual review.

The exact INT32 MPP destination increased generation peak by 0.201 GiB versus
the legacy path. The shared 9.296 GiB total peak occurred in another phase, so
the end-to-end peak did not change.

## Interpretation

This is a memory result with a narrowed speed deficit, not a claim that W4A4
is faster than BF16 on M4 Pro. The public optimization target is to recover the
remaining 21.5% without weakening the exactness gates or materially increasing
peak memory.
