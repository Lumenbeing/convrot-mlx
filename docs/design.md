# Design

## Quantization contract

For each eligible linear projection, ConvRot MLX:

1. Partitions the input feature axis into regular-Hadamard groups.
2. Applies the same normalized, symmetric rotation to activations and weights.
3. Quantizes each rotated weight row and activation row with symmetric
   `absmax / 7` scaling into signed values `[-7, 7]`.
4. Packs two signed four-bit values per byte.
5. Computes the integer product and applies activation scale, weight scale, and
   bias in an explicit epilogue.

The rotation preserves the dense product before quantization because the
regular-Hadamard matrix is orthonormal and symmetric. Group sizes 64 and 256
are implemented. Other powers of four work in the eager path but do not have
fused Metal kernels.

## Weight layouts

- Portable and legacy Metal: `N × packed-K`, low nibble first along K.
- Metal MPP: `K × packed-N`, low nibble first along N.

`repack_nk_weight_for_mpp` transposes signed values and repacks them without
requantization. The reverse operation is tested bit-for-bit.

## Backends

| Backend | Activation | Weight | Compute | Intended use |
|---|---|---|---|---|
| `reference` | packed A4 | packed W4 | eager FP32 oracle | correctness |
| `mlx-expanded-fp32` | packed A4 | packed W4 | transient FP32 MLX GEMM | portable comparison |
| `metal*` | packed A4 | packed W4 | custom simdgroup half MMA | pre-Metal-4 experiment |
| `mpp-w4a4` | signed A4 values in INT8 | packed signed W4 | MPP INT8×INT4→INT32 | primary low-memory path |
| `mpp-w4a16` | rotated BF16/FP16 | packed signed W4 | MPP float×INT4→FP32 | weight-only comparison |

MPP W4A4 prepares activations with a two-pass fused Metal kernel: first find
the row absmax across all feature groups, then rotate and write signed A4
values directly as INT8. The MPP product writes INT32, followed by a separate
scale-and-bias Metal epilogue.

## Correctness invariants

- Signed quantized values use `[-7, 7]`; `-8` is never emitted.
- Packing is two's-complement and low-nibble first.
- Fused rotation and preparation must match the eager MLX contract.
- MPP INT8×INT4 must match an exact integer-dot oracle at signed edge values;
  its floating scale/bias epilogue must remain within `1e-5` in FP32.
- MPP weight repacking must reverse bit-for-bit.
- Every full-model conversion consumes each source tensor exactly once.
- Model adapters must record their source digest and topology revision.

## Remaining optimization target

The native integer product is only part of complete projection latency.
Rotation, row reduction, activation quantization, intermediate allocation,
command submission, and the epilogue remain material. Benchmarking only the
TensorOp can therefore claim a speedup that disappears end to end.

The M4 Pro implementation saves substantial memory but remains 21.5% slower
than dense BF16 in the matched FLUX proof. Promising work includes reducing
activation-preparation passes, avoiding the explicit INT32 destination, fusing
the epilogue, and tuning by shape and Apple GPU generation.
