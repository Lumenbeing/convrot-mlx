# Video-model transfer study

Date: 2026-08-23

ConvRot is mechanically transferable to the tested video-transformer shapes,
but it was not the best production optimization on the M4 Pro. These results
are included to make negative findings reusable and to prevent extrapolating a
small synthetic integer-dot win to a complete video projection.

All rows report median complete projection latency, including the preparation
required by each path.

| Workload and shape | Existing MLX path | MPP W4A4 | W4A4 ratio | Estimated persistent saving |
|---|---:|---:|---:|---:|
| Wan 2.2 A14B, M=7,800, K=N=5,120 | 0.066210 s Q8 | 0.103394 s | 1.562× slower | 14.695 GiB across two experts |
| Wan expansion, 5,120→13,824 | 0.178465 s Q8 | 0.290321 s | 1.627× slower | same model estimate |
| Wan contraction, 13,824→5,120 | 0.173268 s Q8 | 0.352751 s | 2.036× slower | same model estimate |
| LTX 2.5, M=8,800, K=N=4,096 | 0.047239 s Q8 | 0.075218 s | 1.592× slower | 9.692 GiB transformer |
| LTX expansion, 4,096→16,384 | 0.187762 s Q8 | 0.297969 s | 1.587× slower | same model estimate |
| LTX contraction, 16,384→4,096 | 0.183040 s Q8 | 0.408371 s | 2.231× slower | same model estimate |
| H3 REF2VA QKV, M=10,017, K=2,688 | 0.328331 s affine Q8 | 0.731943 s group-64 W4A4 | 2.23× slower | model-wide estimate not promoted |

## Quality caveats

The Wan and H3 transfer tests began from already quantized Q8 weights, so the
W4 candidates incurred a second lossy quantization. They are valid rejection
tests for those installed artifacts, not quality claims about a fresh BF16
conversion.

LTX had a local BF16 source. Its existing Q8 relative RMSE was approximately
0.0093–0.0119, while W4A16 measured approximately 0.153–0.173 and W4A4
approximately 0.218–0.243 on sampled projections. The current Q8 path was both
faster and closer to BF16.

## Better optimizations discovered

- Wan's high- and low-noise experts are not needed simultaneously. Swapping
  exact Q8 experts at the scheduler boundary cut active expert residency by
  14.348 GiB (50%) with approximately 1.04% full-size wall-time cost.
- H3 benefited from a direct BF16-to-affine-Q4 conversion rather than ConvRot.
- LTX's verified largest memory peak occurred during VAE decode after the
  transformer was released, so transformer compression would not fix that
  separate bottleneck.

The general lesson is that freed weight memory can increase denoising headroom,
but it does not automatically reduce latency or a non-overlapping decoder peak.
