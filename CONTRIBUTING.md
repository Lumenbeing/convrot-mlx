# Contributing

Thank you for helping make low-bit MLX inference more useful on Apple Silicon.

## Ground rules

- Do not commit model weights, converted checkpoints, generated media, access
  tokens, private prompts, receipts containing local paths, or gated assets.
- Keep model-specific code under `examples/`; keep `src/convrot_mlx` independent
  of any particular model topology.
- Preserve SPDX headers and the attributions in `NOTICE`.
- Do not copy source from repositories that do not provide an explicit license.
- State whether a change alters numerics. Byte-identical optimizations and
  approximation experiments should never be mixed in one benchmark claim.

## Correctness bar

Run:

```bash
python -m unittest discover -s tests -v
python tools/check_public_tree.py
python -m compileall -q src tests benchmarks tools examples
```

Kernel changes must compare against the eager oracle at signed edge values and
at one representative full projection shape. Layout changes must prove a
bit-exact round trip.

## Performance reports

Include all of the following:

- Apple chip, unified-memory capacity, macOS version, and MLX version.
- Exact `M × K × N`, dtype, group size, backend, warm-up, and sample count.
- Median complete-path latency, not only the TensorOp dispatch.
- Incremental and total peak MLX memory when available.
- Numerical comparison against the reference path.
- Whether the run was cold, warm, compiled, or affected by swap.

For end-to-end model results, identify the exact model and revision but do not
upload gated weights or derived checkpoints.

## Pull requests

Keep changes focused. Explain the intended invariant, include test evidence,
and update the relevant benchmark document when making a performance claim.
