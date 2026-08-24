# Security policy

Please use GitHub's private vulnerability reporting for security-sensitive
issues. Do not publish access tokens, gated-model URLs, private prompts, or
local filesystem details in a public issue.

This repository executes custom Metal kernels and processes user-supplied model
files. Treat untrusted checkpoints and receipts as untrusted input. Conversion
tools refuse to overwrite outputs, but users should still work on copies and
verify source provenance independently.
