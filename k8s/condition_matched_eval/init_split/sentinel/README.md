# Checkpoint -1 split-resource sentinel

This directory contains the exact one-seed gate for the prepared split
initialization-evaluation design. It is a launch plan, not standing launch
authorization.

- Architecture/seed: `gpt2_small:314159`
- GPU stage: one indexed Pod, parallelism 1, one established 24 GB GPU
- CPU stage: one Pod, parallelism 1, no GPU
- Frozen checkpoint inventory SHA-256:
  `25f044a67b9eb9f2f1da168fd2f8a53cf12b6a88f919fbc5f0cc29fa669b5f37`
- Gold stimulus manifest SHA-256:
  `056ca1d2a5df745662ba501c97c434a2dde3a7cc857362d36b34caacd05d5de7`
- Pinned code: `8f6af766d5041f1c291e9fd205a366ce85dddc9e`

The GPU Job must run first. It emits five genuine representative cells, one
per intervention, and telemetry for GPU utilization/VRAM plus the fixed CPU
and RAM allocation. Pull and audit those five run IDs before submitting the
CPU Job. The CPU Job then materializes the sixth stable-inventory identity
(`baseline-h1`) from the verified baseline representative. Audit all six run
IDs after fanout.

Both manifests passed client and server dry-run in context `nautilus`,
namespace `lemn-lab`, on 2026-08-14. Before any future apply, repeat the server
dry-run and confirm that no Job with either exact name exists.

Do not submit the full 70-group fleet until this sentinel establishes that the
GPU request is appropriately utilized and the CPU fanout outputs pass exact
content, provenance, and S3 SHA checks.
