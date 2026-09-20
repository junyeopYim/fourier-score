# Verification records

| Record | Tested scope | Entry point |
|---|---|---|
| **2026-09-20 — companion layout** | Working-tree refactor based on `865e84e`; 170 passed, 2 MPS tests skipped; pre-refactor checkpoint/training equality and public CLI checks | [Summary](2026-09-20-companion/summary.json), [development guide](../docs/DEVELOPMENT.md) |
| 2026-09-20 — ablation verification | `ea56e75`; 143 passed, 2 MPS tests skipped; CPU/CUDA doctor and five-arm full CIFAR-10 architecture audit | [Summary](2026-09-20/summary.json), [environment](2026-09-20/environment.json), [commands](2026-09-20/commands.json), [validation document](../docs/VALIDATION.md) |
| 2026-09-19 — historical | Original CPU-only build; 114 passed, 3 device tests skipped; three-arm architecture audit | [Original summary](summary.json), [original test output](pytest.txt), [preserved validation document](../docs/VALIDATION_2026-09-19.md) |

The original files directly in this directory are retained as historical
evidence. In particular, `pytest.txt`, `summary.json`, `cifar10_architecture.json`
and `uv_lock_attempt.txt` describe the September 19 environment, not the current
patch. New results live in dated subdirectories, so old measurements and
limitations are not silently overwritten.

The ablation record includes exact commands, raw stdout/stderr, JUnit results,
hardware/package versions and the tested commit and source fingerprint. Full
CIFAR-10 model construction is a CPU architecture audit using placeholder
statistics; CUDA doctor checks use the small synthetic model. Neither record
establishes FID/IS or long-training quality. The original archive manifest is
preserved as [2026-09-19-manifest.json](2026-09-19-manifest.json); its file paths
refer to that archived layout, not the current package.
