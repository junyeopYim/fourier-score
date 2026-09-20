# Verification record — 2026-09-20

This record covers code revision `ea56e75e6269a30142e74999053d3f8d66316521`
(`Add Gaussian ablation controls and spectral diagnostics`). Verification began
with a clean working tree. This follow-up changes documentation and evidence
only; it does not change the tested training source or configuration.

The machine-readable [summary](../verification/2026-09-20/summary.json) records
the tested commit, source/config/lockfile hashes and scope. The
[command record](../verification/2026-09-20/commands.json) contains exact argv,
working directory, UTC timestamps, exit codes, durations and raw-log hashes.

## Runtime

The existing project `.venv` was used, without reinstalling or synchronizing
packages. Full details: [environment.json](../verification/2026-09-20/environment.json).

| Component | Observed value |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.14.0+cu130 |
| TorchVision | 0.29.0+cu130 |
| NumPy / Pillow / pytest | 2.4.6 / 12.3.0 / 9.1.1 |
| uv | 0.11.8 |
| Platform | Linux / WSL2 x86_64 |
| GPU / CUDA runtime | NVIDIA GeForce RTX 5060 Ti / 13.0 |
| MPS | Unavailable |

`uv.lock` is tracked. `uv lock --check --offline` passed; see the
[raw output](../verification/2026-09-20/uv_lock_check.txt). This verifies lockfile
consistency, not a clean installation on every supported platform.

## Checks actually executed

| Check | Result | Evidence |
|---|---|---|
| Full pytest suite | **143 passed, 2 skipped, 0 failed, 0 errors** | [Raw log](../verification/2026-09-20/pytest.txt), [JUnit](../verification/2026-09-20/pytest.xml) |
| CIFAR-10 ablation architecture audit | All five output parameterizations have identical architecture and initial backbone hashes | [JSON](../verification/2026-09-20/cifar10_ablation_architecture.json), [stdout](../verification/2026-09-20/cifar10_ablation_architecture.txt) |
| CPU doctor | All five parameterizations passed | [JSON](../verification/2026-09-20/doctor_cpu.json), [stdout](../verification/2026-09-20/doctor_cpu.txt) |
| CUDA doctor | All five parameterizations passed; pytest CUDA device test also passed | [JSON](../verification/2026-09-20/doctor_cuda.json), [stdout](../verification/2026-09-20/doctor_cuda.txt) |
| Primary three-arm dry run | Three distinct commands for seed 42; no training started | [Commands](../verification/2026-09-20/primary_comparison_dry_run.txt) |

Both skipped cases are real MPS device tests (`auto` and explicit `cpu` spectral
backends). They were skipped because this machine has no MPS hardware.

The suite covers the scalar/flat-spectrum equivalence and gradients, unchanged
Fourier reference in the unscaled arm, shared backbone initialization, VE/DDPM
loss and sampling, spectral DSM energy reconstruction, empty bands, evaluation
RNG isolation, checkpoint/EMA reload, deterministic CPU resume, cumulative
resume timing and comparison-run configuration checks.

The CIFAR-10 audit uses `configs/cifar10_ablation.json`, constructing the full
NCSN++ on CPU with placeholder positive statistics. It does not load CIFAR-10
or train a full-size model. `score`, `diffusion`, `scalar_gaussian`,
`fourier_gaussian_unscaled` and `fourier_gaussian` all have:

- 62,758,787 trainable parameters; 62,758,915 total parameters.
- 6 attention blocks and 44 BigGAN++ residual blocks.
- The same architecture and initial backbone weight hashes, recorded per arm.

The doctor checks use the small synthetic NCSN++ configuration. They execute
forward/backward, Adam, sampling, real-DFT/FFT comparison and device RNG restore
on the indicated hardware. They are functional checks, not generation-quality
benchmarks or evidence that full CIFAR-10 batches fit GPU memory.

## Source identity

The tested runtime fingerprint from `utils.util.source_hash()` is:

```text
90e4c5839269027c6bddecf3c55ffe8e1d36d404f3f037374f8dfef8409be6dc
```

It hashes `parse_config.py` and Python files under `base`, `model`, `sde`,
`data_loader`, `trainer`, `utils` and `logger`. Documentation and verification
artifacts are outside this fingerprint. The tested Git commit identifies the
entry points and tests as well; separate config and dependency-file hashes
are in the summary. Documentation-only follow-ups do not invalidate source
identity for checkpoints produced by this tested runtime.

## Limits and historical records

This verification did not run real-dataset long training, FID/IS, Inception
weight downloads, TensorBoard or a clean environment installation. It does not
establish improved image quality, training efficiency, cross-device bitwise
reproducibility, or MPS execution.

The [2026-09-19 record](VALIDATION_2026-09-19.md) is preserved unchanged. The
older files directly under `verification/`, including `pytest.txt`,
`summary.json`, the original architecture audits and `uv_lock_attempt.txt`,
belong to that historical CPU-only record (114 passed, 3 skipped). They are not
the current patch's validation evidence. The [record index](../verification/README.md)
links the two scopes explicitly.

## Repeating the checks

Run from the project root after installing the project environment. These
commands use the existing `.venv`; CPU and CUDA doctor checks are separate.

```bash
uv lock --check --offline
.venv/bin/python -m pytest -q -ra
.venv/bin/python inspect_model.py -c configs/cifar10_ablation.json
.venv/bin/python doctor.py --device cpu
.venv/bin/python doctor.py --device cuda
.venv/bin/python scripts/run_comparison.py -c configs/cifar10_ablation.json \
  --device cuda --objectives score scalar_gaussian fourier_gaussian \
  --seeds 42 --dry-run
```

For another recorded run, save raw outputs and environment metadata in a new
verification directory, preserving the dated records already committed.
