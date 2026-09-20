# Development and migration

## Reading order

1. `fourier_score/method.py`: Gaussian reference and residual scaling.
2. `fourier_score/statistics.py`: training-only mean and Fourier power estimation.
3. `fourier_score/model.py` and `loss.py`: raw backbone output → scaled score → DSM.
4. `fourier_score/training.py`: a single Trainer, without base-class inheritance.
5. `fourier_score/diffusion.py`: the shared process and samplers.

Device support and serialization are in `utils.py`; EMA, terminal logging, and
image output have separate small modules. `spectral.py` contains FFT / real-DFT
implementations. These support the method without changing its equations.
`backbones/` contains attributed upstream code; do not reformat those files.

## Checks

```bash
uv run --locked python -m pytest -q
uv run --locked python scripts/doctor.py --device cpu
uv run --locked python scripts/inspect_model.py -c configs/cifar10_ablation.json
bash scripts/reproduce_cifar10.sh 950k --seeds 42 43 --dry-run
```

The tests cover reference algebra, matching backbone initialization, EMA loading,
consumed-batch/RNG resume, frequency diagnostics, config validation, and CLI
protocols. CUDA/MPS checks depend on available hardware. A smoke test or structural
audit is not evidence of long-training convergence or generative quality.

Dated records under `verification/` and `docs/VALIDATION*` describe the source
revisions they name. Old paths and commands in those records are historical.
See [the verification index](../verification/README.md) for their scope.

## Entry point changes

| Before | Now |
|---|---|
| `train.py`, `sample.py` | Same entry points and existing flags |
| `test.py -r ... -o ...` | `evaluate.py dsm -r ... -o ...` |
| `metrics.py --real ... --generated ...` | `evaluate.py fid --real ... --generated ...` |
| `prepare.py`, `export_real.py` | `scripts/prepare.py`, `scripts/export_real.py` |
| `doctor.py`, `inspect_model.py` | `scripts/doctor.py`, `scripts/inspect_model.py` |
| `parse_config`, `model.*`, `trainer.*`, etc. | Modules under `fourier_score` |

`train.py --download` covers the common MNIST/CIFAR-10 preparation workflow.
`--parameterization` is a public alias for the saved v1 `loss.type` field;
`--objectives` remains an alias for comparison-runner `--parameterizations`.
Old Python import paths are not maintained as a second implementation.

## Checkpoint compatibility

The v1 tensor checkpoint format, backbone parameter keys, EMA keys, and Gaussian
buffers are unchanged. Earlier v1 checkpoints can be used by `sample.py` and
`evaluate.py dsm`; source differences produce the existing inference warning.

Training resume still requires the same source hash and runtime protocol. This
refactor changes that hash, so continue pre-refactor training runs in their original
revision. The latest pre-refactor revision is `865e84e`. No compatibility flag
silently bypasses that check. New runs preserve deterministic CPU resume across
interruptions within this revision. Changing a run's update limit does not change
its explicit preset name or output path.
