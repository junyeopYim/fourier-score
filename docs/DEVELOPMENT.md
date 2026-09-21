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
# Include the optional native LDM integration tests:
uv run --locked --extra ldm --extra metrics python -m pytest -q
# Optional CUDA parity audit; downloads small, pinned original source files.
uv run --locked --extra ldm --extra metrics python scripts/verify_ldm_upstream.py --output saved/ldm_upstream_parity.json
```

The tests cover reference algebra, matching backbone initialization, EMA loading,
consumed-batch/RNG resume, frequency diagnostics, config validation, and CLI
protocols. CUDA/MPS checks depend on available hardware. A smoke test or structural
audit is not evidence of long-training convergence or generative quality.

The LDM path lives in `fourier_score/ldm/` and uses `ldm.py` as its CLI. Pinned
computational modules under `ldm/upstream/` retain upstream arithmetic and keys;
do not reformat them. Local LDM checkpoints have a separate format. Its tests
exercise both KL and VQ stages, analytic posterior statistics, encoded pixel
flips, training-only splits, paired initialization, exact CPU resume, public EMA
selection, native DDIM endpoints, and decoded image artifacts.

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

## Editing while training is running

Keep a training checkout at its starting revision until that process finishes.
Use a separate Git worktree for development; do not pull, switch revisions, or
move Python modules inside an active training checkout. Lazy imports can still
read files after startup.

Checkpoints now reuse the source fingerprint recorded in the run's startup
environment. Saving no longer rereads source files: a moved file cannot prevent
checkpoint saving, and edited files cannot relabel the loaded model code as a
different source revision. Resume still checks the current source against the
checkpoint's fingerprint. This fix only applies to newly started processes;
already running Python processes keep their old checkpoint implementation.

If an older run failed while saving after files were moved, use the last complete
`last.pt` with a separate checkout of its matching source revision. Restore the
same locked environment, data, and device, and change `trainer.save_dir` when
keeping the failed attempt's logs separate. A save failure does not establish that
the step shown in the log reached disk; inspect the checkpoint's `step` field.
