# Development and debugging

## Repository design

The layout follows the useful parts of the PyTorch Template pattern: explicit
entry points, versioned configs, separate data/model/training/evaluation modules,
and complete checkpoints. One trainer per domain keeps the numerical path easy
to inspect without a hierarchy of generic classifier base classes.

Read the implementation in this order:

1. `fourier_score/method.py`: fixed Gaussian reference and residual scale.
2. `fourier_score/statistics.py`: training-only mean and Fourier power.
3. `fourier_score/model.py` and `loss.py`: backbone output → total score → DSM.
4. `fourier_score/training.py`: consumed batches, optimization, EMA and resume.
5. `fourier_score/diffusion.py`: shared forward process and samplers.

The native latent equivalents live under `fourier_score/ldm/`. Vendored code in
`backbones/` and `ldm/upstream/` retains source attribution and should not be
reformatted as part of local infrastructure changes.

## Checks before a long experiment

```bash
uv sync --locked --all-extras
uv run --locked --all-extras python -m pytest -q
uv run --locked python scripts/doctor.py --device cpu
uv run --locked python scripts/doctor.py --device cuda
uv run --locked python scripts/inspect_model.py -c configs/cifar10_950k.json
bash scripts/reproduce_cifar10.sh 950k --seeds 42 43 44 \
  --parameterizations scalar_gaussian fourier_gaussian --dry-run
uv run --locked --extra notebooks python scripts/check_notebooks.py --execute
```

The CUDA command requires an NVIDIA device; use `--device mps` on a supported
Mac. Pytest uses small local fixtures, not downloaded datasets or public weights.
It covers Gaussian algebra/gradients, matched initialization, objective/process
conventions, consumed-batch resume, EMA, statistics, provenance and native latent
workflows. The notebook smoke run checks analytic controls and short training;
it is not evidence of convergence or image quality.

## Diagnose failures at the smallest useful level

| Symptom | Inspect / action |
|---|---|
| Invalid config or unexpected preset | `train.py -c <config> --dry-run`; compare the resolved config saved with the run. |
| CUDA/MPS unavailable or FFT problem | `scripts/doctor.py --device <device>`; record the actual device and wheel versions. |
| Out of memory | Run a separate short experiment with smaller microbatches at the same effective batch; keep that setting fixed across paired arms. |
| Non-finite loss or gradients | Check input range, cached mean/power, noise range and power floor; reproduce on `configs/smoke.json` before modifying the objective. |
| Statistics/cache identity mismatch | Verify dataset, split and preprocessing. Prepare a distinct cache for a changed protocol; do not overwrite the cache used by active runs. |
| Resume source/runtime mismatch | Use the original source checkout and locked environment. Do not bypass provenance checks. |
| Existing output directory | Choose a new run/output name or explicitly resume its `last.pt`. |
| A low DSM value but poor samples | Confirm total-score conversion, EMA and sampler settings. Noisy DSM is not FID or exact true-score error. |
| FID differs from a published number | Compare architecture, training history, image preprocessing, real reference, NFE and metric implementation first. |
| Notebook import/filename failure | Launch the notebook from this checkout; install `--extra notebooks`. Use `FOURIER_SCORE_ROOT` only when root discovery is insufficient. |

Keep failure reproductions small. Include the exact command, resolved config,
source revision, environment, final complete checkpoint step and traceback.
Do not infer a saved step from a progress line printed before checkpoint writing.

## Reproducibility artifacts

Each run saves `config.json`, `environment.json`, `architecture.json`,
`metrics.jsonl`, resumable `last.pt` and inference EMA snapshots. The complete
checkpoint contains optimizer state, EMA, RNGs, the consumed-batch cursor and
statistics. Latent checkpoints also identify the immutable first-stage weights
and latent cache. Sampling records checkpoint identity and settings.

When sharing a result, include the source revision **and dirty diff if any**,
`uv.lock`, resolved configs, split/preprocessing/statistics identity, seed, actual
update count, EMA choice, sampler/actual NFE, sampling batch/seed, metric backend,
real reference and completion status. Record missing measurements as missing.
Report training-seed variation separately from sampling-seed variation.

The [figure guide](FIGURES.md) distinguishes analytic illustrations, synthetic
mechanism measurements and image-model results. Keep those labels in captions.

## Public files and local work

Public source belongs in `fourier_score/`, protocols in `configs/`, reusable
commands in `scripts/`, curated notebooks in `notebooks/`, and derivations and
figures in `docs/`. Git tracks these paths, including newly added documentation.

`local/`, `notes/`, `notebooks/local/` and `verification/` are ignored locations
for private notes, executed exploratory notebooks, machine-specific orchestration,
and raw historical checks. Data, weights and generated runs remain under the
ignored `data/`, `pretrained/`, `saved/` and `evaluation/` paths. Historical files
already present locally are preserved, including retired experiments. Removing
tracking does not remove files from earlier Git history.

Publish a curated result table or figure only with its source records and protocol.
Do not copy raw pod addresses, operational logs or unrelated local notes into the
README. Notebook source should be English and use `$...$` / `$$...$$` math so it
renders in both Jupyter and GitHub. Clear stale errors and keep executed copies
under `saved/`; the source notebook remains a clean executable document.

## Checkpoints and active training

Only supported parameterizations can be loaded in this checkout. Removed
experimental variants are not silently converted to a different method; use
an archived matching revision to inspect those checkpoints.

Training resume requires the same source fingerprint and runtime protocol.
Changes to the local method source therefore require a new experiment or an
original matching checkout for resume. Existing supported inference snapshots
can be loaded with the usual source-difference warning. Numerical backbone and
Gaussian-buffer keys remain unchanged for the supported methods.

Keep an active training checkout at its starting revision until training and
queued evaluation finish. Use a separate worktree for development:

```bash
git worktree add ../fourier-score-dev -b codex/research-cleanup
```

Do not pull new source or rename modules inside an active training checkout;
lazy imports may still read files after startup. Checkpoints reuse the source
fingerprint captured at process startup. A source edit must never relabel an
already running model as if it used the new implementation. Remote scheduling
and checkpoints remain operational artifacts rather than part of the public CLI.
