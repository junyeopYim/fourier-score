# Implementation validation

The executable checks live in `tests/`, `scripts/doctor.py` and
`scripts/check_notebooks.py`. The CI workflow runs CPU tests, a real small
NCSN++ forward/backward/sampling check, the GMM notebook smoke experiment and
analytic figure generation. It does not train image models to convergence or
compute their full FID benchmarks.

```bash
uv sync --locked --all-extras
uv lock --check
uv run --locked --all-extras python -m pytest -q
uv run --locked --all-extras python scripts/doctor.py --device cpu
uv run --locked --all-extras python scripts/check_notebooks.py --execute
```

Numerical tests cover reference/filter algebra and gradients, flat-spectrum
controls, paired backbone initialization, score/epsilon conversions, DSM
reductions, data/statistics identity, EMA and exact consumed-batch CPU resume.
Native latent tests cover KL/VQ stages, posterior-aware statistics, original
preprocessing/splits, native sampling, frozen decoder identity and checkpoint
provenance. Download tests use small local fixtures rather than public datasets.

The notebook compares known population moments and analytic noisy scores before
short neural training. It records the source, software environment, random seeds
and settings. Smoke outputs are local execution evidence, not publication claims.
The [figure generator](FIGURES.md) independently checks its analytic calculations
and records the results alongside source arrays and exported PNG/SVG/PDF files.

CUDA and MPS checks are host-specific and should be run explicitly on the target
host. Optional native-source parity checks are available through
`scripts/verify_ldm_upstream.py`; they may download pinned upstream source files.

Raw historical logs remain local under ignored `verification/` paths and apply
only to the source revisions and environments they record. Current test counts
are reported by the command/CI output rather than frozen into this guide.
Implementation correctness checks do not establish speed, convergence or quality
advantages; those require the controlled experiments in [EXPERIMENTS.md](EXPERIMENTS.md).
