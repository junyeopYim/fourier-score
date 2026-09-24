# Mechanism notebooks

The [GMM notebook](gmm_fourier_residual.ipynb) studies residual score learning on
Gaussian and mixture distributions with exactly matched population moments.
It uses the repository's DSM objective and evaluates against the exact noisy
joint score. A spectral sweep compares Scalar and Fourier Gaussian covariance
at fixed average power, including the flat-spectrum control ($\lambda=0$).
This file is the canonical guide to running the notebook; the top-level
README only summarizes it.

The population, MLP and evaluation code are shared with
[`fourier_score/gmm.py`](../fourier_score/gmm.py). The notebook retains its three
DSM arms by default. For the five existing baselines plus gated Scalar/Fourier,
including validation-only switch selection and independent test banks, run:

```bash
uv run --locked --extra figures python scripts/run_gmm_comparison.py \
  --output saved/gmm_gated_example --workers 3
```

Add `--preset smoke --seeds 42` for a short CPU pipeline check. The full comparison
trains 99 runs (three switches, three spectra, three seeds), selects one shared
switch using validation, then evaluates the seven final methods. The default
bank version `gmm-gated-v1` differs from the notebook's original `gmm-oracle-v1`.
Both validation and test observations change when the bank version changes;
the training streams remain paired and unchanged.

## Run

From the repository root:

```bash
uv sync --locked --extra notebooks
uv run --locked --extra notebooks jupyter lab notebooks/gmm_fourier_residual.ipynb
```

| Preset | Grid | Updates per run | Training seeds | Purpose |
|---|---|---:|---:|---|
| `smoke` | 4 × 4 | 80 | 1 | Quick CPU example |
| `pilot` | 8 × 8 | 1,500 | 1 | Explore the protocol and runtime |
| `experiment` | 8 × 8 | 5,000 | 3 | Repeated-seed spectral comparison |

The default smoke preset has 12 training runs. The full experiment has
2 distributions × 3 spectra × 3 seeds × 3 methods = 54 runs.

For a command-line execution:

```bash
GMM_PRESET=smoke GMM_DEVICE=cpu GMM_RUN_TAG=example \
  uv run --locked --extra notebooks jupyter nbconvert \
  --to notebook --execute notebooks/gmm_fourier_residual.ipynb \
  --ExecutePreprocessor.timeout=600 --output gmm_example.executed.ipynb \
  --output-dir saved/gmm_oracle
```

Run cells in order. Choose a new `GMM_RUN_TAG` for a new experiment; reuse the
same tag and configuration to continue an interrupted run. Training supports
CPU and CUDA, and the exact-score evaluation uses CPU float64. Set
`FOURIER_SCORE_ROOT` to select a checkout explicitly.

## Measurements and outputs

The primary metric is final-step held-out, noise-scaled true-score MSE.
Backbone initialization and training streams are paired across methods.
Repeated training seeds provide uncertainty estimates and paired Scalar–Fourier
differences on shared evaluation banks.

Results are saved under `saved/gmm_oracle/<preset>/<run_tag>/`:

- CSV tables: per-seed results, paired differences, learning curves, noise and frequency diagnostics.
- Figures: PNG, SVG and PDF exports.
- Checkpoints: model, EMA, optimizer and training state for continuing a run.
- Run settings and numerical checks.

The [figure generation commands](../README.md#figures) produce the accompanying
analytic illustrations. The GMM studies built on this notebook, comparing
training objectives and noise gates, are reported in
[reports/](../reports/README.md).
