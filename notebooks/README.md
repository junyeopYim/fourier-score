# Mechanism notebooks

[`gmm_fourier_residual.ipynb`](gmm_fourier_residual.ipynb) tests whether a learned residual captures structure beyond a fixed Gaussian reference. It constructs a Gaussian and a non-Gaussian mixture with exactly matched population mean and covariance, trains with the repository's DSM loss, and evaluates against the exact noisy joint score.

This is an explanatory mechanism experiment. It is separate from the CIFAR-10 and Churches latent image benchmarks. A Fourier advantage is a question to measure, not an assumption in the notebook.

## Environment and execution

From the repository root:

```bash
uv sync --locked --extra notebooks --extra figures
uv run --locked --extra notebooks --extra figures jupyter lab notebooks/gmm_fourier_residual.ipynb
```

The default `smoke` preset runs 12 comparisons with 80 updates each on a 4×4 grid. `pilot` uses an 8×8 grid, 1,500 updates, and one training seed. `experiment` uses an 8×8 grid, 5,000 updates, three spectral settings, and three training seeds: 54 runs in total. Estimate cost with a pilot before using that preset. No preset name establishes convergence.

To execute the full smoke notebook on CPU:

```bash
GMM_PRESET=smoke GMM_DEVICE=cpu GMM_RUN_TAG=smoke_v2 \
  uv run --locked --extra notebooks --extra figures jupyter nbconvert \
  --to notebook --execute notebooks/gmm_fourier_residual.ipynb \
  --ExecutePreprocessor.timeout=600 --output gmm_smoke.executed.ipynb \
  --output-dir saved/gmm_oracle
```

Run all cells in order. An existing output directory is reused only when its configuration, numerical source, notebook code, and runtime match. Select a new `GMM_RUN_TAG` for a different protocol. Separate kernels must use separate tags. To resume, use the same tag and unchanged notebook code; saving outputs does not alter the code hash.

Set `FOURIER_SCORE_ROOT` if the notebook cannot find the checkout. A kernel that has imported a different checkout must be restarted. Float64 oracle evaluation runs on CPU; training supports CPU or CUDA. Use `GMM_DEVICE=cpu` on an MPS host.

## What to inspect

- **Numerical checks:** exact population moments, independent dense-density/autograd oracle comparison, Gaussian-reference agreement, flat-spectrum Scalar–Fourier outputs and gradients, Parseval's identity, covariance-mismatch identity, and repository DSM equivalence.
- **Primary metric:** final-step held-out, noise-scaled true-score MSE. The oracle is never a training label.
- **Controls:** direct score DSM, Scalar Gaussian, Fourier Gaussian, and untrained analytic Gaussian references. Backbone initialization and training random streams are paired across methods.
- **Uncertainty:** repeat training seeds and report paired Scalar–Fourier differences. One seed has no defined seed standard deviation. Shared evaluation banks condition the reported training-seed uncertainty.
- **Recovery:** checkpoints contain model, EMA, optimizer, data RNG, completed update, and source/configuration fingerprints. Only trusted local checkpoints should be loaded.

Outputs live under `saved/gmm_oracle/<preset>/<run_tag>/`. `plan.json` records the actual configuration, runtime, and source hashes. `numerical_tests.json` records deterministic checks; CSV files contain all planned methods/seeds; figures are exported as PNG, SVG, and PDF. Do not promote smoke outputs to experimental evidence or hide unfavorable outcomes.

The checked-in notebook intentionally has no execution output. Keep executed copies and full experiment artifacts under `saved/`, and publish selected figures together with their protocol and result tables when the experiment is complete. Original analytic illustrations are regenerated using `python scripts/plot_diagnostics.py`; see the [figure guide](../docs/FIGURES.md).
