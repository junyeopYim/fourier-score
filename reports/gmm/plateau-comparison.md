# GMM plateau comparison

[Back to README](../../README.md#gmm-plateau-comparison) · [All reports](../README.md) · Provenance: [epoch 0](../provenance.md)

**The fixed plateau did not improve the sigmoid gate in this experiment.**
At $\lambda=1$, Plateau Fourier's overall test error increased by **12.6%**
relative to Gated Fourier, and its high-noise error increased by **25.8%**.
The overall regression occurs in all three paired seeds at every spectrum.
Plateau Fourier still beats Scalar/DSM by 16.4% at $\lambda=1$, but does not
retain the better result of the existing sigmoid gate.

This experiment fixes $\sigma_c=1.5$, $p=4$, $\sigma_L=0.8$ and $\sigma_H=1.0$
**before training and evaluation**, without searching the bounds or selecting
on test results. The backbone, parameter count, initialization, online training
streams, Adam settings, EMA and 5,000-update budget match the gated comparison.
Only the gate changes; its reference, residual scale and normalized target are
updated together. No teacher loss or additional model is used.

There are **18 new training runs** (two covariances × three spectra × three
seeds). The other 63 checkpoints are reused after checking their configurations,
statistics, initial/final EMA hashes and exact reproduction of every archived
validation noise-bin metric. All 81 models are evaluated on a **new common test
bank** (`gmm-plateau-v1`): nine log-noise midpoint bins, 2,048 observations per
bin, shared across methods and training seeds. These new observations explain
the small differences from earlier baseline tables. The 12-worker launcher
runs each job with one CPU thread. No run activated gradient clipping.

Values below are noise-scaled true-score MSE, mean ± one sample standard
deviation over training seeds 42, 43 and 44; lower is better.

| Parameterization / objective | $\lambda=0$ | $\lambda=0.5$ | $\lambda=1$ |
|---|---:|---:|---:|
| Score / DSM | 0.05316 ± 0.00018 | 0.05947 ± 0.00057 | 0.07801 ± 0.00101 |
| Scalar / DSM | 0.13331 ± 0.00487 | 0.08684 ± 0.00332 | 0.07474 ± 0.00078 |
| Fourier / DSM | 0.13332 ± 0.00487 | 0.12939 ± 0.00359 | 0.12823 ± 0.00629 |
| Scalar / normalized residual | 0.12541 ± 0.00638 | 0.09348 ± 0.00203 | 0.07866 ± 0.00159 |
| Fourier / normalized residual | 0.12541 ± 0.00638 | 0.12045 ± 0.00475 | 0.08509 ± 0.00387 |
| Gated Scalar / normalized residual | **0.03901 ± 0.00013** | 0.04296 ± 0.00044 | 0.05853 ± 0.00054 |
| Gated Fourier / normalized residual | **0.03901 ± 0.00013** | **0.04291 ± 0.00044** | **0.05553 ± 0.00064** |
| Plateau Scalar / normalized residual | 0.06117 ± 0.00062 | 0.05649 ± 0.00092 | 0.06914 ± 0.00138 |
| Plateau Fourier / normalized residual | 0.06117 ± 0.00062 | 0.05993 ± 0.00064 | 0.06251 ± 0.00020 |

At $\lambda=1$, the low/middle/high regions use the same boundaries as before
($\sigma<0.3$, $0.3\leq\sigma<1$, and $\sigma\geq1$):

| Noise region | Fourier normalized | Gated Fourier | Plateau Fourier | Plateau vs gated |
|---|---:|---:|---:|---:|
| Low | 0.07816 | 0.03455 | **0.02928** | −15.3% |
| Middle | 0.12659 | **0.07197** | 0.08272 | +14.9% |
| High | **0.05053** | 0.06005 | 0.07554 | +25.8% |

The low-noise improvement is outweighed by middle/high-noise regressions.
Despite exactly reaching $g=1$, the plateau's high-noise error is **49.5%**
above the original Fourier-normalized model. A shared, newly trained backbone
does not recover the original model's predictions merely by restoring its
high-noise adapter equations.

![Plateau GMM comparison across spectra and noise levels.](../../assets/gmm_plateau_comparison/gmm_plateau_comparison.svg)

A separate diagnostic evaluates 15 noise levels around the transition at
$\lambda=1$, with 1,024 fresh observations per level. These points are **not**
pooled into the primary nine-bin average. At $\sigma=1$, the test error is
**0.15923** for Plateau Fourier, versus **0.10203** for Gated Fourier and
**0.10703** for Fourier normalized. The increased error near the transition is
visible in every seed. A narrow transition or shared-backbone optimization may
contribute; this experiment does not isolate those causes, optimize the bounds,
or establish sample/image quality. It uses one fixed GMM geometry and budget.

![Plateau gate and dense transition diagnostic.](../../assets/gmm_plateau_comparison/gmm_plateau_transition.svg)

Reproduce all 81 runs from scratch:

```bash
uv run --locked --extra figures python scripts/run_gmm_plateau.py \
  --output saved/gmm_plateau_reproduction --workers 12
```

If the previous comparison checkpoints are available, add
`--reuse-baselines saved/gmm_gated_20260924` to train only the 18 plateau runs.
Reuse is read-only and requires exact validation reproduction. Use
`--preset smoke --seeds 42` for a short pipeline check. The bounds are configurable
with `--sigma-lo` and `--sigma-hi`; use a new output directory and an independent
test-bank version for subsequent model-selection experiments.

Archived [protocol](../../assets/gmm_plateau_comparison/protocol.json) ·
[summary](../../assets/gmm_plateau_comparison/summary.json) ·
[paired comparisons](../../assets/gmm_plateau_comparison/paired_comparisons.csv) ·
[per-seed results](../../assets/gmm_plateau_comparison/per_seed.csv) ·
[noise regions](../../assets/gmm_plateau_comparison/noise_regions.csv) ·
[transition diagnostics](../../assets/gmm_plateau_comparison/transition_resolved.csv) ·
[checkpoint audit](../../assets/gmm_plateau_comparison/checkpoint_audit.json) ·
[verification](../../assets/gmm_plateau_comparison/verification.json).
Figures: [comparison PNG](../../assets/gmm_plateau_comparison/gmm_plateau_comparison.png) /
[PDF](../../assets/gmm_plateau_comparison/gmm_plateau_comparison.pdf),
[transition PNG](../../assets/gmm_plateau_comparison/gmm_plateau_transition.png) /
[PDF](../../assets/gmm_plateau_comparison/gmm_plateau_transition.pdf).
