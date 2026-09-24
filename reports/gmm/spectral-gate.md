# GMM spectral gate comparison

[Back to README](../../README.md#gmm-spectral-gate-comparison) · [All reports](../README.md) · Provenance: [epoch 0](../provenance.md)

**The spectral gate improves the sigmoid gate at $\lambda=1$, but not at all
spectra.** Overall Fourier test error falls by **4.0%** relative to Gated
Fourier, **14.4%** relative to Plateau Fourier and **31.5%** relative to
Score/DSM. Each improvement holds in all three paired training seeds.
At $\lambda=0$ and $0.5$, error instead rises by **3.5%** and **2.4%** relative
to the sigmoid gate. This is a fixed candidate, not a selected optimum.

The new arms use `spectral_cap` with $\sigma_c=1.5$, $p=4$, $\delta=0.5$ and a
smoothstep transition from $\sigma=1$ to $2$. These settings were fixed before
training. Both Scalar and Fourier use **one 102,016-parameter backbone and one
normalized residual MSE**, trained from the same random initialization and
online sample/noise stream as their controls. There are no teachers, auxiliary
losses or pretrained initializations. The Gaussian reference and target apply
the new gate consistently; Fourier's $g_k$ acts inside the spectral transform.

The experiment adds **18 runs of 5,000 updates**, using three spectra and seeds
42, 43 and 44, with 12 concurrent single-threaded CPU jobs. The 81 previous
control checkpoints are used only for comparison: each configuration, initial
and final EMA hash, statistics and archived validation noise-bin metric is
checked before reuse. All 99 models are evaluated on the same new test bank
per spectrum (`gmm-spectral-cap-v1`), with 2,048 observations at each of nine
log-noise midpoint bins. The baseline numbers therefore differ slightly from
the earlier tables. Optimizer, EMA, data geometry and training budget are
unchanged, and no run activated gradient clipping.

The metric is noise-scaled true-score MSE, mean ± one sample standard deviation
over three training seeds; lower is better.

| Parameterization / objective | $\lambda=0$ | $\lambda=0.5$ | $\lambda=1$ |
|---|---:|---:|---:|
| Score / DSM | 0.05341 ± 0.00008 | 0.05905 ± 0.00062 | 0.07815 ± 0.00120 |
| Scalar / DSM | 0.13345 ± 0.00491 | 0.08603 ± 0.00320 | 0.07527 ± 0.00088 |
| Fourier / DSM | 0.13345 ± 0.00491 | 0.12891 ± 0.00336 | 0.12882 ± 0.00652 |
| Scalar / normalized residual | 0.12554 ± 0.00608 | 0.09260 ± 0.00185 | 0.07917 ± 0.00164 |
| Fourier / normalized residual | 0.12554 ± 0.00608 | 0.12014 ± 0.00454 | 0.08546 ± 0.00389 |
| Gated Scalar / normalized residual | **0.03917 ± 0.00023** | 0.04237 ± 0.00036 | 0.05874 ± 0.00064 |
| Gated Fourier / normalized residual | **0.03917 ± 0.00023** | **0.04233 ± 0.00035** | 0.05575 ± 0.00076 |
| Plateau Scalar / normalized residual | 0.06113 ± 0.00046 | 0.05594 ± 0.00086 | 0.06926 ± 0.00154 |
| Plateau Fourier / normalized residual | 0.06113 ± 0.00046 | 0.05945 ± 0.00063 | 0.06255 ± 0.00029 |
| Spectral Scalar / normalized residual | 0.04052 ± 0.00016 | 0.04285 ± 0.00030 | 0.05815 ± 0.00085 |
| Spectral Fourier / normalized residual | 0.04052 ± 0.00016 | 0.04333 ± 0.00032 | **0.05354 ± 0.00058** |

For $\lambda=1$, the new Fourier gate improves all three region averages
relative to the sigmoid gate, in every paired seed:

| Noise region | Score / DSM | Fourier normalized | Sigmoid Fourier | Plateau Fourier | Spectral Fourier | Spectral vs sigmoid |
|---|---:|---:|---:|---:|---:|---:|
| Low, $\sigma<0.3$ | 0.04915 | 0.07844 | 0.03453 | **0.02926** | 0.03078 | −10.9% |
| Middle, $0.3\leq\sigma<1$ | 0.09086 | 0.12794 | 0.07283 | 0.08316 | **0.07126** | −2.2% |
| High, $\sigma\geq1$ | 0.09443 | **0.05001** | 0.05988 | 0.07522 | 0.05857 | −2.2% |

The intended combination is **only partially recovered**: low-noise error is
5.2% above Plateau Fourier, middle-noise error beats the sigmoid gate, and
high-noise error remains **17.1% above Fourier normalized**. The low/middle
parameterization is unchanged below $\sigma=1$; improvements there arise after
retraining the shared backbone, not from a different local low-noise formula.

![Spectral gate comparison across spectra and noise levels.](../../assets/gmm_spectral_comparison/gmm_spectral_comparison.svg)

A separate diagnostic uses 16 noise levels from $\sigma=0.6$ to $2.4$, with
1,024 independent test observations per level at $\lambda=1$. These extra
points are not included in the primary average. The spectral gate avoids the
earlier plateau's large transition error. At $\sigma\approx2.483$, its maximum
residual scale ratio is about **1.095**, versus **1.412** for the sigmoid gate;
the theoretical bound is $\sqrt{1+0.5^2}\approx1.118$ once $\sigma\geq2$.
The scale constraint is satisfied, but it does not close the learned-score gap.
This study neither isolates the effects of transition width versus frequency
dependence nor establishes results for other geometries, longer training or
sample/image quality.

![Gate range, residual scale bound and transition errors.](../../assets/gmm_spectral_comparison/gmm_spectral_diagnostics.svg)

Reproduce all 99 runs from scratch:

```bash
uv run --locked --extra figures python scripts/run_gmm_spectral_gate.py \
  --output saved/gmm_spectral_reproduction --workers 12
```

To train only the 18 new arms when the previous checkpoints are available, add
`--reuse-baselines saved/gmm_gated_20260924 saved/gmm_plateau_20260924`.
These are evaluation controls only; new arms always start from random weights.
Use `--preset smoke --seeds 42` for a short pipeline check. Future changes to
`--delta`, `--sigma-lo` or `--sigma-hi` need a new output directory and should be
selected on validation before evaluation with an independent `--test-bank-version`.

Archived [protocol](../../assets/gmm_spectral_comparison/protocol.json) ·
[summary](../../assets/gmm_spectral_comparison/summary.json) ·
[paired comparisons](../../assets/gmm_spectral_comparison/paired_comparisons.csv) ·
[per-seed results](../../assets/gmm_spectral_comparison/per_seed.csv) ·
[noise regions](../../assets/gmm_spectral_comparison/noise_regions.csv) ·
[transition diagnostics](../../assets/gmm_spectral_comparison/transition_resolved.csv) ·
[gate and scale curves](../../assets/gmm_spectral_comparison/gate_profile.csv) ·
[checkpoint audit](../../assets/gmm_spectral_comparison/checkpoint_audit.json) ·
[verification](../../assets/gmm_spectral_comparison/verification.json).
Figures: [comparison PNG](../../assets/gmm_spectral_comparison/gmm_spectral_comparison.png) /
[PDF](../../assets/gmm_spectral_comparison/gmm_spectral_comparison.pdf),
[diagnostic PNG](../../assets/gmm_spectral_comparison/gmm_spectral_diagnostics.png) /
[PDF](../../assets/gmm_spectral_comparison/gmm_spectral_diagnostics.pdf).
